"""Centralized fake-cuda compatibility shims for PyTorch distributed."""

from __future__ import annotations

from contextlib import contextmanager
import os
from functools import wraps
from typing import Iterable

try:
    import torch
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d
    import torch.nn as nn
except Exception:  # pragma: no cover - bootstrap should stay tolerant
    torch = None
    dist = None
    c10d = None
    nn = None


_FORCE_ENV = "FLEXSIM_FAKECUDA_DDP_COMPAT"
_FAKECUDA_DEVICE_NAME = "SimGPU"
_INSTALLED = False
_DEVICE_NAME_CACHE: dict[int, bool] = {}
_ORIGINAL_DDP = None


def _force_enabled() -> bool:
    return os.environ.get(_FORCE_ENV, "").lower() in {"1", "true", "yes", "on"}


def _fakecuda_runtime_enabled() -> bool:
    if _force_enabled():
        return True
    return bool(os.environ.get("FAKECUDA_TARGET_ENV_ROOT") or os.environ.get("FAKECUDA_PROOT_BIN"))


def _iter_tensors(value: object) -> Iterable[object]:
    if torch is None or value is None:
        return
    if torch.is_tensor(value):
        yield value
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_tensors(nested)
        return
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _iter_tensors(nested)


def _is_fakecuda_device_index(index: int) -> bool:
    if _fakecuda_runtime_enabled():
        return True
    cached = _DEVICE_NAME_CACHE.get(index)
    if cached is not None:
        return cached
    result = False
    if torch is not None and torch.cuda.is_available():
        try:
            result = torch.cuda.get_device_name(index) == _FAKECUDA_DEVICE_NAME
        except Exception:
            result = False
    _DEVICE_NAME_CACHE[index] = result
    return result


def _tensor_is_fakecuda(tensor: object) -> bool:
    device = getattr(tensor, "device", None)
    if device is None or getattr(device, "type", None) != "cuda":
        return False
    index = getattr(device, "index", None)
    return _is_fakecuda_device_index(0 if index is None else int(index))


def _should_bypass_verify_params(tensors: object) -> bool:
    if _fakecuda_runtime_enabled():
        return True
    return any(_tensor_is_fakecuda(tensor) for tensor in _iter_tensors(tensors))


def _module_uses_fakecuda(module: object) -> bool:
    if nn is None or not isinstance(module, nn.Module):
        return False
    if _fakecuda_runtime_enabled():
        return True
    for parameter in module.parameters(recurse=True):
        if _tensor_is_fakecuda(parameter):
            return True
    for buffer in module.buffers(recurse=True):
        if _tensor_is_fakecuda(buffer):
            return True
    return False


def _resolve_group_root_global_rank(process_group) -> int:
    if dist is None or process_group is None:
        return 0
    get_process_group_ranks = getattr(dist, "get_process_group_ranks", None)
    if callable(get_process_group_ranks):
        try:
            ranks = get_process_group_ranks(process_group)
            if ranks:
                return int(ranks[0])
        except Exception:
            pass
    get_global_rank = getattr(dist, "get_global_rank", None)
    if callable(get_global_rank):
        try:
            return int(get_global_rank(process_group, 0))
        except Exception:
            pass
    return 0


def _broadcast_tensor(tensor, *, process_group=None):
    if process_group is None:
        return dist.broadcast(tensor, src=0, group=None)
    try:
        return dist.broadcast(tensor, group=process_group, group_src=0)
    except TypeError:
        root_rank = _resolve_group_root_global_rank(process_group)
        return dist.broadcast(tensor, src=root_rank, group=process_group)


def _queue_engine_callback(callback) -> bool:
    if torch is None:
        return False
    try:
        queue_callback = getattr(torch.autograd.Variable._execution_engine, "queue_callback", None)
    except Exception:
        queue_callback = None
    if not callable(queue_callback):
        return False
    queue_callback(callback)
    return True


def _wrap_verify_params(original):
    if original is None:
        return None

    @wraps(original)
    def wrapped(process_group, tensors, logger=None, *args, **kwargs):
        if _should_bypass_verify_params(tensors):
            return None
        try:
            return original(process_group, tensors, logger=logger, *args, **kwargs)
        except TypeError:
            return original(process_group, tensors, *args, **kwargs)

    return wrapped


class _FakeCudaHookDistributedDataParallel(nn.Module):
    """Reducer-free DDP shim for fake-cuda devices.

    This preserves DDP-style gradient synchronization for fake-cuda workloads
    without depending on PyTorch reducer internals that assume pinned-memory
    semantics unavailable on SimGPU.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        process_group=None,
        broadcast_buffers: bool = True,
        init_sync: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.broadcast_buffers = bool(broadcast_buffers)
        self.require_backward_grad_sync = True
        self._sync_enabled = True
        self._group_world_size = 1
        self._ddp_parameters = tuple(
            parameter for parameter in self.module.parameters() if parameter.requires_grad
        )
        self._synced_grad_shadow: dict[int, torch.Tensor] = {}
        self._backward_sync_pending = False
        self._post_backward_queued = False
        self._grad_ready_hook_handles = []
        if dist is not None and dist.is_available() and dist.is_initialized():
            self._group_world_size = dist.get_world_size(group=process_group)
        if self._group_world_size > 1 and init_sync:
            self._broadcast_module_state(
                include_parameters=True,
                include_buffers=self.broadcast_buffers,
            )
        if self._group_world_size > 1:
            self._install_grad_ready_hooks()

    def _broadcast_module_state(
        self,
        *,
        include_parameters: bool,
        include_buffers: bool,
    ) -> None:
        if (
            dist is None
            or not dist.is_available()
            or not dist.is_initialized()
            or self._group_world_size <= 1
        ):
            return
        with torch.no_grad():
            if include_parameters:
                for parameter in self.module.parameters():
                    _broadcast_tensor(parameter.data, process_group=self.process_group)
            if include_buffers:
                for buffer in self.module.buffers():
                    _broadcast_tensor(buffer.data, process_group=self.process_group)

    def _prepare_backward_cycle(self) -> None:
        if self._group_world_size <= 1:
            return
        for parameter in self._ddp_parameters:
            if parameter.grad is None:
                self._synced_grad_shadow.pop(id(parameter), None)
        self._backward_sync_pending = True
        self._post_backward_queued = False

    def _install_grad_ready_hooks(self) -> None:
        for parameter in self._ddp_parameters:
            register_post_accumulate = getattr(parameter, "register_post_accumulate_grad_hook", None)
            if callable(register_post_accumulate):
                handle = register_post_accumulate(self._parameter_grad_ready_hook)
            else:
                handle = parameter.register_hook(self._parameter_grad_ready_hook)
            self._grad_ready_hook_handles.append(handle)

    def _parameter_grad_ready_hook(self, *_args):
        self._queue_post_backward_sync()
        return None

    def _queue_post_backward_sync(self) -> None:
        if (
            not self._backward_sync_pending
            or self._post_backward_queued
            or not self._sync_enabled
            or not self.require_backward_grad_sync
        ):
            return
        self._post_backward_queued = True
        if not _queue_engine_callback(self._finalize_backward_sync):
            self._finalize_backward_sync()

    def _finalize_backward_sync(self) -> None:
        if not self._backward_sync_pending:
            return
        self._backward_sync_pending = False
        self._post_backward_queued = False
        self._synchronize_gradients()

    def _synchronize_gradients(self) -> None:
        if (
            self._group_world_size <= 1
            or dist is None
            or not dist.is_available()
            or not dist.is_initialized()
            or not self._sync_enabled
            or not self.require_backward_grad_sync
        ):
            return
        for parameter in self._ddp_parameters:
            shadow = self._synced_grad_shadow.get(id(parameter))
            grad = parameter.grad
            if grad is None:
                delta = torch.zeros_like(parameter)
            elif shadow is None:
                delta = grad.detach().clone()
            else:
                delta = grad.detach() - shadow
            dist.all_reduce(delta, group=self.process_group, op=dist.ReduceOp.SUM)
            delta.div_(self._group_world_size)
            synced_grad = delta if shadow is None else shadow + delta
            if grad is None:
                parameter.grad = synced_grad
            else:
                grad.copy_(synced_grad)
            self._synced_grad_shadow[id(parameter)] = synced_grad.detach().clone()

    @contextmanager
    def no_sync(self):
        previous_enabled = self._sync_enabled
        previous_require = self.require_backward_grad_sync
        self._sync_enabled = False
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self._sync_enabled = previous_enabled
            self.require_backward_grad_sync = previous_require

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = super().__getattr__("module")
            return getattr(module, name)

    def forward(self, *args, **kwargs):
        self._prepare_backward_cycle()
        if self.broadcast_buffers:
            self._broadcast_module_state(include_parameters=False, include_buffers=True)
        return self.module(*args, **kwargs)


def _install_fakecuda_ddp_factory() -> bool:
    global _ORIGINAL_DDP
    if torch is None or nn is None:
        return False
    parallel = getattr(torch.nn, "parallel", None)
    if parallel is None:
        return False
    current = getattr(parallel, "DistributedDataParallel", None)
    if current is None:
        return False
    if _ORIGINAL_DDP is None:
        _ORIGINAL_DDP = current

    def fakecuda_aware_ddp(module, *args, **kwargs):
        if _module_uses_fakecuda(module):
            return _FakeCudaHookDistributedDataParallel(module, **kwargs)
        return _ORIGINAL_DDP(module, *args, **kwargs)

    setattr(parallel, "DistributedDataParallel", fakecuda_aware_ddp)
    distributed_mod = getattr(parallel, "distributed", None)
    if distributed_mod is not None and hasattr(distributed_mod, "DistributedDataParallel"):
        setattr(distributed_mod, "DistributedDataParallel", fakecuda_aware_ddp)
    return True


def install_fakecuda_ddp_compat() -> bool:
    """Install a centralized DDP verification shim for fake-cuda devices."""
    global _INSTALLED
    if _INSTALLED or torch is None or dist is None or c10d is None:
        return False

    verify = getattr(c10d, "_verify_params_across_processes", None)
    if verify is None:
        verify = getattr(dist, "_verify_params_across_processes", None)
    if verify is None:
        return False

    wrapped = _wrap_verify_params(verify)
    setattr(dist, "_verify_params_across_processes", wrapped)
    if hasattr(c10d, "_verify_params_across_processes"):
        setattr(c10d, "_verify_params_across_processes", wrapped)
    _install_fakecuda_ddp_factory()
    _INSTALLED = True
    return True
