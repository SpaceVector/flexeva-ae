from __future__ import annotations

import importlib

import torch
import torch.nn as nn


def _reload_module():
    module = importlib.import_module("flexsim.maya_lite.fakecuda_compat")
    return importlib.reload(module)


def test_verify_wrapper_delegates_for_normal_tensors(monkeypatch):
    module = _reload_module()
    monkeypatch.delenv("FLEXSIM_FAKECUDA_DDP_COMPAT", raising=False)

    called = {}

    def original(process_group, tensors, logger=None):
        called["process_group"] = process_group
        called["tensors"] = tensors
        called["logger"] = logger
        return "delegated"

    wrapped = module._wrap_verify_params(original)
    tensor = torch.zeros(1)

    assert wrapped("pg", [tensor], logger="logger") == "delegated"
    assert called["process_group"] == "pg"
    assert called["tensors"] == [tensor]
    assert called["logger"] == "logger"


def test_verify_wrapper_bypasses_under_force_env(monkeypatch):
    module = _reload_module()
    monkeypatch.setenv("FLEXSIM_FAKECUDA_DDP_COMPAT", "1")

    called = {"count": 0}

    def original(process_group, tensors, logger=None):
        called["count"] += 1
        return "delegated"

    wrapped = module._wrap_verify_params(original)
    tensor = torch.zeros(1)

    assert wrapped("pg", [tensor], logger="logger") is None
    assert called["count"] == 0


def test_install_fakecuda_ddp_compat_patches_dist_and_c10d(monkeypatch):
    module = _reload_module()
    monkeypatch.setattr(module, "_INSTALLED", False)

    original_dist = getattr(module.dist, "_verify_params_across_processes")
    original_c10d = getattr(module.c10d, "_verify_params_across_processes", None)
    try:
        installed = module.install_fakecuda_ddp_compat()
        assert installed is True
        assert module.dist._verify_params_across_processes is not original_dist
        if original_c10d is not None:
            assert module.c10d._verify_params_across_processes is module.dist._verify_params_across_processes
    finally:
        module.dist._verify_params_across_processes = original_dist
        if original_c10d is not None:
            module.c10d._verify_params_across_processes = original_c10d
        module._INSTALLED = False


def test_fakecuda_ddp_wrapper_broadcasts_module_state_and_buffers(monkeypatch):
    module = _reload_module()

    calls = []

    class _StubDist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size(group=None):
            return 2

        @staticmethod
        def broadcast(tensor, src=0, group=None):
            calls.append((tuple(tensor.shape), src, group))
            return tensor

        @staticmethod
        def all_reduce(tensor, group=None, op=None):
            return tensor

    monkeypatch.setattr(module, "dist", _StubDist)
    model = nn.Linear(4, 3)
    model.register_buffer("running_stat", torch.ones(3))

    wrapped = module._FakeCudaHookDistributedDataParallel(
        model,
        process_group="pg",
        broadcast_buffers=True,
        init_sync=True,
    )

    assert wrapped.require_backward_grad_sync is True
    # weight + bias + one buffer on init
    assert len(calls) == 3
    assert all(call[1] == 0 for call in calls)
    assert all(call[2] == "pg" for call in calls)


def test_broadcast_tensor_uses_group_local_root_when_supported(monkeypatch):
    module = _reload_module()
    calls = []

    class _StubDist:
        @staticmethod
        def broadcast(tensor, src=0, group=None, group_src=None):
            calls.append((src, group, group_src))
            if group is not None and group_src is None:
                raise AssertionError("expected group-local broadcast root")
            return tensor

    monkeypatch.setattr(module, "dist", _StubDist)
    tensor = torch.ones(1)

    module._broadcast_tensor(tensor, process_group="pg")

    assert calls == [(0, "pg", 0)]


def test_fakecuda_ddp_wrapper_queues_post_backward_sync_from_grad_ready(monkeypatch):
    module = _reload_module()
    scheduled = []

    class _StubDist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size(group=None):
            return 2

        @staticmethod
        def broadcast(tensor, src=0, group=None, group_src=None):
            return tensor

        @staticmethod
        def all_reduce(tensor, group=None, op=None):
            return tensor

    monkeypatch.setattr(module, "dist", _StubDist)
    monkeypatch.setattr(module, "_queue_engine_callback", lambda cb: scheduled.append(cb) or True)
    wrapped = module._FakeCudaHookDistributedDataParallel(nn.Linear(2, 2))
    weight = wrapped.module.weight

    weight.grad = torch.ones_like(weight)
    wrapped._prepare_backward_cycle()
    wrapped._parameter_grad_ready_hook()

    assert wrapped._post_backward_queued is True
    assert len(scheduled) == 1

    scheduled[0]()

    assert wrapped._backward_sync_pending is False
    assert wrapped._post_backward_queued is False
    assert torch.allclose(weight.grad, torch.full_like(weight, 0.5))


def test_fakecuda_ddp_wrapper_no_sync_does_not_queue_post_backward_sync(monkeypatch):
    module = _reload_module()
    scheduled = []

    class _StubDist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size(group=None):
            return 2

        @staticmethod
        def broadcast(tensor, src=0, group=None, group_src=None):
            return tensor

        @staticmethod
        def all_reduce(tensor, group=None, op=None):
            return tensor

    monkeypatch.setattr(module, "dist", _StubDist)
    monkeypatch.setattr(module, "_queue_engine_callback", lambda cb: scheduled.append(cb) or True)
    wrapped = module._FakeCudaHookDistributedDataParallel(nn.Linear(2, 2))
    wrapped._prepare_backward_cycle()

    with wrapped.no_sync():
        wrapped._parameter_grad_ready_hook()

    assert scheduled == []
    assert wrapped._backward_sync_pending is True


def test_fakecuda_ddp_wrapper_no_sync_toggles_require_backward_grad_sync(monkeypatch):
    module = _reload_module()

    class _StubDist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size(group=None):
            return 2

        @staticmethod
        def broadcast(tensor, src=0, group=None):
            return tensor

        @staticmethod
        def all_reduce(tensor, group=None, op=None):
            return tensor

    monkeypatch.setattr(module, "dist", _StubDist)
    wrapped = module._FakeCudaHookDistributedDataParallel(nn.Linear(2, 2))

    assert wrapped.require_backward_grad_sync is True
    with wrapped.no_sync():
        assert wrapped.require_backward_grad_sync is False
    assert wrapped.require_backward_grad_sync is True


def test_fakecuda_ddp_wrapper_forwards_missing_attributes(monkeypatch):
    module = _reload_module()

    class _StubDist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def is_available():
            return False

        @staticmethod
        def is_initialized():
            return False

    monkeypatch.setattr(module, "dist", _StubDist)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"hidden_size": 128}

        def forward(self, x):
            return x

    wrapped = module._FakeCudaHookDistributedDataParallel(_Model())
    assert wrapped.config == {"hidden_size": 128}


def test_fakecuda_ddp_wrapper_syncs_unused_parameters_with_zero_gradients(monkeypatch):
    module = _reload_module()
    calls = []

    class _StubDist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size(group=None):
            return 2

        @staticmethod
        def broadcast(tensor, src=0, group=None):
            return tensor

        @staticmethod
        def all_reduce(tensor, group=None, op=None):
            calls.append(tuple(tensor.shape))
            return tensor

    monkeypatch.setattr(module, "dist", _StubDist)
    wrapped = module._FakeCudaHookDistributedDataParallel(nn.Linear(4, 3))
    wrapped.module.weight.grad = torch.ones_like(wrapped.module.weight)
    wrapped.module.bias.grad = None

    wrapped._synchronize_gradients()

    assert wrapped.module.weight.grad is not None
    assert torch.allclose(
        wrapped.module.weight.grad,
        torch.full_like(wrapped.module.weight, 0.5),
    )
    assert wrapped.module.bias.grad is not None
    assert torch.allclose(wrapped.module.bias.grad, torch.zeros_like(wrapped.module.bias))
    assert calls == [(3, 4), (3,)]


def test_fakecuda_ddp_wrapper_no_sync_defers_and_then_syncs_accumulated_grads(monkeypatch):
    module = _reload_module()

    class _StubDist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size(group=None):
            return 2

        @staticmethod
        def broadcast(tensor, src=0, group=None):
            return tensor

        @staticmethod
        def all_reduce(tensor, group=None, op=None):
            return tensor

    monkeypatch.setattr(module, "dist", _StubDist)
    wrapped = module._FakeCudaHookDistributedDataParallel(nn.Linear(2, 2))
    weight = wrapped.module.weight

    with wrapped.no_sync():
        weight.grad = torch.ones_like(weight)
        wrapped._synchronize_gradients()
        assert torch.allclose(weight.grad, torch.ones_like(weight))
        assert wrapped._synced_grad_shadow == {}

    weight.grad = torch.full_like(weight, 3.0)
    wrapped._synchronize_gradients()
    assert torch.allclose(weight.grad, torch.full_like(weight, 1.5))
    assert torch.allclose(
        wrapped._synced_grad_shadow[id(weight)],
        torch.full_like(weight, 1.5),
    )


def test_fakecuda_ddp_wrapper_syncs_only_incremental_delta_after_prior_sync(monkeypatch):
    module = _reload_module()

    class _StubDist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size(group=None):
            return 2

        @staticmethod
        def broadcast(tensor, src=0, group=None):
            return tensor

        @staticmethod
        def all_reduce(tensor, group=None, op=None):
            return tensor

    monkeypatch.setattr(module, "dist", _StubDist)
    wrapped = module._FakeCudaHookDistributedDataParallel(nn.Linear(2, 2))
    weight = wrapped.module.weight

    weight.grad = torch.ones_like(weight)
    wrapped._synchronize_gradients()
    assert torch.allclose(weight.grad, torch.full_like(weight, 0.5))

    weight.grad = torch.full_like(weight, 2.5)
    wrapped._synchronize_gradients()
    assert torch.allclose(weight.grad, torch.full_like(weight, 1.5))


def test_fakecuda_ddp_wrapper_prepare_backward_cycle_clears_shadow_after_zero_grad(monkeypatch):
    module = _reload_module()

    class _StubDist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size(group=None):
            return 2

        @staticmethod
        def broadcast(tensor, src=0, group=None):
            return tensor

        @staticmethod
        def all_reduce(tensor, group=None, op=None):
            return tensor

    monkeypatch.setattr(module, "dist", _StubDist)
    wrapped = module._FakeCudaHookDistributedDataParallel(nn.Linear(2, 2))
    weight = wrapped.module.weight

    weight.grad = torch.ones_like(weight)
    wrapped._synchronize_gradients()
    assert id(weight) in wrapped._synced_grad_shadow

    weight.grad = None
    wrapped._prepare_backward_cycle()
    assert id(weight) not in wrapped._synced_grad_shadow
