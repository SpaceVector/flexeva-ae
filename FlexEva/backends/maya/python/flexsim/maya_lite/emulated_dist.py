"""
Blind logical-world distributed shims for Maya-lite emulation.

This module lets one representative worker execute as if it belonged to a
larger logical distributed job. Collective APIs are routed to fake NCCL via
ctypes so phase-1 emulation still emits low-level NCCL traces without launching
the full logical world.
"""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.nn as nn

from .communicators import build_emulated_communicator_id


_NCCL_DATA_TYPES: dict[torch.dtype, int] = {
    torch.int8: 0,
    torch.uint8: 1,
    torch.int32: 2,
    torch.int64: 4,
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
    torch.bfloat16: 9,
}

_NCCL_REDUCE_OPS: dict[object, int] = {
    dist.ReduceOp.SUM: 0,
    dist.ReduceOp.PRODUCT: 1,
    dist.ReduceOp.MAX: 2,
    dist.ReduceOp.MIN: 3,
    dist.ReduceOp.AVG: 4,
}


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _fake_nccl_path() -> Path:
    return _repo_root() / "fake-cuda" / "build" / "liboutput" / "libnccl.so.2"


_ENV_ENABLE = "FLEXSIM_MAYA_EMULATED_DIST"
_ENV_LOGICAL_RANK = "FLEXSIM_MAYA_LOGICAL_RANK"
_ENV_LOGICAL_WORLD_SIZE = "FLEXSIM_MAYA_LOGICAL_WORLD_SIZE"
_ENV_DEFAULT_BACKEND = "FLEXSIM_MAYA_BACKEND"
_ENV_COLLECTIVE_MODE = "FLEXSIM_MAYA_COLLECTIVE_MODE"
_ENV_COMMUNICATORS_PATH = "FLEXSIM_MAYA_COMMUNICATORS_PATH"


def _dtype_code(tensor: torch.Tensor) -> int:
    code = _NCCL_DATA_TYPES.get(tensor.dtype)
    if code is None:
        raise TypeError(f"unsupported tensor dtype for emulated NCCL: {tensor.dtype}")
    return code


def _flatten_numel(tensor: torch.Tensor) -> int:
    return int(tensor.reshape(-1).numel())


def _copy_tensor_contents(dst: torch.Tensor, src: torch.Tensor) -> None:
    with torch.no_grad():
        if dst.shape == src.shape:
            dst.copy_(src)
            return
        flat_dst = dst.reshape(-1)
        flat_src = src.reshape(-1)
        count = min(flat_dst.numel(), flat_src.numel())
        if count > 0:
            flat_dst[:count].copy_(flat_src[:count])
        if flat_dst.numel() > count:
            flat_dst[count:].zero_()


def _zero_tensor_contents(tensor: torch.Tensor) -> None:
    with torch.no_grad():
        tensor.zero_()


def _stream_handle(stream: torch.cuda.Stream | None) -> ctypes.c_void_p | None:
    if stream is None:
        return None
    return ctypes.c_void_p(int(stream.cuda_stream))


class _FakeNcclRuntime:
    def __init__(self) -> None:
        lib_path = _fake_nccl_path()
        if not lib_path.exists():
            raise FileNotFoundError(f"fake NCCL library not built: {lib_path}")
        self._lib = ctypes.CDLL(str(lib_path))
        self._lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]
        self._lib.ncclGetUniqueId.restype = ctypes.c_int
        self._lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._lib.ncclGetVersion.restype = ctypes.c_int
        self._lib.ncclCommInitRankConfig.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            _NcclUniqueId,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.ncclCommInitRankConfig.restype = ctypes.c_int
        self._lib.ncclCommGetAsyncError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self._lib.ncclCommGetAsyncError.restype = ctypes.c_int
        self._lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        self._lib.ncclCommDestroy.restype = ctypes.c_int
        self._lib.ncclGroupStart.argtypes = []
        self._lib.ncclGroupStart.restype = ctypes.c_int
        self._lib.ncclGroupEnd.argtypes = []
        self._lib.ncclGroupEnd.restype = ctypes.c_int
        self._lib.ncclAllReduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.ncclAllReduce.restype = ctypes.c_int
        self._lib.ncclAllGather.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.ncclAllGather.restype = ctypes.c_int
        self._lib.ncclReduceScatter.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.ncclReduceScatter.restype = ctypes.c_int
        self._lib.ncclBroadcast.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.ncclBroadcast.restype = ctypes.c_int
        self._lib.ncclSend.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.ncclSend.restype = ctypes.c_int
        self._lib.ncclRecv.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.ncclRecv.restype = ctypes.c_int
        self._version_initialized = False

    def _check(self, rc: int, opname: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{opname} failed with ncclResult={rc}")

    def _ensure_version_initialized(self) -> None:
        if self._version_initialized:
            return
        version = ctypes.c_int()
        self._check(self._lib.ncclGetVersion(ctypes.byref(version)), "ncclGetVersion")
        self._version_initialized = True

    def init_comm(self, world_size: int, rank: int, *, comm_id: str | None = None) -> ctypes.c_void_p:
        self._ensure_version_initialized()
        uid = _NcclUniqueId()
        # Real traces commonly include ncclGetUniqueId in the first-step control
        # plane even when communicator identity is otherwise determined by higher
        # level orchestration. Emit the same low-level API before overriding the
        # uid with our deterministic emulated communicator id.
        self._check(self._lib.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
        if comm_id:
            encoded = comm_id.encode("utf-8")
            if len(encoded) >= ctypes.sizeof(_NcclUniqueId):
                raise ValueError(f"emulated communicator id exceeds NCCL uid capacity: {comm_id}")
            uid.internal = encoded
        comm = ctypes.c_void_p()
        self._check(
            self._lib.ncclCommInitRankConfig(ctypes.byref(comm), world_size, uid, rank, None),
            "ncclCommInitRankConfig",
        )
        return comm

    def destroy_comm(self, comm: ctypes.c_void_p | None) -> None:
        if comm:
            self._check(self._lib.ncclCommDestroy(comm), "ncclCommDestroy")

    def comm_get_async_error(self, comm: ctypes.c_void_p | None) -> None:
        if not comm:
            return
        async_error = ctypes.c_int()
        self._check(
            self._lib.ncclCommGetAsyncError(comm, ctypes.byref(async_error)),
            "ncclCommGetAsyncError",
        )

    def group_start(self) -> None:
        self._check(self._lib.ncclGroupStart(), "ncclGroupStart")

    def group_end(self) -> None:
        self._check(self._lib.ncclGroupEnd(), "ncclGroupEnd")

    def all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        op: int,
        comm: ctypes.c_void_p | None,
        stream: ctypes.c_void_p | None = None,
    ) -> None:
        self._check(
            self._lib.ncclAllReduce(
                tensor.data_ptr(),
                tensor.data_ptr(),
                _flatten_numel(tensor),
                _dtype_code(tensor),
                op,
                comm,
                stream,
            ),
            "ncclAllReduce",
        )

    def all_gather(
        self,
        tensor: torch.Tensor,
        outputs: Sequence[torch.Tensor],
        *,
        comm: ctypes.c_void_p | None,
        stream: ctypes.c_void_p | None = None,
    ) -> None:
        recv_ptr = outputs[0].data_ptr() if outputs else tensor.data_ptr()
        self._check(
            self._lib.ncclAllGather(
                tensor.data_ptr(),
                recv_ptr,
                _flatten_numel(tensor),
                _dtype_code(tensor),
                comm,
                stream,
            ),
            "ncclAllGather",
        )

    def reduce_scatter(
        self,
        inputs: Sequence[torch.Tensor],
        output: torch.Tensor,
        *,
        op: int,
        comm: ctypes.c_void_p | None,
        stream: ctypes.c_void_p | None = None,
    ) -> None:
        send_tensor = inputs[0] if inputs else output
        self._check(
            self._lib.ncclReduceScatter(
                send_tensor.data_ptr(),
                output.data_ptr(),
                _flatten_numel(output),
                _dtype_code(output),
                op,
                comm,
                stream,
            ),
            "ncclReduceScatter",
        )

    def broadcast(
        self,
        tensor: torch.Tensor,
        *,
        root: int,
        comm: ctypes.c_void_p | None,
        stream: ctypes.c_void_p | None = None,
    ) -> None:
        self._check(
            self._lib.ncclBroadcast(
                tensor.data_ptr(),
                tensor.data_ptr(),
                _flatten_numel(tensor),
                _dtype_code(tensor),
                root,
                comm,
                stream,
            ),
            "ncclBroadcast",
        )

    def send(
        self,
        tensor: torch.Tensor,
        *,
        peer: int,
        comm: ctypes.c_void_p | None,
        stream: ctypes.c_void_p | None = None,
    ) -> None:
        self._check(
            self._lib.ncclSend(
                tensor.data_ptr(),
                _flatten_numel(tensor),
                _dtype_code(tensor),
                peer,
                comm,
                stream,
            ),
            "ncclSend",
        )

    def recv(
        self,
        tensor: torch.Tensor,
        *,
        peer: int,
        comm: ctypes.c_void_p | None,
        stream: ctypes.c_void_p | None = None,
    ) -> None:
        self._check(
            self._lib.ncclRecv(
                tensor.data_ptr(),
                _flatten_numel(tensor),
                _dtype_code(tensor),
                peer,
                comm,
                stream,
            ),
            "ncclRecv",
        )


@dataclass
class EmulatedWork:
    tensor: torch.Tensor | None = None
    event: torch.cuda.Event | None = None
    process_group: EmulatedProcessGroup | None = None

    def _poll_control_plane(self) -> None:
        if self.process_group is None:
            return
        self.process_group.poll_async_error()

    def wait(self) -> torch.Tensor | None:
        if self.event is not None:
            while True:
                self._poll_control_plane()
                if bool(self.event.query()):
                    break
                self._poll_control_plane()
        return self.tensor

    def is_completed(self) -> bool:
        if self.event is None:
            return True
        self._poll_control_plane()
        complete = bool(self.event.query())
        if not complete:
            self._poll_control_plane()
        return complete


@dataclass
class EmulatedProcessGroup:
    ranks: tuple[int, ...]
    local_rank: int | None
    comm: ctypes.c_void_p | None
    runtime: _FakeNcclRuntime
    backend: str
    collective_mode: str
    name: str
    comm_id: str | None = None
    comm_stream: torch.cuda.Stream | None = None

    @property
    def member(self) -> bool:
        return self.local_rank is not None

    @property
    def size(self) -> int:
        return len(self.ranks)

    def destroy(self) -> None:
        self.runtime.destroy_comm(self.comm)
        self.comm = None

    def _collective_ready(self, tensor: torch.Tensor | None = None) -> bool:
        if not self.member or self.size <= 1:
            return False
        if tensor is None:
            return True
        return tensor.is_cuda

    def _ensure_comm(self) -> None:
        if not self.member or self.size <= 1 or self.comm is not None:
            return
        if self.local_rank is None:
            return
        self.comm = self.runtime.init_comm(self.size, self.local_rank, comm_id=self.comm_id)

    def _ensure_comm_stream(self, tensor: torch.Tensor) -> torch.cuda.Stream:
        if self.comm_stream is None:
            self.comm_stream = torch.cuda.Stream(device=tensor.device, priority=0)
        return self.comm_stream

    def _prepare_stream_handoff(
        self,
        tensor: torch.Tensor,
        *,
        wait_on_comm_stream: bool,
    ) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
        current_stream = torch.cuda.current_stream(device=tensor.device)
        comm_stream = self._ensure_comm_stream(tensor)
        if wait_on_comm_stream:
            ready_event = torch.cuda.Event()
            ready_event.record(current_stream)
            comm_stream.wait_event(ready_event)
        return current_stream, comm_stream

    def _record_completion(
        self,
        *,
        current_stream: torch.cuda.Stream,
        comm_stream: torch.cuda.Stream,
        wait_on_current_stream: bool,
    ) -> torch.cuda.Event:
        stream_completion_event = torch.cuda.Event()
        stream_completion_event.record(comm_stream)
        if wait_on_current_stream:
            current_stream.wait_event(stream_completion_event)
        return stream_completion_event

    def poll_async_error(self) -> None:
        if not self.member or self.comm is None or not torch.cuda.is_available():
            return
        # PyTorch/NCCL work completion paths typically touch the active device
        # before polling communicator async-error state.
        torch.cuda.current_device()
        self.runtime.comm_get_async_error(self.comm)

    def _post_group_end_control_plane(self) -> None:
        if not self.member or self.comm is None:
            return
        self.poll_async_error()

    def all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        op: object,
        async_op: bool = False,
    ) -> torch.cuda.Event | None:
        if self._collective_ready(tensor):
            self._ensure_comm()
            current_stream, comm_stream = self._prepare_stream_handoff(
                tensor,
                wait_on_comm_stream=False,
            )
            with torch.cuda.stream(comm_stream):
                self.runtime.group_start()
                self.runtime.all_reduce(
                    tensor,
                    op=_NCCL_REDUCE_OPS.get(op, 0),
                    comm=self.comm,
                    stream=_stream_handle(comm_stream),
                )
                self.runtime.group_end()
            self._post_group_end_control_plane()
            completion = self._record_completion(
                current_stream=current_stream,
                comm_stream=comm_stream,
                wait_on_current_stream=False,
            )
            if async_op:
                return completion
            return None
        return None

    def all_gather(self, output_tensors: Sequence[torch.Tensor], tensor: torch.Tensor) -> None:
        if self._collective_ready(tensor):
            self._ensure_comm()
            _, comm_stream = self._prepare_stream_handoff(
                tensor,
                wait_on_comm_stream=False,
            )
            runtime_outputs = output_tensors
            if self.collective_mode == "trace_only":
                scratch = output_tensors[0] if output_tensors else tensor
                runtime_outputs = [torch.empty_like(scratch)]
            with torch.cuda.stream(comm_stream):
                self.runtime.all_gather(
                    tensor,
                    runtime_outputs,
                    comm=self.comm,
                    stream=_stream_handle(comm_stream),
                )
        if self.collective_mode == "trace_only":
            for output in output_tensors:
                _zero_tensor_contents(output)
            return
        for output in output_tensors:
            _copy_tensor_contents(output, tensor)

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: Sequence[torch.Tensor],
        *,
        op: object,
    ) -> None:
        if self._collective_ready(output):
            self._ensure_comm()
            _, comm_stream = self._prepare_stream_handoff(
                output,
                wait_on_comm_stream=False,
            )
            runtime_output = output
            if self.collective_mode == "trace_only":
                runtime_output = torch.empty_like(output)
            with torch.cuda.stream(comm_stream):
                self.runtime.reduce_scatter(
                    input_list,
                    runtime_output,
                    op=_NCCL_REDUCE_OPS.get(op, 0),
                    comm=self.comm,
                    stream=_stream_handle(comm_stream),
                )
        if self.collective_mode == "trace_only":
            _zero_tensor_contents(output)
            return
        if input_list:
            _copy_tensor_contents(output, input_list[self.local_rank or 0])

    def broadcast(self, tensor: torch.Tensor, *, root_group_rank: int) -> None:
        if self._collective_ready(tensor):
            self._ensure_comm()
            current_stream, comm_stream = self._prepare_stream_handoff(
                tensor,
                wait_on_comm_stream=False,
            )
            with torch.cuda.stream(comm_stream):
                self.runtime.group_start()
                self.runtime.broadcast(
                    tensor,
                    root=root_group_rank,
                    comm=self.comm,
                    stream=_stream_handle(comm_stream),
                )
                self.runtime.group_end()
            self._post_group_end_control_plane()
            self._record_completion(
                current_stream=current_stream,
                comm_stream=comm_stream,
                wait_on_current_stream=False,
            )

    def barrier(self) -> None:
        if self._collective_ready():
            self._ensure_comm()
            self.runtime.group_start()
            self.runtime.group_end()
            self._post_group_end_control_plane()

    def all_to_all(
        self,
        output_tensors: Sequence[torch.Tensor],
        input_tensors: Sequence[torch.Tensor],
    ) -> None:
        if self._collective_ready(input_tensors[0] if input_tensors else None):
            self._ensure_comm()
            _, comm_stream = self._prepare_stream_handoff(
                input_tensors[0],
                wait_on_comm_stream=False,
            )
            runtime_output_tensors = output_tensors
            if self.collective_mode == "trace_only":
                runtime_output_tensors = tuple(torch.empty_like(tensor) for tensor in output_tensors)
            with torch.cuda.stream(comm_stream):
                self.runtime.group_start()
                for peer, tensor in enumerate(input_tensors):
                    self.runtime.send(
                        tensor,
                        peer=peer,
                        comm=self.comm,
                        stream=_stream_handle(comm_stream),
                    )
                for peer, tensor in enumerate(runtime_output_tensors):
                    self.runtime.recv(
                        tensor,
                        peer=peer,
                        comm=self.comm,
                        stream=_stream_handle(comm_stream),
                    )
                self.runtime.group_end()
            self._post_group_end_control_plane()
        if self.collective_mode == "trace_only":
            for output in output_tensors:
                _zero_tensor_contents(output)
            return
        for output, input_tensor in zip(output_tensors, input_tensors):
            _copy_tensor_contents(output, input_tensor)

    def all_to_all_single(
        self,
        output: torch.Tensor,
        input_tensor: torch.Tensor,
        *,
        output_split_sizes: Sequence[int] | None = None,
        input_split_sizes: Sequence[int] | None = None,
    ) -> None:
        world = max(self.size, 1)
        if input_split_sizes is None:
            assert input_tensor.size(0) % world == 0
            input_split_sizes = [input_tensor.size(0) // world] * world
        if output_split_sizes is None:
            assert output.size(0) % world == 0
            output_split_sizes = [output.size(0) // world] * world

        input_chunks = list(input_tensor.split(tuple(int(size) for size in input_split_sizes), dim=0))
        output_chunks = list(output.split(tuple(int(size) for size in output_split_sizes), dim=0))

        runtime_output_chunks = output_chunks
        if self._collective_ready(input_tensor):
            self._ensure_comm()
            _, comm_stream = self._prepare_stream_handoff(
                input_tensor,
                wait_on_comm_stream=False,
            )
            if self.collective_mode == "trace_only":
                scratch_output = torch.empty_like(output)
                runtime_output_chunks = list(
                    scratch_output.split(tuple(int(size) for size in output_split_sizes), dim=0)
                )
            with torch.cuda.stream(comm_stream):
                self.runtime.group_start()
                for peer, chunk in enumerate(input_chunks):
                    self.runtime.send(
                        chunk,
                        peer=peer,
                        comm=self.comm,
                        stream=_stream_handle(comm_stream),
                    )
                for peer, chunk in enumerate(runtime_output_chunks):
                    self.runtime.recv(
                        chunk,
                        peer=peer,
                        comm=self.comm,
                        stream=_stream_handle(comm_stream),
                    )
                self.runtime.group_end()
            self._post_group_end_control_plane()

        if self.collective_mode == "trace_only":
            _zero_tensor_contents(output)
            return

        copy_count = min(output.size(0), input_tensor.size(0))
        with torch.no_grad():
            if copy_count > 0:
                output[:copy_count].copy_(input_tensor[:copy_count])
            if output.size(0) > copy_count:
                output[copy_count:].zero_()

    def send(self, tensor: torch.Tensor, *, peer_group_rank: int) -> None:
        work = self.send_async(tensor, peer_group_rank=peer_group_rank)
        work.wait()

    def send_async(self, tensor: torch.Tensor, *, peer_group_rank: int) -> EmulatedWork:
        completion: torch.cuda.Event | None = None
        if self._collective_ready(tensor):
            self._ensure_comm()
            current_stream, comm_stream = self._prepare_stream_handoff(
                tensor,
                wait_on_comm_stream=True,
            )
            with torch.cuda.stream(comm_stream):
                self.runtime.group_start()
                self.runtime.send(
                    tensor,
                    peer=peer_group_rank,
                    comm=self.comm,
                    stream=_stream_handle(comm_stream),
                )
                self.runtime.group_end()
            self._post_group_end_control_plane()
            completion = self._record_completion(
                current_stream=current_stream,
                comm_stream=comm_stream,
                wait_on_current_stream=True,
            )
        return EmulatedWork(tensor, event=completion, process_group=self)

    def recv(self, tensor: torch.Tensor, *, peer_group_rank: int) -> None:
        work = self.recv_async(tensor, peer_group_rank=peer_group_rank)
        work.wait()

    def recv_async(self, tensor: torch.Tensor, *, peer_group_rank: int) -> EmulatedWork:
        completion: torch.cuda.Event | None = None
        if self._collective_ready(tensor):
            self._ensure_comm()
            current_stream, comm_stream = self._prepare_stream_handoff(
                tensor,
                wait_on_comm_stream=True,
            )
            with torch.cuda.stream(comm_stream):
                self.runtime.group_start()
                self.runtime.recv(
                    tensor,
                    peer=peer_group_rank,
                    comm=self.comm,
                    stream=_stream_handle(comm_stream),
                )
                self.runtime.group_end()
            self._post_group_end_control_plane()
            completion = self._record_completion(
                current_stream=current_stream,
                comm_stream=comm_stream,
                wait_on_current_stream=True,
            )
        if self.collective_mode == "trace_only":
            _zero_tensor_contents(tensor)
        return EmulatedWork(tensor, event=completion, process_group=self)


class EmulatedDistributedDataParallel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        *,
        process_group: EmulatedProcessGroup | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.module = module
        self.process_group = process_group
        active_group = process_group
        if active_group is not None and active_group.size > 1:
            for parameter in self.module.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(self._all_reduce_hook(active_group))

    @staticmethod
    def _all_reduce_hook(group: EmulatedProcessGroup):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            group.all_reduce(grad, op=dist.ReduceOp.SUM)
            return grad

        return hook

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.module(*args, **kwargs)


class EmulatedDistributedEnvironment:
    def __init__(
        self,
        *,
        logical_rank: int,
        logical_world_size: int,
        default_backend: str | None = None,
        collective_mode: str = "compat",
    ) -> None:
        self.logical_rank = logical_rank
        self.logical_world_size = logical_world_size
        self.default_backend = default_backend or ("nccl" if torch.cuda.is_available() else "gloo")
        self.collective_mode = collective_mode
        self.runtime = _FakeNcclRuntime()
        self.groups: dict[tuple[int, ...], EmulatedProcessGroup] = {}
        self.default_group: EmulatedProcessGroup | None = None
        self.communicators_path = self._communicators_path_from_env()
        self.communicators: dict[str, dict[str, object]] = {}
        self.initialized = False

    @staticmethod
    def _communicators_path_from_env() -> Path | None:
        value = os.environ.get(_ENV_COMMUNICATORS_PATH)
        if not value:
            return None
        return Path(value)

    def _persist_communicators(self) -> None:
        if self.communicators_path is None:
            return
        self.communicators_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "logical_rank": self.logical_rank,
            "logical_world_size": self.logical_world_size,
            "communicators": self.communicators,
        }
        self.communicators_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _register_communicator(
        self,
        *,
        ranks: tuple[int, ...],
        name: str,
        backend: str,
        local_rank: int | None,
        is_default: bool,
    ) -> str:
        comm_id = build_emulated_communicator_id(ranks)
        self.communicators[comm_id] = {
            "members": list(ranks),
            "size": len(ranks),
            "local_rank": local_rank,
            "backend": backend,
            "name": name,
            "is_default": is_default,
            "source": "emulated_dist",
        }
        self._persist_communicators()
        return comm_id

    def _resolve_backend(self, args: Sequence[object], kwargs: dict[str, object]) -> str:
        backend = kwargs.get("backend")
        if isinstance(backend, str) and backend:
            return backend
        if args and isinstance(args[0], str) and args[0]:
            return str(args[0])
        return self.default_backend

    def _validate_logical_membership(self, kwargs: dict[str, object]) -> None:
        requested_rank = kwargs.get("rank")
        if requested_rank is not None and int(requested_rank) != self.logical_rank:
            raise ValueError(
                f"emulated logical rank mismatch: requested rank={requested_rank}, "
                f"expected {self.logical_rank}"
            )
        requested_world_size = kwargs.get("world_size")
        if requested_world_size is not None and int(requested_world_size) != self.logical_world_size:
            raise ValueError(
                "emulated logical world size mismatch: "
                f"requested world_size={requested_world_size}, expected {self.logical_world_size}"
            )

    def init_process_group(self, *args: object, **kwargs: object) -> EmulatedProcessGroup:
        kwargs_dict = dict(kwargs)
        self._validate_logical_membership(kwargs_dict)
        backend = self._resolve_backend(args, kwargs_dict)
        if self.default_group is None:
            ranks = tuple(range(self.logical_world_size))
            comm_id = self._register_communicator(
                ranks=ranks,
                name="world",
                backend=backend,
                local_rank=self.logical_rank,
                is_default=True,
            )
            self.default_group = EmulatedProcessGroup(
                ranks=ranks,
                local_rank=self.logical_rank,
                comm=None,
                runtime=self.runtime,
                backend=backend,
                collective_mode=self.collective_mode,
                name="world",
                comm_id=comm_id,
            )
            self.groups[ranks] = self.default_group
        elif backend:
            self.default_group.backend = backend
            if self.default_group.comm_id:
                self.communicators[self.default_group.comm_id]["backend"] = backend
                self._persist_communicators()
        self.initialized = True
        return self.default_group

    def destroy_process_group(self, group: EmulatedProcessGroup | None = None) -> None:
        if group is None:
            seen: set[int] = set()
            for process_group in self.groups.values():
                if id(process_group) in seen:
                    continue
                process_group.destroy()
                seen.add(id(process_group))
            self.groups.clear()
            self.default_group = None
            self.initialized = False
            return
        group.destroy()

    def new_group(self, ranks: Iterable[int], *args: object, **kwargs: object) -> EmulatedProcessGroup:
        normalized = tuple(int(rank) for rank in ranks)
        backend = self._resolve_backend(args, dict(kwargs))
        cached = self.groups.get(normalized)
        if cached is not None:
            cached.backend = backend
            if cached.comm_id:
                self.communicators[cached.comm_id]["backend"] = backend
                self._persist_communicators()
            return cached
        local_rank = normalized.index(self.logical_rank) if self.logical_rank in normalized else None
        comm = None
        comm_id = self._register_communicator(
            ranks=normalized,
            name=f"group:{','.join(str(rank) for rank in normalized)}",
            backend=backend,
            local_rank=local_rank,
            is_default=False,
        )
        if local_rank is not None:
            comm = None
        group = EmulatedProcessGroup(
            ranks=normalized,
            local_rank=local_rank,
            comm=comm,
            runtime=self.runtime,
            backend=backend,
            collective_mode=self.collective_mode,
            name=f"group:{','.join(str(rank) for rank in normalized)}",
            comm_id=comm_id,
        )
        self.groups[normalized] = group
        return group

    def group_or_world(self, group: EmulatedProcessGroup | None) -> EmulatedProcessGroup:
        return group or self.init_process_group()


def install_emulated_distributed(
    *,
    logical_rank: int,
    logical_world_size: int,
    default_backend: str | None = None,
    collective_mode: str = "compat",
) -> EmulatedDistributedEnvironment:
    env = EmulatedDistributedEnvironment(
        logical_rank=logical_rank,
        logical_world_size=logical_world_size,
        default_backend=default_backend,
        collective_mode=collective_mode,
    )

    def init_process_group(*args: object, **kwargs: object) -> EmulatedProcessGroup:
        return env.init_process_group(*args, **kwargs)

    def destroy_process_group(group: EmulatedProcessGroup | None = None) -> None:
        env.destroy_process_group(group)

    def new_group(ranks: Iterable[int], *args: object, **kwargs: object) -> EmulatedProcessGroup:
        return env.new_group(ranks, *args, **kwargs)

    def get_backend(group: EmulatedProcessGroup | None = None) -> str:
        return env.group_or_world(group).backend

    def all_reduce(
        tensor: torch.Tensor,
        op: object = dist.ReduceOp.SUM,
        group: EmulatedProcessGroup | None = None,
        async_op: bool = False,
    ) -> EmulatedWork | None:
        completion = env.group_or_world(group).all_reduce(
            tensor,
            op=op,
            async_op=async_op,
        )
        return EmulatedWork(tensor, event=completion) if async_op else None

    def all_gather(
        tensor_list: Sequence[torch.Tensor],
        tensor: torch.Tensor,
        group: EmulatedProcessGroup | None = None,
        async_op: bool = False,
    ) -> EmulatedWork | None:
        env.group_or_world(group).all_gather(tensor_list, tensor)
        return EmulatedWork(tensor) if async_op else None

    def reduce_scatter(
        output: torch.Tensor,
        input_list: Sequence[torch.Tensor],
        op: object = dist.ReduceOp.SUM,
        group: EmulatedProcessGroup | None = None,
        async_op: bool = False,
    ) -> EmulatedWork | None:
        env.group_or_world(group).reduce_scatter(output, input_list, op=op)
        return EmulatedWork(output) if async_op else None

    def broadcast(
        tensor: torch.Tensor,
        src: int,
        group: EmulatedProcessGroup | None = None,
        async_op: bool = False,
    ) -> EmulatedWork | None:
        process_group = env.group_or_world(group)
        root_group_rank = process_group.ranks.index(src) if src in process_group.ranks else 0
        process_group.broadcast(tensor, root_group_rank=root_group_rank)
        return EmulatedWork(tensor) if async_op else None

    def barrier(
        group: EmulatedProcessGroup | None = None,
        async_op: bool = False,
        **__: object,
    ) -> EmulatedWork | None:
        env.group_or_world(group).barrier()
        return EmulatedWork() if async_op else None

    def all_to_all(
        output_tensor_list: Sequence[torch.Tensor],
        input_tensor_list: Sequence[torch.Tensor],
        group: EmulatedProcessGroup | None = None,
        async_op: bool = False,
    ) -> EmulatedWork | None:
        env.group_or_world(group).all_to_all(output_tensor_list, input_tensor_list)
        return EmulatedWork(output_tensor_list[0] if output_tensor_list else None) if async_op else None

    def all_to_all_single(
        output: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: Sequence[int] | None = None,
        input_split_sizes: Sequence[int] | None = None,
        group: EmulatedProcessGroup | None = None,
        async_op: bool = False,
    ) -> EmulatedWork | None:
        env.group_or_world(group).all_to_all_single(
            output,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )
        return EmulatedWork(output) if async_op else None

    def send(
        tensor: torch.Tensor,
        dst: int | None = None,
        group: EmulatedProcessGroup | None = None,
        group_dst: int | None = None,
        **__: object,
    ) -> None:
        process_group = env.group_or_world(group)
        if dst is not None and group_dst is not None:
            raise ValueError("specify either dst or group_dst, not both")
        if dst is None and group_dst is None:
            raise ValueError("either dst or group_dst is required")
        peer_group_rank = (
            int(group_dst)
            if group_dst is not None
            else process_group.ranks.index(int(dst))
            if int(dst) in process_group.ranks
            else int(dst)
        )
        process_group.send(tensor, peer_group_rank=peer_group_rank)

    def recv(
        tensor: torch.Tensor,
        src: int | None = None,
        group: EmulatedProcessGroup | None = None,
        group_src: int | None = None,
        **__: object,
    ) -> None:
        process_group = env.group_or_world(group)
        if src is not None and group_src is not None:
            raise ValueError("specify either src or group_src, not both")
        if src is None and group_src is None:
            raise ValueError("either src or group_src is required")
        peer_group_rank = (
            int(group_src)
            if group_src is not None
            else process_group.ranks.index(int(src))
            if int(src) in process_group.ranks
            else int(src)
        )
        process_group.recv(tensor, peer_group_rank=peer_group_rank)

    def isend(
        tensor: torch.Tensor,
        dst: int | None = None,
        group: EmulatedProcessGroup | None = None,
        tag: int = 0,
        group_dst: int | None = None,
    ) -> EmulatedWork | None:
        del tag
        process_group = env.group_or_world(group)
        if not process_group.member:
            return None
        if dst is not None and group_dst is not None:
            raise ValueError("specify either dst or group_dst, not both")
        peer_group_rank = (
            int(group_dst)
            if group_dst is not None
            else process_group.ranks.index(int(dst)) if dst in process_group.ranks else int(dst or 0)
        )
        return process_group.send_async(tensor, peer_group_rank=peer_group_rank)

    def irecv(
        tensor: torch.Tensor,
        src: int | None = None,
        group: EmulatedProcessGroup | None = None,
        tag: int = 0,
        group_src: int | None = None,
    ) -> EmulatedWork | None:
        del tag
        process_group = env.group_or_world(group)
        if not process_group.member:
            return None
        if src is None and group_src is None:
            raise NotImplementedError("any-source recv is not supported in Maya-lite emulation")
        if src is not None and group_src is not None:
            raise ValueError("specify either src or group_src, not both")
        peer_group_rank = (
            int(group_src)
            if group_src is not None
            else process_group.ranks.index(int(src)) if src in process_group.ranks else int(src or 0)
        )
        return process_group.recv_async(tensor, peer_group_rank=peer_group_rank)

    def batch_isend_irecv(p2p_op_list: Sequence[object]) -> list[EmulatedWork]:
        requests: list[EmulatedWork] = []
        for op in p2p_op_list:
            operation = getattr(op, "op")
            tensor = getattr(op, "tensor")
            process_group = getattr(op, "group", None)
            tag = int(getattr(op, "tag", 0))
            group_peer = getattr(op, "group_peer", None)
            peer = getattr(op, "peer", None)
            if operation is isend or getattr(operation, "__name__", "") == "isend":
                request = isend(
                    tensor,
                    dst=None if group_peer is not None else peer,
                    group=process_group,
                    tag=tag,
                    group_dst=group_peer,
                )
            elif operation is irecv or getattr(operation, "__name__", "") == "irecv":
                request = irecv(
                    tensor,
                    src=None if group_peer is not None else peer,
                    group=process_group,
                    tag=tag,
                    group_src=group_peer,
                )
            else:
                raise TypeError(f"unsupported P2P op for Maya-lite emulation: {operation!r}")
            if request is not None:
                requests.append(request)
        return requests

    def get_rank(group: EmulatedProcessGroup | None = None) -> int:
        return env.group_or_world(group).local_rank or 0

    def get_world_size(group: EmulatedProcessGroup | None = None) -> int:
        return env.group_or_world(group).size

    def is_initialized() -> bool:
        return env.initialized

    def is_available() -> bool:
        return True

    def is_gloo_available() -> bool:
        return True

    def is_nccl_available() -> bool:
        return True

    def _get_default_group() -> EmulatedProcessGroup:
        return env.init_process_group()

    def _get_group_size(group: EmulatedProcessGroup | None = None) -> int:
        return get_world_size(group)

    def _rank_not_in_group(group: EmulatedProcessGroup | None) -> bool:
        if group is None:
            return False
        return not env.group_or_world(group).member

    def _get_group_rank(group: EmulatedProcessGroup, global_rank: int) -> int:
        return env.group_or_world(group).ranks.index(int(global_rank))

    def _get_global_rank(group: EmulatedProcessGroup, group_rank: int) -> int:
        return env.group_or_world(group).ranks[int(group_rank)]

    def _find_or_create_pg_by_ranks_and_tag(
        tag: str | None,
        ranks: Iterable[int],
        *args: object,
        **kwargs: object,
    ) -> EmulatedProcessGroup:
        del tag
        return env.new_group(ranks, *args, **kwargs)

    world_group = env.init_process_group()
    module_bindings = {
        "init_process_group": init_process_group,
        "destroy_process_group": destroy_process_group,
        "new_group": new_group,
        "all_reduce": all_reduce,
        "all_gather": all_gather,
        "reduce_scatter": reduce_scatter,
        "broadcast": broadcast,
        "barrier": barrier,
        "monitored_barrier": barrier,
        "all_to_all": all_to_all,
        "all_to_all_single": all_to_all_single,
        "send": send,
        "recv": recv,
        "isend": isend,
        "irecv": irecv,
        "batch_isend_irecv": batch_isend_irecv,
        "get_rank": get_rank,
        "get_world_size": get_world_size,
        "get_backend": get_backend,
        "is_initialized": is_initialized,
        "is_available": is_available,
        "is_gloo_available": is_gloo_available,
        "is_nccl_available": is_nccl_available,
        "_verify_params_across_processes": lambda *args, **kwargs: None,
        "_get_default_group": _get_default_group,
        "_get_group_size": _get_group_size,
        "_rank_not_in_group": _rank_not_in_group,
        "_get_group_rank": _get_group_rank,
        "_get_global_rank": _get_global_rank,
        "_find_or_create_pg_by_ranks_and_tag": _find_or_create_pg_by_ranks_and_tag,
    }
    for module in (dist, c10d):
        for name, value in module_bindings.items():
            setattr(module, name, value)

    for owner in (getattr(dist, "group", None), getattr(c10d, "group", None)):
        if owner is None:
            continue
        try:
            owner.WORLD = world_group
        except Exception:
            pass
    for owner in (getattr(dist, "GroupMember", None), getattr(c10d, "GroupMember", None)):
        if owner is None:
            continue
        try:
            owner.WORLD = world_group
        except Exception:
            pass
    torch.nn.parallel.DistributedDataParallel = EmulatedDistributedDataParallel
    return env


def install_emulated_distributed_from_env() -> EmulatedDistributedEnvironment | None:
    if os.environ.get(_ENV_ENABLE, "").lower() not in {"1", "true", "yes", "on"}:
        return None
    logical_rank = int(os.environ.get(_ENV_LOGICAL_RANK, os.environ.get("RANK", "0")))
    logical_world_size = int(
        os.environ.get(_ENV_LOGICAL_WORLD_SIZE, os.environ.get("WORLD_SIZE", "1"))
    )
    default_backend = os.environ.get(_ENV_DEFAULT_BACKEND)
    collective_mode = os.environ.get(_ENV_COLLECTIVE_MODE, "compat")
    return install_emulated_distributed(
        logical_rank=logical_rank,
        logical_world_size=logical_world_size,
        default_backend=default_backend,
        collective_mode=collective_mode,
    )
