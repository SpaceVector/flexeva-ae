import json
import os
from datetime import timedelta
from pathlib import Path

import pytest

cpp_event = pytest.importorskip("cpp_event_py", exc_type=ImportError)
from flexsim.maya_lite.capture_real import (
    _merge_capture_manifest,
    _module_for_api,
    _parse_profiled_rank_groups,
    _parse_rank_list,
    _profiled_rank_groups_from_env,
    _profiled_ranks_from_env,
    _rank_from_env,
    _type_for_event,
    _wrapper_runtime_contract_for_event,
    _world_size_from_env,
    _write_capture_manifest,
)
from flexsim.maya_lite.capture_real import _write_rank_trace
from flexsim.maya_lite.collate import collate_trace_bundle
from flexsim.maya_lite.io import load_trace_directory
from flexsim.maya_lite.planner import plan_profiled_rank_groups, profiled_ranks_for_groups


def test_module_for_api_maps_known_backends():
    assert _module_for_api("ncclAllReduce") == "libnccl.so.2"
    assert _module_for_api("cublasSgemm_v2") == "libcublas.so.12"
    assert _module_for_api("cublasLtMatmul") == "libcublasLt.so.12"
    assert _module_for_api("cuCtxGetCurrent") == "libcuda.so.1"
    assert _module_for_api("cudaLaunchKernel") == "libcudart.so.12"


def test_type_for_event_maps_expected_kinds():
    assert _type_for_event("cublasSgemm_v2", "ComputeKernel") == "blas_compute"
    assert _type_for_event("cublasSetStream_v2", "RuntimeCall") == "stream_op"
    assert _type_for_event("cublasCreate_v2", "RuntimeCall") == "context_op"
    assert _type_for_event("cublasLtCreate", "RuntimeCall") == "context_op"
    assert _type_for_event("cublasLtMatmulDescCreate", "RuntimeCall") == "context_op"
    assert _type_for_event("cublasLtMatmulPreferenceCreate", "RuntimeCall") == "context_op"
    assert _type_for_event("ncclAllReduce", "AllReduce") == "nccl_collective"
    assert _type_for_event("ncclAllReduce", "Unknown") == "nccl_collective"
    assert _type_for_event("ncclReduce", "Unknown") == "nccl_collective"
    assert _type_for_event("ncclAllToAll", "Unknown") == "nccl_collective"
    assert _type_for_event("ncclAllToAllv", "Unknown") == "nccl_collective"
    assert _type_for_event("cudaMemcpyAsync", "MemcpyDeviceToHost") == "mem_copy"
    assert _type_for_event("cudaMemcpyAsync", "Unknown") == "mem_copy"
    assert _type_for_event("cudaMalloc", "MemoryAllocation") == "mem_alloc"
    assert _type_for_event("cudaMalloc", "Unknown") == "mem_alloc"
    assert _type_for_event("cudaLaunchKernel", "ComputeKernel") == "kernel_launch"
    assert _type_for_event("cudaLaunchKernel", "Unknown") == "kernel_launch"
    assert _type_for_event("cudaStreamSynchronize", "RuntimeCall") == "stream_op"
    assert _type_for_event("cudaGetDevice", "RuntimeCall") == "context_op"


def test_wrapper_runtime_contract_for_event_marks_direct_runtime_and_dispatch_only():
    assert _wrapper_runtime_contract_for_event("cudaLaunchKernel", "kernel_launch") == "dispatch_only"
    assert _wrapper_runtime_contract_for_event("cublasGemmEx", "blas_compute") == "dispatch_only"
    assert _wrapper_runtime_contract_for_event("ncclAllReduce", "nccl_collective") == "dispatch_only"
    assert _wrapper_runtime_contract_for_event("cudaMemcpyAsync", "mem_copy") == "dispatch_only"
    assert _wrapper_runtime_contract_for_event("cudaMallocAsync", "mem_alloc") == "dispatch_only"
    assert _wrapper_runtime_contract_for_event("cudaFreeAsync", "mem_alloc") == "dispatch_only"
    assert _wrapper_runtime_contract_for_event("cudaEventRecord", "stream_op") == "dispatch_only"
    assert _wrapper_runtime_contract_for_event("cudaEventRecordWithFlags", "stream_op") == "dispatch_only"
    assert _wrapper_runtime_contract_for_event("cudaStreamWaitEvent", "stream_op") == "dispatch_only"
    assert _wrapper_runtime_contract_for_event("cublasSetStream_v2", "stream_op") == "dispatch_only"
    assert _wrapper_runtime_contract_for_event("cudaMemcpy", "mem_copy") == "direct_runtime"
    assert _wrapper_runtime_contract_for_event("cudaMemGetInfo", "context_op") == "direct_runtime"
    assert _wrapper_runtime_contract_for_event("ncclCommCount", "other") == "direct_runtime"
    assert _wrapper_runtime_contract_for_event("ncclCommUserRank", "other") == "direct_runtime"
    assert _wrapper_runtime_contract_for_event("cublasCreate_v2", "context_op") is None
    assert _wrapper_runtime_contract_for_event("cudaSetDevice", "context_op") is None


def test_rank_from_env_prefers_rank(monkeypatch):
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
    assert _rank_from_env() == 7


def test_rank_from_env_defaults_zero(monkeypatch):
    for key in ("RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
        monkeypatch.delenv(key, raising=False)
    assert _rank_from_env() == 0


def test_world_size_from_env_prefers_world_size(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert _world_size_from_env() == 8


def test_world_size_from_env_defaults_none(monkeypatch):
    for key in ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS"):
        monkeypatch.delenv(key, raising=False)
    assert _world_size_from_env() is None


def test_parse_rank_list_and_groups():
    assert _parse_rank_list("3,1,3,2") == (1, 2, 3)
    assert _parse_profiled_rank_groups("0:0,1;2:2,3") == {0: (0, 1), 2: (2, 3)}


def test_profiled_rank_env_helpers(monkeypatch):
    monkeypatch.setenv("FLEXSIM_PROFILED_RANKS", "0,2")
    monkeypatch.setenv("FLEXSIM_PROFILED_RANK_GROUPS", "0:0,1;2:2,3")
    assert _profiled_ranks_from_env() == (0, 2)
    assert _profiled_rank_groups_from_env() == {0: (0, 1), 2: (2, 3)}


def test_write_rank_trace_preserves_payload_attributes(tmp_path: Path):
    event = cpp_event.EventRecord()
    event.kind = cpp_event.EventKind.AllReduce
    event.api_name = "ncclAllReduce"
    event.timestamp = timedelta(seconds=1)
    event.end_timestamp = timedelta(seconds=1, microseconds=250)
    event.host_duration = timedelta(microseconds=250)
    event.process_id = 11
    event.thread_id = 22
    event.payload.attributes = {
        "count": "1024",
        "dtype_code": "0",
        "collective": "allreduce",
        "reduction": "sum",
    }

    path = _write_rank_trace(tmp_path, [event])
    record = json.loads(path.read_text().strip())

    assert record["api"] == "ncclAllReduce"
    assert record["end_ts"] == 1_000_250
    assert record["host_duration_us"] == 250.0
    assert record["wrapper_runtime_contract"] == "dispatch_only"
    assert "direct_runtime_us" not in record
    assert record["count"] == "1024"
    assert record["dtype_code"] == "0"
    assert record["collective"] == "allreduce"
    assert record["reduction"] == "sum"


def test_write_rank_trace_marks_direct_runtime_for_blocking_wrapper(tmp_path: Path):
    event = cpp_event.EventRecord()
    event.kind = cpp_event.EventKind.RuntimeCall
    event.api_name = "cudaMemGetInfo"
    event.timestamp = timedelta(seconds=2)
    event.end_timestamp = timedelta(seconds=2, microseconds=125)
    event.host_duration = timedelta(microseconds=125)
    event.process_id = 7
    event.thread_id = 9
    event.payload.attributes = {"free_bytes": "1", "total_bytes": "2"}

    path = _write_rank_trace(tmp_path, [event])
    record = json.loads(path.read_text().strip())

    assert record["wrapper_runtime_contract"] == "direct_runtime"
    assert record["direct_runtime_us"] == 125.0


def test_write_rank_trace_marks_async_memcpy_as_dispatch_only_without_direct_runtime(
    tmp_path: Path,
):
    event = cpp_event.EventRecord()
    event.kind = cpp_event.EventKind.RuntimeCall
    event.api_name = "cudaMemcpyAsync"
    event.timestamp = timedelta(seconds=4)
    event.end_timestamp = timedelta(seconds=4, microseconds=37)
    event.host_duration = timedelta(microseconds=37)
    event.process_id = 27
    event.thread_id = 29
    event.payload.attributes = {"bytes": "4096", "kind": "3", "stream_id": "17"}

    path = _write_rank_trace(tmp_path, [event])
    record = json.loads(path.read_text().strip())

    assert record["type"] == "mem_copy"
    assert record["wrapper_runtime_contract"] == "dispatch_only"
    assert record["host_duration_us"] == 37.0
    assert "direct_runtime_us" not in record


def test_write_rank_trace_marks_async_alloc_as_dispatch_only_without_direct_runtime(
    tmp_path: Path,
):
    event = cpp_event.EventRecord()
    event.kind = cpp_event.EventKind.RuntimeCall
    event.api_name = "cudaMallocAsync"
    event.timestamp = timedelta(seconds=5)
    event.end_timestamp = timedelta(seconds=5, microseconds=23)
    event.host_duration = timedelta(microseconds=23)
    event.process_id = 31
    event.thread_id = 37
    event.payload.attributes = {"bytes": "8192", "stream_id": "5", "ptr": "0"}

    path = _write_rank_trace(tmp_path, [event])
    record = json.loads(path.read_text().strip())

    assert record["type"] == "mem_alloc"
    assert record["wrapper_runtime_contract"] == "dispatch_only"
    assert record["host_duration_us"] == 23.0
    assert "direct_runtime_us" not in record


def test_write_rank_trace_marks_event_record_as_dispatch_only_without_direct_runtime(
    tmp_path: Path,
):
    event = cpp_event.EventRecord()
    event.kind = cpp_event.EventKind.RuntimeCall
    event.api_name = "cudaEventRecordWithFlags"
    event.timestamp = timedelta(seconds=6)
    event.end_timestamp = timedelta(seconds=6, microseconds=9)
    event.host_duration = timedelta(microseconds=9)
    event.process_id = 41
    event.thread_id = 43
    event.payload.attributes = {"event_id": "11", "stream_id": "7", "flags": "0"}

    path = _write_rank_trace(tmp_path, [event])
    record = json.loads(path.read_text().strip())

    assert record["api"] == "cudaEventRecordWithFlags"
    assert record["type"] == "stream_op"
    assert record["wrapper_runtime_contract"] == "dispatch_only"
    assert record["host_duration_us"] == 9.0
    assert "direct_runtime_us" not in record


def test_write_rank_trace_marks_cublas_set_stream_as_dispatch_only_without_direct_runtime(
    tmp_path: Path,
):
    event = cpp_event.EventRecord()
    event.kind = cpp_event.EventKind.RuntimeCall
    event.api_name = "cublasSetStream_v2"
    event.timestamp = timedelta(seconds=7)
    event.end_timestamp = timedelta(seconds=7, microseconds=13)
    event.host_duration = timedelta(microseconds=13)
    event.process_id = 47
    event.thread_id = 53
    event.payload.attributes = {"handle_id": "17", "stream_id": "5"}

    path = _write_rank_trace(tmp_path, [event])
    record = json.loads(path.read_text().strip())

    assert record["type"] == "stream_op"
    assert record["wrapper_runtime_contract"] == "dispatch_only"
    assert record["host_duration_us"] == 13.0
    assert "direct_runtime_us" not in record


def test_write_rank_trace_wrapper_timing_fields_override_conflicting_payload_values(
    tmp_path: Path,
):
    event = cpp_event.EventRecord()
    event.kind = cpp_event.EventKind.RuntimeCall
    event.api_name = "cudaEventRecordWithFlags"
    event.timestamp = timedelta(seconds=7)
    event.end_timestamp = timedelta(seconds=7, microseconds=9)
    event.host_duration = timedelta(microseconds=9)
    event.process_id = 47
    event.thread_id = 53
    event.payload.attributes = {
        "event_id": "11",
        "stream_id": "5",
        "wrapper_runtime_contract": "direct_runtime",
        "direct_runtime_us": "999.0",
        "host_duration_us": "777.0",
        "end_ts": "123",
    }

    path = _write_rank_trace(tmp_path, [event])
    record = json.loads(path.read_text().strip())

    assert record["api"] == "cudaEventRecordWithFlags"
    assert record["type"] == "stream_op"
    assert record["wrapper_runtime_contract"] == "dispatch_only"
    assert record["host_duration_us"] == 9.0
    assert record["end_ts"] == 7_000_009
    assert "direct_runtime_us" not in record


def test_write_rank_trace_preserves_authoritative_async_runtime_observation(
    tmp_path: Path,
):
    event = cpp_event.EventRecord()
    event.kind = cpp_event.EventKind.ComputeKernel
    event.api_name = "cudaLaunchKernel"
    event.timestamp = timedelta(seconds=9)
    event.end_timestamp = timedelta(seconds=9, microseconds=14)
    event.host_duration = timedelta(microseconds=14)
    event.process_id = 71
    event.thread_id = 73
    event.payload.attributes = {
        "kernel": "demo_kernel",
        "stream_id": "5",
        "wrapper_runtime_contract": "async_runtime",
        "observed_runtime_us": "1234.5",
        "runtime_observation_source": "capture_real_cuda_event",
    }

    path = _write_rank_trace(tmp_path, [event])
    record = json.loads(path.read_text().strip())

    assert record["type"] == "kernel_launch"
    assert record["wrapper_runtime_contract"] == "async_runtime"
    assert record["observed_runtime_us"] == "1234.5"
    assert record["host_duration_us"] == 14.0
    assert "direct_runtime_us" not in record


def test_write_rank_trace_preserves_launch_boundary_visibility_segments(
    tmp_path: Path,
):
    event = cpp_event.EventRecord()
    event.kind = cpp_event.EventKind.ComputeKernel
    event.api_name = "cudaLaunchKernel"
    event.timestamp = timedelta(seconds=10)
    event.end_timestamp = timedelta(seconds=10, microseconds=21)
    event.host_duration = timedelta(microseconds=21)
    event.process_id = 81
    event.thread_id = 83
    event.payload.attributes = {
        "kernel": "demo_kernel",
        "stream_id": "5",
        "boundary_segment_schema_version": "launch_boundary_visibility_v1",
        "wrapper_segment_coverage": "structural_labels_only_unmeasured",
        "wrapper_segment_sum_us": "0.000",
        "wrapper_segment_unattributed_us": "21.000",
        "caller_visible_elapsed_us": "21.000",
        "actual_launch_visibility_kind": "mixed_or_unresolved",
        "actual_launch_unavailable_reason": (
            "internal_segment_timing_disabled_to_preserve_host_duration"
        ),
        "boundary_visibility_segments": json.dumps(
            [
                {
                    "name": "real_api_body",
                    "visibility_kind": "mixed_or_unresolved",
                    "duration_us": None,
                    "clock": "unmeasured",
                    "included_in_paper_visible_host_duration": False,
                    "included_in_instrumentation_only_duration": False,
                }
            ]
        ),
    }

    path = _write_rank_trace(tmp_path, [event])
    record = json.loads(path.read_text().strip())

    assert record["host_duration_us"] == 21.0
    assert record["end_ts"] == 10_000_021
    assert record["wrapper_runtime_contract"] == "dispatch_only"
    assert record["wrapper_segment_sum_us"] == 0.0
    assert record["wrapper_segment_unattributed_us"] == 21.0
    assert "actual_launch_api_body_us" not in record
    assert record["boundary_visibility_segments"][0]["name"] == "real_api_body"
    assert record["boundary_visibility_segments"][0]["duration_us"] is None
    assert record["boundary_visibility_segments"][0]["included_in_paper_visible_host_duration"] is False


@pytest.mark.parametrize(
    ("api_name", "payload"),
    [
        ("ncclReduce", {"count": "128", "datatype": "7", "op": "0", "root": "0"}),
        ("ncclAllToAll", {"count": "128", "datatype": "7"}),
        ("ncclAllToAllv", {"count": "128", "datatype": "7"}),
    ],
)
def test_write_rank_trace_marks_extended_nccl_collectives_as_dispatch_only(
    tmp_path: Path,
    api_name: str,
    payload: dict[str, str],
):
    event = cpp_event.EventRecord()
    event.kind = cpp_event.EventKind.RuntimeCall
    event.api_name = api_name
    event.timestamp = timedelta(seconds=8)
    event.end_timestamp = timedelta(seconds=8, microseconds=19)
    event.host_duration = timedelta(microseconds=19)
    event.process_id = 59
    event.thread_id = 61
    event.payload.attributes = dict(payload)

    path = _write_rank_trace(tmp_path, [event])
    record = json.loads(path.read_text().strip())

    assert record["api"] == api_name
    assert record["type"] == "nccl_collective"
    assert record["wrapper_runtime_contract"] == "dispatch_only"
    assert record["host_duration_us"] == 19.0
    assert "direct_runtime_us" not in record


def test_write_rank_trace_preserves_handle_and_stream_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MAYA_ENABLE_ACTUAL_CUDA_EVENT_COUNTERPART_DIAGNOSTICS", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_ACTUAL_CUDA_EVENT_COUNTERPART_DIAGNOSTICS", raising=False)
    monkeypatch.delenv("MAYA_ENABLE_HOST_CONTROL_BOUNDARY_COUNTERPART_DIAGNOSTICS", raising=False)
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_HOST_CONTROL_BOUNDARY_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "MAYA_ENABLE_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "MAYA_ENABLE_HOST_CONTROL_ENVELOPE_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_HOST_CONTROL_ENVELOPE_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    event = cpp_event.EventRecord()
    event.kind = cpp_event.EventKind.RuntimeCall
    event.api_name = "cudaStreamWaitEvent"
    event.timestamp = timedelta(seconds=1)
    event.process_id = 33
    event.thread_id = 44
    event.payload.attributes = {
        "stream_id": "123456",
        "event_id": "654321",
        "handle_id": "111",
        "comm_id": "222",
        "world_size": "8",
    }

    path = _write_rank_trace(tmp_path, [event])
    record = json.loads(path.read_text().strip())

    assert record["stream_id"] == "123456"
    assert record["event_id"] == "654321"
    assert record["handle_id"] == "111"
    assert record["comm_id"] == "222"
    assert record["world_size"] == "8"
    assert "actual_cuda_event_counterpart_schema_version" not in record
    assert "host_control_boundary_counterpart_schema_version" not in record
    assert "host_control_envelope_counterpart_schema_version" not in record
    assert "host_control_envelope_counterpart_key" not in record
    assert "host_control_visibility_schema_version" not in record
    assert "selected_occurrence_id" not in record
    assert "actual_boundary_family" not in record
    assert "interval_kind" not in record
    assert "count_once_status" not in record
    assert "safe_to_use_as_repair_evidence" not in record
    assert "host_control_producer_visibility_schema_version" not in record
    assert "host_control_producer_visibility_segments" not in record
    assert "shared_anchor_actual_counterpart_schema_version" not in record
    assert "generic_actual_counterpart_schema_version" not in record
    assert "actual_counterpart_row_id" not in record
    assert "generic_actual_counterpart_row_id" not in record
    assert "actual_timing_status" not in record
    assert "generic_actual_timing_status" not in record
    assert "generic_actual_release_reason" not in record
    assert "actual_endpoint_end_ts_used_as_release" not in record


def _p2p_event(
    api_name: str = "ncclSend",
    *,
    peer: str = "1",
    communicator_members: list[int] | None = None,
) -> cpp_event.EventRecord:
    event = cpp_event.EventRecord()
    event.kind = cpp_event.EventKind.RuntimeCall
    event.api_name = api_name
    event.timestamp = timedelta(seconds=1)
    event.end_timestamp = timedelta(seconds=1, microseconds=13)
    event.host_duration = timedelta(microseconds=13)
    event.process_id = 33
    event.thread_id = 44
    event.payload.attributes = {
        "comm_id": "comm-a",
        "call_idx": "7",
        "peer": peer,
        "count": "1024",
        "numel": "1024",
        "datatype": "9",
        "dtype_code": "9",
        "stream_id": "stream-5",
        "collective": "send" if api_name == "ncclSend" else "recv",
        "runtime_observation_source": "capture_real_cuda_event",
        "observed_runtime_us": "42.5",
    }
    if communicator_members is not None:
        event.payload.attributes["collective_communicator_members"] = (
            communicator_members
        )
    return event


def _allreduce_event() -> cpp_event.EventRecord:
    event = cpp_event.EventRecord()
    event.kind = cpp_event.EventKind.RuntimeCall
    event.api_name = "ncclAllReduce"
    event.timestamp = timedelta(seconds=3)
    event.end_timestamp = timedelta(seconds=3, microseconds=21)
    event.host_duration = timedelta(microseconds=21)
    event.process_id = 33
    event.thread_id = 44
    event.payload.attributes = {
        "comm_id": "comm-a",
        "call_idx": "8",
        "count": "2048",
        "datatype": "9",
        "stream_id": "stream-7",
        "collective": "allreduce",
        "collective_communicator_members": [4, 5, 6, 7],
        "runtime_observation_source": "capture_real_cuda_event",
        "observed_runtime_us": "51.25",
    }
    return event


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_DIAGNOSTICS",
    ],
)
def test_write_rank_trace_opt_in_generic_replay_placement_envelope_actual_counterpart_metadata(
    tmp_path: Path,
    monkeypatch,
    env_key: str,
):
    monkeypatch.setenv("RANK", "6")
    monkeypatch.setenv("FLEXSIM_HOST_MACHINE_ID", "host-g")
    monkeypatch.setenv(env_key, "yes")
    other_env_key = (
        "FLEXSIM_MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_DIAGNOSTICS"
        if env_key
        == "MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_DIAGNOSTICS"
        else "MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_DIAGNOSTICS"
    )
    monkeypatch.delenv(other_env_key, raising=False)

    get_device = cpp_event.EventRecord()
    get_device.kind = cpp_event.EventKind.RuntimeCall
    get_device.api_name = "cudaGetDevice"
    get_device.timestamp = timedelta(seconds=2)
    get_device.end_timestamp = timedelta(seconds=2, microseconds=5)
    get_device.host_duration = timedelta(microseconds=5)
    get_device.process_id = 33
    get_device.thread_id = 44

    allreduce = _allreduce_event()
    allreduce.timestamp = timedelta(seconds=3)
    allreduce.end_timestamp = timedelta(seconds=3, microseconds=21)
    allreduce.host_duration = timedelta(microseconds=21)

    path = _write_rank_trace(tmp_path, [get_device, allreduce])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    first_row, actual_row = rows

    assert len(rows) == 2
    assert first_row["api"] == "cudaGetDevice"
    assert first_row["ts"] == 2_000_000
    assert first_row["end_ts"] == 2_000_005
    assert first_row["host_duration_us"] == 5.0

    assert actual_row["api"] == "ncclAllReduce"
    assert actual_row["ts"] == 3_000_000
    assert actual_row["end_ts"] == 3_000_021
    assert actual_row["host_duration_us"] == 21.0
    assert actual_row["observed_runtime_us"] == "51.25"
    assert actual_row["generic_actual_counterpart_schema_version"] == (
        "generic_replay_placement_envelope_actual_counterpart_metadata_v1"
    )
    assert actual_row["generic_actual_counterpart_opt_in_flag"] is True
    assert actual_row["generic_source_side"] == "actual_endpoint_provenance"
    assert actual_row["generic_actual_counterpart_row_id"].startswith(
        "generic_replay_placement_envelope_actual:"
    )
    assert len(actual_row["generic_actual_counterpart_row_id"].rsplit(":", 1)[1]) == 16
    assert (
        actual_row["generic_actual_counterpart_candidate_kind"]
        == "actual_api_endpoint_row"
    )
    assert actual_row["generic_actual_rank"] == 6
    assert actual_row["generic_actual_api"] == "ncclAllReduce"
    assert actual_row["generic_actual_type"] == "nccl_collective"
    assert actual_row["generic_actual_raw_event_id"] == "rank:6:raw_ordinal:1"
    assert actual_row["generic_actual_raw_ordinal"] == 1
    assert actual_row["generic_actual_trace_id"] is None
    assert actual_row["generic_actual_host_machine_id"] == "host-g"
    assert actual_row["generic_actual_host_dispatch_queue_id"] == "host-g:rank:6"
    assert actual_row["generic_actual_prev_raw_event_id"] == "rank:6:raw_ordinal:0"
    assert actual_row["generic_actual_prev_api"] == "cudaGetDevice"
    assert actual_row["generic_actual_next_raw_event_id"] is None
    assert actual_row["generic_actual_next_api"] is None
    assert actual_row["generic_actual_paper_valid_window_id"] is None
    assert actual_row["generic_actual_in_paper_valid_window"] is None

    assert actual_row["generic_phase1_stable_component_row_id"] is None
    assert actual_row["generic_phase1_component_row_type"] is None
    assert actual_row["generic_phase1_component_kind"] is None
    assert actual_row["generic_phase1_count_once_group_id"] is None
    assert actual_row["generic_phase1_stable_replay_edge_id"] is None
    assert actual_row["generic_counterpart_join_attempted_during_capture"] is False
    assert actual_row["generic_counterpart_join_status"] == (
        "actual_metadata_export_only_predicted_phase1_join_deferred"
    )
    assert actual_row["generic_counterpart_join_confidence"] == "unavailable"
    assert actual_row["generic_counterpart_join_key"] == {
        "generic_actual_rank": 6,
        "generic_actual_api": "ncclAllReduce",
        "generic_actual_type": "nccl_collective",
        "generic_actual_raw_event_id": "rank:6:raw_ordinal:1",
        "generic_actual_raw_ordinal": 1,
        "generic_actual_host_dispatch_queue_id": "host-g:rank:6",
        "generic_actual_stream_resource_id": "rank:6:stream:stream-7",
    }

    assert actual_row["generic_actual_timing_status"] == (
        "endpoint_context_only_strict_counterpart_unavailable"
    )
    assert actual_row["generic_actual_timing_basis"] == "wrapper_endpoint_provenance_only"
    assert actual_row["generic_actual_start_us"] is None
    assert actual_row["generic_actual_end_us"] is None
    assert actual_row["generic_actual_duration_us"] is None
    assert actual_row["generic_actual_wait_start_us"] is None
    assert actual_row["generic_actual_release_us"] is None
    assert actual_row["generic_actual_waited_us"] is None
    assert actual_row["generic_actual_release_reason"] is None
    assert actual_row["generic_actual_released_by_event_id"] is None
    assert actual_row["generic_actual_release_source_kind"] is None
    assert "actual_release_reason" not in actual_row
    assert "actual_released_by_event_id" not in actual_row
    assert actual_row["generic_actual_endpoint_ts_us"] == 3_000_000
    assert actual_row["generic_actual_endpoint_end_ts_us"] == 3_000_021
    assert actual_row["generic_actual_endpoint_host_duration_us"] == 21.0
    assert actual_row["generic_actual_observed_runtime_us"] == "51.25"
    assert actual_row["generic_actual_endpoint_context_only"] is True
    assert (
        actual_row["generic_actual_endpoint_timestamps_used_as_strict_timing"] is False
    )
    assert actual_row["generic_actual_endpoint_end_ts_used_as_wait_release"] is False
    assert actual_row["generic_actual_runtime_direct_substitution"] is False

    assert actual_row["generic_actual_stream_id"] == "stream-7"
    assert actual_row["generic_actual_raw_stream_id"] == "stream-7"
    assert actual_row["generic_actual_stream_resource_id"] == "rank:6:stream:stream-7"
    assert actual_row["generic_actual_stream_namespace_basis"] == (
        "rank_scoped_actual_raw_stream_id_process_local_when_stream_id_present_else_unavailable"
    )
    assert actual_row["generic_predicted_stream_resource_id"] is None
    assert actual_row["generic_predicted_stream_id"] is None
    assert actual_row["generic_stream_namespace_alignment_status"] == (
        "actual_only_unresolved_predicted_namespace_not_joined"
    )
    assert actual_row["generic_actual_count_once_group_id"] is None
    assert actual_row["generic_actual_count_once_interval_id"] is None
    assert actual_row["generic_count_once_status"] == (
        "actual_endpoint_metadata_only_not_strict_non_overlap_proof"
    )
    assert actual_row["generic_count_once_non_overlap_status"] == "unavailable"
    assert actual_row["generic_double_counting_overlap_status"] == "unavailable"
    assert actual_row["generic_wait_map_safety_status"] == "unavailable"
    assert actual_row["generic_diagnostic_only"] is True
    assert actual_row["generic_repair_ready"] is False
    assert actual_row["generic_safe_to_use_as_repair_evidence"] is False
    assert actual_row["generic_safe_to_use_as_subtraction_delta"] is False
    assert actual_row["generic_paper_facing_closure_claimed"] is False
    assert (
        actual_row["generic_native_capture_or_compare_run_for_this_metadata"] is False
    )
    assert actual_row["diagnostic_only"] is True
    assert actual_row["repair_ready"] is False
    assert actual_row["safe_to_use_as_repair_evidence"] is False
    assert actual_row["safe_to_use_as_subtraction_delta"] is False
    assert actual_row["paper_facing_closure_claimed"] is False
    assert actual_row["native_capture_or_compare_run_for_this_metadata"] is False


def test_write_rank_trace_appendix_ab_p2p_actual_counterpart_default_off_absence(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.delenv(
        "MAYA_ENABLE_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )

    path = _write_rank_trace(tmp_path, [_p2p_event()])
    row = json.loads(path.read_text().strip())

    assert row["api"] == "ncclSend"
    assert row["end_ts"] == 1_000_013
    assert row["observed_runtime_us"] == "42.5"
    assert "p2p_actual_counterpart_schema_version" not in row
    assert "p2p_actual_counterpart_opt_in_flag" not in row
    assert "actual_p2p_row_id" not in row
    assert "actual_wait_start_us" not in row
    assert "actual_release_us" not in row
    assert "actual_waited_us" not in row
    assert "actual_release_reason" not in row
    assert "actual_released_by_event_id" not in row
    assert "actual_endpoint_context_only" not in row
    assert "actual_api_end_ts_used_as_release" not in row
    assert "safe_to_use_as_repair_evidence" not in row


def test_write_rank_trace_shared_phase_anchor_actual_counterpart_default_off_absence(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.delenv(
        "MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
        raising=False,
    )

    path = _write_rank_trace(tmp_path, [_p2p_event()])
    row = json.loads(path.read_text().strip())

    assert row["api"] == "ncclSend"
    assert row["end_ts"] == 1_000_013
    assert row["observed_runtime_us"] == "42.5"
    assert "shared_anchor_actual_counterpart_schema_version" not in row
    assert "diagnostic_opt_in_flag" not in row
    assert "strict_actual_release_us" not in row
    assert "actual_endpoint_end_ts_used_as_release" not in row
    assert "actual_endpoint_end_ts_used_as_block_end" not in row
    assert "common_basis_schema_version" not in row
    assert "common_payload_signature" not in row
    assert (
        "selected_allreduce_release_participant_host_dispatch_phase_schema_version"
        not in row
    )


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
    ],
)
def test_write_rank_trace_opt_in_shared_phase_anchor_actual_counterpart_metadata(
    tmp_path: Path,
    monkeypatch,
    env_key: str,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv(env_key, "yes")
    events = [
        _p2p_event("ncclSend", peer="1", communicator_members=[4, 5]),
        _p2p_event("ncclRecv", peer="1", communicator_members=[4, 5]),
        _allreduce_event(),
    ]
    for index, event in enumerate(events):
        event.timestamp = timedelta(seconds=1, microseconds=index * 30)
        event.end_timestamp = timedelta(seconds=1, microseconds=index * 30 + 11)
        event.host_duration = timedelta(microseconds=11)

    path = _write_rank_trace(tmp_path, events)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    send, recv, allreduce = rows

    assert [row["api"] for row in rows] == ["ncclSend", "ncclRecv", "ncclAllReduce"]
    assert all(
        row["shared_anchor_actual_counterpart_schema_version"]
        == "shared_all_rank_phase_anchor_counterpart_replay_causal_edge_metadata_v1"
        for row in rows
    )
    assert all(row["diagnostic_opt_in_flag"] is True for row in rows)
    assert all(row["source_side"] == "actual_endpoint_provenance" for row in rows)
    assert all(row["actual_endpoint_context_only"] is True for row in rows)
    assert all(row["actual_endpoint_end_ts_used_as_release"] is False for row in rows)
    assert all(row["actual_endpoint_end_ts_used_as_block_end"] is False for row in rows)
    assert all(row["strict_actual_wait_start_us"] is None for row in rows)
    assert all(row["strict_actual_release_us"] is None for row in rows)
    assert all(row["strict_actual_release_reason"] is None for row in rows)
    assert all(row["strict_actual_released_by_event_id"] is None for row in rows)
    assert all(row["actual_block_start_us"] is None for row in rows)
    assert all(row["actual_block_end_us"] is None for row in rows)
    assert all(row["actual_block_duration_us"] is None for row in rows)
    assert all(
        "actual endpoint ts/end_ts are retained only as endpoint context"
        in row["actual_block_timing_unavailable_reason"]
        for row in rows
    )
    assert all(
        row["strict_actual_release_observability_status"]
        == "unavailable_without_native_device_stream_release_observer"
        for row in rows
    )
    assert all(row["safe_to_use_as_repair_evidence"] is False for row in rows)
    assert all(row["safe_to_use_as_subtraction_delta"] is False for row in rows)

    assert send["actual_raw_event_id"] == "rank:4:raw_ordinal:0"
    assert send["actual_raw_ordinal"] == 0
    assert send["actual_rank"] == 4
    assert send["actual_api"] == "ncclSend"
    assert send["actual_endpoint_ts_us"] == 1_000_000
    assert send["actual_endpoint_end_ts_us"] == 1_000_011
    assert send["actual_endpoint_host_duration_us"] == 11.0
    assert send["actual_observed_runtime_us"] == "42.5"
    assert send["group_api"] == "ncclP2P"
    assert send["pair_members"] == [4, 5]
    assert send["participant_rank_ids"] == [4, 5]
    assert send["peer_rank"] == 5
    assert send["peer_rank_unavailable_reason"] is None
    assert send["raw_peer"] == "1"
    assert send["raw_peer_local_rank"] == 1
    assert send["raw_peer_semantics"] == "communicator_local_rank_provenance_only"
    assert send["shape_signature"] == (
        "group_api=ncclP2P;collective=p2p;count=1024;datatype=9"
    )
    assert send["normalized_call_order"] == 0
    assert recv["normalized_call_order"] == 0
    assert recv["adjacent_prev_raw_event_id"] == "rank:4:raw_ordinal:0"
    assert recv["actual_endpoint_end_ts_used_as_release"] is False
    assert allreduce["group_api"] == "ncclAllReduce"
    assert allreduce["participant_rank_ids"] == [4, 5, 6, 7]
    assert allreduce["pair_id"] is None
    assert allreduce["pair_members"] is None
    assert allreduce["shape_signature"] == (
        "group_api=ncclAllReduce;collective=allreduce;count=2048;datatype=9"
    )
    assert "endpoint ts/end_ts are provenance only" in allreduce["unavailable_reason"]


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
    ],
)
def test_write_rank_trace_opt_in_shared_phase_anchor_common_basis_fields(
    tmp_path: Path,
    monkeypatch,
    env_key: str,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv(env_key, "1")
    monkeypatch.delenv(
        "MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        raising=False,
    )
    events = [
        _p2p_event("ncclSend", peer="1", communicator_members=[4, 5]),
        _p2p_event("ncclRecv", peer="1", communicator_members=[4, 5]),
        _allreduce_event(),
    ]

    path = _write_rank_trace(tmp_path, events)
    send, recv, allreduce = [json.loads(line) for line in path.read_text().splitlines()]

    assert all(
        row["shared_anchor_actual_counterpart_schema_version"]
        == "shared_all_rank_phase_anchor_counterpart_replay_causal_edge_metadata_v1"
        for row in (send, recv, allreduce)
    )
    assert all(
        row["common_basis_schema_version"]
        == "shared_phase_anchor_common_basis_key_fields_v1"
        for row in (send, recv, allreduce)
    )
    assert [send["common_pair_seq"], recv["common_pair_seq"]] == [0, 1]
    assert [send["common_call_order_index"], recv["common_call_order_index"]] == [0, 1]
    assert send["common_call_order_basis"] == "communicator_pair_sequence"
    assert recv["common_call_order_basis"] == "communicator_pair_sequence"
    assert send["common_group_id_call_index"] is None
    assert send["common_pair_members"] == [4, 5]
    assert send["common_membership_signature"] == "members:4-5"
    assert send["common_payload_signature"] == (
        "group_api=ncclP2P;kind=p2p;api=ncclSend;"
        "members=members:4-5;count=1024;datatype=9;op=null"
    )
    assert send["common_payload_signature_inputs"] == {
        "api": "ncclSend",
        "group_api": "ncclP2P",
        "collective_kind": "p2p",
        "count": "1024",
        "datatype": "9",
        "op": None,
        "membership_signature": "members:4-5",
        "pair_members": [4, 5],
    }
    assert send["payload_basis"] == "raw_operation_semantics_not_stream_only_key"
    assert send["common_api_direction"] == "send"
    assert recv["common_api_direction"] == "recv"
    assert allreduce["common_call_order_basis"] == "raw_semantic_call_idx"
    assert allreduce["common_group_id_call_index"] == 8
    assert allreduce["common_call_order_index"] == 8
    assert allreduce["common_pair_seq"] is None
    assert allreduce["common_collective_kind"] == "allreduce"
    assert allreduce["common_membership_signature"] == "members:4-5-6-7"
    assert allreduce["common_payload_signature"] == (
        "group_api=ncclAllReduce;kind=allreduce;api=ncclAllReduce;"
        "members=members:4-5-6-7;count=2048;datatype=9;op=null"
    )
    assert all(row["actual_endpoint_end_ts_used_as_release"] is False for row in (send, recv, allreduce))
    assert all(row["safe_to_use_as_repair_evidence"] is False for row in (send, recv, allreduce))


def test_write_rank_trace_shared_phase_anchor_allreduce_falls_back_to_group_sequence(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv(
        "MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
        "1",
    )
    event = _allreduce_event()
    event.payload.attributes.pop("call_idx")
    event.payload.attributes["collective_group_id"] = "ncclAllReduce|comm:comm-a|call:13"

    path = _write_rank_trace(tmp_path, [event])
    row = json.loads(path.read_text().strip())

    assert row["common_call_order_basis"] == (
        "recovered_collective_group_sequence_call_ordinal"
    )
    assert row["common_group_id_call_index"] == 13
    assert row["common_call_order_index"] == 13
    assert row["common_key_unavailable_reason"] is None
    assert row["common_pair_seq"] is None
    assert row["safe_to_use_as_repair_evidence"] is False
    assert row["safe_to_use_as_subtraction_delta"] is False


def test_write_rank_trace_shared_phase_anchor_allreduce_keeps_call_idx_without_membership(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv(
        "MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
        "1",
    )
    event = _allreduce_event()
    event.payload.attributes.pop("collective_communicator_members")

    path = _write_rank_trace(tmp_path, [event])
    row = json.loads(path.read_text().strip())

    assert row["participant_rank_ids"] is None
    assert row["common_call_order_basis"] == "raw_semantic_call_idx"
    assert row["common_group_id_call_index"] == 8
    assert row["common_call_order_index"] == 8
    assert row["common_membership_signature"] == "members:unavailable"
    assert row["common_payload_signature"] == (
        "group_api=ncclAllReduce;kind=allreduce;api=ncclAllReduce;"
        "members=members:unavailable;count=2048;datatype=9;op=null"
    )
    assert row["common_payload_signature_inputs"]["membership_signature"] == (
        "members:unavailable"
    )
    assert row["common_key_unavailable_reason"] == "common_membership_unavailable"
    assert row["safe_to_use_as_repair_evidence"] is False
    assert row["safe_to_use_as_subtraction_delta"] is False


def test_write_rank_trace_shared_phase_anchor_allreduce_marks_missing_call_basis_unavailable(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv(
        "MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
        "1",
    )
    event = _allreduce_event()
    event.payload.attributes.pop("call_idx")

    path = _write_rank_trace(tmp_path, [event])
    row = json.loads(path.read_text().strip())

    assert row["common_call_order_basis"] == (
        "unavailable_missing_raw_call_idx_and_group_sequence"
    )
    assert row["common_group_id_call_index"] is None
    assert row["common_call_order_index"] is None
    assert row["common_pair_seq"] is None
    assert row["common_key_unavailable_reason"] == (
        "common_group_call_order_unavailable_missing_raw_call_idx_and_group_sequence"
    )
    assert row["common_rank_window_index"] == 0
    assert row["safe_to_use_as_repair_evidence"] is False
    assert row["safe_to_use_as_subtraction_delta"] is False


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
    ],
)
def test_write_rank_trace_shared_phase_anchor_p2p_raw_peer_is_not_global_rank(
    tmp_path: Path,
    monkeypatch,
    env_key: str,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv(env_key, "1")

    path = _write_rank_trace(tmp_path, [_p2p_event("ncclSend", peer="1")])
    row = json.loads(path.read_text().strip())

    assert row["shared_anchor_actual_counterpart_schema_version"] == (
        "shared_all_rank_phase_anchor_counterpart_replay_causal_edge_metadata_v1"
    )
    assert row["api"] == "ncclSend"
    assert row["pair_members"] is None
    assert row["participant_rank_ids"] is None
    assert row["peer_rank"] is None
    assert row["peer_rank_unavailable_reason"] == (
        "raw_peer_payload_is_communicator_local_and_not_safe_as_global_rank_without_membership"
    )
    assert row["raw_peer"] == "1"
    assert row["raw_peer_local_rank"] == 1
    assert row["raw_peer_semantics"] == "communicator_local_rank_provenance_only"
    assert row["actual_endpoint_context_only"] is True
    assert row["actual_endpoint_end_ts_used_as_release"] is False
    assert row["strict_actual_release_us"] is None
    assert row["safe_to_use_as_repair_evidence"] is False
    assert row["safe_to_use_as_subtraction_delta"] is False


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_COUNTERPART_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_COUNTERPART_DIAGNOSTICS",
    ],
)
def test_write_rank_trace_selected_allreduce_release_participant_phase_metadata(
    tmp_path: Path,
    monkeypatch,
    env_key: str,
):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("FLEXSIM_HOST_MACHINE_ID", "host-a")
    monkeypatch.setenv(env_key, "1")
    monkeypatch.delenv(
        "MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
        raising=False,
    )

    p2p = _p2p_event("ncclSend", communicator_members=[3, 11])
    p2p.timestamp = timedelta(seconds=2)
    p2p.end_timestamp = timedelta(seconds=2, microseconds=9)
    p2p.host_duration = timedelta(microseconds=9)
    allreduce = _allreduce_event()
    allreduce.timestamp = timedelta(seconds=2, microseconds=40)
    allreduce.end_timestamp = timedelta(seconds=2, microseconds=47)
    allreduce.host_duration = timedelta(microseconds=7)
    allreduce.payload.attributes["collective_communicator_members"] = [3, 11]
    allreduce.payload.attributes["call_idx"] = "1022"

    path = _write_rank_trace(tmp_path, [p2p, allreduce])
    p2p_row, allreduce_row = [json.loads(line) for line in path.read_text().splitlines()]

    assert "shared_anchor_actual_counterpart_schema_version" not in p2p_row
    assert (
        "selected_allreduce_release_participant_host_dispatch_phase_schema_version"
        not in p2p_row
    )

    assert allreduce_row["shared_anchor_actual_counterpart_schema_version"] == (
        "shared_all_rank_phase_anchor_counterpart_replay_causal_edge_metadata_v1"
    )
    assert allreduce_row["common_basis_schema_version"] == (
        "shared_phase_anchor_common_basis_key_fields_v1"
    )
    assert allreduce_row["common_call_order_basis"] == "raw_semantic_call_idx"
    assert allreduce_row["common_call_order_index"] == 1022
    assert allreduce_row["common_membership_signature"] == "members:3-11"
    assert allreduce_row["common_payload_signature"] == (
        "group_api=ncclAllReduce;kind=allreduce;api=ncclAllReduce;"
        "members=members:3-11;count=2048;datatype=9;op=null"
    )
    assert allreduce_row[
        "selected_allreduce_release_participant_host_dispatch_phase_schema_version"
    ] == (
        "selected_allreduce_release_participant_host_dispatch_phase_counterpart_metadata_v1"
    )
    assert (
        allreduce_row[
            "selected_allreduce_release_participant_host_dispatch_phase_opt_in_flag"
        ]
        is True
    )
    assert allreduce_row["actual_release_participant_candidate"] is True
    assert allreduce_row["actual_release_participant_api"] == "ncclAllReduce"
    assert allreduce_row["actual_release_participant_rank"] == 3
    assert allreduce_row["actual_release_participant_raw_event_id"] == (
        "rank:3:raw_ordinal:1"
    )
    assert allreduce_row["actual_release_participant_raw_ordinal"] == 1
    assert allreduce_row["actual_release_participant_host_dispatch_queue_id"] == (
        "host-a:rank:3"
    )
    assert allreduce_row["actual_release_participant_host_queue_position"] == 1
    assert allreduce_row["actual_release_participant_prev_raw_event_id"] == (
        "rank:3:raw_ordinal:0"
    )
    assert allreduce_row["actual_release_participant_prev_api"] == "ncclSend"
    assert allreduce_row["actual_release_participant_prev_end_ts_us"] == 2_000_009
    assert allreduce_row["actual_release_participant_endpoint_ts_us"] == 2_000_040
    assert allreduce_row["actual_release_participant_endpoint_end_ts_us"] == 2_000_047
    assert allreduce_row["actual_release_participant_endpoint_host_duration_us"] == 7.0
    assert (
        allreduce_row["actual_release_participant_endpoint_timing_context_only"]
        is True
    )
    assert (
        allreduce_row[
            "actual_release_participant_endpoint_end_ts_used_as_wait_release"
        ]
        is False
    )
    assert (
        allreduce_row[
            "actual_release_participant_endpoint_end_ts_used_as_block_timing"
        ]
        is False
    )
    assert allreduce_row["actual_host_dispatch_phase_arrival_us"] is None
    assert allreduce_row["actual_host_dispatch_phase_release_us"] is None
    assert allreduce_row["actual_host_dispatch_phase_duration_us"] is None
    assert allreduce_row["actual_host_dispatch_phase_queue_wait_us"] is None
    assert allreduce_row["actual_host_dispatch_phase_strict_timing_status"] == (
        "unavailable"
    )
    assert allreduce_row["actual_wait_map_release_timing_status"] == (
        "unavailable_not_endpoint_timing"
    )
    assert allreduce_row["counterpart_join_attempted_during_capture"] is False
    assert allreduce_row["diagnostic_only"] is True
    assert allreduce_row["repair_ready"] is False
    assert allreduce_row["safe_to_use_as_repair_evidence"] is False
    assert allreduce_row["safe_to_use_as_subtraction_delta"] is False
    assert allreduce_row["actual_endpoint_end_ts_used_as_release"] is False
    assert allreduce_row["actual_endpoint_end_ts_used_as_block_end"] is False
    assert allreduce_row["strict_actual_release_us"] is None
    assert allreduce_row["actual_block_start_us"] is None
    assert allreduce_row["actual_block_end_us"] is None
    assert allreduce_row["actual_block_duration_us"] is None


def test_write_rank_trace_nccl_wait_release_counterpart_default_off_absence(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.delenv(
        "MAYA_ENABLE_NCCL_WAIT_RELEASE_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_NCCL_WAIT_RELEASE_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )

    path = _write_rank_trace(tmp_path, [_allreduce_event()])
    row = json.loads(path.read_text().strip())

    assert row["api"] == "ncclAllReduce"
    assert "nccl_wait_release_counterpart_schema_version" not in row
    assert "actual_collective_wait_release_counterpart_id" not in row
    assert "actual_stream_resource_id" not in row
    assert "actual_collective_wait_start_us" not in row
    assert "actual_collective_release_us" not in row
    assert "actual_collective_release_reason" not in row
    assert "strict_runtime_delta_safe" not in row
    assert "strict_waitmap_delta_safe" not in row


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_NCCL_WAIT_RELEASE_COUNTERPART_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_NCCL_WAIT_RELEASE_COUNTERPART_DIAGNOSTICS",
    ],
)
def test_write_rank_trace_opt_in_nccl_wait_release_counterpart_metadata(
    tmp_path: Path,
    monkeypatch,
    env_key: str,
):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv(env_key, "1")
    monkeypatch.delenv(
        "MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "MAYA_ENABLE_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    p2p = _p2p_event("ncclSend", communicator_members=[3, 11])
    allreduce = _allreduce_event()
    allreduce.payload.attributes["collective_communicator_members"] = [3, 11]
    allreduce.payload.attributes["call_idx"] = "1022"

    path = _write_rank_trace(tmp_path, [p2p, allreduce])
    p2p_row, allreduce_row = [json.loads(line) for line in path.read_text().splitlines()]

    assert "nccl_wait_release_counterpart_schema_version" not in p2p_row
    assert "shared_anchor_actual_counterpart_schema_version" not in p2p_row

    assert allreduce_row["nccl_wait_release_counterpart_schema_version"] == (
        "nccl_wait_release_stream_namespace_counterpart_metadata_v1"
    )
    assert allreduce_row["nccl_wait_release_counterpart_opt_in_flag"] is True
    assert allreduce_row["shared_anchor_actual_counterpart_schema_version"] == (
        "shared_all_rank_phase_anchor_counterpart_replay_causal_edge_metadata_v1"
    )
    assert allreduce_row["common_basis_schema_version"] == (
        "shared_phase_anchor_common_basis_key_fields_v1"
    )
    assert allreduce_row["actual_collective_wait_release_scope"] == (
        "actual_ncclAllReduce_endpoint_metadata_for_strict_runtime_waitmap_join"
    )
    assert allreduce_row["actual_collective_wait_release_rank"] == 3
    assert allreduce_row["actual_collective_wait_release_raw_event_id"] == (
        "rank:3:raw_ordinal:1"
    )
    assert allreduce_row["actual_collective_group_api"] == "ncclAllReduce"
    assert allreduce_row["actual_collective_kind"] == "allreduce"
    assert allreduce_row["actual_collective_members"] == [3, 11]
    assert allreduce_row["actual_collective_membership_signature"] == "members:3-11"
    assert allreduce_row["actual_collective_call_order_index"] == 1022
    assert allreduce_row["actual_collective_call_order_basis"] == "raw_semantic_call_idx"
    assert allreduce_row["actual_collective_shape_signature"] == (
        "group_api=ncclAllReduce;collective=allreduce;count=2048;datatype=9"
    )
    assert allreduce_row["actual_collective_payload_signature"] == (
        "group_api=ncclAllReduce;kind=allreduce;api=ncclAllReduce;"
        "members=members:3-11;count=2048;datatype=9;op=null"
    )
    assert allreduce_row["actual_stream_id"] == "stream-7"
    assert allreduce_row["actual_raw_stream_id"] == "stream-7"
    assert allreduce_row["actual_stream_resource_id"] == "rank:3:stream:stream-7"
    assert allreduce_row["actual_stream_namespace_replay_comparable"] is False
    assert allreduce_row["actual_stream_namespace_alignment"] == (
        "actual_only_unresolved_predicted_namespace_not_joined"
    )
    assert allreduce_row["stream_namespace_alignment"] == (
        "actual_only_unresolved_predicted_namespace_not_joined"
    )
    assert allreduce_row["predicted_stream_resource_id"] is None
    assert allreduce_row["actual_collective_wait_start_us"] is None
    assert allreduce_row["actual_collective_release_us"] is None
    assert allreduce_row["actual_collective_waited_us"] is None
    assert allreduce_row["actual_collective_release_reason"] is None
    assert allreduce_row["actual_collective_released_by_event_id"] is None
    assert allreduce_row["actual_collective_release_observability_status"] == (
        "strict_release_timing_unavailable_from_wrapper_endpoint"
    )
    assert "not observable from wrapper endpoint rows" in allreduce_row[
        "actual_collective_release_unavailable_reason"
    ]
    assert allreduce_row["actual_wait_start_us"] is None
    assert allreduce_row["actual_release_us"] is None
    assert allreduce_row["actual_release_reason"] is None
    assert allreduce_row["actual_released_by_event_id"] is None
    assert allreduce_row["wait_map_counterpart_id"] is None
    assert allreduce_row["strict_runtime_delta_safe"] is False
    assert allreduce_row["strict_waitmap_delta_safe"] is False
    assert allreduce_row["strict_delta_safety_status"] == (
        "blocked_missing_stream_alignment_and_actual_wait_release_timing"
    )
    assert allreduce_row["diagnostic_only"] is True
    assert allreduce_row["repair_ready"] is False
    assert allreduce_row["safe_to_use_as_repair_evidence"] is False
    assert allreduce_row["safe_to_use_as_subtraction_delta"] is False


def test_write_rank_trace_selected_allreduce_common_basis_does_not_leak_to_p2p_with_shared_flag(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv(
        "MAYA_ENABLE_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_COUNTERPART_DIAGNOSTICS",
        "1",
    )
    monkeypatch.setenv(
        "MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        "1",
    )
    monkeypatch.delenv(
        "MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
        raising=False,
    )

    p2p = _p2p_event("ncclSend", communicator_members=[3, 11])
    allreduce = _allreduce_event()
    allreduce.payload.attributes["collective_communicator_members"] = [3, 11]
    allreduce.payload.attributes["call_idx"] = "1022"

    path = _write_rank_trace(tmp_path, [p2p, allreduce])
    p2p_row, allreduce_row = [json.loads(line) for line in path.read_text().splitlines()]

    assert p2p_row["shared_anchor_actual_counterpart_schema_version"] == (
        "shared_all_rank_phase_anchor_counterpart_replay_causal_edge_metadata_v1"
    )
    assert p2p_row["api"] == "ncclSend"
    assert "common_basis_schema_version" not in p2p_row
    assert "common_call_order_index" not in p2p_row
    assert "common_payload_signature" not in p2p_row
    assert (
        "selected_allreduce_release_participant_host_dispatch_phase_schema_version"
        not in p2p_row
    )

    assert allreduce_row["shared_anchor_actual_counterpart_schema_version"] == (
        "shared_all_rank_phase_anchor_counterpart_replay_causal_edge_metadata_v1"
    )
    assert allreduce_row["common_basis_schema_version"] == (
        "shared_phase_anchor_common_basis_key_fields_v1"
    )
    assert allreduce_row["common_call_order_index"] == 1022
    assert allreduce_row[
        "selected_allreduce_release_participant_host_dispatch_phase_schema_version"
    ] == (
        "selected_allreduce_release_participant_host_dispatch_phase_counterpart_metadata_v1"
    )
    assert allreduce_row["actual_block_start_us"] is None
    assert allreduce_row["actual_block_end_us"] is None


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_DIAGNOSTICS",
    ],
)
def test_write_rank_trace_opt_in_appendix_ab_p2p_actual_counterpart_metadata(
    tmp_path: Path,
    monkeypatch,
    env_key: str,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv(env_key, "1")

    path = _write_rank_trace(tmp_path, [_p2p_event()])
    row = json.loads(path.read_text().strip())

    assert row["p2p_actual_counterpart_schema_version"] == (
        "appendix_ab_p2p_actual_counterpart_release_metadata_v1"
    )
    assert row["p2p_actual_counterpart_opt_in_flag"] is True
    assert row["actual_p2p_row_id"] == "rank:4:p2p_actual_row:raw_ordinal:0"
    assert row["actual_p2p_occurrence_id"] == (
        "rank:4:p2p:ncclSend:pair_seq:unavailable"
    )
    assert row["actual_rank"] == 4
    assert row["actual_api"] == "ncclSend"
    assert row["actual_raw_event_id"] == "rank:4:raw_ordinal:0"
    assert row["actual_raw_ordinal"] == 0
    assert row["actual_trace_id"] is None
    assert row["actual_paper_valid_window_id"] is None
    assert row["actual_in_paper_valid_window"] is None
    assert row["actual_comm_id"] == "comm-a"
    assert row["actual_canonical_comm_id"] == "comm-a"
    assert row["actual_comm_members"] is None
    assert row["actual_comm_members_unavailable_reason"] == (
        "communicator_membership_recovered_later_by_collate_or_offline_ledger"
    )
    assert row["actual_pair_members"] is None
    assert row["actual_pair_members_basis"] == (
        "unavailable_without_communicator_membership"
    )
    assert row["actual_pair_members_unavailable_reason"] == (
        "raw_peer_payload_is_communicator_local_and_not_safe_as_global_rank_without_membership"
    )
    assert row["actual_peer"] == "1"
    assert row["actual_pair_seq"] is None
    assert row["actual_pair_seq_key"] is None
    assert row["actual_pair_seq_unavailable_reason"] == (
        "communicator_resolved_pair_members_unavailable_during_raw_rank_write"
    )
    assert row["actual_call_idx"] == "7"
    assert row["actual_group_api"] == "ncclP2P"
    assert row["actual_collective"] == "p2p"
    assert row["actual_p2p_direction"] == "send"
    assert row["actual_count_or_numel"] == "1024"
    assert row["actual_datatype_or_dtype_code"] == "9"
    assert row["actual_shape_signature"] == (
        "group_api=ncclP2P;collective=p2p;count=1024;datatype=9"
    )
    assert row["actual_normalized_call_order"] is None
    assert row["actual_stream_id"] == "stream-5"
    assert row["actual_raw_stream_id"] == "stream-5"
    assert row["actual_canonical_stream_id"] == "stream-5"
    assert row["actual_stream_namespace_basis"] == "actual_raw_stream_id_process_local"
    assert row["predicted_stream_resource_id"] is None
    assert row["actual_stream_resource_id"] == "rank:4:stream:stream-5"
    assert row["stream_namespace_alignment"] == (
        "actual_only_unresolved_predicted_namespace_not_joined"
    )
    assert row["actual_api_ts_us"] == 1_000_000
    assert row["actual_api_end_ts_us"] == 1_000_013
    assert row["actual_api_host_duration_us"] == 13.0
    assert row["actual_observed_runtime_us"] == "42.5"
    assert row["actual_endpoint_context_only"] is True
    assert row["actual_api_end_ts_used_as_release"] is False
    assert row["actual_api_end_ts_used_as_block_end"] is False
    assert row["actual_wait_start_us"] is None
    assert row["actual_release_us"] is None
    assert row["actual_waited_us"] is None
    assert row["actual_release_reason"] is None
    assert row["actual_released_by_event_id"] is None
    assert row["actual_released_by_raw_event_id"] is None
    assert row["actual_release_source_kind"] is None
    assert row["actual_release_observability_status"] == (
        "strict_release_timing_unavailable"
    )
    assert "not observable from actual wrapper API endpoints" in row[
        "actual_release_unavailable_reason"
    ]
    assert row["actual_block_start_us"] is None
    assert row["actual_block_end_us"] is None
    assert row["actual_block_duration_us"] is None
    assert row["predicted_stable_block_id"] is None
    assert row["predicted_collective_group_id"] is None
    assert row["predicted_pair_seq"] is None
    assert row["predicted_wait_edge_id"] is None
    assert row["predicted_release_us"] is None
    assert row["actual_counterpart_join_key"] == {
        "rank": 4,
        "api": "ncclSend",
        "group_api": "ncclP2P",
        "canonical_comm_id": "comm-a",
        "pair_members": None,
        "peer": "1",
        "shape_signature": "group_api=ncclP2P;collective=p2p;count=1024;datatype=9",
        "normalized_call_order": None,
    }
    assert row["actual_counterpart_join_method"] == (
        "actual_row_metadata_export_only_predicted_join_not_attempted_during_capture"
    )
    assert row["actual_counterpart_join_confidence"] == "unavailable"
    assert row["actual_counterpart_status"] == (
        "actual_p2p_row_id_exported_predicted_join_not_attempted"
    )
    assert row["double_counting_overlap_status"] == "unavailable"
    assert row["wait_map_safety_status"] == "unavailable"
    assert row["safe_to_use_as_repair_evidence"] is False
    assert row["safe_to_use_as_subtraction_delta"] is False
    assert row["diagnostic_only"] is True
    assert row["repair_ready"] is False
    assert row["native_capture_or_compare_run_for_this_metadata"] is False


def test_write_rank_trace_p2p_pair_sequence_uses_communicator_pair_not_direction(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv(
        "MAYA_ENABLE_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_DIAGNOSTICS",
        "1",
    )
    events = [
        _p2p_event("ncclSend", peer="1", communicator_members=[4, 5]),
        _p2p_event("ncclRecv", peer="1", communicator_members=[4, 5]),
        _p2p_event("ncclSend", peer="1", communicator_members=[4, 5]),
    ]
    for index, event in enumerate(events):
        event.timestamp = timedelta(seconds=1, microseconds=index * 10)
        event.end_timestamp = timedelta(seconds=1, microseconds=index * 10 + 5)

    path = _write_rank_trace(tmp_path, events)
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert [row["actual_api"] for row in rows] == [
        "ncclSend",
        "ncclRecv",
        "ncclSend",
    ]
    assert [row["actual_p2p_direction"] for row in rows] == [
        "send",
        "recv",
        "send",
    ]
    assert [row["actual_collective"] for row in rows] == ["p2p", "p2p", "p2p"]
    assert [row["actual_pair_members"] for row in rows] == [[4, 5], [4, 5], [4, 5]]
    assert [row["actual_pair_seq"] for row in rows] == [0, 1, 2]
    assert [row["actual_normalized_call_order"] for row in rows] == [0, 1, 2]
    assert rows[0]["actual_pair_seq_key"] == [
        "communicator_pair_sequence",
        "p2p",
        "comm-a",
        [4, 5],
    ]
    assert rows[1]["actual_pair_seq_key"] == rows[0]["actual_pair_seq_key"]
    assert rows[0]["actual_counterpart_join_key"]["api"] == "ncclSend"
    assert rows[1]["actual_counterpart_join_key"]["api"] == "ncclRecv"
    assert all(
        row["actual_shape_signature"]
        == "group_api=ncclP2P;collective=p2p;count=1024;datatype=9"
        for row in rows
    )
    assert all(row["actual_pair_seq_unavailable_reason"] is None for row in rows)
    assert all(row["safe_to_use_as_repair_evidence"] is False for row in rows)
    assert all(row["safe_to_use_as_subtraction_delta"] is False for row in rows)


def test_write_rank_trace_appendix_ab_p2p_opt_in_preserves_raw_timing_and_counts(
    tmp_path: Path,
    monkeypatch,
):
    default_dir = tmp_path / "default"
    opt_in_dir = tmp_path / "opt_in"
    events = [_p2p_event("ncclSend"), _p2p_event("ncclRecv")]
    events[1].timestamp = timedelta(seconds=2)
    events[1].end_timestamp = timedelta(seconds=2, microseconds=17)
    events[1].host_duration = timedelta(microseconds=17)

    monkeypatch.setenv("RANK", "5")
    monkeypatch.delenv(
        "MAYA_ENABLE_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    default_path = _write_rank_trace(default_dir, events)
    default_rows = [json.loads(line) for line in default_path.read_text().splitlines()]

    monkeypatch.setenv(
        "MAYA_ENABLE_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_DIAGNOSTICS",
        "1",
    )
    opt_in_path = _write_rank_trace(opt_in_dir, events)
    opt_in_rows = [json.loads(line) for line in opt_in_path.read_text().splitlines()]

    assert [row["api"] for row in opt_in_rows] == [row["api"] for row in default_rows]
    assert len(opt_in_rows) == len(default_rows) == 2
    for default_row, opt_in_row in zip(default_rows, opt_in_rows, strict=True):
        for key in ("ts", "end_ts", "host_duration_us", "observed_runtime_us"):
            assert opt_in_row[key] == default_row[key]
        assert opt_in_row["wrapper_runtime_contract"] == default_row[
            "wrapper_runtime_contract"
        ]
        assert opt_in_row["actual_api_end_ts_us"] == default_row["end_ts"]
        assert opt_in_row["actual_release_us"] is None
        assert opt_in_row["actual_wait_start_us"] is None
        assert opt_in_row["actual_waited_us"] is None
        assert opt_in_row["actual_block_start_us"] is None
        assert opt_in_row["actual_block_end_us"] is None
        assert opt_in_row["actual_block_duration_us"] is None
        assert opt_in_row["actual_api_end_ts_used_as_release"] is False
        assert opt_in_row["actual_api_end_ts_used_as_block_end"] is False
        assert opt_in_row["safe_to_use_as_repair_evidence"] is False
        assert opt_in_row["safe_to_use_as_subtraction_delta"] is False


def test_write_rank_trace_opt_in_host_control_boundary_counterpart_metadata(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("FLEXSIM_HOST_MACHINE_ID", "host-a")
    monkeypatch.setenv("MAYA_ENABLE_HOST_CONTROL_BOUNDARY_COUNTERPART_DIAGNOSTICS", "1")

    get_device = cpp_event.EventRecord()
    get_device.kind = cpp_event.EventKind.RuntimeCall
    get_device.api_name = "cudaGetDevice"
    get_device.timestamp = timedelta(seconds=1)
    get_device.end_timestamp = timedelta(seconds=1, microseconds=3)
    get_device.host_duration = timedelta(microseconds=3)
    get_device.process_id = 33
    get_device.thread_id = 44

    set_stream = cpp_event.EventRecord()
    set_stream.kind = cpp_event.EventKind.RuntimeCall
    set_stream.api_name = "cublasSetStream_v2"
    set_stream.timestamp = timedelta(seconds=1, microseconds=9)
    set_stream.end_timestamp = timedelta(seconds=1, microseconds=14)
    set_stream.host_duration = timedelta(microseconds=5)
    set_stream.process_id = 33
    set_stream.thread_id = 44
    set_stream.payload.attributes = {"stream_id": "7"}

    path = _write_rank_trace(tmp_path, [get_device, set_stream])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    first_row, boundary_row = rows

    assert first_row["host_duration_us"] == 3.0
    assert boundary_row["host_duration_us"] == 5.0
    assert boundary_row["end_ts"] == 1_000_014
    assert boundary_row["host_control_boundary_counterpart_schema_version"] == (
        "host_control_boundary_visibility_unblocker_v2_row_evidence_v1"
    )
    assert boundary_row["host_control_visibility_schema_version"] == (
        "host_control_launch_neighborhood_visibility_counterpart_isolation_v1"
    )
    assert boundary_row["host_control_visibility_opt_in_flag"] is True
    assert boundary_row["host_control_envelope_counterpart_schema_version"] == (
        "host_control_replay_envelope_counterpart_metadata_v1"
    )
    assert boundary_row["host_control_envelope_counterpart_opt_in_flag"] is True
    assert boundary_row["host_control_envelope_counterpart_key"] == (
        "rank:2:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
    )
    assert boundary_row["hostdelay_counterpart_key"] == (
        "rank:2:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
    )
    assert boundary_row["host_control_envelope_actual_row_id"] == (
        "rank:2:raw_ordinal:1"
    )
    assert boundary_row["host_control_envelope_actual_interval_id"] == (
        "rank:2:host_control_boundary:raw_ordinal:0->raw_ordinal:1:"
        "actual_endpoint_gap"
    )
    assert boundary_row["host_control_envelope_prev_raw_event_id"] == (
        "rank:2:raw_ordinal:0"
    )
    assert boundary_row["host_control_envelope_current_raw_event_id"] == (
        "rank:2:raw_ordinal:1"
    )
    assert boundary_row["host_control_envelope_prev_api"] == "cudaGetDevice"
    assert boundary_row["host_control_envelope_current_api"] == "cublasSetStream_v2"
    assert boundary_row["host_control_envelope_rank"] == 2
    assert boundary_row["host_control_envelope_stream_id"] == "7"
    assert boundary_row["host_control_envelope_host_dispatch_queue_id"] == (
        "host-a:rank:2"
    )
    assert boundary_row["host_control_envelope_prev_raw_ordinal"] == 0
    assert boundary_row["host_control_envelope_current_raw_ordinal"] == 1
    assert boundary_row["host_control_envelope_timestamp_basis"] == (
        "actual_previous_end_ts_to_current_ts"
    )
    assert boundary_row["host_control_envelope_interval_start_ts_us"] == 1_000_003
    assert boundary_row["host_control_envelope_interval_end_ts_us"] == 1_000_009
    assert boundary_row["host_control_envelope_interval_duration_us"] == 6.0
    assert boundary_row["host_control_envelope_visibility_kind"] == (
        "mixed_or_unresolved"
    )
    assert boundary_row["host_control_envelope_replay_overlap_status"] == (
        "unavailable"
    )
    assert boundary_row["selected_occurrence_id"] == (
        "rank:2:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
    )
    assert boundary_row["host_control_boundary_occurrence_id"] == (
        "rank:2:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
    )
    assert boundary_row["host_control_boundary_prev_raw_event_id"] == (
        "rank:2:raw_ordinal:0"
    )
    assert boundary_row["host_control_boundary_current_raw_event_id"] == (
        "rank:2:raw_ordinal:1"
    )
    assert boundary_row["host_control_boundary_family"] == (
        "cudaGetDevice -> cublasSetStream_v2"
    )
    assert boundary_row["host_control_boundary_selection_status"] == "selected_family"
    assert boundary_row["actual_counterpart_id"] == (
        "rank:2:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
    )
    assert boundary_row["actual_counterpart_status"] == (
        "actual_boundary_row_id_exported_selected_occurrence_join_not_attempted"
    )
    assert boundary_row["actual_rank"] == 2
    assert boundary_row["actual_raw_prev_event_id"] == "rank:2:raw_ordinal:0"
    assert boundary_row["actual_raw_current_event_id"] == "rank:2:raw_ordinal:1"
    assert boundary_row["actual_boundary_family"] == (
        "cudaGetDevice -> cublasSetStream_v2"
    )
    assert boundary_row["counterpart_join_key"] == (
        "rank:2:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
    )
    assert boundary_row["counterpart_join_method"] == (
        "actual_row_id_only_emulated_selected_occurrence_join_not_attempted"
    )
    assert boundary_row["counterpart_join_confidence"] == "unavailable"
    assert boundary_row["comparable_actual_context_only"] is True
    assert boundary_row["actual_inter_host_op_gap_us"] == (
        boundary_row["host_control_boundary_current_ts_us"]
        - boundary_row["host_control_boundary_prev_end_ts_us"]
    )
    assert boundary_row["host_control_visibility_split_status"] == "unavailable"
    assert "preserve_wrapper_host_timing" in boundary_row[
        "host_control_visibility_split_unavailable_reason"
    ]
    assert boundary_row["host_control_producer_visibility_schema_version"] == (
        "host_control_producer_visibility_nonoverlap_v1"
    )
    assert boundary_row["host_control_producer_visibility_status"] == (
        "structural_unavailable"
    )
    assert "preserve_start_time_end_time" in boundary_row[
        "host_control_producer_visibility_unavailable_reason"
    ]
    assert boundary_row["host_control_producer_visibility_basis"] == (
        "capture_real_default_off_structural_metadata_no_internal_wrapper_clocks"
    )
    assert boundary_row["host_control_producer_visibility_segments"]
    assert {
        segment["duration_us"]
        for segment in boundary_row["host_control_producer_visibility_segments"]
    } == {None}
    assert boundary_row["host_control_producer_numeric_split_status"] == "unavailable"
    assert "not_emitted_without_nonperturbing_brackets" in boundary_row[
        "host_control_producer_numeric_split_unavailable_reason"
    ]
    assert boundary_row["host_control_producer_nonoverlap_status"] == "unavailable"
    assert boundary_row["host_control_producer_wait_map_nonoverlap_status"] == "unavailable"
    assert boundary_row[
        "host_control_producer_double_counting_nonoverlap_status"
    ] == "unavailable"
    assert boundary_row["actual_counterpart_visibility_kind"] == "mixed_or_unresolved"
    assert boundary_row["actual_launch_visibility_kind"] == "mixed_or_unresolved"
    assert boundary_row["runtime_or_framework_duration_us"] is None
    assert boundary_row["payload_enrichment_duration_us"] is None
    assert boundary_row["trace_serialization_duration_us"] is None
    assert boundary_row["mis_materialized_duration_us"] is None
    assert boundary_row["split_sum_check_status"] == "unavailable"
    assert boundary_row["classification_unavailable_reason"] == (
        "mechanical_visibility_split_not_measured_to_preserve_wrapper_host_timing"
    )
    assert boundary_row["affected_interval_id"] is None
    assert boundary_row["count_once_status"] == "unavailable"
    assert "actual_launch_control_dispatch_us" not in boundary_row
    assert "actual_launch_api_body_us" not in boundary_row
    assert "actual_launch_instrumentation_only_us" not in boundary_row
    assert boundary_row["safe_to_use_as_subtraction_delta"] is False
    assert boundary_row["safe_to_use_as_repair_evidence"] is False


def test_write_rank_trace_gemm_adjacent_metadata_default_off_absence(
    tmp_path: Path,
    monkeypatch,
):
    for key in (
        "MAYA_ENABLE_GEMM_ADJACENT_HOSTDELAY_BOUNDARY_ACTUAL_COUNTERPART_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_GEMM_ADJACENT_HOSTDELAY_BOUNDARY_ACTUAL_COUNTERPART_DIAGNOSTICS",
        "MAYA_ENABLE_GEMM_HOSTDELAY_BOUNDARY_COUNTERPART_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_GEMM_HOSTDELAY_BOUNDARY_COUNTERPART_DIAGNOSTICS",
    ):
        monkeypatch.delenv(key, raising=False)

    set_stream = cpp_event.EventRecord()
    set_stream.kind = cpp_event.EventKind.RuntimeCall
    set_stream.api_name = "cublasSetStream_v2"
    set_stream.timestamp = timedelta(seconds=3)
    set_stream.end_timestamp = timedelta(seconds=3, microseconds=4)
    set_stream.host_duration = timedelta(microseconds=4)
    set_stream.process_id = 77
    set_stream.thread_id = 79
    set_stream.payload.attributes = {"stream_id": "5"}

    gemm = cpp_event.EventRecord()
    gemm.kind = cpp_event.EventKind.ComputeKernel
    gemm.api_name = "cublasGemmEx"
    gemm.timestamp = timedelta(seconds=3, microseconds=10)
    gemm.end_timestamp = timedelta(seconds=3, microseconds=16)
    gemm.host_duration = timedelta(microseconds=6)
    gemm.process_id = 77
    gemm.thread_id = 79
    gemm.payload.attributes = {
        "stream_id": "5",
        "m": "128",
        "n": "64",
        "k": "32",
        "algo": "99",
    }

    launch = cpp_event.EventRecord()
    launch.kind = cpp_event.EventKind.ComputeKernel
    launch.api_name = "cudaLaunchKernel"
    launch.timestamp = timedelta(seconds=3, microseconds=24)
    launch.end_timestamp = timedelta(seconds=3, microseconds=31)
    launch.host_duration = timedelta(microseconds=7)
    launch.process_id = 77
    launch.thread_id = 79
    launch.payload.attributes = {"stream_id": "5"}

    path = _write_rank_trace(tmp_path, [set_stream, gemm, launch])
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert [row["api"] for row in rows] == [
        "cublasSetStream_v2",
        "cublasGemmEx",
        "cudaLaunchKernel",
    ]
    assert [row["ts"] for row in rows] == [3_000_000, 3_000_010, 3_000_024]
    assert [row["end_ts"] for row in rows] == [3_000_004, 3_000_016, 3_000_031]
    assert [row["host_duration_us"] for row in rows] == [4.0, 6.0, 7.0]
    assert not any(
        key.startswith("gemm_adjacent_")
        for row in rows
        for key in row
    )


def test_write_rank_trace_opt_in_gemm_adjacent_boundary_metadata(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("FLEXSIM_HOST_MACHINE_ID", "host-gemm")
    monkeypatch.setenv(
        "MAYA_ENABLE_GEMM_HOSTDELAY_BOUNDARY_COUNTERPART_DIAGNOSTICS",
        "true",
    )

    set_stream = cpp_event.EventRecord()
    set_stream.kind = cpp_event.EventKind.RuntimeCall
    set_stream.api_name = "cublasSetStream_v2"
    set_stream.timestamp = timedelta(seconds=4)
    set_stream.end_timestamp = timedelta(seconds=4, microseconds=4)
    set_stream.host_duration = timedelta(microseconds=4)
    set_stream.process_id = 88
    set_stream.thread_id = 90
    set_stream.payload.attributes = {"stream_id": "7"}

    gemm = cpp_event.EventRecord()
    gemm.kind = cpp_event.EventKind.ComputeKernel
    gemm.api_name = "cublasGemmEx"
    gemm.timestamp = timedelta(seconds=4, microseconds=10)
    gemm.end_timestamp = timedelta(seconds=4, microseconds=16)
    gemm.host_duration = timedelta(microseconds=6)
    gemm.process_id = 88
    gemm.thread_id = 90
    gemm.payload.attributes = {
        "stream_id": "7",
        "m": "256",
        "n": "128",
        "k": "64",
        "computeType": "68",
        "transa": "1",
        "transb": "0",
        "algorithm": "23",
    }

    launch = cpp_event.EventRecord()
    launch.kind = cpp_event.EventKind.ComputeKernel
    launch.api_name = "cudaLaunchKernel"
    launch.timestamp = timedelta(seconds=4, microseconds=24)
    launch.end_timestamp = timedelta(seconds=4, microseconds=31)
    launch.host_duration = timedelta(microseconds=7)
    launch.process_id = 88
    launch.thread_id = 90
    launch.payload.attributes = {"stream_id": "7"}

    path = _write_rank_trace(tmp_path, [set_stream, gemm, launch])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    set_stream_row, gemm_row, launch_row = rows

    assert not any(key.startswith("gemm_adjacent_") for key in set_stream_row)
    assert gemm_row["gemm_adjacent_actual_counterpart_schema_version"] == (
        "gemm_adjacent_hostdelay_boundary_counterpart_visibility_count_once_metadata_v1"
    )
    assert gemm_row["gemm_adjacent_actual_counterpart_opt_in_flag"] is True
    assert gemm_row["gemm_adjacent_actual_raw_boundary_family_prev_to_current"] == (
        "cublasSetStream_v2 -> cublasGemmEx"
    )
    assert gemm_row["gemm_adjacent_actual_boundary_family_in_design_scope"] is True
    assert gemm_row["gemm_adjacent_target_gemm_api"] == "cublasGemmEx"
    assert gemm_row["gemm_adjacent_adjacent_api"] == "cublasSetStream_v2"
    assert gemm_row["gemm_adjacent_actual_endpoint_timestamps_used_as_strict_timing"] is False
    assert gemm_row["gemm_adjacent_actual_runtime_direct_substitution"] is False
    assert gemm_row["gemm_adjacent_count_once_status"] == "unavailable"
    assert gemm_row["gemm_adjacent_count_once_non_overlap_status"] == "unavailable"
    assert gemm_row["gemm_adjacent_wait_map_safety_status"] == "unavailable"
    assert gemm_row["gemm_adjacent_safe_to_use_as_repair_evidence"] is False
    assert gemm_row["gemm_adjacent_safe_to_use_as_subtraction_delta"] is False
    assert gemm_row["safe_to_use_as_repair_evidence"] is False
    assert gemm_row["safe_to_use_as_subtraction_delta"] is False
    assert gemm_row["host_control_boundary_family"] == "cublasSetStream_v2 -> cublasGemmEx"
    assert gemm_row["host_control_visibility_split_status"] == "unavailable"

    assert launch_row["gemm_adjacent_actual_raw_boundary_family_prev_to_current"] == (
        "cublasGemmEx -> cudaLaunchKernel"
    )
    assert launch_row["gemm_adjacent_actual_boundary_family_in_design_scope"] is True
    assert launch_row["gemm_adjacent_target_gemm_api"] == "cublasGemmEx"
    assert launch_row["gemm_adjacent_adjacent_api"] == "cudaLaunchKernel"
    assert launch_row["gemm_adjacent_actual_algorithm"] == "23"
    assert launch_row["gemm_adjacent_actual_gemm_shape_signature"] == (
        "m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23"
    )
    assert launch_row["host_duration_us"] == 7.0


def test_write_rank_trace_gemm_adjacent_opt_in_excludes_non_adjacent_gemm(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv(
        "MAYA_ENABLE_GEMM_HOSTDELAY_BOUNDARY_COUNTERPART_DIAGNOSTICS",
        "1",
    )

    allreduce = cpp_event.EventRecord()
    allreduce.kind = cpp_event.EventKind.AllReduce
    allreduce.api_name = "ncclAllReduce"
    allreduce.timestamp = timedelta(seconds=5)
    allreduce.end_timestamp = timedelta(seconds=5, microseconds=4)
    allreduce.host_duration = timedelta(microseconds=4)
    allreduce.process_id = 91
    allreduce.thread_id = 93

    gemm_a = cpp_event.EventRecord()
    gemm_a.kind = cpp_event.EventKind.ComputeKernel
    gemm_a.api_name = "cublasGemmEx"
    gemm_a.timestamp = timedelta(seconds=5, microseconds=10)
    gemm_a.end_timestamp = timedelta(seconds=5, microseconds=16)
    gemm_a.host_duration = timedelta(microseconds=6)
    gemm_a.process_id = 91
    gemm_a.thread_id = 93
    gemm_a.payload.attributes = {"m": "16", "n": "16", "k": "16"}

    gemm_b = cpp_event.EventRecord()
    gemm_b.kind = cpp_event.EventKind.ComputeKernel
    gemm_b.api_name = "cublasGemmStridedBatchedEx"
    gemm_b.timestamp = timedelta(seconds=5, microseconds=22)
    gemm_b.end_timestamp = timedelta(seconds=5, microseconds=30)
    gemm_b.host_duration = timedelta(microseconds=8)
    gemm_b.process_id = 91
    gemm_b.thread_id = 93
    gemm_b.payload.attributes = {
        "m": "16",
        "n": "16",
        "k": "16",
        "batch_count": "2",
    }

    path = _write_rank_trace(tmp_path, [allreduce, gemm_a, gemm_b])
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert [row["api"] for row in rows] == [
        "ncclAllReduce",
        "cublasGemmEx",
        "cublasGemmStridedBatchedEx",
    ]
    assert not any(
        key.startswith("gemm_adjacent_")
        for row in rows
        for key in row
    )


def test_write_rank_trace_cuda_gemm_hostdispatch_strict_occurrence_gap_default_off_absent(
    tmp_path: Path,
    monkeypatch,
):
    for key in (
        "MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
    ):
        monkeypatch.delenv(key, raising=False)

    launch = cpp_event.EventRecord()
    launch.kind = cpp_event.EventKind.ComputeKernel
    launch.api_name = "cudaLaunchKernel"
    launch.timestamp = timedelta(seconds=6)
    launch.end_timestamp = timedelta(seconds=6, microseconds=5)
    launch.host_duration = timedelta(microseconds=5)
    launch.process_id = 101
    launch.thread_id = 103
    launch.payload.attributes = {"stream_id": "0"}

    gemm = cpp_event.EventRecord()
    gemm.kind = cpp_event.EventKind.ComputeKernel
    gemm.api_name = "cublasGemmEx"
    gemm.timestamp = timedelta(seconds=6, microseconds=20)
    gemm.end_timestamp = timedelta(seconds=6, microseconds=31)
    gemm.host_duration = timedelta(microseconds=11)
    gemm.process_id = 101
    gemm.thread_id = 103
    gemm.payload.attributes = {
        "stream_id": "0",
        "m": "256",
        "n": "128",
        "k": "64",
        "computeType": "68",
        "transa": "1",
        "transb": "0",
        "algorithm": "23",
    }

    path = _write_rank_trace(tmp_path, [launch, gemm])
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert [row["api"] for row in rows] == ["cudaLaunchKernel", "cublasGemmEx"]
    assert [row["ts"] for row in rows] == [6_000_000, 6_000_020]
    assert [row["end_ts"] for row in rows] == [6_000_005, 6_000_031]
    assert [row["host_duration_us"] for row in rows] == [5.0, 11.0]
    forbidden_fields = {
        "cuda_gemm_hostdispatch_strict_occurrence_gap_schema_version",
        "cuda_gemm_hostdispatch_strict_occurrence_gap_opt_in_flag",
        "cuda_gemm_hostdispatch_strict_occurrence_gap_source_side",
        "strict_occurrence_common_basis_key",
        "actual_mechanical_dispatch_split_status",
        "actual_endpoint_timestamps_used_as_dispatch_split",
        "actual_host_duration_used_as_dispatch_split",
        "actual_runtime_used_as_dispatch_split",
        "repair_ready",
        "safe_to_use_as_repair_evidence",
        "safe_to_use_as_subtraction_delta",
    }
    assert not any(field in row for row in rows for field in forbidden_fields)


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
    ],
)
def test_write_rank_trace_cuda_gemm_hostdispatch_strict_occurrence_gap_opt_in(
    tmp_path: Path,
    monkeypatch,
    env_key,
):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("FLEXSIM_HOST_MACHINE_ID", "host-strict")
    monkeypatch.setenv(env_key, "true")
    other_key = (
        "FLEXSIM_MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS"
        if env_key
        == "MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS"
        else "MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS"
    )
    monkeypatch.delenv(other_key, raising=False)

    launch = cpp_event.EventRecord()
    launch.kind = cpp_event.EventKind.ComputeKernel
    launch.api_name = "cudaLaunchKernel"
    launch.timestamp = timedelta(seconds=7)
    launch.end_timestamp = timedelta(seconds=7, microseconds=5)
    launch.host_duration = timedelta(microseconds=5)
    launch.process_id = 105
    launch.thread_id = 107
    launch.payload.attributes = {"stream_id": "2"}

    gemm = cpp_event.EventRecord()
    gemm.kind = cpp_event.EventKind.ComputeKernel
    gemm.api_name = "cublasGemmEx"
    gemm.timestamp = timedelta(seconds=7, microseconds=20)
    gemm.end_timestamp = timedelta(seconds=7, microseconds=31)
    gemm.host_duration = timedelta(microseconds=11)
    gemm.process_id = 105
    gemm.thread_id = 107
    gemm.payload.attributes = {
        "stream_id": "2",
        "m": "256",
        "n": "128",
        "k": "64",
        "computeType": "68",
        "transa": "1",
        "transb": "0",
        "algorithm": "23",
    }

    path = _write_rank_trace(tmp_path, [launch, gemm])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    launch_row, gemm_row = rows

    assert [row["ts"] for row in rows] == [7_000_000, 7_000_020]
    assert [row["end_ts"] for row in rows] == [7_000_005, 7_000_031]
    assert [row["host_duration_us"] for row in rows] == [5.0, 11.0]

    for row, api, api_seq, host_queue_seq, stream_seq in (
        (launch_row, "cudaLaunchKernel", 0, 0, 0),
        (gemm_row, "cublasGemmEx", 0, 1, 1),
    ):
        assert row["cuda_gemm_hostdispatch_strict_occurrence_gap_schema_version"] == (
            "cudaLaunch_GEMM_hostdispatch_strict_occurrence_gap_metadata_v1"
        )
        assert row["cuda_gemm_hostdispatch_strict_occurrence_gap_opt_in_flag"] is True
        assert row["cuda_gemm_hostdispatch_strict_occurrence_gap_source_side"] == (
            "actual_endpoint_metadata"
        )
        assert row["strict_occurrence_count_basis_side"] == "actual_endpoint_row"
        assert row["api_family"] == api
        assert row["component_role"] == "actual_mechanical_dispatch_split_candidate"
        assert row["api_sequence_ordinal_in_window"] == api_seq
        assert row["host_queue_sequence_ordinal_in_window"] == host_queue_seq
        assert row["stream_sequence_ordinal_in_window"] == stream_seq
        assert row["host_dispatch_queue_id"] == "host-strict:rank:3"
        assert row["actual_stream_resource_id"] == "rank:3:stream:2"
        assert row["stream_alignment_status"] == (
            "actual_only_unresolved_predicted_namespace_not_joined"
        )
        assert row["actual_mechanical_dispatch_split_status"] == "unavailable"
        assert row["actual_control_dispatch_us"] is None
        assert row["actual_api_body_us"] is None
        assert row["actual_instrumentation_only_us"] is None
        assert row["actual_endpoint_timestamps_used_as_dispatch_split"] is False
        assert row["actual_host_duration_used_as_dispatch_split"] is False
        assert row["actual_runtime_used_as_dispatch_split"] is False
        assert row["strict_occurrence_join_ready"] is False
        assert row["strict_actual_timing_or_mechanical_split_ready"] is False
        assert row["strict_apples_to_apples_delta_ready"] is False
        assert row["repair_ready"] is False
        assert row["safe_to_use_as_repair_evidence"] is False
        assert row["safe_to_use_as_subtraction_delta"] is False
        assert row["safe_to_use_for_runtime_substitution"] is False
        assert row["safe_to_use_for_endpoint_timestamp_substitution"] is False

    assert launch_row["actual_endpoint_ts_us"] == 7_000_000
    assert launch_row["actual_endpoint_end_ts_us"] == 7_000_005
    assert launch_row["actual_endpoint_host_duration_us"] == 5.0
    assert launch_row["actual_start_us"] is None
    assert launch_row["actual_end_us"] is None
    assert launch_row["actual_duration_us"] is None
    assert "api:cudaLaunchKernel" in launch_row["strict_occurrence_common_basis_key"]
    assert "window:" not in launch_row["strict_occurrence_common_basis_key"]
    assert "role:" not in launch_row["strict_occurrence_common_basis_key"]
    assert launch_row["strict_occurrence_boundary_target_side"] == "cudaLaunchKernel"
    assert launch_row["strict_occurrence_projection_keys_status"] == (
        "diagnostic_only_projection_not_strict_join_key"
    )
    assert launch_row["strict_occurrence_projection_keys_repair_ready"] is False
    assert launch_row["strict_occurrence_endpoint_identity_projection_key"] == (
        "rank:3|queue:host-strict:rank:3|api:cudaLaunchKernel|api_seq:0|"
        "material:unavailable|algorithm:unavailable"
    )
    assert launch_row["strict_occurrence_boundary_target_side_projection_key"] == (
        "rank:3|queue:host-strict:rank:3|api:cudaLaunchKernel|api_seq:0|"
        "material:unavailable|algorithm:unavailable|"
        "boundary_target_side:cudaLaunchKernel"
    )

    assert gemm_row["material_signature"] == "m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23"
    assert gemm_row["strict_occurrence_material_without_embedded_algo"] == (
        "m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23"
    )
    assert gemm_row["algorithm"] == "23"
    assert gemm_row["gemm_shape_signature"] == "m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23"
    assert gemm_row["actual_count_once_group_id"] == (
        "actual_endpoint:rank:3:raw_ordinal:1"
    )
    assert gemm_row["count_once_status"] == (
        "metadata_only_count_once_group_not_strict_nonoverlap_proof"
    )
    assert "api:cublasGemmEx" in gemm_row["strict_occurrence_common_basis_key"]
    assert "algorithm:23" in gemm_row["strict_occurrence_common_basis_key"]
    assert "boundary:cudaLaunchKernel -> cublasGemmEx" in gemm_row[
        "strict_occurrence_common_basis_key"
    ]
    assert gemm_row["strict_occurrence_boundary_target_side"] == "target_to_target"
    assert gemm_row["strict_occurrence_endpoint_identity_projection_key"] == (
        "rank:3|queue:host-strict:rank:3|api:cublasGemmEx|api_seq:0|"
        "material:m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23|algorithm:23"
    )
    assert gemm_row["strict_occurrence_boundary_target_side_projection_key"] == (
        "rank:3|queue:host-strict:rank:3|api:cublasGemmEx|api_seq:0|"
        "material:m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23|algorithm:23|"
        "boundary_target_side:target_to_target"
    )

    collated = collate_trace_bundle(load_trace_directory(tmp_path))
    predicted_host_delay = [
        event
        for event in collated.rank_events[3]
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cudaLaunchKernel"
        and event.extras.get("raw_current_api") == "cublasGemmEx"
    ][0]
    assert predicted_host_delay.extras["component_role"] == "hostDelay"
    assert gemm_row["component_role"] == "actual_mechanical_dispatch_split_candidate"
    assert predicted_host_delay.extras["paper_valid_window_id"] is None
    assert predicted_host_delay.extras["strict_occurrence_common_basis_key"] == (
        gemm_row["strict_occurrence_common_basis_key"]
    )
    assert predicted_host_delay.extras[
        "strict_occurrence_endpoint_identity_projection_key"
    ] == gemm_row["strict_occurrence_endpoint_identity_projection_key"]
    assert predicted_host_delay.extras[
        "strict_occurrence_boundary_target_side_projection_key"
    ] == gemm_row["strict_occurrence_boundary_target_side_projection_key"]


def test_write_rank_trace_cuda_gemm_hostdispatch_uses_semantic_predecessor_boundary(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("FLEXSIM_HOST_MACHINE_ID", "host-strict")
    monkeypatch.setenv(
        "MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
        "1",
    )

    first_launch = cpp_event.EventRecord()
    first_launch.kind = cpp_event.EventKind.ComputeKernel
    first_launch.api_name = "cudaLaunchKernel"
    first_launch.timestamp = timedelta(seconds=8)
    first_launch.end_timestamp = timedelta(seconds=8, microseconds=5)
    first_launch.host_duration = timedelta(microseconds=5)
    first_launch.process_id = 105
    first_launch.thread_id = 107
    first_launch.payload.attributes = {"stream_id": "2"}

    get_device = cpp_event.EventRecord()
    get_device.kind = cpp_event.EventKind.RuntimeCall
    get_device.api_name = "cudaGetDevice"
    get_device.timestamp = timedelta(seconds=8, microseconds=12)
    get_device.end_timestamp = timedelta(seconds=8, microseconds=13)
    get_device.host_duration = timedelta(microseconds=1)
    get_device.process_id = 105
    get_device.thread_id = 107

    second_launch = cpp_event.EventRecord()
    second_launch.kind = cpp_event.EventKind.ComputeKernel
    second_launch.api_name = "cudaLaunchKernel"
    second_launch.timestamp = timedelta(seconds=8, microseconds=30)
    second_launch.end_timestamp = timedelta(seconds=8, microseconds=35)
    second_launch.host_duration = timedelta(microseconds=5)
    second_launch.process_id = 105
    second_launch.thread_id = 107
    second_launch.payload.attributes = {"stream_id": "2"}

    path = _write_rank_trace(tmp_path, [first_launch, get_device, second_launch])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _, control_row, second_launch_row = rows

    assert control_row["api"] == "cudaGetDevice"
    assert second_launch_row["api"] == "cudaLaunchKernel"
    assert second_launch_row["ts"] == 8_000_030
    assert second_launch_row["end_ts"] == 8_000_035
    assert second_launch_row["host_duration_us"] == 5.0
    assert second_launch_row["actual_raw_immediate_boundary_family"] == (
        "cudaGetDevice -> cudaLaunchKernel"
    )
    assert second_launch_row["actual_semantic_predecessor_boundary_family"] == (
        "cudaLaunchKernel -> cudaLaunchKernel"
    )
    assert second_launch_row["actual_boundary_namespace_basis"] == (
        "semantic_predecessor_control_query_filtered"
    )
    assert "boundary:cudaLaunchKernel -> cudaLaunchKernel" in second_launch_row[
        "strict_occurrence_common_basis_key"
    ]
    assert second_launch_row["actual_endpoint_timestamps_used_as_strict_timing"] is False
    assert second_launch_row["actual_host_duration_used_as_strict_timing"] is False
    assert second_launch_row["actual_runtime_direct_substitution"] is False
    assert second_launch_row["safe_to_use_as_repair_evidence"] is False


def test_write_rank_trace_cuda_gemm_hostdispatch_strict_occurrence_preserves_zero_algorithm(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("FLEXSIM_HOST_MACHINE_ID", "host-strict")
    monkeypatch.setenv(
        "MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
        "1",
    )

    gemm = cpp_event.EventRecord()
    gemm.kind = cpp_event.EventKind.ComputeKernel
    gemm.api_name = "cublasGemmEx"
    gemm.timestamp = timedelta(seconds=8)
    gemm.end_timestamp = timedelta(seconds=8, microseconds=11)
    gemm.host_duration = timedelta(microseconds=11)
    gemm.process_id = 105
    gemm.thread_id = 107
    gemm.payload.attributes = {
        "stream_id": "2",
        "m": "256",
        "n": "128",
        "k": "64",
        "computeType": "68",
        "transa": "1",
        "transb": "0",
        "algorithm": 0,
    }

    path = _write_rank_trace(tmp_path, [gemm])
    row = json.loads(path.read_text().splitlines()[0])

    assert row["material_signature"] == (
        "m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=0"
    )
    assert row["algorithm"] == 0
    assert "algorithm:0" in row["strict_occurrence_common_basis_key"]
    assert "algorithm:unavailable" not in row["strict_occurrence_common_basis_key"]
    assert row["strict_occurrence_endpoint_identity_projection_key"] == (
        "rank:3|queue:host-strict:rank:3|api:cublasGemmEx|api_seq:0|"
        "material:m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=0|algorithm:0"
    )


def test_write_rank_trace_joined_gemm_stream_queue_wait_actual_metadata_default_off_absent(
    tmp_path: Path,
    monkeypatch,
):
    for key in (
        "MAYA_ENABLE_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_METADATA_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_METADATA_DIAGNOSTICS",
    ):
        monkeypatch.delenv(key, raising=False)

    gemm = cpp_event.EventRecord()
    gemm.kind = cpp_event.EventKind.ComputeKernel
    gemm.api_name = "cublasGemmEx"
    gemm.timestamp = timedelta(seconds=9)
    gemm.end_timestamp = timedelta(seconds=9, microseconds=11)
    gemm.host_duration = timedelta(microseconds=11)
    gemm.process_id = 105
    gemm.thread_id = 107
    gemm.payload.attributes = {
        "stream_id": "2",
        "m": "256",
        "n": "128",
        "k": "64",
        "computeType": "68",
        "transa": "1",
        "transb": "0",
        "algorithm": "23",
    }

    path = _write_rank_trace(tmp_path, [gemm])
    row = json.loads(path.read_text().strip())

    assert not any(
        key.startswith("joined_gemm_stream_queue_wait_") for key in row
    )
    assert row["ts"] == 9_000_000
    assert row["end_ts"] == 9_000_011
    assert row["host_duration_us"] == 11.0


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_METADATA_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_METADATA_DIAGNOSTICS",
    ],
)
def test_write_rank_trace_joined_gemm_stream_queue_wait_actual_metadata_opt_in(
    tmp_path: Path,
    monkeypatch,
    env_key,
):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("FLEXSIM_HOST_MACHINE_ID", "host-stream-wait")
    monkeypatch.setenv(env_key, "1")
    other_key = (
        "FLEXSIM_MAYA_ENABLE_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_METADATA_DIAGNOSTICS"
        if env_key
        == "MAYA_ENABLE_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_METADATA_DIAGNOSTICS"
        else "MAYA_ENABLE_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_METADATA_DIAGNOSTICS"
    )
    monkeypatch.delenv(other_key, raising=False)

    set_stream = cpp_event.EventRecord()
    set_stream.kind = cpp_event.EventKind.RuntimeCall
    set_stream.api_name = "cublasSetStream_v2"
    set_stream.timestamp = timedelta(seconds=9)
    set_stream.end_timestamp = timedelta(seconds=9, microseconds=4)
    set_stream.host_duration = timedelta(microseconds=4)
    set_stream.process_id = 105
    set_stream.thread_id = 107
    set_stream.payload.attributes = {"stream_id": "2"}

    launch = cpp_event.EventRecord()
    launch.kind = cpp_event.EventKind.ComputeKernel
    launch.api_name = "cudaLaunchKernel"
    launch.timestamp = timedelta(seconds=9, microseconds=10)
    launch.end_timestamp = timedelta(seconds=9, microseconds=15)
    launch.host_duration = timedelta(microseconds=5)
    launch.process_id = 105
    launch.thread_id = 107
    launch.payload.attributes = {
        "stream_id": "2",
        "kernel": "void fused_kernel",
        "grid_x": "1",
        "grid_y": "2",
        "grid_z": "3",
        "block_x": "4",
        "block_y": "5",
        "block_z": "6",
        "shared_mem": "0",
    }

    gemm = cpp_event.EventRecord()
    gemm.kind = cpp_event.EventKind.ComputeKernel
    gemm.api_name = "cublasGemmEx"
    gemm.timestamp = timedelta(seconds=9, microseconds=30)
    gemm.end_timestamp = timedelta(seconds=9, microseconds=41)
    gemm.host_duration = timedelta(microseconds=11)
    gemm.process_id = 105
    gemm.thread_id = 107
    gemm.payload.attributes = {
        "stream_id": "2",
        "m": "256",
        "n": "128",
        "k": "64",
        "computeType": "68",
        "transa": "1",
        "transb": "0",
        "algorithm": "23",
    }

    strided_gemm = cpp_event.EventRecord()
    strided_gemm.kind = cpp_event.EventKind.ComputeKernel
    strided_gemm.api_name = "cublasGemmStridedBatchedEx"
    strided_gemm.timestamp = timedelta(seconds=9, microseconds=50)
    strided_gemm.end_timestamp = timedelta(seconds=9, microseconds=65)
    strided_gemm.host_duration = timedelta(microseconds=15)
    strided_gemm.process_id = 105
    strided_gemm.thread_id = 107
    strided_gemm.payload.attributes = {
        "stream_id": "2",
        "m": "512",
        "n": "64",
        "k": "32",
        "batchCount": "8",
        "strideA": "16384",
        "strideB": "2048",
        "strideC": "32768",
        "computeType": "68",
        "transa": "0",
        "transb": "1",
        "algo": "24",
    }

    path = _write_rank_trace(tmp_path, [set_stream, launch, gemm, strided_gemm])
    set_stream_row, launch_row, gemm_row, strided_gemm_row = [
        json.loads(line) for line in path.read_text().splitlines()
    ]

    assert not any(
        key.startswith("joined_gemm_stream_queue_wait_") for key in set_stream_row
    )
    for row, api, stream_seq, prev_api, prev_raw_event_id in (
        (launch_row, "cudaLaunchKernel", 1, "cublasSetStream_v2", "rank:3:raw_ordinal:0"),
        (gemm_row, "cublasGemmEx", 2, "cudaLaunchKernel", "rank:3:raw_ordinal:1"),
        (
            strided_gemm_row,
            "cublasGemmStridedBatchedEx",
            3,
            "cublasGemmEx",
            "rank:3:raw_ordinal:2",
        ),
    ):
        assert row[
            "joined_gemm_stream_queue_wait_actual_counterpart_schema_version"
        ] == "joined_gemm_stream_queue_wait_actual_counterpart_metadata_v1"
        assert row["joined_gemm_stream_queue_wait_actual_counterpart_opt_in_flag"] is True
        assert row["joined_gemm_stream_queue_wait_source_side"] == (
            "actual_stream_order_endpoint_metadata"
        )
        assert row["joined_gemm_stream_queue_wait_actual_api"] == api
        assert row["joined_gemm_stream_queue_wait_actual_stream_id"] == "2"
        assert row["joined_gemm_stream_queue_wait_actual_stream_resource_id"] == (
            "rank:3:stream:2"
        )
        assert row[
            "joined_gemm_stream_queue_wait_actual_stream_sequence_ordinal"
        ] == stream_seq
        assert row[
            "joined_gemm_stream_queue_wait_previous_same_stream_raw_event_id"
        ] == prev_raw_event_id
        assert row["joined_gemm_stream_queue_wait_previous_same_stream_api"] == prev_api
        assert row[
            "joined_gemm_stream_queue_wait_actual_stream_order_pair_id"
        ].endswith(f"previous:{prev_raw_event_id}->current:{row['joined_gemm_stream_queue_wait_actual_raw_event_id']}")
        assert row[
            "joined_gemm_stream_queue_wait_actual_release_timing_status"
        ] == "unavailable"
        assert row[
            "joined_gemm_stream_queue_wait_actual_wait_timing_status"
        ] == "unavailable"
        assert row["joined_gemm_stream_queue_wait_actual_wait_start_us"] is None
        assert row["joined_gemm_stream_queue_wait_actual_release_us"] is None
        assert row["joined_gemm_stream_queue_wait_actual_waited_us"] is None
        assert row[
            "joined_gemm_stream_queue_wait_endpoint_timestamps_used_as_wait_release"
        ] is False
        assert row[
            "joined_gemm_stream_queue_wait_endpoint_timestamps_used_as_strict_delta"
        ] is False
        assert row[
            "joined_gemm_stream_queue_wait_actual_runtime_direct_substitution"
        ] is False
        assert row[
            "joined_gemm_stream_queue_wait_actual_observed_runtime_used_as_prediction"
        ] is False
        assert row[
            "joined_gemm_stream_queue_wait_strict_actual_timing_available"
        ] is False
        assert row["joined_gemm_stream_queue_wait_strict_delta_calculable"] is False
        assert row["joined_gemm_stream_queue_wait_count_once_status"] == "unavailable"
        assert row[
            "joined_gemm_stream_queue_wait_count_once_nonoverlap_status"
        ] == "unavailable"
        assert row[
            "joined_gemm_stream_queue_wait_wait_map_safety_status"
        ] == "unavailable"
        assert row["joined_gemm_stream_queue_wait_wait_map_safety_proven"] is False
        assert row["joined_gemm_stream_queue_wait_repair_ready"] is False
        assert row[
            "joined_gemm_stream_queue_wait_safe_to_use_as_repair_evidence"
        ] is False
        assert row[
            "joined_gemm_stream_queue_wait_safe_to_use_as_subtraction_delta"
        ] is False
        assert row[
            "joined_gemm_stream_queue_wait_safe_to_use_for_runtime_substitution"
        ] is False
        assert row[
            "joined_gemm_stream_queue_wait_safe_to_use_for_endpoint_timestamp_substitution"
        ] is False

    assert gemm_row["joined_gemm_stream_queue_wait_actual_material_signature"] == (
        "m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23"
    )
    assert gemm_row["joined_gemm_stream_queue_wait_actual_gemm_shape_signature"] == (
        "m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23"
    )
    assert gemm_row["joined_gemm_stream_queue_wait_actual_algorithm"] == "23"
    assert (
        "kernel=void fused_kernel"
        in gemm_row[
            "joined_gemm_stream_queue_wait_previous_same_stream_material_signature"
        ]
    )
    assert gemm_row[
        "joined_gemm_stream_queue_wait_previous_same_stream_algorithm"
    ] is None
    assert strided_gemm_row[
        "joined_gemm_stream_queue_wait_actual_material_signature"
    ] == (
        "m=512;n=64;k=32;batch_count=8;stride_a=16384;stride_b=2048;"
        "stride_c=32768;compute_type=68;transa=0;transb=1;algorithm=24"
    )
    assert strided_gemm_row[
        "joined_gemm_stream_queue_wait_actual_gemm_shape_signature"
    ] == (
        "m=512;n=64;k=32;batch_count=8;stride_a=16384;stride_b=2048;"
        "stride_c=32768;compute_type=68;transa=0;transb=1;algorithm=24"
    )
    assert strided_gemm_row["joined_gemm_stream_queue_wait_actual_algorithm"] == "24"
    assert strided_gemm_row[
        "joined_gemm_stream_queue_wait_previous_same_stream_material_signature"
    ] == (
        "m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23"
    )
    assert strided_gemm_row[
        "joined_gemm_stream_queue_wait_previous_same_stream_algorithm"
    ] == "23"


def test_write_rank_trace_joined_gemm_stream_queue_wait_exports_cupti_stream_order_metadata(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv(
        "MAYA_ENABLE_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_METADATA_DIAGNOSTICS",
        "1",
    )

    launch = cpp_event.EventRecord()
    launch.kind = cpp_event.EventKind.ComputeKernel
    launch.api_name = "cudaLaunchKernel"
    launch.timestamp = timedelta(seconds=9, microseconds=10)
    launch.end_timestamp = timedelta(seconds=9, microseconds=15)
    launch.host_duration = timedelta(microseconds=5)
    launch.process_id = 105
    launch.thread_id = 107
    launch.payload.attributes = {
        "stream_id": "2",
        "kernel": "void fused_kernel",
        "grid_x": "1",
        "grid_y": "1",
        "grid_z": "1",
        "block_x": "64",
        "block_y": "1",
        "block_z": "1",
        "shared_mem": "0",
        "cupti_activity_first_kernel_start": "900000",
        "cupti_activity_first_kernel_end": "900900",
        "cupti_activity_last_kernel_start": "900910",
        "cupti_activity_last_kernel_end": "901000",
        "cupti_activity_first_kernel_stream_id": "17",
        "cupti_activity_last_kernel_stream_id": "17",
        "cupti_activity_common_clock_status": "unreviewed",
    }

    set_stream = cpp_event.EventRecord()
    set_stream.kind = cpp_event.EventKind.RuntimeCall
    set_stream.api_name = "cublasSetStream_v2"
    set_stream.timestamp = timedelta(seconds=9, microseconds=20)
    set_stream.end_timestamp = timedelta(seconds=9, microseconds=22)
    set_stream.host_duration = timedelta(microseconds=2)
    set_stream.process_id = 105
    set_stream.thread_id = 107
    set_stream.payload.attributes = {"stream_id": "2"}

    gemm = cpp_event.EventRecord()
    gemm.kind = cpp_event.EventKind.ComputeKernel
    gemm.api_name = "cublasGemmEx"
    gemm.timestamp = timedelta(seconds=9, microseconds=30)
    gemm.end_timestamp = timedelta(seconds=9, microseconds=41)
    gemm.host_duration = timedelta(microseconds=11)
    gemm.process_id = 105
    gemm.thread_id = 107
    gemm.payload.attributes = {
        "stream_id": "2",
        "m": "256",
        "n": "128",
        "k": "64",
        "computeType": "68",
        "transa": "1",
        "transb": "0",
        "algorithm": "23",
        "cupti_activity_first_kernel_start": "901016",
        "cupti_activity_first_kernel_end": "901900",
        "cupti_activity_last_kernel_start": "901920",
        "cupti_activity_last_kernel_end": "902000",
        "cupti_activity_first_kernel_stream_id": "17",
        "cupti_activity_last_kernel_stream_id": "17",
        "cupti_activity_common_clock_status": "unreviewed",
    }

    path = _write_rank_trace(tmp_path, [launch, set_stream, gemm])
    _launch_row, set_stream_row, gemm_row = [
        json.loads(line) for line in path.read_text().splitlines()
    ]

    assert not any(
        key.startswith("joined_gemm_stream_queue_wait_") for key in set_stream_row
    )
    assert gemm_row["joined_gemm_stream_queue_wait_source_side"] == (
        "actual_stream_order_cupti_activity_metadata"
    )
    assert gemm_row[
        "joined_gemm_stream_queue_wait_actual_release_timing_status"
    ] == "unavailable_missing_previous_same_stream_cupti_kernel_end"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_actual_wait_timing_status"
    ] == "partial_available_cupti_current_kernel_start_no_enqueue_wait_start"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_actual_device_timing_source"
    ] == "cupti_activity_concurrent_kernel"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_actual_previous_kernel_end_cupti_timestamp"
    ] is None
    assert gemm_row[
        "joined_gemm_stream_queue_wait_actual_previous_kernel_start_cupti_timestamp"
    ] is None
    assert gemm_row[
        "joined_gemm_stream_queue_wait_previous_same_stream_raw_event_id"
    ] == "rank:3:raw_ordinal:1"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_previous_same_stream_api"
    ] == "cublasSetStream_v2"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_previous_device_predecessor_raw_event_id"
    ] == "rank:3:raw_ordinal:0"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_previous_device_predecessor_api"
    ] == "cudaLaunchKernel"
    assert (
        "kernel=void fused_kernel"
        in gemm_row[
            "joined_gemm_stream_queue_wait_previous_device_predecessor_material_signature"
        ]
    )
    assert gemm_row[
        "joined_gemm_stream_queue_wait_previous_device_predecessor_status"
    ] == "available_previous_same_stream_device_predecessor_gap_unreviewed_clock"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_previous_device_predecessor_source"
    ] == "rank_local_previous_same_stream_cupti_backed_device_predecessor"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_previous_device_predecessor_kernel_start_cupti_timestamp"
    ] == "900910"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_previous_device_predecessor_kernel_end_cupti_timestamp"
    ] == "901000"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_previous_device_predecessor_cupti_kernel_stream_id"
    ] == "17"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_previous_device_predecessor_cupti_kernel_stream_id_pair_status"
    ] == "same_cupti_stream_id_observed"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_previous_device_predecessor_stream_order_gap_cupti_ticks"
    ] == 16
    assert gemm_row[
        "joined_gemm_stream_queue_wait_actual_current_kernel_start_cupti_timestamp"
    ] == "901016"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_actual_current_kernel_end_cupti_timestamp"
    ] == "902000"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_actual_current_cupti_kernel_stream_id"
    ] == "17"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_actual_previous_cupti_kernel_stream_id"
    ] is None
    assert gemm_row[
        "joined_gemm_stream_queue_wait_actual_cupti_kernel_stream_id_pair_status"
    ] == "unavailable"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_actual_cupti_common_clock_status"
    ] == "unreviewed"
    assert gemm_row[
        "joined_gemm_stream_queue_wait_actual_stream_order_gap_cupti_ticks"
    ] is None
    assert gemm_row["joined_gemm_stream_queue_wait_actual_wait_start_us"] is None
    assert gemm_row["joined_gemm_stream_queue_wait_actual_release_us"] is None
    assert gemm_row["joined_gemm_stream_queue_wait_actual_waited_us"] is None
    assert gemm_row[
        "joined_gemm_stream_queue_wait_strict_actual_timing_available"
    ] is False
    assert gemm_row["joined_gemm_stream_queue_wait_strict_delta_calculable"] is False
    assert gemm_row[
        "joined_gemm_stream_queue_wait_actual_runtime_direct_substitution"
    ] is False
    assert gemm_row[
        "joined_gemm_stream_queue_wait_safe_to_use_for_runtime_substitution"
    ] is False


def test_write_rank_trace_opt_in_launch_neighborhood_equivalence_metadata(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("FLEXSIM_HOST_MACHINE_ID", "host-b")
    monkeypatch.setenv("MAYA_ENABLE_LAUNCH_NEIGHBORHOOD_EQUIVALENCE_DIAGNOSTICS", "1")

    launch = cpp_event.EventRecord()
    launch.kind = cpp_event.EventKind.RuntimeCall
    launch.api_name = "cudaLaunchKernel"
    launch.timestamp = timedelta(seconds=2)
    launch.end_timestamp = timedelta(seconds=2, microseconds=5)
    launch.host_duration = timedelta(microseconds=5)
    launch.process_id = 55
    launch.thread_id = 66
    launch.payload.attributes = {"stream_id": "11"}

    get_last_error = cpp_event.EventRecord()
    get_last_error.kind = cpp_event.EventKind.RuntimeCall
    get_last_error.api_name = "cudaGetLastError"
    get_last_error.timestamp = timedelta(seconds=2, microseconds=12)
    get_last_error.end_timestamp = timedelta(seconds=2, microseconds=13)
    get_last_error.host_duration = timedelta(microseconds=1)
    get_last_error.process_id = 55
    get_last_error.thread_id = 66

    path = _write_rank_trace(tmp_path, [launch, get_last_error])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _, boundary_row = rows

    assert boundary_row["api"] == "cudaGetLastError"
    assert boundary_row["launch_neighborhood_equivalence_schema_version"] == (
        "launch_neighborhood_occurrence_equivalence_diagnostics_v1"
    )
    assert boundary_row["launch_neighborhood_equivalence_opt_in_flag"] is True
    assert boundary_row["launch_neighborhood_occurrence_id"] == (
        "rank:3:launch_neighborhood:raw_ordinal:0->raw_ordinal:1"
    )
    assert boundary_row["launch_neighborhood_role"] == (
        "actual_wrapper_control_interleaved_neighborhood"
    )
    assert boundary_row["launch_neighborhood_normalized_signature"] == (
        "paper_visible_operation_boundary -> unresolved_wrapper_control_cpu_work"
    )
    assert boundary_row["launch_neighborhood_prev_raw_event_id"] == (
        "rank:3:raw_ordinal:0"
    )
    assert boundary_row["launch_neighborhood_current_raw_event_id"] == (
        "rank:3:raw_ordinal:1"
    )
    assert boundary_row["launch_neighborhood_prev_api"] == "cudaLaunchKernel"
    assert boundary_row["launch_neighborhood_current_api"] == "cudaGetLastError"
    assert boundary_row["launch_neighborhood_boundary_exclusion_reasons"] == [
        "prev:cudaLaunchKernel:paper_visible_operation_boundary",
        "current:cudaGetLastError:wrapper_control_visibility_unresolved",
    ]
    assert boundary_row["launch_neighborhood_host_dispatch_queue_id"] == (
        "host-b:rank:3"
    )
    assert boundary_row["launch_neighborhood_stream_id"] is None
    assert boundary_row["launch_neighborhood_wait_map_nonoverlap_status"] == (
        "unavailable"
    )
    assert boundary_row["launch_neighborhood_safe_to_use_as_repair_evidence"] is False
    assert boundary_row["host_control_envelope_counterpart_opt_in_flag"] is True


def test_write_rank_trace_opt_in_host_control_records_cuda_pop_blocker_without_wrapper_coverage(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("MAYA_ENABLE_HOST_CONTROL_BOUNDARY_COUNTERPART_DIAGNOSTICS", "1")

    launch = cpp_event.EventRecord()
    launch.kind = cpp_event.EventKind.RuntimeCall
    launch.api_name = "cudaLaunchKernel"
    launch.timestamp = timedelta(seconds=1)
    launch.end_timestamp = timedelta(seconds=1, microseconds=11)
    launch.host_duration = timedelta(microseconds=11)
    launch.process_id = 33
    launch.thread_id = 44

    path = _write_rank_trace(tmp_path, [launch])
    row = json.loads(path.read_text().strip())

    assert row["api"] == "cudaLaunchKernel"
    assert row["host_control_producer_visibility_status"] == "structural_unavailable"
    assert row["host_control_producer_numeric_split_status"] == "unavailable"
    assert row["host_control_compat_launch_pop_coverage_status"] == (
        "unavailable_not_exported_by_current_real_wrapper_producer"
    )
    assert "__cudaPopCallConfiguration_interposition_not_proven" in row[
        "host_control_compat_launch_pop_coverage_unavailable_reason"
    ]
    assert row["safe_to_use_as_repair_evidence"] is False
    assert row["safe_to_use_as_subtraction_delta"] is False


def test_write_rank_trace_host_control_metadata_survives_load_and_collate(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("FLEXSIM_HOST_MACHINE_ID", "host-a")
    monkeypatch.setenv("MAYA_ENABLE_HOST_CONTROL_BOUNDARY_COUNTERPART_DIAGNOSTICS", "1")

    get_device = cpp_event.EventRecord()
    get_device.kind = cpp_event.EventKind.RuntimeCall
    get_device.api_name = "cudaGetDevice"
    get_device.timestamp = timedelta(seconds=1)
    get_device.end_timestamp = timedelta(seconds=1, microseconds=3)
    get_device.host_duration = timedelta(microseconds=3)
    get_device.process_id = 33
    get_device.thread_id = 44

    launch = cpp_event.EventRecord()
    launch.kind = cpp_event.EventKind.RuntimeCall
    launch.api_name = "cudaLaunchKernel"
    launch.timestamp = timedelta(seconds=1, microseconds=9)
    launch.end_timestamp = timedelta(seconds=1, microseconds=14)
    launch.host_duration = timedelta(microseconds=5)
    launch.process_id = 33
    launch.thread_id = 44
    launch.payload.attributes = {"stream_id": "7"}

    _write_rank_trace(tmp_path, [get_device, launch])
    _write_capture_manifest(
        tmp_path,
        world_size=4,
        profiled_ranks=(2,),
        profiled_rank_groups={2: (2,)},
        fidelity_windows={
            2: {"start_ts": 999_990, "end_ts": 1_000_100, "source": "trace_markers"}
        },
        rank_host_machines={2: "host-a"},
        rank_host_dispatch_queues={2: "host-a:rank:2"},
    )

    bundle = load_trace_directory(tmp_path, trace_window="step")
    loaded_prev, loaded_current = bundle.rank_traces[0].events
    leading_occurrence_id = "rank:2:host_control_boundary:leading->raw_ordinal:0"
    current_occurrence_id = (
        "rank:2:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
    )

    assert loaded_prev.extras["host_control_boundary_occurrence_id"] == leading_occurrence_id
    assert loaded_prev.extras["actual_counterpart_id"] == leading_occurrence_id
    assert loaded_current.extras["host_control_boundary_occurrence_id"] == current_occurrence_id
    assert loaded_current.extras["actual_counterpart_id"] == current_occurrence_id
    assert loaded_current.extras["host_control_envelope_counterpart_key"] == (
        current_occurrence_id
    )
    assert loaded_current.extras["hostdelay_counterpart_key"] == current_occurrence_id

    collated = collate_trace_bundle(bundle)
    host_delay = [
        event
        for event in collated.rank_events[2]
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cudaGetDevice"
        and event.extras.get("raw_current_api") == "cudaLaunchKernel"
    ][0]

    assert host_delay.extras["host_control_boundary_occurrence_id"] == current_occurrence_id
    assert host_delay.extras["host_control_envelope_counterpart_schema_version"] == (
        "host_control_replay_envelope_counterpart_metadata_v1"
    )
    assert host_delay.extras["host_control_envelope_counterpart_key"] == (
        current_occurrence_id
    )
    assert host_delay.extras["hostdelay_counterpart_key"] == current_occurrence_id
    assert host_delay.extras["host_control_envelope_hostdelay_interval_id"] == "r2:h1"
    assert host_delay.extras["host_control_envelope_counterpart_interval_id"] == (
        f"{current_occurrence_id}:endpoint_gap"
    )
    assert host_delay.extras["actual_counterpart_id"] == current_occurrence_id
    assert host_delay.extras["selected_occurrence_id"] == current_occurrence_id
    assert host_delay.extras["paper_valid_window_id"] == "rank2:step_window"
    assert host_delay.extras["actual_paper_valid_window_id"] == "rank2:step_window"
    assert host_delay.extras["actual_counterpart_window_id"] == "rank2:step_window"
    assert host_delay.extras["actual_counterpart_window_unavailable_reason"] is None
    assert host_delay.extras["actual_raw_prev_event_id"] == "rank:2:raw_ordinal:0"
    assert host_delay.extras["actual_raw_current_event_id"] == "rank:2:raw_ordinal:1"
    assert host_delay.extras["affected_interval_id"] == (
        f"{current_occurrence_id}:endpoint_gap"
    )
    assert host_delay.extras["interval_kind"] == "actual_endpoint_gap_context_only"
    assert host_delay.extras["count_once_status"] == "unavailable"
    assert host_delay.extras["boundary_origin_prev_fields"][
        "host_control_boundary_occurrence_id"
    ] == leading_occurrence_id
    assert host_delay.extras["boundary_origin_current_fields"][
        "host_control_boundary_occurrence_id"
    ] == current_occurrence_id
    assert host_delay.extras["boundary_origin_field_sources"][
        "actual_counterpart_window_id"
    ] == ["collate"]
    assert host_delay.extras["safe_to_use_as_subtraction_delta"] is False
    assert host_delay.extras["safe_to_use_as_repair_evidence"] is False


def test_write_rank_trace_opt_in_actual_cuda_event_counterpart_metadata(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("MAYA_ENABLE_ACTUAL_CUDA_EVENT_COUNTERPART_DIAGNOSTICS", "1")
    record_event = cpp_event.EventRecord()
    record_event.kind = cpp_event.EventKind.RuntimeCall
    record_event.api_name = "cudaEventRecord"
    record_event.timestamp = timedelta(seconds=1)
    record_event.end_timestamp = timedelta(seconds=1, microseconds=5)
    record_event.host_duration = timedelta(microseconds=5)
    record_event.process_id = 33
    record_event.thread_id = 44
    record_event.payload.attributes = {
        "event_id": "654321",
        "stream_id": "123456",
    }
    wait_event = cpp_event.EventRecord()
    wait_event.kind = cpp_event.EventKind.RuntimeCall
    wait_event.api_name = "cudaStreamWaitEvent"
    wait_event.timestamp = timedelta(seconds=2)
    wait_event.end_timestamp = timedelta(seconds=2, microseconds=7)
    wait_event.host_duration = timedelta(microseconds=7)
    wait_event.process_id = 33
    wait_event.thread_id = 44
    wait_event.payload.attributes = {
        "event_id": "654321",
        "stream_id": "777",
        "flags": "0",
    }

    path = _write_rank_trace(tmp_path, [record_event, wait_event])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    record_row, wait_row = rows

    assert record_row["actual_cuda_event_counterpart_schema_version"] == (
        "actual_cuda_event_record_wait_release_v1"
    )
    assert record_row["actual_cuda_event_handle"] == "654321"
    assert record_row["actual_cuda_event_version"] == 1
    assert record_row["actual_record_wait_pair_id"] == "rank:3:cuda_event:654321:version:1"
    assert record_row["actual_record_raw_event_id"] == "rank:3:raw_ordinal:0"
    assert record_row["actual_record_ts_us"] == 1_000_000
    assert record_row["actual_record_end_ts_us"] == 1_000_005
    assert record_row["actual_released_by_event_id"] == "rank:3:raw_ordinal:0"
    assert record_row["actual_release_us"] is None
    assert record_row["safe_to_use_as_subtraction_delta"] is False
    assert record_row["safe_to_use_as_repair_evidence"] is False

    assert wait_row["actual_cuda_event_counterpart_schema_version"] == (
        "actual_cuda_event_record_wait_release_v1"
    )
    assert wait_row["actual_cuda_event_handle"] == "654321"
    assert wait_row["actual_cuda_event_version"] == 1
    assert wait_row["actual_record_wait_pair_id"] == "rank:3:cuda_event:654321:version:1"
    assert wait_row["actual_wait_raw_event_id"] == "rank:3:raw_ordinal:1"
    assert wait_row["actual_wait_api_ts_us"] == 2_000_000
    assert wait_row["actual_wait_api_end_ts_us"] == 2_000_007
    assert wait_row["actual_released_by_event_id"] == "rank:3:raw_ordinal:0"
    assert wait_row["actual_record_ts_us"] == 1_000_000
    assert wait_row["actual_wait_start_us"] is None
    assert wait_row["actual_release_us"] is None
    assert wait_row["actual_release_reason"] == (
        "record_operation_identified_release_timing_unavailable"
    )
    assert wait_row["actual_stream_namespace_alignment"] == (
        "actual_only_unresolved_predicted_namespace_not_joined"
    )
    assert "not_observable_from_wrapper_endpoint" in wait_row[
        "actual_cuda_event_counterpart_unavailable_reason"
    ]


def test_write_rank_trace_combined_generic_and_cuda_event_counterpart_metadata(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("MAYA_ENABLE_ACTUAL_CUDA_EVENT_COUNTERPART_DIAGNOSTICS", "1")
    monkeypatch.setenv(
        "MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_DIAGNOSTICS",
        "1",
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_ACTUAL_CUDA_EVENT_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )

    record_event = cpp_event.EventRecord()
    record_event.kind = cpp_event.EventKind.RuntimeCall
    record_event.api_name = "cudaEventRecord"
    record_event.timestamp = timedelta(seconds=1)
    record_event.end_timestamp = timedelta(seconds=1, microseconds=5)
    record_event.host_duration = timedelta(microseconds=5)
    record_event.process_id = 33
    record_event.thread_id = 44
    record_event.payload.attributes = {
        "event_id": "654321",
        "stream_id": "123456",
    }
    wait_event = cpp_event.EventRecord()
    wait_event.kind = cpp_event.EventKind.RuntimeCall
    wait_event.api_name = "cudaStreamWaitEvent"
    wait_event.timestamp = timedelta(seconds=2)
    wait_event.end_timestamp = timedelta(seconds=2, microseconds=7)
    wait_event.host_duration = timedelta(microseconds=7)
    wait_event.process_id = 33
    wait_event.thread_id = 44
    wait_event.payload.attributes = {
        "event_id": "654321",
        "stream_id": "777",
        "flags": "0",
    }

    path = _write_rank_trace(tmp_path, [record_event, wait_event])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    record_row, wait_row = rows

    for row in rows:
        assert row["generic_actual_counterpart_schema_version"] == (
            "generic_replay_placement_envelope_actual_counterpart_metadata_v1"
        )
        assert row["actual_cuda_event_counterpart_schema_version"] == (
            "actual_cuda_event_record_wait_release_v1"
        )
        assert row["generic_actual_timing_status"] == (
            "endpoint_context_only_strict_counterpart_unavailable"
        )
        assert row["generic_actual_release_us"] is None
        assert row["generic_actual_release_reason"] is None
        assert row["generic_actual_released_by_event_id"] is None
        assert row["generic_actual_release_source_kind"] is None
        assert row["generic_wait_map_safety_status"] == "unavailable"
        assert row["generic_safe_to_use_as_repair_evidence"] is False
        assert row["generic_safe_to_use_as_subtraction_delta"] is False
        assert row["generic_repair_ready"] is False

    assert record_row["actual_released_by_event_id"] == "rank:3:raw_ordinal:0"
    assert record_row["actual_release_reason"] == (
        "record_operation_identified_release_timing_unavailable"
    )
    assert wait_row["actual_released_by_event_id"] == "rank:3:raw_ordinal:0"
    assert wait_row["actual_release_reason"] == (
        "record_operation_identified_release_timing_unavailable"
    )
    assert wait_row["actual_release_us"] is None
    assert wait_row["actual_wait_start_us"] is None


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
    ],
)
def test_write_rank_trace_component_strict_actual_metadata_opt_in(
    tmp_path: Path,
    monkeypatch,
    env_key,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv(env_key, "true")
    other_key = (
        "FLEXSIM_MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS"
        if env_key == "MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS"
        else "MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS"
    )
    monkeypatch.delenv(other_key, raising=False)

    launch = cpp_event.EventRecord()
    launch.kind = cpp_event.EventKind.ComputeKernel
    launch.api_name = "cudaLaunchKernel"
    launch.timestamp = timedelta(seconds=1)
    launch.end_timestamp = timedelta(seconds=1, microseconds=6)
    launch.host_duration = timedelta(microseconds=6)
    launch.process_id = 33
    launch.thread_id = 44
    launch.payload.attributes = {
        "kernel": "void_fused_attention_kernel",
        "grid_x": "12",
        "grid_y": "3",
        "grid_z": "1",
        "block_x": "256",
        "block_y": "1",
        "block_z": "1",
        "shared_mem": "2048",
        "stream_id": "0",
    }

    path = _write_rank_trace(tmp_path, [launch])
    row = json.loads(path.read_text().splitlines()[0])
    material_signature = (
        "kernel=void_fused_attention_kernel;grid=12x3x1;"
        "block=256x1x1;shared_mem=2048;stream=0"
    )

    assert row["actual_counterpart_schema_version"] == (
        "component_strict_counterpart_actual_metadata_evidence_v1"
    )
    assert row["component_strict_counterpart_opt_in_flag"] is True
    assert row["source_side"] == "actual_counterpart_metadata"
    assert row["actual_counterpart_row_id"].startswith(
        "rank:4:component_strict_actual:raw_ordinal:0:api:cudaLaunchKernel"
    )
    assert row["actual_rank"] == 4
    assert row["actual_world_size"] == 8
    assert row["actual_api"] == "cudaLaunchKernel"
    assert row["actual_stream_resource_id"] == "rank:4:stream:0"
    assert row["actual_material_signature"] == material_signature
    assert row["actual_material_signature_status"] == "available"
    assert row["common_basis_material_signature"] == material_signature
    assert row["common_basis_kind"] == "actual_trace_row_metadata_only"
    assert row["actual_counterpart_join_status"] == (
        "actual_metadata_export_only_predicted_join_deferred"
    )
    assert row["strict_actual_timing_status"] == "unavailable"
    assert row["strict_actual_timing_available"] is False
    assert row["actual_start_us"] is None
    assert row["actual_end_us"] is None
    assert row["actual_duration_us"] is None
    assert row["actual_endpoint_ts_us"] == 1_000_000
    assert row["actual_endpoint_end_ts_us"] == 1_000_006
    assert row["actual_endpoint_host_duration_us"] == 6.0
    assert row["actual_endpoint_timestamps_used_as_strict_timing"] is False
    assert row["actual_host_duration_used_as_strict_timing"] is False
    assert row["actual_runtime_direct_substitution"] is False
    assert row["actual_observed_runtime_used_as_prediction"] is False
    assert row["stream_namespace_alignment_status"] == (
        "actual_only_unresolved_predicted_namespace_not_joined"
    )
    assert row["count_once_status"] == "unavailable"
    assert row["nonoverlap_status"] == "unavailable"
    assert row["wait_map_safety_status"] == "unavailable"
    assert row["producer_visibility_status"] == "unavailable"
    assert row["repair_ready"] is False
    assert row["safe_to_use_as_repair_evidence"] is False
    assert row["safe_to_use_as_subtraction_delta"] is False


def test_write_rank_trace_component_strict_launch_material_signature_missing_fields_null(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv(
        "MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
        "true",
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
        raising=False,
    )

    launch = cpp_event.EventRecord()
    launch.kind = cpp_event.EventKind.ComputeKernel
    launch.api_name = "cudaLaunchKernel"
    launch.timestamp = timedelta(seconds=1)
    launch.end_timestamp = timedelta(seconds=1, microseconds=6)
    launch.host_duration = timedelta(microseconds=6)
    launch.process_id = 33
    launch.thread_id = 44
    launch.payload.attributes = {
        "kernel": "void_fused_attention_kernel",
        "grid_x": "12",
        "grid_y": "3",
        "grid_z": "1",
        "block_x": "256",
        "block_y": "1",
        "shared_mem": "2048",
        "stream_id": "0",
    }

    path = _write_rank_trace(tmp_path, [launch])
    row = json.loads(path.read_text().splitlines()[0])

    assert row["actual_material_signature"] is None
    assert row["actual_material_signature_status"] == "unavailable"
    assert row["common_basis_material_signature"] is None
    assert row["actual_endpoint_ts_us"] == 1_000_000
    assert row["actual_endpoint_end_ts_us"] == 1_000_006
    assert row["actual_endpoint_host_duration_us"] == 6.0
    assert row["actual_runtime_direct_substitution"] is False


def test_write_rank_trace_component_strict_canonicalizes_gemm_material_signature(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv(
        "MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
        "true",
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
        raising=False,
    )

    gemm = cpp_event.EventRecord()
    gemm.kind = cpp_event.EventKind.ComputeKernel
    gemm.api_name = "cublasGemmEx"
    gemm.timestamp = timedelta(seconds=1)
    gemm.end_timestamp = timedelta(seconds=1, microseconds=6)
    gemm.host_duration = timedelta(microseconds=6)
    gemm.process_id = 33
    gemm.thread_id = 44
    gemm.payload.attributes = {
        "material_signature": "preexisting-material-signature",
        "m": "256",
        "n": "128",
        "k": "64",
        "computeType": "68",
        "transa": "1",
        "transb": "0",
        "algorithm": "99",
    }

    path = _write_rank_trace(tmp_path, [gemm])
    row = json.loads(path.read_text().splitlines()[0])

    assert row["actual_api"] == "cublasGemmEx"
    assert row["actual_material_signature"] == (
        "m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=99"
    )
    assert row["common_basis_material_signature"] == row["actual_material_signature"]
    assert row["actual_material_signature_status"] == "available"
    assert row["strict_actual_timing_available"] is False
    assert row["actual_runtime_direct_substitution"] is False
    assert row["actual_observed_runtime_used_as_prediction"] is False


def test_write_rank_trace_component_strict_metadata_default_off_absent(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "4")
    monkeypatch.delenv(
        "MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
        raising=False,
    )

    launch = cpp_event.EventRecord()
    launch.kind = cpp_event.EventKind.ComputeKernel
    launch.api_name = "cudaLaunchKernel"
    launch.timestamp = timedelta(seconds=1)
    launch.end_timestamp = timedelta(seconds=1, microseconds=6)
    launch.host_duration = timedelta(microseconds=6)
    launch.process_id = 33
    launch.thread_id = 44
    launch.payload.attributes = {"stream_id": "0"}

    path = _write_rank_trace(tmp_path, [launch])
    row = json.loads(path.read_text().splitlines()[0])

    assert "actual_counterpart_schema_version" not in row
    assert "component_strict_counterpart_schema_version" not in row
    assert "strict_actual_timing_status" not in row
    assert "safe_to_use_as_repair_evidence" not in row
    assert "actual_material_signature" not in row
    assert "common_basis_material_signature" not in row


def test_write_rank_trace_infers_world_size_from_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    event = cpp_event.EventRecord()
    event.kind = cpp_event.EventKind.AllReduce
    event.api_name = "ncclAllReduce"
    event.timestamp = timedelta(seconds=1)
    event.process_id = 11
    event.thread_id = 22
    event.payload.attributes = {
        "count": "16",
    }

    path = _write_rank_trace(tmp_path, [event])
    record = json.loads(path.read_text().strip())

    assert record["world_size"] == "2"


def test_write_capture_manifest(tmp_path: Path):
    path = _write_capture_manifest(
        tmp_path,
        world_size=4,
        profiled_ranks=(0, 2),
        profiled_rank_groups={0: (0, 1), 2: (2, 3)},
        fidelity_windows={0: {"start_ts": 10, "end_ts": 20, "source": "trace_markers"}},
    )
    payload = json.loads(path.read_text())

    assert payload["original_world_size"] == 4
    assert payload["profiled_ranks"] == [0, 2]
    assert payload["profiled_rank_groups"] == {"0": [0, 1], "2": [2, 3]}
    assert payload["step_windows"] == {
        "0": {
            "start_ts": 10,
            "end_ts": 20,
            "source": "trace_markers",
            "is_paper_valid_step_window": True,
        }
    }
    assert payload["fidelity_windows"] == {
        "0": {
            "start_ts": 10,
            "end_ts": 20,
            "source": "trace_markers",
            "is_paper_valid_step_window": True,
        }
    }


def test_write_capture_manifest_keeps_explicit_non_paper_window_out_of_step_windows(
    tmp_path: Path,
):
    path = _write_capture_manifest(
        tmp_path,
        world_size=1,
        profiled_ranks=(0,),
        profiled_rank_groups={0: (0,)},
        fidelity_windows={
            0: {
                "start_ts": 10,
                "end_ts": 20,
                "source": "trace_markers",
                "is_paper_valid_step_window": False,
            }
        },
    )
    payload = json.loads(path.read_text())

    assert payload["step_windows"] == {}
    assert payload["fidelity_windows"] == {
        "0": {
            "start_ts": 10,
            "end_ts": 20,
            "source": "trace_markers",
            "is_paper_valid_step_window": False,
        }
    }


def test_merge_capture_manifest_tolerates_empty_existing_file(tmp_path: Path):
    manifest_path = tmp_path / "capture_manifest.json"
    manifest_path.write_text("", encoding="utf-8")

    _merge_capture_manifest(
        tmp_path,
        world_size=8,
        profiled_ranks=(0, 4),
        profiled_rank_groups={0: (0, 1, 2, 3), 4: (4, 5, 6, 7)},
        rank=4,
        fidelity_window={"start_ts": 100, "end_ts": 200, "source": "trace_markers"},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["original_world_size"] == 8
    assert payload["profiled_ranks"] == [0, 4]
    assert payload["profiled_rank_groups"] == {"0": [0, 1, 2, 3], "4": [4, 5, 6, 7]}
    assert payload["step_windows"] == {
        "4": {
            "start_ts": 100,
            "end_ts": 200,
            "source": "trace_markers",
            "is_paper_valid_step_window": True,
        }
    }
    assert payload["fidelity_windows"] == {
        "4": {
            "start_ts": 100,
            "end_ts": 200,
            "source": "trace_markers",
            "is_paper_valid_step_window": True,
        }
    }


def test_merge_capture_manifest_tolerates_invalid_existing_json(tmp_path: Path):
    manifest_path = tmp_path / "capture_manifest.json"
    manifest_path.write_text("{not-json", encoding="utf-8")

    _merge_capture_manifest(
        tmp_path,
        world_size=2,
        profiled_ranks=(0,),
        profiled_rank_groups={0: (0, 1)},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["original_world_size"] == 2
    assert payload["profiled_ranks"] == [0]
    assert payload["profiled_rank_groups"] == {"0": [0, 1]}
    assert payload["step_windows"] == {}
    assert payload["fidelity_windows"] == {}


def test_merge_capture_manifest_keeps_boundary_fallback_out_of_step_windows(tmp_path: Path):
    manifest_path = tmp_path / "capture_manifest.json"

    _merge_capture_manifest(
        tmp_path,
        world_size=2,
        profiled_ranks=(0,),
        profiled_rank_groups={0: (0, 1)},
        rank=0,
        fidelity_window={"start_ts": 100, "end_ts": 300, "source": "boundary_fallback"},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["step_windows"] == {}
    assert payload["fidelity_windows"] == {
        "0": {
            "start_ts": 100,
            "end_ts": 300,
            "source": "boundary_fallback",
            "is_paper_valid_step_window": False,
        }
    }


def test_merge_capture_manifest_keeps_explicit_non_paper_marker_window_out_of_step_windows(
    tmp_path: Path,
):
    manifest_path = tmp_path / "capture_manifest.json"

    _merge_capture_manifest(
        tmp_path,
        world_size=2,
        profiled_ranks=(0,),
        profiled_rank_groups={0: (0, 1)},
        rank=0,
        fidelity_window={
            "start_ts": 100,
            "end_ts": 300,
            "source": "trace_markers",
            "is_paper_valid_step_window": False,
        },
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["step_windows"] == {}
    assert payload["fidelity_windows"] == {
        "0": {
            "start_ts": 100,
            "end_ts": 300,
            "source": "trace_markers",
            "is_paper_valid_step_window": False,
        }
    }


def test_planner_pairwise_and_single():
    assert plan_profiled_rank_groups(4, strategy="pairwise") == {0: (0, 1), 2: (2, 3)}
    assert profiled_ranks_for_groups({0: (0, 1), 2: (2, 3)}) == (0, 2)
    assert plan_profiled_rank_groups(4, strategy="single") == {0: (0, 1, 2, 3)}
    assert plan_profiled_rank_groups(4, strategy="identity") == {
        0: (0,),
        1: (1,),
        2: (2,),
        3: (3,),
    }
