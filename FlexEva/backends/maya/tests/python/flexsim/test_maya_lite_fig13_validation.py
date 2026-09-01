from __future__ import annotations

import json
from pathlib import Path

from flexsim.maya_lite.fig13_validation import compare_fig13_step_trace_dirs
from flexsim.maya_lite.filters import (
    TraceApiBucket,
    classify_trace_api_bucket,
    is_host_timing_traced_api,
)


def _write_rank_trace(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_step_manifest(
    trace_dir: Path,
    ranks: list[int],
    *,
    start_ts: int = 10,
    end_ts: int = 20,
    original_world_size: int | None = None,
    profiled_rank_groups: dict[int, list[int]] | None = None,
) -> None:
    payload = {
        "step_windows": {
            str(rank): {"start_ts": start_ts, "end_ts": end_ts, "source": "trace_markers", "step_count": 1}
            for rank in ranks
        }
    }
    if original_world_size is not None:
        payload["original_world_size"] = int(original_world_size)
    if profiled_rank_groups:
        payload["profiled_rank_groups"] = {
            str(rank): [int(member) for member in members]
            for rank, members in profiled_rank_groups.items()
        }
    (trace_dir / "capture_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_trace_api_bucket_policy_matches_maya_validation_contract() -> None:
    assert classify_trace_api_bucket("cudaLaunchKernel", "kernel_launch") is TraceApiBucket.SEMANTIC_TRACED
    assert classify_trace_api_bucket("cublasSetStream_v2", "stream_op") is TraceApiBucket.SEMANTIC_TRACED
    assert classify_trace_api_bucket("ncclCommInitRankConfig", "other") is TraceApiBucket.SEMANTIC_TRACED
    assert classify_trace_api_bucket("ncclAllToAll", "other") is TraceApiBucket.SEMANTIC_TRACED
    assert classify_trace_api_bucket("ncclAllToAllv", "other") is TraceApiBucket.SEMANTIC_TRACED
    assert classify_trace_api_bucket("ncclCommCount", "other") is TraceApiBucket.COMPAT_ONLY
    assert classify_trace_api_bucket("ncclCommUserRank", "other") is TraceApiBucket.COMPAT_ONLY
    assert classify_trace_api_bucket("cudaEventQuery", "stream_op") is TraceApiBucket.SEMANTIC_TRACED
    assert classify_trace_api_bucket("cudaGetDevice", "other") is TraceApiBucket.COMPAT_ONLY
    assert classify_trace_api_bucket("__cudaRegisterFunction", "other") is TraceApiBucket.COMPAT_ONLY
    assert classify_trace_api_bucket("__cudaPushCallConfiguration", "other") is TraceApiBucket.COMPAT_ONLY
    assert classify_trace_api_bucket("cudaStreamCreateWithPriority", "stream_op") is TraceApiBucket.COMPAT_ONLY
    assert classify_trace_api_bucket("cublasLtCreate", "context_op") is TraceApiBucket.COMPAT_ONLY
    assert classify_trace_api_bucket("cublasLtMatmulDescCreate", "context_op") is TraceApiBucket.COMPAT_ONLY
    assert classify_trace_api_bucket("cublasLtMatmulPreferenceCreate", "context_op") is TraceApiBucket.COMPAT_ONLY


def test_host_timing_policy_keeps_compat_only_apis_out_of_semantic_conformance() -> None:
    assert is_host_timing_traced_api("cudaLaunchKernel", "kernel_launch") is True
    assert is_host_timing_traced_api("ncclCommCount", "other") is True
    assert is_host_timing_traced_api("ncclCommUserRank", "other") is True
    assert is_host_timing_traced_api("cudaEventQuery", "stream_op") is True
    assert is_host_timing_traced_api("cudaGetDevice", "other") is True
    assert is_host_timing_traced_api("__cudaRegisterFunction", "other") is True
    assert is_host_timing_traced_api("__cudaPushCallConfiguration", "other") is True
    assert is_host_timing_traced_api("cudaStreamCreateWithPriority", "stream_op") is True
    assert is_host_timing_traced_api("cublasLtCreate", "context_op") is True
    assert is_host_timing_traced_api("cublasLtMatmulDescCreate", "context_op") is True
    assert is_host_timing_traced_api("totallyUnsupportedApi", "other") is False


def test_compare_fig13_step_trace_dirs_uses_step_windows(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    emulated_dir = tmp_path / "emulated"
    real_dir.mkdir()
    emulated_dir.mkdir()

    _write_rank_trace(
        real_dir / "rank_0.jsonl",
        [
            {"ts": 5, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclCommInitRank", "type": "other"},
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclSend", "type": "nccl_collective"},
            {"ts": 13, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaLaunchKernel", "type": "kernel_launch"},
        ],
    )
    _write_rank_trace(
        real_dir / "rank_1.jsonl",
        [
            {"ts": 14, "pid": 1, "tid": 1, "mod": "libnccl.so", "api": "ncclRecv", "type": "nccl_collective"},
        ],
    )
    _write_step_manifest(real_dir, [0, 1])

    _write_rank_trace(
        emulated_dir / "rank_0.jsonl",
        [
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclSend", "type": "nccl_collective"},
            {"ts": 13, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaLaunchKernel", "type": "kernel_launch"},
        ],
    )
    _write_rank_trace(
        emulated_dir / "rank_1.jsonl",
        [
            {"ts": 14, "pid": 1, "tid": 1, "mod": "libnccl.so", "api": "ncclRecv", "type": "nccl_collective"},
        ],
    )
    _write_step_manifest(emulated_dir, [0, 1])

    payload = compare_fig13_step_trace_dirs(real_dir, emulated_dir)

    assert payload["real"]["trace_window"] == "step"
    assert payload["emulated"]["trace_window"] == "step"
    assert payload["real"]["paper_valid_step_window_rank_count"] == 2
    assert payload["emulated"]["paper_valid_step_window_rank_count"] == 2
    assert payload["real"]["step_window_sources"] == ["trace_markers"]
    assert payload["emulated"]["step_window_sources"] == ["trace_markers"]
    assert payload["global_metrics"]["api_cosine_similarity"] == 1.0
    assert payload["global_metrics"]["nccl_sequence_match"] == 1.0
    assert payload["real_only_apis"] == []
    assert payload["emulated_only_apis"] == []


def test_compare_fig13_step_trace_dirs_reports_real_only_api(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    emulated_dir = tmp_path / "emulated"
    real_dir.mkdir()
    emulated_dir.mkdir()

    _write_rank_trace(
        real_dir / "rank_0.jsonl",
        [
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclAllReduce", "type": "nccl_collective"},
            {"ts": 13, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaMemcpy", "type": "mem_copy"},
        ],
    )
    _write_step_manifest(real_dir, [0])

    _write_rank_trace(
        emulated_dir / "rank_0.jsonl",
        [
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclAllReduce", "type": "nccl_collective"},
        ],
    )
    _write_step_manifest(emulated_dir, [0])

    payload = compare_fig13_step_trace_dirs(real_dir, emulated_dir)

    assert payload["global_metrics"]["api_coverage"] == 0.5
    assert payload["real_only_apis"] == ["cudaMemcpy"]
    assert payload["emulated_only_apis"] == []


def test_compare_fig13_step_trace_dirs_reports_semantic_metrics(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    emulated_dir = tmp_path / "emulated"
    real_dir.mkdir()
    emulated_dir.mkdir()

    _write_rank_trace(
        real_dir / "rank_0.jsonl",
        [
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclGroupStart", "type": "other"},
            {"ts": 13, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclAllReduce", "type": "nccl_collective"},
            {"ts": 14, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclGroupEnd", "type": "other"},
            {"ts": 15, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaEventRecord", "type": "stream_op"},
        ],
    )
    _write_step_manifest(real_dir, [0])

    _write_rank_trace(
        emulated_dir / "rank_0.jsonl",
        [
            {"ts": 13, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclAllReduce", "type": "nccl_collective"},
            {
                "ts": 15,
                "pid": 0,
                "tid": 0,
                "mod": "libcudart.so",
                "api": "cudaEventRecordWithFlags",
                "type": "stream_op",
            },
        ],
    )
    _write_step_manifest(emulated_dir, [0])

    payload = compare_fig13_step_trace_dirs(real_dir, emulated_dir)

    assert payload["global_metrics"]["api_coverage"] == 0.5
    assert payload["semantic_metrics"]["api_coverage"] == 1.0
    assert payload["semantic_metrics"]["real_only_apis"] == []
    assert payload["semantic_metrics"]["emulated_only_apis"] == []


def test_compare_fig13_step_trace_dirs_reports_pre_step_context_metrics(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    emulated_dir = tmp_path / "emulated"
    real_dir.mkdir()
    emulated_dir.mkdir()

    _write_rank_trace(
        real_dir / "rank_0.jsonl",
        [
            {"ts": 10, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclGetVersion", "type": "other"},
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclAllReduce", "type": "nccl_collective"},
        ],
    )
    _write_step_manifest(real_dir, [0], start_ts=12, end_ts=20)

    _write_rank_trace(
        emulated_dir / "rank_0.jsonl",
        [
            {"ts": 10, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclGetVersion", "type": "other"},
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclAllReduce", "type": "nccl_collective"},
        ],
    )
    _write_step_manifest(emulated_dir, [0], start_ts=12, end_ts=20)

    payload = compare_fig13_step_trace_dirs(real_dir, emulated_dir, pre_step_context_us=2)

    assert payload["global_metrics"]["api_coverage"] == 1.0
    assert payload["pre_step_context_metrics"]["api_coverage"] == 1.0
    assert payload["pre_step_context_metrics"]["pre_step_context_us"] == 2
    assert payload["stream_event_metrics"]["real_total_calls"] == 0
    assert payload["stream_event_metrics"]["emulated_total_calls"] == 0


def test_compare_fig13_step_trace_dirs_reports_stream_event_metrics(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    emulated_dir = tmp_path / "emulated"
    real_dir.mkdir()
    emulated_dir.mkdir()

    _write_rank_trace(
        real_dir / "rank_0.jsonl",
        [
            {"ts": 10, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaStreamCreateWithPriority", "type": "stream_op"},
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaEventRecord", "type": "stream_op"},
            {"ts": 14, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclAllReduce", "type": "nccl_collective"},
        ],
    )
    _write_step_manifest(real_dir, [0], start_ts=14, end_ts=20)

    _write_rank_trace(
        emulated_dir / "rank_0.jsonl",
        [
            {"ts": 10, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaStreamCreateWithPriority", "type": "stream_op"},
            {"ts": 14, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclAllReduce", "type": "nccl_collective"},
        ],
    )
    _write_step_manifest(emulated_dir, [0], start_ts=14, end_ts=20)

    payload = compare_fig13_step_trace_dirs(real_dir, emulated_dir, pre_step_context_us=5)

    assert payload["stream_event_metrics"]["real_total_calls"] == 2
    assert payload["stream_event_metrics"]["emulated_total_calls"] == 1
    assert payload["stream_event_metrics"]["api_coverage"] == 0.5
    assert payload["stream_event_metrics"]["real_only_apis"] == ["cudaEventRecord"]


def test_compare_fig13_step_trace_dirs_excludes_compat_only_apis_from_conformance(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    emulated_dir = tmp_path / "emulated"
    real_dir.mkdir()
    emulated_dir.mkdir()

    _write_rank_trace(
        real_dir / "rank_0.jsonl",
        [
            {"ts": 10, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaGetDevice", "type": "other"},
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libcublas.so", "api": "cublasSetStream_v2", "type": "stream_op"},
            {"ts": 14, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaLaunchKernel", "type": "kernel_launch"},
        ],
    )
    _write_step_manifest(real_dir, [0], start_ts=10, end_ts=20)

    _write_rank_trace(
        emulated_dir / "rank_0.jsonl",
        [
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libcublas.so", "api": "cublasSetStream_v2", "type": "stream_op"},
            {"ts": 14, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaLaunchKernel", "type": "kernel_launch"},
        ],
    )
    _write_step_manifest(emulated_dir, [0], start_ts=10, end_ts=20)

    payload = compare_fig13_step_trace_dirs(real_dir, emulated_dir)

    assert payload["global_metrics"]["api_coverage"] < 1.0
    assert payload["real_only_apis"] == ["cudaGetDevice"]
    assert payload["conformance_metrics"]["api_coverage"] == 1.0
    assert payload["conformance_metrics"]["real_only_apis"] == []
    assert payload["conformance_metrics"]["emulated_only_apis"] == []
    assert payload["compat_only_metrics"]["real_only_apis"] == ["cudaGetDevice"]


def test_compare_fig13_step_trace_dirs_reports_representative_aligned_metrics_for_compact_emulation(
    tmp_path: Path,
) -> None:
    real_dir = tmp_path / "real"
    emulated_dir = tmp_path / "emulated"
    real_dir.mkdir()
    emulated_dir.mkdir()

    _write_rank_trace(
        real_dir / "rank_0.jsonl",
        [
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclSend", "type": "nccl_collective"},
        ],
    )
    _write_rank_trace(
        real_dir / "rank_1.jsonl",
        [
            {"ts": 12, "pid": 1, "tid": 1, "mod": "libnccl.so", "api": "ncclRecv", "type": "nccl_collective"},
        ],
    )
    _write_rank_trace(
        real_dir / "rank_2.jsonl",
        [
            {"ts": 12, "pid": 2, "tid": 2, "mod": "libnccl.so", "api": "ncclSend", "type": "nccl_collective"},
        ],
    )
    _write_rank_trace(
        real_dir / "rank_3.jsonl",
        [
            {"ts": 12, "pid": 3, "tid": 3, "mod": "libnccl.so", "api": "ncclRecv", "type": "nccl_collective"},
        ],
    )
    _write_step_manifest(real_dir, [0, 1, 2, 3], original_world_size=4)

    _write_rank_trace(
        emulated_dir / "rank_0.jsonl",
        [
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclSend", "type": "nccl_collective"},
        ],
    )
    _write_rank_trace(
        emulated_dir / "rank_1.jsonl",
        [
            {"ts": 12, "pid": 1, "tid": 1, "mod": "libnccl.so", "api": "ncclRecv", "type": "nccl_collective"},
        ],
    )
    _write_step_manifest(
        emulated_dir,
        [0, 1],
        original_world_size=4,
        profiled_rank_groups={0: [0, 2], 1: [1, 3]},
    )

    payload = compare_fig13_step_trace_dirs(real_dir, emulated_dir)

    assert payload["global_metrics"]["nccl_sequence_match"] < 1.0
    assert payload["representative_alignment"]["mode"] == "representative_projection"
    assert payload["representative_alignment"]["compact_side"] == "emulated"
    assert payload["representative_aligned_metrics"]["nccl_sequence_match"] == 1.0
    assert payload["representative_aligned_semantic_metrics"]["nccl_sequence_match"] == 1.0



def test_compare_fig13_step_trace_dirs_reports_normalized_semantic_metrics(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    emulated_dir = tmp_path / "emulated"
    real_dir.mkdir()
    emulated_dir.mkdir()

    _write_rank_trace(
        real_dir / "rank_0.jsonl",
        [
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaLaunchKernel", "type": "kernel_launch"},
            {"ts": 13, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaEventQuery", "type": "stream_op"},
            {"ts": 14, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaMalloc", "type": "mem_alloc"},
            {"ts": 15, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclAllReduce", "type": "nccl_collective"},
        ],
    )
    _write_step_manifest(real_dir, [0], start_ts=10, end_ts=20)

    _write_rank_trace(
        emulated_dir / "rank_0.jsonl",
        [
            {"ts": 11, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "__cudaPushCallConfiguration", "type": "other"},
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaLaunchKernel", "type": "kernel_launch"},
            {"ts": 13, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaEventQuery", "type": "stream_op"},
            {"ts": 14, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "cudaEventQuery", "type": "stream_op"},
            {"ts": 15, "pid": 0, "tid": 0, "mod": "libcudart.so", "api": "__cudaPopCallConfiguration", "type": "other"},
            {"ts": 16, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclAllReduce", "type": "nccl_collective"},
        ],
    )
    _write_step_manifest(emulated_dir, [0], start_ts=10, end_ts=20)

    payload = compare_fig13_step_trace_dirs(real_dir, emulated_dir)
    normalized = payload["normalized_semantic_metrics"]

    assert payload["semantic_metrics"]["api_coverage"] < 1.0
    assert normalized["without_launch_config"]["emulated_only_apis"] == []
    assert normalized["without_launch_config_and_allocator_jitter"]["api_coverage"] == 1.0
    assert normalized["core_workload_nccl"]["api_cosine_similarity"] == 1.0
    assert normalized["core_workload_nccl"]["api_coverage"] == 1.0
    assert normalized["core_workload_nccl"]["call_count_ratio"] == 1.0
    assert normalized["core_workload_nccl"]["excluded_apis"] == [
        "__cudaPopCallConfiguration",
        "__cudaPushCallConfiguration",
        "cudaEventQuery",
        "cudaMalloc",
    ]
