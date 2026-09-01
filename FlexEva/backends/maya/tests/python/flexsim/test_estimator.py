import base64
import json
import math
import pickle
import sys
from pathlib import Path

import pytest

from flexsim.estimator import (
    DEFAULT_GPU_ESTIMATOR_BUNDLE,
    Estimator,
    GPUXGBoostTimingProvider,
    ProviderLoadStatus,
    TraceSignatureTimingProvider,
    TraceLearnedTimingProvider,
    _collapse_collective_group_training_pairs,
    _device_work_weight,
    _signature_key_from_features,
    build_collective_group_timing_features,
    build_learned_timing_features,
    canonicalize_gpu_estimator_event,
    probe_gpu_estimator_provider,
    probe_trace_learned_provider,
    _trace_learned_nccl_collective_support_keys,
)


def _write_trace(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class _TraceLearnedTestNumpy:
    bool_ = bool

    @staticmethod
    def expm1(value):
        if isinstance(value, list):
            return [math.expm1(item) for item in value]
        return math.expm1(value)

    @staticmethod
    def isscalar(value):
        return isinstance(value, (int, float))


class _TraceLearnedTestVectorizer:
    def __init__(self, feature_names: list[str]):
        self.feature_names_ = feature_names

    def transform(self, rows):
        return list(rows)


class _TraceLearnedConstantLogModel:
    def __init__(self, value_us: float):
        self._log_value = math.log1p(value_us)

    def predict(self, matrix):
        return [self._log_value for _ in matrix]


def test_collective_group_timing_features_are_group_stable() -> None:
    features = build_collective_group_timing_features(
        {
            "api": "ncclAllReduce",
            "type": "nccl_collective",
            "rank": 7,
            "ordinal": 12345,
            "pid": 99,
            "tid": 13,
            "prev_api": "cudaLaunchKernel",
            "collective_sequence_number": 42,
            "collective": "allreduce",
            "collective_api": "ncclAllReduce",
            "collective_match_basis": "communicator_sequence",
            "collective_communicator_id": "flexsim-members:0,1,2,3",
            "communicator_size": 4,
            "participant_count": 4,
            "count": 1024,
            "datatype": 9,
            "op": 0,
            "world_size": 16,
            "host_duration_us": 3.0,
            "wrapper_runtime_contract": "dispatch_only",
        }
    )

    assert features["api"] == "ncclAllReduce"
    assert features["collective_communicator_id"] == "flexsim-members:0,1,2,3"
    assert features["world_size"] == 4
    assert features["communicator_size"] == 4
    assert features["participant_count"] == 4
    assert features["numel"] == 1024
    assert features["dtype_code"] == 1
    assert features["reduction"] == "sum"
    assert "rank" not in features
    assert "ordinal_bucket" not in features
    assert "pid_mod_8" not in features
    assert "thread_mod_8" not in features
    assert "prev_api" not in features
    assert "collective_sequence_number" not in features
    assert "observed_wrapper_us" not in features
    assert "wrapper_runtime_contract" not in features


def test_collective_group_timing_features_normalize_p2p_member_direction() -> None:
    send_first = build_collective_group_timing_features(
        {
            "api": "ncclSend",
            "type": "nccl_collective",
            "rank": 0,
            "ordinal": 10,
            "collective": "send",
            "collective_api": "ncclSend",
            "collective_group_id": "ncclSend|comm:p2p|call:7",
            "collective_communicator_id": "p2p",
            "communicator_size": 2,
            "participant_count": 1,
            "count": 4096,
            "datatype": 7,
        }
    )
    recv_first = build_collective_group_timing_features(
        {
            "api": "ncclRecv",
            "type": "nccl_collective",
            "rank": 1,
            "ordinal": 11,
            "collective": "recv",
            "collective_api": "ncclRecv",
            "collective_group_id": "ncclRecv|comm:p2p|call:7",
            "collective_communicator_id": "p2p",
            "communicator_size": 2,
            "participant_count": 1,
            "count": 4096,
            "datatype": 7,
        }
    )

    assert send_first == recv_first
    assert send_first["api"] == "ncclP2P"
    assert send_first["type"] == "nccl_collective"
    assert send_first["collective"] == "p2p"
    assert send_first["collective_api"] == "ncclP2P"
    assert send_first["world_size"] == 2
    assert send_first["communicator_size"] == 2
    assert send_first["participant_count"] == 2
    assert "member_api" not in send_first


def test_fit_from_real_trace_manifest_projects_collective_communicator_size(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    for rank in range(16):
        records: list[dict[str, object]] = []
        if rank in {0, 8}:
            records.append(
                {
                    "ts": 100 + rank,
                    "pid": rank + 1,
                    "tid": rank + 1,
                    "api": "ncclAllReduce",
                    "type": "nccl_collective",
                    "comm_id": f"local-tp-{rank}",
                    "call_idx": 7,
                    "count": 1024,
                    "datatype": 9,
                    "op": 0,
                }
            )
        _write_trace(trace_dir / f"rank_{rank}.jsonl", records)
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "communicators": {"tp-0": {"members": [0, 8]}},
                "communicator_aliases": {
                    "0": {"local-tp-0": "tp-0"},
                    "8": {"local-tp-8": "tp-0"},
                },
            }
        ),
        encoding="utf-8",
    )

    estimator = Estimator.fit_from_traces(
        str(trace_dir),
        learned_method="hybrid",
        fit_workers=1,
    )
    decision = estimator.estimate_collective_group_with_details(
        {
            "api": "ncclAllReduce",
            "type": "nccl_collective",
            "collective": "allreduce",
            "collective_api": "ncclAllReduce",
            "communicator_size": 2,
            "participant_count": 2,
            "count": 1024,
            "datatype": 9,
            "op": 0,
        }
    )

    assert decision is not None
    assert decision.provider_name == "trace_signature_stats"


def test_fit_from_traces_uses_thread_local_microsecond_deltas(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiA",
                "type": "other",
            },
            {
                "ts": 100,
                "pid": 1,
                "tid": 22,
                "mod": "libfoo.so",
                "api": "apiB",
                "type": "other",
            },
            {
                "ts": 250,
                "pid": 1,
                "tid": 22,
                "mod": "libfoo.so",
                "api": "apiC",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert estimator.estimate("apiA", "other") == pytest.approx(0.1)
    assert estimator.estimate("apiB", "other") == pytest.approx(150.0)
    assert estimator.estimate("apiC", "other") == pytest.approx(0.1)


def test_fit_from_traces_prefers_recorded_host_duration_when_available(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "end_ts": 40,
                "host_duration_us": 40.0,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiA",
                "type": "other",
            },
            {
                "ts": 1000,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiB",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert estimator.estimate("apiA", "other") == pytest.approx(40.0)


def test_fit_from_traces_uses_paper_valid_step_window(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "fidelity_windows": {
                    "0": {
                        "start_ts": 100,
                        "end_ts": 200,
                        "source": "trace_markers",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "end_ts": 1000,
                "host_duration_us": 1000.0,
                "pid": 1,
                "tid": 11,
                "api": "apiA",
                "type": "other",
            },
            {
                "ts": 150,
                "end_ts": 160,
                "host_duration_us": 10.0,
                "pid": 1,
                "tid": 11,
                "api": "apiA",
                "type": "other",
            },
            {
                "ts": 160,
                "pid": 1,
                "tid": 11,
                "api": "apiB",
                "type": "other",
            },
            {
                "ts": 300,
                "end_ts": 800,
                "host_duration_us": 500.0,
                "pid": 1,
                "tid": 11,
                "api": "apiA",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1, trace_window="step")

    assert estimator.estimate("apiA", "other") == pytest.approx(10.0)


def test_fit_from_traces_rejects_boundary_fallback_as_step_window(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "fidelity_windows": {
                    "0": {
                        "start_ts": 100,
                        "end_ts": 200,
                        "source": "boundary_fallback",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 150,
                "pid": 1,
                "tid": 11,
                "api": "apiA",
                "type": "other",
            },
        ],
    )

    with pytest.raises(ValueError, match="no paper-valid step windows"):
        Estimator.fit_from_traces(str(trace_dir), max_files=1, trace_window="step")


def test_build_learned_timing_features_keeps_collective_topology_metadata() -> None:
    features = build_learned_timing_features(
        {
            "api": "ncclAllReduce",
            "type": "nccl_collective",
            "module": "libnccl.so.2",
            "rank": 3,
            "world_size": 8,
            "collective_group_id": "ncclAllReduce|comm:comm-0|call:5",
            "collective_api": "ncclAllReduce",
            "collective_match_basis": "communicator_sequence",
            "collective_communicator_id": "comm-0",
            "collective_sequence_number": 5,
            "communicator_size": 8,
            "participant_count": 4,
            "collective_root": 0,
            "numel": 1024,
            "dtype_code": 7,
        }
    )

    assert features["collective_api"] == "ncclAllReduce"
    assert features["collective_match_basis"] == "communicator_sequence"
    assert features["collective_communicator_id"] == "comm-0"
    assert features["collective_sequence_number"] == 5
    assert features["communicator_size"] == 8
    assert features["participant_count"] == 4
    assert features["collective_root"] == 0


def test_device_work_weight_prefers_collective_participant_extent_over_world_size() -> None:
    weight = _device_work_weight(
        "ncclAllReduce",
        "nccl_collective",
        {
            "count": 1024,
            "datatype": 7,
            "world_size": 8,
            "communicator_size": 4,
            "participant_count": 4,
        },
    )

    assert weight == pytest.approx(1024 * 4 * 4)


class _EventOnlyProvider:
    name = "event_only"

    def estimate_us(self, event, percentile="p50"):
        del event, percentile
        return 11.0


class _CollectiveGroupProvider:
    name = "collective_group"
    supports_collective_group_timing = True

    def estimate_us(self, event, percentile="p50"):
        del percentile
        return 42.0 if event.get("collective_group_id") else None


def test_estimate_collective_group_with_details_prefers_group_capable_provider() -> None:
    estimator = Estimator(providers=[_EventOnlyProvider(), _CollectiveGroupProvider()])

    decision = estimator.estimate_collective_group_with_details(
        {
            "api": "ncclAllReduce",
            "type": "nccl_collective",
            "collective_group_id": "group-0",
        }
    )

    assert decision is not None
    assert decision.duration_us == pytest.approx(42.0)
    assert decision.source == "provider"
    assert decision.provider_name == "collective_group"


def test_estimate_collective_group_with_details_ignores_non_group_providers() -> None:
    estimator = Estimator(providers=[_EventOnlyProvider()])

    decision = estimator.estimate_collective_group_with_details(
        {
            "api": "ncclAllReduce",
            "type": "nccl_collective",
            "collective_group_id": "group-0",
        }
    )

    assert decision is None


class _CanonicalizingCollectiveGroupProvider:
    name = "canonicalizing_collective_group"
    supports_collective_group_timing = True

    def __init__(self) -> None:
        self.seen = None

    def estimate_us(self, event, percentile="p50"):
        del percentile
        self.seen = dict(event)
        if event.get("collective_group_id") == "ncclBroadcast|comm:comm-0|call:5":
            return 77.0
        return None


def test_estimate_collective_group_with_details_canonicalizes_alias_prefixed_group_id() -> None:
    provider = _CanonicalizingCollectiveGroupProvider()
    estimator = Estimator(providers=[provider])

    decision = estimator.estimate_collective_group_with_details(
        {
            "api": "ncclBcast",
            "type": "other",
            "collective_api": "ncclBcast",
            "collective_group_id": "ncclBcast|comm:comm-0|call:5",
        }
    )

    assert decision is not None
    assert decision.duration_us == pytest.approx(77.0)
    assert provider.seen is not None
    assert provider.seen["api"] == "ncclBroadcast"
    assert provider.seen["type"] == "nccl_collective"
    assert provider.seen["collective_api"] == "ncclBroadcast"
    assert provider.seen["collective_group_id"] == "ncclBroadcast|comm:comm-0|call:5"


def test_estimate_collective_group_with_details_normalizes_p2p_group_payload() -> None:
    provider = _CanonicalizingCollectiveGroupProvider()
    estimator = Estimator(providers=[provider])

    decision = estimator.estimate_collective_group_with_details(
        {
            "api": "ncclRecv",
            "type": "nccl_collective",
            "collective": "recv",
            "collective_api": "ncclRecv",
            "collective_group_id": "ncclRecv|comm:comm-p2p|call:5",
            "world_size": 16,
            "communicator_size": 2,
            "participant_count": 1,
        }
    )

    assert decision is None
    assert provider.seen is not None
    assert provider.seen["api"] == "ncclP2P"
    assert provider.seen["type"] == "nccl_collective"
    assert provider.seen["collective"] == "p2p"
    assert provider.seen["collective_api"] == "ncclP2P"
    assert provider.seen["world_size"] == 2
    assert provider.seen["communicator_size"] == 2
    assert provider.seen["participant_count"] == 2
    assert provider.seen["trace_world_size"] == 16
    assert provider.seen["member_api"] == "ncclRecv"
    assert provider.seen["member_collective"] == "recv"
    assert provider.seen["collective_group_id"] == "ncclP2P|comm:comm-p2p|call:5"


class _CanonicalizingEventProvider:
    name = "canonicalizing_event"

    def __init__(self) -> None:
        self.seen = None

    def estimate_us(self, event, percentile="p50"):
        del percentile
        self.seen = dict(event)
        if event.get("api") == "ncclBroadcast" and event.get("type") == "nccl_collective":
            return 33.0
        return None


def test_estimate_event_with_details_canonicalizes_alias_event_before_provider_dispatch() -> None:
    provider = _CanonicalizingEventProvider()
    estimator = Estimator(providers=[provider])

    decision = estimator.estimate_event_with_details(
        {
            "api": "ncclBcast",
            "type": "other",
            "collective_api": "ncclBcast",
        }
    )

    assert decision.duration_us == pytest.approx(33.0)
    assert decision.source == "provider"
    assert provider.seen is not None
    assert provider.seen["api"] == "ncclBroadcast"
    assert provider.seen["type"] == "nccl_collective"
    assert provider.seen["collective_api"] == "ncclBroadcast"


def test_collapse_collective_group_training_pairs_uses_group_max_target_and_stable_representative() -> None:
    group_late = {
        "api": "ncclAllReduce",
        "type": "nccl_collective",
        "collective_group_id": "group-0",
        "ts": 40,
        "rank": 2,
        "ordinal": 9,
        "communicator_size": 8,
        "participant_count": 8,
    }
    group_early = {
        "api": "ncclAllReduce",
        "type": "nccl_collective",
        "collective_group_id": "group-0",
        "ts": 10,
        "rank": 1,
        "ordinal": 3,
        "communicator_size": 8,
        "participant_count": 8,
    }
    non_group = {
        "api": "cudaLaunchKernel",
        "type": "kernel_launch",
        "ts": 90,
        "rank": 0,
        "ordinal": 10,
    }

    collapsed = _collapse_collective_group_training_pairs(
        [group_late, group_early, non_group],
        [40.0, 80.0, 5.0],
    )

    assert len(collapsed) == 2
    grouped_sample, grouped_target = collapsed[0]
    assert grouped_sample == {
        key: value
        for key, value in group_early.items()
        if key != "collective_group_id"
    }
    assert grouped_target == pytest.approx(80.0)
    assert collapsed[1] == (non_group, pytest.approx(5.0))


def test_collapse_collective_group_training_pairs_uses_comm_sequence_metadata() -> None:
    rank0 = {
        "api": "ncclAllReduce",
        "type": "nccl_collective",
        "_collective_communicator_id": "flexsim-members:0,1",
        "_collective_sequence_number": 7,
        "collective_communicator_id": "flexsim-members:0,1",
        "communicator_size": 2,
        "participant_count": 2,
        "count": 1024,
    }
    rank1 = {
        "api": "ncclAllReduce",
        "type": "nccl_collective",
        "_collective_communicator_id": "flexsim-members:0,1",
        "_collective_sequence_number": 7,
        "collective_communicator_id": "flexsim-members:0,1",
        "communicator_size": 2,
        "participant_count": 2,
        "count": 1024,
    }

    collapsed = _collapse_collective_group_training_pairs(
        [rank0, rank1],
        [10.0, 25.0],
    )

    assert len(collapsed) == 1
    sample, target = collapsed[0]
    assert target == pytest.approx(25.0)
    assert "_collective_communicator_id" not in sample
    assert "_collective_sequence_number" not in sample
    assert sample["collective_communicator_id"] == "flexsim-members:0,1"


def test_trace_learned_provider_advertises_collective_group_timing_support() -> None:
    assert TraceLearnedTimingProvider.supports_collective_group_timing is True


def test_probe_trace_learned_provider_reports_insufficient_collapsed_samples() -> None:
    status = probe_trace_learned_provider(
        [
            {
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "collective_group_id": "group-0",
                "ts": 10,
                "rank": 0,
                "ordinal": 0,
            }
        ],
        [12.0],
    )

    assert status.provider is None
    assert status.error is not None
    assert "insufficient collapsed training samples" in status.error


def test_trace_learned_nccl_collective_support_guard_falls_through_to_later_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "numpy", _TraceLearnedTestNumpy)

    class RuntimeProvider:
        def __init__(self) -> None:
            self.seen: list[dict[str, object]] = []

        def supports(self, *, api_name, event) -> bool:
            return api_name == "ncclAllReduce" and event.get("type") == "nccl_collective"

        def predict_ms(self, *, api_name, event) -> float:
            self.seen.append(dict(event))
            return 0.11367

    samples = [
        {
            "api": "ncclBroadcast",
            "type": "nccl_collective",
            "collective": "broadcast",
            "collective_api": "ncclBroadcast",
            "count": 1024 + index,
            "world_size": 8,
            "communicator_size": 8,
            "participant_count": 8,
        }
        for index in range(8)
    ]
    trace_provider = TraceLearnedTimingProvider(
        _TraceLearnedTestVectorizer(
            [
                "api=ncclBroadcast",
                "collective_api=ncclBroadcast",
                "type=nccl_collective",
            ]
        ),
        _TraceLearnedConstantLogModel(6.63),
        nccl_collective_support_keys=_trace_learned_nccl_collective_support_keys(samples),
    )

    assert trace_provider._nccl_collective_support_keys == {"ncclBroadcast"}
    assert trace_provider.estimate_us(
        {
            "api": "ncclBroadcast",
            "type": "nccl_collective",
            "collective": "broadcast",
            "collective_api": "ncclBroadcast",
            "count": 2048,
            "world_size": 8,
            "communicator_size": 8,
            "participant_count": 8,
        }
    ) == pytest.approx(6.63)
    allreduce_event = {
        "api": "ncclAllReduce",
        "type": "nccl_collective",
        "collective": "allreduce",
        "collective_api": "ncclAllReduce",
        "count": 2048,
        "world_size": 16,
        "communicator_size": 16,
        "participant_count": 16,
    }
    assert trace_provider.estimate_us(allreduce_event) is None

    runtime_provider = RuntimeProvider()
    estimator = Estimator(
        providers=[
            trace_provider,
            GPUXGBoostTimingProvider(runtime_provider, "/tmp/nonexistent-bundle"),
        ]
    )
    decision = estimator.estimate_collective_group_with_details(allreduce_event)

    assert decision is not None
    assert decision.duration_us == pytest.approx(113.67)
    assert decision.provider_name == "gpu_estimator_xgboost"
    assert runtime_provider.seen[0]["api"] == "ncclAllReduce"


def test_trace_learned_nccl_collective_support_guard_infers_legacy_vectorizer_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "numpy", _TraceLearnedTestNumpy)

    vectorizer = _TraceLearnedTestVectorizer(
        [
            "api=ncclAllReduce",
            "collective_api=ncclAllReduce",
            "type=nccl_collective",
        ]
    )
    model = _TraceLearnedConstantLogModel(77.0)
    legacy_payload = {
        "type": "trace_learned_sklearn",
        "pickle_b64": base64.b64encode(
            pickle.dumps({"vectorizer": vectorizer, "model": model})
        ).decode("ascii"),
    }

    restored = TraceLearnedTimingProvider.from_jsonable(legacy_payload)

    assert restored._nccl_collective_support_keys == {"ncclAllReduce"}
    assert restored.estimate_us(
        {
            "api": "ncclAllReduce",
            "type": "nccl_collective",
            "collective": "allreduce",
            "collective_api": "ncclAllReduce",
            "count": 1024,
            "world_size": 16,
            "communicator_size": 16,
            "participant_count": 16,
        }
    ) == pytest.approx(77.0)
    assert restored.estimate_us(
        {
            "api": "ncclBroadcast",
            "type": "nccl_collective",
            "collective": "broadcast",
            "collective_api": "ncclBroadcast",
            "count": 1024,
            "world_size": 16,
            "communicator_size": 16,
            "participant_count": 16,
        }
    ) is None


def test_trace_learned_nccl_collective_support_guard_infers_legacy_pickle_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "numpy", _TraceLearnedTestNumpy)
    provider = TraceLearnedTimingProvider(
        _TraceLearnedTestVectorizer(
            [
                "api=ncclRecv",
                "api=ncclSend",
                "collective_api=ncclP2P",
                "type=nccl_collective",
            ]
        ),
        _TraceLearnedConstantLogModel(6.63),
        nccl_collective_support_keys={"ncclP2P"},
    )
    # Legacy estimator.pkl instances bypass __init__ during unpickle, so runtime
    # estimation must infer support keys before applying the eligibility guard.
    del provider._nccl_collective_support_keys

    assert provider.estimate_us(
        {
            "api": "ncclRecv",
            "type": "nccl_collective",
            "collective": "recv",
            "collective_api": "ncclP2P",
            "count": 1024,
            "world_size": 2,
            "communicator_size": 2,
            "participant_count": 2,
        }
    ) == pytest.approx(6.63)
    assert provider._nccl_collective_support_keys == {"ncclP2P"}
    assert provider.estimate_us(
        {
            "api": "ncclAllReduce",
            "type": "nccl_collective",
            "collective": "allreduce",
            "collective_api": "ncclAllReduce",
            "count": 1024,
            "world_size": 16,
            "communicator_size": 16,
            "participant_count": 16,
        }
    ) is None


def test_trace_learned_gemm_feature_coverage_guard_falls_through_to_later_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(sys.modules, "numpy", _TraceLearnedTestNumpy)
    monkeypatch.setenv("MAYA_ENABLE_TRACE_LEARNED_GEMM_FEATURE_COVERAGE_GUARD", "1")

    class RuntimeProvider:
        def __init__(self) -> None:
            self.seen: list[dict[str, object]] = []

        def supports(self, *, api_name, event) -> bool:
            return api_name == "cublasGemmStridedBatchedEx"

        def predict_ms(self, *, api_name, event) -> float:
            self.seen.append(dict(event))
            return 0.031

    trace_provider = TraceLearnedTimingProvider(
        _TraceLearnedTestVectorizer(
            [
                "api=cublasGemmStridedBatchedEx",
                "type=blas_compute",
                "m",
                "n",
                "k",
                "lda",
                "ldb",
                "ldc",
                "batch_count",
                "compute_type",
            ]
        ),
        _TraceLearnedConstantLogModel(5.49),
    )
    event = {
        "api": "cublasGemmStridedBatchedEx",
        "type": "blas_compute",
        "m": "256",
        "n": "256",
        "k": "64",
        "lda": "256",
        "ldb": "64",
        "ldc": "256",
        "batchCount": "96",
        "strideA": "16384",
        "strideB": "16384",
        "strideC": "65536",
        "computeType": "68",
        "algo": "99",
    }

    assert trace_provider.estimate_us(event) is None

    runtime_provider = RuntimeProvider()
    estimator = Estimator(
        providers=[
            trace_provider,
            GPUXGBoostTimingProvider(runtime_provider, tmp_path),
        ]
    )
    decision = estimator.estimate_event_with_details(event)

    assert decision.duration_us == pytest.approx(31.0)
    assert decision.provider_name == "gpu_estimator_xgboost"
    assert runtime_provider.seen[0]["api"] == "cublasGemmStridedBatchedEx"
    assert runtime_provider.seen[0]["algorithm"] == "99"


def test_trace_learned_gemm_feature_coverage_guard_allows_modeled_material_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "numpy", _TraceLearnedTestNumpy)
    monkeypatch.setenv("MAYA_ENABLE_TRACE_LEARNED_GEMM_FEATURE_COVERAGE_GUARD", "1")

    provider = TraceLearnedTimingProvider(
        _TraceLearnedTestVectorizer(
            [
                "api=cublasGemmStridedBatchedEx",
                "type=blas_compute",
                "m",
                "n",
                "k",
                "lda",
                "ldb",
                "ldc",
                "batch_count",
                "stride_a",
                "stride_b",
                "stride_c",
                "compute_type",
                "dtype_code",
                "algorithm",
            ]
        ),
        _TraceLearnedConstantLogModel(5.49),
    )

    duration_us = provider.estimate_us(
        {
            "api": "cublasGemmStridedBatchedEx",
            "type": "blas_compute",
            "m": "256",
            "n": "256",
            "k": "64",
            "lda": "256",
            "ldb": "64",
            "ldc": "256",
            "batchCount": "96",
            "strideA": "16384",
            "strideB": "16384",
            "strideC": "65536",
            "computeType": "68",
            "algo": "99",
        }
    )

    assert duration_us is not None
    assert abs(duration_us - 5.49) < 1e-9


def test_gpu_xgboost_provider_advertises_collective_group_timing_and_batches() -> None:
    class RuntimeProvider:
        def __init__(self) -> None:
            self.seen: list[dict[str, object]] = []

        def supports(self, *, api_name, event) -> bool:
            return api_name == "ncclAllReduce" and event.get("type") == "nccl_collective"

        def predict_ms(self, *, api_name, event) -> float:
            self.seen.append(dict(event))
            return 0.11367

    runtime_provider = RuntimeProvider()
    provider = GPUXGBoostTimingProvider(runtime_provider, "/tmp/nonexistent-bundle")
    event = {
        "api": "ncclAllReduce",
        "type": "nccl_collective",
        "collective": "allreduce",
        "collective_api": "ncclAllReduce",
        "count": 2048,
        "world_size": 16,
        "communicator_size": 16,
        "participant_count": 16,
    }

    assert provider.supports_collective_group_timing is True
    assert provider.estimate_many_us([event]) == [pytest.approx(113.67)]

    estimator = Estimator(providers=[provider])
    decision = estimator.estimate_collective_group_with_details(event)

    assert decision is not None
    assert decision.duration_us == pytest.approx(113.67)
    assert decision.provider_name == "gpu_estimator_xgboost"
    assert runtime_provider.seen[0]["api"] == "ncclAllReduce"


def test_gpu_xgboost_provider_pickle_reloads_runtime_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "_flexsim_gpu_estimator_runtime_provider"
    module_path = tmp_path / "src" / "fake_driver_xgboost_provider.py"
    module_path.parent.mkdir()
    module_path.write_text(
        "class FakeDriverXGBoostProvider:\n"
        "    def supports(self, *, api_name, event):\n"
        "        return api_name == 'cudaLaunchKernel'\n"
        "    def predict_ms(self, *, api_name, event):\n"
        "        return 0.123\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    status = probe_gpu_estimator_provider(tmp_path)
    assert isinstance(status.provider, GPUXGBoostTimingProvider)

    payload = pickle.dumps(status.provider)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    restored = pickle.loads(payload)

    assert restored.estimate_us({"api": "cudaLaunchKernel"}) == pytest.approx(123.0)


def test_estimator_load_can_override_saved_gpu_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "estimator.json"
    model.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "type": "gpu_estimator_xgboost",
                        "bundle_dir": "/unavailable/author/path",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    portable_bundle = tmp_path / "portable-bundle"
    seen: list[Path] = []

    def fake_probe(bundle_dir):
        seen.append(Path(bundle_dir))
        return ProviderLoadStatus(provider=None, error="test sentinel")

    monkeypatch.setattr("flexsim.estimator.probe_gpu_estimator_provider", fake_probe)
    Estimator.load(str(model), gpu_estimator_bundle=portable_bundle)

    assert seen == [portable_bundle]


def test_build_learned_timing_features_keeps_dispatch_only_wrapper_signal_out_of_direct_runtime() -> None:
    features = build_learned_timing_features(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "module": "libcudart.so.12",
            "ts": 100,
            "end_ts": 140,
            "host_duration_us": 40.0,
        }
    )

    assert features["has_wrapper_timing_field"] == 1
    assert features["has_positive_observed_wrapper_runtime"] == 0
    assert features["wrapper_runtime_contract"] == "dispatch_only"
    assert "observed_wrapper_us" not in features
    assert "observed_wrapper_log2_bucket" not in features


def test_build_learned_timing_features_normalizes_alias_api_and_type() -> None:
    features = build_learned_timing_features(
        {
            "api": "ncclBcast",
            "prev_api": "cudaEventRecordWithFlags",
            "type": "other",
            "module": "libnccl.so.2",
            "rank": 0,
            "world_size": 8,
        }
    )

    assert features["api"] == "ncclBroadcast"
    assert features["type"] == "nccl_collective"
    assert features["prev_api"] == "cudaEventRecord"
    assert features["prev_api_family"] == "cuda"


def test_build_learned_timing_features_treats_alias_collective_wrapper_signal_as_dispatch_only() -> None:
    features = build_learned_timing_features(
        {
            "api": "ncclBcast",
            "type": "other",
            "module": "libnccl.so.2",
            "ts": 100,
            "end_ts": 140,
            "host_duration_us": 40.0,
        }
    )

    assert features["api"] == "ncclBroadcast"
    assert features["type"] == "nccl_collective"
    assert features["has_wrapper_timing_field"] == 1
    assert features["has_positive_observed_wrapper_runtime"] == 0
    assert features["wrapper_runtime_contract"] == "dispatch_only"
    assert "observed_wrapper_us" not in features


def test_build_learned_timing_features_normalizes_collective_api_alias_field() -> None:
    features = build_learned_timing_features(
        {
            "api": "ncclBcast",
            "type": "other",
            "collective_api": "ncclBcast",
            "rank": 0,
            "world_size": 8,
        }
    )

    assert features["api"] == "ncclBroadcast"
    assert features["collective_api"] == "ncclBroadcast"


def test_build_learned_timing_features_keeps_async_memcpy_dispatch_only_signal_out_of_direct_runtime() -> None:
    features = build_learned_timing_features(
        {
            "api": "cudaMemcpyAsync",
            "type": "mem_copy",
            "module": "libcudart.so.12",
            "ts": 100,
            "end_ts": 140,
            "host_duration_us": 40.0,
            "wrapper_runtime_contract": "dispatch_only",
            "bytes": 4096,
            "kind": 3,
        }
    )

    assert features["has_wrapper_timing_field"] == 1
    assert features["has_positive_observed_wrapper_runtime"] == 0
    assert features["wrapper_runtime_contract"] == "dispatch_only"
    assert "observed_wrapper_us" not in features
    assert "observed_wrapper_log2_bucket" not in features


def test_build_learned_timing_features_keeps_async_alloc_dispatch_only_signal_out_of_direct_runtime() -> None:
    features = build_learned_timing_features(
        {
            "api": "cudaMallocAsync",
            "type": "mem_alloc",
            "module": "libcudart.so.12",
            "ts": 100,
            "end_ts": 140,
            "host_duration_us": 40.0,
            "wrapper_runtime_contract": "dispatch_only",
            "bytes": 4096,
            "stream_id": 7,
        }
    )

    assert features["has_wrapper_timing_field"] == 1
    assert features["has_positive_observed_wrapper_runtime"] == 0
    assert features["wrapper_runtime_contract"] == "dispatch_only"
    assert "observed_wrapper_us" not in features
    assert "observed_wrapper_log2_bucket" not in features


def test_build_learned_timing_features_keeps_event_record_dispatch_only_signal_out_of_direct_runtime() -> None:
    features = build_learned_timing_features(
        {
            "api": "cudaEventRecord",
            "type": "stream_op",
            "module": "libcudart.so.12",
            "ts": 100,
            "end_ts": 140,
            "host_duration_us": 40.0,
            "wrapper_runtime_contract": "dispatch_only",
            "event_id": 11,
            "stream_id": 7,
        }
    )

    assert features["has_wrapper_timing_field"] == 1
    assert features["has_positive_observed_wrapper_runtime"] == 0
    assert features["wrapper_runtime_contract"] == "dispatch_only"
    assert "observed_wrapper_us" not in features
    assert "observed_wrapper_log2_bucket" not in features


def test_build_learned_timing_features_keeps_cublas_set_stream_dispatch_only_signal_out_of_direct_runtime() -> None:
    features = build_learned_timing_features(
        {
            "api": "cublasSetStream_v2",
            "type": "stream_op",
            "module": "libcublas.so.12",
            "ts": 100,
            "end_ts": 140,
            "host_duration_us": 40.0,
            "wrapper_runtime_contract": "dispatch_only",
            "handle_id": 11,
            "stream_id": 7,
        }
    )

    assert features["has_wrapper_timing_field"] == 1
    assert features["has_positive_observed_wrapper_runtime"] == 0
    assert features["wrapper_runtime_contract"] == "dispatch_only"
    assert "observed_wrapper_us" not in features
    assert "observed_wrapper_log2_bucket" not in features


def test_build_learned_timing_features_uses_direct_runtime_contract_for_wrapper_signal() -> None:
    features = build_learned_timing_features(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "module": "libcudart.so.12",
            "ts": 100,
            "end_ts": 140,
            "host_duration_us": 40.0,
            "wrapper_runtime_contract": "direct_runtime",
        }
    )

    assert features["has_wrapper_timing_field"] == 1
    assert features["has_positive_observed_wrapper_runtime"] == 1
    assert features["wrapper_runtime_contract"] == "direct_runtime"
    assert features["observed_wrapper_us"] == pytest.approx(40.0)
    assert features["observed_wrapper_log2_bucket"] >= 5


def test_build_learned_timing_features_falls_back_to_end_ts_delta_for_direct_runtime_contract() -> None:
    features = build_learned_timing_features(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "module": "libcudart.so.12",
            "ts": 100,
            "end_ts": 132,
            "wrapper_runtime_contract": "direct_runtime",
        }
    )

    assert features["has_wrapper_timing_field"] == 1
    assert features["has_positive_observed_wrapper_runtime"] == 1
    assert features["wrapper_runtime_contract"] == "direct_runtime"
    assert features["observed_wrapper_us"] == pytest.approx(32.0)
    assert features["observed_wrapper_log2_bucket"] >= 5


def test_build_learned_timing_features_prefers_explicit_direct_runtime_field() -> None:
    features = build_learned_timing_features(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "module": "libcudart.so.12",
            "ts": 100,
            "end_ts": 140,
            "host_duration_us": 40.0,
            "direct_runtime_us": 96.0,
        }
    )

    assert features["has_wrapper_timing_field"] == 1
    assert features["has_positive_observed_wrapper_runtime"] == 1
    assert features["wrapper_runtime_contract"] == "direct_runtime"
    assert features["observed_wrapper_us"] == pytest.approx(96.0)
    assert features["observed_wrapper_log2_bucket"] >= 6


def test_trace_signature_provider_uses_collective_topology_metadata() -> None:
    provider = TraceSignatureTimingProvider.fit(
        [
            {
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "module": "libnccl.so.2",
                "collective": "allreduce",
                "collective_api": "ncclAllReduce",
                "count": 1024,
                "world_size": 8,
                "communicator_size": 4,
                "participant_count": 4,
            },
            {
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "module": "libnccl.so.2",
                "collective": "allreduce",
                "collective_api": "ncclAllReduce",
                "count": 1024,
                "world_size": 8,
                "communicator_size": 8,
                "participant_count": 8,
            },
        ],
        [40.0, 80.0],
    )

    assert provider is not None
    assert provider.estimate_us(
        {
            "api": "ncclAllReduce",
            "type": "nccl_collective",
            "module": "libnccl.so.2",
            "collective": "allreduce",
            "collective_api": "ncclAllReduce",
            "count": 1024,
            "world_size": 8,
            "communicator_size": 4,
            "participant_count": 4,
        }
    ) == pytest.approx(40.0)
    assert provider.estimate_us(
        {
            "api": "ncclAllReduce",
            "type": "nccl_collective",
            "module": "libnccl.so.2",
            "collective": "allreduce",
            "collective_api": "ncclAllReduce",
            "count": 1024,
            "world_size": 8,
            "communicator_size": 8,
            "participant_count": 8,
        }
    ) == pytest.approx(80.0)


def test_trace_signature_provider_keeps_legacy_collective_signature_fallback() -> None:
    provider = TraceSignatureTimingProvider.fit(
        [
            {
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "module": "libnccl.so.2",
                "collective": "allreduce",
                "count": 1024,
                "world_size": 8,
            }
        ],
        [55.0],
    )

    assert provider is not None
    assert provider.estimate_us(
        {
            "api": "ncclAllReduce",
            "type": "nccl_collective",
            "module": "libnccl.so.2",
            "collective": "allreduce",
            "collective_api": "ncclAllReduce",
            "count": 1024,
            "world_size": 8,
            "communicator_size": 8,
            "participant_count": 8,
        }
    ) == pytest.approx(55.0)


def test_trace_signature_provider_uses_collective_group_max_target() -> None:
    provider = TraceSignatureTimingProvider.fit(
        [
            {
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "module": "libnccl.so.2",
                "collective": "allreduce",
                "collective_api": "ncclAllReduce",
                "count": 1024,
                "world_size": 8,
                "communicator_size": 8,
                "participant_count": 8,
                "collective_group_id": "group-0",
            },
            {
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "module": "libnccl.so.2",
                "collective": "allreduce",
                "collective_api": "ncclAllReduce",
                "count": 1024,
                "world_size": 8,
                "communicator_size": 8,
                "participant_count": 8,
                "collective_group_id": "group-0",
            },
        ],
        [40.0, 80.0],
    )

    assert provider is not None
    assert provider.estimate_us(
        {
            "api": "ncclAllReduce",
            "type": "nccl_collective",
            "module": "libnccl.so.2",
            "collective": "allreduce",
            "collective_api": "ncclAllReduce",
            "count": 1024,
            "world_size": 8,
            "communicator_size": 8,
            "participant_count": 8,
        }
    ) == pytest.approx(80.0)


def test_trace_signature_provider_merges_alias_prefixed_collective_group_ids() -> None:
    provider = TraceSignatureTimingProvider.fit(
        [
            {
                "api": "ncclBcast",
                "type": "other",
                "module": "libnccl.so.2",
                "collective": "broadcast",
                "collective_api": "ncclBcast",
                "count": 1024,
                "world_size": 8,
                "communicator_size": 8,
                "participant_count": 8,
                "collective_group_id": "ncclBcast|comm:comm-0|call:5",
            },
            {
                "api": "ncclBroadcast",
                "type": "nccl_collective",
                "module": "libnccl.so.2",
                "collective": "broadcast",
                "collective_api": "ncclBroadcast",
                "count": 1024,
                "world_size": 8,
                "communicator_size": 8,
                "participant_count": 8,
                "collective_group_id": "ncclBroadcast|comm:comm-0|call:5",
            },
        ],
        [40.0, 80.0],
    )

    assert provider is not None
    assert provider.estimate_us(
        {
            "api": "ncclBroadcast",
            "type": "nccl_collective",
            "module": "libnccl.so.2",
            "collective": "broadcast",
            "collective_api": "ncclBroadcast",
            "count": 1024,
            "world_size": 8,
            "communicator_size": 8,
            "participant_count": 8,
        }
    ) == pytest.approx(80.0)


def test_fit_from_traces_tracks_prev_api_per_thread(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiA",
                "type": "other",
            },
            {
                "ts": 5,
                "pid": 1,
                "tid": 22,
                "mod": "libfoo.so",
                "api": "apiX",
                "type": "other",
            },
            {
                "ts": 10,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel": "apiB",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    kernel_sample = next(
        sample for sample in estimator._feature_samples if sample.get("kernel") == "apiB"
    )
    assert kernel_sample["prev_api"] == "apiA"


def test_fit_from_traces_keeps_kernel_launch_target_when_it_is_the_only_runtime_signal(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            },
            {
                "ts": 5000,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfter",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert estimator.estimate("cudaLaunchKernel", "kernel_launch") == pytest.approx(5000.0)


def test_fit_from_traces_ignores_zero_host_duration_for_async_device_work(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "end_ts": 0,
                "host_duration_us": 0.0,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            },
            {
                "ts": 5000,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfter",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert estimator.estimate("cudaLaunchKernel", "kernel_launch") == pytest.approx(5000.0)


def test_fit_from_traces_keeps_zero_host_duration_for_non_device_work(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "end_ts": 0,
                "host_duration_us": 0.0,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaEventQuery",
                "type": "stream_op",
            },
            {
                "ts": 5000,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfter",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert estimator.estimate("cudaEventQuery", "stream_op") == pytest.approx(0.0)


def test_fit_from_traces_trains_learned_features_only_on_modeled_ops(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaEventQuery",
                "type": "stream_op",
            },
            {
                "ts": 5,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel": "modeled_kernel",
            },
            {
                "ts": 10,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaGetDevice",
                "type": "context_op",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert ("cudaEventQuery", "stream_op") in estimator._stats
    assert [sample.get("api") for sample in estimator._feature_samples] == [
        "cudaLaunchKernel"
    ]


def test_fit_from_traces_uses_end_ts_for_blocking_wrapper_runtime(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 100,
                "end_ts": 350,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaStreamSynchronize",
                "type": "stream_op",
            },
            {
                "ts": 1000,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfter",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert estimator.estimate("cudaStreamSynchronize", "stream_op") == pytest.approx(250.0)


def test_fit_from_traces_does_not_cap_observed_wrapper_runtime_for_context_ops(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "host_duration_us": 400.0,
                "pid": 1,
                "tid": 11,
                "mod": "libnccl.so.2",
                "api": "ncclCommInitRankConfig",
                "type": "other",
            },
            {
                "ts": 1000,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfter",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert estimator.estimate("ncclCommInitRankConfig", "other") == pytest.approx(400.0)


def test_fit_from_traces_uses_observed_wrapper_runtime_for_nccl_comm_query_apis(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "host_duration_us": 275.0,
                "pid": 1,
                "tid": 11,
                "mod": "libnccl.so.2",
                "api": "ncclCommCount",
                "type": "other",
            },
            {
                "ts": 1000,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfter",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert estimator.estimate("ncclCommCount", "other") == pytest.approx(275.0)


def test_fit_from_traces_uses_normalized_op_type_for_cublas_control_apis(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libcublas.so.12",
                "api": "cublasSetStream_v2",
                "type": "blas_compute",
                "handle_id": "7",
                "stream_id": "3",
            },
            {
                "ts": 50,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaGetDevice",
                "type": "context_op",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert ("cublasSetStream_v2", "stream_op") in estimator._stats
    assert ("cublasSetStream_v2", "blas_compute") not in estimator._stats


def test_fit_from_traces_uses_normalized_op_type_for_extended_nccl_collectives(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libnccl.so.2",
                "api": "ncclReduce",
                "type": "other",
                "count": 16,
            },
            {
                "ts": 50,
                "pid": 1,
                "tid": 11,
                "mod": "libnccl.so.2",
                "api": "ncclAllToAll",
                "type": "other",
                "count": 16,
            },
            {
                "ts": 90,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfter",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert ("ncclReduce", "nccl_collective") in estimator._stats
    assert ("ncclAllToAll", "nccl_collective") in estimator._stats
    assert ("ncclReduce", "other") not in estimator._stats
    assert ("ncclAllToAll", "other") not in estimator._stats


def test_fit_from_traces_normalizes_nccl_bcast_alias_into_canonical_collective_key(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libnccl.so.2",
                "api": "ncclBcast",
                "type": "other",
                "end_ts": 40,
                "host_duration_us": 40.0,
            },
            {
                "ts": 100,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfter",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert ("ncclBroadcast", "nccl_collective") in estimator._stats
    assert ("ncclBcast", "other") not in estimator._stats
    assert any(
        sample.get("api") == "ncclBroadcast"
        and sample.get("type") == "nccl_collective"
        and sample.get("wrapper_runtime_contract") == "dispatch_only"
        for sample in estimator._feature_samples
    )

    decision = estimator.estimate_event_with_details(
        {
            "api": "ncclBcast",
            "type": "other",
        }
    )
    assert decision.source == "api_stats"
    assert decision.duration_us == pytest.approx(
        estimator.estimate("ncclBroadcast", "nccl_collective")
    )


def test_estimate_normalizes_alias_api_and_type_for_stats_lookup() -> None:
    estimator = Estimator()
    estimator._stats[("ncclBroadcast", "nccl_collective")] = {
        "p50": 12.0,
        "mean": 12.0,
        "p95": 12.0,
        "count": 1,
    }
    estimator._type_stats["nccl_collective"] = {
        "p50": 9.0,
        "mean": 9.0,
        "p95": 9.0,
        "count": 1,
    }

    assert estimator.estimate("ncclBcast", "other") == pytest.approx(12.0)
    assert estimator.estimate("ncclBroadcast", "other") == pytest.approx(12.0)
    assert estimator.estimate_ns("ncclBcast", "other") == 12_000


def test_merge_fit_partial_canonicalizes_alias_api_keys() -> None:
    estimator = Estimator()

    estimator._merge_fit_partial(
        {
            "raw": {
                "ncclBcast::other": [10.0],
                "ncclBroadcast::nccl_collective": [20.0],
            }
        }
    )

    assert ("ncclBcast", "other") not in estimator._raw
    assert estimator._raw[("ncclBroadcast", "nccl_collective")] == [10.0, 20.0]


def test_load_canonicalizes_alias_api_stats_keys_and_prefers_canonical_entries(
    tmp_path: Path,
) -> None:
    payload = {
        "api_stats": {
            "ncclBcast::other": {
                "p50": 10.0,
                "mean": 10.0,
                "p95": 10.0,
                "count": 1,
            },
            "ncclBroadcast::nccl_collective": {
                "p50": 20.0,
                "mean": 20.0,
                "p95": 20.0,
                "count": 8,
            },
        },
        "type_stats": {
            "nccl_collective": {
                "p50": 7.0,
                "mean": 7.0,
                "p95": 7.0,
                "count": 1,
            }
        },
        "global_p50_us": 1.0,
    }
    path = tmp_path / "estimator.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    estimator = Estimator.load(str(path))

    assert ("ncclBroadcast", "nccl_collective") in estimator._stats
    assert ("ncclBcast", "other") not in estimator._stats
    assert estimator.estimate("ncclBcast", "other") == pytest.approx(20.0)


def test_fit_from_traces_ignores_trace_marker_files_when_counting_world_size(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiA",
                "type": "other",
            },
            {
                "ts": 5,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            },
        ],
    )
    _write_trace(
        trace_dir / "rank_0.markers.jsonl",
        [
            {
                "start_ns": 0,
                "end_ns": 1000,
                "rank": 0,
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert estimator._feature_samples
    assert estimator._feature_samples[0]["world_size"] == 1


def test_operator_family_summary_prioritizes_time_heavy_families(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libcublas.so.12",
                "api": "cublasGemmEx",
                "type": "blas_compute",
                "m": 256,
                "n": 256,
                "k": 256,
            },
            {
                "ts": 1000,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel": "void at::native::elementwise_kernel<float>(...)",
            },
            {
                "ts": 1100,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfter",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    families = estimator.operator_family_summary(limit=5)

    assert families[0]["family"] == "gemm_family"
    assert families[0]["time_share"] > families[1]["time_share"]


class _KernelOnlyProvider:
    name = "kernel_only"

    def estimate_us(
        self,
        event: dict[str, object],
        percentile: str = "p50",
    ) -> float | None:
        del percentile
        if str(event.get("api")) == "cudaLaunchKernel":
            return 7.0
        return None


def test_provider_coverage_summary_reports_weighted_time_share(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel": "void flash_fwd_kernel(...)",
            },
            {
                "ts": 900,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiOther",
                "type": "other",
            },
            {
                "ts": 1000,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiThird",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(
        str(trace_dir),
        max_files=1,
        providers=[_KernelOnlyProvider()],
    )
    coverage = estimator.provider_coverage_summary("kernel_only", limit=5)

    assert coverage["matched_provider_names"] == ["kernel_only"]
    assert coverage["covered_time_share"] > 0.8
    assert coverage["top_covered_families"][0]["family"] == "flash_attention"


def test_kernel_launch_metadata_summary_tracks_missing_real_trace_payload(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "host_duration_us": 12.0,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "stream_id": "0",
            },
            {
                "ts": 50,
                "host_duration_us": 15.0,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel": "void flash_fwd_kernel(...)",
                "grid_x": 16,
                "block_x": 128,
                "shared_mem": 64,
                "stream_id": "7",
            },
            {
                "ts": 100,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfter",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    summary = estimator.kernel_launch_metadata_summary()

    assert summary["total_kernel_launches"] == 2
    assert summary["with_kernel_name"] == 1
    assert summary["with_launch_shape"] == 1
    assert summary["with_stream_id"] == 2
    assert summary["with_host_duration"] == 2
    assert summary["missing_kernel_name_count"] == 1
    assert summary["missing_launch_shape_count"] == 1


def test_fit_from_traces_preserves_observed_wrapper_signal_for_async_modeled_device_work(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "end_ts": 12,
                "host_duration_us": 12.0,
                "wrapper_runtime_contract": "direct_runtime",
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            },
            {
                "ts": 100,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfter",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    kernel_sample = next(
        sample for sample in estimator._feature_samples if sample.get("api") == "cudaLaunchKernel"
    )

    assert kernel_sample["has_wrapper_timing_field"] == 1
    assert kernel_sample["has_positive_observed_wrapper_runtime"] == 1
    assert kernel_sample["observed_wrapper_us"] == pytest.approx(12.0)
    assert kernel_sample["observed_wrapper_log2_bucket"] >= 3


def test_transparent_profiling_summary_reports_async_modeled_wrapper_coverage(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "end_ts": 12,
                "host_duration_us": 12.0,
                "wrapper_runtime_contract": "direct_runtime",
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            },
            {
                "ts": 100,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfterA",
                "type": "other",
            },
            {
                "ts": 200,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            },
            {
                "ts": 300,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfterB",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    summary = estimator.transparent_profiling_summary()

    assert summary["modeled_event_count"] == 2
    assert summary["modeled_event_with_wrapper_timing_field_count"] == 1
    assert summary["modeled_event_with_explicit_direct_runtime_field_count"] == 0
    assert summary["modeled_event_with_direct_wrapper_runtime_count"] == 1
    assert summary["modeled_event_with_dispatch_only_wrapper_contract_count"] == 0
    assert summary["modeled_event_with_direct_runtime_contract_count"] == 1
    assert summary["modeled_explicit_direct_runtime_field_share"] == pytest.approx(0.0)
    assert summary["modeled_direct_wrapper_runtime_share"] == pytest.approx(0.5)
    assert summary["modeled_dispatch_only_wrapper_contract_share"] == pytest.approx(0.0)
    assert summary["modeled_direct_runtime_contract_share"] == pytest.approx(0.5)
    assert summary["async_modeled_device_event_count"] == 2
    assert summary["async_modeled_device_event_with_wrapper_timing_field_count"] == 1
    assert summary["async_modeled_device_event_with_explicit_direct_runtime_field_count"] == 0
    assert summary["async_modeled_device_event_with_direct_wrapper_runtime_count"] == 1
    assert summary["async_modeled_device_event_with_dispatch_only_wrapper_contract_count"] == 0
    assert summary["async_modeled_device_event_with_direct_runtime_contract_count"] == 1
    assert summary["async_modeled_device_explicit_direct_runtime_field_share"] == pytest.approx(0.0)
    assert summary["async_modeled_device_direct_wrapper_runtime_share"] == pytest.approx(0.5)
    assert summary["async_modeled_device_dispatch_only_wrapper_contract_share"] == pytest.approx(0.0)
    assert summary["async_modeled_device_direct_runtime_contract_share"] == pytest.approx(0.5)
    assert summary["modeled_target_time_share_with_direct_wrapper_runtime"] == pytest.approx(0.5)
    assert summary["async_modeled_device_target_time_share_with_direct_wrapper_runtime"] == pytest.approx(0.5)
    assert summary["modeled_observed_wrapper_us_total"] == pytest.approx(12.0)
    assert summary["async_modeled_device_observed_wrapper_us_total"] == pytest.approx(12.0)


def test_transparent_profiling_summary_treats_dispatch_only_wrapper_signal_as_non_direct_runtime(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace_dispatch_only"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "end_ts": 12,
                "host_duration_us": 12.0,
                "wrapper_runtime_contract": "dispatch_only",
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            },
            {
                "ts": 100,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfterA",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    kernel_sample = next(
        sample for sample in estimator._feature_samples if sample.get("api") == "cudaLaunchKernel"
    )
    summary = estimator.transparent_profiling_summary()

    assert kernel_sample["has_wrapper_timing_field"] == 1
    assert kernel_sample["has_positive_observed_wrapper_runtime"] == 0
    assert kernel_sample["wrapper_runtime_contract"] == "dispatch_only"
    assert "observed_wrapper_us" not in kernel_sample
    assert summary["modeled_event_with_wrapper_timing_field_count"] == 1
    assert summary["modeled_event_with_explicit_direct_runtime_field_count"] == 0
    assert summary["modeled_event_with_direct_wrapper_runtime_count"] == 0
    assert summary["modeled_event_with_dispatch_only_wrapper_contract_count"] == 1
    assert summary["modeled_event_with_direct_runtime_contract_count"] == 0
    assert summary["async_modeled_device_event_with_wrapper_timing_field_count"] == 1
    assert summary["async_modeled_device_event_with_explicit_direct_runtime_field_count"] == 0
    assert summary["async_modeled_device_event_with_direct_wrapper_runtime_count"] == 0
    assert summary["async_modeled_device_event_with_dispatch_only_wrapper_contract_count"] == 1
    assert summary["async_modeled_device_event_with_direct_runtime_contract_count"] == 0
    assert summary["modeled_explicit_direct_runtime_field_share"] == pytest.approx(0.0)
    assert summary["modeled_dispatch_only_wrapper_contract_share"] == pytest.approx(1.0)
    assert summary["modeled_direct_runtime_contract_share"] == pytest.approx(0.0)
    assert summary["async_modeled_device_explicit_direct_runtime_field_share"] == pytest.approx(0.0)
    assert summary["async_modeled_device_dispatch_only_wrapper_contract_share"] == pytest.approx(1.0)
    assert summary["async_modeled_device_direct_runtime_contract_share"] == pytest.approx(0.0)
    assert summary["modeled_observed_wrapper_us_total"] == pytest.approx(0.0)
    assert summary["async_modeled_device_observed_wrapper_us_total"] == pytest.approx(0.0)


def test_transparent_profiling_summary_counts_explicit_direct_runtime_field_without_raw_wrapper_delta(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace_explicit_direct_runtime"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "direct_runtime_us": 24.0,
            },
            {
                "ts": 100,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfterA",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    kernel_sample = next(
        sample for sample in estimator._feature_samples if sample.get("api") == "cudaLaunchKernel"
    )
    summary = estimator.transparent_profiling_summary()

    assert kernel_sample["has_wrapper_timing_field"] == 1
    assert kernel_sample["has_positive_observed_wrapper_runtime"] == 1
    assert kernel_sample["wrapper_runtime_contract"] == "direct_runtime"
    assert kernel_sample["observed_wrapper_us"] == pytest.approx(24.0)
    assert summary["modeled_event_with_wrapper_timing_field_count"] == 1
    assert summary["modeled_event_with_explicit_direct_runtime_field_count"] == 1
    assert summary["modeled_event_with_direct_wrapper_runtime_count"] == 1
    assert summary["modeled_event_with_dispatch_only_wrapper_contract_count"] == 0
    assert summary["modeled_event_with_direct_runtime_contract_count"] == 1
    assert summary["async_modeled_device_event_with_wrapper_timing_field_count"] == 1
    assert summary["async_modeled_device_event_with_explicit_direct_runtime_field_count"] == 1
    assert summary["async_modeled_device_event_with_direct_wrapper_runtime_count"] == 1
    assert summary["async_modeled_device_event_with_dispatch_only_wrapper_contract_count"] == 0
    assert summary["async_modeled_device_event_with_direct_runtime_contract_count"] == 1
    assert summary["modeled_explicit_direct_runtime_field_share"] == pytest.approx(1.0)
    assert summary["async_modeled_device_explicit_direct_runtime_field_share"] == pytest.approx(1.0)
    assert summary["modeled_observed_wrapper_us_total"] == pytest.approx(24.0)
    assert summary["async_modeled_device_observed_wrapper_us_total"] == pytest.approx(24.0)


def test_transparent_profiling_summary_counts_observed_runtime_us_async_contract(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace_observed_runtime"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "wrapper_runtime_contract": "async_runtime",
                "observed_runtime_us": 33.0,
            },
            {
                "ts": 100,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfterA",
                "type": "other",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    kernel_sample = next(
        sample for sample in estimator._feature_samples if sample.get("api") == "cudaLaunchKernel"
    )
    summary = estimator.transparent_profiling_summary()

    assert kernel_sample["wrapper_runtime_contract"] == "direct_runtime"
    assert kernel_sample["observed_wrapper_us"] == pytest.approx(33.0)
    assert summary["async_modeled_device_event_with_direct_runtime_contract_count"] == 1
    assert summary["async_modeled_device_observed_wrapper_us_total"] == pytest.approx(33.0)


def test_fit_from_traces_attributes_stream_sync_wait_to_pending_kernel(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "stream_id": "7",
            },
            {
                "ts": 10,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaStreamSynchronize",
                "type": "stream_op",
                "stream_id": "7",
            },
            {
                "ts": 510,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaGetDevice",
                "type": "context_op",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert estimator.estimate("cudaLaunchKernel", "kernel_launch") == pytest.approx(410.0)
    assert estimator.estimate("cudaStreamSynchronize", "stream_op") == pytest.approx(100.0)


def test_fit_from_traces_propagates_event_wait_dependencies_across_streams(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "stream_id": "stream-a",
            },
            {
                "ts": 10,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaEventRecord",
                "type": "stream_op",
                "stream_id": "stream-a",
                "event_id": "evt-1",
            },
            {
                "ts": 20,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaStreamWaitEvent",
                "type": "stream_op",
                "stream_id": "stream-b",
                "event_id": "evt-1",
            },
            {
                "ts": 30,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaStreamSynchronize",
                "type": "stream_op",
                "stream_id": "stream-b",
            },
            {
                "ts": 530,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaGetDevice",
                "type": "context_op",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)

    assert estimator.estimate("cudaLaunchKernel", "kernel_launch") == pytest.approx(410.0)
    assert estimator.estimate("cudaStreamSynchronize", "stream_op") == pytest.approx(100.0)


class _Provider:
    name = "unit_test_provider"

    def estimate_us(self, event, percentile="p50"):
        del event, percentile
        return 7.0


def test_estimate_event_with_details_reports_provider_source() -> None:
    estimator = Estimator(providers=[_Provider()])

    decision = estimator.estimate_event_with_details(
        {"api": "cudaLaunchKernel", "type": "kernel_launch"}
    )

    assert decision.duration_us == 7.0
    assert decision.source == "provider"
    assert decision.provider_name == "unit_test_provider"
    assert decision.calibrated is True


def test_estimate_event_with_details_reports_api_stats_source() -> None:
    estimator = Estimator()
    estimator._stats[("apiA", "other")] = {"p50": 12.0, "mean": 12.0, "p95": 12.0, "count": 1}

    decision = estimator.estimate_event_with_details({"api": "apiA", "type": "other"})

    assert decision.duration_us == 12.0
    assert decision.source == "api_stats"
    assert decision.calibrated is True


def test_estimate_event_with_details_reports_type_stats_source() -> None:
    estimator = Estimator()
    estimator._type_stats["other"] = {"p50": 9.0, "mean": 9.0, "p95": 9.0, "count": 1}

    decision = estimator.estimate_event_with_details({"api": "apiA", "type": "other"})

    assert decision.duration_us == 9.0
    assert decision.source == "type_stats"
    assert decision.calibrated is True


def test_estimate_event_with_details_reports_global_fallback_source() -> None:
    estimator = Estimator()

    decision = estimator.estimate_event_with_details({"api": "apiA", "type": "other"})

    assert decision.duration_us == 1.0
    assert decision.source == "global_fallback"
    assert decision.calibrated is False


def test_fit_from_traces_gpu_xgboost_mode_fails_closed_when_provider_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libcublas.so.12",
                "api": "cublasGemmEx",
                "type": "blas_compute",
                "m": 128,
                "n": 128,
                "k": 128,
            },
            {
                "ts": 10,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaGetDevice",
                "type": "context_op",
            },
        ],
    )

    monkeypatch.setattr(
        "flexsim.estimator.probe_gpu_estimator_provider",
        lambda bundle_dir: ProviderLoadStatus(provider=None, error="xgboost missing"),
    )

    with pytest.raises(RuntimeError, match="gpu_xgboost estimator requested but unavailable"):
        Estimator.fit_from_traces(str(trace_dir), max_files=1, learned_method="gpu_xgboost")


def test_default_gpu_estimator_bundle_points_inside_repo() -> None:
    assert DEFAULT_GPU_ESTIMATOR_BUNDLE.exists()


def test_canonicalize_gpu_estimator_event_prefers_operand_dtype_for_cublas() -> None:
    payload = canonicalize_gpu_estimator_event(
        {
            "api": "cublasGemmEx",
            "dtype_code": "68",
            "Atype": "14",
            "Btype": "14",
            "Ctype": "14",
            "computeType": "68",
        }
    )

    assert payload["dtype_code"] == 2
    assert payload["Atype"] == 2
    assert payload["Btype"] == 2
    assert payload["Ctype"] == 2
    assert payload["compute_type"] == 1


def test_build_learned_timing_features_canonicalizes_direct_cublas_gemm_aliases() -> None:
    features = build_learned_timing_features(
        {
            "api": "cublasGemmStridedBatchedEx",
            "type": "blas_compute",
            "m": "256",
            "n": "128",
            "k": "64",
            "lda": "256",
            "ldb": "64",
            "ldc": "256",
            "batchCount": "4",
            "strideA": "16384",
            "strideB": "8192",
            "strideC": "32768",
            "computeType": "68",
            "algo": "99",
        }
    )

    assert features["lda"] == 256
    assert features["ldb"] == 64
    assert features["ldc"] == 256
    assert features["batch_count"] == 4
    assert features["stride_a"] == 16384
    assert features["stride_b"] == 8192
    assert features["stride_c"] == 32768
    assert features["compute_type"] == 68
    assert features["algorithm"] == 99
    assert "batchCount" not in features
    assert "strideA" not in features
    assert "strideB" not in features
    assert "strideC" not in features
    assert "algo" not in features


def test_build_learned_timing_features_ignores_missing_gemm_fields_for_non_gemm() -> None:
    features = build_learned_timing_features(
        canonicalize_gpu_estimator_event(
            {
                "api": "cudaEventRecord",
                "type": "stream_op",
            }
        )
    )

    assert features["api"] == "cudaEventRecord"
    assert "lda" not in features
    assert "stride_a" not in features
    assert "algorithm" not in features


def test_canonicalize_gpu_estimator_event_normalizes_nccl_dtype_codes() -> None:
    payload = canonicalize_gpu_estimator_event(
        {
            "api": "ncclAllReduce",
            "count": "8",
            "datatype": "7",
            "op": "0",
        }
    )

    assert payload["numel"] == "8"
    assert payload["dtype_code"] == 2
    assert payload["reduction"] == "sum"


def test_canonicalize_gpu_estimator_event_normalizes_nccl_bcast_alias() -> None:
    payload = canonicalize_gpu_estimator_event(
        {
            "api": "ncclBcast",
            "type": "other",
            "collective_api": "ncclBcast",
            "prev_api": "cudaEventRecordWithFlags",
            "count": "8",
            "datatype": "7",
            "root": "0",
            "nranks": "8",
        }
    )

    assert payload["api"] == "ncclBroadcast"
    assert payload["type"] == "nccl_collective"
    assert payload["collective_api"] == "ncclBroadcast"
    assert payload["prev_api"] == "cudaEventRecord"
    assert payload["numel"] == "8"
    assert payload["dtype_code"] == 2
    assert payload["collective"] == "broadcast"
    assert payload["world_size"] == "8"
    assert payload["communicator_size"] == "8"
    assert payload["participant_count"] == "8"
    assert payload["collective_root"] == "0"


def test_canonicalize_gpu_estimator_event_normalizes_nccl_p2p_metadata() -> None:
    payload = canonicalize_gpu_estimator_event(
        {
            "api": "ncclSend",
            "count": "8",
            "datatype": "7",
            "nranks": "8",
            "peer": "3",
        }
    )

    assert payload["numel"] == "8"
    assert payload["dtype_code"] == 2
    assert payload["collective"] == "send"
    assert payload["collective_api"] == "ncclP2P"
    assert payload["communicator_size"] == "8"
    assert payload["participant_count"] == 2


def test_trace_signature_timing_provider_matches_normalized_event_signatures() -> None:
    provider = TraceSignatureTimingProvider.fit(
        samples=[
            {
                "api": "cublasGemmEx",
                "type": "blas_compute",
                "m": 1536,
                "n": 3072,
                "k": 512,
                "Atype": 14,
                "Btype": 14,
                "Ctype": 14,
                "computeType": 68,
            },
            {
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "count": 8,
                "datatype": 7,
                "op": 0,
                "world_size": 8,
            },
        ]
        * 4,
        targets_us=[1200.0, 80.0] * 4,
    )

    assert provider is not None
    assert provider.estimate_us(
        {
            "api": "cublasGemmEx",
            "type": "blas_compute",
            "m": 1536,
            "n": 3072,
            "k": 512,
            "dtype_code": 68,
            "Atype": 14,
            "Btype": 14,
            "Ctype": 14,
            "computeType": 68,
        }
    ) == pytest.approx(1200.0)
    assert provider.estimate_us(
        {
            "api": "ncclAllReduce",
            "type": "nccl_collective",
            "numel": 8,
            "datatype": 7,
            "op": 0,
            "world_size": 8,
        }
    ) == pytest.approx(80.0)


def test_trace_signature_timing_provider_keys_include_gemm_layout_and_algorithm() -> None:
    provider = TraceSignatureTimingProvider.fit(
        samples=[
            {
                "api": "cublasGemmStridedBatchedEx",
                "type": "blas_compute",
                "m": 256,
                "n": 128,
                "k": 64,
                "lda": 256,
                "ldb": 64,
                "ldc": 256,
                "batchCount": 4,
                "strideA": 16384,
                "strideB": 8192,
                "strideC": 32768,
                "Atype": 14,
                "Btype": 14,
                "Ctype": 14,
                "computeType": 68,
                "algo": 99,
            },
            {
                "api": "cublasGemmStridedBatchedEx",
                "type": "blas_compute",
                "m": 256,
                "n": 128,
                "k": 64,
                "lda": 512,
                "ldb": 64,
                "ldc": 512,
                "batch_count": 4,
                "stride_a": 32768,
                "stride_b": 8192,
                "stride_c": 65536,
                "Atype": 14,
                "Btype": 14,
                "Ctype": 14,
                "computeType": 68,
                "algorithm": 23,
            },
        ]
        * 4,
        targets_us=[100.0, 300.0] * 4,
    )

    assert provider is not None
    stored_feature_dicts = [json.loads(key) for key in provider._signature_p50_us]
    exact_keys = [payload for payload in stored_feature_dicts if payload.get("api") == "cublasGemmStridedBatchedEx"]
    assert any(
        payload.get("lda") == 256
        and payload.get("stride_a") == 16384
        and payload.get("algorithm") == 99
        for payload in exact_keys
    )
    assert any(
        payload.get("lda") == 512
        and payload.get("stride_a") == 32768
        and payload.get("algorithm") == 23
        for payload in exact_keys
    )
    assert all("algo" not in payload and "strideA" not in payload for payload in exact_keys)
    assert provider.estimate_us(
        {
            "api": "cublasGemmStridedBatchedEx",
            "type": "blas_compute",
            "m": 256,
            "n": 128,
            "k": 64,
            "lda": 512,
            "ldb": 64,
            "ldc": 512,
            "batchCount": 4,
            "strideA": 32768,
            "strideB": 8192,
            "strideC": 65536,
            "Atype": 14,
            "Btype": 14,
            "Ctype": 14,
            "computeType": 68,
            "algo": 23,
        }
    ) == pytest.approx(300.0)


def test_trace_signature_timing_provider_normalizes_cublas_transpose_fields() -> None:
    provider = TraceSignatureTimingProvider.from_jsonable(
        {
            "type": "trace_signature_stats",
            "signature_p50_us": {
                _signature_key_from_features(
                    {
                        "api": "cublasGemmEx",
                        "type": "blas_compute",
                        "m": 1536,
                        "n": 3072,
                        "k": 512,
                        "transa": "1",
                        "transb": "0",
                        "dtype_code": 2,
                    }
                ): 1234.0,
            },
        }
    )

    assert provider.estimate_us(
        {
            "api": "cublasGemmEx",
            "type": "blas_compute",
            "m": 1536,
            "n": 3072,
            "k": 512,
            "transa": 1,
            "transb": 0,
            "Atype": 2,
        }
    ) == pytest.approx(1234.0)


def test_trace_signature_provider_canonicalizes_loaded_alias_signature_keys() -> None:
    alias_signature_key = _signature_key_from_features(
        {
            "api": "ncclBcast",
            "type": "other",
            "collective_api": "ncclBcast",
            "collective": "broadcast",
            "numel": 8,
            "dtype_code": 2,
            "world_size": 8,
            "communicator_size": 8,
            "participant_count": 8,
        }
    )
    provider = TraceSignatureTimingProvider.from_jsonable(
        {
            "type": "trace_signature_stats",
            "signature_p50_us": {
                alias_signature_key: 42.0,
            },
        }
    )

    estimate = provider.estimate_us(
        {
            "api": "ncclBcast",
            "type": "other",
            "collective_api": "ncclBcast",
            "count": 8,
            "datatype": 7,
            "nranks": 8,
        }
    )

    assert estimate == pytest.approx(42.0)
    assert any("ncclBroadcast" in key for key in provider._signature_p50_us)
    assert not any("ncclBcast" in key for key in provider._signature_p50_us)


def test_trace_signature_p2p_group_provider_matches_raw_rows_to_normalized_payload() -> None:
    raw_member_sample = {
        "api": "ncclRecv",
        "type": "nccl_collective",
        "collective": "recv",
        "collective_api": "ncclRecv",
        "collective_group_id": "ncclRecv|comm:p2p|call:5",
        "collective_communicator_id": "p2p",
        "world_size": 16,
        "communicator_size": 2,
        "participant_count": 1,
        "count": 4096,
        "datatype": 7,
    }
    normalized_group_payload = {
        "api": "ncclP2P",
        "type": "nccl_collective",
        "collective": "p2p",
        "collective_api": "ncclP2P",
        "collective_group_id": "ncclP2P|comm:p2p|call:5",
        "collective_communicator_id": "p2p",
        "world_size": 16,
        "communicator_size": 2,
        "participant_count": 2,
        "count": 4096,
        "datatype": 7,
    }

    raw_features = build_collective_group_timing_features(raw_member_sample)
    normalized_features = build_collective_group_timing_features(normalized_group_payload)
    assert raw_features == normalized_features
    assert raw_features["api"] == "ncclP2P"
    assert raw_features["world_size"] == 2
    assert raw_features["communicator_size"] == 2
    assert raw_features["participant_count"] == 2
    assert "trace_world_size" not in raw_features

    provider = TraceSignatureTimingProvider.fit(
        samples=[raw_member_sample] * 4,
        targets_us=[55.0] * 4,
    )

    assert provider is not None
    stored_feature_dicts = [
        json.loads(key) for key in provider._signature_p50_us
    ]
    assert stored_feature_dicts == [
        {
            "api": "ncclP2P",
            "collective": "p2p",
            "collective_api": "ncclP2P",
            "communicator_size": 2,
            "dtype_code": 2,
            "numel": 4096,
            "participant_count": 2,
            "type": "nccl_collective",
            "world_size": 2,
        }
    ]
    estimator = Estimator(providers=[provider])
    decision = estimator.estimate_collective_group_with_details(normalized_group_payload)
    assert decision is not None
    assert decision.provider_name == "trace_signature_stats"
    assert decision.duration_us == pytest.approx(55.0)


def test_trace_signature_timing_provider_matches_normalized_nccl_p2p_signatures() -> None:
    provider = TraceSignatureTimingProvider.fit(
        samples=[
            {
                "api": "ncclSend",
                "type": "nccl_collective",
                "count": 8,
                "datatype": 7,
                "nranks": 8,
                "peer": 1,
                "world_size": 8,
            }
        ]
        * 4,
        targets_us=[12.0] * 4,
    )

    assert provider is not None
    assert provider.estimate_us(
        {
            "api": "ncclSend",
            "type": "nccl_collective",
            "count": 8,
            "datatype": 7,
            "world_size": 8,
            "collective": "send",
            "collective_api": "ncclP2P",
            "communicator_size": 8,
            "participant_count": 2,
            "collective_sequence_number": 0,
        }
    ) == pytest.approx(12.0)


def test_trace_signature_timing_provider_round_trips_jsonable() -> None:
    provider = TraceSignatureTimingProvider.fit(
        samples=[{"api": "apiA", "type": "other", "bytes": 1024}] * 8,
        targets_us=[5.0] * 8,
    )

    assert provider is not None
    restored = TraceSignatureTimingProvider.from_jsonable(provider.to_jsonable())
    assert restored.estimate_us({"api": "apiA", "type": "other", "bytes": 1024}) == pytest.approx(5.0)


def test_trace_signature_timing_provider_preserves_kernel_shape_keys_on_load() -> None:
    provider = TraceSignatureTimingProvider.fit(
        samples=[
            {
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel": "void foo(c10::BFloat16*)",
                "grid_x": 8,
                "block_x": 128,
            },
            {
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel": "void foo(c10::BFloat16*)",
                "grid_x": 64,
                "block_x": 128,
            },
        ]
        * 8,
        targets_us=[100.0, 900.0] * 8,
    )

    assert provider is not None
    restored = TraceSignatureTimingProvider.from_jsonable(provider.to_jsonable())

    assert any("kernel_signature" in key for key in restored._signature_p50_us)
    assert restored.estimate_us(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel": "void foo(float*)",
            "grid_x": 8,
            "block_x": 128,
        }
    ) == pytest.approx(100.0)
    assert restored.estimate_us(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel": "void foo(float*)",
            "grid_x": 64,
            "block_x": 128,
        }
    ) == pytest.approx(900.0)


def test_trace_signature_timing_provider_distinguishes_kernel_signatures() -> None:
    provider = TraceSignatureTimingProvider.fit(
        samples=(
            [
                {
                    "api": "cudaLaunchKernel",
                    "type": "kernel_launch",
                    "kernel_id": "k-fast",
                    "grid_x": 8,
                    "block_x": 128,
                },
                {
                    "api": "cudaLaunchKernel",
                    "type": "kernel_launch",
                    "kernel_id": "k-slow",
                    "grid_x": 64,
                    "block_x": 256,
                },
            ]
            * 4
        ),
        targets_us=[100.0, 900.0] * 4,
    )

    assert provider is not None
    assert provider.estimate_us(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel_id": "k-fast",
            "grid_x": 8,
            "block_x": 128,
        }
    ) == pytest.approx(100.0)
    assert provider.estimate_us(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel_id": "k-slow",
            "grid_x": 64,
            "block_x": 256,
        }
    ) == pytest.approx(900.0)


def test_trace_signature_timing_provider_blocks_mixed_kernel_signature_fallback() -> None:
    provider = TraceSignatureTimingProvider.fit(
        samples=(
            [
                {
                    "api": "cudaLaunchKernel",
                    "type": "kernel_launch",
                    "kernel_id": "k-heavy",
                    "grid_x": 8,
                    "block_x": 128,
                },
                {
                    "api": "cudaLaunchKernel",
                    "type": "kernel_launch",
                    "kernel_id": "k-light",
                    "grid_x": 2,
                    "block_x": 64,
                },
            ]
            * 4
        ),
        targets_us=[800.0, 100.0] * 4,
    )

    assert provider is not None
    assert provider.estimate_us(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel_id": "k-heavy",
            "grid_x": 64,
            "block_x": 256,
        }
    ) is None
    assert provider.estimate_us(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel_id": "k-light",
            "grid_x": 32,
            "block_x": 128,
        }
    ) is None


def test_trace_signature_timing_provider_allows_singleton_kernel_signature_fallback() -> None:
    provider = TraceSignatureTimingProvider.fit(
        samples=[
            {
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel_id": "k-single",
                "grid_x": 8,
                "block_x": 128,
            }
        ]
        * 4,
        targets_us=[800.0] * 4,
    )

    assert provider is not None
    assert provider.estimate_us(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel_id": "k-single",
            "grid_x": 8,
            "block_x": 128,
        }
    ) == pytest.approx(800.0)
    assert provider.estimate_us(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel_id": "k-single",
            "grid_x": 64,
            "block_x": 256,
        }
    ) is None


def test_trace_signature_kernel_signature_singleton_mismatch_falls_through_to_later_provider() -> None:
    class FallbackProvider:
        name = "fallback_provider"

        def estimate_us(self, event: dict[str, object], percentile: str = "p50") -> float:
            return 123.0

    signature_provider = TraceSignatureTimingProvider.fit(
        samples=[
            {
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel_id": "k-single",
                "grid_x": 8,
                "block_x": 128,
            },
        ],
        targets_us=[10.0],
    )

    assert signature_provider is not None
    estimator = Estimator(providers=[signature_provider, FallbackProvider()])
    decision = estimator.estimate_event_with_details(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel_id": "k-single",
            "grid_x": 128,
            "block_x": 256,
        }
    )

    assert decision.duration_us == pytest.approx(123.0)
    assert decision.provider_name == "fallback_provider"


def test_trace_signature_kernel_signature_guard_falls_through_to_later_provider() -> None:
    class FallbackProvider:
        name = "fallback_provider"

        def estimate_us(self, event: dict[str, object], percentile: str = "p50") -> float:
            return 123.0

    signature_provider = TraceSignatureTimingProvider.fit(
        samples=[
            {
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel_id": "k-mixed",
                "grid_x": 8,
                "block_x": 128,
            },
            {
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel_id": "k-mixed",
                "grid_x": 64,
                "block_x": 128,
            },
        ],
        targets_us=[10.0, 20.0],
    )

    assert signature_provider is not None
    estimator = Estimator(providers=[signature_provider, FallbackProvider()])
    decision = estimator.estimate_event_with_details(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel_id": "k-mixed",
            "grid_x": 128,
            "block_x": 256,
        }
    )

    assert decision.duration_us == pytest.approx(123.0)
    assert decision.provider_name == "fallback_provider"


def test_trace_signature_timing_provider_skips_zero_only_device_work_buckets() -> None:
    provider = TraceSignatureTimingProvider.fit(
        samples=(
            [
                {
                    "api": "cudaLaunchKernel",
                    "type": "kernel_launch",
                    "kernel_id": "k-zero",
                    "grid_x": 8,
                    "block_x": 128,
                },
                {
                    "api": "cudaLaunchKernel",
                    "type": "kernel_launch",
                    "kernel_id": "k-positive",
                    "grid_x": 8,
                    "block_x": 128,
                },
            ]
            * 4
        ),
        targets_us=[0.0, 500.0] * 4,
    )

    assert provider is not None
    assert provider.estimate_us(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel_id": "k-zero",
            "grid_x": 8,
            "block_x": 128,
        }
    ) is None
    assert provider.estimate_us(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel_id": "k-positive",
            "grid_x": 8,
            "block_x": 128,
        }
    ) == pytest.approx(500.0)


def test_trace_signature_timing_provider_prefers_kernel_name_over_runtime_kernel_id() -> None:
    provider = TraceSignatureTimingProvider.fit(
        samples=[
            {
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel_id": "140587169744336",
                "kernel": "void foo(c10::BFloat16*)",
                "grid_x": 8,
                "block_x": 128,
            }
        ]
        * 8,
        targets_us=[700.0] * 8,
    )

    assert provider is not None
    assert provider.estimate_us(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel": "void foo(float*)",
            "grid_x": 8,
            "block_x": 128,
        }
    ) == pytest.approx(700.0)


def test_trace_signature_timing_provider_normalizes_kernel_dtype_and_lambda_variants() -> None:
    provider = TraceSignatureTimingProvider.fit(
        samples=[
            {
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel": "void bar(c10::BFloat16)::{lambda()#4}",
                "grid_x": 8,
                "block_x": 128,
            }
        ]
        * 8,
        targets_us=[321.0] * 8,
    )

    assert provider is not None
    assert provider.estimate_us(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel": "void bar(float)::{lambda()#2}",
            "grid_x": 8,
            "block_x": 128,
        }
    ) == pytest.approx(321.0)


def test_trace_signature_timing_provider_normalizes_query_numeric_feature_types() -> None:
    provider = TraceSignatureTimingProvider.fit(
        samples=[{"api": "cudaLaunchKernel", "type": "kernel_launch", "kernel_id": "k0", "world_size": 8}] * 8,
        targets_us=[42.0] * 8,
    )

    assert provider is not None
    assert provider.estimate_us(
        {
            "api": "cudaLaunchKernel",
            "type": "kernel_launch",
            "kernel_id": "k0",
            "world_size": "8",
        }
    ) == pytest.approx(42.0)


def test_hybrid_estimator_orders_trace_profiling_before_xgboost_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubProvider:
        def __init__(self, name: str):
            self.name = name

        def estimate_us(self, event, percentile="p50"):
            del event, percentile
            return None

    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel_id": "k0",
            },
            {
                "ts": 100,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfter",
                "type": "other",
            },
        ],
    )

    monkeypatch.setattr(
        "flexsim.estimator.probe_trace_learned_provider",
        lambda samples, targets_us, max_samples=20000: ProviderLoadStatus(
            provider=_StubProvider("trace_learned_sklearn"),
            error=None,
        ),
    )
    monkeypatch.setattr(
        "flexsim.estimator.TraceSignatureTimingProvider.fit",
        classmethod(lambda cls, samples, targets_us: _StubProvider("trace_signature_stats")),
    )

    def _attach_gpu(self, bundle_dir, *, prepend=True):
        del bundle_dir
        self.add_provider(_StubProvider("gpu_estimator_xgboost"), prepend=prepend)
        return True

    monkeypatch.setattr(Estimator, "attach_optional_gpu_estimator", _attach_gpu)

    estimator = Estimator.fit_from_traces(
        str(trace_dir),
        max_files=1,
        learned_method="hybrid",
    )

    assert estimator.provider_names() == (
        "trace_signature_stats",
        "trace_learned_sklearn",
        "gpu_estimator_xgboost",
    )


def test_hybrid_estimator_records_trace_learned_probe_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubProvider:
        def __init__(self, name: str):
            self.name = name

        def estimate_us(self, event, percentile="p50"):
            del event, percentile
            return None

    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel_id": "k0",
            },
            {
                "ts": 100,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfter",
                "type": "other",
            },
        ],
    )

    monkeypatch.setattr(
        "flexsim.estimator.probe_trace_learned_provider",
        lambda samples, targets_us, max_samples=20000: ProviderLoadStatus(
            provider=None,
            error=(
                "trace_learned_sklearn unavailable: missing numpy/sklearn "
                "dependency (ModuleNotFoundError(\"No module named 'sklearn'\"))"
            ),
        ),
    )
    monkeypatch.setattr(
        "flexsim.estimator.TraceSignatureTimingProvider.fit",
        classmethod(lambda cls, samples, targets_us: _StubProvider("trace_signature_stats")),
    )
    monkeypatch.setattr(
        Estimator,
        "attach_optional_gpu_estimator",
        lambda self, bundle_dir, *, prepend=True: False,
    )

    estimator = Estimator.fit_from_traces(
        str(trace_dir),
        max_files=1,
        learned_method="hybrid",
    )

    assert estimator.provider_names() == ("trace_signature_stats",)
    assert "missing numpy/sklearn dependency" in estimator.provider_diagnostics()["trace_learned_sklearn"]


def test_gpu_xgboost_estimator_uses_signature_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubProvider:
        def __init__(self, name: str):
            self.name = name

        def estimate_us(self, event, percentile="p50"):
            del event, percentile
            return None

    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 1,
                "tid": 11,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "kernel_id": "k0",
            },
            {
                "ts": 100,
                "pid": 1,
                "tid": 11,
                "mod": "libfoo.so",
                "api": "apiAfter",
                "type": "other",
            },
        ],
    )

    monkeypatch.setattr(
        "flexsim.estimator.TraceSignatureTimingProvider.fit",
        classmethod(lambda cls, samples, targets_us: _StubProvider("trace_signature_stats")),
    )

    def _attach_gpu(self, bundle_dir, *, prepend=True):
        del bundle_dir
        self.add_provider(_StubProvider("gpu_estimator_xgboost"), prepend=prepend)
        return True

    monkeypatch.setattr(Estimator, "attach_optional_gpu_estimator", _attach_gpu)

    estimator = Estimator.fit_from_traces(
        str(trace_dir),
        max_files=1,
        learned_method="gpu_xgboost",
    )

    assert estimator.provider_names() == (
        "gpu_estimator_xgboost",
        "trace_signature_stats",
    )
