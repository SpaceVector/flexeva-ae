import json
from pathlib import Path

import pytest

from flexsim.estimator import ProviderLoadStatus
from flexsim.maya_lite.cli import build_parser, resolve_estimator_mode


def _write_trace(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class _XGBoostLikeProvider:
    name = "gpu_estimator_xgboost"

    def estimate_us(self, event: dict[str, object], percentile: str = "p50") -> float | None:
        del percentile
        if str(event.get("api")) == "cublasSgemm_v2":
            return 50.0
        return None


class _StubEstimator:
    def __init__(self, covered_time_share: float):
        self._covered_time_share = covered_time_share

    def add_provider(self, provider, *, prepend: bool = False):
        del provider, prepend

    def provider_coverage_summary(self, provider_name: str | None = None, *, limit: int = 10):
        del provider_name, limit
        return {
            "covered_time_share": self._covered_time_share,
        }


def test_resolve_estimator_mode_auto_prefers_hybrid_when_xgboost_covers_heavy_hitters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(trace_dir / "rank_0.jsonl", [{"api": "cublasSgemm_v2"}])

    monkeypatch.setattr(
        "flexsim.maya_lite.cli.probe_gpu_estimator_provider",
        lambda bundle_dir: ProviderLoadStatus(provider=_XGBoostLikeProvider(), error=None),
    )
    monkeypatch.setattr(
        "flexsim.maya_lite.cli.Estimator.fit_from_traces",
        lambda *args, **kwargs: _StubEstimator(covered_time_share=0.72),
    )

    assert resolve_estimator_mode(trace_dir, "auto", fit_trace_dir=trace_dir) == "hybrid"


def test_resolve_estimator_mode_auto_falls_back_to_learned_trace_when_xgboost_has_low_time_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(trace_dir / "rank_0.jsonl", [{"api": "cudaLaunchKernel"}])

    monkeypatch.setattr(
        "flexsim.maya_lite.cli.probe_gpu_estimator_provider",
        lambda bundle_dir: ProviderLoadStatus(provider=_XGBoostLikeProvider(), error=None),
    )
    monkeypatch.setattr(
        "flexsim.maya_lite.cli.Estimator.fit_from_traces",
        lambda *args, **kwargs: _StubEstimator(covered_time_share=0.12),
    )

    assert resolve_estimator_mode(trace_dir, "auto", fit_trace_dir=trace_dir) == "learned_trace"


def test_resolve_estimator_mode_respects_non_auto_requests(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(trace_dir / "rank_0.jsonl", [{"api": "cudaLaunchKernel"}])

    assert resolve_estimator_mode(trace_dir, "gpu_xgboost") == "gpu_xgboost"
    assert resolve_estimator_mode(trace_dir, "trace_stats") == "trace_stats"


def test_resolve_estimator_mode_auto_falls_back_when_gpu_xgboost_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(trace_dir / "rank_0.jsonl", [{"api": "cublasSgemm_v2"}])

    monkeypatch.setattr(
        "flexsim.maya_lite.cli.probe_gpu_estimator_provider",
        lambda bundle_dir: ProviderLoadStatus(provider=None, error="xgboost missing"),
    )

    assert resolve_estimator_mode(trace_dir, "auto", fit_trace_dir=trace_dir) == "learned_trace"


def test_resolve_estimator_mode_passes_fit_max_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(trace_dir / "rank_0.jsonl", [{"api": "cublasSgemm_v2"}])

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "flexsim.maya_lite.cli.probe_gpu_estimator_provider",
        lambda bundle_dir: ProviderLoadStatus(provider=_XGBoostLikeProvider(), error=None),
    )

    def _fit_from_traces(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _StubEstimator(covered_time_share=0.72)

    monkeypatch.setattr(
        "flexsim.maya_lite.cli.Estimator.fit_from_traces",
        _fit_from_traces,
    )

    assert (
        resolve_estimator_mode(
            trace_dir,
            "auto",
            fit_trace_dir=trace_dir,
            fit_max_files=5,
        )
        == "hybrid"
    )
    assert captured["kwargs"]["max_files"] == 5


def test_build_parser_defaults_disable_runtime_fallbacks() -> None:
    parser = build_parser()

    args = parser.parse_args(["some/trace/dir", "--model", "estimator.json"])

    assert args.allow_heuristic_kernel_launch_fallback is False
    assert args.allow_weak_runtime_fallback is False
    assert args.use_observed_semantic_wrapper_durations is True


def test_build_parser_rejects_old_runtime_fallback_flags() -> None:
    parser = build_parser()

    for flag in (
        "--allow-heuristic-kernel-launch-fallback",
        "--allow-weak-runtime-fallback",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "some/trace/dir",
                    "--model",
                    "estimator.json",
                    flag,
                ]
            )


def test_build_parser_rejects_old_observed_semantic_wrapper_toggle() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "some/trace/dir",
                "--model",
                "estimator.json",
                "--disable-observed-semantic-wrapper-durations",
            ]
        )
