#!/usr/bin/env python3
"""Build Figure 5 CSVs from paper-run inputs or fresh native runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GPT_SCALES = (8, 16, 32, 64, 128)
NATIVE_GPT_SCALES = (8, 16)
MOE_CASES = (
    ("routed_moe_base", "Base MoE", "none", 1234),
    ("routed_moe_intra_group_0_1", "Intra 0-1", "[0,1]", 5200),
    ("routed_moe_cross_group_0_8", "Cross 0-8", "[0,8]", 5201),
    ("routed_moe_cross_group_0_15", "Cross 0-15", "[0,15]", 5202),
    ("routed_moe_boundary_7_8", "Boundary 7-8", "[7,8]", 5203),
)
COMMON_FIELDS = (
    "predicted_runtime_us",
    "actual_runtime_us",
    "evaluator_error_pct",
    "shared_trace",
    "contract_mode",
    "source_mode",
    "source_sha256",
    "trace_dir",
)


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_trace(path: Path, world_size: int, source_mode: str) -> float:
    if not path.is_dir():
        raise FileNotFoundError(f"missing Figure 5 trace: {path}")
    manifest = load_json(path / "capture_manifest.json")
    if int(manifest.get("original_world_size", 0)) != world_size:
        raise ValueError(f"wrong world size in {path}: expected {world_size}")
    rank_files = [path / f"rank_{rank}.jsonl" for rank in range(world_size)]
    marker_files = [path / f"rank_{rank}.markers.jsonl" for rank in range(world_size)]
    missing = [item for item in rank_files + marker_files if not item.is_file() or item.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"incomplete {world_size}-rank trace: {missing[0]}")
    windows = manifest.get("step_windows")
    if not isinstance(windows, dict) or set(windows) != {str(rank) for rank in range(world_size)}:
        raise ValueError(f"incomplete Figure 5 step windows: {path}")
    durations = []
    for rank in range(world_size):
        window = windows[str(rank)]
        if (
            not isinstance(window, dict)
            or window.get("source") != "trace_markers"
            or window.get("is_paper_valid_step_window") is not True
            or int(window.get("step_count", 0)) != 1
        ):
            raise ValueError(f"invalid Figure 5 step window: {path}, rank {rank}")
        duration = float(window["end_ts"]) - float(window["start_ts"])
        if duration <= 0.0 or not math.isfinite(duration):
            raise ValueError(f"invalid Figure 5 step duration: {path}, rank {rank}")
        durations.append(duration)
    if source_mode != "native_gpu":
        return max(durations)
    route = manifest.get("route_metadata")
    if not isinstance(route, dict) or route.get("figure13_route") != "figure5_native_gpu":
        raise ValueError(f"native Figure 5 trace lacks the native route contract: {path}")
    commands = [route.get("capture_command", ""), *(route.get("node_capture_commands", []) or [])]
    command = " ".join(str(item) for item in commands).lower()
    if "capture_real" not in command or "capture_emulated" in command or "frun" in command:
        raise ValueError(f"native Figure 5 trace was not produced by capture_real: {path}")
    return max(durations)


def measurement(predicted: object, actual: object, label: str) -> tuple[float, float, float]:
    predicted_value = float(predicted)
    actual_value = float(actual)
    if min(predicted_value, actual_value) <= 0.0 or not all(
        math.isfinite(value) for value in (predicted_value, actual_value)
    ):
        raise ValueError(f"invalid Figure 5 runtime: {label}")
    error = 100.0 * abs(predicted_value - actual_value) / actual_value
    return predicted_value, actual_value, error


def source_measurement(row: dict[str, object], label: str) -> tuple[float, float, float, str]:
    source_sha256 = str(row.get("source_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        raise ValueError(f"invalid source SHA-256: {label}")
    predicted, actual, error = measurement(
        row.get("predicted_runtime_us"), row.get("actual_runtime_us"), label
    )
    return predicted, actual, error, source_sha256


def summary_values(path: Path) -> tuple[float, float, float]:
    summary = load_json(path)
    predicted, actual, error = measurement(
        summary.get("predicted_per_iteration_runtime_us"),
        summary.get("actual_per_iteration_runtime_us"),
        str(path),
    )
    stored = float(summary["paper_absolute_error_pct"])
    if not math.isclose(error, stored, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise ValueError(f"error arithmetic differs in {path}")
    return predicted, actual, error


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, gpt_rows: list[dict[str, object]], moe_rows: list[dict[str, object]]) -> None:
    write_csv(output_dir / "figure5a.csv", ("gpu_scale", *COMMON_FIELDS), gpt_rows)
    write_csv(
        output_dir / "figure5b.csv",
        ("case", "route_experts", "seed", "world_size", *COMMON_FIELDS),
        moe_rows,
    )


def collect_trace(source_ledger: Path, output_dir: Path, large_trace_root: Path) -> None:
    source = load_json(source_ledger)
    if source.get("schema") != 1:
        raise ValueError(f"unsupported Figure 5 source schema: {source_ledger}")
    gpt_source = source.get("gpt")
    moe_source = source.get("moe")
    if not isinstance(gpt_source, list) or not isinstance(moe_source, list):
        raise ValueError(f"incomplete Figure 5 source ledger: {source_ledger}")
    if [int(row.get("gpu_scale", 0)) for row in gpt_source if isinstance(row, dict)] != list(GPT_SCALES):
        raise ValueError("Figure 5 source ledger has the wrong GPT scales")
    expected_moe_ids = [case_id for case_id, *_ in MOE_CASES]
    if [str(row.get("case_id", "")) for row in moe_source if isinstance(row, dict)] != expected_moe_ids:
        raise ValueError("Figure 5 source ledger has the wrong MoE cases")

    gpt_rows: list[dict[str, object]] = []
    for row in gpt_source:
        if not isinstance(row, dict):
            raise ValueError("Figure 5 GPT source row is not an object")
        scale = int(row["gpu_scale"])
        expected_mode = "paper_run_record" if scale <= 16 else "provided_trace"
        if row.get("source_mode") != expected_mode:
            raise ValueError(f"wrong Figure 5 source mode: GPT-{scale}")
        trace_dir = ""
        trace_actual = None
        if scale >= 32:
            trace_path = large_trace_root / f"gpt-{scale}"
            trace_actual = validate_trace(trace_path, scale, "provided_trace")
            trace_dir = str(trace_path)
        predicted, actual, error, source_sha256 = source_measurement(row, f"GPT-{scale}")
        if trace_actual is not None:
            if not math.isclose(actual, trace_actual, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(f"Figure 5 source and trace actual runtime differ: GPT-{scale}")
            actual = trace_actual
            error = 100.0 * abs(predicted - actual) / actual
        gpt_rows.append(
            {
                "gpu_scale": scale,
                "predicted_runtime_us": predicted,
                "actual_runtime_us": actual,
                "evaluator_error_pct": error,
                "shared_trace": True,
                "contract_mode": "trace",
                "source_mode": expected_mode,
                "source_sha256": source_sha256,
                "trace_dir": trace_dir,
            }
        )

    moe_rows: list[dict[str, object]] = []
    for row, (case_id, label, route_experts, seed) in zip(moe_source, MOE_CASES, strict=True):
        if not isinstance(row, dict) or row.get("source_mode") != "paper_run_record":
            raise ValueError(f"wrong Figure 5 source mode: {case_id}")
        predicted, actual, error, source_sha256 = source_measurement(row, case_id)
        moe_rows.append(
            {
                "case": label,
                "route_experts": route_experts,
                "seed": seed,
                "world_size": 16,
                "predicted_runtime_us": predicted,
                "actual_runtime_us": actual,
                "evaluator_error_pct": error,
                "shared_trace": True,
                "contract_mode": "trace",
                "source_mode": "paper_run_record",
                "source_sha256": source_sha256,
                "trace_dir": "",
            }
        )
    write_outputs(output_dir, gpt_rows, moe_rows)


def collect_native(summary_root: Path, output_dir: Path, native_trace_root: Path) -> None:
    gpt_rows: list[dict[str, object]] = []
    for scale in NATIVE_GPT_SCALES:
        trace_dir = native_trace_root / "gpt" / str(scale) / "real"
        trace_actual = validate_trace(trace_dir, scale, "native_gpu")
        predicted, actual, error = summary_values(
            summary_root / "gpt" / str(scale) / "simulate_summary.json"
        )
        if not math.isclose(actual, trace_actual, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"Figure 5 summary and trace actual runtime differ: GPT-{scale}")
        gpt_rows.append(
            {
                "gpu_scale": scale,
                "predicted_runtime_us": predicted,
                "actual_runtime_us": actual,
                "evaluator_error_pct": error,
                "shared_trace": True,
                "contract_mode": "native",
                "source_mode": "native_gpu",
                "source_sha256": "",
                "trace_dir": str(trace_dir),
            }
        )

    moe_rows: list[dict[str, object]] = []
    for case_id, label, route_experts, seed in MOE_CASES:
        trace_dir = native_trace_root / "moe" / case_id / "real"
        trace_actual = validate_trace(trace_dir, 16, "native_gpu")
        predicted, actual, error = summary_values(
            summary_root / "moe" / case_id / "simulate_summary.json"
        )
        if not math.isclose(actual, trace_actual, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"Figure 5 summary and trace actual runtime differ: {case_id}")
        moe_rows.append(
            {
                "case": label,
                "route_experts": route_experts,
                "seed": seed,
                "world_size": 16,
                "predicted_runtime_us": predicted,
                "actual_runtime_us": actual,
                "evaluator_error_pct": error,
                "shared_trace": True,
                "contract_mode": "native",
                "source_mode": "native_gpu",
                "source_sha256": "",
                "trace_dir": str(trace_dir),
            }
        )
    write_outputs(output_dir, gpt_rows, moe_rows)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        large = root / "large"
        native = root / "native"
        summaries = root / "summaries"

        def trace_fixture(
            trace_dir: Path, world_size: int, native_gpu: bool, duration: float
        ) -> None:
            trace_dir.mkdir(parents=True)
            route = {
                "figure13_route": "figure5_native_gpu",
                "capture_command": "python -m flexsim.maya_lite.capture_real workload.py",
            }
            (trace_dir / "capture_manifest.json").write_text(
                json.dumps(
                    {
                        "original_world_size": world_size,
                        "route_metadata": route if native_gpu else {},
                        "step_windows": {
                            str(rank): {
                                "start_ts": rank * 1000,
                                "end_ts": rank * 1000 + duration,
                                "source": "trace_markers",
                                "is_paper_valid_step_window": True,
                                "step_count": 1,
                            }
                            for rank in range(world_size)
                        },
                    }
                ),
                encoding="utf-8",
            )
            for rank in range(world_size):
                (trace_dir / f"rank_{rank}.jsonl").write_text("{}\n", encoding="utf-8")
                (trace_dir / f"rank_{rank}.markers.jsonl").write_text("{}\n", encoding="utf-8")

        def summary_fixture(path: Path, error: float) -> None:
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "predicted_per_iteration_runtime_us": 100.0 * (1.0 + error / 100.0),
                        "actual_per_iteration_runtime_us": 100.0,
                        "paper_absolute_error_pct": error,
                    }
                ),
                encoding="utf-8",
            )

        paper_actual = {32: 4558752.0, 64: 4556835.0, 128: 4967189.0}
        for scale in (32, 64, 128):
            trace_fixture(large / f"gpt-{scale}", scale, False, paper_actual[scale])
        trace_output = root / "trace-output"
        collect_trace(ROOT / "large-cluster/e2/figure5-source.json", trace_output, large)

        for scale in NATIVE_GPT_SCALES:
            trace_fixture(native / "gpt" / str(scale) / "real", scale, True, 100.0)
            summary_fixture(
                summaries / "gpt" / str(scale) / "simulate_summary.json",
                115.65 if scale == 8 else 38.21,
            )
        for error, (case_id, *_) in zip((38.61, 45.30, 44.11, 44.80, 48.36), MOE_CASES, strict=True):
            trace_fixture(native / "moe" / case_id / "real", 16, True, 100.0)
            summary_fixture(summaries / "moe" / case_id / "simulate_summary.json", error)
        native_output = root / "native-output"
        collect_native(summaries, native_output, native)

        from validate_results import validate_figure5
        from plot_figure5 import draw

        validate_figure5(trace_output, "trace")
        validate_figure5(native_output, "native")
        assert draw(trace_output, root / "plots", "trace").name == "figure5.pdf"
        assert draw(native_output, root / "plots", "native").name == "figure5-native.pdf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("trace", "native"), default="trace")
    parser.add_argument("--source-ledger", type=Path, default=ROOT / "large-cluster/e2/figure5-source.json")
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--native-trace-root", type=Path)
    parser.add_argument("--large-trace-root", type=Path, default=ROOT / "large-cluster/e2")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("Figure 5 collector self-test: PASS")
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.mode == "trace":
        collect_trace(args.source_ledger, args.output_dir, args.large_trace_root)
    else:
        if args.summary_root is None or args.native_trace_root is None:
            parser.error("native mode requires --summary-root and --native-trace-root")
        collect_native(args.summary_root, args.output_dir, args.native_trace_root)
    print(f"wrote {args.output_dir / 'figure5a.csv'} and {args.output_dir / 'figure5b.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
