#!/usr/bin/env python3
"""Build Figure 5 CSVs from fresh native runs and linked large-cluster traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GPT_SCALES = (8, 16, 32, 64, 128)
MOE_CASES = (
    ("routed_moe_base", "Base MoE", "none", 1234),
    ("routed_moe_intra_group_0_1", "Intra 0-1", "[0,1]", 5200),
    ("routed_moe_cross_group_0_8", "Cross 0-8", "[0,8]", 5201),
    ("routed_moe_cross_group_0_15", "Cross 0-15", "[0,15]", 5202),
    ("routed_moe_boundary_7_8", "Boundary 7-8", "[7,8]", 5203),
)


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_trace(path: Path, world_size: int, source_mode: str) -> None:
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
    if source_mode == "native_gpu":
        route = manifest.get("route_metadata")
        if not isinstance(route, dict) or route.get("figure13_route") != "figure5_native_gpu":
            raise ValueError(f"native Figure 5 trace lacks the native route contract: {path}")
        commands = [route.get("capture_command", ""), *(route.get("node_capture_commands", []) or [])]
        command = " ".join(str(item) for item in commands).lower()
        if "capture_real" not in command or "capture_emulated" in command or "frun" in command:
            raise ValueError(f"native Figure 5 trace was not produced by capture_real: {path}")


def summary_values(path: Path) -> tuple[float, float, float]:
    summary = load_json(path)
    predicted = float(summary["predicted_per_iteration_runtime_us"])
    actual = float(summary["actual_per_iteration_runtime_us"])
    stored = float(summary["paper_absolute_error_pct"])
    if min(predicted, actual) <= 0.0:
        raise ValueError(f"non-positive runtime in {path}")
    error = 100.0 * abs(predicted - actual) / actual
    if not math.isclose(error, stored, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise ValueError(f"error arithmetic differs in {path}")
    return predicted, actual, error


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect(summary_root: Path, output_dir: Path, native_trace_root: Path, large_trace_root: Path) -> None:
    gpt_rows: list[dict[str, object]] = []
    for scale in GPT_SCALES:
        source_mode = "native_gpu" if scale <= 16 else "large_cluster_trace"
        trace_dir = (
            native_trace_root / "gpt" / str(scale) / "real"
            if scale <= 16
            else large_trace_root / f"gpt-{scale}"
        )
        validate_trace(trace_dir, scale, source_mode)
        predicted, actual, error = summary_values(summary_root / "gpt" / str(scale) / "simulate_summary.json")
        gpt_rows.append(
            {
                "gpu_scale": scale,
                "predicted_runtime_us": predicted,
                "actual_runtime_us": actual,
                "oracle_error_pct": error,
                "maya_error_pct": error,
                "flexeva_error_pct": error,
                "source_mode": source_mode,
                "trace_dir": str(trace_dir),
            }
        )

    moe_rows: list[dict[str, object]] = []
    for case_id, label, route_experts, seed in MOE_CASES:
        trace_dir = native_trace_root / "moe" / case_id / "real"
        validate_trace(trace_dir, 16, "native_gpu")
        predicted, actual, error = summary_values(summary_root / "moe" / case_id / "simulate_summary.json")
        moe_rows.append(
            {
                "case": label,
                "route_experts": route_experts,
                "seed": seed,
                "world_size": 16,
                "predicted_runtime_us": predicted,
                "actual_runtime_us": actual,
                "oracle_error_pct": error,
                "maya_error_pct": error,
                "flexeva_error_pct": error,
                "source_mode": "native_gpu",
                "trace_dir": str(trace_dir),
            }
        )

    write_csv(
        output_dir / "figure5a.csv",
        (
            "gpu_scale",
            "predicted_runtime_us",
            "actual_runtime_us",
            "oracle_error_pct",
            "maya_error_pct",
            "flexeva_error_pct",
            "source_mode",
            "trace_dir",
        ),
        gpt_rows,
    )
    write_csv(
        output_dir / "figure5b.csv",
        (
            "case",
            "route_experts",
            "seed",
            "world_size",
            "predicted_runtime_us",
            "actual_runtime_us",
            "oracle_error_pct",
            "maya_error_pct",
            "flexeva_error_pct",
            "source_mode",
            "trace_dir",
        ),
        moe_rows,
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        summaries = root / "summaries"
        native = root / "native"
        large = root / "large"
        output = root / "output"

        def fixture(trace_dir: Path, summary: Path, world_size: int, native_gpu: bool) -> None:
            write_dir = trace_dir
            if native_gpu:
                write_dir.mkdir(parents=True)
            else:
                write_dir = root / "external" / trace_dir.name
                write_dir.mkdir(parents=True)
                trace_dir.parent.mkdir(parents=True, exist_ok=True)
                trace_dir.symlink_to(write_dir, target_is_directory=True)
            route = {
                "figure13_route": "figure5_native_gpu",
                (
                    "capture_command"
                    if world_size == 8
                    else "node_capture_commands"
                ): (
                    "python -m flexsim.maya_lite.capture_real workload.py"
                    if world_size == 8
                    else ["python -m flexsim.maya_lite.capture_real workload.py"]
                ),
            }
            (write_dir / "capture_manifest.json").write_text(
                json.dumps({"original_world_size": world_size, "route_metadata": route if native_gpu else {}}),
                encoding="utf-8",
            )
            for rank in range(world_size):
                (write_dir / f"rank_{rank}.jsonl").write_text("{}\n", encoding="utf-8")
                (write_dir / f"rank_{rank}.markers.jsonl").write_text("{}\n", encoding="utf-8")
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps(
                    {
                        "predicted_per_iteration_runtime_us": 101.0,
                        "actual_per_iteration_runtime_us": 100.0,
                        "paper_absolute_error_pct": 1.0,
                    }
                ),
                encoding="utf-8",
            )

        for scale in GPT_SCALES:
            trace = native / "gpt" / str(scale) / "real" if scale <= 16 else large / f"gpt-{scale}"
            fixture(trace, summaries / "gpt" / str(scale) / "simulate_summary.json", scale, scale <= 16)
        for case_id, *_ in MOE_CASES:
            fixture(
                native / "moe" / case_id / "real",
                summaries / "moe" / case_id / "simulate_summary.json",
                16,
                True,
            )
        collect(summaries, output, native, large)
        with (output / "figure5a.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert [row["source_mode"] for row in rows] == [
            "native_gpu",
            "native_gpu",
            "large_cluster_trace",
            "large_cluster_trace",
            "large_cluster_trace",
        ]
        from validate_results import validate_figure5

        validate_figure5(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--native-trace-root", type=Path, default=ROOT / "trace" / "e2" / "figure5")
    parser.add_argument("--large-trace-root", type=Path, default=ROOT / "large-cluster" / "e2")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("Figure 5 collector self-test: PASS")
        return 0
    if args.summary_root is None or args.output_dir is None:
        parser.error("--summary-root and --output-dir are required")
    collect(args.summary_root, args.output_dir, args.native_trace_root, args.large_trace_root)
    print(f"wrote {args.output_dir / 'figure5a.csv'} and {args.output_dir / 'figure5b.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
