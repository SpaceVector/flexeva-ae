#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path


COMPONENTS = (
    "maya_emulation_s",
    "trace_processing_s",
    "trace_ras_compaction_s",
    "code_analysis_s",
    "source_ras_update_s",
    "grounding_s",
    "selective_emulation_s",
    "trace_patching_collation_s",
    "event_simulation_s",
    "feedback_generation_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the two-panel Figure 6 ledger from the original evaluators.")
    parser.add_argument("--gpt-result", type=Path, required=True)
    parser.add_argument("--gpt-breakdown", type=Path, required=True)
    parser.add_argument("--moe-result", type=Path, required=True)
    parser.add_argument("--moe-breakdown", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_components(panel: str, row: dict[str, str]) -> dict[str, float]:
    values = {name: 0.0 for name in COMPONENTS}
    values.update(
        {
            "maya_emulation_s": float(row.get("maya_emulation_s", 0.0)),
            "code_analysis_s": float(row.get("code_analysis_s", 0.0)),
            "source_ras_update_s": float(row.get("source_ras_partition_update_s", 0.0)),
            "grounding_s": float(row.get("grounding_s", 0.0)),
            "selective_emulation_s": float(row.get("selective_emulation_s", 0.0)),
            "event_simulation_s": float(row.get("event_simulation_s", 0.0)),
            "feedback_generation_s": float(row.get("feedback_generation_s", 0.0)),
        }
    )
    if panel == "gpt":
        values["trace_processing_s"] = float(row.get("trace_processing_s", 0.0))
        values["trace_ras_compaction_s"] = float(row.get("trace_ras_compaction_s", 0.0))
        values["trace_patching_collation_s"] = float(row.get("trace_patching_collation_s", 0.0))
    elif row["system"] == "Maya-full":
        values["trace_processing_s"] = float(row.get("trace_patching_collation_s", 0.0)) + float(
            row.get("lower_trace_ras_compaction_s", 0.0)
        )
    else:
        values["trace_ras_compaction_s"] = float(row.get("lower_trace_ras_compaction_s", 0.0))
        values["trace_patching_collation_s"] = float(row.get("trace_patching_collation_s", 0.0))
    if not math.isclose(sum(values.values()), float(row["total_s"]), rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"phase total mismatch: {panel} R{row['round']} {row['system']}")
    return values


def check_capture_contract(payload: dict[str, object]) -> None:
    for capture in [payload["anchor"], *payload["candidates"]]:
        distributed = capture["run"]["distributed"]
        if (distributed["nnodes"], distributed["node_rank"], distributed["nproc_per_node"]) != (2, 0, 8):
            raise ValueError(f"not a two-node/16-rank capture: {distributed}")
        if capture["run"]["peer_trace_transfer"]["files"] != 16:
            raise ValueError(f"incomplete peer trace transfer: {capture['run']['peer_trace_transfer']}")
        if capture["api_audit"] != {"cudaGetDevice_modeled_count": 0, "cudaGetDevice_replay_count": 0}:
            raise ValueError(f"cudaGetDevice entered modeled/replay events: {capture['api_audit']}")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gpt = json.loads(args.gpt_result.read_text(encoding="utf-8"))
    moe = json.loads(args.moe_result.read_text(encoding="utf-8"))
    check_capture_contract(gpt)
    check_capture_contract(moe)

    definitions = (
        ("gpt", read_csv(args.gpt_breakdown), "Maya-style full"),
        ("moe", read_csv(args.moe_breakdown), "Maya-full"),
    )
    cumulative: list[dict[str, object]] = []
    phases: list[dict[str, object]] = []
    for panel, rows, full_name in definitions:
        selected = [row for row in rows if row["system"] in {full_name, "FlexEva cumulative"}]
        by_key = {(int(row["round"]), row["system"]): row for row in selected}
        base = float(by_key[(1, full_name)]["total_s"])
        for round_id in range(1, 5):
            full = by_key[(round_id, full_name)]
            flexeva = by_key[(round_id, "FlexEva cumulative")]
            full_s = float(full["total_s"])
            flexeva_s = float(flexeva["total_s"])
            cumulative.append(
                {
                    "panel": panel,
                    "round": round_id,
                    "x_label": full["x_label"],
                    "maya_full_cumulative_s": full_s,
                    "flexeva_cumulative_s": flexeva_s,
                    "normalization_base_s": base,
                    "maya_normalized": full_s / base,
                    "flexeva_normalized": flexeva_s / base,
                    "speedup_vs_maya_full": full_s / flexeva_s,
                }
            )
            for system, source in (("Maya-full", full), ("FlexEva", flexeva)):
                components = canonical_components(panel, source)
                if system == "Maya-full" and any(
                    components[name] != 0.0
                    for name in ("trace_ras_compaction_s", "source_ras_update_s", "selective_emulation_s")
                ):
                    raise ValueError(f"Maya-full contains RAS/selective time: {panel} R{round_id}")
                phases.append(
                    {
                        "panel": panel,
                        "round": round_id,
                        "x_label": source["x_label"],
                        "system": system,
                        **components,
                        "total_s": sum(components.values()),
                    }
                )

    raw: list[dict[str, object]] = []
    for round_id, row in enumerate(gpt["candidates"], start=1):
        raw.append(
            {
                "panel": "gpt",
                "round": round_id,
                "x_label": row["label"],
                "maya_full_s": row["metrics"]["maya_full_s"],
                "flexeva_refresh_s": row["metrics"]["flexeva_refresh_s"],
            }
        )
    moe_per_round = {(int(row["round"]), row["system"]): row for row in moe["per_round_breakdown"]}
    for round_id in range(1, 5):
        full = moe_per_round[(round_id, "Maya-full")]
        flexeva = moe_per_round[(round_id, "FlexEva refresh")]
        raw.append(
            {
                "panel": "moe",
                "round": round_id,
                "x_label": full["x_label"],
                "maya_full_s": full["total_s"],
                "flexeva_refresh_s": flexeva["total_s"],
            }
        )

    write_csv(args.out_dir / "figure6.csv", cumulative)
    write_csv(args.out_dir / "raw_seconds.csv", raw)
    write_csv(args.out_dir / "phase_breakdown.csv", phases)
    result = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": args.commit,
        "execution": {
            "nodes": 2,
            "physical_a100s": 16,
            "world_size": 16,
            "ranks_per_node": 8,
            "master_addr": gpt["anchor"]["run"]["distributed"]["master_addr"],
        },
        "method": {
            "normalization": "each panel is normalized to its own Maya-full R1 cumulative time",
            "maya_full": "complete execution, ordinary ungrouped trace construction, and full replay",
            "flexeva": "original selected-partition and selected-replay path",
            "plot_input": "figure6.csv; Maya-full and FlexEva only",
        },
        "gpt_case": gpt["anchor"]["case"],
        "moe_config": moe["method"]["config"],
        "raw_seconds": raw,
        "cumulative": cumulative,
        "phase_breakdown": phases,
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(args.out_dir / "result.json"), "figure6": str(args.out_dir / "figure6.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
