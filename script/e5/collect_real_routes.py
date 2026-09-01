#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for candidate in (
    ROOT / "script" / "e3" / "capture",
    ROOT / "script" / "e3" / "figure6",
    ROOT / "flexmaya_ras" / "scripts",
):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from measure_routed_moe_fakecuda_similarity import (  # noqa: E402
    RouteCase,
    RoutedMoeConfig,
    run_routed_moe_path,
)
from measure_routed_moe_real_eval_breakdown import (  # noqa: E402
    ROUTE_PARTITION,
    write_source_files,
)


CONFIG = RoutedMoeConfig(
    backend="ns3",
    binary="extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default",
    world_size=16,
    ep_size=8,
    dp=2,
    steps=1,
    global_batch_size=16,
    seq_len=64,
    hidden_size=128,
    num_layers=2,
    num_heads=4,
    vocab_size=4096,
    num_experts=8,
    top_k=2,
    capacity_factor=1.25,
    micro_batches=2,
    dtype="bf16",
)
ANCHOR = RouteCase(0, "anchor_route_0_1", "anchor route (0,1)", (0, 1))


def routes() -> list[RouteCase]:
    pairs = [
        (left, right)
        for left in range(CONFIG.num_experts)
        for right in range(CONFIG.num_experts)
        if left != right and (left, right) != ANCHOR.experts
    ][:32]
    return [
        RouteCase(index + 1, f"candidate_{index:02d}_route_{left}_{right}", f"route ({left},{right})", (left, right))
        for index, (left, right) in enumerate(pairs)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect fresh FakeCUDA route grounding for paper-aligned E5.")
    sub = parser.add_subparsers(dest="action", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--out-dir", type=Path, required=True)
    capture.add_argument("--maya-root", type=Path, required=True)
    capture.add_argument("--python", type=Path, default=Path(sys.executable))
    capture.add_argument("--proot", type=Path, required=True)
    capture.add_argument("--local-device-count", type=int, default=8)
    capture.set_defaults(
        reuse_existing_traces=False,
        no_route_p2p_probe=True,
        source_region_markers=True,
        sync_before_step_window=False,
    )
    sub.add_parser("self-test")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fingerprint(trace_dir: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    total_bytes = 0
    for rank in range(CONFIG.world_size):
        for suffix in (".jsonl", "_markers.jsonl"):
            path = trace_dir / f"rank_{rank}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"missing trace: {path}")
            digest.update(path.name.encode() + b"\0")
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
                    total_bytes += len(chunk)
    return {"file_count": CONFIG.world_size * 2, "total_bytes": total_bytes, "sha256": digest.hexdigest()}


def capture_one(args: argparse.Namespace, route: RouteCase, case_dir: Path, seed: int) -> tuple[dict, dict[str, Path]]:
    sources = write_source_files(case_dir, config=CONFIG, route_case=route)
    run = run_routed_moe_path(args, CONFIG, route_case=route, seed=seed, case_dir=case_dir)
    if int(run["return_code"]) != 0:
        raise RuntimeError(f"{route.name} failed; see {run['stderr']}")
    run["trace_fingerprint"] = fingerprint(Path(str(run["trace_dir"])))
    return run, sources


def source_paths(root: Path, paths: dict[str, Path]) -> dict[str, str]:
    return {name: str(path.resolve().relative_to(root)) for name, path in paths.items()}


def run_capture(args: argparse.Namespace) -> int:
    root = args.out_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "capture_manifest.json"
    manifest: dict[str, object] = {
        "status": "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {
            "role": "fresh route/lineage grounding; paper-shape state construction is measured separately",
            "capture": "Routed-MoE through Maya FakeCUDA at 16 logical ranks",
            "candidate_variation": "32 distinct forced top-2 expert routes",
            "raw_trace_integrity": "SHA-256 over 16 rank traces and marker files per candidate",
        },
        "config": asdict(CONFIG),
        "candidate_count": 32,
        "anchor": None,
        "candidates": [],
    }
    write_json(manifest_path, manifest)

    anchor_run, anchor_sources = capture_one(args, ANCHOR, root / "anchor", 8100)
    manifest["anchor"] = {
        "route": asdict(ANCHOR),
        "source_paths": source_paths(root, anchor_sources),
        "run": anchor_run,
    }
    write_json(manifest_path, manifest)

    captured = []
    for index, route in enumerate(routes()):
        case_dir = root / "candidates" / route.name
        sources = write_source_files(
            case_dir,
            config=CONFIG,
            route_case=route,
            anchor_paths=anchor_sources,
            changed_ops=(ROUTE_PARTITION,),
        )
        run = run_routed_moe_path(args, CONFIG, route_case=route, seed=8200 + index, case_dir=case_dir)
        if int(run["return_code"]) != 0:
            raise RuntimeError(f"{route.name} failed; see {run['stderr']}")
        run["trace_fingerprint"] = fingerprint(Path(str(run["trace_dir"])))
        captured.append(
            {
                "candidate_id": route.name,
                "route": asdict(route),
                "source_paths": source_paths(root, sources),
                "run": run,
            }
        )
        manifest["candidates"] = captured
        write_json(manifest_path, manifest)
        print(json.dumps({"captured": len(captured), "total": 32}), flush=True)

    manifest["status"] = "complete"
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(manifest_path, manifest)
    print(json.dumps({"capture_manifest": str(manifest_path), "candidate_count": 32}, sort_keys=True))
    return 0


def self_test() -> int:
    candidates = routes()
    assert len(candidates) == len({item.name for item in candidates}) == len({item.experts for item in candidates}) == 32
    assert all(left != right for left, right in (item.experts for item in candidates))
    print("E5 route collector self-test: PASS")
    return 0


def main() -> int:
    args = parse_args()
    return self_test() if args.action == "self-test" else run_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
