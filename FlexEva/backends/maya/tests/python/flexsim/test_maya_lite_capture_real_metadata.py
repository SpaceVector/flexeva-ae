import json
import sys
import types
from pathlib import Path

import pytest

from flexsim.maya_lite.capture_real import (
    _merge_capture_manifest,
    _parse_route_metadata,
    main as capture_real_main,
    merge_real_trace_nodes,
)


def _required_route_metadata(**overrides):
    metadata = {
        "figure13_route": "paper_native_section42_dynamic_dedup",
        "auto_profiled_strategy": "identity",
        "dynamic_first_iteration_dedup": True,
        "collective_mode": "trace_only",
        "host_timing_mode": "measure",
        "host_timing_dispatch_scope": "host_machine",
        "host_timing_schedule_surface": "semantic",
        "validation_mode": "live_16gpu",
        "workload_args": ["--tp", "2", "--pp", "8", "--dp", "1"],
    }
    metadata.update(overrides)
    return metadata


def test_parse_route_metadata_coerces_booleans_and_rejects_invalid_items():
    assert _parse_route_metadata(
        [
            "figure13_route=paper_native_section42_dynamic_dedup",
            "dynamic_first_iteration_dedup=true",
            "paper_facing_closure=false",
        ]
    ) == {
        "figure13_route": "paper_native_section42_dynamic_dedup",
        "dynamic_first_iteration_dedup": True,
        "paper_facing_closure": False,
    }
    with pytest.raises(ValueError):
        _parse_route_metadata(["not-key-value"])


def test_capture_real_rejects_cpp_event_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workload = tmp_path / "workload.py"
    workload.write_text("print('should not run')\n", encoding="utf-8")
    fallback_cpp_event = types.ModuleType("cpp_event_py")
    fallback_cpp_event.__fallback__ = True
    monkeypatch.setitem(sys.modules, "cpp_event_py", fallback_cpp_event)

    with pytest.raises(RuntimeError, match="requires native cpp_event_py"):
        capture_real_main(
            [
                "--output-dir",
                str(tmp_path / "out"),
                "--profiled-ranks",
                "0",
                str(workload),
            ]
        )

    assert not (tmp_path / "out" / "rank_0.jsonl").exists()


def test_merge_capture_manifest_preserves_route_metadata(tmp_path: Path):
    manifest_path = tmp_path / "capture_manifest.json"

    _merge_capture_manifest(
        tmp_path,
        world_size=16,
        profiled_ranks=tuple(range(16)),
        profiled_rank_groups={rank: (rank,) for rank in range(16)},
        route_metadata={
            "figure13_route": "paper_native_section42_dynamic_dedup",
            "auto_profiled_strategy": "identity",
            "dynamic_first_iteration_dedup": True,
            "validation_mode": "live_16gpu",
        },
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["route_metadata"] == {
        "figure13_route": "paper_native_section42_dynamic_dedup",
        "auto_profiled_strategy": "identity",
        "dynamic_first_iteration_dedup": True,
        "validation_mode": "live_16gpu",
    }


def test_merge_real_trace_nodes_preserves_route_metadata_and_markers(tmp_path: Path):
    node0 = tmp_path / "node0"
    node1 = tmp_path / "node1"
    merged = tmp_path / "gpus_16"
    node0.mkdir()
    node1.mkdir()
    (node0 / "rank_0.jsonl").write_text('{"rank":0}\n', encoding="utf-8")
    (node0 / "rank_0.markers.jsonl").write_text('{"marker":"a"}\n', encoding="utf-8")
    (node1 / "rank_8.jsonl").write_text('{"rank":8}\n', encoding="utf-8")
    (node1 / "rank_8.markers.jsonl").write_text('{"marker":"b"}\n', encoding="utf-8")
    (node0 / "capture_manifest.json").write_text(
        json.dumps(
            {
                "original_world_size": 16,
                "profiled_ranks": [0],
                "profiled_rank_groups": {"0": [0]},
                "fidelity_windows": {
                    "0": {"start_ts": 10, "end_ts": 20, "source": "trace_markers"}
                },
                "route_metadata": {
                    "capture_command": "node0 command",
                    **_required_route_metadata(),
                },
            }
        ),
        encoding="utf-8",
    )
    (node1 / "capture_manifest.json").write_text(
        json.dumps(
            {
                "original_world_size": 16,
                "profiled_ranks": [8],
                "profiled_rank_groups": {"8": [8]},
                "fidelity_windows": {
                    "8": {"start_ts": 30, "end_ts": 40, "source": "trace_markers"}
                },
                "route_metadata": {
                    "capture_command": "node1 command",
                    **_required_route_metadata(),
                },
            }
        ),
        encoding="utf-8",
    )

    merge_real_trace_nodes([node0, node1], merged)

    payload = json.loads((merged / "capture_manifest.json").read_text(encoding="utf-8"))
    assert payload["original_world_size"] == 16
    assert payload["profiled_ranks"] == [0, 8]
    assert payload["profiled_rank_groups"] == {"0": [0], "8": [8]}
    assert payload["route_metadata"]["figure13_route"] == "paper_native_section42_dynamic_dedup"
    assert payload["route_metadata"]["auto_profiled_strategy"] == "identity"
    assert payload["route_metadata"]["node_capture_commands"] == [
        "node0 command",
        "node1 command",
    ]
    assert (merged / "rank_0.markers.jsonl").exists()
    assert (merged / "rank_8.markers.jsonl").exists()


def test_merge_real_trace_nodes_rejects_core_route_metadata_conflict(tmp_path: Path):
    node0 = tmp_path / "node0"
    node1 = tmp_path / "node1"
    merged = tmp_path / "gpus_16"
    node0.mkdir()
    node1.mkdir()
    (node0 / "rank_0.jsonl").write_text('{"rank":0}\n', encoding="utf-8")
    (node1 / "rank_8.jsonl").write_text('{"rank":8}\n', encoding="utf-8")
    (node0 / "capture_manifest.json").write_text(
        json.dumps(
            {
                "original_world_size": 16,
                "profiled_ranks": [0],
                "profiled_rank_groups": {"0": [0]},
                "route_metadata": _required_route_metadata(),
            }
        ),
        encoding="utf-8",
    )
    (node1 / "capture_manifest.json").write_text(
        json.dumps(
            {
                "original_world_size": 16,
                "profiled_ranks": [8],
                "profiled_rank_groups": {"8": [8]},
                "route_metadata": _required_route_metadata(auto_profiled_strategy="pairwise"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="auto_profiled_strategy"):
        merge_real_trace_nodes([node0, node1], merged)
    assert not merged.exists()


def test_merge_real_trace_nodes_rejects_missing_node_manifest(tmp_path: Path):
    node0 = tmp_path / "node0"
    node1 = tmp_path / "node1"
    merged = tmp_path / "gpus_16"
    node0.mkdir()
    node1.mkdir()
    (node0 / "rank_0.jsonl").write_text('{"rank":0}\n', encoding="utf-8")
    (node1 / "rank_8.jsonl").write_text('{"rank":8}\n', encoding="utf-8")
    (node0 / "capture_manifest.json").write_text(
        json.dumps(
            {
                "original_world_size": 16,
                "profiled_ranks": [0],
                "profiled_rank_groups": {"0": [0]},
                "route_metadata": _required_route_metadata(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing capture_manifest"):
        merge_real_trace_nodes([node0, node1], merged)
    assert not merged.exists()


def test_merge_real_trace_nodes_rejects_missing_required_route_metadata(tmp_path: Path):
    node0 = tmp_path / "node0"
    node1 = tmp_path / "node1"
    merged = tmp_path / "gpus_16"
    node0.mkdir()
    node1.mkdir()
    incomplete_metadata = _required_route_metadata()
    incomplete_metadata.pop("dynamic_first_iteration_dedup")
    for node, rank, representative in ((node0, 0, "0"), (node1, 8, "8")):
        (node / f"rank_{rank}.jsonl").write_text(f'{{"rank":{rank}}}\n', encoding="utf-8")
        (node / "capture_manifest.json").write_text(
            json.dumps(
                {
                    "original_world_size": 16,
                    "profiled_ranks": [rank],
                    "profiled_rank_groups": {representative: [rank]},
                    "route_metadata": incomplete_metadata,
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="dynamic_first_iteration_dedup"):
        merge_real_trace_nodes([node0, node1], merged)
    assert not merged.exists()
