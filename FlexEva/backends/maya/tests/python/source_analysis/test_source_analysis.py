from __future__ import annotations

import ast
import json
from pathlib import Path

from astparser.codegen import generate_code
from astparser.parser import parse_entry_files
from flexsim.anchor_reuse import (
    BlackBoxBoundaryRule,
    ControlRegionKind,
    LogicScopeSpec,
    LogicSliceGranularityMode,
    LogicSliceGranularityPolicy,
    build_control_flow_graphs,
    build_resilient_anchor_state,
    capture_program_logic_from_instrumented_code,
    resilient_anchor_state_summary,
)


def _condition_names(condition: str) -> list[str]:
    tree = ast.parse(condition, mode="eval")
    return sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})


def _write_micro_trace(trace_dir: Path) -> None:
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "original_world_size": 4,
                "profiled_ranks": [0, 2],
                "profiled_rank_groups": {"0": [0, 1], "2": [2, 3]},
            }
        ),
        encoding="utf-8",
    )
    for rank, offset in ((0, 0), (2, 1)):
        rows = (
            {
                "ts": 10 + offset,
                "pid": rank + 1,
                "tid": rank + 1,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            },
            {
                "ts": 20 + offset,
                "pid": rank + 1,
                "tid": rank + 1,
                "mod": "libnccl.so.2",
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "count": 1024,
                "datatype": "fp16",
                "op": "sum",
            },
            {
                "ts": 30 + offset,
                "pid": rank + 1,
                "tid": rank + 1,
                "mod": "libcudart.so.12",
                "api": "cudaMemcpyAsync",
                "type": "mem_copy",
            },
        )
        (trace_dir / f"rank_{rank}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )


def test_ast_dryrun_cfg_and_active_rank_partition(tmp_path: Path) -> None:
    source = tmp_path / "toy.py"
    source.write_text(
        """
from pyextend.runtime import SimEnv


def main(env: SimEnv):
    rank = env.default_resolve("rank")
    world_size = env.default_resolve("world_size")
    if rank < world_size // 2:
        return rank + 1
    return rank - 1
""".lstrip(),
        encoding="utf-8",
    )
    cpp_source = tmp_path / "toy_kernel.cpp"
    cpp_source.write_text(
        """
float fast_path(float x);
float slow_path(float x);

float kernel(float x, bool first_half) {
    if (first_half) {
        return fast_path(x);
    }
    return slow_path(x);
}
""".lstrip(),
        encoding="utf-8",
    )

    branches, decorators = parse_entry_files([source])
    assert len(branches) == 1
    branch = branches[0]
    assert branch.condition is not None
    instrumented = tmp_path / "toy_instrumented.py"
    generate_code([source], branches, decorators, instrumented)
    assert f"mark_cond(rank < world_size // 2, {branch.lineno})" in instrumented.read_text(encoding="utf-8")

    capture = capture_program_logic_from_instrumented_code(
        instrumented,
        world_size=4,
        branch_var_map={branch.lineno: _condition_names(branch.condition)},
        logic_scope=LogicScopeSpec(
            scope_id="ae.source_analysis",
            selected_paths=(str(instrumented), str(cpp_source)),
            selected_functions=("main", "kernel"),
        ),
        boundary_rule=BlackBoxBoundaryRule(opaque_call_names=("fast_path", "slow_path")),
        slice_granularity_policy=LogicSliceGranularityPolicy(
            mode=LogicSliceGranularityMode.CFG_BOUNDARY_AWARE,
            max_cfg_blocks_per_slice=2,
            split_on_emission=True,
        ),
    )

    assert len(capture.program_logic.points) == 1
    point = capture.program_logic.points[0]
    assert point.branch_ids == (branch.lineno,)
    assert point.rank_groups == ((0, 1), (2, 3))
    assert "branch_vars=rank,world_size" in point.notes

    python_cfg = build_control_flow_graphs(instrumented, selected_functions=("main",))
    cpp_cfg = build_control_flow_graphs(cpp_source, selected_functions=("kernel",))
    assert any(block.kind.value == "branch_header" for cfg in python_cfg for block in cfg.blocks)
    assert any(block.kind.value == "branch_header" for cfg in cpp_cfg for block in cfg.blocks)
    assert capture.control_region_tree is not None
    assert any(region.kind == ControlRegionKind.BRANCH for region in capture.control_region_tree.regions)
    assert capture.logic_state_store is not None
    assert capture.logic_slice_graph is not None

    trace_dir = tmp_path / "trace"
    _write_micro_trace(trace_dir)
    state = build_resilient_anchor_state(
        anchor_candidate_id="toy-anchor",
        workload_family="gpt",
        trace_dir=trace_dir,
        anchor_capture=capture,
        anchor_code_paths=(instrumented, cpp_source),
    )
    summary = resilient_anchor_state_summary(state)
    assert len(state.code.baseline_files) == 2
    assert state.semantic.logic_slice_count > 0
    assert state.runtime_values.distributions
    assert state.trace.world_size == 4
    assert state.trace.lineage.segment_bundle.segments
    assert summary["anchor_candidate_id"] == "toy-anchor"
    print(
        "source-analysis: python=ast cpp=line-regex "
        f"branch={branch.lineno} active-rank-groups={point.rank_groups} "
        f"logic-slices={len(capture.logic_state_store.slices)} "
        "ras-slots=code,semantic,runtime-values,trace"
    )
