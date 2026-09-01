from __future__ import annotations

from pathlib import Path

import pytest

from paper_resilient_anchor_state import (
    DEFAULT_MAYA_V2_SOURCE_ROOT,
    MAYA_V2_LOAD_SKEW_CASE_ID,
    build_maya_v2_load_skew_case,
    build_moe_load_skew_grounding_snapshot,
    build_optional_live_capture_command,
    build_selective_refresh_plan,
    build_trace_ras_projection,
)


def test_build_maya_v2_load_skew_case_selects_latest_round_and_local_entries() -> None:
    case = build_maya_v2_load_skew_case()

    assert case.case_id == MAYA_V2_LOAD_SKEW_CASE_ID
    assert case.family_id == "moe_workload_family_v2"
    assert case.workload_kind == "routed_moe_training"
    assert case.round.round_id == "round_load_skew_r1"
    assert case.round.goal_id == "straggler_load_skew"
    assert case.round.candidate_ids == (
        "overflow_reroute",
        "layout_striped",
        "local_backup_reroute",
        "balanced_secondary_route",
    )
    assert [candidate.candidate_id for candidate in case.candidates] == [
        "anchor_baseline",
        "overflow_reroute",
        "layout_striped",
        "local_backup_reroute",
        "balanced_secondary_route",
    ]
    assert case.trace_layer_policy["raw_trace_role"] == "audit_input_evidence"
    assert case.trace_layer_policy["modeling_surface"] == "normalized_collated_maya_events"
    assert case.live_capture_mode == "optional_fake_cuda_proot_integration"

    for candidate in case.candidates:
        entry = Path(candidate.entry)
        assert entry.exists()
        assert str(entry).startswith(str(DEFAULT_MAYA_V2_SOURCE_ROOT))
        assert "/home/muxi/test-sim/" not in str(entry)

    payload = case.to_dict()
    assert payload["round"]["parallel_budget"] == 4
    assert payload["grounding_targets"] == [
        "route_decision",
        "expert_load",
        "overflow_state",
        "remote_dispatch",
        "collective_stage",
    ]


def test_moe_load_skew_grounding_snapshot_tracks_runtime_control_values() -> None:
    snapshot = build_moe_load_skew_grounding_snapshot(
        route_decisions_by_rank={
            0: (0, 1),
            1: (1, 3),
            2: (0, 4),
            3: (5,),
        },
        local_experts_by_rank={
            0: (0, 1),
            1: (0, 1),
            2: (4, 5),
            3: (4, 5),
        },
        overflow_tokens=6,
        dropped_tokens=2,
        rerouted_tokens=4,
        collective_stage="dispatch_all_to_all_then_expert_compute",
    )

    assert snapshot.source == "minimal_cpu_semantic_pass"
    assert snapshot.expert_loads == {0: 2, 1: 2, 3: 1, 4: 1, 5: 1}
    assert snapshot.overflow_tokens == 6
    assert snapshot.dropped_tokens == 2
    assert snapshot.rerouted_tokens == 4
    assert snapshot.local_dispatch_tokens == 5
    assert snapshot.remote_dispatch_tokens == 2
    assert snapshot.remote_token_fraction == pytest.approx(2 / 7)
    assert snapshot.required_grounding_points == (
        "route_decision",
        "expert_load",
        "overflow_state",
        "remote_dispatch",
        "collective_stage",
    )


def test_trace_ras_projection_uses_rank_local_host_and_stream_lanes() -> None:
    projection = build_trace_ras_projection(
        {
            0: (
                {"id": "h0", "type": "host_delay", "api": "__hostDelay__", "source_boundary_id": "route"},
                {
                    "id": "k0",
                    "type": "kernel_launch",
                    "api": "cudaLaunchKernel",
                    "stream_id": "7",
                    "source_boundary_id": "route",
                },
                {
                    "id": "g0",
                    "type": "blas_compute",
                    "api": "cublasGemmEx",
                    "stream_id": "7",
                    "source_boundary_id": "route",
                },
                {
                    "id": "sync0",
                    "type": "stream_op",
                    "api": "cudaStreamSynchronize",
                    "stream_id": "7",
                    "source_boundary_id": "route",
                },
                {
                    "id": "nccl0",
                    "type": "nccl_collective",
                    "api": "ncclAllToAll",
                    "stream_id": "comm",
                    "source_boundary_id": "dispatch",
                },
            ),
            1: (
                {"id": "ctx1", "type": "context_op", "api": "cudaGetDevice"},
                {"id": "h1", "type": "host_delay", "api": "__hostDelay__", "source_boundary_id": "route"},
                {
                    "id": "copy1",
                    "type": "mem_copy",
                    "api": "cudaMemcpyAsync",
                    "stream_id": "copy",
                    "source_boundary_id": "dispatch",
                },
            ),
        }
    )

    lanes = {lane.lane_id: lane for lane in projection.lanes}
    assert projection.raw_event_count == 8
    assert projection.modeling_event_count == 7
    assert lanes["rank:0/host"].event_count == 1
    assert lanes["rank:0/stream:7"].event_count == 3
    assert lanes["rank:0/stream:comm"].event_count == 1
    assert lanes["rank:1/host"].event_count == 1
    assert lanes["rank:1/stream:copy"].event_count == 1

    partitions = {partition.partition_id: partition for partition in projection.partitions}
    assert partitions["rank:0/stream:7:p0000"].event_ids == ("k0", "g0")
    assert partitions["rank:0/stream:7:p0001"].event_ids == ("sync0",)
    assert partitions["rank:0/stream:7:p0001"].boundary_kind == "sync"
    assert partitions["rank:0/stream:comm:p0000"].event_ids == ("nccl0",)
    assert partitions["rank:0/stream:comm:p0000"].boundary_kind == "sync"
    assert partitions["rank:0/stream:comm:p0000"].source_boundary_id == "dispatch"


def test_selective_refresh_plan_reruns_representatives_and_escalates_collectives() -> None:
    rank_groups = ((0, 1, 2, 3), (4, 5, 6, 7))
    plan = build_selective_refresh_plan(
        candidate_id="overflow_reroute",
        logical_world_size=8,
        rank_groups=rank_groups,
        affected_group_indices=(1,),
        affected_boundary_ids=("route_overflow",),
        cached_partition_ids=("rank:0/stream:7:p0000", "rank:1/host:p0000"),
    )

    assert plan.affected_rank_groups == ((4, 5, 6, 7),)
    assert plan.refreshed_ranks == (4,)
    assert plan.reused_ranks == (0, 1, 2, 3, 5, 6, 7)
    assert plan.escalation_required is False
    assert plan.cached_partition_ids == ("rank:0/stream:7:p0000", "rank:1/host:p0000")

    escalated = build_selective_refresh_plan(
        candidate_id="overflow_reroute",
        logical_world_size=8,
        rank_groups=rank_groups,
        affected_group_indices=(1,),
        affected_boundary_ids=("dispatch_collective",),
        collective_shape_changed=True,
        collective_member_ranks=(4, 5, 6, 7),
    )

    assert escalated.refreshed_ranks == (4, 5, 6, 7)
    assert escalated.reused_ranks == (0, 1, 2, 3)
    assert escalated.escalation_required is True
    assert escalated.escalation_reason == "collective membership/order/shape changed"


def test_optional_live_capture_command_is_gated_fake_cuda_integration() -> None:
    case = build_maya_v2_load_skew_case()
    command = build_optional_live_capture_command(
        case,
        candidate_id="overflow_reroute",
        output_dir="/tmp/maya-v2-overflow",
        logical_world_size=8,
        profiled_rank_groups="0:0,1,2,3;4:4,5,6,7",
        steps=1,
    )

    assert command[:3] == ("python", "-m", "flexsim.maya_lite.capture_emulated")
    assert "--collective-mode" in command
    assert "trace_only" in command
    assert "--trace-surface" in command
    assert "all" in command
    assert "--frun" in command
    assert any(item.endswith("fake-cuda/frun") for item in command)
    assert any(item.endswith("moe_routed_family_v1/overflow_reroute.py") for item in command)
    assert command[-2:] == ("--steps", "1")
