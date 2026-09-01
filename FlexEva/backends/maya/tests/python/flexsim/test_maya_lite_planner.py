from __future__ import annotations

import pytest

from flexsim.maya_lite.planner import (
    plan_identity_rank_groups,
    plan_megatron_pipeline_stage_groups,
    plan_profiled_rank_groups,
    profiled_ranks_for_groups,
)


def test_plan_identity_rank_groups_profiles_every_rank() -> None:
    groups = plan_identity_rank_groups(4)

    assert groups == {
        0: (0,),
        1: (1,),
        2: (2,),
        3: (3,),
    }
    assert profiled_ranks_for_groups(groups) == (0, 1, 2, 3)


def test_plan_megatron_pipeline_stage_groups_collapses_tp_and_dp_within_stage() -> None:
    groups = plan_megatron_pipeline_stage_groups(
        512,
        tensor_parallel_size=8,
        pipeline_parallel_size=8,
    )

    assert tuple(groups) == tuple(range(8))
    assert all(len(members) == 64 for members in groups.values())
    assert groups[0][:4] == (0, 8, 16, 24)
    assert groups[7][-4:] == (487, 495, 503, 511)


def test_plan_profiled_rank_groups_supports_megatron_stage_strategy() -> None:
    groups = plan_profiled_rank_groups(
        64,
        strategy="megatron_pp_stage",
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
    )

    assert groups == {0: tuple(range(64))}
    assert profiled_ranks_for_groups(groups) == (0,)


def test_plan_profiled_rank_groups_supports_identity_strategy() -> None:
    groups = plan_profiled_rank_groups(4, strategy="identity")

    assert groups == {
        0: (0,),
        1: (1,),
        2: (2,),
        3: (3,),
    }
    assert profiled_ranks_for_groups(groups) == (0, 1, 2, 3)


def test_plan_profiled_rank_groups_supports_full_world_alias() -> None:
    groups = plan_profiled_rank_groups(3, strategy="full_world")

    assert groups == {
        0: (0,),
        1: (1,),
        2: (2,),
    }


def test_plan_profiled_rank_groups_rejects_incomplete_megatron_shape() -> None:
    with pytest.raises(ValueError, match="requires tensor_parallel_size"):
        plan_profiled_rank_groups(64, strategy="megatron_pp_stage")


def test_plan_megatron_pipeline_stage_groups_validates_divisibility() -> None:
    with pytest.raises(ValueError, match="world_size must be divisible"):
        plan_megatron_pipeline_stage_groups(
            10,
            tensor_parallel_size=4,
            pipeline_parallel_size=4,
        )
