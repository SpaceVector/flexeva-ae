"""
Representative-rank planning for Maya-lite fixed-workload scaling.

This is intentionally simple and workload-agnostic. It does not attempt to
recover full workload semantics; it only supplies common fixed-workload rank
selection patterns so capture-side scaling is not hidden in one-off scripts.
"""

from __future__ import annotations


def plan_identity_rank_groups(world_size: int) -> dict[int, tuple[int, ...]]:
    """Return one profiled group per logical rank."""
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    return {rank: (rank,) for rank in range(world_size)}


def plan_megatron_pipeline_stage_groups(
    world_size: int,
    *,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
) -> dict[int, tuple[int, ...]]:
    """
    Group Megatron-style ranks by pipeline stage.

    This follows the local `gpt_3d.py` rank layout:

        rank = dp_rank * (tp * pp) + tp_rank * pp + pp_rank

    and chooses one representative worker per pipeline stage, collapsing all
    tensor-parallel and data-parallel replicas in that stage into the same
    profiled group.
    """
    if tensor_parallel_size <= 0:
        raise ValueError(
            f"tensor_parallel_size must be positive, got {tensor_parallel_size}"
        )
    if pipeline_parallel_size <= 0:
        raise ValueError(
            f"pipeline_parallel_size must be positive, got {pipeline_parallel_size}"
        )

    model_parallel_size = tensor_parallel_size * pipeline_parallel_size
    if world_size % model_parallel_size != 0:
        raise ValueError(
            "world_size must be divisible by tensor_parallel_size * "
            f"pipeline_parallel_size, got world_size={world_size}, "
            f"tp={tensor_parallel_size}, pp={pipeline_parallel_size}"
        )

    data_parallel_size = world_size // model_parallel_size
    groups: dict[int, tuple[int, ...]] = {}
    for pp_rank in range(pipeline_parallel_size):
        members: list[int] = []
        for dp_rank in range(data_parallel_size):
            for tp_rank in range(tensor_parallel_size):
                rank = (
                    dp_rank * model_parallel_size
                    + tp_rank * pipeline_parallel_size
                    + pp_rank
                )
                members.append(rank)
        groups[pp_rank] = tuple(sorted(members))
    return groups


def plan_profiled_rank_groups(
    world_size: int,
    *,
    strategy: str,
    tensor_parallel_size: int | None = None,
    pipeline_parallel_size: int | None = None,
) -> dict[int, tuple[int, ...]]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    mode = strategy.strip().lower()
    if mode in {"identity", "full_world"}:
        return plan_identity_rank_groups(world_size)
    if mode == "single":
        return {0: tuple(range(world_size))}
    if mode == "pairwise":
        groups: dict[int, tuple[int, ...]] = {}
        rank = 0
        while rank < world_size:
            members = (rank, rank + 1) if rank + 1 < world_size else (rank,)
            groups[rank] = members
            rank += 2
        return groups
    if mode == "megatron_pp_stage":
        if tensor_parallel_size is None or pipeline_parallel_size is None:
            raise ValueError(
                "megatron_pp_stage requires tensor_parallel_size and "
                "pipeline_parallel_size"
            )
        return plan_megatron_pipeline_stage_groups(
            world_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )
    raise ValueError(f"unsupported representative-rank strategy: {strategy}")


def profiled_ranks_for_groups(groups: dict[int, tuple[int, ...]]) -> tuple[int, ...]:
    return tuple(sorted(groups))
