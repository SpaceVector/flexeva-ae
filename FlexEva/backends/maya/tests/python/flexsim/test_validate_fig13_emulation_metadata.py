import json
import shlex

from paper.maya_lite.validate_fig13_emulation import (
    _build_real_torchrun_command,
    _route_metadata_for_remote_capture,
)


def test_build_real_torchrun_command_includes_required_route_metadata():
    workload_args = ["--steps", "1", "--tp", "2", "--pp", "8", "--dp", "1"]
    route_metadata = _route_metadata_for_remote_capture(workload_args)

    command = _build_real_torchrun_command(
        remote_root="/repo",
        workload_relpath="tests/workloads/fake_cuda/maya_fig13_megatron.py",
        workload_args=workload_args,
        output_dir="/tmp/out",
        nnodes=2,
        nproc_per_node=8,
        node_rank=1,
        master_addr="192.168.11.179",
        master_port=29531,
        route_metadata=route_metadata,
    )

    assert "--auto-profiled-strategy identity" in command
    for key in [
        "figure13_route",
        "auto_profiled_strategy",
        "dynamic_first_iteration_dedup",
        "collective_mode",
        "host_timing_mode",
        "host_timing_dispatch_scope",
        "host_timing_schedule_surface",
        "validation_mode",
        "workload_args",
    ]:
        assert f"--route-metadata {key}=" in command
    assert "dynamic_first_iteration_dedup=false" in command
    assert "workload_args=" in command
    assert "--master_addr=192.168.11.179" in command

    tokens = shlex.split(command)
    metadata_values = [
        tokens[index + 1]
        for index, token in enumerate(tokens)
        if token == "--route-metadata"
    ]
    metadata = dict(value.split("=", 1) for value in metadata_values)
    assert set(metadata) == set(route_metadata)
    assert len(metadata_values) == 9
    assert metadata["dynamic_first_iteration_dedup"] == "false"
    assert json.loads(metadata["workload_args"]) == workload_args
