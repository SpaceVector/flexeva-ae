#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
artifact_root=$(realpath -e "$script_dir/../..")
guard_script="$script_dir/server_guard.sh"

usage() {
    cat <<'EOF'
Usage:
  script/e3/server.sh self-test
  script/e3/server.sh check
  script/e3/server.sh status RUN_ID
  script/e3/server.sh run RUN_ID GPU_COUNT -- COMMAND [ARG ...]

Run this script directly on the experiment server. It never opens SSH.
Use @python as COMMAND to select the configured canonical Python.

Set:
  AE_NODE_ROOT=/large/server/filesystem
Optional:
  AE_CANONICAL_PYTHON (defaults to the clone-local .venv),
  AE_EXPECTED_HOSTNAME, AE_EXPECTED_FILESYSTEM, AE_GPU_LOAD_CONTROL
EOF
}

valid_run_id() {
    [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
}

show_status() {
    local run_id=$1 node_root_real run_root run_dir
    node_root_real=$(realpath -e "$node_root") \
        || { echo "cannot resolve NODE_ROOT: $node_root" >&2; return 2; }
    run_root="$node_root_real/eurosys27-ae/runs"
    run_dir="$run_root/$run_id"
    [[ -f $run_dir/status.env ]] \
        || { echo "run status does not exist: $run_dir/status.env" >&2; return 2; }
    echo "run_id=$run_id"
    echo "run_dir=$run_dir"
    cat "$run_dir/status.env"
    for name in command.txt environment.txt stdout.log stderr.log interpretation.md; do
        [[ ! -f $run_dir/$name ]] || echo "${name//[.-]/_}=$run_dir/$name"
    done
}

select_profile() {
    local actual_hostname=$1 repo_root default_python
    repo_root=$(git -C "$artifact_root" rev-parse --show-toplevel 2>/dev/null || true)
    default_python="${repo_root:-$artifact_root}/.venv/bin/python"
    expected_hostname=${AE_EXPECTED_HOSTNAME:-$actual_hostname}
    node_root=${AE_NODE_ROOT:-}
    [[ -n $node_root ]] || {
        echo "set AE_NODE_ROOT to the experiment filesystem" >&2
        return 2
    }
    export AE_EXPECTED_FILESYSTEM=${AE_EXPECTED_FILESYSTEM:-$(stat -f -c %T "$node_root")}
    [[ $AE_EXPECTED_FILESYSTEM != overlay && $AE_EXPECTED_FILESYSTEM != tmpfs ]] || {
        echo "artifact data cannot use filesystem type: $AE_EXPECTED_FILESYSTEM" >&2
        return 2
    }
    export AE_GPU_LOAD_CONTROL=${AE_GPU_LOAD_CONTROL:-none}
    canonical_python=${AE_CANONICAL_PYTHON:-$default_python}
}

relative_workdir() {
    local node_root_real
    node_root_real=$(realpath -e "$node_root")
    case "$artifact_root/" in
        "$node_root_real/"*) printf '%s\n' "${artifact_root#"$node_root_real/"}" ;;
        *)
            echo "artifact root must be deployed below AE_NODE_ROOT: $artifact_root" >&2
            return 2
            ;;
    esac
}

self_test() {
    valid_run_id reviewer-quick-01
    ! valid_run_id ../escape
    bash -n "$guard_script"
    "$guard_script" self-test
    echo "direct server entry self-test: PASS"
}

action=${1:-}
case "$action" in
    self-test)
        [[ $# == 1 ]] || { usage >&2; exit 2; }
        self_test
        ;;
    check)
        [[ $# == 1 ]] || { usage >&2; exit 2; }
        select_profile "$(hostname)"
        exec "$guard_script" check "$expected_hostname" "$node_root" "$canonical_python"
        ;;
    status)
        [[ $# == 2 ]] || { usage >&2; exit 2; }
        run_id=$2
        valid_run_id "$run_id" || { echo "invalid RUN_ID: $run_id" >&2; exit 2; }
        select_profile "$(hostname)"
        show_status "$run_id"
        ;;
    run)
        [[ $# -ge 5 ]] || { usage >&2; exit 2; }
        run_id=$2
        gpu_count=$3
        shift 3
        [[ ${1:-} == -- ]] || { usage >&2; exit 2; }
        shift
        (( $# > 0 )) || { usage >&2; exit 2; }
        valid_run_id "$run_id" \
            || { echo "invalid RUN_ID: $run_id" >&2; exit 2; }
        [[ $gpu_count == 1 || $gpu_count == 2 || $gpu_count == 4 || $gpu_count == 8 ]] \
            || { echo "GPU_COUNT must be 1, 2, 4, or 8" >&2; exit 2; }
        select_profile "$(hostname)"
        workdir=$(relative_workdir)
        exec "$guard_script" \
            run "$expected_hostname" "$node_root" "$canonical_python" \
            "$run_id" "$gpu_count" "$workdir" \
            "${GPU_IDLE_MEMORY_LIMIT_MIB:-64}" \
            "${GPU_IDLE_SAMPLES:-5}" \
            "${GPU_IDLE_SAMPLE_INTERVAL_S:-2}" \
            "${MIN_GPFS_FREE_GIB:-20}" \
            -- "$@"
        ;;
    -h|--help|help|"")
        usage
        ;;
    *)
        echo "unknown action: $action" >&2
        usage >&2
        exit 2
        ;;
esac
