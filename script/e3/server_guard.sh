#!/usr/bin/env bash
set -Euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
environment_blocked_exit=75
run_dir=""
run_status="not_started"
load_stopped=0
run_lock_fd=""
load_control="${AE_GPU_LOAD_CONTROL:-none}"
expected_filesystem="${AE_EXPECTED_FILESYSTEM:-gpfs}"

write_status() {
    [[ -n $run_dir && -d $run_dir ]] || return 0
    {
        printf 'status=%s\n' "$run_status"
        printf 'updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        if [[ -n ${command_exit_code:-} ]]; then
            printf 'command_exit_code=%s\n' "$command_exit_code"
        fi
    } > "$run_dir/status.env"
    return 0
}

write_interpretation() {
    local conclusion=$1
    [[ -n $run_dir && -d $run_dir ]] || return 0
    {
        echo "# Interpretation"
        echo
        echo "- Status: \`$run_status\`"
        echo "- Scientific conclusion: $conclusion"
    } > "$run_dir/interpretation.md"
}

blocked() {
    run_status="environment_blocked"
    echo "ENVIRONMENT_BLOCKED: $*" >&2
    set +e
    write_status
    write_interpretation "None. Environment-blocked runs must not be used as negative scientific evidence."
    exit "$environment_blocked_exit"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || blocked "required command is unavailable: $1"
}

acquire_run_lock() {
    local lock_root=$1 run_id=$2 lock_path holder
    require_command flock
    mkdir -p "$lock_root" || blocked "cannot create GPU run lock directory: $lock_root"
    lock_path="$lock_root/gpu-run.lock"
    : >> "$lock_path" || blocked "cannot open GPU run lock: $lock_path"
    exec {run_lock_fd}<>"$lock_path" || blocked "cannot open GPU run lock descriptor"
    if ! flock -n "$run_lock_fd"; then
        holder=$(tr '\n' ' ' < "$lock_path" 2>/dev/null || true)
        blocked "another guarded GPU run holds $lock_path${holder:+ ($holder)}"
    fi
    {
        printf 'run_id=%s\n' "$run_id"
        printf 'pid=%s\n' "$$"
        printf 'hostname=%s\n' "$(hostname)"
        printf 'acquired_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$lock_path"
    cp "$lock_path" "$run_dir/run-lock.txt" \
        || blocked "cannot preserve GPU run lock metadata"
}

release_run_lock() {
    [[ -n $run_lock_fd ]] || return 0
    flock -u "$run_lock_fd" 2>/dev/null || true
    exec {run_lock_fd}>&-
    run_lock_fd=""
}

base_check() {
    local expected_hostname=$1 node_root=$2 canonical_python=$3
    local actual_hostname filesystem_type

    actual_hostname=$(hostname) || blocked "cannot read hostname"
    [[ $actual_hostname == "$expected_hostname" ]] || blocked "hostname=$actual_hostname expected=$expected_hostname"
    [[ -d $node_root ]] || blocked "GPFS root is unavailable: $node_root"
    filesystem_type=$(stat -f -c %T "$node_root") || blocked "cannot inspect GPFS root: $node_root"
    [[ $filesystem_type == "$expected_filesystem" ]] \
        || blocked "NODE_ROOT filesystem=$filesystem_type expected=$expected_filesystem"
    [[ -x $canonical_python ]] || blocked "canonical Python is unavailable: $canonical_python"
    if [[ $load_control != none ]]; then
        [[ -x $load_control ]] || blocked "GPU load controller is unavailable: $load_control"
    fi
}

find_nsys() {
    local candidate
    for candidate in \
        "$(command -v nsys 2>/dev/null || true)" \
        /usr/local/bin/nsys \
        /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys; do
        [[ -n $candidate && -x $candidate ]] && { printf '%s\n' "$candidate"; return 0; }
    done
    return 1
}

static_check() {
    local expected_hostname=$1 node_root=$2 canonical_python=$3 min_free_gib=${4:-20}
    local python_report gpu_report gpu_count topology nsys_bin free_kib required_kib

    base_check "$expected_hostname" "$node_root" "$canonical_python"
    require_command nvidia-smi
    require_command g++
    require_command cmake
    [[ -x /usr/local/cuda/bin/nvcc ]] || blocked "CUDA compiler is unavailable"
    [[ -x /usr/local/cuda/bin/ncu ]] || blocked "Nsight Compute binary is unavailable"
    [[ -x /usr/bin/rdma ]] || blocked "RDMA utility is unavailable"
    nsys_bin=$(find_nsys) || blocked "Nsight Systems is unavailable"

    python_report=$("$canonical_python" - <<'PY'
import platform
import torch

print(f"python={platform.python_version()}")
print(f"python_compatible={int((3, 10) <= tuple(map(int, platform.python_version_tuple()[:2])) < (3, 14))}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={int(torch.cuda.is_available())}")
print(f"cuda_device_count={torch.cuda.device_count()}")
PY
    ) || blocked "canonical Python cannot import PyTorch/CUDA"
    grep -qx 'python_compatible=1' <<< "$python_report" || blocked "Python must be 3.10 through 3.13"
    grep -qx 'torch=2.8.0+cu128' <<< "$python_report" || blocked "unexpected PyTorch version"
    grep -qx 'torch_cuda=12.8' <<< "$python_report" || blocked "unexpected PyTorch CUDA version"
    grep -qx 'cuda_available=1' <<< "$python_report" || blocked "PyTorch CUDA is unavailable"
    grep -qx 'cuda_device_count=8' <<< "$python_report" || blocked "PyTorch does not see 8 GPUs"

    gpu_report=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits) \
        || blocked "nvidia-smi GPU query failed"
    gpu_count=$(wc -l <<< "$gpu_report" | tr -d ' ')
    [[ $gpu_count == 8 ]] || blocked "GPU count=$gpu_count expected=8"
    awk -F, '
        {
            gsub(/^[ \t]+|[ \t]+$/, "", $2)
            gsub(/^[ \t]+|[ \t]+$/, "", $3)
            if ($2 != "NVIDIA A100-SXM4-80GB" || $3 + 0 < 81920) exit 1
        }
    ' <<< "$gpu_report" || blocked "GPU model or memory does not match 8x A100-SXM4-80GB"

    topology=$(nvidia-smi topo -m) || blocked "cannot inspect GPU topology"
    awk '
        /^GPU[0-7][[:space:]]/ {
            row = substr($1, 4) + 0
            seen++
            for (col = 0; col < 8; col++) {
                value = $(col + 2)
                if (col == row) {
                    if (value != "X") bad = 1
                } else if (value != "NV12") {
                    bad = 1
                }
            }
        }
        END { exit !(seen == 8 && !bad) }
    ' <<< "$topology" || blocked "GPU topology is not full NV12"

    /usr/local/cuda/bin/nvcc --version | grep -q 'release 12.8' || blocked "CUDA toolkit is not 12.8"
    [[ $(g++ -dumpfullversion -dumpversion) == 11.4* ]] || blocked "g++ is not 11.4.x"
    [[ $(cmake --version | awk 'NR == 1 { print $3 }') == 3.22.1 ]] || blocked "CMake is not 3.22.1"

    free_kib=$(df -Pk "$node_root" | awk 'NR == 2 { print $4 }')
    required_kib=$((min_free_gib * 1024 * 1024))
    (( free_kib >= required_kib )) || blocked "GPFS has less than ${min_free_gib} GiB free"

    echo "status=STATIC_CHECK_PASSED"
    echo "hostname=$(hostname)"
    echo "node_root=$node_root"
    echo "filesystem=$expected_filesystem"
    echo "$python_report"
    echo "driver_versions=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u | paste -sd, -)"
    echo "gpu_count=8"
    echo "gpu_model=NVIDIA A100-SXM4-80GB"
    echo "gpu_topology=full_NV12"
    echo "cuda_toolkit=$(/usr/local/cuda/bin/nvcc --version | tail -n 1)"
    echo "gxx=$(g++ -dumpfullversion -dumpversion)"
    echo "cmake=$(cmake --version | awk 'NR == 1 { print $3 }')"
    echo "nsys=$nsys_bin"
    TMPDIR="$node_root" "$nsys_bin" --version 2>&1 | sed 's/^/nsys_version=/'
    echo "ncu=/usr/local/cuda/bin/ncu (counters disabled: ERR_NVGPUCTRPERM)"
    echo "rdma=/usr/bin/rdma"
    echo "gpfs_free_kib=$free_kib"
    echo "gpu_snapshot_begin"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
    echo "gpu_snapshot_end"
    echo "topology_begin"
    echo "$topology"
    echo "topology_end"
}

wait_for_idle_gpus() {
    local memory_limit=$1 samples=$2 interval=$3
    local attempt compute_rows snapshot ready=0 sample

    for attempt in $(seq 1 15); do
        compute_rows=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null) \
            || blocked "cannot query GPU compute processes"
        snapshot=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits) \
            || blocked "cannot query GPU idle state"
        {
            echo "attempt=$attempt"
            echo "compute_processes_begin"
            echo "$compute_rows"
            echo "compute_processes_end"
            echo "gpu_state_begin"
            echo "$snapshot"
            echo "gpu_state_end"
        } >> "$run_dir/gpu-idle-check.log"

        if ! grep -q '[^[:space:]]' <<< "$compute_rows" && awk -F, -v limit="$memory_limit" '
            {
                gsub(/ /, "", $2)
                gsub(/ /, "", $3)
                if ($2 + 0 > limit || $3 + 0 != 0) exit 1
            }
        ' <<< "$snapshot"; then
            ready=1
            break
        fi
        sleep 2
    done
    (( ready == 1 )) || blocked "GPU processes, memory, or utilization did not become idle"

    for sample in $(seq 1 "$samples"); do
        snapshot=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits) \
            || blocked "cannot sample sustained GPU idle state"
        {
            echo "sustained_sample=$sample"
            echo "$snapshot"
        } >> "$run_dir/gpu-idle-check.log"
        awk -F, -v limit="$memory_limit" '
            {
                gsub(/ /, "", $2)
                gsub(/ /, "", $3)
                if ($2 + 0 > limit || $3 + 0 != 0) exit 1
            }
        ' <<< "$snapshot" || blocked "GPU state was not continuously idle"
        (( sample == samples )) || sleep "$interval"
    done
}

check_gpfs_write() {
    local canonical_python=$1 probe_path="$run_dir/.gpfs-write-probe"
    "$canonical_python" - "$probe_path" <<'PY'
import os
import sys

path = sys.argv[1]
try:
    with open(path, "wb") as stream:
        stream.write(b"\0" * (1024 * 1024))
        stream.flush()
        os.fsync(stream.fileno())
except BaseException:
    try:
        os.unlink(path)
    except OSError:
        pass
    raise
else:
    os.unlink(path)
PY
}

restore_load() {
    local original_exit=$?
    release_run_lock
    if (( load_stopped == 1 )); then
        set +e
        "$load_control" start > "$run_dir/gpu-load-restore.log" 2>&1
        restore_exit=$?
        set -e
        if (( restore_exit != 0 )); then
            echo "WARNING: failed to restore GPU load task; see $run_dir/gpu-load-restore.log" >&2
        fi
    fi
    return "$original_exit"
}

record_source_version() {
    local workdir=$1
    if git -C "$workdir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        {
            printf 'ae_commit='
            git -C "$workdir" rev-parse HEAD
            git -C "$workdir" status --short
        } > "$run_dir/source-version.txt"
        return
    fi
    find "$workdir" -type f \
        ! -path '*/.git/*' \
        ! -name '.git' \
        ! -path '*/build/*' \
        ! -path '*/.pytest_cache/*' \
        ! -path '*/__pycache__/*' \
        ! -name '*.so' \
        -print0 \
        | sort -z \
        | xargs -0 -r sha256sum > "$run_dir/source.sha256"
}

action=${1:-}
expected_hostname=${2:-}
node_root=${3:-}
canonical_python=${4:-}

case "$action" in
    self-test)
        [[ $# == 1 ]] || { echo "invalid guard self-test arguments" >&2; exit 2; }
        self_test_dir=$(mktemp -d)
        run_dir="$self_test_dir"
        run_status="running"
        unset command_exit_code || true
        write_status
        grep -qx 'status=running' "$run_dir/status.env" \
            || { echo "guard self-test failed: running status" >&2; exit 1; }
        command_exit_code=7
        run_status="command_failed"
        write_status
        grep -qx 'command_exit_code=7' "$run_dir/status.env" \
            || { echo "guard self-test failed: exit code" >&2; exit 1; }
        command -v flock >/dev/null \
            || { echo "guard self-test failed: flock unavailable" >&2; exit 1; }
        exec {test_lock_a}<>"$self_test_dir/lock"
        flock -n "$test_lock_a" \
            || { echo "guard self-test failed: first lock" >&2; exit 1; }
        exec {test_lock_b}<>"$self_test_dir/lock"
        if flock -n "$test_lock_b"; then
            echo "guard self-test failed: concurrent lock was accepted" >&2
            exit 1
        fi
        run_lock_fd=$test_lock_a
        release_run_lock
        flock -n "$test_lock_b" \
            || { echo "guard self-test failed: released lock was not reusable" >&2; exit 1; }
        flock -u "$test_lock_b"
        exec {test_lock_b}>&-
        record_source_version "$script_dir/../.."
        grep -q '^ae_commit=' "$run_dir/source-version.txt" \
            || { echo "guard self-test failed: source revision" >&2; exit 1; }
        find "$self_test_dir" -depth -delete
        echo "server guard self-test: PASS"
        ;;
    check)
        [[ $# == 4 ]] || { echo "invalid remote check arguments" >&2; exit 2; }
        static_check "$expected_hostname" "$node_root" "$canonical_python" 20
        echo "gpu_load_task_begin"
        if [[ $load_control == none ]]; then
            echo "disabled for this server profile"
        else
            "$load_control" status || true
        fi
        echo "gpu_load_task_end"
        echo "NOTE: check is read-only and does not authorize a GPU workload."
        ;;
    run)
        [[ $# -ge 13 ]] || { echo "invalid remote run arguments" >&2; exit 2; }
        run_id=$5
        gpu_count=$6
        workdir_relative=$7
        memory_limit=$8
        idle_samples=$9
        idle_interval=${10}
        min_free_gib=${11}
        shift 11
        [[ ${1:-} == -- ]] || { echo "missing command separator" >&2; exit 2; }
        shift
        (( $# > 0 )) || { echo "missing experiment command" >&2; exit 2; }
        command_args=("$@")

        base_check "$expected_hostname" "$node_root" "$canonical_python"
        node_root_real=$(realpath -e "$node_root") || blocked "cannot resolve NODE_ROOT"
        workdir=$(realpath -e "$node_root/$workdir_relative") || blocked "workdir does not exist in GPFS: $workdir_relative"
        case "$workdir/" in
            "$node_root_real/"*) ;;
            *) blocked "workdir escapes NODE_ROOT: $workdir" ;;
        esac

        run_root="$node_root_real/eurosys27-ae/runs"
        mkdir -p "$run_root" || blocked "cannot create GPFS run root: $run_root"
        run_dir="$run_root/$run_id"
        [[ ! -e $run_dir ]] || blocked "run directory already exists: $run_dir"
        mkdir "$run_dir" || blocked "cannot create run directory: $run_dir"
        mkdir "$run_dir/raw" "$run_dir/results" "$run_dir/traces" "$run_dir/profiler" "$run_dir/cache"

        acquire_run_lock "$node_root_real/eurosys27-ae/.locks" "$run_id"

        check_gpfs_write "$canonical_python" \
            || blocked "GPFS write/fsync probe failed; check quota and fileset state"

        set +e
        static_check "$expected_hostname" "$node_root_real" "$canonical_python" "$min_free_gib" \
            2>&1 | tee "$run_dir/environment.txt"
        preflight_exit=${PIPESTATUS[0]}
        set -e
        (( preflight_exit == 0 )) || exit "$environment_blocked_exit"

        if [[ ${command_args[0]} == @python ]]; then
            command_args[0]="$canonical_python"
        fi
        {
            printf 'cd %q\n' "$workdir"
            printf 'command '
            printf '%q ' "${command_args[@]}"
            printf '\n'
        } > "$run_dir/command.txt"
        record_source_version "$workdir" \
            || blocked "failed to record the source version on GPFS"

        if [[ $load_control != none ]]; then
            load_stopped=1
            trap restore_load EXIT
        fi
        trap 'exit 130' INT TERM HUP
        if [[ $load_control != none ]]; then
            "$load_control" stop > "$run_dir/gpu-load-stop.log" 2>&1 \
                || blocked "failed to stop GPU load task"
        fi
        wait_for_idle_gpus "$memory_limit" "$idle_samples" "$idle_interval"

        gpu_ids=$(seq -s, 0 $((gpu_count - 1)))
        nsys_bin=$(find_nsys) || blocked "Nsight Systems disappeared after preflight"
        mkdir "$run_dir/cache/tmp" "$run_dir/cache/xdg" "$run_dir/cache/torch" "$run_dir/cache/huggingface" "$run_dir/cache/cuda"

        run_status="running"
        write_status
        set +e
        (
            exec {run_lock_fd}>&-
            run_lock_fd=""
            cd "$workdir"
            export CANONICAL_PYTHON="$canonical_python"
            export PATH="$(dirname "$canonical_python"):$PATH"
            export PYTHONNOUSERSITE=1
            export CUDA_VISIBLE_DEVICES="$gpu_ids"
            export RUN_DIR="$run_dir"
            export RAW_DIR="$run_dir/raw"
            export RESULTS_DIR="$run_dir/results"
            export TRACE_DIR="$run_dir/traces"
            export PROFILER_DIR="$run_dir/profiler"
            export NSYS_BIN="$nsys_bin"
            export TMPDIR="$run_dir/cache/tmp"
            export XDG_CACHE_HOME="$run_dir/cache/xdg"
            export TORCH_HOME="$run_dir/cache/torch"
            export HF_HOME="$run_dir/cache/huggingface"
            export CUDA_CACHE_PATH="$run_dir/cache/cuda"
            export PROOT_TMP_DIR="$run_dir/cache/tmp"
            "${command_args[@]}"
        ) > >(tee "$run_dir/stdout.log") 2> >(tee "$run_dir/stderr.log" >&2)
        command_exit_code=$?
        set -e

        if (( command_exit_code == 0 )); then
            run_status="completed"
            write_interpretation "Pending author analysis of the preserved raw results."
        else
            run_status="command_failed"
            write_interpretation "None yet. Diagnose the failure before drawing a scientific conclusion."
        fi
        write_status
        exit "$command_exit_code"
        ;;
    *)
        echo "unknown remote guard action: $action" >&2
        exit 2
        ;;
esac
