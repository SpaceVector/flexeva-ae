#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
MODE="${FIGURE5_MODE:-trace}"
MAYA_ROOT="${MAYA_ROOT:-$ROOT/FlexEva/backends/maya}"
PROOT_BIN="${PROOT_BIN:-$ROOT/.deps/proot-5.3.1/bin/proot}"
LARGE_ROOT="${FIGURE5_LARGE_CLUSTER_ROOT:-$ROOT/large-cluster/e2}"
SOURCE_LEDGER="${FIGURE5_SOURCE_LEDGER:-$ROOT/large-cluster/e2/figure5-source.json}"
ESTIMATOR_MODEL="${FIGURE5_ESTIMATOR_MODEL:-$LARGE_ROOT/estimator.json}"
PEER_WAIT_S="${FLEXMAYA_PEER_WAIT_S:-14400}"
REUSE_NATIVE="${FIGURE5_REUSE_NATIVE:-0}"
REUSE_EVAL="${FIGURE5_REUSE_EVAL:-0}"
RUN_ID="${FIGURE5_RUN_ID:-${FLEXMAYA_RUN_ID:-}}"

[[ "$MODE" == trace || "$MODE" == native ]] || {
    echo "figure5: mode must be trace or native" >&2
    exit 2
}

if [[ "$PYTHON_BIN" != */* ]]; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
[[ -n "$PYTHON_BIN" ]] || { echo "figure5: Python is unavailable" >&2; exit 2; }
PYTHON_BIN="$(cd -- "$(dirname -- "$PYTHON_BIN")" && pwd -P)/$(basename -- "$PYTHON_BIN")"
: "${RUN_ID:?set FIGURE5_RUN_ID to a unique identifier}"
RESULT_BASE="${FIGURE5_RESULT_ROOT:-$ROOT/result/e2/generated/figure5/$RUN_ID}"
RESULT_ROOT="$RESULT_BASE/$MODE"
[[ "$LARGE_ROOT" == /* ]] || LARGE_ROOT="$PWD/$LARGE_ROOT"
[[ "$SOURCE_LEDGER" == /* ]] || SOURCE_LEDGER="$PWD/$SOURCE_LEDGER"
[[ "$RESULT_BASE" == /* ]] || RESULT_BASE="$PWD/$RESULT_BASE"
[[ "$RESULT_ROOT" == /* ]] || RESULT_ROOT="$PWD/$RESULT_ROOT"

if [[ "$MODE" == trace ]]; then
    [[ ! -e "$RESULT_ROOT" ]] || {
        echo "figure5: trace result exists; use a new FIGURE5_RUN_ID: $RESULT_ROOT" >&2
        exit 2
    }
    mkdir -p "$RESULT_ROOT"
    "$PYTHON_BIN" "$ROOT/script/e2/collect_figure5.py" \
        --mode trace --source-ledger "$SOURCE_LEDGER" \
        --large-trace-root "$LARGE_ROOT" --output-dir "$RESULT_ROOT"
    "$PYTHON_BIN" "$ROOT/script/e2/validate_results.py" \
        --figure5-mode trace --figure5-result-dir "$RESULT_ROOT"
    "$PYTHON_BIN" "$ROOT/script/e2/plot_figure5.py" \
        --mode trace --result-dir "$RESULT_ROOT" --output-dir "$ROOT/plot"
    echo "figure5: trace-based limited reproduction complete: $RESULT_ROOT"
    exit 0
fi

if [[ -z "${FLEXMAYA_NODE_RANK:-}" ]]; then
    exec "$PYTHON_BIN" "$ROOT/script/lib/two_node.py" launch \
        --run-id "$RUN_ID" \
        --entry script/run_e2 \
        --local-python "$PYTHON_BIN" \
        --min-free-gib "${FIGURE5_MIN_FREE_GIB:-500}" \
        -- figure5 native
fi

: "${RUN_ID:?figure5: coordinated workers require FLEXMAYA_RUN_ID}"
TRACE_BASE="${FIGURE5_TRACE_ROOT:-$ROOT/trace/e2/figure5/$RUN_ID}"
TRACE_ROOT="$TRACE_BASE/native"
[[ "$PROOT_BIN" == /* ]] || PROOT_BIN="$PWD/$PROOT_BIN"
[[ "$ESTIMATOR_MODEL" == /* ]] || ESTIMATOR_MODEL="$PWD/$ESTIMATOR_MODEL"
[[ "$TRACE_BASE" == /* ]] || TRACE_BASE="$PWD/$TRACE_BASE"
[[ "$TRACE_ROOT" == /* ]] || TRACE_ROOT="$PWD/$TRACE_ROOT"
MAYA_ROOT="$(realpath -e "$MAYA_ROOT")"
cd "$MAYA_ROOT"

for name in FLEXMAYA_NODE_RANK FLEXMAYA_MASTER_ADDR FLEXMAYA_MASTER_PORT FLEXMAYA_CONTROL_PORT; do
    [[ -n "${!name:-}" ]] || { echo "figure5: $name is required" >&2; exit 2; }
done
[[ "${FLEXMAYA_NNODES:-}" == "2" ]] || { echo "figure5: FLEXMAYA_NNODES must be 2" >&2; exit 2; }
[[ "$FLEXMAYA_NODE_RANK" == "0" || "$FLEXMAYA_NODE_RANK" == "1" ]] || {
    echo "figure5: FLEXMAYA_NODE_RANK must be 0 or 1" >&2
    exit 2
}
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+){7}$ ]] || {
    echo "figure5: CUDA_VISIBLE_DEVICES must list eight physical GPUs on each node" >&2
    exit 2
}
[[ "$FLEXMAYA_MASTER_PORT" =~ ^[0-9]+$ && "$FLEXMAYA_CONTROL_PORT" =~ ^[0-9]+$ \
    && "$FLEXMAYA_MASTER_PORT" -ge 1 && "$FLEXMAYA_CONTROL_PORT" -ge 1 \
    && "$FLEXMAYA_MASTER_PORT" -le 65530 && "$FLEXMAYA_CONTROL_PORT" -le 65529 ]] || {
    echo "figure5: master/control ports must leave room for six case ports" >&2
    exit 2
}
[[ -f "$ESTIMATOR_MODEL" ]] || {
    echo "figure5: missing independent estimator: $ESTIMATOR_MODEL" >&2
    echo "set FIGURE5_ESTIMATOR_MODEL to the supplied estimator JSON" >&2
    exit 2
}

MAYA_PYTHON="$MAYA_ROOT/python"
CPPEVENT_DIR="$MAYA_ROOT/CppEvent"
LOWLEVEL_DIR="$MAYA_ROOT/build/lowlevel-interface"
GPT_WORKLOAD="$ROOT/script/e2/workload/megatron/maya_megatron.py"
MOE_WORKLOAD="$ROOT/script/e2/workload/routed_moe/moe_topk.py"
MAYA_PATH="$MAYA_ROOT:$MAYA_PYTHON:$CPPEVENT_DIR:$ROOT/script/e2/workload/megatron:$ROOT/script/e2/workload/routed_moe${PYTHONPATH:+:$PYTHONPATH}"
TORCH_LIB="$($PYTHON_BIN -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda/lib64}"
REAL_LIBRARY_PATH="$CPPEVENT_DIR:$LOWLEVEL_DIR:$TORCH_LIB:$CUDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
REAL_PRELOAD="$LOWLEVEL_DIR/libcuda_real_wrappers.so:$LOWLEVEL_DIR/libnccl_real_wrappers.so:$LOWLEVEL_DIR/libcublas_real_wrappers.so:$LOWLEVEL_DIR/liblowlevel_interface_runtime.so:$CPPEVENT_DIR/libcpp_event_runtime_shared.so:$CPPEVENT_DIR/libcpp_event_tls_shared.so${LD_PRELOAD:+:$LD_PRELOAD}"

for path in \
    "$GPT_WORKLOAD" \
    "$MOE_WORKLOAD" \
    "$LOWLEVEL_DIR/libcuda_real_wrappers.so" \
    "$LOWLEVEL_DIR/libnccl_real_wrappers.so" \
    "$LOWLEVEL_DIR/libcublas_real_wrappers.so" \
    "$LOWLEVEL_DIR/liblowlevel_interface_runtime.so" \
    "$CPPEVENT_DIR/libcpp_event_runtime_shared.so" \
    "$CPPEVENT_DIR/libcpp_event_tls_shared.so"; do
    [[ -f "$path" ]] || { echo "figure5: missing native-capture dependency: $path; run script/setup" >&2; exit 2; }
done

env PYTHONPATH="$MAYA_PATH" LD_LIBRARY_PATH="$REAL_LIBRARY_PATH" \
    "$PYTHON_BIN" -c 'import cpp_event_py, cpp_event_tls' || {
        echo "figure5: native CppEvent bindings are unavailable; run script/setup" >&2
        exit 2
    }
PYTHONPATH="$MAYA_PATH" "$PYTHON_BIN" -c \
    'from flexsim.estimator import Estimator; import sys; estimator=Estimator.load(sys.argv[1]); assert estimator.is_calibrated(); assert "gpu_estimator_xgboost" in estimator.provider_names()' \
    "$ESTIMATOR_MODEL" || {
        echo "figure5: estimator or bundled XGBoost provider is unavailable" >&2
        exit 2
    }

visible_gpus="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
[[ "$visible_gpus" == "8" ]] || {
    echo "figure5: expected eight visible physical GPUs on node $FLEXMAYA_NODE_RANK, found $visible_gpus" >&2
    exit 2
}

mkdir -p "$TRACE_ROOT" "$RESULT_ROOT"

run_native() {
    local name="$1" world_size="$2" output_dir="$3" port="$4" workload="$5"
    shift 5
    local -a launch
    if [[ "$world_size" == "8" ]]; then
        [[ "$FLEXMAYA_NODE_RANK" == "0" ]] || return 0
        launch=(--standalone --nnodes=1 --nproc-per-node=8)
    else
        launch=(
            --nnodes=2
            "--node-rank=$FLEXMAYA_NODE_RANK"
            --nproc-per-node=8
            "--master-addr=$FLEXMAYA_MASTER_ADDR"
            "--master-port=$port"
        )
    fi
    [[ ! -e "$output_dir/capture_manifest.json" ]] || {
        echo "figure5: native output exists: $output_dir (set FIGURE5_REUSE_NATIVE=1 to reuse the complete run)" >&2
        exit 2
    }
    mkdir -p "$output_dir" "$RESULT_ROOT/native-logs"
    env -u FAKECUDA_TARGET_ENV_ROOT -u FAKECUDA_PROOT_BIN \
        -u FAKECUDA_TRACE -u FAKECUDA_TRACE_PATH -u FLEXMAYA_TRACE_DIR \
        PYTHONNOUSERSITE=1 \
        PYTHONPATH="$MAYA_PATH" \
        LD_LIBRARY_PATH="$REAL_LIBRARY_PATH" \
        LD_PRELOAD="$REAL_PRELOAD" \
        FLEXSIM_HOST_MACHINE_ID="node$FLEXMAYA_NODE_RANK" \
        LLI_REAL_EXTERNAL_CUBLAS_PATH=libcublas.so.12 \
        LLI_REAL_EXTERNAL_NCCL_PATH=libnccl.so.2 \
        LLI_REAL_EXTERNAL_CUDA_PATH=libcuda.so.1 \
        TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
        "$PYTHON_BIN" -m torch.distributed.run "${launch[@]}" \
        -m flexsim.maya_lite.capture_real \
        --output-dir "$output_dir" \
        --auto-profiled-strategy identity \
        --route-metadata figure13_route=figure5_native_gpu \
        --route-metadata auto_profiled_strategy=identity \
        --route-metadata dynamic_first_iteration_dedup=true \
        --route-metadata collective_mode=trace_only \
        --route-metadata host_timing_mode=measure \
        --route-metadata host_timing_dispatch_scope=host_machine \
        --route-metadata host_timing_schedule_surface=semantic \
        --route-metadata validation_mode=figure5_fresh_native_gpu \
        --route-metadata "run_name=$name" \
        "$workload" -- "$@" \
        >"$RESULT_ROOT/native-logs/$name.node$FLEXMAYA_NODE_RANK.stdout.txt" \
        2>"$RESULT_ROOT/native-logs/$name.node$FLEXMAYA_NODE_RANK.stderr.txt"
}

merge_two_nodes() {
    local name="$1" base="$2" transfer_port="$3"
    if [[ "$FLEXMAYA_NODE_RANK" == "0" ]]; then
        "$PYTHON_BIN" "$ROOT/script/lib/two_node.py" transfer \
            --node-rank 0 --address "$FLEXMAYA_MASTER_ADDR" --port "$transfer_port" \
            --timeout "$PEER_WAIT_S" --directory "$base/node1"
        [[ ! -e "$base/real" ]] || { echo "figure5: merged native trace exists: $base/real" >&2; exit 2; }
        PYTHONPATH="$MAYA_PATH" LD_LIBRARY_PATH="$REAL_LIBRARY_PATH" \
            "$PYTHON_BIN" -c \
            'from pathlib import Path; from flexsim.maya_lite.capture_real import merge_real_trace_nodes; import sys; merge_real_trace_nodes([Path(sys.argv[1]), Path(sys.argv[2])], Path(sys.argv[3]))' \
            "$base/node0" "$base/node1" "$base/real"
    else
        "$PYTHON_BIN" "$ROOT/script/lib/two_node.py" transfer \
            --node-rank 1 --address "$FLEXMAYA_MASTER_ADDR" --port "$transfer_port" \
            --timeout "$PEER_WAIT_S" --directory "$base/node1"
    fi
    echo "figure5: native $name complete on node $FLEXMAYA_NODE_RANK"
}

gpt_args() {
    local scale="$1" tp dp
    case "$scale" in
        8) tp=1; dp=1 ;;
        16) tp=2; dp=1 ;;
        32) tp=4; dp=1 ;;
        64) tp=8; dp=1 ;;
        128) tp=8; dp=2 ;;
        *) echo "figure5: unsupported GPT scale: $scale" >&2; return 2 ;;
    esac
    GPT_TP="$tp"
    GPT_ARGS=(
        --steps 1 --global-batch-size 768 --seq-len 256 --hidden-size 512
        --num-layers 64 --num-heads 8 --vocab-size 32000
        --tp "$tp" --pp 8 --dp "$dp" --micro-batches 64
        --schedule 1f1b --pipeline-p2p-mode blocking --dtype bf16 --log-interval 1
    )
}

MOE_COMMON=(
    --steps 1 --warmup-steps 1 --global-batch-size 16 --seq-len 64
    --hidden-size 128 --num-layers 2 --num-heads 4 --vocab-size 4096
    --num-experts 16 --top-k 2 --capacity-factor 1.25 --ep-size 16
    --dp 1 --micro-batches 1 --dtype bf16 --log-interval 1
)
MOE_CASES=(
    'routed_moe_base|1234|-1|'
    'routed_moe_intra_group_0_1|5200|0|0,1'
    'routed_moe_cross_group_0_8|5201|1|0,8'
    'routed_moe_cross_group_0_15|5202|2|0,15'
    'routed_moe_boundary_7_8|5203|3|7,8'
)

if [[ "$REUSE_NATIVE" != "1" ]]; then
    gpt_args 8
    if [[ "$FLEXMAYA_NODE_RANK" == "0" ]]; then
        run_native gpt-8 8 "$TRACE_ROOT/gpt/8/real" "$FLEXMAYA_MASTER_PORT" "$GPT_WORKLOAD" "${GPT_ARGS[@]}"
    fi
    "$PYTHON_BIN" "$ROOT/script/lib/two_node.py" barrier \
        --node-rank "$FLEXMAYA_NODE_RANK" --address "$FLEXMAYA_MASTER_ADDR" \
        --port "$FLEXMAYA_CONTROL_PORT" --timeout "$PEER_WAIT_S"

    gpt_args 16
    run_native gpt-16 16 "$TRACE_ROOT/gpt/16/node$FLEXMAYA_NODE_RANK" "$FLEXMAYA_MASTER_PORT" "$GPT_WORKLOAD" "${GPT_ARGS[@]}"
    merge_two_nodes gpt-16 "$TRACE_ROOT/gpt/16" "$((FLEXMAYA_CONTROL_PORT + 1))"

    case_index=0
    for definition in "${MOE_CASES[@]}"; do
        IFS='|' read -r case_name seed route_id route_experts <<<"$definition"
        route_args=()
        if [[ -n "$route_experts" ]]; then
            route_args=(--route-path-id "$route_id" --route-experts "$route_experts" --route-p2p-probe)
        fi
        run_native "$case_name" 16 "$TRACE_ROOT/moe/$case_name/node$FLEXMAYA_NODE_RANK" \
            "$((FLEXMAYA_MASTER_PORT + case_index + 1))" "$MOE_WORKLOAD" \
            "${MOE_COMMON[@]}" --seed "$seed" "${route_args[@]}"
        merge_two_nodes "$case_name" "$TRACE_ROOT/moe/$case_name" "$((FLEXMAYA_CONTROL_PORT + case_index + 2))"
        case_index=$((case_index + 1))
    done
fi

if [[ "$FLEXMAYA_NODE_RANK" == "1" ]]; then
    exit 0
fi

test -x "$MAYA_ROOT/fake-cuda/frun" || { echo "figure5: missing FakeCUDA launcher; run script/setup" >&2; exit 2; }
test -x "$PROOT_BIN" || { echo "figure5: missing PRoot: $PROOT_BIN" >&2; exit 2; }
export PYTHONPATH="$MAYA_PATH"
export FAKECUDA_PROOT_BIN="$PROOT_BIN"
export FAKECUDA_TARGET_ENV_ROOT="${FAKECUDA_TARGET_ENV_ROOT:-$(cd -- "$(dirname -- "$PYTHON_BIN")/.." && pwd)}"
export FAKECUDA_FRUN_QUIET=1

evaluate_case() {
    local name="$1" world_size="$2" tp="$3" pp="$4" real_trace="$5" emulated_trace="$6" output_dir="$7" workload="$8"
    shift 8
    [[ -d "$real_trace" ]] || { echo "figure5: missing real/linked trace: $real_trace" >&2; exit 2; }
    if [[ "$REUSE_EVAL" == "1" && -f "$output_dir/simulate_summary.json" ]]; then
        return 0
    fi
    [[ ! -e "$emulated_trace" && ! -e "$output_dir/simulate_summary.json" ]] || {
        echo "figure5: evaluation output exists for $name (set FIGURE5_REUSE_EVAL=1 to reuse it)" >&2
        exit 2
    }
    mkdir -p "$emulated_trace" "$output_dir/cache"
    FLEXSIM_CLUSTER_RDMA_AFFINITY=0 FLEXSIM_CLUSTER_CPU_AFFINITY=0 \
    "$PYTHON_BIN" "$MAYA_PYTHON/flexsim/maya_lite/capture_emulated.py" \
        --output-dir "$emulated_trace" \
        --logical-world-size "$world_size" \
        --auto-profiled-strategy identity \
        --frun "$MAYA_ROOT/fake-cuda/frun" \
        --python-bin "$PYTHON_BIN" \
        --local-device-span 8 \
        --max-concurrent-workers "$world_size" \
        --tensor-parallel-size "$tp" \
        --pipeline-parallel-size "$pp" \
        --collective-mode trace_only \
        --trace-flush-mode buffered \
        --trace-flush-every 16384 \
        --trace-stdio-buffer-bytes 16777216 \
        --trace-surface all \
        --host-timing-mode measure \
        --host-timing-dispatch-scope host_machine \
        --host-timing-schedule-surface semantic \
        --dynamic-first-iteration-dedup \
        "$workload" -- "$@" \
        >"$output_dir/capture.stdout.txt" 2>"$output_dir/capture.stderr.txt"
    "$PYTHON_BIN" "$MAYA_PYTHON/flexsim/maya_lite/prepare.py" \
        --real "$real_trace" --emu "$emulated_trace" --cache "$output_dir/cache" \
        --estimator-model "$ESTIMATOR_MODEL" --trace-window step \
        >"$output_dir/prepare.stdout.txt" 2>"$output_dir/prepare.stderr.txt"
    emulator_seconds="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["capture_elapsed_seconds"])' "$emulated_trace/capture_manifest.json")"
    "$PYTHON_BIN" "$MAYA_PYTHON/flexsim/maya_lite/simulate.py" \
        --cache "$output_dir/cache" --emulator-wall-seconds "$emulator_seconds" \
        --parallel-workers "${FIGURE5_PARALLEL_WORKERS:-16}" \
        >"$output_dir/simulate.stdout.txt" 2>"$output_dir/simulate.stderr.txt"
    cp "$output_dir/cache/prepare_summary.json" "$output_dir/prepare_summary.json"
    cp "$output_dir/cache/simulate_summary.json" "$output_dir/simulate_summary.json"
}

for scale in 8 16; do
    gpt_args "$scale"
    real_trace="$TRACE_ROOT/gpt/$scale/real"
    evaluate_case "gpt-$scale" "$scale" "$GPT_TP" 8 "$real_trace" "$TRACE_ROOT/gpt/$scale/emulated" \
        "$RESULT_ROOT/gpt/$scale" "$GPT_WORKLOAD" "${GPT_ARGS[@]}"
done

for definition in "${MOE_CASES[@]}"; do
    IFS='|' read -r case_name seed route_id route_experts <<<"$definition"
    route_args=()
    if [[ -n "$route_experts" ]]; then
        route_args=(--route-path-id "$route_id" --route-experts "$route_experts" --route-p2p-probe)
    fi
    evaluate_case "$case_name" 16 1 1 "$TRACE_ROOT/moe/$case_name/real" \
        "$TRACE_ROOT/moe/$case_name/emulated" "$RESULT_ROOT/moe/$case_name" "$MOE_WORKLOAD" \
        "${MOE_COMMON[@]}" --seed "$seed" "${route_args[@]}"
done

"$PYTHON_BIN" "$ROOT/script/e2/collect_figure5.py" \
    --mode native --summary-root "$RESULT_ROOT" --output-dir "$RESULT_ROOT" \
    --native-trace-root "$TRACE_ROOT"
"$PYTHON_BIN" "$ROOT/script/e2/validate_results.py" \
    --figure5-mode native --figure5-result-dir "$RESULT_ROOT"
"$PYTHON_BIN" "$ROOT/script/e2/plot_figure5.py" \
    --mode native --result-dir "$RESULT_ROOT" --output-dir "$ROOT/plot"
echo "figure5: native 8/16-GPU data complete: $RESULT_ROOT"
