#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAYA_ROOT="${MAYA_ROOT:-$ROOT/FlexEva/backends/maya}"
PROOT_BIN="${PROOT_BIN:-$ROOT/.deps/proot-5.3.1/bin/proot}"
SOURCE_COMMIT="${FIGURE6_SOURCE_COMMIT:-0129f024354887ed272fea4a2aa4a661ec38662b}"
RUN_ID="${FIGURE6_RUN_ID:-${FLEXMAYA_RUN_ID:-}}"

if [[ "$PYTHON_BIN" != */* ]]; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
[[ -n "$PYTHON_BIN" ]] || { echo "figure6: Python is unavailable" >&2; exit 2; }
PYTHON_BIN="$(realpath -e "$PYTHON_BIN")"

if [[ -z "${FLEXMAYA_NODE_RANK:-}" ]]; then
    : "${RUN_ID:?set FIGURE6_RUN_ID to a unique guarded-run identifier}"
    exec "$PYTHON_BIN" "$ROOT/script/lib/two_node.py" launch \
        --run-id "$RUN_ID" \
        --entry script/run_e3 \
        --local-python "$PYTHON_BIN" \
        --min-free-gib "${FIGURE6_MIN_FREE_GIB:-500}" \
        -- figure6
fi

: "${RUN_ID:?figure6: coordinated workers require FLEXMAYA_RUN_ID}"
RESULT_ROOT="${FIGURE6_RESULT_ROOT:-$ROOT/result/e3/generated/figure6/$RUN_ID}"
TRACE_ROOT="${FIGURE6_TRACE_ROOT:-$ROOT/trace/e3/figure6/$RUN_ID}"

for name in FLEXMAYA_NODE_RANK FLEXMAYA_MASTER_ADDR FLEXMAYA_MASTER_PORT FLEXMAYA_CONTROL_PORT; do
    [[ -n "${!name:-}" ]] || { echo "figure6: $name is required" >&2; exit 2; }
done
[[ "${FLEXMAYA_NNODES:-}" == "2" ]] || { echo "figure6: FLEXMAYA_NNODES must be 2" >&2; exit 2; }
[[ "$FLEXMAYA_NODE_RANK" == "0" || "$FLEXMAYA_NODE_RANK" == "1" ]] || {
    echo "figure6: FLEXMAYA_NODE_RANK must be 0 or 1" >&2
    exit 2
}
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+){7}$ ]] || {
    echo "figure6: CUDA_VISIBLE_DEVICES must list eight GPUs on each node" >&2
    exit 2
}
test -x "$MAYA_ROOT/fake-cuda/frun" || { echo "figure6: run script/setup first" >&2; exit 2; }
test -x "$PROOT_BIN" || { echo "figure6: missing PRoot: $PROOT_BIN" >&2; exit 2; }
[[ ! -e "$RESULT_ROOT" && ! -e "$TRACE_ROOT" ]] || {
    echo "figure6: run output already exists; use a new FIGURE6_RUN_ID" >&2
    exit 2
}

export PYTHONPATH="$ROOT/FlexEva/flexmaya_ras/src:$ROOT/script/e3/capture:$ROOT/script/e3/figure6${PYTHONPATH:+:$PYTHONPATH}"
export FAKECUDA_TARGET_ENV_ROOT="${FAKECUDA_TARGET_ENV_ROOT:-$(cd -- "$(dirname -- "$PYTHON_BIN")/.." && pwd)}"
export FLEXMAYA_PEER_WAIT_S="${FLEXMAYA_PEER_WAIT_S:-14400}"
SOCKET_IFNAME="${FIGURE6_SOCKET_IFNAME:-eth1}"
export GLOO_SOCKET_IFNAME="$SOCKET_IFNAME"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="=$SOCKET_IFNAME"
mkdir -p "$RESULT_ROOT" "$TRACE_ROOT"

export FLEXMAYA_TRACE_ROOT="$TRACE_ROOT/gpt"
"$PYTHON_BIN" "$ROOT/script/e3/figure6/measure_megatron_18p4b_16gpu_e2e_speedup.py" \
    --out-dir "$RESULT_ROOT/gpt/capture" \
    --maya-root "$MAYA_ROOT" \
    --python "$PYTHON_BIN" \
    --proot "$PROOT_BIN" \
    --local-device-count 8 \
    --keep-raw-traces

if [[ "$FLEXMAYA_NODE_RANK" == "0" ]]; then
    "$PYTHON_BIN" "$ROOT/script/e3/figure6/measure_megatron_18p4b_16gpu_eval_breakdown.py" \
        --input-dir "$RESULT_ROOT/gpt/capture" \
        --out-dir "$RESULT_ROOT/gpt/breakdown"
fi

export FLEXMAYA_TRACE_ROOT="$TRACE_ROOT/moe"
"$PYTHON_BIN" "$ROOT/script/e3/figure6/measure_routed_moe_real_eval_breakdown.py" \
    --out-dir "$RESULT_ROOT/moe" \
    --maya-root "$MAYA_ROOT" \
    --python "$PYTHON_BIN" \
    --proot "$PROOT_BIN" \
    --local-device-count 8 \
    --seed-base 5200 \
    --keep-raw-traces

if [[ "$FLEXMAYA_NODE_RANK" == "0" ]]; then
    "$PYTHON_BIN" "$ROOT/script/e3/figure6/summarize_figure6_scale_only.py" \
        --gpt-result "$RESULT_ROOT/gpt/breakdown/e2e_result.json" \
        --gpt-breakdown "$RESULT_ROOT/gpt/breakdown/breakdown_cumulative_wide.csv" \
        --moe-result "$RESULT_ROOT/moe/result.json" \
        --moe-breakdown "$RESULT_ROOT/moe/breakdown_cumulative_wide.csv" \
        --out-dir "$RESULT_ROOT" \
        --commit "$SOURCE_COMMIT"

    "$PYTHON_BIN" "$ROOT/script/e3/figure6/plot_cumulative_eval_time.py" \
        --csv "$RESULT_ROOT/figure6.csv" --panel gpt \
        --output "$ROOT/plot/figure6a.pdf"
    "$PYTHON_BIN" "$ROOT/script/e3/figure6/plot_cumulative_eval_time.py" \
        --csv "$RESULT_ROOT/figure6.csv" --panel moe \
        --output "$ROOT/plot/figure6b.pdf"
    "$PYTHON_BIN" "$ROOT/script/e3/validate_results.py" \
        --require-generated --generated-dir "$RESULT_ROOT"
    echo "figure6: complete data: $RESULT_ROOT"
fi
