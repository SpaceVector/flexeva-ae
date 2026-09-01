#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACTION="${1:-run}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${FIGURE8_RUN_ID:-${FLEXMAYA_RUN_ID:-}}"

[[ $# -le 1 ]] || { echo "usage: script/run_e3 figure8 [self-test|run|verify]" >&2; exit 2; }
case "$ACTION" in
    self-test|run|verify) ;;
    *) echo "usage: script/run_e3 figure8 [self-test|run|verify]" >&2; exit 2 ;;
esac

if [[ "$PYTHON_BIN" != */* ]]; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
[[ -n "$PYTHON_BIN" ]] || { echo "figure8: Python is unavailable" >&2; exit 2; }
PYTHON_BIN="$(cd -- "$(dirname -- "$PYTHON_BIN")" && pwd -P)/$(basename -- "$PYTHON_BIN")"
[[ -x "$PYTHON_BIN" ]] || { echo "figure8: Python is not executable: $PYTHON_BIN" >&2; exit 2; }
PYTHONPATH_VALUE="$ROOT/FlexEva/flexmaya_ras/src:$ROOT/script/e3/capture:$ROOT/script/e3/figure6:$ROOT/script/e3/figure8:$ROOT/script/e3/workload/megatron${PYTHONPATH:+:$PYTHONPATH}"

plot() {
    "$PYTHON_BIN" "$ROOT/script/e3/plot_figure8.py" \
        --scale "$1/figure8a.csv" --trace "$1/figure8b.csv" --output "$1/figure8.pdf"
}

if [[ "$ACTION" == self-test ]]; then
    "$PYTHON_BIN" "$ROOT/script/e3/plot_figure8.py"
    exec "$PYTHON_BIN" "$ROOT/script/e3/validate_results.py"
fi

if [[ "$ACTION" == verify ]]; then
    : "${FIGURE8_RESULT_ROOT:?set FIGURE8_RESULT_ROOT to a complete Figure 8 result directory}"
    plot "$FIGURE8_RESULT_ROOT"
    "$PYTHON_BIN" "$ROOT/script/e3/validate_results.py" \
        --require-figure8-generated --figure8-generated-dir "$FIGURE8_RESULT_ROOT" \
        --figure8-max-timing-drift-rel "${FIGURE8_MAX_TIMING_DRIFT_REL:-0.10}"
    install -m 0644 "$FIGURE8_RESULT_ROOT/figure8.pdf" "$ROOT/plot/figure8.pdf"
    exit
fi

: "${RUN_ID:?set FIGURE8_RUN_ID to a unique guarded-run identifier}"
if [[ -z "${FLEXMAYA_NODE_RANK:-}" ]]; then
    exec "$PYTHON_BIN" "$ROOT/script/lib/two_node.py" launch \
        --run-id "$RUN_ID" \
        --entry script/run_e3 \
        --local-python "$PYTHON_BIN" \
        --min-free-gib "${FIGURE8_MIN_FREE_GIB:-50}" \
        -- figure8 run
fi

for name in FLEXMAYA_NNODES FLEXMAYA_NODE_RANK FLEXMAYA_MASTER_ADDR FLEXMAYA_MASTER_PORT FLEXMAYA_CONTROL_PORT; do
    [[ -n "${!name:-}" ]] || { echo "figure8: $name is required" >&2; exit 2; }
done
[[ "$FLEXMAYA_NNODES" == 2 ]] || { echo "figure8: FLEXMAYA_NNODES must be 2" >&2; exit 2; }
[[ "$FLEXMAYA_NODE_RANK" == 0 || "$FLEXMAYA_NODE_RANK" == 1 ]] || { echo "figure8: node rank must be 0 or 1" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+){7}$ ]] || { echo "figure8: list eight GPUs in CUDA_VISIBLE_DEVICES" >&2; exit 2; }

MAYA_ROOT="${MAYA_ROOT:-$ROOT/FlexEva/backends/maya}"
PROOT_BIN="${PROOT_BIN:-$ROOT/.deps/proot-5.3.1/bin/proot}"
RESULT_ROOT="${FIGURE8_RESULT_ROOT:-$ROOT/result/e3/generated/figure8/$RUN_ID}"
TRACE_ROOT="${FIGURE8_TRACE_ROOT:-$ROOT/trace/e3/figure8/$RUN_ID}"
test -x "$MAYA_ROOT/fake-cuda/frun" || { echo "figure8: run script/setup first" >&2; exit 2; }
test -x "$PROOT_BIN" || { echo "figure8: missing PRoot: $PROOT_BIN" >&2; exit 2; }
[[ ! -e "$RESULT_ROOT" && ! -e "$TRACE_ROOT" ]] || { echo "figure8: run output already exists; use a new FIGURE8_RUN_ID" >&2; exit 2; }

FAKECUDA_ENV="${FAKECUDA_TARGET_ENV_ROOT:-$(cd -- "$(dirname -- "$PYTHON_BIN")/.." && pwd)}"
SOCKET_IFNAME="${FIGURE8_SOCKET_IFNAME:-eth1}"
command=(
    /usr/bin/env
    "PYTHONPATH=$PYTHONPATH_VALUE"
    "FAKECUDA_TARGET_ENV_ROOT=$FAKECUDA_ENV"
    "FLEXMAYA_NNODES=$FLEXMAYA_NNODES"
    "FLEXMAYA_NODE_RANK=$FLEXMAYA_NODE_RANK"
    "FLEXMAYA_MASTER_ADDR=$FLEXMAYA_MASTER_ADDR"
    "FLEXMAYA_MASTER_PORT=$FLEXMAYA_MASTER_PORT"
    "FLEXMAYA_CONTROL_PORT=$FLEXMAYA_CONTROL_PORT"
    "FLEXMAYA_PEER_WAIT_S=${FLEXMAYA_PEER_WAIT_S:-14400}"
    "FLEXMAYA_TRACE_ROOT=$TRACE_ROOT"
    "GLOO_SOCKET_IFNAME=$SOCKET_IFNAME"
    "NCCL_IB_DISABLE=1"
    "NCCL_SOCKET_IFNAME==$SOCKET_IFNAME"
    "$PYTHON_BIN" "$ROOT/script/e3/figure8/measure_figure8.py"
    --out-dir "$RESULT_ROOT"
    --maya-root "$MAYA_ROOT"
    --python "$PYTHON_BIN"
    --proot "$PROOT_BIN"
    --local-device-count 8
    --keep-raw-traces
)

if [[ "${FLEXMAYA_COORDINATED:-0}" == 1 ]]; then
    "${command[@]}"
else
    AE_CANONICAL_PYTHON="$PYTHON_BIN" \
    MIN_GPFS_FREE_GIB="${FIGURE8_MIN_FREE_GIB:-50}" \
        "$ROOT/script/e3/server.sh" run "$RUN_ID-node$FLEXMAYA_NODE_RANK" 8 -- "${command[@]}"
fi

if [[ "$FLEXMAYA_NODE_RANK" == 0 ]]; then
    plot "$RESULT_ROOT"
    "$PYTHON_BIN" "$ROOT/script/e3/validate_results.py" \
        --require-figure8-generated --figure8-generated-dir "$RESULT_ROOT" \
        --figure8-max-timing-drift-rel "${FIGURE8_MAX_TIMING_DRIFT_REL:-0.10}"
    install -m 0644 "$RESULT_ROOT/figure8.pdf" "$ROOT/plot/figure8.pdf"
    echo "figure8: complete data: $RESULT_ROOT"
fi
