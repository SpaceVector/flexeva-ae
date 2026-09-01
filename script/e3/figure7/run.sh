#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACTION="${1:-run}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRIVER="$ROOT/script/e3/figure7/measure_figure7_production.py"

[[ $# -le 1 ]] || {
    echo "usage: script/run_e3 figure7 [self-test|probe|run|report|verify]" >&2
    exit 2
}
case "$ACTION" in
    self-test|probe|run|report|verify) ;;
    *)
        echo "usage: script/run_e3 figure7 [self-test|probe|run|report|verify]" >&2
        exit 2
        ;;
esac

if [[ "$PYTHON_BIN" != */* ]]; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
[[ -n "$PYTHON_BIN" ]] || { echo "figure7: Python is unavailable" >&2; exit 2; }
PYTHON_BIN="$(realpath -e "$PYTHON_BIN")"

export PYTHONPATH="$ROOT/FlexEva/flexmaya_ras/src:$ROOT/script/e3/capture:$ROOT/script/e3/figure7:$ROOT/script/e3/workload/megatron${PYTHONPATH:+:$PYTHONPATH}"
export FAKECUDA_TARGET_ENV_ROOT="${FAKECUDA_TARGET_ENV_ROOT:-$(cd -- "$(dirname -- "$PYTHON_BIN")/.." && pwd)}"

if [[ "$ACTION" == self-test ]]; then
    "$PYTHON_BIN" "$ROOT/script/e3/figure7/production.py" self-test \
        --out-dir "$ROOT/result/e3/generated/figure7-self-test"
    exec "$PYTHON_BIN" "$DRIVER" self-test \
        --out-dir "$ROOT/result/e3/generated/figure7-self-test"
fi

if [[ "$ACTION" == verify || "$ACTION" == report ]]; then
    : "${FIGURE7_RESULT:?set FIGURE7_RESULT to a complete Figure 7 result.json}"
    if [[ "$ACTION" == verify ]]; then
        exec "$PYTHON_BIN" "$DRIVER" verify \
            --out-dir "$(dirname -- "$FIGURE7_RESULT")" \
            --result "$FIGURE7_RESULT"
    fi
    : "${FIGURE7_REPORT_ROOT:?set FIGURE7_REPORT_ROOT to a new report directory}"
    "$PYTHON_BIN" "$DRIVER" report \
        --out-dir "$FIGURE7_REPORT_ROOT" \
        --result "$FIGURE7_RESULT"
    install -m 0644 "$FIGURE7_REPORT_ROOT/figure7.pdf" "$ROOT/plot/figure7.pdf"
    exit
fi

: "${FIGURE7_RUN_ID:?set FIGURE7_RUN_ID to a unique guarded-run identifier}"
FIGURE7_MASTER_ADDR="${FIGURE7_MASTER_ADDR:-${FLEXMAYA_MASTER_ADDR:-}}"
FIGURE7_PEER_TARGET="${FIGURE7_PEER_TARGET:-${FLEXMAYA_PEER_TARGET:-}}"
FIGURE7_PEER_PORT="${FIGURE7_PEER_PORT:-${FLEXMAYA_PEER_PORT:-22}}"
FIGURE7_PEER_REPO_ROOT="${FIGURE7_PEER_REPO_ROOT:-${FLEXMAYA_PEER_REPO_ROOT:-}}"
FIGURE7_PEER_NODE_ROOT="${FIGURE7_PEER_NODE_ROOT:-${FLEXMAYA_PEER_NODE_ROOT:-}}"
FIGURE7_PEER_PYTHON="${FIGURE7_PEER_PYTHON:-${FLEXMAYA_PEER_PYTHON:-}}"
: "${FIGURE7_MASTER_ADDR:?set FLEXMAYA_MASTER_ADDR to the coordinator address reachable from the peer}"
: "${FIGURE7_PEER_TARGET:?set FLEXMAYA_PEER_TARGET to the peer SSH target}"
: "${FIGURE7_PEER_REPO_ROOT:?set FLEXMAYA_PEER_REPO_ROOT to this checkout on the peer}"
: "${FIGURE7_PEER_NODE_ROOT:?set FLEXMAYA_PEER_NODE_ROOT to the peer large-filesystem root}"
: "${FIGURE7_PEER_PYTHON:?set FLEXMAYA_PEER_PYTHON to the peer canonical Python}"

MAYA_ROOT="${MAYA_ROOT:-$ROOT/FlexEva/backends/maya}"
PROOT_BIN="${PROOT_BIN:-$ROOT/.deps/proot-5.3.1/bin/proot}"
RESULT_ROOT="${FIGURE7_RESULT_ROOT:-$ROOT/result/e3/generated/figure7/$FIGURE7_RUN_ID}"
PEER_MAYA_ROOT="${FIGURE7_PEER_MAYA_ROOT:-${FLEXMAYA_PEER_MAYA_ROOT:-$FIGURE7_PEER_REPO_ROOT/FlexEva/backends/maya}}"
PEER_PROOT="${FIGURE7_PEER_PROOT:-${FLEXMAYA_PEER_PROOT:-$FIGURE7_PEER_REPO_ROOT/.deps/proot-5.3.1/bin/proot}}"
PEER_WORK_ROOT="${FIGURE7_PEER_WORK_ROOT:-$FIGURE7_PEER_REPO_ROOT/result/e3/generated/figure7/$FIGURE7_RUN_ID}"
MASTER_PORT_BASE="${FIGURE7_MASTER_PORT_BASE:-45100}"
SOCKET_IFNAME="${FIGURE7_SOCKET_IFNAME:-eth1}"
PROBE_ROUND="${FIGURE7_PROBE_ROUND:-1}"

test -x "$MAYA_ROOT/fake-cuda/frun" || { echo "figure7: run script/setup first" >&2; exit 2; }
test -x "$PROOT_BIN" || { echo "figure7: missing PRoot: $PROOT_BIN" >&2; exit 2; }
test -x "$ROOT/script/e3/server.sh" || { echo "figure7: guarded server runner is unavailable" >&2; exit 2; }

command=(
    /usr/bin/env
    "PYTHONPATH=$PYTHONPATH"
    "FAKECUDA_TARGET_ENV_ROOT=$FAKECUDA_TARGET_ENV_ROOT"
    "$PYTHON_BIN" "$DRIVER" "$ACTION"
    --out-dir "$RESULT_ROOT"
    --maya-root "$MAYA_ROOT"
    --proot "$PROOT_BIN"
    --python "$PYTHON_BIN"
    --repo-root "$ROOT"
    --peer-target "$FIGURE7_PEER_TARGET"
    --peer-port "$FIGURE7_PEER_PORT"
    --peer-repo-root "$FIGURE7_PEER_REPO_ROOT"
    --peer-python "$FIGURE7_PEER_PYTHON"
    --peer-maya-root "$PEER_MAYA_ROOT"
    --peer-proot "$PEER_PROOT"
    --peer-node-root "$FIGURE7_PEER_NODE_ROOT"
    --peer-work-root "$PEER_WORK_ROOT"
    --run-id "$FIGURE7_RUN_ID"
    --master-addr "$FIGURE7_MASTER_ADDR"
    --master-port-base "$MASTER_PORT_BASE"
    --socket-ifname "$SOCKET_IFNAME"
)
[[ "$ACTION" != probe ]] || command+=(--probe-round "$PROBE_ROUND")

AE_CANONICAL_PYTHON="$PYTHON_BIN" \
MIN_GPFS_FREE_GIB="${FIGURE7_MIN_FREE_GIB:-500}" \
    "$ROOT/script/e3/server.sh" run "$FIGURE7_RUN_ID-coordinator" 8 -- "${command[@]}"

if [[ "$ACTION" == run ]]; then
    install -m 0644 "$RESULT_ROOT/figure7.pdf" "$ROOT/plot/figure7.pdf"
fi
