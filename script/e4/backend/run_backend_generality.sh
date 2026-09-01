#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
ASTRA_ROOT="${ASTRA_TABLE7_ROOT:-$ROOT/.deps/astra-sim-table7}"
RESULT_ROOT="${TABLE7_RESULT_ROOT:-$ROOT/result/e4/generated/table7}"
SETUP="$ROOT/script/e4/backend/setup_astra.sh"
BINARY="$ASTRA_ROOT/extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default"
DRIVER="$ASTRA_ROOT/scripts/measure_table7_backend_generality.py"
ANCHOR_ROOT="$ASTRA_ROOT/inputs/table7_anchor_20260901"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $0"
    echo "  Build/check ASTRA-Sim and run all six Table 7 backend cases."
    exit 0
fi

if [[ "$PYTHON_BIN" != */* ]]; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
[[ -n "$PYTHON_BIN" ]] || { echo "run_backend_generality: Python is unavailable" >&2; exit 2; }
PYTHON_BIN="$(cd -- "$(dirname -- "$PYTHON_BIN")" && pwd -P)/$(basename -- "$PYTHON_BIN")"

if [[ ! -x "$BINARY" ]]; then
    ASTRA_TABLE7_ROOT="$ASTRA_ROOT" PYTHON="$PYTHON_BIN" "$SETUP" build
else
    ASTRA_TABLE7_ROOT="$ASTRA_ROOT" PYTHON="$PYTHON_BIN" "$SETUP" check
fi

test -f "$DRIVER"
test -d "$ANCHOR_ROOT"
mkdir -p "$RESULT_ROOT"
[[ ! -e "$RESULT_ROOT/gpt" && ! -e "$RESULT_ROOT/moe" ]] || {
    echo "Table 7 output already exists: $RESULT_ROOT; choose TABLE7_RESULT_ROOT" >&2
    exit 2
}

"$PYTHON_BIN" "$DRIVER" run \
    --repo-root "$ASTRA_ROOT" \
    --anchor-root "$ANCHOR_ROOT" \
    --binary "$BINARY" \
    --out-dir "$RESULT_ROOT/gpt" \
    --repeats 1 \
    --case gpt_attention_backward \
    --case gpt_mlp_backward \
    --case gpt_optimizer_step

"$PYTHON_BIN" "$DRIVER" run \
    --repo-root "$ASTRA_ROOT" \
    --anchor-root "$ANCHOR_ROOT" \
    --binary "$BINARY" \
    --out-dir "$RESULT_ROOT/moe" \
    --repeats 3 \
    --case moe_optimizer_step \
    --case moe_attention_backward \
    --case moe_router_backward

"$PYTHON_BIN" "$DRIVER" verify --result "$RESULT_ROOT/gpt/result.json"
"$PYTHON_BIN" "$DRIVER" verify --result "$RESULT_ROOT/moe/result.json"
