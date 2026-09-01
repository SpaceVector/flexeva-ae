#!/usr/bin/env bash
set -euo pipefail

E5_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$E5_DIR/../.." && pwd)"
BUNDLE_ROOT="$E5_DIR/bundle"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
OUTPUT_ARG="${1:-$ROOT/result/e5/generated}"

if [[ "$PYTHON_BIN" != */* ]]; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
    echo "Table 8 reproduction requires Python" >&2
    exit 2
}
[[ "$(uname -s)" == "Linux" && -r /proc/self/status ]] || {
    echo "Table 8 RSS reproduction requires Linux /proc" >&2
    exit 2
}
"$PYTHON_BIN" -c 'import pybind11, pytest, setuptools' || {
    echo "missing E5 build dependencies; install requirements.txt first" >&2
    exit 2
}

mkdir -p "$OUTPUT_ARG"
OUTPUT_DIR="$(cd -- "$OUTPUT_ARG" && pwd)"
BUILD_DIR="$OUTPUT_DIR/build"
mkdir -p "$BUILD_DIR"
STAGE_ROOT="$(mktemp -d "$BUILD_DIR/source.XXXXXX")"
cp -a "$BUNDLE_ROOT/flexmaya_ras" "$STAGE_ROOT/"
cp -a "$BUNDLE_ROOT/paper_resilient_anchor_state" "$STAGE_ROOT/"
cp -a "$BUNDLE_ROOT/external" "$STAGE_ROOT/"

{
    date --iso-8601=seconds
    uname -a
    "$PYTHON_BIN" --version
    "${CXX:-c++}" --version | head -n 1
    "$PYTHON_BIN" -c 'import pybind11, setuptools; print(f"pybind11={pybind11.__version__} setuptools={setuptools.__version__}")'
    sha256sum \
        "$BUNDLE_ROOT/flexmaya_ras/scripts/measure_candidate_memory.py" \
        "$BUNDLE_ROOT/flexmaya_ras/scripts/run_moe_v2_matrix.py"
} >"$OUTPUT_DIR/environment.txt" 2>&1

printf '%s\n' \
    'measure_candidate_memory.py --world-size 16 --ep-group-size 8 --micro-batches 64 --layers 64 --seq-len 256 --hidden-size 512 --candidate-counts 1,2,4,8,16,32' \
    >"$OUTPUT_DIR/command.txt"

(
    cd "$STAGE_ROOT/flexmaya_ras"
    PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" setup.py build_ext --inplace
) >"$OUTPUT_DIR/build_stdout.txt" 2>"$OUTPUT_DIR/build_stderr.txt"

TABLE8_PYTHONPATH="$STAGE_ROOT/flexmaya_ras/src:$STAGE_ROOT/paper_resilient_anchor_state/src"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TABLE8_PYTHONPATH" \
    "$PYTHON_BIN" -m pytest -q -p no:cacheprovider \
    "$STAGE_ROOT/flexmaya_ras/tests/test_flexmaya_ras.py" \
    "$STAGE_ROOT/paper_resilient_anchor_state/tests/test_maya_v2_load_skew_case.py" \
    >"$OUTPUT_DIR/tests.txt" 2>&1

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TABLE8_PYTHONPATH" \
    "$PYTHON_BIN" "$STAGE_ROOT/flexmaya_ras/scripts/measure_candidate_memory.py" \
        --world-size 16 \
        --ep-group-size 8 \
        --micro-batches 64 \
        --layers 64 \
        --seq-len 256 \
        --hidden-size 512 \
        --candidate-counts 1,2,4,8,16,32 \
        --out-dir "$OUTPUT_DIR" \
        >"$OUTPUT_DIR/stdout.txt" 2>"$OUTPUT_DIR/stderr.txt"

PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" "$E5_DIR/verify_table8.py" \
    --input "$OUTPUT_DIR/memory_scaling.json" \
    --write-derived-json "$OUTPUT_DIR/table8_derived.json" \
    >"$OUTPUT_DIR/verification.txt"
cat "$OUTPUT_DIR/verification.txt"
