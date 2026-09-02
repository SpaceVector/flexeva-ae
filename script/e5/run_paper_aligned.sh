#!/usr/bin/env bash
set -euo pipefail

E5_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$E5_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${CANONICAL_PYTHON:-$ROOT/.venv/bin/python}}"
MODE="${1:-run}"

if [[ "$PYTHON_BIN" != */* ]]; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { echo "E5: Python is unavailable" >&2; exit 2; }
PYTHON_BIN="$(cd -- "$(dirname -- "$PYTHON_BIN")" && pwd -P)/$(basename -- "$PYTHON_BIN")"

if [[ -d "$ROOT/FlexEva/backends/maya" ]]; then
    MAYA_ROOT="${MAYA_ROOT:-$ROOT/FlexEva/backends/maya}"
    LIVE_CORE_SRC="$ROOT/FlexEva/flexmaya_ras/src"
elif [[ -d "$ROOT/maya/maya-native-source" ]]; then
    MAYA_ROOT="${MAYA_ROOT:-$ROOT/maya/maya-native-source}"
    LIVE_CORE_SRC="$ROOT/flexmaya_ras/src"
else
    echo "E5: Maya source is unavailable" >&2
    exit 2
fi
PROOT_BIN="${PROOT_BIN:-$ROOT/.deps/proot-5.3.1/bin/proot}"
RESULT_ROOT="${E5_RESULT_ROOT:-${RUN_DIR:+$RUN_DIR/results/e5-paper-aligned}}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result/e5/generated/paper-aligned}"

case "$MODE" in
    self-test)
        PYTHONPATH="$LIVE_CORE_SRC${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$E5_DIR/collect_real_routes.py" self-test
        PYTHONPATH="$LIVE_CORE_SRC${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$E5_DIR/measure_paper_aligned.py" self-test
        PYTHONPATH="$LIVE_CORE_SRC${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$E5_DIR/measure_case_speedup.py" self-test
        exit
        ;;
    verify)
        STAGED_SRC="$RESULT_ROOT/build/source/flexmaya_ras/src"
        [[ -d "$STAGED_SRC" ]] || { echo "E5: missing completed build below $RESULT_ROOT" >&2; exit 2; }
        PYTHONPATH="$STAGED_SRC${PYTHONPATH:+:$PYTHONPATH}" \
            "$PYTHON_BIN" "$E5_DIR/measure_paper_aligned.py" verify \
            --result "$RESULT_ROOT/measurement/result.json"
        PYTHONPATH="$STAGED_SRC${PYTHONPATH:+:$PYTHONPATH}" \
            "$PYTHON_BIN" "$E5_DIR/measure_case_speedup.py" verify \
            --result "$RESULT_ROOT/speed/result.json"
        exit
        ;;
    run) ;;
    *) echo "usage: script/e5/run_paper_aligned.sh [run|self-test|verify]" >&2; exit 2 ;;
esac

[[ ! -e "$RESULT_ROOT" ]] || { echo "E5: refusing to overwrite $RESULT_ROOT" >&2; exit 2; }
if [[ -n ${RUN_DIR:-} ]]; then
    case "$(realpath -m "$RESULT_ROOT")/" in
        "$(realpath -e "$RUN_DIR")/"*) ;;
        *) echo "E5: result root must remain below RUN_DIR" >&2; exit 2 ;;
    esac
fi
[[ -x "$MAYA_ROOT/fake-cuda/frun" ]] || { echo "E5: missing FakeCUDA launcher" >&2; exit 2; }
[[ -x "$PROOT_BIN" ]] || { echo "E5: missing PRoot" >&2; exit 2; }

mkdir -p "$RESULT_ROOT/build/source"
cp -a "$E5_DIR/bundle/flexmaya_ras" "$RESULT_ROOT/build/source/"
cp -a "$E5_DIR/bundle/paper_resilient_anchor_state" "$RESULT_ROOT/build/source/"
cp -a "$E5_DIR/bundle/external" "$RESULT_ROOT/build/source/"
STAGE_ROOT="$RESULT_ROOT/build/source"
(
    cd "$STAGE_ROOT/flexmaya_ras"
    PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" setup.py build_ext --inplace
) >"$RESULT_ROOT/build/stdout.txt" 2>"$RESULT_ROOT/build/stderr.txt"

CAPTURE_DEPS="$ROOT/script/e3/capture:$ROOT/script/e3/figure6:$ROOT/flexmaya_ras/scripts"
E5_PYTHONPATH="$STAGE_ROOT/flexmaya_ras/src:$STAGE_ROOT/paper_resilient_anchor_state/src:$CAPTURE_DEPS"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$E5_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
export FLEXMAYA_ROUTED_MOE_SCRIPT="$E5_DIR/workload/moe_topk.py"

"$PYTHON_BIN" "$E5_DIR/collect_real_routes.py" capture \
    --out-dir "$RESULT_ROOT/capture" \
    --maya-root "$MAYA_ROOT" \
    --python "$PYTHON_BIN" \
    --proot "$PROOT_BIN" \
    --local-device-count 8

"$PYTHON_BIN" "$E5_DIR/measure_paper_aligned.py" collect \
    --grounding-manifest "$RESULT_ROOT/capture/capture_manifest.json" \
    --out-dir "$RESULT_ROOT/collection"

"$PYTHON_BIN" "$E5_DIR/measure_paper_aligned.py" measure \
    --manifest "$RESULT_ROOT/collection/candidate_manifest.json" \
    --out-dir "$RESULT_ROOT/measurement" \
    --repeats "${E5_REPEATS:-1}"

"$PYTHON_BIN" "$E5_DIR/measure_paper_aligned.py" verify \
    --result "$RESULT_ROOT/measurement/result.json"

"$PYTHON_BIN" "$E5_DIR/measure_case_speedup.py" run \
    --manifest "$RESULT_ROOT/collection/candidate_manifest.json" \
    --out-dir "$RESULT_ROOT/speed" \
    --repeats "${E5_SPEED_REPEATS:-3}"

"$PYTHON_BIN" "$E5_DIR/measure_case_speedup.py" verify \
    --result "$RESULT_ROOT/speed/result.json"
