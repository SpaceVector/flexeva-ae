#!/usr/bin/env bash
# Build the native CUDA/NCCL/cuBLAS capture interposers from this checkout.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)

python_bin=${PYTHON_BIN:-python3}
cxx=${CXX:-c++}
python_root_dir=${PYTHON_ROOT_DIR:-$(cd -- "$(dirname -- "$python_bin")/.." && pwd)}
CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY="${CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY:-OFF}"
CPPEVENT_BUILD_PYTHON_BINDINGS="${CPPEVENT_BUILD_PYTHON_BINDINGS:-OFF}"
LLI_ENABLE_CUPTI_ACTIVITY_METADATA_WRAPPER_HOOKS="${LLI_ENABLE_CUPTI_ACTIVITY_METADATA_WRAPPER_HOOKS:-OFF}"
cpp_event_out=${CPPEVENT_OUT:-$repo_root/CppEvent}
lli_out=${LLI_OUT:-$repo_root/build/lowlevel-interface}
generated_dir="$lli_out/generated"
cpp_event_build_dir=${CPPEVENT_BUILD_DIR:-$repo_root/build/CppEvent}
runtime_root=${RUNTIME_ROOT:-$repo_root/build/real-wrapper-runtime}

mkdir -p "$runtime_root/tmp" "$runtime_root/cache" "$cpp_event_out" \
  "$lli_out" "$generated_dir"
export TMPDIR=${TMPDIR:-$runtime_root/tmp}
export TMP=${TMP:-$TMPDIR}
export TEMP=${TEMP:-$TMPDIR}

common_includes=(
  "-I$repo_root/cpp/lowlevel_interface/include"
  "-I$repo_root/cpp/lowlevel_interface/include/cuda_stubs"
  "-I$repo_root/cpp/cpp_event/include"
)
common_shared_flags=(-shared -fPIC -std=c++2a)
common_link_flags=(-ldl -lpthread)
lli_runtime_rpath='-Wl,-rpath,$ORIGIN:$ORIGIN/../../CppEvent'

is_enabled() {
  case "${1:-}" in
    1|ON|on|TRUE|true|YES|yes) return 0 ;;
    *) return 1 ;;
  esac
}

cmake_bool() {
  if is_enabled "$1"; then echo ON; else echo OFF; fi
}

CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY_CMAKE=$(cmake_bool "$CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY")
CPPEVENT_BUILD_PYTHON_BINDINGS_CMAKE=$(cmake_bool "$CPPEVENT_BUILD_PYTHON_BINDINGS")
LLI_ENABLE_CUPTI_ACTIVITY_METADATA_WRAPPER_HOOKS_CMAKE=$(cmake_bool "$LLI_ENABLE_CUPTI_ACTIVITY_METADATA_WRAPPER_HOOKS")

echo "[real-wrapper-build] root=$repo_root"
echo "[real-wrapper-build] compiler=$($cxx --version | head -n 1)"
echo "[real-wrapper-build] python=$($python_bin --version 2>&1)"
echo "[real-wrapper-build] CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY=$CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY_CMAKE"
echo "[real-wrapper-build] CPPEVENT_BUILD_PYTHON_BINDINGS=$CPPEVENT_BUILD_PYTHON_BINDINGS_CMAKE"
echo "[real-wrapper-build] LLI_ENABLE_CUPTI_ACTIVITY_METADATA_WRAPPER_HOOKS=$LLI_ENABLE_CUPTI_ACTIVITY_METADATA_WRAPPER_HOOKS_CMAKE"

cmake -S "$repo_root/CppEvent" -B "$cpp_event_build_dir" \
  -DCOMPOSE_ENABLE_TESTS=OFF \
  -DCOMPOSE_ENABLE_PYTHON="$CPPEVENT_BUILD_PYTHON_BINDINGS_CMAKE" \
  -DCPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY="$CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY_CMAKE" \
  -DPython_ROOT_DIR="$python_root_dir" \
  -DPython_EXECUTABLE="$python_bin" \
  -DPython3_ROOT_DIR="$python_root_dir" \
  -DPython3_EXECUTABLE="$python_bin" \
  >/dev/null
cmake_targets=(cpp_event_runtime_shared cpp_event_tls_shared)
if is_enabled "$CPPEVENT_BUILD_PYTHON_BINDINGS_CMAKE"; then
  cmake_targets+=(cpp_event_py cpp_event_tls)
fi
cmake --build "$cpp_event_build_dir" -j --target "${cmake_targets[@]}" >/dev/null

"$cxx" "${common_shared_flags[@]}" "${common_includes[@]}" \
  "$repo_root/cpp/lowlevel_interface/src/debug_event.cpp" \
  -L"$cpp_event_out" "$lli_runtime_rpath" \
  -lcpp_event_runtime_shared -lcpp_event_tls_shared \
  "${common_link_flags[@]}" \
  -o "$lli_out/liblowlevel_interface_runtime.so.tmp"
mv "$lli_out/liblowlevel_interface_runtime.so.tmp" \
  "$lli_out/liblowlevel_interface_runtime.so"

build_wrapper() {
  local prefix=$1 backend_macro=$2 backend_soname=$3
  local generated_source="$generated_dir/${prefix}_real_wrappers.cpp"
  local output_path="$lli_out/lib${prefix}_real_wrappers.so"
  local cupti_args=()
  if is_enabled "$LLI_ENABLE_CUPTI_ACTIVITY_METADATA_WRAPPER_HOOKS_CMAKE"; then
    cupti_args+=(--enable-cupti-activity-metadata)
  fi

  "$python_bin" "$repo_root/cpp/lowlevel_interface/scripts/generate_wrappers.py" \
    --config "$repo_root/cpp/lowlevel_interface/config/${prefix}_wrappers.json" \
    --wrapper-template "$repo_root/cpp/lowlevel_interface/templates/wrapper.cpp.in" \
    --preamble-template "$repo_root/cpp/lowlevel_interface/templates/module_preamble.cpp.in" \
    --backend-macro "$backend_macro" \
    --output-root "$lli_out" \
    --output-relpath "generated/${prefix}_real_wrappers.cpp" \
    "${cupti_args[@]}"

  "$cxx" "${common_shared_flags[@]}" "${common_includes[@]}" \
    "-D${backend_macro}=\"${backend_soname}\"" "$generated_source" \
    -L"$lli_out" -L"$cpp_event_out" "$lli_runtime_rpath" \
    -llowlevel_interface_runtime -lcpp_event_runtime_shared \
    -lcpp_event_tls_shared "${common_link_flags[@]}" \
    -o "${output_path}.tmp"
  mv "${output_path}.tmp" "$output_path"
}

build_wrapper cuda LLI_REAL_EXTERNAL_CUDA_PATH libcudart.so.12
build_wrapper nccl LLI_REAL_EXTERNAL_NCCL_PATH libnccl.so.2
build_wrapper cublas LLI_REAL_EXTERNAL_CUBLAS_PATH libcublas.so.12

PYTHONPATH="$repo_root/python:$repo_root/CppEvent" \
LD_LIBRARY_PATH="$repo_root/CppEvent:$lli_out${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$python_bin" - <<'PY'
import cpp_event_py
import cpp_event_tls

assert not getattr(cpp_event_py, "__fallback__", False)
print("native real-wrapper stack: PASS")
PY
