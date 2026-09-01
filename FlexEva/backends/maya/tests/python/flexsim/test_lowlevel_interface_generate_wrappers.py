from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "cpp" / "lowlevel_interface" / "scripts" / "generate_wrappers.py"


def _load_generate_wrappers_module():
    spec = importlib.util.spec_from_file_location("lli_generate_wrappers", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cuda_launch_kernel_payload_includes_kernel_shape_and_shared_mem(tmp_path):
    module = _load_generate_wrappers_module()
    output_path = module.generate_from_config(
        REPO_ROOT / "cpp" / "lowlevel_interface" / "config" / "cuda_wrappers.json",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "wrapper.cpp.in",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "module_preamble.cpp.in",
        "LLI_REAL_EXTERNAL_CUDA_PATH",
        tmp_path,
        Path("generated/cuda_real_wrappers.cpp"),
    )

    rendered = output_path.read_text(encoding="utf-8")

    assert 'set_payload_attr(payload, "kernel", format_kernel_name(func));' in rendered
    assert 'set_payload_attr(payload, "grid_x", gridDim.x);' in rendered
    assert 'set_payload_attr(payload, "grid_y", gridDim.y);' in rendered
    assert 'set_payload_attr(payload, "grid_z", gridDim.z);' in rendered
    assert 'set_payload_attr(payload, "block_x", blockDim.x);' in rendered
    assert 'set_payload_attr(payload, "block_y", blockDim.y);' in rendered
    assert 'set_payload_attr(payload, "block_z", blockDim.z);' in rendered
    assert 'set_payload_attr(payload, "shared_mem", sharedMem);' in rendered
    assert "std::string format_kernel_name(const void *func)" in rendered
    assert "void remember_registered_kernel(const void *host_fun, const char *device_name," in rendered
    assert 'remember_registered_kernel((const void*)hostFun, deviceName, deviceFun);' in rendered


def test_cuda_pop_call_configuration_wrapper_is_not_generated_by_default(
    tmp_path,
):
    module = _load_generate_wrappers_module()
    output_path = module.generate_from_config(
        REPO_ROOT / "cpp" / "lowlevel_interface" / "config" / "cuda_wrappers.json",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "wrapper.cpp.in",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "module_preamble.cpp.in",
        "LLI_REAL_EXTERNAL_CUDA_PATH",
        tmp_path,
        Path("generated/cuda_real_wrappers.cpp"),
    )

    rendered = output_path.read_text(encoding="utf-8")

    assert "__cudaPopCallConfiguration" not in rendered


def test_generated_wrappers_canonicalize_alias_api_names(tmp_path):
    module = _load_generate_wrappers_module()
    output_path = module.generate_from_config(
        REPO_ROOT / "cpp" / "lowlevel_interface" / "config" / "cuda_wrappers.json",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "wrapper.cpp.in",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "module_preamble.cpp.in",
        "LLI_REAL_EXTERNAL_CUDA_PATH",
        tmp_path,
        Path("generated/cuda_real_wrappers.cpp"),
    )

    rendered = output_path.read_text(encoding="utf-8")

    assert "std::string_view canonicalize_api_name_local" in rendered
    assert 'record_event(canonicalize_api_name_local("cudaEventRecordWithFlags"),' in rendered
    assert 'record_event(canonicalize_api_name_local("cudaStreamCreateWithPriority"),' in rendered


def test_generated_wrappers_record_start_and_end_time(tmp_path):
    module = _load_generate_wrappers_module()
    output_path = module.generate_from_config(
        REPO_ROOT / "cpp" / "lowlevel_interface" / "config" / "cuda_wrappers.json",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "wrapper.cpp.in",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "module_preamble.cpp.in",
        "LLI_REAL_EXTERNAL_CUDA_PATH",
        tmp_path,
        Path("generated/cuda_real_wrappers.cpp"),
    )

    rendered = output_path.read_text(encoding="utf-8")

    assert "auto end_time = lowlevel::interface::EventRecorder::Clock::now();" in rendered
    assert (
        'record_event(canonicalize_api_name_local("cudaLaunchKernel"), '
        'lookup_event_kind_local("cudaLaunchKernel"), start_time, end_time, payload);'
    ) in rendered


def test_generated_cuda_wrappers_default_path_has_no_cupti_metadata_hooks(tmp_path):
    module = _load_generate_wrappers_module()
    output_path = module.generate_from_config(
        REPO_ROOT / "cpp" / "lowlevel_interface" / "config" / "cuda_wrappers.json",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "wrapper.cpp.in",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "module_preamble.cpp.in",
        "LLI_REAL_EXTERNAL_CUDA_PATH",
        tmp_path,
        Path("generated/cuda_real_wrappers.cpp"),
    )

    rendered = output_path.read_text(encoding="utf-8")

    assert '#include "lowlevel/interface/cupti_activity_metadata_observer.hpp"' not in rendered
    assert "CuptiActivityMetadataObservation" not in rendered
    assert "begin_cupti_activity_metadata_observation" not in rendered
    assert "complete_cupti_activity_metadata_observation" not in rendered
    assert "raw_event_id" not in rendered


def test_generated_cuda_wrappers_opt_in_includes_cupti_metadata_hooks_for_target_apis_only(
    tmp_path,
):
    module = _load_generate_wrappers_module()
    output_path = module.generate_from_config(
        REPO_ROOT / "cpp" / "lowlevel_interface" / "config" / "cuda_wrappers.json",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "wrapper.cpp.in",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "module_preamble.cpp.in",
        "LLI_REAL_EXTERNAL_CUDA_PATH",
        tmp_path,
        Path("generated/cuda_real_wrappers.cpp"),
        enable_cupti_activity_metadata=True,
    )

    rendered = output_path.read_text(encoding="utf-8")

    assert '#include "lowlevel/interface/cupti_activity_metadata_observer.hpp"' in rendered
    assert "raw_event_id" not in rendered
    assert "observed_runtime_us" not in rendered
    assert "wrapper_runtime_contract" not in rendered

    for api_name in (
        "cudaLaunchKernel",
        "cudaEventRecord",
        "cudaEventRecordWithFlags",
        "cudaStreamWaitEvent",
    ):
        signature = f'extern "C" cudaError_t {api_name}'
        start = rendered.index(signature)
        end = rendered.find('\nextern "C"', start + 1)
        wrapper = rendered[start:] if end == -1 else rendered[start:end]
        assert (
            "lowlevel::interface::CuptiActivityMetadataObservation "
            "cupti_activity_metadata_observation{};"
        ) in wrapper
        begin = (
            "cupti_activity_metadata_observation = "
            "lowlevel::interface::begin_cupti_activity_metadata_observation("
            f'canonicalize_api_name_local("{api_name}"));'
        )
        complete = (
            "lowlevel::interface::complete_cupti_activity_metadata_observation("
            "cupti_activity_metadata_observation, payload, "
            "static_cast<int>(result) == 0);"
        )
        assert begin in wrapper
        assert complete in wrapper
        assert wrapper.index(begin) < wrapper.index("auto result = forward_call(")
        assert wrapper.index("auto result = forward_call(") < wrapper.index(complete)
        assert wrapper.index(complete) < wrapper.index(
            "auto end_time = lowlevel::interface::EventRecorder::Clock::now();"
        )

    for api_name in ("cudaDeviceSynchronize", "cudaEventCreate", "cudaStreamCreate"):
        signature = f'extern "C" cudaError_t {api_name}'
        start = rendered.index(signature)
        end = rendered.find('\nextern "C"', start + 1)
        wrapper = rendered[start:] if end == -1 else rendered[start:end]
        assert "CuptiActivityMetadataObservation" not in wrapper
        assert "begin_cupti_activity_metadata_observation" not in wrapper
        assert "complete_cupti_activity_metadata_observation" not in wrapper


def test_generated_cublas_wrappers_opt_in_includes_cupti_metadata_hooks_for_gemm_only(
    tmp_path,
):
    module = _load_generate_wrappers_module()
    output_path = module.generate_from_config(
        REPO_ROOT / "cpp" / "lowlevel_interface" / "config" / "cublas_wrappers.json",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "wrapper.cpp.in",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "module_preamble.cpp.in",
        "LLI_REAL_EXTERNAL_CUBLAS_PATH",
        tmp_path,
        Path("generated/cublas_real_wrappers.cpp"),
        enable_cupti_activity_metadata=True,
    )

    rendered = output_path.read_text(encoding="utf-8")

    assert '#include "lowlevel/interface/cupti_activity_metadata_observer.hpp"' in rendered
    for api_name in ("cublasGemmEx", "cublasGemmStridedBatchedEx"):
        signature = f'extern "C" cublasStatus_t {api_name}'
        start = rendered.index(signature)
        end = rendered.find('\nextern "C"', start + 1)
        wrapper = rendered[start:] if end == -1 else rendered[start:end]
        assert "CuptiActivityMetadataObservation" in wrapper
        begin = (
            "cupti_activity_metadata_observation = "
            "lowlevel::interface::begin_cupti_activity_metadata_observation("
            f'canonicalize_api_name_local("{api_name}"));'
        )
        complete = (
            "lowlevel::interface::complete_cupti_activity_metadata_observation("
            "cupti_activity_metadata_observation, payload, "
            "static_cast<int>(result) == 0);"
        )
        assert begin in wrapper
        assert complete in wrapper
        assert wrapper.index(begin) < wrapper.index("auto result = forward_call(")
        assert wrapper.index("auto result = forward_call(") < wrapper.index(complete)

    set_stream_start = rendered.index('extern "C" cublasStatus_t cublasSetStream_v2')
    set_stream_end = rendered.find('\nextern "C"', set_stream_start + 1)
    set_stream_wrapper = (
        rendered[set_stream_start:]
        if set_stream_end == -1
        else rendered[set_stream_start:set_stream_end]
    )
    assert "CuptiActivityMetadataObservation" not in set_stream_wrapper
    assert "begin_cupti_activity_metadata_observation" not in set_stream_wrapper


def test_generated_cublas_gemm_wrappers_export_stream_from_handle_registry(
    tmp_path,
):
    module = _load_generate_wrappers_module()
    output_path = module.generate_from_config(
        REPO_ROOT / "cpp" / "lowlevel_interface" / "config" / "cublas_wrappers.json",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "wrapper.cpp.in",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "module_preamble.cpp.in",
        "LLI_REAL_EXTERNAL_CUBLAS_PATH",
        tmp_path,
        Path("generated/cublas_real_wrappers.cpp"),
    )

    rendered = output_path.read_text(encoding="utf-8")

    for api_name in ("cublasGemmEx", "cublasGemmStridedBatchedEx"):
        signature = f'extern "C" cublasStatus_t {api_name}'
        start = rendered.index(signature)
        end = rendered.find('\nextern "C"', start + 1)
        wrapper = rendered[start:] if end == -1 else rendered[start:end]
        lookup = (
            "if (lowlevel::interface::lookup_registered_cublas_handle_stream_for_async_runtime("
            "reinterpret_cast<void *>(handle), &cublas_stream_handle)) {"
        )
        assert "void *cublas_stream_handle = nullptr;" in wrapper
        assert lookup in wrapper
        assert 'set_payload_attr(payload, "stream_id", format_handle_id(cublas_stream_handle));' in wrapper
        assert 'set_payload_attr(payload, "stream_id_source", "cublas_handle_stream_registry");' in wrapper
        assert wrapper.index(lookup) < wrapper.index("auto result = forward_call(")

    set_stream_start = rendered.index('extern "C" cublasStatus_t cublasSetStream_v2')
    set_stream_end = rendered.find('\nextern "C"', set_stream_start + 1)
    set_stream_wrapper = (
        rendered[set_stream_start:]
        if set_stream_end == -1
        else rendered[set_stream_start:set_stream_end]
    )
    assert "update_cublas_handle_stream_for_async_runtime" in set_stream_wrapper


def test_cupti_metadata_observer_is_default_off_and_does_not_claim_runtime_repair():
    header = (
        REPO_ROOT
        / "cpp"
        / "lowlevel_interface"
        / "include"
        / "lowlevel"
        / "interface"
        / "cupti_activity_metadata_observer.hpp"
    ).read_text(encoding="utf-8")
    source = (
        REPO_ROOT
        / "cpp"
        / "cpp_event"
        / "src"
        / "cupti_activity_metadata_observer.cpp"
    ).read_text(encoding="utf-8")

    assert "MAYA_ENABLE_CUPTI_ACTIVITY_METADATA" in source
    assert "FLEXSIM_MAYA_ENABLE_CUPTI_ACTIVITY_METADATA" in source
    assert "cupti_activity_collector_not_compiled" in source
    assert "cupti_activity_metadata_raw_event_id_source" in source
    assert "capture_real_rank_trace_serialization" in source
    assert "cupti_activity_metadata_runtime_substitution" in source
    assert "cublasGemmEx" in source
    assert "cublasGemmStridedBatchedEx" in source
    assert "cublas_gemm" in source
    assert "cublas_strided_batched_gemm" in source
    assert '"false"' in source
    assert "observed_runtime_us" not in source
    assert "wrapper_runtime_contract" not in source
    assert "raw_event_id" not in header


def test_real_cupti_activity_collection_is_compile_time_opt_in_only():
    cmake = (REPO_ROOT / "cpp" / "cpp_event" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    direct_build = (
        REPO_ROOT / "scripts" / "build_real_wrapper_stack.sh"
    ).read_text(encoding="utf-8")
    source = (
        REPO_ROOT
        / "cpp"
        / "cpp_event"
        / "src"
        / "cupti_activity_metadata_observer.cpp"
    ).read_text(encoding="utf-8")

    assert "option(CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY" in cmake
    assert "find_path(CPPEVENT_CUPTI_INCLUDE_DIR" in cmake
    assert "find_library(CPPEVENT_CUPTI_LIBRARY" in cmake
    assert "CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY=1" in cmake
    assert 'CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY="${CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY:-OFF}"' in direct_build
    assert 'CPPEVENT_BUILD_PYTHON_BINDINGS="${CPPEVENT_BUILD_PYTHON_BINDINGS:-OFF}"' in direct_build
    assert 'LLI_ENABLE_CUPTI_ACTIVITY_METADATA_WRAPPER_HOOKS="${LLI_ENABLE_CUPTI_ACTIVITY_METADATA_WRAPPER_HOOKS:-OFF}"' in direct_build
    assert "CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY_CMAKE=" in direct_build
    assert "CPPEVENT_BUILD_PYTHON_BINDINGS_CMAKE=" in direct_build
    assert "LLI_ENABLE_CUPTI_ACTIVITY_METADATA_WRAPPER_HOOKS_CMAKE=" in direct_build
    assert '-DCOMPOSE_ENABLE_PYTHON="$CPPEVENT_BUILD_PYTHON_BINDINGS_CMAKE"' in direct_build
    assert '-DCPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY="$CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY_CMAKE"' in direct_build
    assert "cmake_targets+=(cpp_event_py cpp_event_tls)" in direct_build
    assert 'is_enabled "$LLI_ENABLE_CUPTI_ACTIVITY_METADATA_WRAPPER_HOOKS_CMAKE"' in direct_build
    assert "--enable-cupti-activity-metadata" in direct_build
    assert "#include <cupti.h>" in source
    assert "cuptiActivityRegisterCallbacks" in source
    assert "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME)" in source
    assert "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)" in source
    assert "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION)" in source
    assert "cuptiSubscribe" in source
    assert "cuptiEnableDomain" in source
    assert "CUPTI_CB_DOMAIN_RUNTIME_API" in source
    assert "callback_data->correlationId" in source
    assert "cupti_activity_callback_correlation_strategy" in source
    assert "cupti_activity_total_runtime_callback_count" in source
    assert "cupti_activity_total_runtime_callback_with_wrapper_count" in source
    assert "cupti_activity_total_runtime_callback_with_thread_window_count" in source
    assert "cupti_activity_first_target_callback_function" in source
    assert "cupti_activity_wrapper_begin_thread_id" in source
    assert "cupti_activity_callback_order_samples" in source
    assert "completed_callsite_fifo_by_thread_api" in source
    assert "cupti_activity_total_callsite_fifo_correlation_record_count" in source
    assert "cupti_activity_total_callsite_fifo_candidate_record_count" in source
    assert "cuptiActivityPushExternalCorrelationId" in source
    assert "cuptiActivityPopExternalCorrelationId" in source
    assert "cuptiActivityFlushAll" in source
    assert "cupti_activity_common_clock_status" in source
    assert "cupti_activity_strict_wait_timing" in source
    assert "cupti_activity_last_kernel_start" in source
    assert "cupti_activity_last_kernel_end" in source
    assert "cupti_activity_kernel_stream_id_unique_count" in source
    assert "cupti_activity_kernel_stream_id_status" in source
    assert "cupti_activity_device_activity_timing_status" in source
    assert "external_correlation_pushed" in source
    assert "if (observation.external_correlation_pushed)" in source
    assert "cupti_activity_global_dropped_record_count" in source
    assert "global_since_last_clear" in source
    assert "correlation_to_external_id().clear()" in source
    assert '"cupti_activity_dropped_record_count"' not in source


def test_generated_cuda_launch_wrapper_exports_boundary_visibility_segments_when_opted_in(tmp_path):
    module = _load_generate_wrappers_module()
    wrapper_header = (
        REPO_ROOT
        / "cpp"
        / "lowlevel_interface"
        / "include"
        / "lowlevel"
        / "interface"
        / "wrapper_template.hpp"
    ).read_text(encoding="utf-8")
    assert "MAYA_ENABLE_LAUNCH_BOUNDARY_VISIBILITY_DIAGNOSTICS" in wrapper_header
    assert "FLEXSIM_MAYA_ENABLE_LAUNCH_BOUNDARY_VISIBILITY_DIAGNOSTICS" in wrapper_header
    assert "launch_boundary_visibility_diagnostics_enabled()" in wrapper_header

    output_path = module.generate_from_config(
        REPO_ROOT / "cpp" / "lowlevel_interface" / "config" / "cuda_wrappers.json",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "wrapper.cpp.in",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "module_preamble.cpp.in",
        "LLI_REAL_EXTERNAL_CUDA_PATH",
        tmp_path,
        Path("generated/cuda_real_wrappers.cpp"),
    )

    rendered = output_path.read_text(encoding="utf-8")
    launch_start = rendered.index('extern "C" cudaError_t cudaLaunchKernel')
    launch_end = rendered.index('extern "C" void __cudaRegisterFunction', launch_start)
    launch_wrapper = rendered[launch_start:launch_end]

    assert "auto end_time = lowlevel::interface::EventRecorder::Clock::now();" in launch_wrapper
    assert (
        "attach_launch_boundary_visibility_metadata(\n"
        '      payload, canonicalize_api_name_local("cudaLaunchKernel"), '
        "start_time, end_time);"
    ) in launch_wrapper
    assert launch_wrapper.index(
        "auto end_time = lowlevel::interface::EventRecorder::Clock::now();"
    ) < launch_wrapper.index(
        "attach_launch_boundary_visibility_metadata("
    )
    assert launch_wrapper.index("attach_launch_boundary_visibility_metadata(") < launch_wrapper.index(
        'record_event(canonicalize_api_name_local("cudaLaunchKernel"),'
    )
    start_line = "auto start_time = lowlevel::interface::EventRecorder::Clock::now();"
    end_line = "auto end_time = lowlevel::interface::EventRecorder::Clock::now();"
    measured_interval = launch_wrapper[
        launch_wrapper.index(start_line) + len(start_line) : launch_wrapper.index(end_line)
    ]
    assert "Clock::now()" not in measured_interval
    assert "real_api_body_start" not in launch_wrapper
    assert "real_api_body_end" not in launch_wrapper
    assert "post_call_payload_build_start" not in launch_wrapper

    cublas_output_path = module.generate_from_config(
        REPO_ROOT / "cpp" / "lowlevel_interface" / "config" / "cublas_wrappers.json",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "wrapper.cpp.in",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "module_preamble.cpp.in",
        "LLI_REAL_EXTERNAL_CUBLAS_PATH",
        tmp_path,
        Path("generated/cublas_real_wrappers.cpp"),
    )
    cublas_rendered = cublas_output_path.read_text(encoding="utf-8")

    selected_wrappers = [
        (rendered, 'extern "C" cudaError_t cudaGetDevice'),
        (rendered, 'extern "C" cudaError_t cudaLaunchKernel'),
        (rendered, 'extern "C" cudaError_t cudaEventRecord'),
        (rendered, 'extern "C" cudaError_t cudaStreamWaitEvent'),
        (cublas_rendered, 'extern "C" cublasStatus_t cublasSetStream_v2'),
    ]
    for source_text, signature in selected_wrappers:
        start = source_text.index(signature)
        end = source_text.find('\nextern "C"', start + 1)
        wrapper = source_text[start:] if end == -1 else source_text[start:end]
        assert "attach_launch_boundary_visibility_metadata(" in wrapper
        start_pos = wrapper.index(start_line) + len(start_line)
        end_pos = wrapper.index(end_line)
        assert "Clock::now()" not in wrapper[start_pos:end_pos]
        assert "real_api_body_start" not in wrapper
        assert "real_api_body_end" not in wrapper
        assert "post_call_payload_build_start" not in wrapper


def test_fake_cuda_launch_boundary_metadata_is_logger_side_only():
    trace_header = (
        REPO_ROOT / "cpp" / "fake_cuda" / "include" / "common" / "trace_log.hpp"
    ).read_text(encoding="utf-8")
    assert "MAYA_ENABLE_LAUNCH_BOUNDARY_VISIBILITY_DIAGNOSTICS" in trace_header
    assert "FLEXSIM_MAYA_ENABLE_LAUNCH_BOUNDARY_VISIBILITY_DIAGNOSTICS" in trace_header
    assert "launch_boundary_visibility_diagnostics_enabled() &&" in trace_header

    selected_sources = [
        REPO_ROOT / "cpp" / "fake_cuda" / "src" / "runtime" / "common.cpp",
        REPO_ROOT / "cpp" / "fake_cuda" / "src" / "runtime" / "cudaLaunch.cpp",
        REPO_ROOT / "cpp" / "fake_cuda" / "src" / "runtime" / "cudaGet.cpp",
        REPO_ROOT / "cpp" / "fake_cuda" / "src" / "runtime" / "cudaEvent.cpp",
        REPO_ROOT / "cpp" / "fake_cuda" / "src" / "runtime" / "cudaStream.cpp",
        REPO_ROOT / "cpp" / "fake_cuda" / "src" / "cublas" / "cublasSet.cpp",
    ]
    forbidden = [
        "record_boundary_origin_body_duration_api",
        "set_boundary_origin_split_timing_api",
        "make_boundary_visibility_segment",
        "add_boundary_visibility_segment",
        "create_launch_boundary_id",
        "push_pending_launch_boundary_id",
        "take_pending_launch_boundary_id",
    ]
    for path in selected_sources:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{token} should not run before trace emission in {path}"


def test_fake_cuda_cublas_gemm_payload_exports_algorithm_aliases():
    source = (
        REPO_ROOT / "cpp" / "fake_cuda" / "src" / "cublas" / "common.cpp"
    ).read_text(encoding="utf-8")

    for signature in (
        "API cublasStatus_t cublasGemmEx",
        "API cublasStatus_t cublasGemmStridedBatchedEx",
    ):
        start = source.index(signature)
        end = source.find("\nAPI cublasStatus_t", start + 1)
        wrapper = source[start:] if end == -1 else source[start:end]

        assert 'payload.add_int("algorithm", static_cast<int>(algo));' in wrapper
        assert 'payload.add_int("algo", static_cast<int>(algo));' in wrapper


def test_generated_nccl_wrappers_emit_reduce_collective_metadata(tmp_path):
    module = _load_generate_wrappers_module()
    output_path = module.generate_from_config(
        REPO_ROOT / "cpp" / "lowlevel_interface" / "config" / "nccl_wrappers.json",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "wrapper.cpp.in",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "module_preamble.cpp.in",
        "LLI_REAL_EXTERNAL_NCCL_PATH",
        tmp_path,
        Path("generated/nccl_real_wrappers.cpp"),
    )

    rendered = output_path.read_text(encoding="utf-8")

    assert 'set_payload_attr(payload, "collective", "reduce");' in rendered
    assert 'set_payload_attr(payload, "numel", count);' in rendered
    assert (
        'normalize_nccl_reduction_name_from_int(static_cast<int>(op));'
    ) in rendered


def test_generated_nccl_wrappers_emit_comm_init_rank_config_metadata(tmp_path):
    module = _load_generate_wrappers_module()
    output_path = module.generate_from_config(
        REPO_ROOT / "cpp" / "lowlevel_interface" / "config" / "nccl_wrappers.json",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "wrapper.cpp.in",
        REPO_ROOT / "cpp" / "lowlevel_interface" / "templates" / "module_preamble.cpp.in",
        "LLI_REAL_EXTERNAL_NCCL_PATH",
        tmp_path,
        Path("generated/nccl_real_wrappers.cpp"),
    )

    rendered = output_path.read_text(encoding="utf-8")

    assert 'set_payload_attr(payload, "rank", rank);' in rendered
    assert 'set_payload_attr(payload, "nranks", nranks);' in rendered
    assert 'set_payload_attr(payload, "world_size", nranks);' in rendered
    assert rendered.count('set_payload_attr(payload, "comm_id", format_created_handle_id(comm));') >= 2
