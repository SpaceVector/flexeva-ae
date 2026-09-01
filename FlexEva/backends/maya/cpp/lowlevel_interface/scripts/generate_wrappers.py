#!/usr/bin/env python3
"""Generate low-level interface wrapper sources from JSON metadata."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

EVENT_KIND_MAP = {
    "Unknown": "cpp_event::EventKind::kUnknown",
    "RuntimeCall": "cpp_event::EventKind::kRuntimeCall",
    "Collective": "cpp_event::EventKind::kCollective",
    "PointToPoint": "cpp_event::EventKind::kPointToPoint",
    "Barrier": "cpp_event::EventKind::kBarrier",
    "AllReduce": "cpp_event::EventKind::kAllReduce",
    "Broadcast": "cpp_event::EventKind::kBroadcast",
    "AllGather": "cpp_event::EventKind::kAllGather",
    "ReduceScatter": "cpp_event::EventKind::kReduceScatter",
    "Send": "cpp_event::EventKind::kSend",
    "Recv": "cpp_event::EventKind::kRecv",
    "MemcpyHostToDevice": "cpp_event::EventKind::kMemcpyHostToDevice",
    "MemcpyDeviceToHost": "cpp_event::EventKind::kMemcpyDeviceToHost",
    "MemcpyDeviceToDevice": "cpp_event::EventKind::kMemcpyDeviceToDevice",
    "MemoryAllocation": "cpp_event::EventKind::kMemoryAllocation",
    "MemoryFree": "cpp_event::EventKind::kMemoryFree",
    "FileRead": "cpp_event::EventKind::kFileRead",
    "FileWrite": "cpp_event::EventKind::kFileWrite",
}


@dataclass
class Parameter:
    name: str
    declaration: str
    type_signature: str


@dataclass
class ApiSpec:
    name: str
    return_type: str
    event_kind: str
    event_kind_expr: str
    parameters: Sequence[Parameter]


@dataclass
class Config:
    api_set: str
    output: Path
    includes: Sequence[str]
    apis: Sequence[ApiSpec]
    enable_cupti_activity_metadata: bool = False


class GeneratorError(RuntimeError):
    """Raised when configuration validation fails."""


def _load_parameters(entries: Iterable[dict]) -> List[Parameter]:
    params: List[Parameter] = []
    for raw in entries:
        try:
            name = raw["name"]
            declaration = raw["declaration"]
            type_signature = raw["type"]
        except KeyError as exc:
            raise GeneratorError(f"parameter entry missing field: {exc}") from exc
        params.append(Parameter(name=name, declaration=declaration, type_signature=type_signature))
    return params


def _event_kind_expr(kind: str) -> str:
    try:
        return EVENT_KIND_MAP[kind]
    except KeyError as exc:
        supported = ", ".join(sorted(EVENT_KIND_MAP))
        raise GeneratorError(f"unsupported event_kind '{kind}'. Supported kinds: {supported}") from exc


def _load_config(path: Path) -> Config:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GeneratorError(f"failed to parse JSON config {path}: {exc}") from exc

    try:
        api_set = data["api_set"]
        output = Path(data["output"])
        includes = list(data.get("includes", []))
        raw_apis = data["apis"]
    except KeyError as exc:
        raise GeneratorError(f"config missing required field: {exc}") from exc

    apis: List[ApiSpec] = []
    for entry in raw_apis:
        try:
            name = entry["name"]
            return_type = entry["return_type"]
            event_kind = entry["event_kind"]
            params = _load_parameters(entry.get("parameters", []))
        except KeyError as exc:
            raise GeneratorError(f"api entry for {entry.get('name', '<unknown>')} missing field: {exc}") from exc
        apis.append(
            ApiSpec(
                name=name,
                return_type=return_type,
                event_kind=event_kind,
                event_kind_expr=_event_kind_expr(event_kind),
                parameters=params,
            )
        )

    return Config(api_set=api_set, output=output, includes=includes, apis=apis)


def _render_dispatch_call(spec: ApiSpec) -> str:
    arg_list = ", ".join(param.name for param in spec.parameters)
    base_call = (
        f'lowlevel::interface::dispatch(canonicalize_api_name_local("{spec.name}"), '
        f'lookup_event_kind_local("{spec.name}"), payload, forward_call'
    )
    if arg_list:
        base_call += f", {arg_list}"
    base_call += ")"

    if spec.return_type.strip() == "void":
        return f"{base_call};"
    return f"return {base_call};"


def _is_launch_boundary_visibility_api(api_set: str, spec: ApiSpec) -> bool:
    if api_set == "cuda":
        return spec.name in {
            "cudaGetDevice",
            "cudaLaunchKernel",
            "cudaEventRecord",
            "cudaStreamWaitEvent",
        }
    if api_set == "cublas":
        return spec.name == "cublasSetStream_v2"
    return False


def _render_call_and_record(api_set: str, spec: ApiSpec) -> str:
    arg_list = ", ".join(param.name for param in spec.parameters)
    call_expr = f"forward_call({arg_list})" if arg_list else "forward_call()"
    canonical_api_expr = f'canonicalize_api_name_local("{spec.name}")'
    event_kind_expr = f'lookup_event_kind_local("{spec.name}")'
    record_expr = (
        f"lowlevel::interface::recorder().record_event({canonical_api_expr}, "
        f"{event_kind_expr}, start_time, end_time, payload);"
    )
    attach_metadata = ""
    if _is_launch_boundary_visibility_api(api_set, spec):
        attach_metadata = (
            "  lowlevel::interface::attach_launch_boundary_visibility_metadata(\n"
            f"      payload, {canonical_api_expr}, start_time, end_time);"
            "\n"
        )
    if spec.return_type.strip() == "void":
        return (
            "auto start_time = lowlevel::interface::EventRecorder::Clock::now();\n"
            "  {{PRE_CALL_HOOK}}\n"
            f"  {call_expr};\n"
            "  {{POST_CALL_HOOK}}\n"
            "  auto end_time = lowlevel::interface::EventRecorder::Clock::now();\n"
            "  {{POST_PAYLOAD_BUILD}}\n"
            f"{attach_metadata}"
            f"  {record_expr}"
        )
    return (
        "auto start_time = lowlevel::interface::EventRecorder::Clock::now();\n"
        "  {{PRE_CALL_HOOK}}\n"
        f"  auto result = {call_expr};\n"
        "  {{POST_CALL_HOOK}}\n"
        "  auto end_time = lowlevel::interface::EventRecorder::Clock::now();\n"
        "  {{POST_PAYLOAD_BUILD}}\n"
        f"{attach_metadata}"
        f"  {record_expr}\n"
        "  return result;"
    )


def _render_fn_pointer_signature(spec: ApiSpec) -> str:
    if not spec.parameters:
        return ""
    return ", ".join(param.type_signature for param in spec.parameters)


def _render_parameter_signature(spec: ApiSpec) -> str:
    if not spec.parameters:
        return "void"
    return ", ".join(param.declaration for param in spec.parameters)


def _render_api(
    template: str,
    api_set: str,
    spec: ApiSpec,
    *,
    enable_cupti_activity_metadata: bool,
) -> str:
    param_names = ", ".join(param.name for param in spec.parameters)
    replacements = {
        "{{RETURN_TYPE}}": spec.return_type,
        "{{API_NAME}}": spec.name,
        "{{PARAM_SIGNATURE}}": _render_parameter_signature(spec),
        "{{PARAM_TYPES}}": _render_fn_pointer_signature(spec),
        "{{PARAM_NAMES}}": param_names,
        "{{DISPATCH_CALL}}": _render_dispatch_call(spec),
        "{{CALL_AND_RECORD}}": _render_call_and_record(api_set, spec),
        "{{PRE_PAYLOAD_BUILD}}": "{{PRE_PAYLOAD_BUILD}}",
        "{{PRE_CALL_HOOK}}": "{{PRE_CALL_HOOK}}",
        "{{POST_CALL_HOOK}}": "{{POST_CALL_HOOK}}",
        "{{POST_PAYLOAD_BUILD}}": "{{POST_PAYLOAD_BUILD}}",
    }
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def _collect_includes(config: Config) -> List[str]:
    includes = [
        "\"lowlevel/interface/async_runtime_observer.hpp\"",
        "\"lowlevel/interface/wrapper_template.hpp\"",
        "\"cpp_event/event_schema.hpp\"",
        "<array>",
        "<cstdlib>",
        "<dlfcn.h>",
        "<iostream>",
        "<stdexcept>",
        "<string>",
        "<string_view>",
        "<utility>",
    ]
    if config.enable_cupti_activity_metadata:
        includes.insert(1, "\"lowlevel/interface/cupti_activity_metadata_observer.hpp\"")
    for item in config.includes:
        if item not in includes:
            includes.append(item)
    return includes


def _render_includes_block(config: Config) -> str:
    return "\n".join(f"#include {include}" for include in _collect_includes(config))


def _render_metadata_block(config: Config) -> str:
    api_count = len(config.apis)
    if api_count == 0:
        return "constexpr std::array<ApiMetadata, 0> kApiMetadata{};"

    entries = ",\n".join(
        f'    {{"{api.name}", {api.event_kind_expr}}}' for api in config.apis
    )
    return (
        f"constexpr std::array<ApiMetadata, {api_count}> kApiMetadata{{{{\n"
        f"{entries}\n"
        f"}}}};"
    )


def _render_preamble(
    preamble_template: str, config: Config, backend_macro: str
) -> str:
    replacements = {
        "{{AUTOGEN_BANNER}}": "// @generated by scripts/generate_wrappers.py; DO NOT EDIT.",
        "{{INCLUDES}}": _render_includes_block(config),
        "{{API_METADATA_BLOCK}}": _render_metadata_block(config),
        "{{BACKEND_LIBRARY_MACRO}}": backend_macro,
    }
    preamble = preamble_template
    for key, value in replacements.items():
        preamble = preamble.replace(key, value)
    return preamble


def _generate_source(
    preamble_template: str,
    wrapper_template: str,
    config: Config,
    backend_macro: str,
) -> str:
    segments = [_render_preamble(preamble_template, config, backend_macro), ""]

    for api in config.apis:
        rendered = _render_api(
            wrapper_template,
            config.api_set,
            api,
            enable_cupti_activity_metadata=config.enable_cupti_activity_metadata,
        )
        pre_payload, pre_call_hook, post_call_hook, post_payload = _render_payload_build(
            config.api_set,
            api,
            enable_cupti_activity_metadata=config.enable_cupti_activity_metadata,
        )
        rendered = rendered.replace("{{PRE_PAYLOAD_BUILD}}", pre_payload)
        rendered = rendered.replace("{{PRE_CALL_HOOK}}", pre_call_hook)
        rendered = rendered.replace("{{POST_CALL_HOOK}}", post_call_hook)
        rendered = rendered.replace("{{POST_PAYLOAD_BUILD}}", post_payload)
        segments.append(rendered)
        segments.append("")

    return "\n".join(segments).rstrip() + "\n"


def generate_from_config(
    config_path: Path,
    wrapper_template_path: Path,
    preamble_template_path: Path,
    backend_macro: str,
    output_root: Path,
    output_relpath: Path | None = None,
    enable_cupti_activity_metadata: bool = False,
) -> Path:
    config = _load_config(config_path)
    if output_relpath is not None:
        config = Config(
            api_set=config.api_set,
            output=output_relpath,
            includes=config.includes,
            apis=config.apis,
            enable_cupti_activity_metadata=enable_cupti_activity_metadata,
        )
    else:
        config = Config(
            api_set=config.api_set,
            output=config.output,
            includes=config.includes,
            apis=config.apis,
            enable_cupti_activity_metadata=enable_cupti_activity_metadata,
        )
    wrapper_template = wrapper_template_path.read_text(encoding="utf-8")
    preamble_template = preamble_template_path.read_text(encoding="utf-8")

    rendered = _generate_source(
        preamble_template, wrapper_template, config, backend_macro
    )

    output_path = output_root / config.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate low-level interface wrappers.")
    parser.add_argument("--config", type=Path, required=True, help="Path to wrapper metadata JSON.")
    parser.add_argument(
        "--wrapper-template",
        dest="wrapper_template",
        type=Path,
        required=False,
        default=Path(__file__).resolve().parent.parent / "templates" / "wrapper.cpp.in",
        help="Path to the wrapper template file.",
    )
    parser.add_argument(
        "--preamble-template",
        dest="preamble_template",
        type=Path,
        required=False,
        default=Path(__file__).resolve().parent.parent / "templates" / "module_preamble.cpp.in",
        help="Path to the module preamble template.",
    )
    parser.add_argument(
        "--backend-macro",
        dest="backend_macro",
        type=str,
        required=True,
        help="Name of the preprocessor macro that expands to the backend library path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=False,
        default=Path(__file__).resolve().parent.parent,
        help="Root directory that output paths are resolved against.",
    )
    parser.add_argument(
        "--output-relpath",
        type=Path,
        required=False,
        default=None,
        help="Optional output path relative to --output-root, overriding config output.",
    )
    parser.add_argument(
        "--enable-cupti-activity-metadata",
        action="store_true",
        help="Emit default-off CUPTI activity metadata observer hooks into generated wrappers.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    try:
        output_path = generate_from_config(
            args.config,
            args.wrapper_template,
            args.preamble_template,
            args.backend_macro,
            args.output_root,
            args.output_relpath,
            enable_cupti_activity_metadata=args.enable_cupti_activity_metadata,
        )
    except GeneratorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"io error: {exc}", file=sys.stderr)
        return 1

    print(f"Generated {output_path}")
    return 0


def _payload_line(key: str, expr: str) -> str:
    return f'lowlevel::interface::set_payload_attr(payload, "{key}", {expr});'


def _find_param(spec: ApiSpec, name: str) -> Parameter | None:
    for param in spec.parameters:
        if param.name == name:
            return param
    return None


def _is_async_runtime_signal_api(api_set: str, spec: ApiSpec) -> bool:
    if api_set == "cuda":
        return spec.name == "cudaLaunchKernel"
    if api_set == "cublas":
        return spec.name in {
            "cublasSgemm_v2",
            "cublasGemmEx",
            "cublasGemmStridedBatchedEx",
            "cublasGemmBatchedEx",
            "cublasLtMatmul",
        }
    if api_set == "nccl":
        return spec.name in {
            "ncclAllReduce",
            "ncclAllGather",
            "ncclAllToAll",
            "ncclAllToAllv",
            "ncclBroadcast",
            "ncclReduce",
            "ncclReduceScatter",
            "ncclSend",
            "ncclRecv",
        }
    return False


def _is_cupti_activity_metadata_api(api_set: str, spec: ApiSpec) -> bool:
    if api_set == "cuda":
        return spec.name in {
            "cudaLaunchKernel",
            "cudaEventRecord",
            "cudaEventRecordWithFlags",
            "cudaStreamWaitEvent",
        }
    if api_set == "cublas":
        return spec.name in {
            "cublasGemmEx",
            "cublasGemmStridedBatchedEx",
        }
    return False


def _is_cublas_handle_stream_payload_api(api_set: str, spec: ApiSpec) -> bool:
    if api_set != "cublas":
        return False
    return spec.name in {
        "cublasSgemm_v2",
        "cublasDgemm_v2",
        "cublasHgemm",
        "cublasGemmEx",
        "cublasGemmStridedBatchedEx",
        "cublasGemmBatchedEx",
    }


def _render_async_runtime_pre_call(api_set: str, spec: ApiSpec) -> str:
    if not _is_async_runtime_signal_api(api_set, spec):
        return "/* no pre-call hook */"
    if api_set in {"cuda", "nccl"}:
        return (
            'async_runtime_observation = lowlevel::interface::begin_async_runtime_observation('
            f'canonicalize_api_name_local("{spec.name}"), reinterpret_cast<void *>(stream));'
        )
    if api_set == "cublas":
        return (
            'async_runtime_observation = lowlevel::interface::begin_async_runtime_observation('
            f'canonicalize_api_name_local("{spec.name}"), '
            'lowlevel::interface::lookup_cublas_handle_stream_for_async_runtime('
            'reinterpret_cast<void *>(handle)));'
        )
    return "/* no pre-call hook */"


def _render_cupti_activity_metadata_pre_call(spec: ApiSpec) -> str:
    return (
        'cupti_activity_metadata_observation = '
        'lowlevel::interface::begin_cupti_activity_metadata_observation('
        f'canonicalize_api_name_local("{spec.name}"));'
    )


def _render_payload_build(
    api_set: str,
    spec: ApiSpec,
    *,
    enable_cupti_activity_metadata: bool,
) -> tuple[str, str, str, str]:
    pre_lines = ["cpp_event::EventPayload payload{};"]
    post_call_lines: list[str] = []
    post_lines: list[str] = []
    param_names = {param.name for param in spec.parameters}
    pre_call_hooks: list[str] = []

    generic_params = (
        "m", "n", "k", "lda", "ldb", "ldc", "count", "size", "sendcount",
        "recvcount", "root", "peer", "batchCount", "strideA", "strideB", "strideC",
        "kind", "op", "datatype", "Atype", "Btype", "Ctype", "computeType", "algo",
        "rank", "color", "key",
        "transa", "transb",
    )
    for name in generic_params:
        if name in param_names:
            pre_lines.append(_payload_line(name, name))

    if "stream" in param_names:
        pre_lines.append(_payload_line("stream_id", "format_handle_id(stream)"))
    if "streamId" in param_names:
        pre_lines.append(_payload_line("stream_id", "format_handle_id(streamId)"))
    if "event" in param_names:
        pre_lines.append(_payload_line("event_id", "format_handle_id(event)"))
    if "start" in param_names:
        pre_lines.append(_payload_line("start_event_id", "format_handle_id(start)"))
    if "end" in param_names:
        pre_lines.append(_payload_line("end_event_id", "format_handle_id(end)"))
    if "handle" in param_names:
        pre_lines.append(_payload_line("handle_id", "format_handle_id(handle)"))
    if "comm" in param_names:
        pre_lines.append(_payload_line("comm_id", "format_handle_id(comm)"))
    if "func" in param_names:
        pre_lines.append(_payload_line("kernel_id", "format_handle_id(func)"))
    if "hostFun" in param_names:
        pre_lines.append(_payload_line("host_fun_id", "format_handle_id((const void*)hostFun)"))

    if api_set == "cuda":
        if spec.name == "__cudaRegisterFunction":
            if "deviceName" in param_names:
                pre_lines.append(_payload_line("kernel", "maybe_demangle_symbol_name(deviceName)"))
                pre_lines.append(_payload_line("device_name", "deviceName"))
            if "deviceFun" in param_names:
                pre_lines.append(_payload_line("device_fun", "deviceFun"))
            if {"hostFun", "deviceName"}.issubset(param_names):
                pre_lines.append("remember_registered_kernel((const void*)hostFun, deviceName, deviceFun);")
        if spec.name in {"cudaMemcpy", "cudaMemcpyAsync"} and "count" in param_names:
            pre_lines.append(_payload_line("bytes", "count"))
        if spec.name in {"cudaMalloc", "cudaMallocAsync"} and "size" in param_names:
            pre_lines.append(_payload_line("bytes", "size"))
        if spec.name in {"cudaMemset", "cudaMemsetAsync"} and "count" in param_names:
            pre_lines.append(_payload_line("bytes", "count"))
        if spec.name == "cudaLaunchKernel":
            if "func" in param_names:
                pre_lines.append(_payload_line("kernel", "format_kernel_name(func)"))
            if "gridDim" in param_names:
                pre_lines.append(_payload_line("grid_x", "gridDim.x"))
                pre_lines.append(_payload_line("grid_y", "gridDim.y"))
                pre_lines.append(_payload_line("grid_z", "gridDim.z"))
            if "blockDim" in param_names:
                pre_lines.append(_payload_line("block_x", "blockDim.x"))
                pre_lines.append(_payload_line("block_y", "blockDim.y"))
                pre_lines.append(_payload_line("block_z", "blockDim.z"))
            if "sharedMem" in param_names:
                pre_lines.append(_payload_line("shared_mem", "sharedMem"))
        if spec.name in {"cudaStreamCreate", "cudaStreamCreateWithFlags", "cudaStreamCreateWithPriority"} and "pStream" in param_names:
            post_lines.append(_payload_line("stream_id", "format_created_handle_id(pStream)"))
        if spec.name in {"cudaEventCreate", "cudaEventCreateWithFlags"} and "event" in param_names:
            post_lines.append(_payload_line("event_id", "format_created_handle_id(event)"))
        if _is_async_runtime_signal_api(api_set, spec):
            pre_lines.append(
                "lowlevel::interface::AsyncRuntimeObservation async_runtime_observation{};"
            )
            pre_call_hooks.append(_render_async_runtime_pre_call(api_set, spec))
            post_lines.append(
                "lowlevel::interface::complete_async_runtime_observation("
                "async_runtime_observation, payload, static_cast<int>(result) == 0);"
            )
        if enable_cupti_activity_metadata and _is_cupti_activity_metadata_api(api_set, spec):
            pre_lines.append(
                "lowlevel::interface::CuptiActivityMetadataObservation "
                "cupti_activity_metadata_observation{};"
            )
            pre_call_hooks.append(_render_cupti_activity_metadata_pre_call(spec))
            post_call_lines.append(
                "lowlevel::interface::complete_cupti_activity_metadata_observation("
                "cupti_activity_metadata_observation, payload, "
                "static_cast<int>(result) == 0);"
            )

    if api_set == "cublas":
        if _is_cublas_handle_stream_payload_api(api_set, spec) and "handle" in param_names:
            pre_lines.append(
                "void *cublas_stream_handle = nullptr;"
            )
            pre_lines.append(
                "if (lowlevel::interface::lookup_registered_cublas_handle_stream_for_async_runtime("
                "reinterpret_cast<void *>(handle), &cublas_stream_handle)) {"
            )
            pre_lines.append(
                "  "
                + _payload_line("stream_id", "format_handle_id(cublas_stream_handle)")
            )
            pre_lines.append(
                '  lowlevel::interface::set_payload_attr(payload, "stream_id_source", '
                '"cublas_handle_stream_registry");'
            )
            pre_lines.append("}")
        if "m" in param_names and "n" in param_names and "k" in param_names:
            pre_lines.append(_payload_line("kind", '"gemm"'))
        if spec.name in {"cublasSgemm_v2", "cublasDgemm_v2", "cublasHgemm"}:
            pre_lines.append(_payload_line("batch_count", "1"))
        if "batchCount" in param_names:
            pre_lines.append(_payload_line("batch_count", "batchCount"))
        if spec.name == "cublasSgemm_v2":
            pre_lines.append(_payload_line("dtype_code", "1"))
        if spec.name == "cublasHgemm":
            pre_lines.append(_payload_line("dtype_code", "0"))
        if "computeType" in param_names:
            pre_lines.append(_payload_line(
                "dtype_code",
                "normalize_cublas_dtype_code_from_int(static_cast<int>(computeType))",
            ))
        if spec.name == "cublasCreate_v2" and "handle" in param_names:
            post_lines.append(_payload_line("handle_id", "format_created_handle_id(handle)"))
            post_lines.append("if (static_cast<int>(result) == 0 && handle != nullptr) {")
            post_lines.append(
                "  lowlevel::interface::register_cublas_handle_for_async_runtime("
                "reinterpret_cast<void *>(*handle));"
            )
            post_lines.append("}")
        if spec.name == "cublasDestroy_v2" and "handle" in param_names:
            post_lines.append("if (static_cast<int>(result) == 0) {")
            post_lines.append(
                "  lowlevel::interface::unregister_cublas_handle_for_async_runtime("
                "reinterpret_cast<void *>(handle));"
            )
            post_lines.append("}")
        if spec.name == "cublasSetStream_v2" and {"handle", "streamId"}.issubset(param_names):
            post_lines.append("if (static_cast<int>(result) == 0) {")
            post_lines.append(
                "  lowlevel::interface::update_cublas_handle_stream_for_async_runtime("
                "reinterpret_cast<void *>(handle), reinterpret_cast<void *>(streamId));"
            )
            post_lines.append("}")
        if _is_async_runtime_signal_api(api_set, spec):
            pre_lines.append(
                "lowlevel::interface::AsyncRuntimeObservation async_runtime_observation{};"
            )
            pre_call_hooks.append(_render_async_runtime_pre_call(api_set, spec))
            post_lines.append(
                "lowlevel::interface::complete_async_runtime_observation("
                "async_runtime_observation, payload, static_cast<int>(result) == 0);"
            )
        if enable_cupti_activity_metadata and _is_cupti_activity_metadata_api(api_set, spec):
            pre_lines.append(
                "lowlevel::interface::CuptiActivityMetadataObservation "
                "cupti_activity_metadata_observation{};"
            )
            pre_call_hooks.append(_render_cupti_activity_metadata_pre_call(spec))
            post_call_lines.append(
                "lowlevel::interface::complete_cupti_activity_metadata_observation("
                "cupti_activity_metadata_observation, payload, "
                "static_cast<int>(result) == 0);"
            )

    if api_set == "nccl":
        collective_name = None
        if spec.name == "ncclAllReduce":
            collective_name = "allreduce"
        elif spec.name == "ncclAllToAll":
            collective_name = "alltoall"
        elif spec.name == "ncclAllToAllv":
            collective_name = "alltoallv"
        elif spec.name == "ncclBroadcast":
            collective_name = "broadcast"
        elif spec.name == "ncclAllGather":
            collective_name = "allgather"
        elif spec.name == "ncclReduce":
            collective_name = "reduce"
        elif spec.name == "ncclReduceScatter":
            collective_name = "reducescatter"
        elif spec.name == "ncclSend":
            collective_name = "send"
        elif spec.name == "ncclRecv":
            collective_name = "recv"
        if collective_name is not None:
            pre_lines.append(_payload_line("collective", f'"{collective_name}"'))
        if "count" in param_names:
            pre_lines.append(_payload_line("numel", "count"))
        elif "sendcount" in param_names:
            pre_lines.append(_payload_line("numel", "sendcount"))
        elif "recvcount" in param_names:
            pre_lines.append(_payload_line("numel", "recvcount"))
        if "datatype" in param_names:
            pre_lines.append(_payload_line(
                "dtype_code",
                "normalize_nccl_dtype_code_from_int(static_cast<int>(datatype))",
            ))
        if "op" in param_names:
            pre_lines.append("if (const char *reduction_name = "
                         "normalize_nccl_reduction_name_from_int(static_cast<int>(op)); "
                         "reduction_name != nullptr) {")
            pre_lines.append('  lowlevel::interface::set_payload_attr(payload, "reduction", reduction_name);')
            pre_lines.append("}")
        if "nranks" in param_names:
            pre_lines.append(_payload_line("nranks", "nranks"))
            pre_lines.append(_payload_line("world_size", "nranks"))
        if spec.name in {"ncclCommInitRank", "ncclCommInitRankConfig"} and "comm" in param_names:
            post_lines.append(_payload_line("comm_id", "format_created_handle_id(comm)"))
        if spec.name == "ncclCommSplit" and "comm" in param_names:
            pre_lines.append(_payload_line("parent_comm_id", "format_handle_id(comm)"))
        if spec.name == "ncclCommSplit" and "newcomm" in param_names:
            post_lines.append(_payload_line("new_comm_id", "format_created_handle_id(newcomm)"))
        if _is_async_runtime_signal_api(api_set, spec):
            pre_lines.append(
                "lowlevel::interface::AsyncRuntimeObservation async_runtime_observation{};"
            )
            pre_call_hooks.append(_render_async_runtime_pre_call(api_set, spec))
            post_lines.append(
                "lowlevel::interface::complete_async_runtime_observation("
                "async_runtime_observation, payload, static_cast<int>(result) == 0);"
            )

    return (
        "\n  ".join(pre_lines),
        "\n  ".join(pre_call_hooks) if pre_call_hooks else "/* no pre-call hook */",
        "\n  ".join(post_call_lines) if post_call_lines else "/* no post-call hook */",
        "\n  ".join(post_lines) if post_lines else "/* no post-call payload */",
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


