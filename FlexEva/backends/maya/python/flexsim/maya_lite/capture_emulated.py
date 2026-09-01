#!/usr/bin/env python3
"""
Capture profiled workers with blind logical-world emulation.

Each profiled rank executes alone under fake-cuda. The repo-local
`sitecustomize.py` installs the Maya-lite emulated c10d bootstrap from env so
the workload runs directly, while phase 1 stays blind and still records
low-level CUDA/NCCL traces for the profiled ranks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import fcntl
from functools import lru_cache
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

DEFAULT_SAFE_CAPTURE_WORKERS = 1
DEFAULT_TRACE_FLUSH_EVERY = 16_384
DEFAULT_TRACE_STDIO_BUFFER_BYTES = 16 * 1024 * 1024
DEFAULT_DYNAMIC_DEDUP_TRACE_FLUSH_EVERY = 64
# Historical prefix classifier kept only for helper compatibility; native
# dynamic dedup now waits for complete first-step sequence hashing.
DEFAULT_DYNAMIC_DEDUP_PREFIX_TOKENS = 32
DEFAULT_DYNAMIC_DEDUP_POLL_INTERVAL_MS = 25
DEFAULT_DYNAMIC_DEDUP_KILL_GRACE_MS = 1_000
DEFAULT_DYNAMIC_DEDUP_ROLLING_WINDOW = 64
_DYNAMIC_DEDUP_VOLATILE_APIS = frozenset(
    {
        "cudaEventQuery",
        "cublasSetStream_v2",
        "cublasLtMatmulDescSetAttribute",
        "cublasLtMatmulPreferenceSetAttribute",
        "cublasLtMatrixLayoutSetAttribute",
    }
)
DEFAULT_WORKLOAD_HEURISTIC_MIN_EXTENSION_US = 100_000
DEFAULT_WORKLOAD_HEURISTIC_MIN_SUPPORTED_TAIL_EVENTS = 32
DEFAULT_WORKLOAD_HEURISTIC_GAP_CUTOFF_US = 100_000
DEFAULT_WORKLOAD_HEURISTIC_MIN_GAP_CANDIDATE_SEMANTIC_EVENTS = 100
_CPU_ONLINE_PATH = Path("/sys/devices/system/cpu/online")
_DEFAULT_CAPTURE_BOOTSTRAP_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

from flexsim.maya_lite.filters import (
    is_ignorable_setup_api,
    is_semantic_traced_api,
    is_supported_trace_api,
    is_teardown_api,
)
from flexsim.maya_lite.augment_emulated_helper_threads import (
    HELPER_THREAD_AUGMENTATION_STATUS_FAILED,
    HELPER_THREAD_AUGMENTATION_STATUS_MISSING_SUMMARY_DIR,
    HELPER_THREAD_AUGMENTATION_STATUS_NOT_REQUIRED,
    HELPER_THREAD_AUGMENTATION_STATUS_SUMMARY_DIR_NOT_FOUND,
    augment_trace_directory,
    record_helper_thread_augmentation_status,
)
from flexsim.maya_lite.io import (
    _coerce_event_end_ts,
    _effective_event_end_ts,
    _pattern_signature_for_event,
    estimate_rank_trace_active_seconds,
    estimate_rank_trace_window,
    is_paper_valid_fidelity_window_source,
    iter_rank_trace_events,
    list_rank_trace_files,
    pattern_tokens_for_rank_trace,
)
from flexsim.maya_lite.markers import (
    TRACE_MARKER_API,
    completed_step_count_from_markers,
    load_step_markers,
    resolve_indexed_step_window_from_marker_trace_timestamps,
    resolve_indexed_step_window_from_markers,
    resolve_indexed_step_window_from_trace_markers,
    resolve_step_window_from_marker_trace_timestamps,
    resolve_step_window_from_markers,
    resolve_step_window_from_trace_markers,
)
from flexsim.maya_lite.planner import plan_profiled_rank_groups, profiled_ranks_for_groups
from flexsim.maya_lite.schema import RankTrace, TraceEvent, TraceSource


_DEVICE_COUNT_RE = re.compile(r'"device_count"\s*:\s*(\d+)')
_FAKECUDA_LIBRARY_NAMES = (
    "libcudart.so.12",
    "libcuda.so.1",
    "libnvidia-ml.so",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libnccl.so.2",
)


@lru_cache(maxsize=None)
def _derive_fakecuda_runtime_env(python_bin: str) -> dict[str, str]:
    resolved_python_bin = Path(python_bin).resolve()
    env_root = resolved_python_bin.parent.parent
    resolved: dict[str, str] = {
        "FAKECUDA_TARGET_ENV_ROOT": str(env_root),
        "FAKECUDA_FRUN_QUIET": "1",
        "FAKECUDA_SKIP_LDCONFIG": "1",
    }

    site_packages_root = env_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if site_packages_root.exists():
        resolved["FAKECUDA_SITE_PACKAGES_ROOT"] = str(site_packages_root)
        runtime_libs = {
            "TARGET_CUDART": site_packages_root / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.12",
            "TARGET_CUBLAS": site_packages_root / "nvidia" / "cublas" / "lib" / "libcublas.so.12",
            "TARGET_CUBLASLT": site_packages_root / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12",
            "TARGET_NCCL": site_packages_root / "nvidia" / "nccl" / "lib" / "libnccl.so.2",
        }
        for key, path in runtime_libs.items():
            if path.exists():
                resolved[key] = str(path)

    for key, candidate in {
        "TARGET_CUDA": Path("/lib/x86_64-linux-gnu/libcuda.so.1"),
        "TARGET_NVML": Path("/lib/x86_64-linux-gnu/libnvidia-ml.so.1"),
    }.items():
        if candidate.exists():
            resolved[key] = str(candidate)

    return resolved


def _existing_path(path: str | Path | None) -> Path | None:
    if path in (None, ""):
        return None
    candidate = Path(path)
    try:
        if candidate.exists():
            return candidate
    except OSError:
        return None
    return None


def _resolved_existing_path(path: str | Path | None) -> str | None:
    candidate = _existing_path(path)
    if candidate is None:
        return None
    try:
        return str(candidate.resolve())
    except OSError:
        return str(candidate)


def _prepend_colon_path(new_entry: str | None, current: str | None) -> str:
    if not new_entry:
        return current or ""
    if not current:
        return new_entry
    entries = [entry for entry in current.split(":") if entry]
    if new_entry in entries:
        return current
    return f"{new_entry}:{current}"


def _maybe_build_direct_proot_command_prefix(
    *,
    frun_path: Path,
    env: Mapping[str, str],
) -> tuple[list[str], str] | None:
    script_dir = frun_path.resolve().parent
    requested_proot = _existing_path(env.get("FAKECUDA_PROOT_BIN"))
    proot_bin = requested_proot
    if proot_bin is None:
        official = _existing_path(script_dir / "proot.official")
        fallback = _existing_path(script_dir / "proot")
        proot_bin = official or fallback
    if proot_bin is None:
        return None

    fake_lib_dir = script_dir / "build" / "liboutput"
    fake_libs = {
        "cudart": fake_lib_dir / "libcudart.so.12",
        "cuda": fake_lib_dir / "libcuda.so.1",
        "nvml": fake_lib_dir / "libnvidia-ml.so",
        "cublas": fake_lib_dir / "libcublas.so.12",
        "cublaslt": fake_lib_dir / "libcublasLt.so.12",
        "nccl": fake_lib_dir / "libnccl.so.2",
    }
    if any(not path.exists() for path in fake_libs.values()):
        return None

    bind_pairs: list[tuple[str, str]] = []
    bound_targets: set[str] = set()
    bound_kinds: set[str] = set()

    def add_bind(
        source: Path,
        target: str | Path | None,
        *,
        bind_kind: str | None = None,
    ) -> None:
        existing = _existing_path(target)
        if existing is None:
            return
        target_str = str(existing)
        if target_str in bound_targets:
            return
        bind_pairs.append((str(source), target_str))
        bound_targets.add(target_str)
        if bind_kind:
            bound_kinds.add(bind_kind)

    def add_bind_group(
        source: Path,
        *targets: str | Path | None,
        bind_kind: str | None = None,
    ) -> None:
        for target in targets:
            add_bind(source, target, bind_kind=bind_kind)

    target_cudart = env.get("TARGET_CUDART")
    target_cuda = env.get("TARGET_CUDA")
    target_nvml = env.get("TARGET_NVML")
    target_cublas = env.get("TARGET_CUBLAS")
    target_cublaslt = env.get("TARGET_CUBLASLT")
    target_nccl = env.get("TARGET_NCCL")

    add_bind_group(
        fake_libs["cudart"],
        target_cudart,
        _resolved_existing_path(target_cudart),
        env.get("TARGET_CUDART_SYS"),
        _resolved_existing_path(env.get("TARGET_CUDART_SYS")),
        "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12",
        "/lib/libcudart.so",
        bind_kind="cudart",
    )
    add_bind_group(
        fake_libs["cuda"],
        target_cuda,
        _resolved_existing_path(target_cuda),
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/local/cuda-12.8/compat/libcuda.so.1",
        "/usr/local/cuda-12.8/compat/libcuda.so",
        "/usr/local/cuda/compat/libcuda.so.1",
        "/usr/local/cuda/compat/libcuda.so",
        bind_kind="cuda",
    )
    add_bind_group(
        fake_libs["nvml"],
        target_nvml,
        _resolved_existing_path(target_nvml),
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        bind_kind="nvml",
    )
    add_bind_group(
        fake_libs["cublas"],
        target_cublas,
        _resolved_existing_path(target_cublas),
        bind_kind="cublas",
    )
    add_bind_group(
        fake_libs["cublaslt"],
        target_cublaslt,
        _resolved_existing_path(target_cublaslt),
        bind_kind="cublaslt",
    )
    add_bind_group(
        fake_libs["nccl"],
        target_nccl,
        _resolved_existing_path(target_nccl),
        bind_kind="nccl",
    )

    if not bind_pairs:
        return None
    if "cuda" not in bound_kinds:
        # direct_proot is only safe when the fake libcuda shim can be bound
        # into the target environment. On CPU-only hosts without any concrete
        # libcuda path, fall back to frun so its bootstrap path can supply the
        # expected CUDA loader contract.
        return None

    proot_extra_lib_dir = (
        _existing_path(script_dir / "third_party" / "proot-runtime" / "usr" / "lib" / "x86_64-linux-gnu")
        or _existing_path(script_dir / "third_party" / "proot-runtime" / "lib" / "x86_64-linux-gnu")
        or _existing_path(env.get("FAKECUDA_PROOT_EXTRA_LIB_DIR"))
    )
    proot_ld_library_path = _prepend_colon_path(
        str(proot_extra_lib_dir) if proot_extra_lib_dir is not None else None,
        env.get("LD_LIBRARY_PATH"),
    )
    # The fake CUDA shims in liboutput depend on libfake_cuda_core.so, which is
    # not installed in the target Python/CUDA environment.  frun prepends this
    # directory itself, but the direct-proot fast path bypasses frun, so keep
    # the shim dependency directory visible to the dynamic loader here too.
    proot_ld_library_path = _prepend_colon_path(str(fake_lib_dir), proot_ld_library_path)

    prefix = [str(proot_bin)]
    for source, target in bind_pairs:
        prefix.extend(["-b", f"{source}:{target}"])
    return prefix, proot_ld_library_path


def _apply_capture_bootstrap_env_defaults(env: dict[str, str]) -> None:
    for key, value in _DEFAULT_CAPTURE_BOOTSTRAP_ENV.items():
        env.setdefault(key, value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_frun() -> Path:
    return _repo_root() / "fake-cuda" / "frun"


def _default_python() -> Path:
    return Path.home() / "miniconda3" / "envs" / "fakecuda-test" / "bin" / "python"


def _fakecuda_liboutput_dir(frun_path: Path) -> Path:
    return frun_path.resolve().parent / "build" / "liboutput"


def _file_fingerprint_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "mtime_ns": None,
            "sha256": None,
        }
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _fakecuda_artifact_fingerprint_metadata(frun_path: Path) -> dict[str, object]:
    frun = frun_path.resolve()
    fakecuda_root = frun.parent
    liboutput_dir = _fakecuda_liboutput_dir(frun)
    libraries = {
        name: _file_fingerprint_metadata(liboutput_dir / name)
        for name in _FAKECUDA_LIBRARY_NAMES
    }
    return {
        "contract_version": "fakecuda_artifact_fingerprint_v1",
        "frun": str(frun),
        "fakecuda_root": str(fakecuda_root),
        "liboutput_dir": str(liboutput_dir),
        "required_libraries_present": all(
            bool(record.get("exists")) for record in libraries.values()
        ),
        "libraries": libraries,
    }


def _default_local_device_span() -> int:
    config_path = _repo_root() / "cpp" / "fake_cuda" / "config" / "config.json"
    if not config_path.exists():
        return 1
    match = _DEVICE_COUNT_RE.search(config_path.read_text(encoding="utf-8"))
    if match is None:
        return 1
    return max(int(match.group(1)), 1)


def _parse_cpu_set(value: str) -> list[int]:
    cpus: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_end = part.split("-", 1)
            start = int(raw_start.strip())
            end = int(raw_end.strip())
            if end < start:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def _online_cpu_ids(path: Path = _CPU_ONLINE_PATH) -> list[int] | None:
    try:
        payload = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        cpus = _parse_cpu_set(payload)
    except ValueError:
        return None
    return cpus or None


def _available_cpu_ids() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_cpu_ids = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        except OSError:
            affinity_cpu_ids = list(range(max(os.cpu_count() or 1, 1)))
    else:
        affinity_cpu_ids = list(range(max(os.cpu_count() or 1, 1)))
    online_cpu_ids = _online_cpu_ids()
    if online_cpu_ids is None:
        return affinity_cpu_ids
    online = set(online_cpu_ids)
    return [cpu for cpu in affinity_cpu_ids if cpu in online]


def _format_cpu_affinity_spec(cpu_ids: list[int]) -> str:
    if not cpu_ids:
        raise ValueError("CPU affinity spec requires at least one CPU")
    ranges: list[str] = []
    start = previous = int(cpu_ids[0])
    for raw_cpu in cpu_ids[1:]:
        cpu = int(raw_cpu)
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = cpu
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(ranges)


def _worker_cpu_affinity_spec(
    *,
    affinity_slot: int,
    cores_per_worker: int,
    available_cpu_ids: list[int] | None = None,
) -> str | None:
    cores = max(0, int(cores_per_worker))
    if cores <= 0:
        return None
    cpus = list(_available_cpu_ids() if available_cpu_ids is None else available_cpu_ids)
    start = max(int(affinity_slot), 0) * cores
    end = start + cores
    selected = cpus[start:end]
    if len(selected) != cores:
        raise RuntimeError(
            "CPU affinity slot exceeds available CPUs: "
            f"slot={affinity_slot}, cores_per_worker={cores}, "
            f"available_cpu_count={len(cpus)}"
        )
    return _format_cpu_affinity_spec(selected)


def _worker_cpu_affinity_manifest_metadata(args: argparse.Namespace) -> dict[str, object | None]:
    cores_per_worker = max(
        0,
        int(getattr(args, "worker_cpu_affinity_cores_per_worker", 0) or 0),
    )
    if cores_per_worker <= 0:
        return {
            "worker_cpu_affinity_cores_per_worker": None,
            "worker_cpu_affinity_available_cpu_count": None,
            "worker_cpu_affinity_topology_basis": None,
        }
    return {
        "worker_cpu_affinity_cores_per_worker": cores_per_worker,
        "worker_cpu_affinity_available_cpu_count": len(_available_cpu_ids()),
        "worker_cpu_affinity_topology_basis": "available_logical_cpu",
    }


def _logical_host_machine_id(logical_rank: int, local_device_span: int) -> str:
    span = max(int(local_device_span), 1)
    return f"logical_host_{int(logical_rank) // span}"


def _default_host_dispatch_queue_id(*, logical_rank: int, host_machine_id: str) -> str:
    # The dispatch queue is the host execution context driving a rank worker.
    # Physical host identity remains separate for communicator/topology inputs.
    return f"{host_machine_id}:rank:{int(logical_rank)}"


def _canonical_host_dispatch_queue_id(
    value: object | None,
    *,
    logical_rank: int,
    host_machine_id: str,
    dispatch_scope: str | None,
) -> str:
    resolved = str(value).strip() if value not in (None, "") else ""
    if resolved:
        return resolved
    return _default_host_dispatch_queue_id(
        logical_rank=logical_rank,
        host_machine_id=host_machine_id,
    )


def _load_rank_host_machines(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rank_host_machines_payload = payload.get("rank_host_machines", {})
    if not isinstance(rank_host_machines_payload, dict):
        return {}
    resolved: dict[int, str] = {}
    for raw_rank, raw_host_machine_id in rank_host_machines_payload.items():
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError):
            continue
        host_machine_id = str(raw_host_machine_id).strip()
        if not host_machine_id:
            continue
        resolved[rank] = host_machine_id
    return resolved


def _load_rank_host_dispatch_queues(
    path: Path | None,
    *,
    rank_host_machines: Mapping[int, str],
    dispatch_scope: str | None,
) -> dict[int, str]:
    resolved: dict[int, str] = {}
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("rank_host_dispatch_queues", {})
        if isinstance(raw, dict):
            for raw_rank, raw_dispatch_queue_id in raw.items():
                try:
                    rank = int(raw_rank)
                except (TypeError, ValueError):
                    continue
                dispatch_queue_id = str(raw_dispatch_queue_id).strip()
                if not dispatch_queue_id:
                    continue
                host_machine_id = rank_host_machines.get(rank)
                if host_machine_id is None:
                    resolved[rank] = dispatch_queue_id
                    continue
                resolved[rank] = _canonical_host_dispatch_queue_id(
                    dispatch_queue_id,
                    logical_rank=rank,
                    host_machine_id=str(host_machine_id),
                    dispatch_scope=dispatch_scope,
                )
    for rank, host_machine_id in sorted(rank_host_machines.items()):
        resolved.setdefault(
            int(rank),
            _default_host_dispatch_queue_id(
                logical_rank=int(rank),
                host_machine_id=str(host_machine_id),
            ),
        )
    return resolved


def _rank_host_machines_for_profiled_groups(
    profiled_rank_groups: dict[int, tuple[int, ...]],
    fallback_rank_host_machines: dict[int, str],
) -> dict[int, str]:
    if not any(len(tuple(members)) > 1 for members in profiled_rank_groups.values()):
        return dict(fallback_rank_host_machines)
    resolved = dict(fallback_rank_host_machines)
    for host_index, representative_rank in enumerate(sorted(profiled_rank_groups)):
        synthetic_host_machine_id = f"logical_profiled_host_{host_index}"
        for member in profiled_rank_groups[representative_rank]:
            resolved[int(member)] = synthetic_host_machine_id
    return resolved


def _parse_profiled_rank_groups(value: str | None) -> dict[int, tuple[int, ...]]:
    if not value:
        return {}
    groups: dict[int, tuple[int, ...]] = {}
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        representative_raw, members_raw = item.split(":", 1)
        representative = int(representative_raw.strip())
        members = tuple(int(member.strip()) for member in members_raw.split(",") if member.strip())
        if not members:
            raise ValueError(f"profiled rank group has no members: {item}")
        groups[representative] = members
    return groups


def _summarize_profiled_capture(
    *,
    logical_world_size: int,
    profiled_rank_groups: dict[int, tuple[int, ...]],
    planning_strategy: str | None,
) -> dict[str, object]:
    # Maya Section 7.4 distinguishes "emulate all workers first" from the
    # hyperscale optimization that selectively launches unique workers.
    covered_logical_ranks = sorted(
        {
            int(member)
            for members in profiled_rank_groups.values()
            for member in members
        }
    )
    full_world_emulation = (
        len(profiled_rank_groups) == logical_world_size
        and all(members == (representative,) for representative, members in profiled_rank_groups.items())
    )
    paper_alignment_mode = "selective_profiled_validation"
    if full_world_emulation:
        paper_alignment_mode = "emulator_full_world_validation"
    elif (planning_strategy or "").strip().lower() == "megatron_pp_stage":
        paper_alignment_mode = "fig13_unique_workers"

    return {
        "profiled_world_size": len(profiled_rank_groups),
        "covered_logical_rank_count": len(covered_logical_ranks),
        "covers_full_logical_world": len(covered_logical_ranks) == logical_world_size,
        "full_world_emulation": full_world_emulation,
        "worker_selection_mode": (
            "full_world"
            if full_world_emulation
            else "selective_profiled"
        ),
        "paper_alignment_mode": paper_alignment_mode,
        "paper_emulator_stage_basis": "capture_elapsed_seconds",
        "diagnostic_active_time_basis": "active_emulator_seconds",
    }


def _capture_elapsed_metadata(
    *,
    capture_elapsed_seconds: float,
    capture_command_elapsed_seconds: float,
    worker_capture_elapsed_seconds: float,
    post_worker_finalize_seconds: float,
) -> dict[str, object]:
    return {
        "capture_elapsed_seconds": float(capture_elapsed_seconds),
        "capture_command_elapsed_seconds": float(capture_command_elapsed_seconds),
        "worker_capture_elapsed_seconds": float(worker_capture_elapsed_seconds),
        "post_worker_finalize_seconds": float(post_worker_finalize_seconds),
        "post_worker_finalize_included_in_capture_elapsed": True,
    }


def _normalize_optional_host_timing_value(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    return normalized or None


def _host_timing_profile_applies(host_timing_mode: str | None) -> bool:
    normalized_mode = _normalize_optional_host_timing_value(host_timing_mode)
    return normalized_mode in {"trace", "sleep"}


def _uses_synthetic_host_timing(host_timing_mode: str | None) -> bool:
    return _normalize_optional_host_timing_value(host_timing_mode) in {"trace", "sleep"}


def _helper_thread_augmentation_expected(
    *,
    host_timing_mode: str | None,
    summary_dir: Path | None,
) -> bool:
    if summary_dir is not None:
        return True
    return _uses_synthetic_host_timing(host_timing_mode)


def _infer_host_timing_summary_dir_from_profile_dir(profile_dir: Path) -> Path:
    name = profile_dir.name
    if name.endswith("_profiles"):
        sibling_name = f"{name[:-len('_profiles')]}_summaries"
    elif name.endswith("profiles"):
        sibling_name = f"{name[:-len('profiles')]}summaries"
    else:
        sibling_name = f"{name}_summaries"
    return profile_dir.parent / sibling_name


def _resolve_host_timing_summary_dir(
    *,
    explicit_summary_dir: Path | None,
    host_timing_profile_dir: Path | None,
    host_timing_profile: Path | None,
) -> Path | None:
    if explicit_summary_dir is not None:
        return explicit_summary_dir.resolve()
    if host_timing_profile_dir is not None:
        return _infer_host_timing_summary_dir_from_profile_dir(
            host_timing_profile_dir.resolve()
        )
    if host_timing_profile is not None:
        return _infer_host_timing_summary_dir_from_profile_dir(
            host_timing_profile.resolve().parent
        )
    return None


def _resolve_host_timing_dispatch_scope(
    *,
    host_timing_mode: str | None,
    requested_dispatch_scope: str | None,
) -> str | None:
    resolved_dispatch_scope = _normalize_optional_host_timing_value(requested_dispatch_scope)
    if resolved_dispatch_scope is not None:
        return resolved_dispatch_scope
    normalized_mode = _normalize_optional_host_timing_value(host_timing_mode)
    if normalized_mode in {"trace", "measure", "sleep"}:
        return "host_machine"
    return None


def _resolve_host_timing_schedule_surface(
    *,
    host_timing_mode: str | None,
    requested_schedule_surface: str | None,
) -> str | None:
    resolved_schedule_surface = _normalize_optional_host_timing_value(requested_schedule_surface)
    if resolved_schedule_surface is not None:
        return resolved_schedule_surface
    normalized_mode = _normalize_optional_host_timing_value(host_timing_mode)
    if normalized_mode in {"trace", "measure", "sleep"}:
        return "semantic"
    return None


def _summarize_host_timing_policy(
    *,
    host_timing_mode: str | None,
    requested_dispatch_scope: str | None,
    resolved_dispatch_scope: str | None,
    requested_schedule_surface: str | None,
    resolved_schedule_surface: str | None,
    host_timing_profile: Path | None,
    host_timing_profile_dir: Path | None,
) -> dict[str, object]:
    normalized_mode = _normalize_optional_host_timing_value(host_timing_mode)
    requested_dispatch_scope_normalized = _normalize_optional_host_timing_value(
        requested_dispatch_scope
    )
    requested_schedule_surface_normalized = _normalize_optional_host_timing_value(
        requested_schedule_surface
    )
    profile_requested = host_timing_profile is not None or host_timing_profile_dir is not None
    profile_backed = profile_requested and _host_timing_profile_applies(normalized_mode)

    paper_alignment_line = "disabled"
    host_timing_line_family = "disabled"
    if normalized_mode == "trace":
        paper_alignment_line = "diagnostic_profile_shaped_host_overhead"
        host_timing_line_family = "synthetic_profile_shaped"
    elif normalized_mode == "measure":
        paper_alignment_line = "direct_emulation_measured_host_overhead"
        host_timing_line_family = "direct_wallclock"
    elif normalized_mode == "sleep":
        paper_alignment_line = "diagnostic_sleep_injected_host_overhead"
        host_timing_line_family = "synthetic_sleep_injected"

    paper_alignment_ready = False
    if normalized_mode == "trace":
        # Profile-shaped host timing is useful for diagnostics and old
        # regression comparison, but the paper describes measuring wall-clock
        # host gaps during emulation.  It is therefore not canonical.
        paper_alignment_ready = False
    elif normalized_mode == "measure":
        paper_alignment_ready = (
            resolved_dispatch_scope == "host_machine"
            and resolved_schedule_surface == "semantic"
        )
    elif normalized_mode == "sleep":
        paper_alignment_ready = False

    return {
        "host_timing_line_contract_version": "phase4_v1",
        "host_timing_mode_resolved": normalized_mode,
        "host_timing_dispatch_scope_requested": requested_dispatch_scope_normalized,
        "host_timing_dispatch_scope_resolved": resolved_dispatch_scope,
        "host_timing_schedule_surface_requested": requested_schedule_surface_normalized,
        "host_timing_schedule_surface_resolved": resolved_schedule_surface,
        "host_timing_dispatch_scope_defaulted": (
            requested_dispatch_scope_normalized is None and resolved_dispatch_scope is not None
        ),
        "host_timing_schedule_surface_defaulted": (
            requested_schedule_surface_normalized is None
            and resolved_schedule_surface is not None
        ),
        "host_timing_profile_requested": profile_requested,
        "host_timing_profile_backed": profile_backed,
        "host_timing_synthetic_shaping": normalized_mode in {"trace", "sleep"},
        "host_timing_paper_alignment_line": paper_alignment_line,
        "host_timing_line_family": host_timing_line_family,
        "host_timing_paper_alignment_ready": paper_alignment_ready,
    }


def _resolve_trace_flush_policy(
    args: argparse.Namespace,
) -> tuple[str, int, int]:
    mode = str(args.trace_flush_mode)
    flush_every = max(int(args.trace_flush_every), 1)
    stdio_buffer_bytes = max(int(args.trace_stdio_buffer_bytes), 0)
    if args.dynamic_first_iteration_dedup and mode == "buffered":
        mode = "per_event"
        flush_every = min(flush_every, DEFAULT_DYNAMIC_DEDUP_TRACE_FLUSH_EVERY)
    return mode, flush_every, stdio_buffer_bytes


@dataclass
class _WorkerProcess:
    representative_rank: int
    local_rank: int
    host_machine_id: str
    host_dispatch_queue_id: str
    process: subprocess.Popen[str]
    stdout_path: Path
    stderr_path: Path
    stdout_handle: object
    stderr_handle: object
    marker_path: Path
    communicator_path: Path
    trace_temp_path: Path
    host_timing_profile: str | None
    host_timing_dispatch_scope: str | None
    host_timing_schedule_surface: str | None
    cpu_affinity_slot: int | None
    cpu_affinity_spec: str | None
    cpu_affinity_cores_per_worker: int | None
    cpu_affinity_topology_basis: str | None
    start_time: float
    start_realtime_ns: int
    first_step_classified: bool = False
    first_step_classified_at: float | None = None
    first_step_tokens: tuple[tuple[object, ...], ...] | None = None
    first_step_token_hash: str | None = None
    first_step_token_count: int | None = None
    first_step_rolling_window: int | None = None
    first_step_rolling_hash_count: int | None = None
    cached_marker_mtime_ns: int | None = None
    cached_marker_records: list[dict[str, object]] | None = None
    cached_communicator_mtime_ns: int | None = None
    cached_communicator_memberships: dict[str, tuple[int, ...]] | None = None
    cached_step_trace_size_bytes: int = -1
    cached_step_tokens: tuple[tuple[object, ...], ...] | None = None
    cached_step_window_key: tuple[int, int] | None = None
    duplicate_of: int | None = None
    termination_reason: str | None = None
    termination_requested_at: float | None = None
    termination_escalated: bool = False


@contextmanager
def _capture_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture blind emulated profiled workers into Maya-lite traces"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logical-world-size", type=int, required=True)
    parser.add_argument(
        "--profiled-rank-groups",
        default=None,
        help="Profiled-rank mapping like '0:0,1;2:2,3'.",
    )
    parser.add_argument(
        "--auto-profiled-strategy",
        choices=["identity", "full_world", "single", "pairwise", "megatron_pp_stage"],
        default=None,
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--pipeline-parallel-size", type=int, default=None)
    parser.add_argument("--frun", type=Path, default=_default_frun())
    parser.add_argument("--python-bin", type=Path, default=_default_python())
    parser.add_argument("--local-device-span", type=int, default=_default_local_device_span())
    parser.add_argument(
        "--rank-host-machines-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file that provides rank_host_machines. When set, "
            "host-machine IDs are taken from that mapping instead of being "
            "inferred from local-device-span."
        ),
    )
    parser.add_argument(
        "--max-concurrent-workers",
        type=int,
        default=DEFAULT_SAFE_CAPTURE_WORKERS,
        help=(
            "Maximum number of profiled workers to emulate in parallel. "
            "The default stays conservative to avoid changing capture semantics "
            "while still exposing real concurrency on cluster runs."
        ),
    )
    parser.add_argument(
        "--worker-cpu-affinity-cores-per-worker",
        type=int,
        default=0,
        help=(
            "If >0, pin each profiled worker to a disjoint CPU core range of "
            "this size using taskset. This avoids cross-worker CPU "
            "contention that would otherwise pollute direct wall-clock "
            "hostDelay during concurrent capture. Set to 0 to disable. "
            "Workers are bound to cores [i*N, (i+1)*N) where i is the "
            "scheduling slot (0..max_concurrent_workers-1)."
        ),
    )
    parser.add_argument(
        "--collective-mode",
        choices=["compat", "trace_only"],
        default="compat",
        help=(
            "How much synthetic tensor behavior the emulated c10d layer provides. "
            "trace_only is the strict Maya-lite mode and only supports "
            "send/recv/all_reduce/barrier/all_to_all/all_to_all_single in the training-step path."
        ),
    )
    parser.add_argument("--master-port", type=int, default=29631)
    parser.add_argument(
        "--trace-flush-mode",
        choices=["per_event", "buffered"],
        default="buffered",
        help="fake-cuda trace logger flush policy used during emulated capture.",
    )
    parser.add_argument(
        "--trace-flush-every",
        type=int,
        default=DEFAULT_TRACE_FLUSH_EVERY,
        help="When trace-flush-mode=buffered, fflush after this many events.",
    )
    parser.add_argument(
        "--trace-stdio-buffer-bytes",
        type=int,
        default=DEFAULT_TRACE_STDIO_BUFFER_BYTES,
        help="stdio buffer size for fake-cuda trace JSONL writes.",
    )
    parser.add_argument(
        "--trace-surface",
        choices=["semantic", "all"],
        default="all",
        help=(
            "fake-cuda trace event surface. The paper-facing default is all: "
            "the device emulator captures the raw API-call stream used to "
            "measure host wall-clock gaps, while downstream semantic "
            "conformance still filters compat-only APIs."
        ),
    )
    parser.add_argument(
        "--host-timing-mode",
        choices=["none", "trace", "measure", "sleep"],
        default="measure",
        help=(
            "fake-cuda host timing mode. The paper-facing default is "
            "measure, which records direct emulation wall-clock deltas in "
            "trace timestamps. trace and sleep are diagnostic modes only."
        ),
    )
    parser.add_argument(
        "--host-timing-dispatch-scope",
        choices=["thread", "process", "host_machine"],
        default=None,
        help=(
            "Optional fake-cuda host timing dispatch scope. When omitted, "
            "trace/measure/sleep default to host_machine scope so the "
            "optional host-timing path matches Maya's single dispatch queue "
            "per host-machine resource model."
        ),
    )
    parser.add_argument(
        "--host-timing-schedule-surface",
        choices=["supported", "semantic"],
        default=None,
        help=(
            "Optional fake-cuda host timing schedule surface. "
            "'semantic' is the paper-facing default and only advances "
            "host-dispatch timing state on semantic-traced APIs while still "
            "recording compat events; 'supported' is a diagnostic surface."
        ),
    )
    parser.add_argument(
        "--host-timing-profile",
        type=Path,
        default=None,
        help=(
            "Optional line-based host timing profile passed to fake-cuda via "
            "FAKECUDA_HOST_TIMING_PROFILE. Format: 'pairocc:<prev_api>-><api>#<occurrence>=<delay_us>', "
            "'pair:<prev_api>-><api>=<delay_us>', "
            "'<api>=<delay_us>', or 'type:<type>=<delay_us>'. "
            "Applied only by trace/sleep host timing modes."
        ),
    )
    parser.add_argument(
        "--host-timing-profile-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of per-rank host timing profiles. "
            "The emulator first looks for rank_<logical_rank>.profile and then "
            "falls back to default.profile. Applied only by trace/sleep modes."
        ),
    )
    parser.add_argument(
        "--host-timing-summary-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of per-rank host timing summaries used to embed "
            "helper/control-plane completion directly into the emulator artifact. "
            "When omitted, the emulator derives a sibling *_summaries directory "
            "from --host-timing-profile-dir or --host-timing-profile when possible."
        ),
    )
    parser.add_argument(
        "--host-timing-default-us",
        type=float,
        default=None,
        help=(
            "Optional default host timing delay in microseconds used when the "
            "profile does not contain a more specific API or type entry. "
            "Applied only by trace/sleep modes."
        ),
    )
    parser.add_argument(
        "--trim-to-step-window",
        action="store_true",
        help="Rewrite each captured rank trace to the resolved step window.",
    )
    parser.add_argument(
        "--capture-step-window-occurrence",
        type=int,
        default=None,
        help=(
            "Optional 1-based training-step occurrence to resolve when trace or "
            "marker windows contain multiple steps. When omitted, capture keeps "
            "the historical whole-trace marker window behavior."
        ),
    )
    parser.add_argument(
        "--capture-step-window-step",
        type=int,
        default=None,
        help=(
            "Optional explicit training-step id to resolve when step markers "
            "carry numeric step metadata. Applies to both trace markers and "
            "marker files."
        ),
    )
    parser.add_argument("--trim-pre-padding-us", type=int, default=0)
    parser.add_argument("--trim-post-padding-us", type=int, default=0)
    parser.add_argument(
        "--capture-lock",
        type=Path,
        default=Path("/tmp/maya_emulated_capture.lock"),
        help="Lock file used to serialize access to shared fake-cuda trace outputs.",
    )
    parser.add_argument(
        "--dynamic-first-iteration-dedup",
        action="store_true",
        help=(
            "Detect duplicate workers from the first completed training-step trace "
            "pattern and terminate redundant workers early. This is the closest "
            "capture-time approximation of Maya's first-iteration dedup path."
        ),
    )
    parser.add_argument(
        "--dynamic-dedup-window-size",
        type=int,
        default=16,
        help="Rolling-hash window size used for first-iteration pattern dedup.",
    )
    parser.add_argument(
        "--dynamic-dedup-poll-interval-ms",
        type=int,
        default=DEFAULT_DYNAMIC_DEDUP_POLL_INTERVAL_MS,
        help="Polling interval for first-iteration dedup marker/trace inspection.",
    )
    parser.add_argument("script", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser


def _resolved_profiled_rank_groups(args: argparse.Namespace) -> dict[int, tuple[int, ...]]:
    groups = _parse_profiled_rank_groups(args.profiled_rank_groups)
    if groups:
        return groups
    if args.auto_profiled_strategy is None:
        raise SystemExit("either --profiled-rank-groups or --auto-profiled-strategy is required")
    return plan_profiled_rank_groups(
        args.logical_world_size,
        strategy=args.auto_profiled_strategy,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )


def _cleanup_fakecuda_traces() -> None:
    for path in Path("/tmp").glob("fakecuda_trace_*.jsonl"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _merge_pythonpath(*entries: str, current: str | None) -> str:
    merged: list[str] = []
    for entry in entries:
        if entry and entry not in merged:
            merged.append(entry)
    for entry in (current or "").split(":"):
        if entry and entry not in merged:
            merged.append(entry)
    return ":".join(merged)


def _trace_temp_path(output_dir: Path, representative_rank: int) -> Path:
    return output_dir / f"rank_{representative_rank}.raw.jsonl"


def _trace_final_path(output_dir: Path, representative_rank: int) -> Path:
    return output_dir / f"rank_{representative_rank}.jsonl"


def _finalize_worker_trace(*, output_dir: Path, representative_rank: int) -> Path:
    temp_path = _trace_temp_path(output_dir, representative_rank)
    destination = _trace_final_path(output_dir, representative_rank)
    if not temp_path.exists():
        raise RuntimeError(
            f"no fake-cuda worker trace captured for profiled rank {representative_rank}"
        )
    shutil.move(str(temp_path), destination)
    return destination


def _marker_path(output_dir: Path, representative_rank: int) -> Path:
    return output_dir / f"rank_{representative_rank}.markers.jsonl"


def _communicator_path(output_dir: Path, representative_rank: int) -> Path:
    return output_dir / f"rank_{representative_rank}.communicators.json"


def _load_worker_communicators(
    path: Path,
    *,
    allow_incomplete: bool = False,
) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        if allow_incomplete:
            return {}
        raise
    raw = payload.get("communicators", {})
    if not isinstance(raw, dict):
        return {}
    resolved: dict[str, dict[str, object]] = {}
    for comm_id, record in raw.items():
        if not isinstance(record, dict):
            continue
        members = record.get("members", [])
        resolved[str(comm_id)] = {
            "members": [int(member) for member in members],
            "size": int(record.get("size", len(members))),
            "name": str(record.get("name", "")),
            "backend": str(record.get("backend", "")),
            "source": str(record.get("source", "emulated_dist")),
        }
    return resolved


def _communicator_memberships_from_records(
    records: Mapping[str, Mapping[str, object]],
) -> dict[str, tuple[int, ...]]:
    return {
        str(comm_id): tuple(int(member) for member in record.get("members", []))
        for comm_id, record in records.items()
    }


def _capture_worker_command_and_env(
    *,
    args: argparse.Namespace,
    profiled_index: int,
    representative_rank: int,
    rank_host_machines: dict[int, str],
    rank_host_dispatch_queues: dict[int, str],
    script_args: list[str],
    repo_root: str,
    python_root: str,
    output_dir: Path,
) -> tuple[
    list[str],
    dict[str, str],
    Path,
    Path,
    Path,
    Path,
    str | None,
    str | None,
    str | None,
    int,
    str,
    str,
]:
    local_rank = profiled_index % max(args.local_device_span, 1)
    host_machine_id = rank_host_machines.get(
        representative_rank,
        _logical_host_machine_id(
            representative_rank,
            args.local_device_span,
        ),
    )
    host_dispatch_queue_id = rank_host_dispatch_queues.get(
        representative_rank,
        _default_host_dispatch_queue_id(
            logical_rank=representative_rank,
            host_machine_id=host_machine_id,
        ),
    )
    marker_path = _marker_path(output_dir, representative_rank)
    communicator_path = _communicator_path(output_dir, representative_rank)
    trace_temp_path = _trace_temp_path(output_dir, representative_rank)
    marker_path.unlink(missing_ok=True)
    communicator_path.unlink(missing_ok=True)
    trace_temp_path.unlink(missing_ok=True)
    frun_path = args.frun.resolve()
    command = [
        str(frun_path),
        str(args.python_bin.resolve()),
        str(args.script.resolve()),
        *script_args,
    ]
    trace_flush_mode, trace_flush_every, trace_stdio_buffer_bytes = _resolve_trace_flush_policy(args)
    env = os.environ.copy()
    env.update(_derive_fakecuda_runtime_env(str(args.python_bin)))
    _apply_capture_bootstrap_env_defaults(env)
    env["FAKECUDA_TRACE"] = "1"
    env["FAKECUDA_TRACE_PATH"] = str(trace_temp_path)
    env["FLEXSIM_MAYA_EMULATED_DIST"] = "1"
    env["FLEXSIM_MAYA_LOGICAL_RANK"] = str(representative_rank)
    env["FLEXSIM_MAYA_LOGICAL_WORLD_SIZE"] = str(args.logical_world_size)
    env["FLEXSIM_MAYA_BACKEND"] = "nccl"
    env["FLEXSIM_MAYA_COLLECTIVE_MODE"] = args.collective_mode
    env["FAKECUDA_TRACE_FLUSH_MODE"] = trace_flush_mode
    env["FAKECUDA_TRACE_FLUSH_EVERY"] = str(trace_flush_every)
    env["FAKECUDA_TRACE_STDIO_BUFFER_BYTES"] = str(trace_stdio_buffer_bytes)
    env["FAKECUDA_TRACE_SURFACE"] = str(args.trace_surface)
    if args.host_timing_mode is not None:
        env["FAKECUDA_HOST_TIMING_MODE"] = args.host_timing_mode
    resolved_dispatch_scope = _resolve_host_timing_dispatch_scope(
        host_timing_mode=args.host_timing_mode,
        requested_dispatch_scope=args.host_timing_dispatch_scope,
    )
    if resolved_dispatch_scope is not None:
        env["FAKECUDA_HOST_TIMING_DISPATCH_SCOPE"] = resolved_dispatch_scope
    resolved_schedule_surface = _resolve_host_timing_schedule_surface(
        host_timing_mode=args.host_timing_mode,
        requested_schedule_surface=args.host_timing_schedule_surface,
    )
    if resolved_schedule_surface is not None:
        env["FAKECUDA_HOST_TIMING_SCHEDULE_SURFACE"] = resolved_schedule_surface
    resolved_host_timing_profile: Path | None = None
    if args.host_timing_profile_dir is not None:
        profile_dir = args.host_timing_profile_dir.resolve()
        rank_profile = profile_dir / f"rank_{representative_rank}.profile"
        default_profile = profile_dir / "default.profile"
        if rank_profile.exists():
            resolved_host_timing_profile = rank_profile
        elif default_profile.exists():
            resolved_host_timing_profile = default_profile
    if resolved_host_timing_profile is None and args.host_timing_profile is not None:
        resolved_host_timing_profile = args.host_timing_profile.resolve()
    if resolved_host_timing_profile is not None and _host_timing_profile_applies(args.host_timing_mode):
        env["FAKECUDA_HOST_TIMING_PROFILE"] = str(resolved_host_timing_profile)
    if args.host_timing_default_us is not None and _host_timing_profile_applies(args.host_timing_mode):
        env["FAKECUDA_HOST_TIMING_DEFAULT_US"] = str(args.host_timing_default_us)
    env["PYTHONPATH"] = _merge_pythonpath(
        repo_root,
        python_root,
        current=env.get("PYTHONPATH"),
    )
    env["RANK"] = str(representative_rank)
    env["WORLD_SIZE"] = str(args.logical_world_size)
    env["LOCAL_RANK"] = str(local_rank)
    env["FLEXSIM_HOST_MACHINE_ID"] = host_machine_id
    env["FLEXSIM_HOST_DISPATCH_QUEUE_ID"] = host_dispatch_queue_id
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(args.master_port)
    env["FLEXSIM_MAYA_MARKERS_PATH"] = str(marker_path)
    env["FLEXSIM_MAYA_COMMUNICATORS_PATH"] = str(communicator_path)
    bootstrap_diag_root = env.get("FLEXSIM_MAYA_BOOTSTRAP_DIAG_DIR")
    if bootstrap_diag_root:
        bootstrap_diag_dir = Path(bootstrap_diag_root)
        bootstrap_diag_dir.mkdir(parents=True, exist_ok=True)
        env["FLEXSIM_MAYA_BOOTSTRAP_DIAG_PATH"] = str(
            bootstrap_diag_dir / f"rank_{representative_rank}.bootstrap.json"
        )

    direct_proot_prefix = _maybe_build_direct_proot_command_prefix(
        frun_path=frun_path,
        env=env,
    )
    if direct_proot_prefix is not None:
        prefix, proot_ld_library_path = direct_proot_prefix
        command = [
            *prefix,
            str(args.python_bin.resolve()),
            str(args.script.resolve()),
            *script_args,
        ]
        env["LD_LIBRARY_PATH"] = proot_ld_library_path
        env["FLEXSIM_MAYA_CAPTURE_LAUNCHER"] = "direct_proot"
    else:
        env["FLEXSIM_MAYA_CAPTURE_LAUNCHER"] = "frun"
    return (
        command,
        env,
        marker_path,
        communicator_path,
        trace_temp_path,
        _trace_final_path(output_dir, representative_rank),
        resolved_dispatch_scope,
        resolved_schedule_surface,
        str(resolved_host_timing_profile) if resolved_host_timing_profile is not None else None,
        local_rank,
        host_machine_id,
        host_dispatch_queue_id,
    )


def _step_window_pattern_tokens(
    *,
    trace_path: Path,
    representative_rank: int,
    step_window: dict[str, int | str],
    communicator_path: Path,
    communicator_memberships: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[tuple[object, ...], ...] | None:
    if not trace_path.exists():
        return None
    start_ts = int(step_window["start_ts"])
    end_ts = int(step_window["end_ts"])
    events_list = []
    max_ts: int | None = None
    with trace_path.open("r", encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            payload = line.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                # Dynamic first-iteration dedup reads rank_*.raw.jsonl while the
                # worker may still be flushing its last buffered line. Skip an
                # incomplete raw-trace record instead of failing the whole
                # classification pass. Finalized rank_*.jsonl traces stay strict.
                if trace_path.name.endswith(".raw.jsonl"):
                    continue
                raise
            event = TraceEvent.from_json_record(
                record,
                rank=representative_rank,
                ordinal=ordinal,
                source=TraceSource.FAKE,
            )
            event_end_ts = _effective_event_end_ts(event)
            if max_ts is None or event_end_ts > max_ts:
                max_ts = event_end_ts
            if event.ts <= end_ts and event_end_ts >= start_ts:
                events_list.append(event)
    if max_ts is None or max_ts < end_ts:
        return None
    events = tuple(events_list)
    if not events:
        return None
    rank_trace = RankTrace(
        rank=representative_rank,
        path=trace_path,
        source=TraceSource.FAKE,
        events=events,
    )
    if communicator_memberships is None:
        communicator_memberships = _communicator_memberships_from_records(
            _load_worker_communicators(communicator_path)
        )
    return pattern_tokens_for_rank_trace(
        rank_trace,
        communicator_memberships=communicator_memberships,
    )


def _canonicalize_dynamic_dedup_tokens(
    tokens: tuple[tuple[object, ...], ...]
) -> tuple[tuple[object, ...], ...]:
    """Canonical first-step operation sequence for Maya runtime dedup."""
    canonical: list[tuple[object, ...]] = []
    for token in tokens:
        if not token:
            continue
        api = str(token[0])
        if api == TRACE_MARKER_API or api in _DYNAMIC_DEDUP_VOLATILE_APIS:
            continue
        op_type = token[1] if len(token) > 1 else ""
        extras = token[2] if len(token) > 2 and isinstance(token[2], tuple) else ()
        filtered_extras = tuple(
            (key, value)
            for key, value in extras
            if key not in {"peer_role"}
        )
        canonical.append((api, op_type, filtered_extras))
    return tuple(canonical)


def _dynamic_dedup_rolling_hash_count(
    token_count: int,
    *,
    window_size: int = DEFAULT_DYNAMIC_DEDUP_ROLLING_WINDOW,
) -> tuple[int, int]:
    width = min(int(window_size), int(token_count)) if token_count > 0 else 0
    count = max(0, int(token_count) - width + 1) if width else 0
    return width, count


def _dynamic_dedup_sequence_hash(
    tokens: tuple[tuple[object, ...], ...]
) -> tuple[str, int, int, int]:
    canonical = _canonicalize_dynamic_dedup_tokens(tokens)
    digest = hashlib.blake2b(
        repr(canonical).encode("utf-8"),
        digest_size=16,
    ).hexdigest()
    width, rolling_count = _dynamic_dedup_rolling_hash_count(len(canonical))
    return digest, len(canonical), width, rolling_count


def _prefix_pattern_tokens(
    *,
    trace_path: Path,
    representative_rank: int,
    marker_path: Path,
    communicator_path: Path,
    prefix_token_count: int = DEFAULT_DYNAMIC_DEDUP_PREFIX_TOKENS,
    marker_records: list[dict[str, object]] | None = None,
    communicator_memberships: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[tuple[object, ...], ...] | None:
    if prefix_token_count <= 0 or not trace_path.exists():
        return None

    prefix_start_ts: int | None = None
    if marker_records is None:
        marker_records = load_step_markers(marker_path)
    for record in marker_records:
        if record.get("kind") != "step_begin" or record.get("label", "training_step") != "training_step":
            continue
        start_ts = record.get("trace_ts")
        if start_ts not in (None, ""):
            prefix_start_ts = int(start_ts)
            break
        start_ns = record.get("realtime_ns")
        if start_ns not in (None, ""):
            prefix_start_ts = int(int(start_ns) / 1_000)
        break
    if prefix_start_ts is None:
        return None

    if communicator_memberships is None:
        communicator_memberships = _communicator_memberships_from_records(
            _load_worker_communicators(communicator_path)
        )
    tokens: list[tuple[object, ...]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            payload = line.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                if trace_path.name.endswith(".raw.jsonl"):
                    continue
                raise
            event = TraceEvent.from_json_record(
                record,
                rank=representative_rank,
                ordinal=ordinal,
                source=TraceSource.FAKE,
            )
            event_end_ts = _effective_event_end_ts(event)
            if event.ts < prefix_start_ts and event_end_ts < prefix_start_ts:
                continue
            if event.api == TRACE_MARKER_API:
                continue
            token = _pattern_signature_for_event(
                event,
                communicator_memberships=communicator_memberships,
            )
            if token is None:
                continue
            tokens.append(token)
            if len(tokens) >= prefix_token_count:
                return tuple(tokens)
    return None


def _cached_marker_records(worker: _WorkerProcess) -> list[dict[str, object]]:
    try:
        marker_stat = worker.marker_path.stat()
    except FileNotFoundError:
        worker.cached_marker_mtime_ns = None
        worker.cached_marker_records = []
        return []
    marker_mtime_ns = marker_stat.st_mtime_ns
    if (
        worker.cached_marker_records is not None
        and worker.cached_marker_mtime_ns == marker_mtime_ns
    ):
        return worker.cached_marker_records
    marker_records = load_step_markers(worker.marker_path)
    worker.cached_marker_mtime_ns = marker_mtime_ns
    worker.cached_marker_records = marker_records
    return marker_records


def _cached_communicator_memberships(worker: _WorkerProcess) -> dict[str, tuple[int, ...]]:
    try:
        communicator_stat = worker.communicator_path.stat()
    except FileNotFoundError:
        worker.cached_communicator_mtime_ns = None
        worker.cached_communicator_memberships = {}
        return {}
    communicator_mtime_ns = communicator_stat.st_mtime_ns
    if (
        worker.cached_communicator_memberships is not None
        and worker.cached_communicator_mtime_ns == communicator_mtime_ns
    ):
        return worker.cached_communicator_memberships
    memberships = _communicator_memberships_from_records(
        _load_worker_communicators(worker.communicator_path, allow_incomplete=True)
    )
    worker.cached_communicator_mtime_ns = communicator_mtime_ns
    worker.cached_communicator_memberships = memberships
    return memberships


def _cached_prefix_pattern_tokens(worker: _WorkerProcess) -> tuple[tuple[object, ...], ...] | None:
    try:
        trace_stat = worker.trace_temp_path.stat()
    except FileNotFoundError:
        worker.cached_prefix_trace_size_bytes = -1
        worker.cached_prefix_tokens = None
        return None
    trace_size = trace_stat.st_size
    if worker.cached_prefix_trace_size_bytes == trace_size:
        return worker.cached_prefix_tokens
    tokens = _prefix_pattern_tokens(
        trace_path=worker.trace_temp_path,
        representative_rank=worker.representative_rank,
        marker_path=worker.marker_path,
        communicator_path=worker.communicator_path,
        marker_records=_cached_marker_records(worker),
        communicator_memberships=_cached_communicator_memberships(worker),
    )
    worker.cached_prefix_trace_size_bytes = trace_size
    worker.cached_prefix_tokens = tokens
    return tokens


def _classify_worker_pattern_tokens(
    worker: _WorkerProcess,
    *,
    tokens: tuple[tuple[object, ...], ...],
    hash_to_representative: dict[str, int],
    dynamic_rank_groups: dict[int, list[int]],
) -> None:
    worker.first_step_classified = True
    if worker.first_step_classified_at is None:
        worker.first_step_classified_at = time.perf_counter()
    worker.first_step_tokens = tokens
    token_hash, token_count, rolling_window, rolling_hash_count = _dynamic_dedup_sequence_hash(tokens)
    worker.first_step_token_hash = token_hash
    worker.first_step_token_count = token_count
    worker.first_step_rolling_window = rolling_window
    worker.first_step_rolling_hash_count = rolling_hash_count
    duplicate_of = hash_to_representative.get(token_hash)
    if duplicate_of is None:
        hash_to_representative[token_hash] = worker.representative_rank
        dynamic_rank_groups.setdefault(worker.representative_rank, [worker.representative_rank])
        return
    if duplicate_of == worker.representative_rank:
        dynamic_rank_groups.setdefault(worker.representative_rank, [worker.representative_rank])
        return
    worker.duplicate_of = duplicate_of
    members = dynamic_rank_groups.setdefault(duplicate_of, [duplicate_of])
    if worker.representative_rank not in members:
        members.append(worker.representative_rank)
    if worker.process.poll() is None:
        _request_dynamic_dedup_termination(worker)


def _signal_worker_process_tree(
    process: subprocess.Popen[str] | object,
    *,
    signum: int,
) -> None:
    pid = getattr(process, "pid", None)
    if pid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(int(pid), signum)
            return
        except ProcessLookupError:
            return
        except (OSError, PermissionError, ValueError):
            pass

    if signum == signal.SIGTERM:
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            terminate()
            return
    kill = getattr(process, "kill", None)
    if callable(kill):
        kill()


def _request_dynamic_dedup_termination(worker: _WorkerProcess) -> None:
    if worker.termination_requested_at is None:
        worker.termination_requested_at = time.perf_counter()
    worker.termination_reason = "dynamic_first_iteration_dedup"
    _signal_worker_process_tree(
        worker.process,
        signum=signal.SIGTERM,
    )


def _maybe_escalate_dynamic_dedup_termination(
    worker: _WorkerProcess,
    *,
    grace_seconds: float,
) -> None:
    if worker.termination_reason != "dynamic_first_iteration_dedup":
        return
    if worker.termination_requested_at is None:
        return
    if worker.termination_escalated:
        return
    if worker.process.poll() is not None:
        return
    if (time.perf_counter() - worker.termination_requested_at) < grace_seconds:
        return
    _signal_worker_process_tree(
        worker.process,
        signum=signal.SIGKILL,
    )
    worker.termination_escalated = True


def _launch_blocking_worker_count(
    active_workers: Mapping[int, _WorkerProcess],
) -> int:
    launch_blocking = 0
    for worker in active_workers.values():
        if (
            worker.duplicate_of is not None
            and worker.termination_reason == "dynamic_first_iteration_dedup"
            and worker.termination_requested_at is not None
        ):
            continue
        launch_blocking += 1
    return launch_blocking


def _start_capture_worker_process(
    *,
    args: argparse.Namespace,
    profiled_index: int,
    representative_rank: int,
    rank_host_machines: dict[int, str],
    rank_host_dispatch_queues: dict[int, str],
    script_args: list[str],
    repo_root: str,
    python_root: str,
    output_dir: Path,
    affinity_slot: int = 0,
) -> _WorkerProcess:
    (
        command,
        env,
        marker_path,
        communicator_path,
        trace_temp_path,
        _final_trace_path,
        resolved_host_timing_dispatch_scope,
        resolved_host_timing_schedule_surface,
        resolved_host_timing_profile,
        local_rank,
        host_machine_id,
        host_dispatch_queue_id,
    ) = _capture_worker_command_and_env(
        args=args,
        profiled_index=profiled_index,
        representative_rank=representative_rank,
        rank_host_machines=rank_host_machines,
        rank_host_dispatch_queues=rank_host_dispatch_queues,
        script_args=script_args,
        repo_root=repo_root,
        python_root=python_root,
        output_dir=output_dir,
    )
    cores_per_worker = max(0, int(getattr(args, "worker_cpu_affinity_cores_per_worker", 0) or 0))
    cpu_affinity_spec = _worker_cpu_affinity_spec(
        affinity_slot=affinity_slot,
        cores_per_worker=cores_per_worker,
    )
    cpu_affinity_topology_basis = "available_logical_cpu" if cpu_affinity_spec else None
    if cores_per_worker > 0:
        assert cpu_affinity_spec is not None
        command = ["taskset", "-c", cpu_affinity_spec, *command]
    stdout_path = output_dir / f"rank_{representative_rank}.stdout.txt"
    stderr_path = output_dir / f"rank_{representative_rank}.stderr.txt"
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(_repo_root()),
        env=env,
        stdout=stdout_handle,
        stderr=stderr_handle,
        start_new_session=True,
        text=True,
    )
    return _WorkerProcess(
        representative_rank=representative_rank,
        local_rank=local_rank,
        host_machine_id=host_machine_id,
        host_dispatch_queue_id=host_dispatch_queue_id,
        process=process,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        marker_path=marker_path,
        communicator_path=communicator_path,
        trace_temp_path=trace_temp_path,
        host_timing_profile=resolved_host_timing_profile,
        host_timing_dispatch_scope=resolved_host_timing_dispatch_scope,
        host_timing_schedule_surface=resolved_host_timing_schedule_surface,
        cpu_affinity_slot=int(affinity_slot) if cores_per_worker > 0 else None,
        cpu_affinity_spec=cpu_affinity_spec,
        cpu_affinity_cores_per_worker=cores_per_worker if cores_per_worker > 0 else None,
        cpu_affinity_topology_basis=cpu_affinity_topology_basis if cores_per_worker > 0 else None,
        start_time=time.perf_counter(),
        start_realtime_ns=time.time_ns(),
    )


def _run_capture_worker(
    *,
    args: argparse.Namespace,
    profiled_index: int,
    representative_rank: int,
    rank_host_machines: dict[int, str],
    rank_host_dispatch_queues: dict[int, str],
    script_args: list[str],
    repo_root: str,
    python_root: str,
    output_dir: Path,
    affinity_slot: int = 0,
) -> dict[str, object]:
    (
        command,
        env,
        marker_path,
        communicator_path,
        _trace_temp_path,
        _final_trace_path,
        resolved_host_timing_dispatch_scope,
        resolved_host_timing_schedule_surface,
        resolved_host_timing_profile,
        local_rank,
        host_machine_id,
        host_dispatch_queue_id,
    ) = _capture_worker_command_and_env(
        args=args,
        profiled_index=profiled_index,
        representative_rank=representative_rank,
        rank_host_machines=rank_host_machines,
        rank_host_dispatch_queues=rank_host_dispatch_queues,
        script_args=script_args,
        repo_root=repo_root,
        python_root=python_root,
        output_dir=output_dir,
    )
    cores_per_worker = max(0, int(getattr(args, "worker_cpu_affinity_cores_per_worker", 0) or 0))
    cpu_affinity_spec = _worker_cpu_affinity_spec(
        affinity_slot=affinity_slot,
        cores_per_worker=cores_per_worker,
    )
    cpu_affinity_topology_basis = "available_logical_cpu" if cpu_affinity_spec else None
    if cores_per_worker > 0:
        assert cpu_affinity_spec is not None
        command = ["taskset", "-c", cpu_affinity_spec, *command]
    worker_start = time.perf_counter()
    worker_start_realtime_ns = time.time_ns()
    proc = subprocess.run(
        command,
        cwd=str(_repo_root()),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    worker_elapsed = time.perf_counter() - worker_start
    worker_end_realtime_ns = time.time_ns()
    return {
        "representative_rank": representative_rank,
        "local_rank": local_rank,
        "launcher": "direct_sitecustomize",
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed_seconds": worker_elapsed,
        "start_realtime_ns": worker_start_realtime_ns,
        "end_realtime_ns": worker_end_realtime_ns,
        "host_machine_id": host_machine_id,
        "host_dispatch_queue_id": host_dispatch_queue_id,
        "marker_path": marker_path,
        "communicator_path": communicator_path,
        "host_timing_profile": (
            str(resolved_host_timing_profile)
            if resolved_host_timing_profile is not None
            else None
        ),
        "host_timing_dispatch_scope": resolved_host_timing_dispatch_scope,
        "host_timing_schedule_surface": resolved_host_timing_schedule_surface,
        "cpu_affinity_slot": int(affinity_slot) if cores_per_worker > 0 else None,
        "cpu_affinity_spec": cpu_affinity_spec,
        "cpu_affinity_cores_per_worker": cores_per_worker if cores_per_worker > 0 else None,
        "cpu_affinity_topology_basis": cpu_affinity_topology_basis if cores_per_worker > 0 else None,
    }


def _trim_trace_file_to_ts_window(
    trace_path: Path,
    *,
    start_ts: int,
    end_ts: int,
    pre_padding_us: int = 0,
    post_padding_us: int = 0,
) -> dict[str, int]:
    trimmed_start = int(start_ts) - max(int(pre_padding_us), 0)
    trimmed_end = int(end_ts) + max(int(post_padding_us), 0)
    temp_path = trace_path.with_suffix(".jsonl.trimmed")
    total_events = 0
    kept_events = 0

    with trace_path.open("r", encoding="utf-8") as source, temp_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            payload = line.strip()
            if not payload:
                continue
            record = json.loads(payload)
            if "ts" not in record:
                raise ValueError(f"trace record missing ts in {trace_path}:{line_number}")
            total_events += 1
            api = str(record.get("api", ""))
            if api == TRACE_MARKER_API or is_teardown_api(api):
                continue
            event_ts = int(record["ts"])
            event_end_ts = _coerce_event_end_ts(record.get("end_ts"), event_ts)
            if event_ts > trimmed_end or event_end_ts < trimmed_start:
                continue
            target.write(line)
            kept_events += 1

    if kept_events <= 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"step-window trim kept no events for {trace_path} "
            f"within [{trimmed_start}, {trimmed_end}]"
        )

    shutil.move(str(temp_path), trace_path)
    return {
        "start_ts": trimmed_start,
        "end_ts": trimmed_end,
        "total_events": total_events,
        "kept_events": kept_events,
    }


def _resolve_capture_step_window(
    trace_path: Path,
    *,
    marker_records: list[dict[str, object]],
    host_timing_mode: str | None,
    capture_step_window_occurrence: int | None = None,
    capture_step_window_step: int | None = None,
) -> dict[str, int | str] | None:
    normalized_host_timing_mode = (host_timing_mode or "").strip().lower()
    use_trace_marker_window = normalized_host_timing_mode in {"trace", "measure"}
    indexed_capture_window_requested = (
        capture_step_window_occurrence is not None or capture_step_window_step is not None
    )
    resolved_occurrence = int(capture_step_window_occurrence or 1)
    trace_marker_window = None
    if use_trace_marker_window:
        if indexed_capture_window_requested:
            trace_marker_window = resolve_indexed_step_window_from_marker_trace_timestamps(
                marker_records,
                occurrence=resolved_occurrence,
                step=capture_step_window_step,
            )
        else:
            trace_marker_window = resolve_step_window_from_marker_trace_timestamps(
                marker_records,
            )
    if trace_marker_window is None:
        if indexed_capture_window_requested:
            trace_marker_window = resolve_indexed_step_window_from_trace_markers(
                trace_path,
                occurrence=resolved_occurrence,
                step=capture_step_window_step,
            )
        else:
            trace_marker_window = resolve_step_window_from_trace_markers(trace_path)
    if use_trace_marker_window and trace_marker_window is not None:
        resolved_step_window = trace_marker_window
    else:
        if indexed_capture_window_requested:
            resolved_step_window = resolve_indexed_step_window_from_markers(
                marker_records,
                source=TraceSource.FAKE,
                occurrence=resolved_occurrence,
                step=capture_step_window_step,
            )
            if resolved_step_window is not None:
                resolved_step_window = {
                    **resolved_step_window,
                    "source": "trace_markers",
                }
        else:
            resolved_step_window = resolve_step_window_from_markers(
                marker_records,
                source=TraceSource.FAKE,
            )
            if resolved_step_window is not None:
                resolved_step_window = {
                    **resolved_step_window,
                    "source": "trace_markers",
                }
    if indexed_capture_window_requested and resolved_step_window is None:
        raise RuntimeError(
            "requested capture step window could not be resolved for "
            f"{trace_path} (occurrence={resolved_occurrence}, "
            f"step={capture_step_window_step})"
        )
    if resolved_step_window is None:
        return estimate_rank_trace_window(trace_path)
    return resolved_step_window


def _resolve_workload_heuristic_step_window(
    trace_path: Path,
    *,
    resolved_step_window: dict[str, int | str] | None,
    host_timing_mode: str | None,
    min_extension_us: int = DEFAULT_WORKLOAD_HEURISTIC_MIN_EXTENSION_US,
    min_supported_tail_events: int = DEFAULT_WORKLOAD_HEURISTIC_MIN_SUPPORTED_TAIL_EVENTS,
    gap_cutoff_us: int = DEFAULT_WORKLOAD_HEURISTIC_GAP_CUTOFF_US,
    min_gap_candidate_semantic_events: int = DEFAULT_WORKLOAD_HEURISTIC_MIN_GAP_CANDIDATE_SEMANTIC_EVENTS,
) -> dict[str, int | str] | None:
    if resolved_step_window is None:
        return None

    normalized_mode = (host_timing_mode or "").strip().lower()
    if normalized_mode == "sleep":
        return None

    base_source = str(resolved_step_window.get("source") or "")
    if base_source != "trace_markers":
        return None

    base_start_ts = int(resolved_step_window["start_ts"])
    base_end_ts = int(resolved_step_window["end_ts"])
    last_relevant_end_ts = base_end_ts
    gap_cutoff_end_ts: int | None = None
    supported_tail_event_count = 0
    semantic_tail_event_count = 0
    previous_relevant_end_ts = base_end_ts

    for event in iter_rank_trace_events(trace_path, source=TraceSource.FAKE):
        event_end_ts = _effective_event_end_ts(event)
        if event_end_ts <= base_end_ts:
            continue
        if event.api == TRACE_MARKER_API:
            continue
        if is_ignorable_setup_api(event.api) or is_teardown_api(event.api):
            continue
        if not is_supported_trace_api(event.api, event.op_type):
            continue
        gap_us = event_end_ts - previous_relevant_end_ts
        supported_tail_event_count += 1
        if is_semantic_traced_api(event.api, event.op_type):
            semantic_tail_event_count += 1
        if (
            int(gap_cutoff_us) > 0
            and gap_us >= int(gap_cutoff_us)
            and semantic_tail_event_count >= max(int(min_gap_candidate_semantic_events), 0)
        ):
            gap_cutoff_end_ts = previous_relevant_end_ts
        if event_end_ts > last_relevant_end_ts:
            last_relevant_end_ts = event_end_ts
        previous_relevant_end_ts = event_end_ts

    selected_end_ts = last_relevant_end_ts
    heuristic_name = "post_step_supported_tail"
    if gap_cutoff_end_ts is not None and gap_cutoff_end_ts > base_end_ts:
        selected_end_ts = gap_cutoff_end_ts
        heuristic_name = "post_step_supported_tail_gap_cutoff"

    extension_us = selected_end_ts - base_end_ts
    if extension_us < max(int(min_extension_us), 0):
        return None
    if semantic_tail_event_count <= 0 and supported_tail_event_count < max(
        int(min_supported_tail_events), 0
    ):
        return None

    heuristic_window = dict(resolved_step_window)
    heuristic_window["start_ts"] = base_start_ts
    heuristic_window["end_ts"] = selected_end_ts
    heuristic_window["source"] = "workload_heuristic"
    heuristic_window["base_source"] = base_source
    heuristic_window["base_end_ts"] = base_end_ts
    heuristic_window["heuristic_name"] = heuristic_name
    heuristic_window["post_step_supported_event_count"] = supported_tail_event_count
    heuristic_window["post_step_semantic_event_count"] = semantic_tail_event_count
    if gap_cutoff_end_ts is not None and gap_cutoff_end_ts > base_end_ts:
        heuristic_window["gap_cutoff_us"] = int(gap_cutoff_us)
        heuristic_window["gap_cutoff_candidate_end_ts"] = gap_cutoff_end_ts
        heuristic_window["gap_cutoff_min_semantic_events"] = int(min_gap_candidate_semantic_events)
    return heuristic_window


def _should_apply_default_workload_heuristic_step_window(
    *,
    args: argparse.Namespace,
    resolved_step_window: dict[str, int | str] | None,
) -> bool:
    """Return whether capture should spend time on post-step heuristic expansion.

    Canonical Figure 13 routes should keep a clean marker-resolved step window
    exact by default. Tail-extension heuristics remain available for explicit
    trace-trimming / diagnostic paths, but should not run on the default
    paper-facing capture line where they both add cost and widen the fidelity
    envelope beyond the measured step.
    """
    if resolved_step_window is None:
        return False
    if not bool(getattr(args, "trim_to_step_window", False)):
        return False
    return True


def _first_and_last_training_step_marker_ns(
    marker_records: list[dict[str, object]],
    *,
    label: str = "training_step",
) -> tuple[int | None, int | None]:
    begin_ns: int | None = None
    end_ns: int | None = None
    for record in marker_records:
        if record.get("label", label) != label:
            continue
        kind = str(record.get("kind") or "")
        realtime_ns = record.get("realtime_ns")
        if realtime_ns in (None, ""):
            continue
        realtime_ns = int(realtime_ns)
        if kind == "step_begin" and begin_ns is None:
            begin_ns = realtime_ns
        elif kind == "step_end":
            end_ns = realtime_ns
    return begin_ns, end_ns


def _worker_step_timing_diagnostics(
    *,
    marker_records: list[dict[str, object]],
    worker_start_realtime_ns: int | None,
    worker_end_realtime_ns: int | None,
    worker_elapsed_seconds: float,
    active_trace_seconds: float,
) -> dict[str, object] | None:
    if worker_start_realtime_ns is None or worker_end_realtime_ns is None:
        return None
    step_begin_ns, step_end_ns = _first_and_last_training_step_marker_ns(marker_records)
    if step_begin_ns is None or step_end_ns is None or step_end_ns < step_begin_ns:
        return None
    bootstrap_before_step_seconds = max(step_begin_ns - worker_start_realtime_ns, 0) / 1_000_000_000.0
    marker_step_seconds = max(step_end_ns - step_begin_ns, 0) / 1_000_000_000.0
    post_step_overhang_seconds = max(worker_end_realtime_ns - step_end_ns, 0) / 1_000_000_000.0
    accounted_seconds = (
        bootstrap_before_step_seconds + marker_step_seconds + post_step_overhang_seconds
    )
    active_trace_minus_marker_step_seconds_signed = (
        float(active_trace_seconds) - marker_step_seconds
    )
    marker_step_minus_active_trace_seconds = max(
        -active_trace_minus_marker_step_seconds_signed,
        0.0,
    )
    return {
        "worker_elapsed_seconds": float(worker_elapsed_seconds),
        "bootstrap_before_step_seconds": bootstrap_before_step_seconds,
        "marker_step_seconds": marker_step_seconds,
        "post_step_overhang_seconds": post_step_overhang_seconds,
        "active_trace_seconds": float(active_trace_seconds),
        "active_trace_minus_marker_step_seconds_signed": (
            active_trace_minus_marker_step_seconds_signed
        ),
        "active_trace_minus_marker_step_seconds": max(
            active_trace_minus_marker_step_seconds_signed,
            0.0,
        ),
        "marker_step_minus_active_trace_seconds": marker_step_minus_active_trace_seconds,
        "worker_unaccounted_seconds": max(float(worker_elapsed_seconds) - accounted_seconds, 0.0),
    }


def _summarize_worker_timing_diagnostics(
    launched_workers: list[dict[str, object]],
) -> dict[str, object] | None:
    unique_diag = [
        worker.get("step_timing_diagnostics")
        for worker in launched_workers
        if worker.get("dynamic_duplicate_of") is None
        and isinstance(worker.get("step_timing_diagnostics"), dict)
    ]
    duplicate_workers = [
        worker
        for worker in launched_workers
        if worker.get("dynamic_duplicate_of") is not None
    ]

    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / float(len(values))

    summary: dict[str, object] = {
        "unique_worker_count": len(unique_diag),
        "duplicate_worker_count": len(duplicate_workers),
    }
    if unique_diag:
        summary.update(
            {
                "unique_worker_elapsed_seconds_mean": _mean(
                    [float(diag["worker_elapsed_seconds"]) for diag in unique_diag]
                ),
                "unique_bootstrap_before_step_seconds_mean": _mean(
                    [float(diag["bootstrap_before_step_seconds"]) for diag in unique_diag]
                ),
                "unique_marker_step_seconds_mean": _mean(
                    [float(diag["marker_step_seconds"]) for diag in unique_diag]
                ),
                "unique_post_step_overhang_seconds_mean": _mean(
                    [float(diag["post_step_overhang_seconds"]) for diag in unique_diag]
                ),
                "unique_active_trace_seconds_mean": _mean(
                    [float(diag["active_trace_seconds"]) for diag in unique_diag]
                ),
                "unique_active_trace_minus_marker_step_seconds_mean": _mean(
                    [float(diag["active_trace_minus_marker_step_seconds"]) for diag in unique_diag]
                ),
                "unique_active_trace_minus_marker_step_seconds_signed_mean": _mean(
                    [
                        float(
                            diag.get(
                                "active_trace_minus_marker_step_seconds_signed",
                                diag["active_trace_minus_marker_step_seconds"],
                            )
                        )
                        for diag in unique_diag
                    ]
                ),
                "unique_marker_step_minus_active_trace_seconds_mean": _mean(
                    [
                        float(diag.get("marker_step_minus_active_trace_seconds", 0.0))
                        for diag in unique_diag
                    ]
                ),
            }
        )
    if duplicate_workers:
        summary.update(
            {
                "duplicate_elapsed_seconds_mean": _mean(
                    [float(worker["elapsed_seconds"]) for worker in duplicate_workers]
                ),
                "duplicate_first_step_classified_elapsed_seconds_mean": _mean(
                    [
                        float(value)
                        for worker in duplicate_workers
                        for value in [worker.get("first_step_classified_elapsed_seconds")]
                        if value is not None
                    ]
                ),
                "duplicate_termination_requested_elapsed_seconds_mean": _mean(
                    [
                        float(value)
                        for worker in duplicate_workers
                        for value in [worker.get("termination_requested_elapsed_seconds")]
                        if value is not None
                    ]
                ),
            }
        )
    return summary


def _summarize_finalize_timing_diagnostics(
    launched_workers: list[dict[str, object]],
) -> dict[str, object] | None:
    diagnostics = [
        worker.get("finalize_timing_diagnostics")
        for worker in launched_workers
        if worker.get("dynamic_duplicate_of") is None
        and isinstance(worker.get("finalize_timing_diagnostics"), dict)
    ]

    if not diagnostics:
        return None

    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / float(len(values))

    def _mean_key(key: str) -> float | None:
        return _mean([float(diag.get(key, 0.0)) for diag in diagnostics])

    return {
        "unique_worker_count": len(diagnostics),
        "finalize_trace_seconds_mean": _mean_key("finalize_trace_seconds"),
        "marker_load_seconds_mean": _mean_key("marker_load_seconds"),
        "resolve_step_window_seconds_mean": _mean_key("resolve_step_window_seconds"),
        "resolve_workload_heuristic_seconds_mean": _mean_key("resolve_workload_heuristic_seconds"),
        "trim_trace_seconds_mean": _mean_key("trim_trace_seconds"),
        "load_communicators_seconds_mean": _mean_key("load_communicators_seconds"),
    }


def _refresh_helper_augmented_step_windows(
    *,
    output_dir: Path,
    host_timing_mode: str | None,
    helper_thread_augmentation: dict[str, object] | None = None,
) -> None:
    manifest_path = output_dir / "capture_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_step_windows = manifest.get("step_windows")
    if not isinstance(raw_step_windows, dict):
        raw_step_windows = {}
        manifest["step_windows"] = raw_step_windows
    raw_fidelity_windows = manifest.get("fidelity_windows")
    if not isinstance(raw_fidelity_windows, dict):
        raw_fidelity_windows = {}
        manifest["fidelity_windows"] = raw_fidelity_windows

    updated = False
    extension_payload = {}
    if isinstance(helper_thread_augmentation, dict):
        raw_extensions = helper_thread_augmentation.get("step_window_extensions_by_rank", {})
        if isinstance(raw_extensions, dict):
            extension_payload = raw_extensions

    for rank_key, extension in sorted(extension_payload.items()):
        if not isinstance(extension, dict):
            continue
        current_step = raw_step_windows.get(rank_key)
        if not isinstance(current_step, dict):
            continue
        current_start = int(current_step.get("start_ts", 0))
        current_end = int(current_step.get("end_ts", current_start))
        updated_start = min(current_start, int(extension.get("start_ts", current_start)))
        updated_end = max(current_end, int(extension.get("end_ts", current_end)))
        if updated_start == current_start and updated_end == current_end:
            continue
        promoted_window = {
            **current_step,
            "start_ts": updated_start,
            "end_ts": updated_end,
            "source": "workload_heuristic",
            "base_source": str(current_step.get("source") or "trace_markers"),
            "base_end_ts": current_end,
            "heuristic_name": "post_step_helper_control_tail",
            "helper_thread_augmented": True,
        }
        raw_step_windows[rank_key] = promoted_window
        raw_fidelity_windows[rank_key] = dict(promoted_window)
        updated = True

    for trace_path in list_rank_trace_files(output_dir):
        rank = int(trace_path.stem.split("_", 1)[1])
        rank_key = str(rank)
        base_window = raw_step_windows.get(rank_key)
        if not isinstance(base_window, dict):
            candidate_window = raw_fidelity_windows.get(rank_key)
            base_window = candidate_window if isinstance(candidate_window, dict) else None
        if not isinstance(base_window, dict):
            continue
        heuristic_window = _resolve_workload_heuristic_step_window(
            trace_path,
            resolved_step_window=base_window,
            host_timing_mode=host_timing_mode,
        )
        if heuristic_window is None:
            continue
        if base_window.get("helper_thread_augmented"):
            heuristic_window["helper_thread_augmented"] = True
        raw_fidelity_windows[rank_key] = dict(heuristic_window)
        if is_paper_valid_fidelity_window_source(str(heuristic_window.get("source"))):
            raw_step_windows[rank_key] = dict(heuristic_window)
        updated = True

    if updated:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _maybe_classify_first_step_pattern(
    worker: _WorkerProcess,
    *,
    first_step_hash_to_representative: dict[str, int],
    dynamic_rank_groups: dict[int, list[int]],
    allow_full_trace_fallback: bool = False,
) -> None:
    if worker.first_step_classified:
        return
    marker_records = _cached_marker_records(worker)
    step_window = None
    if completed_step_count_from_markers(marker_records) >= 1:
        step_window = resolve_indexed_step_window_from_marker_trace_timestamps(
            marker_records,
            occurrence=1,
        )
        if worker.trace_temp_path.exists():
            if step_window is None:
                step_window = resolve_indexed_step_window_from_trace_markers(
                    worker.trace_temp_path,
                    occurrence=1,
                )
        if step_window is None:
            step_window = resolve_indexed_step_window_from_markers(
                marker_records,
                source=TraceSource.FAKE,
                occurrence=1,
            )
    elif allow_full_trace_fallback:
        if not worker.trace_temp_path.exists():
            return
        step_window = estimate_rank_trace_window(
            worker.trace_temp_path,
            source=TraceSource.FAKE,
        )
    if step_window is None:
        return
    tokens = _step_window_pattern_tokens(
        trace_path=worker.trace_temp_path,
        representative_rank=worker.representative_rank,
        step_window=step_window,
        communicator_path=worker.communicator_path,
        communicator_memberships=_cached_communicator_memberships(worker),
    )
    if not tokens:
        return
    _classify_worker_pattern_tokens(
        worker,
        tokens=tokens,
        hash_to_representative=first_step_hash_to_representative,
        dynamic_rank_groups=dynamic_rank_groups,
    )


def _finalize_dynamic_worker_result(worker: _WorkerProcess) -> dict[str, object]:
    returncode = worker.process.wait()
    end_realtime_ns = time.time_ns()
    worker.stdout_handle.close()
    worker.stderr_handle.close()
    return {
        "representative_rank": worker.representative_rank,
        "local_rank": worker.local_rank,
        "launcher": "direct_sitecustomize_dynamic",
        "returncode": returncode,
        "stdout": worker.stdout_path.read_text(encoding="utf-8") if worker.stdout_path.exists() else "",
        "stderr": worker.stderr_path.read_text(encoding="utf-8") if worker.stderr_path.exists() else "",
        "elapsed_seconds": time.perf_counter() - worker.start_time,
        "start_realtime_ns": worker.start_realtime_ns,
        "end_realtime_ns": end_realtime_ns,
        "host_machine_id": worker.host_machine_id,
        "host_dispatch_queue_id": worker.host_dispatch_queue_id,
        "marker_path": worker.marker_path,
        "communicator_path": worker.communicator_path,
        "host_timing_profile": worker.host_timing_profile,
        "host_timing_dispatch_scope": worker.host_timing_dispatch_scope,
        "host_timing_schedule_surface": worker.host_timing_schedule_surface,
        "cpu_affinity_slot": worker.cpu_affinity_slot,
        "cpu_affinity_spec": worker.cpu_affinity_spec,
        "cpu_affinity_cores_per_worker": worker.cpu_affinity_cores_per_worker,
        "cpu_affinity_topology_basis": worker.cpu_affinity_topology_basis,
        "dynamic_duplicate_of": worker.duplicate_of,
        "termination_reason": worker.termination_reason,
        "first_step_classified": worker.first_step_classified,
        "first_step_classified_elapsed_seconds": (
            max(worker.first_step_classified_at - worker.start_time, 0.0)
            if worker.first_step_classified_at is not None
            else None
        ),
        "termination_requested_elapsed_seconds": (
            max(worker.termination_requested_at - worker.start_time, 0.0)
            if worker.termination_requested_at is not None
            else None
        ),
        "first_step_token_hash": worker.first_step_token_hash,
        "first_step_token_count": worker.first_step_token_count,
        "first_step_rolling_window": worker.first_step_rolling_window,
        "first_step_rolling_hash_count": worker.first_step_rolling_hash_count,
    }


def _has_completed_step_markers(marker_path: Path) -> bool:
    try:
        records = load_step_markers(marker_path)
    except Exception:
        return False
    has_begin = False
    has_end = False
    for record in records:
        event = str(record.get("event") or record.get("kind") or record.get("phase") or "")
        if event in {"begin", "start", "step_begin"}:
            has_begin = True
        if event in {"end", "stop", "step_end"}:
            has_end = True
    return has_begin and has_end


def _is_completed_dynamic_first_iteration_worker(worker: Mapping[str, object]) -> bool:
    """A SIGTERM'ed dynamic-dedup worker is valid after its step was captured."""

    if not bool(worker.get("first_step_classified")):
        return False
    marker_path = Path(worker["marker_path"])
    trace_path = _trace_temp_path(marker_path.parent, int(worker["representative_rank"]))
    return marker_path.exists() and trace_path.exists() and _has_completed_step_markers(marker_path)


def _process_completed_unique_worker(
    *,
    worker: Mapping[str, object],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    representative_rank = int(worker["representative_rank"])
    marker_path = Path(worker["marker_path"])
    communicator_path = Path(worker["communicator_path"])
    finalize_step_start = time.perf_counter()
    trace_path = _finalize_worker_trace(
        output_dir=output_dir,
        representative_rank=representative_rank,
    )
    finalize_trace_seconds = time.perf_counter() - finalize_step_start
    marker_load_start = time.perf_counter()
    marker_records = load_step_markers(marker_path)
    marker_load_seconds = time.perf_counter() - marker_load_start
    step_window_start = time.perf_counter()
    resolved_step_window = _resolve_capture_step_window(
        trace_path,
        marker_records=marker_records,
        host_timing_mode=args.host_timing_mode,
        capture_step_window_occurrence=args.capture_step_window_occurrence,
        capture_step_window_step=args.capture_step_window_step,
    )
    resolve_step_window_seconds = time.perf_counter() - step_window_start
    heuristic_window_start = time.perf_counter()
    if _should_apply_default_workload_heuristic_step_window(
        args=args,
        resolved_step_window=resolved_step_window,
    ):
        heuristic_step_window = _resolve_workload_heuristic_step_window(
            trace_path,
            resolved_step_window=resolved_step_window,
            host_timing_mode=args.host_timing_mode,
        )
        if heuristic_step_window is not None:
            resolved_step_window = heuristic_step_window
    resolve_workload_heuristic_seconds = time.perf_counter() - heuristic_window_start
    if resolved_step_window is not None:
        active_trace_seconds = (
            max(
                int(resolved_step_window["end_ts"]) - int(resolved_step_window["start_ts"]),
                0,
            )
            / 1_000_000.0
        )
    else:
        active_trace_seconds = estimate_rank_trace_active_seconds(trace_path) or 0.0

    step_timing_diagnostics = _worker_step_timing_diagnostics(
        marker_records=marker_records,
        worker_start_realtime_ns=(
            int(worker["start_realtime_ns"])
            if worker.get("start_realtime_ns") is not None
            else None
        ),
        worker_end_realtime_ns=(
            int(worker["end_realtime_ns"])
            if worker.get("end_realtime_ns") is not None
            else None
        ),
        worker_elapsed_seconds=float(worker["elapsed_seconds"]),
        active_trace_seconds=active_trace_seconds,
    )

    trim_summary: dict[str, int] | None = None
    trim_trace_seconds = 0.0
    if args.trim_to_step_window and resolved_step_window is not None:
        trim_start = time.perf_counter()
        trim_summary = _trim_trace_file_to_ts_window(
            trace_path,
            start_ts=int(resolved_step_window["start_ts"]),
            end_ts=int(resolved_step_window["end_ts"]),
            pre_padding_us=args.trim_pre_padding_us,
            post_padding_us=args.trim_post_padding_us,
        )
        trim_trace_seconds = time.perf_counter() - trim_start

    load_communicators_start = time.perf_counter()
    worker_communicators = _load_worker_communicators(communicator_path)
    load_communicators_seconds = time.perf_counter() - load_communicators_start
    return {
        "representative_rank": representative_rank,
        "marker_path": marker_path,
        "resolved_step_window": resolved_step_window,
        "active_trace_seconds": active_trace_seconds,
        "step_timing_diagnostics": step_timing_diagnostics,
        "finalize_timing_diagnostics": {
            "finalize_trace_seconds": finalize_trace_seconds,
            "marker_load_seconds": marker_load_seconds,
            "resolve_step_window_seconds": resolve_step_window_seconds,
            "resolve_workload_heuristic_seconds": resolve_workload_heuristic_seconds,
            "trim_trace_seconds": trim_trace_seconds,
            "load_communicators_seconds": load_communicators_seconds,
        },
        "trim_summary": trim_summary,
        "worker_communicators": worker_communicators,
    }


def _run_capture_workers_dynamic(
    *,
    args: argparse.Namespace,
    profiled_ranks: tuple[int, ...],
    rank_host_machines: dict[int, str],
    rank_host_dispatch_queues: dict[int, str],
    script_args: list[str],
    repo_root: str,
    python_root: str,
    output_dir: Path,
) -> tuple[list[dict[str, object]], dict[int, tuple[int, ...]]]:
    max_workers = max(1, args.max_concurrent_workers)
    pending = list(enumerate(profiled_ranks))
    active: dict[int, _WorkerProcess] = {}
    # Track which affinity slot each active worker holds so we can
    # release it when the worker exits. Slots are integers in
    # [0, max_workers) and map to disjoint CPU core ranges via
    # --worker-cpu-affinity-cores-per-worker.
    worker_affinity_slots: dict[int, int] = {}
    free_affinity_slots: list[int] = list(range(max_workers))
    worker_results: list[dict[str, object]] = []
    first_step_hash_to_representative: dict[str, int] = {}
    dynamic_rank_groups: dict[int, list[int]] = {}
    poll_interval = max(float(args.dynamic_dedup_poll_interval_ms), 1.0) / 1000.0
    termination_grace_seconds = DEFAULT_DYNAMIC_DEDUP_KILL_GRACE_MS / 1000.0

    while pending or active:
        while pending and _launch_blocking_worker_count(active) < max_workers:
            profiled_index, representative_rank = pending.pop(0)
            slot = free_affinity_slots.pop(0) if free_affinity_slots else 0
            worker = _start_capture_worker_process(
                args=args,
                profiled_index=profiled_index,
                representative_rank=representative_rank,
                rank_host_machines=rank_host_machines,
                rank_host_dispatch_queues=rank_host_dispatch_queues,
                script_args=script_args,
                repo_root=repo_root,
                python_root=python_root,
                output_dir=output_dir,
                affinity_slot=slot,
            )
            active[representative_rank] = worker
            worker_affinity_slots[representative_rank] = slot

        for worker in list(active.values()):
            _maybe_classify_first_step_pattern(
                worker,
                first_step_hash_to_representative=first_step_hash_to_representative,
                dynamic_rank_groups=dynamic_rank_groups,
            )
            _maybe_escalate_dynamic_dedup_termination(
                worker,
                grace_seconds=termination_grace_seconds,
            )

        completed: list[int] = []
        for representative_rank, worker in active.items():
            if worker.process.poll() is None:
                continue
            if not worker.first_step_classified:
                _maybe_classify_first_step_pattern(
                    worker,
                    first_step_hash_to_representative=first_step_hash_to_representative,
                    dynamic_rank_groups=dynamic_rank_groups,
                    allow_full_trace_fallback=True,
                )
            if not worker.first_step_classified:
                dynamic_rank_groups.setdefault(worker.representative_rank, [worker.representative_rank])
            worker_results.append(_finalize_dynamic_worker_result(worker))
            completed.append(representative_rank)

        for representative_rank in completed:
            active.pop(representative_rank, None)
            # Release this worker's affinity slot back to the free pool so
            # subsequent pending workers can reuse it. Without this, a long
            # capture would exhaust the pool and fall back to slot 0 for all
            # later workers.
            released_slot = worker_affinity_slots.pop(representative_rank, None)
            if released_slot is not None:
                free_affinity_slots.append(released_slot)

        if pending or active:
            time.sleep(poll_interval)

    for representative_rank in profiled_ranks:
        duplicate_of = None
        for root_rank, members in dynamic_rank_groups.items():
            if representative_rank in members:
                duplicate_of = root_rank
                break
        if duplicate_of is None:
            dynamic_rank_groups[representative_rank] = [representative_rank]

    normalized_groups = {
        int(root_rank): tuple(
            [int(root_rank)]
            + sorted(
                int(member)
                for member in members
                if int(member) != int(root_rank)
            )
        )
        for root_rank, members in sorted(dynamic_rank_groups.items())
    }
    return worker_results, normalized_groups


def _finalize_helper_thread_augmentation_contract(
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    summary_dir = _resolve_host_timing_summary_dir(
        explicit_summary_dir=args.host_timing_summary_dir,
        host_timing_profile_dir=args.host_timing_profile_dir,
        host_timing_profile=args.host_timing_profile,
    )
    expected = _helper_thread_augmentation_expected(
        host_timing_mode=args.host_timing_mode,
        summary_dir=summary_dir,
    )
    if not expected:
        return record_helper_thread_augmentation_status(
            output_dir,
            expected=False,
            status=HELPER_THREAD_AUGMENTATION_STATUS_NOT_REQUIRED,
            summary_dir=summary_dir,
        )
    if summary_dir is None:
        return record_helper_thread_augmentation_status(
            output_dir,
            expected=True,
            status=HELPER_THREAD_AUGMENTATION_STATUS_MISSING_SUMMARY_DIR,
            summary_dir=None,
        )
    if not summary_dir.exists():
        return record_helper_thread_augmentation_status(
            output_dir,
            expected=True,
            status=HELPER_THREAD_AUGMENTATION_STATUS_SUMMARY_DIR_NOT_FOUND,
            summary_dir=summary_dir,
        )
    try:
        payload = augment_trace_directory(output_dir, summary_dir=summary_dir)
        _refresh_helper_augmented_step_windows(
            output_dir=output_dir,
            host_timing_mode=args.host_timing_mode,
            helper_thread_augmentation=payload,
        )
        return payload
    except Exception as exc:
        record_helper_thread_augmentation_status(
            output_dir,
            expected=True,
            status=HELPER_THREAD_AUGMENTATION_STATUS_FAILED,
            summary_dir=summary_dir,
            error=str(exc),
        )
        raise


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profiled_rank_groups = _resolved_profiled_rank_groups(args)
    profiled_ranks = profiled_ranks_for_groups(profiled_rank_groups)
    capture_shape = _summarize_profiled_capture(
        logical_world_size=args.logical_world_size,
        profiled_rank_groups=profiled_rank_groups,
        planning_strategy=args.auto_profiled_strategy,
    )
    planned_profiled_rank_groups = dict(profiled_rank_groups)
    script_args = args.script_args[1:] if args.script_args and args.script_args[0] == "--" else args.script_args
    explicit_rank_host_machines = _load_rank_host_machines(args.rank_host_machines_file)

    python_root = str((_repo_root() / "python").resolve())
    repo_root = str(_repo_root().resolve())
    aggregate_stdout: list[str] = []
    aggregate_stderr: list[str] = []
    launched_workers: list[dict[str, object]] = []
    rank_host_machines = dict(explicit_rank_host_machines)
    for rank in range(max(int(args.logical_world_size), 0)):
        rank_host_machines.setdefault(
            int(rank),
            _logical_host_machine_id(int(rank), args.local_device_span),
        )
    resolved_host_timing_dispatch_scope = _resolve_host_timing_dispatch_scope(
        host_timing_mode=args.host_timing_mode,
        requested_dispatch_scope=args.host_timing_dispatch_scope,
    )
    explicit_rank_host_dispatch_queues = _load_rank_host_dispatch_queues(
        args.rank_host_machines_file,
        rank_host_machines=rank_host_machines,
        dispatch_scope=resolved_host_timing_dispatch_scope,
    )
    rank_host_dispatch_queues = dict(explicit_rank_host_dispatch_queues)
    active_emulator_seconds = 0.0
    step_windows: dict[str, dict[str, object]] = {}
    fidelity_windows: dict[str, dict[str, object]] = {}
    communicators: dict[str, dict[str, object]] = {}
    start = time.perf_counter()

    with _capture_lock(args.capture_lock):
        for path in output_dir.glob("rank_*.jsonl"):
            path.unlink()
        for path in output_dir.glob("rank_*.raw.jsonl"):
            path.unlink()
        for path in output_dir.glob("rank_*.markers.jsonl"):
            path.unlink()
        for path in output_dir.glob("rank_*.communicators.json"):
            path.unlink()
        _cleanup_fakecuda_traces()

        worker_results: list[dict[str, object]] = []
        if args.dynamic_first_iteration_dedup:
            worker_results, profiled_rank_groups = _run_capture_workers_dynamic(
                args=args,
                profiled_ranks=profiled_ranks,
                rank_host_machines=rank_host_machines,
                rank_host_dispatch_queues=rank_host_dispatch_queues,
                script_args=script_args,
                repo_root=repo_root,
                python_root=python_root,
                output_dir=output_dir,
            )
            capture_shape = _summarize_profiled_capture(
                logical_world_size=args.logical_world_size,
                profiled_rank_groups=profiled_rank_groups,
                planning_strategy=args.auto_profiled_strategy,
            )
            profiled_ranks = tuple(sorted(profiled_rank_groups))
            # Native Maya dynamic dedup is a runtime profiling optimization:
            # it may reduce the set of physical profiled traces, but it must
            # not change the logical job topology.  Keep the original
            # rank->host and rank->dispatch-queue mapping (e.g. rank//8 for an
            # 8-GPU node) so materialized logical ranks replay on the same host
            # machines as the full job.  The profiled-group remapping helper is
            # only appropriate for ahead-of-time representative-trace routes.
        else:
            max_workers = max(1, args.max_concurrent_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _run_capture_worker,
                        args=args,
                        profiled_index=profiled_index,
                        representative_rank=representative_rank,
                        rank_host_machines=rank_host_machines,
                        rank_host_dispatch_queues=rank_host_dispatch_queues,
                        script_args=script_args,
                        repo_root=repo_root,
                        python_root=python_root,
                        output_dir=output_dir,
                        # Round-robin affinity slot in [0, max_workers).
                        # Disjoint core ranges per concurrent slot avoid
                        # cross-worker CPU contention that would pollute
                        # direct wall-clock hostDelay.
                        affinity_slot=profiled_index % max_workers,
                    )
                    for profiled_index, representative_rank in enumerate(profiled_ranks)
                ]
                for future in as_completed(futures):
                    worker_results.append(future.result())
            if not explicit_rank_host_machines:
                rank_host_machines = _rank_host_machines_for_profiled_groups(
                    profiled_rank_groups,
                    fallback_rank_host_machines=rank_host_machines,
                )
                rank_host_dispatch_queues = _load_rank_host_dispatch_queues(
                    None,
                    rank_host_machines=rank_host_machines,
                    dispatch_scope=resolved_host_timing_dispatch_scope,
                )

        first_failure_rank: int | None = None
        total_worker_elapsed_seconds = 0.0
        successful_unique_workers: list[dict[str, object]] = []
        launched_worker_by_rank: dict[int, dict[str, object]] = {}
        for worker in sorted(worker_results, key=lambda item: int(item["representative_rank"])):
            representative_rank = int(worker["representative_rank"])
            marker_path = Path(worker["marker_path"])
            communicator_path = Path(worker["communicator_path"])
            aggregate_stdout.append(
                f"===== rank {representative_rank} stdout =====\n{worker['stdout']}"
            )
            aggregate_stderr.append(
                f"===== rank {representative_rank} stderr =====\n{worker['stderr']}"
            )
            launched_worker = {
                "representative_rank": representative_rank,
                "local_rank": int(worker["local_rank"]),
                "launcher": str(worker["launcher"]),
                "returncode": int(worker["returncode"]),
                "elapsed_seconds": float(worker["elapsed_seconds"]),
                "start_realtime_ns": int(worker["start_realtime_ns"]),
                "end_realtime_ns": int(worker["end_realtime_ns"]),
                "host_machine_id": str(worker["host_machine_id"]),
                "host_dispatch_queue_id": str(worker["host_dispatch_queue_id"]),
                "host_timing_profile": worker.get("host_timing_profile"),
            }
            if worker.get("cpu_affinity_slot") is not None:
                launched_worker["cpu_affinity_slot"] = int(worker["cpu_affinity_slot"])
            if worker.get("cpu_affinity_spec") is not None:
                launched_worker["cpu_affinity_spec"] = str(worker["cpu_affinity_spec"])
            if worker.get("cpu_affinity_cores_per_worker") is not None:
                launched_worker["cpu_affinity_cores_per_worker"] = int(
                    worker["cpu_affinity_cores_per_worker"]
                )
            if worker.get("cpu_affinity_topology_basis") is not None:
                launched_worker["cpu_affinity_topology_basis"] = str(
                    worker["cpu_affinity_topology_basis"]
                )
            if worker.get("dynamic_duplicate_of") is not None:
                launched_worker["dynamic_duplicate_of"] = int(worker["dynamic_duplicate_of"])
            if worker.get("termination_reason") is not None:
                launched_worker["termination_reason"] = str(worker["termination_reason"])
            if worker.get("first_step_classified") is not None:
                launched_worker["first_step_classified"] = bool(worker["first_step_classified"])
            if worker.get("first_step_classified_elapsed_seconds") is not None:
                launched_worker["first_step_classified_elapsed_seconds"] = float(
                    worker["first_step_classified_elapsed_seconds"]
                )
            if worker.get("termination_requested_elapsed_seconds") is not None:
                launched_worker["termination_requested_elapsed_seconds"] = float(
                    worker["termination_requested_elapsed_seconds"]
                )
            for key in (
                "first_step_token_hash",
                "first_step_token_count",
                "first_step_rolling_window",
                "first_step_rolling_hash_count",
            ):
                if worker.get(key) is not None:
                    launched_worker[key] = worker[key]
            total_worker_elapsed_seconds += float(worker["elapsed_seconds"])
            launched_workers.append(launched_worker)
            launched_worker_by_rank[representative_rank] = launched_worker
            termination_reason = worker.get("termination_reason")
            was_dynamic_termination = termination_reason == "dynamic_first_iteration_dedup"
            was_completed_dynamic_worker = (
                args.dynamic_first_iteration_dedup
                and int(worker["returncode"]) < 0
                and _is_completed_dynamic_first_iteration_worker(worker)
            )
            if int(worker["returncode"]) != 0 and not (was_dynamic_termination or was_completed_dynamic_worker):
                if first_failure_rank is None:
                    first_failure_rank = representative_rank
                marker_path.unlink(missing_ok=True)
                continue

            if worker.get("dynamic_duplicate_of") is not None:
                # Maya's native dynamic dedup profiles every worker for the
                # first iteration, then terminates redundant workers.  Keep the
                # already captured first-iteration trace for duplicate workers;
                # downstream replay needs those rank-specific host timings and
                # critical-path candidates.  Dedup still records the
                # representative groups and prevents profiling beyond the first
                # iteration.
                successful_unique_workers.append(worker)
                continue

            successful_unique_workers.append(worker)

        if first_failure_rank is not None:
            (output_dir / "capture_stdout.txt").write_text(
                "\n".join(aggregate_stdout),
                encoding="utf-8",
            )
            (output_dir / "capture_stderr.txt").write_text(
                "\n".join(aggregate_stderr),
                encoding="utf-8",
            )
            raise SystemExit(
                "emulated worker failed for rank "
                f"{first_failure_rank}; see {output_dir / 'capture_stderr.txt'}"
            )

        worker_capture_elapsed_seconds = time.perf_counter() - start

        finalize_workers = min(len(successful_unique_workers), max(os.cpu_count() or 1, 1))
        processed_unique_workers: list[dict[str, object]] = []
        finalize_start = time.perf_counter()
        if finalize_workers <= 1:
            for worker in successful_unique_workers:
                processed_unique_workers.append(
                    _process_completed_unique_worker(
                        worker=worker,
                        output_dir=output_dir,
                        args=args,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=finalize_workers) as executor:
                futures = [
                    executor.submit(
                        _process_completed_unique_worker,
                        worker=worker,
                        output_dir=output_dir,
                        args=args,
                    )
                    for worker in successful_unique_workers
                ]
                for future in as_completed(futures):
                    processed_unique_workers.append(future.result())
        finalize_elapsed_seconds = time.perf_counter() - finalize_start

        for processed in sorted(
            processed_unique_workers,
            key=lambda item: int(item["representative_rank"]),
        ):
            representative_rank = int(processed["representative_rank"])
            launched_worker = launched_worker_by_rank[representative_rank]
            resolved_step_window = processed["resolved_step_window"]
            active_trace_seconds = float(processed["active_trace_seconds"])
            if resolved_step_window is not None:
                fidelity_windows[str(representative_rank)] = dict(resolved_step_window)
                if is_paper_valid_fidelity_window_source(
                    str(resolved_step_window.get("source"))
                ):
                    step_windows[str(representative_rank)] = dict(resolved_step_window)
            trim_summary = processed["trim_summary"]
            if trim_summary is not None:
                launched_worker["trimmed_trace_events"] = int(trim_summary["kept_events"])
                launched_worker["trimmed_trace_total_events"] = int(trim_summary["total_events"])
            active_emulator_seconds += active_trace_seconds
            launched_worker["active_trace_seconds"] = active_trace_seconds
            if processed["step_timing_diagnostics"] is not None:
                launched_worker["step_timing_diagnostics"] = dict(processed["step_timing_diagnostics"])
            if processed["finalize_timing_diagnostics"] is not None:
                launched_worker["finalize_timing_diagnostics"] = dict(
                    processed["finalize_timing_diagnostics"]
                )
            worker_communicators = processed["worker_communicators"]
            if worker_communicators:
                launched_worker["communicator_count"] = len(worker_communicators)
            for comm_id, record in sorted(worker_communicators.items()):
                existing = communicators.get(comm_id)
                if existing is not None and existing.get("members") != record.get("members"):
                    raise SystemExit(
                        "emulated communicator topology mismatch for "
                        f"{comm_id}: {existing.get('members')} != {record.get('members')}"
                    )
                communicators[comm_id] = record
            Path(processed["marker_path"]).unlink(missing_ok=True)

    capture_command_elapsed_seconds = time.perf_counter() - start
    # Figure 5/6 define the emulator stage as producing worker traces for the
    # collator. Keep worker-only elapsed time as diagnostic metadata, but use
    # the full capture/finalize path for paper-facing Emulator accounting.
    capture_elapsed_seconds = capture_command_elapsed_seconds
    (output_dir / "capture_stdout.txt").write_text("\n".join(aggregate_stdout), encoding="utf-8")
    (output_dir / "capture_stderr.txt").write_text("\n".join(aggregate_stderr), encoding="utf-8")
    (output_dir / "capture_elapsed_seconds.txt").write_text(
        f"{capture_elapsed_seconds:.6f}\n",
        encoding="utf-8",
    )
    (output_dir / "capture_command_elapsed_seconds.txt").write_text(
        f"{capture_command_elapsed_seconds:.6f}\n",
        encoding="utf-8",
    )
    host_timing_policy = _summarize_host_timing_policy(
        host_timing_mode=args.host_timing_mode,
        requested_dispatch_scope=args.host_timing_dispatch_scope,
        resolved_dispatch_scope=_resolve_host_timing_dispatch_scope(
            host_timing_mode=args.host_timing_mode,
            requested_dispatch_scope=args.host_timing_dispatch_scope,
        ),
        requested_schedule_surface=args.host_timing_schedule_surface,
        resolved_schedule_surface=_resolve_host_timing_schedule_surface(
            host_timing_mode=args.host_timing_mode,
            requested_schedule_surface=args.host_timing_schedule_surface,
        ),
        host_timing_profile=args.host_timing_profile,
        host_timing_profile_dir=args.host_timing_profile_dir,
    )
    trace_flush_mode_resolved, trace_flush_every_resolved, trace_stdio_buffer_bytes_resolved = (
        _resolve_trace_flush_policy(args)
    )
    resolved_host_timing_summary_dir = _resolve_host_timing_summary_dir(
        explicit_summary_dir=args.host_timing_summary_dir,
        host_timing_profile_dir=args.host_timing_profile_dir,
        host_timing_profile=args.host_timing_profile,
    )
    manifest = {
        "mode": "emulated_phase1",
        "original_world_size": args.logical_world_size,
        "profiled_ranks": list(profiled_ranks),
        "profiled_rank_groups": {
            str(rank): list(members)
            for rank, members in sorted(profiled_rank_groups.items())
        },
        "rank_host_machines": {
            str(rank): host_machine_id
            for rank, host_machine_id in sorted(rank_host_machines.items())
        },
        "rank_host_dispatch_queues": {
            str(rank): host_dispatch_queue_id
            for rank, host_dispatch_queue_id in sorted(rank_host_dispatch_queues.items())
        },
        **_capture_elapsed_metadata(
            capture_elapsed_seconds=capture_elapsed_seconds,
            capture_command_elapsed_seconds=capture_command_elapsed_seconds,
            worker_capture_elapsed_seconds=worker_capture_elapsed_seconds,
            post_worker_finalize_seconds=finalize_elapsed_seconds,
        ),
        "active_emulator_seconds": active_emulator_seconds,
        "step_windows": step_windows,
        "fidelity_windows": fidelity_windows,
        "communicators": communicators,
        "launched_workers": launched_workers,
        "launcher": "direct_sitecustomize",
        "fakecuda_artifacts": _fakecuda_artifact_fingerprint_metadata(args.frun),
        "collective_mode": args.collective_mode,
        "max_concurrent_workers": max(1, args.max_concurrent_workers),
        **_worker_cpu_affinity_manifest_metadata(args),
        "local_device_span": max(int(args.local_device_span), 1),
        "total_worker_elapsed_seconds": total_worker_elapsed_seconds,
        "worker_timing_diagnostics_summary": _summarize_worker_timing_diagnostics(launched_workers),
        "finalize_timing_diagnostics_summary": _summarize_finalize_timing_diagnostics(
            launched_workers
        ),
        "trace_flush_mode": args.trace_flush_mode,
        "trace_flush_every": args.trace_flush_every,
        "trace_stdio_buffer_bytes": args.trace_stdio_buffer_bytes,
        "trace_surface": args.trace_surface,
        "trace_flush_mode_resolved": trace_flush_mode_resolved,
        "trace_flush_every_resolved": trace_flush_every_resolved,
        "trace_stdio_buffer_bytes_resolved": trace_stdio_buffer_bytes_resolved,
        "host_timing_mode": args.host_timing_mode,
        "host_timing_dispatch_scope": args.host_timing_dispatch_scope,
        "host_timing_schedule_surface": args.host_timing_schedule_surface,
        "dynamic_first_iteration_dedup": args.dynamic_first_iteration_dedup,
        "dynamic_dedup_sequence_hash_mode": (
            "complete_first_step_canonical_operation_sequence"
            if args.dynamic_first_iteration_dedup
            else None
        ),
        "dynamic_dedup_first_iteration_trace_retention": (
            "retain_all_workers_first_iteration"
            if args.dynamic_first_iteration_dedup
            else None
        ),
        "dynamic_dedup_rolling_window": (
            DEFAULT_DYNAMIC_DEDUP_ROLLING_WINDOW
            if args.dynamic_first_iteration_dedup
            else None
        ),
        "planned_profiled_rank_groups": {
            str(rank): list(members)
            for rank, members in sorted(planned_profiled_rank_groups.items())
        },
        "host_timing_profile": (
            str(args.host_timing_profile.resolve()) if args.host_timing_profile is not None else None
        ),
        "host_timing_profile_dir": (
            str(args.host_timing_profile_dir.resolve())
            if args.host_timing_profile_dir is not None
            else None
        ),
        "host_timing_summary_dir": (
            str(resolved_host_timing_summary_dir)
            if resolved_host_timing_summary_dir is not None
            else None
        ),
        "host_timing_default_us": args.host_timing_default_us,
        "trim_to_step_window": args.trim_to_step_window,
        "capture_step_window_occurrence": args.capture_step_window_occurrence,
        "capture_step_window_step": args.capture_step_window_step,
        "trim_pre_padding_us": args.trim_pre_padding_us,
        "trim_post_padding_us": args.trim_post_padding_us,
        **host_timing_policy,
        **capture_shape,
    }
    (output_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    helper_thread_augmentation = _finalize_helper_thread_augmentation_contract(
        output_dir=output_dir,
        args=args,
    )
    manifest["helper_thread_augmentation"] = helper_thread_augmentation
    manifest_path = output_dir / "capture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
