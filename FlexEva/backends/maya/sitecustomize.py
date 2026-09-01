"""Repo-local Python path bootstrap for direct execution from the workspace."""

from __future__ import annotations

import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent
_TRUTHY_ENV = {"1", "true", "yes", "on"}
_AUTO_DDP_COMPAT_ENV = "FLEXSIM_AUTO_INSTALL_FAKECUDA_DDP_COMPAT"
_PYTHON_ROOT = _REPO_ROOT / "python"
_IMPORT_PATHS = [
    _PYTHON_ROOT,
    _REPO_ROOT / "CppEvent",
    _REPO_ROOT / "build" / "CppEvent",
    _REPO_ROOT / "build" / "bindlayer",
]
_LIBRARY_PATHS = [
    _REPO_ROOT / "CppEvent",
    _REPO_ROOT / "build" / "CppEvent",
    _REPO_ROOT / "lowlevel-interface",
    _REPO_ROOT / "build" / "lowlevel-interface",
]


def _prepend_sys_path(path: Path) -> None:
    value = str(path)
    if path.exists() and value not in sys.path:
        sys.path.insert(0, value)


def _prepend_env_path(name: str, paths: list[Path]) -> None:
    current = [entry for entry in os.environ.get(name, "").split(":") if entry]
    merged: list[str] = []

    for path in paths:
        value = str(path)
        if path.exists() and value not in merged:
            merged.append(value)

    for entry in current:
        if entry not in merged:
            merged.append(entry)

    if merged:
        os.environ[name] = ":".join(merged)


for _path in _IMPORT_PATHS:
    _prepend_sys_path(_path)

_prepend_env_path("PYTHONPATH", _IMPORT_PATHS)
_prepend_env_path("LD_LIBRARY_PATH", _LIBRARY_PATHS)


def _normalized_env_value(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip().lower()


def _maybe_install_maya_emulated_dist() -> None:
    if _normalized_env_value("FLEXSIM_MAYA_EMULATED_DIST") not in _TRUTHY_ENV:
        return
    try:
        from flexsim.maya_lite.emulated_dist import install_emulated_distributed_from_env

        install_emulated_distributed_from_env()
    except Exception as exc:
        raise RuntimeError("failed to install Maya-lite emulated distributed bootstrap") from exc


def _entrypoint_script_path() -> Path | None:
    argv0 = sys.argv[0].strip() if sys.argv else ""
    if not argv0 or argv0 in {"-c", "-m"}:
        return None
    path = Path(argv0)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    try:
        if path.exists() and path.is_file():
            return path
    except OSError:
        return None
    return None


def _workload_likely_uses_ddp(entrypoint: Path | None = None) -> bool:
    path = entrypoint or _entrypoint_script_path()
    if path is None:
        return False
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    ddp_markers = (
        "DistributedDataParallel",
        "install_fakecuda_ddp_compat",
    )
    return any(marker in source for marker in ddp_markers)


def _should_install_fakecuda_ddp_compat(entrypoint: Path | None = None) -> bool:
    if _normalized_env_value("FLEXSIM_FAKECUDA_DDP_COMPAT") in _TRUTHY_ENV:
        return True
    policy = _normalized_env_value(_AUTO_DDP_COMPAT_ENV, "auto")
    if policy in {"0", "false", "no", "off", "never"}:
        return False
    if policy in _TRUTHY_ENV or policy == "always":
        return True
    return _workload_likely_uses_ddp(entrypoint)


def _maybe_install_fakecuda_ddp_compat() -> None:
    if not _should_install_fakecuda_ddp_compat():
        return
    try:
        from flexsim.maya_lite.fakecuda_compat import install_fakecuda_ddp_compat

        install_fakecuda_ddp_compat()
    except Exception:
        # Keep repo-local bootstrap tolerant for non-PyTorch utility invocations.
        return


_maybe_install_fakecuda_ddp_compat()
_maybe_install_maya_emulated_dist()
