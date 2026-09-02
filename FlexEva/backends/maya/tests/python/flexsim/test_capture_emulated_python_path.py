from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from flexsim.maya_lite.capture_emulated import (
    _build_parser,
    _capture_worker_command_and_env,
)


@pytest.mark.parametrize("direct_proot", [False, True])
def test_capture_worker_preserves_uv_venv_python_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    direct_proot: bool,
) -> None:
    fakecuda_root = tmp_path / "fake-cuda"
    fakecuda_root.mkdir()
    frun = fakecuda_root / "frun"
    frun.write_text("", encoding="utf-8")

    base_python = Path(sys.executable).resolve()
    venv_root = tmp_path / ".venv"
    python_bin = venv_root / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.symlink_to(base_python)
    (venv_root / "pyvenv.cfg").write_text(
        f"home = {base_python.parent}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    site_packages = (
        venv_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages.mkdir(parents=True)
    (site_packages / "venv_probe.py").write_text("VALUE = 'venv'\n", encoding="utf-8")

    monkeypatch.setattr(
        "flexsim.maya_lite.capture_emulated._maybe_build_direct_proot_command_prefix",
        (lambda **_: (["/fake/proot"], "/fake/lib")) if direct_proot else (lambda **_: None),
    )
    repo_root = Path(__file__).resolve().parents[3]
    args = _build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--logical-world-size",
            "4",
            "--auto-profiled-strategy",
            "identity",
            "tests/workloads/fake_cuda/maya_fig13_megatron.py",
        ]
    )
    args.frun = frun
    args.python_bin = python_bin

    command, env, *_ = _capture_worker_command_and_env(
        args=args,
        profiled_index=0,
        representative_rank=0,
        rank_host_machines={},
        rank_host_dispatch_queues={},
        script_args=[],
        repo_root=str(repo_root),
        python_root=str(repo_root / "python"),
        output_dir=tmp_path / "out",
    )

    python_command = str(python_bin.absolute())
    assert python_command in command
    assert str(python_bin.resolve()) not in command
    assert env["FAKECUDA_TARGET_ENV_ROOT"] == str(venv_root.absolute())
    assert env["FAKECUDA_SITE_PACKAGES_ROOT"] == str(site_packages.absolute())
    completed = subprocess.run(
        [python_command, "-c", "import venv_probe; assert venv_probe.VALUE == 'venv'"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
