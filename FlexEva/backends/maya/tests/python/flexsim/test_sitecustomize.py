from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_sitecustomize(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("FLEXSIM_AUTO_INSTALL_FAKECUDA_DDP_COMPAT", "never")
    monkeypatch.delenv("FLEXSIM_MAYA_EMULATED_DIST", raising=False)
    spec = importlib.util.spec_from_file_location(
        "sitecustomize_under_test",
        repo_root / "sitecustomize.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sitecustomize_auto_ddp_detection_distinguishes_megatron_and_ddp_workloads(
    monkeypatch,
) -> None:
    module = _load_sitecustomize(monkeypatch)
    repo_root = Path(__file__).resolve().parents[3]

    megatron_workload = repo_root / "tests" / "workloads" / "fake_cuda" / "maya_fig13_megatron.py"
    gpt2_workload = repo_root / "tests" / "workloads" / "fake_cuda" / "gpt2.py"

    assert module._workload_likely_uses_ddp(megatron_workload) is False
    assert module._workload_likely_uses_ddp(gpt2_workload) is True


def test_sitecustomize_ddp_policy_respects_auto_never_and_always(monkeypatch) -> None:
    module = _load_sitecustomize(monkeypatch)
    repo_root = Path(__file__).resolve().parents[3]
    gpt2_workload = repo_root / "tests" / "workloads" / "fake_cuda" / "gpt2.py"

    monkeypatch.setenv("FLEXSIM_AUTO_INSTALL_FAKECUDA_DDP_COMPAT", "auto")
    assert module._should_install_fakecuda_ddp_compat() is False
    assert module._should_install_fakecuda_ddp_compat(gpt2_workload) is True

    monkeypatch.setenv("FLEXSIM_AUTO_INSTALL_FAKECUDA_DDP_COMPAT", "never")
    assert module._should_install_fakecuda_ddp_compat(gpt2_workload) is False

    monkeypatch.setenv("FLEXSIM_AUTO_INSTALL_FAKECUDA_DDP_COMPAT", "always")
    assert module._should_install_fakecuda_ddp_compat() is True
