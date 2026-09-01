import pytest

from flexsim.maya_lite.capture_real import _env_flag_enabled, _env_flag_truthy


_ACTUAL_CUDA_EVENT_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_ACTUAL_CUDA_EVENT_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_ACTUAL_CUDA_EVENT_COUNTERPART_DIAGNOSTICS",
)
_HOST_CONTROL_BOUNDARY_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_HOST_CONTROL_BOUNDARY_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_HOST_CONTROL_BOUNDARY_COUNTERPART_DIAGNOSTICS",
)
_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_DIAGNOSTICS",
)
_SHARED_PHASE_ANCHOR_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
)
_SHARED_PHASE_ANCHOR_COMMON_BASIS_ENV_KEYS = (
    "MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
)
_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_ENV_KEYS = (
    "MAYA_ENABLE_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_COUNTERPART_DIAGNOSTICS",
)
_NCCL_WAIT_RELEASE_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_NCCL_WAIT_RELEASE_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_NCCL_WAIT_RELEASE_COUNTERPART_DIAGNOSTICS",
)
_LAUNCH_NEIGHBORHOOD_EQUIVALENCE_ENV_KEYS = (
    "MAYA_ENABLE_LAUNCH_NEIGHBORHOOD_EQUIVALENCE_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_LAUNCH_NEIGHBORHOOD_EQUIVALENCE_DIAGNOSTICS",
)
_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_DIAGNOSTICS",
)


def _clear_actual_cuda_event_counterpart_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ACTUAL_CUDA_EVENT_COUNTERPART_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _clear_host_control_boundary_counterpart_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _HOST_CONTROL_BOUNDARY_COUNTERPART_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _clear_appendix_ab_p2p_actual_counterpart_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _APPENDIX_AB_P2P_ACTUAL_COUNTERPART_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _clear_shared_phase_anchor_counterpart_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _SHARED_PHASE_ANCHOR_COUNTERPART_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _clear_shared_phase_anchor_common_basis_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _SHARED_PHASE_ANCHOR_COMMON_BASIS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _clear_selected_allreduce_release_participant_host_dispatch_phase_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in _SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _clear_nccl_wait_release_counterpart_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _NCCL_WAIT_RELEASE_COUNTERPART_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _clear_launch_neighborhood_equivalence_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _LAUNCH_NEIGHBORHOOD_EQUIVALENCE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _clear_generic_replay_placement_envelope_actual_counterpart_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in _GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("key", _ACTUAL_CUDA_EVENT_COUNTERPART_ENV_KEYS)
@pytest.mark.parametrize("value", ["true", "yes", "on", "0", ""])
def test_actual_cuda_event_counterpart_env_requires_exact_one_for_false_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
):
    _clear_actual_cuda_event_counterpart_env(monkeypatch)
    monkeypatch.setenv(key, value)

    assert _env_flag_enabled(*_ACTUAL_CUDA_EVENT_COUNTERPART_ENV_KEYS) is False


def test_actual_cuda_event_counterpart_env_requires_exact_one_when_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_actual_cuda_event_counterpart_env(monkeypatch)

    assert _env_flag_enabled(*_ACTUAL_CUDA_EVENT_COUNTERPART_ENV_KEYS) is False


@pytest.mark.parametrize("key", _ACTUAL_CUDA_EVENT_COUNTERPART_ENV_KEYS)
def test_actual_cuda_event_counterpart_env_enables_exact_one(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
):
    _clear_actual_cuda_event_counterpart_env(monkeypatch)
    monkeypatch.setenv(key, "1")

    assert _env_flag_enabled(*_ACTUAL_CUDA_EVENT_COUNTERPART_ENV_KEYS) is True


@pytest.mark.parametrize("key", _HOST_CONTROL_BOUNDARY_COUNTERPART_ENV_KEYS)
@pytest.mark.parametrize("value", ["true", "yes", "on", "0", ""])
def test_host_control_boundary_counterpart_env_requires_exact_one_for_false_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
):
    _clear_host_control_boundary_counterpart_env(monkeypatch)
    monkeypatch.setenv(key, value)

    assert _env_flag_enabled(*_HOST_CONTROL_BOUNDARY_COUNTERPART_ENV_KEYS) is False


def test_host_control_boundary_counterpart_env_requires_exact_one_when_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_host_control_boundary_counterpart_env(monkeypatch)

    assert _env_flag_enabled(*_HOST_CONTROL_BOUNDARY_COUNTERPART_ENV_KEYS) is False


@pytest.mark.parametrize("key", _HOST_CONTROL_BOUNDARY_COUNTERPART_ENV_KEYS)
def test_host_control_boundary_counterpart_env_enables_exact_one(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
):
    _clear_host_control_boundary_counterpart_env(monkeypatch)
    monkeypatch.setenv(key, "1")

    assert _env_flag_enabled(*_HOST_CONTROL_BOUNDARY_COUNTERPART_ENV_KEYS) is True


@pytest.mark.parametrize("key", _LAUNCH_NEIGHBORHOOD_EQUIVALENCE_ENV_KEYS)
@pytest.mark.parametrize("value", ["true", "yes", "on", "0", ""])
def test_launch_neighborhood_equivalence_env_requires_exact_one_for_false_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
):
    _clear_launch_neighborhood_equivalence_env(monkeypatch)
    monkeypatch.setenv(key, value)

    assert _env_flag_enabled(*_LAUNCH_NEIGHBORHOOD_EQUIVALENCE_ENV_KEYS) is False


@pytest.mark.parametrize("key", _LAUNCH_NEIGHBORHOOD_EQUIVALENCE_ENV_KEYS)
def test_launch_neighborhood_equivalence_env_enables_exact_one(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
):
    _clear_launch_neighborhood_equivalence_env(monkeypatch)
    monkeypatch.setenv(key, "1")

    assert _env_flag_enabled(*_LAUNCH_NEIGHBORHOOD_EQUIVALENCE_ENV_KEYS) is True


@pytest.mark.parametrize("key", _APPENDIX_AB_P2P_ACTUAL_COUNTERPART_ENV_KEYS)
@pytest.mark.parametrize("value", ["true", "yes", "on", "0", ""])
def test_appendix_ab_p2p_actual_counterpart_env_requires_exact_one_for_false_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
):
    _clear_appendix_ab_p2p_actual_counterpart_env(monkeypatch)
    monkeypatch.setenv(key, value)

    assert _env_flag_enabled(*_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_ENV_KEYS) is False


def test_appendix_ab_p2p_actual_counterpart_env_requires_exact_one_when_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_appendix_ab_p2p_actual_counterpart_env(monkeypatch)

    assert _env_flag_enabled(*_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_ENV_KEYS) is False


@pytest.mark.parametrize("key", _APPENDIX_AB_P2P_ACTUAL_COUNTERPART_ENV_KEYS)
def test_appendix_ab_p2p_actual_counterpart_env_enables_exact_one(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
):
    _clear_appendix_ab_p2p_actual_counterpart_env(monkeypatch)
    monkeypatch.setenv(key, "1")

    assert _env_flag_enabled(*_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_ENV_KEYS) is True


@pytest.mark.parametrize("key", _SHARED_PHASE_ANCHOR_COUNTERPART_ENV_KEYS)
@pytest.mark.parametrize("value", ["0", "", "false", "no", "off"])
def test_shared_phase_anchor_counterpart_env_truthy_false_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
):
    _clear_shared_phase_anchor_counterpart_env(monkeypatch)
    monkeypatch.setenv(key, value)

    assert _env_flag_truthy(*_SHARED_PHASE_ANCHOR_COUNTERPART_ENV_KEYS) is False


def test_shared_phase_anchor_counterpart_env_truthy_when_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_phase_anchor_counterpart_env(monkeypatch)

    assert _env_flag_truthy(*_SHARED_PHASE_ANCHOR_COUNTERPART_ENV_KEYS) is False


@pytest.mark.parametrize("key", _SHARED_PHASE_ANCHOR_COUNTERPART_ENV_KEYS)
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_shared_phase_anchor_counterpart_env_truthy_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
):
    _clear_shared_phase_anchor_counterpart_env(monkeypatch)
    monkeypatch.setenv(key, value)

    assert _env_flag_truthy(*_SHARED_PHASE_ANCHOR_COUNTERPART_ENV_KEYS) is True


@pytest.mark.parametrize("key", _SHARED_PHASE_ANCHOR_COMMON_BASIS_ENV_KEYS)
@pytest.mark.parametrize("value", ["0", "", "false", "no", "off"])
def test_shared_phase_anchor_common_basis_env_truthy_false_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
):
    _clear_shared_phase_anchor_common_basis_env(monkeypatch)
    monkeypatch.setenv(key, value)

    assert _env_flag_truthy(*_SHARED_PHASE_ANCHOR_COMMON_BASIS_ENV_KEYS) is False


def test_shared_phase_anchor_common_basis_env_truthy_when_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_shared_phase_anchor_common_basis_env(monkeypatch)

    assert _env_flag_truthy(*_SHARED_PHASE_ANCHOR_COMMON_BASIS_ENV_KEYS) is False


@pytest.mark.parametrize("key", _SHARED_PHASE_ANCHOR_COMMON_BASIS_ENV_KEYS)
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_shared_phase_anchor_common_basis_env_truthy_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
):
    _clear_shared_phase_anchor_common_basis_env(monkeypatch)
    monkeypatch.setenv(key, value)

    assert _env_flag_truthy(*_SHARED_PHASE_ANCHOR_COMMON_BASIS_ENV_KEYS) is True


@pytest.mark.parametrize(
    "key", _SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_ENV_KEYS
)
@pytest.mark.parametrize("value", ["0", "", "false", "no", "off"])
def test_selected_allreduce_release_participant_host_dispatch_phase_env_false_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
):
    _clear_selected_allreduce_release_participant_host_dispatch_phase_env(monkeypatch)
    monkeypatch.setenv(key, value)

    assert (
        _env_flag_truthy(
            *_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_ENV_KEYS
        )
        is False
    )


def test_selected_allreduce_release_participant_host_dispatch_phase_env_when_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_selected_allreduce_release_participant_host_dispatch_phase_env(monkeypatch)

    assert (
        _env_flag_truthy(
            *_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_ENV_KEYS
        )
        is False
    )


@pytest.mark.parametrize(
    "key", _SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_ENV_KEYS
)
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_selected_allreduce_release_participant_host_dispatch_phase_env_truthy_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
):
    _clear_selected_allreduce_release_participant_host_dispatch_phase_env(monkeypatch)
    monkeypatch.setenv(key, value)

    assert (
        _env_flag_truthy(
            *_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_ENV_KEYS
        )
        is True
    )


@pytest.mark.parametrize("key", _NCCL_WAIT_RELEASE_COUNTERPART_ENV_KEYS)
@pytest.mark.parametrize("value", ["true", "yes", "on", "0", ""])
def test_nccl_wait_release_counterpart_env_requires_exact_one_for_false_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
):
    _clear_nccl_wait_release_counterpart_env(monkeypatch)
    monkeypatch.setenv(key, value)

    assert _env_flag_enabled(*_NCCL_WAIT_RELEASE_COUNTERPART_ENV_KEYS) is False


def test_nccl_wait_release_counterpart_env_requires_exact_one_when_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_nccl_wait_release_counterpart_env(monkeypatch)

    assert _env_flag_enabled(*_NCCL_WAIT_RELEASE_COUNTERPART_ENV_KEYS) is False


@pytest.mark.parametrize("key", _NCCL_WAIT_RELEASE_COUNTERPART_ENV_KEYS)
def test_nccl_wait_release_counterpart_env_enables_exact_one(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
):
    _clear_nccl_wait_release_counterpart_env(monkeypatch)
    monkeypatch.setenv(key, "1")

    assert _env_flag_enabled(*_NCCL_WAIT_RELEASE_COUNTERPART_ENV_KEYS) is True


@pytest.mark.parametrize(
    "key", _GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_ENV_KEYS
)
@pytest.mark.parametrize("value", ["0", "", "false", "no", "off"])
def test_generic_replay_placement_envelope_actual_counterpart_env_false_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
):
    _clear_generic_replay_placement_envelope_actual_counterpart_env(monkeypatch)
    monkeypatch.setenv(key, value)

    assert (
        _env_flag_truthy(
            *_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_ENV_KEYS
        )
        is False
    )


def test_generic_replay_placement_envelope_actual_counterpart_env_when_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_generic_replay_placement_envelope_actual_counterpart_env(monkeypatch)

    assert (
        _env_flag_truthy(
            *_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_ENV_KEYS
        )
        is False
    )


@pytest.mark.parametrize(
    "key", _GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_ENV_KEYS
)
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_generic_replay_placement_envelope_actual_counterpart_env_truthy_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
):
    _clear_generic_replay_placement_envelope_actual_counterpart_env(monkeypatch)
    monkeypatch.setenv(key, value)

    assert (
        _env_flag_truthy(
            *_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_ENV_KEYS
        )
        is True
    )
