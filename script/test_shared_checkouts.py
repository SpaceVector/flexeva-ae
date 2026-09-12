#!/usr/bin/env python3
"""Regression checks for same-account checkouts; no SSH or GPU work is run."""
import fcntl
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SENTINELS = {"status.env": "status=completed\n", "interpretation.md": "Original result\n"}


def run(command, env):
    return subprocess.run(command, env=env, text=True, capture_output=True, timeout=15)


def script(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def preserve(path):
    path.mkdir(parents=True)
    for name, text in SENTINELS.items():
        (path / name).write_text(text)


def unchanged(path):
    assert {p.name: p.read_text() for p in path.iterdir()} == SENTINELS, path


def test_guard(home, env):
    first = home / "reviewer-a/flexeva-ae"
    second = home / "reviewer-b/flexeva-ae"
    for checkout in (first, second):
        (checkout / "script/e3").mkdir(parents=True)
        for name in ("server.sh", "server_guard.sh"):
            shutil.copy2(ROOT / "script/e3" / name, checkout / "script/e3" / name)

    def guarded(checkout, run_id, node_root=home, extra=None):
        return run([
            "bash", str(checkout / "script/e3/server_guard.sh"), "run",
            socket.gethostname(), str(node_root), "/bin/false", run_id, "8",
            str(checkout.relative_to(node_root)), "64", "1", "0", "0", "--", "/bin/false",
        ], {**env, **(extra or {})})

    legacy = home / "eurosys27-ae/runs/finished"
    own = first / "result/server-runs/finished"
    preserve(legacy)
    preserve(own)
    result = guarded(first, "finished")
    assert result.returncode == 75, result
    unchanged(legacy)
    unchanged(own)

    (first / "result/server-runs/linked").symlink_to(own, target_is_directory=True)
    result = guarded(first, "linked")
    assert result.returncode == 75, result
    unchanged(own)

    race = first / "result/server-runs/race"
    script(home / "race-bin/mkdir", '''#!/usr/bin/env bash
if [[ "${@: -1}" == "$TEST_RACE_PATH" ]]; then
    "$TEST_REAL_MKDIR" "$TEST_RACE_PATH" || exit
    printf 'status=completed\\n' >"$TEST_RACE_PATH/status.env"
    printf 'Original result\\n' >"$TEST_RACE_PATH/interpretation.md"
    exit 1
fi
exec "$TEST_REAL_MKDIR" "$@"
''')
    result = guarded(first, "race", extra={
        "PATH": f"{home}/race-bin:{env['PATH']}", "TEST_RACE_PATH": str(race),
        "TEST_REAL_MKDIR": shutil.which("mkdir"),
    })
    assert result.returncode == 75, result
    unchanged(race)

    lock = home / "eurosys27-ae/.locks/gpu-run.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("run_id=another-reviewer\n")
    with lock.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Different checkout parents must still use the same account-level lock.
        result = guarded(second, "finished", node_root=second.parent)
        assert result.returncode == 75 and "another guarded GPU run" in result.stderr, result
        assert lock.read_text() == "run_id=another-reviewer\n"
        assert "status=environment_blocked" in (second / "result/server-runs/finished/status.env").read_text()
        unchanged(own)
        unchanged(legacy)

    result = guarded(second, "lock-released", node_root=second.parent)
    assert result.returncode == 75 and "GPFS write/fsync probe failed" in result.stderr, result
    assert "another guarded GPU run" not in result.stderr, result

    result = run(["bash", str(first / "script/e3/server.sh"), "status", "finished"],
                 {**env, "AE_NODE_ROOT": str(home), "AE_EXPECTED_FILESYSTEM": "gpfs"})
    assert result.returncode == 0 and f"run_dir={own}\n" in result.stdout, result
    assert "status=completed\n" in result.stdout, result
    result = guarded(first, "../escape")
    assert result.returncode == 2 and "invalid RUN_ID" in result.stderr, result


def test_peer_checkout(home, env):
    # Stop setup at the first clone attempt: fake SSH never opens a connection.
    fake_bin = home / "setup-bin"
    script(fake_bin / "git", "#!/bin/sh\nprintf '%040d\\n' 1\n")
    script(fake_bin / "ip", "#!/bin/sh\nexit 0\n")
    script(fake_bin / "ssh", '''#!/usr/bin/env bash
case "${@: -1}" in
    *'printf'*) printf '%s\\n' "$TEST_PEER_HOME" ;;
    *) printf '%s\\n' "${@: -1}"; exit 91 ;;
esac
''')
    setup_env = {**env, "PATH": f"{fake_bin}:{env['PATH']}",
                 "FLEXMAYA_PEER_TARGET": "test-peer", "FLEXMAYA_PEER_PORT": "22",
                 "TEST_PEER_HOME": "/peer-home"}
    for relative in ("flexeva-ae", "reviewer-a/flexeva-ae", "reviewer-b/flexeva-ae"):
        checkout = home / relative
        (checkout / "script/lib").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "script/setup", checkout / "script/setup")
        shutil.copy2(ROOT / "script/lib/ae_env.sh", checkout / "script/lib/ae_env.sh")
        result = run(["bash", str(checkout / "script/setup")], setup_env)
        assert result.returncode == 91, result
        assert f"test -d '/peer-home/{relative}/.git'" in result.stdout, result
        assert not (checkout / ".deps").exists()

    command = ["bash", str(checkout / "script/setup")]
    result = run(command, {**setup_env, "FLEXMAYA_PEER_REPO_ROOT": "/explicit/peer"})
    assert result.returncode == 91 and "test -d '/explicit/peer/.git'" in result.stdout, result

    script(checkout / ".deps/ae_env.sh", "export FLEXMAYA_PEER_REPO_ROOT=/saved/peer\n")
    result = run(command, setup_env)
    assert result.returncode == 91 and "test -d '/saved/peer/.git'" in result.stdout, result

    other_home = home / "other-home"
    other_home.mkdir()
    command = ["bash", str(home / "flexeva-ae/script/setup")]
    outside_env = {**setup_env, "HOME": str(other_home)}
    result = run(command, outside_env)
    assert result.returncode == 2 and "checkout outside HOME" in result.stderr, result
    assert "git clone" not in result.stdout, result
    result = run(command, {**outside_env, "FLEXMAYA_PEER_REPO_ROOT": "/explicit/outside"})
    assert result.returncode == 91 and "test -d '/explicit/outside/.git'" in result.stdout, result


def test_e5_export(home, env):
    checkout = home / "reviewer-a/flexeva-ae"
    shutil.copy2(ROOT / "script/run_e5", checkout / "script/run_e5")
    # The stub guarded runner supplies tiny files instead of running E5.
    script(checkout / "script/e3/server.sh", '''#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
result="$root/result/server-runs/$2/results/e5-paper-aligned"
mkdir -p "$result/measurement" "$result/speed"
for name in result.json measurements.csv summary.csv table8.csv integrity.json; do
    printf 'own-checkout\\n' >"$result/measurement/$name"
done
for name in result.json samples.csv per_round.csv summary.csv integrity.json; do
    printf 'own-checkout\\n' >"$result/speed/$name"
done
''')
    result = run(["bash", str(checkout / "script/run_e5")], {
        **env, "PYTHON_BIN": "/bin/false", "E5_RUN_ID": "export-test",
        "FLEXMAYA_PEER_TARGET": "test-peer", "FLEXMAYA_PEER_PORT": "22",
    })
    assert result.returncode == 0, result
    exported = checkout / "result/e5/generated/export-test"
    assert len(list(exported.rglob("*.*"))) == 12
    assert (exported / "table8.csv").read_text() == "own-checkout\n"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="ae-shared-checkouts-") as temporary:
        home = Path(temporary)
        env = {key: value for key, value in os.environ.items() if key in ("PATH", "LANG", "LC_ALL")}
        env.update(HOME=str(home), AE_EXPECTED_FILESYSTEM=subprocess.check_output(
            ["stat", "-f", "-c", "%T", str(home)], text=True).strip())
        for check in (test_guard, test_peer_checkout, test_e5_export):
            check(home, env)
            print(f"{check.__name__}: PASS")
