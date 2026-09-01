#!/usr/bin/env python3
"""Launch a guarded 2x8-GPU run and move flat peer outputs to node 0."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=3",
)
FORWARDED_ENV = (
    "FIGURE5_REUSE_NATIVE",
    "FIGURE5_REUSE_EVAL",
    "FIGURE6_SOCKET_IFNAME",
    "FIGURE6_SOURCE_COMMIT",
    "FIGURE8_MAX_TIMING_DRIFT_REL",
    "FIGURE8_SOCKET_IFNAME",
    "FLEXSIM_CLUSTER_RDMA_AFFINITY",
    "FLEXSIM_CLUSTER_RDMA_SET_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "NCCL_IB_DISABLE",
    "NCCL_SOCKET_IFNAME",
)
PEER_ENV_MAP = (
    ("FLEXMAYA_PEER_MAYA_ROOT", "MAYA_ROOT"),
    ("FLEXMAYA_PEER_PROOT", "PROOT_BIN"),
    ("FIGURE5_PEER_ESTIMATOR_MODEL", "FIGURE5_ESTIMATOR_MODEL"),
    ("FIGURE5_PEER_LARGE_CLUSTER_ROOT", "FIGURE5_LARGE_CLUSTER_ROOT"),
    ("FIGURE5_PEER_RESULT_ROOT", "FIGURE5_RESULT_ROOT"),
    ("FIGURE5_PEER_TRACE_ROOT", "FIGURE5_TRACE_ROOT"),
)
MAX_HEADER_BYTES = 64 * 1024
MAX_TRANSFER_FILES = 64
MAX_FILE_BYTES = 1 << 40


class RunInterrupted(Exception):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"two-node: set {name}")
    return value


def valid_run_id(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is not None


def relative_entry(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise SystemExit(f"two-node: entry must stay below the checkout: {value}")
    return path


def checked_port(value: str, name: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise SystemExit(f"two-node: {name} must be an integer") from error
    if not 1 <= port <= 65535:
        raise SystemExit(f"two-node: {name} must be in [1, 65535]")
    return port


def terminate(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5.0
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(max(deadline - time.monotonic(), 0.0))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def run_pair(local_command: list[str], peer_command: list[str]) -> int:
    processes: list[subprocess.Popen[bytes]] = []
    previous_handlers: dict[signal.Signals, object] = {}

    def interrupt(signum: int, _frame: object) -> None:
        raise RunInterrupted(signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.signal(signum, interrupt)
    try:
        peer = subprocess.Popen(peer_command, start_new_session=True)
        processes.append(peer)
        time.sleep(1.0)
        if (peer_exit := peer.poll()) is not None:
            return int(peer_exit or 1)
        local = subprocess.Popen(local_command, start_new_session=True)
        processes.append(local)
        while True:
            local_exit, peer_exit = local.poll(), peer.poll()
            failed = next(
                (code for code in (local_exit, peer_exit) if code not in (None, 0)),
                None,
            )
            if failed is not None:
                terminate(processes)
                return int(failed)
            if local_exit == peer_exit == 0:
                return 0
            time.sleep(0.2)
    except RunInterrupted as error:
        terminate(processes)
        return 128 + error.signum
    except BaseException:
        terminate(processes)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def launch(args: argparse.Namespace) -> int:
    if not valid_run_id(args.run_id):
        raise SystemExit(f"two-node: invalid run id: {args.run_id}")
    entry = relative_entry(args.entry)
    local_entry = ROOT / entry
    local_server = ROOT / "script/e3/server.sh"
    if (
        not args.local_python.is_absolute()
        or not args.local_python.is_file()
        or not os.access(args.local_python, os.X_OK)
    ):
        raise SystemExit(f"two-node: local Python is not executable: {args.local_python}")
    if args.min_free_gib <= 0:
        raise SystemExit("two-node: --min-free-gib must be positive")
    if not local_entry.is_file() or not local_server.is_file():
        raise SystemExit(f"two-node: missing local entry or server runner: {local_entry}")

    master_addr = required_env("FLEXMAYA_MASTER_ADDR")
    master_port = checked_port(
        required_env("FLEXMAYA_MASTER_PORT"), "FLEXMAYA_MASTER_PORT"
    )
    control_port = checked_port(
        required_env("FLEXMAYA_CONTROL_PORT"), "FLEXMAYA_CONTROL_PORT"
    )
    peer_target = required_env("FLEXMAYA_PEER_TARGET")
    peer_port = checked_port(os.environ.get("FLEXMAYA_PEER_PORT", "22"), "FLEXMAYA_PEER_PORT")
    peer_root = Path(required_env("FLEXMAYA_PEER_REPO_ROOT"))
    peer_node_root = Path(required_env("FLEXMAYA_PEER_NODE_ROOT"))
    peer_python = Path(
        os.environ.get("FLEXMAYA_PEER_PYTHON") or peer_root / ".venv/bin/python"
    )
    if not all(path.is_absolute() for path in (peer_root, peer_node_root, peer_python)):
        raise SystemExit("two-node: peer repo, node root, and Python must be absolute paths")

    common_env = {
        "FLEXMAYA_COORDINATED": "1",
        "FLEXMAYA_CONTROL_PORT": str(control_port),
        "FLEXMAYA_MASTER_ADDR": master_addr,
        "FLEXMAYA_MASTER_PORT": str(master_port),
        "FLEXMAYA_NNODES": "2",
        "FLEXMAYA_PEER_WAIT_S": os.environ.get("FLEXMAYA_PEER_WAIT_S", "14400"),
        "FLEXMAYA_RUN_ID": args.run_id,
    }
    common_env.update(
        (name, os.environ[name]) for name in FORWARDED_ENV if name in os.environ
    )
    local_env = {
        **common_env,
        "FLEXMAYA_NODE_RANK": "0",
        "PYTHON_BIN": str(args.local_python),
    }
    peer_env = {
        **common_env,
        "FLEXMAYA_NODE_RANK": "1",
        "PYTHON_BIN": str(peer_python),
    }
    for source, destination in PEER_ENV_MAP:
        if source in os.environ:
            peer_env[destination] = os.environ[source]

    entry_args = list(args.entry_args)
    if entry_args[:1] == ["--"]:
        entry_args.pop(0)

    def guarded(
        server: Path,
        command_entry: Path,
        env: dict[str, str],
        node: int,
    ) -> list[str]:
        return [
            str(server),
            "run",
            f"{args.run_id}-node{node}",
            "8",
            "--",
            "/usr/bin/env",
            *(f"{key}={value}" for key, value in sorted(env.items())),
            str(command_entry),
            *entry_args,
        ]

    local_command = [
        "/usr/bin/env",
        f"AE_CANONICAL_PYTHON={args.local_python}",
        f"MIN_GPFS_FREE_GIB={args.min_free_gib}",
        *guarded(local_server, local_entry, local_env, 0),
    ]
    remote_command = [
        "/usr/bin/env",
        f"AE_NODE_ROOT={peer_node_root}",
        f"AE_CANONICAL_PYTHON={peer_python}",
        f"MIN_GPFS_FREE_GIB={args.min_free_gib}",
        *guarded(peer_root / "script/e3/server.sh", peer_root / entry, peer_env, 1),
    ]
    ssh = shutil.which("ssh")
    if ssh is None:
        raise SystemExit("two-node: ssh is unavailable")
    peer_command = [
        ssh,
        *SSH_OPTIONS,
        "-p",
        str(peer_port),
        peer_target,
        shlex.join(remote_command),
    ]
    print(f"two-node: starting {args.run_id} on node 0 and {peer_target}", flush=True)
    return run_pair(local_command, peer_command)


def connect(address: str, port: int, timeout_s: float) -> socket.socket:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            stream = socket.create_connection(
                (address, port), timeout=min(timeout_s, 10.0)
            )
            stream.settimeout(timeout_s)
            return stream
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out connecting to {address}:{port}")
            time.sleep(0.2)


def receive_exact(stream: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    while size:
        chunk = stream.recv(size)
        if not chunk:
            raise EOFError("peer transfer ended early")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def send_files(stream: socket.socket, paths: list[Path]) -> tuple[int, int]:
    total = 0
    for path in paths:
        size = path.stat().st_size
        header = json.dumps({"name": path.name, "size": size}).encode()
        stream.sendall(struct.pack("!I", len(header)) + header)
        with path.open("rb") as source:
            while chunk := source.read(4 * 1024 * 1024):
                stream.sendall(chunk)
        total += size
    stream.sendall(struct.pack("!I", 0))
    return len(paths), total


def receive_files(stream: socket.socket, destination: Path) -> tuple[int, int]:
    seen: set[str] = set()
    total = 0
    while header_size := struct.unpack("!I", receive_exact(stream, 4))[0]:
        if header_size > MAX_HEADER_BYTES or len(seen) >= MAX_TRANSFER_FILES:
            raise ValueError("invalid peer transfer size")
        header = json.loads(receive_exact(stream, header_size))
        name, size = str(header["name"]), int(header["size"])
        valid_name = name == "capture_manifest.json" or re.fullmatch(
            r"rank_[0-9]+(?:\.markers)?\.jsonl", name
        )
        if (
            Path(name).name != name
            or name in seen
            or valid_name is None
            or not 0 <= size <= MAX_FILE_BYTES
        ):
            raise ValueError(f"invalid peer file header: {header}")
        seen.add(name)
        temporary = destination / f".{name}.part"
        with temporary.open("wb") as output:
            remaining = size
            while remaining:
                chunk = stream.recv(min(4 * 1024 * 1024, remaining))
                if not chunk:
                    raise EOFError(f"peer file ended early: {name}")
                output.write(chunk)
                remaining -= len(chunk)
        temporary.replace(destination / name)
        total += size
    return len(seen), total


def transfer(args: argparse.Namespace) -> int:
    address = args.address
    port = checked_port(str(args.port), "port")
    timeout = float(args.timeout)
    directory = args.directory.resolve()
    if args.node_rank == 0:
        directory.mkdir(parents=True, exist_ok=False)
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((address, port))
            server.listen(1)
            server.settimeout(timeout)
            stream, _ = server.accept()
            with stream:
                stream.settimeout(timeout)
                count, size = receive_files(stream, directory)
    else:
        paths = [
            directory / "capture_manifest.json",
            *sorted(directory.glob("rank_*.jsonl")),
        ]
        if (
            not paths[0].is_file()
            or len(paths) == 1
            or any(not path.is_file() for path in paths)
        ):
            raise SystemExit(f"two-node: incomplete transfer source: {directory}")
        with connect(address, port, timeout) as stream:
            count, size = send_files(stream, paths)
    print(json.dumps({"files": count, "bytes": size, "node_rank": args.node_rank}))
    return 0


def barrier(args: argparse.Namespace) -> int:
    port = checked_port(str(args.port), "port")
    timeout = float(args.timeout)
    if args.node_rank == 0:
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.address, port))
            server.listen(1)
            server.settimeout(timeout)
            stream, _ = server.accept()
            with stream:
                stream.settimeout(timeout)
                if stream.recv(1) != b"1":
                    raise EOFError("peer barrier ended early")
                stream.sendall(b"1")
    else:
        with connect(args.address, port, timeout) as stream:
            stream.sendall(b"1")
            if stream.recv(1) != b"1":
                raise EOFError("coordinator barrier ended early")
    return 0


def self_test() -> int:
    assert valid_run_id("figure6-production-01") and not valid_run_id("../bad")
    assert relative_entry("script/run_e3") == Path("script/run_e3")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source, destination = root / "source", root / "destination"
        source.mkdir()
        destination.mkdir()
        paths = [
            source / "capture_manifest.json",
            source / "rank_8.jsonl",
            source / "rank_8.markers.jsonl",
        ]
        paths[0].write_text('{"ok": true}\n')
        paths[1].write_bytes(b"trace\0data")
        paths[2].write_text('{"marker": true}\n')
        left, right = socket.socketpair()
        sender = threading.Thread(target=send_files, args=(left, paths))
        sender.start()
        count, size = receive_files(right, destination)
        sender.join()
        left.close()
        right.close()
        assert count == 3 and size == sum(path.stat().st_size for path in paths)
        assert all((destination / path.name).read_bytes() == path.read_bytes() for path in paths)
    assert run_pair(["/bin/true"], ["/bin/sleep", "1.2"]) == 0
    assert run_pair(["/bin/sleep", "10"], ["/bin/false"]) == 1
    print("two-node launcher self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--run-id", required=True)
    launch_parser.add_argument("--entry", required=True)
    launch_parser.add_argument("--local-python", type=Path, required=True)
    launch_parser.add_argument("--min-free-gib", type=int, default=20)
    launch_parser.add_argument("entry_args", nargs=argparse.REMAINDER)

    for action in ("barrier", "transfer"):
        current = subparsers.add_parser(action)
        current.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
        current.add_argument("--address", required=True)
        current.add_argument("--port", type=int, required=True)
        current.add_argument("--timeout", type=float, default=14400.0)
        if action == "transfer":
            current.add_argument("--directory", type=Path, required=True)

    subparsers.add_parser("self-test")
    args = parser.parse_args()
    actions = {
        "launch": launch,
        "barrier": barrier,
        "transfer": transfer,
        "self-test": lambda _: self_test(),
    }
    return actions[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
