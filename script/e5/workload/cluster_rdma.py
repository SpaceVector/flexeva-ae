from __future__ import annotations

import ipaddress
import os
import subprocess


def _physical_gpu_index(local_rank: int) -> int:
    if local_rank < 0:
        return local_rank
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return local_rank
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if local_rank >= len(entries):
        return local_rank
    try:
        return int(entries[local_rank])
    except ValueError:
        return local_rank


def _gid_index_to_ipv4(gid_text: str) -> str | None:
    parts = gid_text.strip().split(":")
    if len(parts) < 2:
        return None
    try:
        packed = bytes.fromhex("".join(parts[-2:]))
    except ValueError:
        return None
    if len(packed) != 4:
        return None
    return str(ipaddress.IPv4Address(packed))


def _iface_for_ipv4(ipv4_addr: str) -> str | None:
    try:
        output = subprocess.check_output(
            ["ip", "-o", "-4", "addr", "show"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        iface = parts[1]
        cidr = parts[3]
        address = cidr.split("/", 1)[0]
        if address == ipv4_addr:
            return iface
    return None


def maybe_apply_cluster_cpu_affinity(local_rank: int) -> list[int] | None:
    enabled = os.environ.get("FLEXSIM_CLUSTER_CPU_AFFINITY", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    if local_rank < 0:
        return None
    physical_gpu = _physical_gpu_index(local_rank)
    if physical_gpu % 8 < 4:
        cpus = list(range(0, 32)) + list(range(64, 96))
    else:
        cpus = list(range(32, 64)) + list(range(96, 109))
    try:
        os.sched_setaffinity(0, cpus)
    except (AttributeError, OSError):
        return None
    return cpus


def maybe_apply_cluster_rdma_affinity(
    local_rank: int,
    *,
    set_socket_ifname: bool | None = None,
) -> tuple[str | None, str | None]:
    enabled = os.environ.get("FLEXSIM_CLUSTER_RDMA_AFFINITY", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None, None
    if local_rank < 0:
        return None, None

    if set_socket_ifname is None:
        socket_setting = os.environ.get("FLEXSIM_CLUSTER_RDMA_SET_SOCKET_IFNAME", "")
        set_socket_ifname = socket_setting.strip().lower() in {"1", "true", "yes", "on"}

    physical_gpu = _physical_gpu_index(local_rank)
    hca_index = 1 + (physical_gpu % 8) // 2
    hca_name = f"mlx5_{hca_index}"
    hca_port = f"{hca_name}:1"
    os.environ["NCCL_IB_HCA"] = f"={hca_port}"
    # The dual-node RoCE setup exposes IPv4-mapped GIDs at indices 6 and 7;
    # index 7 is RoCE v2 and is the working NCCL path for cross-node traffic.
    os.environ.setdefault("NCCL_IB_GID_INDEX", "7")
    os.environ.setdefault("NCCL_CROSS_NIC", "0")
    os.environ.setdefault("NCCL_NET_GDR_LEVEL", "0")
    oob_hca = os.environ.get("FLEXSIM_CLUSTER_RDMA_SET_OOB_HCA", "").strip().lower()
    if oob_hca in {"1", "true", "yes", "on"}:
        os.environ["NCCL_NET"] = "IB"
        os.environ["NCCL_OOB_NET_ENABLE"] = "1"
        os.environ["NCCL_OOB_NET_IFNAME"] = f"={hca_port}"

    gid_path = f"/sys/class/infiniband/{hca_name}/ports/1/gids/6"
    rdma_iface = None
    try:
        with open(gid_path, "r", encoding="utf-8") as handle:
            gid_text = handle.read().strip()
    except OSError:
        gid_text = ""
    ipv4_addr = _gid_index_to_ipv4(gid_text)
    if ipv4_addr:
        rdma_iface = _iface_for_ipv4(ipv4_addr)
    if rdma_iface and set_socket_ifname:
        os.environ["NCCL_SOCKET_IFNAME"] = rdma_iface
    return hca_name, rdma_iface
