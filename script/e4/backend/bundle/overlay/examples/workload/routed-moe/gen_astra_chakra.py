#!/usr/bin/env python3
"""Generate ASTRA-sim Chakra ET traces for routed-MoE training.

This generator models the communication shape of ``moe_topk.py`` directly for
ASTRA-sim/RAS. It emits one Chakra ET file per rank and an optional
communicator-group JSON file containing contiguous EP groups and strided DP
groups.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MUTATION_SCHEMA_VERSION = "native-maya-mutation-v1"
CHANGED_EVENTS_SCHEMA_VERSION = "native-maya-changed-events-v1"


SUPPORTED_OPS = {
    "attention_forward",
    "router_forward",
    "moe_dispatch_all_to_all",
    "expert_forward",
    "moe_combine_all_to_all",
    "dense_mlp_forward",
    "loss_forward",
    "dense_mlp_backward",
    "expert_backward",
    "router_backward",
    "attention_backward",
    "optimizer_step",
}
SUPPORTED_SET_KEYS = {"duration_micros"}
INT_MATCH_FIELDS = {
    "ranks": "rank",
    "dp_ranks": "dp_rank",
    "ep_ranks": "rank_in_ep",
    "steps": "step",
    "microbatches": "microbatch",
    "layers": "layer",
}
STRING_MATCH_FIELDS = {"ops": "op"}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def configure_chakra_pythonpath(repo_root: Path) -> None:
    chakra_root = repo_root / "extern" / "graph_frontend"
    proto_dir = chakra_root / "chakra" / "schema" / "protobuf"
    for path in (repo_root, chakra_root, proto_dir):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def ensure_chakra_proto(repo_root: Path) -> None:
    configure_chakra_pythonpath(repo_root)
    try:
        spec = importlib.util.find_spec("chakra.schema.protobuf.et_def_pb2")
    except ModuleNotFoundError:
        spec = None
    if spec is not None:
        return

    generated = (
        repo_root
        / "extern"
        / "graph_frontend"
        / "chakra"
        / "schema"
        / "protobuf"
        / "et_def_pb2.py"
    )
    if generated.exists():
        return

    raise RuntimeError(
        "Chakra protobuf Python bindings were not found. Run "
        "./build/astra_analytical/build.sh first or set PYTHONPATH to include "
        "et_def_pb2.py and its protobuf dependencies."
    )


def dtype_bytes(dtype: str) -> int:
    if dtype == "fp32":
        return 4
    if dtype == "bf16":
        return 2
    raise ValueError(f"unsupported dtype: {dtype}")


def local_micro_batch_size(global_batch_size: int, dp: int, micro_batches: int) -> int:
    return max(1, math.ceil(global_batch_size / max(dp * micro_batches, 1)))


def attr(name: str, value: Any, chakra_attr: Any) -> Any:
    if isinstance(value, bool):
        return chakra_attr(name=name, bool_val=value)
    if isinstance(value, int):
        return chakra_attr(name=name, int64_val=value)
    if isinstance(value, str):
        return chakra_attr(name=name, string_val=value)
    raise TypeError(f"unsupported attr value for {name}: {value!r}")


def add_node(
    et: Any,
    encode_message: Any,
    chakra_node: Any,
    chakra_attr: Any,
    *,
    node_id: int,
    name: str,
    node_type: int,
    duration_micros: int = 0,
    data_deps: list[int] | None = None,
    attrs: dict[str, Any] | None = None,
) -> int:
    node = chakra_node()
    node.id = node_id
    node.name = name
    node.type = node_type
    node.duration_micros = duration_micros
    node.data_deps.extend(data_deps or [])
    for attr_name, value in (attrs or {}).items():
        node.attr.append(attr(attr_name, value, chakra_attr))
    encode_message(et, node)
    return node.id


@dataclass(frozen=True)
class GroupIds:
    ep: dict[int, int]
    dp: dict[int, int]
    groups: dict[str, list[int]]


def build_group_ids(ep_size: int, dp: int) -> GroupIds:
    world_size = ep_size * dp
    next_group_id = 1
    ep_groups: dict[int, int] = {}
    dp_groups: dict[int, int] = {}
    groups: dict[str, list[int]] = {}

    for start in range(0, world_size, ep_size):
        group_id = next_group_id
        next_group_id += 1
        ranks = list(range(start, start + ep_size))
        ep_groups[start // ep_size] = group_id
        groups[str(group_id)] = ranks

    if dp > 1:
        for offset in range(ep_size):
            group_id = next_group_id
            next_group_id += 1
            ranks = list(range(offset, world_size, ep_size))
            dp_groups[offset] = group_id
            groups[str(group_id)] = ranks

    return GroupIds(ep=ep_groups, dp=dp_groups, groups=groups)


def rank_to_coords(rank: int, ep_size: int) -> tuple[int, int, int]:
    dp_rank = rank // ep_size
    rank_in_ep = rank % ep_size
    ep_group_index = dp_rank
    return dp_rank, rank_in_ep, ep_group_index


def owned_expert_count(num_experts: int, ep_size: int, rank_in_ep: int) -> int:
    base = num_experts // ep_size
    remainder = num_experts % ep_size
    return base + (1 if rank_in_ep < remainder else 0)


@dataclass(frozen=True)
class ModelShape:
    local_micro_batch: int
    dtype_size: int
    tokens_per_microbatch: int
    activation_bytes: int
    logits_bytes: int
    dispatch_bytes: int
    layer_param_bytes: int
    expert_param_bytes: int
    stage_param_bytes: int
    attention_ops: int
    router_ops: int
    expert_forward_ops: int
    dense_mlp_ops: int
    loss_ops: int
    optimizer_ops: int


def build_model_shape(args: argparse.Namespace, rank_in_ep: int) -> ModelShape:
    mb = local_micro_batch_size(args.global_batch_size, args.dp, args.micro_batches)
    elem_bytes = dtype_bytes(args.dtype)
    tokens = mb * args.seq_len
    hidden = args.hidden_size
    intermediate = hidden * 4
    topk_tokens = tokens * args.top_k
    capacity_per_expert = math.ceil(args.capacity_factor * topk_tokens / args.num_experts)
    routed_tokens = min(topk_tokens, capacity_per_expert * args.num_experts)
    local_experts = owned_expert_count(args.num_experts, args.ep_size, rank_in_ep)
    local_expert_tokens = min(routed_tokens, capacity_per_expert * local_experts)

    activation_elems = tokens * hidden
    activation_bytes = activation_elems * elem_bytes
    dispatch_bytes = routed_tokens * hidden * elem_bytes
    logits_bytes = tokens * args.vocab_size * elem_bytes

    head_dim = hidden // args.num_heads
    qkv_ops = 2 * tokens * hidden * 3 * hidden
    attn_score_ops = 2 * mb * args.num_heads * args.seq_len * args.seq_len * head_dim
    attn_value_ops = attn_score_ops
    attn_out_ops = 2 * tokens * hidden * hidden
    attention_ops = qkv_ops + attn_score_ops + attn_value_ops + attn_out_ops
    router_ops = 2 * tokens * hidden * args.num_experts
    expert_forward_ops = 4 * local_expert_tokens * hidden * intermediate
    dense_mlp_ops = 4 * tokens * hidden * intermediate
    loss_ops = tokens * args.vocab_size

    attention_params = 4 * hidden * hidden + 4 * hidden
    dense_mlp_params = 2 * hidden * intermediate + 2 * intermediate
    router_params = hidden * args.num_experts + args.num_experts
    expert_params = local_experts * (2 * hidden * intermediate + 2 * intermediate)
    layer_param_bytes = (
        attention_params + dense_mlp_params + router_params + expert_params
    ) * elem_bytes
    stage_param_bytes = args.num_layers * layer_param_bytes
    stage_param_bytes += 2 * args.vocab_size * hidden * elem_bytes

    return ModelShape(
        local_micro_batch=mb,
        dtype_size=elem_bytes,
        tokens_per_microbatch=max(tokens, 1),
        activation_bytes=max(activation_bytes, 1),
        logits_bytes=max(logits_bytes, 1),
        dispatch_bytes=max(dispatch_bytes, 1),
        layer_param_bytes=max(layer_param_bytes, 1),
        expert_param_bytes=max(expert_params * elem_bytes, 1),
        stage_param_bytes=max(stage_param_bytes, 1),
        attention_ops=max(attention_ops, 1),
        router_ops=max(router_ops, 1),
        expert_forward_ops=max(expert_forward_ops, 1),
        dense_mlp_ops=max(dense_mlp_ops, 1),
        loss_ops=max(loss_ops, 1),
        optimizer_ops=max(stage_param_bytes // elem_bytes * 2, 1),
    )


@dataclass(frozen=True)
class MutationRule:
    filters: dict[str, frozenset[int | str]]
    duration_micros: int

    def matches(self, context: dict[str, int | str | None]) -> bool:
        for key, accepted in self.filters.items():
            value = context.get(key)
            if value is None or value not in accepted:
                return False
        return True


@dataclass(frozen=True)
class MutationConfig:
    rules: tuple[MutationRule, ...]

    def apply(self, duration: int, context: dict[str, int | str | None]) -> int:
        for rule in self.rules:
            if rule.matches(context):
                duration = rule.duration_micros
        return duration


@dataclass
class MutationEventRecorder:
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        rank: int,
        dp_rank: int,
        rank_in_ep: int,
        node_id: int,
        name: str,
        node_type: int,
        data_deps: list[int],
        baseline_duration_micros: int,
        duration_micros: int,
        attrs: dict[str, Any],
        step: int | None,
        microbatch: int | None,
        layer: int | None,
        op: str,
    ) -> None:
        self.events.append(
            {
                "rank": rank,
                "dp_rank": dp_rank,
                "rank_in_ep": rank_in_ep,
                "node_id": node_id,
                "node_name": f"rank{rank}.{name}",
                "node_type": node_type,
                "data_deps": data_deps,
                "duration_micros": duration_micros,
                "baseline_duration_micros": baseline_duration_micros,
                "attrs": attrs,
                "context": {
                    "step": step,
                    "microbatch": microbatch,
                    "layer": layer,
                    "op": op,
                },
            }
        )


def parse_int_match_values(path: Path, key: str, value: Any) -> frozenset[int]:
    if not isinstance(value, list):
        raise SystemExit(f"mutation config {path}: match {key!r} must be a list")
    values: set[int] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise SystemExit(
                f"mutation config {path}: match {key!r} must contain "
                "non-negative integers"
            )
        values.add(item)
    return frozenset(values)


def parse_ops_match_values(path: Path, key: str, value: Any) -> frozenset[str]:
    if not isinstance(value, list):
        raise SystemExit(f"mutation config {path}: match {key!r} must be a list")
    values: set[str] = set()
    for item in value:
        if not isinstance(item, str) or item not in SUPPORTED_OPS:
            raise SystemExit(
                f"mutation config {path}: match {key!r} has unknown op {item!r}"
            )
        values.add(item)
    return frozenset(values)


def parse_mutation_rule(path: Path, raw_rule: Any, index: int) -> MutationRule:
    if not isinstance(raw_rule, dict):
        raise SystemExit(f"mutation config {path}: rule {index} must be an object")

    unknown_rule_keys = set(raw_rule) - {"match", "set"}
    if unknown_rule_keys:
        keys = ", ".join(sorted(unknown_rule_keys))
        raise SystemExit(
            f"mutation config {path}: rule {index} has unknown keys: {keys}"
        )

    raw_match = raw_rule.get("match", {})
    if not isinstance(raw_match, dict):
        raise SystemExit(
            f"mutation config {path}: rule {index} match must be an object"
        )
    filters: dict[str, frozenset[int | str]] = {}
    for key, value in raw_match.items():
        if key in INT_MATCH_FIELDS:
            filters[INT_MATCH_FIELDS[key]] = parse_int_match_values(path, key, value)
        elif key in STRING_MATCH_FIELDS:
            filters[STRING_MATCH_FIELDS[key]] = parse_ops_match_values(
                path, key, value
            )
        else:
            raise SystemExit(
                f"mutation config {path}: rule {index} has unknown match {key!r}"
            )

    raw_set = raw_rule.get("set")
    if not isinstance(raw_set, dict):
        raise SystemExit(f"mutation config {path}: rule {index} set must be an object")
    unknown_set = set(raw_set) - SUPPORTED_SET_KEYS
    if unknown_set:
        keys = ", ".join(sorted(unknown_set))
        raise SystemExit(
            f"mutation config {path}: rule {index} has unknown set keys: {keys}"
        )
    duration_micros = raw_set.get("duration_micros")
    if (
        isinstance(duration_micros, bool)
        or not isinstance(duration_micros, int)
        or duration_micros < 0
    ):
        raise SystemExit(
            f"mutation config {path}: rule {index} "
            "set.duration_micros must be a non-negative integer"
        )

    return MutationRule(filters=filters, duration_micros=duration_micros)


def load_mutation_config(path_text: str | MutationConfig | None) -> MutationConfig:
    if isinstance(path_text, MutationConfig):
        return path_text
    if not path_text:
        return MutationConfig(rules=())

    path = Path(path_text)
    try:
        with path.open("r", encoding="utf-8") as config_file:
            root = json.load(config_file)
    except FileNotFoundError as exc:
        raise SystemExit(f"mutation config {path}: file does not exist") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"mutation config {path}:{exc.lineno}:{exc.colno}: malformed JSON: "
            f"{exc.msg}"
        ) from exc
    except OSError as exc:
        raise SystemExit(f"mutation config {path}: unable to read: {exc}") from exc

    if not isinstance(root, dict):
        raise SystemExit(f"mutation config {path}: JSON root must be an object")
    if root.get("schema_version") != MUTATION_SCHEMA_VERSION:
        raise SystemExit(
            f"mutation config {path}: unsupported schema_version "
            f"{root.get('schema_version')!r}"
        )
    unknown_root_keys = set(root) - {"schema_version", "rules"}
    if unknown_root_keys:
        keys = ", ".join(sorted(unknown_root_keys))
        raise SystemExit(f"mutation config {path}: unknown keys: {keys}")
    raw_rules = root.get("rules")
    if not isinstance(raw_rules, list):
        raise SystemExit(f"mutation config {path}: rules must be a list")

    return MutationConfig(
        rules=tuple(
            parse_mutation_rule(path, raw_rule, index)
            for index, raw_rule in enumerate(raw_rules)
        )
    )


class RankTraceBuilder:
    def __init__(
        self,
        *,
        et: Any,
        encode_message: Any,
        chakra_node: Any,
        chakra_attr: Any,
        args: argparse.Namespace,
        rank: int,
        dp_rank: int,
        rank_in_ep: int,
        ep_group_index: int,
        group_ids: GroupIds,
    ) -> None:
        self.et = et
        self.encode_message = encode_message
        self.chakra_node = chakra_node
        self.chakra_attr = chakra_attr
        self.args = args
        self.rank = rank
        self.dp_rank = dp_rank
        self.rank_in_ep = rank_in_ep
        self.ep_group_index = ep_group_index
        self.group_ids = group_ids
        self.shape = build_model_shape(args, rank_in_ep)
        self.node_id = 1
        self.last_node: int | None = None

    def deps(self, *extra: int | None) -> list[int]:
        deps = []
        if self.last_node is not None:
            deps.append(self.last_node)
        deps.extend(dep for dep in extra if dep is not None)
        return list(dict.fromkeys(deps))

    def add(
        self,
        name: str,
        node_type: int,
        *,
        data_deps: list[int] | None = None,
        duration: int = 0,
        attrs: dict[str, Any] | None = None,
    ) -> int:
        node_id = self.node_id
        self.node_id += 1
        self.last_node = add_node(
            self.et,
            self.encode_message,
            self.chakra_node,
            self.chakra_attr,
            node_id=node_id,
            name=f"rank{self.rank}.{name}",
            node_type=node_type,
            data_deps=self.deps() if data_deps is None else data_deps,
            duration_micros=duration,
            attrs=attrs or {},
        )
        return self.last_node

    def add_compute(
        self,
        name: str,
        num_ops: int,
        tensor_size: int,
        *,
        duration: int | None = None,
        step: int | None,
        microbatch: int | None = None,
        layer: int | None = None,
        op: str,
    ) -> int:
        baseline_duration = self.args.compute_us if duration is None else duration
        context = {
            "rank": self.rank,
            "dp_rank": self.dp_rank,
            "rank_in_ep": self.rank_in_ep,
            "step": step,
            "microbatch": microbatch,
            "layer": layer,
            "op": op,
        }
        effective_duration = self.args.mutation_config.apply(
            baseline_duration, context
        )
        attrs = {
            "is_cpu_op": False,
            "num_ops": num_ops,
            "tensor_size": tensor_size,
        }
        data_deps = self.deps()
        node_id = self.add(
            name,
            self.args.COMP_NODE,
            data_deps=data_deps,
            duration=effective_duration,
            attrs=attrs,
        )
        if effective_duration != baseline_duration:
            self.args.mutation_event_recorder.record(
                rank=self.rank,
                dp_rank=self.dp_rank,
                rank_in_ep=self.rank_in_ep,
                node_id=node_id,
                name=name,
                node_type=self.args.COMP_NODE,
                data_deps=data_deps,
                baseline_duration_micros=baseline_duration,
                duration_micros=effective_duration,
                attrs=attrs,
                step=step,
                microbatch=microbatch,
                layer=layer,
                op=op,
            )
        return node_id

    def add_ep_all_to_all(self, name: str, comm_size: int) -> int:
        return self.add(
            name,
            self.args.COMM_COLL_NODE,
            attrs={
                "is_cpu_op": False,
                "comm_type": self.args.ALL_TO_ALL,
                "comm_size": comm_size,
                "pg_name": str(self.group_ids.ep[self.ep_group_index]),
            },
        )

    def add_dp_all_reduce(self, name: str, comm_size: int) -> int | None:
        if self.args.dp <= 1:
            return None
        return self.add(
            name,
            self.args.COMM_COLL_NODE,
            attrs={
                "is_cpu_op": False,
                "comm_type": self.args.ALL_REDUCE,
                "comm_size": comm_size,
                "pg_name": str(self.group_ids.dp[self.rank_in_ep]),
            },
        )

    def forward_layer(self, step: int, microbatch: int, layer: int) -> None:
        prefix = f"step{step}.mb{microbatch}.layer{layer}"
        self.add_compute(
            f"{prefix}.attention_forward",
            self.shape.attention_ops,
            self.shape.activation_bytes,
            step=step,
            microbatch=microbatch,
            layer=layer,
            op="attention_forward",
        )
        self.add_compute(
            f"{prefix}.router_forward",
            self.shape.router_ops,
            self.shape.tokens_per_microbatch * self.args.num_experts * self.shape.dtype_size,
            step=step,
            microbatch=microbatch,
            layer=layer,
            op="router_forward",
        )
        self.add_ep_all_to_all(
            f"{prefix}.moe_dispatch_all_to_all",
            self.shape.dispatch_bytes,
        )
        self.add_compute(
            f"{prefix}.expert_forward",
            self.shape.expert_forward_ops,
            self.shape.dispatch_bytes,
            step=step,
            microbatch=microbatch,
            layer=layer,
            op="expert_forward",
        )
        self.add_ep_all_to_all(
            f"{prefix}.moe_combine_all_to_all",
            self.shape.dispatch_bytes,
        )
        self.add_compute(
            f"{prefix}.dense_mlp_forward",
            self.shape.dense_mlp_ops,
            self.shape.activation_bytes,
            step=step,
            microbatch=microbatch,
            layer=layer,
            op="dense_mlp_forward",
        )

    def backward_layer(self, step: int, microbatch: int, layer: int) -> None:
        prefix = f"step{step}.mb{microbatch}.layer{layer}"
        self.add_compute(
            f"{prefix}.dense_mlp_backward",
            2 * self.shape.dense_mlp_ops,
            self.shape.activation_bytes,
            step=step,
            microbatch=microbatch,
            layer=layer,
            op="dense_mlp_backward",
        )
        self.add_ep_all_to_all(
            f"{prefix}.moe_combine_grad_all_to_all",
            self.shape.dispatch_bytes,
        )
        self.add_compute(
            f"{prefix}.expert_backward",
            2 * self.shape.expert_forward_ops,
            self.shape.dispatch_bytes,
            step=step,
            microbatch=microbatch,
            layer=layer,
            op="expert_backward",
        )
        self.add_ep_all_to_all(
            f"{prefix}.moe_dispatch_grad_all_to_all",
            self.shape.dispatch_bytes,
        )
        self.add_compute(
            f"{prefix}.router_backward",
            2 * self.shape.router_ops,
            self.shape.tokens_per_microbatch * self.args.num_experts * self.shape.dtype_size,
            step=step,
            microbatch=microbatch,
            layer=layer,
            op="router_backward",
        )
        self.add_compute(
            f"{prefix}.attention_backward",
            2 * self.shape.attention_ops,
            self.shape.activation_bytes,
            step=step,
            microbatch=microbatch,
            layer=layer,
            op="attention_backward",
        )

    def generate(self) -> None:
        for step in range(self.args.steps):
            for microbatch in range(self.args.micro_batches):
                for layer in range(self.args.num_layers):
                    self.forward_layer(step, microbatch, layer)
                self.add_compute(
                    f"step{step}.mb{microbatch}.loss_forward",
                    self.shape.loss_ops,
                    self.shape.logits_bytes,
                    step=step,
                    microbatch=microbatch,
                    op="loss_forward",
                )
                for layer in reversed(range(self.args.num_layers)):
                    self.backward_layer(step, microbatch, layer)

            self.add_dp_all_reduce(
                f"step{step}.data_parallel_gradient_all_reduce",
                self.shape.stage_param_bytes,
            )
            self.add_compute(
                f"step{step}.optimizer_step",
                self.shape.optimizer_ops,
                self.shape.stage_param_bytes,
                duration=self.args.optimizer_us,
                step=step,
                op="optimizer_step",
            )


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        "steps",
        "global_batch_size",
        "seq_len",
        "hidden_size",
        "num_layers",
        "num_heads",
        "vocab_size",
        "num_experts",
        "top_k",
        "ep_size",
        "dp",
        "micro_batches",
        "compute_us",
        "optimizer_us",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise SystemExit(f"--{field.replace('_', '-')} must be positive")
    if args.hidden_size % args.num_heads != 0:
        raise SystemExit("--hidden-size must be divisible by --num-heads")
    if args.capacity_factor <= 0:
        raise SystemExit("--capacity-factor must be positive")
    if args.top_k > args.num_experts:
        raise SystemExit("--top-k must be <= --num-experts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ASTRA-sim Chakra ET traces for routed MoE"
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--comm-group-output",
        help="Write EP/DP communicator groups to this JSON path.",
    )
    parser.add_argument("--repo-root", default=str(repo_root_from_script()))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--micro-batches", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--compute-us", type=int, default=10)
    parser.add_argument("--optimizer-us", type=int, default=5)
    parser.add_argument(
        "--mutation-config",
        help=(
            "Optional native-maya-mutation-v1 JSON file with rules that "
            "override generated compute node duration_micros."
        ),
    )
    parser.add_argument(
        "--mutation-events-output",
        help=(
            "Write native-maya-changed-events-v1 records for changed compute "
            "nodes."
        ),
    )
    return parser.parse_args()


def generate(args: argparse.Namespace) -> None:
    validate_args(args)
    args.mutation_config = load_mutation_config(args.mutation_config)
    args.mutation_event_recorder = MutationEventRecorder()

    repo_root = Path(args.repo_root).resolve()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    comm_group_output = Path(args.comm_group_output) if args.comm_group_output else None
    if comm_group_output is not None:
        comm_group_output.parent.mkdir(parents=True, exist_ok=True)

    configure_chakra_pythonpath(repo_root)
    ensure_chakra_proto(repo_root)

    from chakra.schema.protobuf.et_def_pb2 import (  # pylint: disable=import-error
        ALL_REDUCE,
        ALL_TO_ALL,
        COMM_COLL_NODE,
        COMP_NODE,
        AttributeProto as ChakraAttr,
        GlobalMetadata,
        Node as ChakraNode,
    )
    from chakra.src.third_party.utils.protolib import (  # pylint: disable=import-error
        encodeMessage as encode_message,
    )

    args.ALL_REDUCE = ALL_REDUCE
    args.ALL_TO_ALL = ALL_TO_ALL
    args.COMM_COLL_NODE = COMM_COLL_NODE
    args.COMP_NODE = COMP_NODE

    group_ids = build_group_ids(args.ep_size, args.dp)
    if comm_group_output is not None:
        comm_group_output.write_text(
            json.dumps(group_ids.groups, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    world_size = args.ep_size * args.dp
    for rank in range(world_size):
        dp_rank, rank_in_ep, ep_group_index = rank_to_coords(rank, args.ep_size)
        output_path = output_prefix.with_name(f"{output_prefix.name}.{rank}.et")
        with output_path.open("wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))
            builder = RankTraceBuilder(
                et=et,
                encode_message=encode_message,
                chakra_node=ChakraNode,
                chakra_attr=ChakraAttr,
                args=args,
                rank=rank,
                dp_rank=dp_rank,
                rank_in_ep=rank_in_ep,
                ep_group_index=ep_group_index,
                group_ids=group_ids,
            )
            builder.generate()

    if args.mutation_events_output:
        output_path = Path(args.mutation_events_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CHANGED_EVENTS_SCHEMA_VERSION,
            "source": "routed-moe-generator",
            "output_prefix": str(output_prefix),
            "world_size": world_size,
            "events": args.mutation_event_recorder.events,
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    args = parse_args()
    generate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
