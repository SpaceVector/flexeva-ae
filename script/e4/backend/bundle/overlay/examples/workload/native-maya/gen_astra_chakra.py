#!/usr/bin/env python3
"""Generate ASTRA-sim Chakra ET traces for the native Maya Megatron workload.

The generator mirrors the communication shape of maya_megatron.py without
running PyTorch or the fake-CUDA tracing path.  It emits one ET file per rank
and a communicator group JSON file for TP/DP collectives.
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


def configure_chakra_pythonpath(repo_root: Path) -> None:
    chakra_root = repo_root / "extern" / "graph_frontend"
    proto_dir = chakra_root / "chakra" / "schema" / "protobuf"
    for path in (chakra_root, proto_dir):
        sys.path.insert(0, str(path))


def ensure_chakra_proto(repo_root: Path) -> None:
    proto_dir = (
        repo_root
        / "extern"
        / "graph_frontend"
        / "chakra"
        / "schema"
        / "protobuf"
    )
    try:
        spec = importlib.util.find_spec("chakra.schema.protobuf.et_def_pb2")
        if spec is not None:
            return
    except ModuleNotFoundError:
        pass

    generated = proto_dir / "et_def_pb2.py"
    if generated.exists():
        return

    raise RuntimeError(
        "Chakra protobuf Python bindings were not found. Run "
        "./build/astra_analytical/build.sh to generate et_def_pb2.py, or set "
        "PYTHONPATH to include the generated et_def_pb2.py location and its "
        "protobuf Python dependencies."
    )


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def rank_to_coords(rank: int, dp: int, tp: int, pp: int) -> tuple[int, int, int]:
    pp_rank = rank % pp
    tp_rank = (rank // pp) % tp
    dp_rank = rank // (tp * pp)
    return dp_rank, tp_rank, pp_rank


def coords_to_rank(dp_rank: int, tp_rank: int, pp_rank: int, tp: int, pp: int) -> int:
    return dp_rank * (tp * pp) + tp_rank * pp + pp_rank


def stage_layer_count(num_layers: int, pp: int, pp_rank: int) -> int:
    base = num_layers // pp
    remainder = num_layers % pp
    return base + (1 if pp_rank < remainder else 0)


def local_micro_batch_size(global_batch_size: int, dp: int, micro_batches: int) -> int:
    return max(1, math.ceil(global_batch_size / max(dp * micro_batches, 1)))


def dtype_bytes(dtype: str) -> int:
    if dtype == "fp32":
        return 4
    if dtype == "bf16":
        return 2
    raise ValueError(f"unsupported dtype: {dtype}")


def attr(name, value, chakra_attr):
    if isinstance(value, bool):
        return chakra_attr(name=name, bool_val=value)
    if isinstance(value, int):
        return chakra_attr(name=name, int64_val=value)
    if isinstance(value, str):
        return chakra_attr(name=name, string_val=value)
    raise TypeError(f"unsupported attr value for {name}: {value!r}")


def add_node(et, encode_message, chakra_node, chakra_attr, **kwargs) -> int:
    node = chakra_node()
    node.id = kwargs["node_id"]
    node.name = kwargs["name"]
    node.type = kwargs["node_type"]
    node.duration_micros = kwargs.get("duration_micros", 0)
    node.data_deps.extend(kwargs.get("data_deps", []))
    for name, value in kwargs.get("attrs", {}).items():
        node.attr.append(attr(name, value, chakra_attr))
    encode_message(et, node)
    return node.id


@dataclass(frozen=True)
class GroupIds:
    tp: dict[tuple[int, int], int]
    dp: dict[tuple[int, int], int]
    groups: dict[str, list[int]]


@dataclass(frozen=True)
class ModelShape:
    local_micro_batch: int
    dtype_size: int
    activation_bytes: int
    logits_bytes: int
    embedding_bytes: int
    layer_param_bytes: int
    stage_param_bytes: int
    forward_ops_per_layer: int
    backward_ops_per_layer: int
    optimizer_ops: int


def build_group_ids(dp: int, tp: int, pp: int) -> GroupIds:
    next_group_id = 1
    tp_groups: dict[tuple[int, int], int] = {}
    dp_groups: dict[tuple[int, int], int] = {}
    groups: dict[str, list[int]] = {}

    if tp > 1:
        for dp_rank in range(dp):
            for pp_rank in range(pp):
                group_id = next_group_id
                next_group_id += 1
                ranks = [
                    coords_to_rank(dp_rank, tp_rank, pp_rank, tp, pp)
                    for tp_rank in range(tp)
                ]
                tp_groups[(dp_rank, pp_rank)] = group_id
                groups[str(group_id)] = ranks

    if dp > 1:
        for tp_rank in range(tp):
            for pp_rank in range(pp):
                group_id = next_group_id
                next_group_id += 1
                ranks = [
                    coords_to_rank(dp_rank, tp_rank, pp_rank, tp, pp)
                    for dp_rank in range(dp)
                ]
                dp_groups[(tp_rank, pp_rank)] = group_id
                groups[str(group_id)] = ranks

    return GroupIds(tp=tp_groups, dp=dp_groups, groups=groups)


def build_model_shape(args: argparse.Namespace, pp_rank: int) -> ModelShape:
    mb = local_micro_batch_size(args.global_batch_size, args.dp, args.micro_batches)
    elem_bytes = dtype_bytes(args.dtype)
    activation_elems = mb * args.seq_len * args.hidden_size
    activation_bytes = activation_elems * elem_bytes
    logits_bytes = mb * args.seq_len * args.vocab_size * elem_bytes

    hidden = args.hidden_size
    intermediate = hidden * 4
    local_layers = stage_layer_count(args.num_layers, args.pp, pp_rank)

    attention_params = (3 * hidden * hidden + hidden * hidden) // args.tp
    mlp_params = (hidden * intermediate + intermediate * hidden) // args.tp
    layer_param_elems = attention_params + mlp_params + 4 * hidden
    layer_param_bytes = layer_param_elems * elem_bytes
    stage_param_bytes = local_layers * layer_param_bytes
    if pp_rank == 0:
        stage_param_bytes += args.vocab_size * hidden * elem_bytes
    if pp_rank == args.pp - 1:
        stage_param_bytes += args.vocab_size * hidden * elem_bytes

    qkv_ops = 2 * activation_elems * (3 * hidden // args.tp)
    attn_score_ops = 2 * mb * (args.num_heads // args.tp) * args.seq_len * args.seq_len
    attn_value_ops = attn_score_ops
    attn_out_ops = 2 * activation_elems * hidden // args.tp
    mlp_up_ops = 2 * activation_elems * intermediate // args.tp
    mlp_down_ops = 2 * activation_elems * intermediate // args.tp
    forward_ops = qkv_ops + attn_score_ops + attn_value_ops + attn_out_ops + mlp_up_ops + mlp_down_ops
    backward_ops = 2 * forward_ops
    optimizer_ops = max(stage_param_bytes // elem_bytes * 2, 1)

    return ModelShape(
        local_micro_batch=mb,
        dtype_size=elem_bytes,
        activation_bytes=max(activation_bytes, 1),
        logits_bytes=max(logits_bytes, 1),
        embedding_bytes=max(args.vocab_size * hidden * elem_bytes, 1),
        layer_param_bytes=max(layer_param_bytes, 1),
        stage_param_bytes=max(stage_param_bytes, 1),
        forward_ops_per_layer=max(forward_ops, 1),
        backward_ops_per_layer=max(backward_ops, 1),
        optimizer_ops=max(optimizer_ops, 1),
    )


def p2p_tag(step: int, microbatch: int, src_pp_rank: int, is_backward: bool) -> int:
    direction = 1 if is_backward else 0
    return step * 1_000_000 + microbatch * 1_000 + direction * 100 + src_pp_rank


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

    def apply(
        self,
        duration: int,
        context: dict[str, int | str | None],
    ) -> int:
        for rule in self.rules:
            if not rule.matches(context):
                continue
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
        tp_rank: int,
        pp_rank: int,
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
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
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


INT_MATCH_FIELDS = {
    "ranks": "rank",
    "dp_ranks": "dp_rank",
    "tp_ranks": "tp_rank",
    "pp_ranks": "pp_rank",
    "steps": "step",
    "microbatches": "microbatch",
    "layers": "layer",
}
STRING_MATCH_FIELDS = {
    "ops": "op",
}
SUPPORTED_OPS = {
    "embedding_lookup",
    "attention_forward",
    "mlp_forward",
    "loss_forward",
    "mlp_backward",
    "attention_backward",
    "optimizer_step",
}
SUPPORTED_SET_KEYS = {"duration_micros"}


def parse_int_match_values(path: Path, key: str, value: Any) -> frozenset[int]:
    if not isinstance(value, list):
        raise SystemExit(
            f"mutation config {path}: match {key!r} must be a list"
        )
    values: set[int] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise SystemExit(
                f"mutation config {path}: match {key!r} must contain integers"
            )
        if item < 0:
            raise SystemExit(
                f"mutation config {path}: match {key!r} must be non-negative"
            )
        values.add(item)
    return frozenset(values)


def parse_ops_match_values(path: Path, key: str, value: Any) -> frozenset[str]:
    if not isinstance(value, list):
        raise SystemExit(
            f"mutation config {path}: match {key!r} must be a list"
        )
    values: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item:
            raise SystemExit(
                f"mutation config {path}: match {key!r} must contain strings"
            )
        if item not in SUPPORTED_OPS:
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
            canonical = INT_MATCH_FIELDS[key]
            filters[canonical] = parse_int_match_values(path, key, value)
        elif key in STRING_MATCH_FIELDS:
            canonical = STRING_MATCH_FIELDS[key]
            filters[canonical] = parse_ops_match_values(path, key, value)
        else:
            raise SystemExit(
                f"mutation config {path}: rule {index} has unknown match {key!r}"
            )

    raw_set = raw_rule.get("set")
    if not isinstance(raw_set, dict):
        raise SystemExit(
            f"mutation config {path}: rule {index} set must be an object"
        )
    unknown_set = set(raw_set) - SUPPORTED_SET_KEYS
    if unknown_set:
        keys = ", ".join(sorted(unknown_set))
        raise SystemExit(
            f"mutation config {path}: rule {index} has unknown set keys: {keys}"
        )
    if "duration_micros" not in raw_set:
        raise SystemExit(
            f"mutation config {path}: rule {index} set.duration_micros is required"
        )
    duration_micros = raw_set["duration_micros"]
    if isinstance(duration_micros, bool) or not isinstance(duration_micros, int):
        raise SystemExit(
            f"mutation config {path}: rule {index} duration_micros must be an integer"
        )
    if duration_micros < 0:
        raise SystemExit(
            f"mutation config {path}: rule {index} duration_micros must be non-negative"
        )

    return MutationRule(
        filters=filters,
        duration_micros=duration_micros,
    )


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
    schema_version = root.get("schema_version")
    if schema_version != "native-maya-mutation-v1":
        raise SystemExit(
            f"mutation config {path}: unsupported schema_version {schema_version!r}"
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
        et,
        encode_message,
        chakra_node,
        chakra_attr,
        args: argparse.Namespace,
        rank: int,
        dp_rank: int,
        tp_rank: int,
        pp_rank: int,
        group_ids: GroupIds,
    ) -> None:
        self.et = et
        self.encode_message = encode_message
        self.chakra_node = chakra_node
        self.chakra_attr = chakra_attr
        self.args = args
        self.rank = rank
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.group_ids = group_ids
        self.shape = build_model_shape(args, pp_rank)
        self.node_id = 1
        self.last_node: int | None = None
        self.forward_done: dict[tuple[int, int], int] = {}
        self.mutation_config: MutationConfig = args.mutation_config

    def deps(self, *extra: int | None) -> list[int]:
        deps = []
        if self.last_node is not None:
            deps.append(self.last_node)
        deps.extend(dep for dep in extra if dep is not None)
        return list(dict.fromkeys(deps))

    def add(self, name: str, node_type: int, *, data_deps=None, duration=0, attrs=None) -> int:
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
            data_deps=data_deps if data_deps is not None else self.deps(),
            duration_micros=duration,
            attrs=attrs or {},
        )
        return self.last_node

    def add_compute(
        self,
        name: str,
        num_ops: int,
        tensor_size: int,
        *deps: int | None,
        duration: int | None = None,
        step: int | None = None,
        microbatch: int | None = None,
        layer: int | None = None,
        op: str,
    ) -> int:
        baseline_duration = self.args.compute_us if duration is None else duration
        context = {
            "rank": self.rank,
            "dp_rank": self.dp_rank,
            "tp_rank": self.tp_rank,
            "pp_rank": self.pp_rank,
            "step": step,
            "microbatch": microbatch,
            "layer": layer,
            "op": op,
        }
        effective_duration = self.mutation_config.apply(
            baseline_duration,
            context,
        )
        attrs = {
            "is_cpu_op": False,
            "num_ops": num_ops,
            "tensor_size": tensor_size,
        }
        data_deps = self.deps(*deps)
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
                tp_rank=self.tp_rank,
                pp_rank=self.pp_rank,
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

    def add_tp_all_reduce(self, name: str, comm_size: int) -> int | None:
        if self.args.tp <= 1:
            return None
        group_id = self.group_ids.tp[(self.dp_rank, self.pp_rank)]
        return self.add(
            name,
            self.args.COMM_COLL_NODE,
            attrs={
                "is_cpu_op": False,
                "comm_type": self.args.ALL_REDUCE,
                "comm_size": comm_size,
                "pg_name": str(group_id),
            },
        )

    def add_dp_all_reduce(self, name: str, comm_size: int) -> int | None:
        if self.args.dp <= 1:
            return None
        group_id = self.group_ids.dp[(self.tp_rank, self.pp_rank)]
        return self.add(
            name,
            self.args.COMM_COLL_NODE,
            attrs={
                "is_cpu_op": False,
                "comm_type": self.args.ALL_REDUCE,
                "comm_size": comm_size,
                "pg_name": str(group_id),
            },
        )

    def add_send(self, name: str, dst_rank: int, size: int, tag: int) -> int:
        return self.add(
            name,
            self.args.COMM_SEND_NODE,
            attrs={
                "is_cpu_op": False,
                "comm_size": size,
                "comm_src": self.rank,
                "comm_dst": dst_rank,
                "comm_tag": tag,
            },
        )

    def add_recv(self, name: str, src_rank: int, size: int, tag: int) -> int:
        return self.add(
            name,
            self.args.COMM_RECV_NODE,
            attrs={
                "is_cpu_op": False,
                "comm_size": size,
                "comm_src": src_rank,
                "comm_dst": self.rank,
                "comm_tag": tag,
            },
        )

    def forward_microbatch(self, step: int, microbatch: int) -> None:
        if self.pp_rank > 0:
            src_rank = coords_to_rank(
                self.dp_rank, self.tp_rank, self.pp_rank - 1, self.args.tp, self.args.pp
            )
            self.add_recv(
                f"step{step}.mb{microbatch}.recv_activation",
                src_rank,
                self.shape.activation_bytes,
                p2p_tag(step, microbatch, self.pp_rank - 1, False),
            )
        elif step == 0 and microbatch == 0:
            self.add_compute(
                "embedding_lookup",
                self.shape.local_micro_batch * self.args.seq_len,
                self.shape.embedding_bytes,
                step=step,
                microbatch=microbatch,
                op="embedding_lookup",
            )

        local_layers = stage_layer_count(self.args.num_layers, self.args.pp, self.pp_rank)
        for layer in range(local_layers):
            self.add_compute(
                f"step{step}.mb{microbatch}.layer{layer}.attention_forward",
                self.shape.forward_ops_per_layer // 2,
                self.shape.activation_bytes,
                step=step,
                microbatch=microbatch,
                layer=layer,
                op="attention_forward",
            )
            self.add_tp_all_reduce(
                f"step{step}.mb{microbatch}.layer{layer}.attention_output_all_reduce",
                self.shape.activation_bytes,
            )
            self.add_compute(
                f"step{step}.mb{microbatch}.layer{layer}.mlp_forward",
                self.shape.forward_ops_per_layer // 2,
                self.shape.activation_bytes,
                step=step,
                microbatch=microbatch,
                layer=layer,
                op="mlp_forward",
            )
            self.add_tp_all_reduce(
                f"step{step}.mb{microbatch}.layer{layer}.mlp_output_all_reduce",
                self.shape.activation_bytes,
            )

        if self.pp_rank == self.args.pp - 1:
            self.add_compute(
                f"step{step}.mb{microbatch}.loss_forward",
                self.shape.local_micro_batch * self.args.seq_len * self.args.vocab_size,
                self.shape.logits_bytes,
                step=step,
                microbatch=microbatch,
                op="loss_forward",
            )
        else:
            dst_rank = coords_to_rank(
                self.dp_rank, self.tp_rank, self.pp_rank + 1, self.args.tp, self.args.pp
            )
            self.add_send(
                f"step{step}.mb{microbatch}.send_activation",
                dst_rank,
                self.shape.activation_bytes,
                p2p_tag(step, microbatch, self.pp_rank, False),
            )
        self.forward_done[(step, microbatch)] = self.last_node or 0

    def backward_microbatch(self, step: int, microbatch: int) -> None:
        if self.pp_rank < self.args.pp - 1:
            src_rank = coords_to_rank(
                self.dp_rank, self.tp_rank, self.pp_rank + 1, self.args.tp, self.args.pp
            )
            self.add_recv(
                f"step{step}.mb{microbatch}.recv_grad",
                src_rank,
                self.shape.activation_bytes,
                p2p_tag(step, microbatch, self.pp_rank + 1, True),
            )

        forward_dep = self.forward_done[(step, microbatch)]
        local_layers = stage_layer_count(self.args.num_layers, self.args.pp, self.pp_rank)
        for layer in reversed(range(local_layers)):
            self.add_compute(
                f"step{step}.mb{microbatch}.layer{layer}.mlp_backward",
                self.shape.backward_ops_per_layer // 2,
                self.shape.activation_bytes,
                forward_dep,
                step=step,
                microbatch=microbatch,
                layer=layer,
                op="mlp_backward",
            )
            self.add_compute(
                f"step{step}.mb{microbatch}.layer{layer}.attention_backward",
                self.shape.backward_ops_per_layer // 2,
                self.shape.activation_bytes,
                step=step,
                microbatch=microbatch,
                layer=layer,
                op="attention_backward",
            )

        if self.pp_rank > 0:
            dst_rank = coords_to_rank(
                self.dp_rank, self.tp_rank, self.pp_rank - 1, self.args.tp, self.args.pp
            )
            self.add_send(
                f"step{step}.mb{microbatch}.send_grad",
                dst_rank,
                self.shape.activation_bytes,
                p2p_tag(step, microbatch, self.pp_rank, True),
            )

    def finish_step(self, step: int) -> None:
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

    def generate(self) -> None:
        for step in range(self.args.steps):
            if self.args.schedule == "gpipe":
                for microbatch in range(self.args.micro_batches):
                    self.forward_microbatch(step, microbatch)
                for microbatch in range(self.args.micro_batches):
                    self.backward_microbatch(step, microbatch)
            else:
                warmup = min(self.args.pp - self.pp_rank - 1, self.args.micro_batches)
                queue: list[int] = []
                for microbatch in range(warmup):
                    self.forward_microbatch(step, microbatch)
                    queue.append(microbatch)
                for microbatch in range(warmup, self.args.micro_batches):
                    self.forward_microbatch(step, microbatch)
                    queue.append(microbatch)
                    self.backward_microbatch(step, queue.pop(0))
                while queue:
                    self.backward_microbatch(step, queue.pop(0))
            self.finish_step(step)


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        "steps",
        "global_batch_size",
        "seq_len",
        "hidden_size",
        "num_layers",
        "num_heads",
        "vocab_size",
        "tp",
        "pp",
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
    if args.num_heads % args.tp != 0:
        raise SystemExit("--num-heads must be divisible by --tp")
    if args.hidden_size % args.tp != 0:
        raise SystemExit("--hidden-size must be divisible by --tp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ASTRA-sim Chakra ET traces for native Maya Megatron"
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--comm-group-output",
        help="Write TP/DP communicator groups to this JSON path.",
    )
    parser.add_argument("--repo-root", default=str(repo_root_from_script()))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=12_000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--pp", type=int, default=8)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--micro-batches", type=int, default=64)
    parser.add_argument("--schedule", choices=["1f1b", "gpipe"], default="1f1b")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--pipeline-p2p-mode",
        choices=["blocking", "async", "batch"],
        default="blocking",
        help="Accepted for CLI parity with maya_megatron.py; ET shape is unchanged.",
    )
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
            "Write native-maya-changed-events-v1 records for nodes whose "
            "generated payload changed under --mutation-config."
        ),
    )
    return parser.parse_args()


def generate(args: argparse.Namespace) -> None:
    validate_args(args)
    args.mutation_config = load_mutation_config(
        getattr(args, "mutation_config", None)
    )
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
        COMM_COLL_NODE,
        COMM_RECV_NODE,
        COMM_SEND_NODE,
        COMP_NODE,
        AttributeProto as ChakraAttr,
        GlobalMetadata,
        Node as ChakraNode,
    )
    from chakra.src.third_party.utils.protolib import (  # pylint: disable=import-error
        encodeMessage as encode_message,
    )

    args.ALL_REDUCE = ALL_REDUCE
    args.COMM_COLL_NODE = COMM_COLL_NODE
    args.COMM_RECV_NODE = COMM_RECV_NODE
    args.COMM_SEND_NODE = COMM_SEND_NODE
    args.COMP_NODE = COMP_NODE

    group_ids = build_group_ids(args.dp, args.tp, args.pp)
    if comm_group_output is not None:
        comm_group_output.write_text(
            json.dumps(group_ids.groups, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    world_size = args.dp * args.tp * args.pp
    for rank in range(world_size):
        dp_rank, tp_rank, pp_rank = rank_to_coords(rank, args.dp, args.tp, args.pp)
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
                tp_rank=tp_rank,
                pp_rank=pp_rank,
                group_ids=group_ids,
            )
            builder.generate()

    mutation_events_output = getattr(args, "mutation_events_output", None)
    if mutation_events_output:
        output_path = Path(mutation_events_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "native-maya-changed-events-v1",
            "source": "native-maya-generator",
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
