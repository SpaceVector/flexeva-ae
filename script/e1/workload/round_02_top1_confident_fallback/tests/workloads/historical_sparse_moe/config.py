from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass
class HistoricalSparseMoEModelConfig:
    vocab_size: int = 50304
    sequence_length: int = 2048
    num_layers: int = 24
    hidden_size: int = 1024
    num_attention_heads: int = 16
    ffw_multiplier: int = 4
    dropout: float = 0.0
    moe_every_other_ffn: bool = True
    num_experts: int = 128
    top_k: int = 2
    router_aux_loss_coef: float = 0.01
    capacity_factor_train: float = 1.25
    capacity_factor_eval: float = 2.0
    min_capacity: int = 4
    drop_tokens: bool = True
    init_std: float = 0.014
    historical_reference: str = "gshard_style_top2_padded_dispatch"


@dataclass
class HistoricalSparseMoEOptimizationConfig:
    micro_batch_size_per_gpu: int = 1
    global_batch_size: int = 16
    train_iters: int = 10
    learning_rate: float = 1.2e-4
    min_learning_rate: float = 1.0e-6
    lr_decay_style: str = "cosine"
    lr_warmup_tokens: int = 375_000_000
    lr_decay_tokens: int = 300_000_000_000
    train_tokens: int = 300_000_000_000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    clip_grad: float = 1.0
    fp16: bool = True
    activation_checkpoint: bool = False
    log_interval: int = 1
    eval_interval: int = 100
    eval_iters: int = 10
    save_interval: int = 10_000


@dataclass
class SyntheticDataConfig:
    seed: int = 1234
    split: str = "98,2,0"
    vocab_file: str = "<synthetic>"
    merge_file: str = "<synthetic>"
    dataset_note: str = "Fixed-seed synthetic token stream for historically grounded sparse-MoE experiments"


@dataclass
class RuntimeConfig:
    nodes: int = 16
    gpus_per_node: int = 8
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 128
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    distributed_backend: str = "nccl"


@dataclass
class HistoricalSparseMoEWorkloadConfig:
    target_id: str
    model: HistoricalSparseMoEModelConfig
    optimization: HistoricalSparseMoEOptimizationConfig
    data: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def world_size(self) -> int:
        return self.runtime.nodes * self.runtime.gpus_per_node

    @property
    def model_parallel_size(self) -> int:
        return self.runtime.tensor_parallel_size * self.runtime.pipeline_parallel_size

    @property
    def expert_parallel_size(self) -> int:
        return self.runtime.expert_parallel_size

    @property
    def moe_layers_expected(self) -> int:
        if self.model.moe_every_other_ffn:
            return self.model.num_layers // 2
        return self.model.num_layers

    @property
    def num_local_experts(self) -> int:
        return self.model.num_experts // self.expert_parallel_size

    @property
    def data_parallel_size(self) -> int:
        denom = self.model_parallel_size * self.expert_parallel_size
        return self.world_size // denom

    @property
    def gradient_accumulation_steps(self) -> int:
        denom = self.optimization.micro_batch_size_per_gpu * self.data_parallel_size
        return self.optimization.global_batch_size // denom

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HistoricalSparseMoEWorkloadConfig":
        return cls(
            target_id=payload["target_id"],
            model=HistoricalSparseMoEModelConfig(**payload["model"]),
            optimization=HistoricalSparseMoEOptimizationConfig(**payload["optimization"]),
            data=SyntheticDataConfig(**payload["data"]),
            runtime=RuntimeConfig(**payload["runtime"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "HistoricalSparseMoEWorkloadConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "model": asdict(self.model),
            "optimization": asdict(self.optimization),
            "data": asdict(self.data),
            "runtime": asdict(self.runtime),
        }

    def validate(self) -> None:
        if self.model.hidden_size % self.model.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.model.top_k != 2:
            raise ValueError("historical sparse-MoE baseline is defined for top-2 gating")
        if self.model.num_experts % self.expert_parallel_size != 0:
            raise ValueError("num_experts must be divisible by expert_parallel_size")
        if self.num_local_experts < 1:
            raise ValueError("num_local_experts must be at least 1")
        denom = self.model_parallel_size * self.expert_parallel_size
        if self.world_size % denom != 0:
            raise ValueError("world_size must be divisible by TP * PP * EP")
        if self.data_parallel_size != 1:
            raise ValueError("historical sparse-MoE workload currently targets DP=1 for controllability")
        denom = self.optimization.micro_batch_size_per_gpu * self.data_parallel_size
        if self.optimization.global_batch_size % denom != 0:
            raise ValueError("global_batch_size must be divisible by micro_batch_size_per_gpu * DP")

    def derived_summary(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "model_parallel_size": self.model_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "num_local_experts": self.num_local_experts,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "moe_layers_expected": self.moe_layers_expected,
            "historical_reference": self.model.historical_reference,
        }

    def make_smoke_variant(self) -> "HistoricalSparseMoEWorkloadConfig":
        return HistoricalSparseMoEWorkloadConfig(
            target_id=f"{self.target_id}_smoke",
            model=replace(
                self.model,
                sequence_length=128,
                num_layers=4,
                hidden_size=256,
                num_attention_heads=8,
                num_experts=8,
            ),
            optimization=replace(
                self.optimization,
                micro_batch_size_per_gpu=2,
                global_batch_size=2,
                train_iters=2,
                fp16=False,
            ),
            data=self.data,
            runtime=replace(
                self.runtime,
                nodes=1,
                gpus_per_node=1,
                expert_parallel_size=1,
                distributed_backend="gloo",
            ),
        )
