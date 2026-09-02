#!/usr/bin/env python3
"""Pure PyTorch Table 4 style model workloads for Maya fake-cuda capture.

The goal is workload generality coverage, not exact upstream reference
implementations.  The four model families are dependency-free PyTorch
modules selected from the models Maya reports for PyTorch Table 4 validation:
ResNet, ViT, BERT, and Llama.  They exercise representative CNN, vision
transformer, encoder-LM, and decoder-LM behavior while keeping the entrypoint
usable under the repo's fake-cuda path without torchvision, transformers, or
DeepSpeed.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from flexsim.maya_lite.markers import step_window
except Exception:  # pragma: no cover - allows standalone PyTorch smoke runs.
    def step_window(step: int | None = None, *, label: str = "training_step"):
        del step, label
        return nullcontext()


_DIST_TIMEOUT = timedelta(hours=2)


@dataclass(frozen=True)
class WorkloadConfig:
    batch_size: int
    image_size: int
    seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int
    vocab_size: int
    num_classes: int
    patch_size: int
    resnet_layers: tuple[int, int, int, int]
    resnet_width: int
    mlp_ratio: float


_SMOKE = {
    "resnet": WorkloadConfig(2, 64, 64, 128, 2, 4, 4096, 1000, 16, (1, 1, 1, 1), 32, 4.0),
    "vit": WorkloadConfig(2, 64, 64, 128, 2, 4, 4096, 1000, 16, (1, 1, 1, 1), 32, 4.0),
    "bert": WorkloadConfig(2, 64, 64, 128, 2, 4, 8192, 2, 16, (1, 1, 1, 1), 32, 4.0),
    "llama": WorkloadConfig(2, 64, 64, 128, 2, 4, 8192, 2, 16, (1, 1, 1, 1), 32, 4.0),
}

_TABLE4_LITE = {
    "resnet": WorkloadConfig(8, 128, 128, 256, 4, 8, 16384, 1000, 16, (2, 2, 2, 2), 48, 4.0),
    "vit": WorkloadConfig(4, 128, 128, 384, 6, 6, 16384, 1000, 16, (2, 2, 2, 2), 48, 4.0),
    "bert": WorkloadConfig(4, 256, 256, 512, 6, 8, 30522, 2, 16, (2, 2, 2, 2), 48, 4.0),
    "llama": WorkloadConfig(2, 256, 256, 512, 8, 8, 32000, 2, 16, (2, 2, 2, 2), 48, 4.0),
}

_TABLE4_SHAPE = {
    "resnet": WorkloadConfig(16, 224, 128, 512, 8, 8, 32000, 1000, 16, (3, 8, 36, 3), 64, 4.0),
    "vit": WorkloadConfig(8, 224, 128, 768, 12, 12, 32000, 1000, 16, (3, 4, 6, 3), 64, 4.0),
    "bert": WorkloadConfig(8, 224, 512, 768, 12, 12, 30522, 2, 16, (3, 4, 6, 3), 64, 4.0),
    "llama": WorkloadConfig(1, 224, 2048, 4096, 32, 32, 32000, 2, 16, (3, 4, 6, 3), 64, 2.6875),
}

_PRESETS = {
    "smoke": _SMOKE,
    "table4-lite": _TABLE4_LITE,
    "table4-shape": _TABLE4_SHAPE,
}
_MODEL_CHOICES = ("resnet", "vit", "bert", "llama")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch Table 4 model workloads")
    parser.add_argument("--model", choices=_MODEL_CHOICES, required=True)
    parser.add_argument("--preset", choices=sorted(_PRESETS), default="smoke")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--resnet-layers", default=None, help="Comma-separated ResNet stage depths, e.g. 3,8,36,3.")
    parser.add_argument("--resnet-width", type=int, default=None)
    parser.add_argument("--mlp-ratio", type=float, default=None)
    parser.add_argument("--parallel", choices=["auto", "none", "ddp", "fsdp"], default="auto")
    parser.add_argument("--compile", action="store_true", help="Apply torch.compile before distributed wrapping.")
    parser.add_argument(
        "--compile-backend",
        default="inductor",
        help="Backend passed to torch.compile. The default is PyTorch's inductor backend.",
    )
    parser.add_argument(
        "--activation-checkpoint",
        action="store_true",
        help="Checkpoint ViT transformer blocks during the forward pass.",
    )
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--cudnn-mode",
        choices=["default", "disabled"],
        default="default",
        help="Diagnostic backend mode. 'disabled' turns off torch.backends.cudnn without changing the model.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--sync-before-step-window", action="store_true")
    parser.add_argument("--no-step-end-synchronize", action="store_true")
    return parser.parse_args()


def _parse_resnet_layers(value: str | None, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if value is None:
        return default
    parts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(parts) != 4 or any(item <= 0 for item in parts):
        raise ValueError("--resnet-layers must contain four positive integers")
    return parts


def resolve_config(args: argparse.Namespace) -> WorkloadConfig:
    base = _PRESETS[args.preset][args.model]
    return WorkloadConfig(
        batch_size=args.batch_size or base.batch_size,
        image_size=args.image_size or base.image_size,
        seq_len=args.seq_len or base.seq_len,
        hidden_size=args.hidden_size or base.hidden_size,
        num_layers=args.num_layers or base.num_layers,
        num_heads=args.num_heads or base.num_heads,
        vocab_size=args.vocab_size or base.vocab_size,
        num_classes=args.num_classes or base.num_classes,
        patch_size=args.patch_size or base.patch_size,
        resnet_layers=_parse_resnet_layers(args.resnet_layers, base.resnet_layers),
        resnet_width=args.resnet_width or base.resnet_width,
        mlp_ratio=args.mlp_ratio or base.mlp_ratio,
    )


def runtime_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.bfloat16 if args.dtype == "bf16" else torch.float32


def configure_cudnn_backend(mode: str) -> bool | None:
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is None:
        return None
    if mode == "disabled":
        cudnn.enabled = False
    elif mode != "default":
        raise ValueError(f"unsupported cuDNN mode: {mode}")
    return bool(cudnn.enabled)


def setup_dist(force: bool = False) -> tuple[int, int, int, torch.device, bool]:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    should_init = force or env_world_size > 1
    if should_init:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", str(env_world_size))
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    initialized = False
    if should_init and not dist.is_initialized():
        if device.type == "cuda":
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                device_id=device,
                timeout=_DIST_TIMEOUT,
            )
        else:
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                timeout=_DIST_TIMEOUT,
            )
        initialized = True

    return rank, world_size, local_rank, device, initialized


def cleanup_dist(initialized: bool) -> None:
    if initialized and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def synchronize_completed_iteration(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def synchronize_before_step_window() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, width: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = width * self.expansion
        self.conv1 = _conv1x1(in_channels, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = _conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = _conv1x1(width, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(_conv1x1(in_channels, out_channels, stride), nn.BatchNorm2d(out_channels))
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        return F.relu(out + identity, inplace=True)


class ResNetWorkload(nn.Module):
    def __init__(self, layers: tuple[int, int, int, int], width: int, num_classes: int) -> None:
        super().__init__()
        if len(layers) != 4:
            raise ValueError("ResNet requires four stage depths")
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_channels = width
        self.layer1 = self._make_layer(width, layers[0], stride=1)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 8 * BottleneckBlock.expansion, num_classes)

    def _make_layer(self, width: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BottleneckBlock(self.in_channels, width, stride)]
        self.in_channels = width * BottleneckBlock.expansion
        layers.extend(BottleneckBlock(self.in_channels, width) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, inner)
        self.fc2 = nn.Linear(inner, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def _rotate_half_pairs(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _apply_rotary_position_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = (q.size(-1) // 2) * 2
    if rotary_dim == 0:
        return q, k
    positions = torch.arange(q.size(-2), device=q.device, dtype=torch.float32)
    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(0, rotary_dim, 2, device=q.device, dtype=torch.float32)
            / rotary_dim
        )
    )
    freqs = torch.outer(positions, inv_freq)
    cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1).to(dtype=q.dtype)
    sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1).to(dtype=q.dtype)
    cos = cos.view(1, 1, q.size(-2), rotary_dim)
    sin = sin.view(1, 1, q.size(-2), rotary_dim)

    def apply(x: torch.Tensor) -> torch.Tensor:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        rotated = (x_rot * cos) + (_rotate_half_pairs(x_rot) * sin)
        return torch.cat([rotated, x_pass], dim=-1) if x_pass.numel() else rotated

    return apply(q), apply(k)


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        causal: bool = False,
        rotary: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        self.rotary = rotary
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.rotary:
            q, k = _apply_rotary_position_embedding(q, k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            mask = torch.ones(seq, seq, device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, hidden)
        return self.out(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = SelfAttention(hidden_size, num_heads, causal=False)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViTWorkload(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_classes: int,
        mlp_ratio: float,
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.activation_checkpoint = activation_checkpoint
        num_patches = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(hidden_size, num_heads, mlp_ratio) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch(images).flatten(2).transpose(1, 2)
        cls = self.cls.expand(images.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        for block in self.blocks:
            if self.activation_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.head(self.norm(x[:, 0]))


class BERTWorkload(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_size)
        self.position = nn.Embedding(seq_len, hidden_size)
        self.token_type = nn.Embedding(2, hidden_size)
        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(hidden_size, num_heads, mlp_ratio) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.mlm = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq = input_ids.shape
        pos = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, seq)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = self.token(input_ids) + self.position(pos) + self.token_type(token_type_ids)
        for block in self.blocks:
            x = block(x)
        return self.mlm(self.norm(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.gate = nn.Linear(hidden_size, inner, bias=False)
        self.up = nn.Linear(hidden_size, inner, bias=False)
        self.down = nn.Linear(inner, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LlamaDecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = SelfAttention(hidden_size, num_heads, causal=True, rotary=True)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LlamaWorkload(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList(
            LlamaDecoderBlock(hidden_size, num_heads, mlp_ratio) for _ in range(num_layers)
        )
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


def build_model(model_name: str, cfg: WorkloadConfig, *, activation_checkpoint: bool = False) -> nn.Module:
    if model_name == "resnet":
        return ResNetWorkload(cfg.resnet_layers, cfg.resnet_width, cfg.num_classes)
    if model_name == "vit":
        return ViTWorkload(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            num_classes=cfg.num_classes,
            mlp_ratio=cfg.mlp_ratio,
            activation_checkpoint=activation_checkpoint,
        )
    if model_name == "bert":
        return BERTWorkload(
            vocab_size=cfg.vocab_size,
            seq_len=cfg.seq_len,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
        )
    if model_name == "llama":
        return LlamaWorkload(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
        )
    raise ValueError(f"unknown model: {model_name}")


def maybe_wrap_model(
    model: nn.Module,
    *,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    local_rank: int,
    device: torch.device,
) -> nn.Module:
    parallel = args.parallel
    if parallel == "auto":
        parallel = "ddp" if world_size > 1 else "none"

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")
        model = torch.compile(model, backend=args.compile_backend)

    if parallel == "none":
        return model
    if parallel == "ddp":
        if not dist.is_initialized():
            raise RuntimeError("DDP requested but process group is not initialized")
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    if parallel == "fsdp":
        if not dist.is_initialized():
            raise RuntimeError("FSDP requested but process group is not initialized")
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except Exception as exc:  # pragma: no cover - depends on torch build.
            raise RuntimeError("FSDP is unavailable in this PyTorch build") from exc
        return FSDP(model)
    raise ValueError(f"unsupported parallel mode: {parallel}")


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def synthetic_batch(
    model_name: str,
    cfg: WorkloadConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model_name in {"resnet", "vit"}:
        images = torch.randn(cfg.batch_size, 3, cfg.image_size, cfg.image_size, device=device, dtype=dtype)
        labels = torch.randint(0, cfg.num_classes, (cfg.batch_size,), device=device)
        return images, labels
    tokens = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
    labels = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
    return tokens, labels


def loss_for_batch(
    model_name: str,
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = model(inputs)
    if model_name in {"resnet", "vit"}:
        return F.cross_entropy(logits.float(), labels)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.reshape(-1))


def train_one_step(
    *,
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: WorkloadConfig,
    device: torch.device,
    dtype: torch.dtype,
    grad_accum_steps: int,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    last_loss = 0.0
    for _ in range(max(int(grad_accum_steps), 1)):
        inputs, labels = synthetic_batch(model_name, cfg, device=device, dtype=dtype)
        loss = loss_for_batch(model_name, model, inputs, labels) / max(int(grad_accum_steps), 1)
        loss.backward()
        last_loss = float(loss.detach().float().item())
    optimizer.step()
    return last_loss


def main() -> None:
    args = parse_args()
    cudnn_enabled = configure_cudnn_backend(args.cudnn_mode)
    cfg = resolve_config(args)
    force_dist = args.parallel in {"ddp", "fsdp"}
    rank, world_size, local_rank, device, initialized = setup_dist(force=force_dist)

    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    dtype = runtime_dtype(args)
    model = build_model(
        args.model,
        cfg,
        activation_checkpoint=bool(args.activation_checkpoint and args.model == "vit"),
    ).to(device=device, dtype=dtype)
    model.train()
    raw_param_count = parameter_count(model)
    model = maybe_wrap_model(
        model,
        args=args,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if rank == 0:
        print(
            {
                "model": args.model,
                "preset": args.preset,
                "parallel": args.parallel,
                "compile": bool(args.compile),
                "compile_backend": args.compile_backend if args.compile else None,
                "activation_checkpoint": bool(args.activation_checkpoint and args.model == "vit"),
                "world_size": world_size,
                "device": str(device),
                "dtype": args.dtype,
                "cudnn_mode": args.cudnn_mode,
                "cudnn_enabled": cudnn_enabled,
                "params": raw_param_count,
                "params_m": round(raw_param_count / 1_000_000, 3),
                "config": cfg.__dict__,
            },
            flush=True,
        )

    for warmup in range(max(int(args.warmup_steps), 0)):
        train_one_step(
            model_name=args.model,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            dtype=dtype,
            grad_accum_steps=args.grad_accum_steps,
        )
        if not args.no_step_end_synchronize:
            synchronize_completed_iteration(device)
        if rank == 0 and (warmup + 1 == args.warmup_steps):
            print(f"warmup {warmup + 1}/{args.warmup_steps} complete", flush=True)

    start = time.time()
    for step in range(1, max(int(args.steps), 0) + 1):
        if args.sync_before_step_window:
            synchronize_before_step_window()
        with step_window(step):
            loss = train_one_step(
                model_name=args.model,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                device=device,
                dtype=dtype,
                grad_accum_steps=args.grad_accum_steps,
            )
            if not args.no_step_end_synchronize:
                synchronize_completed_iteration(device)
        if rank == 0 and (step % args.log_interval == 0 or step == args.steps):
            elapsed = time.time() - start
            print(
                f"step {step:4d}/{args.steps} | model={args.model} | "
                f"loss={loss:.6f} | elapsed_s={elapsed:.3f}",
                flush=True,
            )

    cleanup_dist(initialized)


if __name__ == "__main__":
    main()
