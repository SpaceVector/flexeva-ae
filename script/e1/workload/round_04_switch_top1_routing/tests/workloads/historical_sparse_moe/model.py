from __future__ import annotations

import os
import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .config import HistoricalSparseMoEModelConfig, HistoricalSparseMoEWorkloadConfig


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_to_all_counts(send_counts: list[int], device: torch.device, group: dist.ProcessGroup | None) -> list[int]:
    if not _dist_ready():
        return send_counts
    send = torch.tensor(send_counts, dtype=torch.int64, device=device)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv.tolist()


def _all_to_all_1d(
    tensor: torch.Tensor,
    send_splits: list[int],
    recv_splits: list[int],
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    if not _dist_ready():
        return tensor
    total_recv = sum(recv_splits)
    recv = torch.empty(total_recv, dtype=tensor.dtype, device=tensor.device)
    dist.all_to_all_single(recv, tensor.contiguous(), recv_splits, send_splits, group=group)
    return recv


def _all_to_all_2d(
    tensor: torch.Tensor,
    send_splits: list[int],
    recv_splits: list[int],
    hidden_size: int,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    if not _dist_ready():
        return tensor
    total_recv = sum(recv_splits)
    recv = torch.empty((total_recv, hidden_size), dtype=tensor.dtype, device=tensor.device)
    dist.all_to_all_single(recv, tensor.contiguous(), recv_splits, send_splits, group=group)
    return recv


def _safe_topk(
    tensor: torch.Tensor,
    k: int,
    *,
    dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.is_cuda and os.environ.get("FLEXSIM_MAYA_SAFE_TOPK", "").lower() in {"1", "true", "yes", "on"}:
        cpu_tensor = tensor.detach().cpu()
        sorted_values, sorted_indices = torch.sort(cpu_tensor, dim=dim, descending=True)
        slices = [slice(None)] * sorted_values.dim()
        slices[dim] = slice(0, k)
        index = tuple(slices)
        return sorted_values[index].to(tensor.device), sorted_indices[index].to(tensor.device)
    return torch.topk(tensor, k, dim=dim)


def _fakecuda_eval_compat_enabled() -> bool:
    for key in ("FLEXSIM_MAYA_SAFE_ROUTING", "FLEXSIM_MAYA_SAFE_TOPK"):
        if os.environ.get(key, "").lower() in {"1", "true", "yes", "on"}:
            return True
    return False


@dataclass
class MoEForwardStats:
    router_aux_loss: torch.Tensor
    tokens_dropped: float
    tokens_rerouted: float
    load_balance_cv: float
    remote_dispatch_ratio: float
    dispatch_imbalance_cv: float
    estimated_a2a_bytes: float


class CausalSelfAttention(nn.Module):
    """Modern SDPA-based causal self-attention."""

    def __init__(self, model_cfg: HistoricalSparseMoEModelConfig):
        super().__init__()
        self.num_heads = model_cfg.num_attention_heads
        self.head_dim = model_cfg.hidden_size // model_cfg.num_attention_heads
        self.qkv = nn.Linear(model_cfg.hidden_size, model_cfg.hidden_size * 3)
        self.proj = nn.Linear(model_cfg.hidden_size, model_cfg.hidden_size)
        self.dropout = model_cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.proj(attn)


class ExpertMLP(nn.Module):
    def __init__(self, model_cfg: HistoricalSparseMoEModelConfig):
        super().__init__()
        inner = model_cfg.hidden_size * model_cfg.ffw_multiplier
        self.net = nn.Sequential(
            nn.Linear(model_cfg.hidden_size, inner, bias=False),
            nn.GELU(),
            nn.Linear(inner, model_cfg.hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseFFN(nn.Module):
    def __init__(self, model_cfg: HistoricalSparseMoEModelConfig):
        super().__init__()
        inner = model_cfg.hidden_size * model_cfg.ffw_multiplier
        self.w1 = nn.Linear(model_cfg.hidden_size, inner, bias=False)
        self.w2 = nn.Linear(inner, model_cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, MoEForwardStats]:
        out = self.w2(F.gelu(self.w1(x)))
        zero = x.new_zeros(())
        return out, MoEForwardStats(zero, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class HistoricalSwitchTop1PackedMoE(nn.Module):
    """Switch-style top-1 routed MoE with packed variable-length dispatch."""

    def __init__(self, cfg: HistoricalSparseMoEWorkloadConfig, ep_group: dist.ProcessGroup | None):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.ep_group = ep_group
        self.ep_size = cfg.runtime.expert_parallel_size
        self.num_local_experts = cfg.num_local_experts
        self.router = nn.Linear(self.model_cfg.hidden_size, self.model_cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertMLP(self.model_cfg) for _ in range(self.num_local_experts)])

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _capacity(self, num_tokens: int) -> int:
        factor = self.model_cfg.capacity_factor_train if self.training else self.model_cfg.capacity_factor_eval
        return max(
            self.model_cfg.min_capacity,
            math.ceil(factor * num_tokens * self.model_cfg.top_k / self.model_cfg.num_experts),
        )

    def _packed_dispatch(self, flat: torch.Tensor, assigned_tokens: list[int], assigned_experts: list[int], assigned_scores: list[float]) -> tuple[torch.Tensor, float]:
        if not assigned_tokens:
            return flat.new_zeros(flat.shape), 0.0

        device = flat.device
        compat_routing = flat.is_cuda and _fakecuda_eval_compat_enabled()
        metadata_device = torch.device("cpu") if compat_routing else device
        hidden_size = flat.size(1)
        ep_rank = dist.get_rank(group=self.ep_group) if _dist_ready() else 0

        token_tensor = torch.tensor(assigned_tokens, dtype=torch.long, device=metadata_device)
        expert_tensor = torch.tensor(assigned_experts, dtype=torch.long, device=metadata_device)
        score_tensor = torch.tensor(assigned_scores, dtype=flat.dtype, device=metadata_device)

        dest_ranks = torch.div(expert_tensor, self.num_local_experts, rounding_mode="floor")
        local_expert_ids = torch.remainder(expert_tensor, self.num_local_experts)

        order = torch.arange(dest_ranks.numel(), device=metadata_device)
        order = order[torch.argsort(token_tensor[order], stable=True)]
        order = order[torch.argsort(local_expert_ids[order], stable=True)]
        order = order[torch.argsort(dest_ranks[order], stable=True)]

        ordered_origin_metadata = token_tensor[order]
        ordered_dest_metadata = dest_ranks[order]
        ordered_local_experts_metadata = local_expert_ids[order]
        ordered_sources_metadata = torch.full_like(ordered_dest_metadata, ep_rank)
        ordered_hidden = flat[ordered_origin_metadata.to(device)]
        ordered_scores = score_tensor[order].to(device)
        ordered_origin = ordered_origin_metadata.to(device)
        ordered_local_experts = ordered_local_experts_metadata.to(device)
        ordered_sources = ordered_sources_metadata.to(device)

        send_splits = torch.bincount(ordered_dest_metadata, minlength=self.ep_size).tolist()
        if sum(send_splits) != ordered_hidden.size(0):
            raise RuntimeError("packed dispatch send splits do not match the payload")
        recv_splits = _all_to_all_counts(send_splits, metadata_device, self.ep_group)

        recv_hidden = _all_to_all_2d(ordered_hidden, send_splits, recv_splits, hidden_size, self.ep_group)
        recv_scores = _all_to_all_1d(ordered_scores, send_splits, recv_splits, self.ep_group)
        recv_origin = _all_to_all_1d(ordered_origin, send_splits, recv_splits, self.ep_group)
        recv_local_experts = _all_to_all_1d(ordered_local_experts, send_splits, recv_splits, self.ep_group)
        recv_sources = _all_to_all_1d(ordered_sources, send_splits, recv_splits, self.ep_group)

        local_outputs = torch.zeros_like(recv_hidden)
        for local_idx, expert in enumerate(self.experts):
            if compat_routing:
                selected = torch.nonzero(ordered_local_experts_metadata == local_idx).flatten()
                if selected.numel():
                    selected = selected.to(device)
                    local_outputs[selected] = expert(recv_hidden[selected]) * recv_scores[selected].unsqueeze(-1)
            else:
                mask = recv_local_experts == local_idx
                if mask.any():
                    local_outputs[mask] = expert(recv_hidden[mask]) * recv_scores[mask].unsqueeze(-1)

        recv_sources_metadata = ordered_sources_metadata if compat_routing else recv_sources
        send_back_order = torch.argsort(recv_sources_metadata, stable=True)
        send_back_splits = torch.bincount(recv_sources_metadata, minlength=self.ep_size).tolist()
        if sum(send_back_splits) != local_outputs.size(0):
            raise RuntimeError("packed dispatch return splits do not match the payload")
        recv_back_splits = _all_to_all_counts(send_back_splits, metadata_device, self.ep_group)
        send_back_order = send_back_order.to(device)

        returned_hidden = _all_to_all_2d(
            local_outputs[send_back_order],
            send_back_splits,
            recv_back_splits,
            hidden_size,
            self.ep_group,
        )
        returned_origin = _all_to_all_1d(
            recv_origin[send_back_order],
            send_back_splits,
            recv_back_splits,
            self.ep_group,
        )

        estimated_a2a_bytes = float(
            self._tensor_bytes(ordered_hidden)
            + self._tensor_bytes(ordered_scores)
            + self._tensor_bytes(ordered_origin)
            + self._tensor_bytes(ordered_local_experts)
            + self._tensor_bytes(ordered_sources)
            + self._tensor_bytes(local_outputs[send_back_order])
            + self._tensor_bytes(recv_origin[send_back_order])
        )

        out = flat.new_zeros(flat.shape)
        out.index_add_(0, returned_origin.long(), returned_hidden)
        return out, estimated_a2a_bytes

    def _route_tokens(self, flat: torch.Tensor) -> tuple[torch.Tensor, MoEForwardStats]:
        device = flat.device
        logits = self.router(flat)
        probs = torch.softmax(logits, dim=-1)
        compat_routing = flat.is_cuda and _fakecuda_eval_compat_enabled()
        routing_probs = probs.detach().cpu() if compat_routing else probs

        top_scores, top_indices = torch.max(routing_probs, dim=-1)

        num_tokens = flat.shape[0]
        capacity = self._capacity(num_tokens)
        count_device = torch.device("cpu") if compat_routing else device
        expert_counts = torch.zeros(self.model_cfg.num_experts, dtype=torch.long, device=count_device)

        assigned_tokens: list[int] = []
        assigned_experts: list[int] = []
        assigned_scores: list[float] = []
        dropped = 0

        for token_idx in range(num_tokens):
            expert = int(top_indices[token_idx].item())
            current_load = int(expert_counts[expert].item())
            if current_load >= capacity:
                dropped += 1
                continue
            assigned_tokens.append(token_idx)
            assigned_experts.append(expert)
            assigned_scores.append(float(top_scores[token_idx].item()))
            expert_counts[expert] += 1

        expert_counts_float = expert_counts.float()
        mean_probs = probs.mean(dim=0)
        normalized_load = expert_counts_float.to(device=mean_probs.device, dtype=mean_probs.dtype) / max(
            float(len(assigned_experts)), 1.0
        )
        aux = (
            (normalized_load * mean_probs).sum()
            * self.model_cfg.num_experts
            * self.model_cfg.router_aux_loss_coef
        )
        load_balance_cv = float((expert_counts_float.std() / (expert_counts_float.mean() + 1e-9)).item())
        send_counts = expert_counts.view(self.ep_size, self.num_local_experts).sum(dim=1).float()
        ep_rank = dist.get_rank(group=self.ep_group) if _dist_ready() else 0
        remote_assignments = float(send_counts.sum().item() - send_counts[ep_rank].item())
        total_assignments = float(send_counts.sum().item())
        remote_dispatch_ratio = remote_assignments / total_assignments if total_assignments > 0 else 0.0
        dispatch_imbalance_cv = float((send_counts.std() / (send_counts.mean() + 1e-9)).item()) if self.ep_size > 1 else 0.0

        if not assigned_tokens:
            out = flat.new_zeros(flat.shape)
            estimated_a2a_bytes = 0.0
        elif self.ep_size == 1:
            out = flat.new_zeros(flat.shape)
            for token_idx, expert, score in zip(assigned_tokens, assigned_experts, assigned_scores, strict=True):
                out[token_idx] += self.experts[expert](flat[token_idx : token_idx + 1])[0] * score
            estimated_a2a_bytes = 0.0
        else:
            out, estimated_a2a_bytes = self._packed_dispatch(flat, assigned_tokens, assigned_experts, assigned_scores)

        return out, MoEForwardStats(
            router_aux_loss=aux,
            tokens_dropped=float(dropped),
            tokens_rerouted=0.0,
            load_balance_cv=load_balance_cv,
            remote_dispatch_ratio=remote_dispatch_ratio,
            dispatch_imbalance_cv=dispatch_imbalance_cv,
            estimated_a2a_bytes=estimated_a2a_bytes,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, MoEForwardStats]:
        batch, seq, hidden = x.shape
        flat = x.reshape(batch * seq, hidden)
        out, stats = self._route_tokens(flat)
        return out.view(batch, seq, hidden), stats


class TransformerBlock(nn.Module):
    def __init__(self, cfg: HistoricalSparseMoEWorkloadConfig, ep_group: dist.ProcessGroup | None, use_moe: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.model.hidden_size)
        self.ln2 = nn.LayerNorm(cfg.model.hidden_size)
        self.attn = CausalSelfAttention(cfg.model)
        self.ffn = HistoricalSwitchTop1PackedMoE(cfg, ep_group) if use_moe else DenseFFN(cfg.model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, MoEForwardStats]:
        x = x + self.attn(self.ln1(x))
        ffn_out, stats = self.ffn(self.ln2(x))
        x = x + ffn_out
        return x, stats


class HistoricalSparseMoEModel(nn.Module):
    def __init__(self, cfg: HistoricalSparseMoEWorkloadConfig, ep_group: dist.ProcessGroup | None = None):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.model.vocab_size, cfg.model.hidden_size)
        self.pos_embed = nn.Embedding(cfg.model.sequence_length, cfg.model.hidden_size)
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.blocks = nn.ModuleList()
        for layer_idx in range(cfg.model.num_layers):
            use_moe = (layer_idx % 2 == 1) if cfg.model.moe_every_other_ffn else True
            self.blocks.append(TransformerBlock(cfg, ep_group, use_moe))
        self.final_ln = nn.LayerNorm(cfg.model.hidden_size)
        self.lm_head = nn.Linear(cfg.model.hidden_size, cfg.model.vocab_size, bias=False)
        self.apply(self._init_weights)

    def parameter_summary(self) -> dict[str, int]:
        local_total = 0
        local_expert = 0
        for name, param in self.named_parameters():
            count = param.numel()
            local_total += count
            if ".experts." in name:
                local_expert += count
        dense_shared = local_total - local_expert
        logical_global = dense_shared + local_expert * self.cfg.runtime.expert_parallel_size
        return {
            "local_total_params": local_total,
            "local_expert_params": local_expert,
            "shared_dense_params": dense_shared,
            "logical_global_params": logical_global,
        }

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.model.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor | float]:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)

        aux_total = x.new_zeros(())
        dropped_total = 0.0
        rerouted_total = 0.0
        cv_values: list[float] = []
        remote_ratio_values: list[float] = []
        dispatch_cv_values: list[float] = []
        estimated_a2a_bytes_total = 0.0

        for block in self.blocks:
            x, stats = block(x)
            aux_total = aux_total + stats.router_aux_loss
            dropped_total += stats.tokens_dropped
            rerouted_total += stats.tokens_rerouted
            if stats.load_balance_cv > 0.0:
                cv_values.append(stats.load_balance_cv)
            if stats.remote_dispatch_ratio > 0.0:
                remote_ratio_values.append(stats.remote_dispatch_ratio)
            if stats.dispatch_imbalance_cv > 0.0:
                dispatch_cv_values.append(stats.dispatch_imbalance_cv)
            estimated_a2a_bytes_total += stats.estimated_a2a_bytes

        x = self.final_ln(x)
        logits = self.lm_head(x)

        result: dict[str, torch.Tensor | float] = {
            "logits": logits,
            "router_aux_loss": aux_total,
            "tokens_dropped": dropped_total,
            "tokens_rerouted": rerouted_total,
            "load_balance_cv": float(sum(cv_values) / max(len(cv_values), 1)),
            "remote_dispatch_ratio": float(sum(remote_ratio_values) / max(len(remote_ratio_values), 1)),
            "dispatch_imbalance_cv": float(sum(dispatch_cv_values) / max(len(dispatch_cv_values), 1)),
            "estimated_a2a_bytes": float(estimated_a2a_bytes_total),
        }
        if labels is not None:
            lm_loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
            result["loss"] = lm_loss + aux_total
            result["lm_loss"] = lm_loss
        return result
