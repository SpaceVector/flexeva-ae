#!/usr/bin/env python3
"""Sibling that stages expert-parallel communication into two priority bursts."""

from __future__ import annotations

import torch
import torch.distributed as dist

from common import BASE, FamilyVariant, run_variant


class StagedPriorityDispatchExpertParallelMoELayer(BASE.ExpertParallelMoELayer):
    """Split expert-parallel traffic into head and tail stages.

    The anchor emits one large expert-parallel exchange per pass. This sibling
    uses a simple two-stage plan so the trace shape reflects a different
    communication critical path.
    """

    def forward(self, x):
        batch_size, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        num_tokens = x_flat.shape[0]

        dispatch_mask, combine_weights, aux_loss, metadata = self.gate(x_flat)
        metadata = dict(metadata)
        metadata["dispatch_stages"] = 2
        expert_counts_list = list(metadata.get("expert_counts", []))
        expert_token_indices = metadata.get("expert_token_indices")

        if self.ep_group is not None and self.ep_size > 1:
            expert_counts = torch.tensor(metadata["expert_counts"], device=x.device)
            send_counts = []
            per_rank_selected: list[torch.Tensor] = []
            for r in range(self.ep_size):
                owned = [idx for idx, owner in enumerate(self.expert_owner) if owner == r]
                if expert_token_indices is not None and owned:
                    count = sum(int(expert_counts_list[idx]) for idx in owned if idx < len(expert_counts_list))
                else:
                    count = expert_counts[owned].sum().item() if owned else 0
                send_counts.append(count)
                if expert_token_indices is not None and owned:
                    selected, _token_counts = BASE.build_owned_token_tensor(
                        owned,
                        expert_token_indices,
                        device=x.device,
                        sort_by_density=True,
                    )
                else:
                    r_mask = dispatch_mask[owned].sum(dim=0).sum(dim=0) if owned else torch.zeros(num_tokens, device=x.device)
                    selected = (r_mask > 0).nonzero(as_tuple=True)[0]
                    if len(selected) > 0:
                        priorities = r_mask[selected]
                        order = torch.argsort(priorities, descending=True)
                        selected = selected[order]
                per_rank_selected.append(selected)

            send_counts_tensor = torch.tensor(send_counts, device=x.device, dtype=torch.long)
            recv_counts_tensor = torch.zeros_like(send_counts_tensor)
            dist.all_to_all_single(recv_counts_tensor, send_counts_tensor, group=self.ep_group)

            max_tokens = int(num_tokens * 1.5)
            head_send = torch.zeros(self.ep_size, max_tokens, hidden, device=x.device)
            tail_send = torch.zeros(self.ep_size, max_tokens, hidden, device=x.device)
            head_recv = torch.zeros(self.ep_size, max_tokens, hidden, device=x.device)
            tail_recv = torch.zeros(self.ep_size, max_tokens, hidden, device=x.device)

            head_tokens = 0
            for r, selected in enumerate(per_rank_selected):
                if len(selected) == 0:
                    continue
                split = max(1, len(selected) // 2)
                head = selected[:split]
                tail = selected[split:]
                head_tokens += len(head)
                if len(head) > 0:
                    head_send[r, : len(head)] = x_flat[head]
                if len(tail) > 0:
                    tail_send[r, : len(tail)] = x_flat[tail]

            metadata["dispatch_stage0_tokens"] = head_tokens
            metadata["dispatch_stage1_tokens"] = int(sum(len(sel) - max(1, len(sel) // 2) for sel in per_rank_selected if len(sel) > 0))

            dist.all_to_all(list(head_recv.unbind(0)), list(head_send.unbind(0)), group=self.ep_group)
            dist.all_to_all(list(tail_recv.unbind(0)), list(tail_send.unbind(0)), group=self.ep_group)

            expert_outputs_local = torch.zeros_like(head_recv[self.rank_in_ep])
            for local_idx, global_idx in enumerate(self.local_global_experts):
                mask = dispatch_mask[global_idx]
                expert_input = torch.matmul(mask, x_flat)
                if global_idx < len(expert_counts_list) and int(expert_counts_list[global_idx]) > 0:
                    expert_out = self.experts[local_idx](expert_input)
                    expert_outputs_local[: expert_out.shape[0]] += expert_out

            result_send_head = torch.zeros_like(head_send)
            result_send_tail = torch.zeros_like(tail_send)
            result_recv_head = torch.zeros_like(head_recv)
            result_recv_tail = torch.zeros_like(tail_recv)
            split = max(1, expert_outputs_local.shape[0] // 2)
            result_send_head[self.rank_in_ep, :split] = expert_outputs_local[:split]
            result_send_tail[self.rank_in_ep, : expert_outputs_local.shape[0] - split] = expert_outputs_local[split:]

            dist.all_to_all(list(result_recv_head.unbind(0)), list(result_send_head.unbind(0)), group=self.ep_group)
            dist.all_to_all(list(result_recv_tail.unbind(0)), list(result_send_tail.unbind(0)), group=self.ep_group)
            output = result_recv_head.sum(dim=0)[:num_tokens] + result_recv_tail.sum(dim=0)[:num_tokens]
        else:
            output = torch.zeros_like(x_flat)
            for expert_id in range(self.num_experts):
                mask = dispatch_mask[expert_id]
                expert_input = torch.matmul(mask, x_flat)
                if expert_id < len(expert_counts_list) and int(expert_counts_list[expert_id]) > 0:
                    expert_out = self.experts[expert_id](expert_input)
                    output += torch.matmul(mask.t(), expert_out)

        output = output.view(batch_size, seq_len, hidden)
        return output, aux_loss, metadata


if __name__ == "__main__":
    run_variant(
        FamilyVariant(
            variant_id="staged_priority_dispatch",
            description="Code-level sibling: expert-parallel dispatch is split into two priority stages.",
            default_overrides={
                "batch_size": 2,
                "seq_len": 64,
                "hidden_size": 128,
                "num_layers": 2,
                "num_experts": 8,
                "top_k": 2,
                "capacity_factor": 1.25,
                "ep_size": 2,
                "micro_batches": 2,
                "recompute": True,
                "expert_layout": "contiguous",
                "log_interval": 1,
            },
            ep_layer_cls=StagedPriorityDispatchExpertParallelMoELayer,
        )
    )
