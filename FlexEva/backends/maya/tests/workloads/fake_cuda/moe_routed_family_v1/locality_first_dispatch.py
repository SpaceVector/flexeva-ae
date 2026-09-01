#!/usr/bin/env python3
"""Sibling with a local code change in dispatch packing order."""

from __future__ import annotations

import torch
import torch.distributed as dist

from common import BASE, FamilyVariant, run_variant


class LocalityFirstExpertParallelMoELayer(BASE.ExpertParallelMoELayer):
    """Prioritize dense/local tokens when packing all-to-all buffers.

    The anchor preserves token order when building send buffers.
    This sibling changes the dispatch policy:
    - destination-rank token lists are reordered by routing density
    - tokens with stronger destination affinity are packed first
    """

    def forward(self, x):
        batch_size, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        num_tokens = x_flat.shape[0]

        dispatch_mask, combine_weights, aux_loss, metadata = self.gate(x_flat)
        expert_counts_list = list(metadata.get("expert_counts", []))
        expert_token_indices = metadata.get("expert_token_indices")

        if self.ep_group is not None and self.ep_size > 1:
            expert_counts = torch.tensor(metadata["expert_counts"], device=x.device)
            send_counts = []
            for r in range(self.ep_size):
                owned = [idx for idx, owner in enumerate(self.expert_owner) if owner == r]
                if expert_token_indices is not None and owned:
                    count = sum(int(expert_counts_list[idx]) for idx in owned if idx < len(expert_counts_list))
                else:
                    count = expert_counts[owned].sum().item() if owned else 0
                send_counts.append(count)

            send_counts_tensor = torch.tensor(send_counts, device=x.device, dtype=torch.long)
            recv_counts_tensor = torch.zeros_like(send_counts_tensor)
            dist.all_to_all_single(recv_counts_tensor, send_counts_tensor, group=self.ep_group)

            max_tokens = int(num_tokens * 1.5)
            send_buf = torch.zeros(self.ep_size, max_tokens, hidden, device=x.device)
            recv_buf = torch.zeros(self.ep_size, max_tokens, hidden, device=x.device)

            for r in range(self.ep_size):
                owned = [idx for idx, owner in enumerate(self.expert_owner) if owner == r]
                if not owned:
                    continue
                if expert_token_indices is not None:
                    selected, _token_counts = BASE.build_owned_token_tensor(
                        owned,
                        expert_token_indices,
                        device=x.device,
                        sort_by_density=True,
                    )
                else:
                    r_mask = dispatch_mask[owned].sum(dim=0).sum(dim=0)
                    selected = (r_mask > 0).nonzero(as_tuple=True)[0]
                    if len(selected) == 0:
                        continue
                    # Key difference from anchor: pack tokens with higher routing
                    # density to this destination first.
                    priorities = r_mask[selected]
                    order = torch.argsort(priorities, descending=True)
                    selected = selected[order]
                if len(selected) == 0:
                    continue
                n = min(len(selected), max_tokens)
                send_buf[r, :n] = x_flat[selected[:n]]

            dist.all_to_all(list(recv_buf.unbind(0)), list(send_buf.unbind(0)), group=self.ep_group)

            expert_outputs_local = torch.zeros_like(recv_buf[self.rank_in_ep])
            for local_idx, global_idx in enumerate(self.local_global_experts):
                mask = dispatch_mask[global_idx]
                expert_input = torch.matmul(mask, x_flat)
                if global_idx < len(expert_counts_list) and int(expert_counts_list[global_idx]) > 0:
                    expert_out = self.experts[local_idx](expert_input)
                    expert_outputs_local[: expert_out.shape[0]] += expert_out

            result_send = torch.zeros_like(send_buf)
            result_recv = torch.zeros_like(recv_buf)
            result_send[self.rank_in_ep] = expert_outputs_local
            dist.all_to_all(list(result_recv.unbind(0)), list(result_send.unbind(0)), group=self.ep_group)
            output = result_recv.sum(dim=0)[:num_tokens]
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
            variant_id="locality_first_dispatch",
            description="Code-level sibling: destination-rank send buffers are packed by routing density rather than token order.",
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
            ep_layer_cls=LocalityFirstExpertParallelMoELayer,
        )
    )
