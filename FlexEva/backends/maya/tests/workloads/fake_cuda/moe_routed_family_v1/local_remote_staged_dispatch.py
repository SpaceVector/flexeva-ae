#!/usr/bin/env python3
"""Sibling that stages local-owner and remote-owner traffic separately."""

from __future__ import annotations

import torch
import torch.distributed as dist

from common import BASE, FamilyVariant, run_variant


class LocalRemoteStagedDispatchMoELayer(BASE.ExpertParallelMoELayer):
    """Split dispatch into local-owner and remote-owner phases."""

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
            max_tokens = int(num_tokens * 1.5)
            local_send = torch.zeros(self.ep_size, max_tokens, hidden, device=x.device)
            remote_send = torch.zeros(self.ep_size, max_tokens, hidden, device=x.device)
            local_recv = torch.zeros(self.ep_size, max_tokens, hidden, device=x.device)
            remote_recv = torch.zeros(self.ep_size, max_tokens, hidden, device=x.device)

            local_stage_tokens = 0
            remote_stage_tokens = 0
            for r in range(self.ep_size):
                owned = [idx for idx, owner in enumerate(self.expert_owner) if owner == r]
                if not owned:
                    continue
                if expert_token_indices is not None:
                    selected, _token_counts = BASE.build_owned_token_tensor(
                        owned,
                        expert_token_indices,
                        device=x.device,
                    )
                else:
                    r_mask = dispatch_mask[owned].sum(dim=0).sum(dim=0)
                    selected = (r_mask > 0).nonzero(as_tuple=True)[0]
                if len(selected) == 0:
                    continue
                n = min(len(selected), max_tokens)
                if r == self.rank_in_ep:
                    local_send[r, :n] = x_flat[selected[:n]]
                    local_stage_tokens += n
                else:
                    remote_send[r, :n] = x_flat[selected[:n]]
                    remote_stage_tokens += n

            metadata["local_stage_tokens"] = local_stage_tokens
            metadata["remote_stage_tokens"] = remote_stage_tokens

            dist.all_to_all(list(local_recv.unbind(0)), list(local_send.unbind(0)), group=self.ep_group)
            dist.all_to_all(list(remote_recv.unbind(0)), list(remote_send.unbind(0)), group=self.ep_group)

            expert_outputs_local = torch.zeros_like(local_recv[self.rank_in_ep])
            for local_idx, global_idx in enumerate(self.local_global_experts):
                mask = dispatch_mask[global_idx]
                expert_input = torch.matmul(mask, x_flat)
                if global_idx < len(expert_counts_list) and int(expert_counts_list[global_idx]) > 0:
                    expert_out = self.experts[local_idx](expert_input)
                    expert_outputs_local[: expert_out.shape[0]] += expert_out

            result_send_local = torch.zeros_like(local_send)
            result_send_remote = torch.zeros_like(remote_send)
            result_recv_local = torch.zeros_like(local_recv)
            result_recv_remote = torch.zeros_like(remote_recv)
            split = max(1, expert_outputs_local.shape[0] // 2)
            result_send_local[self.rank_in_ep, :split] = expert_outputs_local[:split]
            result_send_remote[self.rank_in_ep, : expert_outputs_local.shape[0] - split] = expert_outputs_local[split:]
            dist.all_to_all(list(result_recv_local.unbind(0)), list(result_send_local.unbind(0)), group=self.ep_group)
            dist.all_to_all(list(result_recv_remote.unbind(0)), list(result_send_remote.unbind(0)), group=self.ep_group)
            output = result_recv_local.sum(dim=0)[:num_tokens] + result_recv_remote.sum(dim=0)[:num_tokens]
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
            variant_id="local_remote_staged_dispatch",
            description="Code-level sibling: exchange local-owner traffic before remote-owner traffic.",
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
            ep_layer_cls=LocalRemoteStagedDispatchMoELayer,
        )
    )
