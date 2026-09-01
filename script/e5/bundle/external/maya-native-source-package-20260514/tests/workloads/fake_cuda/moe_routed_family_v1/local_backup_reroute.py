#!/usr/bin/env python3
"""Sibling that uses local experts as overflow backups before remote fallback."""

from __future__ import annotations

import torch
import torch.distributed as dist

from common import BASE, FamilyVariant, run_variant


class LocalBackupRerouteExpertParallelMoELayer(BASE.ExpertParallelMoELayer):
    """Prefer local experts when recovering dropped tokens.

    This is a code-semantic sibling meant to look like a real systems design
    choice: try to relieve overflow with local capacity first before paying for
    broader rerouting.
    """

    def forward(self, x):
        batch_size, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        num_tokens = x_flat.shape[0]

        dispatch_mask, combine_weights, aux_loss, metadata = self.gate(x_flat)
        fakecuda_cpu_routing = BASE.is_fakecuda_device(x.device) and self.gate.top_k > 1
        expert_counts = (
            torch.tensor(metadata["expert_counts"], dtype=torch.long)
            if fakecuda_cpu_routing
            else torch.tensor(metadata["expert_counts"], dtype=torch.long, device=x.device)
        )
        expert_token_indices = metadata.get("expert_token_indices")
        if fakecuda_cpu_routing and expert_token_indices is None:
            expert_token_indices = [[] for _ in range(self.num_experts)]
        capacity = int(metadata["capacity"])
        rerouted_local = 0
        rerouted_remote = 0

        if capacity > 0:
            local_experts = list(self.local_global_experts)
            if expert_token_indices is not None:
                dropped_tokens = BASE.infer_dropped_token_list(dispatch_mask.shape[-1], expert_token_indices)
            else:
                dropped_tokens = (combine_weights.sum(dim=-1) == 0).nonzero(as_tuple=True)[0].tolist()
            for token_idx in dropped_tokens:
                chosen: int | None = None
                for expert_id in sorted(local_experts, key=lambda idx: int(expert_counts[idx].item())):
                    if int(expert_counts[expert_id].item()) < capacity:
                        chosen = expert_id
                        rerouted_local += 1
                        break
                if chosen is None:
                    for expert_id in torch.argsort(expert_counts).tolist():
                        if int(expert_counts[expert_id].item()) < capacity:
                            chosen = int(expert_id)
                            rerouted_remote += 1
                            break
                if chosen is None:
                    continue
                pos = int(expert_counts[chosen].item())
                dispatch_mask[chosen, pos, token_idx] = 1.0
                combine_weights[token_idx, chosen] = 1.0
                expert_counts[chosen] += 1
                if expert_token_indices is not None:
                    expert_token_indices[chosen].append(int(token_idx))

        metadata = dict(metadata)
        metadata["expert_counts"] = expert_counts.tolist()
        if expert_token_indices is not None:
            metadata["expert_token_indices"] = expert_token_indices
        metadata["rerouted_tokens"] = rerouted_local + rerouted_remote
        metadata["rerouted_local_tokens"] = rerouted_local
        metadata["rerouted_remote_tokens"] = rerouted_remote
        metadata["tokens_dropped"] = max(int(metadata["tokens_dropped"]) - rerouted_local - rerouted_remote, 0)
        expert_counts_list = list(metadata.get("expert_counts", []))

        if self.ep_group is not None and self.ep_size > 1:
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
                if expert_token_indices is not None and owned:
                    selected, _token_counts = BASE.build_owned_token_tensor(
                        owned,
                        expert_token_indices,
                        device=x.device,
                    )
                else:
                    r_mask = dispatch_mask[owned].sum(dim=0).sum(dim=0) if owned else torch.zeros(num_tokens, device=x.device)
                    selected = (r_mask > 0).nonzero(as_tuple=True)[0]
                n = min(len(selected), max_tokens)
                if n > 0:
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
            variant_id="local_backup_reroute",
            description="Code-level sibling: overflow tokens spill to local experts before remote fallback.",
            default_overrides={
                "batch_size": 2,
                "seq_len": 64,
                "hidden_size": 128,
                "num_layers": 2,
                "num_experts": 8,
                "top_k": 2,
                "capacity_factor": 1.0,
                "ep_size": 2,
                "micro_batches": 2,
                "recompute": True,
                "expert_layout": "contiguous",
                "log_interval": 1,
            },
            ep_layer_cls=LocalBackupRerouteExpertParallelMoELayer,
        )
    )
