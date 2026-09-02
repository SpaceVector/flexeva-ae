#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from dataclasses import replace
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.workloads.historical_sparse_moe.config import HistoricalSparseMoEWorkloadConfig
from tests.workloads.historical_sparse_moe.model import HistoricalSparseMoEModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historically grounded sparse-MoE workload")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs").joinpath("gshard_style_fastloop.json"),
        help="Path to the workload config JSON.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a reduced local smoke configuration.")
    parser.add_argument("--dry-summary", action="store_true", help="Validate and print the config summary only.")
    parser.add_argument("--train-iters", type=int, default=None, help="Override train_iters.")
    parser.add_argument(
        "--benchmark-warmup-steps",
        type=int,
        default=0,
        help="Number of initial steps to exclude from the final benchmark timing summary.",
    )
    parser.add_argument(
        "--benchmark-measure-steps",
        type=int,
        default=0,
        help="Number of post-warmup steps to include in the final benchmark timing summary. 0 means all remaining steps.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write the final summary JSON.")
    return parser.parse_args()


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _is_master() -> bool:
    return _rank() == 0


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_step_window(step: int, should_measure: bool):
    if not should_measure:
        return nullcontext()
    try:
        from flexsim.maya_lite.markers import step_window as maya_step_window
    except Exception:
        return nullcontext()
    return maya_step_window(step=step, label="training_step")


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def init_runtime(cfg: HistoricalSparseMoEWorkloadConfig) -> tuple[torch.device, dist.ProcessGroup | None]:
    use_cuda = torch.cuda.is_available()
    backend = cfg.runtime.distributed_backend if use_cuda else "gloo"

    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    ep_group = dist.group.WORLD if dist.is_initialized() else None
    return device, ep_group


def build_config(args: argparse.Namespace) -> HistoricalSparseMoEWorkloadConfig:
    cfg = HistoricalSparseMoEWorkloadConfig.from_json(args.config)
    if args.smoke:
        cfg = cfg.make_smoke_variant()
    if args.train_iters is not None:
        cfg = HistoricalSparseMoEWorkloadConfig(
            target_id=cfg.target_id,
            model=cfg.model,
            optimization=replace(cfg.optimization, train_iters=args.train_iters),
            data=cfg.data,
            runtime=cfg.runtime,
        )
    cfg.validate()
    return cfg


def generate_synthetic_batch(
    cfg: HistoricalSparseMoEWorkloadConfig,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.data.seed + step)
    batch = cfg.optimization.micro_batch_size_per_gpu
    seq_len = cfg.model.sequence_length
    vocab = cfg.model.vocab_size
    inputs = torch.randint(0, vocab, (batch, seq_len), generator=generator, dtype=torch.long)
    labels = torch.randint(0, vocab, (batch, seq_len), generator=generator, dtype=torch.long)
    return inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    torch.manual_seed(cfg.data.seed)
    device, ep_group = init_runtime(cfg)

    summary = {"target_id": cfg.target_id, **cfg.derived_summary(), "device": str(device)}
    if _is_master():
        print(json.dumps(summary, indent=2))
    if args.dry_summary:
        return

    model = HistoricalSparseMoEModel(cfg, ep_group=ep_group).to(device)
    parameter_summary = model.parameter_summary()
    if _is_master():
        print(json.dumps(parameter_summary, indent=2))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimization.learning_rate,
        betas=(cfg.optimization.adam_beta1, cfg.optimization.adam_beta2),
        weight_decay=cfg.optimization.weight_decay,
    )

    use_autocast = device.type == "cuda" and cfg.optimization.fp16
    if hasattr(torch, "amp") and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=use_autocast)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_autocast)

    final_metrics: dict[str, float | int | str | list[str]] = {
        "target_id": cfg.target_id,
        "world_size": _world_size(),
        "rank": _rank(),
    }
    final_metrics.update(parameter_summary)
    benchmark_warmup_steps = max(0, args.benchmark_warmup_steps)
    if benchmark_warmup_steps >= cfg.optimization.train_iters:
        raise ValueError("benchmark_warmup_steps must be smaller than train_iters")
    benchmark_measure_steps = args.benchmark_measure_steps
    measured_step_times: list[float] = []
    measured_step_ids: list[int] = []

    model.train()
    if dist.is_initialized():
        dist.barrier()
    for step in range(cfg.optimization.train_iters):
        should_measure = step >= benchmark_warmup_steps and (
            benchmark_measure_steps <= 0 or len(measured_step_times) < benchmark_measure_steps
        )
        _sync_device(device)
        with _benchmark_step_window(step + 1, should_measure):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accum_steps = cfg.gradient_accumulation_steps
            loss = None
            last_outputs = None
            for micro_step in range(accum_steps):
                inputs, labels = generate_synthetic_batch(cfg, step * accum_steps + micro_step, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                    outputs = model(inputs, labels=labels)
                    micro_loss = outputs["loss"] / accum_steps
                scaler.scale(micro_loss).backward()
                loss = micro_loss if loss is None else loss + micro_loss
                last_outputs = outputs

            assert loss is not None
            assert last_outputs is not None
            outputs = last_outputs
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimization.clip_grad)
            scaler.step(optimizer)
            scaler.update()

        _sync_device(device)
        local_step_time = time.perf_counter() - step_start
        if dist.is_initialized():
            max_step_time = torch.tensor([local_step_time], device=device, dtype=torch.float64)
            dist.all_reduce(max_step_time, op=dist.ReduceOp.MAX)
            step_time = float(max_step_time.item())
        else:
            step_time = local_step_time
        if should_measure:
            measured_step_times.append(step_time)
            measured_step_ids.append(step + 1)
        if _is_master() and (
            step == 0
            or (step + 1) == cfg.optimization.train_iters
            or (step + 1) % cfg.optimization.log_interval == 0
        ):
            payload = {
                "step": step + 1,
                "train_iters": cfg.optimization.train_iters,
                "loss": float(loss.item()),
                "lm_loss": float(outputs["lm_loss"].item()),
                "router_aux_loss": float(outputs["router_aux_loss"].item()),
                "tokens_dropped": float(outputs["tokens_dropped"]),
                "tokens_rerouted": float(outputs["tokens_rerouted"]),
                "load_balance_cv": float(outputs["load_balance_cv"]),
                "remote_dispatch_ratio": float(outputs["remote_dispatch_ratio"]),
                "dispatch_imbalance_cv": float(outputs["dispatch_imbalance_cv"]),
                "estimated_a2a_bytes": float(outputs["estimated_a2a_bytes"]),
                "gradient_accumulation_steps": accum_steps,
                "local_step_time_s": round(local_step_time, 4),
                "step_time_s": round(step_time, 4),
                "measured_for_benchmark": should_measure,
            }
            print(json.dumps(payload))
            final_metrics.update(payload)

    if dist.is_initialized():
        dist.barrier()

    if _is_master():
        benchmark_step_time = _median(measured_step_times) if measured_step_times else float(final_metrics["step_time_s"])
        final_metrics["last_step_time_s"] = float(final_metrics["step_time_s"])
        final_metrics["step_time_s"] = round(benchmark_step_time, 4)
        final_metrics["benchmark_step_times_s"] = [round(value, 4) for value in measured_step_times]
        final_metrics["benchmark_step_ids"] = measured_step_ids
        final_metrics["benchmark_warmup_steps"] = benchmark_warmup_steps
        final_metrics["benchmark_measure_steps"] = benchmark_measure_steps if benchmark_measure_steps > 0 else len(measured_step_times)
        final_metrics["benchmark_samples"] = len(measured_step_times)
        final_metrics["status"] = "complete"
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
        print(json.dumps(final_metrics, indent=2))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
