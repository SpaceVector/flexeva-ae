"""Table 4: anchor capture, selected output generation, merge, full reference."""

from collections import defaultdict
from itertools import chain
import json
from pathlib import Path
from types import SimpleNamespace

import flexmaya_ras as fm
from flexmaya_ras._flexmaya_ras import TraceSuffixBuilder
from flexmaya_ras.simulator import replay_trace_segment


OPTIMIZER = "torch.optim.AdamW(model.parameters(), lr=args.lr)"
BASE_OPTIMIZER = "torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False, fused=False)"


def optimizer_sources(source):
    """The fixed optimizer mutation leaves the entire training prefix intact."""
    candidate = source.read_text()
    if candidate.count(OPTIMIZER) != 1:
        raise ValueError("Table 4 requires the supplied single AdamW constructor")
    return candidate.replace(OPTIMIZER, BASE_OPTIMIZER), candidate


def optimizer_start(markers_path):
    records = [json.loads(line) for line in markers_path.read_text().splitlines() if line.strip()]
    starts = [int(row["trace_ts"]) for row in records
              if row.get("kind") == "region_begin" and row.get("label") == "optimizer_step"]
    if len(starts) != 1:
        raise ValueError(f"expected one optimizer boundary: {markers_path}")
    return starts[0]


def rebase_suffix(prefix_end, replacement, world_size):
    """Only new output gets new temporal bindings."""
    suffixes = defaultdict(list)
    for event in replacement:
        if not event.code_partition.endswith("_optimizer_step"):
            raise ValueError("selective capture published an unselected trace region")
        suffixes[event.rank].append(event)
    if set(prefix_end) != set(range(world_size)) or set(suffixes) != set(prefix_end):
        raise ValueError("anchor or selective capture has incomplete rank coverage")
    for rank in range(world_size):
        suffix = suffixes[rank]
        # Place the new tail after the anchor prefix without changing its
        # internal timestamp intervals or any timestamp in the cached prefix.
        offset = prefix_end[rank] + 1 - suffix[0].timestamp_ns
        for event in suffix:
            event.timestamp_ns += offset


class Joined:
    """A two-segment view; assembling a candidate does not copy its prefix."""
    def __init__(self, prefix, suffix):
        self.prefix, self.suffix = prefix, suffix

    def __len__(self):
        return len(self.prefix) + len(self.suffix)

    def __iter__(self):
        return chain(self.prefix, self.suffix)

GRAPH_FIELDS = ("events", "edges", "sync_partitions", "lineage_edges")


def join_graph(prefix, suffix):
    groups = [SimpleNamespace(
        id=old.id, representative_rank=old.representative_rank, ranks=old.ranks,
        representative_event_count=old.representative_event_count + new.representative_event_count,
        logical_event_count=old.logical_event_count + new.logical_event_count,
    ) for old, new in zip(prefix.dedup_groups, suffix.dedup_groups, strict=True)]
    return SimpleNamespace(
        **{field: Joined(getattr(prefix, field), tuple(getattr(suffix, field)))
           for field in GRAPH_FIELDS},
        lanes=suffix.lanes, dedup_groups=groups, deduplicated=True,
        logical_event_count=prefix.logical_event_count + suffix.logical_event_count,
    )


def check_case(row):
    pipeline = row["anchor_pipeline"]
    built, reused = pipeline["candidate_graph_events_built"], pipeline["prefix_graph_events_reused"]
    if (min(built, reused) <= 0 or built != pipeline["candidate_events_replayed"]
            or built + reused != row["flexeva"]["trace"]["event_count"]):
        raise ValueError(f"Table 4 incremental feedback accounting differs: {row['case']}")
    similarity = row["similarity"]
    if min(similarity["logical_event_coverage"], similarity["weighted_event_jaccard"]) < 0.9999:
        raise ValueError(f"Table 4 selective/full similarity below threshold: {row['case']}")
    if similarity["modeled_kernel_nccl_jaccard"] != 1.0:
        raise ValueError(f"Table 4 selective/full kernel or NCCL trace differs: {row['case']}")
    if similarity["event_wait_dependency_jaccard"] != 1.0 or not row["lane_order"]["all_equal"]:
        raise ValueError(f"Table 4 selective/full dependencies or lane order differ: {row['case']}")


def run_pipeline(args, name, source, world_size, rank_groups, launch, parse, *,
                 context=None, projection_policies=()):
    """Neither selection nor merge receives the independent reference trace."""
    case_dir = args.out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    trace_base = (args.input_trace_root or args.trace_root or args.out_dir) / name
    baseline, candidate = optimizer_sources(source)
    runs = {}

    def capture(phase, text):
        phase_dir = case_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        frozen = phase_dir / "workload.py"
        frozen.write_text(text)
        options = SimpleNamespace(**vars(args))
        options.capture_trace_dir = trace_base / ("traces" if phase == "reference" else f"{phase}/traces")
        options.workload_source = frozen
        options.table4_phase = phase
        options.source_region_markers = True
        recipe_path = options.capture_trace_dir / "capture.json"
        recipe = {"source": text, "context": context, "phase": phase,
                  "autograd_multithreading": False, "nccl_blocking_wait": True}
        recipe = json.loads(json.dumps(recipe))
        if args.reuse_existing_traces and json.loads(recipe_path.read_text()) != recipe:
            raise ValueError(f"{name}: saved {phase} capture uses different source or inputs")
        print(f"Table 4 {name}: {phase} capture", flush=True)
        run = launch(options, phase_dir)
        runs[phase] = run
        if run["return_code"] != 0:
            raise RuntimeError(f"{name}: {phase} capture failed; see {run['stderr']}")
        if not args.reuse_existing_traces:
            recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
        return Path(run["trace_dir"])

    old_dir = capture("anchor", baseline)
    old = parse(old_dir, {})
    # The optimizer mutation leaves all work before this boundary unchanged.
    prefix_raw = [event for event in old if not event.code_partition.endswith("_optimizer_step")]
    old_tail = [event for event in old if event.code_partition.endswith("_optimizer_step")]
    prefix_end = {}
    for event in prefix_raw:
        prefix_end[event.rank] = max(prefix_end.get(event.rank, 0), event.timestamp_ns)
    reused = len(prefix_raw)
    builder = TraceSuffixBuilder(prefix_raw, rank_groups)
    prefix_trace = builder.prefix
    prefix = SimpleNamespace(
        **{field: tuple(getattr(prefix_trace, field)) for field in GRAPH_FIELDS},
        dedup_groups=prefix_trace.dedup_groups, logical_event_count=prefix_trace.logical_event_count,
    )
    # Both optimizer variants resume from the same prefix completion/lane clocks.
    checkpoint = replay_trace_segment(prefix_trace)
    projection_checkpoints = {
        policy: replay_trace_segment(
            prefix_trace, predictor=fm.ReplayRandomForestPredictor(fm.ReplayRFConfig(enabled=False)),
            use_duration_hints=False, zero_duration_apis=policy,
        ) for policy in projection_policies
    }
    anchor_feedback = replay_trace_segment(builder.append(old_tail), checkpoint=checkpoint).report
    if anchor_feedback.cycle_detected:
        raise ValueError("incomplete anchor feedback")
    del old, prefix_raw, old_tail
    selected_dir = capture("selective", candidate)
    # The new tail must refer to the same logical streams and events as the anchor.
    for rank in range(world_size):
        receipts = [json.loads((directory / f"rank_{rank}_refresh.json").read_text())
                    for directory in (old_dir, selected_dir)]
        if [row["phase"] for row in receipts] != ["anchor", "selective"]:
            raise ValueError("expected separate anchor and selective captures")
        if receipts[0]["entry_resources"] != receipts[1]["entry_resources"]:
            raise ValueError(f"{name}: rank {rank} optimizer entry resource bindings changed")
    selected = parse(selected_dir, {})
    rebase_suffix(prefix_end, selected, world_size)
    delta = builder.append(selected)
    feedback = replay_trace_segment(delta, checkpoint=checkpoint).report
    if feedback.cycle_detected:
        raise ValueError("incomplete candidate feedback")
    # Whole-candidate metrics see the cached prefix and new tail as one trace.
    trace = join_graph(prefix, delta)
    trace.projection_feedback = {
        policy: replay_trace_segment(
            delta, checkpoint=saved, use_duration_hints=False, zero_duration_apis=policy,
        ).report for policy, saved in projection_checkpoints.items()
    }
    pipeline = {
        "anchor_operation": "refresh", "parent_anchor": name + ":anchor",
        "candidate_id": name, "reused_events": reused,
        "refreshed_events": len(selected), "merged_events": reused + len(selected),
        "feedback_complete": True, "selective_capture": True,
        "independent_reference": False,
        "mutation": {"before": BASE_OPTIMIZER, "after": OPTIMIZER},
        "changed_partitions": sorted({event.code_partition for event in selected}),
        "anchor_feedback": anchor_feedback.to_dict(),
        "candidate_feedback": feedback.to_dict(),
        "candidate_graph_events_built": len(delta.events),
        "candidate_events_replayed": feedback.replayed_events,
        "candidate_projection_events_replayed": {
            ",".join(sorted(policy)): result.replayed_events
            for policy, result in trace.projection_feedback.items()
        },
        "prefix_graph_events_reused": checkpoint.report.event_count,
        "prefix_execution_skipped": False,
        "runs": runs,
    }
    # Publish candidate feedback before collecting the independent full run.
    # That run is only a comparison reference, not an input to selective refresh.
    (case_dir / "candidate.json").write_text(json.dumps(pipeline, indent=2) + "\n")
    del selected
    reference_dir = capture("reference", candidate)
    audit = {}
    reference = parse(reference_dir, audit)
    pipeline["independent_reference"] = True
    return runs["reference"], reference, trace, feedback, pipeline, audit
