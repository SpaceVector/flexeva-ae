#!/usr/bin/env python3
"""Plot both Figure 8 panels from their paper-facing CSV ledgers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "plot" / "figure8.pdf"
SYSTEMS = (
    ("Maya-full", "Maya", "#4C78A8", "o"),
    ("Maya-trace-RAS", "Maya-trace-RAS", "#72B7B2", "s"),
    ("FlexEva refresh", "FlexEva", "#F28E2B", "^"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty Figure 8 ledger: {path}")
    return rows


def indexed(rows: list[dict[str, str]]) -> tuple[list[str], dict[tuple[str, str], dict[str, str]]]:
    labels = list(dict.fromkeys(row["label"] for row in rows))
    by_key = {(row["label"], row["system"]): row for row in rows}
    expected = {(label, system) for label in labels for system, *_ in SYSTEMS}
    if len(labels) != 3 or set(by_key) != expected:
        raise ValueError("Figure 8 requires three cases and all three systems")
    return labels, by_key


def short_label(label: str) -> str:
    model, rest = label.split("/", 1)
    gpu = rest.split()[0]
    return f"{model}\n{gpu}"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.8,
            "axes.labelsize": 7.1,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(axis: object, *, show_ylabel: bool) -> None:
    axis.set_yscale("log")
    axis.set_ylim(20, 800)
    axis.yaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500]))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
    axis.yaxis.set_minor_locator(FixedLocator([]))
    axis.set_ylabel("Evaluation time (s)" if show_ylabel else "")
    axis.grid(axis="y", linestyle="--", linewidth=0.35, color="#d8d8d8", zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=0.55, length=2.0, pad=1.0)
    for spine in axis.spines.values():
        spine.set_linewidth(0.55)


def plot_figure8(scale_path: Path, trace_path: Path, output: Path) -> None:
    scale_labels, scale = indexed(read_rows(scale_path))
    trace_labels, trace = indexed(read_rows(trace_path))
    if scale_labels != trace_labels:
        raise ValueError("Figure 8 panels use different case order")
    for key in scale:
        if float(scale[key]["seconds"]) <= 0 or float(scale[key]["seconds"]) != float(trace[key]["seconds"]):
            raise ValueError(f"invalid or inconsistent Figure 8 time: {key}")

    configure()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, (left, right) = plt.subplots(1, 2, figsize=(3.35, 1.62), sharey=True)
    x = [0.0, 1.0, 2.0]
    width = 0.24
    for offset, (system, display, color, _) in zip((-width, 0.0, width), SYSTEMS, strict=True):
        left.bar(
            [value + offset for value in x],
            [float(scale[(label, system)]["seconds"]) for label in scale_labels],
            width=width,
            label=display,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )
    left.set_xticks(x, [short_label(label) for label in scale_labels])
    left.set_xlim(-0.55, 2.55)
    style_axis(left, show_ylabel=True)

    for system, display, color, marker in SYSTEMS:
        events = [float(trace[(label, system)]["total_trace_events"]) / 1.0e6 for label in trace_labels]
        seconds = [float(trace[(label, system)]["seconds"]) for label in trace_labels]
        right.plot(
            events,
            seconds,
            label=display,
            color=color,
            marker=marker,
            markersize=3.2,
            markeredgecolor="black",
            markeredgewidth=0.3,
            linewidth=1.0,
            zorder=3,
        )
    right.set_xlabel("Total trace events (M)", labelpad=1.0)
    right.set_xlim(3.5, 28.5)
    right.set_xticks([5, 10, 20])
    style_axis(right, show_ylabel=False)

    for axis in (left, right):
        axis.legend(frameon=False, loc="upper left", handlelength=1.0, handletextpad=0.25, labelspacing=0.08)
    figure.text(0.30, 0.025, "(a) Scale-out targets", ha="center", va="bottom", fontsize=6.8)
    figure.text(0.77, 0.025, "(b) Trace-size sensitivity", ha="center", va="bottom", fontsize=6.8)
    figure.subplots_adjust(left=0.16, right=0.995, bottom=0.34, top=0.98, wspace=0.16)
    figure.savefig(output)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_figure8(args.scale.resolve(), args.trace.resolve(), args.output.resolve())
    print(f"wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
