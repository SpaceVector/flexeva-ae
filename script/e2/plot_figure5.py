#!/usr/bin/env python3
"""Plot Figure 5 from trace-mode or native-mode CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "plot"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def draw_panel(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    label_key: str,
    xlabel: str,
    mode: str,
    paper_ymax: float,
    display_labels: list[str] | None = None,
) -> None:
    labels = display_labels or [row[label_key] for row in rows]
    errors = [float(row["evaluator_error_pct"]) for row in rows]
    x = list(range(len(labels)))
    if mode == "trace":
        width = 0.32
        ax.bar(
            [value - width / 2 for value in x],
            errors,
            width,
            label="Maya",
            color="white",
            edgecolor="black",
            linewidth=0.55,
        )
        ax.bar(
            [value + width / 2 for value in x],
            errors,
            width,
            label="FlexEva",
            color="white",
            edgecolor="black",
            linewidth=0.55,
            hatch="xx",
        )
        ymax = paper_ymax
    else:
        ax.bar(
            x,
            errors,
            0.48,
            label="Maya / FlexEva",
            color="white",
            edgecolor="black",
            linewidth=0.55,
            hatch="xx",
        )
        ymax = max(1.0, max(errors) * 1.15)
    ax.set_ylabel("Error (%)")
    ax.set_xlabel(xlabel)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=5.8 if display_labels else 6.4)
    ax.set_ylim(0.0, ymax)
    ax.grid(axis="y", linestyle="--", linewidth=0.35, color="#d6d6d6", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=6.5)


def draw(result_dir: Path, output_dir: Path, mode: str) -> Path:
    gpt_rows = read_rows(result_dir / "figure5a.csv")
    moe_rows = read_rows(result_dir / "figure5b.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax_gpt, ax_moe) = plt.subplots(1, 2, figsize=(4.0, 2.3), gridspec_kw={"wspace": 0.46})
    draw_panel(ax_gpt, gpt_rows, "gpu_scale", "Number of GPUs", mode, 5.0)
    draw_panel(
        ax_moe,
        moe_rows,
        "case",
        "MoE case",
        mode,
        8.0,
        display_labels=["Base\nMoE", "Intra\n0-1", "Cross\n0-8", "Cross\n0-15", "Bndry.\n7-8"],
    )
    if mode == "trace":
        ax_gpt.set_yticks([0, 1, 2, 3, 4, 5])
        ax_moe.set_yticks([0, 2, 4, 6, 8])
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.39, top=0.93)
    fig.text(0.27, 0.07, "(a) GPT configurations", ha="center", family="serif", fontsize=10.0)
    fig.text(0.74, 0.07, "(b) MoE rounds", ha="center", family="serif", fontsize=10.0)
    output_path = output_dir / ("figure5.pdf" if mode == "trace" else "figure5-native.pdf")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--mode", choices=("trace", "native"), default="trace")
    args = parser.parse_args()
    print(f"wrote {draw(args.result_dir, args.output_dir, args.mode)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
