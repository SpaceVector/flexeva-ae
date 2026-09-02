#!/usr/bin/env python3
"""Draw the data-backed Figure 1(b/c) panels from the E1 result ledgers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "plot"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def draw(result_dir: Path, output_dir: Path) -> Path:
    code_rows = read_csv(result_dir / "figure1b.csv")
    improvement_rows = read_csv(result_dir / "figure1c.csv")
    if len(code_rows) != len(improvement_rows) or not code_rows:
        raise ValueError("Figure 1 ledgers must have the same non-zero row count")

    labels = [row["Optimization Stage"] for row in code_rows]
    x = list(range(len(labels)))
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.8,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax_code, ax_improvement) = plt.subplots(
        1,
        2,
        figsize=(3.45, 1.55),
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.42},
    )
    fig.patch.set_facecolor("white")

    width = 0.27
    added = [int(row["Added"]) for row in code_rows]
    deleted = [int(row["Deleted"]) for row in code_rows]
    total = [int(row["Total"]) for row in code_rows]
    ax_code.bar(
        [value - width / 2 for value in x],
        added,
        width,
        label="Added",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.45,
        alpha=0.82,
    )
    ax_code.bar(
        [value + width / 2 for value in x],
        deleted,
        width,
        label="Deleted",
        color="#d62728",
        edgecolor="black",
        linewidth=0.45,
        alpha=0.82,
    )
    ax_code.plot(x, total, color="black", marker="o", markersize=2.3, linewidth=0.8, label="Total")
    ax_code.set_title("(b) Code modifications", loc="left", pad=1.5)
    ax_code.set_ylabel("Lines of Code", labelpad=1.5)
    ax_code.set_xticks(x, labels)
    ax_code.set_ylim(0, 220)
    ax_code.set_yticks([0, 100, 200])
    ax_code.grid(axis="y", linestyle="--", linewidth=0.35, color="#d9d9d9")
    ax_code.legend(frameon=False, loc="upper right", handlelength=1.1, borderpad=0.1, labelspacing=0.2)

    metrics = (
        ("Time", "#1f77b4", "o", "-"),
        ("A2A", "#ff7f0e", "s", "--"),
        ("Drop", "#2ca02c", "^", "-"),
        ("Reroute", "#d62728", "D", ":"),
    )
    for metric, color, marker, linestyle in metrics:
        values = [float(row[metric]) for row in improvement_rows]
        ax_improvement.plot(
            x,
            values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=2.6,
            linewidth=0.9,
            markeredgecolor="black",
            markeredgewidth=0.25,
            label=metric,
        )
    ax_improvement.set_title("(c) Optimization improvement", loc="left", pad=1.5)
    ax_improvement.set_ylabel("Improvement", labelpad=1.5)
    ax_improvement.set_xticks(x, labels)
    ax_improvement.set_ylim(0.4, 2.08)
    ax_improvement.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax_improvement.set_yticklabels(["0.5x", "1.0x", "1.5x", "2.0x"])
    ax_improvement.axhline(1.0, color="#b8b8b8", linestyle="--", linewidth=0.55)
    ax_improvement.text(0.04, 0.88, "↑ better", transform=ax_improvement.transAxes, fontsize=5.8, color="#333333")

    for axis in (ax_code, ax_improvement):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.55)
        axis.tick_params(axis="both", width=0.55, length=1.8, pad=1.5)

    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.20, top=0.91)
    output_path = output_dir / "figure1.pdf"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_path = draw(args.result_dir, args.output_dir)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
