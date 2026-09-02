from __future__ import annotations

import csv
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
CSV_PATH = ROOT / "line_plot.csv"
FIGSIZE = (2.10, 1.58)

def resolve_series(rows: list[dict[str, str]]) -> list[tuple[str, str, str, str, str]]:
    if "amortized_flexeva_total_s" in rows[0]:
        return [
            ("maya_full_s", "Maya", "#1f77b4", "o", "-"),
            ("amortized_flexeva_total_s", "FlexEva", "#ff7f0e", "s", "-"),
        ]
    flexeva_column = "flexeva_cumulative_s" if "flexeva_cumulative_s" in rows[0] else "flexeva_refresh_cumulative_s"
    return [
        ("maya_full_cumulative_s", "Maya", "#1f77b4", "o", "-"),
        (flexeva_column, "FlexEva", "#ff7f0e", "s", "-"),
    ]


def resolve_baseline_column(rows: list[dict[str, str]]) -> str:
    if "maya_full_s" in rows[0]:
        return "maya_full_s"
    return "maya_full_cumulative_s"


def load_rows(csv_path: Path, panel: str | None = None) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if rows and "Cumulative Round" in rows[0]:
        rows = [
            {
                "round": row["Cumulative Round"].removeprefix("R"),
                "maya_full_cumulative_s": row["Maya"],
                "flexeva_cumulative_s": row["FlexEva"],
            }
            for row in rows
        ]
    if panel is not None:
        rows = [row for row in rows if row.get("panel") == panel]
    if not rows:
        raise ValueError(f"no plot rows in {csv_path} for panel={panel!r}")
    return rows


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 7.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--panel", choices=("gpt", "moe"))
    parser.add_argument("--output", default="cumulative_eval_time.pdf")
    return parser.parse_args()


def plot_cumulative_eval_time(csv_path: Path, output_name: str, panel: str | None = None) -> None:
    output = Path(output_name)
    if not output.is_absolute():
        output = FIGURE_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    rows = load_rows(csv_path, panel)
    x = np.arange(len(rows))
    labels = [f"R{row['round']}" for row in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    baseline_column = resolve_baseline_column(rows)
    baseline = float(rows[0][baseline_column])
    all_values: list[float] = []
    for column, label, color, marker, linestyle in resolve_series(rows):
        values = [float(row[column]) / baseline for row in rows]
        all_values.extend(values)
        ax.plot(
            x,
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4.4,
            linewidth=1.3,
            markeredgecolor="black",
            markeredgewidth=0.35,
            zorder=3,
        )

    ax.set_xlabel("Cumulative Round", labelpad=1.0)
    ax.set_ylabel("Norm. Eval. Time")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.25, len(rows) - 0.75)
    y_max = float(np.ceil(max(all_values) * 2.0) / 2.0)
    ax.set_ylim(0, y_max + 0.15)
    y_ticks = np.arange(0, y_max + 0.01, 0.5 if y_max <= 4 else 1.0)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{tick:.1f}x" for tick in y_ticks])

    ax.text(
        0.05,
        0.08,
        "Lower is better",
        transform=ax.transAxes,
        fontsize=6.8,
        color="#333333",
        ha="left",
        va="bottom",
    )

    ax.grid(axis="y", linestyle="--", linewidth=0.35, color="#d6d6d6", zorder=0)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    ax.tick_params(axis="both", width=0.65, length=2.2, pad=1.0)

    ax.legend(
        frameon=False,
        loc="upper left",
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.1,
        borderaxespad=0.1,
    )

    fig.subplots_adjust(left=0.24, right=0.99, bottom=0.25, top=0.95)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    plot_cumulative_eval_time(Path(args.csv), args.output, args.panel)
