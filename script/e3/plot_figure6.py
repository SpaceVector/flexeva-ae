#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIGURE6_DIR = Path(__file__).resolve().parent / "figure6"
sys.path.insert(0, str(FIGURE6_DIR))

from plot_cumulative_eval_time import plot_cumulative_eval_time


def main() -> int:
    for panel in ("a", "b"):
        plot_cumulative_eval_time(
            ROOT / "result" / "e3" / f"figure6{panel}.csv",
            str(ROOT / "plot" / f"figure6{panel}.pdf"),
        )
    print(f"wrote {ROOT / 'plot' / 'figure6a.pdf'} and {ROOT / 'plot' / 'figure6b.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
