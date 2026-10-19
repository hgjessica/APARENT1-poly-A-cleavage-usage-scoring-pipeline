#!/usr/bin/env python3
"""Step 3: bar plots of the APARENT1 poly(A) change per oligo.

    python 03_plot_bars.py --config config.json --input scores/hexamer_scores_n.csv
    python 03_plot_bars.py --config config.json --input scores/hexamer_scores_n.csv --metric delta_iso
    python 03_plot_bars.py --config config.json --input scores/hexamer_scores_n.csv --style grouped
    python 03_plot_bars.py --config config.json --input scores/peak_scores_Nmask_score_regions.csv

style best    : one bar per oligo, taken from the site/peak with the largest |delta| (default);
                green = increased poly(A) signal, red = reduced. Sorted high to low.
style grouped : (hexamer CSV only) one bar per oligo and hexamer, two panels
                (composite and isoform-only).
For peak CSVs the bar edge colour encodes the region of the peak.
--input is relative to work_dir unless absolute. Output: figures/<input name>_<style>.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from common import base_parser, load_config, Ctx, need

POS, NEG, ZERO = "#4CAF50", "#F44336", "#9E9E9E"
EDGES = ["#FF9800", "#7B1FA2", "#1E88E5", "#00897B", "#6D4C41"]


def pick_columns(df, metric):
    """-> (delta column, |delta| column, region column or None, y label, value format)."""
    if "delta" in df.columns:                       # peak CSV (log-odds)
        return "delta", "abs_delta", "peak_region", "delta log-odds (masked - reference)", "{:+.2f}"
    if metric == "delta_iso":
        return "delta_iso", "abs_delta_iso", "hexamer_region", "delta isoform prob.", "{:+.3f}"
    return ("delta_composite", "abs_delta_composite", "hexamer_region",
            "delta composite (iso x peak cut)", "{:+.3f}")


def plot_best(df, metric, out, title):
    d, a, reg, ylabel, fmt = pick_columns(df, metric)
    best = df.loc[df.groupby("oligo")[a].idxmax()].sort_values(d, ascending=False)
    vals = best[d].tolist()
    n = len(vals)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), 5), facecolor="white")
    kw = {"edgecolor": "white", "linewidth": 0.5}
    legend = [Patch(facecolor=POS, label="increased (delta > 0)"),
              Patch(facecolor=NEG, label="reduced (delta < 0)")]
    if "peak_region" in best.columns:
        regs = sorted(best[reg].unique())
        cmap = {r: EDGES[i % len(EDGES)] for i, r in enumerate(regs)}
        kw = {"edgecolor": [cmap[r] for r in best[reg]], "linewidth": 1.8}
        legend += [Patch(facecolor="white", edgecolor=cmap[r], label="peak in %s" % r) for r in regs]
    bars = ax.bar(range(n), vals, width=0.7,
                  color=[POS if v > 0 else NEG if v < 0 else ZERO for v in vals], **kw)
    ax.axhline(0, color="#333333", linewidth=0.8)
    span = max([abs(v) for v in vals] + [1e-9])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.02) * span,
                fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=6.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(best["oligo"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("oligo")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-span * 1.3, span * 1.3)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=legend, fontsize=8, loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_grouped(df, out, title):
    oligos = list(df["oligo"].unique())
    hexs = list(df["polyA_hexamer"].unique())
    color = dict(zip(hexs, plt.cm.tab10(np.linspace(0, 1, len(hexs)))))
    width = 0.8 / len(hexs)
    x = np.arange(len(oligos))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(oligos) * 1.2), 10), facecolor="white")
    for ax, col, yl in ((axes[0], "delta_composite", "delta (iso x peak cut)"),
                        (axes[1], "delta_iso", "delta iso prob.")):
        ax.axhline(0, color="black", linewidth=0.8)
        for hi, hx in enumerate(hexs):
            vals = []
            for o in oligos:
                r = df[(df["oligo"] == o) & (df["polyA_hexamer"] == hx)]
                vals.append(float(r[col].iloc[0]) if len(r) else 0.0)
            ax.bar(x + (hi - len(hexs) / 2 + 0.5) * width, vals, width=width * 0.9,
                   color=color[hx], label=hx, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(oligos, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(yl)
        ax.legend(title="polyA hexamer", fontsize=7, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--input", required=True, help="CSV from step 02")
    ap.add_argument("--metric", choices=("delta_composite", "delta_iso"),
                    default="delta_composite", help="hexamer CSV only (default delta_composite)")
    ap.add_argument("--style", choices=("best", "grouped"), default="best")
    ap.add_argument("--title", default="APARENT1 poly(A) change per oligo")
    args = ap.parse_args()
    ctx = Ctx(load_config(args.config), args.dry_run)
    inp = Path(args.input)
    if not inp.is_absolute():
        inp = ctx.path(inp)
    need(ctx, inp)
    out = ctx.path("figures", "%s_%s.png" % (inp.stem, args.style))
    print("plot %s -> %s" % (inp, out))
    if ctx.dry:
        return
    df = pd.read_csv(str(inp))
    if df.empty:
        sys.exit("[ERROR] %s has no rows" % inp)
    if "oligo" not in df.columns:
        sys.exit("[ERROR] %s does not look like a step-02 output (no 'oligo' column)" % inp)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.style == "grouped":
        if "polyA_hexamer" not in df.columns:
            sys.exit("[ERROR] --style grouped needs a hexamer CSV")
        plot_grouped(df, out, args.title)
    else:
        plot_best(df, args.metric, out, args.title)
    print("Saved %s" % out)


if __name__ == "__main__":
    main()
