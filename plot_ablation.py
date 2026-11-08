#!/usr/bin/env python3
"""plot_ablation.py -- graphs for ablation.sh (one scaling fix at a time).

usage: python3 plot_ablation.py ablation_<id>/raw.csv [OUTDIR]
Needs only matplotlib.

  A_time_per_iter.png   ms per CG iteration vs threads, one panel per matrix
  B_speedup.png         speedup over cpu_sparse at 1 thread, one panel per matrix
  C_best_by_config.png  best time per iteration of each configuration
  D_bandwidth.png       achieved memory bandwidth vs threads (largest matrix)
Values are medians of the repeats.
"""
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["v1_close", "v1_spread", "v2_sync", "v2_touch", "v2_balance", "v2_grain"]
LABEL = {
    "v1_close": "cpu_sparse, close (baseline)",
    "v1_spread": "cpu_sparse, spread",
    "v2_sync": "+ padded sums, 3 barriers",
    "v2_touch": "+ first-touch",
    "v2_balance": "+ nnz-balanced rows",
    "v2_grain": "+ size-based threads",
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948", "#008300"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 6, "legend.frameon": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(path):
    runs = defaultdict(list)
    info = {}
    for r in csv.DictReader(open(path)):
        k = (r["config"], r["matrix"], int(r["threads"]))
        runs[k].append(float(r["ms_per_iter"]))
        info[r["matrix"]] = (int(r["N"]), int(r["nnz"]))
    med = {k: statistics.median(v) for k, v in runs.items()}
    return med, info


def bytes_per_iter(N, nnz):
    # matrix values + column indices, row pointers, ~13 vector streams of 8 bytes
    return 12.0 * nnz + 8.0 * N + 104.0 * N


def save(fig, outdir, name):
    fig.tight_layout()
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


def panels(n):
    cols = min(n, 2)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 4.4 * rows), squeeze=False)
    flat = [a for row in axes for a in row]
    for a in flat[n:]:
        a.set_visible(False)
    return fig, flat[:n]


def thread_axis(ax, threads):
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads, [str(t) for t in threads], fontsize=8)
    ax.set_xlabel("threads")
    if max(threads) > 72:
        ax.axvline(72, color=INK2, linewidth=0.8, linestyle=":")  # one full socket


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "raw.csv"
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(csv_path) or ".", "plots")
    os.makedirs(outdir, exist_ok=True)
    med, info = load(csv_path)
    configs = [c for c in ORDER if any(k[0] == c for k in med)]
    configs += sorted({k[0] for k in med} - set(configs))
    color = {c: SERIES[i % len(SERIES)] for i, c in enumerate(configs)}
    mats = sorted(info, key=lambda m: info[m][0])
    threads = sorted({k[2] for k in med})
    base_cfg = configs[0]

    def series(cfg, m):
        ts = [t for t in threads if (cfg, m, t) in med]
        # every configuration shares the baseline's 1-thread point
        if 1 not in ts and (base_cfg, m, 1) in med:
            return [1] + ts, [med[(base_cfg, m, 1)]] + [med[(cfg, m, t)] for t in ts]
        return ts, [med[(cfg, m, t)] for t in ts]

    # A: time per iteration
    fig, axes = panels(len(mats))
    for ax, m in zip(axes, mats):
        for c in configs:
            ts, ys = series(c, m)
            if ts:
                ax.plot(ts, ys, "-o", color=color[c], label=LABEL.get(c, c))
        ax.set_yscale("log")
        thread_axis(ax, threads)
        ax.set_ylabel("ms per iteration")
        ax.set_title(f"{m}  (N = {info[m][0]:,})")
    axes[0].legend(fontsize=8)
    fig.suptitle("Time per CG iteration (lower is better; dotted line = one full socket)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    save(fig, outdir, "A_time_per_iter.png")

    # B: speedup over cpu_sparse at 1 thread
    fig, axes = panels(len(mats))
    for ax, m in zip(axes, mats):
        if (base_cfg, m, 1) not in med:
            continue
        base = med[(base_cfg, m, 1)]
        ax.plot(threads, threads, "--", color=INK2, linewidth=1.1, label="ideal")
        for c in configs:
            ts, ys = series(c, m)
            if ts:
                ax.plot(ts, [base / y for y in ys], "-o", color=color[c], label=LABEL.get(c, c))
        thread_axis(ax, threads)
        ax.set_yscale("log", base=2)
        ax.set_ylabel("speedup vs cpu_sparse, 1 thread")
        ax.set_title(f"{m}  (N = {info[m][0]:,})")
    axes[0].legend(fontsize=8)
    save(fig, outdir, "B_speedup.png")

    # C: best time per iteration for each configuration
    fig, axes = panels(len(mats))
    for ax, m in zip(axes, mats):
        rows = []
        for c in configs:
            pts = [(med[(c, m, t)], t) for t in threads if (c, m, t) in med and t > 1]
            if pts:
                rows.append((c, *min(pts)))
        if not rows:
            continue
        base = rows[0][1]
        y = list(range(len(rows)))[::-1]
        ax.barh(y, [r[1] for r in rows], color=[color[r[0]] for r in rows], height=0.6)
        ax.set_yticks(y, [LABEL.get(r[0], r[0]) for r in rows], fontsize=9)
        for yi, (c, v, t) in zip(y, rows):
            ax.text(v, yi, f"  {v:.3g} ms @ {t} thr  ({base / v:.2f}x)", va="center", fontsize=8)
        ax.set_xlim(0, max(r[1] for r in rows) * 1.6)
        ax.set_xlabel("best ms per iteration (over thread counts)")
        ax.set_title(f"{m}  (N = {info[m][0]:,})")
        ax.grid(axis="y", visible=False)
    save(fig, outdir, "C_best_by_config.png")

    # D: achieved bandwidth on the largest matrix
    m = mats[-1]
    N, nnz = info[m]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for c in configs:
        ts, ys = series(c, m)
        if ts:
            ax.plot(ts, [bytes_per_iter(N, nnz) / (y * 1e-3) / 1e9 for y in ys], "-o",
                    color=color[c], label=LABEL.get(c, c))
    thread_axis(ax, threads)
    ax.set_ylabel("effective memory bandwidth (GB/s)")
    ax.set_title(f"Achieved bandwidth, {m} (N = {N:,})")
    ax.legend(fontsize=8)
    save(fig, outdir, "D_bandwidth.png")


if __name__ == "__main__":
    main()
