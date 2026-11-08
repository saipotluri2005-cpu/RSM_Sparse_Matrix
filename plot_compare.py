#!/usr/bin/env python3
"""plot_compare.py -- graphs comparing cpu_only (dense) and cpu_sparse (CSR).

usage: python3 plot_compare.py benchall_<id>/raw.csv [OUTDIR]
Needs only matplotlib (no pandas), so it runs on a Vista login node.

  1_solve_time_vs_size.png    solve time vs N, both programs        (the headline)
  2_time_per_iter_vs_size.png time per CG iteration vs N            (O(N^2) vs O(nnz))
  3_sparse_speedup.png        how many times faster CSR is, per matrix
  4_memory_vs_size.png        matrix storage: dense vs CSR
  5_thread_scaling.png        speedup vs threads, one panel per program
Times are the median of the repeats; size plots use the highest thread count.
"""
import csv
import os
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed series colors: dense = orange, sparse = blue, every plot
COLOR = {"cpu_only": "#eb6834", "cpu_sparse": "#2a78d6", "cpu_sparse_v2": "#1baf7a"}
NAME = {"cpu_only": "dense (cpu_only)", "cpu_sparse": "sparse CSR (cpu_sparse)",
        "cpu_sparse_v2": "sparse CSR v2 (cpu_sparse_v2)"}
EXTRA = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 7, "legend.frameon": False, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(path):
    runs = defaultdict(list)          # (impl, matrix, threads) -> [ms]
    info = {}                          # (impl, matrix) -> dict
    for r in csv.DictReader(open(path)):
        k = (r["impl"], r["matrix"], int(r["threads"]))
        runs[k].append(float(r["solve_ms"]))
        try:
            res = float(r["rel_residual"])
        except ValueError:
            res = float("nan")
        it = int(r["iterations"])
        d = info.setdefault((r["impl"], r["matrix"]),
                            {"N": int(r["N"]), "nnz": r.get("nnz") or "", "iters": it, "ok": True})
        d["iters"] = it
        if not (res == res) or res > 1e-4 or it >= 5000:
            d["ok"] = False
    med = {k: statistics.median(v) for k, v in runs.items()}
    return med, info


def save(fig, outdir, name):
    fig.tight_layout()
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


def size_series(med, info, impl, tmax, per_iter=False):
    pts = []
    for (i, m), d in info.items():
        if i != impl or (i, m, tmax) not in med:
            continue
        y = med[(i, m, tmax)]
        if per_iter:
            y /= max(1, d["iters"])
        pts.append((d["N"], y, d["ok"], m))
    return sorted(pts)


def plot_vs_size(med, info, impls, tmax, per_iter, title, ylabel, outdir, name):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    any_bad = False
    for impl in impls:
        pts = size_series(med, info, impl, tmax, per_iter)
        if not pts:
            continue
        # the line joins converged runs only: a run that hit MAX_ITERS has no comparable time
        line = [p for p in pts if p[2] or per_iter]
        ax.plot([p[0] for p in line], [p[1] for p in line], "-", color=COLOR.get(impl, INK2),
                label=NAME.get(impl, impl), zorder=2)
        for n, y, ok, _ in pts:
            any_bad |= not ok
            ax.plot(n, y, "o", color=COLOR.get(impl, INK2),
                    markerfacecolor=COLOR.get(impl, INK2) if ok else SURFACE,
                    markeredgewidth=2, zorder=3)
    ax.set(xscale="log", yscale="log", xlabel="unknowns N", ylabel=ylabel, title=title)
    if any_bad:
        ax.plot([], [], "o", color=INK2, markerfacecolor=SURFACE, markeredgewidth=2,
                label="did not converge (not SPD)")
    ax.legend()
    save(fig, outdir, name)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "raw.csv"
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(csv_path) or ".", "plots")
    os.makedirs(outdir, exist_ok=True)
    med, info = load(csv_path)
    impls = [i for i in ("cpu_only", "cpu_sparse", "cpu_sparse_v2") if any(k[0] == i for k in info)]
    impls += sorted({k[0] for k in info} - set(impls))
    tmax = max(k[2] for k in med)

    # 1, 2: time vs size
    plot_vs_size(med, info, impls, tmax, False,
                 f"Solve time vs matrix size ({tmax} threads)", "solve time (ms)",
                 outdir, "1_solve_time_vs_size.png")
    plot_vs_size(med, info, impls, tmax, True,
                 f"Cost of one CG iteration ({tmax} threads)", "ms per iteration",
                 outdir, "2_time_per_iter_vs_size.png")

    # 3: sparse speedup per matrix (only where both ran)
    if "cpu_only" in impls and "cpu_sparse" in impls:
        rows = []
        for (i, m), d in info.items():
            if i == "cpu_sparse" and ("cpu_only", m, tmax) in med and (i, m, tmax) in med:
                lab = m if d["ok"] else m + " [not converged]"
                rows.append((d["N"], lab, med[("cpu_only", m, tmax)] / med[(i, m, tmax)]))
        rows.sort()
        if rows:
            fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(rows) + 1.6))
            y = range(len(rows))
            ax.barh(list(y), [r[2] for r in rows], color=COLOR["cpu_sparse"], height=0.6)
            ax.set_yticks(list(y), [f"{r[1]}  (N={r[0]})" for r in rows])
            ax.set_xscale("log")
            ax.axvline(1, color=INK2, linewidth=1)
            for yi, r in zip(y, rows):
                ax.text(r[2], yi, f"  {r[2]:.0f}x" if r[2] >= 10 else f"  {r[2]:.1f}x",
                        va="center", color=INK, fontsize=10)
            ax.set(xlabel="dense time / sparse time (higher = CSR faster)",
                   title=f"How much faster CSR is ({tmax} threads)")
            ax.grid(axis="y", visible=False)
            save(fig, outdir, "3_sparse_speedup.png")

    # 4: memory needed for the matrix
    pts = sorted({(d["N"], d["nnz"]) for (i, m), d in info.items() if d["nnz"]})
    if pts:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        N = [p[0] for p in pts]
        ax.plot(N, [n * n * 8 / 1e6 for n in N], "-o", color=COLOR["cpu_only"], label="dense: 8·N² bytes")
        ax.plot(N, [(12 * int(z) + 8 * (n + 1)) / 1e6 for n, z in pts], "-o",
                color=COLOR["cpu_sparse"], label="CSR: 12·nnz + 8·N bytes")
        ax.set(xscale="log", yscale="log", xlabel="unknowns N", ylabel="matrix storage (MB)",
               title="Memory the matrix needs")
        ax.legend()
        save(fig, outdir, "4_memory_vs_size.png")

    # 5: thread scaling, one panel per program, the largest converged matrices
    threads = sorted({k[2] for k in med})
    if len(threads) > 1:
        fig, axes = plt.subplots(1, len(impls), figsize=(6.2 * len(impls), 4.6), sharey=True,
                                 squeeze=False)
        for ax, impl in zip(axes[0], impls):
            mats = sorted([(d["N"], m) for (i, m), d in info.items() if i == impl and d["ok"]])[-5:]
            ax.plot(threads, threads, "--", color=INK2, linewidth=1.2, label="ideal")
            for c, (n, m) in zip(EXTRA, reversed(mats)):
                ts = [t for t in threads if (impl, m, t) in med]
                if len(ts) < 2:
                    continue
                base = med[(impl, m, ts[0])]
                ax.plot(ts, [base / med[(impl, m, t)] for t in ts], "-o", color=c,
                        label=f"{m} (N={n})")
            ax.set(xlabel="threads", title=NAME.get(impl, impl))
            ax.set_xscale("log", base=2)
            ax.set_xticks(threads, [str(t) for t in threads])
            ax.legend(fontsize=9)
        axes[0][0].set_ylabel("speedup vs 1 thread")
        save(fig, outdir, "5_thread_scaling.png")


if __name__ == "__main__":
    main()
