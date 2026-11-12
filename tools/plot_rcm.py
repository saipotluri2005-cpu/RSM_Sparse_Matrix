#!/usr/bin/env python3
"""Plots and a summary table from scripts/vista_cpu.slurm / vista_gpu.slurm results.

usage: python3 tools/plot_rcm.py RESULTS_DIR [RESULTS_DIR ...]

RESULTS_DIR is an unpacked qa_rcm_cpu_<job>/ or qa_rcm_gpu_<job>/ folder.
Writes RESULTS_DIR/plots/*.png and RESULTS_DIR/summary.md from whatever files
are present (rcm_cpu.csv, baseline.csv, study_*.csv, spy/, barrier.csv,
stream.txt, gpu_info.txt, spmv/spmv.csv). Needs numpy and matplotlib.

Generated with assistance of claude code
"""
import csv
import os
import re
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ["original", "shuffled", "rcm", "cm", "rcm_mindeg", "rcm_first", "shuffled_rcm"]
COLORS = {"original": "#4c72b0", "shuffled": "#c44e52", "rcm": "#55a868", "cm": "#8172b3",
          "rcm_mindeg": "#ccb974", "rcm_first": "#64b5cd", "shuffled_rcm": "#2f7d4f"}


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=130)
    plt.close(fig)


def rcm_cpu(d, plots, md):
    p = os.path.join(d, "rcm_cpu.csv")
    if not os.path.exists(p):
        return
    groups = defaultdict(list)
    for r in rows(p):
        groups[(r["matrix"], r["input"], r["reorder"], int(r["threads"]))].append(num(r["ms_per_iter"]))
    med = {k: statistics.median(v) for k, v in groups.items()}
    mats = sorted({k[0] for k in med})
    threads = sorted({k[3] for k in med})
    combos = [("natural", "none"), ("natural", "rcm"), ("shuffled", "none"), ("shuffled", "rcm")]
    md.append("## Time per CG iteration (ms, median), cpu_sparse_v2 + Jacobi\n")
    md.append("| matrix | threads | natural | natural + RCM | shuffled | shuffled + RCM | shuffled / shuffled+RCM |")
    md.append("|---|---|---|---|---|---|---|")
    for m in mats:
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.2
        x = np.arange(len(threads))
        for i, (inp, ro) in enumerate(combos):
            vals = [med.get((m, inp, ro, t), np.nan) for t in threads]
            ax.bar(x + (i - 1.5) * w, vals, w, label=f"{inp}{' + RCM' if ro == 'rcm' else ''}")
        ax.set_xticks(x, [str(t) for t in threads])
        ax.set_xlabel("threads")
        ax.set_ylabel("ms per CG iteration")
        ax.set_yscale("log")
        ax.set_title(f"{m}: numbering vs time per iteration")
        ax.legend(fontsize=8)
        save(fig, plots, f"rcm_cpu_{m}.png")
        for t in threads:
            v = [med.get((m, inp, ro, t), np.nan) for inp, ro in combos]
            md.append(f"| {m} | {t} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {v[2] / v[3]:.2f}x |")
    md.append("")


def baseline(d, md):
    p = os.path.join(d, "baseline.csv")
    if not os.path.exists(p):
        return
    md.append("## Full solves (ms): read + build (incl. RCM) + solve\n")
    md.append("| matrix | input | config | N | iterations | read | build | solve | total | residual |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows(p):
        tot = num(r["read_ms"]) + num(r["build_ms"]) + num(r["solve_ms"])
        md.append(f"| {r['matrix']} | {r['input']} | {r['config']} | {r['N']} | {r['iterations']} | "
                  f"{num(r['read_ms']):.0f} | {num(r['build_ms']):.0f} | {num(r['solve_ms']):.0f} | "
                  f"{tot:.0f} | {r['rel_residual']} |")
    md.append("")


def studies(d, plots, md):
    for name in sorted(os.listdir(d)):
        if not (name.startswith("study_") and name.endswith(".csv")):
            continue
        data = rows(os.path.join(d, name))
        if not data:
            continue
        tag = name[len("study_"):-4]
        by = defaultdict(dict)
        for r in data:
            by[r["matrix"]][r["ordering"]] = r
        mats = list(by)
        md.append(f"## Orderings ({name})\n")
        md.append("| matrix | N | nnz | " + " | ".join(f"bw {o}" for o in ORDER if any(o in by[m] for m in mats))
                  + " | SpMV ms original | SpMV ms shuffled | SpMV ms rcm | RCM ms |")
        cols = [o for o in ORDER if any(o in by[m] for m in mats)]
        md.append("|" + "---|" * (3 + len(cols) + 4))
        helped = hurt = same = 0
        for m in mats:
            o = by[m]
            first = next(iter(o.values()))
            bws = [o[c]["bandwidth"] if c in o else "" for c in cols]
            sp = [num(o[c]["spmv_ms"]) if c in o else float("nan") for c in ("original", "shuffled", "rcm")]
            rcm_ms = num(o["rcm"]["order_ms"]) if "rcm" in o else float("nan")
            md.append(f"| {m} | {first['N']} | {first['nnz']} | " + " | ".join(bws) +
                      f" | {sp[0]:.4g} | {sp[1]:.4g} | {sp[2]:.4g} | {rcm_ms:.1f} |")
            if "rcm" in o and "original" in o:
                a, b = int(o["original"]["bandwidth"]), int(o["rcm"]["bandwidth"])
                helped += b < a
                hurt += b > a
                same += b == a
        md.append(f"\nRCM vs original bandwidth: smaller on {helped}, larger on {hurt}, equal on {same} matrices.\n")
        # bandwidth and SpMV time per ordering, one panel per matrix (first 12)
        show = mats[:12]
        if not show:
            continue
        fig, axes = plt.subplots(len(show), 2, figsize=(10, 2.2 * len(show)), squeeze=False)
        for i, m in enumerate(show):
            o = by[m]
            labs = [c for c in cols if c in o]
            axes[i][0].bar(labs, [int(o[c]["bandwidth"]) for c in labs], color=[COLORS[c] for c in labs])
            axes[i][0].set_yscale("log")
            axes[i][0].set_ylabel(m, fontsize=8)
            axes[i][1].bar(labs, [num(o[c]["spmv_ms"]) for c in labs], color=[COLORS[c] for c in labs])
            for ax in axes[i]:
                ax.tick_params(axis="x", labelsize=7, rotation=30)
        axes[0][0].set_title("bandwidth (log)")
        axes[0][1].set_title("SpMV ms")
        save(fig, plots, f"orderings_{tag}.png")


def spy(d, plots):
    sd = os.path.join(d, "spy")
    if not os.path.isdir(sd):
        return
    grids = defaultdict(dict)
    for f in os.listdir(sd):
        m = re.match(r"spy_(.+)_(" + "|".join(sorted(ORDER, key=len, reverse=True)) + r")\.csv$", f)
        if m:
            grids[m.group(1)][m.group(2)] = os.path.join(sd, f)
    for mat, files in grids.items():
        labs = [o for o in ORDER if o in files]
        fig, axes = plt.subplots(1, len(labs), figsize=(2.2 * len(labs), 2.5), squeeze=False)
        for ax, o in zip(axes[0], labs):
            g = np.loadtxt(files[o], delimiter=",")
            ax.imshow(np.log1p(g), cmap="Greys", interpolation="nearest")
            ax.set_title(o, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{mat}: nonzero pattern per ordering", fontsize=9)
        save(fig, os.path.join(plots, "spy"), f"spy_{mat}.png")


def barrier(d, plots, md):
    p = os.path.join(d, "barrier.csv")
    if not os.path.exists(p):
        return
    data = [r for r in rows(p) if r.get("threads", "").isdigit()]
    fig, ax = plt.subplots(figsize=(6, 4))
    md.append("## OpenMP synchronization cost (ns)\n")
    md.append("| threads | bind | barrier | reduction |")
    md.append("|---|---|---|---|")
    for bind in ("close", "spread"):
        pts = sorted((int(r["threads"]), num(r["barrier_ns"]), num(r["reduction_ns"])) for r in data
                     if r["proc_bind"] == bind)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"barrier, {bind}")
            ax.plot([p[0] for p in pts], [p[2] for p in pts], "s--", label=f"reduction, {bind}")
            md.extend(f"| {t} | {bind} | {b:.0f} | {r:.0f} |" for t, b, r in pts)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("threads")
    ax.set_ylabel("ns per operation")
    ax.legend(fontsize=8)
    ax.set_title("cost of one barrier / reduction")
    save(fig, plots, "barrier.png")
    md.append("")


def stream(d, md):
    p = os.path.join(d, "stream.txt")
    if not os.path.exists(p):
        return
    md.append("## STREAM triad (best MB/s)\n")
    cfg = None
    for line in open(p):
        if line.startswith("###"):
            cfg = line[3:].strip()
        elif line.startswith("Triad:") and cfg:
            md.append(f"- {cfg}: {line.split()[1]} MB/s")
    md.append("")


def gpu(d, plots, md):
    info = os.path.join(d, "gpu_info.txt")
    if os.path.exists(info):
        md.append("## GPU\n")
        md.extend(f"- {line.strip()}" for line in open(info) if "=" in line)
        md.append("")
    p = os.path.join(d, "spmv", "spmv.csv")
    if not os.path.exists(p):
        return
    times = {}
    for r in rows(p):
        if r.get("strategy") != "hbm" or r.get("status") != "ok":
            continue
        base = os.path.basename(r["matrix"]).replace(".mtx", "").replace(".gz", "")
        order = "original"
        for suffix in ("_shuffled_rcm", "_shuffled"):
            if base.endswith(suffix):
                order, base = suffix[1:], base[: -len(suffix)]
                break
        times[(base, order)] = (num(r["repeat_ms"]), r["rows"], r["nnz"])
    cpu = {}
    sp = os.path.join(d, "study_gh_72t.csv")
    if os.path.exists(sp):
        for r in rows(sp):
            cpu[(r["matrix"], r["ordering"])] = num(r["spmv_ms"])
    mats = sorted({k[0] for k in times}, key=lambda m: int(times[(m, "original")][1]) if (m, "original") in times else 0)
    orders = ["original", "shuffled", "shuffled_rcm"]
    md.append("## SpMV time (ms): GPU (cuSPARSE, matrix in HBM) and Grace CPU (72 threads)\n")
    md.append("| matrix | rows | nnz | GPU original | GPU shuffled | GPU shuffled+RCM | CPU original | CPU shuffled | CPU shuffled+RCM |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for m in mats:
        g = [times.get((m, o), (float("nan"),))[0] for o in orders]
        c = [cpu.get((m, o), float("nan")) for o in orders]
        rws, nnz = times.get((m, "original"), (0, "", ""))[1:]
        md.append(f"| {m} | {rws} | {nnz} | " + " | ".join(f"{x:.4g}" for x in g + c) + " |")
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.25
    x = np.arange(len(mats))
    for i, o in enumerate(orders):
        ax.bar(x + (i - 1) * w, [times.get((m, o), (np.nan,))[0] for m in mats], w, label=o, color=COLORS[o])
    ax.set_xticks(x, mats, fontsize=8)
    ax.set_ylabel("GPU SpMV ms (warm mean)")
    ax.set_title("GPU SpMV time by matrix numbering")
    ax.legend(fontsize=8)
    save(fig, plots, "gpu_spmv.png")
    md.append("")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        plots = os.path.join(d, "plots")
        md = [f"# Results: {os.path.basename(os.path.normpath(d))}\n"]
        prog = os.path.join(d, "progress.txt")
        if os.path.exists(prog):
            md.append("```\n" + open(prog).read().strip() + "\n```\n")
        rcm_cpu(d, plots, md)
        baseline(d, md)
        studies(d, plots, md)
        spy(d, plots)
        barrier(d, plots, md)
        stream(d, md)
        gpu(d, plots, md)
        with open(os.path.join(d, "summary.md"), "w") as f:
            f.write("\n".join(md) + "\n")
        print(f"wrote {os.path.join(d, 'summary.md')} and {plots}/")


if __name__ == "__main__":
    main()
