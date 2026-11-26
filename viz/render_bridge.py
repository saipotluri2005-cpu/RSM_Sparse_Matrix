#!/usr/bin/env python3
"""Animate a solved truss bridge: the load ramps up, the bridge sags, bars change color.

  python3 render_bridge.py viz_cases/bridge_mesh.txt viz_out/bridge_gpu_solution.txt \
          --matrix viz_cases/bridge.mtx --out viz_out/bridge.mp4

Inputs
  *_mesh.txt            from gen_bridge.py (joints, bars, unknown numbering)
  *_gpu_solution.txt    from gpu_cg with SAVE_SOLUTION=1 RHS=file ("i value" per line,
                        original numbering)
  --matrix (optional)   K and <name>_b.mtx: checks ||K u - b|| / ||b|| independently of
                        the GPU code, and compares with a direct solve if scipy is present

The structure is linear, so the displacement at load factor s is just s * u: one GPU
solve gives the whole animation. Writes MP4 if ffmpeg is installed, otherwise a GIF,
plus a PNG of the last frame.
Generated with assistance of claude code
"""
import argparse
import math
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

COMP, MID, TENS = "#2f6db5", "#b4b8bf", "#d1442c"     # compression, unloaded, tension
CMAP = LinearSegmentedColormap.from_list("force", [COMP, MID, TENS])


def read_mesh(path):
    with open(path) as f:
        lines = [l.split() for l in f if l.strip() and not l.startswith("#")]
    k = 0
    nj = int(lines[k][1]); k += 1
    xy = np.array([[float(v) for v in lines[k + j][:2]] for j in range(nj)])
    dof = np.array([[int(v) for v in lines[k + j][2:4]] for j in range(nj)])
    k += nj
    nb = int(lines[k][1]); k += 1
    bars = [(int(l[0]), int(l[1]), float(l[2]), l[3]) for l in lines[k:k + nb]]
    k += nb
    E = float(lines[k][1]) if k < len(lines) and lines[k][0] == "E" else 200e9
    return xy, dof, bars, E


def read_solution(path, n):
    u = np.full(n, np.nan)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                u[int(parts[0])] = float(parts[1])
    if np.isnan(u).any():
        sys.exit(f"{path}: expected {n} values, some are missing")
    return u


def check(matrix, u):
    try:
        import scipy.io
        import scipy.sparse.linalg as sla
    except ImportError:
        print("check: scipy not available, skipped")
        return
    K = scipy.io.mmread(matrix).tocsr()
    b = np.asarray(scipy.io.mmread(matrix.replace(".mtx", "_b.mtx"))).ravel()
    rel = np.linalg.norm(K @ u - b) / np.linalg.norm(b)
    ud = sla.spsolve(K.tocsc(), b)
    err = np.abs(u - ud).max() / np.abs(ud).max()
    print(f"check: ||K u - b|| / ||b|| = {rel:.2e}; GPU vs direct solve, max relative difference = {err:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("solution")
    ap.add_argument("--matrix", help="bridge.mtx, for the residual / direct-solve check")
    ap.add_argument("--out", default="bridge.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=4.0, help="length of the load ramp")
    ap.add_argument("--title", default="Steel truss bridge under load, solved with CG on the GPU")
    a = ap.parse_args()

    xy, dof, bars, E = read_mesh(a.mesh)
    n = int(dof.max()) + 1
    u = read_solution(a.solution, n)
    if a.matrix:
        check(a.matrix, u)

    # joint displacements (supports stay at 0)
    d = np.zeros_like(xy)
    for j in range(len(xy)):
        for c in range(2):
            if dof[j, c] >= 0:
                d[j, c] = u[dof[j, c]]
    # axial force per bar at full load, positive = tension
    force = []
    for a_, b_, area, _ in bars:
        L0 = np.hypot(*(xy[b_] - xy[a_]))
        e = (xy[b_] - xy[a_]) / L0
        force.append(E * area / L0 * np.dot(e, d[b_] - d[a_]))
    force = np.array(force)
    fmax = np.abs(force).max()
    amax = max(b_[2] for b_ in bars)
    width = np.array([2.6 if area > 0.75 * amax else 1.6 for _, _, area, _ in bars])

    span = xy[:, 0].max() - xy[:, 0].min()
    height = xy[:, 1].max()
    dmax = np.abs(d).max()
    raw = 0.25 * height / dmax if dmax > 0 else 1.0
    exag = 10 ** math.floor(math.log10(raw))
    exag *= 5 if raw / exag >= 5 else 2 if raw / exag >= 2 else 1
    deck = [j for j in range(len(xy)) if xy[j, 1] == 0 and dof[j, 1] >= 0]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(12, 5.4), dpi=110)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.85, bottom=0.17)
    ax.set_aspect("equal")
    ax.axis("off")
    pad = 0.06 * span
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(-0.95 * height, height * 1.12)

    seg0 = [[xy[a_], xy[b_]] for a_, b_, _, _ in bars]
    ax.add_collection(LineCollection(seg0, colors="#d6d8dc", linewidths=1, linestyles=(0, (4, 3)), zorder=1))
    lc = LineCollection(seg0, cmap=CMAP, norm=Normalize(-fmax, fmax), linewidths=width, capstyle="round", zorder=3)
    lc.set_array(np.zeros(len(bars)))
    ax.add_collection(lc)
    joints = ax.scatter(xy[:, 0], xy[:, 1], s=14, color="#2a2c30", zorder=4)
    # supports: pin (left), roller (right)
    s = 0.03 * span
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    ax.add_patch(plt.Polygon([[x0, 0], [x0 - s, -1.6 * s], [x0 + s, -1.6 * s]], color="#5d6068", zorder=2))
    ax.add_patch(plt.Polygon([[x1, 0], [x1 - s, -1.3 * s], [x1 + s, -1.3 * s]], color="#5d6068", zorder=2))
    ax.add_patch(plt.Circle((x1, -1.55 * s), 0.25 * s, color="#5d6068", zorder=2))
    ax.plot([x0 - 2 * s, x0 + 2 * s], [-1.65 * s] * 2, color="#5d6068", lw=2, zorder=2)
    ax.plot([x1 - 2 * s, x1 + 2 * s], [-1.85 * s] * 2, color="#5d6068", lw=2, zorder=2)

    # loads: read back from the right-hand side if we have it, else uniform deck arrows
    loads = np.zeros(len(xy))
    if a.matrix:
        import scipy.io
        b = np.asarray(scipy.io.mmread(a.matrix.replace(".mtx", "_b.mtx"))).ravel()
        for j in deck:
            loads[j] = -b[dof[j, 1]]
    else:
        loads[deck] = 1.0
    lmax = loads.max() if loads.max() > 0 else 1.0
    arrows = [ax.annotate("", xy=(xy[j, 0], 0), xytext=(xy[j, 0], 0),
                          arrowprops=dict(arrowstyle="-|>", color="#ff4f00" if loads[j] > 1.5 * np.median(loads[deck]) else "#7a7d84",
                                          lw=2, mutation_scale=14), zorder=5, annotation_clip=False) for j in deck]

    fig.text(0.03, 0.93, a.title, fontsize=16, weight="bold")
    info = fig.text(0.03, 0.875, "", fontsize=12, color="#3a3c40")
    cax = fig.add_axes([0.30, 0.09, 0.40, 0.035])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-fmax / 1e3, fmax / 1e3), cmap=CMAP), cax=cax, orientation="horizontal")
    cb.set_label("axial force at full load (kN):   compression  ←  0  →  tension", fontsize=11)
    cb.outline.set_visible(False)

    nramp = int(a.fps * a.seconds)
    nhold = int(a.fps * 1.5)

    def frame(k):
        t = min(1.0, k / max(1, nramp))
        lam = t * t * (3 - 2 * t)                       # ease in and out
        p = xy + lam * exag * d
        lc.set_segments([[p[a_], p[b_]] for a_, b_, _, _ in bars])
        lc.set_array(lam * force)
        joints.set_offsets(p)
        for arr, j in zip(arrows, deck):
            # weight hanging from the deck: tail just under the joint, head pointing down
            y0 = p[j, 1] - 0.04 * height
            arr.set_position((p[j, 0], y0))
            arr.xy = (p[j, 0], y0 - max(1e-3, lam) * 0.4 * height * loads[j] / lmax)
        info.set_text(f"load {lam * 100:3.0f}%   |   largest deflection {lam * dmax * 1000:6.1f} mm (drawn ×{exag:g})"
                      f"   |   largest bar force {lam * fmax / 1e3:6.0f} kN   |   {n} unknowns")
        return [lc, joints, info] + arrows

    anim = animation.FuncAnimation(fig, frame, frames=nramp + nhold + 1, blit=False)
    out = a.out
    if out.endswith(".mp4") and shutil.which("ffmpeg"):
        anim.save(out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=2400))
    else:
        if out.endswith(".mp4"):
            print("ffmpeg not found: writing a GIF instead")
            out = out[:-4] + ".gif"
        anim.save(out, writer=animation.PillowWriter(fps=a.fps))
    frame(nramp)
    fig.savefig(out.rsplit(".", 1)[0] + ".png")
    print(f"wrote {out} and {out.rsplit('.', 1)[0]}.png  (max deflection {dmax * 1000:.1f} mm, "
          f"max force {fmax / 1e3:.0f} kN)")


if __name__ == "__main__":
    main()
