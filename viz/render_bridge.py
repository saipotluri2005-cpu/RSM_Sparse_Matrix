#!/usr/bin/env python3
"""Render a solved truss bridge as an engineering animation.

  # truck driving across (gen_bridge.py --moving: one GPU solve per deck joint)
  python3 render_bridge.py viz_cases/bridge_mesh.txt --solutions viz_out \
          --cases viz_cases --report viz_out/gpu_report.txt --out viz_out/bridge.mp4

  # single load case (gen_bridge.py without --moving): the load ramps up
  python3 render_bridge.py viz_cases/bridge_mesh.txt --solutions viz_out --out bridge.mp4

Inputs
  *_mesh.txt               joints, bars, unknown numbering and load-case names (gen_bridge.py)
  <case>_gpu_solution.txt  displacements from gpu_cg (RHS=file SAVE_SOLUTION=1), per load case
  --cases DIR              the .mtx / _b.mtx files: checks ||K u - b|| / ||b|| and compares
                           with a direct solve (needs scipy), independently of the GPU code
  --report FILE            gpu_cg's printed report: GPU name, iterations and solve times
                           are stamped on the video

The structure is linear, so a truck between joints p and p+1 gives exactly the blend of
the two solved cases. Writes MP4 when ffmpeg works, otherwise a GIF, plus a PNG still.
Generated with assistance of claude code
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

# member force colors: compression / unstressed steel / tension
CMAP = LinearSegmentedColormap.from_list("force", ["#2f6fb0", "#8e969e", "#c43c2c"])
STEEL_EDGE, CONCRETE, CONCRETE_EDGE = "#22262b", "#c2beb6", "#8c877f"


# ---------------------------------------------------------------------------- input
def read_mesh(path):
    with open(path) as f:
        rows = [l.split() for l in f if l.strip() and not l.startswith("#")]
    k = 0
    nj = int(rows[k][1]); k += 1
    xy = np.array([[float(v) for v in rows[k + j][:2]] for j in range(nj)])
    dof = np.array([[int(v) for v in rows[k + j][2:4]] for j in range(nj)]); k += nj
    nb = int(rows[k][1]); k += 1
    bars = [(int(r[0]), int(r[1]), float(r[2])) for r in rows[k:k + nb]]; k += nb
    meta = {r[0]: r[1:] for r in rows[k:]}
    E = float(meta.get("E", ["200e9"])[0])
    stem = os.path.basename(path)[:-len("_mesh.txt")]
    cases = meta.get("cases", [stem])
    truck = meta.get("truck", ["0", "0"])
    return xy, dof, bars, E, cases, float(truck[0]), truck[1] == "moving"


def read_solution(path, n):
    u = np.full(n, np.nan)
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                u[int(p[0])] = float(p[1])
    if np.isnan(u).any():
        sys.exit(f"{path}: expected {n} values")
    return u


def read_report(path):
    """gpu_cg report -> (gpu name, {case: (iterations, solve_ms, residual)})"""
    gpu, cases, cur = "GPU", {}, None
    for line in open(path):
        if line.startswith("GPU:"):
            gpu = line[4:].split(",")[0].strip()
        m = re.match(r"Processing File:\s*(.*)", line)
        if m:
            cur = os.path.basename(m.group(1).strip()).replace(".gz", "").replace(".mtx", "")
            cases[cur] = [0, 0.0, 0.0]
        elif cur and line.startswith("Iterations Taken"):
            cases[cur][0] = int(line.split()[2])
        elif cur and line.startswith("Solve Time"):
            cases[cur][1] = float(line.split()[2])
        elif cur and line.startswith("Relative Residual"):
            cases[cur][2] = float(line.split()[2])
    return gpu, cases


def check(cases_dir, names, U):
    try:
        import scipy.io
        import scipy.sparse.linalg as sla
    except ImportError:
        print("check: scipy not available, skipped")
        return
    worst_res = worst_err = 0.0
    for name, u in zip(names, U):
        K = scipy.io.mmread(os.path.join(cases_dir, name + ".mtx")).tocsr()
        b = np.asarray(scipy.io.mmread(os.path.join(cases_dir, name + "_b.mtx"))).ravel()
        worst_res = max(worst_res, np.linalg.norm(K @ u - b) / np.linalg.norm(b))
        ud = sla.spsolve(K.tocsc(), b)
        worst_err = max(worst_err, np.abs(u - ud).max() / np.abs(ud).max())
    print(f"check over {len(names)} load case(s): worst ||K u - b||/||b|| = {worst_res:.2e}, "
          f"worst difference from a direct solve = {worst_err:.2e}")


def ffmpeg_works():
    # an ffmpeg on PATH can still fail to start (Vista: missing libunwind.so.8)
    if not shutil.which("ffmpeg"):
        return False
    try:
        return subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# ---------------------------------------------------------------------------- scene
def nice(x):
    e = 10 ** math.floor(math.log10(x))
    return e * (5 if x / e >= 5 else 2 if x / e >= 2 else 1)


def truck_patches(ax):
    """semi-truck facing right; local coords: x from -7 to +7 m, wheels on y = 0"""
    parts = {
        "trailer": FancyBboxPatch((-7.0, 1.05), 10.2, 2.75, boxstyle="round,pad=0,rounding_size=0.15",
                                  fc="#eceae6", ec="#5c6168", lw=1.2, zorder=6),
        "chassis": Rectangle((-7.0, 0.75), 14.0, 0.32, fc="#2b2e33", ec="none", zorder=6),
        "cab": Polygon([[3.6, 0.85], [6.9, 0.85], [6.9, 2.3], [6.2, 3.45], [3.6, 3.45]], closed=True,
                       fc="#28466a", ec="#1b2f47", lw=1.2, zorder=6),
        "window": Polygon([[5.35, 2.35], [6.55, 2.35], [6.05, 3.15], [5.35, 3.15]], closed=True,
                          fc="#a9c0d3", ec="none", zorder=7),
    }
    wheels = [Circle((x, 0.48), 0.48, fc="#1d1f22", ec="#55595f", lw=1.0, zorder=7) for x in (-5.9, -4.8, 1.6, 2.7, 5.9)]
    for p in list(parts.values()) + wheels:
        ax.add_patch(p)
    base = {k: (p.get_xy() if not isinstance(p, FancyBboxPatch) else None) for k, p in parts.items()}
    return parts, wheels, base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("--solutions", default=".", help="folder with <case>_gpu_solution.txt")
    ap.add_argument("--cases", help="folder with the .mtx / _b.mtx files, for the check")
    ap.add_argument("--report", help="gpu_cg report, to stamp the GPU run on the video")
    ap.add_argument("--out", default="bridge.mp4")
    ap.add_argument("--seconds", type=float, default=7.0)
    a = ap.parse_args()

    xy, dof, bars, E, names, truck_N, moving = read_mesh(a.mesh)
    n = int(dof.max()) + 1
    U = [read_solution(os.path.join(a.solutions, nm + "_gpu_solution.txt"), n) for nm in names]
    if a.cases:
        check(a.cases, names, U)

    def joint_disp(u):
        d = np.zeros_like(xy)
        for j in range(len(xy)):
            for c in range(2):
                if dof[j, c] >= 0:
                    d[j, c] = u[dof[j, c]]
        return d

    D = np.array([joint_disp(u) for u in U])            # (cases, joints, 2)
    span = xy[:, 0].max()
    H = xy[:, 1].max()
    deck = sorted([j for j in range(len(xy)) if xy[j, 1] == 0], key=lambda j: xy[j, 0])
    deck_x = xy[deck, 0]
    panel = deck_x[1] - deck_x[0]

    def disp_at(xt):
        """displacements with the truck centred at xt (exact blend of neighbouring cases)"""
        if not moving:
            return D[0]
        if xt <= 0 or xt >= span:
            return D[0]                                   # truck off the bridge: dead load only
        p = min(len(D) - 2, int(xt // panel))
        t = (xt - p * panel) / panel
        return (1 - t) * D[p] + t * D[p + 1]

    def forces(d):
        f = []
        for a_, b_, area in bars:
            L0 = np.hypot(*(xy[b_] - xy[a_]))
            e = (xy[b_] - xy[a_]) / L0
            f.append(E * area / L0 * np.dot(e, d[b_] - d[a_]))
        return np.array(f)

    F_all = np.array([forces(d) for d in D])
    fmax = np.abs(F_all).max()
    dmax = np.abs(D).max()
    exag = nice(0.16 * H / dmax) if dmax > 0 else 1.0
    amax = max(b[2] for b in bars)

    # GPU stamp
    stamp = ""
    if a.report and os.path.exists(a.report):
        gpu, rep = read_report(a.report)
        got = [rep[nm] for nm in names if nm in rep]
        if got:
            its = [g[0] for g in got]
            ms = sum(g[1] for g in got)
            stamp = (f"Solved on {gpu} with Conjugate Gradient: {len(got)} load case{'s' if len(got) > 1 else ''}, "
                     f"{n} unknowns each, {min(its)}–{max(its)} iterations, {ms:.1f} ms total")
    if not stamp:
        stamp = f"Finite-element model: {n} unknowns, solved with Conjugate Gradient"

    # ---- figure: full-bleed scene ---------------------------------------------------
    gif = not (a.out.endswith(".mp4") and ffmpeg_works())
    out = a.out[:-4] + ".gif" if (gif and a.out.endswith(".mp4")) else a.out
    W_in, H_in, dpi = (12.8, 6.4, 100) if gif else (16, 8, 120)
    fps = 20 if gif else 30
    fig = plt.figure(figsize=(W_in, H_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    x0, x1 = -0.22 * span, 1.22 * span
    yspan = (x1 - x0) * H_in / W_in
    y0 = -0.62 * yspan
    y1 = y0 + yspan
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    water_y = -0.9 * H
    plt.rcParams["font.family"] = "DejaVu Sans"

    # sky and distant hills
    sky = np.linspace(0, 1, 256)[:, None]
    ax.imshow(sky, extent=(x0, x1, water_y, y1), origin="lower", aspect="auto", zorder=0,
              cmap=LinearSegmentedColormap.from_list("sky", ["#eef1f3", "#c9d6e0"]))
    hx = np.linspace(x0, x1, 200)
    hills = water_y + 0.55 * H + 0.22 * H * np.sin(hx / span * 5.1) + 0.12 * H * np.sin(hx / span * 13.7 + 1)
    ax.fill_between(hx, water_y, hills, color="#b7c3c9", zorder=0.5, lw=0)
    # water with ripples
    ax.imshow(np.linspace(1, 0, 64)[:, None], extent=(x0, x1, y0, water_y), origin="lower", aspect="auto",
              zorder=0.6, cmap=LinearSegmentedColormap.from_list("w", ["#3c5566", "#6c8798"]))
    rng = np.random.default_rng(7)
    for _ in range(70):
        rx, ry = rng.uniform(x0, x1), rng.uniform(y0 + 0.5, water_y - 0.3)
        ax.plot([rx, rx + rng.uniform(1.5, 5)], [ry, ry], color="#9fb4c2", lw=0.7, alpha=0.5, zorder=0.7)

    # abutments, embankments and approach roads
    for side in (0, 1):
        xs = 0.0 if side == 0 else span
        sgn = -1 if side == 0 else 1
        wall = [[xs + sgn * 3.2, water_y - 1.0], [xs - sgn * 1.0, water_y - 1.0],
                [xs - sgn * 1.0, -1.0], [xs + sgn * 3.2, -1.0]]
        bank = [[xs + sgn * 3.0, -1.0], [x1 if side else x0, -1.0], [x1 if side else x0, y0],
                [xs + sgn * 0.5, y0], [xs + sgn * 0.5, water_y - 1.0], [xs + sgn * 3.0, water_y - 1.0]]
        ax.add_patch(Polygon(bank, closed=True, fc="#86796a", ec="none", zorder=1))
        ax.plot([xs + sgn * 3.0, x1 if side else x0], [-0.95, -0.95], color="#6e7d57", lw=3, zorder=1.1)
        ax.add_patch(Polygon(wall, closed=True, fc=CONCRETE, ec=CONCRETE_EDGE, lw=1.2, zorder=2))
        ax.add_patch(Rectangle((min(xs + sgn * 3.0, x1 if side else x0), -0.75),
                               abs((x1 if side else x0) - (xs + sgn * 3.0)), 0.75, fc="#3a3d42", zorder=2))
        ax.add_patch(Rectangle((xs - 0.45, -0.95), 0.9, 0.45, fc="#2b2e33", zorder=3))   # bearing

    # far truss (behind the deck), deck slab, railing, near truss (in front)
    seg = lambda p: [[p[a_], p[b_]] for a_, b_, _ in bars]
    widths = np.array([6.5 if area > 0.75 * amax else 4.2 for _, _, area in bars]) * (W_in / 16)
    far = LineCollection([], colors="#a9b0b7", linewidths=widths * 0.55, zorder=3, capstyle="round")
    ax.add_collection(far)
    slab = Polygon(np.zeros((4, 2)), closed=True, fc="#b3afa8", ec="#7f7a73", lw=1.0, zorder=4)
    road = Polygon(np.zeros((4, 2)), closed=True, fc="#35383d", ec="none", zorder=4.2)
    ax.add_patch(slab); ax.add_patch(road)
    rail, = ax.plot([], [], color="#5d636a", lw=1.4, zorder=4.3)
    posts = LineCollection([], colors="#5d636a", linewidths=1.0, zorder=4.3)
    ax.add_collection(posts)
    parts, wheels, _ = truck_patches(ax)
    outline = LineCollection([], colors=STEEL_EDGE, linewidths=widths + 2.4, zorder=8, capstyle="round")
    near = LineCollection([], cmap=CMAP, norm=Normalize(-fmax, fmax), linewidths=widths, zorder=8.1, capstyle="round")
    ax.add_collection(outline); ax.add_collection(near)
    gussets = ax.scatter(xy[:, 0], xy[:, 1], marker="s", s=60 * (W_in / 16) ** 2, color="#30353b", zorder=8.2)
    ghost = LineCollection(seg(xy), colors="#ffffff", linewidths=1.0, linestyles=(0, (3, 3)), alpha=0.55, zorder=8.3)
    ax.add_collection(ghost)

    # titles and HUD
    fs = W_in / 16
    fig.text(0.03, 0.935, f"{span:.0f} m steel Pratt truss bridge: truck crossing" if moving else
             f"{span:.0f} m steel Pratt truss bridge under load", fontsize=21 * fs, weight="bold", color="#1b1e22")
    fig.text(0.03, 0.895, stamp, fontsize=12.5 * fs, color="#30353b")
    fig.add_artist(Rectangle((0, 0), 1, 0.115, transform=fig.transFigure, fc="#101317", alpha=0.82, zorder=20))
    hud = fig.text(0.03, 0.040, "", fontsize=12.5 * fs, color="#f2f3f5", family="DejaVu Sans Mono", zorder=21)
    fig.text(0.03, 0.078, f"Deflections drawn ×{exag:g}   ·   dashed white = unloaded shape",
             fontsize=11 * fs, color="#aab1b9", zorder=21)
    cax = fig.add_axes([0.70, 0.052, 0.26, 0.022], zorder=22)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-fmax / 1e3, fmax / 1e3), cmap=CMAP), cax=cax,
                      orientation="horizontal")
    cb.outline.set_visible(False)
    cax.tick_params(colors="#d5d9de", labelsize=10 * fs, length=0)
    cax.set_title("member axial force (kN)   compression ← → tension", fontsize=10.5 * fs,
                  color="#d5d9de", pad=4)

    n_frames = int(fps * a.seconds)
    t_start, t_end = -0.2 * span, 1.2 * span

    def deck_y(x, p):
        return np.interp(x, p[deck, 0], p[deck, 1])

    def frame(k):
        s = k / max(1, n_frames - 1)
        if moving:
            xt = t_start + (t_end - t_start) * s
            d = disp_at(xt)
            lam = 1.0
        else:
            lam = min(1.0, s / 0.75)
            lam = lam * lam * (3 - 2 * lam)
            d = lam * D[0]
            xt = deck_x[np.argmin(np.abs(deck_x - 0.33 * span))]
        p = xy + exag * d
        near.set_segments(seg(p)); outline.set_segments(seg(p))
        near.set_array(forces(d))
        far.set_segments([[q + [0.55, 0.45] for q in s_] for s_ in seg(p)])
        gussets.set_offsets(p)
        bx, by = p[deck, 0], p[deck, 1]
        slab.set_xy(np.vstack([np.c_[bx, by + 0.15], np.c_[bx[::-1], by[::-1] - 0.75]]))
        road.set_xy(np.vstack([np.c_[bx, by + 0.28], np.c_[bx[::-1], by[::-1] + 0.12]]))
        rail.set_data(bx, by + 1.25)
        posts.set_segments([[[x, deck_y(x, p) + 0.28], [x, deck_y(x, p) + 1.25]] for x in np.arange(0, span + 0.1, 2.5)])
        # truck sits on the deformed road (or the approach road when off the bridge)
        wy = lambda x: deck_y(x, p) + 0.28 if 0 <= x <= span else 0.0
        base = 0.5 * (wy(xt - 5.4) + wy(xt + 4.2))
        show = moving or truck_N > 0
        for w, wx in zip(wheels, (-5.9, -4.8, 1.6, 2.7, 5.9)):
            w.center = (xt + wx, wy(xt + wx) + 0.48); w.set_visible(show)
        parts["trailer"].set_bounds(xt - 7.0, base + 1.05, 10.2, 2.75)
        parts["chassis"].set_xy((xt - 7.0, base + 0.75))
        parts["cab"].set_xy(np.array([[3.6, 0.85], [6.9, 0.85], [6.9, 2.3], [6.2, 3.45], [3.6, 3.45]]) + [xt, base])
        parts["window"].set_xy(np.array([[5.35, 2.35], [6.55, 2.35], [6.05, 3.15], [5.35, 3.15]]) + [xt, base])
        for prt in parts.values():
            prt.set_visible(show)
        f = forces(d)
        under = -np.interp(np.clip(xt, 0, span), xy[deck, 0], d[deck, 1]) * 1000
        where = f"{np.clip(xt, 0, span):5.1f} m" if 0 <= xt <= span else "off span"
        hud.set_text(f"truck {where:>8} | sag under truck {under:5.1f} mm | "
                     f"tension {f.max() / 1e3:4.0f} kN | compression {-f.min() / 1e3:4.0f} kN"
                     if moving else
                     f"load {lam * 100:3.0f}% | max sag {np.abs(d).max() * 1000:5.1f} mm | "
                     f"tension {f.max() / 1e3:4.0f} kN | compression {-f.min() / 1e3:4.0f} kN")
        return []

    anim = animation.FuncAnimation(fig, frame, frames=n_frames, blit=False)
    if gif:
        if a.out.endswith(".mp4"):
            print("ffmpeg not usable here: writing a GIF instead")
        anim.save(out, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=6000))
    frame(int(n_frames * (0.45 if moving else 0.99)))
    fig.savefig(out.rsplit(".", 1)[0] + ".png", dpi=dpi)
    print(f"wrote {out} and {out.rsplit('.', 1)[0]}.png  ({len(names)} load case(s), "
          f"max deflection {dmax * 1000:.1f} mm, max force {fmax / 1e3:.0f} kN, drawn x{exag:g})")


if __name__ == "__main__":
    main()
