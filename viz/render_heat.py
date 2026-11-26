#!/usr/bin/env python3
"""Render a transient heat run from gpu_cg (HEAT_STEPS mode) as a cutaway 3D animation.

  python3 render_heat.py viz_cases/heat64_mesh.txt --frames viz_out \
          --report viz_out/heat_report.txt --out viz_out/heat64.mp4

Inputs
  <name>_mesh.txt     grid size and heat sources (gen_heat.py). For a plain poisson3d_N
                      matrix pass --grid N instead.
  <name>_frames.txt   index of temperature frames written by gpu_cg (FRAME_DIR), one
                      float32 value per grid point, x fastest, then y, then z
  --report FILE       gpu_cg's printed report, stamped on the video

The block is drawn with the corner nearest the viewer cut away, through the hottest
point, so the inside shows; the camera turns slowly while the block heats up. Writes
MP4 when ffmpeg works, otherwise a GIF, plus a PNG of the last frame.
Generated with assistance of claude code
"""
import argparse
import os
import re
import shutil
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from matplotlib.patches import Rectangle

# cold metal (slate) through red-orange to white-hot; sequential, lightness rises with temperature
CMAP = LinearSegmentedColormap.from_list(
    "thermal", ["#27313d", "#3d3f63", "#6b2f6e", "#a8323f", "#dc5a1e", "#f59d26", "#fbe39a", "#fffbf0"])


def ffmpeg_works():
    if not shutil.which("ffmpeg"):
        return False
    try:
        return subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def read_report(path, stem):
    gpu, info, cur = "the GPU", {}, None
    for line in open(path):
        if line.startswith("GPU:"):
            gpu = line[4:].split(",")[0].strip()
        m = re.match(r"Processing File:\s*(.*)", line)
        if m:
            cur = os.path.basename(m.group(1).strip()).replace(".gz", "").replace(".mtx", "")
        if cur != stem:
            continue
        if line.startswith("Iterations Taken"):
            m = re.search(r"\((\d+)-(\d+) per step\)", line)
            if m:
                info["its"] = (int(m.group(1)), int(m.group(2)))
        elif line.startswith("Solve Time"):
            info["ms"] = float(line.split()[2])
        elif line.startswith("Heat Steps"):
            info["steps"] = int(line.split()[2])
    return gpu, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", nargs="?", help="<name>_mesh.txt from gen_heat.py")
    ap.add_argument("--grid", type=int, help="grid size n, if there is no mesh file")
    ap.add_argument("--name", help="matrix name (default: from the mesh file name)")
    ap.add_argument("--frames", default=".", help="folder with <name>_frames.txt and the frame files")
    ap.add_argument("--report", help="gpu_cg report, to stamp the GPU run on the video")
    ap.add_argument("--out", default="heat.mp4")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds to hold the last frame")
    a = ap.parse_args()

    n, chips = a.grid, []
    name = a.name
    if a.mesh:
        for line in open(a.mesh):
            p = line.split()
            if p and p[0] == "grid":
                n = int(p[1])
            elif p and p[0] == "chip":
                chips.append([float(v) for v in p[1:]])
        name = name or os.path.basename(a.mesh)[:-len("_mesh.txt")]
    if not n or not name:
        ap.error("give a mesh file, or --grid and --name")

    rows = [l.split() for l in open(os.path.join(a.frames, name + "_frames.txt")) if not l.startswith("#")]
    steps = [int(r[0]) for r in rows]
    times = [float(r[1]) for r in rows]
    iters = [int(r[3]) for r in rows]
    frames = [np.fromfile(os.path.join(a.frames, r[4]), dtype=np.float32).reshape(n, n, n) for r in rows]  # [z, y, x]
    tpeak = [float(f.max()) for f in frames]
    tmax = tpeak[-1] if tpeak[-1] > 0 else 1.0
    norm = PowerNorm(0.55, 0, tmax)     # stretches the low end so the spreading heat shows

    stamp = f"Implicit time stepping, {n}³ = {n ** 3:,} unknowns per step, each step solved with Conjugate Gradient"
    if a.report and os.path.exists(a.report):
        gpu, info = read_report(a.report, name)
        if "ms" in info and "its" in info:
            stamp = (f"Solved on {gpu}: {info.get('steps', steps[-1])} time steps × {n ** 3:,} unknowns, "
                     f"{info['its'][0]}–{info['its'][1]} CG iterations per step, {info['ms']:.0f} ms of GPU solving")

    # cut through the hottest point (kept away from the walls so the notch reads)
    z0, y0, x0 = np.unravel_index(np.argmax(frames[-1]), frames[-1].shape)
    cx = int(np.clip(x0, 0.25 * n, 0.75 * n))
    cy = int(np.clip(y0 + 1, 0.3 * n, 0.75 * n))
    cz = int(np.clip(z0 - 1, 0.25 * n, 0.8 * n))
    s = max(1, int(np.ceil(n / 72)))           # draw at most ~72 cells per side

    # faces: (plane axis, plane coordinate, cell index arrays, shade, value lookup)
    faces = []

    def add(axis, coord, u_rng, v_rng, keep, shade, look):
        U, V = np.meshgrid(np.arange(*u_rng, s), np.arange(*v_rng, s), indexing="ij")
        m = keep(U, V)
        faces.append((axis, coord, U[m], V[m], shade, look))

    last = n - 1
    # outer faces (front y=0, right x=last, top z=last), minus the removed corner
    add("y", -0.5, (0, n), (0, n), lambda X, Z: ~((X > cx) & (Z > cz)), 0.90, lambda F, X, Z: F[Z, 0, X])
    add("x", last + 0.5, (0, n), (0, n), lambda Y, Z: ~((Y < cy) & (Z > cz)), 0.78, lambda F, Y, Z: F[Z, Y, last])
    add("z", last + 0.5, (0, n), (0, n), lambda X, Y: ~((X > cx) & (Y < cy)), 1.00, lambda F, X, Y: F[last, Y, X])
    # the three cut faces of the notch
    add("x", cx + 0.5, (0, cy), (cz + 1, n), lambda Y, Z: np.ones_like(Y, bool), 0.80, lambda F, Y, Z: F[Z, Y, cx])
    add("y", cy - 0.5, (cx + 1, n), (cz + 1, n), lambda X, Z: np.ones_like(X, bool), 0.92, lambda F, X, Z: F[Z, cy, X])
    add("z", cz + 0.5, (cx + 1, n), (0, cy), lambda X, Y: np.ones_like(X, bool), 1.00, lambda F, X, Y: F[cz, Y, X])

    def quads(axis, c, U, V):
        h = s / 2
        du = np.array([-h, h, h, -h]); dv = np.array([-h, -h, h, h])
        A = U[:, None] + du + (s - 1) / 2
        B = V[:, None] + dv + (s - 1) / 2
        Cc = np.full_like(A, c, dtype=float)
        if axis == "x":
            return np.stack([Cc, A, B], -1)
        if axis == "y":
            return np.stack([A, Cc, B], -1)
        return np.stack([A, B, Cc], -1)

    Q = [quads(ax_, c, U, V) for ax_, c, U, V, _, _ in faces]
    shade = np.concatenate([np.full(len(f[2]), f[4]) for f in faces])

    # outline edges of the visible regions
    e = -0.5
    L = last + 0.5
    edges3 = [
        [(e, e, e), (L, e, e)], [(L, e, e), (L, e, cz + .5)], [(e, e, e), (e, e, L)], [(e, e, L), (cx + .5, e, L)],
        [(L, e, e), (L, L, e)], [(L, L, e), (L, L, L)], [(L, L, L), (e, L, L)], [(e, L, L), (e, e, L)],
        [(L, L, L), (L, cy - .5, L)], [(cx + .5, e, L), (cx + .5, e, cz + .5)], [(cx + .5, e, cz + .5), (L, e, cz + .5)],
        [(L, e, cz + .5), (L, cy - .5, cz + .5)], [(L, cy - .5, cz + .5), (L, cy - .5, L)],
        [(cx + .5, e, cz + .5), (cx + .5, cy - .5, cz + .5)], [(cx + .5, cy - .5, cz + .5), (L, cy - .5, cz + .5)],
        [(cx + .5, cy - .5, cz + .5), (cx + .5, cy - .5, L)], [(cx + .5, cy - .5, L), (L, cy - .5, L)],
        [(cx + .5, e, L), (cx + .5, cy - .5, L)],
    ]
    edges3 = np.array(edges3, float)

    def project(P, az, el):
        c = (n - 1) / 2
        X, Y, Z = P[..., 0] - c, P[..., 1] - c, P[..., 2] - c
        x1 = X * np.cos(az) - Y * np.sin(az)
        y1 = X * np.sin(az) + Y * np.cos(az)
        sx = x1
        sy = Z * np.cos(el) + y1 * np.sin(el)
        depth = y1 * np.cos(el) - Z * np.sin(el)
        return np.stack([sx, sy], -1), depth

    # ---- figure -------------------------------------------------------------------
    gif = not (a.out.endswith(".mp4") and ffmpeg_works())
    out = a.out[:-4] + ".gif" if (gif and a.out.endswith(".mp4")) else a.out
    W_in, H_in, dpi = (12.8, 7.2, 100) if gif else (16, 9, 120)
    fps = 20 if gif else 24
    fs = W_in / 16
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(W_in, H_in), dpi=dpi)
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off")
    bg.imshow(np.linspace(0, 1, 256)[:, None], extent=(0, 1, 0, 1), aspect="auto", origin="lower",
              cmap=LinearSegmentedColormap.from_list("bg", ["#d9dde1", "#f4f5f6"]))
    ax = fig.add_axes([0.02, 0.12, 0.62, 0.78]); ax.axis("off"); ax.set_aspect("equal")
    r = 0.95 * n
    ax.set_xlim(-r, r); ax.set_ylim(-0.85 * r, 0.85 * r)
    shadow = PolyCollection([], facecolors="#000000", alpha=0.12, edgecolors="none")
    ax.add_collection(shadow)
    pc = PolyCollection([], edgecolors="none", antialiased=False)
    ax.add_collection(pc)
    ec = LineCollection([], colors="#1d1f23", linewidths=1.1)
    ax.add_collection(ec)

    fig.text(0.03, 0.94, "Heat spreading through a cooled metal block", fontsize=21 * fs, weight="bold", color="#1b1e22")
    fig.text(0.03, 0.905, stamp, fontsize=12 * fs, color="#30353b")
    # inset: peak temperature over time
    ia = fig.add_axes([0.70, 0.52, 0.26, 0.28])
    ia.set_facecolor("none")
    for sp_ in ("top", "right"):
        ia.spines[sp_].set_visible(False)
    for sp_ in ("left", "bottom"):
        ia.spines[sp_].set_color("#8a8f96")
    ia.tick_params(colors="#4a4f55", labelsize=10 * fs)
    ia.set_xlim(0, times[-1]); ia.set_ylim(0, 1.05)
    ia.set_title("hottest point, % of final", fontsize=11.5 * fs, color="#30353b", loc="left")
    ia.set_xlabel("simulated time", fontsize=10.5 * fs, color="#4a4f55")
    ia.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    line, = ia.plot([], [], color="#e0571f", lw=2)
    dot, = ia.plot([], [], "o", color="#e0571f", ms=6, mec="#f4f5f6", mew=1.5)
    # colorbar
    cax = fig.add_axes([0.70, 0.30, 0.26, 0.03])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=PowerNorm(0.55, 0, 100), cmap=CMAP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cax.tick_params(colors="#4a4f55", labelsize=10 * fs, length=0)
    cb.set_ticks([0, 5, 20, 50, 100])
    cax.set_title("temperature, % of final peak", fontsize=11.5 * fs, color="#30353b", loc="left", pad=6)
    fig.text(0.70, 0.18, "Walls held cold (0). Two chips inside produce heat;\nthe front corner is cut away through the hottest point.",
             fontsize=10.5 * fs, color="#4a4f55", linespacing=1.4)
    fig.add_artist(Rectangle((0, 0), 1, 0.085, transform=fig.transFigure, fc="#101317", alpha=0.85))
    hud = fig.text(0.03, 0.032, "", fontsize=13 * fs, color="#f2f3f5", family="DejaVu Sans Mono")

    n_hold = int(a.hold * fps)
    total = len(frames) + n_hold
    el = np.radians(26)

    def frame(k):
        i = min(k, len(frames) - 1)
        F = frames[i]
        az = np.radians(-62 + 30 * k / max(1, total - 1))
        polys, depths, cols = [], [], []
        for (axis, c, U, V, sh, look), Qf in zip(faces, Q):
            P2, d = project(Qf, az, el)
            polys.append(P2); depths.append(d.mean(1))
            rgba = CMAP(norm(look(F, U, V)))
            rgba[:, :3] *= sh
            cols.append(rgba)
        P2 = np.concatenate(polys); dep = np.concatenate(depths); C = np.concatenate(cols)
        o = np.argsort(-dep)
        pc.set_verts(P2[o]); pc.set_facecolors(C[o])
        E2, _ = project(edges3, az, el)
        ec.set_segments(E2)
        foot = np.array([[-0.5, -0.5, -0.5], [last + .5, -0.5, -0.5], [last + .5, last + .5, -0.5], [-0.5, last + .5, -0.5]])
        f2, _ = project(foot, az, el)
        shadow.set_verts([f2 + [0.06 * n, -0.05 * n]])
        line.set_data(times[:i + 1], np.array(tpeak[:i + 1]) / tmax)
        dot.set_data([times[i]], [tpeak[i] / tmax])
        hud.set_text(f"step {steps[i]:4d} / {steps[-1]}  |  time {times[i]:8.2f}  |  hottest point "
                     f"{100 * tpeak[i] / tmax:5.1f}% of final  |  CG iterations this step {iters[i]:3d}")
        return []

    anim = animation.FuncAnimation(fig, frame, frames=total, blit=False)
    if gif:
        if a.out.endswith(".mp4"):
            print("ffmpeg not usable here: writing a GIF instead")
        anim.save(out, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=8000))
    frame(total - 1)
    fig.savefig(out.rsplit(".", 1)[0] + ".png", dpi=dpi)
    print(f"wrote {out} and {out.rsplit('.', 1)[0]}.png  ({len(frames)} frames, grid {n}^3, cut at x={cx} y={cy} z={cz})")


if __name__ == "__main__":
    main()
