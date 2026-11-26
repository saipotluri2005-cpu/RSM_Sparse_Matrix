#!/usr/bin/env python3
"""Generate a 2D steel truss bridge as a sparse FEM problem K u = b.

Writes three files (same Matrix Market conventions as gen_poisson.py):
  <name>.mtx        stiffness matrix K (symmetric, lower triangle)   -> gpu_cg / cpu_sparse_v2
  <name>_b.mtx      load vector b (array format)                     -> gpu_cg uses it as the RHS
  <name>_mesh.txt   joints, bars and unknown numbering               -> render_bridge.py

The bridge is a Pratt truss: bottom chord (the deck), top chord, verticals and
diagonals. Left end pinned, right end on a roller, so the structure is stable and
K is symmetric positive definite (CG works). Units: metres, newtons, pascals.

  python3 gen_bridge.py                       # 12 panels, 60 m span
  python3 gen_bridge.py --panels 40 --truck-pos 0.33 --out viz_cases/bridge40
  python3 gen_bridge.py --moving              # truck drives across: one GPU solve per joint
Generated with assistance of claude code
"""
import argparse
import math
import os


def build(panels, panel_len, height, E, A_chord, A_web, deck_load, truck, truck_pos):
    # joints: bottom 0..P, then top 1..P-1
    joints = [(i * panel_len, 0.0) for i in range(panels + 1)]
    top = {i: len(joints) + i - 1 for i in range(1, panels)}
    joints += [(i * panel_len, height) for i in range(1, panels)]

    bars = []  # (a, b, area, kind)
    for i in range(panels):
        bars.append((i, i + 1, A_chord, "bottom chord"))
    for i in range(1, panels - 1):
        bars.append((top[i], top[i + 1], A_chord, "top chord"))
    for i in range(1, panels):
        bars.append((i, top[i], A_web, "vertical"))
    bars.append((0, top[1], A_chord, "end post"))
    bars.append((panels, top[panels - 1], A_chord, "end post"))
    mid = panels / 2
    for i in range(1, panels - 1):  # one diagonal per inner panel, leaning toward mid-span
        if i + 0.5 < mid:
            bars.append((top[i], i + 1, A_web, "diagonal"))
        else:
            bars.append((top[i + 1], i, A_web, "diagonal"))

    # unknowns: (ux, uy) per joint, minus supports (pin at 0, roller at P)
    fixed = {(0, 0), (0, 1), (panels, 1)}
    dof = {}
    for j in range(len(joints)):
        for d in range(2):
            if (j, d) not in fixed:
                dof[(j, d)] = len(dof)
    n = len(dof)

    K = {}
    for a, b, area, _ in bars:
        (xa, ya), (xb, yb) = joints[a], joints[b]
        L = math.hypot(xb - xa, yb - ya)
        c, s = (xb - xa) / L, (yb - ya) / L
        k = E * area / L
        g = [(a, 0, -c), (a, 1, -s), (b, 0, c), (b, 1, s)]  # bar strain = g . u / L
        for (j1, d1, v1) in g:
            r = dof.get((j1, d1))
            if r is None:
                continue
            for (j2, d2, v2) in g:
                q = dof.get((j2, d2))
                if q is None or q > r:  # lower triangle only
                    continue
                K[(r, q)] = K.get((r, q), 0.0) + k * v1 * v2

    rhs = [0.0] * n
    for i in range(1, panels):  # deck load shared by the bottom joints
        rhs[dof[(i, 1)]] -= deck_load * panel_len
    if truck > 0:
        j = max(1, min(panels - 1, round(truck_pos * panels)))
        rhs[dof[(j, 1)]] -= truck
    return joints, bars, dof, K, rhs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", type=int, default=12)
    ap.add_argument("--panel-len", type=float, default=5.0)
    ap.add_argument("--height", type=float, default=0.0, help="truss height in m (default: span / 10, at least 6)")
    ap.add_argument("--deck-load", type=float, default=20e3, help="N per metre of deck")
    ap.add_argument("--truck", type=float, default=400e3, help="point load in N (0 = none)")
    ap.add_argument("--truck-pos", type=float, default=0.33, help="fraction of span")
    ap.add_argument("--moving", action="store_true",
                    help="truck drives across: one load case per deck joint (<out>_tNN.mtx + _b.mtx)")
    ap.add_argument("--out", default="viz_cases/bridge")
    a = ap.parse_args()
    if a.panels < 2:
        ap.error("--panels must be at least 2")

    if a.height <= 0:
        a.height = max(6.0, a.panels * a.panel_len / 10)
    grow = max(1.0, (a.panels * a.panel_len / 60.0) ** 1.5)   # longer spans get heavier members
    joints, bars, dof, K, rhs = build(a.panels, a.panel_len, a.height, 200e9, 0.012 * grow, 0.006 * grow,
                                      a.deck_load, a.truck, a.truck_pos)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n = len(dof)

    def write_case(path, b):
        with open(path + ".mtx", "w") as f:
            f.write("%%MatrixMarket matrix coordinate real symmetric\n")
            f.write("% 2D Pratt truss bridge stiffness matrix, from gen_bridge.py\n")
            f.write(f"{n} {n} {len(K)}\n")
            for (r, q), v in sorted(K.items()):
                f.write(f"{r + 1} {q + 1} {v:.17g}\n")
        with open(path + "_b.mtx", "w") as f:
            f.write("%%MatrixMarket matrix array real general\n")
            f.write(f"{n} 1\n")
            f.writelines(f"{v:.17g}\n" for v in b)

    cases = []
    if a.moving:
        # one load case per deck joint: the truck standing on joint p (p = 0 and p = panels
        # are the supports, where the truck's weight goes straight into the bearing)
        dead = list(rhs)
        if a.truck > 0:
            j = max(1, min(a.panels - 1, round(a.truck_pos * a.panels)))
            dead[dof[(j, 1)]] += a.truck          # take the fixed truck back out
        for p in range(a.panels + 1):
            b = list(dead)
            if (p, 1) in dof:
                b[dof[(p, 1)]] -= a.truck
            write_case(f"{a.out}_t{p:02d}", b)
            cases.append(f"{os.path.basename(a.out)}_t{p:02d}")
    else:
        write_case(a.out, rhs)
        cases.append(os.path.basename(a.out))

    with open(a.out + "_mesh.txt", "w") as f:
        f.write(f"# bridge mesh from gen_bridge.py  panels={a.panels} span={a.panels * a.panel_len} m\n")
        f.write(f"joints {len(joints)}\n")
        for j, (x, y) in enumerate(joints):
            ux, uy = dof.get((j, 0), -1), dof.get((j, 1), -1)  # unknown index, -1 = support
            f.write(f"{x} {y} {ux} {uy}\n")
        f.write(f"bars {len(bars)}\n")
        for b_ in bars:
            f.write(f"{b_[0]} {b_[1]} {b_[2]} {b_[3].replace(' ', '_')}\n")
        f.write("E 200e9\n")
        f.write(f"truck {a.truck} {'moving' if a.moving else a.truck_pos}\n")
        f.write("cases " + " ".join(cases) + "\n")
    print(f"wrote {len(cases)} load case(s) {cases[0]}..{cases[-1]} ({n} unknowns, {2 * len(K) - n} nonzeros each)"
          f" and {a.out}_mesh.txt")


if __name__ == "__main__":
    main()
