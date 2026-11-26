#!/usr/bin/env python3
"""Generate a 3D heat-conduction problem on the same 7-point grid as the poisson3d benchmarks.

A cube of metal, every wall held at 0 (cooled), with heat-producing "chips" inside.
Same matrix as gen_poisson.py --3d n (6 on the diagonal, -1 to each grid neighbour,
node (x, y, z) = row x + n*y + n*n*z); only the right-hand side changes.

Writes
  <out>.mtx         K (steady state:  K T = q)
  <out>_b.mtx       q, the heat sources (gpu_cg RHS=file)
  <out>_mesh.txt    grid size, sources and a time step for the transient run

Transient heating (gpu_cg HEAT_STEPS=..., HEAT_DT=dt) solves, every step,
  (K + I/dt) T_new = T_old/dt + q
which is implicit Euler for dT/dt = -K T + q, starting from T = 0.

  python3 gen_heat.py --n 64 --out viz_cases/heat64
Generated with assistance of claude code
"""
import argparse
import math
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=64, help="grid points per side (N = n^3 unknowns)")
    ap.add_argument("--steps", type=int, default=120, help="time steps that reach about 95%% of steady state")
    ap.add_argument("--out", default="viz_cases/heat64")
    a = ap.parse_args()
    n = a.n
    N = n ** 3
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # two chips near the top of the block, one larger and hotter (fractions of the side)
    chips = [((0.18, 0.40), (0.30, 0.70), (0.66, 0.80), 1.0),
             ((0.58, 0.82), (0.38, 0.62), (0.70, 0.78), 0.6)]

    nnz_lower = N + 3 * n * n * (n - 1)
    with open(a.out + ".mtx", "w") as f:
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
        f.write(f"% 3D heat conduction, 7-point stencil, {n}^3 grid, from gen_heat.py\n")
        f.write(f"{N} {N} {nnz_lower}\n")
        out = []
        for z in range(n):
            for y in range(n):
                for x in range(n):
                    r = x + n * (y + n * z) + 1
                    out.append(f"{r} {r} 6\n")
                    if x > 0: out.append(f"{r} {r - 1} -1\n")
                    if y > 0: out.append(f"{r} {r - n} -1\n")
                    if z > 0: out.append(f"{r} {r - n * n} -1\n")
            f.write("".join(out))
            out.clear()

    q = [0.0] * N
    for (x0, x1), (y0, y1), (z0, z1), power in chips:
        for z in range(int(z0 * n), int(z1 * n)):
            for y in range(int(y0 * n), int(y1 * n)):
                for x in range(int(x0 * n), int(x1 * n)):
                    q[x + n * (y + n * z)] = power
    with open(a.out + "_b.mtx", "w") as f:
        f.write("%%MatrixMarket matrix array real general\n")
        f.write(f"{N} 1\n")
        f.writelines("0\n" if v == 0 else f"{v}\n" for v in q)

    # slowest decay time of the cube is (n+1)^2 / (3 pi^2); three of them is ~95% of steady
    dt = 3 * (n + 1) ** 2 / (3 * math.pi ** 2) / a.steps
    with open(a.out + "_mesh.txt", "w") as f:
        f.write(f"# heat problem from gen_heat.py\n")
        f.write(f"grid {n}\n")
        f.write(f"dt {dt:.6g}\n")
        f.write(f"steps {a.steps}\n")
        for (x0, x1), (y0, y1), (z0, z1), power in chips:
            f.write(f"chip {x0} {x1} {y0} {y1} {z0} {z1} {power}\n")
    print(f"wrote {a.out}.mtx ({N} unknowns, {2 * nnz_lower - N} nonzeros), {a.out}_b.mtx, "
          f"{a.out}_mesh.txt (dt = {dt:.4g}, {a.steps} steps)")


if __name__ == "__main__":
    main()
