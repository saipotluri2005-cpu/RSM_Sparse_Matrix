#!/usr/bin/env python3
"""Generate SPD Poisson test matrices (Matrix Market, lower triangle, symmetric).

usage:
  python3 gen_poisson.py 30            -> poisson2d_30.mtx   (30x30 grid, N = 900)
  python3 gen_poisson.py 30 60 90      -> one file per size
  python3 gen_poisson.py --3d 16 32    -> poisson3d_16.mtx, poisson3d_32.mtx (N = n^3)
"""
import sys


def poisson2d(n):
    e = []
    for i in range(n):
        for j in range(n):
            k = i * n + j + 1
            e.append((k, k, 4.0))
            if j > 0: e.append((k, k - 1, -1.0))
            if i > 0: e.append((k, k - n, -1.0))
    return n * n, e


def poisson3d(n):
    e = []
    for i in range(n):
        for j in range(n):
            for l in range(n):
                k = (i * n + j) * n + l + 1
                e.append((k, k, 6.0))
                if l > 0: e.append((k, k - 1, -1.0))
                if j > 0: e.append((k, k - n, -1.0))
                if i > 0: e.append((k, k - n * n, -1.0))
    return n ** 3, e


if __name__ == "__main__":
    args = sys.argv[1:]
    three_d = "--3d" in args
    sizes = [int(a) for a in args if a != "--3d"]
    if not sizes:
        sys.exit(__doc__)
    for n in sizes:
        N, ent = poisson3d(n) if three_d else poisson2d(n)
        name = f"poisson{'3d' if three_d else '2d'}_{n}.mtx"
        with open(name, "w") as f:
            f.write("%%MatrixMarket matrix coordinate real symmetric\n")
            f.write(f"{N} {N} {len(ent)}\n")
            f.writelines(f"{r} {c} {v}\n" for r, c, v in ent)
        print(f"wrote {name}  N = {N}")
