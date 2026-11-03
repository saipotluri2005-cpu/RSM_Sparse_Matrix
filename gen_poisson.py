import sys
n = int(sys.argv[1]); N = n * n
ent = []
for i in range(n):
    for j in range(n):
        k = i * n + j
        ent.append((k + 1, k + 1, 4.0))
        if j > 0: ent.append((k + 1, k, -1.0))
        if i > 0: ent.append((k + 1, k - n + 1, -1.0))
with open(f"poisson2d_{n}.mtx", "w") as f:
    f.write("%%MatrixMarket matrix coordinate real symmetric\n")
    f.write(f"{N} {N} {len(ent)}\n")
    for r, c, v in ent:
        f.write(f"{r} {c} {v}\n")
