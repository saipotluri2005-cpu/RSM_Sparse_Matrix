# SPD test matrices (NIST Matrix Market)

38 real symmetric positive definite (SPD) matrices from https://math.nist.gov/MatrixMarket/,
found with the search tool's Symmetry = "Symmetric positive definite" filter. CG should
converge on all of these; the matrices in asymmetric_matrices/ are expected to fail CG.

Folders are split by number of rows N (unknowns):

| Folder | Rule | Count |
|---|---|---|
| small/ | N < 1,000 | 18 |
| medium/ | 1,000 <= N < 5,000 | 11 |
| large/ | N >= 5,000 | 9 |

Left out: the 13 bcsstm* matrices (diagonal, so CG finishes in one step) and
s3dkq4m2 / s3dkt3m2 (no convergence in 20,000 Jacobi-CG iterations; s3dkq4m2 is 120 MB).

Most need PRECOND=jacobi. With plain CG and MAX_ITERS=5000, 16 of the 38 do not converge.
With Jacobi, 34 converge within 5000 iterations; nos2, s3rmq4m1, s3rmt3m1 and s3rmt3m3
need MAX_ITERS=20000.
