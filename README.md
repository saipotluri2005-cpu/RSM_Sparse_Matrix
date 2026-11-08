# RSM_Sparse_Matrix

Conjugate Gradient (CG) solvers for sparse FEM matrices from the
[SuiteSparse collection](https://sparse.tamu.edu/), multithreaded with OpenMP
and benchmarked on TACC Vista.

| Program | Matrix storage | Work per CG iteration |
|---|---|---|
| `cpu_only` | dense N×N array (baseline) | O(N²) |
| `cpu_sparse` | CSR + RCM reordering | O(nnz) |

Both programs read `.mtx` / `.mtx.gz` files, solve `A x = b` with `b = 1`,
and print the same report (matrix size, iterations, solve time, relative residual).

## Pipeline (`cpu_sparse`, see `cpu_high_level.txt`)

1. Read the Matrix Market file into a COO list (row, col, value)
2. Report the sparsity ratio, nnz / N²
3. Reverse Cuthill-McKee reordering: moves nonzeros toward the diagonal
4. Convert COO → CSR, storing only the nonzeros
5. CG with a CSR sparse matrix-vector product: three fused passes per
   iteration, all inside one OpenMP parallel region, NUMA first-touch
6. *(planned)* cuSPARSE SpMV on the Vista GPU

`cpu_sparse` options (environment variables):

| Variable | Default | Effect |
|---|---|---|
| `REORDER` | `rcm` | `none` skips RCM |
| `PRECOND` | `none` | `jacobi` uses Jacobi-preconditioned CG |
| `MAX_ITERS` / `TOL` | `5000` / `1e-5` | stopping criteria |
| `SAVE_SOLUTION` | `1` | `0` skips writing `<matrix>_csr_solution.txt` |

## Build and run

```bash
make                                    # builds cpu_only and cpu_sparse
OMP_NUM_THREADS=8 ./cpu_sparse path/to/matrix.mtx.gz
OMP_NUM_THREADS=8 ./cpu_sparse some_folder      # every .mtx / .mtx.gz inside
```

Run on a compute node (`idev`, or `sbatch`), not a login node.

## Test matrices

Test matrices live on the **`test_matrices`** branch. The scripts check that
branch out into a separate folder, `../RSM_test_matrices` (a git worktree), so
your working branch never changes. To use another folder: `TEST_DIR=path`.

`gen_poisson.py` makes SPD Poisson matrices of any size:
`python3 gen_poisson.py 30 60 90` or `python3 gen_poisson.py --3d 16 32`.

CG converges only on **symmetric positive definite** matrices. Others still
run, but are reported as `NO_CONVERGE`.

## Benchmark: dense vs sparse

```bash
sbatch benchmark.sh          # or: make bench   (light defaults on a login node)
```

Output in `benchall_<jobid>/`:

- `raw.csv` — every run (`impl, matrix, N, nnz, threads, rep, iterations, solve_ms, rel_residual`)
- `summary.txt` — per-matrix scaling tables and the dense/sparse speedup
- `plots/` — solve time vs N, time per iteration vs N, sparse speedup per matrix,
  memory (dense vs CSR), thread scaling

Settings: `THREADS="1 2 4 8"`, `REPS=3`, `POISSON_SIZES="30 60"`, `MAX_DENSE_GB=100`,
`IMPLS="cpu_only cpu_sparse"`, `PERF=1`. Re-plot an old run with
`python3 plot_compare.py benchall_<id>/raw.csv`.

## Profiling

```bash
sbatch cg_perf.slurm                    # cpu_sparse
EXE=./cpu_only sbatch cg_perf.slurm     # dense version
make tsan MATRIX=small.mtx              # ThreadSanitizer (PROG=cpu_only for dense)
make perf MATRIX=big.mtx.gz             # quick perf stat
```

`cg_perf.slurm` runs a thread sweep, then perf stat (1 vs N threads, stalls,
caches), per-thread counts, spin vs passive waiting, `perf record`, and a
NUMA check on the largest converged matrix. Output: `perfrun_<program>_<jobid>/`.

## Files

| File | Purpose |
|---|---|
| `cpu_only.cpp` | dense CG baseline |
| `cpu_sparse.cpp` | CSR + RCM CG |
| `benchmark.sh` | dense vs sparse benchmark (Slurm or interactive) |
| `plot_compare.py` | graphs from `raw.csv` (needs matplotlib) |
| `cg_perf.slurm` | benchmark + staged perf analysis for one program |
| `gen_poisson.py` | Poisson test-matrix generator |
| `Makefile` | build, bench, tsan, perf, clean |
| `cpu_high_level.txt` | pipeline plan |
| `resources.md` | where to get matrices |
