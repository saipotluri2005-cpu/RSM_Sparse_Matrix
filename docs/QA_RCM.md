# Branch `qa_rcm`: tests, fixes, matrix generator, RCM study, Vista scripts

Branched from `CPU_multithreaded`. Default behaviour of the three solvers is unchanged
except where a fix says otherwise; under normal settings `cpu_sparse_v2` gives identical
iterations, residuals and bandwidths to the original (96 of 96 runs compared: 6 matrices,
1/4/8/16 threads, plain and Jacobi, FIRST_TOUCH 0/1).

<!-- Generated with assistance of claude code -->

## Fixes to the solvers

| Change | Files | Why |
|---|---|---|
| The row partition is sized from the threads OpenMP actually creates; dynamic thread adjustment is turned off; each partitioned region checks its team size | `cpu_sparse_v2.cpp` | The partition assumed `omp_get_max_threads()` threads. With `OMP_THREAD_LIMIT` below `OMP_NUM_THREADS`, or `OMP_DYNAMIC=true` on a loaded machine, fewer threads ran and their rows were skipped: a segmentation fault with `FIRST_TOUCH=1`, and with `FIRST_TOUCH=0` a wrong answer reported as converged (poisson2d_60: 38 or 135 iterations, residual 1.24 / 1.92). No team script sets either variable, so existing results are not affected. |
| `Threads:` prints the number of threads actually created | all three | It printed the requested number even when fewer ran. |
| Solution files are written with 17 significant digits | all three | With the default 6 digits, the residual recomputed from the saved file was 2.0e-3 (bcsstk14), 2.3e-2 (bcsstk18), 0.85 (1138_bus) and 1.0e-3 (poisson2d_60) although each solve reached 1e-5; with 17 digits it matches the solver. Needed to check GPU answers against CPU answers. |
| Repeated entries in a file are summed | `cpu_only.cpp` | `cpu_only` kept only the last value while `cpu_sparse` sums them, so the two solved different systems for such files. None of the current test matrices has repeated entries. |
| `MAX_ITERS` and `TOL` environment variables | `cpu_only.cpp` | Same options as the sparse versions; needed for the known-answer tests. |
| `REORDER=auto` (opt-in) | `cpu_sparse.cpp`, `cpu_sparse_v2.cpp` | Keeps the input order when RCM would not shrink the bandwidth. Default stays `REORDER=rcm`. |

## New files

| File | What it does |
|---|---|
| `tools/gen_fem.cpp` | SPD matrices of any size: `poisson1d N`, `poisson2d n`, `poisson3d n`, `elasticity nx [ny nz]` (8-node hexahedra, E = 1, nu = 0.3, x = 0 face clamped, 3 unknowns per node). `--shuffle SEED` renumbers nodes randomly; `--check` verifies the element stiffness (symmetric, rigid-body motions give zero force). `elasticity 70` has 1,058,610 unknowns. |
| `tools/rcm_study.cpp` | Per matrix and ordering (original, shuffled, rcm, cm, rcm_mindeg, rcm_first, shuffled_rcm): bandwidth, profile, mean distance from the diagonal, time to compute the ordering, SpMV time; checks every ordering is a permutation and gives the same A x; writes spy grids (`--spy`) and reordered matrices (`--write`). Its `rcm` reproduces `cpu_sparse`'s permutation. |
| `tools/check_solution.cpp` | Recomputes ‖b − Ax‖/‖b‖ from a matrix file and a saved solution file; `--ref` compares against another solution. |
| `tools/omp_barrier_cost.cpp` | Cost of one `omp barrier` and one small `omp for reduction` at the current thread count and binding. |
| `tools/gpu_info.cu` | GPU memory size, bus width and L2 cache size. |
| `tools/plot_rcm.py` | Plots and `summary.md` from the Vista result folders. |
| `tools/sparse_io.hpp`, `tools/rcm.hpp` | Shared Matrix Market I/O, CSR building, ordering metrics, RCM variants. |
| `qa/run_tests.sh` (`make test`) | 59 checks, about 2 s: exact answers (1D Poisson closed form, diagonal, 1x1, repeated entries, pattern file) for all three solvers; the three solvers agree; same iterations at 1/2/4/8 threads; v2 bit-identical when rerun; thread limit below the requested count; every v2 option combination; RCM keeps the answer; `REORDER=auto`; `rcm_study` matches `cpu_sparse`'s RCM; Jacobi on NIST and elasticity matrices with the residual rechecked from the file; complex, non-square, gzip, unsymmetric and `MAX_ITERS` inputs. |
| `qa/race_check.sh` (`make race-check`) | 44 runs of the three solvers built with clang ThreadSanitizer and LLVM OpenMP's Archer tool (2 and 4 threads, plain and Jacobi, FIRST_TOUCH 0/1, thread limit, BALANCE/THREAD_GRAIN): no data races reported. A deliberately racy test program is reported by the same setup. Needs `clang-18` and `libomp-18-dev`. |
| `qa/data/` | Small test matrices: bcsstk01, nos4, 494_bus (NIST SPD), young1c (complex). |
| `.github/workflows/tests.yml` | Runs `make test` and `make race-check` on every push. |
| `scripts/vista_cpu.slurm` | One gg-node job: environment (including any `OMP_*` TACC sets), `make test`, 1M-unknown matrices (2D/3D Poisson, elasticity; natural and shuffled), time per iteration with/without RCM at 1/72/144 threads, full solves (1 thread without reordering = success-criterion baseline; 1 thread with RCM; 144 threads with RCM) with read/build/solve times, `rcm_study` on the 38 NIST SPD matrices and the generated ones, STREAM triad (if `tools/stream.c` was downloaded), barrier cost. |
| `scripts/vista_gpu.slurm` | One gh-node job: GPU properties; 3D Poisson with x = 8, 33 and 97 MB and the 1M elasticity matrix, each as original, shuffled and shuffled + RCM; GPU SpMV time with `benchmarks/vista_memory/spmv` from `test_matrices`, and Grace CPU SpMV time with `rcm_study`. |

Also changed: `Makefile` (`-march=native` on x86, `tools/`, `test`, `race-check`),
`benchmark.sh` (`raw.csv` gains threads_used, read_ms, build_ms, bw_before, bw_after,
reorder, precond, total_ms; the summary's not-converged test uses `MAX_ITERS`), `.gitignore`.

## Running

```bash
make                 # solvers + tools (Vista or x86 Linux)
make test            # 59 correctness checks
make race-check      # needs clang-18 + libomp-18-dev (runs automatically on GitHub)

tools/gen_fem elasticity 70 --shuffle 1 -o el70_shuf.mtx
OMP_NUM_THREADS=1 tools/rcm_study --spy 150 --outdir spy --csv study.csv el70_shuf.mtx
PRECOND=jacobi ./cpu_sparse_v2 el70_shuf.mtx
tools/check_solution el70_shuf.mtx el70_shuf_csr_v2_solution.txt

# Vista (from the repo root, branch qa_rcm)
git fetch origin
curl -fsSLo tools/stream.c https://www.cs.virginia.edu/stream/FTP/Code/stream.c
sbatch scripts/vista_cpu.slurm
sbatch scripts/vista_gpu.slurm
# results: ~/qa_rcm_results_{cpu,gpu}_<jobid>.tar.gz
python3 tools/plot_rcm.py qa_rcm_cpu_<jobid> qa_rcm_gpu_<jobid>
```

## Observations so far (x86 laptop; to be repeated on Vista)

- The generated grids are already well numbered: RCM leaves the bandwidth of
  `poisson2d 1000` at 1,000 and raises it on the structured elasticity mesh
  (`elasticity 8`: 245 → 599). On `poisson2d 1000` with random numbering, RCM brings
  the bandwidth from 999,146 to 1,000 and halves the time per iteration on one thread.
- On the NIST matrices, iteration counts change slightly with thread count
  (bcsstk18: 1,534–1,552) because floating-point additions happen in a different order.
  `cpu_sparse_v2` is bit-reproducible at a fixed thread count; `cpu_sparse` is not.
- Plain CG does not converge within 5,000 iterations on 16 of the 38 NIST SPD matrices;
  with `PRECOND=jacobi`, 34 converge within 5,000.
- Elasticity iteration counts with Jacobi grow about linearly with mesh size
  (83, 164, 247 for n = 10, 20, 30), so `elasticity 70` needs roughly 600.
