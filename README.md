# RSM_Sparse_Matrix

Conjugate Gradient (CG) solvers for sparse FEM matrices from the
[SuiteSparse collection](https://sparse.tamu.edu/), multithreaded with OpenMP
and benchmarked on TACC Vista.

| Program | Matrix storage | Work per CG iteration |
|---|---|---|
| `cpu_only` | dense N×N array (baseline) | O(N²) |
| `cpu_sparse` | CSR + RCM reordering | O(nnz) |
| `cpu_sparse_v2` | `cpu_sparse` + scaling fixes for many cores | O(nnz) |
| `gpu_cg` | CSR on the H200 GPU, prepared by the Grace CPU (GH200 node) | O(nnz) |

Results, graphs and the performance analysis: [`docs/RESULTS.md`](docs/RESULTS.md).

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
`IMPLS="cpu_only cpu_sparse"` (add `cpu_sparse_v2` to include it), `PERF=1`. Re-plot an old run with
`python3 plot_compare.py benchall_<id>/raw.csv`.

## Scaling fixes (`cpu_sparse_v2`) and the ablation study

`cpu_sparse_v2` keeps `cpu_sparse`'s algorithm and output and changes how threads work
together (details in `docs/RESULTS.md` §3):

| Fix | What changes | Switch |
|---|---|---|
| Synchronization | per-thread partial sums on their own 64-byte cache lines, 3 barriers per iteration instead of 3 reductions + `single` + barrier | always on |
| NUMA first-touch | matrix and all vectors written first by the thread that uses them | `FIRST_TOUCH=0/1` (default 1) |
| Load balance | rows split so each thread gets equal nonzeros | `BALANCE=nnz/rows` (default nnz) |
| Small matrices | at most one thread per N nonzeros | `THREAD_GRAIN=N` (default 0 = off) |
| Placement | both sockets from the first thread | `OMP_PROC_BIND=spread` at launch |

`cpu_sparse_v2` also prints an estimated `Bandwidth Achieved` in GB/s.

```bash
make
sbatch ablation.sh                      # Poisson 120, 500, 1000, 2000; 1-144 threads
```

`ablation.sh` adds one fix at a time (`cpu_sparse` close → `cpu_sparse` spread → sync → first-touch →
balance → thread grain), so the difference between neighbouring configurations is
the effect of that one change. Output in `ablation_<jobid>/`: `raw.csv` and
`plots/` (time per iteration and speedup vs threads, best time per configuration,
achieved bandwidth). Settings: `SIZES`, `MATRICES`, `THREADS`, `REPS`, `CONFIGS`.
Re-plot with `python3 plot_ablation.py ablation_<id>/raw.csv`.

## GPU solver (`gpu_cg`, GH200 nodes)

On a Vista `gh` node the 72-core Grace CPU and the H200 GPU are joined by
NVLink-C2C (~450 GB/s each way, cache coherent). CG is memory-bound and runs
hundreds of iterations on the same matrix, so the split is:

| Step | Where | Why |
|---|---|---|
| read `.mtx`, COO, RCM, CSR | CPU (OpenMP) | branchy, one pass; reuses the CPU code (`sparse_prep.hpp`) |
| copy CSR to GPU memory | NVLink-C2C, **once** | ~400 GB/s, a few ms even for 100M nonzeros |
| every CG iteration | GPU (HBM3, ~4 TB/s) | the matrix is read every iteration: ~8× the CPU's memory bandwidth |
| alpha, beta, convergence test | GPU (one thread) | the CPU never waits on a dot product |
| "done yet?" | 48 bytes to the CPU every `CHECK_EVERY` iterations | the GPU runs one chunk ahead, so it never idles |
| solution x | NVLink-C2C, once, into pinned memory | then the CPU checks the residual |
| next matrix | CPU prepares it while the GPU solves this one | `PIPELINE=1` |

The SpMV is not split between CPU and GPU: the CPU would add ~12% more
bandwidth but force a CPU–GPU sync every iteration.

```bash
make gpu-test            # checks the GPU logic on the CPU (no GPU needed)
make gpu                 # builds ./gpu_cg; finds nvcc, gcc 14 and cuSPARSE on Vista itself
make gpu-info            # shows which nvcc / g++ / cuSPARSE it picked
make gpu-quick           # sbatch: one 64^3 matrix, 1 rep (a few minutes)
make gpu-bench           # sbatch: all modes below + cpu_sparse_v2 on 72 cores, median table
make gpu-run MATRIX=Matrices   # run directly, on a gh node (idev -p gh ...)
```

On Vista, cuSPARSE is in the HPC SDK's `math_libs/13.3/` folder, not inside
`cuda/13.3/`; the Makefile looks there and bakes the path into `gpu_cg`, so it
runs without modules or `LD_LIBRARY_PATH`. Elsewhere: `make gpu CUSPARSE=/path/libcusparse.so`.

| Option | Values (default first) |
|---|---|
| `GPU_MATRIX` | `hbm` copy once · `managed` unified memory + prefetch · `host` GPU reads CPU memory over C2C every iteration |
| `SPMV` | `cusparse` · `fused` (our kernel: Ap and p·Ap in one pass) |
| `CUDA_GRAPH` | `1` one graph launch per chunk · `0` plain launches |
| `CHECK_EVERY` | `16` iterations between "done?" checks |
| `STAGE` | `pageable` · `pinned` (cudaHostRegister the CSR before the copy) |
| `PIPELINE` | `1` prepare the next matrix during this solve · `0` |
| `VERIFY` | `1` recompute the residual on the CPU from the copied-back x |
| `RHS` | `ones` (b = 1, as on the CPU) · `file` (b from `<name>_b.mtx`) |
| `SOLUTION_DIR` | where `SAVE_SOLUTION=1` writes `<name>_gpu_solution.txt` (default `.`) |
| `PRECOND`, `REORDER`, `MAX_ITERS`, `TOL`, `SAVE_SOLUTION` | as in the CPU programs |

The iteration count matches `cpu_sparse_v2` (same stopping test, same MAX_ITERS
rule; `test_gpu_logic` checks this). x can differ in the last digits because
GPU sums are added in a different order.

## FEM visualization (`viz/`, branch `FEM_visualization`)

Generate a structure, solve it with `gpu_cg`, and turn the answer into a video.

```bash
sbatch viz.slurm          # or: make viz   -> viz_out/bridge.mp4, viz_out/bridge_long.mp4
```

| Step | File | What it does |
|---|---|---|
| 1. Generate | `viz/gen_bridge.py` | 2D steel Pratt truss: writes `<name>.mtx` (stiffness K), `<name>_b.mtx` (loads b), `<name>_mesh.txt` (joints, bars) |
| 2. Solve | `gpu_cg` with `RHS=file SAVE_SOLUTION=1` | reads b from `<name>_b.mtx` (reordered with the matrix by RCM), writes u in the original numbering |
| 3. Render | `viz/render_bridge.py` | checks `‖Ku − b‖/‖b‖` and compares with a direct solve, then animates the load ramp: the bridge sags (exaggerated), bars colored compression → tension |

The structure is linear, so the displacement at x% load is x% of u: one GPU solve
makes the whole animation. `render_bridge.py` needs numpy, scipy and matplotlib
(ffmpeg for MP4, otherwise it writes a GIF); if the node lacks them, copy
`viz_cases/` and `viz_out/` to a laptop and run it there.

`RHS=file` also works for SuiteSparse matrices that ship a `<name>_b.mtx`. Folder
scans skip `*_b.mtx` and `*_coord.mtx` files, which are not matrices.

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
| `cpu_sparse_v2.cpp` | `cpu_sparse` + switchable scaling fixes |
| `benchmark.sh` | dense vs sparse benchmark (Slurm or interactive) |
| `plot_compare.py` | graphs from `raw.csv` (needs matplotlib) |
| `ablation.sh` | measures each scaling fix on its own (Slurm) |
| `plot_ablation.py` | graphs from the ablation `raw.csv` |
| `docs/RESULTS.md` | results, graphs, performance analysis |
| `cg_perf.slurm` | benchmark + staged perf analysis for one program |
| `gen_poisson.py` | Poisson test-matrix generator |
| `gpu_cg.cu` | GPU CG (CUDA + cuSPARSE), CPU prepares, GPU solves |
| `sparse_prep.hpp` | read/RCM/CSR code shared by the GPU solver and its test |
| `gpu_cg_logic.hpp` | per-iteration scalar logic run on the GPU (and in the test) |
| `test_gpu_logic.cpp` | CPU replay of the GPU kernels vs CPU CG |
| `gpu_bench.slurm` | GPU modes vs CPU benchmark on a gh node |
| `viz/gen_bridge.py` | truss bridge generator (matrix + loads + mesh) |
| `viz/render_bridge.py` | bridge animation from a solved displacement vector |
| `viz.slurm` | generate → GPU solve → render |
| `Makefile` | build, bench, tsan, perf, clean |
| `cpu_high_level.txt` | pipeline plan |
| `resources.md` | where to get matrices |
