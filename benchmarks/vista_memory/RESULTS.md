# Vista GH200 benchmark results — job 1134703

Measured on NVIDIA GH200 120GB, compute capability 9.0, CUDA runtime 13.3
and driver API version 13.1. Source measurements and regenerated plots are in
`results-1134703/`. All 90 rows pass correctness checks; no memory modes were skipped.
These are results from one job, without confidence intervals.

## CPU–GPU transfers

For 256 MiB copies, CUDA-event bandwidth was:

| Host allocation/API | CPU → GPU (GB/s) | GPU → CPU (GB/s) |
| --- | ---: | ---: |
| Pageable, synchronous | 407.0 | 293.6 |
| Pinned, synchronous | 412.7 | 295.5 |
| Pinned, asynchronous | 416.1 | 296.9 |

GB/s is decimal. Large transfers amortize API and synchronization overhead.
Pinned asynchronous copies were fastest here, but their large-buffer advantage
over pageable copies was modest. The asynchronous tests wait for completion;
they do not demonstrate overlap with computation. Including host/event overhead,
their 256 MiB bandwidth is about 411.7 GB/s H2D and 294.7 GB/s D2H.
This measures each direction separately, not simultaneous bidirectional bandwidth.

## HBM versus Grace CPU memory

The GPU memory kernel reads every FP64 input and writes an HBM output. For 256 MiB
of input, repeated kernel times were 0.1603 ms in explicit HBM, 0.1602 ms in
managed memory after GPU prefetch, and 0.6425 ms with mapped CPU-resident input.
Thus direct CPU-memory access worked, but took about 4.0× as long for this kernel.
Managed prefetch provides HBM-like steady-state execution; it still costs time
up front (1.905 ms host movement interval here, versus 0.810 ms for the HBM copy).
The host strategy's tiny movement interval is event overhead with no input copy.

The implication is that C2C makes direct access useful, but does not make CPU
memory equivalent to GPU-local HBM. Avoiding a copy trades an initial movement
cost for potentially more expensive GPU reads. Allocation/initialization is
reported separately and must also be considered for short-lived allocations.

## Sparse matrix-vector multiplication

Same cuSPARSE algorithm, FP64 values, and HBM vectors in every placement:

| Matrix | CSR bytes | HBM SpMV (ms) | Prefetched managed (ms) | CPU-resident CSR (ms) |
| --- | ---: | ---: | ---: | ---: |
| bfw62a | 5,652 | 0.00750 | 0.00743 | 0.01268 |
| memplus | 1,584,836 | 0.01053 | 0.01059 | 0.02291 |
| af23560 | 5,905,316 | 0.01171 | 0.01174 | 0.03249 |

These are warm repeated means. CPU-resident CSR was about 1.7–2.8× slower.
Numerical scaled errors were below 2.7e-16. The matrices are small enough that
cache reuse and launch overhead matter; these results do not establish sustained
C2C bandwidth for matrices larger than GPU cache or memory.

Including the initial CSR movement, host-resident CSR won at one SpMV and HBM
won at ten for all three matrices. The measured crossover is therefore between
1 and 10 operations, not a precisely identified count. The one-operation bfw62a
difference was only 0.00089 ms and needs repeated runs before drawing a strong
conclusion. For af23560, totals were 0.0704 ms HBM versus 0.0502 ms host at one
operation, 0.1739 versus 0.3422 ms at ten, and 11.65 versus 32.42 ms at 1000.
Managed had higher initial movement cost but nearly identical repeated execution.
Totals exclude allocation, parsing, descriptor/vector setup, and validation;
trials reuse warmed runtime/library state.

The very first HBM memory kernel (0.433 ms) and very first HBM SpMV (28.13 ms)
are consistent with initial kernel/library loading overhead. Strategy order is
fixed, so these first-call outliers are not fair measures of placement alone.

## CPU and GPU use

The script runs one process at a time on one GPU, with `OMP_NUM_THREADS=1`.
Parsing, initialization, and CPU-reference SpMV are serial in the application;
it does not intentionally use most Grace CPU cores. Runtime helper threads may
exist. Slurm node allocation is distinct from application core utilization.

GPU kernels execute in parallel: the 256 MiB memory test launches 131,072 blocks
of 256 threads, scheduled across the GPU; the 4 KiB test launches only two blocks.
cuSPARSE chooses its own GPU execution configuration. Large launches provide
enough work for broad GPU participation, but these short, memory-focused tests
do not establish full SM occupancy, peak FLOP use, or sustained GPU utilization.
Copies exercise memory/interconnect resources and need not show high SM activity.
No utilization profiler was recorded, so an exact percentage cannot be inferred.

For a reused sparse matrix, these results favor copying CSR to HBM once (or
prefetching managed CSR once), keeping vectors on the GPU, and avoiding repeated
CPU–GPU movement. This is evidence for a memory-placement choice, not a measured
full FEM solver speedup.

Reproduce plots from the repository root:

```bash
benchmarks/vista_memory/.venv/bin/python benchmarks/vista_memory/plot.py \
  benchmarks/vista_memory/results-1134703
```
