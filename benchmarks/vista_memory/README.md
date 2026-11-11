# Vista memory and cuSPARSE benchmarks

For measured job 1134703 results and CPU/GPU utilization interpretation, see
[RESULTS.md](RESULTS.md).

Standalone C++17/CUDA experiments; existing repository code and matrices are untouched.
Use a single GPU on one Grace–Hopper node. Results are measurements, not theoretical
NVLink bandwidth estimates. No benchmark results are included: local GPU initialization
fails, so CSVs must be generated on a working CUDA node.

## Build

On Vista, inspect the currently available modules and their compatibility first:

```bash
module avail
module spider cuda
module spider gcc
module spider cmake
# Load the GCC, CUDA, and CMake modules actually available at your site.
module load <compatible-gcc-module> <cuda-module> <cmake-module>
which g++ nvcc cmake
uname -m
g++ --version
nvcc --version
nvidia-smi
```

Use the CUDA Toolkit's supported host compiler table; GCC 13/14 compatibility depends
on the toolkit, not the GPU. Do not use `--allow-unsupported-compiler`.
CMake uses the active compilers, without fixed paths or toolkit versions. If needed,
set `CXX`, `CUDACXX`, and `CUDAHOSTCXX` to the compatible tools before a fresh configure.
Dependencies are CUDA/cuSPARSE, zlib development headers/library, and CMake >=3.18.
Build natively on Vista's aarch64 host; no x86 cross-compilation is configured.

From the repository root:

```bash
cmake -S benchmarks/vista_memory -B benchmarks/vista_memory/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build benchmarks/vista_memory/build -j
ctest --test-dir benchmarks/vista_memory/build --output-on-failure
```

Architecture 90 targets Hopper. Override for another GPU. The CPU reader tests require
no GPU; they cover symmetric expansion, duplicates, pattern entries, bad indices,
truncation, and unsupported formats.

## Run

Run these commands inside an allocated GH compute node, from the repository root:

```bash
mkdir -p benchmarks/vista_memory/results
benchmarks/vista_memory/build/transfer benchmarks/vista_memory/results/transfer.csv 30
benchmarks/vista_memory/build/memory benchmarks/vista_memory/results/memory.csv 30
VISTA_REPS=50 benchmarks/vista_memory/build/spmv benchmarks/vista_memory/results \
  asymmetric_matrices/small/bfw62a.mtx \
  asymmetric_matrices/large/memplus.mtx \
  asymmetric_matrices/large/af23560.mtx
# gzip works too; choose one copy of each matrix for a clean comparison:
VISTA_REPS=50 benchmarks/vista_memory/build/spmv benchmarks/vista_memory/results \
  asymmetric_matrices/beacxc.mtx.gz asymmetric_matrices/af23560.mtx.gz
```

SpMV overwrites both `spmv.csv` and `amortization.csv` each invocation; use a new output
directory for each run. Transfer and memory programs also overwrite their output path.
Each program selects CUDA device 0 within `CUDA_VISIBLE_DEVICES`; keep only the
allocated GPU visible. It exits nonzero on CUDA errors or numerical validation failure.
Mapped cuSPARSE API rejection is recorded as skipped, with the actual error message.
Illegal GPU accesses remain fatal; they are not presented as supported measurements.

A sample batch script is `vista.slurm`. Verify queues with `sinfo` and `qlimits` and
supply your allocation; no partition or allocation is hardcoded:

```bash
# Load the compatible build/runtime modules first. Review the script before submission.
sbatch --partition=<verified-GH-partition> --account=<your-allocation> \
  benchmarks/vista_memory/vista.slurm
```

The script runs the three programs sequentially without MPI and records `module list`
and `nvidia-smi`. It creates `benchmarks/vista_memory/results-JOBID/`. No jobs are
submitted by the build or by the benchmark executables.

## What is measured

All times are milliseconds; bandwidth is decimal GB/s (`bytes / ms / 1e6`). Sizes
4 KiB, 1 MiB, 16 MiB, 64 MiB, and 256 MiB are exactly powers of two. CSV rows include
GPU and CPU model, CUDA runtime/driver versions, compute capability, memory-access
capabilities, benchmark parameters, and correctness. GPU initialization is excluded.
Resource destruction, parsing, reference computation, validation, and CSV I/O are
excluded from measured GPU intervals. One sample per configuration is the mean of
the recorded repetition count, not independent statistical confidence bounds.

* **Transfer:** five warmups per direction; 30 measured copies by default. Pageable
  and pinned synchronous copies use the default stream; pinned asynchronous copies
  use a private stream. Events surround each copy, with the ending event synchronized.
  `event_ms` is the CUDA stream interval; it can include idle time caused by host-side
  staging/submission, especially for pageable memory. `host_ms` includes event
  submission, copy API latency, and completion synchronization. For short copies this
  overhead dominates. `GB_s` uses `event_ms`; also examine `bytes/(host_ms*1e6)` for
  end-to-end bandwidth. H2D data is copied back for exact verification; D2H is checked
  byte-for-byte against a deterministic pattern. Allocation is excluded.
* **Memory:** input is FP64; every element is read and transformed into a separate
  HBM output (`out[i]=2*input[i]+1`). All outputs are checked exactly. The output
  write traffic is identical in each strategy; these are kernel times, not isolated
  input bandwidth. `setup_ms` covers input allocation plus CPU initialization for
  host/managed allocations; staging-data construction and output allocation are
  outside it. `movement_event_ms`/`movement_host_ms` cover the HBM copy or managed
  prefetch; for host they measure an empty interval (no movement). `first_ms` is
  the first kernel after setup/movement; five subsequent warmups precede the mean
  of repeated launches. Kernel launch overhead is included in event batch intervals.
* **SpMV:** CPU CSR is shared across strategies, with FP64 values, signed 32-bit
  indices, zero-based indexing, and `CUSPARSE_SPMV_CSR_ALG1` for every mode. x and y
  always reside in HBM; beta=0. No custom kernel fallback or preprocessing is used.
  `setup_ms` covers CSR allocation plus host/managed initialization, separately from
  CSR movement. Descriptor, workspace, and vector setup are outside `first_ms`.
  `first_ms` measures the first SpMV after buffer-size querying; five warmups precede
  the mean repeated SpMV time. First and repeated results are checked against a CPU
  reference with nonuniform x. `max_scaled_error` is max absolute error divided by
  `(1 + sum(abs(Aij*xj)))` per row, with tolerance 1e-10 and nonfinite rejection.
* **Amortization:** three actual measured trials per count (1, 10, 100, 1000).
  `total_host_ms` spans CSR copy/prefetch, its synchronization, N SpMVs, and final
  event synchronization. `movement_host_ms` reports its movement portion;
  `spmv_event_ms` is the SpMV-only GPU interval.
  Each managed trial resets CSR to CPU using CPU prefetch outside timing, so the
  next GPU prefetch is real. HBM CSR is recopied each trial; host CSR has no movement.
  Allocations, descriptor creation, vector setup, CPU reset, and verification are
  excluded. This estimates reuse of an already allocated matrix, not full process
  startup. Trials follow previous SpMV runs, so kernels/workspaces and caches can be
  warm; these totals are not cold-start timings. No synthetic multiplication of a
  single SpMV time is used.

## Placement and interpretation

HBM uses `cudaMalloc`; managed uses `cudaMallocManaged` and explicit GPU prefetch.
Managed experiments are skipped unless managed allocation and concurrent managed
access are supported. Oversubscription or competing jobs can still change residency.

CPU-resident storage uses `cudaHostAllocMapped` with `cudaHostGetDevicePointer`, only
when host mapping is supported (SpMV additionally requires unified addressing). The
allocation remains host memory: there is no explicit input copy to HBM or managed
migration. GPU caching of host data is still possible. Mapped pinned memory may
consume substantial host page-locking resources; allocation failures are fatal.
This is a portable mapped-memory experiment, not an assumption that an ordinary
`malloc` pointer works on every CUDA GPU. The logged pageable and host-page-table
capabilities help identify ATS systems. On a Vista GH node, the GPU accesses Grace
host memory across C2C. On other systems this may be PCIe; software mapping capability
alone does not identify the physical interconnect. Inspect the model/topology logs.
cuSPARSE acceptance of mapped CSR must be established on the actual runtime; skipped
rows never silently substitute managed or HBM storage.

The repository contains 24 supported real files (including compressed/uncompressed
copies). The largest supported CSR here is `af23560`, 23,560 square, 484,256 nonzeros,
about 5.9 MB. Such matrices can fit in GPU cache: repeated host CSR timings need not
represent streaming Grace DRAM/C2C bandwidth. Use larger external real matrices when
available, without replacing repository files. `young1c.mtx.gz` is **complex symmetric**
and is rejected explicitly; it cannot be represented faithfully by real FP64 values.
Coordinate real, integer, and pattern general/symmetric formats are supported;
array, complex, Hermitian, and skew-symmetric formats are rejected. Symmetric entries
are mirrored off diagonal, duplicates are summed, and explicit zeros are retained.

Repeat entire runs in separate directories for variability. Keep load, clock/power
conditions and CPU/NUMA affinity consistent, and preserve the job logs. Compare first
access to steady state before attributing a difference to interconnect bandwidth.

## Plots and crossover

With pandas and matplotlib installed in your Python environment:

```bash
python3 benchmarks/vista_memory/plot.py benchmarks/vista_memory/results
```

Outputs: `bandwidth.png`, `memory.png`, `spmv.png`, `amortization.png`, and
`crossover.txt`. Missing CSVs and skipped rows are omitted. The crossover report
compares measured host totals, gives a bracket between tested counts when HBM becomes
faster, and says when HBM already wins at the first measured count. It does not
interpolate or claim statistical significance. Rerun nearby counts or extend the
source's count list if a finer crossover estimate is needed.

## Local validation and references

Local environment: x86_64 WSL, CUDA 12.0.140, GCC 13.3 default; GCC 12 successfully
compiled all CUDA programs without an unsupported-compiler override. All CPU reader
unit tests and all 24 supported repository matrix loads passed; the complex matrix
was rejected as intended. GPU executables stop at initialization with
`CUDA driver version is insufficient for CUDA runtime version`; `nvidia-smi` reports
OS-blocked GPU access. No CUDA timings, cuSPARSE placement acceptance, numerical GPU
checks, C2C behavior, or Vista Slurm execution were validated locally. CMake and
pandas/matplotlib are absent locally, so the CMake configure and plot rendering still
need validation; source compilation was checked directly with nvcc and g++.

Official references consulted:
- [TACC Vista guide](https://docs.tacc.utexas.edu/hpc/vista/)
- [CUDA memory management API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
- [CUDA Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)
- [CUDA host compiler support](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy)
