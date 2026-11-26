# Build and run on TACC Vista (Arm Grace CPU).
#
#   make                      build cpu_only (dense), cpu_sparse (CSR v1) and cpu_sparse_v2
#   make gpu                  build gpu_cg (needs: module load gcc/14.2.0 cuda/13.3, gh node)
#   make gpu-test             logic test of the GPU solver, runs on any CPU
#   make gpu-bench            sbatch gpu_bench.slurm (all GPU modes vs CPU on a gh node)
#   make bench                dense vs sparse benchmark + plots (test_matrices branch)
#   make ablation             measure each v2 scaling fix (sbatch ablation.sh)
#   make tsan                 ThreadSanitizer check            (PROG=cpu_only for dense)
#   make perf                 perf stat, pinned to $(CORES)     (PROG=cpu_only for dense)
#   make clean                remove binaries
#   make clean-results        remove benchmark/perf output folders and solution files

CXX      := g++
CXXFLAGS := -std=c++20 -O3 -mcpu=native -fopenmp -Wall -Wextra -pthread -g -fno-omit-frame-pointer
TSANFLAGS := -std=c++20 -O1 -g -fsanitize=thread -mcpu=native -fopenmp -Wall -Wextra -pthread
LDFLAGS  := -fopenmp

PROGS    := cpu_only cpu_sparse cpu_sparse_v2

# GPU (Hopper H200 = sm_90). Host code is compiled by g++ with OpenMP.
NVCC     := nvcc
NVFLAGS  := -std=c++17 -O3 -arch=sm_90 -lineinfo -Xcompiler -fopenmp,-O3,-mcpu=native
GPU_HDRS := sparse_prep.hpp gpu_cg_logic.hpp
PROG     ?= cpu_sparse

# Matrix used by tsan/perf (small for tsan; it runs 5-15x slower)
MATRIX   ?= Matrices
CORES    := 0-7
THREADS  := 8

# Arm Neoverse events (verify names with: perf list)
PERF_EVENTS := cycles,instructions,l1d_cache,l1d_cache_refill,l2d_cache_refill,ll_cache_miss_rd,stall_backend,context-switches

all: $(PROGS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

gpu_cg: gpu_cg.cu $(GPU_HDRS)
	$(NVCC) $(NVFLAGS) $< -o $@ -lcusparse -lgomp

test_gpu_logic: test_gpu_logic.cpp $(GPU_HDRS)
	$(CXX) -std=c++17 -O2 -fopenmp $< -o $@

gpu: gpu_cg

gpu-test: test_gpu_logic
	./test_gpu_logic

gpu-bench: gpu_cg cpu_sparse_v2 test_gpu_logic
	sbatch gpu_bench.slurm

%_tsan: %.cpp
	$(CXX) $(TSANFLAGS) $< -o $@ $(LDFLAGS)

ablation: all
	sbatch ablation.sh

bench: all
	bash benchmark.sh

tsan: $(PROG)_tsan
	OMP_NUM_THREADS=4 setarch $$(uname -m) -R ./$(PROG)_tsan $(MATRIX)

perf: $(PROG)
	OMP_NUM_THREADS=$(THREADS) OMP_PLACES=cores OMP_PROC_BIND=close \
	perf stat -e $(PERF_EVENTS) taskset -c $(CORES) ./$(PROG) $(MATRIX)

clean:
	rm -f $(PROGS) $(addsuffix _tsan,$(PROGS)) gpu_cg test_gpu_logic

clean-results:
	rm -rf benchall_*/ perfrun_*/ ablation_*/ bench_matrices/ gpu_bench_*/ *_solution.txt perf.data* cachegrind.out.*

.PHONY: all ablation bench tsan perf clean clean-results gpu gpu-test gpu-bench
