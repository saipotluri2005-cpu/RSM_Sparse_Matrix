# Build and run on TACC Vista (Arm Grace CPU).
#
#   make                      build cpu_only (dense), cpu_sparse (CSR v1) and cpu_sparse_v2
#   make gpu                  build gpu_cg (finds nvcc, gcc 14 and cuSPARSE itself on Vista)
#   make gpu-info             show which nvcc / g++ / cuSPARSE the GPU build will use
#   make gpu-test             logic test of the GPU solver, runs on any CPU
#   make gpu-run              run gpu_cg on $(MATRIX) (on a gh node, e.g. inside idev)
#   make gpu-quick            sbatch a short benchmark (one 64^3 matrix, 1 rep)
#   make gpu-bench            sbatch the full benchmark (all GPU modes vs CPU on a gh node)
#   make viz                  sbatch viz.slurm: generate bridges, solve on the GPU, render MP4s
#   make bench                dense vs sparse benchmark + plots (test_matrices branch)
#   make ablation             measure each v2 scaling fix (sbatch ablation.sh)
#   make tsan                 ThreadSanitizer check            (PROG=cpu_only for dense)
#   make perf                 perf stat, pinned to $(CORES)     (PROG=cpu_only for dense)
#   make clean                remove binaries
#   make clean-results        remove benchmark/perf output folders and solution files

# Vista install locations, used when the modules are not loaded.
# Override on another machine, e.g.  make gpu VISTA_SDK=/opt/nvidia/hpc_sdk/Linux_aarch64/25.1
VISTA_SDK  ?= /home1/apps/nvidia/Linux_aarch64/26.9
VISTA_GCC  ?= /opt/apps/gcc/14.2.0
CUDA_VER   ?= 13.3

# gcc 14 from the module if loaded, else its install path, else whatever g++ is
CXX      := $(or $(wildcard $(VISTA_GCC)/bin/g++),g++)
comma    := ,
GCC_RPATH := $(if $(wildcard $(VISTA_GCC)/lib64),-Wl$(comma)-rpath$(comma)$(VISTA_GCC)/lib64)
CXXFLAGS := -std=c++20 -O3 -mcpu=native -fopenmp -Wall -Wextra -pthread -g -fno-omit-frame-pointer
TSANFLAGS := -std=c++20 -O1 -g -fsanitize=thread -mcpu=native -fopenmp -Wall -Wextra -pthread
LDFLAGS  := -fopenmp $(GCC_RPATH)

PROGS    := cpu_only cpu_sparse cpu_sparse_v2

# GPU (Hopper H200 = sm_90). Host code is compiled by gcc 14 with OpenMP.
# nvcc: from PATH (module loaded) or straight from the Vista HPC SDK.
NVCC     := $(or $(shell which nvcc 2>/dev/null),$(wildcard $(VISTA_SDK)/cuda/$(CUDA_VER)/bin/nvcc),nvcc)
NVCC_DIR := $(patsubst %/bin/nvcc,%,$(NVCC))
CCBIN    := $(if $(wildcard $(VISTA_GCC)/bin/g++),-ccbin $(VISTA_GCC)/bin/g++)
NVFLAGS  := -std=c++17 -O3 -arch=sm_90 -lineinfo $(CCBIN) \
            -Xcompiler -fopenmp,-O3,-mcpu=native,-Wno-stringop-overflow
# cuSPARSE: the HPC SDK keeps it in math_libs/<ver>/, next to cuda/<ver>/, not
# inside it. Order: CUSPARSE=... on the command line, the known Vista path,
# then a search. stubs/ copies only link and fail at run time, so skip them.
CUSPARSE ?= $(firstword \
    $(wildcard $(NVCC_DIR)/../../math_libs/$(CUDA_VER)/targets/sbsa-linux/lib/libcusparse.so) \
    $(wildcard $(VISTA_SDK)/math_libs/$(CUDA_VER)/targets/sbsa-linux/lib/libcusparse.so) \
    $(shell find $$TACC_CUDA_LIB $$TACC_CUDA_DIR $$TACC_CUDA_MATH_LIB \
        $(if $(NVCC_DIR),"$(NVCC_DIR)" "$(NVCC_DIR)/.." "$(NVCC_DIR)/../..") \
        -maxdepth 6 -name 'libcusparse.so*' -not -path '*/stubs/*' -not -path '*/REDIST/*' \
        2>/dev/null | sort | head -1))
CUSPARSE_DIR := $(patsubst %/,%,$(dir $(CUSPARSE)))
# rpath: gpu_cg finds cuSPARSE and gcc 14's libstdc++/libgomp at run time
# without any module or LD_LIBRARY_PATH
GPU_LIBS := $(CUSPARSE) -Xlinker -rpath=$(CUSPARSE_DIR) \
            $(if $(wildcard $(VISTA_GCC)/lib64),-Xlinker -rpath=$(VISTA_GCC)/lib64) -lgomp
GPU_ENV  := LD_LIBRARY_PATH=$(CUSPARSE_DIR):$(VISTA_GCC)/lib64:$$LD_LIBRARY_PATH
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
	@if [ ! -x "$(NVCC)" ] && ! command -v "$(NVCC)" > /dev/null; then \
	    echo "ERROR: nvcc not found. Run: module load gcc/14.2.0 cuda/$(CUDA_VER)"; exit 1; fi
	@if [ -z "$(CUSPARSE)" ]; then \
	    echo "ERROR: libcusparse not found. Find it with:"; \
	    echo "  find $(NVCC_DIR)/../.. -name 'libcusparse.so*' | grep -v stubs"; \
	    echo "then: make gpu CUSPARSE=/full/path/libcusparse.so"; exit 1; fi
	@echo "nvcc:     $(NVCC)"
	@echo "cuSPARSE: $(CUSPARSE)"
	$(NVCC) $(NVFLAGS) $< -o $@ $(GPU_LIBS)

test_gpu_logic: test_gpu_logic.cpp $(GPU_HDRS)
	$(CXX) -std=c++17 -O2 -fopenmp -Wno-stringop-overflow $< -o $@ $(GCC_RPATH)

gpu: gpu_cg

gpu-info:
	@echo "nvcc:      $(NVCC)"
	@echo "host g++:  $(CXX)"
	@echo "cuSPARSE:  $(or $(CUSPARSE),NOT FOUND)"
	@$(NVCC) --version 2>/dev/null | tail -2 || true

gpu-test: test_gpu_logic
	./test_gpu_logic

gpu-run: gpu_cg
	@nvidia-smi -L > /dev/null 2>&1 || { echo "No GPU here. Use a gh node: idev -p gh -N 1 -n 1 -t 00:30:00 -A TRA25006"; exit 1; }
	$(GPU_ENV) ./gpu_cg $(MATRIX)

gpu-quick: gpu_cg cpu_sparse_v2 test_gpu_logic
	SIZES="64" REPS=1 sbatch gpu_bench.slurm

gpu-bench: gpu_cg cpu_sparse_v2 test_gpu_logic
	sbatch gpu_bench.slurm

viz: gpu_cg
	sbatch viz.slurm

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
	rm -rf benchall_*/ perfrun_*/ ablation_*/ bench_matrices/ gpu_bench_*/ viz_cases/ viz_out/ *_solution.txt perf.data* cachegrind.out.*

.PHONY: all ablation bench tsan perf clean clean-results gpu gpu-info gpu-test gpu-run gpu-quick gpu-bench viz
