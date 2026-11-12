# Build and run on TACC Vista (Arm Grace CPU); also builds on x86-64 Linux.
#
#   make                      build cpu_only (dense), cpu_sparse (CSR v1), cpu_sparse_v2 and tools/
#   make test                 correctness tests (qa/run_tests.sh), about a minute
#   make race-check           ThreadSanitizer + Archer race check with clang (qa/race_check.sh)
#   make bench                dense vs sparse benchmark + plots (test_matrices branch)
#   make ablation             measure each v2 scaling fix (sbatch ablation.sh)
#   make tsan                 GCC ThreadSanitizer run          (PROG=cpu_only for dense)
#   make perf                 perf stat, pinned to $(CORES)     (PROG=cpu_only for dense)
#   make clean                remove binaries
#   make clean-results        remove benchmark/perf output folders and solution files
#
# qa_rcm branch edits (x86 flag, tools/, test, race-check): generated with
# assistance of claude code

# -mcpu=native is the Arm spelling; x86 GCC wants -march=native
ifeq ($(shell uname -m),aarch64)
  ARCHFLAG ?= -mcpu=native
else
  ARCHFLAG ?= -march=native
endif

CXX      := g++
CXXFLAGS := -std=c++20 -O3 $(ARCHFLAG) -fopenmp -Wall -Wextra -pthread -g -fno-omit-frame-pointer
TSANFLAGS := -std=c++20 -O1 -g -fsanitize=thread $(ARCHFLAG) -fopenmp -Wall -Wextra -pthread
LDFLAGS  := -fopenmp

PROGS    := cpu_only cpu_sparse cpu_sparse_v2
TOOLS    := tools/gen_fem tools/rcm_study tools/check_solution tools/omp_barrier_cost
PROG     ?= cpu_sparse

# Race check: clang with LLVM's OpenMP runtime (libomp), whose Archer tool
# tells ThreadSanitizer about OpenMP synchronization. GCC's libgomp cannot.
LLVM_DIR ?= /usr/lib/llvm-18
CLANGXX  ?= clang++-18
CLANGTSAN := -std=c++20 -O1 -g -fsanitize=thread -fopenmp -I$(LLVM_DIR)/include \
             -L$(LLVM_DIR)/lib -Wl,-rpath,$(LLVM_DIR)/lib

# Matrix used by tsan/perf (small for tsan; it runs 5-15x slower)
MATRIX   ?= Matrices
CORES    := 0-7
THREADS  := 8

# Arm Neoverse events (verify names with: perf list)
PERF_EVENTS := cycles,instructions,l1d_cache,l1d_cache_refill,l2d_cache_refill,ll_cache_miss_rd,stall_backend,context-switches

all: $(PROGS) $(TOOLS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

tools/%: tools/%.cpp tools/sparse_io.hpp tools/rcm.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

%_tsan: %.cpp
	$(CXX) $(TSANFLAGS) $< -o $@ $(LDFLAGS)

%_clangtsan: %.cpp
	$(CLANGXX) $(CLANGTSAN) $< -o $@

test: all
	bash qa/run_tests.sh

race-check: $(TOOLS) $(addsuffix _clangtsan,$(PROGS))
	LLVM_DIR=$(LLVM_DIR) bash qa/race_check.sh

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
	rm -f $(PROGS) $(TOOLS) $(addsuffix _tsan,$(PROGS)) $(addsuffix _clangtsan,$(PROGS))

clean-results:
	rm -rf benchall_*/ perfrun_*/ ablation_*/ bench_matrices/ *_solution.txt perf.data* cachegrind.out.*

.PHONY: all test race-check ablation bench tsan perf clean clean-results
