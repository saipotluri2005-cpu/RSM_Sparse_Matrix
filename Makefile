# Build and run on TACC Vista (Arm Grace CPU).
#
#   make                      build cpu_only (dense) and cpu_sparse (CSR)
#   make bench                dense vs sparse benchmark + plots (test_matrices branch)
#   make tsan                 ThreadSanitizer check            (PROG=cpu_only for dense)
#   make perf                 perf stat, pinned to $(CORES)     (PROG=cpu_only for dense)
#   make clean                remove binaries
#   make clean-results        remove benchmark/perf output folders and solution files

CXX      := g++
CXXFLAGS := -std=c++20 -O3 -mcpu=native -fopenmp -Wall -Wextra -pthread -g -fno-omit-frame-pointer
TSANFLAGS := -std=c++20 -O1 -g -fsanitize=thread -mcpu=native -fopenmp -Wall -Wextra -pthread
LDFLAGS  := -fopenmp

PROGS    := cpu_only cpu_sparse
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

%_tsan: %.cpp
	$(CXX) $(TSANFLAGS) $< -o $@ $(LDFLAGS)

bench: all
	bash benchmark.sh

tsan: $(PROG)_tsan
	OMP_NUM_THREADS=4 setarch $$(uname -m) -R ./$(PROG)_tsan $(MATRIX)

perf: $(PROG)
	OMP_NUM_THREADS=$(THREADS) OMP_PLACES=cores OMP_PROC_BIND=close \
	perf stat -e $(PERF_EVENTS) taskset -c $(CORES) ./$(PROG) $(MATRIX)

clean:
	rm -f $(PROGS) $(addsuffix _tsan,$(PROGS))

clean-results:
	rm -rf benchall_*/ perfrun_*/ bench_matrices/ *_solution.txt perf.data* cachegrind.out.*

.PHONY: all bench tsan perf clean clean-results
