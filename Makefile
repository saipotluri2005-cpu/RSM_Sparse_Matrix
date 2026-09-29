# Compiler and Flags for TACC Vista (72-Core ARM Grace CPU)
CXX      := g++
CXXFLAGS := -std=c++20 -O3 -mcpu=native -fopenmp -Wall -Wextra -pthread
LDFLAGS  := -fopenmp

TARGET   := cpu_only
SRC      := cpu_only.cpp

# Cores to pin to and thread count for perf runs
CORES    := 0-7
THREADS  := 8

# Arm Neoverse events (verify names with: perf list)
PERF_EVENTS := cycles,instructions,l1d_cache,l1d_cache_refill,l2d_cache_refill,ll_cache_miss_rd,stall_backend,context-switches

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# ThreadSanitizer build: -O1 -g, no -O3, separate binary
$(TARGET)_tsan: $(SRC)
	$(CXX) -std=c++20 -O1 -g -fsanitize=thread -mcpu=native -fopenmp -Wall -Wextra -pthread $(SRC) -o $@ -fopenmp

tsan: $(TARGET)_tsan
	setarch $$(uname -m) -R ./$(TARGET)_tsan

# Perf stat on the optimized binary, pinned to 8 cores
perf: $(TARGET)
	OMP_NUM_THREADS=$(THREADS) OMP_PLACES=cores OMP_PROC_BIND=close \
	perf stat -e $(PERF_EVENTS) taskset -c $(CORES) ./$(TARGET)

clean:
	rm -f $(TARGET) $(TARGET)_tsan

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run tsan perf