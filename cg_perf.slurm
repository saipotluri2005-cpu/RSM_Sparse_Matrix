#!/bin/bash
#SBATCH -J cg_perf
#SBATCH -o cg_perf.%j.out
#SBATCH -e cg_perf.%j.err
#SBATCH -p gg                  # Grace-Grace (144 cores). Use gh for 72-core Grace-Hopper.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A YOUR_ALLOCATION

# ---------------------------------------------------------------------------
# Staged perf analysis of the OpenMP dense CG solver.
# Each stage is a SEPARATE run: perf adds overhead, so timings are taken with
# no profiler attached (stage 1) and everything else is for diagnosis only.
#
# Build first (frame pointers + -g so perf can attribute time and walk stacks):
#   g++ -O3 -g -fno-omit-frame-pointer -mcpu=native -std=c++17 -fopenmp \
#       cg_omp.cpp -o cg_omp
# ---------------------------------------------------------------------------

EXE=${EXE:-./cg_omp}
MATDIR=${MATDIR:-./matrices_one}     # use ONE representative matrix here
NT_MAX=${NT_MAX:-144}                # 144 on gg, 72 on gh
SWEEP="1 2 4 8 18 36 72 144"
REPS=${REPS:-3}                      # repeats for perf stat variation
NUMA_DELAY=${NUMA_DELAY:-20}         # seconds into the run to sample numastat

OUT=perf_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"

export OMP_PLACES=cores
export OMP_PROC_BIND=close
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 99)

# ---------------- Stage 0: environment -------------------------------------
{
  date; hostname
  echo "perf_event_paranoid = $PARANOID"
  which perf; perf --version
  echo "nproc = $(nproc)"
  lscpu
  numactl -H
} > "$OUT/00_env.txt" 2>&1
perf list > "$OUT/00_perf_list.txt" 2>&1
perf stat -e cycles,instructions true 2> "$OUT/00_hwcheck.txt"
echo "== env written to $OUT/00_env.txt (paranoid=$PARANOID) =="
cat "$OUT/00_hwcheck.txt"   # "<not supported>" here means no HW counters

# ---------------- Stage 1: clean scaling sweep (NO perf attached) ----------
echo "== Stage 1: scaling sweep =="
: > "$OUT/01_scaling.txt"
for t in $SWEEP; do
  [ "$t" -le "$NT_MAX" ] || continue
  /usr/bin/time -v env OMP_NUM_THREADS=$t "$EXE" "$MATDIR" \
      > "$OUT/01_t${t}.out" 2> "$OUT/01_t${t}.time"
  ms=$(grep -h 'Solve Time' "$OUT/01_t${t}.out" | awk '{s+=$3} END{print s}')
  echo "$t $ms" >> "$OUT/01_scaling.txt"
done
awk 'NR==1{base=$2} {printf "threads=%-4d solve_ms=%-12.1f speedup=%.2fx\n",$1,$2,base/$2}' \
    "$OUT/01_scaling.txt" | tee "$OUT/01_scaling_summary.txt"

export OMP_NUM_THREADS=$NT_MAX

# ---------------- Stage 2: perf stat, 1 thread vs N threads ----------------
# task-clock should stay ~constant from 1 -> N threads. If it grows, threads
# are spinning / waiting on memory instead of doing useful work.
echo "== Stage 2: perf stat =="
BASE=task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions
OMP_NUM_THREADS=1 perf stat -r "$REPS" -e $BASE -o "$OUT/02_stat_1thread.txt" \
    -- "$EXE" "$MATDIR" > /dev/null 2>&1
perf stat -r "$REPS" -e $BASE -o "$OUT/02_stat_Nthreads.txt" \
    -- "$EXE" "$MATDIR" > /dev/null 2>&1

# Arm Neoverse V2 events (check names against 00_perf_list.txt).
# Few events per run so counters are not multiplexed.
perf stat -r "$REPS" -e cycles,instructions,stall_frontend,stall_backend \
    -o "$OUT/02_stat_stalls.txt" -- "$EXE" "$MATDIR" > /dev/null 2>&1
perf stat -r "$REPS" -e l1d_cache,l1d_cache_refill,l2d_cache,l2d_cache_refill \
    -o "$OUT/02_stat_cache.txt"  -- "$EXE" "$MATDIR" > /dev/null 2>&1
perf stat -r "$REPS" -e ll_cache_rd,ll_cache_miss_rd,bus_access,dtlb_walk \
    -o "$OUT/02_stat_mem.txt"    -- "$EXE" "$MATDIR" > /dev/null 2>&1

# ---------------- Stage 3: per-thread counts (imbalance / starvation) ------
echo "== Stage 3: per-thread =="
perf stat --per-thread -e task-clock,cycles,instructions \
    -o "$OUT/03_per_thread.txt" -- "$EXE" "$MATDIR" > /dev/null 2>&1

# ---------------- Stage 4: passive waiting, make blocking visible ----------
# Default libgomp SPINS at barriers, so waiting threads look "busy".
# With passive policy they sleep, so waiting shows up as context switches
# and as lower "CPUs utilized".
echo "== Stage 4: passive wait policy =="
OMP_WAIT_POLICY=passive /usr/bin/time -v "$EXE" "$MATDIR" \
    > "$OUT/04_passive.out" 2> "$OUT/04_passive.time"
OMP_WAIT_POLICY=passive perf stat -e task-clock,context-switches,cpu-migrations \
    -o "$OUT/04_passive_stat.txt" -- "$EXE" "$MATDIR" > /dev/null 2>&1
grep -E 'Elapsed|Percent of CPU|context switches' "$OUT/04_passive.time" \
    > "$OUT/04_passive_summary.txt"
grep -E 'Elapsed|Percent of CPU|context switches' "$OUT/01_t${NT_MAX}.time" \
    > "$OUT/04_spin_summary.txt"

# ---------------- Stage 5: where on-CPU time goes --------------------------
echo "== Stage 5: perf record =="
D="$OUT/perf.data"
perf record -F 499 --call-graph fp -o "$D" -- "$EXE" "$MATDIR" \
    > "$OUT/05_record.out" 2>&1
perf report -i "$D" --stdio --no-children --sort dso,sym -g none \
    --percent-limit 0.3 > "$OUT/05_report_symbols.txt" 2>&1
perf report -i "$D" --stdio --no-children --sort tid -g none \
    > "$OUT/05_report_per_thread.txt" 2>&1
perf report -i "$D" --stdio --no-children --sort sym -g caller \
    --percent-limit 1 > "$OUT/05_report_callers.txt" 2>&1
perf annotate -i "$D" --stdio 2> /dev/null | head -n 600 \
    > "$OUT/05_annotate_top.txt"
# Share of samples spent waiting in the OpenMP runtime (barrier spin/futex):
grep -E 'libgomp|gomp_|do_spin|futex|wait' "$OUT/05_report_symbols.txt" \
    > "$OUT/05_openmp_wait_share.txt"

# ---------------- Stage 6: where threads BLOCK (needs paranoid <= 1) -------
if [ "$PARANOID" -le 1 ]; then
  echo "== Stage 6: context-switch stacks =="
  OMP_WAIT_POLICY=passive perf record -e context-switches -c 100 \
      --call-graph dwarf -o "$OUT/perf_cs.data" -- "$EXE" "$MATDIR" \
      > /dev/null 2>&1
  perf report -i "$OUT/perf_cs.data" --stdio --no-children \
      > "$OUT/06_ctx_switch_stacks.txt" 2>&1
else
  echo "== Stage 6 skipped: perf_event_paranoid=$PARANOID (>1); use stage 4 =="
fi

# ---------------- Stage 7: NUMA first-touch + pinning check ---------------
echo "== Stage 7: NUMA / affinity =="
OMP_DISPLAY_AFFINITY=TRUE "$EXE" "$MATDIR" > "$OUT/07_affinity.out" 2>&1 &
PID=$!
sleep "$NUMA_DELAY"
if command -v numastat > /dev/null; then
  numastat -p $PID > "$OUT/07_numastat.txt" 2>&1   # memory split across nodes
fi
wait $PID

echo "== Done. Results in $OUT/ =="
echo "-- scaling --";  cat "$OUT/01_scaling_summary.txt"
echo "-- 1 thread vs N threads (task-clock, CPUs utilized, IPC) --"
grep -E 'task-clock|CPUs utilized|insn per cycle|elapsed' \
    "$OUT/02_stat_1thread.txt" "$OUT/02_stat_Nthreads.txt"
