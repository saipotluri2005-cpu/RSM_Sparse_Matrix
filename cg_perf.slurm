#!/bin/bash
#SBATCH -J cg_bench_perf
#SBATCH -o cg_bench_perf.%j.txt
#SBATCH -p gh                  # Vista Grace-Hopper node: 72 cores (use gg for 144-core Grace-Grace)
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TRA25006

# ---------------------------------------------------------------------------
# Target: Vista gh node (72 cores). Allocation: TRA25006.
#
# PART A  Benchmark: every matrix in ./Matrices (.mtx / .mtx.gz, recursive)
#         across a thread sweep -> raw.csv, summary.txt, not_solved.csv
#         (identical output to bench_all.sh)
# PART B  Staged perf analysis on ONE representative matrix, chosen
#         automatically from Part A (largest converged matrix), or set
#         REP_MATRIX=<label> to pick one.
#
# Build first:  make            (cpu_sparse is built with -g -fno-omit-frame-pointer)
#
#   sbatch cg_perf.slurm                     profiles cpu_sparse
#   EXE=./cpu_only sbatch cg_perf.slurm      profiles the dense version
#
# Matrices come from the test_matrices branch (same worktree as benchmark.sh).
# For the dense-vs-sparse comparison and plots, use benchmark.sh instead.
#
# Optional environment variables:
#   EXE=path               solver binary (default: ./cpu_sparse, else ./cpu_only)
#   TEST_DIR=path          folder to scan  (default: test_matrices branch worktree)
#   TEST_BRANCH=name       branch holding the matrices (default: test_matrices)
#   POISSON_SIZES="30 60"  also generate and run these Poisson grids
#   THREADS="1 2 4 8"      thread counts (default: powers of two up to cores)
#   REPS=3                 repeats per (matrix, threads)
#   MAX_DENSE_GB=80        skip matrices needing more dense memory than this
#   REP_MATRIX=label       matrix used for the perf stages
#   SKIP_PERF=1            run only Part A
#   NUMA_DELAY=20          seconds into the run to sample numastat
# ---------------------------------------------------------------------------

set -u

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    ROOT="$SLURM_SUBMIT_DIR"
else
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [ -z "${EXE:-}" ]; then
    if   [ -x "$ROOT/cpu_sparse" ]; then EXE="$ROOT/cpu_sparse"
    elif [ -x "$ROOT/cpu_only" ];   then EXE="$ROOT/cpu_only"
    else EXE="$ROOT/cpu_sparse"; fi
fi
GEN="${GEN:-$ROOT/gen_poisson.py}"
TEST_BRANCH="${TEST_BRANCH:-test_matrices}"
export SAVE_SOLUTION=0

# ---- Test matrices from the test_matrices branch (git worktree) -----------
if [ -n "${TEST_DIR:-}" ]; then
    MATRICES="$TEST_DIR"
else
    WT="$(dirname "$ROOT")/RSM_test_matrices"
    git -C "$ROOT" fetch -q origin "$TEST_BRANCH" 2> /dev/null
    if git -C "$ROOT" rev-parse -q --verify "origin/$TEST_BRANCH" > /dev/null; then
        REF="origin/$TEST_BRANCH"
    else
        REF="$TEST_BRANCH"
    fi
    if [ -d "$WT" ]; then
        git -C "$WT" checkout -q --detach "$REF" || { echo "ERROR: cannot update $WT"; exit 1; }
    else
        git -C "$ROOT" worktree add -q --detach "$WT" "$REF" ||
            { echo "ERROR: cannot check out branch $TEST_BRANCH into $WT"; exit 1; }
    fi
    MATRICES="$WT"
fi
POISSON_SIZES="${POISSON_SIZES:-}"
NUMA_DELAY="${NUMA_DELAY:-20}"

[ -x "$EXE" ] || { echo "ERROR: $EXE not found. Build it first."; exit 1; }
[ -d "$MATRICES" ] || { echo "ERROR: folder $MATRICES not found."; exit 1; }

# ---- Defaults: gentle on login nodes ---------------------------------------
HOST=$(hostname -s)
NCORES=$(nproc)
if [[ "$HOST" == login* && "${FORCE:-0}" != "1" ]]; then
    echo "NOTE: login node detected ($HOST). Light defaults; nproc reports $NCORES core(s)."
    MAXT=$(( NCORES < 8 ? NCORES : 8 ))
    export MAX_DENSE_GB="${MAX_DENSE_GB:-2}"
else
    MAXT=$NCORES
    export MAX_DENSE_GB="${MAX_DENSE_GB:-80}"   # gh nodes have ~120 GB CPU memory
fi
REPS="${REPS:-3}"

if [ -z "${THREADS:-}" ]; then
    THREADS=""
    t=1
    while [ "$t" -lt "$MAXT" ]; do THREADS="$THREADS $t"; t=$((t * 2)); done
    THREADS="$THREADS $MAXT"
fi

export OMP_PLACES=cores
export OMP_PROC_BIND=close

# ---- Collect matrices -------------------------------------------------------
mapfile -t FILES < <(find "$MATRICES" -path '*/.git' -prune -o -type f \
                          \( -name '*.mtx' -o -name '*.mtx.gz' \) -print | sort)

if [ -n "$POISSON_SIZES" ]; then
    GENDIR="$ROOT/bench_matrices"
    mkdir -p "$GENDIR"
    for n in $POISSON_SIZES; do
        pf="$GENDIR/poisson2d_${n}.mtx"
        [ -f "$pf" ] || (cd "$GENDIR" && python3 "$GEN" "$n")
        FILES+=("$pf")
    done
fi

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "No .mtx / .mtx.gz files found in $MATRICES"
    exit 1
fi

# ---- Output folder ----------------------------------------------------------
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/perfrun_$(basename "$EXE")_${SLURM_JOB_ID:-$STAMP}"
WORK="$OUT/work"
PERF="$OUT/perf"
mkdir -p "$OUT" "$WORK" "$PERF"
CSV="$OUT/raw.csv"
SKIPPED="$OUT/not_solved.csv"
echo "matrix,N,threads,rep,iterations,solve_ms,rel_residual" > "$CSV"
echo "matrix,reason" > "$SKIPPED"

{
    echo "date:      $(date)"
    echo "host:      $HOST   cores: $NCORES"
    echo "exe:       $EXE"
    echo "threads:   $THREADS"
    echo "reps:      $REPS"
    echo "matrices:  ${#FILES[@]} file(s)"
    printf '  %s\n' "${FILES[@]}"
    echo "MAX_DENSE_GB=$MAX_DENSE_GB OMP_PLACES=$OMP_PLACES OMP_PROC_BIND=$OMP_PROC_BIND"
} | tee "$OUT/config.txt"
echo

# ============================================================================
# PART A: benchmark sweep (NO profiler attached, so timings are clean)
# ============================================================================
cd "$WORK" || exit 1

for f in "${FILES[@]}"; do
    label=$(basename "$f"); label=${label%.gz}; label=${label%.mtx}
    echo "---- $label ----"
    skip=0
    for t in $THREADS; do
        [ "$skip" = 1 ] && break
        for rep in $(seq 1 "$REPS"); do
            out=$(OMP_NUM_THREADS=$t "$EXE" "$f" 2>&1)
            ms=$(awk '/Solve Time/{print $3}' <<< "$out")
            if [ -z "$ms" ]; then
                reason=$(grep -m1 -E 'SKIPPED|BREAKDOWN|Error' <<< "$out")
                reason=${reason//,/;}
                echo "  NOT SOLVED: ${reason:-unknown reason}"
                echo "$label,${reason:-unknown}" >> "$SKIPPED"
                skip=1
                break
            fi
            N=$(awk '/Matrix Size/{print $3}' <<< "$out")
            iters=$(awk '/Iterations Taken/{print $3}' <<< "$out")
            res=$(awk '/Relative Residual/{print $3}' <<< "$out")
            echo "$label,$N,$t,$rep,$iters,$ms,$res" >> "$CSV"
            printf "  threads=%-4s rep=%s  N=%-6s iters=%-5s solve=%-10s ms  resid=%s\n" \
                   "$t" "$rep" "$N" "$iters" "$ms" "$res"
        done
    done
done

# ---- Summary ----------------------------------------------------------------
echo
python3 - "$CSV" "$SKIPPED" << 'PYEOF' | tee "$OUT/summary.txt"
import csv, sys, statistics
from collections import defaultdict, OrderedDict

rows = list(csv.DictReader(open(sys.argv[1])))
skipped = list(csv.DictReader(open(sys.argv[2])))

times = defaultdict(list)
iters = defaultdict(set)
resid = defaultdict(float)
size = OrderedDict()
for r in rows:
    k = (r["matrix"], int(r["threads"]))
    times[k].append(float(r["solve_ms"]))
    iters[k].add(r["iterations"])
    try:
        resid[k] = max(resid[k], float(r["rel_residual"]))
    except ValueError:
        resid[k] = float("nan")
    size[r["matrix"]] = r["N"]

def status(k):
    rs = resid[k]
    its = [int(i) for i in iters[k]]
    if rs != rs or rs == float("inf"):
        return "BAD (nan/inf)"
    if max(its) >= 5000 or rs > 1e-4:
        return "NO_CONVERGE"
    if len(its) > 1:
        return "ITERS VARY"
    return "ok"

overview = []
for m in size:
    ts = sorted(t for (mm, t) in times if mm == m)
    base = min(times[(m, ts[0])])
    print(f"=== {m}  ({size[m]} x {size[m]}) ===")
    print(f"{'threads':>7} {'min ms':>11} {'median ms':>11} {'speedup':>8} {'effic':>7} {'iters':>8} {'max resid':>11}  status")
    for t in ts:
        v = times[(m, t)]
        mn, md = min(v), statistics.median(v)
        sp = base / mn
        eff = sp / (t / ts[0])
        it = ",".join(sorted(iters[(m, t)]))
        print(f"{t:>7} {mn:>11.3f} {md:>11.3f} {sp:>7.2f}x {eff*100:>6.0f}% {it:>8} {resid[(m, t)]:>11.2e}  {status((m, t))}")
    last = ts[-1]
    overview.append((m, size[m], ",".join(sorted(iters[(m, last)])), status((m, last)),
                     ts[0], min(times[(m, ts[0])]), last, min(times[(m, last)])))
    print()

print("=== OVERVIEW: all solved matrices ===")
print(f"{'matrix':<22} {'N':>7} {'iters':>6} {'status':<14} {'T_lo':>5} {'ms@lo':>11} {'T_hi':>5} {'ms@hi':>11}")
for (m, n, it, st, tlo, mlo, thi, mhi) in overview:
    print(f"{m:<22} {n:>7} {it:>6} {st:<14} {tlo:>5} {mlo:>11.3f} {thi:>5} {mhi:>11.3f}")

if skipped:
    print()
    print("=== NOT SOLVED (skipped or broke down) ===")
    seen = set()
    for s in skipped:
        if s["matrix"] in seen:
            continue
        seen.add(s["matrix"])
        print(f"{s['matrix']:<22} {s['reason']}")

print()
print("speedup = min time at lowest thread count / min time at N threads")
print("status NO_CONVERGE / BAD usually means the matrix is not symmetric positive definite.")
PYEOF

echo
echo "== Part A done. Results in: $OUT  (raw.csv summary.txt not_solved.csv config.txt) =="

if [ "${SKIP_PERF:-0}" = "1" ]; then
    echo "SKIP_PERF=1: skipping perf stages."
    exit 0
fi

# ============================================================================
# PART B: perf analysis on one representative matrix
# ============================================================================
if ! command -v perf > /dev/null; then
    echo "perf not found in PATH (try: module load perf, or check your cluster modules). Skipping Part B."
    exit 0
fi

# Pick the representative matrix: largest N among matrices that converged.
if [ -z "${REP_MATRIX:-}" ]; then
    REP_MATRIX=$(python3 - "$CSV" << 'PYEOF'
import csv, sys
best = None
for r in csv.DictReader(open(sys.argv[1])):
    try:
        ok = int(r["iterations"]) < 5000 and float(r["rel_residual"]) <= 1e-4
        n = int(r["N"])
    except ValueError:
        continue
    if ok and (best is None or n > best[0]):
        best = (n, r["matrix"])
print(best[1] if best else "")
PYEOF
)
fi
if [ -z "$REP_MATRIX" ]; then
    echo "No converged matrix found for perf analysis. Skipping Part B."
    exit 0
fi

REPFILE=""
for f in "${FILES[@]}"; do
    label=$(basename "$f"); label=${label%.gz}; label=${label%.mtx}
    if [ "$label" = "$REP_MATRIX" ]; then REPFILE="$f"; break; fi
done
if [ -z "$REPFILE" ]; then
    echo "REP_MATRIX=$REP_MATRIX not found among the scanned files. Skipping Part B."
    exit 0
fi

NT_MAX=$MAXT
echo "== Part B: perf analysis on $REP_MATRIX ($REPFILE), NT_MAX=$NT_MAX =="
echo "$REP_MATRIX" > "$PERF/rep_matrix.txt"

PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 99)
cd "$WORK" || exit 1

# ---------------- Stage 0: environment -------------------------------------
{
  date; hostname
  echo "perf_event_paranoid = $PARANOID"
  which perf; perf --version
  echo "nproc = $(nproc)"
  lscpu
  numactl -H
} > "$PERF/00_env.txt" 2>&1
perf list > "$PERF/00_perf_list.txt" 2>&1
perf stat -e cycles,instructions true 2> "$PERF/00_hwcheck.txt"
echo "== env written to $PERF/00_env.txt (paranoid=$PARANOID) =="
cat "$PERF/00_hwcheck.txt"   # "<not supported>" here means no HW counters

export OMP_NUM_THREADS=$NT_MAX

# ---------------- Stage 2: perf stat, 1 thread vs N threads ----------------
# task-clock should stay ~constant from 1 -> N threads. If it grows, threads
# are spinning / waiting on memory instead of doing useful work.
echo "== Stage 2: perf stat =="
BASE=task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions
OMP_NUM_THREADS=1 perf stat -r "$REPS" -e $BASE -o "$PERF/02_stat_1thread.txt" \
    -- "$EXE" "$REPFILE" > /dev/null 2>&1
perf stat -r "$REPS" -e $BASE -o "$PERF/02_stat_Nthreads.txt" \
    -- "$EXE" "$REPFILE" > /dev/null 2>&1

# Arm Neoverse V2 events (check names against 00_perf_list.txt).
# Few events per run so counters are not multiplexed.
perf stat -r "$REPS" -e cycles,instructions,stall_frontend,stall_backend \
    -o "$PERF/02_stat_stalls.txt" -- "$EXE" "$REPFILE" > /dev/null 2>&1
perf stat -r "$REPS" -e l1d_cache,l1d_cache_refill,l2d_cache,l2d_cache_refill \
    -o "$PERF/02_stat_cache.txt"  -- "$EXE" "$REPFILE" > /dev/null 2>&1
perf stat -r "$REPS" -e ll_cache_rd,ll_cache_miss_rd,bus_access,dtlb_walk \
    -o "$PERF/02_stat_mem.txt"    -- "$EXE" "$REPFILE" > /dev/null 2>&1

# ---------------- Stage 3: per-thread counts (imbalance / starvation) ------
echo "== Stage 3: per-thread =="
perf stat --per-thread -e task-clock,cycles,instructions \
    -o "$PERF/03_per_thread.txt" -- "$EXE" "$REPFILE" > /dev/null 2>&1

# ---------------- Stage 4: passive waiting, make blocking visible ----------
# Default libgomp SPINS at barriers, so waiting threads look "busy".
# With passive policy they sleep, so waiting shows up as context switches
# and as lower "CPUs utilized".
echo "== Stage 4: spin vs passive wait policy =="
/usr/bin/time -v "$EXE" "$REPFILE" > "$PERF/04_spin.out" 2> "$PERF/04_spin.time"
OMP_WAIT_POLICY=passive /usr/bin/time -v "$EXE" "$REPFILE" \
    > "$PERF/04_passive.out" 2> "$PERF/04_passive.time"
OMP_WAIT_POLICY=passive perf stat -e task-clock,context-switches,cpu-migrations \
    -o "$PERF/04_passive_stat.txt" -- "$EXE" "$REPFILE" > /dev/null 2>&1
grep -E 'Elapsed|Percent of CPU|context switches' "$PERF/04_passive.time" \
    > "$PERF/04_passive_summary.txt"
grep -E 'Elapsed|Percent of CPU|context switches' "$PERF/04_spin.time" \
    > "$PERF/04_spin_summary.txt"
echo "spin solve:    $(awk '/Solve Time/{print $3, $4}' "$PERF/04_spin.out")"
echo "passive solve: $(awk '/Solve Time/{print $3, $4}' "$PERF/04_passive.out")"

# ---------------- Stage 5: where on-CPU time goes --------------------------
echo "== Stage 5: perf record =="
D="$PERF/perf.data"
perf record -F 499 --call-graph fp -o "$D" -- "$EXE" "$REPFILE" \
    > "$PERF/05_record.out" 2>&1
perf report -i "$D" --stdio --no-children --sort dso,sym -g none \
    --percent-limit 0.3 > "$PERF/05_report_symbols.txt" 2>&1
perf report -i "$D" --stdio --no-children --sort tid -g none \
    > "$PERF/05_report_per_thread.txt" 2>&1
perf report -i "$D" --stdio --no-children --sort sym -g caller \
    --percent-limit 1 > "$PERF/05_report_callers.txt" 2>&1
perf annotate -i "$D" --stdio 2> /dev/null | head -n 600 \
    > "$PERF/05_annotate_top.txt"
# Share of samples spent waiting in the OpenMP runtime (barrier spin/futex):
grep -E 'libgomp|gomp_|do_spin|futex|wait' "$PERF/05_report_symbols.txt" \
    > "$PERF/05_openmp_wait_share.txt"

# ---------------- Stage 6: where threads BLOCK (needs paranoid <= 1) -------
if [ "$PARANOID" -le 1 ]; then
  echo "== Stage 6: context-switch stacks =="
  OMP_WAIT_POLICY=passive perf record -e context-switches -c 100 \
      --call-graph dwarf -o "$PERF/perf_cs.data" -- "$EXE" "$REPFILE" \
      > /dev/null 2>&1
  perf report -i "$PERF/perf_cs.data" --stdio --no-children \
      > "$PERF/06_ctx_switch_stacks.txt" 2>&1
else
  echo "== Stage 6 skipped: perf_event_paranoid=$PARANOID (>1); use stage 4 =="
fi

# ---------------- Stage 7: NUMA first-touch + pinning check ---------------
# NOTE: if the solve finishes in under NUMA_DELAY seconds, numastat will see
# no process. Lower NUMA_DELAY for small matrices.
echo "== Stage 7: NUMA / affinity =="
OMP_DISPLAY_AFFINITY=TRUE "$EXE" "$REPFILE" > "$PERF/07_affinity.out" 2>&1 &
PID=$!
sleep "$NUMA_DELAY"
if command -v numastat > /dev/null; then
  numastat -p $PID > "$PERF/07_numastat.txt" 2>&1   # memory split across nodes
fi
wait $PID

echo
echo "== Done. Benchmark in $OUT, perf in $PERF =="
echo "-- 1 thread vs N threads (task-clock, CPUs utilized, IPC) --"
grep -E 'task-clock|CPUs utilized|insn per cycle|elapsed' \
    "$PERF/02_stat_1thread.txt" "$PERF/02_stat_Nthreads.txt"