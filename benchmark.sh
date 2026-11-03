#!/bin/bash
#SBATCH -J cg_bench_all
#SBATCH -o cg_bench_all.%j.txt
#SBATCH -p gg                  # Grace-Grace (144 cores). Use gh for 72-core Grace-Hopper.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A YOUR_ALLOCATION

# ---------------------------------------------------------------------------
# Benchmark the OpenMP dense CG solver on EVERY matrix in the Matrices folder
# (.mtx and .mtx.gz, searched recursively), across a thread sweep.
#
#   bash bench_all.sh        light defaults (login node)
#   sbatch bench_all.sh      full defaults (compute node)
#
# Optional settings (environment variables):
#   MATRICES=path          folder to scan            (default: ./Matrices)
#   POISSON_SIZES="30 60"  also generate and run these Poisson grids
#   THREADS="1 2 4 8"      thread counts (default: powers of two up to cores)
#   REPS=3                 repeats per (matrix, threads)
#   PERF=1                 also run one perf stat per (matrix, threads)
#   MAX_DENSE_GB=100       skip matrices needing more dense memory than this
#
# CG only converges on symmetric positive definite matrices. Other matrices
# are still run and timed, but are labeled NO_CONVERGE / BAD in the summary.
# ---------------------------------------------------------------------------

set -u

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    ROOT="$SLURM_SUBMIT_DIR"
else
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
EXE="${EXE:-$ROOT/cpu_only}"
GEN="${GEN:-$ROOT/gen_poisson.py}"
MATRICES="${MATRICES:-$ROOT/Matrices}"
POISSON_SIZES="${POISSON_SIZES:-}"

[ -x "$EXE" ] || { echo "ERROR: $EXE not found. Run make first."; exit 1; }
[ -d "$MATRICES" ] || { echo "ERROR: folder $MATRICES not found."; exit 1; }

# ---- Defaults: gentle on login nodes ---------------------------------------
HOST=$(hostname -s)
NCORES=$(nproc)
if [[ "$HOST" == login* && "${FORCE:-0}" != "1" ]]; then
    echo "NOTE: login node detected ($HOST). Light defaults; nproc reports $NCORES core(s)."
    echo "      Login nodes limit you to very few cores, so a thread sweep needs a compute node."
    MAXT=$(( NCORES < 8 ? NCORES : 8 ))
    export MAX_DENSE_GB="${MAX_DENSE_GB:-2}"
else
    MAXT=$NCORES
    export MAX_DENSE_GB="${MAX_DENSE_GB:-100}"
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
mapfile -t FILES < <(find "$MATRICES" -type f \( -name '*.mtx' -o -name '*.mtx.gz' \) | sort)

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
OUT="$ROOT/benchall_${SLURM_JOB_ID:-$STAMP}"
WORK="$OUT/work"
mkdir -p "$OUT" "$WORK"
CSV="$OUT/raw.csv"
SKIPPED="$OUT/not_solved.csv"
echo "matrix,N,threads,rep,iterations,solve_ms,rel_residual" > "$CSV"
echo "matrix,reason" > "$SKIPPED"

{
    echo "date:      $(date)"
    echo "host:      $HOST   cores: $NCORES"
    echo "threads:   $THREADS"
    echo "reps:      $REPS"
    echo "matrices:  ${#FILES[@]} file(s)"
    printf '  %s\n' "${FILES[@]}"
    echo "MAX_DENSE_GB=$MAX_DENSE_GB OMP_PLACES=$OMP_PLACES OMP_PROC_BIND=$OMP_PROC_BIND"
} | tee "$OUT/config.txt"
echo

# ---- Run --------------------------------------------------------------------
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
        if [ "$skip" = 0 ] && [ "${PERF:-0}" = "1" ] && command -v perf > /dev/null; then
            OMP_NUM_THREADS=$t perf stat \
                -e task-clock,context-switches,cpu-migrations,cycles,instructions \
                -o "$OUT/perf_${label}_t${t}.txt" -- "$EXE" "$f" > /dev/null 2>&1
        fi
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
echo "Results in: $OUT"
echo "  raw.csv  summary.txt  not_solved.csv  config.txt"