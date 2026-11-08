#!/bin/bash
#SBATCH -J cg_bench_all
#SBATCH -o cg_bench_all.%j.txt
#SBATCH -p gg                  # Grace-Grace (144 cores). Use gh for 72-core Grace-Hopper.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TRA25006

# ---------------------------------------------------------------------------
# Benchmark dense CG (cpu_only) against sparse CSR CG (cpu_sparse) on every
# matrix in the test_matrices branch, across a thread sweep, then plot both.
#
#   make                     builds cpu_only and cpu_sparse
#   bash benchmark.sh        light defaults (login node)
#   sbatch benchmark.sh      full defaults (compute node)
#
# Test matrices come from the test_matrices branch WITHOUT switching your
# branch: the script checks that branch out into a separate git worktree
# (../RSM_test_matrices) and reads the matrices from there.
#
# Optional settings (environment variables):
#   TEST_BRANCH=test_matrices  branch holding the test matrices
#   TEST_DIR=path              use this folder instead of the branch
#   IMPLS="cpu_only cpu_sparse"  programs to compare
#   POISSON_SIZES="30 60"      also generate and run these Poisson grids
#   THREADS="1 2 4 8"          thread counts (default: powers of two up to cores)
#   REPS=3                     repeats per (program, matrix, threads)
#   PERF=1                     also run one perf stat per (program, matrix, threads)
#   MAX_DENSE_GB=100           cpu_only skips matrices needing more than this
#   REORDER=none PRECOND=jacobi  passed through to cpu_sparse
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
IMPLS="${IMPLS:-cpu_only cpu_sparse}"
GEN="${GEN:-$ROOT/gen_poisson.py}"
POISSON_SIZES="${POISSON_SIZES:-}"
TEST_BRANCH="${TEST_BRANCH:-test_matrices}"
export SAVE_SOLUTION=0         # cpu_sparse: don't write a solution file per run

for impl in $IMPLS; do
    [ -x "$ROOT/$impl" ] || { echo "ERROR: $ROOT/$impl not found. Run make first."; exit 1; }
done

# ---- Test matrices from the test_matrices branch (git worktree) -----------
if [ -n "${TEST_DIR:-}" ]; then
    MATRICES="$TEST_DIR"
else
    WT="$(dirname "$ROOT")/RSM_test_matrices"
    git -C "$ROOT" fetch -q origin "$TEST_BRANCH" 2> /dev/null   # newest copy, if a remote exists
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
OUT="$ROOT/benchall_${SLURM_JOB_ID:-$STAMP}"
WORK="$OUT/work"
mkdir -p "$OUT" "$WORK"
CSV="$OUT/raw.csv"
SKIPPED="$OUT/not_solved.csv"
echo "impl,matrix,N,nnz,threads,rep,iterations,solve_ms,rel_residual" > "$CSV"
echo "impl,matrix,reason" > "$SKIPPED"

{
    echo "date:      $(date)"
    echo "host:      $HOST   cores: $NCORES"
    echo "programs:  $IMPLS"
    echo "matrices:  $MATRICES  (branch $TEST_BRANCH @ $(git -C "$MATRICES" rev-parse --short HEAD 2> /dev/null || echo n/a))"
    echo "threads:   $THREADS"
    echo "reps:      $REPS"
    echo "files:     ${#FILES[@]}"
    printf '  %s\n' "${FILES[@]}"
    echo "MAX_DENSE_GB=$MAX_DENSE_GB OMP_PLACES=$OMP_PLACES OMP_PROC_BIND=$OMP_PROC_BIND"
    echo "REORDER=${REORDER:-rcm} PRECOND=${PRECOND:-none}"
} | tee "$OUT/config.txt"
echo

# ---- Run --------------------------------------------------------------------
cd "$WORK" || exit 1

for f in "${FILES[@]}"; do
    label=$(basename "$f"); label=${label%.gz}; label=${label%.mtx}
    for impl in $IMPLS; do
        EXE="$ROOT/$impl"
        echo "---- $label  [$impl] ----"
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
                    echo "$impl,$label,${reason:-unknown}" >> "$SKIPPED"
                    skip=1
                    break
                fi
                N=$(awk '/Matrix Size/{print $3}' <<< "$out")
                nnz=$(awk '/Nonzeros/{print $2}' <<< "$out")
                iters=$(awk '/Iterations Taken/{print $3}' <<< "$out")
                res=$(awk '/Relative Residual/{print $3}' <<< "$out")
                echo "$impl,$label,$N,$nnz,$t,$rep,$iters,$ms,$res" >> "$CSV"
                printf "  threads=%-4s rep=%s  N=%-7s iters=%-5s solve=%-10s ms  resid=%s\n" \
                       "$t" "$rep" "$N" "$iters" "$ms" "$res"
            done
            if [ "$skip" = 0 ] && [ "${PERF:-0}" = "1" ] && command -v perf > /dev/null; then
                OMP_NUM_THREADS=$t perf stat \
                    -e task-clock,context-switches,cpu-migrations,cycles,instructions \
                    -o "$OUT/perf_${impl}_${label}_t${t}.txt" -- "$EXE" "$f" > /dev/null 2>&1
            fi
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
    k = (r["impl"], r["matrix"], int(r["threads"]))
    times[k].append(float(r["solve_ms"]))
    iters[k].add(r["iterations"])
    try:
        resid[k] = max(resid[k], float(r["rel_residual"]))
    except ValueError:
        resid[k] = float("nan")
    size[(r["impl"], r["matrix"])] = r["N"]

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

best = {}   # (impl, matrix) -> min ms at the highest thread count
for (impl, m) in size:
    ts = sorted(t for (i, mm, t) in times if i == impl and mm == m)
    base = min(times[(impl, m, ts[0])])
    print(f"=== {m}  [{impl}]  ({size[(impl, m)]} x {size[(impl, m)]}) ===")
    print(f"{'threads':>7} {'min ms':>11} {'median ms':>11} {'speedup':>8} {'effic':>7} {'iters':>8} {'max resid':>11}  status")
    for t in ts:
        v = times[(impl, m, t)]
        mn, md = min(v), statistics.median(v)
        sp = base / mn
        eff = sp / (t / ts[0])
        it = ",".join(sorted(iters[(impl, m, t)]))
        print(f"{t:>7} {mn:>11.3f} {md:>11.3f} {sp:>7.2f}x {eff*100:>6.0f}% {it:>8} {resid[(impl, m, t)]:>11.2e}  {status((impl, m, t))}")
    best[(impl, m)] = (ts[-1], min(times[(impl, m, ts[-1])]), status((impl, m, ts[-1])))
    print()

impls = sorted({i for (i, _) in size})
mats = list(OrderedDict.fromkeys(m for (_, m) in size))
print("=== OVERVIEW: best time at the highest thread count ===")
head = f"{'matrix':<24} {'N':>8}" + "".join(f" {i + ' ms':>16} {'status':<12}" for i in impls)
if "cpu_only" in impls and "cpu_sparse" in impls:
    head += f" {'sparse speedup':>15}"
print(head)
for m in mats:
    n = next(size[(i, m)] for i in impls if (i, m) in size)
    line = f"{m:<24} {n:>8}"
    for i in impls:
        if (i, m) in best:
            line += f" {best[(i, m)][1]:>16.3f} {best[(i, m)][2]:<12}"
        else:
            line += f" {'skipped':>16} {'':<12}"
    if ("cpu_only", m) in best and ("cpu_sparse", m) in best:
        line += f" {best[('cpu_only', m)][1] / best[('cpu_sparse', m)][1]:>14.1f}x"
    print(line)

if skipped:
    print()
    print("=== NOT SOLVED (skipped or broke down) ===")
    seen = set()
    for s in skipped:
        key = (s["impl"], s["matrix"])
        if key in seen:
            continue
        seen.add(key)
        print(f"{s['matrix']:<24} [{s['impl']}] {s['reason']}")

print()
print("speedup = min time at lowest thread count / min time at N threads")
print("sparse speedup = cpu_only time / cpu_sparse time, both at the highest thread count")
print("status NO_CONVERGE / BAD usually means the matrix is not symmetric positive definite.")
PYEOF

# ---- Plots ------------------------------------------------------------------
if python3 -c "import matplotlib" 2> /dev/null; then
    python3 "$ROOT/plot_compare.py" "$CSV" "$OUT/plots"
else
    echo "matplotlib not found: skipping plots (module load python3, or pip install --user matplotlib)"
fi

echo
echo "Results in: $OUT"
echo "  raw.csv  summary.txt  not_solved.csv  config.txt  plots/"
