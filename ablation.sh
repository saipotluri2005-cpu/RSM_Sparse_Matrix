#!/bin/bash
#SBATCH -J cg_ablation
#SBATCH -o cg_ablation.%j.txt
#SBATCH -p gg                  # Grace-Grace, 144 cores (2 sockets x 72)
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -t 03:00:00
#SBATCH -A TRA25006

# ---------------------------------------------------------------------------
# Ablation: measure each scaling fix on its own (docs/RESULTS.md section 3).
#
# Each configuration adds ONE change to the one before it, so the difference
# between neighbours is the effect of that change alone:
#
#   v1_close      cpu_sparse (v1) as it is, threads packed onto socket 0 first
#   v1_spread     v1, threads spread over both sockets           (placement)
#   v2_sync       v2: padded per-thread sums, 3 barriers/iter    (sync fix)
#   v2_touch      + parallel first-touch of matrix and vectors   (NUMA fix)
#   v2_balance    + rows split by nonzeros, not row count        (balance)
#   v2_grain      + at most one thread per 16384 nonzeros        (small sizes)
#
#   make && sbatch ablation.sh
#
# Settings (environment variables):
#   SIZES="120 500 1000 2000"   Poisson grid sizes n (N = n^2)
#   MATRICES="a.mtx b.mtx.gz"   extra matrix files (SPD) to include
#   THREADS="1 2 4 ... 144"     thread counts
#   REPS=3                      repeats per point (plots use the median)
#   CONFIGS="v1_close v2_sync"  run only some configurations
# The 1-thread point is run only for v1_close (all configurations are the
# same code path at 1 thread); it is the baseline for every speedup.
# ---------------------------------------------------------------------------

set -u
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT" || exit 1
module load python3 > /dev/null 2>&1   # for matplotlib, if the module exists

for p in cpu_sparse cpu_sparse_v2; do
    [ -x "$ROOT/$p" ] || { echo "ERROR: $ROOT/$p not found. Run make first."; exit 1; }
done

NCORES=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
SIZES="${SIZES:-120 500 1000 2000}"
REPS="${REPS:-3}"
if [ -z "${THREADS:-}" ]; then
    THREADS=""
    for t in 1 2 4 8 16 32 64 72 128 144; do [ "$t" -le "$NCORES" ] && THREADS="$THREADS $t"; done
fi

# name | program | environment
ALL_CONFIGS=(
    "v1_close|cpu_sparse|OMP_PROC_BIND=close"
    "v1_spread|cpu_sparse|OMP_PROC_BIND=spread"
    "v2_sync|cpu_sparse_v2|OMP_PROC_BIND=spread FIRST_TOUCH=0 BALANCE=rows THREAD_GRAIN=0"
    "v2_touch|cpu_sparse_v2|OMP_PROC_BIND=spread FIRST_TOUCH=1 BALANCE=rows THREAD_GRAIN=0"
    "v2_balance|cpu_sparse_v2|OMP_PROC_BIND=spread FIRST_TOUCH=1 BALANCE=nnz THREAD_GRAIN=0"
    "v2_grain|cpu_sparse_v2|OMP_PROC_BIND=spread FIRST_TOUCH=1 BALANCE=nnz THREAD_GRAIN=16384"
)
CONFIGS="${CONFIGS:-}"
RUN=()
for c in "${ALL_CONFIGS[@]}"; do
    name=${c%%|*}
    if [ -z "$CONFIGS" ] || [[ " $CONFIGS " == *" $name "* ]]; then RUN+=("$c"); fi
done

# ---- matrices ---------------------------------------------------------------
GENDIR="$ROOT/bench_matrices"
mkdir -p "$GENDIR"
FILES=()
for n in $SIZES; do
    f="$GENDIR/poisson2d_${n}.mtx"
    [ -f "$f" ] || (cd "$GENDIR" && python3 "$ROOT/gen_poisson.py" "$n")
    FILES+=("$f")
done
for f in ${MATRICES:-}; do FILES+=("$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"); done

# ---- output -----------------------------------------------------------------
OUT="$ROOT/ablation_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
CSV="$OUT/raw.csv"
echo "config,matrix,N,nnz,threads,threads_used,rep,iterations,solve_ms,ms_per_iter,rel_residual" > "$CSV"
{
    echo "date:     $(date)"
    echo "host:     $(hostname -s)   cores: $NCORES"
    echo "threads:  $THREADS"
    echo "reps:     $REPS"
    echo "configs:"
    printf '  %s\n' "${RUN[@]}"
    echo "matrices:"
    printf '  %s\n' "${FILES[@]}"
} | tee "$OUT/config.txt"

export OMP_PLACES=cores SAVE_SOLUTION=0
first=${RUN[0]%%|*}

for f in "${FILES[@]}"; do
    label=$(basename "$f"); label=${label%.gz}; label=${label%.mtx}
    for c in "${RUN[@]}"; do
        IFS='|' read -r name prog envs <<< "$c"
        echo "---- $label  [$name] ----"
        for t in $THREADS; do
            [ "$t" = 1 ] && [ "$name" != "$first" ] && continue
            for rep in $(seq 1 "$REPS"); do
                out=$(env $envs OMP_NUM_THREADS=$t "$ROOT/$prog" "$f" 2>&1)
                ms=$(awk '/Solve Time/{print $3}' <<< "$out")
                if [ -z "$ms" ]; then
                    echo "  FAILED: $(grep -m1 -E 'SKIPPED|Error' <<< "$out")"
                    break 2
                fi
                N=$(awk '/Matrix Size/{print $3}' <<< "$out")
                nnz=$(awk '/Nonzeros/{print $2}' <<< "$out")
                used=$(awk '/^Threads:/{print $2}' <<< "$out")
                it=$(awk '/Iterations Taken/{print $3}' <<< "$out")
                mpi=$(awk '/Time per Iteration/{print $4}' <<< "$out")
                res=$(awk '/Relative Residual/{print $3}' <<< "$out")
                echo "$name,$label,$N,$nnz,$t,$used,$rep,$it,$ms,$mpi,$res" >> "$CSV"
                printf "  threads=%-4s (used %-4s) rep=%s  iters=%-5s  %10.4f ms/iter\n" \
                       "$t" "$used" "$rep" "$it" "$mpi"
            done
        done
    done
done

echo
if python3 -c "import matplotlib" 2> /dev/null; then
    python3 "$ROOT/plot_ablation.py" "$CSV" "$OUT/plots"
else
    echo "matplotlib not found on this node: plot on your laptop with"
    echo "  python plot_ablation.py $(basename "$OUT")/raw.csv"
fi
echo "Results in: $OUT"
