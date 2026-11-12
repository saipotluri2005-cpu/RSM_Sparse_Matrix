#!/bin/bash
#SBATCH -J qa_rcm_cpu
#SBATCH -o vista_qa_cpu.%j.txt
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -t 02:30:00
#SBATCH -A TRA25006

# ---------------------------------------------------------------------------
# vista_cpu.slurm -- qa_rcm measurements on one Vista Grace-Grace (gg) node.
#
#   cd $WORK/RSM_Sparse_Matrix          (repo root, branch qa_rcm)
#   git fetch origin                    (so origin/test_matrices exists)
#   curl -fsSLo tools/stream.c https://www.cs.virginia.edu/stream/FTP/Code/stream.c   (optional)
#   sbatch scripts/vista_cpu.slurm
#
# Results: qa_rcm_cpu_<jobid>/ in the repo, packed as ~/qa_rcm_results_cpu_<jobid>.tar.gz
#
#   env.txt         OMP_* variables the environment sets, modules, lscpu, numactl
#   tests.txt       make test on this node
#   rcm_cpu.csv     time per CG iteration: natural vs shuffled numbering, with
#                   and without RCM, 1 / 72 / 144 threads (cpu_sparse_v2, Jacobi)
#   baseline.csv    full solves: 1 thread without reordering (success-criterion
#                   baseline), 1 thread with RCM, 144 threads with RCM; read,
#                   build (incl. RCM) and solve times
#   study_*.csv     rcm_study: bandwidth/profile/SpMV time per ordering, with
#                   spy grids in spy/
#   stream.txt      STREAM triad, if tools/stream.c is present
#   barrier.csv     OpenMP barrier and reduction cost, 2..144 threads, close/spread
#
# Settings: REPS (default 3), ITERS (CG iterations for rcm_cpu, default 200),
#           P2 / P3 / EL grid sizes (defaults 1000 / 100 / 70, about 1M unknowns
#           each), RCM_THREADS (default "1 72 144"), PAR (parallel thread count
#           in baseline.csv, default 144), BAR_THREADS (barrier sweep), KEEP=1
#           keeps the generated matrices in $SCRATCH.
#
# Generated with assistance of claude code
# ---------------------------------------------------------------------------

set -u
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT" || exit 1
[ -f Makefile ] && [ -d tools ] || { echo "run sbatch from the repo root (branch qa_rcm)"; exit 1; }

JOB="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="$ROOT/qa_rcm_cpu_$JOB"
MAT="${SCRATCH:-$ROOT}/qa_rcm_mats_$JOB"
mkdir -p "$OUT/spy" "$MAT"
REPS="${REPS:-3}"
ITERS="${ITERS:-200}"
P2="${P2:-1000}"
P3="${P3:-100}"
EL="${EL:-70}"
RCM_THREADS="${RCM_THREADS:-1 72 144}"
PAR="${PAR:-144}"
BAR_THREADS="${BAR_THREADS:-2 4 8 16 32 64 72 128 144}"
NCORES=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.txt"; }

# ---- 0. environment --------------------------------------------------------
{
    echo "job $JOB on $(hostname)  $(date)"
    echo "git: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
    echo
    echo "## OMP_* / GOMP_* variables set before this script changes anything:"
    env | grep -E '^(OMP|GOMP|KMP)_' || echo "(none)"
    echo
    module list 2>&1
    echo
    g++ --version | head -1
    echo "nproc (unrestricted): $NCORES"
    echo
    lscpu
    echo
    numactl -H 2> /dev/null
} > "$OUT/env.txt"
log "environment saved; OMP variables: $(env | grep -E '^OMP_' | tr '\n' ' ')"

# ---- 1. build and test -----------------------------------------------------
make clean > /dev/null
if ! make -j 32 > "$OUT/build.txt" 2>&1; then log "BUILD FAILED, see build.txt"; exit 1; fi
make test > "$OUT/tests.txt" 2>&1
log "tests: $(tail -1 "$OUT/tests.txt")"

export OMP_PLACES=cores OMP_PROC_BIND=spread SAVE_SOLUTION=0
unset OMP_THREAD_LIMIT OMP_DYNAMIC REORDER PRECOND FIRST_TOUCH BALANCE THREAD_GRAIN

# ---- 2. matrices -----------------------------------------------------------
gen() { [ -f "$MAT/$1.mtx" ] || tools/gen_fem "${@:2}" -o "$MAT/$1.mtx" >> "$OUT/gen.txt"; }
gen p2d_$P2         poisson2d "$P2"
gen p2d_${P2}_shuf  poisson2d "$P2" --shuffle 1
gen p3d_$P3         poisson3d "$P3"
gen p3d_${P3}_shuf  poisson3d "$P3" --shuffle 1
gen el_$EL          elasticity "$EL"
gen el_${EL}_shuf   elasticity "$EL" --shuffle 1
GENERATED="p2d_$P2 p3d_$P3 el_$EL"
log "matrices: $(ls "$MAT" | tr '\n' ' ')"

# parse one cpu_sparse_v2 run into CSV fields
fields() {
    awk '/^Matrix Size:/{n=$3} /^Nonzeros:/{z=$2} /^Bandwidth:/{b0=$2; b1=$4}
         /^Threads:/{t=$2} /^Iterations Taken:/{it=$3} /^Read Time:/{rd=$3}
         /^Build Time:/{bd=$3} /^Solve Time:/{sv=$3} /^Time per Iteration:/{pi=$4}
         /^Relative Residual:/{rr=$3}
         END {printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s", n, z, b0, b1, t, it, rd, bd, sv, pi, rr}' <<< "$1"
}

# ---- 3. RCM on the CPU: time per iteration ---------------------------------
CSV="$OUT/rcm_cpu.csv"
echo "matrix,input,reorder,threads,rep,N,nnz,bw_before,bw_after,threads_used,iterations,read_ms,build_ms,solve_ms,ms_per_iter,rel_residual" > "$CSV"
for m in $GENERATED; do
    for input in natural shuffled; do
        f="$MAT/$m.mtx"; [ $input = shuffled ] && f="$MAT/${m}_shuf.mtx"
        for ro in none rcm; do
            for t in $RCM_THREADS; do
                for rep in $(seq 1 "$REPS"); do
                    out=$(OMP_NUM_THREADS=$t REORDER=$ro PRECOND=jacobi MAX_ITERS=$ITERS ./cpu_sparse_v2 "$f" 2>&1)
                    echo "$m,$input,$ro,$t,$rep,$(fields "$out")" >> "$CSV"
                done
            done
            log "rcm_cpu $m $input REORDER=$ro done"
        done
    done
done

# ---- 4. baseline for the success criterion (full solves) --------------------
CSV="$OUT/baseline.csv"
echo "matrix,input,config,reorder,threads,N,nnz,bw_before,bw_after,threads_used,iterations,read_ms,build_ms,solve_ms,ms_per_iter,rel_residual" > "$CSV"
for m in $GENERATED; do
    for input in natural shuffled; do
        f="$MAT/$m.mtx"; [ $input = shuffled ] && f="$MAT/${m}_shuf.mtx"
        for cfg in "baseline_1t_none|none|1" "cpu_1t_rcm|rcm|1" "cpu_${PAR}t_rcm|rcm|$PAR"; do
            IFS='|' read -r name ro t <<< "$cfg"
            out=$(OMP_NUM_THREADS=$t REORDER=$ro PRECOND=jacobi MAX_ITERS=50000 ./cpu_sparse_v2 "$f" 2>&1)
            echo "$m,$input,$name,$ro,$t,$(fields "$out")" >> "$CSV"
        done
        log "baseline $m $input done"
    done
done

# ---- 5. rcm_study ------------------------------------------------------------
WT="$(dirname "$ROOT")/RSM_test_matrices"
if git rev-parse -q --verify origin/test_matrices > /dev/null; then
    if [ -d "$WT" ]; then git -C "$WT" checkout -q --detach origin/test_matrices
    else git worktree add -q --detach "$WT" origin/test_matrices; fi
fi
if [ -d "$WT/spd_matrices" ]; then
    mapfile -t NIST < <(find "$WT/spd_matrices" -name '*.mtx' | sort)
    OMP_NUM_THREADS=1 tools/rcm_study --reps 20 --spy 150 --outdir "$OUT/spy" \
        --csv "$OUT/study_nist_1t.csv" "${NIST[@]}" 2>> "$OUT/study.log"
    log "rcm_study NIST: ${#NIST[@]} matrices"
else
    log "no spd_matrices folder (git fetch origin on the login node first); NIST study skipped"
fi
for t in 1 $PAR; do
    OMP_NUM_THREADS=$t tools/rcm_study --reps 10 --spy 150 --outdir "$OUT/spy" \
        --csv "$OUT/study_generated_${t}t.csv" "$MAT/p2d_$P2.mtx" "$MAT/p3d_$P3.mtx" "$MAT/el_$EL.mtx" \
        2>> "$OUT/study.log"
    log "rcm_study generated, $t threads"
done

# ---- 6. STREAM ----------------------------------------------------------------
if [ -f tools/stream.c ]; then
    # 125M doubles = 1 GB per array, more than 4x the two chips' L3 caches;
    # fall back to 80M if the static arrays do not link
    ARCH=-mcpu=native
    [ "$(uname -m)" = aarch64 ] || ARCH=-march=native
    for size in 125000000 80000000; do
        gcc -O3 $ARCH -fopenmp -DSTREAM_ARRAY_SIZE=$size -DNTIMES=20 tools/stream.c -o "$OUT/stream" \
            >> "$OUT/stream_build.txt" 2>&1 && break
    done
    if [ -x "$OUT/stream" ]; then
        for cfg in "1|close" "72|close" "72|spread" "144|spread"; do
            IFS='|' read -r t bind <<< "$cfg"
            echo "### OMP_NUM_THREADS=$t OMP_PROC_BIND=$bind" >> "$OUT/stream.txt"
            OMP_NUM_THREADS=$t OMP_PROC_BIND=$bind "$OUT/stream" >> "$OUT/stream.txt" 2>&1
        done
        log "STREAM: $(grep -E '^###|^Triad' "$OUT/stream.txt" | paste - - | awk '{print $2, $3, "triad", $5, "MB/s;"}' | tr '\n' ' ')"
    else
        log "STREAM build failed, see stream_build.txt"
    fi
else
    log "tools/stream.c not present; STREAM skipped"
fi

# ---- 7. barrier cost ------------------------------------------------------------
echo "threads,proc_bind,barrier_ns,reduction_ns" > "$OUT/barrier.csv"
for bind in close spread; do
    for t in $BAR_THREADS; do
        OMP_NUM_THREADS=$t OMP_PROC_BIND=$bind tools/omp_barrier_cost 50000 | tail -1 >> "$OUT/barrier.csv"
    done
done
log "barrier cost done"

# ---- pack ---------------------------------------------------------------------
rm -f "$OUT/stream"
[ "${KEEP:-0}" = 1 ] || rm -rf "$MAT"
tar czf "$HOME/qa_rcm_results_cpu_$JOB.tar.gz" -C "$ROOT" "$(basename "$OUT")"
log "packed: $HOME/qa_rcm_results_cpu_$JOB.tar.gz"
