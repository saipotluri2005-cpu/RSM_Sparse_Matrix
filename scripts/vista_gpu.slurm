#!/bin/bash
#SBATCH -J qa_rcm_gpu
#SBATCH -o vista_qa_gpu.%j.txt
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 01:30:00
#SBATCH -A TRA25006

# ---------------------------------------------------------------------------
# vista_gpu.slurm -- does RCM change GPU SpMV time? One Vista Grace-Hopper node.
#
#   cd $WORK/RSM_Sparse_Matrix          (repo root, branch qa_rcm)
#   git fetch origin                    (so origin/test_matrices exists)
#   sbatch scripts/vista_gpu.slurm
#
# For 3D Poisson grids whose vector x is smaller and larger than the GPU's L2
# cache (n = 100, 160, 230 -> x = 8, 33, 97 MB) and the 1M-unknown elasticity
# matrix, it writes three orderings of each matrix (original, shuffled,
# shuffled then RCM) and times y = A x with Jonathan's cuSPARSE benchmark
# (test_matrices: benchmarks/vista_memory/spmv). The same orderings are also
# timed on this node's Grace CPU with rcm_study (72 threads).
#
# Results: qa_rcm_gpu_<jobid>/ in the repo, packed as ~/qa_rcm_results_gpu_<jobid>.tar.gz
#   env.txt, gpu_info.txt   modules, nvidia-smi, GPU L2 cache size
#   study_gh_72t.csv        orderings: bandwidth, profile, CPU SpMV time
#   spmv/spmv.csv           GPU SpMV time per matrix file (strategy hbm = matrix in GPU memory)
#
# Settings: SIZES (default "100 160 230"), EL (default 70), KEEP=1 keeps matrices.
#
# Generated with assistance of claude code
# ---------------------------------------------------------------------------

set -u
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT" || exit 1
[ -f Makefile ] && [ -d tools ] || { echo "run sbatch from the repo root (branch qa_rcm)"; exit 1; }

JOB="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="$ROOT/qa_rcm_gpu_$JOB"
MAT="${SCRATCH:-$ROOT}/qa_rcm_gpumats_$JOB"
mkdir -p "$OUT" "$MAT"
SIZES="${SIZES:-100 160 230}"
EL="${EL:-70}"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.txt"; }

# Same modules as Jonathan's benchmarks/vista_memory/vista.slurm (job 1134703)
module unload nvidia/24.7 > /dev/null 2>&1
module load gcc/14.2.0 > /dev/null 2>&1
module load cuda/13.3 > /dev/null 2>&1
module load cmake > /dev/null 2>&1
{
    echo "job $JOB on $(hostname)  $(date)"
    echo "git: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
    module list 2>&1
    g++ --version | head -1
    nvcc --version 2>&1 | tail -2
    cmake --version 2>&1 | head -1
    nvidia-smi 2>&1
} > "$OUT/env.txt"

# ---- build ---------------------------------------------------------------------
make clean > /dev/null
make -j 32 tools/gen_fem tools/rcm_study > "$OUT/build.txt" 2>&1 || { log "CPU tools build failed"; exit 1; }
nvcc -O2 tools/gpu_info.cu -o "$OUT/gpu_info" >> "$OUT/build.txt" 2>&1 && "$OUT/gpu_info" > "$OUT/gpu_info.txt" 2>&1
log "GPU: $(tr '\n' ' ' < "$OUT/gpu_info.txt" 2> /dev/null)"

WT="$(dirname "$ROOT")/RSM_test_matrices"
if git rev-parse -q --verify origin/test_matrices > /dev/null; then
    if [ -d "$WT" ]; then git -C "$WT" checkout -q --detach origin/test_matrices
    else git worktree add -q --detach "$WT" origin/test_matrices; fi
fi
JB="$OUT/jb_build"
SPMV=""
if [ -f "$WT/benchmarks/vista_memory/CMakeLists.txt" ] && command -v cmake > /dev/null; then
    cmake -S "$WT/benchmarks/vista_memory" -B "$JB" -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CUDA_ARCHITECTURES=90 >> "$OUT/build.txt" 2>&1 &&
        cmake --build "$JB" -j 32 >> "$OUT/build.txt" 2>&1 && SPMV="$JB/spmv"
fi
[ -x "$SPMV" ] && log "built Jonathan's spmv" || log "could not build Jonathan's spmv (see build.txt); GPU timing skipped"

# ---- matrices: original, shuffled, shuffled + RCM --------------------------------
export OMP_PLACES=cores OMP_PROC_BIND=spread OMP_NUM_THREADS=72
FILES=()
for n in $SIZES; do
    tools/gen_fem poisson3d "$n" -o "$MAT/p3d_$n.mtx" >> "$OUT/gen.txt"
done
tools/gen_fem elasticity "$EL" -o "$MAT/el_$EL.mtx" >> "$OUT/gen.txt"
INPUTS=()
for n in $SIZES; do INPUTS+=("$MAT/p3d_$n.mtx"); done
INPUTS+=("$MAT/el_$EL.mtx")
tools/rcm_study --orderings original,shuffled,shuffled_rcm --write shuffled,shuffled_rcm --reps 10 \
    --outdir "$MAT" --csv "$OUT/study_gh_72t.csv" "${INPUTS[@]}" 2>> "$OUT/study.log"
for f in "${INPUTS[@]}"; do
    s=$(basename "$f" .mtx)
    FILES+=("$f" "$MAT/${s}_shuffled.mtx" "$MAT/${s}_shuffled_rcm.mtx")
done
log "orderings written: $(ls "$MAT" | wc -l) files"

# ---- GPU SpMV ---------------------------------------------------------------------
if [ -n "$SPMV" ]; then
    mkdir -p "$OUT/spmv"
    export OMP_NUM_THREADS=1
    VISTA_REPS=50 "$SPMV" "$OUT/spmv" "${FILES[@]}" > "$OUT/spmv/stdout.txt" 2>&1
    log "GPU spmv exit $?: $(wc -l < "$OUT/spmv/spmv.csv" 2> /dev/null) lines in spmv.csv"
fi

# ---- pack -----------------------------------------------------------------------------
rm -rf "$JB" "$OUT/gpu_info"
[ "${KEEP:-0}" = 1 ] || rm -rf "$MAT"
tar czf "$HOME/qa_rcm_results_gpu_$JOB.tar.gz" -C "$ROOT" "$(basename "$OUT")"
log "packed: $HOME/qa_rcm_results_gpu_$JOB.tar.gz"
