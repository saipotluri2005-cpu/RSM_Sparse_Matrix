#!/bin/bash
#SBATCH -J gpu_cg_bench
#SBATCH -o gpu_cg_bench.%j.txt
#SBATCH -p gh                  # Grace-Hopper: 72 Grace cores + 1 H200, NVLink-C2C
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 01:00:00
#SBATCH -A TRA25006

# ---------------------------------------------------------------------------
# GPU CG benchmark on one GH200 node:
#   1. builds gpu_cg, cpu_sparse_v2 and the logic test, runs the test
#   2. generates 3D Poisson matrices (FEM-like, 7 nonzeros/row)
#   3. runs every way of placing the matrix (GPU_MATRIX), both SpMV kernels,
#      CUDA graphs on/off and several CHECK_EVERY values
#   4. runs cpu_sparse_v2 on all 72 Grace cores of the same node for reference
#   5. times the whole folder with PIPELINE=1 vs 0 (CPU prep overlapped or not)
#
#   sbatch gpu_bench.slurm
#   SIZES="64 128" REPS=5 sbatch gpu_bench.slurm
#
# Output: gpu_bench_<jobid>/results.csv and a table at the end of the log.
# Generated with assistance of claude code
# ---------------------------------------------------------------------------
set -u
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT"
SIZES="${SIZES:-64 100 128 160}"
REPS="${REPS:-3}"
OUT="gpu_bench_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

module unload nvidia/24.7 > /dev/null 2>&1
module load gcc/14.2.0 cuda/13.3
module load python3 > /dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv

# ---- 1. build + logic test ------------------------------------------------
make gpu_cg cpu_sparse_v2 test_gpu_logic || exit 1
./test_gpu_logic | tail -3 || exit 1

# ---- 2. matrices -----------------------------------------------------------
MDIR="$ROOT/gpu_bench_matrices"
mkdir -p "$MDIR"
for s in $SIZES; do
    [ -f "$MDIR/poisson3d_$s.mtx" ] || (cd "$MDIR" && python3 "$ROOT/gen_poisson.py" --3d "$s")
done
ls -la "$MDIR"

# ---- helpers ---------------------------------------------------------------
CSV="$OUT/results.csv"
echo "config,matrix,N,nnz,iters,transfer_ms,solve_ms,ms_per_iter,GBps,d2h_ms,rel_res" > "$CSV"
export SAVE_SOLUTION=0
field() { awk -v k="$1" -v c="$2" 'index($0,k)==1{print $c; exit}' <<< "$3"; }
run() { # run <config-name> <command...>   (env vars set by caller)
    local name="$1"; shift
    for f in "$MDIR"/poisson3d_*.mtx; do
        for r in $(seq "$REPS"); do
            out=$("$@" "$f" 2>&1)
            echo "$out" > "$OUT/${name}_$(basename "$f" .mtx)_r$r.log"
            echo "$name,$(basename "$f" .mtx),$(field 'Matrix Size' 3 "$out"),$(field 'Nonzeros' 2 "$out"),$(field 'Iterations Taken' 3 "$out"),$(field 'Transfer Time' 3 "$out"),$(field 'Solve Time' 3 "$out"),$(field 'Time per Iteration' 4 "$out"),$(field 'Bandwidth Achieved' 3 "$out"),$(field 'Result Copy' 3 "$out"),$(field 'Relative Residual' 3 "$out")" >> "$CSV"
        done
    done
}

# ---- 3. GPU configurations -------------------------------------------------
for M in hbm managed host; do
    for S in cusparse fused; do
        GPU_MATRIX=$M SPMV=$S run "gpu_${M}_${S}" ./gpu_cg
    done
done
GPU_MATRIX=hbm SPMV=fused STAGE=pinned run gpu_hbm_fused_pinned ./gpu_cg
GPU_MATRIX=hbm SPMV=fused CUDA_GRAPH=0 run gpu_hbm_fused_nograph ./gpu_cg
GPU_MATRIX=hbm SPMV=cusparse CUDA_GRAPH=0 run gpu_hbm_cusparse_nograph ./gpu_cg
for C in 1 4 64; do
    GPU_MATRIX=hbm SPMV=fused CHECK_EVERY=$C run "gpu_hbm_fused_check$C" ./gpu_cg
done
GPU_MATRIX=hbm SPMV=fused PRECOND=jacobi run gpu_hbm_fused_jacobi ./gpu_cg

# ---- 4. CPU reference on the same node (72 Grace cores) -------------------
OMP_NUM_THREADS=72 OMP_PLACES=cores OMP_PROC_BIND=close run cpu_v2_72t ./cpu_sparse_v2

# ---- 5. pipeline: whole folder, CPU prep overlapped with GPU solve or not --
for P in 0 1; do
    t0=$(date +%s.%N)
    PIPELINE=$P GPU_MATRIX=hbm SPMV=fused ./gpu_cg "$MDIR" > "$OUT/pipeline_$P.log" 2>&1
    t1=$(date +%s.%N)
    echo "PIPELINE=$P  whole folder: $(awk "BEGIN{printf \"%.2f\", $t1 - $t0}") s"
done

# ---- summary: median per (config, matrix) ----------------------------------
echo
echo "Median over $REPS runs (times in ms; GBps = estimated memory bandwidth of the solve)"
python3 - "$CSV" <<'EOF'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
groups = {}
for r in rows:
    groups.setdefault((r["config"], r["matrix"]), []).append(r)
def med(rs, k):
    v = [float(r[k]) for r in rs if r[k] not in ("", None)]
    return st.median(v) if v else float("nan")
cpu = {m: med(rs, "solve_ms") for (c, m), rs in groups.items() if c.startswith("cpu")}
print(f"{'config':28s} {'matrix':14s} {'iters':>6s} {'xfer':>8s} {'solve':>9s} {'ms/it':>8s} {'GB/s':>7s} {'d2h':>7s} {'vs CPU':>7s} {'res':>9s}")
for (c, m), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    s = med(rs, "solve_ms")
    sp = cpu.get(m, float("nan")) / s if s else float("nan")
    print(f"{c:28s} {m:14s} {rs[0]['iters']:>6s} {med(rs,'transfer_ms'):8.2f} {s:9.2f} "
          f"{med(rs,'ms_per_iter'):8.4f} {med(rs,'GBps'):7.0f} {med(rs,'d2h_ms'):7.2f} {sp:6.1f}x {rs[0]['rel_res']:>9s}")
EOF
echo "Full CSV: $CSV"
