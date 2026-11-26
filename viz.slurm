#!/bin/bash
#SBATCH -J fem_viz
#SBATCH -o fem_viz.%j.txt
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 00:30:00
#SBATCH -A TRA25006

# ---------------------------------------------------------------------------
# FEM visualization: generate -> solve on the GPU -> render a video.
#
#   sbatch viz.slurm
#
#   1. viz/gen_bridge.py writes truss bridges to viz_cases/ (matrix, loads, mesh)
#   2. gpu_cg solves K u = b on the H200 (RHS=file reads <name>_b.mtx) and saves u
#   3. viz/render_bridge.py checks u (residual + direct solve) and makes the MP4
#      (a GIF if ffmpeg is missing). If the Python libraries are missing on the
#      node, step 3 prints the command to run on your laptop instead.
#
# Output: viz_out/<name>.mp4 (or .gif) and .png, plus the GPU report per bridge.
# Generated with assistance of claude code
# ---------------------------------------------------------------------------
set -u
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT"
module unload nvidia/24.7 > /dev/null 2>&1
module load gcc/14.2.0 cuda/13.3
module load python3 > /dev/null 2>&1

make gpu_cg || exit 1
mkdir -p viz_cases viz_out

# ---- 1. generate ------------------------------------------------------------
python3 viz/gen_bridge.py --panels 12 --truck 400e3 --truck-pos 0.33 --out viz_cases/bridge || exit 1
python3 viz/gen_bridge.py --panels 30 --truck 600e3 --truck-pos 0.5 --out viz_cases/bridge_long || exit 1

# ---- 2. solve on the GPU ----------------------------------------------------
# tight tolerance: the picture shows displacements, not just a small residual
RHS=file SAVE_SOLUTION=1 SOLUTION_DIR=viz_out PRECOND=jacobi TOL=1e-10 MAX_ITERS=200000 \
    ./gpu_cg viz_cases | tee viz_out/gpu_report.txt

# ---- 3. render --------------------------------------------------------------
if python3 -c "import numpy, scipy, matplotlib" 2> /dev/null; then
    for name in bridge bridge_long; do
        python3 viz/render_bridge.py viz_cases/${name}_mesh.txt viz_out/${name}_gpu_solution.txt \
            --matrix viz_cases/${name}.mtx --out viz_out/${name}.mp4
    done
    ls -la viz_out
else
    echo "numpy/scipy/matplotlib not available here. Copy viz_cases/ and viz_out/ to your laptop and run:"
    echo "  python3 viz/render_bridge.py viz_cases/bridge_mesh.txt viz_out/bridge_gpu_solution.txt --matrix viz_cases/bridge.mtx --out bridge.mp4"
fi
