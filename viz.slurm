#!/bin/bash
#SBATCH -J fem_viz
#SBATCH -o fem_viz.%j.txt
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 00:45:00
#SBATCH -A TRA25006

# ---------------------------------------------------------------------------
# FEM visualization: generate -> solve on the GPU -> render a video.
#
#   sbatch viz.slurm                        # bridges and heat
#   VIZ=heat HEAT_N=96 sbatch viz.slurm     # only the heat block, finer grid
#
# Bridges (viz/gen_bridge.py --moving): one load case per truck position, each
#   solved by gpu_cg (RHS=file); render_bridge.py blends them as the truck drives.
# Heat (viz/gen_heat.py): the poisson3d-style 7-point matrix with two heat sources;
#   gpu_cg HEAT_STEPS mode solves one implicit time step per CG solve and writes
#   the temperature every step; render_heat.py makes the cutaway animation.
#
# Output in viz_out/: bridge.*, bridge_long.*, heat<N>.* (MP4, or GIF when the
# node's ffmpeg does not run), PNG stills, and the GPU reports.
# Generated with assistance of claude code
# ---------------------------------------------------------------------------
set -u
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT"
VIZ="${VIZ:-bridge heat}"
HEAT_N="${HEAT_N:-64}"
module unload nvidia/24.7 > /dev/null 2>&1
module load gcc/14.2.0 cuda/13.3
module load python3 > /dev/null 2>&1

make gpu_cg || exit 1
mkdir -p viz_out
render_ok=1
python3 -c "import numpy, scipy, matplotlib" 2> /dev/null || render_ok=0

if [[ " $VIZ " == *" bridge "* ]]; then
    rm -rf viz_cases && mkdir -p viz_cases
    python3 viz/gen_bridge.py --moving --panels 12 --truck 400e3 --out viz_cases/bridge || exit 1
    python3 viz/gen_bridge.py --moving --panels 30 --truck 600e3 --out viz_cases/bridge_long || exit 1
    # tight tolerance: the picture shows displacements, not just a small residual
    RHS=file SAVE_SOLUTION=1 SOLUTION_DIR=viz_out PRECOND=jacobi TOL=1e-10 MAX_ITERS=200000 \
        ./gpu_cg viz_cases > viz_out/gpu_report.txt
    grep -E "Processing File|Iterations Taken|Solve Time|Relative Residual" viz_out/gpu_report.txt | tail -8
    if [ $render_ok = 1 ]; then
        for name in bridge bridge_long; do
            python3 viz/render_bridge.py viz_cases/${name}_mesh.txt --solutions viz_out --cases viz_cases \
                --report viz_out/gpu_report.txt --out viz_out/${name}.mp4
        done
    fi
fi

if [[ " $VIZ " == *" heat "* ]]; then
    mkdir -p viz_heat
    H=viz_heat/heat${HEAT_N}
    [ -f $H.mtx ] || python3 viz/gen_heat.py --n $HEAT_N --out $H || exit 1
    DT=$(awk '/^dt/{print $2}' ${H}_mesh.txt)
    STEPS=$(awk '/^steps/{print $2}' ${H}_mesh.txt)
    RHS=file HEAT_STEPS=$STEPS HEAT_DT=$DT FRAME_DIR=viz_out TOL=1e-8 \
        ./gpu_cg $H.mtx | tee viz_out/heat_report.txt
    if [ $render_ok = 1 ]; then
        python3 viz/render_heat.py ${H}_mesh.txt --frames viz_out --report viz_out/heat_report.txt \
            --out viz_out/heat${HEAT_N}.mp4
    fi
fi

if [ $render_ok = 0 ]; then
    echo "numpy/scipy/matplotlib not available: copy viz_cases/, viz_heat/ and viz_out/ to a laptop and run"
    echo "the render_bridge.py / render_heat.py commands from this script there."
fi
ls -la viz_out | grep -vE "_frame_[0-9]+\.bin"
