#!/bin/bash
#SBATCH --job-name=vista-memory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=vista-memory-%j.log
# Submit with --partition=<verified-GH-partition> --account=<your-allocation>.
# Before submission, load site modules matching the compiler used to build.
# Run from the repository root. No job is submitted by this script itself.
set -euo pipefail
module unload nvidia/24.7
module load gcc/14.2.0
module load cuda/13.3
module list
nvcc --version
nvidia-smi
export OMP_NUM_THREADS=1
build_dir="${VISTA_BUILD_DIR:-benchmarks/vista_memory/build}"
results_dir="${VISTA_RESULTS_DIR:-benchmarks/vista_memory/results-${SLURM_JOB_ID}}"
mkdir -p "$results_dir"
"$build_dir/transfer" "$results_dir/transfer.csv" 30
"$build_dir/memory" "$results_dir/memory.csv" 30
VISTA_REPS=50 "$build_dir/spmv" "$results_dir" \
  asymmetric_matrices/small/bfw62a.mtx \
  asymmetric_matrices/large/memplus.mtx \
  asymmetric_matrices/large/af23560.mtx
