#!/usr/bin/env bash
#SBATCH --job-name=bitsev-stage1-toy
#SBATCH --partition=EPYC
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
# Stage 1 on hierarchical equality with the H-A2 trajectory. Resumable: finished
# (configuration, seed) records are skipped. Extra arguments are passed through, e.g.
#   sbatch slurm/stage1_toy.sbatch --seeds 5 --growth 1.5 --out results/stage1_toy_r1.5
source "${BITSEV_ROOT:-$HOME/codes/description_length}/slurm/_lib.sh"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
bitsev-stage1-toy --trajectory --threads "$SLURM_CPUS_PER_TASK" "$@"
