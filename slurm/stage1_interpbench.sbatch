#!/usr/bin/env bash
#SBATCH --job-name=bitsev-stage1-ib
#SBATCH --partition=EPYC
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
# Stage 1 on an InterpBench case. Example:
#   sbatch slurm/stage1_interpbench.sbatch --case 18 --variable freq --d-v 4 --members 4 --out results/stage1_ib18_dv4_mix4
source "${BITSEV_ROOT:-$HOME/codes/description_length}/slurm/_lib.sh"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
bitsev-stage1-interpbench --threads "$SLURM_CPUS_PER_TASK" "$@"
