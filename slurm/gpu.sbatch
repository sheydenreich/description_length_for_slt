#!/usr/bin/env bash
#SBATCH --job-name=bitsev-gpu
#SBATCH --partition=DGX
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
# Single-GPU job on an A100. Usage: sbatch slurm/gpu.sbatch <command ...>
# Use -p H100 --gres=gpu:H100:1 only if an A100 is not sufficient (the GPU type is
# mandatory on this cluster). The GPU (V100) partition is not usable
# with the cu130 torch build in the bitsev env (no sm_70 kernels).
# Jobs must checkpoint per prequential block so that a resubmission resumes.
source "${BITSEV_ROOT:-$HOME/codes/description_length}/slurm/_lib.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "+ $*"
"$@"
