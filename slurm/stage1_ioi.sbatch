#!/usr/bin/env bash
#SBATCH --job-name=bitsev-stage1-ioi
#SBATCH --partition=DGX
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
# Stage 1 on IOI in a Hugging Face model with the Sutter-style adversary (A100).
# One job per seed (about 6 h each on an A100; records share the output directory):
#   for k in 0 1 2; do sbatch slurm/stage1_ioi.sbatch --model gpt2 --seeds 1 --seed-offset $k --out results/stage1_ioi_gpt2_v2; done
source "${BITSEV_ROOT:-$HOME/codes/description_length}/slurm/_lib.sh"
export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bitsev-stage1-ioi "$@"
