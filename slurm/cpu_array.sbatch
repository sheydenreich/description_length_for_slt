#!/usr/bin/env bash
#SBATCH --job-name=bitsev-cpu
#SBATCH --partition=EPYC
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=slurm_logs/%x_%A_%a.out
# CPU array job. Usage:
#   sbatch --array=0-$((N-1)) slurm/cpu_array.sbatch TASKFILE
# where line k of TASKFILE is a shell command run by array element k. Use GENOA instead of
# EPYC for memory-hungry tasks: sbatch -p GENOA --mem=64G ...
source "${BITSEV_ROOT:-$HOME/codes/description_length}/slurm/_lib.sh"
TASKFILE="$1"
CMD="$(nth_line "$TASKFILE" "${SLURM_ARRAY_TASK_ID}")"
[ -z "$CMD" ] && { echo "no task ${SLURM_ARRAY_TASK_ID} in $TASKFILE"; exit 0; }
echo "+ $CMD"
eval "$CMD"
