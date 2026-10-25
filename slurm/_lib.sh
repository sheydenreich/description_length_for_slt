# Shared shell helpers for bitsev SLURM scripts. Source this file; do not run it.
set -euo pipefail
export BITSEV_ROOT="${BITSEV_ROOT:-$HOME/codes/description_length}"
export BITSEV_VENV="${BITSEV_VENV:-$HOME/venvs/bitsev}"
export BITSEV_RESULTS="${BITSEV_RESULTS:-$BITSEV_ROOT/results}"
export PYTHONUNBUFFERED=1
# shellcheck disable=SC1091
source "$BITSEV_VENV/bin/activate"
mkdir -p "$BITSEV_RESULTS" "$BITSEV_ROOT/slurm_logs"
cd "$BITSEV_ROOT"
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} python=$(python -V 2>&1)"
# nth_line FILE K: print line K (0-based) of FILE, empty if past the end.
nth_line() { sed -n "$(( $2 + 1 ))p" "$1"; }
export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
