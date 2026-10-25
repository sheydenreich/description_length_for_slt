#!/usr/bin/env bash
# Smoke test on the current (THIN) node: environment check and the test suite.
# Allowed to run directly; everything heavier goes through sbatch.
source "${BITSEV_ROOT:-$HOME/codes/description_length}/slurm/_lib.sh"
bitsev-check-env
pytest -q
