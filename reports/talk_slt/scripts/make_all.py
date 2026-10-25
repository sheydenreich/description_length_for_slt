"""Regenerate every SLT-talk figure in reports/talk_slt/ (PDF and PNG). Run from any directory:
    source ~/venvs/bitsev/bin/activate && python reports/talk_slt/scripts/make_all.py
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODULES = ["fig_iia_vs_code", "fig_ladder_panels", "fig_kappa_vs_dv", "fig_predicted_vs_measured",
           "fig_fisher_hessian", "fig_bayes_ladder", "fig_pythia_onset", "fig_models", "fig_rrr",
           "fig_oracle_coefficients", "fig_mib", "fig_backup_learner_jc", "fig_gain_decomposition",
           "fig_gain_large_n"]

if __name__ == "__main__":
    for name in MODULES:
        importlib.import_module(name).main()
