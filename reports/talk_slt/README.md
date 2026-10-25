# Figures and deck for the singular-learning-theory talk (SLT days, week of 2026-10-26)

## Deck
`slt_days_talk.pptx`: 20 main slides + 13 backup slides (marked "BACKUP"), speaker notes on every slide.
Built by `scripts/build_deck.js` (pptxgenjs; see the header of the script for the command and the Node
environment). Every coefficient is tagged on the slide by its learner: blue pill "plug-in κ (MAP)", orange
pill "Bayes λ". Sources are named on each figure slide.

## Regenerating
`source ~/venvs/bitsev/bin/activate && python reports/talk_slt/scripts/make_all.py` rewrites every figure
below (PDF and PNG) from files under `reports/` only; `scripts/common.py` holds the style, colours and the
text-table parsers (each parser names its source file).

## Figures made for the talk (2026-10-21; script in brackets)
- `iia_vs_code.*` (fig_iia_vs_code.py): IIA against gain over label-only for every IOI configuration on GPT-2,
  Qwen3-1.7B, Qwen2.5-3B, Llama-3.1-8B; manufactured alignments pass an IIA threshold and lose 1700-2000 bits.
- `ladder_panels.*` (fig_ladder_panels.py): the complexity ladder on GPT-2 as four panels: fitted parameters,
  IIA, plug-in κ (0 / 49 / 226 / 988) and gain over label-only for the Jacobian lens, the learned subspace,
  the coupling map on the trained net and the manufactured alignment (replaces `gpt2_complexity_ladder.png`).
- `kappa_vs_dv.*` (fig_kappa_vs_dv.py): κ of the MAP code at d_V 4/16/64 (flat) against the nominal Bayes
  λ = d_V(d − d_V)/2 (1528 → 22528) and the plug-in predictions.
- `predicted_vs_measured.*` (fig_predicted_vs_measured.py): ½ tr(F H⁻¹) against measured κ on the oracle
  (k sweep, n ≤ 16384) and GPT-2 layer 11 (d_V 4, 16), with the nominal λ drawn at the same κ (85× gap at
  d_V 16). Pending (job 1820081): measured κ at layers 10 and 12, the d_V 64 prediction.
- `gpt2_fisher_vs_hessian_slide.*` (fig_fisher_hessian.py): Hessian eigenvalues and held-out Fisher in the
  same directions; accumulated coefficient over directions, Bayes (Laplace) → 6016 vs plug-in → 70.6.
- `bayes_vs_plugin_ladder.*` (fig_bayes_ladder.py): code length / label-only for the Bayes code (HMC) and the
  plug-in mixture against the family's nominal λ: hierarchical equality (56), InterpBench 18 (512), GPT-2
  (6016, Laplace prediction, not run).
- `oracle_kappa_vs_lambda.*`, `oracle_ksweep_slide.*`, `oracle_sharpness_slide.*`
  (fig_oracle_coefficients.py): MAP κ and Bayes λ on the oracle, arrows from n ≤ 2048 to n ≤ 16384, k sweep
  with ½ tr(F H⁺) and sharpness sweep.
- `rrr_exact_check.*` (fig_rrr.py): reduced-rank regression, Bayes slope vs Aoyagi–Watanabe λ, plug-in at MN/2;
  between-family gain slopes vs Δλ.
- `models_verdicts.*` (fig_models.py): valid subspace, coupling, twin and manufactured alignment on four
  language models with the manufactured alignment's IIA.
- `pythia_onset.*` (fig_pythia_onset.py): Pythia-1.4b checkpoints, task accuracy and codes.
- `mib_arithmetic.*` (fig_mib.py): MIB two-digit addition (carry), gain over the trivial floor per layer for
  Llama-3.1-8B and Gemma-2-2b.
- `learner_decides_verdict.*`, `jacobian_consistency.*` (fig_backup_learner_jc.py): backup figures.
- `gain_decomposition.*`, `code_rate_per_block.*` (fig_gain_decomposition.py, 2026-10-22): the gain of the valid
  GPT-2 subspace over label-only, the manufactured alignment and the trained coupling split into the final fits'
  loss difference (≈ n·Δℓ) and the regret difference (Δκ ln n + O(1)), 32768 targets, 3 seeds; and the per-block
  code rate above the channel floor. Against the manufactured alignment the fit term is negative and the whole
  margin is regret. Source: `stage1_ioi/gain_decomposition_gpt2_n32768.json` (reading in the `.txt` next to it).

Learner note: the Llama-3.1-8B and Qwen2.5-3B IOI runs, the Pythia series and the MIB runs used the
single-map MAP code (settings: learner "laplace", 0 posterior samples), Qwen3-1.7B the mixture of four;
the headers of `stage1_ioi/tier2_models.txt` and `stage1_mib/arithmetic.txt` say "mixture-of-4" for all.

## Earlier figures (2026-10-19/20, still used in backup slides)
- `oracle_effective_coefficient.*`: oracle (d = 12) with k of 11 readable directions; MAP-code regret slope
  (`reports/oracle/effective_lambda_sweep.json`) and SGLD local learning coefficient (`reports/oracle/
  llc_check.json`, step 1e-4, pull 10) both track k/2, not the nominal 5.5. (Both are plug-in-side
  quantities; the Bayes coefficient, k/2 + (11 - k)/4, is in `oracle_kappa_vs_lambda.*`.)
- `interpbench_regret_vs_n.*`: regret of the MAP subspace code on InterpBench 18 at the valid and the
  misspecified locus, 1024- and 8192-target runs. Flat below the frame's parameter count (1040), growth at
  about 45 per ln n beyond it; the misspecified locus grows linearly from the excess loss.
- `gpt2_regret_vs_n.*`, `gpt2_hessian_vs_fisher.*`, `gpt2_complexity_ladder.*`, `oracle_bayes_vs_map.*`,
  `oracle_bayes_vs_map_n.*`, `oracle_sharpness_vs_n.*`, `sharp_readout_1d_quadrature.txt`: see
  `reports/oracle/bayes_code_hmc.txt`, `reports/stage1_ioi/curvature_gpt2.txt`, `h_a3_gpt2_large_n.txt`.
  `oracle_bayes_vs_map.*` and `oracle_bayes_vs_map_n.*` (hand-made, no script) are superseded: they draw the
  nominal (d - 1)/2 = 5.5 as the Bayes reference for every k, but the d - 1 - k unreadable directions are
  quartic and the Bayes coefficient is k/2 + (d - 1 - k)/4 (correction of 2026-10-22, end of
  `reports/oracle/bayes_code_hmc.txt`). Use `oracle_kappa_vs_lambda.*` / `oracle_ksweep_slide.*`.
Source tables: `reports/stage1_interpbench/case18_h_a3_n8192.txt`, `case18_llc.txt`,
`reports/stage1_ioi/regret_slopes_map_code.txt`.
