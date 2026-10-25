# Implementation plan: Stage 1 of "Bits of Evidence", weeks 0–6

Scope: everything the research plan (`research_plan_synthesis_v7_2.tex`) needs up to and
including the two decision gates G-A0 (end of week 2) and G-A1 (end of week 6, mid-November
2026). Month 0 is October 2026. Later stages (Stage 2, Project B, stretch goals) are out of
scope here but the architecture below is designed so they slot in without rework.

Code priorities (from `Claude.md`): modular and easily expandable, minimal tech debt,
human-understandable, efficient computation.

---

## Progress log

- 2026-10-14/15: git initialised; package skeleton; env `~/venvs/bitsev` (Python 3.12,
  numpy, scipy, torch 2.14 cu130); `codes/` schedule, learner protocols, conjugate
  families, prequential coder, gains and log-slope fits, e-values; `exact/` Gaussian mean
  (batching table reproduced), learning curves, identifiability; `preregistration/G-A0.md`;
  SLURM templates. Week 1 deliverables complete; 31 tests pass.
- 2026-10-15: week 2 exact work done ahead of schedule: `exact/germs.py` (four singular
  germs reduced to exact 1-D integrals, cross-checked against Gauss-Legendre),
  `exact/thermometer.py` (reproduces Appendix C), `exact/batching.py`, `exact/rrr.py`
  (Aoyagi-Watanabe λ against least-squares κ), `experiments/run_exact.py` (`bitsev-exact`).
  **Gate G-A0 passed** on all five preregistered criteria; record in `reports/G-A0/`.
  Still open from weeks 1-2: dynamics-note re-runs and the `devinterp` issue text
  (writing tasks), and the sampled Bayes code for reduced-rank regression (needed for
  H-A3 in week 5, not for G-A0). 47 tests pass.
- 2026-10-15 (afternoon): week 3 pipeline built: `alignment/` (network loci, Rotation and
  AdditiveCoupling maps with certified distortion bounds, interchange on cached states,
  gradient learner, controls with declared information access, score with faithfulness),
  `testbeds/hierarchical_equality.py` + training loop with dense checkpoints,
  `experiments/run_stage1_toy.py` (`bitsev-stage1-toy`, SLURM template), figures
  (`bitsev-stage1-figures`), diagnostics D1-D3 (`diagnostics/`), baselines (online probe,
  Pareto). Writing agent delivered `dynamics_note/dynamics_note.tex` (two TODOs need the
  version-5 appendix) and `tooling/devinterp_issue.md`. First full run (d_V = 1) on EPYC:
  rotation at h1 reaches IIA 0.98 i.i.d. and 0.96 shifted but its code is *longer* than the
  untrained control's, because the intervened logits barely leave 50/50 (final log loss
  0.70 nats) and the early blocks pay 10-30 bits per target from confident errors. The
  coupling map aligns h2 (IIA 0.98) where the rotation cannot. Follow-up run with d_V = 4
  submitted (`results/stage1_toy_dv4`). Open design question for the preregistration of
  G-A1: the learner specification (budget, weight decay, schedule start) governs the
  early-block cost and therefore the sign of gains; see `results/stage1_toy/figures/`.
- 2026-10-15 (evening): week 4 started. `testbeds/tl_transformer.py` re-implements the
  TransformerLens 1.x forward (validated to 1e-3 against TL 1.19 in a scratch env
  `~/venvs/tl1`); `testbeds/interpbench.py` loads InterpBench models and re-implements the
  programs of cases 52, 7, 13, 18 with named variables. Token conventions found by brute
  force (ints: BOS after the values; strings: BOS 0, PAD 1, letters from 2). Cases 52, 7, 18
  verified at 100%; case 13's model is only ~80% faithful to its program and is not used.
  `bitsev-stage1-interpbench` + SLURM template; case 18 (variable `freq`, computed by the
  first MLP) submitted at d_V 4 and 8 with the mixture learner. Mixture learner added
  (`--members`): averaging the predictions of independently trained maps cuts the early-
  block cost (coupling at h1: 2300 -> 975 bits in the first partial comparison). Baselines
  wired: online probe, Pareto, causal abstraction error with unmapped interventions; D1-D3
  recorded per configuration. Survey of testbeds in `notes/testbeds_survey.md`.
- 2026-10-15 (late): d_V = 4 toy run finished (`reports/stage1_toy/summary_dv4.md`). Same
  picture as d_V = 1: the h1 rotation (IIA 1.0 / 0.9) gains 720 bits over the input locus and
  520 over the coupling map but loses 480 bits to the untrained control. Caveat found: on
  hierarchical equality the input locus is not "before V is computed" in the plan's sense,
  because a rotation that swaps the a, b slots copies V1 exactly (IIA 0.9 at d_V = 4); the
  wrong-locus control needs a locus where V is not linearly present. H-A2 trajectory at
  h1 (`reports/stage1_toy/trajectory_dv4.png`): the gain against the step-0 network rises
  to +375 bits exactly when held-out IIA jumps from 0.5 to 0.97 (step 128-256), then falls
  to -480 bits as training continues while IIA stays at its maximum: competence onset is
  visible in the code, but the readout's growing confidence then penalises the plug-in
  learner.
- 2026-10-15 (night): mixture learner (4 members, d_V = 4) finished
  (`reports/stage1_toy/summary_dv4_mix4.md`). Hedging removes most of the early-block cost:
  h1 rotation -97 ± 141 bits vs label-only (was -495), h1 coupling +85 ± 18, h2 coupling
  -106 ± 11 with the best faithfulness (log loss 0.4 nats), while the coupling map on the
  untrained network stays at -3149 ± 197. Same expressive family, trained vs untrained
  network: about 3000 bits of gain. The manufactured alignment itself is not yet
  "convincing" (IIA 0.5): a Sutter-style adversary needs a larger map and budget.
  InterpBench case 18 (variable `freq`, single map, d_V = 4, 3 seeds;
  `reports/stage1_interpbench/case18_summary_dv4.md`): rotation at resid_2 reaches IIA 1.0
  and gains 19,200 bits over the embedding locus (a genuine wrong locus here: IIA 0.5) and
  4,900 over the coupling map, but is still 301 ± 10 bits behind the untrained twin; at
  resid_1 (IIA 0.9) the deficit is 1,674. The two mixture runs (d_V 4 and 8) are still in
  progress and should show whether hedging closes the gap as it did on the toy.
- 2026-10-15 (decisions implemented): the user fixed (1) the mixture learner (4 members,
  d_V = 4) as the canonical Stage 1 learner, with a Laplace-Bayes learner for H-A3; (2) a
  Sutter-style adversary on a real LM; (3) the wrong-locus control as the locus where V is
  least linearly decodable; (4) a PyTorch translation of the Méloux CAE systems. Done:
  `alignment/learners.py` Laplace learner (MAP + Hessian posterior, L-BFGS), `testbeds/oracle.py`
  (exactly realizable law, λ = d_V(d − d_V)/2) on which the Laplace code's log-slope is
  2.75 ± 0.13 against λ = 2.5 once the map is gauge-free (`SubspaceMap`; the rotation's
  exponential parametrisation has flat directions only to first order and ruins Bayes codes);
  `controls.least_decodable_locus`; `alignment/hf_network.py` (Hugging Face residual loci at
  one position) + `testbeds/ioi.py` (GPT-2 small 99.7% IO-over-S after a template fix) +
  `bitsev-stage1-ioi` on the DGX partition (coupling map, 512 wide, 4 blocks, 4000 steps on
  the twin as the adversary); `cae_torch/` (nine systems, 113 equivalence tests, separate
  package). Diagnostics on the toy mixture run: CAE 0.25 for valid alignments vs 0.47-0.49
  for untrained/manufactured; D2 share≤0 is 1.0 for trained networks and 0.0 for untrained
  (H-A8); the unmapped-variable error does not separate single-variable alignments on
  hierarchical equality because the complement legitimately carries V2.
- 2026-10-15 (noise channel): the per-block curves on InterpBench case 18 showed the correct
  alignment coding clean targets at 0.13 bits but paying 39 bits for each flipped label (92%
  of its code after position 128; the fixed target seed happens to put only three flips in
  the first 128 draws). The frozen readout is far sharper than the 5% randomized response and
  a rotation cannot temper it, so the "untrained twin wins" verdicts of all earlier runs were
  an artefact of coding the clean law. Every predictor now codes the observed law
  (1 − ε)p + ε/K through `targets.NoiseChannel` (learner names carry `rr=ε`); the label-only
  code is unchanged. All Stage 1 runs before commit c39b208 are superseded; re-runs:
  `results/stage1_toy_dv4_mix4_rr` (with trajectory), `results/stage1_ib18_dv{4,8}_mix4_rr`,
  and the GPT-2 IOI run picks the channel up when it starts.
- 2026-10-16: channel re-runs finished (`reports/stage1_toy/summary_dv4_mix4_rr.md`,
  `reports/stage1_interpbench/case18_summary_dv{4,8}_mix4_rr.md`). Every verdict flips to the
  predicted sign. Toy (5 seeds): h1 rotation +506 ± 19 bits over label-only and +515 ± 19 over
  the untrained twin, +1251 over the input locus; the coupling on the untrained network
  (the toy adversary) is at -571 ± 20. H-A2 trajectory: the gain against the step-0 network
  rises to +650 bits exactly at competence onset and stays positive (+515 at the end); the
  d_V = 4 rotation at h2 (IIA 0.72) is the mis-specified case and stays negative. InterpBench
  18 (3 seeds): resid_2 rotation +983 (d_V 4) / +996 (d_V 8) bits over label-only, +1278/+1291
  over the twin, +1946/+1890 over the embedding locus, +929/+813 over the coupling map; per-
  block curves fall to 0.25-0.4 bits per target while every control stays above 1 bit.
  GPT-2 IOI (seeds 0-1 of 3; the 12 h job timed out at 38/60 records): label-only 4325 bits;
  the Sutter-style adversary (coupling, 512 wide, 4000 steps, on the untrained twin at
  layer 6) reaches IIA 0.86-0.90 on the i.i.d. split, 0.00 on disjoint names, and codes 83
  bits *longer* than label-only, while the coupling on the trained network at layer 12 gains
  +1847 bits (IIA 0.93 / 0.05). The subspace hypotheses (d_V 16, lr 1e-3, 400 steps) never
  learn at any layer (IIA ≈ 1/16, calibration error 0.76, about 7 bits per target: GPT-2 keeps
  predicting the base name), so the IOI learner specification is unfixed; a GPU sweep over
  locus, d_V, learning rate and steps (`experiments/ioi_map_sweep.py`, job 1785790) decides
  it before seed 2 is run. Probe baselines moved to their own table (they code V noise-free).
- 2026-10-16 (evening): IOI map sweep (`results/ioi_map_sweep`): subspace maps need lr 1e-2
  (the 1e-3 default never left the base name); with it, d_V = 16 reaches IIA 0.96-1.00 at
  layers 10-12 at the final position and stays at chance at layers ≤ 9 for every width up to
  256; coupling maps diverge at 1e-2. No subspace map transfers to unseen names (shifted IIA
  at chance below d_V 256). GPT-2 run v2 (3 seeds, one A100 job each, 8 h): valid hypotheses
  at layers 10/11/12 gain +2975/+3299/+3142 bits over label-only (4325), about +3000 over
  the twin, +6500-6800 over the embedding locus, all seeds positive; the Sutter-style
  adversary (coupling on the twin at layer 6) reaches IIA 0.86 on every seed but gains only
  +22 ± 98 bits (per seed -83, +218, -69) and is beaten by the valid hypothesis by ~3000
  bits. Preregistration draft `preregistration/G-A1.md` (decisive gain = max(100 bits, 5%)
  and 3 s.e. with all seeds positive): criteria 1, 2, 3, 5 met on every testbed; criterion
  4's "equivalent hypotheses' codes agree" clause fails on GPT-2 (layer 11 codes 324 ± 5 bits
  shorter than layer 10; the more faithful map), which is the one open decision.
  Reports in `reports/stage1_ioi/`.
- 2026-10-17: overnight: GPT-2 seeds 3-4 (five seeds; gains unchanged, s.e. ≈ 3 bits) and the
  adversary sweep. A coupling map on the *untrained* twin at layer 10 (4000 steps) gains
  +704 ± 63 bits over label-only on every seed (layer 9 +290 ± 43), i.e. decisive under the
  preregistered rule: the untrained residual stream still carries the name tokens and a
  1.9 M-parameter map decodes them within the training names (IIA 0.86 i.i.d., 0.00 on new
  names, distortion bound 4e16). More budget (16000 steps) overfits the early blocks
  (−744 bits). The comparative verdicts stand (valid alignment ahead by ~2270 bits; the
  same family on the trained network +2903), but criterion 2 of `preregistration/G-A1.md`
  fails in its absolute form. Submitted: the same adversary on a second random
  initialisation (twin seed 8) to test the comparative form. Decision needed on the form
  of criterion 2 (absolute vs comparative) and on criterion 4's agreement clause.
- 2026-10-17 (afternoon): user decisions: criterion 2 becomes comparative (manufactured
  alignment must not decisively beat the same map on a second untrained twin; distortion
  bound reported as diagnostic) and criterion 4 is restricted to "all equivalent valid
  hypotheses are retained"; both logged as post hoc with the original outcomes kept.
  Second-twin runs on all testbeds (`--adversary-only --second-twin-seed 8`): toy −19 ± 36
  bits, InterpBench +67 ± 24 / −16 ± 42, GPT-2 layer 9 +10 ± 82, layer 10 +349 ± 195 (per
  seed +702/+30/+316, fails only the 3-s.e. test). **Gate G-A1 passes** under the amended
  criteria; the decision and both original failures are recorded in
  `preregistration/G-A1.md`. Next: Bayes/Laplace code for the map family (charges map
  complexity; the plug-in code does not), write-up of the Stage 1 results, and the H-A3
  check on InterpBench with the Laplace learner.
- 2026-10-17 (evening): Bayes code for the map family. The Laplace Gaussian fails on
  saturated readouts (InterpBench, GPT-2): the curvature at the MAP is ~0 in most directions,
  the posterior collapses to the prior and samples are random subspaces (held-out loss 2
  nats, worse than uniform). Two fixes: (i) `SubspaceMap.retract()` fixes the gauge (the
  Gaussian prior had driven the frame norm to 1.6e-4 at n = 64, after which a 1e-6
  perturbation flips the span); (ii) a plain Langevin sampler from the MAP (`sampler=sgld`;
  RMSprop preconditioning amplifies noise in flat directions and is off). With both, the
  sampled predictive matches the MAP at step 1e-6 and hedges more at larger steps, also at
  the valid alignment's expense: with 200 steps the step size is a hedge knob, not a
  converged posterior, which is stated as a limitation. Submitted: H-A3 on InterpBench 18
  (MAP-only subspace, d_V 4 and 1, 3 seeds; `bitsev-h-a3` on the records) and GPT-2 under the
  sampled code at step 1e-5 (layer-11 subspace with controls, layer-10 adversary on twins 7
  and 8). Write-up draft in `reports/stage1_writeup/`.
- 2026-10-17 (night): H-A3 on InterpBench 18 (MAP subspace code, prior 0.01, L-BFGS, warm
  start; `bitsev-h-a3`, regret = prequential loss minus the final fit's loss). At the
  realizable locus resid_2 the regret saturates: d_V 4 reaches 0.0 nats by n = 16 and 49 nats
  by n = 1024 (corrected slope 8 ± 3 against λ = 512, three seeds); d_V 1 spends ~100 nats finding the
  direction by n = 64 and then flattens (148 nats at 1024; slope 12 ± 1 against λ = 129.5).
  There is no λ ln n regime to observe: the readout is deterministic given the span and the
  residual loss is the channel's floor, which does not depend on the parameters. At the
  misspecified locus resid_1 the regret grows steadily (slope 107 ± 15 / 133 ± 38), which is
  the excess-loss drift R_n, not λ. The learning-coefficient law requires a likelihood that
  stays parameter-dependent at order 1/sqrt(n) (as on the oracle); on trained deterministic
  networks with external label noise the Stage 1 code is governed by the search cost and the
  drift instead. L-BFGS overflows on the coupling family; coupling maps use Adam there.
- 2026-10-17 (late): the first GPT-2 run under the sampled code was wrong and cancelled: the
  MAP learner's Adam path fed the prior through Adam's normalisation, where a negligible prior
  gradient dominates the near-zero data gradient of a saturated readout and shrinks the map to
  zero within tens of steps (coupling IIA 0.42 against 0.99 for the plug-in with the same
  budget; layer-11 subspace IIA 0.24). Fixed with decoupled weight decay (AdamW); the MAP now
  matches the plug-in on both families and the Langevin hedge is mild (0.25 vs 0.26 nats).
  Also fixed: the loss scaled by n overflowed float32 backward passes (mean loss now), and
  L-BFGS line-search overflows fall back to Adam. GPT-2 sampled-code jobs resubmitted (six
  A100s). Lesson for the paper's methods section: on saturated readouts every ingredient of a
  Bayes code (prior, curvature, sampler, optimiser) interacts with the near-zero gradients.
- 2026-10-18: second GPT-2 sampled-code attempt (absolute Langevin step 1e-5) was also wrong
  and set aside (`results/stage1_ioi_gpt2_sgld_unscaled_superseded`): a step that is a mild
  hedge on a 260-wide InterpBench frame wipes out the 768-wide GPT-2 subspace (IIA 0.99 ->
  0.25). The sampler now steps in units of each tensor's scale after retraction. Validation
  on GPT-2 layer 11 with 256 targets (one seed): MAP alone IIA 0.98 (348 bits); relative
  step 1e-5 leaves the code unchanged (349); 1e-4 is a mild hedge (361, IIA 0.98); the
  absolute 1e-5 step had given IIA 0.09. The warm start of the MAP learner is now a runner
  option (cold on GPT-2 to match the plug-in's cold-start members; warm on InterpBench for
  H-A3). Full run submitted: step 1e-4, cold start, layer-11 subspace with controls and the
  layer-10 adversary on twins 7 and 8, 3 seeds (`results/stage1_ioi_gpt2_sgld4`). Also in
  flight on EPYC: InterpBench cases 52 and 7 with the canonical learner, and stronger
  manufactured alignments (coupling 256 x 4, 4000 steps) on the toy and case 18; the toy one
  still reaches only IIA 0.53. Summaries now report anytime evidence per comparison.
- 2026-10-18 (morning): GPT-2 under the sampled code (single cold-start MAP + Langevin,
  relative step 1e-4, 3 seeds; `reports/stage1_ioi/sampled_code_comparison_sgld4.txt`): valid
  layer-11 subspace +3350 ± 7 bits over label-only (plug-in mixture: +3302), IIA 0.99; the
  layer-10 manufactured alignment on the untrained twin codes −1985 ± 602 bits (seeds −2784,
  −2365, −806; twin 8: −1777 ± 554, all negative) while its final IIA is still 0.83. Per
  block it pays ~7 bits per target (confidently wrong) until the last block, where the
  plug-in mixture's adversary had reached 2.6. So under this learner the *absolute* form of
  criterion 2 holds on GPT-2. Open question before claiming a Bayes-code complexity charge:
  is it the Langevin hedge or the single cold-start map (the mixture averages four)? The
  MAP-only arm (`results/stage1_ioi_gpt2_map0`) decides; whichever it is, the learner
  specification governs whether a convincing manufactured alignment beats label-only, which
  is itself a result for the paper (the mixture is the stronger learner for the adversary).
- 2026-10-18 (midday): MAP-only arm (`reports/stage1_ioi/single_map_comparison_map0.txt`)
  reproduces the sampled arm to the bit (adversary −2786/−2365/−806 per seed in both; valid
  +3385 ± 5 vs +3350 ± 7): the Langevin step did nothing to the coupling because its scale was
  measured at initialisation, where the coupling's weights are ~0.003 and the fit grows them
  a hundredfold. So the rejection of the layer-10 adversary under this learner is due to the
  single cold-start map: the four-member mixture's averaging rescues the adversary's
  confidently wrong predictions (7 bits per target for one map vs 2.6 at the end for the
  mixture) and barely matters for the valid map. Finding: the absolute rejection criterion
  holds for the single-map learner and fails for the mixture; the comparative criterion
  holds for both. The sampler scale is now measured at the MAP; two confirmation arms
  submitted on the adversary (twin 7, 3 seeds): the plug-in with one member (should
  reproduce −1985) and the Langevin hedge with MAP-scale steps (the actual Bayes-hedge test).
- 2026-10-19: confirmation arms (`reports/stage1_ioi/adversary_by_learner.txt`): the
  layer-10 manufactured alignment gains +704 ± 63 bits under the 4-member mixture, −1085 ± 319
  under a single plug-in member (IIA 0.58), −1986 ± 602 under the single MAP (IIA 0.83) and
  −1948 ± 590 under MAP + Langevin with MAP-scale steps. The Langevin hedge moves the code by
  ~40 bits at the scales we can afford; the absolute verdict is decided by mixture-vs-single
  map: averaging four confidently wrong maps is what lets the adversary beat label-only.
  Recorded as a result (learner specification governs the absolute criterion; the comparative
  criterion holds for all learners); no evidence of a Bayes complexity charge here.
  InterpBench case 7 (`count`): resid_2 rotation +2814 ± 5 over the twin, +4021 over the
  embedding locus, +312 over the coupling; adversary rejected. Case 52 (`value`): resid_2
  rotation +1187 over label-only and +1152 over the twin (IIA 1.0), the coupling does better
  (+1986); the resid_1 hypothesis is mis-specified (IIA 0.2) and loses 1280 bits to the twin,
  correctly; adversary −93. Limitation found: `value` is linearly decodable everywhere (probe
  1.0 at the embedding), so the least-decodable rule picks resid_2, the locus of the valid
  hypothesis, as the "wrong locus"; the rule needs a floor (decodability below chance + margin)
  or a causal criterion for copy variables. Toy strong adversary (256 x 4, 4000 steps): IIA
  0.5, −608 bits: the toy admits no convincing false explanation. Case-18 strong adversary
  timed out at 6 h (2 of 6 records); resubmitted with a longer limit.
- 2026-10-19 (later): anytime-evidence caveat. Between a manufactured alignment and its
  second twin the running log Bayes factor wanders: on the toy it crosses the α = 0.01 bar
  (6.6 bits) at n = 58 with a peak of 51 ± 16 bits and ends at −19 ± 36; on InterpBench 18 it
  crosses at n = 969 (d_V 4) and never (d_V 8). The first crossing is therefore not a usable
  verdict when both sides are noise; the decisive quantities are the size and persistence
  of the evidence (valid alignments: thousands of bits that never return). The write-up
  states this; the preregistered rule (size floor and 3 s.e. with all seeds positive) is
  unaffected because it is not an anytime test.
- 2026-10-19 (decision): wrong-locus control redefined causally
  (`controls.least_intervenable_locus`: best held-out IIA of two cold-start fits of the
  hypothesis's map family per locus; lowest wins). Selections: toy input (0.50 / h1 0.99 /
  h2 0.73), case 18 resid_0 (0.46 / 0.84 / 1.00), case 7 resid_0 (0.17 / 0.99 / 1.00),
  case 52 resid_1 (resid_0 0.99 / resid_1 0.11 / resid_2 0.64; the old rule had picked
  resid_2). A single 256-target fit was not reliable (case 52 resid_2: 0.18 for one seed),
  hence two seeds on the full set. GPT-2: attainable 0.072 / 0.069 / 0.093 / 0.971 / 0.997 / 0.973 at
  layers 0 / 6 / 9 / 10 / 11 / 12; a 0.02 tolerance (ties -> earliest) keeps layer 0, so no
  GPT-2 record changes. Case 52's wrong-locus records recomputed. Preregistration amended and logged as post hoc.
- 2026-10-19 (afternoon): investigation of the learning-coefficient "failure". Hypothesis:
  the nominal λ = d_V(d − d_V)/2 counts every Grassmannian direction, but the trained
  readout responds to only a few; flat directions do not count in singular learning theory,
  so the measured slope is the *effective* λ. Evidence so far
  (`reports/stage1_interpbench/case18_fisher_spectrum.txt`): the empirical Fisher at the fitted
  d_V = 4 subspace on the realizable locus resid_2 puts 90% of its trace in 7 of 1024
  directions (99% in 22; participation ratio 4.6; 5 eigenvalues above a tenth of the top),
  i.e. λ_eff ≈ 3-10 against the measured slope 8 ± 3; per-example gradient norms are 1e-4
  (saturated readout). At the misspecified resid_1 the d_V = 1 family has 24-49 constrained
  directions (participation 15.8), consistent with its large drift slope. The Hessian is
  unusable here (non-finite through the QR at d_V = 1; half the eigenvalues negative at
  d_V = 4). Running: the oracle with only k of d − 1 directions readable (λ_eff = k/2; does the
  slope track k?), InterpBench with 8192 targets (log growth or true saturation?), and the
  bigger-model IOI check (Qwen2.5-0.5B/1.5B, Pythia-410m/1b/1.4b, GPT-2 medium).
- 2026-10-19 (evening): larger models (`reports/stage1_ioi/model_check.txt`,
  `bitsev.experiments.ioi_model_check`): on our 14-token prompts with 32 single-token names,
  IO-over-S accuracy among 16 names is 1.00 for Qwen2.5-0.5B (24 layers, width 896, 3.4 GB),
  Pythia-410m (24, 1024), Pythia-1b (16, 2048) and Pythia-1.4b (24, 2048, 7.4 GB), 0.94 for
  Qwen2.5-1.5B (28, 1536). Chosen for Stage 1: Qwen2.5-0.5B and Pythia-1.4b (two families;
  the latter ~12x GPT-2 small). Map sweeps (single-map learner, layers 0-24, d_V 16/64,
  lr 1e-2, 400 steps) submitted to choose loci before the full runs.
- 2026-10-19 (night): oracle with only k of d − 1 directions readable (d = 12, n = 2048, 4
  seeds, validated MAP setting without the learner's channel; `reports/oracle/`): corrected
  slopes 0.53 ± 0.15, 1.46 ± 0.38, 1.94 ± 0.57, 2.24 ± 0.25, 4.49 ± 0.68 for k = 1, 2, 4, 7, 11
  against effective λ = k/2 = 0.5, 1.0, 2.0, 3.5, 5.5 and nominal 5.5 throughout. The law
  holds with the effective coefficient; the transformer's flat regret measures how few
  directions its readout uses (Fisher: 5-7 dominant of 1024). Pitfall found on the way: the
  oracle readout already carries the noise, so applying the learner's channel on top is a
  misspecification that inflates the slope (12 vs 3.4 at d = 6). A sharpened readout with
  the +-1 V coordinate is trivially separable (zero regret at 10x, failed MAP at 100x), so
  the saturation study uses a continuous V coordinate and the oracle's own counterfactual
  law as the regret reference (running).
- 2026-10-20: readout-sharpness sweep on the oracle with a continuous V coordinate (d = 6,
  k = 5, λ = 2.5, n = 2048, regret against the oracle's own counterfactual law;
  `reports/oracle/readout_sharpness_continuous_v.txt`): corrected slope 1.5 ± 0.5 at the
  matched sharpness (no channel; 1.0 with the channel), then 1.3, 2.1, 6.5 ± 0.8, 14.2 ± 1.7 at
  sharpness x3, x10, x30, x100 with the channel (regret 7 -> 114 nats). A sharper readout
  *raises* the cost per log n: a small angular error flips boundary points confidently at the
  channel floor (~3 nats each), a pre-asymptotic "hard mistakes" regime. With the +-1
  coordinate the same sharp readout gave zero regret. So the transformer's flat regret is the
  large-margin case: discrete tokens leave no points near the boundary, the span is found in
  the first blocks and nothing is paid afterwards; the λ ln n regime needs continuous features
  with boundary mass (then λ_eff = readable directions / 2, per the k-sweep). Margin control
  running.
- 2026-10-20 (margin control, `reports/oracle/margin_control_per_seed.txt`): continuous V with
  a forbidden band |v| < margin, sharpness x10/x30, 6 seeds. Margin 0.3 lowers the slope (x30:
  6.5 -> 2.7); margin 1.0 at x10 gives regret exactly 0.0 on three of six seeds (0.7, 2.7 and 19.8
  on the others) and at x30 splits into three saturating seeds (6, 15, 24 nats) and three collapsed
  MAP fits (~2700 nats, the vanishing-gradient failure seen on the transformers). Conclusion
  of the investigation: the λ ln n law holds with the effective coefficient when the variable
  has boundary mass under a smooth readout; a sharp readout raises the cost per log n while
  boundary points exist; with a margin (discrete tokens) the regret saturates once the span is
  found. Trained transformers are the last case with few readable directions, which is why
  their regret is flat and small. The bfloat16 adapter path ran end to end on Qwen2.5-0.5B.
- 2026-10-20 (LLC cross-check, `reports/oracle/llc_vs_regret_slope.txt`): an SGLD estimator
  of the local learning coefficient (`alignment/llc.py`, Lau et al. 2023: tempered posterior
  at beta = 1/ln n localised at the MAP, lambda = n beta (E[L_n] − L_n*)) on the oracle
  k-sweep (d = 12, n = 2048, 4 chains x 1000 steps). At step 1e-4 and localisation 10 it reads
  0.85 ± 0.13, 1.14 ± 0.17, 1.78 ± 0.37, 2.57 ± 0.19, 4.20 ± 0.21 for k = 1, 2, 4, 7, 11
  (lambda_eff 0.5, 1, 2, 3.5, 5.5; regret slopes 0.5, 1.5, 1.9, 2.2, 4.5): the two independent
  estimators agree with each other and with the truth to within 20-30%; smaller steps or
  tighter localisation under-mix and underestimate (standard LLC-estimator sensitivity). So
  the Stage 1 code's regret slope measures the LLC of the explanation (map family composed
  with the frozen network), and 2 lambda is a readable "effective dimension of the
  explanation". Next: the estimator on the InterpBench realizable subspace (slope 8 ± 3;
  Fisher 5-7 dominant directions) and as a summary column.
- 2026-10-20 (models): Gemma access granted (2-2b, 3-1b-pt downloaded; 2-9b accessible).
  Suitability (`reports/stage1_ioi/model_check.txt`): Qwen3-1.7B-Base 1.00 among names (28
  layers, width 2048, 11 GB fp32), Qwen2.5-3B 0.98 (36 layers, 15 GB), Qwen2.5-7B 1.00 in
  bfloat16 (28 layers, width 3584, 16 GB), Llama-3.1-8B 1.00 in bfloat16 (32 layers, width
  4096, 18 GB). Gemma-2-2b 0.24 and Gemma-3-1b 0.03 with top-1 accuracy ~0: a tokenisation
  issue (beginning-of-sequence token) under test, not a capability verdict. Tiers: 1 =
  Qwen3-1.7B-Base (sweep running); 2 = Qwen2.5-3B/7B (+ Llama-3.1-8B); 3 = Pythia-1.4b
  checkpoints (pilot sweep running).
- 2026-10-20 (Gemma, LLC on a transformer): with the beginning-of-sequence token
  prepended (commit b419fd9) gemma-2-2b and gemma-3-1b-pt solve IOI among names at 1.00
  (16 GB and 9 GB fp32) and Llama-3.1-8B stays at 1.00; the earlier Gemma failure was a
  tokenisation artefact. The SGLD learning-coefficient estimator applied to the fitted
  d_V = 4 subspace on InterpBench 18 (`reports/stage1_interpbench/case18_llc.txt`) is not
  stable: at the valid locus it reads 1.0 ± 0.7 (step 1e-4, localisation 10, 1000 steps),
  12.7 ± 12.3 with 2000 steps (one chain drifts to 50), 27 ± 14 at step 1e-3 and 2.2 ± 0.2
  under weaker localisation (gamma 1); at the misspecified locus 6.4 / 18 / 65 / 13 in the same
  settings. The ordering (misspecified > valid) and the order of magnitude (a few readable
  directions, matching the Fisher spectrum's 5-7) are reproduced; the number is not, and
  the growth with chain length says the chain is drifting along flat directions rather than
  equilibrating. Decision: the LLC does not become a summary column; the regret slope stays
  the estimator of the effective coefficient, the LLC is reported as a diagnostic with its
  sensitivity. Loss traces per chain added to the estimator to separate drift from plateau.
- 2026-10-20 (8192 targets, `reports/stage1_interpbench/case18_h_a3_n8192.txt`): the MAP
  subspace code's regret at the valid locus does not saturate after all: slope 16.8 ± 2.8 (n ≥ 64)
  and 21.1 ± 1.7 (n ≥ 512) for d_V 4 against 8 ± 3 on the 1024-target run, local slope 41-47 over
  the last two doublings; d_V 1: 31.5 ± 8.8 against 12. The MAP's training loss rises from 0.149
  to 0.170 between 1024 and 8192 targets: the 1024-target fits were partly interpolating (1040
  frame parameters), so the earlier "saturation" was the interpolation regime, and the λ ln n
  growth resumes once n exceeds the parameter count, at an effective coefficient of roughly 45
  (about 90 constrained directions, 9% of the frame, nominal 512). Runs to 32768 targets submitted
  (EPYC, 24 h) to see whether the slope settles. Misspecified locus 91-115 as before.
- 2026-10-20 (SLT focus; talk in the week of 2026-10-26): per-chain traces of the LLC sampler
  on InterpBench 18 show no plateau at n = 1024 or 8192: chains sit at zero inside the flat region
  of the saturated readout or escape and grow without bound; the sampler measures the flat region's
  extent, not a power law at a point (`reports/stage1_interpbench/case18_llc*.txt`). Learner
  coefficients from the recorded MAP runs (`reports/stage1_ioi/regret_slopes_map_code.txt`):
  manufactured alignment 988 ± 93 vs valid subspace 49 ± 2 on GPT-2 (255 vs 8 on InterpBench);
  the coupling family on the trained network 225 / 474. Talk figures in `reports/talk_slt/`.
  GPT-2 MAP runs with 8192 and 32768 targets submitted (layers 0 and 11, adversary at layer 10,
  one seed, 48 h) for the scale dependence of the coefficient on a language model.
- 2026-10-20 (SLT investigations, user's decision: investigations only until the talk, both
  framings, direct SLT connections preferred). Three lines opened:
  (1) *T4 numerically*: reduced-rank regression as an alignment family with known λ
  (`testbeds/rrr.py`, `bitsev-rrr-codes`): Bayes code by HMC on the sufficient statistics,
  least-squares plug-in code, and a Langevin LLC at the MAP. The LLC reads Aoyagi–Watanabe's
  λ within 5% (2.18 / 1.50 / 3.35 for 2 / 1.5 / 3.5) at step 1e-4; the code comparison (7
  shapes, 16 seeds, n = 65536) is job 1813528. Expected: Bayes slope = λ, plug-in = MN/2.
  (2) *Sharp readouts*: with a near-deterministic readout behind a noise channel the KL grows
  linearly (not quadratically) in the decision-relevant directions, so the RLCT is 1 per
  direction instead of ½ (λ = k, not k/2) until n resolves the smooth core. The MAP code's
  sharpness sweep (1.0 → 14.2 from ×1 to ×100 at λ_eff 2.5) overshoots even that, so the
  MAP learner's κ and λ must be separated: a genuine Bayes code of the subspace map by HMC
  (`alignment/hmc.py`, `--learner hmc` in the oracle sweep) runs as array 1813587 on the
  sharpness and k sweeps.
  (3) *Finite-n Laplace prediction* from the training-set Fisher spectrum, ½ Σ log(1 + n h σ²),
  underestimates the InterpBench regret 5-6× (24 vs 144 nats at 8192, slope 10 vs ~45): the
  regret is not a curvature count at the fitted point but comes from targets the fit gets
  confidently wrong, i.e. the boundary; consistent with the sampler's failure. Recorded in
  this entry only (no report file yet).
  Large-n GPT-2 MAP runs: 8192 running, 32768 resubmitted after an out-of-memory fix (the
  adapter now evaluates intervened prompts in chunks of 1024).
- 2026-10-20 (Jacobian lens, appendix v7.3, `reports/stage1_ioi/jlens_gpt2.txt`): the
  parameter-free J-lens alignment (fixed frame = span of the names' averaged-Jacobian directions)
  reaches IIA 0.98 / 1.00 at GPT-2 layers 11 / 12 and codes the targets in +3529 / +3626 bits over
  label-only, shorter than the learned subspace (+3302 / +3145 mixture, +3385 MAP); layer 10 +2821
  (IIA 0.80, learned map wins); layer 9 and below fail; the untrained twin fails everywhere
  (0.04). Jacobian consistency: manufactured 2.0-2.2 median vs 0.15-0.38 for the valid maps.
  H-A10 holds on all three claims. For the talk: the floor of the complexity ladder, learning
  coefficient exactly 0, is also the shortest code of the variable (`reports/talk_slt/
  gpt2_complexity_ladder.png`). Module `alignment/jlens.py`, experiment `ioi_jlens.py`.
- 2026-10-20 (oracle Bayes code by HMC, `reports/oracle/bayes_code_hmc.txt`, figure
  `reports/talk_slt/oracle_bayes_vs_map.png`): (a) sharpness sweep (k = 5): the Bayes coefficient
  rises from 2.4 (k/2) at ×1 to 5.3 (k) at ×100, exactly the 1-D quadrature's finite-n prediction;
  the MAP code's 6.5 / 14.2 at ×30 / ×100 is therefore κ > λ (the plug-in pays 2-3× Bayes in a
  sharp landscape). (b) k sweep: the Bayes coefficient does *not* follow k/2 (2.0, 2.9, 3.2, 3.2,
  4.7 for k = 1, 2, 4, 7, 11) while the MAP code does (0.5 … 4.5). Hessian/Fisher check at the MAP
  (`reports/oracle/hessian_vs_fisher.txt`): the 11 − k directions outside the data subspace are
  curved in the Hessian but have zero Fisher (every example constrains them identically), so the
  plug-in pays (1/2) tr(F H⁻¹) = k/2 while Bayes pays for every curved direction once n resolves
  it; the information equality fails there. Three regimes of the plan's T4 now measured: κ_MAP < λ
  (smooth, deterministically constrained), κ_MAP > λ (sharp readout), κ_OLS > λ (reduced-rank
  regression, pending job). "λ_eff = k/2" was a statement about the plug-in code.
- 2026-10-20 (GPT-2 MAP code to 32768 targets, `reports/stage1_ioi/h_a3_gpt2_large_n.txt`):
  valid subspace coefficient stable at ~45-50 per ln n from n = 32 to 16384 (365 nats at 32768,
  ~100 constrained directions of 12288), no interpolation plateau; manufactured alignment 6775
  nats (18x), local coefficient peaking at 1700-2450 around n = 1000; coupling on the trained
  network 165-195; untrained twin's subspace rising 41 -> 66. The 32768 job ran in 2.3 h after the
  chunking fix. Figure `reports/talk_slt/gpt2_regret_vs_n.png`.
- 2026-10-20 (late): the InterpBench runner's default map is `rotation`; the 32768-target
  MAP runs and the first HMC jobs were launched without `--map subspace` and so used the
  rotation family (its L-BFGS warm-started MAP does not track across blocks: regret slope
  ~1000-1250 at n = 32768 with IIA 1.00, identical over seeds because the rotation's init is
  seed-independent; kept in `results/stage1_ib18_dv{4,1}_map_n32768` as a learner-failure
  note, not used). Resubmitted with `--map subspace` (jobs 1814013/4 for 32768 targets,
  1814015-9 for the HMC Bayes code). Pilot sweeps archived (`reports/stage1_ioi/
  map_sweep_{qwen05,pythia14}.json`): Qwen2.5-0.5B faithful at layers 22-24 of 24, Pythia-1.4b
  from layer 16 of 24 (IIA ~0.9 at 16, ~0.97 at 18-24); the coupling map diverges at lr 1e-2 on
  both, as on GPT-2. Qwen3-1.7B sweep still running.
- 2026-10-20 (Qwen3-1.7B-Base sweep, `reports/stage1_ioi/map_sweep_qwen3_17b.json`, 3.5 h on
  one A100): subspace d_V 16 at lr 1e-2, 400 steps reaches IIA 0.95 / 0.96 at
  layers 26 / 28 of 28, 0.37 / 0.45 at 22 / 24, chance below; d_V 64 slightly
  better. One 400-step fit costs ~10 min (15x GPT-2), so a full Stage 1 seed with the mixture learner
  would take days: the tier-1 run should use the single-map MAP learner, loci 0 / 26 / 28 and the
  4000-step adversary at one locus, after the talk. Coupling diverges at lr 1e-2 as elsewhere.
- 2026-10-20 (evening launches, user: multi-day jobs are fine, 2 H100s available for a few
  days): Qwen3-1.7B-Base Stage 1 with the canonical mixture learner, 3 seeds in parallel
  (jobs 1814205-7, layers 0/22/26/28, adversary at 26, 96 h limit); tier-2 map sweeps on
  Qwen2.5-3B and Gemma-2-2b (A100) and, in full precision on the H100s, Llama-3.1-8B and
  Qwen2.5-7B (jobs 1814210/1); Pythia-1.4b checkpoint competence check (step 1000 ... 143000,
  job 1814216) as the entry to the tier-3 competence-onset series (`--revision` added to the
  IOI runner). Still never started from the plan: the MIB causal-variable tasks.
- 2026-10-20 (GPT-2 curvature, `reports/stage1_ioi/curvature_gpt2.txt`): the dense Hessian of
  the learned subspace's loss at layer 11 has 12032 positive eigenvalues (nearly flat spectrum,
  12014 above 1% of the top) and 256 zeros (the gauge): the family is regular on the Grassmannian
  with nominal λ = 6016, but the Fisher is 2% of the Hessian in trace and F/H has median 0.005 per
  direction. Plug-in prediction (1/2) tr(F H⁻¹) = 71 vs measured 45-50; Laplace Bayes 6016 ln n
  (65k nats at n = 1024, 20× label-only). The information equality fails almost everywhere: the
  plug-in finds the plane along noise-free directions and pays nothing for them; a Bayes code of
  the family would be hopeless. This explains the absent Bayes complexity charge and the collapse
  of the Laplace/Langevin codes. Confirmed with the Fisher on fresh targets: trace ratio 2.0%,
  plug-in prediction 70.6 (training-set Fisher 71.2).
- 2026-10-20 (MIB causal-variable track, assessment): datasets are public and load from the
  cluster (`mib-bench/{ioi,copycolors_mcqa,arithmetic_addition,arithmetic_subtraction,arc_easy,
  arc_challenge,ravel}`; rows carry the base prompt plus named counterfactual prompts, e.g.
  arithmetic: `ones_carry_counterfactual`, `random_counterfactual`; MCQA: `answerPosition_*`).
  Track protocol (Mueller et al. 2025): residual stream at the last token (and one task-specific
  token), every layer; featurizers Full/PCA/SAE/DBM/DAS; IIA on pairs where the model is right on
  base and counterfactual; models GPT-2, Qwen2.5-0.5B, Gemma-2-2B, Llama-3.1-8B; headline DAS
  IIA: MCQA 86-95 (answer) / 76-79 (order), ARC 88 / 74-76, arithmetic carry 31 (Gemma) / 52
  (Llama), RAVEL 54-75. First task for us: two-digit addition with the carry variable (a computed
  variable, weak DAS baselines, fixed-length prompts; Llama-3.1-8B tokenises 2-3 digit numbers as
  single tokens so the full answer is one output token; Gemma needs a first-digit readout) on
  the H100; then MCQA (answer position) on Qwen2.5-0.5B / Gemma-2-2b, which needs variable-length
  prompt support (left padding, attention mask) in `HFNetwork`. Loader `testbeds/mib.py` to be
  written after the SLT write-up is handed off (2026-10-21).
- 2026-10-20 (Pythia-1.4b checkpoints, `reports/stage1_ioi/pythia14_checkpoints.txt`): IOI
  competence appears between steps 1000 (0.09) and 4000 (0.95); 1.00 from step 8000. Competence-
  onset Stage 1 series submitted as a GPU array (two at a time): checkpoints 1000-143000, MAP
  learner, loci 0/16/20, 400-step adversary at 20.
- 2026-10-21 (overnight results). *Reduced-rank regression* (`reports/rrr/`): the Bayes code's
  slope equals Aoyagi–Watanabe's λ on all 7 + 5 shapes (e.g. 6.04 ± 0.27 for λ = 6, 1.53 for 1.5),
  the plug-in's equals MN/2 whatever the rank (T4 numerical); gains between two families on the
  same data grow like Δλ ln n (1.52 / 3.40 / 4.05 measured for 1.5 / 3.5 / 4.0); per-sample and
  block schedules agree after the (r−1)/ln r correction (T3 for a singular Bayes code); the
  Langevin LLC reads λ at n = 4096 but diverges at 65536 unless the step shrinks with n.
  *InterpBench 32768 targets (subspace)*: valid coefficient 23-32 (d_V 4), 33-42 (d_V 1), still
  rising slowly; misspecified 120-186. *GPT-2 32768, 3 seeds*: valid 35-42, coupling 130-146,
  manufactured 730-830. *InterpBench Bayes code by HMC*
  (`reports/stage1_interpbench/case18_bayes_code_hmc.txt`): the genuine Bayes code of the valid
  alignment is longer than label-only at 1024 targets (1526-2370 vs 1342 bits; slope ~270 vs the
  plug-in's 8), concentrates only by 8192 (IIA 1.00, 3707 bits), exactly as the GPT-2 Hessian/Fisher
  analysis predicts. Qwen3 Stage 1 (18 of ~40 records per seed) and toy HMC (slow: trajectory
  checkpoints included, will need a resubmission after the 24 h limit) still running.
- 2026-10-21 (strong adversaries, tier-2 sweeps): the 4000-step manufactured alignment fails on
  InterpBench (IIA 0.45-0.50, code 130-1300 bits longer than label-only, both twins, 3 seeds) and on
  the toy (IIA ~0.5, 235-715 bits longer, 5 seeds): `reports/stage1_interpbench/
  case18_strong_adversary.txt`, `reports/stage1_toy/strong_adversary.txt`; gate criterion 1 stays
  unmet on the small testbeds. Tier-2 sweeps (`reports/stage1_ioi/map_sweep_{llama31_8b,qwen25_7b,
  qwen25_3b}.json`): Llama-3.1-8B faithful at layers 28-32 of 32 (0.96-0.98, fp32 on H100, 10 min
  per fit), Qwen2.5-7B at 28 of 28 (0.93; 26: 0.78), Qwen2.5-3B at 34-36 of 36 (0.94-0.96). Gemma-2-2b
  ran out of memory on the A100 and was resubmitted on an H100.
- 2026-10-21 (competence onset, `reports/stage1_ioi/pythia14_checkpoint_series.txt`, partial):
  the valid subspace's code on Pythia-1.4b falls from 4750-4800 bits (step 1000, worse than
  label-only, IIA 0.15) to 2830-2860 (2000, IIA 0.57), 1450-1540 (3000, 0.93) and 1020-1030 (8000,
  0.99), tracking the task accuracy 0.09 -> 0.43 -> 0.95 -> 1.00; the untrained twin's subspace is
  at 3750-3800 bits throughout (IIA 0.38, the raw name copy) and beats the trained network at step
  1000; the 400-step manufactured alignment has IIA 0.78 but a code 640 bits longer than label-only
  at every checkpoint. Later checkpoints running. Strong-adversary report headers corrected: the
  family was the GPT-2-sized coupling (width 256, 4 blocks), per settings.json.
- 2026-10-21 (afternoon launches): InterpBench full Hessian/Fisher (`bitsev-ib-curvature`, EPYC);
  oracle Bayes and MAP k-sweeps at 16384 targets (array 1815759: does the Bayes coefficient rise
  towards 5.5 with n?); Jacobian lens on Qwen3-1.7B, Pythia-1.4b, Qwen2.5-0.5B (A100); Llama-3.1-8B
  IOI Stage 1 in fp32 on an H100 (job 1815763, loci 0/28/30/32, adversary at 30, MAP learner).
  MIB arithmetic testbed implemented (`testbeds/mib.py`, `--task arith`): prompts are fixed-length
  under Llama/Gemma/Qwen tokenizers; Llama tokenises every answer as one token (alphabet 199),
  Gemma/Qwen read the first digit (alphabet 10); unit tests pass; CPU smoke completed the build
  and locus search (carry probe 0.875 at Qwen2.5-0.5B layer 20); GPU smoke queued, and on success
  the Llama-3.1-8B arithmetic run (bf16 on an A100, layers 0-32, adversary at 24, 3000 prompts)
  is submitted automatically.
- 2026-10-21 (Jacobian lens across models, `reports/stage1_ioi/jlens_models.txt`): faithful at
  the last 2-4 layers of Qwen2.5-0.5B (0.94-1.00), Pythia-1.4b (1.00) and Qwen3-1.7B (0.94-0.98)
  with gains of 3330-3715 bits over label-only, failing on every untrained twin; the all-names
  frame generalises to unseen names (0.90-1.00), the training-name frame does not; cond(V_J) is
  2-6 on the newer models against 68-121 on GPT-2. H-A10 holds on four models. InterpBench
  curvature job resubmitted (the entry-point script outside the venv lacked torch; use
  `python -m`).
- 2026-10-21 (MIB arithmetic): GPU smoke of `--task arith` on Qwen2.5-0.5B completed end to end
  (records, controls, adversary); the Llama-3.1-8B arithmetic Stage 1 (bf16, A100, layers
  0/8/16/20/24/28/32, carry variable, adversary at 24, MAP learner, 3000 prompts) is job 1815864.
  Caveat to verify from its log: the model's own accuracy on the carry-determined answer token
  (the small Qwen was near chance under the first-digit readout in the smoke).
- 2026-10-21 (InterpBench curvature, `reports/stage1_interpbench/case18_curvature.txt`): at the
  valid locus the fitted point is a flat, indefinite plateau (top eigenvalue 6e-4, 654 negative
  eigenvalues of 1024), so neither the Laplace count (33-129) nor the plug-in trace (~0) predicts
  the measured 8-45; the misspecified-locus cold-start fit did not converge and is not used. The
  local analysis works on GPT-2 (regular point) and not on the saturated InterpBench readout.
- 2026-10-21 (oracle at 16384 targets, `reports/oracle/bayes_code_hmc.txt`): the Bayes coefficient
  rises with n towards the nominal 5.5 at every k (k = 1: 2.0 -> 3.35; k = 11: 4.7 -> 5.43) while
  the plug-in stays at tr(F H⁻¹)/2 (k = 1: 1.14, the predicted 1.1; k = 11: 5.3): λ(n) of the Bayes
  code is scale dependent and tends to the Grassmannian dimension / 2, the plug-in's κ does not.
  Figure `reports/talk_slt/oracle_bayes_vs_map_n.png`.
- 2026-10-21 (Llama-3.1-8B runs on the H100s, bf16, batch 32): IOI accuracy 0.999 among names,
  attainable IIA 0.98-0.99 at layers 28-32, 0.07 at layer 0 (wrong locus). Arithmetic: accuracy
  0.985 on the single answer token (the Llama readout works); the carry is linearly decodable at
  1.00 from layer 20 but the attainable IIA of a 16-dim subspace is 0.48-0.54 at every layer,
  chance for the binary variable, consistent with MIB's 52% for DAS on this model: the carry is
  not realised by a last-position subspace swap at this budget. Both runs continue (jobs
  1816979/1816980).
- 2026-10-21 (evening launches, queue refilled): sharp-readout oracle sweep (×1/×10/×30/×100,
  Bayes and MAP) to 16384 targets (array 1817688: does the Bayes coefficient fall back from k
  towards k/2 at large n as the quadrature predicts?); GPT-2 Fisher/Hessian at layers 10 and 12
  and at layer 11 with d_V 4 and 64 plus MAP-code runs at d_V 4 and 64 (array 1817689, two at a
  time) for a predicted-vs-measured plug-in coefficient scatter; Qwen2.5-3B IOI Stage 1 (bf16,
  layers 0/32/34/36, job 1817690); Gemma-2-2b arithmetic Stage 1 with the first-digit readout
  (job 1817691). Watchers report accuracies and results.
- 2026-10-21 (late): Qwen2.5-3B IOI run past setup (accuracy 0.982; attainable IIA 0.98 / 0.96 at
  layers 34 / 36, 0.24 at 32, 0.07 at 0). Gemma-2-2b arithmetic past setup (0.993 on the first answer
  digit; attainable IIA for the carry 0.78-0.81 already at layers 0-16, which is mostly the trivial
  share of pairs whose carries agree under the first-digit readout: judge the carry by the code and
  the controls, not by that number). Handoff note: `reports/HANDOFF_2026-10-21.md`.
- 2026-10-21 (sharp readout at 16384 targets, `reports/oracle/bayes_code_hmc.txt`): the Bayes
  coefficient at ×100 falls from 5.3 back to 3.55 (towards k/2) and reads 2.51 ± 0.08 at ×1; the
  MAP's stays at 10 ± 4.5 at ×100. The sharp-readout excess is a finite-n crossover for the Bayes code
  and a persistent learner effect for the plug-in. Figure `reports/talk_slt/oracle_sharpness_vs_n.png`.
- 2026-10-21 (afternoon, archive of the overnight batch). *Tier-2 models*
  (`reports/stage1_ioi/tier2_models.txt`): Llama-3.1-8B, Qwen2.5-3B and Qwen3-1.7B reproduce the GPT-2
  verdicts without tuning (valid subspace +3000-3100 bits at the last 2-4 layers, controls rejected by
  n <= 20, manufactured alignment -1700 to -1800 bits at IIA 0.51-0.59); Qwen2.5-3B layer 32 is the first
  locus where the twin beats the trained network (-445 vs +183). Gemma IOI sweep: loci 22-26; the coupling
  map needs lr 1e-3 there. *MIB arithmetic* (`reports/stage1_mib/arithmetic.txt`): the trivial floor
  (model's own answer; +2715 Llama, +719 Gemma over label-only) is the reference; Gemma reads the carry
  linearly at layer 20 (IIA 0.96 vs floor 0.78, +754 bits vs floor); Llama's 16-dim subspace never leaves
  the floor (as MIB's linear methods) but the coupling family on the trained network reaches IIA 0.94-0.96
  at layers 20-24 with the shortest code at 24 (+5063), while on the twin it reaches 0.14 (-2160). *Toy
  Bayes code by HMC* (`reports/stage1_toy/bayes_code_hmc.txt`): the Bayes code beats the plug-in on the
  valid alignment (+733 vs +506 bits), charges the coupling family what the plug-in does not (-322 vs
  +586) and has regret slope 12.9 at nominal 56: the n-vs-lambda ladder toy / InterpBench / GPT-2.
  *Pythia series* complete (`pythia14_checkpoint_series.txt`): code keeps shortening after accuracy
  saturates (4000 -> 8000), layer 16 degrades at the final checkpoint while layer 20 does not. *GPT-2
  curvature across loci and d_V* (`curvature_gpt2.txt`, `regret_slopes_map_code.txt`): plug-in prediction
  75 vs measured 79 at d_V 4, 71 vs 49 at d_V 16; the valid coefficient is flat in d_V (79/49/56) while
  the nominal Bayes coefficient grows 1528 -> 22528; layer 12 predicted 112 (960 negative Hessian
  directions behind the final norm). Submitted (1820081): the d_V 64 eigendecomposition with 24 h / 200 GB
  (timed out at 6 h after all 49152 Hessian columns) and MAP-code runs at layers 10 and 12 to complete the
  predicted-vs-measured scatter. Qwen3 seeds 1814205-7 still on their 4000-step adversary arm. Summary
  title fix for `--task arith`.
- 2026-10-22 (gain decomposition, `reports/stage1_ioi/gain_decomposition_gpt2_n32768.txt`): new tool
  `bitsev.experiments.gain_decomposition` splits a gain into the final fits' loss difference (≈ n·Δℓ) and the
  regret difference (Δκ ln n + O(1)); the IOI runner's task construction is now callable without the model
  (`ioi_task`, `ioi_targets`), so the label-only code can be rebuilt. GPT-2, 32768 targets: against label-only
  and the twin the gain is all fit term; against the manufactured alignment the fit term is −1950 bits (the flow
  codes the targets better: held-out 0.390 vs 0.427 nats, a calibration effect, GPT-2 puts ≈ 96 % on the name)
  and the whole +7662 margin is regret (+9612); against the trained coupling the terms cross near n ≈ 20k
  (−371 at 32768). "Manufactured longer than label-only" is an n = 1024 statement. Talk: new main slide 14 and
  backup slide 27. Untested remedy: a temperature parameter for every predictor.
- 2026-10-22 (review of the write-up, items 1.1-1.8 and 2.1-2.3, all accepted and fixed). Text: the channel
  bounds every target's cost (log2(K/eps) bits) but realizes the law only at infinite logits on the networks;
  the language-model "MAP" is a maximum-likelihood single-map learner (the prior is inert on the subspace
  map); D2 and the new D1 are network/locus properties, blind to the alignment; InterpBench and IOI variables
  determine the output (positive controls). Code: CAE complement prediction is A(s; V <- V(b)), so the CAE of
  a one-variable alignment is 1 - IIA (0.015 valid vs 0.50; `reports/stage1_toy/diagnostics_corrected.txt`);
  D1 permutation null replaced by margin ratios (nn ratio 1.3 untrained, 2.1/7.8 trained h1/h2). New check:
  no subspace illusion for the learned IOI subspaces on GPT-2, Pythia-1.4b, Qwen2.5-0.5B
  (`reports/stage1_ioi/subspace_illusion.txt`). Oracle (item 1.8): the d-1-k unreadable directions are
  quartic, so the Bayes lambda is k/2 + (d-1-k)/4, not (d-1)/2; the 16384-target HMC slopes match it
  (`reports/oracle/bayes_code_hmc.txt`, end); records carry `lambda_bayes`, talk figures redrawn. Running:
  array 1831396 (HMC, 65536 targets, k = 1, 4, 7; prediction 3.0, 3.75, 4.5 vs 5.5 under the old reading;
  open: at k = 1 and 4 one extra direction has F = H, which would make k = 1's value 3.25). Deck slide 7 and
  the summary slide still carry the old oracle reading (hand-edited deck, not regenerated).
- 2026-10-23 (temperature arm and held-out Hessian, `reports/stage1_ioi/temperature_and_heldout_hessian.txt`):
  `maps.Tempered` (one learned inverse temperature on the readout's logits, MAP code only; runner flag
  `--temperature`). GPT-2, 32768 targets, 3 seeds: the valid subspace becomes calibrated (clean-target loss at
  the channel floor), the fit term against the manufactured alignment is −4 of +9265 bits and the trained
  coupling stays 1.1k behind: with calibration fixed the verdict is the regret term alone. 131072 targets (one
  seed per arm) show the gain shrinking after 16384-32768, but the fixed 400-step learner under-optimises there;
  seeds 1-2 submitted (1842451-4). Held-out Hessian at the 1024-target MAP: tr H 776 vs 683, F/H 1.7 %, Laplace
  count 5651: population curvature, not interpolation. Deck: slide 7 and summary carry the corrected oracle
  coefficient k/2 + (d−1−k)/4; slide 14 shows both rows; slide 9 the held-out Hessian.
- 2026-10-23 (afternoon, `reports/stage1_ioi/large_n_gains.txt`): 131072 targets, 3 seeds per arm, fixed and
  10-pass (`--min-epochs 10`) Adam budgets agree. Tempered: margin over the manufactured alignment plateaus at
  9-10k bits from 32768 on (no crossover), tie with the trained coupling; as is: margin falls to ~6k, trained
  coupling wins by 6-7k (calibration deficit linear). Same-family coupling pair keeps separating (+12.8k / +9.9k).
  Beyond ~32768 targets constant-rate Adam leaves ~0.01 nats/target excess loss, comparable to the κ ln n growth,
  and the final fits are noisy endpoints (fit/regret split unreliable). Next if wanted: decaying learning rate or
  L-BFGS polish for the block fits. Deck: backup slide "To 131072 targets"; 32768-target 10-pass runs still running.
- Deviation from the layout below: `codes/bayes.py` and `codes/plugin.py` were merged
  into `codes/conjugate.py` because both learners share the sufficient-statistics path.
- Compute: the cu130 torch build has no V100 kernels, so GPU work targets DGX (A100).

## 0. What exists and what is missing

| Item | Status |
|---|---|
| Project directory | Empty apart from the plan. Not a git repository. |
| Archived checks cited in the plan (`batching_check.py`, `thermometer_check.py`, `identifiability.py`, `topo_*_check.py`, `common.py`, dynamics-note scripts) | **Not on this machine.** Re-implement inside the new package; the tables in the plan are the regression targets. If the originals are retrieved later, they become extra fixtures. |
| Reusable sibling code | `codes/slt_validation_hpc/slt_numerics` (parallel-tempering HMC sampler, `rhat`/`ess`/`mcse`, log-slope fitting with jackknife and block bootstrap, exact reduced-rank-regression λ in `formulas.aw_rrr`); `codes/geometry-of-safety/core/hooks.py` (hook manager); `codes/lm_geometry_hpc/slurm` (array-job pattern with skip-if-done). Copy or vendor the small pieces; do not import across repos. |
| Compute | SLURM. CPU: `epyc`, `genoa`, `thin`. GPU: `gpu` (2×V100 per node, 18 h limit), `dgx` (dgx001-002), `h100` (8×H100 per node). Account `lade`. Scratch at `~/scratch` (CephFS). |
| Python | `uv` available; miniconda at `~/scratch/miniconda`. No env has transformers, gudhi, devinterp or pyvene. Build a fresh env. |

---

## 1. Repository layout

One installable package, `bitsev`, with thin experiment scripts around it. Each subpackage
is independent of the ones below it, so the core code-length library can be released on its
own (the plan promises a library with the Stage 1 paper and a `devinterp` contribution).

```
description_length/
├── pyproject.toml                # package bitsev; extras: [interp], [topo], [dev]
├── README.md
├── PLAN.md                       # this file
├── preregistration/
│   ├── G-A0.md                   # criterion for the exact-examples gate (frozen before week 1 runs)
│   └── G-A1.md                   # criterion, thresholds, seeds, config hashes (frozen before week 3 runs)
├── src/bitsev/
│   ├── codes/                    # layer 1: code lengths, no ML dependencies beyond numpy/torch
│   │   ├── schedule.py           #   prequential schedules: per-sample, geometric blocks n_j=ceil(r^j); batching factor (r-1)/ln r
│   │   ├── prequential.py        #   generic coder: (learner, schedule, data) -> per-block losses, DL(n) curve
│   │   ├── learner.py            #   Learner/Predictor protocols; SequentialLearner for one-pass evaluation
│   │   ├── conjugate.py          #   conjugate families (Gaussian mean, Beta-Bernoulli, Dirichlet, Gaussian linear regression) with Bayes and plug-in learners
│   │   ├── gain.py               #   G(n) = DL_control - DL; bootstrap intervals; log-slope fit with identifiability warning
│   │   └── evalues.py            #   Bayes-factor e-values; universal inference with certified dual upper bound (convex families); weighted-average combination
│   ├── exact/                    # layer 1b: exactly solvable models for G-A0 and H-A3
│   │   ├── learning_curve.py     #   expected cumulative excess of a known learning curve under a schedule
│   │   ├── gaussian_mean.py      #   closed-form excess 1/(2m) (plug-in) and exact KL for Bayes with N(0,1) prior
│   │   ├── germs.py              #   singular germs (x^2+y^2, x^2+y^4, x^2y^2, (ab-c)^2) as exact 1-D integrals: F_n, learning curve, Gibbs moments, λ fits
│   │   ├── batching.py           #   block-code coefficient tables for any known learning curve
│   │   ├── rrr.py                #   reduced-rank regression: exact λ (Aoyagi–Watanabe) against least-squares κ
│   │   ├── thermometer.py        #   Γ(λ, T) moment estimators, with the Gibbs-only caveat
│   │   └── identifiability.py    #   VIF of (log n)/n vs 1/n regressors
│   ├── alignment/                # layer 2: Stage 1 score
│   │   ├── network.py            #   Network protocol: forward(x) -> logits; read/write hidden state at locus; adapters for nn.Module, HookedTransformer, HF
│   │   ├── algorithms.py         #   Algorithm protocol: run(x) -> output, variables V(x), counterfactual A(b; V<-V(s))
│   │   ├── targets.py            #   frozen counterfactual target sets with randomized response (noise rate ε_e), seeded, hashed
│   │   ├── maps.py               #   invertible map families Φ: rotation (DAS), low-rank rotation, additive coupling (RevNet), with Lipschitz accessors
│   │   ├── intervene.py          #   interchange intervention h'(b,s) = Φ^{-1}(z_V(s), z_⊥(b)) at a locus
│   │   ├── learners.py           #   alignment learners: gradient-trained (cold/warm start, recorded), Laplace-Bayes and SGLD-Bayes for low-dim families
│   │   ├── controls.py           #   untrained network, wrong locus, other map family, label-only; each declares its information access
│   │   └── score.py              #   DL(n) for a configuration; G(n) against each control; faithfulness (log loss + calibration)
│   ├── baselines/                # layer 2b: rivals
│   │   ├── iia.py                #   i.i.d. and distribution-shifted held-out IIA
│   │   ├── cae.py                #   causal abstraction error with interventions on unmapped variables
│   │   ├── probing_mdl.py        #   online (Voita–Titov) probing codes without controls
│   │   └── pareto.py             #   accuracy against map size
│   ├── diagnostics/              # layer 2c: D1–D3 from the topological appendix
│   │   ├── separation.py         #   sep_V via MST + union-find kernel barcode in degree 0; ρ_V; label-shuffle null
│   │   ├── stretch.py            #   forward/inverse stretch, segment sensitivity by power iteration on JVPs, sample distortion Dist_H
│   │   └── realizability.py      #   counterfactual error, one-sided Hausdorff, persistence of H ↪ H∪I (gudhi, optional extra)
│   ├── testbeds/                 # layer 3: models and tasks
│   │   ├── toy_tasks.py          #   hierarchical equality, modular arithmetic, IOI-lite with disjoint-vocabulary splits
│   │   ├── toy_transformer.py    #   small trainable transformer/MLP with dense checkpointing (for H-A2)
│   │   ├── interpbench.py        #   loader for InterpBench semi-synthetic models and their tracr ground truth
│   │   ├── mib.py                #   loader for MIB causal-variable track tasks
│   │   ├── meloux.py             #   simulated systems with valid/invalid abstractions (Méloux et al.) or our re-implementation
│   │   └── manufactured.py       #   recipes: expressive map on untrained net; wrong locus passing i.i.d. splits; wrong algorithm on correlated variable
│   ├── experiments/              # layer 4: configs -> runs
│   │   ├── config.py             #   dataclass configs, YAML load/dump, content hash
│   │   ├── registry.py           #   name -> (network, algorithm, locus, map family, learner, controls)
│   │   ├── run_exact.py          #   G-A0 battery
│   │   ├── run_stage1.py         #   one configuration x controls x seeds -> results/<hash>/
│   │   └── run_baselines.py
│   └── report/
│       ├── tables.py             #   regression tables (batching, thermometer) and gate tables
│       ├── figures.py
│       └── gate.py               #   applies preregistration/G-A1.md criterion to results -> verdict
├── configs/                      # YAML; one file per experiment family
├── slurm/                        # sbatch templates (CPU array for exact/diagnostics; GPU for alignment training)
├── scripts/                      # thin CLIs: `bitsev-exact`, `bitsev-stage1`, `bitsev-gate`
├── tests/                        # pytest; every formula in the plan with a displayed number is a test
└── results/ -> ~/scratch/bitsev_results   (symlink; gitignored)
```

Design rules:
- **Protocols, not inheritance.** `Learner`, `Network`, `Algorithm`, `MapFamily` are small
  `typing.Protocol`s. New testbeds or map families are added by implementing a protocol and
  registering a name; nothing else changes.
- **Schedules are first-class.** Every prequential result stores the schedule, the raw
  per-block losses and the learner identity. Batching correction is applied only in analysis,
  never baked into stored numbers.
- **Information access is declared.** Every control carries a `sees = {"network", "locus",
  "labels", "algorithm"}` subset; `score.py` refuses a comparison whose controls are not
  documented.
- **Frozen artefacts are hashed.** Target sets, intervention protocols and preregistration
  files get content hashes recorded in every result file.
- **Results are append-only JSON/Parquet** under `results/<config-hash>/<seed>/`; runs skip
  if the output exists (same pattern as `lm_geometry_hpc`). Reports are generated from
  results, never edited by hand.
- **Efficiency.** Alignment learners retrain per block, so blocks are geometric (default
  r = 2, also 1.5 for the H-A3 check); the per-block training reuses cached hidden states at
  the locus (compute once per network and locus). Exact examples are vectorised numpy and run
  on CPU nodes.

---

## 2. Environment (day 1)

```bash
cd ~/codes/description_length && git init
uv venv --python 3.12 ~/venvs/bitsev && source ~/venvs/bitsev/bin/activate
uv pip install -e ".[dev]"            # numpy scipy torch pyyaml pandas pyarrow pytest matplotlib
uv pip install -e ".[interp]"          # transformer_lens, pyvene (DAS reference), huggingface_hub
uv pip install -e ".[topo]"            # gudhi==3.13.*
```
- Torch wheel: CUDA build for `gpu`/`h100` partitions; the same env works on CPU nodes.
- InterpBench models are PyTorch (`HookedTransformer`); loading them avoids a JAX/tracr
  dependency. tracr itself is only needed if we compile new programs; defer.
- Pin versions in `pyproject.toml`; write `scripts/check_env.py` in the style of
  `slt_validation_hpc/check_env.py` (lighter: versions + a smoke test).
- `slurm/cpu_array.sbatch` (epyc, 1 core, 4 GB) and `slurm/gpu.sbatch` (gpu partition,
  1 V100, ≤18 h, checkpoint per block so a run resumes).

---

## 3. Week-by-week plan

### Week 0 (2–3 days, overlaps with week 1): skeleton and preregistration of G-A0
- Package skeleton, `pyproject.toml`, CI-style `pytest` run, SLURM templates, results symlink.
- `preregistration/G-A0.md`: the gate text from the plan plus concrete tolerances:
  batching coefficients within 1e-3 of ½(r−1)/ln r for r ∈ {1, 1.5, 2, 4}; Bayes per-sample
  coefficient within 1e-3 of λ in the Gaussian and quadrature models; the three effects
  (Bayes redundancy, batching, optimizer) each shown as a separate, nonzero, identifiable
  term in at least one model.

### Week 1: code-length core and the Gaussian exact examples
Deliverables (all under test):
1. `codes/schedule.py`, `codes/prequential.py`, `codes/learner.py`: generic prequential coder
   over any `Learner`, per-sample and geometric schedules, returns a `CodeLength` object
   (losses per block, cumulative DL(n), schedule, learner id).
2. `codes/bayes.py`, `codes/plugin.py`: conjugate families. Test: chain rule, i.e.
   prequential Bayes code equals −log marginal likelihood to 1e-10.
3. `exact/gaussian_mean.py`: closed-form expected excess per sample for plug-in (1/(2m)) and
   Bayes (exact KL). Summing over schedules reproduces the plan's table
   (0.5001, 0.6166, 0.7213, 1.0820, 0.4998, 0.7208) at block boundaries 2^8–2^22. This is the
   first regression test.
4. `codes/gain.py`: G(n), bootstrap over seeds/orderings, log-slope fit with the
   (r−1)/ln r correction and the identifiability warning (VIF from `exact/identifiability.py`).
5. `codes/evalues.py`: Bayes-factor e-value, sequential martingale check by simulation
   (E[E_n] ≤ 1 under the null to Monte Carlo error), universal inference for a logistic readout
   with a certified upper bound from the dual, weighted-average combination. Test: optimizer
   lower bound inflates E; dual bound does not.

### Week 2: singular exact examples, optimizer effects, G-A0 report
1. `exact/quadrature.py`: 2-D tensor-product quadrature of e^{−nK(w)}φ(w) on a box for
   K ∈ {x²y², x²+y⁴, x²+y², (ab−c*)²}. Outputs F_n, the Bayes learning curve F_{n+1}−F_n,
   λ and m by fitting over a decade of n. Tests: λ = ½ (m = 2), ¾, 1, ½ (m = 1 or 2 for
   c* ≠ 0 or = 0); the 1/(m log m) term visible for x²y².
2. `exact/thermometer.py`: reproduces T̂/T = 0.93, 0.97, 0.98 and λ̂ = 0.37, 0.40, 0.42 for
   x²y² at T = 1e−2, 1e−3, 1e−4; machine precision for the regular cases.
3. `exact/rrr.py`: exact λ for reduced-rank regression (port `formulas.aw_rrr`); Bayes code
   by SGLD/PT-HMC (vendor sampler from `slt_numerics`) vs plug-in code; shows κ_plug-in ≠ λ
   in a singular model. This is the operational content of "optimizer effects" and the first
   evidence on T4.
4. `exact/batching.py`: block-scheme sums for Bayes learners in the singular models, checking
   that corrected block slopes recover λ and that uncorrected ones do not.
5. `report/tables.py` → `results/G-A0/report.md`: one table per claim; `report/gate.py`
   applies `preregistration/G-A0.md`. **Gate G-A0 decision.**
6. Non-code items from the plan for weeks 1–2: draft of the dynamics note (needs the
   archived dynamics scripts or a re-run; schedule the re-run on an EPYC node, it is cheap),
   and the `devinterp` issue text (regression tests against exact λ; code-length utility).
   These are writing tasks; keep them out of the package.

### Week 3: Stage 1 score on a toy ground truth
Goal: a complete DL(n) / G(n) pipeline on one task where we know the answer.
1. `testbeds/toy_tasks.py`: hierarchical equality (Geiger et al.) with variables V = the two
   equality bits; input distribution with i.i.d. and disjoint-vocabulary splits.
2. `testbeds/toy_transformer.py`: small MLP and 2-layer transformer; train with dense
   checkpoints (every checkpoint saved for H-A2); untrained twin with the same seed.
3. `alignment/network.py`, `intervene.py`: hook-based read/write at a locus; cached hidden
   states; batched interchange interventions.
4. `alignment/maps.py`: rotation (orthogonal via Cayley or matrix exponential, DAS-style),
   low-rank rotation, additive coupling with B blocks. Each exposes `forward`, `inverse`,
   `lipschitz_bounds()` and a Gaussian prior.
5. `alignment/targets.py`: frozen target sets y*(b,s) with randomized response ε_e = 0.05
   (frozen), stored with hash.
6. `alignment/learners.py`: gradient-trained learner (Adam, fixed budget per block, cold
   start by default; warm start as a named variant). Laplace-Bayes learner for the rotation
   family (for H-A3 later).
7. `alignment/controls.py` + `score.py`: untrained network, wrong locus (embedding layer),
   other map family, label-only. Faithfulness by log loss with a reliability diagram.
8. Tests: a configuration that reproduces the target law exactly has Δℓ = 0 against an
   equally exact control (R_n = 0 on identical predictions); the label-only code under a
   relabelling-invariant prior equals n log|Y| (T2); gains are in bits and consistent across
   schedules after correction.
Output: first G(n) curves, trained vs untrained, on hierarchical equality.

### Week 4: ground-truth testbeds and manufactured alignments
1. `testbeds/interpbench.py`: load InterpBench models + their tracr-derived high-level
   variables; map variables to loci. `testbeds/mib.py`: MIB causal-variable track tasks
   (IOI, arithmetic, ARC-easy, RAVEL) on the small models the track supports.
2. `testbeds/meloux.py`: check for released code of the simulated-systems benchmark
   (arXiv:2607.00267). If unavailable, implement 3–4 of the ten systems from the paper's
   descriptions with their valid and invalid abstractions, enough for H-A4's pilot.
3. `testbeds/manufactured.py`: three recipes from the plan, each producing an alignment
   that passes a stated existing test: (a) expressive coupling map on the untrained twin
   reaching high i.i.d. held-out IIA; (b) alignment at a locus before V is computed that
   passes i.i.d. splits; (c) wrong algorithm aligned onto a network computing a correlated
   variable.
4. `diagnostics/`: D1 (sep_V, ρ_V, shuffle null), D2 (stretch terms, segment sensitivity,
   Dist_H), D3 (counterfactual error, Hausdorff, gudhi persistence on ≤ few thousand states).
   Regression tests re-derive the four topo-check results in the plan (interleaving bound,
   Poisson slopes −2/k, realizability bound, circle death/birth ratios).
5. Pilot grid on SLURM: {trained, untrained} × {correct locus, wrong locus} × {rotation,
   coupling} × 3 seeds on hierarchical equality and two InterpBench tasks.
6. **Write and freeze `preregistration/G-A1.md`** before the week-5 runs. Contents: the
   list of configurations; the definition of "convincing false explanation" (passes
   distribution-shifted held-out IIA at ≥ threshold *and* CAE at ≤ threshold); the Stage 1
   rejection rule (gain against the untrained-network and wrong-locus controls; the regime
   test from the slope of G(n), with bootstrap intervals over orderings and seeds); the
   retention rule for valid explanations up to the equivalence class listed per task; pass
   criterion as "share rejected that the strongest baseline accepts" and "false-rejection
   rate" thresholds; sample sizes n, ε_e, schedule r, seeds, learner budgets. Commit its hash.
   The thresholds are the researcher's call and must be set here, not after seeing results.

### Week 5: baselines, full grid, training trajectories
1. `baselines/iia.py` (i.i.d. and disjoint-vocabulary shifted), `baselines/cae.py`
   (interventions on unmapped variables; wire into the same `intervene.py`),
   `baselines/probing_mdl.py` (online probing codes without controls), `baselines/pareto.py`.
2. Full grid on SLURM: all configurations from `preregistration/G-A1.md`, all baselines, D1–D3,
   ≥ 5 seeds. GPU array on `gpu` (V100) with per-block checkpoints; diagnostics on CPU array.
3. H-A2: Stage 1 gain vs i.i.d. held-out IIA along the dense training checkpoints from week 3.
4. H-A3 pilot: Laplace/SGLD Bayes code for the rotation family on a toy where λ is known
   (rrr-type map), per-sample and r = 1.5, 2 blocks; check corrected slopes against Δλ.
5. Daily monitoring: `report/tables.py` regenerates the running gate table from whatever
   results exist.

### Week 6: analysis, gate G-A1, and sharing
1. `report/gate.py` applies the frozen criterion; outputs verdict, per-configuration table,
   effect sizes with intervals, and the list of manufactured alignments each rival accepted.
2. `report/figures.py`: G(n) curves per regime; gain vs IIA along training; D1–D3 panels.
3. Written gate report (`results/G-A1/report.md`), with the pass/fail consequences from
   the plan: pass → ICML 2027 method paper and library release; fail → report or post,
   ≤ 3-week bounded feasibility budget only with a new hypothesis and preliminary evidence.
4. Dynamics note to the Alignment Forum (weeks 6–8; writing, not code).

---

## 4. Decisions made here (flag if you disagree)

1. **Re-implement the archived checks** inside `bitsev` instead of waiting for the old
   scripts; the plan's tables are exact regression targets.
2. **Geometric blocks by default** for gradient-trained learners (r = 2), with raw per-block
   losses stored and the batching correction applied in analysis. Per-sample codes only for
   conjugate or Laplace-Bayes learners.
3. **Cold-start retraining per block** as the canonical gradient learner; warm start as a
   named variant. Both are valid prequential codes, but they are different learners and the
   plan requires the learner to be specified.
4. **InterpBench over raw tracr** to stay in PyTorch; MIB causal-variable tasks via its
   released data and small models; Méloux systems re-implemented if no code is released.
5. **Randomized-response noise ε_e = 0.05**, frozen with the targets.
6. **Package name `bitsev`**; layers `codes` → `exact` → `alignment`/`baselines`/
   `diagnostics` → `testbeds` → `experiments`/`report`, with no upward imports.

## 5. Open items that need the researcher

- G-A1 thresholds (IIA and CAE cut-offs for "convincing", rejection rule, pass rates). They
  go into `preregistration/G-A1.md` by end of week 4.
- Location of the archived v5–v7 scripts, if they should be imported as fixtures.
- Which MIB tasks and InterpBench cases to prioritise if week 4 is tight (default: IOI-lite
  and two InterpBench cases with computed, not copied, variables; the plan expects copied
  variables such as the IOI name not to separate).
- Whether the dynamics-note re-runs should live in this repository (default: a separate
  `dynamics_note/` directory, not part of `bitsev`).

## 6. Later stages the layout already anticipates

Stage 2 adds `alignment/responses.py` (network responses as data, unit-of-observation
choice) and black-box learners under the same `Learner` protocol. Project B adds `depth/`
(local free-energy depth, reweighting flip point) reusing `exact/quadrature.py` and the
vendored sampler. The `devinterp` contribution is `codes/` plus `exact/` with their tests.
