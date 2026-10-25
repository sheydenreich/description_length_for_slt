# Talk preparation notes: "Bits of evidence: the free energy of an explanation"

SLT days, week of 2026-10-26. Deck: `reports/talk_slt/slt_days_talk.pptx` (33 pages: 20 main slides, a
divider and 12 backup slides, built by `scripts/build_deck.js`). Slide numbers below follow this rebuild,
which added main slide 14 ("Fit or Occam") and backup slide 27 ("Code rate per block"). These notes are for
the speaker only. Every number below was checked against the file named next to it. Where these notes go
beyond the repo (a citation, or an analysis of my own), they say so. Written 2026-10-22.

**The most important new fact, which the deck now states on slide 14:** the talk's headline "the
manufactured alignment codes longer than label-only" is a statement about **n = 1024**. With 32768 targets
(GPT-2, single-map MAP code, 3 seeds; `results/stage1_ioi_gpt2_map_n32768/summary.md`) the manufactured
alignment beats label-only by **+108,956 bits**, and the coupling map on the trained network beats the valid
subspace by **371 bits**. The valid subspace still beats the manufactured alignment, by +7.7k bits. That
margin is **entirely learning cost**: the fit term favours the manufactured alignment by 2.0k bits. Slides 2 and
15 now say "at 1024 targets" (fixed 2026-10-22); say it aloud anyway.

**Updated 2026-10-23 (temperature arm, held-out Hessian, oracle correction).** Read these before the slide sections.

- **Slide 14 now shows both rows** (`gain_decomposition.png`, 2×2): readout as is (top) and one learned inverse
  temperature per predictor (bottom; `reports/stage1_ioi/temperature_and_heldout_hessian.txt`). With the
  temperature the valid subspace's clean-target loss drops to the channel floor (median 0.051 vs 0.050), the fit
  term against the manufactured alignment is **−4 bits** of a **+9265-bit** gain (32768 targets, 3 seeds), and the
  trained coupling stays 1.1k bits behind (no crossover). The calibration confound explains the whole fit-term
  advantage of the flows; once removed, the verdict is the learning cost alone. Expect: "so you need κ to reject
  forgeries" — yes, at least for this testbed and learner.
- **New slide 15, "Two coupling maps, the same fit: only the regret differs"** (`gain_coupling_pair.png`). Every later
  slide number in this file is one higher in the deck (34 pages). The coupling map on the trained network (layer 11)
  against the same family on the untrained twin (layer 10, the manufactured alignment): 1,869,440 parameters each,
  IIA ≥ 0.99 for both at 32768 targets. Fit term +15 bits (+95 with the temperature), gain +8.0k bits, all regret
  (κ 146 vs 826 per ln n, n ≥ 64, `h_a3_gpt2_large_n.txt`). This needs no temperature: both flows saturate the
  readout. Best answer to "is the manufactured alignment just a worse fit?": no, same fit, different learning cost.
- **Large n (131072 targets, 3 seeds per arm; backup slide "To 131072 targets", `gain_large_n.png`,
  `reports/stage1_ioi/large_n_gains.txt`).** Raising the Adam steps to 10 passes per block reproduces the
  fixed-budget curves, so the late behaviour is not an optimisation-budget artefact. Tempered: the margin over the
  forgery plateaus at 9–10k bits from 32768 on (+9.1k ± 1.1k at 131072), no crossover; the trained coupling ties.
  As is: the margin falls to +5.9k ± 2.1k and the trained coupling wins by 6.9k (calibration deficit ≈ 0.055 nats
  per target, linear). The same-family coupling pair keeps separating (+12.8k / +9.9k). Why the plateau: beyond
  ~32768 targets constant-rate Adam leaves ≈ 0.01 nats per target of excess loss, comparable to the κ ln n growth
  (≈ 1100 bits per doubling); the fit/regret split is unreliable there. Say: "at 32768 targets, calibrated, the
  verdict is all learning cost; further out the plug-in learner's optimisation noise matters as much as its κ."
  Slide numbers after the new backup slide shift by one (35 pages).
- **λ = 6016 is population curvature** (danger question 1 below is answered): the dense Hessian on 4096 fresh
  targets at the same MAP gives tr H 776 (683 on the training targets), F/H 1.7 %, Laplace count 5651,
  ½ tr(F H⁻¹) 73.7. Slide 9 has a bullet on it. The 983 negative held-out directions are overfitting of the
  1024-target MAP.
- **Oracle corrected (slide 7, summary):** the unread directions are quartic (F = H = 0), so the Bayes coefficient
  is k/2 + (11 − k)/4: 3.35 vs 3.0 at k = 1, 5.43 vs 5.5 at k = 11 at 16384 targets. Drop every "λ climbs towards
  the nominal 5.5" and "curved but Fisher-free" phrasing for the oracle; the slide-7 section below predates this.
  On GPT-2 the held-out Hessian shows the directions are genuinely curved, so the oracle mechanism is not the
  GPT-2 one.

**Fixed in the deck on 2026-10-22 (after these notes were first written; commits 0d72ca6, 5d8740b, 11728f6,
c3e3d96).** The slide sections below still describe the earlier text where it changed; each affected section
starts with an "Update" line, and items 0, 4, 5, 6, 7 (slide text only) and 9 of the inconsistency list are
marked fixed.

- Job 1820081 is archived (`reports/stage1_ioi/h_a3_gpt2_map_l10_l12.json`,
  `curvature_gpt2_layer11_dv64_full_heldout_fisher.json`, rows in `curvature_gpt2.txt` and
  `regret_slopes_map_code.txt`). MAP-code κ at layer 10 is 65.1 ± 2.9 (predicted 52.5), at layer 12
  65.1 ± 0.7 (predicted 111.7); the d_V 64 prediction is 176.7 against a measured 55.9 ± 1.6. Slides 10, 13,
  18, 19 and 20 now say that ½ tr(F H⁻¹) gets the order of magnitude but not the trend (measured over
  predicted 0.32–1.24).
- The Qwen3-1.7B manufactured arm is archived (`tier2_models.txt`): −147 ± 53 bits against label-only, mixture
  learner, layer 26, IIA 0.71. Slides 2, 15 and 33 and both verdict figures show it.
- The "≈ 65,000 nats, ≈ 22×" Laplace number is gone (slides 9, 11, ladder figure, `curvature_gpt2.txt`): any code
  of the observed law is capped at 6.46 nats per target (≤ 2.2× label-only), and the Laplace count 6016 ln n
  exceeds that cap until n ≈ 10⁴. The figure draws the cap as an upper bound.
- "Longer than label-only" and "coupling ~1000 bits behind" are scoped to n = 1024 on slides 2, 15 and 20 and
  on the y-axes of `iia_vs_code.png` and `models_verdicts.png`, with a pointer to slide 14.

Read these before you rehearse. Each could change a sentence on a slide.

---

## 1. The argument on one page

The talk makes seven claims. Each is listed with the slide that carries its evidence and how firm it is.

1. **Code length rejects alignments that interchange accuracy accepts.** The score is the prequential code
   length of the algorithm's noisy counterfactual outputs, with the intervened network's frozen readout as
   the predictor. On GPT-2 a 1.9 M-parameter coupling map on an untrained twin reaches IIA 0.83 and codes
   1986 bits longer than label-only. The valid subspace codes 3385 bits shorter. Evidence: slides 2, 12
   and 15. Firmness: the **comparative** verdict (valid beats manufactured) holds under every learner tried
   and at every n measured (+7.7k bits at 32768, though it peaked at 16384). The **absolute** verdict
   (manufactured loses to label-only) is an n = 1024 statement. It fails under the preregistered canonical
   mixture learner at GPT-2 layer 10 (+704 bits; slide 29). It also fails at n = 32768 under the single-map
   code (+108,956 bits; slide 14).
2. **For a Bayes code the gain is a difference of free energies,** so Δλ ln n is the Occam term. Evidence:
   slides 3 and 4 (chain rule plus Watanabe). Firmness: standard theory, applied here.
3. **Where λ is known, the Bayes code reads it, and a plug-in reads its own coefficient.** In reduced-rank
   regression the HMC Bayes slope equals Aoyagi–Watanabe λ on 12 shapes. Least squares gives MN/2, and gains
   between families give Δλ ln n. Evidence: slide 6, and slide 23 for the batching check. Firmness: solid
   for the Bayes code. The plug-in is unconstrained least squares, which makes its MN/2 close to automatic
   (see the questions on slide 6).
4. **The plug-in coefficient κ is not λ, and the gap is set by the Fisher against the Hessian.** On the
   oracle the MAP code follows ½ tr(F H⁺) ≈ k/2, while the HMC Bayes slope lies above it and moves with n.
   On sharp readouts κ > λ. Evidence: slides 7 and 8. Firmness: the measurements are solid. The
   interpretation "curved but Fisher-free, so λ(n) tends to 5.5" is open to a better reading (the quartic
   KL; see slide 7).
5. **On GPT-2 the empirical Fisher is 2 % of the empirical Hessian.** The plug-in prediction ½ tr(F H⁻¹) is
   about 71 and the measured κ is 49. The nominal Grassmannian λ is 6016. Evidence: slides 9 and 10.
   Firmness: the factor-of-100 gap to the nominal value is robust. The formula's accuracy is a factor
   1.2–3 across the five configurations now available, and it does not track differences between loci or
   widths.
6. **κ, not parameter count or λ, orders explanations.** The ladder runs Jacobian lens 0, learned subspace
   49, trained coupling 226, manufactured 988. κ stays flat in d_V (79, 49 and 56 at d_V 4, 16 and 64)
   while the nominal λ grows fifteenfold. Evidence: slides 12, 13, 14 and 26. Firmness: the κ ordering
   persists to 32768 targets. But **the code ordering does not follow κ alone at large n**. Against the
   manufactured alignment the fit term turns negative, so the margin is pure learning cost (Occam regime). It
   peaked at 16384, and the speaker notes extrapolate that the manufactured alignment overtakes near 2·10⁵
   targets. Against the trained coupling map the ordering already flips near n ≈ 20k. The mechanism is
   calibration: the orthogonal subspace swap inherits GPT-2's ≈ 96 % on the name, while expressive flows
   saturate the readout. Reading the coupling-map numbers as "coefficients" is also weak, because those maps
   sit far in the interpolation regime (1.9 M parameters against at most 32768 targets).
7. **The Bayes code is the right instrument only while λ ln n is small against n Δℓ.** At λ = 56 it beats
   the plug-in. At 512 it is longer than label-only at 1024 targets. At 6016 it was not run. The talk
   therefore asks SLT for a theory of plug-in learners. Evidence: slides 11, 18 and 19. Firmness: the
   λ = 56 and 512 rungs are measured. The 6016 rung is an extrapolation whose "22×" number is impossible
   as stated (slide 11).

Supporting material: the protocol transfers to three larger models and to MIB addition without retuning
(slides 15 and 17), and the code tracks competence onset in Pythia (slide 16).

---

## 2. Glossary (as defined in this repo)

- **Target.** A base/source input pair (b, s) with the algorithm's counterfactual output
  y* = A(b; V ← V(s)), passed through the noise channel. **n counts coded targets.** Each is one label, not
  a token. The main runs have 1024 targets, of which the first 8 are not coded, so 1016 are coded.
- **Noise channel / randomized response.** With probability ε = 0.05 a label is replaced by a uniform draw
  from the K classes. Every predictor codes the **observed law** p_obs = (1 − ε) p + ε/K. This bounds
  the cost of any target by log₂(K/ε): 9.3 bits for IOI's K = 32. Coding the clean law flipped every
  verdict (slide 22).
- **Label-only code.** The Bayes code of a Dirichlet(1,…,1)-categorical model of the label sequence alone
  (Laplace's rule (1 + c_y)/(K + m)), frozen at block boundaries. It sees neither the network nor the
  algorithm. On GPT-2 IOI it is 4324.5 bits for 1016 targets. (The project record once called it a
  plug-in; it is a posterior-predictive code.)
- **Gain over label-only.** DL(label-only) − DL(configuration) in bits on identical targets. A positive
  gain means a shorter code. Gains against other controls are defined the same way.
- **Trivial floor (MIB).** The layer-0 interchange does nothing, so the predictor is the model's own answer
  on the base. It is right whenever base and source share the carry. This floor is the reference on MIB.
- **Configuration.** A network, a locus, a map family and a learner.
- **Controls.** (i) **Untrained twin**: the same architecture, locus, map family and learner on a randomly
  initialised network (seed 7; the step-0 checkpoint on hierarchical equality). (ii) **Wrong locus**: the
  locus with the lowest *attainable IIA* under the hypothesis's own map family. This is the causal rule,
  adopted post hoc after the probe rule failed on InterpBench case 52. It picks layer 0 on GPT-2. (iii)
  **Other map family**: the coupling map at the same locus. (iv) **Label-only**. (v) **Second twin** (seed
  8), used only for the comparative rejection criterion.
- **Manufactured alignment.** The construction of Sutter et al.: an additive coupling map (width 512, 4
  blocks, 1,869,440 parameters on GPT-2, 4000 Adam steps) trained on the untrained twin. On Pythia it is a
  400-step version.
- **Map families.** *Gauge-free subspace map*: a d × d_V frame W completed by QR, which depends only on
  span(W). It has d·d_V parameters and nominal λ = d_V(d − d_V)/2, half the Grassmannian dimension; 6016
  for GPT-2 (d = 768, d_V = 16). *Rotation* exp(S): the DAS parametrisation, whose flat directions are flat
  only to first order. *Additive coupling map*: an invertible non-linear map with a certified distortion
  bound.
- **Learners.** *Plug-in*: a fresh map per block, 400 Adam steps. *Mixture of four*: the canonical
  (preregistered) learner, which averages four plug-in members' predictive distributions. *MAP / single-map
  MAP code*: one map per block. On the subspace family the prior is inert (decoupled decay plus QR
  retraction), so this is a **maximum-likelihood single-map learner**; on GPT-2 it is fitted by cold-start
  AdamW. *Laplace*: MAP plus a Gaussian; it fails on saturated readouts. *MAP + Langevin hedge*: 200 SGLD
  steps from the MAP; the step size is a hedge knob, not a converged posterior. *Bayes code by HMC*: a
  genuine posterior predictive. *Jacobian lens*: fits nothing.
- **Regret.** The cumulative prequential loss minus the cumulative loss of a reference on the same targets,
  in nats. On networks the reference is the **final fit on all targets of the run**, an in-sample
  best-in-family proxy. On the oracle and in RRR it is the true law.
- **Plug-in κ (learner coefficient).** The batching-corrected least-squares slope of regret against ln n
  (default fit range n ≥ 64 on transformers, n ≥ 256 on the oracle and RRR), for a point-estimate learner.
  Its prediction is κ ≈ ½ tr(F H⁻¹) (Takeuchi/sandwich). F is the **empirical** Fisher: the uncentred
  second moment of per-example gradients at the MAP, on training or held-out targets. H is the Hessian of
  the mean training loss at the MAP.
- **Bayes λ.** Used in two senses. (a) The **nominal λ** d_V(d − d_V)/2, the regular-model value on the
  Grassmannian and an upper bound on the true RLCT. (b) The **measured Bayes slope**: the regret slope of an
  HMC Bayes code at finite n, which the repo writes λ or λ(n). Say which one you mean every time.
- **Laplace count.** ½ Σ n hᵢ/(n hᵢ + α) over Hessian eigenvalues: a finite-n effective number of resolved
  directions (MacKay's γ/2).
- **Batching correction.** Divide raw slopes by (r − 1)/ln r: 1.4427 for doubling blocks, 1.1204 for
  r = 1.25, 2.1640 for r = 4.
- **Interpolation regime.** The map has about as many parameters as there are targets or more, so the final
  fit can interpolate (including the flipped labels) and the regret is flat or distorted for reasons that
  have nothing to do with learning coefficients.
- **Excess-loss drift R_n.** The part of a misspecified family's regret that comes from the prefix fits'
  errors moving relative to the final fit. It is neither n Δℓ nor λ ln n.
- **Regimes of the gain.** *Falsification*: Δℓ > 0, linear gain. *Occam*: both families reproduce the law,
  logarithmic gain via Δλ. *Approximation*: neither fits.
- **Fit term / regret term (gain decomposition, slide 14).** For a hypothesis H and control C on the same
  targets, G(n) = [Fit_C(n) − Fit_H(n)] + [Reg_C(n) − Reg_H(n)]. This is an exact identity. Fit_X(n) is the
  loss of X's final fit on the first n targets; it grows like n·Δℓ. Reg_X = DL_X − Fit_X; it carries
  Δκ ln n + O(1). Tool: `src/bitsev/experiments/gain_decomposition.py`.
- **Channel floor.** The entropy of the observed law when the clean label is known: 0.360 nats/target for
  ε = 0.05 and K = 32. On a clean (unflipped) target the floor's loss is −ln(0.95 + 0.05/32) = 0.050 nats.
- **Code rate per block.** The nats per target within one doubling block. Each block is coded before it is
  fitted, so this is an honest held-out loss of the predictor fitted on the prefix, with no final fit
  involved (slide 27).
- **Decisive gain.** G ≥ max(100 bits, 5 % of label-only) (216 bits on IOI), at least 3 s.e. over seeds,
  and positive on every seed. Fixed 2026-10-16.
- **Anytime evidence / e-process.** The running gain against a control is a running log Bayes factor. It is
  an e-process only under the **simple** null that the control's sequential forecasts are the law of the
  labels. "Rejects α = 0.01 by n" means every seed's running evidence exceeds log₂ 100 ≈ 6.6 bits by that
  index.
- **IIA.** Interchange-intervention accuracy: the agreement of the intervened network's argmax with the
  clean counterfactual. *i.i.d. split*: 1000 new pairs from the training pool, whose inputs may recur.
  *Shifted split*: disjoint names or vocabulary.
- **Attainable IIA.** The best held-out IIA of two cold-start fits of the family at a locus. It defines the
  wrong locus.
- **Jacobian lens (J-lens).** A frame spanned by the names' averaged-Jacobian directions (W_U J_ℓ). There
  are no fitted parameters, so κ = λ = 0. In "train" names it is built from the 16 training names, in
  "all" from all 32.
- **Jacobian consistency (JC).** ‖δ − W_U J Δ‖/‖δ‖: how far an intervention's actual output change departs
  from the averaged Jacobian's prediction.
- **Readable directions k (oracle).** The data vary along V and along k of the other d − 1 coordinates. The
  readout reads only the V coordinate.
- **Sharpness (oracle).** The readout logit is scaled by ×1 to ×100 relative to the matched law. At ×1,
  β = ½ ln 19 ≈ 1.47 for ε = 0.1, so the logit difference is ≈ 2.9 v.
- **Competence onset.** The training step at which task accuracy rises. Pythia-1.4b checkpoints are used.

---

## 3. Literature: what is known and what is yours

These are the citations you are likely to need. Mark anything you are unsure of as "as I recall".

- **Prequential / predictive MDL.** Dawid (1984, JRSS A, "the prequential approach"); Rissanen (1984,
  IEEE-IT, "Universal coding, information, prediction, and estimation"). Grünwald, *The MDL Principle*
  (MIT Press 2007), covers the NML regret, the fact that "a family that fits anything compresses nothing"
  in the worst case, and prequential plug-in codes.
- **The prequential code of the Bayes predictive equals the Bayes mixture code** by the chain rule. This is
  standard; it is in Grünwald (2007) and Dawid.
- **Regular-model free energy** F_n = n L_n + (d/2) log n + O_p(1): Rissanen (1986, Ann. Stat.), Clarke &
  Barron (1990, IEEE-IT).
- **Watanabe's expansion** F_n = n L_n(w₀) + λ ln n − (m − 1) ln ln n + O_p(1), and the Bayes generalisation
  error λ/n: Watanabe, *Algebraic Geometry and Statistical Learning Theory* (CUP 2009) and *Mathematical
  Theory of Bayesian Statistics* (CRC 2018). It holds under realizability or a relatively-finite-variance
  condition.
- **Reduced-rank regression λ and m:** Aoyagi & Watanabe (2005, *Neural Networks* 18, "Stochastic
  complexities of reduced rank regression in Bayesian estimation").
- **Plug-in regret ½ tr(F H⁻¹) ln n.** The sandwich covariance is from Huber (1967) and White (1982,
  Econometrica). Takeuchi's TIC (1976, in Japanese) is the same trace. For prequential plug-in ML codes under
  misspecification, Grünwald & de Rooij (COLT 2005, "Asymptotic log-loss of prequential maximum likelihood
  codes") show, as I recall, a regret coefficient of ½·Var_true/Var_model in exponential families: the 1-D
  version of ½ tr(F H⁻¹). Check the exact statement before citing. The repo's own derivation is in
  `reports/notes/sandwich_learner_coefficient.tex`.
- **Maximum likelihood in singular models** has generalisation behaviour different from both Bayes and d/2.
  Watanabe's book has chapters on ML/MAP in singular models (the error is governed by the maximum of a
  Gaussian process), and Fukumizu (2003, Ann. Stat., "Likelihood ratio of unidentifiable models and
  multilayer neural networks") treats the ML likelihood ratio. Cite cautiously. The rank-constrained ML
  estimator in RRR was **not** run here.
- **Effective number of parameters at finite n**, Σ h/(h + α/n): MacKay (1992, "Bayesian interpolation"),
  Spiegelhalter et al. p_D (2002). The repo's "Laplace count" is this quantity.
- **LLC estimation:** Lau, Furman, Wang, Murfet & Wei, "The local learning coefficient: a singularity-aware
  complexity measure" (AISTATS 2025): λ̂ = nβ(E_β L_n − L_n*), β = 1/ln n, SGLD with localisation. The
  write-up cites further work on reliability (`hitchcock2025`, `cai2026`) that I could not verify.
  Developmental use: Hoogland et al. (2024, "The developmental landscape of in-context learning") showed
  LLC changes at stage boundaries. Relevant to the Pythia question; cite as "as I recall".
- **MDL / online probing:** Voita & Titov (EMNLP 2020); Blier & Ollivier (NeurIPS 2018, "The description
  length of deep learning models"), which shows prequential codes of SGD-trained nets are valid codes that
  are not Bayes codes; Hewitt & Liang (2019) control tasks; Pimentel et al. (2020) Pareto probing.
- **Causal abstraction and DAS:** Geiger et al. (NeurIPS 2021, "Causal abstractions of neural networks"),
  the JMLR/arXiv foundation paper (the write-up cites `geiger2025ca`), and Geiger, Wu, Potts, Icard &
  Goodman (CLeaR 2024, DAS).
- **Expressive maps fake alignments:** Sutter, Minder, Hofmann & Pimentel (2025, "The non-linear
  representation dilemma"; check the exact title and venue). They report perfect IIA on IOI in randomly
  initialised LMs with non-linear maps, and failure on disjoint names.
- **Illusions in subspace patching:** Makelov, Lange & Nanda (ICLR 2024, "Is this the subspace you are
  looking for?").
- **IOI:** Wang et al. (ICLR 2023; arXiv 2022).
- **InterpBench:** Gupta, Arcuschin, Kwa & Garriga-Alonso (NeurIPS 2024 D&B).
- **MIB:** Mueller et al. (ICML 2025). The repo reads MIB's causal-variable track as finding DAS strongest
  overall and linear methods near chance on carry; the "52 %" figure for DAS on this task is the repo's
  reading. Check it before quoting.
- **Randomized response:** Warner (1965).
- **E-values, Ville, safe testing:** Ville (1939); Grünwald, de Heide & Koolen (JRSS B 2024, "Safe
  testing"); Ramdas et al. (2023, Stat. Sci.); Wasserman, Ramdas & Balakrishnan (PNAS 2020, universal
  inference, for composite nulls).
- **Jacobian lens:** the write-up cites Gurnee et al. (2026, `gurnee2026workspace`). I could not verify it.
  Say "the Jacobian lens of Gurnee et al., as used here".
- **Méloux et al.** (causal metrics with unmapped interventions; identifiability of abstractions): cited in
  the write-up; not verified here.

**What is the speaker's own (this project):** the score itself (noisy counterfactual targets, frozen readout,
controls with declared information access, the decisive-gain rule); the observed-law finding; the learner
dependence of the absolute verdict; the HMC Bayes codes on oracle, InterpBench, HE and RRR; the GPT-2 dense
Hessian and Fisher; κ flat in d_V; the ladder; the tier-2, MIB and Pythia runs; the J-lens as an alignment
family scored by code.

**Standard facts applied in a new setting:** the Bayes code equals the free energy (chain rule); Watanabe's
expansion as a decomposition of gains; ½ tr(F H⁻¹) for a plug-in; the batching factor (a three-line
derivation, checked numerically here); Laplace-count effective dimension; e-process via Ville.

---

## 4. Slide by slide

### Slide 1: Title, "Bits of evidence: the free energy of an explanation"

**On the slide.** The title, the subtitle ("Scoring interpretability alignments by the code length of
counterfactuals, and what singular learning theory says about the complexity term") and the author line.

**Why it matters.** It frames the talk as an application of description length, where the complexity
term is the SLT object. The punchline in the notes is that the codes that work are plug-in codes, whose
complexity is a Fisher-weighted count rather than λ.

**Pitfalls.**
- "Free energy of an explanation" is literally true only for Bayes codes, and the operative codes are
  plug-in codes. Say this in the first minute so nobody feels baited: "for a Bayes code it *is* a free
  energy; the twist of the talk is that the codes that work are not Bayes codes".
- Set expectations: one seed on several of the larger-model results, and two post hoc amendments to the
  preregistered evaluation (write-up §6).

---

### Slide 2: Interchange accuracy accepts explanations that the code rejects (pill: plug-in κ (MAP))

Update 2026-10-22: the text now says "at 1016 targets", adds Qwen3 (mixture, IIA 0.71, 147 bits longer) and a third
bullet "with 32768 targets both beat label-only; what separates them then is the learning cost alone (slide 14)"; the
notes carry both caveats. The figure has the Qwen3 red point and a y-axis label "1024 targets".

**What the figure shows** (`iia_vs_code.png`). The x-axis is IIA on the i.i.d. split, from 0 to 1. The
y-axis is gain over label-only in bits (−4000 to +4300). A black horizontal line marks the label-only code
at 0. A grey band covers x ≥ 0.8, labelled "accepted by an IIA ≥ 0.8 rule".
- Colour gives the configuration: blue = learned subspace on the trained net; yellow = subspace on the
  untrained twin; green = coupling map on the trained net; grey = wrong locus (embedding / layer 0); red =
  manufactured (coupling on the untrained twin).
- Marker shape gives the model: circle GPT-2, diamond Qwen3-1.7B, square Qwen2.5-3B, triangle Llama-3.1-8B.
- Red points: GPT-2 at (0.83, −1986), Qwen2.5-3B at (0.59, −1723), Llama at (0.51, −1830). There is no Qwen3
  red point; it was pending at build time and is now available (−147, mixture learner).
- Top right: blue points at about +3000 to +3400, IIA 0.9–1.0. Green points at +1400 to +2460, IIA 0.87–1.0.
- Grey points sit near (0.07–0.1, −3100 to −3550).
- Yellow (twin) points run from −640 to +690. The Qwen3 twin is strong (+620–690, IIA 0.54–0.57).
- Some blue points sit low: Qwen2.5 L32 at (0.24, −445) and Qwen3 L22 at (0.40, +712). These are
  non-intervenable loci.
- One sentence to read it: the right edge holds both the best and the worst explanations; the y-axis
  separates them and the x-axis does not.
- Sources: `stage1_ioi/tier2_models.txt`, `gpt2_summary_single_map_map0.md`, `adversary_by_learner.txt`.
  GPT-2 IIAs come from a summary rounded to one decimal, except the manufactured 0.83.

**Why it matters.** It is the motivation: IIA, the standard acceptance statistic, cannot tell a manufactured
alignment from a faithful one, and the code can. It sets up slide 3 (what the score is).

**Known vs ours.** Known: IIA and causal abstraction (Geiger et al.); expressive maps on untrained nets
(Sutter et al. 2025). Ours: the code-length score and all numbers on the plot.

**Likely questions.**
1. *"Is this result learner-dependent?"* Yes, and say so here, not only in backup. Under the canonical,
   preregistered mixture-of-four learner the GPT-2 layer-10 adversary **beats** label-only by +704 ± 63
   bits (IIA 0.86). Under single-map learners it loses by 1085–1986 bits (`adversary_by_learner.txt`). The
   comparison with the valid alignment holds under every learner: 2594 ± 65 bits paired under the mixture,
   about 5400 under the MAP (3385 + 1986). The layer-6 adversary in the main run (mixture) is at
   +60 ± 138, which is not decisive.
2. *"IIA 0.51 and 0.59 would not pass any sensible threshold. Isn't 'same on Qwen and Llama' an
   overstatement?"* Yes. They pass only a 0.5 threshold. The existence of a convincing (IIA ≥ 0.8) false
   explanation rests on GPT-2 alone (write-up §5). Say: "on the larger models the adversary does not even
   fool IIA at 0.8; the point there is that the code still rejects it".
3. *"How noisy is −1986?"* Per seed −2786, −2365 and −806 (s.e. 602). The "one member" arm, the same learner
   up to random-number streams, gives −1085 ± 319. Quote "1000 to 2000 bits worse than label-only" rather
   than 1986.
4. *"Why does the manufactured alignment get IIA 0.83 at all?"* There are 16 training names, and the
   i.i.d. split reuses the training pool, so a 1.9 M-parameter map can memorise name directions in the
   untrained residual stream (which still carries the name tokens). On disjoint names its IIA is 0.00.
   But the valid subspace also fails on disjoint names (shifted IIA 0.1, chance), so the shift test does not
   separate them on GPT-2 (write-up §11, "disjoint-name failure").
5. *"Interpretability illusion: could the 'valid' subspace be an illusion à la Makelov?"* There is a fresh
   check (`stage1_ioi/subspace_illusion.txt`, 2026-10-22). On GPT-2, Pythia-1.4b and Qwen2.5-0.5B the
   disconnected part of the learned subspace alone gives IIA 0.072–0.073, the no-effect level, and the
   output-connected part alone reproduces the full IIA (0.93–1.00, except Pythia L16 at 0.80). The most
   sensitive direction carries the name (82–91 % of its variance). Caveat: this covers the learned subspace
   on IOI only, not the coupling maps.
6. *"What happens with more targets?"* (Expect this after slide 14, or earlier from a sharp listener.) At
   32768 targets the manufactured alignment beats label-only by +108,956 bits, with i.i.d. IIA that rounds
   to 1.0 and shifted IIA 0.0 (`results/stage1_ioi_gpt2_map_n32768/summary.md`). The valid subspace still
   beats it by +7.7k bits. Answer: "the absolute statement on this slide is an n = 1024 statement; the
   comparative one survives to 32768, and slide 14 shows it is carried entirely by learning cost".

**Pitfalls.**
- **"Its code is 1986 bits longer than label-only" holds at n = 1024 only.** Say "at a thousand targets".
  Better: lead with the comparison to the valid subspace, which survives.
- This slide uses the single-map MAP code on GPT-2, Qwen2.5 and Llama, and the mixture on Qwen3. The
  canonical learner of the preregistered evaluation is the mixture. Say aloud: "these numbers are the
  single-map code; under the averaging learner one GPT-2 adversary beats label-only; the comparison with
  the valid alignment survives every learner (slide 29)".
- On IOI the variable `io` determines the output, so interchange at the final position is close to output
  patching (write-up §11, "variables that determine the output"). Interp people may raise this.

---

### Slide 3: The score: prequential code length of counterfactual targets

**On the slide.**
- A pipeline: base and source pair → algorithm's counterfactual y* → randomized response ε = 0.05 → label y.
  Separately: network N, locus ℓ, map φ fitted on the prefix, interchange, frozen readout p(y | b, s).
- The formula DL(n) = Σ −log₂ p_θ̂(i−1)(yᵢ | bᵢ, sᵢ).
- The controls: label-only, untrained twin, wrong locus, other map family.
- "Gain = DL(control) − DL(hyp.) = running log Bayes factor (an e-process against the control)."
- "By the chain rule, the prequential code of the Bayes posterior predictive is the Bayes free energy Fₙ."

**Why it matters.** It defines the object. The two claims SLT people care about are that (i) the gain is
evidence, and (ii) for Bayes predictives it is a free energy. This leads into slide 4.

**Known vs ours.** Known: the prequential code (Dawid), chain rule equivalence, MDL probing as a cousin
(Voita & Titov), the e-value reading (Ville, safe testing). Ours: coding the *algorithm's counterfactuals*
through the network's *frozen readout* under an interchange, with controls of specified information access.

**Likely questions.**
1. *"How does this differ from MDL probing?"* A probe codes the variable itself from the representation,
   noise-free, with a fitted classifier. Here the fitted part is only the alignment map, the readout is the
   network's own (frozen), and what is coded is the algorithm's counterfactual output under intervention.
   A short code needs the network's downstream computation to use the aligned subspace. The repo also
   computes the online probe code as a diagnostic on hierarchical equality; it codes a different quantity
   and is not a criterion.
2. *"What is n: targets or tokens?"* Coded targets (base/source pairs with one noisy label each). n = 1016
   coded of 1024 in the main runs, up to 32768 in the scale runs. The network's own training data never
   enters; λ and κ here are those of the *map family* on the *target distribution*.
3. *"Why is label-only the baseline?"* It is the Bayes code of the labels alone (Dirichlet(1,…,1)), so it
   normalises for target difficulty and knows neither network nor algorithm. A relabelling-invariant Bayes
   code over any family gains nothing over it. The informative comparisons are with the controls (twin,
   wrong locus, second twin); label-only only fixes the zero.
4. *"In what sense is it an e-process?"* Both predictors are fixed before each target, so exp(running
   gain) is a non-negative martingale under the **simple** null "the labels follow the control's forecasts".
   That null is a specific forecaster, not "the control family". A composite-null guarantee needs the
   family's maximised likelihood, which only bounds from one side (universal inference). Slide 19 lists
   this as open.
5. *"Why ε = 0.05, and does the verdict depend on it?"* It was fixed in the research plan and frozen with
   the targets; it was not swept. It sets the per-target cap log₂(K/ε) and the channel floor. A larger ε
   would shrink every gain. Answer: "not tested; the sign of the comparisons should not depend on it, but I
   have no sweep".

**Pitfalls.**
- The speaker notes say randomized response "makes deterministic targets realizable at finite logits".
  **The corrected write-up says the opposite** (§2.5 and §3.1, after review fix 1.1). The target law is
  exactly realizable at finite logits only on the oracle. A network reaches it only in the limit of infinite
  margins, and Watanabe's expansion applies to networks only approximately. The practical role of the
  channel is the **floor** of log₂(K/ε) per target. Change the spoken sentence to: "the channel puts a floor
  under every predictor's cost and makes the target law have full support; networks approach it only at
  infinite logit margin".
- "A family that can fit anything compresses nothing" is a worst-case (NML) statement. An expressive family
  with a lucky inductive bias can still beat label-only on a particular sequence (the layer-10 mixture
  adversary does).

---

### Slide 4: For a Bayes code, the complexity term is a learning-coefficient difference (pill: Bayes λ)

**On the slide.** G(n) = n Δℓ + Δλ ln n − Δ(m − 1) ln ln n + R_n + O_p(1), and Watanabe's Fₙ for each
configuration. Three boxes: Falsification (Δℓ > 0, linear gain), Occam (Δℓ = 0, logarithmic gain via Δλ),
Approximation (neither fits; R_n can dominate). The question at the bottom: what replaces λ for plug-in
codes?

**Why it matters.** This is the hinge from the interpretability application to SLT. The original research
plan wanted to measure Δλ (the Occam regime). The rest of the talk shows the operative codes are plug-in and
measures their coefficient.

**Known vs ours.** The expansion is Watanabe's. Writing a gain as a difference of two expansions, and the
three-regime reading, is the plan's (a standard application). R_n as "excess-loss drift" is the project's
term.

**Likely questions.**
1. *"The two configurations live on different networks, loci and families. Is Δλ meaningful?"* Each
   configuration is its own statistical model of the same target sequence, so each has an Fₙ and the
   difference is well defined. Δλ is a difference of two models' RLCTs, not a local quantity of one model.
2. *"The O_p(1) terms depend on priors. At n = 1024, are they negligible?"* No guarantee. The write-up
   says the prior-dependent constants "need not be small". On RRR the Δλ ln n law is visible at n up to
   65536; on networks nobody measured it.
3. *"Realizability?"* On networks Δℓ and R_n vanish only up to terms exponentially small in the logit
   margins (corrected write-up §3.1). The Occam reading holds to that precision.
4. *"Which regime are your network verdicts in?"* Mostly falsification. On GPT-2 the valid code's gain of
   3385 bits is about 3790 bits of loss term minus about 400 bits of learning cost. Derivation: the
   final-fit loss is ≈ 372 nats ≈ 537 bits, close to the channel entropy of 0.36 nats/target × 1016 =
   528 bits, and the regret at 1024 is 279 nats = 403 bits. So valid vs label-only is n Δℓ-dominated. The
   *complexity* part matters for valid vs coupling vs manufactured. Slide 14 now measures this split
   directly. Against label-only at 32768 targets: fit +117.0k, regret −0.4k bits (falsification). Against
   the manufactured alignment: fit −2.0k, regret +9.6k (Occam, with the sign of Δℓ *against* the valid
   alignment).

**Pitfalls.**
- Do not suggest that any measured network gain is a Δλ ln n gain. None was shown to be.

---

### Slide 5: Two coefficients: the Bayes λ and the learner's κ (pills: both)

**On the slide.** Left column: the Bayes predictive has excess loss λ/m, so regret λ ln n; λ is the RLCT,
d/2 if regular; the subspace family's nominal λ is d_V(d − d_V)/2. Right column: a plug-in's excess loss
κ/m gives regret κ ln n; κ = ½ tr(F H⁻¹) in a regular, correctly specified model (Takeuchi-type), with
F = H giving d/2; F is the second moment of per-example gradients and H the Hessian, both at the MAP. Bottom
line: how κ is measured (regret against best-in-family fit, LS slope on ln n, divided by 1.4427).

**Why it matters.** It fixes the labelling convention used for every number in the talk (blue pill = κ,
orange pill = λ).

**Known vs ours.** Both columns are standard: Watanabe for λ/n; Takeuchi, White and Huber for the sandwich.
The batching factor is the plan's derivation (three lines; Gaussian-mean check 0.7213 vs 0.7213).

**Likely questions.**
1. *"Isn't ½ tr(F H⁻¹) a regular-model formula? Your models are singular."* Yes. It is used as a *local
   prediction at the fitted point*, with the pseudo-inverse on the gauge zeros. On GPT-2 the empirical
   Hessian at the MAP has no zero eigenvalues apart from the gauge, so the formula is at least
   well-defined. On InterpBench the Hessian is indefinite (654 negative eigenvalues), so the formula is
   unusable there (`case18_curvature.txt`).
2. *"'Correctly specified' plus ½ tr(F H⁻¹) is self-contradictory: if correctly specified, F = H."* Good
   catch. The slide's "regular, correctly specified: κ = ½ tr(F H⁻¹)" should read "regular (possibly
   misspecified)". Under correct specification it reduces to d/2. The sandwich formula is the misspecified
   (Takeuchi) version (see `reports/notes/sandwich_learner_coefficient.tex`).
3. *"Your regret is against the in-sample final fit, not the population optimum. Doesn't that bias the
   slope?"* Yes, in a known direction. The final fit's in-sample loss is below the population best by its
   own optimism (about κ nats in total in a regular model), spread over all n targets. That adds a term
   linear in n to the regret and raises the fitted slope. It is large when the final fit interpolates; for
   the manufactured alignment the in-sample final-fit loss is about 0.05 nats/target against a channel
   entropy of 0.36. It is small for the valid subspace, whose final fit sits at about the channel floor.
   On the oracle and RRR the reference is the true law, so there it is exact.
4. *"Is the 'MAP' a MAP?"* On the subspace family, no: the prior acts only along the gauge fibre and the QR
   retraction removes it, so it is maximum likelihood with an optimiser budget (400 cold-start AdamW steps
   per block on GPT-2). Say "plug-in (ML) code" if pressed.

**Pitfalls.**
- Fix the "correctly specified" wording aloud.
- "Almost never on trained networks" (notes): fine, but say "κ = λ was not observed on any trained network".

---

### Slide 6: Exact check: the Bayes code reads Aoyagi–Watanabe λ, the plug-in reads MN/2 (pills: both)

**What the figure shows** (`rrr_exact_check.png`).
- Left panel: x = Aoyagi–Watanabe λ of the shape (M, N, H, r), from 1 to 8; y = batching-corrected regret
  slope per ln n. A grey diagonal marks slope = λ. Black horizontal ticks mark the regular count MN/2 for
  each shape. Blue circles with error bars are the least-squares plug-in; orange squares are the HMC Bayes
  code. All 12 shapes appear; x = 3.5 has three shapes and x = 2 two.
- Orange sits on the diagonal. Blue sits at the black ticks (2.11, 4.31, 3.81 or 8.29) whatever the true
  rank.
- Right panel: x = Δλ between a rank-restricted and a full family on the same data; y = the measured Bayes
  gain slope. Four orange squares near the diagonal: 1.23 ± 0.15 at 1.5, 1.52 ± 0.13 at 1.5, 3.40 ± 0.21
  at 3.5, 4.05 ± 0.14 at 4.0.
- n = 65536, 16 seeds, slopes on n ≥ 256. Sources: `rrr/codes_summary.txt`, `gain_summary.txt`,
  `gain_between_families.txt`.

**Why it matters.** It validates the whole pipeline (HMC Bayes code, batching correction, slope fit) where
λ is exactly known, and gives the first, cleanest instance of κ ≠ λ. It sets up the oracle (slide 7).

**Known vs ours.** λ and m are from Aoyagi & Watanabe (2005). The numerical confirmation by a prequential
HMC code and the between-family Δλ ln n gains are the project's (a standard prediction, measured). That OLS
has regret (MN/2) ln n is a classical regular-model fact (unconstrained least squares is a regular
MN-parameter model).

**Likely questions.**
1. *"Your plug-in is least squares on the unconstrained M × N matrix. That is a regular MN-parameter model,
   so MN/2 is automatic. The singular comparison is the rank-H maximum-likelihood estimator."* This is
   correct, and probably the most likely expert objection on this slide. `src/bitsev/testbeds/rrr.py`
   (`OLSLearner.fit`) fits the full matrix and ignores H, which is why the plug-in slope is identical for
   every H and r at fixed (M, N). Honest answer: "the least-squares plug-in was chosen as the simplest
   point-estimate learner that ignores the singular structure; the rank-constrained ML plug-in was not run,
   and its coefficient is exactly the open problem on slide 18 (ML in singular models; cf. Watanabe's
   book, Fukumizu 2003)". Consider changing the spoken title to "the least-squares plug-in reads MN/2".
2. *"Do the multiplicity-2 shapes really show the ln ln n term?"* Weakly at best. Of the three m = 2
   shapes, (3,3,3,0) sits 0.45 ± 0.21 low, (4,4,4,1) 0.09 ± 0.27 low and (4,4,1,0) 0.01 ± 0.20 low. Two
   m = 1 shapes also sit about 0.28 low: (3,3,3,1) 3.72 ± 0.23 vs 4, and (4,2,4,0) 3.22 ± 0.16 vs 3.5. My
   own estimate (not in the repo) of the slope bias from −(m − 1) ln ln n over n ∈ [256, 65536] is about
   −0.12 per unit of (m − 1), since d ln ln n / d ln n = 1/ln n ≈ 0.09–0.18. Answer: "consistent in sign,
   not resolved at these n".
3. *"Is the HMC posterior converged?"* There are 64 chains × 200 samples per block on the sufficient
   statistics, with acceptance 0.67–0.74 (`codes_summary.txt`). It is an exact Gaussian likelihood, so it is
   cheap and reliable relative to the network cases.
4. *"Why does the between-family gain slope for (3,3,1,1) vs (3,3,3,1) come out at 1.23 instead of
   1.5?"* It is 1.8 s.e. low, not explained. Three of the four comparisons are within one s.e.
5. *"Does the batching correction hold for singular Bayes codes?"* Yes, numerically (slide 23).

**Pitfalls.**
- The text quotes the two good between-family comparisons. The figure shows all four. Mention the 1.23.
- Do not say "multiplicity-2 shapes sit low because of ln ln n" as a finding. Say "the one clear deficit
  is in a multiplicity-2 shape, in the direction of the ln ln n term".
- Pill discipline: the blue numbers are the *least-squares* κ, not a MAP of a rank-H family.

---

### Slide 7: Oracle: the plug-in pays for readable directions, the Bayes code for curved ones (pills: both)

**What the figure shows** (`oracle_ksweep_slide.png`).
- x = readable directions k (1, 2, 4, 7, 11, of 11); y = regret slope per ln n (0 to 8.4).
- A grey dashed line at the nominal (d − 1)/2 = 5.5, and a dotted diagonal at k/2.
- Black crosses mark ½ tr(F H⁺) at the MAP: 1.10, 1.08, 2.41, 3.26 and 4.47.
- Blue circles are the MAP code κ; orange squares the HMC Bayes code. Open markers are n ≤ 2048, filled
  markers n ≤ 16384, and arrows join them.
- MAP: 0.53 → 1.14 (k = 1), 1.46 → 1.39, 1.94 → 2.37, 2.24 → 3.73, 4.49 → 5.32.
- Bayes: 1.98 → 3.35, 2.88 → 3.05, 3.16 → 3.46, 3.19 → 4.44, 4.72 → 5.43.
- d = 12, d_V = 1, no channel (the oracle readout carries the noise), 4 seeds. Sources:
  `oracle/bayes_code_hmc.txt`, `oracle/hmc_n16384/*.json`, `oracle/hessian_vs_fisher.txt`.

**Why it matters.** It is the first setting where κ and the Bayes slope can be measured side by side on the
alignment family itself. The MAP follows the plug-in prediction (about k/2); the Bayes code does not. It
corrects the project's earlier slogan (λ_eff = readable/2 is a plug-in statement) and sets up GPT-2
(slide 9).

**Known vs ours.** That a plug-in in a direction with no gradient noise has no estimation variance there
follows from the sandwich formula (standard). The oracle, the measurements and the Hessian/Fisher check are
the project's.

**Likely questions** (this slide gets the deepest SLT questions).
1. *"The oracle is exactly realizable. At a realizable optimum the population Hessian equals the Fisher
   (information equality, which your own sandwich note proves). How can a direction be 'curved in H but
   Fisher 10⁻¹⁶'?"* **This is a dangerous question, and I think the slide's reading is wrong in an
   interesting way.**
   - In the oracle the readout reads only the V coordinate. Rotating the d_V = 1 frame by an angle φ
     towards an *outside* direction (one along which the data do not vary) changes the swapped V by a
     factor cos²φ and nothing else: v₁′ = v_b + cos²φ ΔV. The prediction moves only at second order in φ.
     So every per-example gradient vanishes (F = 0), and the population KL is **quartic**: K ∝ (1 − cos²φ)²
     ≈ φ⁴.
   - I checked this numerically with a script in my scratchpad that mirrors `bitsev/testbeds/oracle.py`
     (d = 12, k = 4, ±1 V, matched β). A log–log fit of the population KL over φ ∈ [0.02, 0.16] gives
     exponent 4.01 along an outside direction and 2.06 along a readable one.
   - The positive "curvature" 9·10⁻⁵ to 0.03 in `hessian_vs_fisher.txt` is therefore the **empirical**
     Hessian at a finite-sample MAP. The likely source is the O(n^{−1/2}) first-order term in cos²φ of the
     empirical loss at the boundary cos²φ = 1. It is not population curvature. Consistent with this, it is
     identical across all outside directions for a given k and not monotone in k: 8.9e-5, 4.3e-3, 4.3e-4
     and 2.8e-2 for k = 1, 2, 4, 7.
   - Consequence (my derivation, not in the repo): the population KL is K ≈ t_Rᵀ A t_R + B|t|⁴. The cross
     term vanishes in expectation because the noise coordinates are independent of V. So the oracle's true
     RLCT is **λ = k/2 + (11 − k)/4, not 5.5**: 3.0, 3.25, 3.75, 4.5 and 5.5 for k = 1, 2, 4, 7, 11. The
     HMC slopes at n ≤ 16384 are 3.35 ± 0.50, 3.05 ± 0.56, 3.46 ± 0.24, 4.44 ± 0.43 and 5.43 ± 0.36: all
     within about one s.e. of this prediction. The repo's reading "λ(n) moves towards 5.5" is 4–8 s.e. away
     at k = 1 and k = 4.
   - Suggested answer: "the outside directions are degenerate: the KL is quartic there, so they contribute
     a quarter each, and the Bayes slope at 16384 matches k/2 + (11 − k)/4. The plug-in pays nothing for
     them because the empirical minimiser has no first-order noise there". This makes the oracle a
     genuinely singular example where κ_MAP < λ < d/2. **Consider re-wording the slide before the talk.**
     The cheap confirmation is to evaluate the population KL along an outside direction with the repo's
     own oracle code and check the exponent. My script is not in the repo.
2. *"Why does the Bayes slope at k = 1 start at 1.98 at n ≤ 2048?"* It is the finite-n crossover of the
   quartic directions (or, in the repo's language, directions with nh ≲ 1 not yet resolved). Either way it
   is scale dependent.
3. *"The MAP at k = 11 (5.32 ± 0.20 at n ≤ 16384) is 4 s.e. above the plug-in prediction 4.47. Why?"* Not
   explained in the repo. Possibly inexact minimisation (100 warm-started L-BFGS steps per block), or the
   prediction was computed from one MAP at n = 2048. At n ≤ 2048 the agreement is 4.49 vs 4.47.
4. *"Prior on the raw frame vs uniform on the sphere?"* A N(0,1) prior on the free frame W pushed through
   the span map is uniform on the Grassmannian (rotation invariance), so the prior is fine. The Laplace
   numbers are rough for other reasons (write-up §10.2).

**Pitfalls.**
- "Bayes λ … climbs towards the nominal 5.5" is the claim most likely to be contested; see question 1.
- "MAP κ ≈ ½ tr(F H⁺): 1.14 vs 1.10 at k = 1" is the n ≤ 16384 value. At n ≤ 2048 it is 0.53, which the
  repo attributes to an unresolved second inside direction (h = 0.064).
- Always say "Bayes-code slope at n ≤ …", not "λ", for the orange points.

---

### Slide 8: Sharp readouts: a finite-n crossover for Bayes, a persistent excess for the MAP (pills: both)

**What the figure shows** (`oracle_sharpness_slide.png`).
- x = readout sharpness on a log scale (×1, ×10, ×30, ×100); y = regret slope per ln n (0 to 16).
- A dotted line at k/2 = 2.5 and a dashed line at k = 5 ("one per direction").
- Blue = MAP κ, orange = HMC Bayes. Open markers are n ≤ 2048, filled n ≤ 16384, with arrows between them.
- MAP: 1.03 → 1.20 (×1), 2.09 → 2.00 (×10), 6.53 → 2.99 (×30), 14.21 → 10.34 ± 4.49 (×100).
- Bayes: 2.40 → 2.51 ± 0.08, 2.95 → 3.03, 4.41 → 3.41, 5.32 → 3.55 ± 0.47.
- k = 5, d = 6, continuous V ~ N(0,1), learner channel 0.1, 4 seeds. Sources: `oracle/bayes_code_hmc.txt`,
  `oracle/sharp_n16384/*.json`, `talk_slt/sharp_readout_1d_quadrature.txt`. The ×3 point was not repeated
  at 16384 and is not plotted.

**Why it matters.** It shows the second direction of κ ≠ λ (κ > λ). It also shows that even the Bayes
coefficient is scale dependent when the readout is sharp: the effective coefficient is about one per
direction while n cannot resolve the smooth core, because the KL is linear there. It connects to saturated
transformer readouts.

**Known vs ours.** That a KL ∝ |w| gives local coefficient 1 rather than ½ in 1-D is a standard
volume-scaling fact (λ = 1/a for K ∝ |w|^a). That an analytic model's asymptotic λ is unchanged but the
finite-n slope changes is standard SLT lore. The quadrature, the s^1.3 crossover, and the oracle
measurements are the project's.

**Likely questions.**
1. *"Is λ changing with sharpness? The RLCT of an analytic model doesn't."* Correct. The asymptotic
   coefficient of the sigmoid model is ½ per direction. What changes is the finite-n slope (the repo's
   "λ(n)"). Use "Bayes-code slope", not "λ".
2. *"What is s in the 1-D model for your ×100 oracle?"* My own computation, not in the repo: in the oracle
   the logit difference is 2βv with β = ½ ln 19 ≈ 1.47 at ×1, so ×100 corresponds to s ≈ 290. The
   quadrature at s = 100 is already back near 0.51 per direction for n ≥ 256, and at s = 1000 it averages
   about 0.6 over n ∈ [256, 2048]. Taken literally, the 1-D model predicts about 2.7–3 for k = 5 at ×100,
   not the measured 5.3. The oracle's angular geometry is not a 1-D threshold shift, so treat the quadrature
   as a **qualitative** explanation. Say "qualitatively the same crossover", not "predicts".
3. *"Why does the MAP stay at 10 at ×100?"* A point estimate at a slightly wrong angle flips the boundary
   points confidently, and each costs the channel floor (about 3 nats at 10 %). The number of such
   mistakes falls only as fast as the angular error does. The seed spread is as large as the mean
   (± 4.49). The margin control supports this reading (`oracle/margin_control_per_seed.txt`).
4. *"Why is the MAP κ at ×1 (1.0–1.2) below k/2 = 2.5?"* Not explained in the repo. The ×1 row applies the
   learner's channel on top of a readout already matched to the law (the "double-noise" situation of
   write-up §9.3), so it is not a clean test of k/2. Answer: "not understood".
5. *"Does the s^1.3 scaling have a closed form?"* Open (slide 19). It was read off three points: crossover
   at about 2^10, 2^14 and 2^19 for s = 10², 10³ and 10⁴.

**Pitfalls.**
- "Bayes λ rises from 2.40 (k/2) to 5.32 (k)" means finite-n slopes, not RLCTs.
- The MAP at ×30 drops from 6.53 to 2.99 with more data, so "persistent excess" is true only at ×100.

---

### Slide 9: GPT-2: the information equality fails; the Fisher is 2 % of the Hessian (pills: both)

Update 2026-10-22: the second bullet now reads "The Laplace count 6016 ln n exceeds any code's ceiling (6.46 nats
per target, 2.2× label-only) until n ≈ 10⁴"; the 65,000-nat / 22× claim is gone from text and notes.

**What the figure shows** (`gpt2_fisher_vs_hessian_slide.png`).
- Left panel: directions sorted by Hessian curvature on a log x-axis (1 to about 1.2·10⁴); log y-axis
  "curvature / gradient variance". The black line is the Hessian eigenvalue hᵢ: it starts at about 1, falls
  slowly to about 0.05 over 10⁴ directions, then plunges at the tail (minimum 2·10⁻⁵). Blue dots are the
  held-out empirical Fisher in the same eigendirections, from about 0.3 at the top to 10⁻³–10⁻⁴.
  Annotation: tr F / tr H = 2.0 %, median Fᵢᵢ/hᵢ = 0.007.
- Right panel: accumulated coefficient per ln n against the number of directions included, log–log. The
  orange line ½ × count reaches λ = 6016; the blue line ½ Σ Fᵢᵢ/hᵢ reaches 70.6, annotated "κ predicted
  70.6 (measured 49 ± 2)".
- Layer 11, d_V = 16, 12,288 frame parameters, 1024 targets. 12,032 positive eigenvalues and 256 gauge
  zeros. Source: `curvature_gpt2_layer11_full_heldout_fisher.json`.

**Why it matters.** It is the real-network version of the oracle mechanism: the plug-in pays for very few
Fisher-weighted directions and the Laplace count pays for all of them. It sets up slide 10 (prediction vs
measurement).

**Known vs ours.** The dense Hessian and Fisher computation and its result are the project's. The
interpretation through ½ tr(F H⁻¹) is standard.

**Likely questions.**
1. *"This H is the empirical Hessian of the training loss at a MAP fitted to 1024 targets with 12,288
   parameters, which is the interpolation regime (train NLL 0.323 nats/target, below the channel entropy
   of ≈ 0.36). Under near-realizability the population Hessian is close to F. So is the family 'regular
   with λ = 6016', or is most of H fitting curvature?"* **The most dangerous question of the talk, in my
   view.** It is the GPT-2 version of the oracle point (slide 7, question 1).
   - What the repo has: the held-out Fisher matches the training Fisher (trace 14.0 vs 15.0). There is no
     held-out Hessian.
   - Honest answer: "6016 is the regular count, an upper bound on λ. The training Hessian has no zero
     modes beyond the gauge, but I have not shown that the population Hessian is full rank; F ≪ H with
     F ≈ H_population would make the true λ much smaller than 6016". Then say what is robust: the plug-in
     pays about 50, and a Bayes code of this family at this n is uncompetitive in practice (InterpBench HMC,
     slide 11).
   - If there is GPU time before the talk, the decisive check is the Hessian of the mean loss on the 1000
     held-out targets at the same point (about 42 min on one A100; same code path).
2. *"Mean gradient 10⁻² (training), 0.24 (held-out): the MAP is not stationary. Does the formula still
   apply?"* Approximately. ½ tr(F H⁻¹) assumes the exact minimiser. The repo uses the uncentred second
   moment; with ‖ḡ‖² ≈ 0.06 on held-out targets against tr F ≈ 14, the mean-gradient contribution is small.
3. *"Why is F so small? Is the model misspecified?"* The note `reports/notes/sandwich_learner_coefficient.tex`
   gives two reasons: the target law is reached only at infinite logits (approximate realizability), and the
   saturated readout makes per-example gradients tiny. Both are plausible. Neither has been separated
   experimentally.
4. *"Would a Laplace Bayes code really pay 65,000 nats?"* No. See slide 11: the code is capped at
   1016 × ln(640) ≈ 6,570 nats.

**Pitfalls.**
- Say "nominal λ = 6016 (the regular count, an upper bound)" every time. Do not say "λ = 6016".
- "Regular on the Grassmannian" is a statement about the empirical Hessian at one fitted point.
- The slide's "≈ 65,000 nats … ≈ 22× the label-only code" should not be said as a prediction of a code
  length (see slide 11).

---

### Slide 10: ½ tr(F H⁻¹) predicts the measured κ; the nominal λ is two orders of magnitude off (pills: both)

Update 2026-10-22: title now "½ tr(F H⁻¹) gets the order of κ; the nominal λ is 20–400× off". Text: oracle within
the seed errors for k ≤ 7, 5.32 ± 0.20 vs 4.47 at k = 11; GPT-2 measured vs predicted L11 79/75 (d_V 4), 49/71 (16),
56/177 (64), L10 65/52, L12 65/112; "κ stays at 49–79 … prediction moves 52–177 (ratios 0.32–1.24)"; untested
candidates listed. The figure plots all five GPT-2 points (no pending ticks), labels with leader lines, legend lower right.

**What the figure shows** (`predicted_vs_measured.png`).
- Log–log axes: x = predicted coefficient per ln n, y = measured κ (MAP code). A grey diagonal marks
  measured = predicted.
- Blue circles are the oracle k-sweep: x = ½ tr(F H⁺), y = the MAP slope at n ≤ 16384, with error bars.
  Open grey circles sit at x = 5.5 (the nominal) at the same y, joined by dotted lines.
- Blue diamonds are GPT-2 L11 d_V 4 at (75.0, 79.0) and d_V 16 at (70.6, 48.9). Open grey diamonds sit at
  the nominal 1528 and 6016, and there is an open diamond for d_V 64 at (22528, 55.9), "prediction pending".
- A double arrow labelled "85× at d_V 16" (6016/70.6).
- Ticks on the x-axis at 52 (L10) and 112 (L12), "predicted, κ pending".
- Sources: `oracle/hessian_vs_fisher.txt`, `oracle/hmc_n16384/ksweep_map_k*.json`,
  `stage1_ioi/curvature_gpt2_layer*_heldout_fisher.json`, `regret_slopes_map_code.txt`.

**Why it matters.** It is the quantitative heart of the talk: a local Fisher/Hessian formula from regular
theory predicts the plug-in's coefficient within a small factor, while the nominal Bayes count is off by
about 100.

**Known vs ours.** The formula is standard. Its test on trained networks is the project's.

**New numbers since the deck was built** (unarchived; see the top of this file):

| configuration | predicted ½ tr(F H⁻¹) | measured κ (MAP code, n ≤ 1024) |
|---|---|---|
| L10, d_V 16 | 52.5 | 65.1 ± 2.9 (2 seeds) |
| L11, d_V 4 | 75.0 | 79.0 ± 4.6 |
| L11, d_V 16 | 70.6 | 48.9 ± 1.7 |
| L11, d_V 64 | 176.7 | 55.9 ± 1.6 |
| L12, d_V 16 | 111.7 | 65.1 ± 0.7 (2 seeds) |

The prediction is now within a factor 0.6–3.2 of the measurement. It over-predicts at L12 (which has 960
negative eigenvalues, so the formula is ill-posed there) and at d_V 64 (49,152 parameters for 1024 targets,
train NLL 0.19, deep interpolation). It under-predicts at L10. The measured κ hardly varies (49–79); the
prediction varies 3.4×.

**Likely questions.**
1. *"How good is the prediction really?"* With all five GPT-2 points: right order of magnitude, wrong
   ordering across loci and widths. Say: "it gets the two orders of magnitude, not the details".
2. *"Why is κ below the prediction at d_V 16?"* Candidates: inexact minimiser (400 cold-start AdamW steps
   per block), finite n, a singular correction, or H being an interpolation-regime empirical Hessian
   (slide 9). The new d_V 64 point (177 vs 56) supports the last: the more parameters per target, the more
   the training Hessian overstates the population curvature in the denominator… or understates it. Do not
   commit to a direction; say "unexplained, possibly the empirical Hessian at an interpolating fit".
3. *"Oracle within the seed errors?"* Not at k = 11 (5.32 ± 0.20 vs 4.47, about 4 s.e.). At k = 7 it is
   3.73 ± 0.37 vs 3.26 (1.3 s.e.).
4. *"The 85× — prediction or measurement?"* It is nominal over predicted (6016/70.6). Nominal over measured
   is 6016/48.9 ≈ 123×.

**Pitfalls.**
- The slide text "Pending (job 1820081)" is out of date, and "κ < prediction … unexplained" is now
  "prediction off by up to 3× in both directions". Update the spoken version.
- The d_V 4 point is not a valid alignment: IIA 0.27, gain +132, train NLL 2.43. It is a misspecified
  family, which is exactly where the Takeuchi formula is designed to work. Do not call it the valid subspace.

---

### Slide 11: The Bayes code is the right instrument only while λ ln n is small against n Δℓ (pills: both)

Update 2026-10-22: GPT-2 bullet now "not run. Any code is capped at 6.46 nats per target (≤ 2.2× label-only,
drawn); the Laplace count exceeds the cap until n ≈ 10⁴". The figure draws a hollow downward triangle at 2.2×
labelled "≤ 2.2×" (legend: "Bayes code, not run: its ceiling"). The legend issue of item 8 is not fixed.

**What the figure shows** (`bayes_vs_plugin_ladder.png`).
- Log–log axes: x = nominal Bayes λ of the alignment family (56, 512, 6016); y = code length divided by the
  label-only code (0.1 to 30). A line at 1 is labelled "label-only code", and the area above it is shaded
  "longer than label-only".
- Blue circles, labelled "plug-in code (mixture of 4)": HE 0.505, InterpBench 0.216, GPT-2 0.237.
- Orange squares are the HMC Bayes code at 1024 targets: HE 0.282; InterpBench mean 1.44 (seeds 1.14, 1.77,
  1.41 as open squares).
- A dotted orange line runs to an open square at GPT-2, "≈ 22×" ("Laplace prediction, not run").
- An open triangle "Bayes at 8192 targets (1 seed)" at InterpBench sits at 0.35 (3707 / ~10,700).
- Sources: `stage1_toy/bayes_code_hmc.txt`, `stage1_interpbench/case18_bayes_code_hmc.txt`,
  `stage1_ioi/gpt2_summary_v2.md`, `curvature_gpt2.txt`.

**Why it matters.** It places the Bayes code honestly: it works and charges the expressive family when λ is
small (HE: +733 Bayes vs +506 plug-in; coupling +586 → −322), and it fails at the widths of real
alignment problems. It motivates "we need a theory of plug-in learners" (slide 18).

**Known vs ours.** All measurements are the project's. The statement that the Bayes code needs λ ln n ≪
n Δℓ is a direct reading of Watanabe's expansion.

**Likely questions.**
1. *"A Bayes code with the observed-law likelihood cannot cost 65,000 nats on 1016 targets: every factor is
   at least ε/K, so Fₙ ≤ n ln(K/ε) = 1016 × ln 640 ≈ 6,570 nats (≈ 9,470 bits, 2.2× label-only)."*
   **Dangerous, and correct.** λ ln n at n = 1024 is far outside the asymptotic regime: the expansion
   requires λ ln n ≪ the total loss scale. Honest answer: "the 65,000 is the asymptotic λ ln n
   extrapolated; the real Bayes code is capped at about 2.2× label-only, and the prediction is that it sits
   at the prior predictive, near that cap. The point is that it does not concentrate at this n, as
   InterpBench shows directly". Do not show "≈ 22×" as a code length. Say "λ ln n alone is ≈ 22× the
   label-only code; the code itself is capped at ≈ 2.2×".
2. *"InterpBench HMC: chains restart at every block with 200 warm-up iterations. Is that a posterior?"* It
   is an approximate posterior predictive per block. Acceptance is 0.52–0.82. The final IIA of the sampled
   predictive is 0.98 on two seeds and 0.48 on the third, so it is not concentrated at 1024. The
   untrained-twin Bayes code equals the prior predictive (1633 bits on every seed), which is evidence the
   sampler is at least sampling the prior correctly.
3. *"The InterpBench Bayes slope (269 ± 43) is measured against the MAP's final fit. What does it
   estimate?"* The slope of a not-yet-concentrated Bayes code against a plug-in reference. It is not an
   RLCT estimate. All Bayes configurations there have slopes 236–271, valid and invalid alike.
4. *"HE: the Bayes code's slope is 12.9, a quarter of 56. Is λ = 56?"* 56 is the regular count. The measured
   Bayes slope is in the interpolation regime (496 rotation parameters, n ≤ 1024) against its own final fit.
   It is consistent with λ < 56 but does not measure λ.
5. *"Isn't 'Bayes charges the coupling family' just that the coupling posterior hasn't concentrated?"* Yes,
   that is what "charges" means here: the prior mass of faithful coupling maps is small, so the predictive
   stays near the prior (IIA 0.5) at 1024 targets.

**Pitfalls.**
- The legend says the blue line is the "mixture of 4". The InterpBench point is the single-map MAP code
  (+1052, ratio 0.216). The mixture's +983 would give 0.267. A minor mislabel.
- "Not run" for GPT-2: say it every time the 22× appears.

---

### Slide 12: A complexity ladder of four explanations of the same variable (pill: plug-in κ (MAP))

**What the figure shows** (`ladder_panels.png`). Four horizontal bar panels share the same four rows:
Jacobian lens (no fitted map), learned subspace (trained GPT-2), coupling map (trained GPT-2), manufactured
(coupling, untrained). The panels are:
1. Fitted parameters on a log axis: 0, 12,288, 1,869,440, 1,869,440.
2. IIA: 0.98, 1.00, 1.00, 0.83, with a dashed line "accept at 0.8".
3. Plug-in κ on a log axis: 0 (exact), 49, 226, 988, with error bars.
4. Gain over label-only in bits: +3525, +3385, +2459, −1986.

Footer: GPT-2 IOI, single-map MAP code, 1016 coded targets, κ at n ≤ 1024, 3 seeds. Sources:
`regret_slopes_map_code.txt`, `gpt2_summary_single_map_map0.md`, `adversary_by_learner.txt`,
`jlens_gpt2.json`.

**Why it matters.** It is the slide "to keep if you keep one": IIA accepts all four, the parameter count
cannot separate the two coupling maps, and κ orders them in the same order as the code.

**Known vs ours.** All the project's. The J-lens construction is from Gurnee et al. (as cited); scoring it
as a zero-parameter alignment family is the project's.

**Likely questions.**
1. *"The coupling maps have 1.9 M parameters and n ≤ 1024. Isn't κ = 988 measured deep in the interpolation
   regime?"* Yes. The manufactured final fit has an in-sample loss of about 0.05 nats/target against a
   channel entropy of 0.36. It memorises the flipped labels, so the reference is optimistic and inflates
   the regret. What survives: at 32768 targets (still far below 1.9 M) the manufactured slope is 726–826
   (3 seeds) against 35–42 for the valid subspace, and the ordering is unchanged (slide 26). Say "κ here is
   a summary slope of the regret, not an asymptotic coefficient, for the coupling maps".
2. *"Is the manufactured regret logarithmic at all?"* Not cleanly. In the one-seed 32768 run the
   per-doubling regret increments after n = 2048 are 431, 522, 709 and 921 nats, then 249 at the last block.
   That is growing (log would be constant, linear would double each time). This is my reading of
   `h_a3_gpt2_large_n.txt`. Slide 14 settles the sign of the loss term: the fit term *favours* the
   manufactured alignment (−2.0k bits at 32768; held-out final-fit loss 0.390 vs 0.427 nats). So the whole
   separation is learning cost, and the cost is not a clean κ ln n. Answer: "a summary slope; the fit term
   goes the other way".
6. *"So does κ really order the code at large n?"* Only in a window of n. At 32768 the trained coupling map
   (κ 226) codes 371 bits *shorter* than the valid subspace (κ 49), because it fits better (calibration,
   slide 14). The ladder's code column is an n = 1024 statement. The κ column is the robust one.
3. *"The J-lens is 'parameter-free', but you chose its frame using the names' unembeddings. That is task
   knowledge the learned map has to learn."* Fair. The lens uses the output alphabet (names) and the
   network's averaged Jacobian; it codes nothing about them, and the code charges nothing for the choice.
   An honest two-part code would add the description of "which tokens and which layer", a few bits.
4. *"κ = 0 is a learner coefficient or a learning coefficient?"* Both: a fixed predictor has κ = λ = 0.
5. *"J-lens +3525 vs learned +3385: is 140 bits significant?"* The lens is deterministic, and the learned
   subspace has s.e. ±4.9 over seeds. The 140 bits is about what the learned map pays to learn: its regret
   at 1024 is 279 nats ≈ 403 bits, so its final fit is slightly better than the lens.

**Pitfalls.**
- The "1.00" IIAs are from a summary rounded to one decimal (displayed with two). Say "≈ 1".
- The notes say "the code length follows" the κ order. That is true at 1016 targets, not at 32768 for the
  trained coupling map (slide 14). Say "at a thousand targets".
- The κ panel's ordering is clean. The claim "the explanation with learning coefficient exactly zero is
  also the shortest" holds on GPT-2 at layers 11–12. At layer 10 the learned map wins (+2975 mixture vs
  +2821 lens; IIA 0.97 vs 0.80).

---

### Slide 13: κ is set by the data, not by the dimension of the family (pills: both)

Update 2026-10-22: new bullet "Plug-in prediction 75 / 71 / 177 (diamonds): it grows with d_V, the measured κ
does not"; the figure shows the d_V 64 diamond (176.7). Item 13 (d_V 4 is misspecified) is not addressed.

**What the figure shows** (`kappa_vs_dv.png`).
- x = subspace dimension d_V (4, 16, 64, log₂ scale); y = per ln n on a log scale (10 to 10⁵).
- The grey dashed nominal λ = d_V(d − d_V)/2: 1528, 6016, 22,528.
- Measured κ lines: red manufactured 817 / 988 / 652; green trained coupling 389 / 226 / 160; blue valid
  subspace 79.0 / 48.9 / 55.9; yellow untrained-twin subspace 26.7 / 44.4 / 48.4.
- Open blue diamonds mark the plug-in prediction at d_V 4 (75) and 16 (70.6), "d_V 64 pending".
- GPT-2 layer 11, n ≤ 1024; 2 seeds at d_V 4 and 64, 3 at 16. Sources: `regret_slopes_map_code.txt`,
  `curvature_gpt2_layer11*_heldout_fisher.json`.

**Why it matters.** It shows κ is not a parameter count: a fifteenfold increase in nominal λ leaves κ flat.
It is the cleanest single argument that the operative complexity is "Fisher-weighted directions".

**Known vs ours.** The project's.

**Likely questions.**
1. *"At d_V 64 you have 49,152 parameters and 1024 targets. Isn't the flatness the interpolation regime?"*
   The d_V 64 fit is in-sample below the channel floor (train NLL 0.19 vs 0.36). There is no 32768-target
   run at d_V 64. At d_V 16 the 32768 run shows no plateau (slide 26). Answer: "only shown below the
   parameter count at d_V 64".
2. *"The untrained twin's κ (27–48) is as small as the valid one's. So κ does not separate valid from twin?"*
   Correct. The twin is rejected by the loss term (+59 bits vs +3385), not by κ. The notes say so; say it
   aloud.
3. *"Is d_V 4 a valid alignment?"* No: IIA 0.27, gain +132, train NLL 2.43. Its κ = 79 includes
   misspecification drift. Do not present it as "the valid subspace at d_V 4".
4. *"Prediction at d_V 64?"* Now available: 176.7 vs 55.9 measured (see slide 10).

**Pitfalls.**
- "Valid subspace: κ = 79, 49, 56 at d_V = 4, 16, 64": qualify the 79.
- The "3–20×" claim covers manufactured and trained coupling, not the twin.

---

### Slide 14: Fit or Occam: which term carries the gain at 32768 targets (pill: plug-in κ (MAP))

**What the figure shows** (`gain_decomposition.png`). Three panels, one per control. Each has x = targets n
on a log scale (16 to 32768) and y = thousands of bits, with a dotted vertical line at n = 1024 ("headline
runs"). In each panel, black is the gain G(n) of the valid layer-11 subspace, blue squares are the regret
term (Δκ ln n + O(1)), and pink diamonds (dashed) are the fit term (≈ n·Δℓ). Bands are across seeds.
- **vs label-only.** Gain and fit lie on top of each other, rising to +116.6k (fit +117.0k). The regret term
  is flat near −0.4k. The valid subspace pays more learning cost than the Dirichlet code but wins on fit:
  falsification.
- **vs manufactured alignment** (coupling, untrained twin, layer 10). The regret term rises to +9.6k. The
  fit term is ≈ 0 until about 1024, then falls to −2.0k. The gain peaks at +8.3k at 16384 and ends at
  +7.7k.
- **vs coupling map on the trained net** (layer 11). The regret rises to +1.6k and the fit falls to −2.0k.
  The gain peaks near +0.9k around n = 1–8k, crosses zero between 16384 and 32768, and ends at −0.4k.
- Slide text: against the untrained twin the result is the same as against label-only (+112.4k, regret −57
  bits). Held-out final-fit losses: 0.390 (manufactured) vs 0.427 (valid) nats. The mechanism is the
  readout's ≈ 96 % on the name, inherited by the subspace swap, while flows saturate. At 32768 the
  manufactured alignment beats label-only by 109k bits.
- Source: `reports/stage1_ioi/gain_decomposition_gpt2_n32768.json`, made by
  `bitsev.experiments.gain_decomposition` on `results/stage1_ioi_gpt2_map_n32768` (single-map MAP code,
  3 seeds; targets hash 221ae7763d8eb720, a different draw from the 1024-target runs). I recomputed the seed
  means from the JSON. Gain / fit / regret in bits at 32768: label-only 116,619 / 116,984 / −365;
  manufactured 7,662 / −1,950 / 9,612; trained coupling −371 / −1,965 / 1,594; twin 112,388 / 112,445 /
  −57. The manufactured gain peaks at 8,285 at n = 16384; the trained-coupling gain peaks at 935 at 4096.

**Why it matters.** This is slide 4's decomposition measured. It is also the slide where the talk is most
honest. Against the controls that cannot reach the loss (label-only, twin, wrong locus) the verdict is
falsification and grows linearly. Against the controls that *can* (the expressive flows) the valid
alignment is the worse fitter. Its whole advantage is the learning cost, which grows only logarithmically
and is overtaken: by the trained coupling near n ≈ 20k, and by the manufactured alignment near 2·10⁵ on a
rough extrapolation (notes). This is the bridge to the SLT ask: the Occam term is real, and it is a
learner's term.

**Known vs ours.** The identity G = ΔFit + ΔReg is trivial bookkeeping. The measurement and the calibration
explanation are the project's.

**Likely questions.**
1. *"So at large n your score prefers the manufactured explanation?"* Eventually, on i.i.d. pairs from this
   pool, yes: at 32768 its gain over label-only is +109k bits and its i.i.d. IIA rounds to 1.0. Its
   disjoint-name IIA is 0.0, but so is the valid subspace's (0.1). Answer: "with enough i.i.d. targets an
   expressive map on an untrained network learns to read the name tokens out of the residual stream, and a
   code over i.i.d. pairs eventually rewards that. The complexity charge only wins in a window of n. This is
   the Sutter dilemma reappearing as n grows, priced in bits: the code charges a learning cost that grows
   like ln n, while a better fit pays back linearly."
2. *"Isn't the valid alignment's disadvantage just miscalibration? Give every predictor a temperature."*
   Yes, that is the mechanism, and the temperature run has not been done; the notes say so. The arithmetic
   is consistent (my check): with clean p = 0.96 on the name, the observed-law probability is
   0.95·0.96 + 0.05/32 ≈ 0.914, a loss of 0.090 nats, against the floor's 0.050. Over about 95 % clean
   targets that is ≈ 0.04 nats/target. That matches the held-out gap 0.427 − 0.390 = 0.037 and the fit term
   of about −2.0k bits over 32760 targets. Answer: "a one-parameter temperature per predictor would remove
   it; not run yet". Be ready to be asked why it was not built in. An orthogonal swap with a frozen readout
   cannot rescale logits, and that was a design choice: the map only routes.
3. *"Why does the fit term use final fits of 1.9 M-parameter flows that are in-sample?"* It does, but the
   held-out final-fit losses (0.390 vs 0.427) and the per-block code rates (slide 27, coded before fitting)
   tell the same story. At n = 1024 the in-sample optimism shows: valid vs manufactured is +5370 = fit
   −464 + regret +5834 (headline run `results/stage1_ioi_gpt2_map0`, per the deck). That split was not
   re-derived here.
4. *"Where does the 2·10⁵ come from?"* It is a rough extrapolation in the speaker notes: a regret
   difference growing like ln n against a fit difference growing like n. It is not a fit with an error bar.
   Say "roughly 10⁵ to 10⁶".
5. *"Does this undermine slide 12?"* It refines it. κ still orders the explanations: learning cost 0 < 49 <
   226 < 988. The code length is κ-ordered only while ln n·Δκ dominates n·|Δℓ|.
6. *"Does the calibration gap bear on F vs H (slide 9)?"* Possibly. This is my heuristic: the subspace
   predictor is systematically *under*-confident relative to the observed law, a concrete misspecification.
   In the sandwich note's Gaussian example an under-confident model has F < H. That is in the direction of
   F ≈ 0.02 H, but it is not a quantitative explanation.

**Pitfalls.**
- Say "Occam regime" only for the manufactured comparison, where Δℓ has the *wrong* sign for the valid
  alignment. The textbook Occam regime has Δℓ = 0.
- Slides 2 and 15 still say "longer than label-only" without "at 1024 targets". Say it.
- This is one learner (single-map MAP) on one testbed (GPT-2 IOI). The mixture learner at 32768 was not run.

---

### Slide 15: Four language models, one verdict, no retuning (pill: plug-in κ (MAP))

Update 2026-10-22: bullets now "Coupling on the trained net: 925–1665 bits behind the subspace at n = 1024 (on GPT-2
it overtakes near 20k targets, slide 14)" (the "equal or higher IIA" phrase is dropped from text and notes) and
"Manufactured, n = 1024: 1723–1986 bits longer than label-only (single-map code); 147 longer on Qwen3 (mixture,
IIA 0.71)". The figure has the Qwen3 manufactured bar (−147, IIA 0.71).

**What the figure shows** (`models_verdicts.png`). Grouped bars of gain over label-only (bits, −2900 to
4400) for four models.
- GPT-2 small, layer 11, MAP: valid +3385, coupling +2459, twin +59, manufactured −1986 (IIA 0.83).
- Qwen3-1.7B, layer 28, mixture: +3041, +2047, +692, manufactured "pending".
- Qwen2.5-3B, layer 34, MAP: +3129, +1690, +223, −1723 (IIA 0.59).
- Llama-3.1-8B, layer 30, MAP: +3101, +1436, −639, −1830 (IIA 0.51).
- Sources: `tier2_models.txt`, `gpt2_summary_single_map_map0.md`.

**Why it matters.** It shows external validity: the protocol transfers without retuning to models 20–60×
larger, and the label-only code is the same 4324.5 bits because the targets are identical.

**Known vs ours.** The project's.

**Likely questions.**
1. *"Seeds?"* Llama and Qwen2.5: one seed each. Qwen3: 3 seeds (mixture). GPT-2: 3 seeds. The decisive-gain
   rule cannot be applied to one-seed runs; their statistics are the anytime ones.
2. *"Why is the gain nearly identical across models?"* Because it is dominated by n Δℓ of a near-perfect
   predictor behind the 5 % channel on the same 1016 targets (`tier2_models.txt`, "Reading (3)"). It is
   not a measure of model quality.
3. *"'At equal or higher IIA' — really?"* At the plotted loci, no. Llama L30: coupling 0.87 vs subspace
   0.92. Qwen2.5 L34: 0.89 vs 0.94. Qwen3 L28: 0.96 vs 0.99. The write-up's conclusion itself says "at an
   IIA 0.05 lower". It is true at some other loci (e.g. Llama L32: 0.93 vs 0.71). Say "at similar IIA".
4. *"Coupling 925–1665 bits behind?"* At the plotted loci: GPT-2 925, Qwen3 994, Qwen2.5 1439, Llama 1665.
   But Qwen3 at L26 is only 683 behind.
5. *"Qwen3 manufactured?"* It has now finished (unarchived): −147 ± 53 bits (3 seeds, **mixture** learner,
   layer 26, IIA ≈ 0.7). It is notable because it is the only manufactured alignment on a large model
   scored under the canonical mixture, and it still does not beat label-only, unlike GPT-2 L10.
6. *"Last 2–4 layers?"* The valid gains of +3000 to +3400 are at Llama L28–30, Qwen2.5 L34, Qwen3 L26–28.
   The very last layers do worse in two models: Llama L32 +1709 and Qwen2.5 L36 +2702.

**Pitfalls.**
- "Manufactured: 1723–1986 bits longer than label-only" is an n = 1024 statement. On GPT-2 at 32768 it is
  +108,956 bits *shorter* (slide 14). The larger models were run only at 1024.
- Mixed learners across bars; the x-labels say so. Repeat it.
- Qwen2.5 L32 is a nice extra fact: a probe reads the name (0.90), but the valid family loses to its own
  twin there (−445 vs +183). The code distinguishes decodable from intervenable.

---

### Slide 16: Competence onset: the code tracks the task and keeps improving after it is solved (pill: plug-in κ (MAP))

**What the figure shows** (`pythia_onset.png`).
- Top panel: IOI accuracy against training step: 0.09 (1k), 0.43 (2k), 0.95 (4k), 1.00 (8k, 16k, 143k).
  Step 3k was not measured; the line interpolates through it.
- Bottom panel: code length in bits against step. The valid subspace at L20 (solid blue) runs 4748, 2833,
  1451, 1551, 896, 853, 910. L16 (dashed, open) runs 4800, 2857, 1544, 1755, 1023, 1032, 1545. The trained
  coupling at L20 (green) runs 5524 → 1796 → 2226.
- Horizontal lines: untrained twin 3751 (yellow dashed), manufactured 4968 (red dashed, IIA 0.78),
  label-only about 4325 (grey).
- Blue numbers give the IIA of L20.
- Single-map MAP code, one seed. Source: `pythia14_checkpoint_series.txt`.

**Why it matters.** It is the only training-dynamics result and the natural bridge to developmental
interpretability (LLC over training). The notes end with a question to the audience.

**Known vs ours.** Known: LLC and stage transitions (Hoogland et al. 2024, as I recall); Pythia checkpoints
(Biderman et al. 2023). Ours: the measurements.

**Likely questions.**
1. *"One seed: how much is noise?"* On GPT-2 the valid subspace's seed s.e. is about 5 bits, so the 655-bit
   drop between 4k and 8k is very unlikely to be seed noise. But there is no Pythia replicate.
2. *"Why does the trained net lose to the twin at step 1k?"* The repo's reading is that the untrained
   residual stream still carries the raw name tokens (twin IIA 0.39), while early training disrupts this
   copy before the task is computed. It is an interpretation, not tested.
3. *"Have you measured the network's own LLC along these checkpoints?"* No. That is the question for the
   audience.
4. *"Manufactured: 400 steps, not 4000?"* Yes, it is the weaker adversary. It reaches IIA 0.78 and loses
   643 bits. It does not depend on the checkpoint, because it lives on the twin.
5. *"Why does L16 get worse at 143k?"* The repo reads this as the name representation migrating to later
   layers late in training (1545 bits, IIA 0.94). This is speculative.

**Pitfalls.**
- "Keeps improving after it is solved": from 4k (0.95) to 8k (1.00) the accuracy still rises, so "after it
  is solved" is partly "while the last 5 % is solved". The cleaner phrase is "keeps improving after
  accuracy is at 0.95".
- The interpolated accuracy at 3k on the plot is not data.

---

### Slide 17: MIB arithmetic: the code picks the family that is faithful at each locus (pill: plug-in κ (MAP))

**What the figure shows** (`mib_arithmetic.png`). Two panels: Llama-3.1-8B (whole-number answer) and
Gemma-2-2b (first answer digit).
- x = layer; y = gain over the trivial floor in bits (−3000 to 3000). The floor is the layer-0 subspace,
  i.e. the model's own answer on the base.
- Blue = 16-dim subspace; green = coupling map on the trained network. Numbers at the points are IIAs.
- Llama: the subspace stays at IIA 0.44–0.51 everywhere but rises to +1053 (L20) and +1049 (L24) over the
  floor. The coupling runs +1714 (L16, IIA 0.69), **−839 (L20, IIA 0.96)**, +2348 (L24, 0.94), +1370
  (L28, 0.86), −2300 (L32, 0.43).
- Gemma: the subspace peaks at L20 (+754, IIA 0.96); the coupling at L20 is +499 (0.97).
- Red text at the bottom: manufactured, Llama L24 −4875 vs floor (IIA 0.14); Gemma L20 −1794 vs floor
  (IIA 0.62).
- Source: `stage1_mib/arithmetic.txt`; one seed; single-map MAP code.

**Why it matters.** It is a task where the variable (carry) is a genuine intermediate variable, not the
output, and where the code ranks the linear and non-linear families in opposite orders on two models.

**Known vs ours.** MIB (Mueller et al. 2025). The rest is the project's.

**Likely questions.**
1. *"On Llama the subspace gains about 1000 bits over the floor at chance IIA. What does the gain
   measure?"* Calibration of the predictor on the noisy law, not reading the carry
   (`arithmetic.txt`, Reading (2)). **This shows a positive gain is not evidence that the variable is read.**
   The meaningful comparison is the coupling's +2348 vs the subspace's +1049 at L24.
2. *"At L20 the coupling reaches IIA 0.96 but codes 839 bits worse than the floor."* The non-linear learner
   needs most of the 1024 targets to find the map there, and the prequential code charges for that. So the
   code measures *sample efficiency*, not faithfulness alone. Have this ready.
3. *"Does the manufactured alignment matter here?"* Weakly. It does not fool IIA either (0.14 and 0.62,
   below the 0.78 floor).
4. *"MIB found DAS at chance on carry?"* The repo says the floor (0.47–0.54) matches "the 52 % reported for
   DAS on this task in MIB". Check the MIB paper before quoting.

**Pitfalls.**
- "The coupling map reads the carry at layers 20–24": at L20 its code loses to the floor. Say "at layer 24".
- "Picks the faithful family" relies on IIA as the arbiter of faithfulness, which the talk otherwise
  criticises. The consistency argument is that the expressive family on the untrained twin fails (IIA 0.14).

---

### Slide 18: What this asks of singular learning theory

Update 2026-10-22: card 1 now says "κ has the order of ½ tr(F H⁻¹) … but not its trend (within 3×)".

**On the slide.** Four boxes.
1. Regret of plug-in learners in singular models: κ ≈ ½ tr(F H⁻¹), with κ < λ, κ > λ and κ = MN/2 > λ
   measured.
2. "Regular, yet Fisher-free": F ≈ 0.01 H; what is the right effective coefficient, and how does λ(n)
   cross over?
3. Estimators: SGLD LLC tracked κ on the oracle and read λ on RRR at n = 4096, diverges at 65536, never
   plateaus on saturated readouts.
4. The free energy of an explanation: the J-lens (λ = 0) is shortest; which priors on alignment maps would
   make a Bayes code competitive; is λ(n) the right complexity?

**Why it matters.** It is the ask. It must be crisp and must not overclaim the numbers behind it.

**Known vs ours.** ML/MAP generalisation in singular models is partly known (Watanabe's book; Fukumizu).
Say "there are results for ML in singular models, but I don't know one that covers a gradient-trained,
budget-limited plug-in with a saturated readout".

**Likely questions.**
1. *"Isn't there already a theory: Watanabe's results for the MLE/MAP in singular models?"* Partially. Those
   results concern the exact ML/MAP and are expressed through Gaussian-process maxima. Here the learner is
   budget-limited Adam with cold starts, so the measured κ includes optimisation. The RRR case uses
   *unconstrained* least squares (slide 6), which is regular.
2. *"'Regular yet Fisher-free' is a contradiction under realizability."* Yes. See slides 7 and 9: the oracle
   directions are quartic, so they are singular, not regular. Re-frame box 2 as "Fisher-free directions:
   are they singular (higher-order KL) or misspecification?". It becomes a better question for this
   audience.
3. *"Why should we believe the oracle LLC 'tracks κ'?"* 0.85, 1.14, 1.78, 2.57, 4.20 against k/2 = 0.5, 1,
   2, 3.5, 5.5 at n = 2048. It is within 25 % at k ≥ 2, but 70 % high at k = 1. If the true λ is
   k/2 + (11 − k)/4, the LLC at β = 1/ln n with tight localisation would be expected to see something
   between k/2 and that. It is one setting (step 10⁻⁴, localisation 10); smaller steps under-mix (0.1–1.5).

**Pitfalls.**
- "κ = MN/2 > λ (least squares in RRR)": say "unconstrained least squares".

---

### Slide 19: Open questions for the discussion

Update 2026-10-22: first question now "κ vs ½ tr(F H⁻¹): measured 49–79 at every GPT-2 locus and width,
predicted 52–177. Inexact minimiser, training-set Hessian of an interpolating fit, or a singular correction?"

**On the slide.** Six bullets: κ below prediction (49 vs 71; 79 vs 75); does κ settle (InterpBench 23–42
at 32768, rising); a closed form for sharp-readout λ(n) (s^1.3); anytime validity against composite
controls; training dynamics (LLC vs Pythia 1k–4k); priors for explanations (Bayes better at λ = 56,
hopeless at 512 and 6016).

**Update before the talk.** The first bullet is out of date: L10 is now 65 measured vs 52.5 predicted, L12
65 vs 112, d_V 64 56 vs 177. Rephrase as "prediction right to a factor 1.2–3, not in the details".
"Hopeless at 6016" was not run, so say "predicted hopeless". A seventh question is worth adding aloud
after slide 14: *is there a principled way to make the complexity charge win at all n?* For example, a
calibration-invariant score (a temperature per predictor), or a learner whose coefficient grows with the
map's capacity.

**Likely questions.**
- *"InterpBench '23–42'?"* It mixes widths: d_V 4 gives 22.6–31.7 and d_V 1 gives 32.6–42.0 at 32768,
  depending on the fit range.
- *"Composite null?"* Universal inference (Wasserman et al. 2020) needs the control family's maximised
  likelihood. For neural control families only a lower bound from the optimiser is available, so no
  guarantee is claimed.

---

### Slide 20: Summary

Update 2026-10-22: bullet 3 now "κ counts Fisher-weighted directions (½ tr(F H⁻¹) gets its order) … (n ≤ 1024)";
new bullet "Against a manufactured alignment the fit favours the forgery; the verdict is the learning cost alone
(32768 targets)"; font 20.

**On the slide.** Four statements: gains are log Bayes factors against controls; for Bayes codes the
complexity is λ (exact on RRR, scale dependent on the oracle, "huge on trained networks (6016 on GPT-2)");
the codes that work are plug-in, κ ≈ ½ tr(F H⁻¹), 0 / 49 / 988; a theory of plug-in learners is needed.

**Pitfalls.**
- "6016 on GPT-2" should be "nominally 6016". "Scale dependent on the oracle" may be better said as
  "singular on the oracle (quartic directions)", if the quartic reading holds.
- "Gains are log Bayes factors": for plug-in codes it is a ratio of two sequential forecasters'
  likelihoods. It is an e-value only against the simple null.
- "988 for the manufactured alignment" ordering explanations "by a factor of twenty" (notes): this is the
  learning-cost ordering. The total code ordering holds only in a window of n (slide 14). Consider adding
  "at the sample sizes of interpretability practice" to the summary sentence.

---

### Slide 21: Backup divider

Order: noise channel, batching, Bayes-code settings, InterpBench, GPT-2 to 32768, code rate per block,
J-lens, learner dependence, LLC, Pythia and MIB tables, compute.

---

### Slide 22 (backup): Every predictor must code the observed label law

**On the slide.** p_obs = (1 − ε)p + ε/K with ε = 0.05. "What went wrong first": coding the clean law, the
correct alignment paid about 39 bits per flipped label (92 % of its code after position 128) on InterpBench
18, and every verdict had the wrong sign. "Why randomized response": full support, "representable at finite
logits, so the model is realizable and Watanabe's expansion applies".

**Pitfalls.**
- **The "Why randomized response" box contradicts the corrected write-up** (§2.5 and §3.1). Realizability at
  finite logits holds only on the oracle; networks reach the law only at infinite margin, and Watanabe's
  expansion applies approximately. The speaker notes repeat the old claim. Say instead: "it caps the cost
  per target at log₂(K/ε) and gives the target law full support; on networks the law is approached, not
  attained".
- Useful extra fact: the fixed target seed puts only 3 flips in the first 128 targets and 53 in 1024
  (write-up §9.2, "quiet prefix"). This hid the problem early.

**Likely question.** *"Isn't coding the observed law just using the right likelihood?"* Yes. The finding is
that leaving it out reversed every verdict: the frozen readout is sharper than the channel and an orthogonal
map cannot temper it.

---

### Slide 23 (backup): Batching correction for geometric block schedules

**On the slide.** Cumulative excess ≈ κ (r − 1)/ln r · ln n, with the derivation. Factors 1.4427
(r = 2), 1.1204 (r = 1.25), 2.1640 (r = 4). Gaussian-mean check: 0.5001 / 0.7213 / Bayes 0.7208. RRR
singular Bayes code (n = 4096, 8 seeds): per-sample / doubling / quadrupling 3.15/3.13/2.84 for (3,3,3,0),
5.75/6.00/6.24 for (4,4,4,0), 2.11/1.94/1.98 for (2,2,2,2), 7.18/7.44/7.68 for (4,4,4,2).

**Known vs ours.** The derivation is elementary; the checks are the project's.

**Pitfalls / questions.**
- The (3,3,3,0) values sit around 3.0, below λ = 3.5, under all three schedules. The schedules agree with
  each other (s.e. 0.25–1.0), which is the claim; they do not all agree with λ at n = 4096.
- **Not every slope in the talk is corrected.** The "local slopes" quoted on slide 26 and the "slope 45 per
  ln n" annotation on the InterpBench figure (slide 25) are raw (see the inconsistency list).

---

### Slide 24 (backup): Bayes codes: settings and what had to be repaired

**On the slide.** HMC settings per testbed, and four failures on saturated readouts: Laplace collapses to the
prior (held-out loss 2.27 nats, worse than uniform 1.10 = ln 3); the gauge shrinks the frame to norm
1.6·10⁻⁴, fixed by QR retraction; Langevin from the MAP with the step as a hedge knob (moves the GPT-2
adversary by about 40 bits); exp(S) is flat only to first order (the Laplace code needs the gauge-free map,
2.75 ± 0.13 vs λ = 2.5 on the oracle).

**Pitfalls.**
- **"Hierarchical equality: … chains continue" is wrong.** `stage1_toy/bayes_code_hmc.txt` and the
  write-up say the chains **restart** at every block (warm_start False), as on InterpBench.
- **The speaker note is outdated.** It says the archived InterpBench text header says the chains continue.
  The current `case18_bayes_code_hmc.txt` header says "chains restart at every block (warm_start False)"
  (corrected in commit 99efeac). There is no discrepancy left to mention.
- The oracle check (2.75 ± 0.13 vs 2.5) was done at d = 6 with all k = 5 directions readable, a regular
  case. It does not test the singular k < d − 1 situation.

---

### Slide 25 (backup): InterpBench case 18: interpolation regime, then slow growth (pill: plug-in κ (MAP))

**What the figure shows** (`interpbench_regret_vs_n.png`, an earlier figure in a different style).
- x = targets coded n (2⁴ to 2¹³, log₂); y = regret of the MAP code in nats (0–700).
- Blue = valid locus resid_2; orange = misspecified locus resid_1. Dashed lines are the 1024-target run and
  solid lines the 8192-target run; the legend does not show the difference.
- A vertical dotted line marks "frame parameters (1040)". A black dotted segment over the last two blue
  points is annotated "slope 45 per ln n".
- Valid, 1024 run: 49 nats at 1024. Valid, 8192 run: 144 nats at 8192. Misspecified: 529 (1024 run), 684
  (8192 run).

**Slide text.** d_V 4, width 260, 1040 frame parameters, nominal λ 512. MAP κ: 8.4 ± 2.9 (n ≤ 1024),
16.8 ± 2.8 at 8192, 22.6–31.7 at 32768 depending on the fit range. Misspecified 91–186. The Fisher puts 90 %
of its trace in 7 of 1024 directions; the Hessian is indefinite (654 negative eigenvalues).

**Why it matters.** It is the cautionary tale: a flat regret first read as "saturation" was the
interpolation regime. It supports the project's insistence on the "below the parameter count" caveat.

**Likely questions.**
1. *"So which number is κ?"* None has settled. The corrected slope rises with the fit range: 22.6, 27.1,
   31.7 on n ≥ 64, 512, 2048 at d_V 4.
2. *"Why is the 8192 run's regret at n = 1024 different from the 1024 run's?"* The two runs use different
   target draws and different references (the final fit on 8192 vs 1024 targets).
3. *"Why can't you evaluate ½ tr(F H⁻¹)?"* The fitted point is a flat, indefinite plateau. The mean gradient
   is 4.5·10⁻⁵, the top Hessian eigenvalue 6.1·10⁻⁴, and there are 654 negative eigenvalues
   (`case18_curvature.txt`).

**Pitfalls.**
- The "slope 45 per ln n" annotation is a **raw** local slope; batching-corrected it is about 31 (Δregret
  of about 32 nats over the last doubling). The text and the figure disagree by the batching factor.
- The colours do not follow the deck's convention (orange = Bayes); here orange is the misspecified locus.

---

### Slide 26 (backup): GPT-2 to 32768 targets: the separation persists (pill: plug-in κ (MAP))

**What the figure shows** (`gpt2_regret_vs_n.png`, earlier style).
- Log–log axes: regret of the MAP code (nats) against n (2⁴ to 2¹⁵). Dotted lines are the n = 1024 runs and
  solid lines the n = 32768 runs.
- Series: blue valid subspace L11; orange coupling L11 (trained); purple untrained-twin subspace L11; red
  manufactured (twin, L10). A vertical line marks the frame parameters (12288).
- The manufactured curve reaches about 6775 at 32768; the coupling about 1800; valid 365; twin 442.

**Slide text.** Valid 45–50 per ln n from n = 32 to 16384 (one seed), 35–42 at 32768 (3 seeds).
Manufactured 6775 nats by 32768, with a local coefficient peaking at 1700–2450 near n = 1000. No plateau at
12,288.

**Likely questions.**
1. *"Is 45–50 corrected?"* The fitted, corrected slopes are 48.9 (n ≤ 1024, 3 seeds), 44.6 (8192), and 37.0 /
   30.9 / 19 (32768, one seed, three fit ranges). The per-doubling regret increments of the valid curve from
   32 to 16384 average about 39 nats, which is the corrected local coefficient. The quoted "local slopes
   20 … 77" and the manufactured "1700–2450" are **raw** (per ln n without the 1.4427 factor); corrected,
   the manufactured peak is about 1180–1700.
2. *"Why does the valid regret drop at the last block?"* The final fit on 32768 targets (400 cold-start Adam
   steps on minibatches of 256) is slightly worse than the block predictor: a limitation of the
   best-in-family proxy.
3. *"The twin's regret grows faster at large n (41 → 66)?"* The repo reads this as the twin's copy of the
   name tokens being read out progressively.

**Pitfalls.**
- "18× the valid alignment's" (notes) is 6775/365 = 18.6 on one seed, or 7081/419 = 16.9 on three.
- The title "the separation persists" is true of the **regret** (learning cost). The total code separation
  from the trained coupling map does not persist: it flips sign between 16384 and 32768 (slide 14).

---

### Slide 27 (backup): Code rate per block: the fit term without final fits (pill: plug-in κ (MAP))

**What the figure shows** (`code_rate_per_block.png`).
- x = targets n at the end of each doubling block (16 to 32768, log scale). y = code rate in the block minus
  the channel floor, in nats per target, on a log scale (10⁻² to about 6). Title: "channel floor 0.360 nats
  per target (ε = 0.05, 32 names)".
- Blue = valid subspace (layer 11); green = coupling map on the trained network; red = manufactured
  alignment. There are error bars across 3 seeds.
- Valid: falls fast (≈ 0.7 above the floor by n = 32, ≈ 0.17 by 1024), then levels at 0.04–0.05; last
  block 0.048.
- Coupling: stays at 2–3 until about 256, crosses the valid curve around 2–4k, and ends at 0.011.
- Manufactured: stays at 3–5 above the floor until 1024 (the headline regime), drops to ≈ 0.5 at 2048, and
  ends at 0.021.
- Slide text: on clean targets the final fits' median loss is 0.092 nats for valid vs 0.050–0.053 for the
  flows (floor 0.050); IIA 0.997–1.0 for all three; at n = 1024, valid vs manufactured is +5370 = fit −464
  + regret +5834.
- Sources: `gain_decomposition_gpt2_n32768.json` (my recomputation of the last-block rates: 0.4079,
  0.3815, 0.3713 nats, i.e. 0.048, 0.021 and 0.011 above 0.3602) and `results/stage1_ioi_gpt2_map0`
  for n = 1024.

**Why it matters.** It removes the objection that slide 14 depends on in-sample final fits. Each block is
coded before it is fitted, so its rate is a held-out loss. The flows end up *below* the valid subspace's
rate. The valid subspace is under-confident, not wrong.

**Likely questions.**
1. *"Is the valid subspace's 0.048-nat excess irreducible within its family?"* Within the orthogonal-swap
   family with a frozen readout, yes, as far as the data show. It levels off from n ≈ 8192. The readout's
   own confidence (≈ 96 % on the name) caps it. A temperature would remove it (not run).
2. *"Then which explanation is 'right'?"* The code over i.i.d. pairs measures how cheaply a configuration
   reproduces the algorithm's outputs. At large n that rewards calibration. Faithfulness questions still
   need the controls (untrained twin, second twin) and shifted tests. On disjoint names all learned maps
   fail on GPT-2.
3. *"Why is the manufactured rate flat at 3–5 nats until 1024 and then drops?"* This is my reading of the
   figure: with 16 names and a 1.9 M-parameter flow, the map needs on the order of a thousand pairs before
   it generalises across base/source combinations. This is the "learning cost" that carries the
   n = 1024 verdict.

**Pitfall.** The median clean-target loss numbers come from the final fits (in-sample) per the slide text;
the block rates are the held-out evidence. Keep the two apart if asked.

---

### Slide 28 (backup): Jacobian lens and Jacobian consistency

**What the figure shows** (`jacobian_consistency.png`).
- Bars of the median JC with whiskers to the 90th percentile, for layer 10 and layer 11.
- Green J-lens: 0.26 (IIA 0.80) and 0.15 (0.98). Blue learned subspace: 0.38 (0.97) and 0.24 (0.99). Red
  manufactured: 2.05 (0.80) and 2.24 (0.84).
- 90th percentiles reach 4.84 and 4.04 for the manufactured alignment. Source:
  `jlens_gpt2_jacobian_consistency.json`.

**Slide text.** J-lens: no fitted parameters, κ = 0. GPT-2 L11: IIA 0.98, +3525 bits (+3529 with all 32
names); L12 +3626. Learned subspace +3302 (mixture) or +3385 (MAP). It fails on every untrained twin (IIA
0.04). It is faithful at the last 2–4 layers of Qwen2.5-0.5B, Pythia-1.4b and Qwen3-1.7B.

**Likely questions.**
1. *"Is JC on the manufactured alignment computed with the twin's own Jacobian?"* Yes, on the twin's
   averaged Jacobian.
2. *"The lens with training names fails on unseen names (0.07): so does the learned map?"* Yes. Generalisation
   across names is a property of which directions are in the frame.
3. *"Is a successful J-lens interchange evidence of use?"* No. It shows the span *can* carry the variable;
   the Makelov caveat applies to every interchange alignment (write-up §2.6).

**Pitfalls.** "Six to ten times": the median ratios range from 5.4 (2.05/0.38) to 15 (2.24/0.15).

---

### Slide 29 (backup): The learner, not a complexity charge, decides the absolute verdict (pill: plug-in κ (MAP))

**What the figure shows** (`learner_decides_verdict.png`). Horizontal bars of gain over label-only for the
GPT-2 layer-10 manufactured alignment (3 seeds):
- Mixture of 4 (canonical): +704 (IIA 0.86).
- One member: −1085 (0.58).
- MAP only (AdamW cold start): −1986 (0.83).
- MAP + Langevin (MAP-scale step 10⁻⁴): −1948 (0.83).
- Error bars are s.e. Footer: the valid L11 subspace is +3302 (mixture), +3385 (MAP), +3350 (MAP +
  Langevin). Source: `adversary_by_learner.txt`.

**Why it matters.** It is the honesty slide. The absolute verdict depends on whether one map or four are
averaged. The hedge moves about 40 bits, so this is not a Bayes complexity charge.

**Likely questions.**
1. *"Did you switch from the preregistered mixture to the MAP after seeing the adversary win?"* The
   chronology: the mixture is canonical; the adversary sweep under it found +704 (L10) and +290 (L9), so the
   absolute rejection criterion **failed as first drafted**. It was amended post hoc to a comparative
   criterion against a second twin, which passes only narrowly at L10 (+349 ± 195, 1.8 s.e.). The
   single-map runs came after, while investigating a Bayes/Laplace charge. All are reported (write-up §6).
   Say this plainly.
2. *"One member vs MAP-only differ by 900 bits?"* Per the write-up they are the same learner up to
   random-number streams, so the difference is between-run spread (1.3 combined s.e.).
3. *"Why does averaging help the adversary?"* Four confidently wrong maps are wrong on different pairs, so
   the average spreads its mass: 2.6 bits/target in the last block vs 5.6 for the MAP. Maps that have found
   the valid subspace agree, so averaging barely matters for them.
4. *"The comparative margin?"* About 2600 bits (paired: 2594 ± 65) under the mixture; about 5400 (3385 +
   1986) under the MAP. The slide says "≈ 5300".

**Pitfall.** "A single map pays 6–7 bits per target throughout": the MAP's last block is 5.6.

---

### Slide 30 (backup): The SGLD local learning coefficient as a cross-check (pills: both)

**What the figure shows** (`oracle_effective_coefficient.png`, earlier style).
- x = "effective coefficient = readable directions / 2" (0.5 to 5.5); y = estimate.
- A dashed identity line and a dotted line at the nominal λ = 5.5.
- Blue dots are the MAP regret slope at n ≤ 2048: 0.53, 1.46, 1.94, 2.24, 4.49. Red open squares are the
  SGLD LLC: 0.85, 1.14, 1.78, 2.57, 4.20.
- d = 12, n = 2048; step 10⁻⁴, localisation 10.

**Slide text.** Estimator λ̂ = nβ(E[Lₙ] − Lₙ*) with β = 1/ln n, localised at the MAP. RRR at n = 4096:
3.38 / 6.25 / 2.16 / 7.91 for λ = 3.5 / 6 / 2 / 7.5; it diverges at 65536. InterpBench: 0.96 ± 0.69 to
27 ± 14 depending on settings, with no plateau.

**Likely questions.**
1. *"Why should the LLC agree with κ rather than with the Bayes slope?"* At β = 1/ln n with localisation,
   the estimator measures the local volume scaling of the empirical loss around the MAP at one scale. In the
   oracle's outside directions the empirical loss rises very slowly (quartic, or a tiny empirical
   curvature), so the localised chains may not explore far enough to see them at this n. It is plausible
   that it reads something between k/2 and the true λ. Say "one setting, interesting, not understood".
2. *"Why does it diverge at large n?"* The drift nβ∇L grows with n, so the step must shrink with n. This is
   standard SGLD tuning.
3. *"Isn't the RRR check at n = 4096 with a 4000-step chain somewhat tuned?"* One step size was used for all
   shapes. The value at 7.5 is 7.91 (5 % high), and at 3.5 it is 3.38.
4. *"InterpBench: localisation 1 vs 10?"* In `case18_llc.txt` γ = 1 is the weaker pull (the more stable
   reading, 2.22 ± 0.18). The write-up's text calls it "tight localisation (strength 1)". Avoid the word
   "tight".

**Pitfall.** "Close to k/2": at k = 1 the LLC is 0.85 against 0.5 (70 % high).

---

### Slide 31 (backup): Pythia-1.4b checkpoints table

This is a transcription of `pythia14_checkpoint_series.txt`. I checked every cell and they match.
Label-only is about 4325 bits. There is one seed, step 3000 accuracy was not measured, and checkpoints 32k
and 64k were not run.

---

### Slide 32 (backup): MIB addition table

This is a transcription of `stage1_mib/arithmetic.txt`. I checked every cell and they match. Label-only is
7602.7 bits (Llama, 199 symbols) and 2280.9 bits (Gemma, 10 symbols). Manufactured: Llama −2160 (IIA
0.14), Gemma −1075 (IIA 0.62). Have ready: the subspace vs the floor on Llama rejects the floor only after
41–145 targets, and the Gemma L20 subspace rejects every control by n ≤ 18.

---

### Slide 33 (backup): Compute

Update 2026-10-22: the Qwen3 line now reads "three seeds with the mixture learner on A100s, including the 4000-step
manufactured-alignment arm".

The times match the write-up (§4.8). Out of date: "Qwen3-1.7B … still running their manufactured-alignment
arm"; it has finished.

---

## 5. The questions most likely to hurt, ranked

1. **"Is λ = 6016 real?"** The Hessian is an empirical Hessian at an interpolating MAP (12,288 parameters,
   1024 targets, train NLL below the channel entropy). Under (near-)realizability the population Hessian
   equals the Fisher, so F ≈ 0.02 H says most of H is not population curvature. The oracle shows the same
   effect concretely: the "curved, Fisher-free" directions there have **quartic** population KL (exponent
   4.0, checked numerically), so the oracle's true λ is k/2 + (11 − k)/4, which fits the HMC slopes at
   n ≤ 16384. Prepared answer: "6016 is the regular count and an upper bound. On the oracle the Fisher-free
   directions are quartic, so they cost a quarter each, not a half. On GPT-2 I have not computed the
   held-out Hessian, so 'regular' is not established. What is established is that the plug-in pays about
   50 and that a Bayes code at this n is uncompetitive (InterpBench HMC)." Best defence: run the held-out
   Hessian (one A100, about 42 min) and the oracle KL-exponent check before the talk.
2. **"So your headline is an n = 1024 artefact?"** At 32768 targets the manufactured alignment beats
   label-only by 109k bits, and the trained coupling map beats the valid subspace by 371 bits. Against the
   manufactured alignment the valid subspace is the *worse* fitter (held-out 0.427 vs 0.390 nats), and its
   +7.7k margin is pure learning cost, peaking at 16384 and extrapolated to vanish near 2·10⁵ targets. The
   deck now shows this (slide 14). The danger is that it undercuts slides 2, 12, 15 and 20 unless every
   "longer than label-only" is said with "at a thousand targets". Prepared answer: "At large n the code
   ranks explanations by how well they reproduce the algorithm on i.i.d. pairs, and expressive maps win on
   calibration, because GPT-2 itself is only 96 % sure and an orthogonal swap inherits that. The complexity
   charge decides in a window of n that covers the sample sizes used in practice. A per-predictor
   temperature would remove the calibration advantage; I have not run it. The comparative verdict holds at
   every n measured." Do not claim the comparative verdict holds asymptotically.
3. **"Your RRR plug-in is unconstrained least squares."** `OLSLearner` ignores H, so it is a regular
   MN-parameter model and MN/2 is automatic. The singular comparison (the rank-H ML plug-in) was not run.
   Prepared answer: "yes, the slide validates the Bayes pipeline and shows what a plug-in that ignores
   singular structure pays; the rank-constrained ML plug-in is exactly the open question on slide 18".

Also likely, in rough order:
- **"A Bayes code cannot cost 65,000 nats"** (slides 9 and 11). This is a fix to make rather than a question
  to field. With the observed-law floor every target costs at most ln(K/ε) = 6.46 nats, so
  Fₙ ≤ 6,570 nats ≈ 2.2× label-only at n = 1016. The "≈ 22×" is λ ln n outside its regime of validity.
  Prepared answer: "λ ln n is 22× label-only; the code itself would sit near the prior predictive, at most
  2.2×. InterpBench shows the same directly: 1.1–1.8× label-only at 1024."
- *Learner dependence and post hoc amendments* (slides 2 and 29). Prepared answer above.
- *Interpolation regime for the coupling κ values* (slides 12 and 13). At 1024 the manufactured final fit
  is in-sample at about 0.05 nats/target against a 0.36 floor. The regret separation persists to 32768,
  but those maps are still 50× over-parametrised and their regret grows faster than log.
- *"IIA 0.51/0.59 is not 'accepted'"* (slide 2).
- *"A gain at chance IIA"* (slide 17, Llama subspace about +1000 over the floor).
- *"IOI's variable determines the output, so this is output patching"*. The write-up concedes it; the
  intermediate-variable tests are HE and MIB carry only.
- *"Strawman adversary?"* It is Sutter et al.'s construction at their scale (1.9 M parameters, 4000 steps,
  IIA 0.86). More budget makes it worse (16000 steps: −744). On HE and InterpBench even a 10× adversary is
  not convincing (IIA ≤ 0.6). No *code-aware* adversary (one that chooses a map family with a good
  inductive bias for this code) was tried. Say so.

---

## 6. Inconsistencies found (slides, notes, figures, sources)

Slide numbers refer to the deck; the text and notes live in `reports/talk_slt/scripts/build_deck.js`.

0. **"Longer than label-only" is an n = 1024 statement (slides 2 and 15 text and notes; slides 12 and 20
   notes; `iia_vs_code.png`; `models_verdicts.png`; write-up abstract and §5).** [FIXED 2026-10-22 on slides 2, 15, 20 and both verdict figures (write-up not changed).] At 32768 targets (GPT-2,
   single-map MAP code, 3 seeds; `results/stage1_ioi_gpt2_map_n32768/summary.md`) the manufactured
   alignment gains **+108,956 ± 239 bits** over label-only. The trained coupling map gains +116,989 ± 126
   against the valid subspace's +116,619 ± 83, so it beats the valid subspace by **371 ± 190 bits**. Slide
   14 states this; the other slides do not qualify it.
1. **Realizability claim (slides 3 notes, 22 text and notes).** "Makes deterministic targets realizable at
   finite logits … Watanabe's expansion applies" contradicts the corrected write-up
   (`stage1_results.tex` §2.5 and §3.1, commit abfa067 "channel floor instead of finite-logit
   realizability").
2. **HMC chains on hierarchical equality (slide 24 text).** "Chains continue" is wrong:
   `reports/stage1_toy/bayes_code_hmc.txt` and write-up §10.4 say the chains restart at every block.
3. **Outdated speaker note (slide 24).** It says the archived InterpBench header says "continue". The
   current `reports/stage1_interpbench/case18_bayes_code_hmc.txt` says "restart" (fixed in 99efeac).
4. **Pending results now exist (slides 10, 13, 19; `predicted_vs_measured.png`; `kappa_vs_dv.png`;
   write-up §10.3).** [FIXED 2026-10-22: archived under reports/stage1_ioi/, slides 10, 13, 18, 19, 20 and both figures updated (write-up not changed).] L10 κ = 65.1 ± 2.9 (pred 52.5) and L12 κ = 65.1 ± 0.7 (pred 111.7), from
   `results/stage1_ioi_gpt2_map_l10_l12/`. d_V 64 prediction 176.7 vs measured 55.9
   (`results/ioi_curvature_gpt2_l11_dv64/curvature.json`). Neither is archived under `reports/`. The slide
   19 claim "κ below its prediction" no longer holds in general.
5. **Qwen3 manufactured arm finished (slides 2, 15, 33; `models_verdicts.png`; `tier2_models.txt`; write-up
   §4.4, §5, §11).** [FIXED 2026-10-22: archived in tier2_models.txt; slides 2, 15, 33 and both verdict figures (write-up not changed).] −147.0 ± 53.3 bits, mixture, L26, IIA ≈ 0.7, from
   `results/stage1_ioi_qwen3_17b/summary.md`. Not archived.
6. **"Prediction within the seed errors" on the oracle (slide 10 text).** [FIXED 2026-10-22: slide 10 now says k ≤ 7 and gives the k = 11 miss.] k = 11: 5.32 ± 0.20 vs 4.47
   (about 4 s.e.); k = 7: 3.73 ± 0.37 vs 3.26. Sources: `oracle/hmc_n16384/ksweep_map_k*.json`,
   `oracle/hessian_vs_fisher.txt`. The figure shows the k = 11 miss.
7. **"At equal or higher IIA" (slide 15 text and notes; `tier2_models.txt` Reading (2)).** [PARTLY FIXED 2026-10-22: phrase dropped from slide 15 text and notes; tier2_models.txt Reading (2) and the Qwen3 L26 range unchanged.] At the plotted
   loci the coupling IIA is lower: Llama 0.87 vs 0.92, Qwen2.5 0.89 vs 0.94, Qwen3 0.96 vs 0.99. The
   write-up's own conclusion says "at an IIA 0.05 lower". The range "925–1665" also excludes Qwen3 L26 (683).
8. **Figure legend (`bayes_vs_plugin_ladder.png`, slide 11).** The blue line is labelled "mixture of 4", but
   the InterpBench point is the MAP code (+1052; `fig_bayes_ladder.py` regex picks it from
   `case18_bayes_code_hmc.txt`). The mixture gives +983.
9. **"≈ 65,000 nats, ≈ 22× label-only" (slides 9, 11, figure; `curvature_gpt2.txt`; write-up §10.3,
   §10.4).** [FIXED 2026-10-22 on slides 9, 11, the ladder figure and curvature_gpt2.txt (bracketed correction); write-up §10.3/§10.4 not changed.] Exceeds the hard bound n ln(K/ε) ≈ 6,570 nats (about 2.2× label-only) for an observed-law code
   on 1016 targets. It is λ ln n, not a code length.
10. **Raw vs batching-corrected slopes.** Slide 23 says "every slope in the talk is divided by it". But:
    the slide 26 text ("45–50 per ln n from n = 32 to 16384"; manufactured "1700–2450") uses raw local
    slopes from `h_a3_gpt2_large_n.txt`, where the corrected local values are about 39 and about 1180–1700.
    The "slope 45 per ln n" annotation in `interpbench_regret_vs_n.png` (slide 25), the slide 25 notes
    ("20 to 45") and `reports/talk_slt/README.md` ("about 45 per ln n") are raw; corrected about 31. Write-up
    §9.4 turns the raw 45 into "about 90 constrained directions" (corrected: about 64).
11. **Slide 25 internal mismatch.** The text gives 16.8 ± 2.8 at 8192 (a corrected fit); the figure gives
    "slope 45" (a raw local slope over the last two doublings). The figure's legend does not distinguish
    the dashed 1024-target run from the solid 8192-target run.
12. **"Multiplicity-2 shapes sit low" (slide 6 text and notes; `rrr/README.md` "the two m = 2 cases").**
    There are three m = 2 shapes. Only (3,3,3,0) is clearly low (−0.45). (4,4,4,1) is −0.09 and (4,4,1,0)
    −0.01, while the m = 1 shapes (3,3,3,1) and (4,2,4,0) are −0.28. Source `rrr/codes_summary.txt`,
    `gain_summary.txt`.
13. **"Valid subspace κ = 79 at d_V 4" (slide 13; `kappa_vs_dv.png` label "valid subspace").** At d_V 4 the
    subspace has IIA 0.27, gain +132 and train NLL 2.43 (`curvature_gpt2_layer11_dv4_full_heldout_fisher.json`):
    misspecified, not valid.
14. **Oracle outside-direction curvature (slide 7; `oracle/hessian_vs_fisher.txt` reading; write-up
    §10.2).** "Curved, growing with k": the values 8.9e-5, 4.3e-3, 4.3e-4, 2.8e-2 are not monotone. My
    numerical check gives a quartic population KL along these directions, so the "curvature" is empirical.
    This is an interpretation issue rather than a typo.
15. **LLC localisation wording (write-up §9.4).** "2.22 ± 0.18 under tight localisation (strength 1)" vs
    `stage1_interpbench/case18_llc.txt`: γ = 1 is "a weaker pull to the MAP than γ 10".
16. **Fisher definition in a source file.** `stage1_ioi/curvature_gpt2.txt` says "covariance of the
    per-example gradients"; the code and the write-up use the uncentred second moment (the write-up notes
    the correction).
17. **IIA of the manufactured alignment in `regret_slopes_map_code.txt`.** The note says "IIA 0.86 on the
    training names" for the MAP-code run. Under the MAP code it is 0.83 (`adversary_by_learner.txt`); 0.86
    is the mixture run.
18. **Outdated learner note (`reports/talk_slt/README.md`, docstrings of `fig_models.py` and `fig_mib.py`).**
    These say the headers of `tier2_models.txt` and `stage1_mib/arithmetic.txt` claim "mixture-of-4" for
    all runs. Both headers now state the single-map MAP learner correctly (mixture for Qwen3 only).
19. **Rounded IIAs shown with two decimals (`ladder_panels.png`, `iia_vs_code.png`).** The GPT-2 IIAs come
    from `gpt2_summary_single_map_map0.md`, rounded to one decimal ("1.00" means 1.0 ± 0.05; the twin "0.2",
    the wrong locus "0.1"), while the manufactured IIA 0.83 has two decimals.
20. **Comparative margin under the MAP (slide 29 text).** "≈ 5300 bits": 3385 + 1986 = 5371, so ≈ 5400.
    The write-up also says "about 5300".
21. **"A single map pays 6–7 bits per target throughout" (slide 29).** The MAP's last block is 5.6
    bits/target (`adversary_by_learner.txt`).
22. **JC ratio "six to ten times" (slide 28 notes; `jlens_gpt2.txt`).** The median ratios run 5.4–15.
23. **LLC "close to k/2" (slide 30; write-up "within 20–30 %").** At k = 1 it is 0.85 vs 0.5 (+70 %).
24. **Slide 5 wording.** "Regular, correctly specified: κ = ½ tr(F H⁻¹)": under correct specification
    F = H and κ = d/2. The sandwich formula is the misspecified case.
25. **Slide 17.** "The coupling map reads the carry at layers 20–24": at L20 its code is 839 bits *worse*
    than the floor (+1876 vs +2715) despite IIA 0.96 (`arithmetic.txt`, and visible in the figure).
26. **Pythia figure (`pythia_onset.png`).** The accuracy line is drawn through step 3000, where accuracy
    was not measured.
27. **Slide 2 text "Same on Qwen2.5-3B and Llama-3.1-8B: IIA 0.59 / 0.51"** sits next to a figure that
    shades only IIA ≥ 0.8 as "accepted". Neither point is in the band.

---

## 7. Numbers to have in your head

- Label-only (IOI): 4324.5 bits over 1016 targets (4.26 bits/target). Channel floor (K = 32, ε = 0.05):
  0.36 nats = 0.52 bits per target (about 528 bits in total). Cap per target: 9.3 bits.
- GPT-2 valid L11: +3385 (MAP), +3302 (mixture); κ 48.9 ± 1.7; regret at 1024 is 279 nats; 35–42 at
  32768.
- Manufactured L10: −1986 ± 602 (MAP), +704 ± 63 (mixture); κ 988 ± 93; 726–826 at 32768; IIA 0.83 / 0.86;
  shifted 0.00.
- Coupling (trained) L11: +2459; κ 226; 129–146 at 32768 (3 seeds).
- At 32768 targets (3 seeds, MAP code), gain over label-only: valid +116,619; trained coupling +116,989;
  manufactured +108,956; twin +4,230. Valid vs manufactured +7.7k = fit −2.0k + regret +9.6k (peak +8.3k at
  16384). Valid vs coupling −0.4k = fit −2.0k + regret +1.6k. Held-out final-fit losses: 0.427 (valid),
  0.390 (manufactured), 0.387 (coupling) nats. Clean-target median: 0.092 (valid) vs 0.050 (floor).
- J-lens L11: +3525 (train names), IIA 0.98, κ = 0.
- Curvature L11 d_V 16: 12,032 positive and 256 gauge eigenvalues; tr F / tr H = 2.0 % (held out); median
  F/h 0.007; prediction 70.6.
- Nominal λ: 1528 / 6016 / 22,528 at d_V 4 / 16 / 64. κ: 79 / 49 / 56.
- RRR: Bayes 6.04 ± 0.27 (λ 6), 1.53 ± 0.15 (1.5), 1.99 ± 0.20 (2); OLS 8.29 ± 0.31 (MN/2 = 8).
- Oracle k-sweep (n ≤ 16384): MAP 1.14 / 1.39 / 2.37 / 3.73 / 5.32; Bayes 3.35 / 3.05 / 3.46 / 4.44 /
  5.43; my λ = k/2 + (11 − k)/4 gives 3.0 / 3.25 / 3.75 / 4.5 / 5.5.
- HE: Bayes +733 vs plug-in +506; coupling −322 vs +586; Bayes slope 12.9 (nominal 56).
- InterpBench HMC at 1024: 1526 / 2370 / 1898 bits vs label-only 1342; MAP +1052 (κ 8.4, interpolation
  regime).
