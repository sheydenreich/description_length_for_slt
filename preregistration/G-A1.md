# Preregistration: gate G-A1 (end of week 6, mid-November 2026)

Plan text (Section "Gates and outputs"): *Does the Stage 1 score reject convincing false
explanations that the strongest baselines accept, while retaining valid explanations up to
equivalence?*

This file fixes, before the deciding runs, what "convincing", "reject", "retain" and
"up to equivalence" mean in numbers. Status of the evidence at the time of writing
(2026-10-16) is recorded honestly in the "Status" lines: the toy and InterpBench runs
listed below were already observed with the final scoring code before this file was
frozen; the GPT-2 IOI run with the final learner specification was submitted the same day
and had not returned. Changes after freezing go in the change log at the bottom.

## The score

For a configuration (network, locus, map family, learner) the Stage 1 score is the
prequential code length of the counterfactual targets of one algorithm variable, passed
through randomized response with rate ε = 0.05, on a geometric schedule (r = 2, first
block at 8, 1024 targets, 1016 coded). Every predictor codes the observed label law
(1 − ε)·p + ε/K through the declared channel (`alignment/targets.NoiseChannel`). The
*gain* of a configuration against a control is the control's code length minus the
configuration's, in bits, on identical targets. The reference code is the label-only
code (Dirichlet–categorical plug-in on the labels alone).

Canonical learner: equal-weight mixture of four cold-start gradient-trained maps
(`MixtureAlignmentLearner`, 400 Adam steps per block, batch 256). Map families: the
gauge-free subspace map (`SubspaceMap`, or `Rotation` where the testbed's runner uses it)
with d_V dimensions, and the additive coupling map as the expressive family.

## Testbeds, hypotheses and controls

| testbed | network | variable | valid hypothesis | learner spec | seeds |
|---|---|---|---|---|---|
| hierarchical equality | MLP 16-32-32-2 trained to 100% | V1 | rotation at h1, d_V = 4 | lr 1e-2 | 5 |
| InterpBench case 18 | released transformer, 100% faithful to its program | freq | rotation at resid_2, d_V = 4 and d_V = 8 | lr 1e-2 | 3 |
| IOI in GPT-2 small | pretrained, 99.6% IO-over-S on 32 single-token names | io | subspace at layers 10, 11, 12 (final position), d_V = 16 | subspace lr 1e-2, coupling lr 1e-3 | 3 |

Controls for every hypothesis (same targets, same learner, same budget):

1. **untrained twin**: same architecture, weights re-initialised with seed 7;
2. **wrong locus** *(amended 2026-10-19, see change log)*: the locus where the hypothesis's
   map family can least realise the counterfactual: for every locus, the best held-out
   interchange intervention accuracy of two cold-start fits with the plug-in budget on the
   full target set; the control is the locus with the lowest attainable accuracy, loci within 0.02 of the
   minimum counting as tied with the earliest winning, and a hypothesis at that locus has
   no wrong-locus control. Linear
   decodability is reported as a diagnostic. *Original rule: the locus where a held-out
   logistic probe for the variable is least accurate. It chose input / resid_0 / layer 0 on
   the toy, InterpBench 18 and GPT-2, which the causal rule reproduces (attainable IIA 0.50
   / 0.46 / 0.072, the last tied with layer 6 at 0.069 and layer 9 at 0.093, all chance
   level for 32 names, against 0.97-1.00 at layers 10-12), but on InterpBench case 52 the variable is a copy of an input token,
   decodable everywhere (probe 1.0 at the embedding), and the probe rule selected resid_2,
   the valid locus; the causal rule selects resid_1 (attainable IIA 0.11 vs 0.99 at the
   embedding and 0.64 at resid_2).*
3. **other map family**: the coupling map at the hypothesis locus;
4. **label-only** reference.

Manufactured alignment (the "convincing false explanation"): the coupling map (width 512,
4 blocks; width 64, 2 blocks on the two small testbeds) trained for 4000 steps (toy and
InterpBench: 400) on the *untrained twin*, at h2 / resid_2 / layer 6 respectively. Its
accuracy (interchange intervention accuracy, IIA) is what the strongest baseline, DAS-style
alignment search judged by IIA, would accept.

## Criteria

Thresholds follow the rule agreed on 2026-10-16: a gain is **decisive** when it exceeds
both an absolute and a relative floor **and** is statistically clear:

- size: gain ≥ max(100 bits, 5% of the label-only code on the coded targets), i.e. 100 bits
  on the toy and InterpBench (label-only 1021 and 1342 bits), 216 bits on IOI (4325 bits);
- significance: mean gain over seeds ≥ 3 standard errors of that mean, and the gain is
  positive for every seed.

1. **Convincing false explanations exist.** On at least one testbed the manufactured
   alignment reaches IIA ≥ 0.80 on the i.i.d. evaluation split, so the accuracy baseline
   would accept it. *Status: met on GPT-2 (IIA 0.86 on all three seeds of the final run); not met on the
   toy (0.5) or InterpBench (0.6), where it is nevertheless scored.*
2. **Manufactured alignments are rejected** *(amended 2026-10-17, see change log)*. On
   every testbed the manufactured alignment's gain over its own untrained-twin control is
   **not** decisive. For an alignment of an untrained network the matched control is the
   same map family, locus, learner and budget on a *second* random initialisation (twin
   seed 8); for every alignment the control is computable without ground truth. Its
   certified distortion bound is reported next to the verdict as a diagnostic, not a
   criterion. *Original absolute form and its status: "gain over label-only is not
   decisive" held on the toy (−571 ± 20 bits), InterpBench (−437 ± 31, −430 ± 33) and
   GPT-2 at layer 6 (+60 ± 138), but FAILED for the coupling map on the untrained GPT-2 twin
   at layer 10 (+704 ± 63, all seeds) and layer 9 (+290 ± 43): the untrained residual
   stream still carries the name tokens and a 1.9 M-parameter map decodes them within the
   16 training names (IIA 0.86 i.i.d., 0.00 on new names; distortion bound 4·10¹⁶ against
   1 for the valid subspace). A plug-in code with a fixed budget does not charge for map
   complexity; this is reported as a result. Status of the comparative form (gain of the
   manufactured alignment over the same map on twin seed 8; `reports/stage1_ioi/
   second_twin_check.txt`): toy −19 ± 36 bits (5 seeds), InterpBench +67 ± 24 and −16 ± 42,
   GPT-2 layer 9 +10 ± 82, GPT-2 layer 10 +349 ± 195 (per seed +702, +30, +316): none
   decisive, so **met** everywhere. The layer-10 case is narrow: it clears the size floor
   and every seed is positive, and fails only the 3-s.e. test. Two random initialisations
   of GPT-2 can differ by several hundred bits under an expressive map, which is the noise
   floor of this control and is reported with the result.*
3. **Valid explanations are retained.** On every testbed the valid hypothesis's gains over
   label-only, over the untrained twin and over the wrong locus are decisive. *Status: toy
   +506 ± 19 / +515 ± 19 / +1251 ± 26; InterpBench d_V 4 +983 ± 11 / +1278 ± 11 /
   +1946 ± 11; GPT-2 layer 11 +3299 ± 3 / +3253 ± 3 / +6816 ± 3 (layers 10 and 12
   similar, all seeds positive): met everywhere.*
4. **Up to equivalence** *(amended 2026-10-17, see change log)*. Every valid hypothesis
   for the same variable that differs only in a representational choice (InterpBench d_V 4
   vs 8; GPT-2 layers 10, 11, 12) satisfies criterion 3. The ordering of their codes is
   reported as a result, not judged. *Status: met (InterpBench +983 and +996 bits; GPT-2
   +2975, +3302, +3145 over label-only, every seed decisive). Original clause and its
   status: the drafted version also required that the differences between equivalent
   hypotheses' gains not be decisive; InterpBench met it (13 ± 14 bits) but GPT-2 did not
   (layer 11 codes 327 ± 3 bits shorter than layer 10 and 157 ± 4 shorter than layer 12
   over five seeds). The shorter code belongs to the more faithful map (IIA 0.99 against
   0.97, log loss 0.50 against 0.62 nats), so the clause penalised the score for ranking
   loci correctly.*
5. **Separation the baselines lack.** On the testbed(s) of criterion 1, the accuracy
   baseline accepts the manufactured alignment (IIA ≥ 0.80) while the valid hypothesis's
   gain over the manufactured alignment's code is decisive. The online-probe code is
   reported alongside but codes a different quantity (the variable itself, noise-free) and
   is not a pass/fail criterion.

The gate **passes** if criteria 1–5 all hold. If criterion 1 fails everywhere, the gate
is **undecided**, not failed: the plan's premise (that convincing false explanations
exist) was not reproduced, and a stronger adversary is the first item of the bounded
feasibility budget. Criterion 3 failing on any testbed, or criterion 2 failing on any
testbed, fails the gate.

## Reporting

For each testbed: the summary table (`results/<run>/summary.md`, archived under
`reports/`), the per-block code figure, the gains table with standard errors, the
faithfulness on the i.i.d. and shifted splits, and the diagnostics where computed. The
decision is recorded at the bottom of this file with the commit hash of the code.

## Decision

**2026-10-17: the gate PASSES under the amended criteria** (code at commit 6af7b3f plus this
file). Criteria 1, 2 (comparative), 3, 4 (restricted) and 5 hold on hierarchical equality,
InterpBench case 18 and GPT-2 IOI. Under the criteria as first drafted the gate would have
failed on GPT-2 on two counts, both recorded above: an expressive map on an untrained
network can beat the label-only code (criterion 2, absolute form), and equivalent loci do
not code equally (criterion 4, agreement clause). Both are results for the Stage 1 paper,
not footnotes: the first says plug-in codes with a fixed budget do not charge for map
complexity and motivates Bayes or Laplace codes for the map family; the second says the
code orders loci by faithfulness. The narrow GPT-2 layer-10 margin on the comparative
criterion is reported as such.

## Change log

- 2026-10-19 (decision): the wrong-locus control redefined causally (least attainable
  interchange accuracy of the hypothesis's map family) by the user's decision after
  InterpBench case 52 showed the decodability rule selecting the valid locus for a copy
  variable. Post hoc; the original rule's choices stay recorded. Controls selected by the
  old rule are unchanged on the toy, case 18, case 7 and GPT-2; case 52's wrong-locus
  records are recomputed.
- 2026-10-17 (decision): criterion 4 restricted to "all equivalent valid hypotheses are
  retained", by the user's decision; the agreement clause and its GPT-2 outcome stay
  recorded above.
- 2026-10-17 (decision): criterion 2 changed from the absolute form to the comparative form
  above, with the distortion bound as a diagnostic, by the user's decision after seeing the
  GPT-2 adversary sweep. This is post hoc; the absolute form's outcome stays recorded.
- 2026-10-17: five GPT-2 seeds (gains unchanged to within 3 bits; s.e. about 3 bits) and the
  adversary sweep recorded. Criterion 2 fails as written on GPT-2 (layer-10 adversary); the
  twin-versus-twin comparison was added as a *proposed* comparative form, not yet adopted.
- 2026-10-16 (later): GPT-2 statuses filled in from `results/stage1_ioi_gpt2_v2`
  (3 seeds, learner spec from the sweep). Criterion 4's agreement clause flagged as the one
  open point; no criterion was changed.
- 2026-10-16: drafted after the channel fix (commit c39b208) and the IOI map sweep
  (`results/ioi_map_sweep`), with the toy and InterpBench statuses as observed.
