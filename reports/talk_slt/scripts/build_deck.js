// Build reports/talk_slt/slt_days_talk.pptx (SLT days, week of 2026-10-26).
// Figures are read from reports/talk_slt/*.png (made by make_all.py and earlier scripts).
// Run:  PATH=~/scratch/miniconda/envs/slides/bin:$PATH NODE_PATH=~/scratch/tools/node_pptx/node_modules \
//         node reports/talk_slt/scripts/build_deck.js
// Every number on a slide is taken from a file under reports/ (named in the slide's source line or notes).
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const SKILL = process.env.PPTX_SKILL_DIR ||
  "/u/lade/sheydenreich/.claude/skills/synced/34486e66-ec48-49aa-bb03-13986aa7c18f_7da7ec62-9ec7-4869-b790-48713d4fd41e/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const FIG = path.resolve(__dirname, "..");
const OUT = path.join(FIG, "slt_days_talk.pptx");

const THEME = {
  name: "Bits of evidence",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F2A", lt1: "FFFFFF", dk2: "52514E", lt2: "EEF1F5",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100",
    accent5: "E34948", accent6: "008300", hlink: "2A78D6", folHlink: "6250D6",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.author = "Sven Heydenreich";
pres.title = "Bits of evidence: the free energy of an explanation";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, H = 7.5;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: THEME.colors.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.9, w: 11.7, h: 1.6, fontFace: "Cambria",
      fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.75, w: 11.0, h: 1.4, fontFace: "Calibri",
      fontSize: 22, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: THEME.colors.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.3, fontFace: "Cambria",
      fontSize: 40, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.95, w: 11.0, h: 1.0, fontFace: "Calibri",
      fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: THEME.colors.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 0.6, w: 11.7, h: 1.0, fontFace: "Cambria",
      fontSize: 40, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
});
function contentLayout(name, tag) {
  const objects = [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.38, w: 12.3, h: 1.0, fontFace: "Cambria",
      fontSize: 28, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Bits of evidence · SLT days 2026", options: { x: 0.5, y: 7.08, w: 5, h: 0.3, fontSize: 10,
      color: C.text2, margin: 0 } } },
  ];
  if (tag) objects.push({ text: { text: tag, options: { x: 0.5, y: 0.05, w: 3, h: 0.25, fontSize: 11, bold: true,
    color: C.accent2, margin: 0, charSpacing: 2 } } });
  pres.defineSlideMaster({
    title: name, background: { color: THEME.colors.lt1 }, objects,
    slideNumber: { x: 12.3, y: 7.08, w: 0.6, h: 0.3, fontSize: 10, color: THEME.colors.dk2, align: "right" },
  });
}
contentLayout("CONTENT", null);
contentLayout("BACKUP", "BACKUP");

// ------------------------------------------------------------------ helpers
function pngAspect(file) {
  const b = fs.readFileSync(file);
  return b.readUInt32BE(16) / b.readUInt32BE(20);
}
function slideOf(layout, section, title) {
  const s = pres.addSlide({ masterName: layout, sectionTitle: section });
  if (title) s.addText(title, { placeholder: "title" });
  return s;
}
const PILL = { plug: ["plug-in κ (MAP)", C.accent1], bayes: ["Bayes λ", C.accent2] };
function pills(s, kinds) {
  let x = W - 0.45;
  for (const k of [...kinds].reverse()) {
    const [txt, col] = PILL[k];
    const w = 0.1 * txt.length + 0.35;
    x -= w;
    s.addText(txt, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.15, x, y: 0.08, w, h: 0.3, fill: { color: col },
      color: C.background1, fontSize: 11, bold: true, align: "center", valign: "middle", margin: 0,
      objectName: "learner tag " + k });
    x -= 0.12;
  }
}
// figure at (x, y) with width w (or height h); returns its height
function fig(s, name, x, y, { w = null, h = null } = {}) {
  const file = path.join(FIG, name);
  const a = pngAspect(file);
  if (w === null) w = h * a;
  if (h === null) h = w / a;
  s.addImage({ path: file, x, y, w, h, objectName: "figure " + name, altText: name });
  return h;
}
function source(s, txt, y = 6.72) {
  s.addText("Source: " + txt, { x: 0.5, y, w: 12.3, h: 0.3, fontSize: 10, color: C.text2, margin: 0, isTextBox: true,
    objectName: "source line" });
}
// Paragraph builder. pptxgenjs writes paragraph properties per run, so bullets and spacing go on the first run
// of each paragraph only, and nothing paragraph-level is set on the shape.
function paras(items, { bullet = false, space = 9 } = {}) {
  return items.map((t, i) => {
    const runs = Array.isArray(t) ? t : (typeof t === "string" ? [{ text: t }] : [t]);
    return runs.map((r, j) => ({ text: r.text, options: Object.assign({}, r.options || {},
      j === 0 ? Object.assign({ paraSpaceAfter: space }, bullet ? { bullet: { indent: 16 } } : {}) : {},
      j === runs.length - 1 && i < items.length - 1 ? { breakLine: true } : {}) }));
  }).flat();
}
// split a flat run list at empty breakLine runs into paragraphs
function splitParas(body) {
  if (typeof body === "string") return [body];
  const out = []; let cur = [];
  for (const r of body) {
    if (r.text === "" && r.options && r.options.breakLine) { out.push(cur); cur = []; continue; }
    const o = Object.assign({}, r.options || {}); delete o.breakLine;
    cur.push({ text: r.text, options: o });
  }
  if (cur.length) out.push(cur);
  return out;
}
function bullets(s, items, x, y, w, h, fontSize = 16) {
  s.addText(paras(items, { bullet: true }), { x, y, w, h, fontSize, color: C.text1, valign: "top", margin: 0,
    isTextBox: true, objectName: "bullets" });
}
function card(s, x, y, w, h, head, body, { headColor = C.text1, fill = C.background2, fontSize = 15 } = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill },
    objectName: "card " + head });
  s.addText(head, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.45, fontSize: 17, bold: true, color: headColor, margin: 0,
    fontFace: "Cambria", isTextBox: true, valign: "top" });
  s.addText(paras(splitParas(body), { space: 8 }), { x: x + 0.2, y: y + 0.6, w: w - 0.4, h: h - 0.7, fontSize,
    color: C.text1, margin: 0, valign: "top", isTextBox: true });
}
function formula(s, txt, x, y, w, h, fontSize = 24) {
  if (Array.isArray(txt)) txt = txt.map((r) => ({ text: r.text, options: Object.assign({ align: "center" }, r.options || {}) }));
  s.addText(txt, { x, y, w, h, fontSize, fontFace: "Cambria", italic: true, color: C.text1, align: "center",
    valign: "middle", margin: 0, isTextBox: true, objectName: "formula" });
}
const B = (t) => ({ text: t, options: { bold: true } });
const T = (t) => ({ text: t });

// ================================================================== slides
const S1 = "Problem and score", S2 = "Coefficients where we know the answer", S3 = "Language models",
  S4 = "Scale", S5 = "For singular learning theory", SB = "Backup";

// 1 title
pres.addSection({ title: S1 });
{
  const s = slideOf("TITLE_DARK", S1, "Bits of evidence: the free energy of an explanation");
  s.addText("Scoring interpretability alignments by the code length of counterfactuals, and what singular learning theory says about the complexity term",
    { placeholder: "body" });
  s.addText("Sven Heydenreich  ·  Singular Learning Theory days  ·  October 2026", { x: 0.8, y: 6.2, w: 11, h: 0.4,
    fontSize: 16, color: C.background2, margin: 0, isTextBox: true });
  s.addNotes("This talk is about an application of description length to interpretability, and about what singular learning theory does and does not explain in it. The application: score a claim that a network implements an algorithm by how many bits it takes to code the algorithm's counterfactual predictions with the network as the predictor. For a Bayes code that score is a free energy, so learning coefficients enter directly. The punchline is that the codes that actually work are plug-in codes, and their complexity term is not the Bayes learning coefficient but a Fisher-weighted count of directions. I would like to leave you with concrete questions about plug-in learners in singular models.");
}

// 2 problem
{
  const s = slideOf("CONTENT", S1, "Interchange accuracy accepts explanations that the code rejects");
  pills(s, ["plug"]);
  fig(s, "iia_vs_code.png", 0.4, 1.4, { w: 8.4 });
  bullets(s, [
    [B("Causal abstraction: "), T("swap the part of a hidden state aligned with variable V; compare the output with the algorithm's counterfactual. IIA = fraction that agree.")],
    [B("Expressive maps fake it "), T("(Sutter et al.): a 1.9 M-parameter coupling map on an untrained GPT-2 reaches IIA 0.83.")],
    [T("At 1016 targets its code is "), B("1986 bits longer"), T(" than label-only; the valid subspace is "), B("3385 bits shorter"), T(".")],
    T("Same on Qwen2.5-3B and Llama-3.1-8B: IIA 0.59 / 0.51, code 1723 / 1830 bits longer; Qwen3 (mixture): IIA 0.71, 147 longer."),
    T("With 32768 targets both beat label-only; what separates them then is the learning cost alone (slide 14)."),
  ], 9.05, 1.5, 3.85, 5.1, 15);
  source(s, "stage1_ioi/gpt2_summary_single_map_map0.md, tier2_models.txt, adversary_by_learner.txt; single-map MAP code except Qwen3 (mixture).");
  s.addNotes("Interchange-intervention accuracy is the standard acceptance statistic for causal abstraction: overwrite the aligned part of a hidden state with its value on another input and check the output against the algorithm. Sutter et al. showed that with expressive invertible maps any network can be made to look like it implements any algorithm, including untrained ones. Each dot here is one configuration on the indirect-object task in four language models: the red points are such manufactured alignments on untrained networks, with IIA up to 0.83, which an accuracy threshold would accept. On the vertical axis is the score of this talk, the code-length gain over a code that sees only the labels: at a thousand targets the manufactured alignments are 1700 to 2000 bits worse than knowing nothing under the single-map learner (150 bits worse on Qwen3 under the averaging learner), the faithful subspaces about 3000 bits better. Two honest caveats. Under the canonical averaging learner the GPT-2 layer-10 adversary beats label-only by 704 bits. And 'worse than label-only' is a statement at a thousand targets: with 32768 targets the GPT-2 forgery beats label-only by about 109 thousand bits. What survives at every n and under every learner is the comparison with the valid alignment, and slide 14 shows that it is carried entirely by the learning cost.");
}

// 3 score
{
  const s = slideOf("CONTENT", S1, "The score: prequential code length of counterfactual targets");
  const box = (x, y, w, txt, fill) => s.addText(txt, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x, y, w, h: 0.95,
    fill: { color: fill }, line: { color: fill }, fontSize: 14, color: C.text1, align: "center", valign: "middle", margin: 4 });
  const arrow = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color: THEME.colors.dk2, width: 1.75, endArrowType: "triangle" } });
  box(0.5, 1.55, 2.4, "pair of inputs\n(base b, source s)", C.background2);
  box(3.4, 1.55, 3.0, "algorithm's counterfactual\ny* = A(b; V ← V(s))", C.background2);
  box(6.9, 1.55, 2.6, "randomized response\nε = 0.05 → label y", C.background2);
  arrow(2.9, 2.02, 3.4, 2.02); arrow(6.4, 2.02, 6.9, 2.02);
  box(0.5, 3.05, 2.4, "network N, locus ℓ\nmap φ fitted on prefix", C.background2);
  box(3.4, 3.05, 3.0, "interchange: aligned part\nof h_ℓ(b) ← that of h_ℓ(s)", C.background2);
  box(6.9, 3.05, 2.6, "frozen readout\np(y | b, s)", C.background2);
  arrow(2.9, 3.52, 3.4, 3.52); arrow(6.4, 3.52, 6.9, 3.52);
  arrow(8.2, 2.5, 8.2, 3.05);
  formula(s, [T("DL(n) = Σᵢ −log₂ p"), { text: "θ̂(i−1)", options: { subscript: true } }, T("(yᵢ | bᵢ, sᵢ)")], 0.5, 4.3, 9.0, 0.8, 26);
  s.addText("each target is coded before it is used for fitting: a family that can fit anything compresses nothing",
    { x: 0.5, y: 5.1, w: 9.0, h: 0.4, fontSize: 14, italic: true, color: C.text2, align: "center", margin: 0, isTextBox: true });
  card(s, 9.9, 1.55, 3.0, 4.6, "Controls, same targets",
    [B("label-only"), T(" (Laplace's rule, sees neither net nor algorithm)"), { text: "", options: { breakLine: true } },
     B("untrained twin"), T(" (same architecture, random weights)"), { text: "", options: { breakLine: true } },
     B("wrong locus"), T(" (lowest attainable IIA)"), { text: "", options: { breakLine: true } },
     B("other map family"), T(" (coupling flow)"), { text: "", options: { breakLine: true } },
     { text: "Gain = DL(control) − DL(hyp.) = running log Bayes factor (an e-process against the control).", options: { breakLine: false } }],
    { fontSize: 14 });
  s.addText("By the chain rule, the prequential code of the Bayes posterior predictive is the Bayes free energy Fₙ.",
    { x: 0.5, y: 5.7, w: 9.0, h: 0.5, fontSize: 16, bold: true, color: C.accent2, margin: 0, isTextBox: true });
  s.addNotes("Concretely: draw base and source inputs, compute the algorithm's counterfactual output, pass it through randomized response with a known 5 percent rate, and code the resulting label sequentially. The predictor is the network itself with the aligned part of its hidden state swapped; only the map is fitted, on the targets seen so far, and the readout stays frozen. The code length is the cumulative log loss, and differences to controls on identical targets are gains in bits, which read as running log Bayes factors. Randomized response matters: it makes deterministic targets realizable at finite logits, and every predictor must code the observed noisy law (backup). For a Bayes predictive this code length is exactly the free energy, which is where your theory comes in.");
}

// 4 gain decomposition
{
  const s = slideOf("CONTENT", S1, "For a Bayes code, the complexity term is a learning-coefficient difference");
  pills(s, ["bayes"]);
  formula(s, "G(n) = n·Δℓ + Δλ·ln n − Δ(m−1)·ln ln n + Rₙ + Oₚ(1)", 0.5, 1.45, 12.3, 0.9, 30);
  s.addText("Watanabe: Fₙ = n Lₙ(w₀) + λ ln n − (m−1) ln ln n + Oₚ(1), for each of the two configurations",
    { x: 0.5, y: 2.35, w: 12.3, h: 0.4, fontSize: 15, color: C.text2, align: "center", margin: 0, isTextBox: true });
  card(s, 0.5, 3.0, 3.95, 2.45, "Falsification", "Δℓ > 0: the control cannot match the hypothesis's loss. Gain grows linearly; the control is rejected.", { headColor: C.accent1 });
  card(s, 4.69, 3.0, 3.95, 2.45, "Occam", "Δℓ = 0: both reproduce the target law. Gain at most logarithmic, carried by Δλ: evidence relative to priors or learners.", { headColor: C.accent2 });
  card(s, 8.88, 3.0, 3.95, 2.45, "Approximation", "Neither family fits; Rₙ (excess-loss drift of a misspecified family) can dominate the log term.", { headColor: C.text2 });
  s.addText([B("The question of this talk: "), T("the codes that decide our verdicts are plug-in codes, not Bayes codes. What replaces λ in the second term for them, and does singular learning theory describe it?")],
    { x: 0.5, y: 5.75, w: 12.3, h: 0.8, fontSize: 17, color: C.text1, margin: 0, isTextBox: true });
  s.addNotes("If both configurations are coded by Bayes predictives, the gain between them is a difference of two free energies, and Watanabe's expansion gives this decomposition. The linear term is the falsification regime, where a control is rejected outright. When both families contain the target law, the gain is logarithmic and the coefficient is a difference of learning coefficients: that is the Occam regime, and the original plan was to measure exactly this. But the codes we use in practice are plug-in codes, a map refitted by gradient descent on each prefix, so the second term has to be re-derived for the learner. The rest of the talk measures that coefficient in settings of decreasing control: an exact example, an oracle, then transformers.");
}

// 5 two coefficients
{
  const s = slideOf("CONTENT", S1, "Two coefficients: the Bayes λ and the learner's κ");
  pills(s, ["plug", "bayes"]);
  card(s, 0.5, 1.5, 6.0, 3.6, "Bayes predictive: λ",
    [T("excess loss after m targets λ/m  ⇒  regret λ ln n"), { text: "", options: { breakLine: true } },
     T("λ = real log canonical threshold of the family at the truth; regular: d/2"), { text: "", options: { breakLine: true } },
     T("subspace family on a locus of width d: nominal λ = d_V(d − d_V)/2, half the Grassmannian dimension")],
    { headColor: C.accent2, fontSize: 18 });
  card(s, 6.83, 1.5, 6.0, 3.6, "Plug-in (MAP) refitted on each prefix: κ",
    [T("excess loss κ/m  ⇒  regret κ ln n: a learner coefficient"), { text: "", options: { breakLine: true } },
     T("regular, correctly specified: κ = ½ tr(F H⁻¹) (Takeuchi-type); F = H gives d/2"), { text: "", options: { breakLine: true } },
     T("F: second moment of per-example gradients; H: Hessian of the mean loss, both at the MAP")],
    { headColor: C.accent1, fontSize: 18 });
  s.addText([B("Measured as: "), T("regret = prequential loss − loss of the best-in-family fit on the same targets, least-squares slope against ln n, divided by the batching factor (r − 1)/ln r = 1.4427 for doubling blocks.")],
    { x: 0.5, y: 5.4, w: 12.3, h: 0.9, fontSize: 16, color: C.text1, margin: 0, isTextBox: true });
  s.addNotes("Two coefficients appear throughout and I will label every number with one of them. Lambda is the Bayes learning coefficient: it governs the regret of the posterior predictive. Kappa is the coefficient of whatever learner builds the code; for a plug-in estimator in a regular model it is the classical one half trace of F times H inverse, which equals d over two only when the information equality holds. Both are measured the same way, as the slope of the regret against log n, corrected for the block schedule; that correction is checked exactly in the backup. The plan's open question was when kappa equals lambda; the answer turns out to be: almost never on trained networks.");
}

// 6 RRR
pres.addSection({ title: S2 });
{
  const s = slideOf("CONTENT", S2, "Exact check: the Bayes code reads Aoyagi–Watanabe λ, the plug-in reads MN/2");
  pills(s, ["plug", "bayes"]);
  fig(s, "rrr_exact_check.png", 0.4, 1.5, { w: 8.9 });
  bullets(s, [
    T("Reduced-rank regression y = UVx + e; 12 shapes (M, N, H, r), n = 65536, 16 seeds."),
    [B("Bayes (HMC): "), T("6.04 ± 0.27 for λ = 6; 1.53 ± 0.15 for 1.5; 1.99 ± 0.20 for 2.")],
    [B("Least squares: "), T("8.29 ± 0.31 for MN/2 = 8, whatever the true rank.")],
    T("Gains between two families on the same data: Δλ ln n (3.40 ± 0.21 vs 3.5; 4.05 ± 0.14 vs 4.0)."),
    T("Multiplicity-2 shapes sit low: the −(m−1) ln ln n term."),
  ], 9.55, 1.5, 3.35, 5.1, 15);
  source(s, "rrr/codes_summary.txt, gain_summary.txt, gain_between_families.txt (slopes on n ≥ 256, batching-corrected).");
  s.addNotes("Before any network, the exact example. In reduced-rank regression the learning coefficient is known from Aoyagi and Watanabe, so we ran a genuine Bayes code by HMC and a least-squares plug-in code on the same data. The Bayes code's regret slope is lambda on all twelve shapes, within one or two standard errors; the plug-in's slope is the regular MN over 2 whatever the true rank, so whenever the rank is deficient the two differ by the predicted amount. Gains between a rank-restricted and a full family on the same data grow like delta lambda log n, which is the Occam-regime prediction of the previous slide. The two multiplicity-2 shapes sit slightly below lambda, which we attribute to the log log n term at these n.");
}

// 7 oracle k sweep
{
  const s = slideOf("CONTENT", S2, "Oracle: the plug-in pays for readable directions, the Bayes code for quartic ones too");
  pills(s, ["plug", "bayes"]);
  fig(s, "oracle_ksweep_slide.png", 0.4, 1.4, { h: 5.25 });
  bullets(s, [
    T("Oracle network, d = 12, map of a 1-dim subspace; the data vary along only k of the 11 Grassmannian directions."),
    [B("MAP κ ≈ ½ tr(F H⁺): "), T("1.14 vs 1.10 at k = 1; 5.32 at k = 11 (n ≤ 16384).")],
    [B("Bayes λ "), T("reaches k/2 + (11 − k)/4 at 16384 targets: 3.35 vs 3.0 at k = 1, 5.43 vs 5.5 at k = 11.")],
    T("Unread directions are quartic: F = H = 0; the plug-in pays nothing, the Bayes code ¼ ln n each."),
  ], 8.45, 1.5, 4.45, 5.1, 15);
  source(s, "oracle/hmc_n16384/*.json, oracle/bayes_code_hmc.txt, oracle/hessian_vs_fisher.txt.");
  s.addNotes("The oracle is a network whose hidden state varies along the variable's direction and only k of the other eleven directions; nominal lambda for the one-dimensional subspace map is 5.5 regardless of k. The MAP code's slope follows one half trace F H-plus computed at the MAP, which is essentially k over 2: it pays only for directions the data read. The Bayes code by HMC pays for the other directions too, but not one half each: tilting the frame towards a coordinate the data do not span changes the readout only at second order in the angle, so the KL is quartic there, Fisher and Hessian both vanish, and each such direction contributes a quarter. The Bayes coefficient is k over 2 plus (11 minus k) over 4, which the 16384-target slopes match within about one standard error: 3.35 against 3.0 at k = 1, 5.43 against 5.5 at k = 11. The plug-in pays nothing for quartic directions. So my earlier slogan 'effective lambda equals readable directions over two' was a statement about the plug-in code, not about lambda.");
}

// 8 sharp readouts
{
  const s = slideOf("CONTENT", S2, "Sharp readouts: a finite-n crossover for Bayes, a persistent excess for the MAP");
  pills(s, ["plug", "bayes"]);
  fig(s, "oracle_sharpness_slide.png", 0.4, 1.4, { h: 5.25 });
  bullets(s, [
    T("k = 5 readable directions, continuous V, channel 0.1; readout logits scaled ×1 … ×100."),
    [B("n ≤ 2048: "), T("Bayes λ rises from 2.40 (k/2) to 5.32 (k). The KL is linear, not quadratic, until n resolves the smooth core (1-D quadrature: crossover n ~ s^1.3).")],
    [B("n ≤ 16384: "), T("Bayes falls back to 3.55 at ×100; 2.51 ± 0.08 at ×1.")],
    [B("MAP κ "), T("stays at 10.3 ± 4.5 at ×100 (14.2 at n ≤ 2048): κ > λ, a learner effect.")],
  ], 8.45, 1.5, 4.45, 5.1, 15);
  source(s, "oracle/bayes_code_hmc.txt, oracle/sharp_n16384/*.json, talk_slt/sharp_readout_1d_quadrature.txt.");
  s.addNotes("Second oracle experiment: make the readout sharper. For a near-deterministic readout behind a noise channel the KL divergence grows linearly rather than quadratically in the decision-relevant directions over a range of parameters, so the finite-n coefficient is one per direction instead of one half. A one-dimensional quadrature predicts this and predicts a crossover sample size growing roughly like the sharpness to the power 1.3. The Bayes code follows that script: from k over 2 to k at small n, and back towards k over 2 at sixteen thousand targets. The MAP code overshoots both and does not come back: a point estimate at the wrong place in a sharp landscape pays boundary points at the channel floor, so here kappa exceeds lambda.");
}

// 9 GPT-2 Fisher vs Hessian
pres.addSection({ title: S3 });
{
  const s = slideOf("CONTENT", S3, "GPT-2: the information equality fails; the Fisher is 2 % of the Hessian");
  pills(s, ["plug", "bayes"]);
  fig(s, "gpt2_fisher_vs_hessian_slide.png", 1.6, 1.35, { w: 10.0 });
  bullets(s, [
    T("Layer 11, d_V = 16 subspace (12,288 parameters) at its MAP: 12,032 curved directions + 256 gauge zeros. Regular on the Grassmannian, nominal λ = 6016."),
    T("Plug-in ½ tr(F H⁻¹) = 70.6 (held-out Fisher; 71.2 on training targets), measured κ = 49 ± 2. The Laplace count 6016 ln n exceeds any code's ceiling (6.46 nats per target, 2.2× label-only) until n ≈ 10⁴."),
    T("Hessian on 4096 fresh targets at the same MAP: Laplace count 5651, F/H 1.7 %, ½ tr(F H⁻¹) 73.7. The curvature is population curvature, not interpolation."),
  ], 0.5, 5.4, 12.3, 1.3, 13);
  source(s, "stage1_ioi/curvature_gpt2_layer11_full_heldout_fisher.json, _full_heldout_hessian.json, curvature_gpt2.txt, regret_slopes_map_code.txt.");
  s.addNotes("Now the real network. We computed the dense Hessian of the subspace map's loss at layer 11 of GPT-2, 12288 parameters, together with the empirical Fisher on fresh targets in the same eigendirections. The Hessian spectrum is nearly flat with exactly d_V squared gauge zeros: at this point the family is regular, so a Bayes code would pay all 6016 per log n, asymptotically. At a thousand targets that count is larger than any code can pay: every predictor mixes in the 5 percent channel, so a target costs at most 6.46 nats, 2.2 times the label-only code; the honest reading is that a Bayes code has not left its prior there. But the Fisher is two percent of the Hessian, with a median ratio below one percent per direction: moving the subspace changes every target's loss in the same way. Accumulating one half F over h direction by direction gives 70.6 for the plug-in, against 6016 for the Laplace count, and the measured slope is 49. This is the oracle mechanism, a hundredfold stronger.");
}

// 10 predicted vs measured
{
  const s = slideOf("CONTENT", S3, "½ tr(F H⁻¹) gets the order of κ; the nominal λ is 20–400× off");
  pills(s, ["plug", "bayes"]);
  fig(s, "predicted_vs_measured.png", 0.4, 1.4, { h: 5.2 });
  bullets(s, [
    T("Oracle: within the seed errors for k ≤ 7; 5.32 ± 0.20 vs 4.47 at k = 11."),
    [B("GPT-2, measured vs predicted: "), T("L11 79 vs 75 (d_V 4), 49 vs 71 (16), 56 vs 177 (64); L10 65 vs 52; L12 65 vs 112.")],
    T("κ stays at 49–79 at every locus and width; the prediction moves 52–177 (ratios 0.32–1.24)."),
    T("Not tested: inexact minimiser, training-set Hessian of an interpolating fit, local formula vs regret accumulated from n = 64."),
  ], 9.2, 1.5, 3.7, 5.1, 15);
  source(s, "oracle/hessian_vs_fisher.txt, oracle/hmc_n16384/ksweep_map_k*.json, stage1_ioi/curvature_gpt2_layer*_heldout_fisher.json, regret_slopes_map_code.txt.");
  s.addNotes("Putting the oracle and GPT-2 on one plot: horizontal axis the predicted coefficient, vertical the measured slope of the MAP code. The filled points are the plug-in prediction one half trace F H inverse; the open grey points are the same measurements placed at the nominal Bayes coefficient, and the dotted lines show the gap, 85-fold at d_V 16. With all five GPT-2 points in, the honest summary is: the plug-in formula gets the order of magnitude, where the nominal Bayes coefficient is off by a factor of 20 to 400, but it is not quantitatively predictive. The measured coefficient stays between 49 and 79 at every locus and every width, while the prediction moves from 52 to 177; measured over predicted ranges from 0.3 to 1.2. On the oracle the agreement is within the seed errors up to k = 7 and off by four standard errors at k = 11. Candidates I have not tested: the MAP is not the exact minimiser, the Hessian is the training-set Hessian of a fit that interpolates its thousand targets, and the formula is local while the regret is accumulated from n = 64.");
}

// 11 Bayes ladder
{
  const s = slideOf("CONTENT", S3, "The Bayes code is the right instrument only while λ ln n is small against n Δℓ");
  pills(s, ["plug", "bayes"]);
  fig(s, "bayes_vs_plugin_ladder.png", 0.4, 1.4, { h: 5.2 });
  bullets(s, [
    [B("Hierarchical equality (λ = 56): "), T("Bayes +733 vs plug-in +506 bits; Bayes charges the coupling family (−322) where the plug-in does not (+586).")],
    [B("InterpBench 18 (λ = 512): "), T("the Bayes code of the valid alignment is longer than label-only at 1024 targets (1526–2370 vs 1342 bits); concentrated by 8192.")],
    [B("GPT-2 (λ = 6016): "), T("not run. Any code is capped at 6.46 nats per target (≤ 2.2× label-only, drawn); the Laplace count exceeds the cap until n ≈ 10⁴.")],
  ], 9.3, 1.5, 3.6, 5.1, 15);
  source(s, "stage1_toy/bayes_code_hmc.txt, stage1_interpbench/case18_bayes_code_hmc.txt, stage1_ioi/gpt2_summary_v2.md, curvature_gpt2.txt.");
  s.addNotes("Here is where the Bayes code is and is not the right tool, as a ladder in lambda. On the small hierarchical-equality MLP, nominal lambda 56, a genuine HMC Bayes code beats the plug-in and also charges the expressive coupling family what the plug-in does not: exactly what the plan wanted from a Bayes code. On InterpBench, lambda 512, the Bayes code of the valid alignment is longer than knowing nothing at a thousand targets and only concentrates by eight thousand. On GPT-2 we did not run it; the Laplace count 6016 log n exceeds the hard ceiling of any code of the observed law, 6.46 nats per target or 2.2 times label-only, until about ten thousand targets, so the drawn point is that ceiling, an upper bound. The plug-in stays at a quarter to a half of the label-only code throughout. The Bayes coefficient of an alignment family is real but it charges valid explanations for prior mass on a huge Grassmannian.");
}

// 12 ladder panels
{
  const s = slideOf("CONTENT", S3, "A complexity ladder of four explanations of the same variable");
  pills(s, ["plug"]);
  fig(s, "ladder_panels.png", 0.45, 1.4, { w: 12.4 });
  bullets(s, [
    T("IIA passes all four; the parameter count cannot tell the coupling map on the trained network from the manufactured one (1,869,440 each)."),
    T("κ orders them, 0 → 49 → 226 → 988 per ln n, and the parameter-free Jacobian lens codes the name shortest (+3525 bits)."),
  ], 0.5, 6.05, 12.3, 0.95, 15);
  s.addNotes("This is the slide I would keep if I could keep one. Four explanations of the indirect-object name in GPT-2: the Jacobian lens, which reads the subspace off the network's own averaged Jacobian and fits nothing; a learned 16-dimensional subspace; an expressive coupling flow on the trained network; and the same flow on an untrained network, the manufactured alignment. Accuracy accepts all four. Raw parameter count puts the last two at the same 1.9 million. The plug-in coefficient orders them 0, 49, 226, 988, and the code length follows: the explanation with learning coefficient exactly zero is also the shortest. The coefficients here are the MAP code's kappa at n up to 1024, three seeds.");
}

// 13 kappa vs dV
{
  const s = slideOf("CONTENT", S3, "κ is set by the data, not by the dimension of the family");
  pills(s, ["plug", "bayes"]);
  fig(s, "kappa_vs_dv.png", 0.4, 1.4, { h: 5.2 });
  bullets(s, [
    T("Valid subspace: κ = 79, 49, 56 at d_V = 4, 16, 64, while the nominal λ grows 1528 → 6016 → 22528."),
    T("Controls stay 3–20× above at every d_V: manufactured 817 / 988 / 652, trained coupling 389 / 226 / 160."),
    T("Plug-in prediction 75 / 71 / 177 (diamonds): it grows with d_V, the measured κ does not."),
    T("To n = 32768 (3 seeds): valid 35–42, manufactured 726–826 per ln n; no interpolation plateau on GPT-2."),
  ], 9.5, 1.5, 3.4, 5.1, 15);
  source(s, "stage1_ioi/regret_slopes_map_code.txt, h_a3_gpt2_n32768_3seeds.json, curvature_gpt2_layer11*_heldout_fisher.json.");
  s.addNotes("If kappa were a parameter count it would grow with the subspace dimension. It does not: the valid alignment's coefficient is 79, 49 and 56 at d_V 4, 16 and 64, while the nominal Bayes coefficient grows fifteenfold. The number of Fisher-weighted directions is set by how many name directions carry sampling noise, a property of the data and the network. The plug-in prediction, the diamonds, does not follow either: 75, 71 and 177, so it gets the order of magnitude but not the trend. The controls stay an order of magnitude above at every d_V. With 32768 targets the valid slope is 35 to 42 depending on the fit range and the manufactured one 726 to 826, so the separation is not a small-n artefact. Note the untrained twin's subspace has a small kappa too; it is rejected by the loss term, not the complexity term.");
}

// 13b which term carries the gain
{
  const s = slideOf("CONTENT", S3, "Fit or Occam: calibrated, only the regret separates them");
  pills(s, ["plug"]);
  fig(s, "gain_decomposition.png", 0.4, 1.3, { h: 5.3 });
  bullets(s, [
    [B("Readout as is: "), T("the flows fit better (fit −2.0k bits): GPT-2 puts ≈ 96 % on the name, the subspace inherits it, the flows saturate the readout. The trained coupling overtakes near 20k targets.")],
    [B("One learned temperature per predictor: "), T("fit term 0.0k vs manufactured, −0.1k vs coupling. The verdict is the learning cost alone: +9.3k and +1.1k bits at 32768, no crossover.")],
    [B("Label-only, twin: "), T("all fit term in both (falsification); the manufactured alignment beats label-only by 109k bits at 32768.")],
  ], 8.15, 1.4, 4.75, 5.2, 14);
  source(s, "stage1_ioi/gain_decomposition_gpt2_n32768{,_temp}.json (bitsev.experiments.gain_decomposition; MAP code, 32768 targets, 3 seeds; --temperature).");
  s.addNotes("Slide 4 split a gain into a falsification term, linear in n, and an Occam term, logarithmic. Here is that split measured, for the valid layer-11 subspace against the manufactured alignment on the left and against the coupling map on the trained network on the right. Black is the gain, pink dashed the fit term, the difference of the two best-in-family fits on the same targets, blue the regret term, the difference of the learning costs, which carries kappa log n. Single-map MAP code, 32768 targets, three seeds. Top row, the readout as it is: the flows fit the targets better than the faithful subspace, on held-out pairs as well. That is calibration: the targets are the algorithm's deterministic outputs behind a 5 percent channel, GPT-2 itself puts about 96 percent on the name, the subspace swap inherits that, the flows push the readout to saturation. So the fit term runs against the valid alignment, and the trained coupling overtakes it near twenty thousand targets. Bottom row: give every predictor, valid, controls and forgery alike, one learned inverse temperature on the readout. The calibration gap closes, the valid subspace's clean-target loss drops to the channel floor, and the fit term is zero against the manufactured alignment and within noise against the coupling. What separates them is only the cost of learning: 9300 bits against the forgery and still growing, 1100 against the coupling. Against label-only and the untrained twin the gain is all fit term in both rows. So with the calibration confound removed, the verdict between a valid and a manufactured explanation is a pure learning-coefficient statement. Caveat for large n, backup slide: at 131072 targets, three seeds, the tempered margin against the forgery stops growing and plateaus at nine to ten thousand bits, no crossover; without the temperature it shrinks to about six thousand, and the trained coupling wins. Beyond about thirty thousand targets the excess loss of this learner is set by the optimiser's noise, about a hundredth of a nat per target, not by its learning coefficient.");
}

// 13c same family, trained vs untrained network
{
  const s = slideOf("CONTENT", S3, "Two coupling maps, the same fit: only the regret differs");
  pills(s, ["plug"]);
  fig(s, "gain_coupling_pair.png", 1.4, 1.3, { h: 4.3 });
  bullets(s, [
    [B("Same family, same size: "), T("the coupling map on the trained network (layer 11) against the same map on its untrained twin (layer 10, the manufactured alignment); 1,869,440 parameters each, IIA ≥ 0.99 for both at 32768 targets.")],
    [B("Fit term +15 bits "), T("(+95 with a temperature): both flows saturate the readout. The +8.0k-bit gain is all regret: κ 146 vs 826 per ln n (n ≥ 64).")],
  ], 0.6, 5.7, 12.2, 0.95, 14);
  source(s, "difference of the two splits in stage1_ioi/gain_decomposition_gpt2_n32768{,_temp}.json (3 seeds); κ from h_a3_gpt2_large_n.txt.");
  s.addNotes("The cleanest version of the previous slide needs no temperature. Compare two explanations from the same family: the expressive coupling flow on the trained GPT-2 at layer 11, and the same flow on the untrained twin at layer 10, which is the manufactured alignment. Same architecture of map, same parameter count, nearly perfect interchange accuracy for both once they have seen enough pairs. Both saturate the readout, so the fit term is zero within noise: 15 bits at 32768 targets, 95 with the temperature. The 8000-bit gain is entirely the regret term, the difference in learning cost: the flow on the trained network has a learner coefficient of about 150 per log n, the one on the untrained network about 830. Parameter count, accuracy and the best achievable loss cannot tell these two apart; the learning coefficient can. This is the Occam regime of slide 4 in its pure form.");
}

// 14 models
pres.addSection({ title: S4 });
{
  const s = slideOf("CONTENT", S4, "Four language models, one verdict, no retuning");
  pills(s, ["plug"]);
  fig(s, "models_verdicts.png", 0.4, 1.45, { w: 9.0 });
  bullets(s, [
    T("Identical 1024 targets on every model; label-only 4324.5 bits."),
    T("Valid subspace at the last 2–4 layers: +3000 to +3400 bits; twin and wrong locus rejected at α = 0.01 within 20 targets (GPT-2, Qwen2.5, Llama)."),
    T("Coupling on the trained net: 925–1665 bits behind the subspace at n = 1024 (on GPT-2 it overtakes near 20k targets, slide 14)."),
    T("Manufactured, n = 1024: 1723–1986 bits longer than label-only (single-map code); 147 longer on Qwen3 (mixture, IIA 0.71)."),
  ], 9.65, 1.5, 3.25, 5.1, 15);
  source(s, "stage1_ioi/tier2_models.txt, gpt2_summary_single_map_map0.md, *_summary.md (gains vs controls).");
  s.addNotes("The GPT-2 protocol carries to models twenty to sixty times larger without any change. At the loci where the name is intervenable the valid subspace gains three thousand bits or more on every model, and the controls are rejected within the first twenty targets by the anytime criterion. At a thousand targets the coupling family on the trained network codes about a thousand bits behind the subspace, which is the complexity charge that IIA does not see; on GPT-2 it overtakes the subspace near twenty thousand targets, by calibration (slide 14). At a thousand targets the manufactured alignment never beats the label-only code: 1700 to 2000 bits worse under the single-map learner, 150 bits worse on Qwen3 under the averaging learner, which rescues confident errors. Learners: GPT-2, Qwen2.5 and Llama use the single-map MAP code, Qwen3 the mixture of four.");
}

// 15 Pythia
{
  const s = slideOf("CONTENT", S4, "Competence onset: the code tracks the task and keeps improving after it is solved");
  pills(s, ["plug"]);
  fig(s, "pythia_onset.png", 0.4, 1.35, { h: 5.3 });
  bullets(s, [
    T("Pythia-1.4b training checkpoints, MAP code, one seed."),
    T("Valid layer-20 code: 4748 bits at step 1k (worse than label-only) → 1451 at 3k → 853–910 from 8k on."),
    T("Steps 4k → 8k: accuracy already 0.95, code still drops 655 bits."),
    T("At step 1k the untrained twin's raw name copy (3751 bits) beats the trained net."),
    T("Manufactured (IIA 0.78): 643 bits longer than label-only at every step."),
  ], 9.15, 1.5, 3.75, 5.1, 15);
  source(s, "stage1_ioi/pythia14_checkpoint_series.txt (accuracy at step 3000 not measured).");
  s.addNotes("A training trajectory: Pythia-1.4b at seven released checkpoints. The code of the valid alignment falls with the task accuracy, from worse than label-only at step 1000 to about 900 bits once the task is solved, and keeps falling between steps 4000 and 8000 when accuracy is already saturated: the representation keeps getting cleaner after competence. At step 1000 the trained network is worse than its own untrained twin, which still carries the raw name tokens; training first destroys the copy before computing the task. A question for this audience: do local learning coefficient estimates of the network itself show a transition between steps 1000 and 4000, and does it line up with this?");
}

// 16 MIB
{
  const s = slideOf("CONTENT", S4, "MIB arithmetic: the code picks the family that is faithful at each locus");
  pills(s, ["plug"]);
  fig(s, "mib_arithmetic.png", 1.5, 1.3, { w: 10.2 });
  bullets(s, [
    T("Gemma-2-2b: the carry is linearly intervenable at layer 20 only (IIA 0.96 vs floor 0.78; +754 bits over the floor)."),
    T("Llama-3.1-8B: the 16-dim subspace never leaves the floor (IIA ≈ 0.5, as for DAS in MIB); the coupling map reads the carry at layers 20–24 (IIA 0.94–0.96; +2348 bits over the floor at 24) and reaches IIA 0.14 on the untrained twin."),
  ], 0.5, 5.98, 12.3, 1.05, 14);
  s.addNotes("A second task from the MIB benchmark: two-digit addition, variable the carry. The reference here is not label-only but the trivial floor, the model's own answer on the base input, which is right whenever the two carries agree. On Gemma a linear subspace reads the carry at exactly one layer, by 750 bits over the floor. On Llama no 16-dimensional subspace leaves the floor, consistent with linear methods scoring at chance on this task in MIB, but the nonlinear coupling map does, by 2350 bits at layer 24, and the same map on an untrained Llama gets nowhere. So the code ranks the families in opposite orders on the two models and in each case picks the faithful one; a single family or an accuracy threshold would have missed one of them. Learner: single-map MAP code.");
}

// 17 asks for SLT
pres.addSection({ title: S5 });
{
  const s = slideOf("CONTENT", S5, "What this asks of singular learning theory");
  const cw = 6.05, ch = 2.55;
  card(s, 0.5, 1.45, cw, ch, "1. Regret of plug-in learners in singular models",
    "κ has the order of ½ tr(F H⁻¹), a Takeuchi-type, Fisher-weighted count of directions, but not its trend (within 3×). Measured: κ < λ (smooth, deterministic constraints), κ > λ (sharp readouts), κ = MN/2 > λ (least squares in RRR). Is there a singular-learning account of the MAP's regret?",
    { headColor: C.accent1, fontSize: 16 });
  card(s, 6.78, 1.45, cw, ch, "2. Regular, yet Fisher-free",
    "Alignment families on trained networks are regular on the Grassmannian (H full rank) but F ≈ 0.01 H: the targets are deterministic functions of the state behind a noise channel. What is the right effective coefficient when F ≪ H, and how does λ(n) cross over?",
    { headColor: C.accent1, fontSize: 16 });
  card(s, 0.5, 4.2, cw, ch, "3. Estimators",
    "SGLD LLC tracked the plug-in coefficient on the oracle (0.85–4.20 for k/2 = 0.5–5.5) and read λ on RRR at n = 4096, but diverges at n = 65536 unless the step shrinks with n; on saturated transformer readouts its chains never plateau.",
    { headColor: C.accent2, fontSize: 16 });
  card(s, 6.78, 4.2, cw, ch, "4. The free energy of an explanation",
    "Explanations as programs: the Jacobian lens (λ = 0) is the shortest code of the IOI name. Which priors on alignment maps would make a Bayes code competitive, and is a scale-dependent λ(n) the right complexity for explanations?",
    { headColor: C.accent2, fontSize: 16 });
  s.addNotes("Four places where I think this audience can help. First, the operative complexity is the regret coefficient of a plug-in learner, and we have measured it on both sides of lambda; I do not know a singular-learning-theory account of MAP regret that covers all three regimes. Second, the alignment families are a curious case: regular in the Hessian sense, but almost every direction is Fisher-free because the targets are deterministic functions of the hidden state; what is the right effective coefficient there? Third, the SGLD local learning coefficient behaves as you would expect on the exact and oracle cases but its step must shrink with n, and on saturated readouts the chains drift along flat regions. Fourth, the shortest explanation in GPT-2 has learning coefficient zero; I would like to know what prior would make a Bayes code of explanations behave like the plug-in.");
}

// 18 open questions
{
  const s = slideOf("CONTENT", S5, "Open questions for the discussion");
  bullets(s, [
    [B("κ vs ½ tr(F H⁻¹): "), T("measured 49–79 at every GPT-2 locus and width, predicted 52–177. Inexact minimiser, training-set Hessian of an interpolating fit, or a singular correction?")],
    [B("Does κ settle? "), T("InterpBench 18: 23–42 per ln n at 32768 targets, still rising with the fit range; flat below the frame's 1040 parameters.")],
    [B("Closed form for λ(n) "), T("under sharp readouts: the 1-D crossover scales like s^1.3, not s².")],
    [B("Anytime validity: "), T("the gain is an e-process against a simple null; against a composite control family only a bound exists.")],
    [B("Training dynamics: "), T("does the network's own LLC change where the code of the explanation drops (Pythia steps 1k–4k)?")],
    [B("Priors for explanations: "), T("the Bayes code on λ = 56 was better than the plug-in; at 512 and 6016 it is hopeless. Is there a prior that interpolates?")],
  ], 0.7, 1.5, 11.9, 5.2, 17);
  s.addNotes("These are the questions I would most like to discuss, in rough order of how much I think theory could say. The first is quantitative: the plug-in prediction is good on the oracle up to k = 7, at d_V 4 and at layer 10, but up to three times too high at d_V 64 and layer 12, and it does not follow the measured coefficient across loci or widths. The second and third are about scale dependence: whether the plug-in coefficient on InterpBench settles, and whether the sharp-readout crossover has a closed form. The anytime point is about turning the gain into a valid test against a whole family of controls. The last two connect to developmental interpretability and to priors on explanations.");
}

// 19 summary
{
  const s = slideOf("CLOSING_DARK", S5, "Summary");
  s.addText([
    { text: "Score an explanation by the code length of the algorithm's counterfactuals; gains are log Bayes factors against controls.", options: { bullet: true, breakLine: true } },
    { text: "For Bayes codes the complexity is λ: exact on reduced-rank regression, k/2 + (d−1−k)/4 on the oracle, huge on trained networks (6016 on GPT-2), where it charges valid explanations too.", options: { bullet: true, breakLine: true } },
    { text: "The codes that work are plug-in codes; their complexity κ counts Fisher-weighted directions (½ tr(F H⁻¹) gets its order): 0 for the Jacobian lens, 49 for the valid subspace, 988 for the manufactured alignment (n ≤ 1024).", options: { bullet: true, breakLine: true } },
    { text: "Against a manufactured alignment, once every predictor is calibrated, the fit term vanishes: the verdict is the learning cost alone (32768 targets).", options: { bullet: true, breakLine: true } },
    { text: "What this application needs is a singular learning theory of plug-in learners.", options: { bullet: true, bold: true } },
  ].map((r) => Object.assign(r, { options: Object.assign({ paraSpaceAfter: 14 }, r.options) })),
  { x: 0.8, y: 1.9, w: 11.7, h: 4.6, fontSize: 20, color: C.background1, margin: 0, valign: "top",
    isTextBox: true });
  s.addNotes("To summarise. The score is a prequential code length, and the gain against a control is a log Bayes factor. For a Bayes code the complexity is the learning coefficient, which we can read exactly where it is known, but on trained networks it is enormous and charges the valid explanation as much as the false one. The codes that separate valid from manufactured explanations are plug-in codes, and their complexity is a Fisher-weighted count of directions that orders explanations by a factor of twenty. That ordering is what decides the verdict: with enough data the forgery fits the targets as well as the faithful explanation or better, and the whole margin is the learning cost. That is a learner coefficient, and I would like a theory of it. Thank you.");
}

// ================================================================== backup
pres.addSection({ title: SB });
{
  const s = slideOf("SECTION_DARK", SB, "Backup");
  s.addText("Noise channel · batching correction · Bayes-code settings · InterpBench · GPT-2 to 32768 targets · Jacobian lens · learner dependence · LLC estimator · Pythia and MIB tables · compute",
    { placeholder: "body" });
  s.addNotes("Backup slides, in the order the questions usually come.");
}
{
  const s = slideOf("BACKUP", SB, "Every predictor must code the observed label law");
  formula(s, [T("p"), { text: "obs", options: { subscript: true } }, T("(y) = (1 − ε)·p(y) + ε/K,   ε = 0.05")], 0.5, 1.5, 12.3, 0.8, 28);
  card(s, 0.5, 2.6, 6.0, 3.6, "What went wrong first",
    "The first implementation coded the clean law p(y). On InterpBench 18 the correct alignment coded clean targets at 0.13 bits each but paid about 39 bits per flipped label: 92 % of its code after position 128. Every verdict came out with the wrong sign.",
    { headColor: C.accent5, fontSize: 17 });
  card(s, 6.83, 2.6, 6.0, 3.6, "Why randomized response",
    "It gives deterministic counterfactual targets a conditional law with full support, representable at finite logits, so the model is realizable and Watanabe's expansion applies. Two configurations that both reproduce the law make identical predictions: no term linear in n between them.",
    { headColor: C.accent1, fontSize: 17 });
  source(s, "paper draft, sections 2.5 and 7.1 (reports/stage1_writeup/stage1_results.tex).");
  s.addNotes("The methods finding that preceded all results. The frozen readout is far sharper than a 5 percent randomized response, and an orthogonal map cannot temper it, so if the predictor codes the clean law it is punished exactly where it is right. After the fix every predictor codes the observed law, and every verdict flipped to the expected sign. The channel is also what makes the target law realizable, which is the condition for the free-energy expansion.");
}
{
  const s = slideOf("BACKUP", SB, "Batching correction for geometric block schedules");
  formula(s, "cumulative excess ≈ κ · (r − 1)/ln r · ln n", 0.5, 1.5, 12.3, 0.8, 28);
  bullets(s, [
    T("Block j codes ≈ (r − 1) b_j targets at excess κ/b_j each: κ(r − 1) nats per block, ln n / ln r blocks."),
    T("Factor 1.4427 for doubling blocks, 1.1204 for r = 1.25, 2.1640 for r = 4; every slope in the talk is divided by it."),
    T("Exact check, Gaussian mean (regular): plug-in per sample 0.5001 (0.5000); doubling blocks 0.7213 (0.7213); Bayes with doubling blocks 0.7208."),
    T("Singular Bayes code, RRR (n = 4096, 8 seeds): per sample / doubling / quadrupling 3.15 / 3.13 / 2.84 for (3,3,3,0); 5.75 / 6.00 / 6.24 for (4,4,4,0); 2.11 / 1.94 / 1.98 for (2,2,2,2); 7.18 / 7.44 / 7.68 for (4,4,4,2)."),
  ], 0.7, 2.6, 11.9, 4.0, 17);
  source(s, "paper draft, section 3.7; rrr/README.md (schedule check).");
  s.addNotes("Codes are computed in blocks with geometrically growing boundaries, which inflates the logarithmic coefficient by a known factor. The derivation is three lines; it was checked exactly for the Gaussian mean in the research plan, and here for a singular Bayes code on reduced-rank regression, where per-sample, doubling and quadrupling schedules agree after the correction within their standard errors.");
}
{
  const s = slideOf("BACKUP", SB, "Bayes codes: settings and what had to be repaired");
  pills(s, ["bayes"]);
  card(s, 0.5, 1.45, 6.0, 4.9, "HMC settings",
    [B("Oracle: "), T("N(0,1) prior on the raw frame, 4 chains × 100 samples after 200 warm-up, 10 leapfrog steps, step adapted to ≈ 0.7 acceptance, chains continue across blocks."), { text: "", options: { breakLine: true } },
     B("InterpBench 18: "), T("same prior, 20 leapfrog steps; chains restart from the prior at every block (acceptance 0.52–0.82)."), { text: "", options: { breakLine: true } },
     B("Hierarchical equality: "), T("4 × 100, 200 warm-up, 20 leapfrog, step 0.01, chains continue."), { text: "", options: { breakLine: true } },
     B("RRR: "), T("64 chains × 200 samples per block on the sufficient statistics, 300 warm-up, thinning 2.")],
    { headColor: C.accent2, fontSize: 14 });
  card(s, 6.83, 1.45, 6.0, 4.9, "Failures on saturated readouts",
    [B("Laplace collapses to the prior: "), T("curvature ≈ 0 in most directions; held-out loss 2.27 nats, worse than uniform (1.10)."), { text: "", options: { breakLine: true } },
     B("Gauge: "), T("a Gaussian prior on the raw frame shrinks it to norm 1.6·10⁻⁴; fixed by a QR retraction after every step."), { text: "", options: { breakLine: true } },
     B("Langevin from the MAP: "), T("200 steps; the step size is a hedge knob, not a converged posterior (hedge moves the GPT-2 adversary by ≈ 40 bits)."), { text: "", options: { breakLine: true } },
     B("Rotation exp(S): "), T("flat only to first order; the Laplace code needs the gauge-free subspace map (2.75 ± 0.13 vs λ = 2.5 on the oracle).")],
    { headColor: C.accent5, fontSize: 14 });
  source(s, "oracle/bayes_code_hmc.txt, stage1_interpbench/case18_bayes_code_hmc.txt, stage1_toy/bayes_code_hmc.txt, rrr/README.md; paper sections 7.3–7.4.");
  s.addNotes("The settings of every Bayes code in the talk, and the four things that broke when we took a Laplace code validated on the oracle to trained networks. The common cause is that the data gradient of the map is nearly zero on a saturated readout, so the prior, the loss scaling and the sampler noise all dominate it. One correction to the archived InterpBench text file: the paper draft states that the InterpBench HMC chains restart at every block, while the archived table header says they continue; the paper is the corrected version.");
}
{
  const s = slideOf("BACKUP", SB, "InterpBench case 18: interpolation regime, then slow growth");
  pills(s, ["plug"]);
  fig(s, "interpbench_regret_vs_n.png", 0.5, 1.4, { h: 5.2 });
  bullets(s, [
    T("Variable freq, locus resid_2 (10 positions × 26 = 260 wide), d_V 4: 1040 frame parameters, nominal λ = 512."),
    T("MAP κ: 8.4 ± 2.9 on n ≤ 1024 (interpolation); 16.8 ± 2.8 at 8192; 22.6–31.7 at 32768 depending on fit range."),
    T("Misspecified locus: 91–186 (excess-loss drift)."),
    T("Fisher: 90 % of its trace in 7 of 1024 directions; Hessian indefinite (654 negative eigenvalues), so ½ tr(F H⁻¹) is unusable here."),
  ], 7.5, 1.5, 5.4, 5.1, 15);
  source(s, "stage1_interpbench/case18_h_a3_n8192.txt, case18_h_a3_n32768.txt, case18_fisher_spectrum.txt, case18_curvature.txt.");
  s.addNotes("InterpBench is where the first reading went wrong: the regret of the valid alignment was flat up to a thousand targets, which I first read as saturation. It was the interpolation regime of a 1040-parameter frame; beyond it the regret grows at 20 to 45 per log n, still far below the nominal 512, and not yet settled at 32768 targets. The fitted point here is a flat, indefinite plateau, so the local curvature formula cannot be evaluated; GPT-2 is the clean case.");
}
{
  const s = slideOf("BACKUP", SB, "GPT-2 to 32768 targets: the separation persists");
  pills(s, ["plug"]);
  fig(s, "gpt2_regret_vs_n.png", 0.5, 1.4, { h: 5.2 });
  bullets(s, [
    T("Single-map MAP code, layers 11 (valid, coupling, twin) and 10 (manufactured)."),
    T("Valid subspace: 45–50 per ln n from n = 32 to 16384 on one seed; 35–42 on three seeds at 32768."),
    T("Manufactured: 6775 nats by 32768 (one seed), local coefficient peaking at 1700–2450 near n = 1000."),
    T("No plateau at the frame's 12,288 parameters."),
  ], 7.9, 1.5, 5.0, 5.1, 15);
  source(s, "stage1_ioi/h_a3_gpt2_large_n.txt, h_a3_gpt2_n32768_3seeds.json; paper table 'GPT-2 IOI, single-map MAP code'.");
  s.addNotes("The regret curves behind the coefficients. On GPT-2 there is no interpolation plateau: the valid subspace pays roughly 45 to 50 per log n throughout, and the manufactured alignment pays its largest cost between 500 and 2000 targets, where new base-source combinations expose what the flow memorised. At 32768 targets the manufactured alignment's regret is about eighteen times the valid one's.");
}
{
  const s = slideOf("BACKUP", SB, "Code rate per block: the fit term without final fits");
  pills(s, ["plug"]);
  fig(s, "code_rate_per_block.png", 0.4, 1.45, { h: 4.7 });
  bullets(s, [
    T("Each doubling block is coded before it is fitted: its rate is an honest held-out loss of the predictor fitted on the prefix."),
    T("Last block (16384 → 32768) above the 0.360-nat floor: valid 0.048, manufactured 0.021, trained coupling 0.011."),
    T("Clean targets, final fits (median): valid 0.092 nats, flows 0.050–0.053 (floor 0.050): the subspace is under-confident, not wrong (IIA 0.997–1.0)."),
    T("At n = 1024 (headline runs): valid vs manufactured +5370 bits = fit −464 + regret +5834."),
  ], 8.4, 1.5, 4.5, 5.1, 15);
  source(s, "stage1_ioi/gain_decomposition_gpt2_n32768.json; results/stage1_ioi_gpt2_map0 for n = 1024.");
  s.addNotes("The same comparison without relying on the final fits. Each block of the prequential code is coded before the predictor sees it, so the block rate is a held-out loss of the map fitted on everything before. The valid subspace levels off about five hundredths of a nat above the channel floor; both flows keep falling below it. The gap is calibration on clean targets: the valid subspace's median loss on an unflipped target is 0.092 nats against the channel's 0.050, the flows sit at the floor. At the talk's headline 1024 targets the split is the same: the fit term already slightly favours the manufactured alignment and the regret carries the verdict.");
}
{
  const s = slideOf("BACKUP", SB, "To 131072 targets: a plateau with calibration, a decline without");
  pills(s, ["plug"]);
  fig(s, "gain_large_n.png", 0.95, 1.3, { w: 11.4 });
  bullets(s, [
    T("3 seeds, Adam steps raised to 10 passes per block (the fixed 400-step budget gives the same curves)."),
    T("Tempered: +9.1k ± 1.1k vs the forgery (peak ≈ 9.9k at 32768–65536), tie with the trained coupling. As is: +5.9k ± 2.1k and −6.9k (the 0.055-nat calibration deficit grows linearly)."),
    T("Same-family pair keeps separating: +12.8k / +9.9k. Beyond ~32768 targets constant-rate Adam leaves ≈ 0.01 nats per target of excess loss, comparable to the κ ln n growth; the fit/regret split is not reliable there."),
  ], 0.6, 5.15, 12.2, 1.5, 13);
  source(s, "stage1_ioi/large_n_gains.txt; gain_decomposition_gpt2_n131072_ep10{,_temp}.json.");
  s.addNotes("What happens far beyond the talk's headline sample sizes. Three seeds each at 131072 targets, with the learner's step budget raised to ten passes over the prefix; the fixed budget gives the same curves, so this is not an optimisation-budget artefact. With a temperature, the margin of the valid subspace over the forgery stops growing after about thirty thousand targets and stays at nine to ten thousand bits; against the coupling on the trained network it is a tie. Without the temperature, the calibration deficit of the subspace, about five hundredths of a nat per target, accumulates linearly and the margin over the forgery shrinks, while the trained coupling wins by seven thousand bits. The same-family pair, coupling on the trained versus the untrained network, keeps separating. Why the plateau: at this size the plug-in learner's excess loss per target is set by the noise of a constant-learning-rate optimiser, about a hundredth of a nat, which is as large as the kappa over n term. So the asymptotic regime of a plug-in code depends on the optimiser as much as on the model: one more reason a theory of plug-in learners is needed.");
}
{
  const s = slideOf("BACKUP", SB, "Jacobian lens and Jacobian consistency");
  fig(s, "jacobian_consistency.png", 0.5, 1.5, { h: 4.3 });
  bullets(s, [
    T("J-lens frame: span of the names' averaged-Jacobian directions (W_U J_ℓ); no fitted parameters, κ = 0."),
    T("GPT-2 layer 11: IIA 0.98, +3525 bits (16 training names; +3529 with all 32); layer 12: +3626. Learned subspace: +3302 (mixture), +3385 (MAP)."),
    T("Fails on every untrained twin (IIA 0.04); faithful at the last 2–4 layers of Qwen2.5-0.5B, Pythia-1.4b, Qwen3-1.7B."),
    T("JC = ‖δ − W_U J Δ‖/‖δ‖: manufactured 2.05–2.24 vs 0.15–0.38 for valid maps."),
  ], 8.85, 1.5, 4.05, 5.1, 15);
  source(s, "stage1_ioi/jlens_gpt2.txt, jlens_gpt2_jacobian_consistency.json, jlens_models.txt.");
  s.addNotes("The Jacobian lens builds the alignment analytically from the network's averaged Jacobian, so its code has no learning cost at all, and on GPT-2 it is the shortest code of the name at the layers where the name is readable. The consistency statistic compares the actual output change of an intervention with what the averaged Jacobian predicts: the manufactured alignment moves the output through local sensitivity the average does not show, six to ten times more than the valid maps.");
}
{
  const s = slideOf("BACKUP", SB, "The learner, not a complexity charge, decides the absolute verdict");
  pills(s, ["plug"]);
  fig(s, "learner_decides_verdict.png", 0.5, 1.45, { w: 12.3 });
  bullets(s, [
    T("Averaging four confidently wrong maps hedges for the adversary (2.6 bits per target in the last block); a single map pays 6–7 bits per target throughout."),
    T("The comparative verdict (valid beats manufactured) holds under every learner: ≈ 2600 bits (mixture), ≈ 5300 bits (MAP)."),
  ], 0.5, 6.15, 12.3, 0.85, 14);
  s.addNotes("The one result where the absolute verdict depends on the learner. Under the canonical mixture of four plug-in members the layer-10 manufactured alignment beats label-only by 704 bits; under one member, the MAP alone, or the MAP with a Langevin hedge it is a thousand to two thousand bits worse. The hedge itself moves the code by 40 bits, so this is not a Bayes complexity charge: it is the averaging. The valid alignment barely changes across learners because maps that have found the subspace agree.");
}
{
  const s = slideOf("BACKUP", SB, "The SGLD local learning coefficient as a cross-check");
  pills(s, ["plug", "bayes"]);
  fig(s, "oracle_effective_coefficient.png", 0.5, 1.4, { h: 5.0 });
  bullets(s, [
    T("Estimator: tempered posterior at β = 1/ln n localised at the MAP; λ̂ = nβ (E[Lₙ] − Lₙ*)."),
    [B("Oracle (n = 2048): "), T("0.85, 1.14, 1.78, 2.57, 4.20 for k = 1, 2, 4, 7, 11: close to k/2 and to the MAP slope, below the Bayes λ at small k.")],
    [B("RRR, n = 4096: "), T("3.38 / 6.25 / 2.16 / 7.91 for λ = 3.5 / 6 / 2 / 7.5. At 65536 the same step diverges.")],
    [B("InterpBench: "), T("0.96 ± 0.69 to 27 ± 14 depending on step and chain length; no chain plateaus.")],
  ], 6.2, 1.5, 6.7, 5.1, 15);
  source(s, "oracle/llc_vs_regret_slope.txt, rrr/README.md, stage1_interpbench/case18_llc.txt, case18_llc_traces*.txt.");
  s.addNotes("We used the Lau et al. estimator as an independent check. Where the answer is known it works with the usual step sensitivity, and on the oracle it agrees with the plug-in coefficient rather than with the Bayes one at small k, which I find interesting in itself. On the saturated InterpBench readout the loss is exactly flat around the MAP until a target crosses the decision boundary, so the chains either sit still or drift away; the estimator measures the extent of the flat region, not a power law at a point.");
}
{
  const s = slideOf("BACKUP", SB, "Pythia-1.4b checkpoints: code in bits (IIA)");
  pills(s, ["plug"]);
  const hdr = ["step", "IOI accuracy", "valid L16", "valid L20", "twin L20", "coupling L20", "manufactured L20"];
  const rows = [
    ["1000", "0.09", "4800 (0.15)", "4748 (0.15)", "3751 (0.39)", "5524 (0.61)", "4968 (0.78)"],
    ["2000", "0.43", "2857 (0.57)", "2833 (0.58)", "3751 (0.39)", "4306 (0.75)", "4968 (0.78)"],
    ["3000", "–", "1544 (0.94)", "1451 (0.93)", "3751 (0.39)", "2778 (0.92)", "4968 (0.78)"],
    ["4000", "0.95", "1755 (0.93)", "1551 (0.91)", "3751 (0.39)", "2873 (0.87)", "4968 (0.78)"],
    ["8000", "1.00", "1023 (0.99)", "896 (1.00)", "3751 (0.39)", "1953 (0.97)", "4968 (0.78)"],
    ["16000", "1.00", "1032 (0.99)", "853 (1.00)", "3751 (0.39)", "1796 (0.95)", "4968 (0.78)"],
    ["143000", "1.00", "1545 (0.94)", "910 (0.99)", "3751 (0.39)", "2226 (0.95)", "4968 (0.78)"],
  ];
  const head = hdr.map((h) => ({ text: h, options: { bold: true, color: C.background1, fill: { color: THEME.colors.dk1 } } }));
  s.addTable([head, ...rows.map((r) => r.map((c) => ({ text: c })))], { x: 0.5, y: 1.5, w: 12.3, fontSize: 15,
    color: C.text1, border: { type: "solid", pt: 0.5, color: "C3C2B7" }, rowH: 0.48, align: "center", valign: "middle" });
  s.addText("Label-only ≈ 4325 bits; single-map MAP code, one seed, 1024 targets. Twin and manufactured do not depend on the checkpoint.",
    { x: 0.5, y: 5.6, w: 12.3, h: 0.5, fontSize: 14, color: C.text2, margin: 0, isTextBox: true });
  source(s, "stage1_ioi/pythia14_checkpoint_series.txt (transcribed).");
  s.addNotes("The full table behind the competence-onset figure. Accuracy at step 3000 was not measured. Checkpoints 32000 and 64000 were not run.");
}
{
  const s = slideOf("BACKUP", SB, "MIB two-digit addition, carry: gain over label-only in bits (IIA)");
  pills(s, ["plug"]);
  const hdr = ["layer", "Llama subspace", "Llama coupling", "Llama twin", "", "layer", "Gemma subspace", "Gemma coupling", "Gemma twin"];
  const L = [["0 (floor)", "+2715 (0.48)", "+2863 (0.01)", "−1038 (0.00)"], ["8", "+2787 (0.48)", "+2818 (0.00)", "−904 (0.00)"],
    ["16", "+3375 (0.47)", "+4429 (0.69)", "−929 (0.00)"], ["20", "+3768 (0.49)", "+1876 (0.96)", "−997 (0.00)"],
    ["24", "+3764 (0.51)", "+5063 (0.94)", "−1035 (0.00)"], ["28", "+3528 (0.49)", "+4085 (0.86)", "−1056 (0.00)"],
    ["32", "+3156 (0.44)", "+415 (0.43)", "−1083 (0.00)"]];
  const G = [["0 (floor)", "+719 (0.78)", "+686 (0.78)", "−1354 (0.09)"], ["8", "+744 (0.78)", "+800 (0.88)", "−439 (0.52)"],
    ["12", "+770 (0.78)", "+919 (0.89)", "−542 (0.50)"], ["16", "+888 (0.79)", "+814 (0.86)", "−765 (0.44)"],
    ["20", "+1473 (0.96)", "+1218 (0.97)", "−878 (0.42)"], ["24", "+1116 (0.84)", "+1054 (0.95)", "−1082 (0.35)"],
    ["26", "+528 (0.78)", "+643 (0.92)", "−1320 (0.33)"]];
  const head = hdr.map((h) => ({ text: h, options: { bold: true, color: C.background1, fill: { color: THEME.colors.dk1 } } }));
  const body = L.map((r, i) => [...r, "", ...G[i]].map((c) => ({ text: c })));
  s.addTable([head, ...body], { x: 0.5, y: 1.5, w: 12.3, colW: [1.05, 1.55, 1.55, 1.4, 0.2, 1.05, 1.55, 1.55, 1.4], fontSize: 13,
    color: C.text1, border: { type: "solid", pt: 0.5, color: "C3C2B7" }, rowH: 0.45, align: "center", valign: "middle" });
  s.addText("Label-only 7602.7 bits (Llama, whole-number answer, 199 symbols) and 2280.9 bits (Gemma, first digit, 10 symbols). Manufactured: Llama −2160 (IIA 0.14), Gemma −1075 (IIA 0.62).",
    { x: 0.5, y: 5.35, w: 12.3, h: 0.7, fontSize: 14, color: C.text2, margin: 0, isTextBox: true });
  source(s, "stage1_mib/arithmetic.txt (transcribed); single-map MAP code (run settings), one seed.");
  s.addNotes("The table behind the MIB figure. The floor row is the wrong-locus control here: the interchange at layer 0 does nothing, so the predictor is the model's own answer, which is right whenever the two carries agree.");
}
{
  const s = slideOf("BACKUP", SB, "Compute");
  bullets(s, [
    T("Hierarchical equality, InterpBench, reduced-rank regression: CPU nodes (AMD EPYC). HE main run 2 h 34 min; InterpBench 18 at d_V 4, 3 h 56 min."),
    T("GPT-2: one A100 (40 GB) per job; main run ≈ 8 h per seed; adversary sweep ≈ 4 h 52 min per seed; one dense 12,288 × 12,288 Hessian ≈ 42 min; Jacobian lens 17 min."),
    T("Llama-3.1-8B IOI 12 h and MIB addition 16.5 h on one H100 (bf16); Qwen2.5-3B IOI 6 h and Gemma-2-2b addition 9.5 h on one A100."),
    T("Qwen3-1.7B IOI: three seeds with the mixture learner on A100s, including the 4000-step manufactured-alignment arm."),
    T("Code: Python package on PyTorch and Hugging Face transformers; figures and this deck from reports/talk_slt/scripts."),
  ], 0.7, 1.5, 11.9, 5.0, 17);
  source(s, "paper draft, section 'Compute and software'.");
  s.addNotes("Everything in the talk ran on a university cluster with one GPU per job. The expensive pieces are the language-model runs with the mixture learner and the dense Hessians.");
}

// pptxgenjs writes a <a:pPr> before every run of a multi-run paragraph (with <a:buNone/> on runs that carry
// no bullet), which drops the bullet and is not schema-clean. Keep only the first pPr of each paragraph.
async function fixParagraphProps(file) {
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  let n = 0;
  for (const name of Object.keys(zip.files)) {
    if (!/^ppt\/slides\/slide\d+\.xml$/.test(name)) continue;
    const xml = await zip.file(name).async("string");
    const out = xml.replace(/(<\/a:r>)<a:pPr\b[^>]*?(?:\/>|>[\s\S]*?<\/a:pPr>)/g, (m, r) => { n++; return r; });
    zip.file(name, out);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log("removed", n, "duplicate paragraph-property blocks");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await fixParagraphProps(OUT);
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
