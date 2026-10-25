// Build reports/talk_slt/slt_days_talk_7min.pptx: the 7-minute version of the SLT-days talk (9 main slides and a
// backup section), following presentation_comments.md. The long deck is build_deck.js / slt_days_talk.pptx.
// Figures are read from reports/talk_slt/*.png (made by make_all.py). Main slides carry at most two or three
// claims and their references; the numbers live in the figures, the speaker notes and the backup slides, whose
// text is taken over from build_deck.js (every number there is named with its source file).
// Run:  PATH=~/scratch/miniconda/envs/slides/bin:$PATH NODE_PATH=~/scratch/tools/node_pptx/node_modules \
//         node reports/talk_slt/scripts/build_deck_7min.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const SKILL = process.env.PPTX_SKILL_DIR ||
  "/u/lade/sheydenreich/.claude/skills/synced/34486e66-ec48-49aa-bb03-13986aa7c18f_7da7ec62-9ec7-4869-b790-48713d4fd41e/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const FIG = path.resolve(__dirname, "..");
const OUT = path.join(FIG, "slt_days_talk_7min.pptx");

// The long deck's theme; accent6 (unused there) is the fit-term magenta of gain_decomposition*.png.
const THEME = {
  name: "Bits of evidence",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F2A", lt1: "FFFFFF", dk2: "52514E", lt2: "EEF1F5",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100",
    accent5: "E34948", accent6: "D55181", hlink: "2A78D6", folHlink: "6250D6",
  },
};
// Tints for the two terms of the gain (fit = accent6, complexity = accent2), hex because they are fills only.
const FIT_TINT = "F9E3EC", CPLX_TINT = "FCE7DD";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.author = "Sven Heydenreich";
pres.title = "Bits of evidence: the free energy of an explanation";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, H = 7.5;

// ------------------------------------------------------------------ layouts
const darkTitle = (y, h, fontSize, valign) => ({ placeholder: { options: { name: "title", type: "title", x: 0.8, y,
  w: 11.7, h, fontFace: "Cambria", fontSize, bold: true, color: C.background1, valign, align: "left", margin: 0 },
  text: "" } });
pres.defineSlideMaster({
  title: "HOOK_DARK",
  background: { color: THEME.colors.dk1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.4, w: 7.4, h: 4.4,
    fontFace: "Cambria", fontSize: 38, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 },
    text: "" } }],
});
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: THEME.colors.dk1 },
  objects: [
    darkTitle(1.2, 1.6, 44, "bottom"),
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.0, w: 11.0, h: 1.1, fontFace: "Calibri",
      fontSize: 22, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: THEME.colors.dk1 },
  objects: [darkTitle(1.0, 1.2, 54, "bottom")],
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: THEME.colors.dk1 },
  objects: [
    darkTitle(2.6, 1.3, 40, "middle"),
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.95, w: 11.0, h: 1.0, fontFace: "Calibri",
      fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
function contentLayout(name, tag) {
  const objects = [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.38, w: 12.3, h: 0.9, fontFace: "Cambria",
      fontSize: 28, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ];
  if (tag) objects.push({ text: { text: tag, options: { x: 0.5, y: 0.05, w: 3, h: 0.25, fontSize: 11, bold: true,
    color: C.accent2, margin: 0, charSpacing: 2 } } });
  pres.defineSlideMaster({
    title: name, background: { color: THEME.colors.lt1 }, objects,
    slideNumber: { x: 0.5, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: THEME.colors.dk2, align: "left" },
  });
}
contentLayout("CONTENT", null);
contentLayout("BACKUP", "BACKUP");

// ------------------------------------------------------------------ helpers (as in build_deck.js)
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
// references of a main slide, bottom right, one per line
function refs(s, lines) {
  s.addText(lines.map((t, i) => ({ text: t, options: i < lines.length - 1 ? { breakLine: true } : {} })),
    { x: 5.6, y: 6.35, w: 7.25, h: 0.9, fontSize: 11, color: C.text2, align: "right", valign: "bottom", margin: 0,
      isTextBox: true, objectName: "references" });
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

// ------------------------------------------------------------------ hidden-state drawing (slides 1 and 3)
// Illustration colours (hex: they encode the picture, not the theme). Base sentence red, source sentence green;
// "random" hidden-state cells avoid both hues.
const RED_DARK = "B2182B", RED_LIGHT = "F6C4C4", GREEN_DARK = "1B7837", GREEN_LIGHT = "C3E6C4";
const CELL_PALETTE = ["5B6C8F", "8E9CC2", "C7CEE6", "A39BC9", "6E5A9E", "D9C27A", "B8A04E", "7FA7C9", "4E7FA8",
  "9AA5B1", "C9B8DD", "E3D9A6"];
let seed = 7;
function rand() { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; }
const randomCells = (n) => Array.from({ length: n }, () => CELL_PALETTE[Math.floor(rand() * CELL_PALETTE.length)]);
// a vertical stack of cells (one hidden-state vector) with its top-left corner at (x, y)
function stack(s, x, y, colors, { cell = 0.25, border = THEME.colors.lt1, name = "hidden state" } = {}) {
  colors.forEach((c, i) => s.addShape(pres.shapes.RECTANGLE, { x, y: y + i * cell, w: cell, h: cell,
    fill: { color: c }, line: { color: border, width: 1 }, objectName: `${name} cell ${i + 1}` }));
}
function arrow(s, x1, y1, x2, y2, { color = THEME.colors.dk2, width = 1.75 } = {}) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1),
    h: Math.abs(y2 - y1), flipH: x2 < x1, flipV: y2 < y1,
    line: { color, width, endArrowType: "triangle" }, objectName: "arrow" });
}
// the learned map: a blue tag on an arrow (blue is the plug-in learner's colour throughout)
function mapTag(s, x, y, label) {
  s.addText(label, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x, y, w: 0.7, h: 0.48,
    fill: { color: THEME.colors.accent1 }, line: { color: THEME.colors.accent1 }, color: C.background1,
    fontFace: "Cambria", fontSize: 20, bold: true, italic: true, align: "center", valign: "middle", margin: 0,
    objectName: "alignment map " + label });
}

// ================================================================== main slides
const SM = "Talk", SB = "Backup";
pres.addSection({ title: SM });

// 1 hook
{
  const s = slideOf("HOOK_DARK", SM, null);
  s.addText([
    { text: "We probe a network for an algorithm, and we find it.", options: { color: C.background2, bold: false,
      fontSize: 30, breakLine: true } },
    { text: " ", options: { fontSize: 18, breakLine: true } },
    { text: "Does the network implement it, or did we manufacture it?", options: {} },
  ], { placeholder: "title" });
  // an activation grid with a claimed four-cell variable: is it there, or did the probe put it there?
  const cell = 0.42, x0 = 8.95, y0 = 1.85;
  for (let col = 0; col < 7; col++) stack(s, x0 + col * cell, y0, randomCells(8),
    { cell, border: THEME.colors.dk1, name: `grid column ${col + 1}` });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x0 + 6 * cell - 0.07, y: y0 + 2 * cell - 0.07, w: cell + 0.14,
    h: 4 * cell + 0.14, rectRadius: 0.06, fill: { type: "none" },
    line: { color: THEME.colors.accent4, width: 3, dashType: "dash" }, objectName: "claimed variable" });
  s.addText("?", { x: x0 + 7 * cell + 0.3, y: y0 + 4 * cell - 0.5, w: 0.7, h: 1.0, fontSize: 60, bold: true,
    fontFace: "Cambria", color: THEME.colors.accent4, margin: 0, valign: "middle", isTextBox: true });
  s.addNotes("Open with the question. Interpretability methods look inside a network for the variables of an algorithm, and they report a hit with an accuracy. When they find it: does the network compute that variable, or did the probe, which we trained, put it there? This talk: description length can tell the two apart, and the term that decides it is a learning coefficient, though not the Bayes one.");
}

// 2 title
{
  const s = slideOf("TITLE_DARK", SM, "Bits of evidence: the free energy of an explanation");
  s.addText("Scoring interpretability alignments by the code length of counterfactuals, and what singular learning theory says about the complexity term",
    { placeholder: "body" });
  s.addText([
    { text: "Sven Heydenreich", options: { fontSize: 22, bold: true, color: C.background1, breakLine: true } },
    { text: "Postdoc, Laboratory of Data Engineering, AREA Science Park", options: { fontSize: 17,
      color: C.background2, breakLine: true } },
    { text: "[email]   ·   [website]", options: { fontSize: 17, color: "8C93A3" } },
  ], { x: 0.8, y: 4.75, w: 7.5, h: 1.3, margin: 0, valign: "top", isTextBox: true, objectName: "speaker" });
  s.addText("Singular Learning Theory days  ·  October 2026", { x: 0.8, y: 6.55, w: 8, h: 0.4, fontSize: 14,
    color: "8C93A3", margin: 0, isTextBox: true });
  s.addNotes("I am Sven Heydenreich, a postdoc at the Laboratory of Data Engineering at AREA Science Park. The talk scores interpretability explanations by description length and asks what singular learning theory says about the complexity term of that score.");
}

// 3 causal abstraction
{
  const s = slideOf("CONTENT", SM, "Causal abstraction: swap the variable, check the output");
  const yA = 2.55, yB = 5.0, yM = (yA + yB) / 2, cell = 0.25, half = 4 * cell;
  const sentence = (y, tag, tagColor, runs) => {
    s.addText(tag, { x: 0.5, y: y - 0.95, w: 3.8, h: 0.3, fontSize: 13, bold: true, color: tagColor, margin: 0,
      isTextBox: true, objectName: "sentence tag" });
    s.addText(runs.map(([t, o]) => ({ text: t, options: o || {} })), { x: 0.5, y: y - 0.6, w: 3.85, h: 1.2,
      fontSize: 17, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: "sentence" });
  };
  const io = (name, color) => [name, { bold: true, color }];
  sentence(yA, "base  ·  ABAB", RED_DARK, [["When Mary and "], io("John", RED_DARK),
    [" went to the store,", { breakLine: true }], ["Mary gave a drink to …"]]);
  sentence(yB, "source  ·  ABBA", GREEN_DARK, [["When "], io("Anna", GREEN_DARK),
    [" and Tom went to the park,", { breakLine: true }], ["Tom gave a book to …"]]);
  // GPT-2 up to layer l: a hidden state per sentence
  for (const y of [yA, yB]) {
    arrow(s, 4.4, y, 4.95, y);
    stack(s, 5.0, y - half, randomCells(8), { name: "hidden state" });
  }
  s.addText("GPT-2, layer ℓ", { x: 4.3, y: yA - half - 0.38, w: 1.5, h: 0.3, fontSize: 12, color: C.text2,
    margin: 0, align: "center", isTextBox: true });
  // the learned alignment map: the aligned coordinates (dark) and the rest (light)
  for (const [y, dark, light] of [[yA, RED_DARK, RED_LIGHT], [yB, GREEN_DARK, GREEN_LIGHT]]) {
    arrow(s, 5.3, y, 6.7, y);
    mapTag(s, 5.65, y - 0.24, "φ");
    stack(s, 6.75, y - half, [dark, dark, dark, dark, light, light, light, light], { name: "aligned state" });
  }
  s.addText("learned\nalignment map", { x: 5.3, y: yA - 0.86, w: 1.4, h: 0.55, fontSize: 12, bold: true,
    color: C.accent1, margin: 0, align: "center", valign: "bottom", isTextBox: true });
  // interchange: the source's aligned coordinates, the base's rest
  arrow(s, 7.05, yA, 7.95, yM - 0.45);
  arrow(s, 7.05, yB, 7.95, yM + 0.45);
  stack(s, 8.0, yM - half, [GREEN_DARK, GREEN_DARK, GREEN_DARK, GREEN_DARK, RED_LIGHT, RED_LIGHT, RED_LIGHT,
    RED_LIGHT], { name: "interchanged state" });
  s.addText("interchange", { x: 7.5, y: yM - half - 0.38, w: 1.25, h: 0.3, fontSize: 12, color: C.text2,
    margin: 0, align: "center", isTextBox: true });
  // back through the inverse map and the rest of the network
  arrow(s, 8.3, yM, 9.7, yM);
  mapTag(s, 8.65, yM - 0.24, "φ⁻¹");
  stack(s, 9.75, yM - half, randomCells(8), { name: "new hidden state" });
  arrow(s, 10.05, yM, 11.05, yM);
  s.addText("rest of GPT-2", { x: 9.95, y: yM - 0.42, w: 1.2, h: 0.3, fontSize: 12, color: C.text2, margin: 0,
    align: "center", isTextBox: true });
  s.addText("Anna", { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x: 11.1, y: yM - 0.4, w: 1.75, h: 0.8,
    fill: { color: GREEN_LIGHT }, line: { color: GREEN_DARK, width: 1.5 }, fontSize: 28, bold: true,
    color: GREEN_DARK, align: "center", valign: "middle", margin: 0, objectName: "counterfactual output" });
  s.addText("counterfactual output", { x: 11.0, y: yM + 0.5, w: 1.95, h: 0.3, fontSize: 12, color: C.text2,
    margin: 0, align: "center", isTextBox: true });
  s.addText([B("IIA"), T(": how often the output is the algorithm's counterfactual")],
    { x: 0.5, y: 6.45, w: 6.0, h: 0.4, fontSize: 16, color: C.text1, margin: 0, isTextBox: true });
  refs(s, ["Causal abstraction: Geiger et al., NeurIPS 2021 · DAS: Geiger et al., CLeaR 2024",
    "IOI: Wang et al., ICLR 2023",
    "Manufactured alignments: Sutter et al., NeurIPS 2025"]);
  s.addNotes("Causal abstraction on the indirect-object task in GPT-2. Two prompts: the base has the pattern ABAB, so the answer is the second name, John; the source has ABBA, so the answer is the first name, Anna. Run both through GPT-2 up to a layer and take the hidden states. An alignment map phi rotates each hidden state so that its first coordinates carry the variable, here the indirect-object name. Swap those coordinates from the source into the base, map back with phi inverse and run the rest of the network. If the network computes the variable there, the output is the counterfactual, Anna: the base said John, and Anna sits at a different position than John did, so this is not about positions. Interchange-intervention accuracy, IIA, is the fraction of swaps whose output is the algorithm's counterfactual. The catch is that phi is learned. An expressive phi, such as a coupling flow, can memorise features and reach high IIA even on an untrained GPT-2 (Sutter et al.). Accuracy alone cannot tell a found variable from a manufactured one.");
}

// 4 description length and the gain
{
  const s = slideOf("CONTENT", SM, "Description length: code the counterfactuals, count the bits");
  pills(s, ["plug", "bayes"]);
  formula(s, [T("DL(n) = Σᵢ −log₂ p"), { text: "φ̂(i−1)", options: { subscript: true } }, T("(yᵢ)")],
    0.5, 1.4, 6.0, 0.6, 24);
  formula(s, [T("G(n) = DL"), { text: "control", options: { subscript: true } }, T("(n) − DL"),
    { text: "hyp", options: { subscript: true } }, T("(n)")], 6.85, 1.4, 6.0, 0.6, 24);
  s.addText("each counterfactual label is coded before the map is fitted on it  ·  for a Bayes predictive, DL is the free energy Fₙ",
    { x: 0.5, y: 2.05, w: 12.3, h: 0.35, fontSize: 15, italic: true, color: C.text2, align: "center", margin: 0,
      isTextBox: true });
  // centerpiece: the gain, its two terms as coloured chips, each above its card (chip and card share a centre)
  const y = 2.75, h = 0.95, fs = 38, cardW = 4.6, c1 = 4.25, c2 = c1 + cardW + 0.3;
  const chip = (txt, cx, w, color, tint) => s.addText(txt, { shape: pres.shapes.ROUNDED_RECTANGLE,
    rectRadius: 0.12, x: cx - w / 2, y, w, h, fill: { color: tint }, line: { color: tint }, fontFace: "Cambria",
    italic: true, fontSize: fs, color, align: "center", valign: "middle", margin: 0, objectName: "gain term" });
  const plain = (txt, x, w, align = "center") => s.addText(txt, { x, y, w, h, fontFace: "Cambria", italic: true,
    fontSize: fs, color: C.text1, align, valign: "middle", margin: 0, isTextBox: true });
  plain("G(n) ≈", c1 - 1.1 - 2.0, 1.9, "right");
  chip("n Δℓ", c1, 2.0, C.accent6, FIT_TINT);
  plain("+", (c1 + c2) / 2 - 0.3, 0.6);
  chip("Δλ ln n", c2, 2.6, C.accent2, CPLX_TINT);
  plain("+ …", c2 + 1.4, 1.2, "left");
  const termCard = (cx, head, body, color, tint) => {
    const x = cx - cardW / 2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.1, w: cardW, h: 1.85, rectRadius: 0.1, fill: { color: tint },
      line: { color: tint }, objectName: "card " + head });
    s.addText(head, { x: x + 0.25, y: 4.25, w: cardW - 0.5, h: 0.45, fontSize: 20, bold: true, fontFace: "Cambria",
      color, margin: 0, isTextBox: true });
    s.addText(body, { x: x + 0.25, y: 4.75, w: cardW - 0.5, h: 1.1, fontSize: 16, color: C.text1, margin: 0,
      valign: "top", isTextBox: true });
  };
  termCard(c1, "Linear: realizability", "Can the family reproduce the counterfactual law at all? If not, it falls behind linearly in n: rejected outright.",
    C.accent6, FIT_TINT);
  termCard(c2, "Logarithmic: complexity", [T("Both fit: the gain is a learning-coefficient difference. Our codes are plug-in: "),
    { text: "λ → κ", options: { bold: true, color: C.accent1 } }, T(".")], C.accent2, CPLX_TINT);
  refs(s, ["Prequential codes: Dawid 1984; Grünwald, The MDL Principle 2007",
    "Free energy: Watanabe 2009 · MDL meets SLT: Urdshals et al. 2025",
    "MDL probing: Voita & Titov, EMNLP 2020"]);
  s.addNotes("Instead of counting agreements, count bits. Draw base and source pairs, compute the algorithm's counterfactual label, pass it through a 5 percent randomized-response channel, and code the labels one by one with the intervened network as the predictor; the map is fitted only on the labels seen so far. That is a prequential code, and for a Bayes predictive it is exactly the Bayes free energy, so Watanabe's expansion applies: F_n = n L_n(w0) + lambda ln n - (m-1) ln ln n + O_p(1). The gain of a hypothesis over a control on the same labels is then a difference of free energies with two terms. The linear term, n times the loss gap of the best fits, is nonzero only when one family cannot reproduce the counterfactual law at all: a realizability question, and it rejects outright. The logarithmic term is a difference of learning coefficients: the complexity of learning the alignment, Occam's razor in bits. The codes we actually use are plug-in codes, a map refitted by gradient descent, so lambda is replaced by a learner coefficient kappa; that is the last slide.");
}

// 5 IIA vs code
{
  const s = slideOf("CONTENT", SM, "Interchange accuracy accepts explanations that the code rejects");
  pills(s, ["plug"]);
  fig(s, "iia_vs_code.png", 0.45, 1.45, { w: 8.9 });
  s.addText(paras([
    [{ text: "Manufactured alignments pass an IIA threshold, ", options: {} },
      { text: "and the code rejects them", options: { bold: true, color: C.accent5 } }],
    [{ text: "More expressive alignment maps ", options: {} }, { text: "pay more bits", options: { bold: true } }],
  ], { space: 22 }), { x: 9.75, y: 2.0, w: 3.1, h: 3.4, fontSize: 20, color: C.text1, valign: "top", margin: 0,
    isTextBox: true, objectName: "takeaways" });
  refs(s, ["Sutter et al., NeurIPS 2025 · Makelov et al., ICLR 2024"]);
  s.addNotes("Each point is one alignment on the indirect-object task in four language models: horizontally its interchange accuracy, vertically its gain over a code that ignores the network (label-only), at 1024 targets. The red points are manufactured alignments, an expressive coupling map on an untrained twin: IIA up to 0.83, so an accuracy threshold of 0.8 accepts them, yet they code longer than knowing nothing. The faithful subspaces gain around 3000 bits. And the more expressive family, the coupling map on the trained network, pays more bits than the subspace for the same variable. Say aloud: 'longer than label-only' is a statement at 1024 targets under the single-map learner; with 32768 targets the forgery beats label-only too, and what survives at every n is the comparison with the valid alignment (slide 7, backup).");
}

// 6 complexity ladder
{
  const s = slideOf("CONTENT", SM, "The complexity ladder: κ orders four explanations");
  pills(s, ["plug"]);
  fig(s, "ladder_panels.png", 0.75, 1.4, { w: 11.8 });
  s.addText([T("IIA accepts all four; the parameter count cannot separate the two coupling maps. "),
    { text: "κ orders them, and the code follows.", options: { bold: true } }],
    { x: 0.5, y: 5.95, w: 12.3, h: 0.45, fontSize: 18, color: C.text1, margin: 0, align: "left", isTextBox: true });
  refs(s, ["Jacobian lens: Gurnee et al., Transformer Circuits 2026"]);
  s.addNotes("Four explanations of the same variable, the indirect-object name in GPT-2: the Jacobian lens, which reads the subspace off the network's own averaged Jacobian and fits nothing; a learned 16-dimensional subspace; an expressive coupling map on the trained network; and the same map on the untrained twin, the manufactured alignment. Interchange accuracy accepts all four. The parameter count puts the two coupling maps at the same 1.9 million. The plug-in learning coefficient kappa, the regret per ln n, orders them 0, 49, 226, 988, and the code length follows: the explanation with kappa exactly zero codes the name shortest. (MAP code, kappa at n up to 1024, three seeds.)");
}

// 7 fit or Occam (readout as is): two comparisons with the valid subspace and the fair same-family one
{
  const s = slideOf("CONTENT", SM, "Fit or Occam: the learning cost carries the verdict");
  pills(s, ["plug"]);
  fig(s, "gain_decomposition_slide.png", 0.5, 1.5, { w: 12.3 });
  refs(s, ["Free-energy expansion: Watanabe 2009", "Coupling-map alignments: Sutter et al., NeurIPS 2025"]);
  s.addNotes("Which term of slide 4 carries the verdict? Each panel splits a gain into the fit term (pink: the difference of the best-in-family losses, linear in n) and the regret term (blue: the difference of the learning costs, delta kappa ln n); single-map MAP code, 32768 targets, three seeds. Left: the valid subspace against the manufactured alignment. The gain is carried by the regret term, and that term scales with the learning coefficient: this is where the investigation becomes a singular-learning-theory question. Middle: against the coupling map on the trained network. The gain is not monotonic in n: the fit term, linear in n, runs against the subspace at large n while the regret term grows only like ln n, so the gain peaks and turns over, and the coupling overtakes near 20k targets. With G(n) = n delta-l + delta-kappa ln n and a small negative delta-l the peak sits near n = delta-kappa / |delta-l| (back of the envelope for the left panel: delta-kappa about 750, |delta-l| about 0.04 nats per target, so about 2e4; the peak is at 16384). Right: the fair comparison, the same coupling map with the same parameter count on the trained and on the untrained network. Both fit equally well, the fit term is zero, and the whole 8k-bit gain is regret: kappa about 150 against about 830 per ln n. Parameter count, interchange accuracy and the best achievable loss cannot tell these two apart; the learning coefficient can. If asked why the flows fit better in the middle panel: calibration of the readout (backup 'Fit or Occam: calibrated'); beyond about 32768 targets the optimiser's noise matters as much as kappa (backup 'To 131072 targets').");
}

// 8 kappa vs lambda
{
  const s = slideOf("CONTENT", SM, "κ is not λ: the plug-in pays for Fisher-weighted directions");
  pills(s, ["plug", "bayes"]);
  const a = pngAspect(path.join(FIG, "kappa_lambda_panels.png"));
  const w = Math.min(12.3, 5.3 * a);
  fig(s, "kappa_lambda_panels.png", (W - w) / 2, 1.3, { w });
  refs(s, ["½ tr(F H⁻¹): Takeuchi 1976; White 1982 · λ: Watanabe 2009", "LLC: Lau et al., AISTATS 2025"]);
  s.addNotes("What is kappa, and how does it relate to lambda? (a) On GPT-2 layer 11 the valid subspace's kappa stays at 50 to 80 as the subspace dimension goes from 4 to 64, while the nominal Bayes coefficient d_V(d - d_V)/2 grows from 1528 to 22528; the controls stay above at every width. (b) The classical plug-in prediction, half the trace of F H inverse, gets the order of kappa on the oracle and on GPT-2, but not its trend (measured over predicted 0.3 to 1.2); the nominal lambda is 20 to 400 times off. (c) The reason: at the MAP the empirical Fisher is about 2 percent of the Hessian; the targets are deterministic functions of the hidden state behind a noise channel, so moving the subspace changes every target's loss the same way. (d) A Laplace (Bayes) code pays one half per resolved direction, 6016; the plug-in pays one half F over h per direction, about 70; measured 49. Discussion: plug-in against Bayes codes (on reduced-rank regression the HMC Bayes code reads the Aoyagi-Watanabe lambda and least squares reads MN/2; on the oracle the Bayes code pays k/2 + (11 - k)/4, the plug-in k/2; backup), local kappa against lambda and the SGLD LLC (backup), and what a singular learning theory of plug-in learners would have to say (backup 'What this asks of SLT').");
}

// 9 thank you
{
  const s = slideOf("CLOSING_DARK", SM, "Thank you");
  s.addText("Let's talk: about ideas, open questions, or a mistake you want to yell at me about.",
    { x: 0.8, y: 2.5, w: 7.6, h: 1.3, fontSize: 26, color: C.background2, margin: 0, valign: "top", isTextBox: true });
  s.addText([
    { text: "Sven Heydenreich", options: { fontSize: 20, bold: true, color: C.background1, breakLine: true } },
    { text: "Laboratory of Data Engineering, AREA Science Park", options: { fontSize: 17, color: C.background2,
      breakLine: true } },
    { text: "[email]", options: { fontSize: 17, color: "8C93A3", breakLine: true } },
    { text: "[website]", options: { fontSize: 17, color: "8C93A3" } },
  ], { x: 0.8, y: 4.4, w: 7.6, h: 1.8, margin: 0, valign: "top", isTextBox: true, objectName: "contact" });
  s.addText("[QR code or link to the slides]", { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x: 9.4,
    y: 2.5, w: 3.1, h: 3.1, fill: { type: "none" }, line: { color: "5A6275", width: 1.5, dashType: "dash" },
    fontSize: 14, color: "8C93A3", align: "center", valign: "middle", margin: 8, objectName: "slides link space" });
  s.addNotes("Thank you. If you want to discuss this, have ideas, or found a mistake, please talk to me or write. The slides, with a backup section on the Bayes codes, the oracle and the larger models, are online.");
}

// ================================================================== backup (slide text from build_deck.js)
pres.addSection({ title: SB });
{
  const s = slideOf("SECTION_DARK", SB, "Backup");
  s.addText("The score in detail · observed label law · the two coefficients · exact and oracle checks · Bayes codes · fit and regret, calibrated · large n · four language models · Jacobian lens · learner dependence · Pythia · SGLD LLC · questions for SLT",
    { placeholder: "body" });
  s.addNotes("Backup slides, for questions and for reading after the talk.");
}
{
  const s = slideOf("BACKUP", SB, "The score: prequential code length of counterfactual targets");
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
  const s = slideOf("BACKUP", SB, "Two coefficients: the Bayes λ and the learner's κ");
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
  s.addNotes("Two coefficients appear throughout and I will label every number with one of them. Lambda is the Bayes learning coefficient: it governs the regret of the posterior predictive. Kappa is the coefficient of whatever learner builds the code; for a plug-in estimator in a regular model it is the classical one half trace of F times H inverse, which equals d over two only when the information equality holds. Both are measured the same way, as the slope of the regret against log n, corrected for the block schedule (checked exactly in the long deck, slt_days_talk.pptx). The plan's open question was when kappa equals lambda; the answer turns out to be: almost never on trained networks.");
}
{
  const s = slideOf("BACKUP", SB, "Exact check: the Bayes code reads Aoyagi–Watanabe λ, the plug-in reads MN/2");
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
  s.addNotes("Before any network, the exact example. In reduced-rank regression the learning coefficient is known from Aoyagi and Watanabe, so we ran a genuine Bayes code by HMC and a least-squares plug-in code on the same data. The Bayes code's regret slope is lambda on all twelve shapes, within one or two standard errors; the plug-in's slope is the regular MN over 2 whatever the true rank, so whenever the rank is deficient the two differ by the predicted amount. Gains between a rank-restricted and a full family on the same data grow like delta lambda log n, which is the Occam-regime prediction of slide 4. The two multiplicity-2 shapes sit slightly below lambda, which we attribute to the log log n term at these n.");
}
{
  const s = slideOf("BACKUP", SB, "Oracle: the plug-in pays for readable directions, the Bayes code for quartic ones too");
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
{
  const s = slideOf("BACKUP", SB, "Sharp readouts: a finite-n crossover for Bayes, a persistent excess for the MAP");
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
{
  const s = slideOf("BACKUP", SB, "The Bayes code is the right instrument only while λ ln n is small against n Δℓ");
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
{
  const s = slideOf("BACKUP", SB, "Fit or Occam: calibrated, only the regret separates them");
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
{
  const s = slideOf("BACKUP", SB, "Two coupling maps, the same fit: only the regret differs");
  pills(s, ["plug"]);
  fig(s, "gain_coupling_pair.png", 1.4, 1.3, { h: 4.3 });
  bullets(s, [
    [B("Same family, same size: "), T("the coupling map on the trained network (layer 11) against the same map on its untrained twin (layer 10, the manufactured alignment); 1,869,440 parameters each, IIA ≥ 0.99 for both at 32768 targets.")],
    [B("Fit term +15 bits "), T("(+95 with a temperature): both flows saturate the readout. The +8.0k-bit gain is all regret: κ 146 vs 826 per ln n (n ≥ 64).")],
  ], 0.6, 5.7, 12.2, 0.95, 14);
  source(s, "difference of the two splits in stage1_ioi/gain_decomposition_gpt2_n32768{,_temp}.json (3 seeds); κ from h_a3_gpt2_large_n.txt.");
  s.addNotes("The cleanest version of the previous slide needs no temperature. Compare two explanations from the same family: the expressive coupling flow on the trained GPT-2 at layer 11, and the same flow on the untrained twin at layer 10, which is the manufactured alignment. Same architecture of map, same parameter count, nearly perfect interchange accuracy for both once they have seen enough pairs. Both saturate the readout, so the fit term is zero within noise: 15 bits at 32768 targets, 95 with the temperature. The 8000-bit gain is entirely the regret term, the difference in learning cost: the flow on the trained network has a learner coefficient of about 150 per log n, the one on the untrained network about 830. Parameter count, accuracy and the best achievable loss cannot tell these two apart; the learning coefficient can. This is the Occam regime of slide 4 in its pure form.");
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
  const s = slideOf("BACKUP", SB, "Four language models, one verdict, no retuning");
  pills(s, ["plug"]);
  fig(s, "models_verdicts.png", 0.4, 1.45, { w: 9.0 });
  bullets(s, [
    T("Identical 1024 targets on every model; label-only 4324.5 bits."),
    T("Valid subspace at the last 2–4 layers: +3000 to +3400 bits; twin and wrong locus rejected at α = 0.01 within 20 targets (GPT-2, Qwen2.5, Llama)."),
    T("Coupling on the trained net: 925–1665 bits behind the subspace at n = 1024 (on GPT-2 it overtakes near 20k targets, slide 7)."),
    T("Manufactured, n = 1024: 1723–1986 bits longer than label-only (single-map code); 147 longer on Qwen3 (mixture, IIA 0.71)."),
  ], 9.65, 1.5, 3.25, 5.1, 15);
  source(s, "stage1_ioi/tier2_models.txt, gpt2_summary_single_map_map0.md, *_summary.md (gains vs controls).");
  s.addNotes("The GPT-2 protocol carries to models twenty to sixty times larger without any change. At the loci where the name is intervenable the valid subspace gains three thousand bits or more on every model, and the controls are rejected within the first twenty targets by the anytime criterion. At a thousand targets the coupling family on the trained network codes about a thousand bits behind the subspace, which is the complexity charge that IIA does not see; on GPT-2 it overtakes the subspace near twenty thousand targets, by calibration (slide 7). At a thousand targets the manufactured alignment never beats the label-only code: 1700 to 2000 bits worse under the single-map learner, 150 bits worse on Qwen3 under the averaging learner, which rescues confident errors. Learners: GPT-2, Qwen2.5 and Llama use the single-map MAP code, Qwen3 the mixture of four.");
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
  const s = slideOf("BACKUP", SB, "Competence onset: the code tracks the task and keeps improving after it is solved");
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
  const s = slideOf("BACKUP", SB, "What this asks of singular learning theory");
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
