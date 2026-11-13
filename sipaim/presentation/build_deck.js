// SIPAIM 2026 Paper 71 — oral presentation (10 min talk + backup for Q&A).
// Every number on these slides is taken from the camera-ready paper
// (sipaim/camera_ready_2026-10-21/) or computed from its formulas in make_assets.py.
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(
  "/home/tals/.claude/skills/synced/75cbb7e8-89d1-4082-b723-8cb4074f6e55_155a93a2-4334-40c9-a502-ed1a8419e5de/pptx/scripts/apply_theme.js");
const data = require("./assets/data.json");

const OUT = path.join(__dirname, "SIPAIM2026_paper71_oral.pptx");
const A = (f) => path.join(__dirname, "assets", f);

const THEME = {
  name: "PEM Oral",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "1E1B2E", lt1: "FFFFFF", dk2: "24163A", lt2: "F2F0F6",
    accent1: "0E9AA7",   // PEM (cyan, as in the paper's figure)
    accent2: "E8780C",   // entropy (magma orange)
    accent3: "B73779",   // magma magenta
    accent4: "6E6B7B",   // baselines
    accent5: "B8B5C4",   // light baseline
    accent6: "C8384A",   // degradation / errors
    hlink: "0E9AA7", folHlink: "B73779",
  },
};
const X = THEME.colors;                  // hex, only where an option is hex-only
const LIGHT_RED = "F8DDE0", LIGHT_TEAL = "D6EFF1";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";             // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Post-hoc Entropy Minimization Improves Semi-Supervised Medical Image Segmentation";
pres.author = "Tal Shaharabany, Natalie Carlebach, Lior Wolf";
pres.subject = "SIPAIM 2026 oral presentation, Paper 71";
const C = pres.SchemeColor;

// ── Layouts ──────────────────────────────────────────────────────────────────
const FOOT = "Shaharabany, Carlebach, Wolf  ·  SIPAIM 2026";
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: X.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.35, w: 7.4, h: 2.6,
        fontSize: 36, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 4.25, w: 7.4, h: 1.1,
        fontSize: 20, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: X.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.85,
        fontSize: 30, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent4, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: X.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.85,
        fontSize: 30, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

// ── Helpers ──────────────────────────────────────────────────────────────────
const txt = (slide, text, o) => slide.addText(text, { isTextBox: true, margin: 0, valign: "top", ...o });
const card = (slide, x, y, w, h, name, fill = C.background2) =>
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12,
    fill: { color: fill }, line: { color: fill, width: 0 }, objectName: name });
const numberBadge = (slide, x, y, n, name) => {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.6, h: 0.6, fill: { color: C.accent1 },
    line: { color: C.accent1, width: 0 }, objectName: name });
  txt(slide, String(n), { x, y, w: 0.6, h: 0.6, fontSize: 20, bold: true, color: C.background1,
    align: "center", valign: "middle", objectName: name + " number" });
};
const eqImage = (slide, key, x, y, w, name) => {
  const [pw, ph] = data.eq_sizes[key];
  slide.addImage({ path: A(`${key}.png`), x, y, w, h: w * ph / pw, objectName: name,
    altText: name });
};
const caption = (slide, text, o) => txt(slide, text, { fontSize: 12, color: C.accent4, italic: true, ...o });
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  catAxisTitleFontFace: "+mn-lt", valAxisTitleFontFace: "+mn-lt" };
const quietAxes = { catAxisLabelColor: X.accent4, valAxisLabelColor: X.accent4,
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 12,
  valGridLine: { color: "E4E2EA", size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: true, valAxisLineShow: false };

// ════════════════════════════════════════════════════════════════════════════
pres.addSection({ title: "Talk" });

// 1 ── Title ─────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Talk" });
  s.addText("Post-hoc Entropy Minimization Improves Semi-Supervised Medical Image Segmentation",
    { placeholder: "title" });
  s.addText([
    { text: "Tal Shaharabany  ·  Natalie Carlebach  ·  Lior Wolf", options: { breakLine: true } },
    { text: "Tel Aviv University", options: { fontSize: 16, color: C.accent5 } },
  ], { placeholder: "body" });
  txt(s, "SIPAIM 2026  ·  Lima, Peru  ·  Paper 71", { x: 0.7, y: 5.75, w: 7.4, h: 0.4,
    fontSize: 16, bold: true, color: C.accent2, objectName: "Venue" });
  const p = data.panels.la_5pct_entropy.size;
  s.addImage({ path: data.panels.la_5pct_entropy.path, x: 8.55, y: 1.75, w: 4.2, h: 4.2 * p[1] / p[0],
    rounding: false, objectName: "Entropy map", altText: "Residual entropy of a converged SSL model on a left-atrium MRI slice: a thin bright rim on the organ boundary" });
  caption(s, "Residual entropy of a converged SSL model", { x: 8.55, y: 1.75 + 4.2 * p[1] / p[0] + 0.12,
    w: 4.2, h: 0.3, color: C.accent5, objectName: "Entropy caption" });
  s.addNotes(
`[0:00–0:20] Thank you. I'm Tal Shaharabany from Tel Aviv University; this is joint work with Natalie Carlebach and Lior Wolf.
The image on the right is the residual uncertainty of a fully trained semi-supervised model. The talk is about the thin bright rim you see, and how a few minutes of label-free fine-tuning can use it.`);
}

// 2 ── Observation ──────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  s.addText("Converged SSL leaves entropy at the boundary", { placeholder: "title" });
  const p = data.panels.la_5pct_entropy.size, w = 6.3, h = w * p[1] / p[0];
  s.addImage({ path: data.panels.la_5pct_entropy.path, x: 0.6, y: 1.45, w, h, objectName: "Entropy map",
    altText: "Per-voxel entropy of the BCP model on a left-atrium test case: a thin bright rim along the predicted surface over a dark interior" });
  caption(s, "BCP on an LA 5% test case: entropy forms a thin rim on the predicted surface; the interior is dark.",
    { x: 0.6, y: 1.45 + h + 0.12, w, h: 0.5, objectName: "Map caption" });

  txt(s, "89–95%", { x: 7.5, y: 1.45, w: 5.2, h: 0.95, fontSize: 60, bold: true, color: C.accent2, objectName: "Stat entropy share" });
  txt(s, "of the prediction entropy on unlabeled volumes…", { x: 7.5, y: 2.42, w: 5.2, h: 0.4, fontSize: 18, color: C.text1, objectName: "Stat entropy label" });
  txt(s, "5–9%", { x: 7.5, y: 3.05, w: 5.2, h: 0.95, fontSize: 60, bold: true, color: C.accent2, objectName: "Stat voxel share" });
  txt(s, "…sits in this share of the voxels (c ≤ 0.99)", { x: 7.5, y: 4.02, w: 5.2, h: 0.4, fontSize: 18, color: C.text1, objectName: "Stat voxel label" });
  txt(s, "SSL training stops using the unlabeled data at convergence. The uncertainty that remains is concentrated where segmentation errors are, and finding it needs no labels.",
    { x: 7.5, y: 4.75, w: 5.2, h: 1.2, fontSize: 16, color: C.text1, objectName: "Observation text" });
  caption(s, "Measured on three BCP checkpoints: Pancreas-CT 20%, LA 5%, LA 10% labeled.",
    { x: 7.5, y: 6.1, w: 5.2, h: 0.3, objectName: "Checkpoint caption" });
  s.addNotes(
`[0:20–1:20] Semi-supervised methods such as BCP use a large unlabeled set during training, and then stop using it once training converges.
We looked at what a converged model still "doesn't know" about that unlabeled data. On three BCP checkpoints — pancreas CT at 20% labels, and left atrium MRI at 5% and 10% — 89 to 95 percent of the prediction entropy sits in only 5 to 9 percent of the voxels.
And those voxels are not random: they form a thin shell on the predicted organ surface, which is exactly where segmentation errors live. The interior is saturated.
So the question of this paper is: can we use that leftover, spatially structured uncertainty after training, without any labels?`);
}

// 3 ── Method ───────────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  s.addText("PEM: entropy minimization after convergence", { placeholder: "title" });
  const steps = [
    ["Load a converged checkpoint", "Any SSL method. Here: BCP, released for LA and reproduced for Pancreas-CT."],
    ["Minimize entropy on unlabeled data", "Confident-shell mask τm = 0.90. No labels, no pseudo-labels, no new architecture."],
    ["Stop after 2–5 epochs", "Budget fixed per dataset; the final epoch is reported. Fine-tuning takes under a minute."],
  ];
  steps.forEach(([head, body], i) => {
    const x = 0.6 + i * 4.15;
    card(s, x, 1.45, 3.8, 2.75, `Step ${i + 1} card`);
    numberBadge(s, x + 0.3, 1.7, i + 1, `Step ${i + 1} badge`);
    txt(s, head, { x: x + 0.3, y: 2.42, w: 3.2, h: 0.62, fontSize: 18, bold: true, color: C.text1, valign: "bottom", objectName: `Step ${i + 1} heading` });
    txt(s, body, { x: x + 0.3, y: 3.12, w: 3.2, h: 0.95, fontSize: 15, color: C.text1, objectName: `Step ${i + 1} text` });
  });
  eqImage(s, "eq_loss", 0.75, 4.65, 6.3, "PEM loss equation");
  caption(s, "Averaged over unlabeled volumes; ε guards an empty mask.", { x: 0.75, y: 5.8, w: 6.3, h: 0.3, objectName: "Equation caption" });
  txt(s, [
    { text: "Hᵢ", options: { bold: true } }, { text: " entropy of voxel i;  ", options: {} },
    { text: "cᵢ", options: { bold: true } }, { text: " its confidence (max class probability).", options: { breakLine: true } },
    { text: "The mask keeps the voxels the model is not yet sure about, and the loss is their mean entropy." },
  ], { x: 7.6, y: 4.6, w: 5.1, h: 1.6, fontSize: 16, color: C.text1, paraSpaceAfter: 6, objectName: "Equation explanation" });
  s.addNotes(
`[1:20–2:20] The method, Post-hoc Entropy Minimization or PEM, is deliberately simple.
One: take any converged SSL checkpoint. In our experiments that is BCP — the released checkpoints for left atrium, and our reproduction of BCP for pancreas.
Two: fine-tune it on the unlabeled volumes with a single loss — the mean entropy of the voxels whose confidence is below a threshold tau-m. We use tau-m = 0.90. There are no labels, no pseudo-labels, no teacher, no architectural change.
Three: stop after a fixed budget — two epochs for pancreas, five for left atrium — and report the final epoch. The fine-tuning itself takes under a minute on one GPU.
Entropy minimization is classical. What is new is the regime: applying it after convergence, when the remaining entropy already sits on the boundary.`);
}

// 4 ── Why it works: gradient ───────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  s.addText("The entropy gradient targets the boundary", { placeholder: "title" });
  eqImage(s, "eq_grad", 0.6, 1.5, 3.4, "Entropy gradient equation");
  txt(s, [
    { text: "Vanishes on the saturated interior (p → 0 or 1)", options: { bullet: true, breakLine: true } },
    { text: "Vanishes at maximum ambiguity (p = 0.5)", options: { bullet: true, breakLine: true } },
    { text: "Peaks on partly decided voxels (p ≈ 0.82), the band beside the boundary", options: { bullet: true } },
  ], { x: 0.6, y: 2.75, w: 5.3, h: 1.9, fontSize: 17, color: C.text1, paraSpaceAfter: 8, objectName: "Gradient properties" });
  txt(s, "So the unmasked loss already focuses on the boundary band. The mask adds insurance: it drops voxels above τm and renormalizes the loss, and our ablation does not separate those two effects.",
    { x: 0.6, y: 4.85, w: 5.3, h: 1.4, fontSize: 15, color: C.text1, objectName: "Mask explanation" });

  const g = data.grad, cut = 0.90;
  const labels = g.c.map((c) => (Math.abs(c * 10 - Math.round(c * 10)) < 1e-6 ? c.toFixed(1) : ""));
  const kept = g.c.map((c, i) => (c <= cut + 1e-9 ? g.pct[i] : null));
  const removed = g.c.map((c, i) => (c >= cut - 1e-9 ? g.pct[i] : null));
  s.addChart(pres.charts.LINE, [
    { name: "kept by the mask (c ≤ 0.90)", labels, values: kept },
    { name: "removed by the mask (c > 0.90)", labels, values: removed },
  ], {
    x: 6.3, y: 1.4, w: 6.4, h: 4.75, objectName: "Gradient chart",
    altText: "Line chart of entropy-gradient magnitude against voxel confidence: zero at 0.5, peak near 0.82, about 88 percent of the peak at 0.90, falling to zero as confidence approaches 1",
    chartColors: [X.accent1, X.accent5], lineSize: 3, lineDataSymbol: "none", displayBlanksAs: "gap",
    showTitle: true, title: "|∂H/∂z| vs. voxel confidence (% of peak)", titleFontSize: 14, titleColor: X.dk1,
    showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: X.dk1,
    catAxisLabelRot: 0, catAxisMajorTickMark: "none", valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25,
    showCatAxisTitle: true, catAxisTitle: "confidence c = max(p, 1 − p)", catAxisTitleFontSize: 12, catAxisTitleColor: X.accent4,
    ...quietAxes, ...chartText,
  });
  caption(s, "At c = 0.90 the gradient is still 88% of its peak, so the mask changes which voxels get a gradient, not only the normalization.",
    { x: 6.3, y: 6.25, w: 6.4, h: 0.55, objectName: "Gradient caption" });
  s.addNotes(
`[2:20–3:30] Why does this work without any stabilizer? Look at the gradient of a voxel's entropy with respect to its logit: minus z times p times one-minus-p.
It vanishes in three places: on the saturated interior, where p is near 0 or 1, and at maximum ambiguity, p = 0.5. It peaks at p of about 0.82.
So the gradient is automatically supported on a band of partly-decided voxels next to the decision surface — the boundary shell from the previous slide — with no mask at all.
What does the mask add, then? At tau-m = 0.90 the gradient is still 88 percent of its peak, so the mask both removes those voxels and renormalizes the loss over the ones it keeps. Our ablation does not separate these two effects, so we describe the mask as insurance against collapse — we'll see the evidence for that in a moment.`);
}

// 5 ── Diagnostic ───────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  s.addText("A label-free check before fine-tuning", { placeholder: "title" });
  eqImage(s, "eq_rho", 0.6, 1.45, 3.6, "Shell entropy concentration equation");
  txt(s, "Share of the prediction entropy held by low-confidence voxels (c ≤ 0.99), from one forward pass on unlabeled data. No labels, no validation set.",
    { x: 4.5, y: 1.55, w: 2.2, h: 1.6, fontSize: 13, color: C.text1, objectName: "Rho definition" });
  const cats = ["Supervised (no SSL)", "BCP Pancreas 20%", "BCP LA 5%", "BCP LA 10%"];
  s.addChart([
    { type: pres.charts.BAR, data: [{ name: "ρf(0.99)", labels: cats, values: [0.552, 0.894, 0.948, 0.945] }],
      options: { barDir: "col", barGapWidthPct: 55, chartColors: [X.accent6, X.accent1, X.accent1, X.accent1], showValue: true } },
    { type: pres.charts.LINE, data: [{ name: "ρ* = 0.85 (provisional)", labels: cats, values: [0.85, 0.85, 0.85, 0.85] }],
      options: { chartColors: [X.accent2], lineSize: 2, lineDash: "dash", lineDataSymbol: "none", showValue: false } },
  ], {
    x: 0.6, y: 3.2, w: 6.1, h: 3.55, objectName: "Rho chart",
    altText: "Column chart of shell entropy concentration: supervised-only 0.552, BCP Pancreas 0.894, BCP LA 5% 0.948, BCP LA 10% 0.945, with a dashed threshold at 0.85",
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00",
    dataLabelPosition: "ctr", dataLabelFontSize: 12, dataLabelColor: X.lt1, dataLabelFormatCode: "0.000",
    showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: X.dk1,
    ...quietAxes, ...chartText,
  });

  card(s, 7.1, 1.45, 5.6, 5.3, "Negative control card");
  txt(s, "Negative control: supervised only", { x: 7.45, y: 1.7, w: 5.0, h: 0.45, fontSize: 18, bold: true, color: C.text1, objectName: "Control heading" });
  txt(s, "ρf = 0.55, below ρ* = 0.85", { x: 7.45, y: 2.18, w: 5.0, h: 0.4, fontSize: 16, color: C.text1, objectName: "Control rho" });
  txt(s, "75.66 → 17.29", { x: 7.45, y: 2.75, w: 5.0, h: 0.85, fontSize: 44, bold: true, color: C.accent6, objectName: "Control stat" });
  txt(s, "Dice after PEM at the Pancreas-selected setting (τm = 0.90, 5×10⁻⁵, E = 2); mean of 3 seeds, SD 5.47",
    { x: 7.45, y: 3.68, w: 5.0, h: 0.75, fontSize: 14, color: C.text1, objectName: "Control stat label" });
  txt(s, "The same setting improves BCP Pancreas-CT by +1.02 Dice. Without a boundary shell, sharpening reinforces errors.",
    { x: 7.45, y: 4.55, w: 5.0, h: 0.9, fontSize: 16, color: C.text1, objectName: "Control interpretation" });
  caption(s, "ρ* is provisional: it rests on these four checkpoints. ρf has no spatial term; boundary localization comes from distance to the boundary (5–11 vs 14.5 voxels).",
    { x: 7.45, y: 5.6, w: 5.0, h: 0.95, objectName: "Control caveat" });
  s.addNotes(
`[3:30–4:40] Because the method hurts when the uncertainty is not boundary-shaped, we want a check we can run before fine-tuning, without labels.
That is rho-f: the fraction of total prediction entropy held by the low-confidence voxels, from a single forward pass on the unlabeled data.
The three BCP checkpoints all score about 0.9 or higher. A supervised-only network trained on the same 12 pancreas labels scores 0.55 — its uncertainty is scattered through the interior.
And on that checkpoint PEM fails badly: the very setting that improves BCP pancreas by about one Dice point collapses the supervised model from 75.7 to 17.3 Dice.
Two caveats, stated honestly: the threshold of 0.85 rests on these four checkpoints only, and rho-f itself has no spatial term — that the uncertain voxels sit on the boundary we show separately, with their distance to the boundary.`);
}

// 6 ── Main results ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  s.addText("PEM sharpens boundaries: HD95 down 25–44%", { placeholder: "title" });
  const cats = ["Pancreas-CT 20%", "LA 5%", "LA 10%"];
  s.addChart(pres.charts.BAR, [
    { name: "BCP", labels: cats, values: [7.81, 13.91, 9.88] },
    { name: "+ PL-FT", labels: cats, values: [7.68, 10.11, 6.37] },
    { name: "+ SC", labels: cats, values: [7.62, 11.77, 8.04] },
    { name: "+ PEM", labels: cats, values: [5.87, 7.79, 6.87] },
  ], {
    x: 0.6, y: 1.4, w: 7.7, h: 4.7, barDir: "col", barGrouping: "clustered", barGapWidthPct: 70,
    objectName: "HD95 chart",
    altText: "Clustered column chart of HD95 in voxels. Pancreas: BCP 7.81, PL-FT 7.68, SC 7.62, PEM 5.87. LA 5%: 13.91, 10.11, 11.77, 7.79. LA 10%: 9.88, 6.37, 8.04, 6.87",
    chartColors: [X.accent5, X.accent4, X.accent3, X.accent1],
    showTitle: true, title: "HD95 in voxels (lower is better), 3-seed means", titleFontSize: 14, titleColor: X.dk1,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: X.dk1, dataLabelFormatCode: "0.00",
    showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: X.dk1,
    valAxisMinVal: 0, valAxisMaxVal: 16, valAxisMajorUnit: 4,
    ...quietAxes, ...chartText,
  });
  caption(s, "PEM has the lowest mean HD95 on Pancreas-CT and LA 5%; pseudo-label fine-tuning (PL-FT) is lowest on LA 10%.",
    { x: 0.6, y: 6.2, w: 7.7, h: 0.55, objectName: "HD95 caption" });

  card(s, 8.75, 1.45, 3.95, 5.3, "Dice card");
  txt(s, "Dice gain over BCP", { x: 9.1, y: 1.7, w: 3.3, h: 0.45, fontSize: 18, bold: true, color: C.text1, objectName: "Dice heading" });
  [["+1.02", "Pancreas-CT 20%"], ["+0.66", "LA 5%"], ["+0.62", "LA 10%"]].forEach(([v, l], i) => {
    const y = 2.3 + i * 1.05;
    txt(s, v, { x: 9.05, y, w: 2.05, h: 0.75, fontSize: 40, bold: true, color: C.accent1, objectName: `Dice gain ${l}` });
    txt(s, l, { x: 11.15, y: y + 0.1, w: 1.45, h: 0.6, fontSize: 15, color: C.text1, valign: "middle", objectName: `Dice label ${l}` });
  });
  caption(s, "All methods chose their learning rate by the same rule on the labeled volumes; 3-seed means.",
    { x: 9.1, y: 5.55, w: 3.3, h: 0.95, objectName: "Dice caption" });
  s.addNotes(
`[4:40–5:50] Here are the main results, on the three BCP checkpoints, against two post-hoc baselines run under the identical protocol: pseudo-label fine-tuning and self-consistency. Temperature scaling is omitted because it cannot move the argmax — it changes Dice by less than 0.04.
PEM reduces the 95th-percentile Hausdorff distance by 25 percent on pancreas, 44 percent on LA 5%, and 30 percent on LA 10%, and improves Dice by about 1.0, 0.7 and 0.6 points.
Against the baselines: PEM has the lowest mean HD95 on pancreas and LA 5%; on LA 10% pseudo-label fine-tuning is lower.
Every method — PEM and both baselines — selected its learning rate with the same rule on the labeled training volumes, never on the test set, and all numbers are three-seed means.`);
}

// 7 ── Qualitative ──────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  s.addText("Corrections land on the organ surface", { placeholder: "title" });
  const cols = ["Image", "GT", "BCP", "PEM", "Entropy", "Corr."];
  const keys = ["image", "gt", "bcp", "pem", "entropy", "corr"];
  const rows = [["pancreas_ct_20pct", "Pancreas\n20%"], ["la_5pct", "LA 5%"], ["la_10pct", "LA 10%"]];
  const x0 = 1.75, cw = 1.83, y0 = 1.85, rh = 1.42, pad = 0.06;
  cols.forEach((c, j) => txt(s, c, { x: x0 + j * cw, y: 1.42, w: cw, h: 0.35, fontSize: 15, bold: true,
    color: C.text1, align: "center", objectName: `Column ${c}` }));
  rows.forEach(([st, label], i) => {
    const y = y0 + i * rh;
    txt(s, label, { x: 0.6, y, w: 1.05, h: rh, fontSize: 14, bold: true, color: C.text1, align: "right",
      valign: "middle", objectName: `Row ${label}` });
    keys.forEach((k, j) => {
      const p = data.panels[`${st}_${k}`];
      const bw = cw - 2 * pad, bh = rh - 2 * pad;
      const sc = Math.min(bw / p.size[0], bh / p.size[1]);
      const w = p.size[0] * sc, h = p.size[1] * sc;
      s.addImage({ path: p.path, x: x0 + j * cw + (cw - w) / 2, y: y + (rh - h) / 2, w, h,
        objectName: `${label} ${cols[j]}`, altText: `${label}, ${cols[j]} panel` });
    });
  });
  caption(s, "Median-improvement test case per setting (seed 2020), so nothing is cherry-picked. GT green, BCP red, PEM cyan, Dice overlaid; Corr.: green = PEM fixes, red = PEM breaks.",
    { x: 0.6, y: 6.2, w: 12.1, h: 0.55, objectName: "Qualitative caption" });
  s.addNotes(
`[5:50–6:40] Qualitatively, these are the median-improvement test cases for each setting — chosen by rule, so nothing is cherry-picked.
The entropy column shows the thin bright rim the gradient acts on. In the correction column, green is where PEM fixes BCP and red is where it breaks it.
Corrections of either sign lie on the surface, not in the interior — that is what the mechanism predicts. Their balance follows the headroom: green clearly dominates on LA 5%, corrections are sparse on pancreas, and on the most saturated setting, LA 10%, red covers part of the rim even though this case improves overall.`);
}

// 8 ── Significance ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  s.addText("Where the gain holds up: Pancreas-CT, not LA", { placeholder: "title" });
  caption(s, "Exploratory retrospective analysis  ·  per-case Wilcoxon on seed-averaged cases  ·  Holm across the 3 settings  ·  HD95 primary",
    { x: 0.6, y: 1.3, w: 12.1, h: 0.35, objectName: "Test description" });
  const sig = { fill: { color: C.accent1 }, color: C.background1, bold: true };
  const ns = { fill: { color: C.background2 }, color: C.text1 };
  const head = { fill: { color: C.text2 }, color: C.background1, bold: true };
  const cell = (t, st) => ({ text: t, options: { ...st, align: "center", valign: "middle" } });
  const rowHead = (t) => ({ text: t, options: { bold: true, color: C.text1, valign: "middle", fill: { color: C.background1 } } });
  s.addTable([
    [cell("Holm-adjusted p, HD95", head), cell("PEM vs BCP", head), cell("PEM vs PL-FT", head), cell("PEM vs SC", head)],
    [rowHead("Pancreas-CT 20%\nN = 18"), cell("< .001  (17/18 cases)", sig), cell(".003", sig), cell(".003", sig)],
    [rowHead("LA 5%\nN = 20"), cell(".059", ns), cell(".701", ns), cell(".153", ns)],
    [rowHead("LA 10%\nN = 20"), cell(".059", ns), cell(".405", ns), cell(".153", ns)],
  ], { x: 0.6, y: 1.9, w: 8.1, colW: [2.6, 2.1, 1.75, 1.65], rowH: [0.7, 0.9, 0.9, 0.9], fontSize: 16,
    border: { type: "solid", pt: 2, color: X.lt1 }, objectName: "Significance table" });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 5.55, w: 0.3, h: 0.3, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "Legend significant" });
  txt(s, "p < 0.05 after Holm correction", { x: 1.0, y: 5.55, w: 3.5, h: 0.3, fontSize: 14, color: C.text1, valign: "middle", objectName: "Legend label" });

  card(s, 9.1, 1.9, 3.6, 4.85, "Interpretation card");
  txt(s, [
    { text: "Pancreas-CT: ", options: { bold: true } },
    { text: "lower HD95 than BCP and both post-hoc baselines.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "LA: ", options: { bold: true } },
    { text: "gains over BCP are nominal (raw p = .030, .033); no significant difference from PL-FT on either metric.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Dice vs BCP: ", options: { bold: true } },
    { text: "Holm p = .042, .090, .041." },
  ], { x: 9.4, y: 2.15, w: 3.0, h: 4.4, fontSize: 16, color: C.text1, objectName: "Interpretation text" });
  caption(s, "Budgets, grid bounds and the Pancreas base checkpoint were informed by test results; Holm correction does not remove that bias.",
    { x: 0.6, y: 6.1, w: 8.1, h: 0.6, objectName: "Exploratory caveat" });
  s.addNotes(
`[6:40–7:50] Are these gains reliable? We ran per-case paired Wilcoxon tests, averaging each case over the three seeds first, with HD95 as the primary endpoint and Holm correction across the three settings.
On pancreas the boundary gain is clear: PEM lowers HD95 against BCP on 17 of 18 cases, and beats both pseudo-labeling and self-consistency after correction.
On left atrium it is not: the HD95 gains over BCP are nominal — raw p of .03, but .059 after correction — and we detect no significant difference from pseudo-label fine-tuning on either metric.
We call these tests exploratory, and I want to be explicit why: the epoch budgets, the learning-rate grid bounds and the pancreas base checkpoint were informed by earlier test-set results, and a multiplicity correction does not remove that kind of selection bias.
Our reading: PEM's advantage over pseudo-labeling shrinks as the checkpoint saturates — which is what the gradient analysis predicts, since the entropy gradient vanishes as predictions commit.`);
}

// 9 ── Mask ablation ────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  s.addText("The mask is insurance, not a source of gain", { placeholder: "title" });
  const grid = [
    ["Full (no mask)", [0.66, -0.53, -0.86, -11.3, 0.53, 1.12], 3],
    ["Confident τm = 0.90", [0.70, 1.20, 0.63, 0.36, 0.41, 1.04], 0],
    ["Confident τm = 0.95", [0.87, 1.43, 0.55, -1.58, 0.44, 1.04], 1],
    ["Confident τm = 0.99", [0.94, 0.40, -1.17, -16.8, 0.50, 1.07], 2],
    ["Confident τm = 0.999", [0.74, -0.81, -2.57, -30.2, 0.53, 1.11], 3],
  ];
  const h = { fill: { color: C.text2 }, color: C.background1, bold: true, align: "center", valign: "middle" };
  const fmt = (v) => (v >= 0 ? "+" : "−") + (Math.abs(v) >= 10 ? Math.abs(v).toFixed(1) : Math.abs(v).toFixed(2));
  const rows = [
    [{ text: "Mask", options: { ...h, rowspan: 2, align: "left" } }, { text: "LA 5%", options: { ...h, colspan: 2 } },
     { text: "LA 10%", options: { ...h, colspan: 2 } }, { text: "Pancreas-CT 20%", options: { ...h, colspan: 2 } },
     { text: "Degraded", options: { ...h, rowspan: 2 } }],
    ["5×10⁻⁶", "1×10⁻⁵", "5×10⁻⁶", "1×10⁻⁵", "1×10⁻⁵", "5×10⁻⁵"].map((t) => ({ text: t, options: { ...h, fontSize: 13 } })),
    ...grid.map(([name, vals, deg]) => {
      const sel = name.includes("0.90");
      const base = sel ? { fill: { color: LIGHT_TEAL }, bold: true } : { fill: { color: X.lt1 } };
      return [
        { text: name, options: { ...base, color: C.text1, valign: "middle" } },
        ...vals.map((v) => ({ text: fmt(v), options: { ...base, align: "center", valign: "middle",
          color: v < 0 ? C.accent6 : C.text1, ...(v < 0 ? { fill: { color: LIGHT_RED } } : {}) } })),
        { text: String(deg), options: { ...base, align: "center", valign: "middle", bold: true, color: deg === 0 ? C.accent1 : C.accent6 } },
      ];
    }),
  ];
  s.addTable(rows, { x: 0.6, y: 1.4, w: 12.1, colW: [2.65, 1.33, 1.33, 1.33, 1.33, 1.33, 1.33, 1.47],
    rowH: [0.42, 0.38, 0.47, 0.47, 0.47, 0.47, 0.47], fontSize: 15,
    border: { type: "solid", pt: 1, color: "E4E2EA" }, objectName: "Ablation table" });
  caption(s, "ΔDice vs BCP at each dataset's fixed endpoint, seed 2020; learning rates in the second header row. Red cells: the checkpoint got worse.",
    { x: 0.6, y: 4.88, w: 12.1, h: 0.3, objectName: "Ablation caption" });

  const stats = [
    ["0 of 6", C.accent1, "cells degraded at τm = 0.90, which also has the highest mean gain (+0.72)"],
    ["up to −30", C.accent6, "Dice when the mask is too loose (τm = 0.999, LA 10%)"],
    ["+0.12 / +0.08", C.accent4, "Dice in favour of no mask on Pancreas-CT, a robust instance-norm checkpoint"],
  ];
  stats.forEach(([v, col, l], i) => {
    const x = 0.6 + i * 4.15;
    card(s, x, 5.3, 3.8, 1.5, `Ablation stat ${i + 1} card`);
    txt(s, v, { x: x + 0.25, y: 5.38, w: 3.3, h: 0.55, fontSize: 28, bold: true, color: col, objectName: `Ablation stat ${i + 1}` });
    txt(s, l, { x: x + 0.25, y: 5.95, w: 3.3, h: 0.8, fontSize: 14, color: C.text1, objectName: `Ablation stat ${i + 1} label` });
  });
  s.addNotes(
`[7:50–8:50] What does the mask buy? This is the ablation, read at each dataset's fixed endpoint, not at a peak.
Loosening the mask from 0.90 toward 0.999 readmits the saturated interior and approaches the unmasked loss — and the number of degraded cells, in red, rises along that path from zero to three. The failures are large: the loosest mask costs 30 Dice points on LA 10%.
tau-m = 0.90 is the only threshold that never degrades a BCP checkpoint, and it also has the highest mean gain. tau-m = 0.95 has the single largest cell, but it beats 0.99 in only three of six cells, so we don't recommend it.
Interestingly, on the robust instance-normalized pancreas checkpoint the unmasked loss is slightly better. So the mask is insurance against collapse on brittle checkpoints, not a source of gain.`);
}

// 10 ── Takeaways ───────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Talk" });
  s.addText("Takeaways", { placeholder: "title" });
  const items = [
    ["Residual entropy is structured", "On converged BCP checkpoints it sits in a few low-confidence voxels on the organ surface. ρf measures the concentration from one forward pass."],
    ["A cheap post-hoc step", "2–5 epochs, no labels, under a minute of fine-tuning: HD95 −25% to −44%, Dice +0.6 to +1.0. Clearest gain on Pancreas-CT."],
    ["Use it with care", "Check ρf first, keep τm = 0.90, and set the learning rate by normalization type: 5×10⁻⁶ batch-norm, up to 5×10⁻⁵ instance-norm."],
  ];
  items.forEach(([head, body], i) => {
    const x = 0.6 + i * 4.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.45, w: 3.8, h: 3.4, rectRadius: 0.12,
      fill: { color: C.background1, transparency: 88 }, line: { color: C.background1, width: 0, transparency: 100 }, objectName: `Takeaway ${i + 1} card` });
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.7, w: 0.6, h: 0.6, fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 }, objectName: `Takeaway ${i + 1} badge` });
    txt(s, String(i + 1), { x: x + 0.3, y: 1.7, w: 0.6, h: 0.6, fontSize: 20, bold: true, color: C.text2, align: "center", valign: "middle", objectName: `Takeaway ${i + 1} number` });
    txt(s, head, { x: x + 0.3, y: 2.42, w: 3.2, h: 0.62, fontSize: 18, bold: true, valign: "bottom", color: C.background1, objectName: `Takeaway ${i + 1} heading` });
    txt(s, body, { x: x + 0.3, y: 3.12, w: 3.2, h: 1.65, fontSize: 15, color: C.background1, objectName: `Takeaway ${i + 1} text` });
  });
  txt(s, "Limitations: the significance tests are exploratory; ρ* rests on four checkpoints; 18–20 test volumes.",
    { x: 0.6, y: 5.1, w: 12.1, h: 0.4, fontSize: 15, color: C.accent5, objectName: "Limitations" });
  txt(s, "Thank you — questions?", { x: 0.6, y: 5.7, w: 8, h: 0.6, fontSize: 30, bold: true, color: C.background1, objectName: "Closing" });
  txt(s, "shaharabany@mail.tau.ac.il", { x: 0.6, y: 6.35, w: 8, h: 0.4, fontSize: 16, color: C.accent2, objectName: "Contact" });
  s.addNotes(
`[8:50–9:45] To summarize. First, converged SSL checkpoints leave residual entropy that is spatially structured — concentrated in a few low-confidence voxels on the organ surface — and rho-f measures that concentration without labels.
Second, Post-hoc Entropy Minimization exploits it cheaply: two to five epochs, no labels, under a minute of fine-tuning, reducing HD95 by 25 to 44 percent, with the clearest and statistically strongest gain on pancreas.
Third, use it with care: check rho-f first, keep the tight mask, and set the learning rate by normalization type — batch-normalized checkpoints break at rates instance-normalized ones tolerate.
The main limitations: our tests are exploratory, the rho-star threshold rests on four checkpoints, and the test sets have 18 to 20 volumes.
Thank you — I'm happy to take questions.`);
}

// ════════════════════════════════════════════════════════════════════════════
pres.addSection({ title: "Backup" });

// B1 ── Protocol ────────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("Backup: how the settings were chosen", { placeholder: "title" });
  const col = (x, head, items, name) => {
    card(s, x, 1.45, 5.85, 5.3, `${name} card`);
    txt(s, head, { x: x + 0.3, y: 1.65, w: 5.25, h: 0.45, fontSize: 19, bold: true, color: C.text1, objectName: `${name} heading` });
    txt(s, items.map((t, i) => ({ text: t, options: { bullet: true, ...(i < items.length - 1 ? { breakLine: true } : {}) } })),
      { x: x + 0.3, y: 2.2, w: 5.25, h: 4.4, fontSize: 15, color: C.text1, paraSpaceAfter: 9, objectName: `${name} text` });
  };
  col(0.6, "Selection rule", [
    "Labeled training volumes (12 Pancreas; 4 and 8 LA) are excluded from every post-hoc objective and used only to choose the learning rate and τm.",
    "The rule maximizes their Dice change, not absolute Dice, since they also trained the base checkpoint.",
    "The same rule and grid apply to PL-FT and SC; every method reports 3-seed means.",
    "Five-fold cross-validation is impossible with four volumes, so this is a non-independent selection set; selection overfitting remains possible.",
  ], "Selection rule");
  col(6.85, "What is retrospective", [
    "This is a corrected retrospective reanalysis of the submitted study.",
    "Budgets E = 2 (Pancreas) and E = 5 (LA) are carried over from the submitted protocol; grid bounds came from earlier sweeps.",
    "Pancreas BCP is our reproduction (82.89 vs 82.91 reported), saved at its best test Dice, so its base number is optimistic.",
    "Retrospective rank check over 33 configurations: validation vs. test ΔDice, Spearman ρ = 0.78 / 0.75 / 0.99 (Pancreas / LA 5% / LA 10%).",
  ], "Retrospective");
  s.addNotes(`Backup for questions on the protocol. Key points: no learning rate or threshold is chosen on test data; the labeled volumes are excluded from the post-hoc loss but are not independent of the base checkpoint; the budgets came from the submitted protocol; the rank check uses test data and is retrospective.`);
}

// B2 ── Full Table II ───────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("Backup: full results (3-seed means)", { placeholder: "title" });
  const h = { fill: { color: C.text2 }, color: C.background1, bold: true, align: "center", valign: "middle" };
  const grp = (t) => [{ text: t, options: { colspan: 5, bold: true, italic: true, color: C.text1, fill: { color: C.background2 } } }];
  const r = (m, v, best) => [{ text: m, options: { color: C.text1, bold: m.includes("PEM") } },
    ...v.map((x, i) => ({ text: x, options: { align: "center", color: C.text1, bold: best[i] } }))];
  const B = [true, true, true, true], N = [false, false, false, false];
  s.addTable([
    [{ text: "Method", options: { ...h, align: "left" } }, { text: "Dice ↑", options: h }, { text: "Jaccard ↑", options: h },
     { text: "HD95 ↓", options: h }, { text: "ASD ↓", options: h }],
    grp("Pancreas-CT 20%  (12 labeled / 50 unlabeled / 18 test)"),
    r("BCP (reproduced)", ["82.89", "71.01", "7.81", "2.62"], N), r("+ PL-FT", ["83.18", "71.43", "7.68", "2.44"], N),
    r("+ SC", ["83.04", "71.23", "7.62", "2.49"], N), r("+ PEM", ["83.91", "72.50", "5.87", "1.74"], B),
    grp("LA 5%  (4 labeled / 76 unlabeled / 20 test)"),
    r("BCP (released)", ["87.32", "77.65", "13.91", "3.60"], N), r("+ PL-FT", ["88.24", "79.06", "10.11", "2.68"], [true, true, false, false]),
    r("+ SC", ["87.75", "78.29", "11.77", "3.08"], N), r("+ PEM", ["87.99", "78.68", "7.79", "2.28"], [false, false, true, true]),
    grp("LA 10%  (8 labeled / 72 unlabeled / 20 test)"),
    r("BCP (released)", ["89.40", "80.92", "9.88", "2.86"], N), r("+ PL-FT", ["90.09", "82.05", "6.37", "1.84"], B),
    r("+ SC", ["89.58", "81.24", "8.04", "2.30"], N), r("+ PEM", ["90.02", "81.95", "6.87", "2.19"], N),
  ], { x: 0.6, y: 1.4, w: 12.1, colW: [4.1, 2.0, 2.0, 2.0, 2.0], rowH: 0.3, fontSize: 14, margin: [0.02, 0.08, 0.02, 0.08],
    border: { type: "solid", pt: 1, color: "E4E2EA" }, objectName: "Full results table" });
  caption(s, "Bold: best per column within each setting. Temperature scaling omitted (|ΔDice| < 0.04). No significant PEM–PL-FT difference is detected on LA.",
    { x: 0.6, y: 6.62, w: 12.1, h: 0.3, objectName: "Full results caption" });
  s.addNotes(`Backup: Table II of the paper. Bold marks the best per column within each setting.`);
}

// B3 ── Full Table IV ───────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("Backup: per-case tests (exploratory)", { placeholder: "title" });
  const h = { fill: { color: C.text2 }, color: C.background1, bold: true, align: "center", valign: "middle" };
  const grp = (t) => [{ text: t, options: { colspan: 7, bold: true, italic: true, color: C.text1, fill: { color: C.background2 } } }];
  const sigCell = (t) => ({ text: t, options: { align: "center", color: C.text1, bold: t === "< .001" || (t.startsWith(".") && parseFloat("0" + t) < 0.05) } });
  const r = (setting, v) => [{ text: setting, options: { color: C.text1 } }, ...v.map(sigCell)];
  s.addTable([
    [{ text: "", options: h }, { text: "HD95 (primary)", options: { ...h, colspan: 3 } }, { text: "Dice (secondary)", options: { ...h, colspan: 3 } }],
    ["Setting", "Δ", "p", "Holm p", "Δ", "p", "Holm p"].map((t, i) => ({ text: t, options: { ...h, ...(i === 0 ? { align: "left" } : {}) } })),
    grp("PEM − BCP"),
    r("Pancreas-CT 20%", ["−1.94", "< .001", "< .001", "+1.02", ".021", ".042"]),
    r("LA 5%", ["−6.12", ".030", ".059", "+0.66", ".090", ".090"]),
    r("LA 10%", ["−3.02", ".033", ".059", "+0.62", ".014", ".041"]),
    grp("PEM − PL-FT"),
    r("Pancreas-CT 20%", ["−1.81", ".001", ".003", "+0.73", ".060", ".180"]),
    r("LA 5%", ["−2.33", ".701", ".701", "−0.26", ".927", "1.00"]),
    r("LA 10%", ["+0.50", ".202", ".405", "−0.07", ".648", "1.00"]),
    grp("PEM − SC"),
    r("Pancreas-CT 20%", ["−1.75", "< .001", ".003", "+0.87", ".048", ".145"]),
    r("LA 5%", ["−3.99", ".077", ".153", "+0.24", ".261", ".261"]),
    r("LA 10%", ["−1.17", ".097", ".153", "+0.44", ".058", ".145"]),
  ], { x: 0.6, y: 1.4, w: 12.1, colW: [3.4, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45], rowH: 0.34, fontSize: 14,
    border: { type: "solid", pt: 1, color: "E4E2EA" }, objectName: "Per-case tests table" });
  caption(s, "Two-sided paired Wilcoxon on seed-averaged cases (N = 18 Pancreas, N = 20 LA); for each comparator and metric, Holm across the three settings. Bold: p < .05.",
    { x: 0.6, y: 6.3, w: 12.1, h: 0.5, objectName: "Per-case tests caption" });
  s.addNotes(`Backup: Table IV of the paper, with raw and Holm-adjusted p-values. These are exploratory because budgets, grid bounds and the Pancreas base checkpoint were informed by test results.`);
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
