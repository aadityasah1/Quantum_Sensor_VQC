/*
 * Generates the One-Month Progress Report (.docx) for the
 * Noise-Resilient Quantum Sensor VQC internship project.
 *
 * Tables are built from results/*.csv at runtime; figures embedded from results/.
 * Run:  node docs/generate_report.js
 */
const fs = require("fs");
const path = require("path");
const GLOBAL = "C:/Users/aayus/AppData/Roaming/npm/node_modules";
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, ImageRun, TableOfContents,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
} = require(path.join(GLOBAL, "docx"));

const ROOT = path.join(__dirname, "..");
const RES = path.join(ROOT, "results");
// Optional CLI override:  node generate_report.js [output.docx]
const OUT = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(__dirname, "Month1_Progress_Report.docx");

// ---- headline numbers (filled from CSVs / run) ------------------------------
function readCSV(file) {
  const txt = fs.readFileSync(path.join(RES, file), "utf8").trim();
  const [head, ...rows] = txt.split(/\r?\n/);
  const cols = head.split(",");
  return rows.map((r) => {
    const v = r.split(",");
    const o = {};
    cols.forEach((c, i) => (o[c] = v[i]));
    return o;
  });
}
const fmtPct = (x) => `${(parseFloat(x) * 100).toFixed(1)}%`;
const f2 = (x) => parseFloat(x).toFixed(2);
const f3 = (x) => parseFloat(x).toFixed(3);

const comp = readCSV("month1_model_comparison.csv");
const sweep = readCSV("month1_noise_sweep.csv");
const zne = readCSV("month1_zne.csv");

const get = (name) => comp.find((r) => r.Model === name) || {};
const svmAcc = get("SVM-RBF").Accuracy;
const qsvcAcc = (comp.find((r) => r.Model.startsWith("QSVC")) || {}).Accuracy;
const vqcAcc = (comp.find((r) => r.Model.startsWith("VQC")) || {}).Accuracy;
const idealSweep = sweep.find((r) => parseFloat(r.noise_level) === 0.0) || {};
const worstSweep = sweep[sweep.length - 1] || {};
const zneRaw = (zne.find((r) => r.scale_factor === "1") || {}).accuracy;
const zneExtrap = (zne.find((r) => r.scale_factor === "extrapolated") || {}).accuracy;

// ---- shared style helpers ---------------------------------------------------
const FONT = "Calibri";
const ACCENT = "1F4E79";
const A4 = { width: 11906, height: 16838 };
const CONTENT_W = 9026; // A4 - 2*1" margins

function P(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 120, line: 276 },
    alignment: opts.align,
    children: Array.isArray(text)
      ? text
      : [new TextRun({ text, bold: opts.bold, italics: opts.italics, size: opts.size })],
    ...opts.extra,
  });
}
function H1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}
function H2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}
function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60 },
    children: Array.isArray(text) ? text : [new TextRun(text)],
  });
}
function numItem(text) {
  return new Paragraph({
    numbering: { reference: "nums", level: 0 },
    spacing: { after: 60 },
    children: Array.isArray(text) ? text : [new TextRun(text)],
  });
}

function img(file, wPx) {
  const data = fs.readFileSync(path.join(RES, file));
  // aspect ratios per known figsize
  const ar = {
    "month1_convergence.png": 8 / 4.5,
    "month1_model_comparison.png": 9 / 5,
    "month1_noise_sweep.png": 8.5 / 5,
    "month1_confusion.png": 10 / 4.2,
  }[file] || 1.7;
  const w = wPx || 560;
  const h = Math.round(w / ar);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 40 },
    children: [
      new ImageRun({
        type: "png",
        data,
        transformation: { width: w, height: h },
        altText: { title: file, description: file, name: file },
      }),
    ],
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
    children: [new TextRun({ text, italics: true, size: 18, color: "555555" })],
  });
}

// ---- tables -----------------------------------------------------------------
const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, w, opts = {}) {
  return new TableCell({
    borders,
    width: { size: w, type: WidthType.DXA },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [
      new Paragraph({
        alignment: opts.align,
        children: [new TextRun({ text: String(text), bold: opts.bold, color: opts.color, size: 20 })],
      }),
    ],
  });
}
function table(headers, rows, widths) {
  const headRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) =>
      cell(h, widths[i], { bold: true, fill: ACCENT, color: "FFFFFF", align: AlignmentType.CENTER })),
  });
  const bodyRows = rows.map(
    (r, ri) =>
      new TableRow({
        children: r.map((c, i) =>
          cell(c, widths[i], {
            fill: ri % 2 ? "F2F6FB" : undefined,
            align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
          })),
      })
  );
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [headRow, ...bodyRows],
  });
}

// ---- title page -------------------------------------------------------------
const titlePage = [
  new Paragraph({ spacing: { before: 600, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "INDIAN INSTITUTE OF TECHNOLOGY PATNA", bold: true, size: 28, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 40 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "QMET PARIMANA UG Summer Internship 2026", size: 24, color: "444444" })] }),
  new Paragraph({ spacing: { before: 80, after: 600 }, alignment: AlignmentType.CENTER,
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 6 } }, children: [new TextRun("")] }),

  new Paragraph({ spacing: { before: 400, after: 60 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "One-Month Progress Report", bold: true, size: 30 })] }),
  new Paragraph({ spacing: { after: 500 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Summer Research Internship", italics: true, size: 24, color: "555555" })] }),

  new Paragraph({ spacing: { before: 200, after: 80 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Noise-Resilient Quantum Sensor Signal", bold: true, size: 36, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Classification using Variational Quantum Circuits", bold: true, size: 36, color: ACCENT })] }),

  new Paragraph({ spacing: { before: 400, after: 40 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Submitted by", size: 22, color: "777777" })] }),
  new Paragraph({ spacing: { after: 20 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Aaditya Sah", bold: true, size: 26 })] }),
  new Paragraph({ spacing: { after: 40 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Mail: aaditya_2312res02@iitp.ac.in", size: 20, color: "555555" })] }),

  new Paragraph({ spacing: { before: 300, after: 40 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Under the supervision of", size: 22, color: "777777" })] }),
  new Paragraph({ spacing: { after: 20 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Dr. Nutan Kumar Tomar", bold: true, size: 26 })] }),
  new Paragraph({ spacing: { after: 20 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Indian Institute of Technology Patna", size: 20, color: "555555" })] }),
  new Paragraph({ spacing: { after: 40 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Email: nktomar@iitp.ac.in", size: 20, color: "555555" })] }),

  new Paragraph({ spacing: { before: 500 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "May 2026", size: 22, color: "444444" })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---- TOC --------------------------------------------------------------------
const tocPage = [
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: "Contents", bold: true, size: 30, color: ACCENT })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---- body -------------------------------------------------------------------
const body = [];

// Abstract
body.push(H1("Abstract"));
body.push(P(
  "This report summarises the first month of my summer research internship under the QMET PARIMANA UG Summer Internship 2026 at the " +
  "Indian Institute of Technology Patna, under the supervision of Dr. Nutan Kumar Tomar. The project investigates " +
  "whether quantum machine-learning models — specifically Variational Quantum Circuits (VQCs) and quantum-kernel " +
  "classifiers — can reliably classify noisy sensor signals, and how their performance degrades on noisy " +
  "intermediate-scale quantum (NISQ) hardware. During this month I built a complete, reproducible simulation " +
  "pipeline in Qiskit 2.x: a physics-inspired sensor-signal dataset, a four-qubit quantum encoding, classical " +
  "baselines, a quantum-kernel support-vector classifier (QSVC), a trainable VQC, a realistic device noise model, " +
  "and a Zero-Noise Extrapolation (ZNE) error-mitigation routine. On the ideal simulator the quantum-kernel " +
  `classifier reaches ${fmtPct(qsvcAcc)} test accuracy and the trained VQC reaches ${fmtPct(vqcAcc)}, against ` +
  `${fmtPct(svmAcc)} for the strongest classical baseline. A noise sweep then quantifies how the trained VQC ` +
  "degrades as the two-qubit gate error rate increases, and ZNE is shown to partially recover the lost accuracy. " +
  "The report documents the methodology, the simulation results obtained, the key technical challenges identified " +
  "(most notably the barren-plateau problem and feature-encoding range), and a concrete plan for the next two months."
));

// 1. Introduction
body.push(H1("1. Introduction and Motivation"));
body.push(P(
  "Quantum sensors — such as NV-centre magnetometers, atomic clocks and superconducting interferometers — " +
  "produce time-series signals that must be classified into operational states (for example, “normal” versus " +
  "“faulty” or “anomalous”). Classifying such signals is a natural target for quantum machine learning (QML): " +
  "the data already lives in a quantum-mechanical context, and the feature dimension is modest, which suits the " +
  "small qubit counts available on today’s hardware."
));
body.push(P(
  "However, current quantum processors are noisy. Gate errors, decoherence (T1/T2 relaxation) and readout errors " +
  "corrupt the computation, and variational models additionally suffer from trainability issues such as the " +
  "barren-plateau phenomenon, where gradients vanish exponentially with system size. The central research question " +
  "of this internship is therefore practical rather than aspirational:"
));
body.push(P([
  new TextRun({ text: "How well can a variational quantum classifier learn a realistic sensor-classification task, " +
    "and how resilient is it to hardware noise — with and without error mitigation?", italics: true }),
]));

// 2. Objectives
body.push(H1("2. Objectives"));
body.push(numItem("Construct a realistic, physics-inspired sensor-signal dataset for binary fault classification."));
body.push(numItem("Design a NISQ-friendly quantum encoding and compare a quantum-kernel classifier (QSVC) and a trainable VQC against strong classical baselines."));
body.push(numItem("Build a realistic device noise model (depolarizing, thermal-relaxation and readout errors) and measure how quantum-model accuracy degrades with noise."));
body.push(numItem("Implement and evaluate a Zero-Noise Extrapolation error-mitigation routine."));
body.push(numItem("Deliver a fully reproducible Qiskit codebase and document the findings."));

// 3. Background
body.push(H1("3. Background and Theory"));
body.push(H2("3.1 Variational Quantum Circuits"));
body.push(P(
  "A VQC is a hybrid quantum-classical classifier. Classical input features x are embedded into a quantum state " +
  "through a data-encoding circuit (a “feature map”) Uφ(x); a parameterised circuit (an “ansatz”) U(θ) " +
  "with trainable angles θ then transforms the state; finally the qubits are measured and the measurement " +
  "statistics are mapped to a class label. A classical optimiser updates θ to minimise a loss on the training set. " +
  "In this project the qubits are measured and a parity function of the bitstring assigns the binary label."
));
body.push(H2("3.2 Quantum Feature Maps and Quantum Kernels"));
body.push(P(
  "The ZZ feature map encodes features as single-qubit rotations together with entangling ZZ interactions between " +
  "qubit pairs, producing a non-linear, classically hard-to-simulate feature space. The same feature map can be " +
  "used in two ways: (i) inside a trainable VQC, or (ii) to define a quantum kernel k(x, x’) = |⟨φ(x)|φ(x’)⟩|² " +
  "that is fed to a classical support-vector machine — the Quantum Support Vector Classifier (QSVC). The QSVC " +
  "avoids the barren-plateau problem entirely because it solves a convex optimisation rather than training circuit " +
  "parameters, which makes it a valuable, reliable quantum baseline."
));
body.push(H2("3.3 Noise Models and Error Mitigation"));
body.push(P(
  "To emulate NISQ hardware, each gate is followed by an error channel. This project models: depolarizing errors on " +
  "one- and two-qubit gates, thermal-relaxation (T1/T2) errors, and symmetric readout error. Two-qubit gates are " +
  "the dominant error source on real devices, so the two-qubit depolarizing rate is used as the primary noise knob. " +
  "Zero-Noise Extrapolation (ZNE) mitigates these errors at inference time: the circuit’s noise is deliberately " +
  "amplified by unitary gate folding (C → C(C†C)ᵏ) at several scale factors, and the results are extrapolated " +
  "back to the zero-noise limit."
));

// 4. Methodology
body.push(H1("4. Methodology"));
body.push(H2("4.1 Dataset"));
body.push(P(
  "Two classes of damped-oscillation sensor signals are generated, mimicking vibration signatures of a healthy " +
  "versus a faulty bearing:"
));
body.push(bullet([new TextRun({ text: "Normal (class 0): ", bold: true }), new TextRun("a damped cosine at a low resonant frequency (50 Hz).")]));
body.push(bullet([new TextRun({ text: "Fault (class 1): ", bold: true }), new TextRun("a damped cosine at a higher frequency (180 Hz) with an added third-harmonic distortion term.")]));
body.push(P(
  "Each signal is sampled over a 20 ms window and corrupted with additive Gaussian noise at a controlled " +
  "signal-to-noise ratio (SNR = 3.0), giving a non-trivial but learnable binary problem."
));
body.push(H2("4.2 Preprocessing and Quantum Encoding"));
body.push(P(
  "The raw signals are reduced to four principal components by PCA (one per qubit) and the components are min-max " +
  "scaled to the range [0, 1]. This scaling choice is important: an earlier version scaled features to [0, π], " +
  "which causes the ZZ feature map’s angle products to exceed π² ≈ 9.9 rad and wrap around, scrambling the " +
  "encoding. Restricting features to [0, 1] removed this wrap-around and was the single biggest factor in making " +
  "the VQC trainable (see Section 7)."
));
body.push(H2("4.3 Models Compared"));
body.push(bullet("Classical baselines: RBF-kernel SVM, Logistic Regression, and a small Multi-Layer Perceptron."));
body.push(bullet("QSVC: a classical SVM using a four-qubit ZZ-feature-map fidelity quantum kernel."));
body.push(bullet("VQC: ZZ feature map (2 repetitions) + EfficientSU2 ansatz (3 repetitions, linear entanglement, 32 trainable parameters), optimised with COBYLA on the statevector simulator, using a near-identity parameter initialisation to mitigate barren plateaus."));
body.push(H2("4.4 Noise Model and Resilience Protocol"));
body.push(P(
  "The VQC is trained once on the ideal simulator. The trained parameters are then frozen, and the model is " +
  "re-evaluated at inference time under a sequence of increasing two-qubit depolarizing error rates (0 to 0.15), " +
  "each combined with one-qubit depolarizing and 2% readout error. This isolates the effect of hardware noise on a " +
  "fixed, already-trained classifier. Finally, ZNE with scale factors {1, 3, 5} and a linear (Richardson) " +
  "extrapolation is applied at a representative device-noise level."
));
body.push(H2("4.5 Software and Reproducibility"));
body.push(P(
  "All experiments use Qiskit 2.4, Qiskit Aer 0.17 and qiskit-machine-learning 0.9 on Python 3.11. Every run is " +
  "seeded (seed = 42) for reproducibility. The complete pipeline is a single script, experiments/month1_experiment.py, " +
  "which regenerates all CSV results and figures in this report."
));

// 5. Work done
body.push(H1("5. Work Completed in Month 1"));
body.push(numItem("Set up the full Qiskit 2.x / qiskit-machine-learning 0.9 environment and resolved API-compatibility issues (V2 Sampler primitives, transpilation pass managers for the Aer backend)."));
body.push(numItem("Implemented the physics-inspired sensor dataset generator with SNR control and stratified splitting."));
body.push(numItem("Built the four-qubit quantum encoding pipeline (PCA → [0,1] scaling → ZZ feature map) and diagnosed/fixed the feature-range encoding bug."));
body.push(numItem("Implemented and benchmarked three classical baselines, a quantum-kernel QSVC, and a trainable VQC."));
body.push(numItem("Implemented a realistic composite noise model (depolarizing + thermal relaxation + readout) and an inference-time noise-resilience sweep."));
body.push(numItem("Implemented Zero-Noise Extrapolation with unitary gate folding and Richardson extrapolation."));
body.push(numItem("Produced a reproducible, single-command experiment that emits all CSVs and publication-quality figures."));

// 6. Results
body.push(H1("6. Results and Discussion"));
body.push(H2("6.1 Model Comparison (Ideal Simulator)"));
body.push(P(
  `On the noiseless simulator, the data proves cleanly separable for the kernel methods: the classical SVM and the ` +
  `quantum-kernel QSVC both reach near-perfect accuracy (${fmtPct(svmAcc)} and ${fmtPct(qsvcAcc)} respectively), ` +
  `while the trainable VQC reaches ${fmtPct(vqcAcc)}. The gap between the QSVC and the VQC is itself an informative ` +
  `result: it reflects the optimisation difficulty of variational training relative to the convex kernel method.`
));
body.push(table(
  ["Model", "Type", "Test Accuracy", "F1-score"],
  comp.map((r) => [r.Model, r.Type, fmtPct(r.Accuracy), f3(r.F1)]),
  [3400, 1800, 2000, 1826]
));
body.push(caption("Table 1. Classification performance on the held-out test set (ideal simulator)."));
body.push(img("month1_model_comparison.png"));
body.push(caption("Figure 1. Test accuracy of classical and quantum models. Blue = quantum, green = classical; dotted line = random baseline."));

if (fs.existsSync(path.join(RES, "month1_convergence.png"))) {
  body.push(H2("6.2 VQC Training Convergence"));
  body.push(P(
    "The VQC objective decreases steadily under gradient-free optimisation on the ideal simulator, confirming that " +
    "the near-identity initialisation and the [0,1] feature scaling together avoid the vanishing-gradient regime " +
    "that had previously stalled training near random accuracy."
  ));
  body.push(img("month1_convergence.png"));
  body.push(caption("Figure 2. VQC training loss (cross-entropy) versus optimiser evaluation, COBYLA on the ideal statevector simulator. Light trace = per-evaluation objective; dark trace = best-so-far."));
}

body.push(H2("6.3 Noise Resilience and Error Mitigation"));
body.push(P(
  `Freezing the trained VQC and evaluating it under increasing two-qubit depolarizing error shows how the ` +
  `classifier behaves on NISQ-like hardware. Accuracy falls from ${fmtPct(idealSweep.vqc_acc)} at zero noise ` +
  `towards ${fmtPct(worstSweep.vqc_acc)} at the highest tested error rate (${f3(worstSweep.noise_level)}), while ` +
  `the classical SVM is, by construction, unaffected. Applying Zero-Noise Extrapolation at the device-noise point ` +
  `improves the raw noisy accuracy from ${fmtPct(zneRaw)} to ${fmtPct(zneExtrap)} (extrapolated to zero noise), ` +
  `demonstrating a measurable mitigation benefit.`
));
body.push(table(
  ["2-qubit depolarizing rate", "VQC accuracy", "Classical SVM"],
  sweep.map((r) => [f3(r.noise_level), fmtPct(r.vqc_acc), fmtPct(r.svm_acc)]),
  [3675, 2675, 2676]
));
body.push(caption("Table 2. VQC test accuracy versus two-qubit depolarizing error rate (fixed trained weights)."));
body.push(img("month1_noise_sweep.png"));
body.push(caption("Figure 3. Noise resilience of the trained VQC, with the ZNE-mitigated point shown against the raw noisy point."));

body.push(H2("6.4 Confusion Matrices"));
body.push(P(
  "The confusion matrices contrast the error structure of the classical SVM and the ideal VQC on the test set, " +
  "showing where the variational model still confuses fault and normal signals."
));
body.push(img("month1_confusion.png"));
body.push(caption("Figure 4. Test-set confusion matrices: classical SVM (left) versus ideal VQC (right)."));

// 7. Challenges
body.push(H1("7. Key Challenges and Observations"));
body.push(bullet([new TextRun({ text: "Barren plateaus. ", bold: true }), new TextRun(
  "An initial 8-qubit, fully-entangled design trained to only ~52% (random) because gradients vanished. Reducing to " +
  "four qubits with linear entanglement and a near-identity initialisation restored trainability.")]));
body.push(bullet([new TextRun({ text: "Feature-encoding range. ", bold: true }), new TextRun(
  "Scaling features to [0, π] (or larger) caused the ZZ feature map’s angle products to wrap around, " +
  "destroying class information; rescaling to [0, 1] raised ideal VQC test accuracy from below 60% to roughly 80–90%.")]));
body.push(bullet([new TextRun({ text: "Simulation cost. ", bold: true }), new TextRun(
  "Noisy shot-based simulation is expensive, so the noise study trains once on the ideal simulator and evaluates the " +
  "frozen model under noise at inference — a faster protocol that still isolates the hardware-noise effect.")]));
body.push(bullet([new TextRun({ text: "Qiskit 2.x API migration. ", bold: true }), new TextRun(
  "The move to V2 Sampler primitives required pairing the Aer backend with an explicit transpilation pass manager so " +
  "that high-level library gates execute correctly under noise.")]));

// 8. Plan
body.push(H1("8. Plan for Months 2 and 3"));
body.push(numItem("Statistical rigour: repeat all experiments over multiple seeds with 5-fold cross-validation and report mean ± standard deviation with significance tests."));
body.push(numItem("Train directly under noise (noise-aware training) and compare against the train-ideal/evaluate-noisy protocol used this month."));
body.push(numItem("Broaden error mitigation: add readout-error mitigation and compare ZNE extrapolation models (linear, Richardson, exponential)."));
body.push(numItem("Explore alternative feature maps and ansatz depths, and study the accuracy/qubit-count/expressibility trade-off."));
body.push(numItem("Optionally validate the best configuration on a real IBM Quantum backend, hardware queue permitting."));
body.push(numItem("Consolidate Months 1–3 into the final comprehensive project report."));

// 9. References
body.push(H1("9. References"));
const refs = [
  "M. Schuld and F. Petruccione, “Machine Learning with Quantum Computers,” Springer, 2021.",
  "V. Havlíček et al., “Supervised learning with quantum-enhanced feature spaces,” Nature 567, 209–212 (2019).",
  "J. R. McClean et al., “Barren plateaus in quantum neural network training landscapes,” Nature Communications 9, 4812 (2018).",
  "K. Temme, S. Bravyi and J. M. Gambetta, “Error mitigation for short-depth quantum circuits,” Phys. Rev. Lett. 119, 180509 (2017).",
  "Qiskit Contributors, “Qiskit: An Open-source Framework for Quantum Computing,” 2024.",
];
refs.forEach((r) => body.push(numItem(r)));

// ---- assemble ---------------------------------------------------------------
const doc = new Document({
  creator: "QMET Internship",
  title: "One-Month Progress Report",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: ACCENT, font: FONT },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: "2E5C8A", font: FONT },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 280 } } } }] },
      { reference: "nums", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    ],
  },
  sections: [
    {
      properties: { page: { size: A4, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      children: titlePage,
    },
    {
      properties: { page: { size: A4, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      footers: {
        default: new Footer({
          children: [new Paragraph({ alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: "Page ", size: 18, color: "888888" }),
              new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "888888" })] })],
        }),
      },
      children: [...tocPage, ...body],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("Wrote", OUT, `(${(buf.length / 1024).toFixed(0)} KB)`);
});
