/*
 * Easy-to-understand project report (.docx), plain language, with the live-app
 * screenshot and deployment details.
 *   node docs/generate_simple_report.js   ->  docs/Project_Report.docx
 */
const fs = require("fs");
const path = require("path");
const GLOBAL = "C:/Users/aayus/AppData/Roaming/npm/node_modules";
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, ImageRun, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageNumber, PageBreak,
} = require(path.join(GLOBAL, "docx"));

const ROOT = path.join(__dirname, "..");
const RES = path.join(ROOT, "results");
const OUT = path.join(__dirname, "Project_Report.docx");

const FONT = "Calibri";
const ACCENT = "1F4E79";
const A4 = { width: 11906, height: 16838 };
const CW = 9026;

function P(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 130, line: 288 },
    alignment: opts.align,
    children: Array.isArray(text) ? text
      : [new TextRun({ text, bold: opts.bold, italics: opts.italics, size: opts.size, color: opts.color })],
  });
}
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
function bullet(text) {
  return new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 70 },
    children: Array.isArray(text) ? text : [new TextRun(text)] });
}
const AR = { "fig_pipeline.png": 2.668, "fig_signals.png": 2.622, "ui_app.png": 0.967 };
function img(file, wPx) {
  const w = wPx || 560, h = Math.round(w / (AR[file] || 1.7));
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 90, after: 50 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(RES, file)),
      transformation: { width: w, height: h }, altText: { title: file, description: file, name: file } })] });
}
function caption(t) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 170 },
    children: [new TextRun({ text: t, italics: true, size: 18, color: "555555" })] });
}
const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, w, o = {}) {
  return new TableCell({ borders, width: { size: w, type: WidthType.DXA },
    shading: o.fill ? { fill: o.fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({ alignment: o.align,
      children: [new TextRun({ text: String(text), bold: o.bold, color: o.color, size: 21 })] })] });
}
function table(headers, rows, widths) {
  const head = new TableRow({ tableHeader: true,
    children: headers.map((h, i) => cell(h, widths[i], { bold: true, fill: ACCENT, color: "FFFFFF", align: AlignmentType.CENTER })) });
  const body = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => cell(c, widths[i], { fill: ri % 2 ? "F2F6FB" : undefined, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) }));
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] });
}

// ---- title page ----
const title = [
  new Paragraph({ spacing: { before: 800, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "INDIAN INSTITUTE OF TECHNOLOGY PATNA", bold: true, size: 26, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 40 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "QMET PARIMANA UG Summer Internship 2026", size: 22, color: "444444" })] }),
  new Paragraph({ spacing: { before: 80, after: 700 }, alignment: AlignmentType.CENTER,
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 6 } }, children: [new TextRun("")] }),
  new Paragraph({ spacing: { before: 300, after: 80 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Quantum Sensor Fault Classifier", bold: true, size: 40, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 500 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "A Simple Project Report", italics: true, size: 26, color: "555555" })] }),
  new Paragraph({ spacing: { before: 400, after: 30 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Aaditya Sah", bold: true, size: 26 })] }),
  new Paragraph({ spacing: { after: 30 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "aaditya_2312res02@iitp.ac.in", size: 20, color: "555555" })] }),
  new Paragraph({ spacing: { before: 260, after: 30 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Under the supervision of", size: 20, color: "777777" })] }),
  new Paragraph({ spacing: { after: 20 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Dr. Nutan Kumar Tomar", bold: true, size: 24 })] }),
  new Paragraph({ spacing: { after: 30 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "nktomar@iitp.ac.in", size: 20, color: "555555" })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

const body = [];

// 1
body.push(H1("1. What This Project Does (In Plain English)"));
body.push(P("Machines have parts called bearings that let things spin smoothly. When a bearing starts to wear out, the machine vibrates differently, a bit like how a washing machine sounds wrong when something is stuck inside. If you can notice that change early, you can repair the machine before it breaks down."));
body.push(P("This project listens to that vibration and decides whether the machine is healthy (normal) or damaged (fault). The interesting part is how it decides: it uses a quantum computer's way of processing information, a field called quantum machine learning."));
body.push(P([new TextRun({ text: "In one line: ", bold: true }), new TextRun("a system that reads a machine's vibration and warns you if the machine is about to fail, using a small quantum model.")]));

// 2
body.push(H1("2. Classical vs Quantum (the only theory you need)"));
body.push(P("A normal computer stores information in bits. Each bit is either 0 or 1, like a light switch that is off or on."));
body.push(P("A quantum computer uses qubits. A qubit can be a blend of 0 and 1 at the same time, like a spinning coin that is part heads and part tails until it lands. This lets a quantum model represent data in a richer way. This project uses four qubits."));

// 3
body.push(H1("3. How It Works"));
body.push(P("The signal travels through a short pipeline and comes out as an answer:"));
body.push(img("fig_pipeline.png"));
body.push(caption("Figure 1. The full pipeline, from a raw vibration signal to a normal or fault decision."));
body.push(bullet([new TextRun({ text: "Read the signal. ", bold: true }), new TextRun("The vibration is measured as 8 numbers over a tiny time window.")]));
body.push(bullet([new TextRun({ text: "Shrink it. ", bold: true }), new TextRun("A standard method (PCA) keeps only the 4 most useful numbers, one for each qubit, and throws away the rest.")]));
body.push(bullet([new TextRun({ text: "Load into qubits. ", bold: true }), new TextRun("Those 4 numbers set how the 4 qubits behave.")]));
body.push(bullet([new TextRun({ text: "Run the trained circuit. ", bold: true }), new TextRun("A quantum circuit with 48 adjustable settings transforms the qubits. These 48 settings are the model's learned knowledge.")]));
body.push(bullet([new TextRun({ text: "Read the answer. ", bold: true }), new TextRun("Measuring the qubits gives a number: negative means normal, positive means fault.")]));
body.push(P("The two signal types it learns to tell apart look like this:"));
body.push(img("fig_signals.png"));
body.push(caption("Figure 2. A healthy signal (left) and a faulty signal (right). The faulty one vibrates faster and has extra wobble."));
body.push(P([new TextRun({ text: "Training (done once): ", bold: true }), new TextRun("the model was shown 168 example signals that were already labelled healthy or faulty, and an algorithm slowly adjusted the 48 settings until it got them right. The trained model is then saved and simply reused, which is why each new prediction is fast (about 50 milliseconds).")]));

// 4
body.push(H1("4. Results"));
body.push(P("The models were tested on signals they had never seen before. Three approaches were compared (all four-qubit models on the same task):"));
body.push(table(
  ["Method", "What it is", "Accuracy"],
  [["Classical (SVM)", "ordinary, non-quantum", "100%"],
   ["Quantum kernel (QSVC)", "quantum-assisted", "98.6%"],
   ["Quantum circuit (VQC)", "the model in the live app", "95.8%"]],
  [2900, 3900, 2226]));
body.push(caption("Table 1. Accuracy on unseen test signals."));
body.push(P([new TextRun({ text: "Honest takeaway: ", bold: true }), new TextRun("the ordinary classical method actually scores highest here, and that is completely fine. The goal of the project is not to prove quantum beats everything. It is to show that a quantum model can genuinely learn this task, and it does, reaching about 96%.")]));
body.push(P([new TextRun({ text: "Surviving noise: ", bold: true }), new TextRun("real quantum computers make random errors, called noise. The project also tested how the model holds up as noise increases. It stays strong up to a point, then drops, and a correction trick (Zero-Noise Extrapolation) recovers part of the lost accuracy. This shows the model could work on real, imperfect hardware.")]));

// 5
body.push(H1("5. The Live Application"));
body.push(P("The whole project is wrapped into a real, usable web service. You open a page in the browser, pick or generate a signal, and get an instant answer."));
body.push(img("ui_app.png", 360));
body.push(caption("Figure 3. The live web app classifying a faulty signal (running on the quantum simulator)."));
body.push(H2("How to use it"));
body.push(bullet("Click Normal sample or Fault sample to load a signal (you can also paste your own 8 numbers)."));
body.push(bullet("Choose the model: VQC (quantum) or Classical (fast)."));
body.push(bullet("Click Classify. You get a coloured verdict (green NORMAL or red FAULT), a confidence bar, and the quantum output value."));
body.push(H2("How it is deployed"));
body.push(P("Under the hood it is a web API built with FastAPI, so other software can also send a signal and get a prediction automatically. It can be packaged with Docker and run on any server. To start it locally:"));
body.push(new Paragraph({ spacing: { after: 80 }, shading: { fill: "F3F4F6", type: ShadingType.CLEAR },
  children: [new TextRun({ text: "python -m qsensor.training      (once, trains and saves the model)", font: "Consolas", size: 19 })] }));
body.push(new Paragraph({ spacing: { after: 130 }, shading: { fill: "F3F4F6", type: ShadingType.CLEAR },
  children: [new TextRun({ text: "uvicorn qsensor.api:app         (starts the web app at localhost:8000)", font: "Consolas", size: 19 })] }));

// 6
body.push(H1("6. Challenges I Solved"));
body.push(bullet([new TextRun({ text: "Too many qubits failed. ", bold: true }), new TextRun("An early 8-qubit design only reached about 52% (no better than guessing) because the model could not be trained, a known problem called barren plateaus. Dropping to 4 qubits fixed it.")]));
body.push(bullet([new TextRun({ text: "Wrong number range. ", bold: true }), new TextRun("Feeding the data in the wrong numeric range scrambled it inside the quantum circuit. Rescaling the numbers to the range 0 to 1 was the single biggest improvement.")]));
body.push(bullet([new TextRun({ text: "Repeatable results. ", bold: true }), new TextRun("Quantum simulations can give slightly different answers each run. Pinning the computation to a single thread made the results identical every time.")]));

// 7
body.push(H1("7. What's Next"));
body.push(bullet("Train on real recorded sensor data instead of synthetic signals."));
body.push(bullet("Try the model on a real IBM quantum computer, not just a simulator."));
body.push(bullet("Add more thorough testing (multiple runs, averages and error bars)."));

// 8
body.push(H1("8. Summary"));
body.push(P("I built a quantum machine-learning system that detects machine faults from vibration signals. I first tried a large 8-qubit design that failed, redesigned it into a working 4-qubit model that reaches about 96%, studied how it survives hardware noise, and deployed it as a web service that anyone can use."));

// assemble
const doc = new Document({
  creator: "Aaditya Sah", title: "Quantum Sensor Fault Classifier - Simple Report",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: ACCENT, font: FONT }, paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: "2E5C8A", font: FONT }, paragraph: { spacing: { before: 170, after: 90 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 280 } } } }] }] },
  sections: [
    { properties: { page: { size: A4, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } }, children: title },
    { properties: { page: { size: A4, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      footers: { default: new (require(path.join(GLOBAL, "docx")).Footer)({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Page ", size: 18, color: "888888" }), new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "888888" })] })] }) },
      children: body },
  ],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("Wrote", OUT, `(${(buf.length / 1024).toFixed(0)} KB)`); });
