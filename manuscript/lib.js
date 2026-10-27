// Layout helpers for an IEEE Access-style manuscript (two-column body, Letter paper).
const fs = require("fs");
const d = require("docx");
const {
  Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle, AlignmentType,
  ShadingType, TabStopType, Tab, HeadingLevel, VerticalAlign, HighlightColor,
  MathRun, MathSubScript, MathSuperScript, MathSubSuperScript, MathFraction, MathSum,
} = d;

const FONT = "Times New Roman", SANS = "Arial", BLUE = "00629B";
const PAGE_W = 12240, MARGIN_LR = 900, COL_GAP = 360;
const FULL_W = PAGE_W - 2 * MARGIN_LR;          // 10440 dxa
const COL_W = (FULL_W - COL_GAP) / 2;           // 5040 dxa = 3.5 in

// ---------------------------------------------------------------- inline markup
// **bold**  *italic*  _{sub}  ^{sup}  [TBD ...] / [VERIFY ...] highlighted
const TOKEN = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]*\}|\^\{[^}]*\}|\[(?:TBD|VERIFY)[^\]]*\])/;
function runs(text, base = {}) {
  const out = [];
  for (const part of text.split(TOKEN)) {
    if (!part) continue;
    const o = { font: FONT, ...base };
    if (part.startsWith("**")) out.push(new TextRun({ ...o, text: part.slice(2, -2), bold: true }));
    else if (part.startsWith("*")) out.push(new TextRun({ ...o, text: part.slice(1, -1), italics: true }));
    else if (part.startsWith("_{")) {
      const t = part.slice(2, -1);
      out.push(new TextRun({ ...o, text: t, subScript: true, italics: /^[a-zA-Z]{1,2}$/.test(t) && t !== "pool" }));
    } else if (part.startsWith("^{")) out.push(new TextRun({ ...o, text: part.slice(2, -1), superScript: true }));
    else if (part.startsWith("[TBD") || part.startsWith("[VERIFY"))
      out.push(new TextRun({ ...o, text: part, highlight: HighlightColor.YELLOW }));
    else out.push(new TextRun({ ...o, text: part }));
  }
  return out;
}

const body = (text, opts = {}) => new Paragraph({
  children: runs(text, { size: 20 }), alignment: AlignmentType.JUSTIFIED,
  indent: opts.noIndent ? undefined : { firstLine: 202 }, spacing: { after: 0, line: 240 }, ...opts.p,
});
const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1, spacing: { before: 200, after: 80 }, keepNext: true,
  children: [new TextRun({ text, font: SANS, size: 19, bold: true, color: BLUE, allCaps: true })],
});
const h2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing: { before: 120, after: 60 }, keepNext: true,
  children: [new TextRun({ text, font: SANS, size: 19, italics: true, color: BLUE })],
});
// numbered or bulleted item rendered with a hanging indent (no auto-numbering to keep it editable)
const item = (label, text) => new Paragraph({
  children: [new TextRun({ text: label, font: FONT, size: 20 }), new TextRun({ children: [new Tab()] }),
    ...runs(text, { size: 20 })],
  alignment: AlignmentType.JUSTIFIED, indent: { left: 300, hanging: 300 },
  tabStops: [{ type: TabStopType.LEFT, position: 300 }], spacing: { after: 30, line: 240 },
});

// ---------------------------------------------------------------- math
const mr = (t) => new MathRun(t);
const sub = (b, s) => new MathSubScript({ children: [mr(b)], subScript: [mr(s)] });
const sup = (b, s) => new MathSuperScript({ children: [mr(b)], superScript: [mr(s)] });
const subsup = (b, s, p) => new MathSubSuperScript({ children: [mr(b)], subScript: [mr(s)], superScript: [mr(p)] });
const frac = (n, dd) => new MathFraction({ numerator: n, denominator: dd });
const sum = (lower, children) => new MathSum({ children, subScript: lower ? [mr(lower)] : undefined });
function equation(children, number, width = COL_W) {
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(width / 2) },
      { type: TabStopType.RIGHT, position: width }],
    spacing: { before: 80, after: 80 },
    children: [new TextRun({ children: [new Tab()] }), new d.Math({ children }),
      new TextRun({ font: FONT, size: 20, children: [new Tab(), `(${number})`] })],
  });
}

// ---------------------------------------------------------------- figures & captions
const sizes = JSON.parse(fs.readFileSync("figsizes.json"));
function figure(file, widthIn, number, caption) {
  const [w, h] = sizes[file];
  const px = Math.round(widthIn * 96);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(`figures/${file}`),
        transformation: { width: px, height: Math.round(px * h / w) },
        altText: { title: `Fig. ${number}`, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 },
      children: [new TextRun({ text: `FIGURE ${number}. `, font: SANS, size: 16, bold: true, color: BLUE }),
        ...runs(caption, { size: 16 })] }),
  ];
}
function tableCaption(roman, title) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 0 }, keepNext: true,
      children: [new TextRun({ text: `TABLE ${roman}`, font: SANS, size: 16, bold: true, color: BLUE })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true,
      children: runs(title, { size: 16 }) }),
  ];
}
const tableNote = (text) => new Paragraph({ spacing: { before: 40, after: 160 }, alignment: AlignmentType.JUSTIFIED,
  children: runs(text, { size: 14 }) });

// ---------------------------------------------------------------- tables (IEEE rules: top, header, bottom)
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const RULE = (sz) => ({ style: BorderStyle.SINGLE, size: sz, color: "000000" });
function table(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const fs_ = opts.size || 15;
  const cell = (text, w, { bold = false, top = NONE, bottom = NONE, align, span, shade } = {}) =>
    new TableCell({
      width: { size: w, type: WidthType.DXA }, columnSpan: span, verticalAlign: VerticalAlign.CENTER,
      borders: { top, bottom, left: NONE, right: NONE },
      shading: shade ? { type: ShadingType.CLEAR, fill: shade, color: "auto" } : undefined,
      margins: { top: 30, bottom: 30, left: 60, right: 60 },
      children: [new Paragraph({ alignment: align || AlignmentType.LEFT, spacing: { after: 0 },
        children: runs(String(text), { size: fs_, bold }) })],
    });
  const trs = [new TableRow({ tableHeader: true, children: header.map((h, i) =>
    cell(h, widths[i], { bold: true, top: RULE(8), bottom: RULE(4),
      align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) })];
  rows.forEach((r, k) => {
    const last = k === rows.length - 1;
    if (!Array.isArray(r)) {     // group row spanning all columns
      trs.push(new TableRow({ children: [cell(r.group, total, { bold: true, span: widths.length,
        shade: "EEF3F8", bottom: last ? RULE(8) : NONE })] }));
      return;
    }
    trs.push(new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, widths[i], {
      bottom: last ? RULE(8) : (opts.ruleAfter && opts.ruleAfter.includes(k) ? RULE(2) : NONE),
      align: i < (opts.leftCols || 1) ? AlignmentType.LEFT : AlignmentType.CENTER })) }));
  });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: trs,
    alignment: AlignmentType.CENTER, borders: { top: NONE, bottom: NONE, left: NONE, right: NONE,
      insideHorizontal: NONE, insideVertical: NONE } });
}

module.exports = { d, FONT, SANS, BLUE, PAGE_W, MARGIN_LR, COL_GAP, FULL_W, COL_W, runs, body, h1, h2, item,
  mr, sub, sup, subsup, frac, sum, equation, figure, tableCaption, tableNote, table };
