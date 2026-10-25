import fs from "node:fs/promises";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";
import { createCanvas } from "@napi-rs/canvas";
await fs.mkdir("tmp/pdf-review", { recursive: true });
async function render(file, pages, prefix) {
  const pdf = await getDocument({
    data: new Uint8Array(await fs.readFile(file)),
    useSystemFonts: true,
    isEvalSupported: false,
  }).promise;
  for (const n of pages) {
    const page = await pdf.getPage(n);
    const viewport = page.getViewport({ scale: 1.4 });
    const canvas = createCanvas(Math.ceil(viewport.width), Math.ceil(viewport.height));
    await page.render({
      canvas,
      canvasContext: canvas.getContext("2d"),
      viewport,
    }).promise;
    await fs.writeFile(`tmp/pdf-review/${prefix}-${n}.png`, canvas.toBuffer("image/png"));
  }
  console.log(`Rendered ${prefix}: ${pages.join(", ")} (${pdf.numPages} total pages).`);
}
await render("PYTORCH Guide - Google Gemini.pdf", [3, 8, 13, 28, 104], "source");
await render("tmp/print-1440.pdf", [1, 2, 3], "print");
