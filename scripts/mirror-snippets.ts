import fs from "node:fs/promises";
import { allSnippets } from "../src/content/lessons";
import { phase02Snippets } from "../src/content/phase02";
import { phase03Snippets } from "../src/content/phase03";
import { phase04Snippets } from "../src/content/phase04";
import { phase05Snippets } from "../src/content/phase05";
import { phase06Snippets } from "../src/content/phase06";
import { phase07Snippets } from "../src/content/phase07";
import { phase08Snippets } from "../src/content/phase08";
import { phase09Snippets } from "../src/content/phase09";
import { phase10Snippets } from "../src/content/phase10";
import { phase11Snippets } from "../src/content/phase11";
import { phase12Snippets } from "../src/content/phase12";
import fsSync from "node:fs";
await fs.mkdir("python_examples/01_previews", { recursive: true });
await fs.mkdir("python_examples/02_foundations_autograd", { recursive: true });
await fs.mkdir("python_examples/03_manual_training", { recursive: true });
await fs.mkdir("python_examples/04_modules_data", { recursive: true });
await fs.mkdir("python_examples/05_ann_gpu", { recursive: true });
await fs.mkdir("python_examples/06_regularization_optuna", { recursive: true });
await fs.mkdir("python_examples/07_convolutional_networks", { recursive: true });
await fs.mkdir("python_examples/08_transfer_learning", { recursive: true });
await fs.mkdir("python_examples/09_recurrent_networks", { recursive: true });
await fs.mkdir("python_examples/10_faq_classifier", { recursive: true });
await fs.mkdir("python_examples/11_lstm_internals", { recursive: true });
await fs.mkdir("python_examples/12_lstm_next_word", { recursive: true });
for (const snippet of allSnippets.slice(0, 6))
  await fs.writeFile(`python_examples/01_previews/${snippet.id}.py`, snippet.code + "\n");
for (const snippet of phase02Snippets)
  await fs.writeFile(
    `python_examples/02_foundations_autograd/${snippet.id}.py`,
    snippet.code + "\n",
  );
for (const snippet of phase03Snippets)
  await fs.writeFile(
    `python_examples/03_manual_training/${snippet.id}.py`,
    snippet.code + "\n",
  );
for (const snippet of phase04Snippets)
  await fs.writeFile(
    `python_examples/04_modules_data/${snippet.id}.py`,
    snippet.code + "\n",
  );
for (const snippet of phase05Snippets)
  await fs.writeFile(`python_examples/05_ann_gpu/${snippet.id}.py`, snippet.code + "\n");
for (const snippet of phase06Snippets)
  await fs.writeFile(
    `python_examples/06_regularization_optuna/${snippet.id}.py`,
    snippet.code + "\n",
  );
for (const snippet of phase07Snippets)
  await fs.writeFile(
    `python_examples/07_convolutional_networks/${snippet.id}.py`,
    snippet.code + "\n",
  );
for (const snippet of phase08Snippets)
  await fs.writeFile(
    `python_examples/08_transfer_learning/${snippet.id}.py`,
    snippet.code + "\n",
  );
for (const snippet of phase09Snippets)
  await fs.writeFile(`python_examples/09_recurrent_networks/${snippet.id}.py`, snippet.code + "\n");
for (const snippet of phase10Snippets)
  await fs.writeFile(`python_examples/10_faq_classifier/${snippet.id}.py`, snippet.code + "\n");
for (const snippet of phase11Snippets)
  await fs.writeFile(`python_examples/11_lstm_internals/${snippet.id}.py`, snippet.code + "\n");
for (const snippet of phase12Snippets)
  await fs.writeFile(`python_examples/12_lstm_next_word/${snippet.id}.py`, snippet.code + "\n");
console.log(
  `Mirrored ${allSnippets.length} lesson slots (${phase02Snippets.length} Phase 02, ${phase03Snippets.length} Phase 03, ${phase04Snippets.length} Phase 04, ${phase05Snippets.length} Phase 05, ${phase06Snippets.length} Phase 06, ${phase07Snippets.length} Phase 07, ${phase08Snippets.length} Phase 08, ${phase09Snippets.length} Phase 09, ${phase10Snippets.length} Phase 10, ${phase11Snippets.length} Phase 11, ${phase12Snippets.length} Phase 12 snippets).`,
);
