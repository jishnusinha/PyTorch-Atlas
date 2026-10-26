import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { lessons, modules, allSnippets, searchable } from "../src/content/lessons";
import { LessonSections, lessonToc } from "../src/components/LessonSections";
import type { Lesson } from "../src/content/types";
import { phase02Snippets, phase02Lessons } from "../src/content/phase02";
import { phase03Snippets, phase03Lessons } from "../src/content/phase03";
import { phase04Snippets, phase04Lessons } from "../src/content/phase04";
import { phase05Snippets, phase05Lessons } from "../src/content/phase05";
import { phase06Snippets, phase06Lessons } from "../src/content/phase06";
import { phase07Snippets, phase07Lessons } from "../src/content/phase07";
import { phase08Snippets, phase08Lessons } from "../src/content/phase08";
import { phase09Snippets, phase09Lessons } from "../src/content/phase09";
import { phase10Snippets, phase10Lessons } from "../src/content/phase10";
import { phase11Snippets, phase11Lessons } from "../src/content/phase11";
import { phase12Snippets, phase12Lessons } from "../src/content/phase12";
import { annotatedSnippetIds, codeAnnotations } from "../src/content/codeAnnotations";
import { learningChecks } from "../src/content/learningQuality";
import { comparisons } from "../src/content/comparisons";
import { recipeCategories, referenceRecipes, snippetById } from "../src/content/fieldKit";
import { recipeSearchable } from "../src/content/fieldKit";

test("the three preserved previews retain complete structures and stable IDs", () => {
  const previews = lessons.slice(0, 3);
  assert.equal(previews.length, 3);
  assert.equal(new Set(lessons.map((l) => l.id)).size, 94);
  for (const l of previews) {
    assert(l.objectives.length >= 3);
    assert(l.sections.every((s) => s.paragraphs.length >= 2));
    assert(l.shapes.length >= 3);
    assert(l.questions.length >= 3);
    assert(l.questions.every((q) => q.answer.length > 25));
    assert(l.revision.length >= 4);
    assert(l.pitfalls.length >= 4);
    assert(l.numerical.length > 100);
    assert(l.diagram.description.length > 70);
  }
});
test("ready navigation resolves; pending chapters are explicit", () => {
  const ready = modules.flatMap((m) => m.lessons).filter((l) => l.ready);
  assert.equal(ready.length, 94);
  for (const l of ready) assert(lessons.some((x) => x.id === l.id));
  assert.equal(ready.length, lessons.length);
});
test("Phase 01–04 snippets mirror their exact standalone Python files", () => {
  assert.equal(allSnippets.length, 236);
  assert.equal(phase02Snippets.length, 14);
  assert.equal(phase03Snippets.length, 8);
  assert.equal(phase04Snippets.length, 18);
  assert.equal(phase05Snippets.length, 10);
  assert.equal(phase06Snippets.length, 10);
  assert.equal(phase07Snippets.length, 15);
  assert.equal(phase08Snippets.length, 9);
  assert.equal(phase09Snippets.length, 10);
  assert.equal(phase10Snippets.length, 12);
  assert.equal(phase11Snippets.length, 12);
  assert.equal(phase12Snippets.length, 14);
  assert.equal(allSnippets.length, 236);
  assert.equal(new Set(allSnippets.map((s) => s.id)).size, 135);
  for (const s of allSnippets) {
    const folder = phase12Snippets.some((x) => x.id === s.id)
      ? "12_lstm_next_word"
      : phase11Snippets.some((x) => x.id === s.id)
      ? "11_lstm_internals"
      : phase10Snippets.some((x) => x.id === s.id)
      ? "10_faq_classifier"
      : phase09Snippets.some((x) => x.id === s.id)
      ? "09_recurrent_networks"
      : phase08Snippets.some((x) => x.id === s.id)
      ? "08_transfer_learning"
      : phase07Snippets.some((x) => x.id === s.id)
      ? "07_convolutional_networks"
      : phase06Snippets.some((x) => x.id === s.id)
      ? "06_regularization_optuna"
      : phase05Snippets.some((x) => x.id === s.id)
        ? "05_ann_gpu"
        : phase04Snippets.some((x) => x.id === s.id)
          ? "04_modules_data"
          : phase03Snippets.some((x) => x.id === s.id)
            ? "03_manual_training"
            : phase02Snippets.some((x) => x.id === s.id)
              ? "02_foundations_autograd"
              : "01_previews";
    assert.equal(
      fs.readFileSync(`python_examples/${folder}/${s.id}.py`, "utf8"),
      s.code + "\n",
    );
    assert(s.code.includes("import torch") || s.code.includes("import numpy"));
    assert(s.code.includes("assert "));
    assert(s.expected);
    assert(s.lines.length >= 5);
    assert(!s.code.includes("download("));
  }
});
test("search covers substantive text including revision and questions", () => {
  assert.equal(searchable.length, 94);
  for (const l of lessons) {
    const result = searchable.find((r) => r.id === l.id)!;
    assert(result.text.includes(l.questions[0].answer));
    assert(result.text.includes(l.snippet.code));
  }
});
test("F15 annotations resolve to unique in-range code lines and preserve snippet IDs", () => {
  assert(annotatedSnippetIds.length >= 30);
  const unique = new Map(allSnippets.map((snippet) => [snippet.id, snippet]));
  for (const id of annotatedSnippetIds) {
    const snippet = unique.get(id);
    assert(snippet, `missing annotated snippet ${id}`);
    const annotations = codeAnnotations(snippet);
    assert(annotations.length > 0);
    for (const annotation of annotations) {
      assert(annotation.startLine >= 1 && annotation.endLine >= annotation.startLine);
      assert(annotation.endLine <= snippet.code.split("\n").length);
      assert(annotation.why.length > 30 && annotation.inputShape && annotation.outputShape);
    }
  }
  for (const id of ["module-hooks", "state-dict-modes", "optimizer-lifecycle", "loss-shapes", "optuna-ann-safe", "cnn-shape-trace", "vgg-feature-freeze", "rnn-padding-pack", "gradient-clipping", "faq-checkpoint", "lstm-parameter-count", "nextword-sampling"]) assert(unique.has(id));
});
test("F15 learning checks and source comparisons point at stable lessons", () => {
  assert(learningChecks.length >= 20);
  assert.equal(new Set(learningChecks.map((item) => item.lessonId)).size, learningChecks.length);
  for (const item of learningChecks) {
    assert(lessons.some((lesson) => lesson.id === item.lessonId), item.lessonId);
    assert(allSnippets.some((snippet) => snippet.id === item.practiceSnippet), item.practiceSnippet);
    assert(item.worked.length > 50 && item.shapePath.includes("→"));
  }
  assert.equal(comparisons.length, 6);
  for (const item of comparisons) assert(lessons.some((lesson) => lesson.id === item.lessonId));
});
test("F16 Field Kit recipes are curated, typed, canonical, and lesson-linked", () => {
  assert.equal(referenceRecipes.length, 94);
  assert.equal(new Set(referenceRecipes.map((recipe) => recipe.id)).size, 94);
  assert.equal(recipeCategories.length, 9);
  assert.equal(recipeSearchable.length, referenceRecipes.length);
  for (const recipe of referenceRecipes) {
    const snippet = snippetById.get(recipe.snippetId);
    assert(snippet, `missing canonical snippet ${recipe.snippetId}`);
    assert.equal(recipe.id, recipe.snippetId);
    assert.equal(recipe.title, snippet.title);
    assert.equal(recipe.shapes, snippet.shapes);
    assert.equal(recipe.expected, snippet.expected);
    assert(recipe.whenToUse.length > 20 && recipe.why.length > 20);
    assert(recipe.dependencies && recipe.runEnvironment && recipe.pitfalls.length);
    assert(lessons.some((lesson) => lesson.id === recipe.relatedLessonId));
    assert.equal(recipe.runnable, recipe.id !== "local-csv-loader");
  }
  assert(referenceRecipes.some((recipe) => recipe.id === "optuna-synthetic"));
  assert(referenceRecipes.some((recipe) => recipe.id === "vgg-feature-freeze"));
  assert(referenceRecipes.some((recipe) => recipe.id === "fashion-shapes"));
  assert(referenceRecipes.some((recipe) => recipe.id === "faq-end-to-end"));
  assert(referenceRecipes.some((recipe) => recipe.id === "nextword-train-step"));
});
test("search includes an additional-snippet-only term in its owning lesson", () => {
  const baseText = (lesson: Lesson) =>
    [
      lesson.title,
      lesson.subtitle,
      ...lesson.objectives,
      lesson.intuition,
      lesson.equation,
      lesson.numerical,
      ...lesson.sections.flatMap((section) => [section.title, ...section.paragraphs]),
      ...lesson.pitfalls,
      ...lesson.revision,
      lesson.snippet.code,
      lesson.template.code,
      ...lesson.questions.flatMap((question) => [question.question, question.answer]),
    ]
      .join(" ")
      .toLowerCase();
  let match: { id: string; term: string } | undefined;
  for (const lesson of lessons) {
    for (const snippet of lesson.additionalSnippets ?? []) {
      const candidates = snippet.code.match(/[A-Za-z_][A-Za-z0-9_]{7,}/g) ?? [];
      const term = candidates.find(
        (candidate) =>
          !baseText(lesson).includes(candidate.toLowerCase()) &&
          !searchable.some(
            (result) =>
              result.id !== lesson.id &&
              result.text.toLowerCase().includes(candidate.toLowerCase()),
          ),
      );
      if (term) {
        match = { id: lesson.id, term };
        break;
      }
    }
    if (match) break;
  }
  assert(match, "expected a term unique to an additional Python example");
  const result = searchable.find((item) => item.id === match!.id)!;
  assert(result.text.toLowerCase().includes(match!.term.toLowerCase()));
  assert.equal(
    searchable.filter((item) =>
      item.text.toLowerCase().includes(match!.term.toLowerCase()),
    ).length,
    1,
  );
});
test("four authored sections render once in order and have matching TOC anchors", () => {
  const fixture = {
    id: "four-section-fixture",
    intuition: "A separate intuition panel.",
    sections: ["first", "second", "third", "fourth"].map((name, index) => ({
      id: index === 2 ? "visual" : `section-${name}`,
      title: `Heading ${name}`,
      paragraphs: [`Body ${name} appears once.`],
    })),
  } as unknown as Lesson;
  const html = renderToStaticMarkup(createElement(LessonSections, { lesson: fixture }));
  const headings = [...html.matchAll(/<h2[^>]*>(?:<span>\d+<\/span>)?([^<]+)/g)].map(
    (match) => match[1],
  );
  assert.deepEqual(headings, [
    "Heading first",
    "Heading second",
    "Heading third",
    "Heading fourth",
  ]);
  for (const name of ["first", "second", "third", "fourth"])
    assert.equal(html.split(`Body ${name} appears once.`).length - 1, 1);
  const ids = [...html.matchAll(/<section id="([^"]+)"/g)].map((match) => match[1]);
  const tocIds = lessonToc(fixture).map((entry) => entry.id);
  assert.deepEqual(ids, ["understand", "mechanics", "lesson-four-section-fixture-visual", "lesson-four-section-fixture-section-fourth"]);
  assert.deepEqual(tocIds.slice(0, 4), ids);
  assert.equal(new Set(tocIds).size, tocIds.length);
});
test("Phase 11 covers LSTM equations, numerical work and PyTorch API context", () => {
  assert.deepEqual(phase11Lessons.map(lesson => lesson.id), ["lstm-intuition","lstm-states","lstm-vectors","lstm-gates","lstm-blueprint","lstm-numerical","lstm-gradient-route","lstm-pytorch-shapes","lstm-parameters"]);
  for (const lesson of phase11Lessons) {
    assert(lesson.objectives.length >= 3 && lesson.shapes.length >= 3);
    assert.equal(lesson.questions.length, 4);
    assert(lesson.sections.length >= 2 && lesson.sections.every(section => section.paragraphs.length >= 2));
    assert(lesson.snippet.code.includes("import torch") && lesson.snippet.code.includes("assert "));
    assert(lesson.template.code.includes("assert "));
    assert(lesson.diagram.description.length > 50);
  }
  assert.equal(phase11Lessons.filter(lesson => lesson.source.kind === "supplemental").length, 2);
  assert(phase11Snippets.find(snippet => snippet.id === "lstm-cell-manual")!.code.includes("i_pre, f_pre, g_pre, o_pre"));
  assert(phase11Snippets.find(snippet => snippet.id === "lstm-parameter-count")!.code.includes("== 480"));
});
test("Phase 12 covers the complete local next-word project and source fixes", () => {
  assert.deepEqual(phase12Lessons.map(lesson=>lesson.id),["next-word-data","next-word-tokenization","next-word-vocabulary","next-word-prefixes","next-word-padding","next-word-architecture","next-word-training","next-word-generation","next-word-sampling","next-word-project"]);
  assert.equal(phase12Snippets.length,14);
  for(const lesson of phase12Lessons){
    assert(lesson.objectives.length>=3&&lesson.shapes.length>=3);
    assert.equal(lesson.questions.length,4);
    assert(lesson.sections.length>=2&&lesson.sections.every(section=>section.paragraphs.length>=2));
    assert(lesson.snippet.code.includes("import ")&&lesson.snippet.code.includes("assert "));
    assert(lesson.template.code.includes("assert "));
    assert(lesson.diagram.description.length>50);
  }
  assert.equal(phase12Lessons.filter(lesson=>lesson.source.kind==="supplemental").length,1);
  assert(phase12Lessons.find(lesson=>lesson.id==="next-word-vocabulary")!.sections.some(section=>section.paragraphs.some(paragraph=>paragraph.includes("PAD=0, UNK=1"))));
  assert(phase12Snippets.find(snippet=>snippet.id==="nextword-shapes")!.code.includes("[32,61,150]"));
  assert(phase12Snippets.find(snippet=>snippet.id==="nextword-cli")!.code.includes("--epochs"));
  const project=fs.readFileSync("python_examples/12_lstm_next_word/lstm_next_word.py","utf8");
  assert(project.includes("class NextWordLSTM(nn.Module)"));
  assert(project.includes("weights_only=True")&&project.includes("--corpus-file"));
  assert(project.includes("clip_grad_norm_"));
});
test("Phase 04 covers registered models, losses, optimizers and the data pipeline", () => {
  assert.deepEqual(
    phase04Lessons.map((l) => l.id),
    [
      "manual-vs-idiomatic",
      "nn-module",
      "sequential-architecture",
      "loss-conventions",
      "optimizer-lifecycle",
      "custom-dataset",
      "dataloader-pipeline",
      "collation-workers",
      "idiomatic-classifier",
      "batched-classifier",
    ],
  );
  assert.equal(phase04Snippets.length, 18);
  for (const l of phase04Lessons) {
    assert(l.objectives.length >= 3 && l.shapes.length >= 3 && l.questions.length === 4);
    assert(l.sections.length >= 2 && l.sections.every((s) => s.paragraphs.length >= 2));
    assert(l.snippet.code.includes("import torch") && l.snippet.code.includes("assert "));
    assert(l.template.code.includes("assert "));
    assert(
      ["module", "lifecycle", "loader", "collate", "pipeline"].includes(l.diagram.kind),
    );
    assert(l.diagram.description.length > 50);
  }
  assert(phase04Snippets.some((s) => s.id === "text-collate"));
  assert(phase04Snippets.some((s) => s.id === "state-dict-modes"));
  assert(phase04Snippets.some((s) => s.id === "windows-workers"));
  assert(phase04Snippets.some((s) => s.id === "manual-class-vs-module"));
  assert(phase04Snippets.some((s) => s.id === "dataset-vs-dataloader"));
});
test("Phase 05 covers Fashion-MNIST, ANN training/evaluation and device placement", () => {
  assert.deepEqual(
    phase05Lessons.map((lesson) => lesson.id),
    [
      "fashion-mnist",
      "ann-architecture",
      "multiclass-training",
      "ann-evaluation",
      "device-placement",
    ],
  );
  assert.equal(phase05Snippets.length, 10);
  for (const lesson of phase05Lessons) {
    assert(
      lesson.objectives.length >= 3 &&
        lesson.shapes.length >= 3 &&
        lesson.questions.length === 4,
    );
    assert(
      lesson.sections.length >= 2 &&
        lesson.sections.every((section) => section.paragraphs.length >= 1),
    );
    assert(
      lesson.snippet.code.includes("import torch") &&
        lesson.snippet.code.includes("assert "),
    );
    assert(lesson.template.code.includes("assert "));
    assert(["ann", "ann-data", "metrics", "device"].includes(lesson.diagram.kind));
    assert(lesson.diagram.description.length > 50);
  }
  assert(phase05Snippets.some((snippet) => snippet.id === "synthetic-ann-smoke"));
  assert(phase05Snippets.some((snippet) => snippet.id === "fashion-mnist-optional"));
});
test("Phase 06 covers regularization, optimizers and Optuna with safe splits", () => {
  assert.deepEqual(
    phase06Lessons.map((lesson) => lesson.id),
    [
      "overfitting",
      "dropout",
      "batch-normalization",
      "weight-decay",
      "regularization-comparison",
      "hyperparameter-search",
      "optuna-basics",
      "optuna-tabular",
      "optuna-samplers",
      "optuna-plots",
      "dynamic-search",
      "optuna-tracking",
      "ann-search-space",
      "dynamic-ann",
      "ann-objective",
    ],
  );
  assert.equal(phase06Snippets.length, 10);
  for (const lesson of phase06Lessons) {
    assert(
      lesson.objectives.length >= 3 &&
        lesson.shapes.length >= 3 &&
        lesson.questions.length === 4,
    );
    assert(
      lesson.sections.length >= 2 &&
        lesson.sections.every((section) => section.paragraphs.length >= 1),
    );
    assert(
      lesson.snippet.code.includes("import ") && lesson.snippet.code.includes("assert "),
    );
    assert(lesson.template.code.includes("assert "));
    assert(
      [
        "generalization",
        "dropout",
        "batchnorm",
        "weight-decay",
        "optimizer-landscape",
        "search-space",
        "trial-comparison",
      ].includes(lesson.diagram.kind),
    );
    assert(lesson.diagram.description.length > 50);
  }
  assert(phase06Snippets.some((snippet) => snippet.id === "optuna-synthetic"));
  assert(phase06Snippets.some((snippet) => snippet.id === "optuna-ann-safe"));
  assert(
    phase06Snippets
      .find((snippet) => snippet.id === "optuna-ann-safe")!
      .code.includes("Not passed into objective"),
  );
});
test("Phase 07 covers the ordered CNN math and Fashion-MNIST implementation", () => {
  assert.deepEqual(phase07Lessons.map((lesson) => lesson.id), [
    "cnn-motivation", "image-tensors", "convolution-math", "convolution-shapes",
    "convolution-channels", "cnn-activations", "pooling", "receptive-field", "cnn-classifier",
  ]);
  assert.equal(phase07Snippets.length, 15);
  for (const lesson of phase07Lessons) {
    assert(lesson.objectives.length >= 3 && lesson.shapes.length >= 3);
    assert(lesson.questions.length === 4 && lesson.revision.length >= 4);
    assert(lesson.sections.length >= 2 && lesson.sections.every((section) => section.paragraphs.length >= 2));
    assert(lesson.snippet.code.includes("import torch") && lesson.snippet.code.includes("assert "));
    assert(lesson.template.code.includes("assert "));
    assert(lesson.diagram.description.length > 50);
    assert(lesson.source.kind === "source-derived");
  }
  assert(phase07Snippets.some((snippet) => snippet.id === "manual-cross-correlation"));
  assert(phase07Snippets.some((snippet) => snippet.id === "output-size-formula"));
  assert(phase07Snippets.some((snippet) => snippet.id === "cnn-shape-trace"));
  const optional = phase07Snippets.find((snippet) => snippet.id === "fashionmnist-cnn-optional")!;
  assert(optional.code.includes("DOWNLOAD_DATA = False"));
});
test("Phase 08 covers transfer learning and VGG preprocessing in order", () => {
  assert.deepEqual(phase08Lessons.map(lesson => lesson.id), [
    "transfer-learning", "frozen-features", "vgg-preprocessing", "vgg-transform",
    "vgg-normalization", "vgg16-lab", "weights-transforms",
  ]);
  assert.equal(phase08Snippets.length, 9);
  for (const lesson of phase08Lessons) {
    assert(lesson.objectives.length >= 3 && lesson.shapes.length >= 3);
    assert.equal(lesson.questions.length, 4);
    assert(lesson.sections.length >= 2 && lesson.sections.every(section => section.paragraphs.length >= 2));
    assert(lesson.snippet.code.includes("import ") && lesson.snippet.code.includes("assert "));
    assert(lesson.template.code.includes("assert "));
    assert(lesson.diagram.description.length > 50);
  }
  const optional = phase08Snippets.find(snippet => snippet.id === "vgg16-transfer-optional")!;
  assert(optional.code.includes("ALLOW_DOWNLOADS = False"));
  assert(optional.code.includes("weights.transforms()"));
  assert(optional.code.includes("val_loader") && optional.code.includes("test_loader"));
});
test("Phase 09 covers sequence motivation, recurrent math, output patterns and BPTT", () => {
  assert.deepEqual(phase09Lessons.map(lesson => lesson.id), [
    "rnn-motivation", "rnn-use-cases", "rnn-shapes", "recurrent-cell", "rnn-variants", "bptt", "gradient-pathologies",
  ]);
  assert.equal(phase09Snippets.length, 10);
  for (const lesson of phase09Lessons) {
    assert(lesson.objectives.length >= 3 && lesson.shapes.length >= 3);
    assert.equal(lesson.questions.length, 4);
    assert(lesson.sections.length >= 2 && lesson.sections.every(section => section.paragraphs.length >= 2));
    assert(lesson.snippet.code.includes("import torch") && lesson.snippet.code.includes("assert "));
    assert(lesson.template.code.includes("assert "));
    assert(lesson.diagram.description.length > 50);
    assert(lesson.source.kind === "source-derived");
  }
  assert(phase09Snippets.some(snippet => snippet.id === "rnn-parameter-count"));
  assert(phase09Snippets.some(snippet => snippet.id === "rnn-padding-pack"));
  assert(phase09Snippets.some(snippet => snippet.id === "bptt-shared-gradients"));
  assert(phase09Snippets.some(snippet => snippet.id === "gradient-clipping"));
});
test("Phase 10 covers the FAQ classifier, robust text pipeline and honest abstention", () => {
  assert.deepEqual(phase10Lessons.map(lesson => lesson.id), [
    "faq-classifier", "faq-vocabulary", "faq-vocabulary-map", "faq-data", "faq-architecture", "faq-training", "faq-inference",
  ]);
  assert.equal(phase10Snippets.length, 12);
  for (const lesson of phase10Lessons) {
    assert(lesson.objectives.length >= 3 && lesson.shapes.length >= 3);
    assert.equal(lesson.questions.length, 4);
    assert(lesson.sections.length >= 2 && lesson.sections.every(section => section.paragraphs.length >= 2));
    assert(lesson.snippet.code.includes("import ") && lesson.snippet.code.includes("assert "));
    assert(lesson.template.code.includes("assert "));
    assert(lesson.diagram.description.length > 50);
    assert(["source-derived", "supplemental"].includes(lesson.source.kind));
  }
  assert.equal(phase10Lessons.find(lesson => lesson.id === "faq-vocabulary-map")!.source.kind, "supplemental");
  assert(phase10Snippets.find(snippet => snippet.id === "faq-end-to-end")!.code.includes("pack_padded_sequence"));
  assert(phase10Snippets.find(snippet => snippet.id === "faq-end-to-end")!.code.includes("weights_only=True"));
  assert(phase10Snippets.find(snippet => snippet.id === "faq-source-target")!.code.includes("answer_tokens[0]"));
  assert(phase10Lessons.find(lesson => lesson.id === "faq-classifier")!.source.note?.includes("supplemental"));
});
test("Phase 03 lessons cover the manual binary-classification sequence", () => {
  assert.deepEqual(
    phase03Lessons.map((l) => l.id),
    [
      "binary-classification",
      "preprocessing",
      "manual-forward",
      "manual-loss",
      "manual-training-loop",
      "binary-evaluation",
    ],
  );
  for (const l of phase03Lessons) {
    assert(l.objectives.length >= 3 && l.shapes.length >= 3 && l.questions.length === 4);
    assert(l.sections.every((s) => s.paragraphs.length >= 2));
    assert(l.snippet.code.includes("import torch") && l.snippet.code.includes("assert "));
    assert(l.diagram.kind === "pipeline" && l.diagram.description.length > 40);
  }
  assert(
    phase03Lessons.find((l) => l.id === "manual-training-loop")!.additionalSnippets
      ?.length,
  );
});
test("Phase 02 lessons are complete, ordered and use accessible authored diagrams", () => {
  assert.deepEqual(
    phase02Lessons.map((l) => l.id),
    [
      "tensor-prerequisites",
      "derivatives-chain-rule",
      "computation-graph",
      "scalar-autograd",
      "vector-gradients",
      "gradient-contexts",
    ],
  );
  for (const l of phase02Lessons) {
    assert(l.objectives.length >= 3 && l.shapes.length >= 3 && l.questions.length === 4);
    assert(l.sections.every((s) => s.paragraphs.length >= 2));
    assert(l.snippet.code.includes("assert ") && l.template.code.includes("assert "));
    assert(l.diagram.description.length > 30);
  }
});
test("bundled content excludes source chrome, links and time references", () => {
  const text = JSON.stringify({ lessons, modules, searchable });
  assert(!/youtube|gemini|\b\d{1,2}:\d{2}(?::\d{2})?\b|&t=/i.test(text));
});
test("final source index has reviewed lesson links and records corrections", () => {
  const manifest=JSON.parse(fs.readFileSync("docs/source-coverage.json","utf8"));
  assert.equal(manifest.phase,12);
  assert.equal(manifest.corrections.length,9);
  for(const block of manifest.blocks) for(const topic of block.topics){
    assert.equal(topic.status,"reviewed",`${topic.lesson} remains unreviewed`);
    assert(lessons.some(lesson=>lesson.id===topic.lesson),`${topic.lesson} has no authored lesson`);
  }
  assert(manifest.supplements.every((item:any)=>item.status==="reviewed"));
  for(const item of manifest.supplements) assert(lessons.some(lesson=>lesson.id===item.lesson),`${item.lesson} supplemental entry has no authored lesson`);
  assert(manifest.supplements.some((item:any)=>item.topic.includes("PAD/UNK/EOS")));
});
test("all lesson and source records have valid fields and registry links", () => {
  assert.equal(new Set(lessons.map((lesson) => lesson.id)).size, lessons.length);
  assert.equal(lessons.length, 94);
  const manifest = JSON.parse(fs.readFileSync("docs/source-coverage.json", "utf8"));
  for (const lesson of lessons) {
    assert(/^[a-z0-9-]+$/.test(lesson.id));
    assert(lesson.objectives.length > 0, `${lesson.id} has no learning objective`);
    assert(lesson.sections.length > 0, `${lesson.id} has no authored sections`);
    assert.equal(
      new Set(lesson.sections.map((section) => section.id)).size,
      lesson.sections.length,
      `${lesson.id} has duplicate section IDs`,
    );
    for (const section of lesson.sections) {
      assert(section.title.trim(), `${lesson.id} has a blank section title`);
      assert(section.paragraphs.length > 0, `${lesson.id}/${section.id} has no body`);
    }
    assert(lesson.source.start >= 1 && lesson.source.end <= 211);
    assert(lesson.source.start <= lesson.source.end);
    for (const snippet of [lesson.snippet, lesson.template, ...(lesson.additionalSnippets ?? [])]) {
      assert(snippet.id && allSnippets.some((item) => item.id === snippet.id));
    }
  }
  for (const block of manifest.blocks) {
    assert(block.start >= 1 && block.end <= 211 && block.start <= block.end);
    for (const topic of block.topics) {
      assert(lessons.some((lesson) => lesson.id === topic.lesson), `${topic.lesson} is missing`);
      assert(topic.start >= 1 && topic.end <= 211 && topic.start <= topic.end);
      assert(topic.start <= block.end && topic.end >= block.start);
    }
  }
  for (const item of manifest.supplements)
    assert(lessons.some((lesson) => lesson.id === item.lesson), `${item.lesson} is missing`);
  for (const correction of manifest.corrections) {
    const pages = [...correction.pages.matchAll(/\d+/g)].map((match) => Number(match[0]));
    assert(pages.length > 0 && pages.every((page) => page >= 1 && page <= 211));
  }
});
test("the shipped reader contains no obsolete preview or future-phase copy", () => {
  const app = fs.readFileSync("src/app/App.tsx", "utf8");
  const search = fs.readFileSync("src/components/Search.tsx", "utf8");
  const lessonsSource = fs.readFileSync("src/content/lessons.ts", "utf8");
  const releaseCopy = `${app}\n${search}\n${lessonsSource}`;
  assert(!/three authored previews|more chapters arrive phase by phase|pending chapters are not included/i.test(releaseCopy));
  assert(!/PHASE 01|chapters remain planned|remain planned/i.test(releaseCopy));
});
test("standalone package has no external asset links or runtime fetch", () => {
  const html = fs.readFileSync("dist/PyTorch-Atlas.html", "utf8");
  assert(html.length > 100000);
  assert(!/<(?:script|link|img)[^>]+(?:src|href)=["'](?:https?:|\/assets)/i.test(html));
  assert(!/\bfetch\(/.test(html));
  assert(!/\.pdf(?:["'])/i.test(html));
});
