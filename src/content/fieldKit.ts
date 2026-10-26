import { allSnippets, lessons } from "./lessons";
import type { RecipeCategory, ReferenceRecipe } from "./types";

// One curated entry per stable code example. Category lists are intentionally
// explicit so future lesson additions do not silently inflate or reshape the kit.
const selections: [RecipeCategory, string[]][] = [
  ["Tensors", ["tensor-shapes", "tensor-shape-template", "tensor-broadcasting", "broadcasting-template", "image-tensor-shapes", "cnn-transform", "fashion-shapes", "tensor-dataset"]],
  ["Autograd", ["scalar-gradient", "derivative-check", "autograd-numeric-check", "autograd-multi-input-retain", "autograd-scalar-functions", "autograd-vector-reduction", "autograd-vector-template", "autograd-accumulate-clear", "autograd-disabled-contexts", "autograd-evaluation-template"]],
  ["nn.Module", ["manual-forward", "manual-class-vs-module", "module-registration", "module-hooks", "sequential-summary", "ann-architecture", "synthetic-ann-smoke", "cnn-model"]],
  ["Dataset/Loader", ["first-batches", "batch-factory", "local-csv-loader", "custom-dataset", "dataset-vs-dataloader", "dataloader-pipeline", "loader-options", "text-collate", "windows-workers", "faq-dataset"]],
  ["Losses/Metrics", ["manual-bce", "binary-evaluation", "loss-shapes", "loss-error-repair", "cross-entropy-contract", "metrics-confusion", "epoch-metrics", "faq-loss-targets", "faq-source-target"]],
  ["Training/Optimizers", ["manual-step", "manual-loop", "full-manual-classifier", "manual-one-step", "gradient-accumulation-demo", "manual-vs-idiomatic", "optimizer-lifecycle", "batched-classifier", "seed-training-curves", "optimizer-comparison", "regularized-ann", "gradient-clipping", "optuna-synthetic", "optuna-ann-safe"]],
  ["CNN/Transfer", ["cnn-locality", "manual-cross-correlation", "conv2d-basics", "output-size-formula", "multichannel-conv", "cnn-activations", "pooling-ops", "receptive-field-calc", "cnn-shape-trace", "cnn-smoke-train", "transfer-learning-toy", "vgg-feature-freeze"]],
  ["RNN/LSTM/Text", ["rnn-batch-shape", "rnn-padding-pack", "rnn-many-to-one", "rnn-many-to-many", "rnn-cell-manual", "bptt-shared-gradients", "rnn-train-eval", "faq-tokenize", "faq-vocab", "faq-collate", "faq-model", "faq-inference", "faq-end-to-end", "lstm-gate-numerical", "lstm-classifier-smoke", "nextword-prefix-pairs", "nextword-train-step"]],
  ["Checkpoints/Debugging", ["state-dict-modes", "checkpoint-seed", "cnn-shape-debug", "vgg-shape-debug", "faq-checkpoint", "nextword-validation"]],
];
const uniqueSnippets = new Map(allSnippets.map((snippet) => [snippet.id, snippet]));
const contextualIds = new Set(["local-csv-loader"]);
const advancedIds = new Set([
  "autograd-disabled-contexts", "module-hooks",
  "windows-workers", "optuna-synthetic", "optuna-ann-safe", "gradient-clipping",
  "vgg-feature-freeze", "bptt-shared-gradients", "faq-inference", "faq-end-to-end",
  "nextword-train-step", "nextword-validation",
]);
const projectBySnippet: Record<string, string[]> = {
  "full-manual-classifier": ["BS02"],
  "idiomatic-classifier": ["BS02"],
  "synthetic-ann-smoke": ["BS03"],
  "fashion-mnist-optional": ["BS04"],
  "cnn-smoke-train": ["BS05"],
  "vgg-feature-freeze": ["BS06"],
  "faq-end-to-end": ["BS08"],
  "nextword-train-step": ["BS10"],
};
const purposeBySnippet: Record<string, string> = {
  "rnn-many-to-many": "Use this pattern when every sequence step needs a prediction rather than one label for the whole sequence.",
};

function findLesson(snippetId: string) {
  return lessons.find((lesson) =>
    [lesson.snippet, lesson.template, ...(lesson.additionalSnippets ?? [])].some(
      (snippet) => snippet.id === snippetId,
    ),
  );
}
function tagsFor(title: string, id: string, category: RecipeCategory) {
  const words = `${id.replaceAll("-", " ")} ${title}`
    .toLowerCase()
    .match(/[a-z][a-z0-9]{2,}/g) ?? [];
  return [...new Set([category.toLowerCase(), ...words])].slice(0, 7);
}

export const referenceRecipes: ReferenceRecipe[] = selections.flatMap(
  ([category, ids]) =>
    ids.map((snippetId): ReferenceRecipe => {
      const snippet = uniqueSnippets.get(snippetId);
      if (!snippet) throw new Error(`Field Kit recipe references missing snippet: ${snippetId}`);
      const lesson = findLesson(snippetId);
      if (!lesson) throw new Error(`Field Kit recipe has no lesson backlink: ${snippetId}`);
      const isPrimarySourceExample = lesson.snippet.id === snippetId;
      const sourceKind = isPrimarySourceExample && lesson.source.kind === "source-derived"
        ? "source-derived"
        : "supplemental";
      const contextual = contextualIds.has(snippetId);
      const gotcha = lesson.pitfalls[0] ?? "Check the input and target contract before adapting this pattern.";
      return {
        id: snippetId,
        snippetId,
        title: snippet.title,
        category,
        tags: tagsFor(snippet.title, snippetId, category),
        level: advancedIds.has(snippetId) ? "Advanced" : category === "Tensors" || category === "Autograd" ? "Foundations" : "Applied",
        sourceKind,
        whenToUse: purposeBySnippet[snippetId] ?? snippet.purpose,
        why: lesson.intuition,
        shapes: snippet.shapes,
        dependencies: snippet.dependencies,
        pitfalls: [gotcha],
        expected: snippet.expected,
        relatedLessonId: lesson.id,
        runnable: !contextual,
        runEnvironment: contextual
          ? "Copy into a local Python file after setting its input path to an existing CSV; the reader does not execute Python."
          : "Copy into a local Python file or Colab Python cell with the listed dependencies; the reader does not execute Python.",
        ...(projectBySnippet[snippetId] ? { projectIds: projectBySnippet[snippetId] } : {}),
      };
    }),
);
export const recipeById = new Map(referenceRecipes.map((recipe) => [recipe.id, recipe]));
export const snippetById = uniqueSnippets;
export const recipeCategories = selections.map(([category]) => category);
export const recipeSearchable = referenceRecipes.map((recipe) => ({
  id: `field-kit/${recipe.id}`,
  title: recipe.title,
  path: `Field Kit · ${recipe.category}`,
  kind: "recipe" as const,
  text: [recipe.title, recipe.category, ...recipe.tags, recipe.whenToUse, recipe.why, recipe.shapes, recipe.dependencies, ...recipe.pitfalls, snippetById.get(recipe.snippetId)!.expected, ...new Set(snippetById.get(recipe.snippetId)!.code.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [])].join(" "),
}));
