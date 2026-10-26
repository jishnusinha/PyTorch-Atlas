export interface SourceReference {
  start: number;
  end: number;
  kind: "source-derived" | "supplemental";
  note?: string;
}
export interface DiagramSpec {
  kind:
    | "graph"
    | "update"
    | "batch"
    | "tensor"
    | "chain-rule"
    | "tracking"
    | "gradients"
    | "pipeline"
    | "module"
    | "lifecycle"
    | "loader"
    | "collate"
    | "ann-data"
    | "ann"
    | "metrics"
    | "device"
    | "generalization"
    | "dropout"
    | "batchnorm"
    | "weight-decay"
    | "search-space"
    | "trial-comparison"
    | "optimizer-landscape"
    | "cnn-locality"
    | "conv-grid"
    | "conv-shapes"
    | "conv-channels"
    | "pooling"
    | "receptive-field"
    | "cnn-flow"
    | "transfer-stages"
    | "preprocess-pipeline"
    | "vgg-adaptation"
    | "normalization"
    | "sequence-shapes"
    | "rnn-unroll"
    | "rnn-variants"
    | "bptt"
    | "gradient-path"
    | "faq-tokens"
    | "faq-vocabulary"
    | "faq-batching"
    | "faq-rnn"
    | "faq-probabilities"
    | "faq-demo"
    | "lstm-dual-state"
    | "lstm-cell"
    | "lstm-vectors"
    | "lstm-gradient-route"
    | "lstm-shapes"
    | "lstm-parameters"
    | "nextword-prefix"
    | "nextword-padding"
    | "nextword-shapes"
    | "nextword-topk"
    | "nextword-append";
  title: string;
  description: string;
}
export interface Snippet {
  id: string;
  title: string;
  purpose: string;
  dependencies: string;
  shapes: string;
  code: string;
  expected: string;
  lines: string[];
}
export type RecipeCategory =
  | "Tensors"
  | "Autograd"
  | "nn.Module"
  | "Dataset/Loader"
  | "Losses/Metrics"
  | "Training/Optimizers"
  | "CNN/Transfer"
  | "RNN/LSTM/Text"
  | "Checkpoints/Debugging";
export interface ReferenceRecipe {
  id: string;
  snippetId: string;
  title: string;
  category: RecipeCategory;
  tags: string[];
  level: "Foundations" | "Applied" | "Advanced";
  sourceKind: "source-derived" | "supplemental";
  whenToUse: string;
  why: string;
  shapes: string;
  dependencies: string;
  pitfalls: string[];
  expected: string;
  relatedLessonId: string;
  runnable: boolean;
  runEnvironment: string;
  projectIds?: string[];
}
export interface CodeAnnotation {
  id: string;
  startLine: number;
  endLine: number;
  anchor: string;
  why: string;
  inputShape: string;
  outputShape: string;
  gotcha?: string;
}
export interface RevisionQuestion {
  question: string;
  answer: string;
}
export interface LessonSection {
  id: string;
  title: string;
  paragraphs: string[];
}
export interface Lesson {
  id: string;
  moduleId: string;
  title: string;
  subtitle: string;
  minutes: number;
  source: SourceReference;
  objectives: string[];
  intuition: string;
  equation: string;
  numerical: string;
  shapes: { name: string; shape: string; meaning: string }[];
  diagram: DiagramSpec;
  sections: LessonSection[];
  snippet: Snippet;
  template: Snippet;
  additionalSnippets?: Snippet[];
  pitfalls: string[];
  production: string;
  revision: string[];
  questions: RevisionQuestion[];
}
export interface Module {
  id: string;
  number: string;
  title: string;
  description: string;
  lessons: { id: string; title: string; ready: boolean }[];
}
