export interface Comparison {
  lessonId: string;
  title: string;
  source: string;
  sourcePages: string;
  repair: string;
  label: "PDF pattern + engineering repair" | "Supplemental framework comparison";
}

export const comparisons: Comparison[] = [
  { lessonId: "manual-training-loop", title: "Manual update and optimizer lifecycle", source: "The source first subtracts learning_rate × gradient under no_grad, then introduces nn.Module and torch.optim.", sourcePages: "PDF pp. 12–14, 17–22", repair: "Keep the hand calculation for intuition; use optimizer.zero_grad(), loss.backward(), and optimizer.step() for normal training.", label: "PDF pattern + engineering repair" },
  { lessonId: "loss-conventions", title: "Sigmoid + BCELoss and BCEWithLogitsLoss", source: "The source demonstrates sigmoid probabilities with BCELoss.", sourcePages: "PDF pp. 12–13, 18–20", repair: "Both express binary cross entropy. Prefer raw logits with BCEWithLogitsLoss for stable computation; targets remain float and shape-matched.", label: "PDF pattern + engineering repair" },
  { lessonId: "ann-objective", title: "Optuna test score and validation score", source: "The source objective evaluates trial models using the test loader and returns test accuracy to Optuna.", sourcePages: "PDF pp. 81–87", repair: "Use validation data inside the objective; select settings there, then report the untouched test score once.", label: "PDF pattern + engineering repair" },
  { lessonId: "faq-classifier", title: "Answer token and full response class", source: "The source predicts one word ID from the vocabulary as the answer.", sourcePages: "PDF pp. 170–179", repair: "For this tiny FAQ demonstration, classify among known answer IDs and return a stored full response. This remains a toy classifier, not general question answering.", label: "PDF pattern + engineering repair" },
  { lessonId: "next-word-vocabulary", title: "PAD, UNK and EOS identifiers", source: "The source initializes one unknown token at ID 0 and also uses zero for left padding.", sourcePages: "PDF pp. 200–202", repair: "Reserve distinct PAD, UNK and EOS IDs so missing positions, unseen words and sequence termination have separate meanings.", label: "PDF pattern + engineering repair" },
  { lessonId: "lstm-parameters", title: "Keras and PyTorch recurrent parameter counts", source: "The source's Keras SimpleRNN example counts input, recurrent and output matrices with two bias groups.", sourcePages: "PDF pp. 150–151", repair: "Native PyTorch nn.LSTM has four gates and two bias vectors per layer and direction by default. This is a framework-specific supplemental count.", label: "Supplemental framework comparison" },
];

export const comparisonByLesson = new Map(comparisons.map((item) => [item.lessonId, item]));
