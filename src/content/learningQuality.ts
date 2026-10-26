export interface LearningCheck {
  lessonId: string;
  worked: string;
  shapePath: string;
  practiceSnippet: string;
}

export const learningChecks: LearningCheck[] = [
  { lessonId: "tensor-prerequisites", worked: "Two examples with three features form [2,3]. Multiplying by a [3,2] matrix contracts the feature axis and returns [2,2].", shapePath: "x [2,3] · W [3,2] → output [2,2]", practiceSnippet: "tensor-shapes" },
  { lessonId: "scalar-autograd", worked: "At x=3, y=x²=9 and dy/dx=2x=6. Backward stores 6 on the scalar leaf; it does not update x.", shapePath: "x [] → y [] → x.grad []", practiceSnippet: "autograd-numeric-check" },
  { lessonId: "vector-gradients", worked: "For squared residuals [1,1,2], the summed loss is 6 and the gradient is [2,2,4]. Mean reduction scales each gradient by 1/3.", shapePath: "x [3] → element losses [3] → scalar mean", practiceSnippet: "autograd-accumulate-clear" },
  { lessonId: "manual-loss", worked: "A scalar update with w=.2, gradient=.4 and η=.05 gives .2−.05×.4=.18.", shapePath: "w [] and ∇w [] → w′ []", practiceSnippet: "manual-bce" },
  { lessonId: "manual-training-loop", worked: "One Wisconsin example with 30 measurements enters as [1,30], produces one logit [1,1], and uses a float binary target [1,1].", shapePath: "[1,30] → linear → [1,1]", practiceSnippet: "manual-vs-idiomatic" },
  { lessonId: "loss-conventions", worked: "For z=2,y=1, sigmoid(z)≈.8808 and BCE≈.1269. BCEWithLogitsLoss computes the same objective more stably from z.", shapePath: "logits [B,1], targets [B,1] → scalar", practiceSnippet: "loss-shapes" },
  { lessonId: "optimizer-lifecycle", worked: "At every independent batch, clear gradients, calculate the current loss, backpropagate, then update parameters once.", shapePath: "batch [B,F] → logits [B,1] → scalar loss", practiceSnippet: "optimizer-lifecycle" },
  { lessonId: "dataloader-pipeline", worked: "With N=10 and batch size 4, sequential batches contain 4, 4, and 2 samples. drop_last=True omits the last two.", shapePath: "[10,F] → [4,F], [4,F], [2,F]", practiceSnippet: "dataloader-pipeline" },
  { lessonId: "preprocessing", worked: "Fit each scaler statistic on training rows only; apply the same stored transform to validation and test rows.", shapePath: "train [N,F] → scaler → valid/test [M,F]", practiceSnippet: "manual-preprocessing" },
  { lessonId: "ann-evaluation", worked: "In a confusion matrix, rows identify true labels and columns predictions; inspect class-specific mistakes as well as total accuracy.", shapePath: "labels [B] + predictions [B] → counts [C,C]", practiceSnippet: "metrics-confusion" },
  { lessonId: "ann-objective", worked: "Trials compare validation scores. After choosing settings, retrain the final model and evaluate the untouched test set once.", shapePath: "train → parameters; validation → trial scalar; test → final metric", practiceSnippet: "optuna-ann-safe" },
  { lessonId: "cnn-classifier", worked: "For two grayscale images, the two pooling stages give [2,32,14,14] then [2,64,7,7]; the head returns [2,10] logits.", shapePath: "[2,1,28,28] → [2,32,14,14] → [2,64,7,7] → [2,10]", practiceSnippet: "cnn-shape-trace" },
  { lessonId: "convolution-shapes", worked: "For n=28,k=3,p=1,s=2,d=1, floor((28+2−3)/2)+1=14 positions per axis.", shapePath: "[B,C,28,28] → [B,Cout,14,14]", practiceSnippet: "output-size-formula" },
  { lessonId: "vgg16-lab", worked: "A VGG16 feature grid [B,512,7,7] flattens to 25,088 values per image before the replacement 10-class head.", shapePath: "[B,3,224,224] → [B,512,7,7] → [B,25088] → [B,10]", practiceSnippet: "vgg-shape-debug" },
  { lessonId: "rnn-shapes", worked: "With B=2,T=4,F=3,H=5, batch-first unidirectional output is [2,4,5], while final state is [1,2,5].", shapePath: "[2,4,3] → output [2,4,5]; h_n [1,2,5]", practiceSnippet: "rnn-padding-pack" },
  { lessonId: "bptt", worked: "A shared recurrent weight receives contributions through each of three time steps; autograd sums those paths into its one parameter gradient.", shapePath: "x [B,3,F] → final logits [B,1] → scalar loss", practiceSnippet: "bptt-shared-gradients" },
  { lessonId: "gradient-pathologies", worked: "Four repeated factors of .5 shrink to .0625. Gradient clipping instead caps an oversized norm; it does not repair vanishing gradients.", shapePath: "recurrent Jacobian products → parameter gradients", practiceSnippet: "gradient-clipping" },
  { lessonId: "faq-classifier", worked: "The toy system maps question IDs [1,T] to answer-class logits [1,C], then maps the winning class ID to a stored full response.", shapePath: "question [1,T] → logits [1,C] → response lookup", practiceSnippet: "faq-loss-targets" },
  { lessonId: "lstm-gates", worked: "For one cell coordinate, f=.5,c_prev=4,i=.25,g=.8 gives c=.5×4+.25×.8=2.2. With o=.6, h=.6×tanh(2.2)≈.585.", shapePath: "gate vectors and c_prev [H] → c_t [H] → h_t [H]", practiceSnippet: "lstm-cell-manual" },
  { lessonId: "next-word-vocabulary", worked: "IDs [4,8,3,9] produce prefixes [4]→8, [4,8]→3, and [4,8,3]→9. PAD is a separate ID from UNK and EOS.", shapePath: "IDs [T] → prefix [T′] and scalar target", practiceSnippet: "nextword-prefix-pairs" },
  { lessonId: "next-word-training", worked: "For batch B=32 and vocabulary V=289, logits are [32,289] and integer next-token targets are [32].", shapePath: "X [B,T] → embedding [B,T,E] → logits [B,V]", practiceSnippet: "nextword-train-step" },
];

export const learningCheckByLesson = new Map(learningChecks.map((check) => [check.lessonId, check]));
