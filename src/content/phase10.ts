import type { Lesson, Snippet } from "./types";

const make = (id: string, title: string, shapes: string, lines: string[], expected: string): Snippet => ({
  id, title, purpose: title, shapes,
  dependencies: "Python 3.10+ · PyTorch · CPU · no downloads",
  code: lines.join("\n"), expected,
  lines: lines.map((line, index) => {
    const code = line.trim();
    let explanation = `Apply this statement in the flow ${shapes}.`;
    if (!code) explanation = "Blank line separates related steps.";
    else if (code.startsWith("import ") || code.startsWith("from ")) explanation = "Import the Python or PyTorch tool used below.";
    else if (code.startsWith("#")) explanation = code.replace(/^#+\s*/, "");
    else if (code.startsWith("assert ")) explanation = `Check that ${code.slice(7)} holds for this example.`;
    else if (code.startsWith("class ")) explanation = `Define ${code.slice(6).split(/[(:]/)[0]} as a reusable type.`;
    else if (code.startsWith("def ")) explanation = `Define the helper ${code.slice(4).split("(")[0]} for this pipeline step.`;
    else if (code.startsWith("return ")) explanation = `Return ${code.slice(7)} to the caller.`;
    else if (code.startsWith("if ") || code.startsWith("elif ")) explanation = `Apply the conditional rule ${code.replace(/:$/, "")}.`;
    else if (code.startsWith("for ")) explanation = `Repeat the following operation for ${code.replace(/:$/, "").slice(4)}.`;
    else if (code.includes("manual_seed")) explanation = "Set deterministic initialization for this teaching run.";
    else if (code.includes("tokenize(") || code.includes("re.findall")) explanation = "Normalize or split the text using the configured tokenizer.";
    else if (code.includes("<pad>") || code.includes("<unk>")) explanation = "Reserve or use the special padding/unknown vocabulary index.";
    else if (code.includes("vocab")) explanation = "Use the saved token-to-index mapping to encode or decode text.";
    else if (code.includes("answer_to_id") || code.includes("answer_text")) explanation = "Map between a stable answer-class ID and its stored response.";
    else if (code.includes("Embedding") || code.includes("embedding(")) explanation = "Convert integer token IDs into dense feature vectors.";
    else if (code.includes("nn.RNN") || code.includes("self.rnn(")) explanation = "Read the embedded question in order and update the hidden state.";
    else if (code.includes("Linear") || code.includes("self.head(")) explanation = "Project the final hidden representation to answer-class logits.";
    else if (code.includes("pad_sequence")) explanation = "Pad variable-length token sequences to a common batch width.";
    else if (code.includes("pack_padded_sequence")) explanation = "Mark real lengths so padding does not advance the recurrent state.";
    else if (code.includes("CrossEntropyLoss") || code.includes("loss_fn(")) explanation = "Compare raw class logits with integer answer labels.";
    else if (code.includes("softmax")) explanation = "Normalize logits for relative class-score interpretation.";
    else if (code.includes("argmax") || code.includes("max(dim")) explanation = "Select the highest-scoring known answer class.";
    else if (code.includes("optimizer.zero_grad")) explanation = "Clear accumulated gradients before this batch.";
    else if (code.includes("backward()")) explanation = "Differentiate the loss through the model.";
    else if (code.includes("optimizer.step")) explanation = "Update trainable parameters using their current gradients.";
    else if (code.includes("model.train()")) explanation = "Put the model in training mode before optimization.";
    else if (code.includes("model.eval()")) explanation = "Switch to evaluation behavior for validation or inference.";
    else if (code.includes("inference_mode")) explanation = "Disable gradient recording for this evaluation pass.";
    else if (code.includes("DataLoader")) explanation = "Create an iterator that groups dataset rows into batches.";
    else if (code.includes("torch.save")) explanation = "Persist model weights together with their text mappings.";
    else if (code.includes("torch.load")) explanation = "Restore the checkpoint on CPU with restricted weight loading.";
    else if (code.includes("torch.tensor") || code.includes("torch.stack")) explanation = `Construct a typed tensor for ${shapes}.`;
    else if (code.includes("print(")) explanation = "Display the checked values for this CPU example.";
    else if (code.includes("optimizer =") || code.includes("loss_fn =")) explanation = "Configure the optimizer or logits-compatible objective.";
    else if (code.startsWith("with ")) explanation = `Enter the ${code.replace(/:$/, "")} context for safe execution.`;
    else if (code.startsWith("self.")) explanation = "Store this layer or value as part of the model.";
    else if (code.includes(" = ")) explanation = `Compute and store ${code.split(" = ")[0]} as part of ${shapes}.`;
    else explanation = `Execute ${code} in the sequence classification pipeline.`;
    return `Line ${index + 1} · ${explanation}`;
  }),
});

const toyData = [
  "('What is the capital of France?', 'france_capital'),",
  "('Name the capital city of France', 'france_capital'),",
  "('What is the capital of Japan?', 'japan_capital'),",
  "('Which city is Japan capital?', 'japan_capital'),",
  "('Which planet is called the red planet?', 'red_planet'),",
  "('Name the red planet', 'red_planet'),",
  "('How many planets orbit the Sun?', 'planet_count'),",
  "('Count planets in our solar system', 'planet_count'),",
  "('Who created Python?', 'python_creator'),",
  "('Who developed the Python language?', 'python_creator'),",
  "('What does CPU stand for?', 'cpu_expansion'),",
  "('Expand the abbreviation CPU', 'cpu_expansion'),",
];

export const phase10Snippets: Snippet[] = [
  make("faq-tokenize", "Normalize and tokenize a question", "raw string → lowercase normalized text → token list", [
  "import re", "import torch", "", "def tokenize(text):", "    return re.findall(r\"[a-z0-9']+\", text.lower())",
    "tokens = tokenize('What is the capital of France?')", "assert tokens == ['what', 'is', 'the', 'capital', 'of', 'france']",
    "print(tokens)",
  ], "['what', 'is', 'the', 'capital', 'of', 'france']"),
  make("faq-vocab", "Build a stable PAD/UNK vocabulary", "question tokens → token-to-index map; PAD=0, UNK=1", [
    "import re", "import torch", "", "def tokenize(text):", "    return re.findall(r\"[a-z0-9']+\", text.lower())",
    "def build_vocab(training_questions):", "    vocab = {'<pad>': 0, '<unk>': 1}",
    "    for text in training_questions:", "        for token in tokenize(text):", "            if token not in vocab:", "                vocab[token] = len(vocab)", "    return vocab",
    "vocab = build_vocab(['Capital of France?', 'Red planet'])", "encoded = torch.tensor([vocab.get(token, vocab['<unk>']) for token in tokenize('Capital of Mars?')], dtype=torch.long)",
    "assert vocab['<pad>'] == 0 and vocab['<unk>'] == 1", "assert encoded[-1].item() == vocab['<unk>']",
    "print('vocabulary:', len(vocab), 'encoded:', encoded.tolist())",
  ], "Vocabulary reserves padding and unknown IDs; unknown input maps safely to <unk>."),
  make("faq-dataset", "Return question IDs and answer-class IDs", "Dataset item: tokens [T] long + class [] long", [
    "import re", "import torch", "from torch.utils.data import Dataset", "",
    "def tokenize(text):", "    return re.findall(r\"[a-z0-9']+\", text.lower())",
    "class FAQDataset(Dataset):", "    def __init__(self, rows, vocab, answer_to_id):", "        self.rows, self.vocab, self.answer_to_id = rows, vocab, answer_to_id",
    "    def __len__(self):", "        return len(self.rows)", "    def __getitem__(self, index):",
    "        question, answer = self.rows[index]", "        ids = [self.vocab.get(t, self.vocab['<unk>']) for t in tokenize(question)]",
    "        if not ids:", "            ids = [self.vocab['<unk>']]",
    "        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.answer_to_id[answer], dtype=torch.long)",
    "rows = [('Capital of France?', 'geography')]", "vocab = {'<pad>': 0, '<unk>': 1, 'capital': 2, 'of': 3, 'france': 4}",
    "dataset = FAQDataset(rows, vocab, {'geography': 0})", "question_ids, target = dataset[0]",
    "assert question_ids.dtype == torch.long and target.shape == () and target.item() == 0", "print(question_ids.tolist(), target.item())",
  ], "An item returns integer token IDs and one scalar long answer-class target."),
  make("faq-collate", "Pad variable-length questions in a batch", "items [T₁],[T₂] → tokens [B,Tmax], lengths [B], labels [B]", [
    "import torch", "from torch.nn.utils.rnn import pad_sequence", "",
    "def collate_faq(batch, pad_id=0):", "    questions, labels = zip(*batch)",
    "    lengths = torch.tensor([len(q) for q in questions], dtype=torch.long)",
    "    tokens = pad_sequence(questions, batch_first=True, padding_value=pad_id)",
    "    return tokens, lengths, torch.stack(labels)",
    "batch = [(torch.tensor([2, 3, 4]), torch.tensor(1)), (torch.tensor([5]), torch.tensor(0))]",
    "tokens, lengths, labels = collate_faq(batch)", "assert tokens.tolist() == [[2, 3, 4], [5, 0, 0]]",
    "assert lengths.tolist() == [3, 1] and labels.shape == (2,)", "print(tokens.shape, lengths.tolist(), labels.tolist())",
  ], "Tokens [2,3], lengths [2], class labels [2]; batch_first padding keeps rows aligned."),
  make("faq-model", "Map an embedded question to answer-class logits", "IDs [B,T] → embedding [B,T,E] → RNN [B,T,H] → h_n [1,B,H] → logits [B,K]", [
    "import torch", "from torch import nn", "",
    "class FAQClassifier(nn.Module):", "    def __init__(self, vocab_size, num_answers, embed_dim=8, hidden_size=12, pad_id=0):",
    "        super().__init__()", "        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)",
    "        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)", "        self.output = nn.Linear(hidden_size, num_answers)",
    "    def forward(self, tokens):", "        embedded = self.embedding(tokens)", "        _, hidden = self.rnn(embedded)", "        return self.output(hidden[-1])",
    "model = FAQClassifier(vocab_size=20, num_answers=4)", "logits = model(torch.tensor([[2, 3, 4], [5, 6, 0]], dtype=torch.long))",
    "assert logits.shape == (2, 4) and torch.isfinite(logits).all()", "print(logits.shape)",
  ], "Two questions produce four raw class logits each; the final hidden state feeds the class head."),
  make("faq-padding-pack", "Use true lengths in the recurrent state", "tokens [B,T] → packed embedded sequence → final valid h_n [1,B,H]", [
    "import torch", "from torch import nn", "from torch.nn.utils.rnn import pack_padded_sequence", "",
    "torch.manual_seed(10)", "tokens = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]], dtype=torch.long)",
    "lengths = torch.tensor([3, 2], dtype=torch.long)", "embedding = nn.Embedding(8, 5, padding_idx=0)", "rnn = nn.RNN(5, 7, batch_first=True)",
    "embedded = embedding(tokens)", "packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)",
    "_, hidden = rnn(packed)", "assert embedded.shape == (2, 4, 5) and hidden.shape == (1, 2, 7)",
    "print('embedded:', embedded.shape, 'final valid states:', hidden.shape)",
  ], "Packed lengths prevent padding steps from becoming part of the final state; packing is supplemental to the PDF example."),
  make("faq-loss-targets", "Pair CrossEntropyLoss with class targets", "logits [B,K] float → CrossEntropyLoss; labels [B] long", [
    "import torch", "from torch import nn", "", "logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.5, 0.2]], dtype=torch.float32)",
    "labels = torch.tensor([0, 1], dtype=torch.long)", "loss = nn.CrossEntropyLoss()(logits, labels)",
    "probabilities = logits.softmax(dim=1)", "assert labels.shape == (2,) and labels.dtype == torch.long",
    "assert torch.allclose(probabilities.sum(dim=1), torch.ones(2)) and loss.item() > 0", "print('loss:', round(loss.item(), 4), 'probability sums:', probabilities.sum(1))",
  ], "Cross entropy consumes raw logits and long class IDs; softmax is for interpreting predictions."),
  make("faq-inference", "Predict an answer class with an abstention heuristic", "one question → IDs [1,T] → logits/probabilities [1,K] → answer text or abstain", [
    "import re", "import torch", "from torch import nn", "", "def predict_answer(model, text, vocab, answer_by_id, threshold=0.5):",
    "    tokens = re.findall(r\"[a-z0-9']+\", text.lower())", "    if not tokens:", "        return \"I don't know\", 0.0",
    "    unknown_only = all(token not in vocab for token in tokens)",
    "    ids = [vocab.get(token, vocab['<unk>']) for token in tokens]", "    model.eval()",
    "    with torch.inference_mode():", "        tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)",
    "        lengths = torch.tensor([len(ids)], dtype=torch.long)", "        logits = model(tokens, lengths)",
    "        probabilities = logits.softmax(dim=1).squeeze(0)", "        confidence, answer_id = probabilities.max(dim=0)",
    "    if unknown_only or confidence.item() < threshold:", "        return \"I don't know\", confidence.item()",
    "    return answer_by_id[answer_id.item()], confidence.item()", "",
    "assert callable(predict_answer)", "print('Use a trained classifier and a calibrated threshold before inference.')",
  ], "Inference returns one stored answer class only; max-softmax threshold is not a reliable OOD detector."),
  make("faq-checkpoint", "Save weights with their text mappings", "state_dict + vocab + answer map → local checkpoint → restored model", [
    "import tempfile", "from pathlib import Path", "import torch", "from torch import nn", "",
    "model = nn.Linear(3, 2)", "vocab = {'<pad>': 0, '<unk>': 1, 'hello': 2}", "answer_by_id = {0: 'Paris', 1: 'Tokyo'}",
    "with tempfile.TemporaryDirectory() as folder:", "    path = Path(folder) / 'faq.pt'",
    "    torch.save({'model_state': model.state_dict(), 'vocab': vocab, 'answer_by_id': answer_by_id}, path)",
    "    saved = torch.load(path, map_location='cpu', weights_only=True)", "    restored = nn.Linear(3, 2)", "    restored.load_state_dict(saved['model_state'])",
    "    assert saved['vocab'] == vocab and saved['answer_by_id'] == answer_by_id",
    "    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))", "print('weights and text mappings restored')",
  ], "A checkpoint must keep weights and the exact vocabulary/answer mapping together."),
  make("faq-source-target", "Inspect the source's single-token target", "answer tokens [1,L] → answer[0] [1] → one vocabulary target; improved target is class ID [1]", [
    "import torch", "", "answer_tokens = torch.tensor([[7, 9, 12]], dtype=torch.long)",
    "source_target = answer_tokens[0]", "single_token_target = source_target[0].reshape(1)",
    "answer_class_target = torch.tensor([2], dtype=torch.long)",
    "assert source_target.shape == (3,) and single_token_target.shape == (1,)",
    "assert answer_class_target.shape == (1,) and answer_class_target.dtype == torch.long",
    "print('source first-token target:', single_token_target.tolist(), 'FAQ class target:', answer_class_target.tolist())",
  ], "The exported code selects answer[0], so its vocabulary head predicts one token. Answer-class classification supports a stored multiword answer."),
  make("faq-oov", "Encode a question with unseen words safely", "tokens [T] → known IDs or UNK → embedding indices", [
    "import torch", "from torch import nn", "", "vocab = {'<pad>': 0, '<unk>': 1, 'capital': 2}",
    "words = ['capital', 'of', 'france']", "ids = [vocab.get(word, vocab['<unk>']) for word in words]",
    "embedding = nn.Embedding(len(vocab), 4, padding_idx=vocab['<pad>'])", "vectors = embedding(torch.tensor(ids, dtype=torch.long))",
    "assert ids == [2, 1, 1] and vectors.shape == (3, 4)", "assert vocab['<pad>'] != vocab['<unk>']", "print(ids, tuple(vectors.shape))",
  ], "Unknown words remain valid indices; padding and unknown tokens have distinct IDs."),
  make("faq-end-to-end", "Train and save a tiny FAQ answer-class model", "question IDs [B,T] → [B,T,E] → packed RNN → [B,H] → [B,K]", [
    "import re", "import tempfile", "from pathlib import Path", "import torch", "from torch import nn",
    "from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence", "from torch.utils.data import Dataset, DataLoader", "",
    "torch.manual_seed(27)", "torch.set_num_threads(1)",
    "rows = [" , ...toyData.map(v => `    ${v}`), "]",
    "answer_text = {'france_capital': 'Paris', 'japan_capital': 'Tokyo', 'red_planet': 'Mars', 'planet_count': 'Eight', 'python_creator': 'Guido van Rossum', 'cpu_expansion': 'Central Processing Unit'}",
    "answer_to_id = {name: i for i, name in enumerate(answer_text)}",
    "train_rows = rows[::2]", "valid_rows = rows[1::2]",
    "def tokenize(text):", "    return re.findall(r\"[a-z0-9']+\", text.lower())",
    "vocab = {'<pad>': 0, '<unk>': 1}", "for question, _ in train_rows:", "    for token in tokenize(question):", "        if token not in vocab:", "            vocab[token] = len(vocab)",
    "class FAQDataset(Dataset):", "    def __init__(self, data): self.data = data", "    def __len__(self): return len(self.data)",
    "    def __getitem__(self, i):", "        question, answer = self.data[i]", "        ids = [vocab.get(t, vocab['<unk>']) for t in tokenize(question)]",
    "        return torch.tensor(ids, dtype=torch.long), torch.tensor(answer_to_id[answer], dtype=torch.long)",
    "def collate(batch):", "    questions, targets = zip(*batch)", "    lengths = torch.tensor([len(q) for q in questions], dtype=torch.long)",
    "    tokens = pad_sequence(questions, batch_first=True, padding_value=vocab['<pad>'])", "    return tokens, lengths, torch.stack(targets)",
    "class FAQModel(nn.Module):", "    def __init__(self):", "        super().__init__()", "        self.embedding = nn.Embedding(len(vocab), 8, padding_idx=vocab['<pad>'])",
    "        self.rnn = nn.RNN(8, 12, batch_first=True)", "        self.head = nn.Linear(12, len(answer_to_id))",
    "    def forward(self, tokens, lengths):", "        embedded = self.embedding(tokens)",
    "        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)",
    "        _, hidden = self.rnn(packed)", "        return self.head(hidden[-1])",
    "train_loader = DataLoader(FAQDataset(train_rows), batch_size=4, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(27))",
    "valid_loader = DataLoader(FAQDataset(valid_rows), batch_size=4, shuffle=False, collate_fn=collate)",
    "model = FAQModel()", "optimizer = torch.optim.Adam(model.parameters(), lr=0.03)", "loss_fn = nn.CrossEntropyLoss()",
    "for epoch in range(35):", "    model.train()", "    for tokens, lengths, labels in train_loader:",
    "        optimizer.zero_grad(set_to_none=True)", "        logits = model(tokens, lengths)", "        loss = loss_fn(logits, labels)", "        loss.backward()", "        optimizer.step()",
    "model.eval()", "validation_losses = []", "",
    "with torch.inference_mode():", "    for tokens, lengths, labels in valid_loader:", "        validation_losses.append(loss_fn(model(tokens, lengths), labels).item())",
    "assert validation_losses and all(torch.isfinite(torch.tensor(validation_losses)))",
    "with tempfile.TemporaryDirectory() as folder:", "    checkpoint = Path(folder) / 'faq.pt'",
    "    torch.save({'model_state': model.state_dict(), 'vocab': vocab, 'answer_to_id': answer_to_id, 'answer_text': answer_text}, checkpoint)",
    "    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)", "    restored = FAQModel()", "    restored.load_state_dict(saved['model_state'])",
    "    example = 'What is the capital of France?'", "    ids = [saved['vocab'].get(t, saved['vocab']['<unk>']) for t in tokenize(example)]",
    "    restored.eval()", "    with torch.inference_mode():", "        logits = restored(torch.tensor(ids, dtype=torch.long).unsqueeze(0), torch.tensor([len(ids)]))",
    "        probabilities = logits.softmax(dim=1)", "        answer_id = probabilities.argmax(dim=1).item()",
    "    assert logits.shape == (1, len(answer_to_id))", "    assert saved['answer_text'][next(k for k, v in saved['answer_to_id'].items() if v == answer_id)]",
    "print('vocab:', len(vocab), 'classes:', len(answer_to_id), 'validation loss:', round(sum(validation_losses) / len(validation_losses), 3))",
  ], "CPU toy run trains on six question variants, validates on six held-out paraphrases, and restores model plus vocabulary and answer mapping. Metrics are illustrative only."),
];

const C: Record<string, Snippet> = Object.fromEntries(phase10Snippets.map((snippet) => [snippet.id, snippet]));
const diagrams: Record<string, Lesson["diagram"]> = {
  tokens: { kind: "faq-tokens", title: "From question text to token IDs", description: "Normalize and split the question into tokens, then replace each token with the vocabulary index used by the embedding layer." },
  vocab: { kind: "faq-vocabulary", title: "A stable vocabulary with special IDs", description: "The PAD index fills short rows during batching; the UNK index keeps unseen inference tokens inside the embedding table." },
  batch: { kind: "faq-batching", title: "Batch questions without losing their lengths", description: "Questions can share a padded [B,T] tensor while length metadata lets the recurrent model ignore artificial padding." },
  rnn: { kind: "faq-rnn", title: "Embed, summarize, classify", description: "Integer questions become dense embeddings, a recurrent encoder creates a final hidden summary, and a linear layer scores known answer classes." },
  probabilities: { kind: "faq-probabilities", title: "Interpret class logits carefully", description: "Softmax normalizes scores across known classes; a high maximum is not proof that a question is in scope or factually answered." },
  demo: { kind: "faq-demo", title: "A fixed illustrative FAQ mapping", description: "This conceptual selector displays hand-authored sample mappings; it does not run the PyTorch model or invent model predictions." },
};
const sec = (id: string, title: string, a: string, b: string): Lesson["sections"][number] => ({ id, title, paragraphs: [a, b] });
const lesson = (id: string, title: string, subtitle: string, pages: number[], diagram: Lesson["diagram"], main: string, template: string, shapes: Lesson["shapes"], objectives: string[], intuition: string, equation: string, numerical: string, sections: Lesson["sections"], pitfalls: string[], production: string, revision: string[], questions: [string, string][], note = "Core project follows PDF pages; improvements to answer targets, batching and inference are labeled supplemental.", sourceKind: Lesson["source"]["kind"] = "source-derived"): Lesson => ({
  id, moduleId: "sequences", title, subtitle, minutes: 9,
  source: { start: pages[0], end: pages.at(-1) ?? pages[0], kind: sourceKind, note },
  diagram, snippet: C[main], template: C[template], objectives, intuition, equation, numerical, shapes, sections, pitfalls, production, revision,
  questions: questions.map(([question, answer]) => ({ question, answer })),
});

export const phase10Lessons: Lesson[] = [
  lesson("faq-classifier", "A FAQ classifier is not an open-ended answer engine", "Build a small question-to-known-answer classifier and name its limits.", [168, 168], diagrams.demo, "faq-source-target", "faq-end-to-end",
    [{name:"Question",shape:"text → [B,T]",meaning:"A normalized sequence of integer tokens."},{name:"Classifier",shape:"[B,K] logits",meaning:"One score for each of K answer classes."},{name:"Answer",shape:"class ID → stored text",meaning:"A known answer string selected from a fixed mapping."}],
    ["Define the toy FAQ task as classification among known answers.","Contrast answer classes with vocabulary-token prediction.","Explain why the demonstration cannot answer arbitrary questions."],
    "A FAQ classifier resembles a multiple-choice quiz with a fixed answer key: it chooses one of the answer categories it has seen, rather than composing a new response from facts.",
    "f(question) = logits ∈ ℝᴷ; predicted class = argmaxₖ logitsₖ; answer = answer_text[predicted class]. CrossEntropyLoss trains the K class logits against a scalar class ID.",
    "With K=4 answer classes, one question produces logits [1,4]. Softmax probabilities sum to 1 across those four choices, but the model must choose one even if the question is unrelated.",
    [sec("scope","What the source calls question answering","The PDF describes a small general-knowledge CSV with roughly 90 question/answer pairs and reports a vocabulary size of 324 for that particular data. Those are source-specific project details, not required values or a reusable benchmark. The example is useful for learning text preprocessing and recurrent classification mechanics.","It is not open-ended generative question answering, retrieval-augmented generation, or a factual retrieval engine. A reliable knowledge system must ground answers in an authoritative source and handle unsupported questions explicitly."),sec("target-repair","Use an answer class for stored multiword replies","The exported training code takes answer[0], the first answer token, and trains a vocabulary-sized classifier. This means it predicts one word, despite the project description suggesting a full answer.","This phase makes the target an answer-class ID and stores the corresponding full answer text. That engineering repair supports multiword fixed responses but remains limited to predefined classes. The interactive diagram uses only hand-authored illustrative mappings, never learned predictions.")],
    ["Vocabulary-token logits do not generate an entire answer when only one target token is used.","A softmax maximum always exists, even for an unrelated question.","A fixed answer class cannot produce new factual language.","The small local sample demonstrates mechanics, not FAQ accuracy."],
    "For production support, retrieve a matching approved answer with provenance, calibrate abstention on representative out-of-scope questions, log versioned content and route low-confidence cases to a human. Do not rely on max-softmax alone.",
    ["The source's output is one vocabulary token from its first target position.","This implementation predicts among known answer classes.","Stored answer text can be multiword.","The model is a teaching demo and cannot answer arbitrary facts."],
    [["What task does this repaired demo solve?","Question classification into a finite set of known answer classes."],["Why is it not generative QA?","It selects a stored response and does not compose or retrieve arbitrary factual answers."],["What does the source code target with answer[0]?","The first token of an answer, as a vocabulary classification label."],["Can a high maximum softmax guarantee a valid FAQ match?","No; softmax always normalizes over known classes and can be confident out of distribution."]]),
  lesson("faq-vocabulary", "Normalize questions and build a vocabulary", "Turn text into stable integer IDs with explicit unknown handling.", [169, 169], diagrams.tokens, "faq-tokenize", "faq-vocab",
    [{name:"Raw text",shape:"string",meaning:"User-entered characters and punctuation."},{name:"Tokens",shape:"[T] strings",meaning:"Lowercase lexical units."},{name:"Encoded sequence",shape:"[T] long",meaning:"Indices into the embedding table."}],
    ["Normalize and tokenize text consistently.","Build vocabulary from training questions only.","Reserve distinct PAD and UNK IDs and map unseen words safely."],
    "A vocabulary is a numbered dictionary: each known word gets an index card. UNK is the spare card for a word not in the training dictionary; PAD fills unused positions when questions are batched.",
    "tokenize(s) → (w₁,…,wₜ); encode(wᵢ)=vocab[wᵢ] when known, otherwise vocab[UNK]. Keep PAD distinct from UNK because padding means no token while UNK means an unrecognized token occurred.",
    "For the six tokens in “What is the capital of France?”, T=6 and the encoded tensor has shape [6]. If “Mars” was absent during training, it maps to UNK rather than causing a key error.",
    [sec("normalize","Use one normalization rule","Lowercasing and a documented punctuation tokenizer make “France?”, “FRANCE” and “France” map consistently. A regular expression can retain apostrophes within words; production tokenization may need Unicode, language-specific rules or a subword tokenizer.","The same tokenizer and normalization settings must be saved with the model. If training and inference tokenize differently, familiar questions can become sequences of unknown IDs."),sec("vocabulary","Reserve PAD and UNK","Build the vocabulary from training questions so validation/test vocabulary does not leak into preprocessing. Keep `<pad>` and `<unk>` at fixed indices and store the token-to-index mapping in the checkpoint.","The source uses `<UNK>` at zero and combines question/answer tokens into one vocabulary. This implementation separates `<pad>` and `<unk>` and predicts answer classes, so vocabulary entries need only represent input questions. These are clearly marked robustness/task-target repairs.")],
    ["A PAD token is not an unknown word.","Do not build vocabulary using validation/test text.","The tokenizer must be identical at training and inference.","Avoid treating the special tokens as ordinary answer classes."],
    "Version tokenizer configuration, normalization, vocabulary and answer mapping together. Track OOV rates; a large OOV rate is a data or domain mismatch signal, not something an embedding can fix by itself.",
    ["Normalization must be consistent end-to-end.","Token IDs are integer indices, not float features.","PAD and UNK represent different conditions.","Unseen tokens map safely to UNK."],
    [["Why lowercase text?","To make capitalization variants share token IDs under this normalization scheme."],["Why reserve UNK?","It gives unseen words a valid embedding index instead of raising a key error."],["Why keep PAD separate from UNK?","PAD marks an absent position; UNK marks an actual but unrecognized token."],["Which text should define the vocabulary?","Training questions only, then save that mapping for later use."]]),
  lesson("faq-vocabulary-map", "Keep answer classes separate from word IDs", "Represent each stored answer with a stable class target and reverse mapping.", [169, 170], diagrams.vocab, "faq-vocab", "faq-source-target",
    [{name:"Input word",shape:"token ID",meaning:"An index used only by the embedding vocabulary."},{name:"Target",shape:"answer class ID",meaning:"A scalar long label for CrossEntropyLoss."},{name:"Response map",shape:"ID → text",meaning:"A deterministic mapping to the full approved response."}],
    ["Distinguish token IDs from answer-class labels.","Create deterministic answer mappings.","Support a fixed multiword answer without claiming generation."],
    "Words use one numbered index system to enter the model; answer classes use a separate index system to label the desired response. Mixing the two is like using library shelf numbers as quiz answer choices.",
    "yᵢ = answer_to_id(answerᵢ) ∈ {0,…,K−1}; model logits have final width K. After argmax, answer_text[ŷ] returns the full stored response.",
    "If the class map is geography→0 and astronomy→1, the label for astronomy is scalar `1` with dtype long; the response table might map class 1 to the text “Mars”. The word “Mars” has a separate vocabulary token ID, if present.",
    [sec("mapping","Make class labels reproducible","Create the answer-class map once from the approved set of responses and persist both directions. Each dataset row returns question IDs and one scalar class ID; `nn.CrossEntropyLoss` expects that integer label.","A fixed response can contain several words without making the output layer vocabulary-sized. The classifier selects one response class and then looks up its complete text."),sec("source-repair","Preserve and repair the source design","The PDF includes answer words in a shared vocabulary and uses the first answer token as its class target. That code is a compact next-word-style classification demonstration, not full-answer generation.","The separate answer-class mapping is supplemental: it better matches the advertised fixed FAQ behavior. It still memorizes label boundaries from a tiny dataset and cannot synthesize a novel response.")],
    ["Do not use a token ID as an answer-class ID.","Keep the output layer width equal to number of answer classes K.","The response map is part of the model artifact.","Changing label order without changing the checkpoint corrupts predictions."],
    "Use stable response IDs and immutable mapping versions. If answers are edited, review and revalidate the class mapping rather than silently relabeling saved logits.",
    ["Input vocabulary IDs index embeddings.","Answer-class IDs index the final classifier logits.","The answer map converts a selected class to a full stored response.","Checkpoint mappings must match training label order."],
    [["What does K mean?","The number of supported answer classes."],["What label dtype does cross entropy need?","torch.long class indices."],["Can a class map return a multiword reply?","Yes, the selected ID indexes a stored response string."],["Why save the class mapping?","The meaning and order of logits depend on the exact label map."]], "Supplemental answer-class repair: the PDF predicts a vocabulary token; this lesson separates input word IDs from fixed response-class IDs.", "supplemental"),
  lesson("faq-data", "Batch FAQ questions with a custom Dataset", "Pair encoded question sequences with answer labels and pad them safely.", [174, 176], diagrams.batch, "faq-dataset", "faq-collate",
    [{name:"Dataset row",shape:"question text + class",meaning:"One labeled FAQ example."},{name:"Encoded item",shape:"tokens [T], label []",meaning:"Variable-length input and scalar target."},{name:"Collated batch",shape:"tokens [B,Tmax], lengths [B], labels [B]",meaning:"Padded token rows with original lengths."}],
    ["Implement Dataset length and item access.","Explain why the source chooses batch size one.","Use a collate function to batch variable-length sequences."],
    "A Dataset is a filing cabinet of question/answer pairs. A collator puts several questions on one rectangular page, using PAD for blank cells and recording each question's real length.",
    "For lengths T₁,…,Tᴮ, Tmax=max(Tᵢ); tokens are padded to [B,Tmax], lengths are [B], and class labels are [B]. Padding does not change the underlying answer label.",
    "Batch lengths 3 and 1 become a [2,3] token matrix; the shorter row gets two PAD IDs. The lengths tensor [3,1] lets a packed RNN ignore those two positions.",
    [sec("dataset","Return tensors from a Dataset","The dataset stores rows, vocabulary and answer map. `__getitem__` tokenizes and encodes a question into a long tensor and returns a scalar long class target; `__len__` reports example count.","The source uses batch size one because variable question and answer lengths cannot be stacked without padding. A custom collate function is the standard improvement: it pads tokens while retaining true lengths."),sec("collation","Pad without losing length information","Use `pad_sequence(..., batch_first=True, padding_value=pad_id)` and return lengths and labels. The DataLoader can shuffle training items reproducibly and keep validation order stable.","Do not use padding IDs as evidence. The embedding can set `padding_idx`, and a packed sequence can keep the recurrent final state at the last valid token. For non-packed token-level losses, padded target positions need an ignore mask.")],
    ["Default collation cannot stack variable-length token tensors.","Do not discard original lengths after padding.","Dataset outputs and class targets should use torch.long.","Never shuffle a time-ordered forecasting split across train and validation."],
    "For real FAQ data, group paraphrases and near-duplicates before splitting so variants of the same question do not leak across train and validation. Keep final test examples untouched until evaluation.",
    ["Dataset items pair `[T]` token IDs with a scalar label.","Padding enables rectangular batches.","Lengths identify which tokens are real.","A custom collator permits batch sizes above one."],
    [["Why did the source use batch size one?","Different sequence lengths cannot be stacked without padding."],["What does the collator return besides padded tokens?","True lengths and class labels."],["What is the token batch shape?","[B,Tmax]."],["How can near-duplicate leakage be prevented?","Group related questions before splitting into train/validation/test."]]),
  lesson("faq-architecture", "Encode each question with an embedding and RNN", "Trace integer tokens through the recurrent classifier and respect sequence lengths.", [170, 171, 177], diagrams.rnn, "faq-model", "faq-padding-pack",
    [{name:"Question IDs",shape:"[B,T] long",meaning:"A batch of indexed words."},{name:"Embedding",shape:"[B,T,E] float32",meaning:"Dense vector per token."},{name:"RNN / final state",shape:"[B,T,H] → [B,H]",meaning:"Ordered hidden sequence and last valid summary."},{name:"Classifier",shape:"[B,K] float32",meaning:"Raw answer-class logits."}],
    ["Reconstruct the source nn.Embedding → nn.RNN → Linear model.","Trace each tensor axis and dtype.","Use the final valid state when batches contain padding."],
    "The embedding turns each numbered word into a short learned vector; the RNN reads the vectors in order and carries a summary; a final layer scores answer choices.",
    "X:[B,T] long → Embedding(X):[B,T,E] float → RNN output:[B,T,H], hₙ:[1,B,H] → hₙ[−1]:[B,H] → Linear:[B,K]. With packed inputs, the returned hidden state summarizes only valid tokens.",
    "For B=2,T=5,E=8,H=12,K=4: IDs contain 10 integer slots, the embedding has 2·5·8=80 floats, dense RNN output has 2·5·12=120 floats, final states have 24, and logits have 8.",
    [sec("architecture","The model returns answer scores","The source uses an embedding dimension of 50 and hidden size 64, with `batch_first=True`, then projects to its source-reported vocabulary width 324. `nn.RNN` returns both per-step outputs and final hidden state; its final hidden shape is [layers,batch,hidden]. Squeeze/select the layer axis to get [B,H].", "This repaired classifier maps [B,H] to [B,K] answer-class logits. CrossEntropyLoss consumes logits directly and does its own log-softmax calculation; do not add a softmax to the model's training output."),sec("lengths","Final means last real token","Without padding, the last hidden state corresponds to the end of each question. With right padding, the final array time step may be artificial, so `hidden[-1]` from a packed RNN or a gather using true lengths is needed.","The source's batch-size-one pipeline avoids padding. `pack_padded_sequence` with `enforce_sorted=False` is supplemental and keeps the true sequence endpoints while retaining batches.")],
    ["Embedding inputs must be integer token IDs.","Use batch_first=True to read [B,T,E].","RNN hidden includes a layer dimension; select it intentionally.","Never treat right-padding as a real final word."],
    "Record embedding vocabulary size, padding index, maximum length and model dimensions with the checkpoint. For small FAQ datasets, compare against a keyword or retrieval baseline before adopting an RNN.",
    ["Embedding shape is [B,T,E].","RNN final hidden shape is [layers,B,H].","The class head emits raw logits [B,K].","Packed lengths avoid using padding as the final state."],
    [["What does batch_first=True expect?","[B,T,E] input to the recurrent layer."],["What is the final one-layer hidden layout?","[1,B,H]."],["What shape enters the Linear class head?","[B,H]."],["Why use a packed sequence?","To stop padding positions from altering each final hidden state."]]),
  lesson("faq-training", "Train with answer-class cross entropy", "Connect class labels, the optimizer loop and honest validation.", [172, 177, 178], diagrams.probabilities, "faq-loss-targets", "faq-end-to-end",
    [{name:"Logits",shape:"[B,K] float32",meaning:"Unnormalized score per answer class."},{name:"Labels",shape:"[B] long",meaning:"Correct class index per question."},{name:"Loss",shape:"scalar",meaning:"Mean negative log-likelihood of the correct class."}],
    ["Pair CrossEntropyLoss with raw class logits.","Use long answer-class labels and a clean optimizer lifecycle.","Separate validation from the final test set and avoid toy accuracy claims."],
    "The model places one score beside each answer choice. Cross entropy penalizes it when the correct choice receives a low score; Adam adjusts model parameters from that loss.",
    "L = −(1/B) Σᵢ log softmax(zᵢ)₍yᵢ₎ where zᵢ∈ℝᴷ and yᵢ is a long class index. Training uses zero_grad → model(x) → loss → backward → step.",
    "For one row with logits [2,0,−1], softmax is approximately [0.844,0.114,0.042]. If target class is 0, its loss is −ln(0.844)≈0.170. Labels are [B], not [B,1].",
    [sec("objective","Classify the answer class","The source configures Adam at learning rate 0.001, CrossEntropyLoss and 20 epochs, with one vocabulary-token target per question. The repaired FAQ classifier changes output width to K supported answers and returns one long answer-class ID per question.","Do not apply softmax before CrossEntropyLoss: the loss expects raw logits. For prediction, softmax can help explain relative scores, while argmax selects a class."),sec("loop","Train and validate deliberately","Set `model.train()`, clear gradients, call `model(tokens, lengths)`, compute loss, call backward and optimizer.step. Validation uses `model.eval()` with `torch.inference_mode()` and must not update the optimizer.","The self-contained lab uses a local synthetic set, reproducible seed, train-only vocabulary and held-out paraphrase rows for a validation loss demonstration. It does not report accuracy as a benchmark or claim that two paraphrases per class establish generalization.")],
    ["CrossEntropyLoss expects raw logits and long targets.","Target shape is [B], not a token sequence or [B,1].","Training forward must retain autograd; evaluation can use inference_mode.","A tiny FAQ set can be memorized and cannot substantiate real-world performance."],
    "For a useful evaluation, collect diverse intents and out-of-domain questions, group paraphrases before splitting, use validation for model/threshold choices, and test once on untouched samples. Check class imbalance and per-class errors.",
    ["The output dimension K is the number of answer classes.","CrossEntropyLoss uses `[B,K]` logits and `[B]` long labels.","Zero gradients before backward, then step the optimizer.","Validation is separate and does not update weights."],
    [["Should training logits be passed through softmax first?","No. CrossEntropyLoss expects raw logits."],["What is the label shape for a batch?","[B] long class indices."],["Which model mode is used for optimization?","model.train()."],["Can the small lab's validation loss prove broad FAQ quality?","No; its small synthetic examples only exercise the pipeline."]]),
  lesson("faq-inference", "Load the mappings and decide when to abstain", "Restore the exact model artifact, inspect class probabilities and bound inference claims.", [172, 173, 178, 179], diagrams.demo, "faq-inference", "faq-checkpoint",
    [{name:"Question",shape:"text",meaning:"User input passed through the saved tokenizer."},{name:"Model scores",shape:"[1,K] logits",meaning:"Scores for supported answer categories."},{name:"Selected answer",shape:"answer string or abstention",meaning:"A mapped response, or a cautious fallback."}],
    ["Run inference with eval and inference_mode.","Save and restore model weights with vocabulary and answer mappings.","Treat confidence thresholds as heuristics, not guaranteed OOD detection."],
    "At the quiz desk, the model chooses among the cards it knows. The saved vocabulary and answer map are part of the desk setup; without them, the numeric choice has no meaning.",
    "pₖ=softmax(z)ₖ; k*=argmaxₖ pₖ; answer=answer_text[k*] if an abstention policy accepts the input. A fixed max-probability threshold is only a heuristic unless calibrated and tested on in-scope and out-of-scope data.",
    "For K=4, a probability vector such as [0.62,0.18,0.12,0.08] sums to one and selects class 0. This relative confidence does not prove that the question belongs to any supported class.",
    [sec("inference","Turn a question into one stored response","Tokenize using the saved normalization rule, map unknown words to UNK, add the batch dimension, call the model in eval/inference mode, and select the class. Map the class ID to stored response text.","The source compares maximum softmax probability to 0.5 and says “I don't know” below it. Softmax probabilities are relative to the known output classes; out-of-domain inputs can still receive a high maximum."),sec("artifact","Keep preprocessing beside weights","Save a state_dict together with vocabulary, PAD/UNK indices, answer-class mapping, tokenizer configuration and model dimensions. Restore with `map_location='cpu'` and a weights-only load path where supported, then verify the loaded mapping.","A stronger abstention system needs representative held-out out-of-scope examples and calibration or a retrieval/intent policy. A simple unknown-only token check catches some cases but cannot tell whether a known-word question is factually supported.")],
    ["Do not forget model.eval() and inference_mode at prediction time.","A saved state_dict without vocabulary/map is not a usable text model.","A confident softmax score does not establish factual correctness.","Unknown-token handling avoids crashes but does not solve OOD detection."],
    "Log model, vocabulary and answer-map versions. Add human escalation and approved answer provenance. Test empty input, all-OOV input, adversarial paraphrases, unsupported intents, and threshold behavior before deployment.",
    ["Inference reuses the exact training tokenizer and vocabulary.","Softmax scores sum to one over known classes.","Class ID must be decoded through the stored answer mapping.","Confidence thresholding needs calibration and does not guarantee rejection."],
    [["What context disables training-only layer behavior?","model.eval(), paired with torch.inference_mode() for prediction."],["What belongs beside model weights?","Vocabulary, tokenizer settings, answer-class map and architecture configuration."],["Does maximum softmax guarantee the question is known?","No; it only compares scores among the known classes."],["What does the source threshold do?","It is a heuristic abstention rule based on maximum softmax probability."]]),
];
phase10Lessons.find((item) => item.id === "faq-vocabulary-map")!.additionalSnippets = [C["faq-oov"]];
