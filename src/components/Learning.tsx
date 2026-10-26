import { createElement, useState } from "react";
import Prism from "prismjs";
import "prismjs/components/prism-python";
import type {
  CodeAnnotation,
  Lesson,
  RevisionQuestion,
  Snippet,
  SourceReference,
} from "../content/types";
import { codeAnnotations } from "../content/codeAnnotations";
import { copyText, downloadPython } from "../utils/clipboard";
function tokenText(token: string | Prism.Token | (string | Prism.Token)[]): string {
  if (typeof token === "string") return token;
  if (Array.isArray(token)) return token.map(tokenText).join("");
  return tokenText(token.content);
}
export function CodePanel({ snippet }: { snippet: Snippet }) {
  const [message, setMessage] = useState("");
  const [focused, setFocused] = useState<string | null>(null);
  const annotations = codeAnnotations(snippet);
  const annotationByLine = new Map<number, CodeAnnotation>();
  for (const annotation of annotations)
    for (let line = annotation.startLine; line <= annotation.endLine; line++)
      annotationByLine.set(line, annotation);
  const lines = snippet.code.split("\n");
  return (
    <div className="code-panel">
      <div className="code-title">
        <span>
          <span className="code-dot" /> {snippet.title}
        </span>
        <div>
          <button
            onClick={async () =>
              setMessage(
                (await copyText(snippet.code))
                  ? "Copied to clipboard"
                  : "Copy unavailable. Select the code and copy manually.",
              )
            }
          >
            Copy
          </button>
          <button
            onClick={() => {
              downloadPython(snippet.id, snippet.code);
              setMessage("Python download prepared.");
            }}
          >
            Download .py
          </button>
        </div>
      </div>
      <p className="code-purpose">{snippet.purpose}</p>
      <pre tabIndex={0} aria-label={snippet.title + " Python code"}>
        <code>{lines.map((line, index) => {
          const number = index + 1;
          const annotation = annotationByLine.get(number);
          const tokens = Prism.tokenize(line, Prism.languages.python);
          return <span key={number} id={annotation?.id} tabIndex={annotation ? -1 : undefined} className={`code-line${annotation && focused === annotation.id ? " is-focused" : ""}`}>
            <span className="line-number" aria-hidden="true">{number}</span>
            <span className="line-source">{tokens.map((token, i) => typeof token === "string" ? token : <span key={i} className={`token ${token.type}`}>{tokenText(token)}</span>) || " "}</span>
            {line === "" ? " " : ""}
          </span>;
        })}</code>
      </pre>
      <div className="code-meta">
        <span>{snippet.dependencies}</span>
        <code>{snippet.shapes}</code>
      </div>
      <p className="run-note">
        Run in a local Python file or a Colab Python cell with PyTorch installed. Python
        does not execute in this reader.
      </p>
      <p className="copy-status" role="status">
        {message}
      </p>
      <section className="line-guide" aria-label="Annotated code explanation">
        {annotations.length ? <>
          <h4>Why this line matters</h4>
          <ol className="annotation-list">
            {annotations.map((annotation) => <li key={annotation.id}>
              <button className="annotation-focus" aria-pressed={focused === annotation.id} onClick={() => {
                setFocused(annotation.id);
                document.getElementById(annotation.id)?.focus({ preventScroll: true });
              }}>Focus lines {annotation.startLine}{annotation.endLine !== annotation.startLine ? `–${annotation.endLine}` : ""}</button>
              <p><strong>Why:</strong> {annotation.why}</p>
              <p><strong>Shape:</strong> {annotation.inputShape} → {annotation.outputShape}</p>
              {annotation.gotcha && <p><strong>Gotcha:</strong> {annotation.gotcha}</p>}
            </li>)}
          </ol>
        </> : <details>
          <summary>Read the code, line by line</summary>
          <ol>{snippet.lines.map((line) => <li key={line}>{line}</li>)}</ol>
        </details>}
      </section>
      <div className="expected">
        <span className="eyebrow">EXPECTED OUTPUT</span>
        <pre tabIndex={0}>{snippet.expected}</pre>
      </div>
    </div>
  );
}
function mathTokens(value: string) {
  return value.match(/[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|→|←|≤|≥|≠|[+*=×·−\-/()[\]{}.,:;|]|\s+|[^\s]/gu) ?? [value];
}
export function Equation({ children }: { children: string }) {
  const tokens = mathTokens(children);
  const renderedTokens = tokens.map((token, index) => /^\s+$/u.test(token)
    ? createElement("mspace", { key: index, width: "0.25em" })
    : /^\d/u.test(token)
      ? createElement("mn", { key: index }, token)
      : /^[A-Za-z_]/u.test(token)
        ? createElement("mi", { key: index }, token)
        : createElement("mo", { key: index }, token));
  const math = createElement("math", { xmlns: "http://www.w3.org/1998/Math/MathML", display: "block", tabIndex: 0, "aria-label": "Mathematical expression: " + children }, createElement("mrow", null, renderedTokens));
  return (
    <div className="equation">
      {math}
      <p className="equation-fallback"><span className="visually-hidden">Plain text equivalent: </span>{children}</p>
      <details className="symbol-legend"><summary>Equation symbols</summary><p>Arrows show tensor flow; brackets name dimensions; ×, + and − mean multiply, add and subtract; subscripts and suffixes identify steps, gates or classes.</p></details>
    </div>
  );
}

export function LearningLenses() {
  return <nav className="learning-lenses" aria-label="Learning lenses">
    <span>LEARNING LENSES</span>
    <a href="#understand">Intuition</a><a href="#math">Math</a><a href="#code">Python</a><a href="#pitfalls">Engineering</a>
  </nav>;
}

export function LearningCheck({ worked, shapePath, practiceSnippet }: { worked: string; shapePath: string; practiceSnippet: string }) {
  return <aside className="learning-check">
    <span className="eyebrow">WORKED CHECK · SHAPE WALK</span>
    <p>{worked}</p>
    <code>{shapePath}</code>
    <a href="#code" aria-label={`Practice with ${practiceSnippet}; go to the runnable example`}>Open the runnable example ↓</a>
  </aside>;
}

export function ComparisonNote({ title, source, sourcePages, repair, label }: { title: string; source: string; sourcePages: string; repair: string; label: string }) {
  return <aside className="comparison-note">
    <span className="eyebrow">{label}</span>
    <h3>{title}</h3>
    <p><strong>Source example · {sourcePages}.</strong> {source}</p>
    <p><strong>Engineering context.</strong> {repair}</p>
  </aside>;
}
export function IntuitionPanel({ children }: { children: string }) {
  return (
    <aside className="intuition">
      <span className="eyebrow">THINK OF IT THIS WAY</span>
      <p>{children}</p>
    </aside>
  );
}
export function ShapeTrace({ shapes }: { shapes: Lesson["shapes"] }) {
  return (
    <div className="shape-scroll">
      <table className="shape-table">
        <caption>Follow the tensor shapes</caption>
        <thead>
          <tr>
            <th scope="col">Tensor</th>
            <th scope="col">Shape</th>
            <th scope="col">What it means</th>
          </tr>
        </thead>
        <tbody>
          {shapes.map((s) => (
            <tr key={s.name}>
              <th scope="row">{s.name}</th>
              <td>
                <code>{s.shape}</code>
              </td>
              <td>{s.meaning}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export function SourcePage({ source }: { source: SourceReference }) {
  return (
    <div className="source-note">
      <span>
        PDF pp. {source.start}–{source.end}
      </span>
      <span>
        {source.kind === "supplemental" ? "Supplemental engineering" : "PDF-derived"}
      </span>
      {source.note && (
        <p>
          <strong>Editorial additions.</strong> {source.note}
        </p>
      )}
    </div>
  );
}
export function RevisionCard({ items }: { items: string[] }) {
  return (
    <aside className="revision-card">
      <span className="eyebrow">KEEP THIS MENTAL MODEL</span>
      <h3>The one-minute revision</h3>
      <ul>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </aside>
  );
}
export function Quiz({ questions }: { questions: RevisionQuestion[] }) {
  return (
    <div className="quiz">
      {questions.map((q, i) => (
        <details key={q.question}>
          <summary>
            <span>{String(i + 1).padStart(2, "0")}</span>
            {q.question}
          </summary>
          <p>{q.answer}</p>
        </details>
      ))}
    </div>
  );
}
