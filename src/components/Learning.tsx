import { useState } from "react";
import Prism from "prismjs";
import "prismjs/components/prism-python";
import type {
  Lesson,
  RevisionQuestion,
  Snippet,
  SourceReference,
} from "../content/types";
import { copyText, downloadPython } from "../utils/clipboard";
function tokenText(token: string | Prism.Token | (string | Prism.Token)[]): string {
  if (typeof token === "string") return token;
  if (Array.isArray(token)) return token.map(tokenText).join("");
  return tokenText(token.content);
}
export function CodePanel({ snippet }: { snippet: Snippet }) {
  const [message, setMessage] = useState("");
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
        <code>
          {Prism.tokenize(snippet.code, Prism.languages.python).map((token, i) =>
            typeof token === "string" ? (
              token
            ) : (
              <span key={i} className={`token ${token.type}`}>
                {tokenText(token)}
              </span>
            ),
          )}
        </code>
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
      <details className="line-guide">
        <summary>Read the code, line by line</summary>
        <ol>
          {snippet.lines.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ol>
      </details>
      <div className="expected">
        <span className="eyebrow">EXPECTED OUTPUT</span>
        <pre tabIndex={0}>{snippet.expected}</pre>
      </div>
    </div>
  );
}
export function Equation({ children }: { children: string }) {
  return (
    <div className="equation" aria-label={"Equation: " + children}>
      {children}
    </div>
  );
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
        {source.kind === "supplemental" ? "Supplemental" : "Source-derived concept"}
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
