import { useEffect, useRef, useState } from "react";
import { searchable } from "../content/lessons";
import { recipeSearchable } from "../content/fieldKit";
import { buildProjects, projectSearchText } from "../content/buildStudio";
export function Highlight({ text, query }: { text: string; query: string }) {
  if (!query.trim()) return <>{text}</>;
  const at = text.toLowerCase().indexOf(query.toLowerCase());
  return at < 0 ? (
    <>{text}</>
  ) : (
    <>
      {text.slice(0, at)}
      <mark>{text.slice(at, at + query.length)}</mark>
      {text.slice(at + query.length)}
    </>
  );
}
export function Search({
  close,
  navigate,
}: {
  close: () => void;
  navigate: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const [selected, setSelected] = useState(0);
  const results = [
    ...searchable.map((result) => ({ ...result, kind: "lesson" as const })),
    ...recipeSearchable,
    ...buildProjects.map((project) => ({ id:`build-studio/${project.id}`, title:project.title, path:"Build Studio", kind:"project" as const, text:projectSearchText(project) })),
  ].filter((result) => query.trim().toLowerCase().split(/\s+/).filter(Boolean).every((term) => result.text.toLowerCase().includes(term)));
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    dialog.current?.showModal();
    input.current?.focus();
    return () => previous?.focus();
  }, []);
  function choose(id: string) {
    navigate(id);
    close();
  }
  return (
    <dialog
      className="search-dialog"
      aria-label="Search the atlas"
      ref={dialog}
      onCancel={(e) => {
        e.preventDefault();
        close();
      }}
      onClick={(e) => {
        if (e.target === dialog.current) close();
      }}
      onKeyDown={(e) => {
        if (e.key === "ArrowDown") {
          e.preventDefault();
          setSelected((v) => Math.min(results.length - 1, v + 1));
        }
        if (e.key === "ArrowUp") {
          e.preventDefault();
          setSelected((v) => Math.max(0, v - 1));
        }
        if (e.key === "Enter" && results[selected]) {
          e.preventDefault();
          choose(results[selected].id);
        }
      }}
    >
      <div className="search-top">
        <label htmlFor="global-search">Search the atlas</label>
        <button onClick={close} aria-label="Close search">
          Esc
        </button>
      </div>
      <input
        id="global-search"
        ref={input}
        placeholder="Try gradients, batches, or backward…"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setSelected(0);
        }}
        aria-controls="search-results"
      />
      <p className="search-hint">
        Search lessons, explanations, and Python examples across the atlas.
      </p>
      <div id="search-results" className="search-results" aria-live="polite">
        {results.length ? (
          results.map((r, i) => {
            const index = r.text.toLowerCase().indexOf(query.toLowerCase());
            const excerpt = r.text.slice(
              Math.max(0, index - 30),
              Math.max(0, index - 30) + 180,
            );
            return (
              <button
                key={r.id}
                className={i === selected ? "selected" : ""}
                onClick={() => choose(r.id)}
              >
                <small>{r.path}{r.kind === "recipe" ? " · RECIPE" : r.kind === "project" ? " · PROJECT" : " · LESSON"}</small>
                <strong>
                  <Highlight text={r.title} query={query} />
                </strong>
                <span>
                  <Highlight text={excerpt} query={query} />
                </span>
                <b aria-hidden="true">↗</b>
              </button>
            );
          })
        ) : (
          <div className="empty">
            <strong>No matching lessons yet.</strong>
            <p>
              Try “gradient”, “loss”, or “batch”, or search for an operation in a code
              example.
            </p>
          </div>
        )}
      </div>
      <footer>
        ↑ ↓ to explore <span>↵ to open</span>
      </footer>
    </dialog>
  );
}
