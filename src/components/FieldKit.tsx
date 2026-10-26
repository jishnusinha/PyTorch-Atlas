import { useMemo, useState } from "react";
import { referenceRecipes, recipeById, recipeCategories, snippetById } from "../content/fieldKit";
import type { ReferenceRecipe } from "../content/types";
import { copyText, downloadPython } from "../utils/clipboard";
import { readStored, writeStored } from "../utils/storage";

function RecipeCode({ recipe }: { recipe: ReferenceRecipe }) {
  const [status, setStatus] = useState("");
  const snippet = snippetById.get(recipe.snippetId)!;
  return (
    <section className="field-code" aria-labelledby="recipe-code-title">
      <div className="field-code-heading">
        <div>
          <h3 id="recipe-code-title">{snippet.title}</h3>
          <p>{snippet.purpose}</p>
        </div>
        <div className="field-code-actions">
          <button onClick={async () => setStatus(await copyText(snippet.code)
            ? "Copied to clipboard"
            : "Copy unavailable. Select the code and copy manually.")}>Copy</button>
          {recipe.runnable && <button onClick={() => {
            downloadPython(snippet.id, snippet.code);
            setStatus("Python download prepared.");
          }}>Download .py</button>}
        </div>
      </div>
      <pre tabIndex={0} aria-label={`${recipe.title} Python code`}><code>{snippet.code}</code></pre>
      <p className="field-code-meta"><span>{recipe.dependencies}</span><span>{recipe.shapes}</span></p>
      <p className="run-note">{recipe.runEnvironment}</p>
      <p className="copy-status" role="status">{status}</p>
      <details className="field-lines">
        <summary>Why the steps work</summary>
        <ol>{snippet.lines.map((line) => <li key={line}>{line}</li>)}</ol>
      </details>
    </section>
  );
}

export function FieldKit({
  recipeId,
  navigate,
}: {
  recipeId?: string;
  navigate: (id: string) => void;
}) {
  const [category, setCategory] = useState("All recipes");
  const [query, setQuery] = useState("");
  const [tag, setTag] = useState("All tags");
  const [level, setLevel] = useState("All levels");
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [favorites, setFavorites] = useState<string[]>(() => readStored("field-kit-favorites", []));
  const selected = recipeId ? recipeById.get(recipeId) : undefined;
  const tags = useMemo(() => [...new Set(referenceRecipes.flatMap((recipe) => recipe.tags))].sort(), []);
  const filtered = useMemo(() => referenceRecipes.filter((recipe) => {
    const searchText = [recipe.title, recipe.category, ...recipe.tags, recipe.whenToUse, recipe.why, recipe.shapes, recipe.dependencies, snippetById.get(recipe.snippetId)!.expected, ...new Set(snippetById.get(recipe.snippetId)!.code.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [])].join(" ").toLowerCase();
    return (category === "All recipes" || recipe.category === category)
      && (!query || searchText.includes(query.toLowerCase()))
      && (tag === "All tags" || recipe.tags.includes(tag))
      && (level === "All levels" || recipe.level === level)
      && (!favoritesOnly || favorites.includes(recipe.id));
  }), [category, favorites, favoritesOnly, level, query, tag]);
  function toggleFavorite(id: string) {
    const next = favorites.includes(id) ? favorites.filter((favorite) => favorite !== id) : [...favorites, id];
    setFavorites(next);
    writeStored("field-kit-favorites", next);
  }
  return (
    <div className="field-kit">
      <div className="field-kit-intro">
        <span className="eyebrow">PYTORCH FIELD KIT · {referenceRecipes.length} RECIPES</span>
        <h1>{selected ? selected.title : "Working patterns, ready to adapt."}</h1>
        <p className="dek">{selected ? selected.whenToUse : "A compact library of tested PyTorch patterns, organized by the problem each one solves."}</p>
        {selected && <button className="field-back" onClick={() => navigate("field-kit")}>← Back to all recipes</button>}
      </div>
      {selected ? (
        <article className="recipe-detail">
          <div className="recipe-detail-top">
            <div><span className="recipe-category">{selected.category}</span><span className="recipe-level">{selected.level}</span></div>
            <button className="recipe-favorite" aria-pressed={favorites.includes(selected.id)} onClick={() => toggleFavorite(selected.id)}>{favorites.includes(selected.id) ? "★ Saved" : "☆ Save recipe"}</button>
          </div>
          <dl className="recipe-contract">
            <div><dt>Use it when</dt><dd>{selected.whenToUse}</dd></div>
            <div><dt>Why it works</dt><dd>{selected.why}</dd></div>
            <div><dt>Shape contract</dt><dd><code>{selected.shapes}</code></dd></div>
            <div><dt>Watch for</dt><dd>{selected.pitfalls.join(" ")}</dd></div>
            <div><dt>Expected behavior</dt><dd><code>{selected.expected}</code></dd></div>
            <div><dt>Source status</dt><dd>{selected.sourceKind === "source-derived" ? "Source-derived example" : "Supplemental example"}</dd></div>
          </dl>
          <RecipeCode key={selected.id} recipe={selected} />
          <button className="recipe-lesson-link" onClick={() => navigate(selected.relatedLessonId)}>Read the related lesson →</button>
        </article>
      ) : (
        <div className="field-browser">
          <nav className="field-categories" aria-label="Recipe categories">
            <button className={category === "All recipes" ? "active" : ""} aria-pressed={category === "All recipes"} onClick={() => setCategory("All recipes")}>All recipes <span>{referenceRecipes.length}</span></button>
            {recipeCategories.map((name) => <button key={name} className={category === name ? "active" : ""} aria-pressed={category === name} onClick={() => setCategory(name)}>{name}<span>{referenceRecipes.filter((recipe) => recipe.category === name).length}</span></button>)}
          </nav>
          <section className="field-results" aria-label="Recipe library">
            <div className="field-filters">
              <label>Search recipes<input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="loss, tensor shape, validation…" /></label>
              <label>Tag<select value={tag} onChange={(event) => setTag(event.target.value)}><option>All tags</option>{tags.map((item) => <option key={item}>{item}</option>)}</select></label>
              <label>Level<select value={level} onChange={(event) => setLevel(event.target.value)}><option>All levels</option><option>Foundations</option><option>Applied</option><option>Advanced</option></select></label>
              <button className="favorite-filter" aria-pressed={favoritesOnly} onClick={() => setFavoritesOnly(!favoritesOnly)}>{favoritesOnly ? "★ Favorites" : "☆ Favorites"}</button>
            </div>
            <p className="field-count" aria-live="polite">{filtered.length} {filtered.length === 1 ? "recipe" : "recipes"}</p>
            <div className="field-recipe-list">
              {filtered.map((recipe) => <article key={recipe.id} className="field-recipe-row">
                <button className="field-recipe-open" onClick={() => navigate(`field-kit/${recipe.id}`)}>
                  <span className="recipe-category">{recipe.category} · {recipe.level}</span>
                  <strong>{recipe.title}</strong>
                  <span>{recipe.whenToUse}</span>
                  <code>{recipe.shapes}</code>
                </button>
                <button className="field-row-favorite" aria-label={`${favorites.includes(recipe.id) ? "Remove" : "Save"} ${recipe.title}`} aria-pressed={favorites.includes(recipe.id)} onClick={() => toggleFavorite(recipe.id)}>{favorites.includes(recipe.id) ? "★" : "☆"}</button>
              </article>)}
              {!filtered.length && <p className="empty">No recipes match these filters. Clear a tag or search term to see more.</p>}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
