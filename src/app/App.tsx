import { useEffect, useRef, useState } from "react";
import { lessons, modules } from "../content/lessons";
import { readStored, writeStored } from "../utils/storage";
import {
  CodePanel,
  ComparisonNote,
  Equation,
  LearningCheck,
  LearningLenses,
  Quiz,
  RevisionCard,
  ShapeTrace,
  SourcePage,
} from "../components/Learning";
import { Search } from "../components/Search";
import { ConceptDiagram } from "../diagrams/ConceptDiagram";
import { LessonSections, lessonToc } from "../components/LessonSections";
import coverage from "../../docs/source-coverage.json";
import { learningCheckByLesson } from "../content/learningQuality";
import { comparisonByLesson } from "../content/comparisons";
const valid = (id: string) =>
  lessons.some((l) => l.id === id) || ["reference", "coverage", "bookmarks"].includes(id);
function initialPage() {
  let hash = "";
  try {
    hash = decodeURIComponent(location.hash.slice(1));
  } catch {
    /* Invalid URL fragment. */
  }
  const saved = readStored("atlas-last", "autograd-intuition");
  return valid(hash) ? hash : valid(saved) ? saved : "autograd-intuition";
}
function SymbolMark() {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true">
      <path
        d="M5 25 16 6 27 25M10 18h12M16 6v19"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
      />
      <circle cx="16" cy="6" r="3" fill="currentColor" />
      <circle cx="5" cy="25" r="2.4" fill="currentColor" />
      <circle cx="27" cy="25" r="2.4" fill="currentColor" />
    </svg>
  );
}
export function App() {
  const [page, setPage] = useState(initialPage);
  const [theme, setTheme] = useState(() =>
    readStored<"light" | "dark">("atlas-theme", "light"),
  );
  const [completed, setCompleted] = useState<string[]>(() =>
    readStored("atlas-completed", []),
  );
  const [bookmarks, setBookmarks] = useState<string[]>(() =>
    readStored("atlas-bookmarks", []),
  );
  const [search, setSearch] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const [reading, setReading] = useState(0);
  const [active, setActive] = useState("understand");
  const main = useRef<HTMLElement>(null);
  const menu = useRef<HTMLButtonElement>(null);
  const mobileNav = useRef<HTMLElement>(null);
  const navigated = useRef(false);
  const lesson = lessons.find((l) => l.id === page);
  const module = modules.find((m) => m.id === lesson?.moduleId);
  const index = lessons.findIndex((l) => l.id === page);
  function navigate(id: string) {
    if (!valid(id)) return;
    navigated.current = true;
    setPage(id);
    location.hash = id;
    setDrawer(false);
    window.scrollTo({ top: 0, behavior: "instant" });
  }
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    writeStored("atlas-theme", theme);
  }, [theme]);
  useEffect(() => writeStored("atlas-completed", completed), [completed]);
  useEffect(() => writeStored("atlas-bookmarks", bookmarks), [bookmarks]);
  useEffect(() => {
    if (lesson) writeStored("atlas-last", page);
    if (navigated.current) main.current?.focus();
    setReading(0);
    setActive("understand");
  }, [page, lesson]);
  useEffect(() => {
    function hash() {
      let id = "";
      try {
        id = decodeURIComponent(location.hash.slice(1));
      } catch {
        return;
      }
      if (valid(id)) {
        setPage(id);
        window.scrollTo(0, 0);
      }
    }
    window.addEventListener("hashchange", hash);
    return () => window.removeEventListener("hashchange", hash);
  }, []);
  useEffect(() => {
    function key(e: KeyboardEvent) {
      const target = e.target as HTMLElement;
      const editing = target.matches("input,textarea,select") || target.isContentEditable;
      if ((e.key === "/" && !editing) || ((e.ctrlKey || e.metaKey) && e.key === "k")) {
        e.preventDefault();
        setSearch(true);
      }
    }
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);
  useEffect(() => {
    function scroll() {
      const remaining = document.documentElement.scrollHeight - innerHeight;
      setReading(
        remaining > 0 ? Math.min(100, Math.round((scrollY / remaining) * 100)) : 100,
      );
      const visible = (lesson ? lessonToc(lesson) : [])
        .map(({ id }) => document.getElementById(id))
        .filter((el): el is HTMLElement => !!el && el.getBoundingClientRect().top < 180);
      if (visible.length) setActive(visible[visible.length - 1].id);
    }
    window.addEventListener("scroll", scroll, { passive: true });
    scroll();
    return () => window.removeEventListener("scroll", scroll);
  }, [page]);
  useEffect(() => {
    if (!drawer) return;
    const previous = document.activeElement as HTMLElement | null;
    const nav = mobileNav.current;
    nav?.querySelector<HTMLElement>("button,a")?.focus();
    function trap(e: KeyboardEvent) {
      if (e.key === "Escape") {
        setDrawer(false);
        return;
      }
      if (e.key !== "Tab") return;
      const nodes = nav?.querySelectorAll<HTMLElement>("button:not(:disabled),a,summary");
      if (!nodes?.length) return;
      const first = nodes[0],
        last = nodes[nodes.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", trap);
    const oldOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = oldOverflow;
      document.removeEventListener("keydown", trap);
      previous?.focus();
    };
  }, [drawer]);
  function toggle(id: string, items: string[], set: (v: string[]) => void) {
    set(items.includes(id) ? items.filter((x) => x !== id) : [...items, id]);
  }
  const nav = (
    <>
      <div className="nav-heading">
        <span>THE HANDBOOK</span>
        <button
          className="mobile-close"
          onClick={() => setDrawer(false)}
          aria-label="Close navigation"
        >
          ×
        </button>
      </div>
      <div className="module-list">
        {modules.map((m) => (
          <details
            key={m.id}
            open={m.lessons.some((l) => l.ready) || undefined}
            className={m.id === module?.id ? "current-module" : ""}
          >
            <summary>
              <span className="module-number">{m.number}</span>
              <span>{m.title}</span>
              <span className="chevron">⌄</span>
            </summary>
            <div className="lesson-list">
              {m.lessons.map((l) =>
                l.ready ? (
                  <button
                    key={l.id}
                    className={page === l.id ? "active" : ""}
                    aria-current={page === l.id ? "page" : undefined}
                    onClick={() => navigate(l.id)}
                  >
                    <span aria-hidden="true">{completed.includes(l.id) ? "✓" : "·"}</span>
                    {l.title}
                  </button>
                ) : (
                  <span className="pending" key={l.id}>
                    {l.title}
                    <small>PLANNED</small>
                  </span>
                ),
              )}
            </div>
          </details>
        ))}
      </div>
      <div className="nav-tools">
        <button
          onClick={() => navigate("reference")}
          aria-current={page === "reference" ? "page" : undefined}
        >
          ▤ <span>Quick reference</span>
          <span>↗</span>
        </button>
        <button
          onClick={() => navigate("bookmarks")}
          aria-current={page === "bookmarks" ? "page" : undefined}
        >
          ♧ <span>Saved lessons</span>
          <span>{bookmarks.length}</span>
        </button>
        <button
          onClick={() => navigate("coverage")}
          aria-current={page === "coverage" ? "page" : undefined}
        >
          ◎ <span>Source coverage</span>
          <span>↗</span>
        </button>
      </div>
      <div className="nav-progress">
        <div>
          <strong>Your learning journey</strong>
          <span>
            {completed.length} / {lessons.length}
          </span>
        </div>
        <progress
          max={lessons.length}
          value={completed.length}
          aria-label="Completed lessons"
        />
        <p>12 modules · 94 lessons</p>
      </div>
      <div className="offline">
        <i /> Available entirely offline
      </div>
    </>
  );
  return (
    <>
      <a className="skip-link" href="#reader">
        Skip to lesson
      </a>
      <header className="topbar">
        <button
          className="brand"
          onClick={() => navigate("autograd-intuition")}
          aria-label="PyTorch Atlas home"
        >
          <SymbolMark />
          <span>
            PyTorch <strong>Atlas</strong>
            <small>A FIELD GUIDE TO DEEP LEARNING</small>
          </span>
        </button>
        <div className="header-middle">
          Understand deeply.<span> Build with confidence.</span>
        </div>
        <div className="header-actions">
          <button
            className="search-trigger"
            onClick={() => setSearch(true)}
            aria-label="Search lessons"
          >
            <span aria-hidden="true">⌕</span>
            <span>Search the atlas</span>
            <kbd>Ctrl K</kbd>
          </button>
          <button
            className="theme-button"
            onClick={() => setTheme(theme === "light" ? "dark" : "light")}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
          >
            {theme === "light" ? "☾" : "☀"}
          </button>
          <button
            ref={menu}
            className="menu-button"
            aria-label="Open navigation"
            aria-expanded={drawer}
            onClick={() => setDrawer(true)}
          >
            ☰
          </button>
        </div>
      </header>
      <div className="reading-track">
        <div style={{ width: `${reading}%` }} />
      </div>
      <aside className="sidebar" aria-label="Handbook navigation">
        {nav}
      </aside>
      {drawer && (
        <div className="drawer-scrim" onClick={() => setDrawer(false)}>
          <aside
            ref={mobileNav}
            className="mobile-drawer"
            role="dialog"
            aria-modal="true"
            aria-label="Handbook navigation"
            onClick={(e) => e.stopPropagation()}
          >
            {nav}
          </aside>
        </div>
      )}
      <div className="reader-layout">
        <main id="reader" ref={main} tabIndex={-1}>
          {lesson ? (
            <>
              <div className="breadcrumbs">
                <button onClick={() => navigate("reference")}>The handbook</button>
                <span>/</span>
                <span>{module?.title}</span>
              </div>
              <div className="lesson-meta">
                <span className="eyebrow">
                  {module?.number} / {module?.title.toUpperCase()}
                </span>
                <span>
                  {lesson.minutes} min read ·{" "}
                  {lesson.source.kind === "supplemental"
                    ? "Supplemental lesson"
                    : "Authored lesson"}
                </span>
              </div>
              <h1>{lesson.title}</h1>
              <p className="dek">{lesson.subtitle}</p>
              <div className="article-toolbar">
                <SourcePage source={lesson.source} />
                <button
                  className={bookmarks.includes(page) ? "bookmark saved" : "bookmark"}
                  onClick={() => toggle(page, bookmarks, setBookmarks)}
                  aria-pressed={bookmarks.includes(page)}
                  aria-label={
                    bookmarks.includes(page) ? "Remove bookmark" : "Bookmark lesson"
                  }
                >
                  {bookmarks.includes(page) ? "◆ Saved" : "◇ Save"}
                </button>
              </div>
              <div className="learning-goals">
                <span className="eyebrow">BY THE END, YOU’LL BE ABLE TO</span>
                <ul>
                  {lesson.objectives.map((goal) => (
                    <li key={goal}>{goal}</li>
                  ))}
                </ul>
              </div>
              <LearningLenses />
              <LessonSections lesson={lesson} />
              <section id="visual">
                <h2>
                  <span>{String(lesson.sections.length + 1).padStart(2, "0")}</span>
                  Make the idea tangible
                </h2>
                <ConceptDiagram key={page} spec={lesson.diagram} />
              </section>
              <section id="math">
                <h2>
                  <span>{String(lesson.sections.length + 2).padStart(2, "0")}</span>
                  Work the numbers
                </h2>
                <Equation>{lesson.equation}</Equation>
                <p>{lesson.numerical}</p>
                <ShapeTrace shapes={lesson.shapes} />
                {learningCheckByLesson.get(lesson.id) && <LearningCheck {...learningCheckByLesson.get(lesson.id)!} />}
                {comparisonByLesson.get(lesson.id) && <ComparisonNote {...comparisonByLesson.get(lesson.id)!} />}
              </section>
              <section id="code">
                <h2>
                  <span>{String(lesson.sections.length + 3).padStart(2, "0")}</span>
                  Try it in Python
                </h2>
                <p>
                  A complete, CPU-first example. Copy it into your own Python environment;
                  no data download is needed.
                </p>
                <CodePanel key={lesson.snippet.id} snippet={lesson.snippet} />
              </section>
              <section id="template">
                <h2>
                  <span>{String(lesson.sections.length + 4).padStart(2, "0")}</span>
                  A pattern to take with you
                </h2>
                <p className="supplement-label">
                  Supplemental · Reusable engineering template
                </p>
                <CodePanel key={lesson.template.id} snippet={lesson.template} />
                {lesson.additionalSnippets?.map((snippet) => (
                  <CodePanel key={snippet.id} snippet={snippet} />
                ))}
              </section>
              <section id="pitfalls">
                <h2>
                  <span>{String(lesson.sections.length + 5).padStart(2, "0")}</span>
                  Small mistakes, useful clues
                </h2>
                <ul className="pitfalls">
                  {lesson.pitfalls.map((p) => (
                    <li key={p}>{p}</li>
                  ))}
                </ul>
                <aside className="production">
                  <strong>In practice</strong>
                  <p>{lesson.production}</p>
                </aside>
              </section>
              <section id="revision">
                <RevisionCard items={lesson.revision} />
              </section>
              <section id="quiz">
                <h2>
                  <span>{String(lesson.sections.length + 6).padStart(2, "0")}</span>
                  Close the notes. Check yourself.
                </h2>
                <p>Try an answer before opening each question.</p>
                <Quiz key={page} questions={lesson.questions} />
              </section>
              <div className="complete-row">
                <label>
                  <input
                    type="checkbox"
                    checked={completed.includes(page)}
                    onChange={() => toggle(page, completed, setCompleted)}
                  />
                  <span>
                    <strong>
                      {completed.includes(page)
                        ? "Lesson completed"
                        : "I understand this lesson"}
                    </strong>
                    <small>Your progress is saved on this device.</small>
                  </span>
                </label>
                <span>{completed.includes(page) ? "✓ Nicely done" : "Keep going →"}</span>
              </div>
              <nav className="prev-next" aria-label="Lesson pagination">
                {index > 0 ? (
                  <button onClick={() => navigate(lessons[index - 1].id)}>
                    <small>← PREVIOUS</small>
                    {lessons[index - 1].title}
                  </button>
                ) : (
                  <div>
                    <small>YOU’RE AT THE START</small>A good place to begin.
                  </div>
                )}
                {index < lessons.length - 1 ? (
                  <button onClick={() => navigate(lessons[index + 1].id)}>
                    <small>NEXT LESSON →</small>
                    {lessons[index + 1].title}
                  </button>
                ) : (
                  <button onClick={() => navigate("reference")}>
                    <small>NEXT →</small>Review what you’ve learned
                  </button>
                )}
              </nav>
            </>
          ) : (
            <>
              <div className="breadcrumbs">
                <button onClick={() => navigate("autograd-intuition")}>
                  The handbook
                </button>
                <span>/</span>
                <span>
                  {page === "coverage"
                    ? "Source coverage"
                    : page === "bookmarks"
                      ? "Saved lessons"
                      : "Quick reference"}
                </span>
              </div>
              <span className="eyebrow">YOUR STUDY COMPANION</span>
              <h1>
                {page === "coverage"
                  ? "A source you can trace."
                  : page === "bookmarks"
                    ? "Keep the useful ideas close."
                    : "Less searching. More remembering."}
              </h1>
              <p className="dek">
                {page === "coverage"
                  ? `211 PDF pages mapped across ${coverage.blocks.length} source blocks and ${lessons.length} authored lessons. See each page span, supplemental item and source repair below.`
                  : page === "bookmarks"
                    ? "Your saved lessons stay on this device, ready for another look."
                    : "Core ideas from the authored lessons, in one place."}
              </p>
              {page === "coverage" ? (
                <div className="coverage-list">
                  <p>
                    The source is an export of notes and code. Each indexed topic links to
                    its authored lesson; supplemental engineering context and corrections
                    to clipped or inaccurate examples are listed separately.
                  </p>
                  {coverage.blocks.map((b) => (
                    <details key={b.id}>
                      <summary>
                        <span>{b.title}</span>
                        <small>
                          PDF pp. {b.start}–{b.end}
                        </small>
                      </summary>
                      <ul>
                        {b.topics.map((t, i) => (
                          <li key={i}>
                            <strong>{t.topic}</strong>
                            <span>
                              PDF pp. {t.start}–{t.end} · {t.status} · {t.kind}
                            </span>
                            <code>{t.lesson}</code>
                            {lessons.some((l) => l.id === t.lesson) && (
                              <button onClick={() => navigate(t.lesson)}>
                                Open lesson →
                              </button>
                            )}
                          </li>
                        ))}
                      </ul>
                    </details>
                  ))}
                  <h2>Editorial supplements</h2>
                  {coverage.supplements.map((s) => (
                    <p key={s.topic}>
                      <strong>{s.topic}</strong> · {s.status} · Supplemental
                    </p>
                  ))}
                  <h2>Source issues and corrections</h2>
                  {coverage.corrections.map((item) => (
                    <p key={item.pages}>
                      <span className="source-correction-label">Corrected source example</span>{" "}
                      <strong>{item.pages}</strong> · {item.issue} {item.correction}
                    </p>
                  ))}
                </div>
              ) : page === "bookmarks" ? (
                <div className="saved-list">
                  {bookmarks.length ? (
                    lessons
                      .filter((l) => bookmarks.includes(l.id))
                      .map((l) => (
                        <button key={l.id} onClick={() => navigate(l.id)}>
                          <small>{l.minutes} MIN READ</small>
                          <strong>{l.title}</strong>
                          <span>Continue reading →</span>
                        </button>
                      ))
                  ) : (
                    <div className="empty">
                      <strong>A place for ideas worth returning to.</strong>
                      <p>Use Save at the top of a lesson to keep it here.</p>
                      <button onClick={() => navigate("autograd-intuition")}>
                        Start reading →
                      </button>
                    </div>
                  )}
                </div>
              ) : (
                <div className="reference-list">
                  <section className="phase-note">
                    <h2>Autograd · one-screen cheat sheet</h2>
                    <ul>
                      <li>
                        Create floating-point parameters with{" "}
                        <code>requires_grad=True</code>.
                      </li>
                      <li>
                        Run the forward pass and reduce the objective to a scalar loss.
                      </li>
                      <li>
                        Call <code>loss.backward()</code>; gradients accumulate in leaf{" "}
                        <code>.grad</code>.
                      </li>
                      <li>
                        Clear gradients between independent steps, then update under{" "}
                        <code>no_grad()</code> or an optimizer.
                      </li>
                      <li>
                        For prediction use <code>model.eval()</code> with{" "}
                        <code>inference_mode()</code>; use <code>detach()</code> to cut
                        graph history.
                      </li>
                    </ul>
                  </section>
                  {lessons.map((l) => (
                    <section key={l.id}>
                      <h2>{l.title}</h2>
                      <Equation>{l.equation}</Equation>
                      <RevisionCard items={l.revision} />
                      <button onClick={() => navigate(l.id)}>Read the lesson →</button>
                    </section>
                  ))}
                  <div className="phase-note">
                    <strong>The atlas is taking shape.</strong>
                    <p>
                      All twelve learning modules are available in the lesson navigation.
                      Use search or the source index to move directly to a topic.
                    </p>
                  </div>
                </div>
              )}
            </>
          )}
          <footer className="article-footer">
            <SymbolMark />
            <span>
              PyTorch Atlas <small>Built for understanding. Ready for revision.</small>
            </span>
            <span>OFFLINE EDITION</span>
          </footer>
        </main>
        <aside className="toc" aria-label="On this page">
          {lesson ? (
            <>
              <span className="eyebrow">ON THIS PAGE</span>
              <nav>
              {(lesson ? lessonToc(lesson) : []).map(({ id, title }) => (
                  <a
                    key={id}
                    href={`#${id}`}
                    aria-current={active === id ? "location" : undefined}
                    className={active === id ? "active" : ""}
                    onClick={(e) => {
                      e.preventDefault();
                      document.getElementById(id)?.scrollIntoView({
                        behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
                          ? "instant"
                          : "smooth",
                      });
                    }}
                  >
                    {title}
                  </a>
                ))}
              </nav>
              <div className="toc-note">
                <span>LEARN BY DOING</span>
                <p>
                  Move the controls.
                  <br />
                  Trace the shapes.
                  <br />
                  Then write the code.
                </p>
              </div>
              <span className="reading-label">{reading}% of this page</span>
            </>
          ) : (
            <div className="toc-note">
              <span>AN EVOLVING HANDBOOK</span>
              <p>
                Read deeply.
                <br />
                Return often.
              </p>
              <button onClick={() => navigate(readStored("atlas-last", lessons[0].id))}>
                Resume reading →
              </button>
            </div>
          )}
        </aside>
      </div>
      {search && <Search close={() => setSearch(false)} navigate={navigate} />}
    </>
  );
}
