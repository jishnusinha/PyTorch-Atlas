import type { Lesson } from "../content/types";
import { IntuitionPanel } from "./Learning";

export interface LessonTocEntry {
  id: string;
  title: string;
}

const fixedToc: LessonTocEntry[] = [
  ["visual", "See it in motion"],
  ["math", "Work the numbers"],
  ["code", "Try it in Python"],
  ["template", "Reusable template"],
  ["pitfalls", "Watch out for"],
  ["revision", "Quick revision"],
  ["quiz", "Check yourself"],
].map(([id, title]) => ({ id, title }));

export function authoredSectionAnchor(lesson: Lesson, index: number) {
  if (index === 0) return "understand";
  if (index === 1) return "mechanics";
  return `lesson-${lesson.id}-${lesson.sections[index].id}`;
}

export function lessonToc(lesson: Lesson): LessonTocEntry[] {
  const authored = lesson.sections.map((section, index) => ({
    id: authoredSectionAnchor(lesson, index),
    title: section.title,
  }));
  return [...authored, ...fixedToc];
}

export function LessonSections({ lesson }: { lesson: Lesson }) {
  return (
    <>
      {lesson.sections.map((section, index) => (
        <section id={authoredSectionAnchor(lesson, index)} key={section.id}>
          <h2>
            <span>{String(index + 1).padStart(2, "0")}</span>
            {section.title}
          </h2>
          {section.paragraphs.map((paragraph, paragraphIndex) => (
            <p key={`${section.id}-${paragraphIndex}`}>{paragraph}</p>
          ))}
          {index === 0 && <IntuitionPanel>{lesson.intuition}</IntuitionPanel>}
        </section>
      ))}
    </>
  );
}
