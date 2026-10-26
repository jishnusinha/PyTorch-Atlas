import { build } from "esbuild";
import fs from "node:fs/promises";
import { spawnSync } from "node:child_process";

await build({
  stdin: {
    contents: `import { allSnippets } from "../src/content/lessons.ts";\nimport { annotatedSnippetIds } from "../src/content/codeAnnotations.ts";\nconst selected = new Map(allSnippets.map((snippet) => [snippet.id, snippet]));\nprocess.stdout.write(JSON.stringify(annotatedSnippetIds.map((id) => selected.get(id))));`,
    resolveDir: `${process.cwd()}/scripts`,
    sourcefile: "annotated-snippets-entry.ts",
    loader: "ts",
  },
  outfile: "tmp/annotated-snippets.mjs",
  bundle: true,
  platform: "node",
  format: "esm",
});
const entry = spawnSync(process.execPath, ["tmp/annotated-snippets.mjs"], { encoding: "utf8", maxBuffer: 10 * 1024 * 1024 });
if (entry.status !== 0) throw new Error(entry.stderr || "Could not load annotated snippets");
const snippets = JSON.parse(entry.stdout);
const python = String.raw`
import json, sys, traceback
snippets=json.load(sys.stdin)
passed=[]; skipped=[]; failed=[]
for item in snippets:
    sid=item["id"]
    try:
        exec(compile(item["code"], sid + ".py", "exec"), {"__name__":"__main__"})
        passed.append(sid)
    except ModuleNotFoundError as exc:
        skipped.append((sid, exc.name))
    except Exception as exc:
        failed.append((sid, repr(exc)))
print(f"PASS {len(passed)}: {', '.join(passed)}")
print(f"SKIP {len(skipped)} optional dependency snippets: {skipped}")
if failed:
    print(f"FAIL {failed}")
    sys.exit(1)
`;
const run = spawnSync("python", ["-c", python], {
  input: JSON.stringify(snippets),
  encoding: "utf8",
  maxBuffer: 10 * 1024 * 1024,
});
process.stdout.write(run.stdout);
if (run.stderr) process.stderr.write(run.stderr);
process.exitCode = run.status ?? 1;
