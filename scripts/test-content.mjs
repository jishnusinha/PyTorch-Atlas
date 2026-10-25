import { build } from "esbuild";
import { spawnSync } from "node:child_process";
import fs from "node:fs/promises";
await fs.mkdir("tmp", { recursive: true });
if (process.argv.includes("--mirror-only")) {
  await build({
    entryPoints: ["scripts/mirror-snippets.ts"],
    outfile: "tmp/mirror.mjs",
    bundle: true,
    platform: "node",
    format: "esm",
  });
  await import("../tmp/mirror.mjs");
} else {
  await build({
    entryPoints: ["tests/content.test.ts"],
    outfile: "tmp/content-test.mjs",
    bundle: true,
    platform: "node",
    format: "esm",
  });
  const result = spawnSync(process.execPath, ["--test", "tmp/content-test.mjs"], {
    stdio: "inherit",
  });
  process.exitCode = result.status ?? 1;
}
