import { test, expect } from "@playwright/test";
import { pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";
import AxeBuilder from "@axe-core/playwright";
const url = pathToFileURL(path.resolve("dist/PyTorch-Atlas.html")).href;
const pagesUrl = pathToFileURL(path.resolve("dist/index.html")).href;
test.beforeEach(async ({ page, context }) => {
  await context.setOffline(true);
  await page.goto(url);
});
test("invalid URL and stale saved slug recover to an authored lesson", async ({
  page,
}) => {
  await page.evaluate(() =>
    localStorage.setItem("atlas-last", JSON.stringify("old-missing-lesson")),
  );
  await page.goto(url + "#%ZZ");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Let the graph do the calculus.",
  );
});
test("all six examples copy and download their exact mirrored Python code", async ({
  page,
}) => {
  await page.evaluate(() =>
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (text: string) => {
          (window as any).__copied = text;
        },
      },
    }),
  );
  for (const id of ["autograd-intuition", "one-training-step", "mini-batches"]) {
    await page.goto(url + "#" + id);
    await page.evaluate(() =>
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: {
          writeText: async (text: string) => {
            (window as any).__copied = text;
          },
        },
      }),
    );
    const panels = page.locator(".code-panel");
    for (let i = 0; i < 2; i++) {
      const panel = panels.nth(i);
      await panel.getByRole("button", { name: "Copy", exact: true }).click();
      const text = await page.evaluate(() => (window as any).__copied);
      const promise = page.waitForEvent("download");
      await panel.getByRole("button", { name: "Download .py" }).click();
      const download = await promise;
      const expected = fs.readFileSync(
        path.resolve("python_examples/01_previews", download.suggestedFilename()),
        "utf8",
      );
      expect(text + "\n").toBe(expected);
      expect(fs.readFileSync((await download.path())!, "utf8")).toBe(expected);
    }
  }
});
test("WCAG A/AA scan across all authored lessons, dark mode and search", async ({
  page,
}) => {
  test.setTimeout(240000);
  for (const id of [
    "autograd-intuition",
    "one-training-step",
    "mini-batches",
    "tensor-prerequisites",
    "derivatives-chain-rule",
    "computation-graph",
    "scalar-autograd",
    "vector-gradients",
    "gradient-contexts",
    "binary-classification",
    "preprocessing",
    "manual-forward",
    "manual-loss",
    "manual-training-loop",
    "binary-evaluation",
    "manual-vs-idiomatic",
    "nn-module",
    "sequential-architecture",
    "loss-conventions",
    "optimizer-lifecycle",
    "custom-dataset",
    "dataloader-pipeline",
    "collation-workers",
    "idiomatic-classifier",
    "batched-classifier",
    "fashion-mnist",
    "ann-architecture",
    "multiclass-training",
    "ann-evaluation",
    "device-placement",
    "overfitting",
    "dropout",
    "batch-normalization",
    "weight-decay",
    "regularization-comparison",
    "hyperparameter-search",
    "optuna-basics",
    "optuna-tabular",
    "optuna-samplers",
    "optuna-plots",
    "dynamic-search",
    "optuna-tracking",
    "ann-search-space",
    "dynamic-ann",
    "ann-objective",
    "cnn-motivation",
    "image-tensors",
    "convolution-math",
    "convolution-shapes",
    "convolution-channels",
    "cnn-activations",
    "pooling",
    "receptive-field",
    "cnn-classifier",
    "transfer-learning",
    "frozen-features",
    "vgg-preprocessing",
    "vgg-transform",
    "vgg-normalization",
    "vgg16-lab",
    "weights-transforms",
    "rnn-motivation",
    "rnn-use-cases",
    "rnn-shapes",
    "recurrent-cell",
    "rnn-variants",
    "bptt",
    "gradient-pathologies",
    "faq-classifier",
    "faq-vocabulary",
    "faq-vocabulary-map",
    "faq-data",
    "faq-architecture",
    "faq-training",
    "faq-inference",
    "lstm-intuition",
    "lstm-states",
    "lstm-vectors",
    "lstm-gates",
    "lstm-blueprint",
    "lstm-numerical",
    "lstm-gradient-route",
    "lstm-pytorch-shapes",
    "lstm-parameters",
    "next-word-data",
    "next-word-tokenization",
    "next-word-vocabulary",
    "next-word-prefixes",
    "next-word-padding",
    "next-word-architecture",
    "next-word-training",
    "next-word-generation",
    "next-word-sampling",
    "next-word-project",
  ]) {
    await page.goto(url + "#" + id);
    const result = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(
      result.violations.map((v) => ({
        id: v.id,
        nodes: v.nodes.map((n) => n.target),
      })),
    ).toEqual([]);
  }
  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  expect(
    (await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze())
      .violations,
  ).toEqual([]);
  await page.getByRole("button", { name: "Search lessons" }).click();
  expect(
    (await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze())
      .violations,
  ).toEqual([]);
});
test("file URL opens offline, no remote requests or script errors, valid links and diagrams", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const remote: string[] = [];
  page.on("request", (r) => {
    if (/^https?:/.test(r.url())) remote.push(r.url());
  });
  await page.reload();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Let the graph do the calculus.",
  );
  await expect(
    page.getByRole("img", { name: "Scalar square computation graph" }),
  ).toBeVisible();
  expect(remote).toEqual([]);
  expect(errors).toEqual([]);
  const missing = await page
    .locator('a[href^="#"]')
    .evaluateAll((links) =>
      links
        .map((a) => a.getAttribute("href")!)
        .filter((h) => !document.getElementById(h.slice(1))),
    );
  expect(missing).toEqual([]);
  await page.screenshot({
    path: "docs/screenshots/desktop-light.png",
    fullPage: true,
  });
});
test("GitHub Pages index entrypoint boots independently from a file URL", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const remote: string[] = [];
  page.on("request", (request) => {
    if (/^https?:/.test(request.url())) remote.push(request.url());
  });
  await page.goto(pagesUrl);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Let the graph do the calculus.",
  );
  expect(remote).toEqual([]);
  expect(errors).toEqual([]);
});
test("navigation, previous/next, resume and browser history", async ({ page }) => {
  await page.getByRole("button", { name: "One training step", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "One step toward a smaller loss.",
  );
  await page.reload();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "One step toward a smaller loss.",
  );
  await page
    .getByRole("navigation", { name: "Lesson pagination" })
    .getByRole("button")
    .last()
    .click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Small batches. A steady pipeline.",
  );
  await page.goBack();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "One step toward a smaller loss.",
  );
  await page.goto(url);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "One step toward a smaller loss.",
  );
});
test("keyboard search, highlighted matches, empty result, escape and focus return", async ({
  page,
}) => {
  await page.getByRole("button", { name: "Search lessons" }).focus();
  await page.keyboard.press("/");
  const search = page.getByRole("textbox", { name: "Search the atlas", exact: true });
  await expect(search).toBeFocused();
  await search.fill("collate");
  await expect(page.locator(".search-results mark").first()).toContainText(/collate/i);
  await search.fill("xyznoresults");
  await expect(page.getByText("No matching lessons yet.")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Search lessons" })).toBeFocused();
  await page.keyboard.press("Control+k");
  await search.fill("hillside");
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "One step toward a smaller loss.",
  );
});
test("search discovers a term present only in an additional Python example", async ({ page }) => {
  await page.getByRole("button", { name: "Search lessons" }).click();
  await page
    .getByRole("textbox", { name: "Search the atlas", exact: true })
    .fill("add_offset");
  const result = page.locator(".search-results button");
  await expect(result).toHaveCount(2);
  const lessonResult = result.filter({ hasText: "Tensors are the shape of the computation." });
  await expect(lessonResult).toHaveCount(1);
  await expect(lessonResult.locator("strong")).toHaveText(
    "Tensors are the shape of the computation.",
  );
  await lessonResult.click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Tensors are the shape of the computation.",
  );
});
test("completion, bookmarks, theme and quiz answers persist correctly", async ({
  page,
}) => {
  await page.getByRole("checkbox", { name: /I understand this lesson/ }).check();
  await page.getByRole("button", { name: "Bookmark lesson", exact: true }).click();
  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.getByRole("checkbox", { name: /Lesson completed/ })).toBeChecked();
  await expect(page.getByRole("button", { name: "Remove bookmark" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.getByText("What are y and x.grad when x = −2?", { exact: false }).click();
  await expect(page.getByText("y = 4 and x.grad = −4.", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: /Saved lessons/ }).click();
  await expect(page.locator(".saved-list").getByRole("button")).toHaveCount(1);
  await page.locator(".saved-list").getByRole("button").click();
  await page.getByRole("checkbox", { name: /Lesson completed/ }).uncheck();
  await page.getByRole("button", { name: "Remove bookmark" }).click();
  await page.reload();
  await expect(
    page.getByRole("checkbox", { name: /I understand this lesson/ }),
  ).not.toBeChecked();
  await page.getByRole("button", { name: "Switch to light theme" }).click();
});
test("copy success, denied-copy fallback failure and Python download", async ({
  page,
}) => {
  await page.evaluate(() =>
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (text: string) => {
          (window as any).__copied = text;
        },
      },
    }),
  );
  await page.getByRole("button", { name: "Copy", exact: true }).first().click();
  await expect(page.locator(".copy-status").first()).toHaveText("Copied to clipboard");
  expect(await page.evaluate(() => (window as any).__copied)).toContain(
    "requires_grad=True",
  );
  await page.evaluate(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async () => {
          throw Error("denied");
        },
      },
    });
    document.execCommand = () => false;
  });
  await page.getByRole("button", { name: "Copy", exact: true }).first().click();
  await expect(page.locator(".copy-status").first()).toContainText("Copy unavailable");
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download .py", exact: true }).first().click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("scalar-gradient.py");
  const file = await download.path();
  expect(fs.readFileSync(file!, "utf8")).toContain("assert torch.allclose");
});
test("file clipboard fallback is used when permission is denied", async ({ page }) => {
  await page.evaluate(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async () => {
          throw Error("denied");
        },
      },
    });
    (window as any).__selected = "";
    document.execCommand = (cmd: string) => {
      (window as any).__selected = (document.activeElement as HTMLTextAreaElement).value;
      return cmd === "copy";
    };
  });
  await page.getByRole("button", { name: "Copy", exact: true }).first().click();
  await expect(page.locator(".copy-status").first()).toHaveText("Copied to clipboard");
  expect(await page.evaluate(() => (window as any).__selected)).toContain("import torch");
});
test("real clipboard path can report a successful copy on file URL", async ({ page }) => {
  await page.getByRole("button", { name: "Copy", exact: true }).first().click();
  await expect(page.locator(".copy-status").first()).toHaveText("Copied to clipboard");
});
test("localStorage rejection uses in-memory state without breaking reader", async ({
  page,
}) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, "localStorage", {
      get() {
        throw new DOMException("Blocked", "SecurityError");
      },
    });
  });
  await page.reload();
  await page.getByRole("checkbox", { name: /I understand this lesson/ }).check();
  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  await page.getByRole("button", { name: "One training step", exact: true }).click();
  await page.getByRole("button", { name: "Autograd intuition", exact: true }).click();
  await expect(page.getByRole("checkbox", { name: /Lesson completed/ })).toBeChecked();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
});
test("SVG controls compute real conceptual values and TOC tracks sections", async ({
  page,
}) => {
  const slider = page.getByRole("slider", { name: "Input x" });
  await slider.fill("-2");
  await expect(page.locator(".live-result")).toContainText(
    "output is 4 and the gradient is -4",
  );
  await page.getByRole("button", { name: "One training step", exact: true }).click();
  await page.getByRole("slider", { name: "Learning rate" }).fill("0.25");
  await expect(page.locator(".live-result")).toContainText("2.00 · loss 1.00");
  await page.getByRole("button", { name: "Mini-batch intuition", exact: true }).click();
  await page.getByRole("slider", { name: "Batch size" }).fill("3");
  await expect(page.locator(".live-result")).toHaveText(
    "4 batches: 3 + 3 + 3 + 1 samples.",
  );
  await page.getByRole("checkbox", { name: "Drop the last incomplete batch" }).check();
  await expect(page.locator(".live-result")).toHaveText("3 batches: 3 + 3 + 3 samples.");
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.getByRole("link", { name: "Try it in Python", exact: true }).click();
  await expect(
    page.getByRole("link", { name: "Try it in Python", exact: true }),
  ).toHaveAttribute("aria-current", "location");
});
test("Field Kit alias and source coverage views remain populated and traceable", async ({ page }) => {
  await page.goto(url + "#reference");
  await expect(page).toHaveURL(/#field-kit$/);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Working patterns, ready to adapt.");
  await page.getByRole("button", { name: /Field Kit/ }).first().click();
  await expect(page.locator(".field-recipe-row")).toHaveCount(94);
  await page.getByRole("button", { name: /Source coverage/ }).click();
  await expect(page.locator(".coverage-list details")).toHaveCount(16);
  await page.locator(".coverage-list summary").first().click();
  await expect(page.getByText("Automatic differentiation and chain rule")).toBeVisible();
  await expect(page.getByText("Corrected source example").first()).toBeVisible();
  await page
    .locator(".coverage-list")
    .getByRole("button", { name: "Open lesson →" })
    .first()
    .click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "A gradient carries local sensitivity backward.",
  );
  await expect(page.locator(".source-note")).toContainText("PDF-derived");
  await page.goto(url + "#tensor-prerequisites");
  await expect(page.locator(".source-note")).toContainText("Supplemental engineering");
});
test("Field Kit filters, favorites, direct recipe route and contextual download rules work", async ({ page }) => {
  await page.goto(url + "#field-kit");
  await page.locator(".field-categories").getByRole("button", { name: /Autograd/ }).click();
  await expect(page.locator(".field-recipe-row")).toHaveCount(10);
  await page.getByLabel("Search recipes").fill("derivative");
  await expect(page.locator(".field-recipe-row")).toHaveCount(1);
  await page.locator(".field-recipe-row .field-recipe-open").click();
  await expect(page).toHaveURL(/#field-kit\/derivative-check$/);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Reusable derivative check");
  await expect(page.getByRole("button", { name: "Download .py" })).toBeVisible();
  await page.evaluate(() => Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: async (text: string) => { (window as any).__recipeCopied = text; } },
  }));
  await page.getByRole("button", { name: "Copy", exact: true }).click();
  expect(await page.evaluate(() => (window as any).__recipeCopied)).toBe(
    fs.readFileSync(path.resolve("python_examples/01_previews/derivative-check.py"), "utf8").trimEnd(),
  );
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download .py" }).click();
  const recipeDownload = await downloadPromise;
  expect(fs.readFileSync((await recipeDownload.path())!, "utf8")).toBe(
    fs.readFileSync(path.resolve("python_examples/01_previews/derivative-check.py"), "utf8"),
  );
  await page.getByRole("button", { name: "☆ Save recipe" }).click();
  await expect(page.getByRole("button", { name: "★ Saved" })).toHaveAttribute("aria-pressed", "true");
  await page.goto(url + "#field-kit/local-csv-loader");
  await expect(page.getByText(/after setting its input path to an existing CSV/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Download .py" })).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(url + "#field-kit");
  await expect(page.locator(".field-categories")).toBeVisible();
  await expect(page.getByRole("button", { name: /CNN\/Transfer/ })).toBeVisible();
});
test("Field Kit home and recipe details pass the accessibility scan offline", async ({ page }) => {
  fs.mkdirSync(path.resolve("docs/screenshots"), { recursive: true });
  await page.goto(url + "#field-kit");
  await page.screenshot({ path: path.resolve("docs/screenshots/field-kit-home.png") });
  let results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
  expect(results.violations).toEqual([]);
  await page.goto(url + "#field-kit/nextword-train-step");
  await page.screenshot({ path: path.resolve("docs/screenshots/field-kit-recipe.png") });
  results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
  expect(results.violations).toEqual([]);
});
test("Phase 02 lessons expose the tensor, chain-rule and tracking controls", async ({
  page,
}) => {
  await page.getByRole("button", { name: /Tensor essentials/ }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Tensors are the shape of the computation.",
  );
  await page.getByRole("slider", { name: "Tensor rank" }).fill("3");
  await expect(page.locator(".live-result")).toContainText("[2, 1, 3]");
  await page.getByRole("button", { name: /Derivatives & chain rule/ }).click();
  await page.getByRole("slider", { name: "Input x" }).fill("1.2");
  await expect(page.locator(".live-result")).toContainText("f′(x)");
  await page.getByRole("button", { name: /Gradient control/ }).click();
  const tracking = page.getByRole("checkbox", { name: "Track gradients" });
  await expect(tracking).toBeChecked();
  await tracking.uncheck();
  await expect(page.locator(".live-result")).toContainText(
    "no backward path is recorded",
  );
});
test("Phase 03 pipeline explorer updates stage details and the SGD curve", async ({
  page,
}) => {
  await page
    .getByRole("button", { name: "Manual training pipeline", exact: true })
    .click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Backward, update and clear: the manual training loop.",
  );
  const stages = page.getByRole("group", { name: "Training pipeline stages" });
  await stages.getByRole("button", { name: /Linear score/ }).click();
  await expect(page.locator(".pipeline-detail")).toContainText(
    "X [N,30] @ W [30,1] + b [1]",
  );
  await stages.getByRole("button", { name: /Backward/ }).click();
  await expect(page.locator(".pipeline-detail")).toContainText("loss.backward()");
  await page.getByRole("slider", { name: "Learning rate" }).fill("0.25");
  await expect(page.locator(".pipeline-update .live-result")).toContainText(
    "w′=3−η×4=2.00",
  );
  await page
    .locator(".topbar")
    .evaluate((el) => ((el as HTMLElement).style.display = "none"));
  await page
    .locator(".diagram")
    .screenshot({ path: "docs/screenshots/phase03-pipeline.png" });
  await page.setViewportSize({ width: 360, height: 800 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
  ).toBe(true);
});
test("Phase 04 module, optimizer, loader and collation diagrams are authored and usable", async ({
  page,
}) => {
  await page.goto(url + "#nn-module");
  await expect(
    page.getByRole("img", { name: /BinaryHead contains a registered Linear module/ }),
  ).toBeVisible();
  await page
    .locator(".diagram svg")
    .screenshot({ path: "docs/screenshots/phase04-module-tree.png" });
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Build a model with nn.Module",
  );

  await page.goto(url + "#optimizer-lifecycle");
  await expect(page.getByRole("img", { name: /Training update order/ })).toBeVisible();

  await page.goto(url + "#dataloader-pipeline");
  await expect(
    page.getByRole("img", { name: /The sampler chooses indices/ }),
  ).toBeVisible();

  await page.goto(url + "#collation-workers");
  await expect(
    page.getByRole("img", { name: /Three token sequences of lengths two/ }),
  ).toBeVisible();
  const code = page.locator(".code-panel").first();
  const downloadPromise = page.waitForEvent("download");
  await code.getByRole("button", { name: "Download .py" }).click();
  const download = await downloadPromise;
  const expected = fs.readFileSync(
    path.resolve("python_examples/04_modules_data", download.suggestedFilename()),
    "utf8",
  );
  expect(fs.readFileSync((await download.path())!, "utf8")).toBe(expected);

  await page.goto(url + "#mini-batches");
  await page.getByRole("slider", { name: "Batch size" }).fill("5");
  await expect(page.locator(".live-result")).toContainText("2 batches: 5 + 5 samples");
});
test("Phase 05 ANN width, confusion, data-flow and device diagrams are usable", async ({
  page,
}) => {
  await page.goto(url + "#ann-architecture");
  await expect(page.getByRole("img", { name: /ANN tensor path/ })).toHaveAttribute(
    "aria-label",
    /109,386 trainable parameters/,
  );
  await page.getByRole("slider", { name: "First hidden width" }).fill("64");
  await expect(page.locator(".live-result")).toContainText(
    "64 → 64 hidden units · 55,050 trainable parameters",
  );
  await page.goto(url + "#ann-evaluation");
  await expect(
    page.getByRole("img", { name: /Three-class confusion matrix/ }),
  ).toBeVisible();
  await expect(page.locator(".live-result")).toContainText(
    "Row totals determine class recall",
  );
  await page.goto(url + "#fashion-mnist");
  await expect(
    page.getByRole("img", { name: /Disjoint train, validation and test/ }),
  ).toBeVisible();
  const code = page.locator(".code-panel").last();
  const downloadPromise = page.waitForEvent("download");
  await code.getByRole("button", { name: "Download .py" }).click();
  const download = await downloadPromise;
  expect(fs.readFileSync((await download.path())!, "utf8")).toBe(
    fs.readFileSync(
      path.resolve("python_examples/05_ann_gpu", download.suggestedFilename()),
      "utf8",
    ),
  );
  await page.goto(url + "#device-placement");
  await expect(page.getByRole("img", { name: /Check CUDA availability/ })).toBeVisible();
  await expect(page.locator(".live-result")).toContainText("the copy still happens");
});
test("Phase 06 regularization and Optuna diagrams respond to controls", async ({
  page,
}) => {
  await page.goto(url + "#overfitting");
  await page.getByRole("slider", { name: "Selected checkpoint epoch" }).fill("4");
  await expect(page.locator(".live-result")).toContainText("validation loss 1.25");
  await page.goto(url + "#dropout");
  await page
    .locator(".diagram")
    .screenshot({ path: "docs/screenshots/phase06-dropout.png" });
  await page.getByRole("slider", { name: "Drop probability" }).fill("0.5");
  await expect(page.locator(".live-result")).toContainText("kept activation scale=2.00");
  await page.goto(url + "#batch-normalization");
  await page
    .locator(".diagram")
    .screenshot({ path: "docs/screenshots/phase06-batchnorm.png" });
  await page.getByRole("slider", { name: "BatchNorm gamma" }).press("End");
  await page.getByRole("slider", { name: "BatchNorm beta" }).press("End");
  await expect(page.locator(".live-result")).toContainText("3.0×1.00 + 2.0 = 5.00");
  await page.goto(url + "#weight-decay");
  await page.getByRole("slider", { name: "Weight decay lambda" }).press("ArrowRight");
  await expect(page.locator(".live-result")).toContainText("w′ = 2 − 0.1×0.30×2 = 1.940");
  await page.goto(url + "#regularization-comparison");
  await page
    .getByRole("combobox", { name: "Illustrative optimizer" })
    .selectOption("AdamW");
  await expect(page.locator(".live-result")).toContainText(
    "AdamW: illustrative trajectory only",
  );
  await page.goto(url + "#hyperparameter-search");
  await page
    .locator(".diagram")
    .screenshot({ path: "docs/screenshots/phase06-search-space.png" });
  await page.getByRole("combobox", { name: "Search sampler" }).selectOption("TPE");
  await page.getByRole("slider", { name: "Trial budget" }).press("ArrowRight");
  await expect(page.locator(".live-result")).toContainText("no global-optimum guarantee");
  await page.goto(url + "#optuna-basics");
  await expect(
    page.getByRole("img", { name: /Illustrative parallel-coordinate trial comparison/ }),
  ).toBeVisible();
  await page.goto(url + "#ann-objective");
  const snippets = page.locator(".code-panel");
  await expect(snippets.first()).toContainText("Not passed into objective");
  await expect(snippets.first()).toContainText("test accuracy");
});
test("Phase 07 convolution diagrams calculate shapes, values and parameters", async ({ page }) => {
  await page.goto(url + "#convolution-math");
  await expect(page.getByRole("img", { name: /selected 3 by 3 patch at row 1, column 1.*output -6/ })).toBeVisible();
  await page.getByRole("slider", { name: "Kernel column" }).fill("2");
  await expect(page.locator(".live-result")).toContainText("= -6 before bias");
  await page.locator(".diagram").screenshot({ path: "docs/screenshots/phase07-convolution.png" });
  await page.goto(url + "#convolution-shapes");
  await page.getByRole("slider", { name: "Convolution padding" }).fill("0");
  await expect(page.locator(".live-result")).toContainText("Output: 26 × 26");
  await page.getByRole("slider", { name: "Convolution stride" }).fill("2");
  await expect(page.locator(".live-result")).toContainText("Output: 13 × 13");
  await page.goto(url + "#convolution-channels");
  await page.getByRole("slider", { name: "Input channels" }).fill("1");
  await expect(page.locator(".live-result")).toContainText("160 Conv2d parameters");
  await page.goto(url + "#pooling");
  await page.getByRole("combobox", { name: "Pooling operation" }).selectOption("average");
  await expect(page.locator(".live-result")).toContainText("3.75, 4, 2.5, 3.25");
  await page.goto(url + "#receptive-field");
  await page.getByRole("slider", { name: "Receptive field blocks" }).fill("2");
  await expect(page.locator(".live-result")).toContainText("receptive field 16, jump 4");
  await page.goto(url + "#cnn-classifier");
  await expect(page.getByRole("img", { name: /Fashion-MNIST CNN tensor flow.*Total trainable parameters 429450/ })).toBeVisible();
  await page.locator(".diagram").screenshot({ path: "docs/screenshots/phase07-cnn-flow.png" });
});
test("Phase 08 transfer diagrams update strategy and normalization arithmetic", async ({ page }) => {
  await page.goto(url + "#frozen-features");
  await expect(page.getByRole("img", { name: /Frozen VGG feature extractor/ })).toBeVisible();
  await page.getByRole("combobox", { name: "Transfer strategy" }).selectOption("finetune");
  await expect(page.locator(".live-result")).toContainText("Selected feature weights");
  await page.goto(url + "#vgg-normalization");
  await page.getByRole("slider", { name: "Red pixel value" }).fill("0.5");
  await expect(page.locator(".live-result")).toContainText("(0.50 − 0.485) / 0.229 = 0.066");
  await page.goto(url + "#vgg-preprocessing");
  await expect(page.getByRole("img", { name: /Nine-step Fashion-MNIST/ })).toBeVisible();
  await page.locator(".diagram").screenshot({ path: "docs/screenshots/phase08-preprocessing.png" });
  await page.goto(url + "#weights-transforms");
  await expect(page.getByRole("img", { name: /Red channel normalization/ })).toBeVisible();
});
test("Phase 09 sequence controls update tensor steps, task mapping and gradient products", async ({ page }) => {
  await page.goto(url + "#rnn-shapes");
  await page.getByRole("slider", { name: "Sequence time step" }).fill("4");
  await expect(page.locator(".live-result")).toContainText("x4");
  await page.goto(url + "#recurrent-cell");
  await expect(page.getByRole("img", { name: /RNN cell repeated across four/ })).toBeVisible();
  await page.getByRole("slider", { name: "RNN highlighted time step" }).fill("3");
  await expect(page.locator(".live-result")).toContainText("reused at all four steps");
  await page.goto(url + "#rnn-variants");
  await page.getByRole("combobox", { name: "RNN prediction pattern" }).selectOption("Encoder-decoder");
  await expect(page.locator(".live-result")).toContainText("output length T′");
  await page.goto(url + "#gradient-pathologies");
  await page.getByRole("slider", { name: "Illustrative gradient factor" }).fill("0.5");
  await expect(page.locator(".live-result")).toContainText("0.50^");
});
test("Phase 10 FAQ diagrams explain text flow and keep the interactive mapping illustrative", async ({ page }) => {
  await page.goto(url + "#faq-vocabulary");
  await expect(page.getByRole("img", { name: /lowercased, split into six tokens/ })).toBeVisible();
  await page.goto(url + "#faq-data");
  await expect(page.getByRole("img", { name: /Two variable-length questions are padded/ })).toBeVisible();
  await page.goto(url + "#faq-architecture");
  await expect(page.getByRole("img", { name: /token IDs pass through an embedding/ })).toBeVisible();
  await page.goto(url + "#faq-inference");
  await page.getByRole("combobox", { name: "Illustrative FAQ example" }).selectOption("What is CampusX?");
  await expect(page.locator(".live-result")).toContainText("Illustrative fixed mapping — not a PyTorch prediction");
  await expect(page.locator(".live-result")).toContainText("this demo will not guess");
  await page.locator(".diagram").screenshot({ path: "docs/screenshots/phase10-faq-demo.png" });
  await page.goto(url + "#faq-training");
  await expect(page.getByRole("img", { name: /Three illustrative logits yield normalized softmax/ })).toBeVisible();
});
test("Phase 11 LSTM explorers update cell arithmetic, shapes and parameters", async ({ page }) => {
  await page.goto(url + "#lstm-numerical");
  await expect(page.getByRole("img", { name: /LSTM cell arithmetic for coordinate/ })).toBeVisible();
  await expect(page.locator(".live-result")).toContainText("c=2.400");
  await page.getByRole("slider", { name: "LSTM forget gate" }).fill("0.25");
  await expect(page.locator(".live-result")).toContainText("c=1.400");
  await page.getByRole("slider", { name: "LSTM coordinate" }).fill("2");
  await expect(page.locator(".live-result")).toContainText("c=1.600");
  await page.locator(".diagram").screenshot({ path: "docs/screenshots/phase11-lstm-cell.png" });
  await page.goto(url + "#lstm-gradient-route");
  await page.getByRole("slider", { name: "LSTM direct forget factor" }).fill("0.95");
  await page.getByRole("slider", { name: "LSTM gradient steps" }).fill("20");
  await expect(page.locator(".live-result")).toContainText("0.3585");
  await page.goto(url + "#lstm-pytorch-shapes");
  await expect(page.getByRole("img", { name: /Batch-first input \[2,5,4\]/ })).toBeVisible();
  await page.getByRole("slider", { name: "LSTM shape Directions" }).fill("1");
  await expect(page.locator(".live-result")).toContainText("final states retain layer and direction axes");
  await page.goto(url + "#lstm-parameters");
  await expect(page.locator(".live-result")).toContainText("480 recurrent parameters");
});
test("Phase 12 diagrams, correction index and all stage snippets work", async ({ page }) => {
  for (const id of ["next-word-prefixes", "next-word-padding", "next-word-architecture", "next-word-sampling", "next-word-generation"]) {
    await page.goto(url + "#" + id);
    if (id === "next-word-prefixes") {
      await page.getByRole("slider", { name: "Next-word prefix length" }).fill("4");
      await expect(page.locator(".live-result")).toContainText("immediately following token “science”");
      await page.locator(".diagram").screenshot({path:"docs/screenshots/phase12-nextword-prefix.png"});
    } else if (id === "next-word-padding") {
      await page.getByRole("slider", { name: "Next-word padding width" }).fill("8");
      await expect(page.locator(".live-result")).toContainText("Left padding matches the source example");
    } else if (id === "next-word-architecture") {
      await expect(page.getByRole("img", { name: /Batch-first next-word model trace/ })).toBeVisible();
    } else if (id === "next-word-sampling") {
      await page.getByRole("slider", { name: "Next-word top-k count" }).fill("5");
      await expect(page.locator(".live-result")).toContainText("Illustrative distribution only");
    } else {
      await page.getByRole("slider", { name: "Next-word append steps" }).fill("4");
      await expect(page.locator(".live-result")).toContainText("python libraries the data science tools");
    }
  }
  const lessonIds=["next-word-data","next-word-tokenization","next-word-vocabulary","next-word-prefixes","next-word-padding","next-word-architecture","next-word-training","next-word-generation","next-word-sampling","next-word-project"];
  for (const id of lessonIds) {
    await page.goto(url + "#" + id);
    await page.evaluate(() => Object.defineProperty(navigator, "clipboard", {configurable:true,value:{writeText:async(text:string)=>{(window as any).__copied=text;}}}));
    for(let i=0;i<2;i++){
      const panel=page.locator(".code-panel").nth(i);
      await panel.getByRole("button",{name:"Copy",exact:true}).click();
      const copied=await page.evaluate(()=> (window as any).__copied as string);
      const pendingDownload=page.waitForEvent("download");
      await panel.getByRole("button",{name:"Download .py"}).click();
      const download=await pendingDownload;
      const expected=fs.readFileSync(path.resolve("python_examples/12_lstm_next_word",download.suggestedFilename()),"utf8");
      expect(copied+"\n").toBe(expected);
      expect(fs.readFileSync((await download.path())!,"utf8")).toBe(expected);
    }
  }
  await page.goto(url+"#coverage");
  await expect(page.getByRole("heading",{name:"Source issues and corrections"})).toBeVisible();
  await expect(page.getByText(/PAD\/UNK\/EOS IDs/).first()).toBeVisible();
});
test("F15 annotations focus the exact code line and retain math alternatives offline", async ({ page }) => {
  await page.goto(url + "#optimizer-lifecycle");
  const panel = page.locator(".code-panel").filter({ hasText: "Clear gradients, backpropagate and step" }).first();
  await panel.getByRole("button", { name: "Focus lines 14" }).click();
  await expect(panel.locator(".code-line.is-focused")).toHaveAttribute("id", "optimizer-lifecycle-line-14");
  await expect(panel.getByText(/The standard order is zero_grad/)).toBeVisible();
  await page.goto(url + "#convolution-shapes");
  await expect(page.locator(".equation math mrow")).toBeVisible();
  await expect(page.locator(".equation-fallback")).toContainText("floor");
  await expect(page.getByRole("navigation", { name: "Learning lenses" })).toContainText("Intuition");
  await expect(page.locator(".comparison-note")).toHaveCount(0);
  await page.goto(url + "#loss-conventions");
  await expect(page.locator(".comparison-note")).toContainText("BCEWithLogitsLoss");
  await page.emulateMedia({ media: "print" });
  await expect(page.locator(".annotation-list")).toBeVisible();
  await expect(page.locator(".annotation-focus")).toBeHidden();
});
for (const width of [360, 768, 1440])
  test(`responsive ${width}px, print and reduced motion`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.emulateMedia({ reducedMotion: "reduce" });
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
    ).toBe(true);
    if (width === 360) {
      await page.getByRole("button", { name: "Open navigation" }).click();
      await expect(
        page.getByRole("dialog", { name: "Handbook navigation" }),
      ).toBeVisible();
      await page.keyboard.press("Shift+Tab");
      await expect(page.getByRole("button", { name: /Source coverage/ })).toBeFocused();
      await page.keyboard.press("Escape");
      await expect(page.getByRole("button", { name: "Open navigation" })).toBeFocused();
      await page.getByRole("button", { name: "Open navigation" }).click();
      await page
        .getByRole("button", { name: "Mini-batch intuition", exact: true })
        .click();
      await expect(page.getByRole("heading", { level: 1 })).toHaveText(
        "Small batches. A steady pipeline.",
      );
      await expect(page.locator(".drawer-scrim")).toHaveCount(0);
      await page.screenshot({
        path: "docs/screenshots/mobile-light.png",
        fullPage: true,
      });
      await page.screenshot({ path: "docs/screenshots/mobile-overview.png" });
      await page
        .locator(".diagram")
        .screenshot({ path: "docs/screenshots/mobile-diagram.png" });
    }
    await page.getByRole("button", { name: "Switch to dark theme" }).click();
    await page.screenshot({
      path: `docs/screenshots/${width}-dark.png`,
      fullPage: true,
    });
    await page.emulateMedia({ media: "print" });
    await expect(page.locator(".topbar")).toBeHidden();
    await expect(page.locator("#understand > p").first()).toHaveCSS(
      "color",
      "rgb(40, 56, 74)",
    );
    await expect(page.locator(".intuition")).toHaveCSS(
      "background-color",
      "rgb(243, 244, 246)",
    );
    await page.pdf({
      path: `tmp/print-${width}.pdf`,
      format: "A4",
      printBackground: true,
      margin: { top: "16mm", right: "16mm", bottom: "16mm", left: "16mm" },
    });
  });
