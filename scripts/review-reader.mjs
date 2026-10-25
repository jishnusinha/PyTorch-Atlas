import { chromium } from "@playwright/test";
import { pathToFileURL } from "node:url";
import path from "node:path";
const browser = await chromium.launch();
const context = await browser.newContext({
  offline: true,
  viewport: { width: 1440, height: 1000 },
  reducedMotion: "reduce",
});
const page = await context.newPage();
const url = pathToFileURL(path.resolve("dist/PyTorch-Atlas.html")).href;
await page.goto(url);
await page.screenshot({ path: "docs/screenshots/desktop-overview.png" });
await page.getByRole("button", { name: "Switch to dark theme" }).click();
await page.screenshot({ path: "docs/screenshots/desktop-dark-overview.png" });
await page.getByRole("button", { name: "Switch to light theme" }).click();
await page.setViewportSize({ width: 360, height: 900 });
await page
  .locator(".diagram")
  .screenshot({ path: "docs/screenshots/mobile-autograd.png" });
await page.goto(url + "#one-training-step");
await page.locator(".diagram").screenshot({ path: "docs/screenshots/mobile-update.png" });
await browser.close();
console.log("Captured desktop overviews and mobile graph/update diagrams offline.");
