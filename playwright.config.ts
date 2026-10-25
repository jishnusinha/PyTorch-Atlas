import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests/browser",
  fullyParallel: false,
  workers: 1,
  timeout: 30000,
  use: {
    headless: true,
    viewport: { width: 1440, height: 1000 },
    acceptDownloads: true,
  },
  reporter: [["list"], ["json", { outputFile: "test-results/browser-results.json" }]],
});
