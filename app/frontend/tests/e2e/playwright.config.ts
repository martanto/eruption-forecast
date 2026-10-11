import { defineConfig, devices } from "@playwright/test";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "../../../..");
export default defineConfig({
  testDir: ".",
  outputDir: "../output/playwright",
  reporter: [
    ["list"],
    ["html", { outputFolder: "../output/playwright-report", open: "never" }],
  ],
  fullyParallel: false,
  workers: 1,
  use: { baseURL: "http://127.0.0.1:5173", trace: "retain-on-failure" },
  projects: [
    { name: "edge", use: { ...devices["Desktop Edge"], channel: "msedge" } },
  ],
  webServer: [
    {
      command: "uv run --project app/backend eruption-ui serve",
      cwd: root,
      url: "http://127.0.0.1:8000/api/v1/health",
      reuseExistingServer: !process.env.CI,
      env: {
        ERUPTION_APP_STATE: path.join(
          root,
          "app/backend/tests/output/browser-state",
        ),
      },
    },
    {
      command: "npm run dev",
      cwd: path.join(root, "app/frontend"),
      url: "http://127.0.0.1:5173",
      reuseExistingServer: !process.env.CI,
    },
  ],
});
