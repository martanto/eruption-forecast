import { test, expect } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

test("all ten pages load without browser errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  for (const name of [
    "General setup",
    "Scenario setup",
    "Calculate",
    "Training",
    "Prediction",
    "Evaluation",
    "Explanation",
    "Results",
    "Jobs",
    "System",
  ]) {
    await page
      .getByRole("navigation", { name: "Main navigation" })
      .getByRole("link", { name, exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name, exact: true, level: 1 }),
    ).toBeVisible();
  }
  expect(errors).toEqual([]);
});

test("example setup persists and method parameters are editable", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "OJN example" }).click();
  await expect(page.getByLabel("SDS directory", { exact: true })).toHaveValue(
    "D:\\Data\\OJN",
  );
  await page.getByRole("button", { name: "Save setup", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("General setup saved");
  await page.getByRole("link", { name: "Training", exact: true }).click();
  await page.getByLabel("Find a parameter").fill("seeds");
  await expect(page.getByLabel("Seeds", { exact: true })).toBeVisible();
  await page.getByLabel("Seeds", { exact: true }).fill("2");
  await page.reload();
  await page.getByLabel("Find a parameter").fill("seeds");
  await expect(page.getByLabel("Seeds", { exact: true })).toHaveValue("2");
});

test("scenario overrides do not change shared settings", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "OJN example" }).click();
  await page.getByRole("link", { name: "Scenario setup", exact: true }).click();
  await page.getByRole("button", { name: /OJN smoke/ }).click();
  await page
    .getByRole("button", { name: "Duplicate scenario", exact: true })
    .click();
  await expect(page.getByLabel("Scenario name", { exact: true })).toHaveValue(
    "OJN smoke copy",
  );
  await page.getByLabel("Find a parameter").fill("seeds");
  await page.getByLabel("Seeds", { exact: true }).fill("3");
  await page.getByRole("button", { name: /Shared defaults/ }).click();
  await page.getByLabel("Find a parameter").fill("seeds");
  await expect(page.getByLabel("Seeds", { exact: true })).toHaveValue("1");
});

test("jobs can be submitted, cancelled and retried", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "OJN example" }).click();
  await page.getByRole("button", { name: "Save setup", exact: true }).click();
  await page.getByRole("link", { name: "Calculate", exact: true }).click();
  await page
    .getByRole("button", { name: "Run calculate", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Jobs", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Cancel job", exact: true }).click();
  await expect(page.locator(".job-detail .status-cancelled")).toBeVisible();
  await page
    .getByRole("button", { name: "Retry with valid caches", exact: true })
    .click();
  await expect(
    page.getByText("Waiting for the worker to claim this job."),
  ).toBeVisible();
  await page.getByRole("button", { name: "Cancel job", exact: true }).click();
});

test("mobile navigation and width remain usable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByLabel("Load saved configuration")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "General setup", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Toggle navigation" }).click();
  await page.getByRole("link", { name: "Scenario setup", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Scenario setup", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.keyboard.press("Tab");
  expect(
    await page.evaluate(() => document.activeElement !== document.body),
  ).toBe(true);
});

test("JSON fields retain focus while editing valid values", async ({ page }) => {
  await page.goto("/#/calculate");
  const bands = page.getByLabel("Frequency bands (Hz)", { exact: true });
  await bands.fill("[[1, 2]]");
  await expect(bands).toBeFocused();
  await page.keyboard.press("End");
  await page.keyboard.type(" ");
  await expect(bands).toHaveValue("[[1, 2]] ");
  await expect(bands).toBeFocused();
  await bands.blur();
  await page.reload();
  expect(JSON.parse(await bands.inputValue())).toEqual([[1, 2]]);
});

test("registered CSV artifacts render a chart, table and download", async ({ page, request }) => {
  const directory = path.resolve("tests/output/browser-artifacts");
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, "tremor-browser.csv"), "datetime,rsam_f2\n2025-03-16T00:00:00,12\n2025-03-16T00:10:00,15\n");
  const saved = await request.put("http://127.0.0.1:8000/api/v1/settings", { data: { output_dir: directory } });
  expect(saved.ok()).toBe(true);
  const indexed = await request.post("http://127.0.0.1:8000/api/v1/artifacts/discover", { data: { path: directory } });
  expect(indexed.ok()).toBe(true);
  await page.goto("/#/results");
  await page.getByRole("button", { name: /tremor-browser.csv/ }).click();
  await expect(page.getByRole("img", { name: "tremor-browser.csv" })).toBeVisible();
  await page.getByRole("button", { name: "Data table", exact: true }).click();
  await expect(page.getByRole("cell", { name: "12", exact: true })).toBeVisible();
  const href = await page.getByRole("link", { name: "Download", exact: true }).getAttribute("href");
  const file = await request.get(`http://127.0.0.1:8000${href}`);
  expect(await file.text()).toContain("rsam_f2");
});
