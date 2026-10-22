import { test, expect } from "@playwright/test";

test("training walkthroughs explain sampling, seed models and consensus", async ({
  page,
}, testInfo) => {
  await page.goto("/labelling");
  await page
    .getByRole("link", { name: /Continue: Random Under Sampler/ })
    .click();
  await expect(
    page.getByRole("heading", { name: "Random Under Sampler", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Previous", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: /Step 3:/ }).click();
  await expect(page.locator("output")).toContainText("14 retained windows");
  const initial = await page
    .locator(".sample-discarded")
    .first()
    .getAttribute("title");
  await page.getByLabel("Sampling seed").selectOption("1");
  expect(
    await page.locator(".sample-discarded").first().getAttribute("title"),
  ).not.toBe(initial);
  await expect(page.locator(".sample-positive")).toHaveCount(6);
  await expect(page.locator(".sample-discarded")).toHaveCount(40);
  await page.getByRole("button", { name: "Next step", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Next step", exact: true }),
  ).toBeDisabled();
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({
    path: `tests/review/training/rus-${testInfo.project.name}.png`,
    fullPage: true,
  });

  await page
    .getByRole("navigation", { name: "Pipeline walkthroughs" })
    .getByRole("link", { name: "Seed Ensemble", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Previous", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: /Step 4:/ }).click();
  await expect(page.locator("output")).toContainText("68.0%");
  await page.getByLabel("Classifier family").selectOption("1");
  await expect(page.locator("output")).toContainText("54.0%");
  await page.screenshot({
    path: `tests/review/training/seeds-${testInfo.project.name}.png`,
    fullPage: true,
  });

  await page
    .getByRole("navigation", { name: "Pipeline walkthroughs" })
    .getByRole("link", { name: "Classifier Ensemble", exact: true })
    .click();
  await page.getByRole("button", { name: /Step 4:/ }).click();
  await expect(page.locator("output")).toContainText("64.3%");
  await page.getByLabel("GB SeedEnsemble", { exact: true }).uncheck();
  await expect(page.locator("output")).toContainText("69.5%");
  await page.getByLabel("XGB SeedEnsemble", { exact: true }).uncheck();
  await expect(
    page.getByLabel("RF SeedEnsemble", { exact: true }),
  ).toBeDisabled();
  await expect(page.locator("output")).toContainText("0.0 percentage points");
  await page.getByRole("button", { name: "Restart walkthrough" }).click();
  await expect(
    page.getByLabel("GB SeedEnsemble", { exact: true }),
  ).toBeChecked();
  await expect(
    page.getByRole("button", { name: "Previous", exact: true }),
  ).toBeDisabled();
  await page.screenshot({
    path: `tests/review/training/classifiers-${testInfo.project.name}.png`,
    fullPage: true,
  });
  await page
    .getByText("View the supplied Classifier Ensemble reference", {
      exact: true,
    })
    .click();
  const image = page.getByRole("img", { name: /Original Classifier Ensemble/ });
  await expect(image).toBeVisible();
  await expect
    .poll(() => image.evaluate((el: HTMLImageElement) => el.naturalWidth))
    .toBeGreaterThan(0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Classifier Ensemble", exact: true }),
  ).toBeVisible();
});
