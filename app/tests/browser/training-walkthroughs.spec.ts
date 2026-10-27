import { test, expect } from "@playwright/test";

test("training walkthroughs explain sampling, seed models and consensus", async ({
  page,
}, testInfo) => {
  await page.goto("/labelling");
  await page.getByRole("link", { name: /Continue to Training/ }).click();
  // The training guideline opens on the Seed Ensemble tab first.
  await expect(
    page.getByRole("heading", { name: "Seed Ensemble", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("navigation", { name: "Training lessons" })
    .getByRole("link", { name: "Random Under Sampler", exact: true })
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
    .getByRole("navigation", { name: "Training lessons" })
    .getByRole("link", { name: "Seed Ensemble", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Previous", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: /Step 4:/ }).click();
  await expect(page.locator(".seed-lane")).toHaveCount(3);
  await page.getByRole("button", { name: "Trace seed 1" }).click();
  await expect(
    page.getByRole("button", { name: "Trace seed 1" }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".seed-history-cells .seed-omitted")).toHaveCount(
    40,
  );
  await expect(
    page.locator(".seed-lane").first().locator(".sample-positive"),
  ).toHaveCount(6);
  await expect(
    page.getByRole("img", { name: /Original Seed Ensemble/ }),
  ).toHaveCount(0);
  await expect(page.locator("output")).toContainText("68.0%");
  await page.getByLabel("Classifier family").selectOption("1");
  await expect(page.locator("output")).toContainText("54.0%");
  await page.screenshot({
    path: `tests/review/training/seeds-${testInfo.project.name}.png`,
    fullPage: true,
  });

  await page
    .getByRole("navigation", { name: "Training lessons" })
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
  await expect(
    page.getByRole("img", { name: /Original Classifier Ensemble/ }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "Model prediction", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: /Step 4:/ }).click();
  await expect(page.locator(".classifier-prediction-stages")).toContainText(
    "Window #55  /  probability 64.3%",
  );
  await expect(page.locator(".classifier-prediction-stages")).toContainText(
    "Prediction vote fraction: 88.9%",
  );
  await page.screenshot({
    path: `tests/review/training/classifier-flow-${testInfo.project.name}.png`,
    fullPage: true,
  });
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
