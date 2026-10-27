import { test, expect } from "@playwright/test";

test("walk through both label builders and validate a tremor slice", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("link", { name: "Labelling", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "From eruption dates to learning windows",
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Previous", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Next step", exact: true }).click();
  await expect(
    page.getByText("80 labels · 24 positive · 56 negative"),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Step 3: Slide a two-day look-back" })
    .click();
  await expect(
    page.getByRole("img", {
      name: /Selected window from 03-16 00:00 to 03-18 00:00/,
    }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Next window", exact: true }).click();
  await expect(page.locator("output")).toContainText("03-18 06:00");
  await page.getByRole("button", { name: "2 Dynamic labelling" }).click();
  await expect(
    page.getByRole("heading", { name: "Two event grids before merging" }),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "local id" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Next step", exact: true }).click();
  await expect(
    page.getByRole("columnheader", { name: "local id" }),
  ).toHaveCount(0);
  await expect(page.getByLabel("History before each eruption")).toHaveValue(
    "7",
  );
  await expect(
    page.getByText(/64 event-grid rows − 12 duplicate timestamps/),
  ).toBeVisible();
  await page.getByLabel("History before each eruption").selectOption("3");
  await expect(
    page.getByText(/32 event-grid rows − 0 duplicate timestamps/),
  ).toBeVisible();
  await page.getByRole("button", { name: "3 Tremor matrix" }).click();
  await page.getByLabel("Label source").selectOption("dynamic");
  await page
    .getByRole("button", { name: "Step 3: Check data completeness" })
    .click();
  await page.getByLabel("Remove every fourth sample").check();
  await expect(
    page.getByText(/216 \/ 288 samples = 75% completeness. Skipped/),
  ).toBeVisible();
  await page.getByRole("button", { name: "Next step", exact: true }).click();
  await expect(page.getByRole("table", { name: "Rejected slice · preview only" })).toBeVisible();
  await page.getByLabel("Remove every fourth sample").uncheck();
  await expect(
    page.getByRole("columnheader", { name: "id", exact: true }),
  ).toBeVisible();
  await expect(page.getByText("9216 matrix rows")).toBeVisible();
  const ruler = page.getByRole("region", { name: "Scrollable time ruler" });
  await expect(ruler).toHaveAttribute("tabindex", "0");
  expect(
    await ruler.evaluate(
      (host) => host.scrollWidth === host.clientWidth || host.scrollLeft > 0,
    ),
  ).toBe(true);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: "Restart chapter" }).click();
  await expect(
    page.getByRole("heading", { name: "Pair tremor with a label timestamp" }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Back to Pipeline Atlas" }).click();
  await expect(
    page.getByRole("heading", { name: "Pipeline Atlas", exact: true }),
  ).toBeVisible();
});
