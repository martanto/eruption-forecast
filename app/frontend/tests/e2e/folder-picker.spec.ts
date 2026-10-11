import { test, expect } from "@playwright/test";

test("SDS selection, cancellation, and manual fallback", async ({ page }) => {
  await page.goto("/#/setup");
  const field = page.getByLabel("SDS directory", { exact: true });
  await expect(field).toBeVisible();
  await expect(field).toHaveAttribute("placeholder", /Absolute path/);
  await page.route("**/api/v1/paths/pick-directory", (route) =>
    route.fulfill({ json: { path: "D:\\Data\\OJN" } }),
  );
  await page.getByRole("button", { name: "Choose SDS directory" }).click();
  await expect(field).toHaveValue("D:\\Data\\OJN");
  await expect(field).toBeFocused();
  await page.unroute("**/api/v1/paths/pick-directory");
  await page.route("**/api/v1/paths/pick-directory", (route) =>
    route.fulfill({ json: { path: null } }),
  );
  await page.getByRole("button", { name: "Choose SDS directory" }).click();
  await expect(
    page.getByText("Folder selection cancelled. Your path is unchanged."),
  ).toBeVisible();
  await expect(field).toHaveValue("D:\\Data\\OJN");
  await page.unroute("**/api/v1/paths/pick-directory");
  await page.route("**/api/v1/paths/pick-directory", (route) =>
    route.fulfill({
      status: 503,
      json: { detail: "Enter an absolute path instead." },
    }),
  );
  await page.getByRole("button", { name: "Choose SDS directory" }).click();
  await expect(page.getByRole("alert")).toContainText("Enter an absolute path");
  await field.fill("");
  await field.fill("D:\\Data\\OJN");
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(
    page.getByRole("region", { name: "Directories", exact: true }),
  ).toBeVisible();
});

test("a busy picker can be cancelled and reopened", async ({ page }) => {
  await page.goto("/#/setup");
  let cancelled = false;
  await page.route("**/api/v1/paths/pick-directory", (route) => {
    if (route.request().method() === "DELETE") {
      cancelled = true;
      return route.fulfill({ json: { cancelled: true } });
    }
    return cancelled
      ? route.fulfill({ json: { path: null } })
      : route.fulfill({
          status: 409,
          json: {
            detail:
              "A folder dialog is already open. Complete or cancel it first.",
          },
        });
  });
  await page.getByRole("button", { name: "Choose SDS directory" }).click();
  await expect(page.getByRole("alert")).toContainText("already open");
  await page.getByRole("button", { name: "Cancel folder selection" }).click();
  await expect(page.getByRole("alert")).toHaveCount(0);
  await page.getByRole("button", { name: "Choose SDS directory" }).click();
  await expect(
    page.getByText("Folder selection cancelled. Your path is unchanged."),
  ).toBeVisible();
});
