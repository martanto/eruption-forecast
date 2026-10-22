import { expect, test } from "@playwright/test";

for (const view of ["pipeline", "timeline"]) {
  test(`tour centers every selected step in the ${view} canvas`, async ({ page }) => {
    await page.goto(`/?view=${view}`);
    if (view === "pipeline") {
      await page.locator('.react-flow__node[data-id="tremor"]')
        .getByRole("button", { name: "Explore steps" }).click();
    }
    await page.getByRole("button", { name: "Take the guided tour" }).click();
    for (let step = 0; step < 7; step++) {
      await expect(async () => {
        const selected = await page.locator(".react-flow__node.selected").boundingBox();
        const canvas = await page.locator(".react-flow").boundingBox();
        expect(selected).not.toBeNull();
        expect(canvas).not.toBeNull();
        expect(Math.abs(selected!.x + selected!.width / 2 - (canvas!.x + canvas!.width / 2))).toBeLessThan(4);
        expect(Math.abs(selected!.y + selected!.height / 2 - (canvas!.y + (canvas!.height - 64) / 2))).toBeLessThan(4);
      }).toPass();
      if (step < 6) await page.getByRole("button", { name: "Next", exact: true }).click();
    }
    await page.getByRole("button", { name: "Finish tour" }).click();
    await expect(page.getByRole("button", { name: "Take the guided tour" })).toBeVisible();
  });
}
