import { expect, test } from "@playwright/test";

test("selected stage stays readable when the canvas width changes", async ({ page }) => {
  await page.goto("/");
  const canvas = page.locator(".react-flow");
  const selected = page.locator('.react-flow__node[data-id="tremor"]');
  for (const width of [375, 1440, 320, 1920]) {
    await page.setViewportSize({ width, height: 900 });
    const shell = await page.locator(".atlas-shell").boundingBox();
    expect(shell!.width).toBe(width);
    await expect(async () => {
      const frame = await canvas.boundingBox();
      const node = await selected.boundingBox();
      expect(frame).not.toBeNull();
      expect(node).not.toBeNull();
      expect(node!.x).toBeGreaterThanOrEqual(frame!.x);
      expect(node!.x + node!.width).toBeLessThanOrEqual(frame!.x + frame!.width);
      expect(node!.y).toBeGreaterThanOrEqual(frame!.y);
      expect(node!.y + node!.height).toBeLessThanOrEqual(frame!.y + frame!.height - 64);
    }).toPass();
    const zoom = await page.locator(".react-flow__controls").boundingBox();
    const actions = await page.locator(".canvas-actions").boundingBox();
    expect(zoom).not.toBeNull();
    expect(actions).not.toBeNull();
    expect(zoom!.x + zoom!.width).toBeLessThanOrEqual(actions!.x);
  }
});
