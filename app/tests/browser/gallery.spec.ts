import { test, expect } from '@playwright/test';

test('gallery lists every shared component and fits the viewport', async ({ page }) => {
  await page.goto('/gallery');
  await expect(page.getByRole('heading', { name: 'Component gallery' })).toBeVisible();
  for (const name of ['ChapterNav', 'StepControls', 'DataTable', 'ChartFrame', 'ExplanationDialog', 'ThemeToggle']) {
    await expect(page.getByRole('heading', { name, exact: true })).toBeVisible();
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('gallery demos are interactive', async ({ page }) => {
  await page.goto('/gallery');
  await page.getByRole('button', { name: 'rsam_f3' }).click();
  await expect(page.getByRole('button', { name: 'rsam_f3' })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Open explanation' }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
});
