import { test, expect } from '@playwright/test';

test('theme follows system initially and saves an explicit choice', async ({ page, isMobile }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto('/');
  await expect(page.locator('html')).toHaveAttribute('data-mode', 'dark');
  await page.screenshot({ path: `tests/review/theme-dark-${isMobile ? 'mobile' : 'desktop'}.png` });
  await page.getByRole('button', { name: 'Switch to light theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-mode', 'light');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-mode', 'light');
  await page.screenshot({ path: `tests/review/theme-light-${isMobile ? 'mobile' : 'desktop'}.png` });
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-mode', 'dark');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('system changes apply until the user chooses a theme', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto('/');
  await page.emulateMedia({ colorScheme: 'dark' });
  await expect(page.locator('html')).toHaveAttribute('data-mode', 'dark');
  await page.getByRole('button', { name: 'Switch to light theme' }).click();
  await page.emulateMedia({ colorScheme: 'light' });
  await page.emulateMedia({ colorScheme: 'dark' });
  await expect(page.locator('html')).toHaveAttribute('data-mode', 'light');
});
