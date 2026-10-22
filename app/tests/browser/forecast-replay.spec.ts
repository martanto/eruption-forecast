import { test, expect } from '@playwright/test';

test('recorded prediction opens, replays and switches classifiers', async ({ page, isMobile }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/?node=prediction&scenario=scenario-5-2');
  if (isMobile) await page.getByRole('button', { name: 'Inspect selection', exact: true }).click();
  await page.getByRole('button', { name: 'Open prediction replay' }).click();
  const dialog = page.getByRole('dialog').last();
  await expect(dialog.getByRole('heading', { name: 'OJN · scenario 5 prediction' })).toBeVisible();
  await expect(dialog.locator('canvas')).toBeVisible();
  await dialog.getByLabel('Model', { exact: true }).selectOption('XGBClassifier');
  await dialog.getByRole('button', { name: 'Replay', exact: true }).click();
  await expect(dialog.getByRole('button', { name: 'Pause', exact: true })).toBeVisible();
  await dialog.getByRole('button', { name: 'Show all' }).click();
  await expect(dialog.getByRole('slider', { name: /Replay position/ })).toHaveValue('3168');
  await page.screenshot({ path: `tests/review/forecast-${isMobile ? 'mobile' : 'desktop'}.png` });
  await dialog.getByRole('button', { name: 'Close prediction replay' }).click();
  expect(errors).toEqual([]);
});
