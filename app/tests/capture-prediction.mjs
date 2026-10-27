import { chromium } from '@playwright/test';
import { mkdir } from 'node:fs/promises';

const base = process.env.ATLAS_URL ?? 'http://127.0.0.1:5174';
const output = 'tests/review/prediction';
await mkdir(output, { recursive: true });
const browser = await chromium.launch();
const findings = [];
try {
  for (const [device, viewport] of [['desktop', { width: 1440, height: 1000 }], ['mobile', { width: 390, height: 844 }]]) {
    for (const mode of ['light', 'dark']) {
      const context = await browser.newContext({ viewport, colorScheme: mode, reducedMotion: 'reduce' });
      const page = await context.newPage();
      const errors = [];
      page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
      page.on('pageerror', (e) => errors.push(String(e)));
      await page.goto(`${base}/prediction`);
      await page.evaluate(() => document.fonts.ready);
      const steps = mode === 'light' ? [1, 2, 3, 4, 5] : [4];
      for (const n of steps) {
        await page.getByRole('button', { name: new RegExp(`^Step ${n}:`) }).click();
        await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); });
        await page.screenshot({ path: `${output}/${device}-${mode}-step${n}.png`, fullPage: true });
      }
      await page.getByRole('button', { name: /Several scenarios/ }).click();
      await page.getByRole('button', { name: 'Run scenarios' }).click();
      await page.getByRole('button', { name: 'Next operation' }).click();
      await page.screenshot({ path: `${output}/${device}-${mode}-scenarios.png`, fullPage: true });
      findings.push({ device, mode, overflow: await page.evaluate(() => document.documentElement.scrollWidth > innerWidth) });
      findings.push({ device, mode, errors });
      await context.close();
    }
  }
  console.log(JSON.stringify(findings, null, 2));
} finally {
  await browser.close();
}
