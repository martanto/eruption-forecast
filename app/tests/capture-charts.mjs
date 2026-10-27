// Screenshots every chart SVG on the pages that draw one, for before/after comparison.
// Usage: node tests/capture-charts.mjs <output-dir> [base-url]
import { chromium } from '@playwright/test';
import { mkdir } from 'node:fs/promises';

const output = process.argv[2] ?? 'tests/charts/after';
const base = process.argv[3] ?? 'http://127.0.0.1:5179';
await mkdir(output, { recursive: true });

const ROUTES = [
  ['feature-selection', '/feature-selection'],
  ['prediction', '/prediction'],
  ['prediction-scenarios', '/prediction?part=scenarios'],
];
const browser = await chromium.launch();
const report = [];
try {
  for (const [device, viewport] of [
    ['desktop', { width: 1280, height: 900 }],
    ['mobile', { width: 390, height: 844 }],
  ]) {
    for (const mode of ['light', 'dark']) {
      const context = await browser.newContext({ viewport, colorScheme: mode, reducedMotion: 'reduce' });
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      page.on('console', (message) => message.type() === 'error' && errors.push(message.text()));
      for (const [slug, route] of ROUTES) {
        await page.goto(base + route);
        await page.evaluate(() => document.fonts.ready);
        // The forecast and ROC charts only mount from step 5 of the prediction walkthrough.
        if (slug === 'prediction') await page.getByRole('button', { name: /^Step 5:/ }).click();
        const charts = page.locator('svg[role="img"]');
        const count = await charts.count();
        for (let i = 0; i < count; i++) {
          const chart = charts.nth(i);
          await chart.scrollIntoViewIfNeeded();
          await chart.screenshot({ path: `${output}/${slug}-${device}-${mode}-${String(i).padStart(2, '0')}.png` });
        }
        report.push({ slug, device, mode, charts: count, overflow: await page.evaluate(() => document.documentElement.scrollWidth > innerWidth) });
      }
      if (errors.length) report.push({ device, mode, errors });
      await context.close();
    }
  }
} finally {
  await browser.close();
}
console.log(JSON.stringify(report, null, 1));
