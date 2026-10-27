import { chromium } from '@playwright/test';
import { mkdir } from 'node:fs/promises';

// Renders every page that uses KaTeX and reports parse errors and layout overflow.
const base = process.env.BASE_URL ?? 'http://127.0.0.1:5199';
const output = 'tests/review/math';
await mkdir(output, { recursive: true });
const browser = await chromium.launch();
const report = [];
const consoleErrors = [];
try {
  for (const [device, viewport] of [['desktop', { width: 1440, height: 1000 }], ['mobile', { width: 390, height: 844 }]]) {
    const context = await browser.newContext({ viewport, reducedMotion: 'reduce' });
    const page = await context.newPage();
    page.on('console', (m) => m.type() === 'error' && consoleErrors.push(`${device} ${m.text()}`));
    page.on('pageerror', (e) => consoleErrors.push(`${device} pageerror ${e.message}`));
    const visit = async (slug, url, prepare) => {
      await page.goto(`${base}${url}`);
      await page.evaluate(() => document.fonts.ready);
      if (prepare) await prepare();
      await page.waitForTimeout(300);
      const info = await page.evaluate(() => ({
        math: document.querySelectorAll('.katex').length,
        errors: document.querySelectorAll('.katex-error').length,
        overflow: document.documentElement.scrollWidth > innerWidth,
        scrolling: [...document.querySelectorAll('[data-math="display"]')].filter((el) => el.scrollWidth > el.clientWidth + 1).length,
        clipped: [...document.querySelectorAll('.katex')].filter((el) => !el.closest('[data-math="display"]') && el.getBoundingClientRect().right > innerWidth).length,
      }));
      report.push({ device, slug, ...info });
      await page.screenshot({ path: `${output}/${slug}-${device}.png`, fullPage: true });
    };
    const step = (n) => () => page.getByRole('button', { name: new RegExp(`Step ${n}:`) }).click();
    await visit('under-sampler', '/random-under-sampler', step(2));
    await visit('seed-ensemble', '/seed-ensemble', step(4));
    await visit('classifier-ensemble', '/classifier-ensemble', step(4));
    await visit('feature-selection', '/feature-selection');
    await visit('gallery', '/gallery');
    await visit('consensus-node', '/?node=consensus');
    await visit('confidence-gap', '/?node=interpretation');
    await context.close();
  }
} finally {
  await browser.close();
}
console.table(report);
console.log('console errors:', consoleErrors.length ? consoleErrors : 'none');
