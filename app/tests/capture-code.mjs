import { chromium } from '@playwright/test';
import { mkdir } from 'node:fs/promises';

// Renders every page that shows Python snippets and reports highlighting and overflow.
const base = process.env.BASE_URL ?? 'http://127.0.0.1:5199';
const output = 'tests/review/code';
await mkdir(output, { recursive: true });
const browser = await chromium.launch();
const report = [];
const consoleErrors = [];
try {
  for (const [device, viewport] of [['desktop', { width: 1440, height: 1000 }], ['mobile', { width: 390, height: 844 }]]) {
    for (const mode of ['light', 'dark']) {
      const context = await browser.newContext({ viewport, colorScheme: mode, reducedMotion: 'reduce' });
      const page = await context.newPage();
      page.on('console', (m) => m.type() === 'error' && consoleErrors.push(`${device}/${mode} ${m.text()}`));
      page.on('pageerror', (e) => consoleErrors.push(`${device}/${mode} pageerror ${e.message}`));
      const visit = async (slug, url, prepare) => {
        await page.goto(`${base}${url}`);
        await page.evaluate(() => document.fonts.ready);
        if (prepare) await prepare();
        await page.waitForTimeout(300);
        const info = await page.evaluate(() => ({
          blocks: document.querySelectorAll('.code-snippet').length,
          inline: document.querySelectorAll('.code-inline').length,
          tokens: document.querySelectorAll('.code-snippet .token, .code-inline .token').length,
          wikiBlocks: document.querySelectorAll('.wiki-guide .code-snippet').length, plainPre: document.querySelectorAll('.wiki-guide pre:not(.code-snippet)').length,
          overflow: document.documentElement.scrollWidth > innerWidth,
        }));
        report.push({ device, mode, slug, ...info });
        await page.screenshot({ path: `${output}/${slug}-${device}-${mode}.png`, fullPage: true });
      };
      await visit('labelling', '/labelling');
      await visit('scenarios', '/prediction?part=scenarios');
      // On narrow screens the inspector lives in a drawer.
      const openInspector = async () => {
        if (viewport.width < 1000) await page.getByRole('button', { name: 'Inspect selection' }).click();
      };
      await visit('source-excerpt', '/?node=training', async () => {
        await openInspector();
        await page.locator('.inspector-tabs button:visible', { hasText: 'Sources' }).first().click();
        await page.locator('.source-list summary:visible').first().click();
      });
      for (const node of ['labels', 'training', 'prediction', 'tremor']) await visit('wiki-' + node, '/?node=' + node, openInspector);
      await visit('grid-cv', '/grid-cv', () => page.getByRole('button', { name: /Step 4:/ }).click());
      await visit('gallery', '/gallery');
      await context.close();
    }
  }
} finally {
  await browser.close();
}
console.table(report);
console.log('console errors:', consoleErrors.length ? consoleErrors : 'none');
