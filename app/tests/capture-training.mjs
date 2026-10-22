import { chromium } from '@playwright/test';
import { mkdir } from 'node:fs/promises';

const output = 'tests/review/training';
await mkdir(output, { recursive: true });
const browser = await chromium.launch();
const findings = [];
try {
  for (const [device, viewport] of [['desktop', { width: 1440, height: 1000 }], ['mobile', { width: 390, height: 844 }]]) {
    for (const mode of ['light', 'dark']) {
      const context = await browser.newContext({ viewport, colorScheme: mode, reducedMotion: 'reduce' });
      const page = await context.newPage();
      for (const [slug, route] of [['rus', 'random-under-sampler'], ['seeds', 'seed-ensemble'], ['classifiers', 'classifier-ensemble']]) {
        await page.goto(`http://127.0.0.1:5178/${route}`);
        await page.evaluate(() => document.fonts.ready);
        if (mode === 'light') {
          await page.screenshot({ path: `${output}/${slug}-${device}-start.png`, fullPage: true });
        }
        await page.getByRole('button', { name: /Step 4:/ }).click();
        await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); });
        const suffix = mode === 'dark' ? '-dark' : '';
        await page.screenshot({ path: `${output}/${slug}-${device}${suffix}.png`, fullPage: true });
        findings.push(await page.evaluate(() => ({ route: location.pathname, mode: document.documentElement.dataset.mode, width: innerWidth, overflow: document.documentElement.scrollWidth > innerWidth, skipClipped: getComputedStyle(document.querySelector('.skip-link')).clipPath })));
        await page.locator('.skip-link').focus();
        if (await page.locator('.skip-link').evaluate((el) => getComputedStyle(el).clipPath !== 'none')) throw new Error('Focused skip link is clipped');
      }
      if (mode === 'light') {
        await page.goto('http://127.0.0.1:5178/labelling');
        await page.evaluate(() => document.fonts.ready);
        await page.screenshot({ path: `${output}/incumbent-labelling-${device}.png`, fullPage: true });
        await page.goto('http://127.0.0.1:5178/');
        await page.getByRole('link', { name: 'Training walkthroughs' }).waitFor();
        if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Atlas toolbar overflows');
        await page.getByRole('link', { name: 'Training walkthroughs' }).click();
        if (!page.url().endsWith('/random-under-sampler')) throw new Error('Atlas training link failed');
      }
      await context.close();
    }
  }
  console.log(JSON.stringify(findings, null, 2));
} finally {
  await browser.close();
}
