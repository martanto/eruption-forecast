async (page) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('http://127.0.0.1:5179/');
  await page.evaluate(() => document.fonts.ready);
  await page.locator('.react-flow__node[data-id="tremor"]').waitFor();
  await page.screenshot({ path: 'review/desktop.png', fullPage: true });
  await page.getByRole('button', { name: 'Timeline', exact: true }).click();
  await page.locator('.timeline-periods').waitFor();
  await page.screenshot({ path: 'review/timeline.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('http://127.0.0.1:5179/');
  await page.evaluate(() => document.fonts.ready);
  await page.locator('.react-flow__node[data-id="tremor"]').waitFor();
  await page.screenshot({ path: 'review/mobile.png', fullPage: true });
  await page.getByRole('button', { name: 'Inspect selection', exact: true }).click();
  await page.getByRole('dialog').waitFor();
  await page.screenshot({ path: 'review/mobile-inspector.png', fullPage: true });
  return { overflow: await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), title: await page.title() };
}
