import { test, expect, type Page } from "@playwright/test";

async function inspect(page: Page, mobile: boolean) {
  if (mobile)
    await page
      .getByRole("button", { name: "Inspect selection", exact: true })
      .click();
  return mobile
    ? page.getByRole("dialog").getByRole("region", { name: "Node details" })
    : page.locator(".desktop-inspector");
}

test("selection, view, scenario and history survive navigation and reload", async ({
  page,
  isMobile,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/?node=consensus&scenario=scenario-3-2");
  await expect(
    page.locator('.react-flow__node[data-id="consensus"]'),
  ).toHaveClass(/selected/);
  await page.getByRole("button", { name: "Timeline", exact: true }).click();
  await expect(page).toHaveURL(/view=timeline/);
  await expect(page).toHaveURL(/node=consensus/);
  await expect(page.getByLabel("Example scenario")).toHaveValue("scenario-3-2");
  await page.getByLabel("Example scenario").selectOption("scenario-5-2");
  await page.goBack();
  await expect(page.getByLabel("Example scenario")).toHaveValue("scenario-3-2");
  await page.goForward();
  await expect(page.getByLabel("Example scenario")).toHaveValue("scenario-5-2");
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Timeline", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  const details = await inspect(page, isMobile);
  await expect(
    details.getByRole("heading", { name: "Illustrative calculation" }),
  ).toBeVisible();
  await expect(
    details.getByRole("cell", { name: "0.600", exact: true }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("search, caveats, source excerpts, expansion, keyboard selection and tour", async ({
  page,
  isMobile,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Caveats" }).click();
  const results = page.getByRole("region", { name: "Search results" });
  await expect(
    results.getByRole("button", { name: "P-value ranking & top-N" }),
  ).toBeVisible();
  await expect(
    results.getByRole("button", { name: "SDS / FDSN sources" }),
  ).toHaveCount(0);
  await page
    .getByLabel("Search stages, parameters and sources")
    .fill("p-value");
  await results
    .getByRole("button", { name: "P-value ranking & top-N" })
    .click();
  await expect(page).toHaveURL(/node=ranking/);
  const details = isMobile
    ? page.getByRole("dialog")
    : page.locator(".desktop-inspector");
  await expect(
    details.getByText("Top 20 is not an FDR guarantee", { exact: true }),
  ).toBeVisible();
  await details.getByRole("button", { name: /Sources/ }).click();
  await details
    .getByText("src/eruption_forecast/utils/ml.py", { exact: false })
    .first()
    .click();
  await expect(details.locator("pre").first()).toContainText(
    "selector.p_values",
  );
  if (isMobile) await page.getByLabel("Close Inspect selection").click();
  await page.getByRole("button", { name: "Take the guided tour" }).click();
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page).toHaveURL(/node=labels/);
  await page.getByLabel("Previous tour step").click();
  await expect(page).toHaveURL(/node=tremor/);
  await page.getByLabel("Close tour").click();
  await page.getByRole("button", { name: "Focus stage", exact: true }).click();
  const node = page.locator('.react-flow__node[data-id="tremor"]');
  await node.getByRole("button", { name: "Explore steps" }).click();
  await expect(
    page.locator('.react-flow__node[data-id="sources"]'),
  ).toBeVisible();
  const child = page.locator('.react-flow__node[data-id="sources"]');
  await child.focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/node=sources/);
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await expect(
    page.locator('.react-flow__node[data-id="metrics"]'),
  ).toHaveCount(0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test("pan, zoom, minimap and per-view viewport retention", async ({ page }) => {
  await page.goto("/");
  const viewport = page.locator(".react-flow__viewport");
  await expect(
    page.locator('.react-flow__node[data-id="tremor"]'),
  ).toBeVisible();
  const initial = await viewport.getAttribute("style");
  await page.getByRole("button", { name: "Zoom In" }).click();
  await expect(viewport).not.toHaveAttribute("style", initial!);
  const pane = await page.locator(".react-flow__pane").boundingBox();
  if (!pane) throw new Error("Missing canvas pane");
  const beforePan = await viewport.getAttribute("style");
  await page.mouse.move(pane.x + pane.width * 0.5, pane.y + 95);
  await page.mouse.down();
  await page.mouse.move(pane.x + pane.width * 0.5 + 65, pane.y + 125, {
    steps: 6,
  });
  await page.mouse.up();
  await expect(viewport).not.toHaveAttribute("style", beforePan!);
  const mini = page.locator(".react-flow__minimap");
  await expect(mini.locator(".react-flow__minimap-node")).toHaveCount(7);
  const beforeMini = await viewport.getAttribute("style");
  await mini.click({ position: { x: 15, y: 15 } });
  await expect(viewport).not.toHaveAttribute("style", beforeMini!);
  const pipelineViewport = await viewport.getAttribute("style");
  await page.getByRole("button", { name: "Timeline", exact: true }).click();
  await expect(page.locator(".timeline-periods")).toBeVisible();
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await expect(viewport).toHaveAttribute("style", pipelineViewport!);
});

test("mobile drawers trap focus and return focus to their trigger", async ({
  page,
  isMobile,
}) => {
  test.skip(!isMobile, "Drawer-specific touch layout");
  await page.goto("/");
  const trigger = page.getByRole("button", {
    name: "Stages & scenarios",
    exact: true,
  });
  await trigger.click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await dialog
    .getByRole("button", { name: "04 Seed preparation & training" })
    .click();
  await expect(dialog).not.toBeVisible();
  await expect(page).toHaveURL(/node=training/);
  await page
    .getByRole("button", { name: "Inspect selection", exact: true })
    .click();
  await expect(
    page
      .getByRole("dialog")
      .getByText("Randomized CV is not future validation", { exact: true }),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("button", { name: "Inspect selection", exact: true }),
  ).toBeFocused();
});
