import assert from "node:assert/strict";
import { ESLint } from "eslint";
const eslint = new ESLint();
const cases = [
  ["src/features/explorer/Probe.tsx", "@/features/architecture/content", false],
  ["src/features/explorer/Probe.tsx", "../architecture/content", false],
  [
    "src/features/explorer/components/Probe.tsx",
    "../../architecture/content",
    false,
  ],
  ["src/features/explorer/Probe.tsx", "@/features/architecture", true],
  ["src/features/explorer/Probe.tsx", "../architecture/index", true],
  ["src/features/explorer/Probe.tsx", "./components/AtlasNode", true],
  ["src/components/common/Probe.tsx", "@/features/architecture", false],
  ["src/components/common/Probe.tsx", "../../features/architecture", false],
  [
    "src/components/common/Probe.tsx",
    "../../features/../features/explorer",
    false,
  ],
  ["src/lib/probe.ts", "../components/ui/button", false],
  ["src/features/architecture/probe.ts", "@/routes/ExplorerPage", false],
  ["src/features/explorer/components/Probe.tsx", "@/lib/utils", true],
  ["src/features/architecture/probe.ts", "../explorer", false],
  ["src/features/architecture/probe.ts", "@/features/scenarios", false],
];
for (const [filePath, target, allowed] of cases) {
  const [result] = await eslint.lintText(`import '${target}';`, { filePath });
  const errors = result.messages.filter((message) =>
    ["local/boundaries", "no-restricted-imports"].includes(message.ruleId),
  );
  assert.equal(errors.length === 0, allowed, `${filePath}: ${target}`);
}
for (const filePath of [
  "src/main.tsx",
  "src/app/providers.tsx",
  "src/app/router.tsx",
  "src/features/explorer/components/ValidNode.tsx",
]) {
  const [result] = await eslint.lintText("export const value = 1;", {
    filePath,
  });
  assert.ok(
    !result.messages.some((message) =>
      message.ruleId?.startsWith("check-file/"),
    ),
    filePath,
  );
}
const [invalid] = await eslint.lintText("export const value = 1;", {
  filePath: "src/features/explorer/components/bad-component.tsx",
});
assert.ok(
  invalid.messages.some(
    (message) => message.ruleId === "check-file/filename-naming-convention",
  ),
);
console.log(
  `Boundary checks passed: ${cases.length} imports, fixed entrypoints and nested component naming.`,
);
