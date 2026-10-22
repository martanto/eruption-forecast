import path from "node:path";
import { fileURLToPath } from "node:url";
import js from "@eslint/js";
import tseslint from "typescript-eslint";
import globals from "globals";
import checkFile from "eslint-plugin-check-file";
import reactHooks from "eslint-plugin-react-hooks";

const SOURCE_ROOT = fileURLToPath(new URL("./src/", import.meta.url));
const FEATURES = ["architecture", "explorer", "scenarios", "guided-tour"];
const boundaryRule = {
  meta: {
    type: "problem",
    schema: [],
    messages: {
      upward:
        "Import violates the app → routes/layouts → features → shared → lib → config dependency direction.",
      deep: "Cross-feature imports must use the target feature index.ts.",
    },
  },
  create(context) {
    function check(node) {
      const value = node.source?.value;
      if (
        typeof value !== "string" ||
        (!value.startsWith(".") && !value.startsWith("@/"))
      )
        return;
      const from = path
        .relative(SOURCE_ROOT, context.filename)
        .replaceAll("\\", "/");
      const absolute = value.startsWith("@/")
        ? path.resolve(SOURCE_ROOT, value.slice(2))
        : path.resolve(path.dirname(context.filename), value);
      const to = path.relative(SOURCE_ROOT, absolute).replaceAll("\\", "/");
      if (
        from.startsWith("features/architecture/") &&
        to.startsWith("features/") &&
        !to.startsWith("features/architecture/")
      )
        context.report({ node, messageId: "upward" });
      const rank = (file) =>
        ({
          app: 5,
          layouts: 4,
          routes: 4,
          features: 3,
          components: 2,
          hooks: 2,
          stores: 2,
          types: 2,
          lib: 1,
          config: 0,
        })[file.split("/")[0]];
      if (
        rank(from) !== undefined &&
        rank(to) !== undefined &&
        rank(to) > rank(from)
      )
        context.report({ node, messageId: "upward" });
      if (to.startsWith("features/")) {
        const target = to.split("/")[1];
        const own = from.startsWith(`features/${target}/`);
        if (
          !own &&
          to.split("/").length > 2 &&
          !/^features\/[^/]+\/index(?:\.ts)?$/.test(to)
        )
          context.report({ node, messageId: "deep" });
      }
    }
    return {
      ImportDeclaration: check,
      ExportNamedDeclaration: check,
      ExportAllDeclaration: check,
      ImportExpression(node) {
        check({ ...node, source: node.source });
      },
    };
  },
};
export default tseslint.config(
  {
    ignores: [
      "node_modules/**",
      "dist/**",
      "tests/results/**",
      "tests/reports/**",
      "tests/coverage/**",
      "tests/pages-build/**",
      "tests/.playwright-cli/**",
    ],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  { languageOptions: { globals: { ...globals.browser, ...globals.node } } },
  {
    files: ["tests/review-capture.js", "tests/cli-verification.js"],
    rules: { "@typescript-eslint/no-unused-expressions": "off" },
  },
  {
    files: ["src/**/*.{ts,tsx}"],
    plugins: {
      "check-file": checkFile,
      "react-hooks": reactHooks,
      local: { rules: { boundaries: boundaryRule } },
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "local/boundaries": "error",
      "no-restricted-syntax": [
        "error",
        {
          selector: "ExportDefaultDeclaration",
          message: "Use named exports for application modules.",
        },
      ],
      "check-file/folder-naming-convention": [
        "error",
        { "src/**/": "KEBAB_CASE" },
      ],
      "check-file/filename-naming-convention": [
        "error",
        { "src/**/*.ts": "CAMEL_CASE" },
        { ignoreMiddleExtensions: true },
      ],
    },
  },
  {
    files: ["src/**/*.tsx"],
    ignores: [
      "src/main.tsx",
      "src/app/providers.tsx",
      "src/app/router.tsx",
      "src/components/ui/**",
    ],
    rules: {
      "check-file/filename-naming-convention": [
        "error",
        { "**/*.tsx": "PASCAL_CASE" },
      ],
    },
  },
  {
    files: ["src/components/ui/*.tsx"],
    rules: {
      "check-file/filename-naming-convention": [
        "error",
        { "**/*.tsx": "KEBAB_CASE" },
      ],
    },
  },
  ...FEATURES.map((feature) => ({
    files: [`src/features/${feature}/**/*.{ts,tsx}`],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              regex: `^(?:@/features/|(?:\\.\\./)+)(?:${FEATURES.filter((item) => item !== feature).join("|")})/(?!index(?:\\.ts)?$)`,
              message: "Use the feature public API.",
            },
            {
              regex: "^@/(?:app|routes|layouts)(?:/|$)",
              message: "Features must not import upward.",
            },
          ],
        },
      ],
    },
  })),
  {
    files: ["src/{components,lib,hooks,config,types,stores}/**/*.{ts,tsx}"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              regex:
                "^(?:@/|(?:\\.\\./)+)(?:features|routes|layouts|app)(?:/|$)",
              message:
                "Shared modules cannot import application features or routes.",
            },
          ],
        },
      ],
    },
  },
);
