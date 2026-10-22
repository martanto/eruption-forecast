import { fileURLToPath, URL } from "node:url";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  server: {
    watch: {
      ignored: ["**/tests/results/**", "**/tests/reports/**", "**/tests/hallmark-runtime/**"],
    },
    fs: {
      allow: [
        fileURLToPath(new URL(".", import.meta.url)),
        fileURLToPath(new URL("../wiki", import.meta.url)),
        fileURLToPath(new URL("../assets", import.meta.url)),
      ],
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
    coverage: { reportsDirectory: "./tests/coverage" },
  },
});
