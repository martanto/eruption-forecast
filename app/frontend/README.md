# Eruption Forecast frontend

React, TypeScript, Vite, Tailwind v4 and modular ECharts. Start the backend API and worker first, then:

```powershell
npm --prefix app/frontend install
npm --prefix app/frontend run dev
```

Open `http://127.0.0.1:5173`. Build with `npm --prefix app/frontend run build`; the backend serves the resulting `dist/` directory on restart. Tests live in `tests/`, including `tests/e2e/` for Playwright. Generated API types are derived from root `openapi.json`.

See [UI Workflow](../../wiki/UI-Workflow.md) and the root [design system](../../DESIGN.md).
