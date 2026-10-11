# Eruption Forecast backend

Independent `eruption-forecast-app` Python package built with `uv_build`. It depends on the scientific `eruption-forecast` package; the UV development source maps that dependency to the repository root.

From the repository root:

```powershell
uv sync --project app/backend --group dev
uv run --project app/backend eruption-ui dev
```

The API binds `127.0.0.1:8000`. `serve` runs HTTP only; `worker` runs the persistent queue; `export-openapi` updates root `openapi.json`. Set `ERUPTION_APP_STATE` to isolate metadata. Full workflows, PyPI dependency mode, directory structure and testing commands are in [UI Workflow](../../wiki/UI-Workflow.md).
