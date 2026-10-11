"""FastAPI application factory; importing it performs no scientific work."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request
from pydantic import ValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from eruption_forecast_app.settings import AppSettings
from eruption_forecast_app.routes.api import router
from eruption_forecast_app.repositories.database import Database


def create_app(runtime: AppSettings | None = None) -> FastAPI:
    """Construct the local API without opening the database until startup.

    Example:
        >>> app = create_app()
        >>> app.title
        'Eruption Forecast'
    """
    runtime = runtime or AppSettings()
    db = Database(runtime.database)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        db.initialize()
        yield

    application = FastAPI(
        title="Eruption Forecast",
        version="1.0.0",
        description="Local research workflows, configurations, jobs, artifacts, and machine resources.",
        lifespan=lifespan,
    )
    application.state.database = db
    application.state.settings = runtime
    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    @application.middleware("http")
    async def origin_guard(request: Request, call_next):
        origin = request.headers.get("origin")
        if (
            request.method not in {"GET", "HEAD", "OPTIONS"}
            and origin
            and origin
            not in {
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:8000",
                "http://127.0.0.1:8000",
                str(request.base_url).rstrip("/"),
            }
        ):
            return JSONResponse({"detail": "Untrusted request origin"}, status_code=403)
        return await call_next(request)

    @application.exception_handler(ValueError)
    async def value_error(request: Request, exc: ValueError):
        if isinstance(exc, ValidationError):
            return JSONResponse(
                {"detail": exc.errors(include_context=False)}, status_code=422
            )
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @application.exception_handler(KeyError)
    async def missing(request: Request, exc: KeyError):
        return JSONResponse({"detail": "Resource not found"}, status_code=404)

    application.include_router(router)
    if (runtime.frontend / "assets").is_dir():
        application.mount(
            "/assets", StaticFiles(directory=runtime.frontend / "assets"), name="assets"
        )

    @application.get("/", include_in_schema=False)
    def index():
        if (runtime.frontend / "index.html").exists():
            return FileResponse(runtime.frontend / "index.html")
        return JSONResponse(
            {
                "message": "Build app/frontend/ or start Vite on localhost:5173.",
                "docs": "/docs",
            }
        )

    return application
