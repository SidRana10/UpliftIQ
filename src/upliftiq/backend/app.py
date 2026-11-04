"""Application factory: loads the approved model once, mounts routers, handles errors and CORS."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from upliftiq.backend.reports import Reports, ReportUnavailable
from upliftiq.backend.routes_inference import router as inference_router
from upliftiq.backend.routes_reports import router as reports_router
from upliftiq.backend.settings import Settings
from upliftiq.modeling.criteo_package import (
    CriteoUpliftModel,
    InputValidationError,
    PackageIntegrityError,
)

API_VERSION = "v1"


class ModelUnavailable(RuntimeError):
    """The approved Criteo model could not be loaded."""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model, app.state.model_error = None, None
        try:  # loaded once; requests reuse it. A missing/corrupt package degrades, not crashes.
            app.state.model = CriteoUpliftModel.load(settings.package_dir)
        except (OSError, ValueError, KeyError, PackageIntegrityError) as e:
            app.state.model_error = f"{type(e).__name__}: {e}"
        yield

    app = FastAPI(
        title="UpliftIQ API",
        version="1.0.0",
        description=(
            "Read-only results from validated Hillstrom and Criteo experiments plus inference "
            "from the approved Criteo S-learner. Criteo is an anonymized benchmark: no ROI claims."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.reports = Reports(settings.root)
    app.state.model, app.state.model_error = None, None
    app.state.model_lock = threading.Lock()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(ReportUnavailable)
    async def _report_missing(_: Request, exc: ReportUnavailable):
        return JSONResponse({"detail": str(exc), "code": "report_unavailable"}, status_code=404)

    @app.exception_handler(ModelUnavailable)
    async def _model_missing(_: Request, exc: ModelUnavailable):
        return JSONResponse({"detail": str(exc), "code": "model_unavailable"}, status_code=503)

    @app.exception_handler(InputValidationError)
    async def _bad_input(_: Request, exc: InputValidationError):
        return JSONResponse({"detail": str(exc), "code": "invalid_input"}, status_code=422)

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(_: Request, exc: RequestValidationError):
        # report location and message only: echoing raw input could break JSON (NaN) or leak data
        errors = [
            {"loc": [str(x) for x in e["loc"]], "msg": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        return JSONResponse({"detail": errors, "code": "invalid_request"}, status_code=422)

    app.include_router(reports_router, prefix=f"/api/{API_VERSION}")
    app.include_router(inference_router, prefix=f"/api/{API_VERSION}")
    return app
