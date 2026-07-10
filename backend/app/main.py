"""FastAPI application factory and wiring.

Built as a factory (``create_app``) rather than a module-level ``app = FastAPI()``
so tests can construct an isolated instance per test session with their own
settings and database, without import-order side effects.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_settings
from app.core.exceptions import FormVisionError
from app.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare runtime state on startup.

    Directory creation and schema creation happen here rather than at import
    time so that merely importing the module (as tooling and tests do) never
    writes to disk.
    """
    settings: Settings = get_settings()
    settings.ensure_directories()

    # Imported lazily: keeps `import app.main` free of database side effects
    # and avoids a circular import between the app factory and the db layer.
    from app.db.database import init_database

    init_database()

    logger.info(
        "%s v%s starting (environment=%s, estimator=%s, data_dir=%s)",
        settings.app_name,
        settings.app_version,
        settings.environment,
        settings.pose_estimator,
        settings.data_dir,
    )
    if not settings.pose_model_path.exists():
        logger.warning(
            "Pose model not found at %s. It will be downloaded on first analysis.",
            settings.pose_model_path,
        )

    yield

    logger.info("%s shutting down", settings.app_name)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Computer-vision squat analysis. Upload a back-squat video, receive "
            "pose landmarks, repetition counts, joint angles, squat metrics, and "
            "rule-based coaching feedback."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # The browser cannot read Content-Range on a cross-origin media request
        # unless it is explicitly exposed; without this, video seeking breaks.
        expose_headers=["Content-Range", "Content-Length", "Accept-Ranges"],
    )

    _register_exception_handlers(app)
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    """Mount every route module."""
    from app.api.routes import health

    app.include_router(health.router)


def _register_exception_handlers(app: FastAPI) -> None:
    """Translate exceptions into the single error envelope shape.

    Handlers are registered once here so no route handler needs try/except
    around domain calls — they raise a typed error and it is rendered
    consistently.
    """

    @app.exception_handler(FormVisionError)
    async def handle_domain_error(_: Request, exc: FormVisionError) -> JSONResponse:
        # Client mistakes are expected traffic and logged at INFO; anything 5xx
        # is a real defect and gets a full stack trace.
        if exc.status_code >= 500:
            logger.exception("Domain error: %s", exc.message)
        else:
            logger.info("Rejected request (%s): %s", exc.code, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "REQUEST_VALIDATION_ERROR",
                    "message": "The request body or parameters were invalid.",
                    # jsonable_encoder-safe: pydantic errors can hold exceptions.
                    "detail": {"errors": exc.errors()},
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        _: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "HTTP_ERROR",
                    "message": str(exc.detail),
                    "detail": {},
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Deliberately does not leak the exception text to the client.
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred.",
                    "detail": {},
                }
            },
        )


app = create_app()
