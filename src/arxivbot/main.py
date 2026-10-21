"""FastAPI application for ArxivBot."""

import logging
from contextlib import asynccontextmanager


from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from arxivbot.limiter import limiter
from arxivbot.routes import api, pages
from arxivbot.services.db import check_db, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized")
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="ArxivBot",
    description="Chat with arXiv papers using AI",
    version="0.1.0",
    lifespan=lifespan,
)

# Attach limiter to app state
app.state.limiter = limiter

# Add SlowAPIMiddleware
from slowapi.middleware import SlowAPIMiddleware

app.add_middleware(SlowAPIMiddleware)

WWW_HOST = "www.arxivbot.org"
APEX_ORIGIN = "https://arxivbot.org"


@app.middleware("http")
async def redirect_www_to_apex(request: Request, call_next):
    """Redirect www.arxivbot.org to the apex domain so each page has one URL.

    308 preserves method and body, so POSTs to www keep working. Every other
    host (apex, arxivbot.fly.dev, internal health probes) passes through.
    """
    host = request.headers.get("host", "").split(":")[0].lower()
    if host == WWW_HOST:
        location = APEX_ORIGIN + request.url.path
        if request.url.query:
            location += "?" + request.url.query
        return RedirectResponse(location, status_code=308)
    return await call_next(request)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Return JSON for rate limit errors instead of plain text."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please wait a moment before trying again.",
            "retry_after": exc.retry_after if hasattr(exc, "retry_after") else 60,
        },
        headers={
            "Retry-After": str(exc.retry_after if hasattr(exc, "retry_after") else 60)
        },
    )


# Paths whose errors are not styled pages: the API stays JSON for its clients,
# /ar5iv/ has its own marker error page for the iframe, /static/ serves assets.
PLAIN_ERROR_PREFIXES = ("/api/", "/ar5iv/", "/static/")

NOT_FOUND_MESSAGE = (
    'ArxivBot works on paper pages. Add "bot" after "arxiv" in a paper URL, '
    "or paste an ID below."
)


@app.exception_handler(StarletteHTTPException)
async def html_error_page_handler(request: Request, exc: StarletteHTTPException):
    """Render a styled error page with the paper form for non-API paths.

    Decided by path, not by Accept header. Status codes are unchanged.
    """
    path = request.url.path
    if path.startswith(PLAIN_ERROR_PREFIXES):
        return await http_exception_handler(request, exc)

    context = {
        "status_code": exc.status_code,
        "show_search_link": False,
        "show_url_example": False,
    }
    if exc.status_code == 404:
        context.update(
            heading="Page not found",
            message=NOT_FOUND_MESSAGE,
            show_url_example=True,
        )
    elif exc.status_code == 400 and path.startswith("/abs/"):
        context.update(
            heading="That does not look like an arXiv ID",
            message="Paste an arXiv ID like 1706.03762, or a paper URL, below.",
            show_search_link=True,
        )
    else:
        context.update(
            heading="Something went wrong",
            message="ArxivBot couldn't open that page. Start again with a paper ID below.",
        )

    return pages.templates.TemplateResponse(
        request,
        "error.html",
        context,
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
    )


@app.get("/health")
async def health():
    """Health check endpoint for Fly.io: verifies the database is reachable."""
    try:
        await check_db()
    except Exception:
        logger.exception("Health check failed: database unavailable")
        return JSONResponse(
            {"status": "error", "database": "unavailable"}, status_code=503
        )
    return {"status": "ok", "database": "connected"}


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(api.router)
app.include_router(pages.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
