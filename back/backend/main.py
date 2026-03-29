"""
FastAPI entry point – lifespan, CORS, metrics middleware, router registration.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()  # load .env before anything else

from dependencies import ensure_collection, get_async_qdrant  # noqa: E402
from routers.content import router as content_router  # noqa: E402
from routers.enterprise import router as enterprise_router  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
logger = logging.getLogger(__name__)

# ── Lifespan ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up – initialising Qdrant collection…")
    try:
        client = get_async_qdrant()
        await ensure_collection(client)
        logger.info("Qdrant collection ready.")
    except Exception as exc:
        logger.warning("Qdrant init warning (non-fatal): %s", exc)
    yield
    logger.info("Shutting down.")


# ── App factory ──────────────────────────────────────────────────

app = FastAPI(
    title="EngageTech AI Content Pipeline",
    description=(
        "8-agent LangGraph pipeline for social media content generation. "
        "Supports SSE streaming, human-in-the-loop approval, and enterprise RAG via Qdrant."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Content Pipeline",
            "description": "Generate, stream, and approve social media content through the 8-agent pipeline.",
        },
        {
            "name": "Enterprise Knowledge",
            "description": "Upload, list, and delete enterprise documents indexed in Qdrant.",
        },
    ],
)

# ── CORS ─────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Metrics middleware ────────────────────────────────────────────

@app.middleware("http")
async def metrics_middleware(request: Request, call_next) -> Response:
    start = time.monotonic()
    response: Response = await call_next(request)
    elapsed = round((time.monotonic() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    logger.info("%s %s → %s  (%.2f ms)", request.method, request.url.path, response.status_code, elapsed)
    return response


# ── Routers ───────────────────────────────────────────────────────

app.include_router(content_router)
app.include_router(enterprise_router)


# ── Health check ─────────────────────────────────────────────────

@app.get("/health", tags=["Health"], summary="Health check")
async def health():
    return {"status": "ok", "service": "EngageTech AI Pipeline"}


# ── Global exception handler ──────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
