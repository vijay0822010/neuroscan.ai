"""
NeuroScan AI — FastAPI application.
Single process serves both the REST API and the built React SPA.
Railway injects PORT automatically; health check hits /health.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.routers import analysis, report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# frontend/dist is created by `npm run build` during Railway's build phase
_ROOT       = Path(__file__).resolve().parent.parent
_STATIC_DIR = _ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure upload/report dirs exist (use /tmp on Railway)
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.REPORT_DIR).mkdir(parents=True, exist_ok=True)

    port = os.environ.get("PORT", settings.PORT)
    key  = settings.GROQ_API_KEY
    logger.info(f"NeuroScan AI v{settings.APP_VERSION} — port={port}")
    logger.info(f"Groq key: {key[:18]}… | model: {settings.GROQ_MODEL}")
    logger.info(
        f"Frontend: {'OK — ' + str(_STATIC_DIR) if _STATIC_DIR.is_dir() else 'NOT FOUND — build first'}"
    )
    yield


app = FastAPI(
    title="NeuroScan AI",
    description="Neurological Risk Assessment — ResNet-512 + Wav2Vec2-768 → NSS",
    version=settings.APP_VERSION,
    docs_url="/docs",
    lifespan=lifespan,
)

# CORS — needed for local dev; Railway same-origin doesn't need it
# but keep it open so Swagger UI and external tools work
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REST API routes ────────────────────────────────────────────────────────────
app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])
app.include_router(report.router,   prefix="/api/v1", tags=["Report"])


# ── Health check ── Railway hits this to confirm the app is alive ─────────────
@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": settings.APP_VERSION}


# ── Error handlers ─────────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def _val_err(request: Request, exc: RequestValidationError):
    return JSONResponse(422, {"detail": exc.errors()})

@app.exception_handler(Exception)
async def _generic_err(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(500, {"detail": f"{type(exc).__name__}: {exc}"})


# ── Serve React SPA ────────────────────────────────────────────────────────────
# After `npm run build`, frontend/dist contains:
#   index.html          ← SPA entry point
#   assets/             ← hashed JS/CSS chunks
#
# FastAPI serves:
#   /           → index.html
#   /assets/*   → hashed chunks (immutable cache)
#   /api/v1/*   → Python API  (registered above, matched first)
#   /*          → index.html  (React Router client-side navigation)

if _STATIC_DIR.is_dir():
    _assets_dir = _STATIC_DIR / "assets"
    if _assets_dir.is_dir():
        # Mount /assets as static files — served directly with long cache headers
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

    @app.get("/favicon.ico", include_in_schema=False)
    async def _favicon():
        ico = _STATIC_DIR / "favicon.ico"
        return FileResponse(str(ico)) if ico.exists() else JSONResponse({}, status_code=204)

    # SPA catch-all — must be last route registered
    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_catchall(full_path: str):
        index = _STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse(
            {"detail": "Frontend not built yet."},
            status_code=503,
        )

else:
    # Frontend not built — return a plain JSON response at /
    logger.warning("frontend/dist not found — frontend not built")

    @app.get("/", include_in_schema=False)
    async def _no_frontend():
        return JSONResponse({
            "status":  "API running",
            "version": settings.APP_VERSION,
            "docs":    "/docs",
            "note":    "Frontend not built. Railway build should run `npm run build` automatically.",
        })
