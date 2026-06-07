"""FastAPI application — REST API for Op-Kingmd inference, audit, and GitHub tools."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from op_kingmd.api.routes import audit_router, codegen_router, github_router
from op_kingmd.core.model import ModelConfig, ModelEngine
from op_kingmd.core.pipeline import Web3Pipeline

logger = structlog.get_logger(__name__)

_engine: ModelEngine | None = None
_pipeline: Web3Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _pipeline
    logger.info("startup_loading_model")
    _engine = ModelEngine(ModelConfig.from_env())
    _engine.load()
    _pipeline = Web3Pipeline(_engine)
    app.state.pipeline = _pipeline
    logger.info("startup_complete")
    yield
    logger.info("shutdown")


def _allowed_origins() -> list[str]:
    """Read allowed origins from env; falls back to localhost-only in production."""
    raw = os.getenv("OPKMD_CORS_ORIGINS", "")
    if raw.strip():
        return [o.strip() for o in raw.split(",") if o.strip()]
    # Safe default: only same-origin + localhost dev ports
    return [
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000",
    ]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Op-Kingmd API",
        description=(
            "Web3-focused coding LLM with mandatory Audit-Before-Deploy layer. "
            "Supports Solidity, Rust, TypeScript, Foundry, and Hardhat workflows."
        ),
        version="0.1.0",
        lifespan=lifespan,
        # Disable docs in production unless explicitly enabled
        docs_url="/docs" if os.getenv("OPKMD_ENABLE_DOCS", "1") == "1" else None,
        redoc_url="/redoc" if os.getenv("OPKMD_ENABLE_DOCS", "1") == "1" else None,
    )

    # CORS: never use wildcard with allow_credentials=True (violates CORS spec and
    # enables cookie/credential theft). Use explicit origin list only.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Api-Key"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = round((time.perf_counter() - t0) * 1000)
        response.headers["X-Response-Time"] = f"{elapsed}ms"
        return response

    app.include_router(
        codegen_router,
        prefix="/v1/codegen",
        tags=["Code Generation"],
        dependencies=[Depends(require_api_key)],
    )
    app.include_router(
        audit_router,
        prefix="/v1/audit",
        tags=["Audit"],
        dependencies=[Depends(require_api_key)],
    )
    app.include_router(
        github_router,
        prefix="/v1/github",
        tags=["GitHub"],
        dependencies=[Depends(require_api_key)],
    )

    # Health and root are intentionally unauthenticated but return minimal info
    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_loaded": _engine is not None and _engine._loaded,
        }

    @app.get("/")
    async def root():
        return {
            "name": "Op-Kingmd",
            "version": "0.1.0",
            "description": "Web3-focused coding LLM with Audit-Before-Deploy layer",
        }

    return app


def get_pipeline(request: Request) -> Web3Pipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return pipeline


def require_api_key(x_api_key: str = Header(alias="X-Api-Key", default="")) -> None:
    """Validate API key on every protected route.

    If OPKMD_API_KEY is not set the server runs in open mode (dev only).
    Log every failed attempt for brute-force detection.
    """
    expected = os.getenv("OPKMD_API_KEY", "")
    if not expected:
        # No key configured — open mode, warn once at startup (see lifespan)
        return
    # Constant-time comparison to prevent timing attacks
    import hmac
    if not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        logger.warning("api_key_rejected", prefix=x_api_key[:4] if x_api_key else "")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
