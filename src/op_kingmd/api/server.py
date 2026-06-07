"""FastAPI application — REST API for Op-Kingmd inference, audit, and GitHub tools."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

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


def create_app() -> FastAPI:
    app = FastAPI(
        title="Op-Kingmd API",
        description=(
            "Web3-focused coding LLM with mandatory Audit-Before-Deploy layer. "
            "Supports Solidity, Rust, TypeScript, Foundry, and Hardhat workflows."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = round((time.perf_counter() - t0) * 1000)
        response.headers["X-Response-Time"] = f"{elapsed}ms"
        return response

    app.include_router(codegen_router, prefix="/v1/codegen", tags=["Code Generation"])
    app.include_router(audit_router, prefix="/v1/audit", tags=["Audit"])
    app.include_router(github_router, prefix="/v1/github", tags=["GitHub"])

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": os.getenv("OPKMD_MODEL_ID", "Qwen/Qwen2.5-Coder-7B-Instruct"),
            "device": _engine.get_device() if _engine else "not_loaded",
        }

    @app.get("/")
    async def root():
        return {
            "name": "Op-Kingmd",
            "version": "0.1.0",
            "description": "Web3-focused coding LLM with Audit-Before-Deploy layer",
            "docs": "/docs",
        }

    return app


def get_pipeline(request: Request) -> Web3Pipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return pipeline


def require_api_key(x_api_key: str = Header(...)):
    expected = os.getenv("OPKMD_API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
