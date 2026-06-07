"""Code generation endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from op_kingmd.api.server import get_pipeline
from op_kingmd.core.pipeline import (
    CodeGenRequest,
    Framework,
    Language,
    Web3Pipeline,
)

router = APIRouter()


_MAX_CODE  = 500_000   # ~500 KB source limit
_MAX_DESC  = 8_000
_MAX_CTX   = 100_000


class GenerateRequest(BaseModel):
    description: str = Field(..., min_length=10, max_length=_MAX_DESC,
                             description="Natural language description of what to build")
    language: Language = Language.SOLIDITY
    framework: Framework = Framework.FOUNDRY
    include_tests: bool = True
    include_docs: bool = True
    include_deploy_script: bool = True
    gas_optimize: bool = True
    context: str | None = Field(None, max_length=_MAX_CTX, description="Extra code context or ABI")

    model_config = {"use_enum_values": True}


class ReviewRequest(BaseModel):
    code: str = Field(..., min_length=20, max_length=_MAX_CODE)
    language: Language = Language.SOLIDITY

    model_config = {"use_enum_values": True}


class ExplainRequest(BaseModel):
    code: str = Field(..., min_length=10, max_length=_MAX_CODE)
    language: Language = Language.SOLIDITY

    model_config = {"use_enum_values": True}


class OptimizeRequest(BaseModel):
    code: str = Field(..., min_length=10, max_length=_MAX_CODE)
    language: Language = Language.SOLIDITY

    model_config = {"use_enum_values": True}


class FreeformRequest(BaseModel):
    prompt: str = Field(..., min_length=5, max_length=_MAX_DESC)
    system: str | None = Field(None, max_length=4_000)
    stream: bool = False


@router.post("/generate")
async def generate_contract(req: GenerateRequest, request: Request):
    pipeline: Web3Pipeline = get_pipeline(request)
    codegen_req = CodeGenRequest(
        description=req.description,
        language=Language(req.language),
        framework=Framework(req.framework),
        include_tests=req.include_tests,
        include_docs=req.include_docs,
        include_deploy_script=req.include_deploy_script,
        gas_optimize=req.gas_optimize,
        context=req.context,
    )
    result = pipeline.generate_contract(codegen_req)
    return {
        "code": result.code,
        "tests": result.tests,
        "docs": result.docs,
        "deploy_script": result.deploy_script,
        "warnings": result.warnings,
        "model_id": result.model_id,
        "language": result.language.value,
        "framework": result.framework.value,
    }


@router.post("/review")
async def review_code(req: ReviewRequest, request: Request):
    pipeline: Web3Pipeline = get_pipeline(request)
    review = pipeline.review_contract(req.code, Language(req.language))
    return {"review": review}


@router.post("/explain")
async def explain_code(req: ExplainRequest, request: Request):
    pipeline: Web3Pipeline = get_pipeline(request)
    explanation = pipeline.explain_code(req.code, Language(req.language))
    return {"explanation": explanation}


@router.post("/optimize-gas")
async def optimize_gas(req: OptimizeRequest, request: Request):
    pipeline: Web3Pipeline = get_pipeline(request)
    optimized = pipeline.optimize_gas(req.code, Language(req.language))
    return {"optimized_code": optimized}


@router.post("/freeform")
async def freeform(req: FreeformRequest, request: Request):
    from fastapi.responses import StreamingResponse

    pipeline: Web3Pipeline = get_pipeline(request)
    if req.stream:
        def token_stream():
            for token in pipeline.stream_freeform(req.prompt, req.system):
                yield token
        return StreamingResponse(token_stream(), media_type="text/plain")
    return {"response": pipeline.freeform(req.prompt, req.system)}
