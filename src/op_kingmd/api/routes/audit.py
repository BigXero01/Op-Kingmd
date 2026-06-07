"""Audit-Before-Deploy API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from op_kingmd.audit.analyzer import AuditEngine
from op_kingmd.audit.report import AuditReport, ReportGenerator

router = APIRouter()

_engine = AuditEngine()
_reporter = ReportGenerator()

_ALLOWED_FORMATS = {"json", "markdown", "html"}
_MAX_SOURCE = 500_000   # 500 KB hard cap — prevent memory exhaustion / ReDoS
_MAX_NAME   = 100


class AuditSourceRequest(BaseModel):
    source: str = Field(..., min_length=20, max_length=_MAX_SOURCE,
                        description="Solidity source code (max 500 KB)")
    contract_name: str | None = Field(None, max_length=_MAX_NAME,
                                      pattern=r"^[A-Za-z_]\w*$")
    chain: str = Field("ethereum", max_length=50, pattern=r"^[a-z0-9_-]+$")
    network: str = Field("mainnet", max_length=50, pattern=r"^[a-z0-9_-]+$")
    formats: list[str] = ["json", "markdown"]

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, v: list[str]) -> list[str]:
        invalid = set(v) - _ALLOWED_FORMATS
        if invalid:
            raise ValueError(f"Unsupported formats: {invalid}. Allowed: {_ALLOWED_FORMATS}")
        return v


class AuditPathRequest(BaseModel):
    path: str = Field(..., description="Path to .sol file or directory (server-side)")
    chain: str = Field("ethereum", max_length=50, pattern=r"^[a-z0-9_-]+$")
    network: str = Field("mainnet", max_length=50, pattern=r"^[a-z0-9_-]+$")


@router.post("/source")
async def audit_source(req: AuditSourceRequest):
    """Audit a Solidity source string. Returns 422 if deploy should be blocked."""
    result = _engine.audit(req.source, req.contract_name)
    report = AuditReport.from_result(result, chain=req.chain, network=req.network)

    if result.blocks_deploy:
        # Return score and risk label only — do not leak full finding details in error
        # body to avoid giving attackers a roadmap. Findings are available via /source/report.
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Deployment blocked — audit score below threshold",
                "score": result.score,
                "risk_label": result.risk_label,
                "critical": result.critical_count,
                "high": result.high_count,
            },
        )
    return report.to_dict()


@router.post("/source/report")
async def audit_source_report(req: AuditSourceRequest):
    """Audit source and return full findings + formatted report paths."""
    result = _engine.audit(req.source, req.contract_name)
    report = AuditReport.from_result(result, chain=req.chain, network=req.network)
    paths = _reporter.generate(report, formats=req.formats)
    return {
        "report": report.to_dict(),
        "output_paths": {k: str(v) for k, v in paths.items()},
    }


@router.get("/score/{score}")
async def explain_score(score: int):
    """Explain what a given audit score means."""
    if not 0 <= score <= 100:
        raise HTTPException(status_code=400, detail="Score must be 0-100")
    label = AuditEngine._score_label(score)
    blocked = score < AuditEngine.BLOCK_THRESHOLD
    return {
        "score": score,
        "risk_label": label,
        "blocks_deploy": blocked,
        "explanation": _score_explanation(score),
    }


def _score_explanation(score: int) -> str:
    if score >= 90:
        return "No significant security issues detected. Contract appears safe for deployment."
    elif score >= 75:
        return "Minor issues detected. Review low/medium findings before deployment."
    elif score >= 60:
        return "Medium-risk issues present. Address all high findings before deployment."
    elif score >= 40:
        return "High-risk vulnerabilities detected. Deployment strongly discouraged."
    else:
        return "Critical vulnerabilities present. Deployment BLOCKED. Fix all critical/high findings."
