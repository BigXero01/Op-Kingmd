"""Audit-Before-Deploy API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from op_kingmd.audit.analyzer import AuditEngine
from op_kingmd.audit.report import AuditReport, ReportGenerator

router = APIRouter()

_engine = AuditEngine()
_reporter = ReportGenerator()


class AuditSourceRequest(BaseModel):
    source: str = Field(..., min_length=20, description="Solidity source code string")
    contract_name: str | None = None
    chain: str = "ethereum"
    network: str = "mainnet"
    formats: list[str] = ["json", "markdown"]


class AuditPathRequest(BaseModel):
    path: str = Field(..., description="Absolute or relative path to .sol file or directory")
    chain: str = "ethereum"
    network: str = "mainnet"


@router.post("/source")
async def audit_source(req: AuditSourceRequest):
    """Audit a Solidity source string. Blocks response if deploy should be blocked."""
    result = _engine.audit(req.source, req.contract_name)
    report = AuditReport.from_result(result, chain=req.chain, network=req.network)
    data = report.to_dict()

    if result.blocks_deploy:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Deployment blocked by audit engine",
                "score": result.score,
                "risk_label": result.risk_label,
                "findings": data["findings"],
            },
        )
    return data


@router.post("/source/report")
async def audit_source_report(req: AuditSourceRequest):
    """Audit source and return formatted reports (JSON + Markdown by default)."""
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
    from op_kingmd.audit.analyzer import AuditEngine
    label = AuditEngine._score_label(score)
    blocked = score < AuditEngine.BLOCK_THRESHOLD
    return {
        "score": score,
        "risk_label": label,
        "blocks_deploy": blocked,
        "explanation": _score_explanation(score, label),
    }


def _score_explanation(score: int, label: str) -> str:
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
