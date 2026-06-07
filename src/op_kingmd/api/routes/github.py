"""GitHub integration API endpoints."""

from __future__ import annotations

import os
import re

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from op_kingmd.github.client import GitHubClient

router = APIRouter()

_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$")
_QUERY_MAX = 500


def _get_client() -> GitHubClient:
    token = os.getenv("GITHUB_TOKEN", "")
    # Basic sanity check — GitHub tokens are at minimum 20 chars
    if len(token) < 20:
        raise HTTPException(status_code=503, detail="GITHUB_TOKEN not configured")
    return GitHubClient(token)


class RepoAnalyzeRequest(BaseModel):
    repo: str = Field(..., description="owner/repo format")

    @field_validator("repo")
    @classmethod
    def validate_repo(cls, v: str) -> str:
        if not _REPO_RE.match(v):
            raise ValueError("repo must be in owner/name format (alphanumeric, hyphens, dots)")
        return v


class IssueRequest(BaseModel):
    repo: str = Field(..., max_length=200)
    title: str = Field(..., min_length=1, max_length=256)
    body: str = Field(..., min_length=1, max_length=65_536)
    labels: list[str] = Field(default=[], max_length=10)

    @field_validator("repo")
    @classmethod
    def validate_repo(cls, v: str) -> str:
        if not _REPO_RE.match(v):
            raise ValueError("repo must be in owner/name format")
        return v

    @field_validator("labels", mode="before")
    @classmethod
    def validate_labels(cls, v: list) -> list:
        return [str(lbl)[:50] for lbl in v[:10]]


class PRReviewRequest(BaseModel):
    repo: str = Field(..., max_length=200)
    pr_number: int = Field(..., gt=0, lt=1_000_000)
    post_review: bool = False

    @field_validator("repo")
    @classmethod
    def validate_repo(cls, v: str) -> str:
        if not _REPO_RE.match(v):
            raise ValueError("repo must be in owner/name format")
        return v


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=_QUERY_MAX)
    language: str = Field("Solidity", max_length=50, pattern=r"^[A-Za-z0-9 #+_-]+$")
    org: str | None = Field(None, max_length=100, pattern=r"^[A-Za-z0-9_-]*$")
    # Cap at 50 — GitHub API has rate limits
    max_results: int = Field(20, ge=1, le=50)


@router.post("/analyze")
async def analyze_repo(req: RepoAnalyzeRequest, request: Request):
    from op_kingmd.github.analyzer import RepoAnalyzer
    with _get_client() as client:
        analyzer = RepoAnalyzer(client)
        profile = analyzer.analyze(req.repo)
    return {
        "full_name": profile.full_name,
        "framework": profile.framework,
        "chain": profile.chain,
        "solidity_version": profile.solidity_version,
        "total_contracts": profile.total_contracts,
        "has_tests": profile.has_tests,
        "has_ci": profile.has_ci,
        "has_audits": profile.has_audits,
        "complexity_score": profile.complexity_score,
        "oz_version": profile.oz_version,
        "sol_files": profile.sol_files[:20],
    }


@router.post("/issues/create")
async def create_issue(req: IssueRequest):
    with _get_client() as client:
        repo = client.get_repo(req.repo)
        issue = client.create_issue(repo, req.title, req.body, req.labels or None)
    return {"number": issue.number, "url": issue.html_url, "title": issue.title}


@router.post("/pr/review")
async def review_pr(req: PRReviewRequest, request: Request):
    from op_kingmd.github.pr_review import PRReviewer
    pipeline = getattr(request.app.state, "pipeline", None)
    with _get_client() as client:
        reviewer = PRReviewer(client, pipeline)
        result = reviewer.review_pr(req.repo, req.pr_number)
        if req.post_review:
            reviewer.post_review(result)
    return {
        "pr_number": result.pr_number,
        "overall": result.overall,
        "security_issues": result.security_issues,
        "gas_suggestions": result.gas_suggestions,
        "summary": result.summary,
        "comments": [
            {"path": c.path, "line": c.line, "body": c.body, "severity": c.severity}
            for c in result.comments
        ],
    }


@router.post("/search")
async def search_code(req: SearchRequest):
    with _get_client() as client:
        results = client.search_code(req.query, req.language, req.org, req.max_results)
    return {
        "count": len(results),
        "results": [
            {"path": r.path, "repo": r.repo, "url": r.url, "score": r.score}
            for r in results
        ],
    }
