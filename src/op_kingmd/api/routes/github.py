"""GitHub integration API endpoints."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from op_kingmd.github.client import GitHubClient

router = APIRouter()


def _get_client() -> GitHubClient:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="GITHUB_TOKEN not configured")
    return GitHubClient(token)


class RepoAnalyzeRequest(BaseModel):
    repo: str = Field(..., description="owner/repo format")


class IssueRequest(BaseModel):
    repo: str
    title: str
    body: str
    labels: list[str] = []


class PRReviewRequest(BaseModel):
    repo: str
    pr_number: int
    post_review: bool = False


class SearchRequest(BaseModel):
    query: str
    language: str = "Solidity"
    org: str | None = None
    max_results: int = 20


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
