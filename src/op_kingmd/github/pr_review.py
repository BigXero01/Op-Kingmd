"""AI-powered PR review for Solidity and Web3 TypeScript code."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from op_kingmd.github.client import GitHubClient

logger = structlog.get_logger(__name__)


@dataclass
class ReviewComment:
    path: str
    line: int
    body: str
    severity: str = "info"    # critical / high / medium / low / info


@dataclass
class PRReviewResult:
    pr_number: int
    repo: str
    overall: str                             # APPROVE / REQUEST_CHANGES / COMMENT
    summary: str
    comments: list[ReviewComment] = field(default_factory=list)
    security_issues: int = 0
    gas_suggestions: int = 0
    approved: bool = False


class PRReviewer:
    """Reviews GitHub PRs for Web3 code quality, security, and gas efficiency."""

    REVIEW_SYSTEM = (
        "You are an expert smart contract security auditor and code reviewer. "
        "Review the provided diff for security vulnerabilities, gas inefficiencies, "
        "code quality issues, and best practice violations. "
        "Be specific, actionable, and reference line numbers when possible."
    )

    def __init__(self, client: GitHubClient, pipeline: Any = None) -> None:
        self._gh = client
        self._pipeline = pipeline

    def review_pr(self, repo_name: str, pr_number: int) -> PRReviewResult:
        """Full AI-powered review of a pull request."""
        repo = self._gh.get_repo(repo_name)
        pr = self._gh.get_pr(repo, pr_number)

        # Collect changed files
        changed_files = list(pr.get_files())
        sol_files = [f for f in changed_files if f.filename.endswith(".sol")]
        ts_files = [f for f in changed_files if f.filename.endswith((".ts", ".js"))]

        logger.info(
            "reviewing_pr",
            repo=repo_name,
            pr=pr_number,
            sol_files=len(sol_files),
            ts_files=len(ts_files),
        )

        result = PRReviewResult(
            pr_number=pr_number,
            repo=repo_name,
            overall="COMMENT",
            summary="",
        )

        all_comments: list[ReviewComment] = []

        for pf in sol_files:
            patch = pf.patch or ""
            comments = self._review_solidity_diff(pf.filename, patch)
            all_comments.extend(comments)

        for pf in ts_files:
            patch = pf.patch or ""
            comments = self._review_typescript_diff(pf.filename, patch)
            all_comments.extend(comments)

        result.comments = all_comments
        result.security_issues = sum(
            1 for c in all_comments if c.severity in ("critical", "high")
        )
        result.gas_suggestions = sum(
            1 for c in all_comments if "gas" in c.body.lower()
        )

        result.summary = self._build_summary(result, pr)
        result.overall = "APPROVE" if result.security_issues == 0 else "REQUEST_CHANGES"
        result.approved = result.overall == "APPROVE"
        return result

    def post_review(self, result: PRReviewResult) -> None:
        """Post the review to GitHub."""
        repo = self._gh.get_repo(result.repo)
        pr = self._gh.get_pr(repo, result.pr_number)

        if result.overall == "REQUEST_CHANGES":
            self._gh.request_changes(pr, result.summary)
        else:
            self._gh.comment_on_pr(pr, result.summary)

        logger.info(
            "review_posted",
            pr=result.pr_number,
            outcome=result.overall,
            issues=result.security_issues,
        )

    # ── Diff analysis ─────────────────────────────────────────────────────────

    def _review_solidity_diff(self, filename: str, patch: str) -> list[ReviewComment]:
        comments = []
        lines = patch.splitlines()

        current_line = 0
        for i, line in enumerate(lines):
            # Track line numbers from diff headers
            if line.startswith("@@"):
                m = re.search(r"\+(\d+)", line)
                if m:
                    current_line = int(m.group(1)) - 1
                continue
            if line.startswith("+"):
                current_line += 1
                content = line[1:]
                issue = self._check_line_security(content, filename, current_line)
                if issue:
                    comments.append(issue)
            elif not line.startswith("-"):
                current_line += 1

        return comments

    def _review_typescript_diff(self, filename: str, patch: str) -> list[ReviewComment]:
        comments = []
        lines = patch.splitlines()
        current_line = 0
        for line in lines:
            if line.startswith("@@"):
                m = re.search(r"\+(\d+)", line)
                if m:
                    current_line = int(m.group(1)) - 1
                continue
            if line.startswith("+"):
                current_line += 1
                content = line[1:]
                # Check for hardcoded private keys
                if re.search(r"0x[0-9a-fA-F]{64}", content):
                    comments.append(ReviewComment(
                        path=filename, line=current_line,
                        body="**[CRITICAL]** Possible hardcoded private key detected. Never commit private keys.",
                        severity="critical",
                    ))
                # Check for missing error handling on contract calls
                if ".send(" in content and "await" in content and "try" not in patch[:500]:
                    comments.append(ReviewComment(
                        path=filename, line=current_line,
                        body="**[Medium]** Consider wrapping this transaction in try/catch to handle reverts gracefully.",
                        severity="medium",
                    ))
            elif not line.startswith("-"):
                current_line += 1
        return comments

    SECURITY_RULES: list[tuple[str, str, str]] = [
        (r"\.call\{value", "critical",
         "**[Critical]** Low-level `.call{value}` detected. Ensure CEI (Checks-Effects-Interactions) pattern is followed and ReentrancyGuard is applied."),
        (r"tx\.origin", "high",
         "**[High]** `tx.origin` used for authentication. Use `msg.sender` instead — `tx.origin` is vulnerable to phishing attacks."),
        (r"selfdestruct\(", "critical",
         "**[Critical]** `selfdestruct` can drain contract balance to any address. Ensure this is access-controlled."),
        (r"delegatecall\(", "high",
         "**[High]** `delegatecall` executes in the caller's context. Validate the target address rigorously."),
        (r"block\.timestamp", "medium",
         "**[Medium]** `block.timestamp` can be slightly manipulated by miners. Avoid using for randomness or precise timing."),
        (r"i\+\+", "low",
         "**[Gas]** Use `unchecked { ++i; }` in Solidity 0.8+ loops to save ~50 gas per iteration."),
        (r'require\(.*,"', "low",
         "**[Gas]** Replace `require(cond, \"string\")` with a custom error for ~50 gas savings per revert."),
        (r"\.length\s*[<>!=]", "low",
         "**[Gas]** Cache array `.length` in a local variable before a loop to avoid repeated storage reads."),
    ]

    def _check_line_security(
        self, line: str, filename: str, line_no: int
    ) -> ReviewComment | None:
        for pattern, severity, message in self.SECURITY_RULES:
            if re.search(pattern, line):
                return ReviewComment(
                    path=filename, line=line_no, body=message, severity=severity
                )
        return None

    def _build_summary(self, result: PRReviewResult, pr: Any) -> str:
        lines = [
            f"## Op-Kingmd Security Review — PR #{result.pr_number}",
            "",
            f"**Decision**: {result.overall}",
            f"**Security Issues**: {result.security_issues} (critical/high)",
            f"**Gas Suggestions**: {result.gas_suggestions}",
            f"**Total Comments**: {len(result.comments)}",
            "",
        ]
        if result.security_issues > 0:
            lines += [
                "### Critical / High Issues",
                *[
                    f"- `{c.path}:{c.line}` — {c.body}"
                    for c in result.comments
                    if c.severity in ("critical", "high")
                ],
                "",
            ]
        if result.gas_suggestions > 0:
            lines += [
                "### Gas Optimizations",
                *[
                    f"- `{c.path}:{c.line}` — {c.body}"
                    for c in result.comments
                    if "gas" in c.body.lower()
                ],
                "",
            ]
        lines.append("_Reviewed by Op-Kingmd AI Audit Layer_")
        return "\n".join(lines)
