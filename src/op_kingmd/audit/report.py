"""Audit report generation — JSON, Markdown, and HTML output formats."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path
from typing import Any

import structlog

from op_kingmd.audit.analyzer import AuditResult, Finding, GasOptimization, Severity

logger = structlog.get_logger(__name__)


@dataclass
class AuditReport:
    """Serializable report wrapping an AuditResult."""

    result: AuditResult
    timestamp: str
    model_version: str
    chain: str
    network: str

    @classmethod
    def from_result(
        cls,
        result: AuditResult,
        chain: str = "ethereum",
        network: str = "mainnet",
    ) -> "AuditReport":
        return cls(
            result=result,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            model_version=os.getenv("OPKMD_MODEL_ID", "Qwen/Qwen2.5-Coder-7B-Instruct"),
            chain=chain,
            network=network,
        )

    def to_dict(self) -> dict:
        r = self.result
        return {
            "meta": {
                "timestamp": self.timestamp,
                "model_version": self.model_version,
                "chain": self.chain,
                "network": self.network,
                "tool": "op-kingmd",
            },
            "contract": {
                "name": r.contract_name,
                "path": r.contract_path,
            },
            "score": r.score,
            "risk_label": r.risk_label,
            "blocks_deploy": r.blocks_deploy,
            "summary": {
                "critical": r.critical_count,
                "high": r.high_count,
                "medium": r.medium_count,
                "low": r.low_count,
                "gas_optimizations": len(r.gas_optimizations),
                "dependency_issues": len(r.dependency_issues),
            },
            "findings": [
                {
                    "name": f.name,
                    "severity": f.severity.value,
                    "description": f.description,
                    "location": f.location,
                    "line": f.line,
                    "recommendation": f.recommendation,
                    "detector": f.detector,
                    "snippet": f.snippet,
                }
                for f in r.findings
            ],
            "gas_optimizations": [
                {
                    "rule": g.rule,
                    "description": g.description,
                    "savings_estimate": g.savings_estimate,
                    "location": g.location,
                    "line": g.line,
                }
                for g in r.gas_optimizations
            ],
            "dependency_issues": [
                {
                    "package": d.package,
                    "current_version": d.current_version,
                    "min_safe_version": d.min_safe_version,
                    "severity": d.severity.value,
                }
                for d in r.dependency_issues
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class ReportGenerator:
    """Generates audit reports in multiple formats."""

    SEVERITY_COLORS = {
        "critical": "#dc2626",
        "high": "#ea580c",
        "medium": "#d97706",
        "low": "#65a30d",
        "informational": "#6b7280",
    }

    SEVERITY_EMOJI = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢",
        "informational": "⚪",
    }

    def __init__(self, output_dir: str | None = None) -> None:
        self.output_dir = Path(output_dir or os.getenv("OPKMD_REPORT_DIR", "./audit_reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        report: AuditReport,
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        formats = formats or ["json", "markdown", "html"]
        outputs = {}
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{report.result.contract_name}_{ts}"

        if "json" in formats:
            path = self.output_dir / f"{base_name}.json"
            path.write_text(report.to_json())
            outputs["json"] = path

        if "markdown" in formats:
            path = self.output_dir / f"{base_name}.md"
            path.write_text(self.to_markdown(report))
            outputs["markdown"] = path

        if "html" in formats:
            path = self.output_dir / f"{base_name}.html"
            path.write_text(self.to_html(report))
            outputs["html"] = path

        logger.info("reports_generated", formats=list(outputs.keys()), dir=str(self.output_dir))
        return outputs

    # ── Markdown ──────────────────────────────────────────────────────────────

    def to_markdown(self, report: AuditReport) -> str:
        r = report.result
        data = report.to_dict()
        lines = [
            f"# Security Audit Report — {r.contract_name}",
            "",
            f"> Generated by **Op-Kingmd** · {report.timestamp}",
            "",
            "## Executive Summary",
            "",
            f"| | |",
            f"|---|---|",
            f"| **Contract** | `{r.contract_name}` |",
            f"| **Risk Score** | **{r.score}/100** |",
            f"| **Risk Level** | {r.risk_label} |",
            f"| **Deploy Blocked** | {'YES ⛔' if r.blocks_deploy else 'NO ✅'} |",
            f"| **Chain** | {report.chain} / {report.network} |",
            "",
            "## Finding Summary",
            "",
            "| Severity | Count |",
            "|----------|-------|",
            f"| 🔴 Critical | {r.critical_count} |",
            f"| 🟠 High | {r.high_count} |",
            f"| 🟡 Medium | {r.medium_count} |",
            f"| 🟢 Low | {r.low_count} |",
            f"| ⛽ Gas Optimizations | {len(r.gas_optimizations)} |",
            "",
        ]

        if r.findings:
            lines.append("## Detailed Findings")
            lines.append("")
            for i, f in enumerate(r.findings, 1):
                emoji = self.SEVERITY_EMOJI.get(f.severity.value, "⚪")
                lines += [
                    f"### {i}. {emoji} [{f.severity.value.upper()}] {f.name}",
                    "",
                    f"**Location**: `{f.location}:{f.line or '?'}`",
                    "",
                    f"**Description**: {f.description}",
                    "",
                ]
                if f.snippet:
                    lines += [f"```solidity", f.snippet, "```", ""]
                if f.recommendation:
                    lines += [f"**Recommendation**: {f.recommendation}", ""]

        if r.gas_optimizations:
            lines += [
                "## Gas Optimizations",
                "",
                "| Rule | Description | Savings |",
                "|------|-------------|---------|",
            ]
            for g in r.gas_optimizations:
                lines.append(f"| `{g.rule}` | {g.description} | {g.savings_estimate or '—'} |")
            lines.append("")

        if r.dependency_issues:
            lines += ["## Dependency Issues", ""]
            for d in r.dependency_issues:
                lines.append(
                    f"- **[{d.severity.value.upper()}]** `{d.package}`: "
                    f"version `{d.current_version}` < min safe `{d.min_safe_version}`"
                )
            lines.append("")

        lines += [
            "---",
            f"*Audit by Op-Kingmd · Model: {report.model_version}*",
        ]
        return "\n".join(lines)

    # ── HTML ──────────────────────────────────────────────────────────────────

    def to_html(self, report: AuditReport) -> str:
        r = report.result
        score_color = (
            "#16a34a" if r.score >= 75 else
            "#d97706" if r.score >= 60 else
            "#dc2626"
        )

        # All dynamic values are HTML-escaped before insertion to prevent XSS
        # if contract source or finding descriptions contain HTML-special characters.
        findings_html = ""
        for f in r.findings:
            color = self.SEVERITY_COLORS.get(f.severity.value, "#6b7280")
            emoji = self.SEVERITY_EMOJI.get(f.severity.value, "⚪")
            snippet_block = (
                f"<pre><code>{html_escape(f.snippet)}</code></pre>" if f.snippet else ""
            )
            findings_html += f"""
            <div class="finding" style="border-left: 4px solid {color}; padding: 12px; margin: 8px 0; background: #f9f9f9;">
              <h4>{emoji} [{html_escape(f.severity.value.upper())}] {html_escape(f.name)}</h4>
              <p><strong>Location:</strong> <code>{html_escape(str(f.location))}:{html_escape(str(f.line or '?'))}</code></p>
              <p>{html_escape(f.description)}</p>
              {snippet_block}
              <p><em><strong>Recommendation:</strong> {html_escape(f.recommendation or 'N/A')}</em></p>
            </div>"""

        gas_rows = "".join(
            f"<tr><td><code>{html_escape(g.rule)}</code></td>"
            f"<td>{html_escape(g.description)}</td>"
            f"<td>{html_escape(g.savings_estimate or '—')}</td></tr>"
            for g in r.gas_optimizations
        )

        deploy_badge_class = "blocked" if r.blocks_deploy else "safe"
        deploy_text = "BLOCKED ⛔" if r.blocks_deploy else "ALLOWED ✅"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline';">
  <title>Audit Report — {html_escape(r.contract_name)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }}
    .score-badge {{ display: inline-block; font-size: 3rem; font-weight: 900; color: {score_color}; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }}
    .blocked {{ background: #fef2f2; color: #dc2626; border: 1px solid #dc2626; }}
    .safe {{ background: #f0fdf4; color: #16a34a; border: 1px solid #16a34a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }}
    pre {{ background: #1e293b; color: #e2e8f0; padding: 12px; border-radius: 6px; overflow-x: auto; }}
    h1, h2, h3 {{ border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }}
  </style>
</head>
<body>
  <h1>Security Audit Report</h1>
  <p>Contract: <strong>{html_escape(r.contract_name)}</strong> &nbsp;·&nbsp; Generated: {html_escape(report.timestamp)}</p>

  <h2>Executive Summary</h2>
  <div class="score-badge">{r.score}/100</div>
  <span class="badge {deploy_badge_class}">{html_escape(r.risk_label)} — Deploy {deploy_text}</span>

  <table style="margin-top: 16px;">
    <tr><th>Severity</th><th>Count</th></tr>
    <tr><td>🔴 Critical</td><td>{r.critical_count}</td></tr>
    <tr><td>🟠 High</td><td>{r.high_count}</td></tr>
    <tr><td>🟡 Medium</td><td>{r.medium_count}</td></tr>
    <tr><td>🟢 Low</td><td>{r.low_count}</td></tr>
    <tr><td>⛽ Gas Optimizations</td><td>{len(r.gas_optimizations)}</td></tr>
  </table>

  <h2>Findings</h2>
  {findings_html if findings_html else '<p>No issues detected.</p>'}

  <h2>Gas Optimizations</h2>
  {'<table><tr><th>Rule</th><th>Description</th><th>Est. Savings</th></tr>' + gas_rows + '</table>' if gas_rows else '<p>No optimizations detected.</p>'}

  <hr>
  <small>Audit by <strong>Op-Kingmd</strong> · Chain: {html_escape(report.chain)}/{html_escape(report.network)}</small>
</body>
</html>"""
