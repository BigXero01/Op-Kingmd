"""Audit-Before-Deploy Layer.

Mandatory pipeline that runs before any contract deployment:
  1. Static vulnerability detection (pattern matching + AST analysis)
  2. Slither static analyzer integration
  3. Gas optimization review
  4. Dependency vulnerability scanning
  5. Risk scoring (0-100, blocks deploy if score < threshold)

Usage:
    engine = AuditEngine()
    report = engine.audit("./src/Token.sol")
    if report.blocks_deploy:
        raise SystemExit("Deployment blocked by audit")
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "audit_config.yaml"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "informational"


@dataclass
class Finding:
    name: str
    severity: Severity
    description: str
    location: str | None = None
    line: int | None = None
    recommendation: str | None = None
    detector: str = "static"
    snippet: str | None = None


@dataclass
class GasOptimization:
    rule: str
    description: str
    savings_estimate: str | None = None
    location: str | None = None
    line: int | None = None


@dataclass
class DependencyIssue:
    package: str
    current_version: str | None
    min_safe_version: str
    severity: Severity


@dataclass
class AuditResult:
    contract_name: str
    contract_path: str
    findings: list[Finding] = field(default_factory=list)
    gas_optimizations: list[GasOptimization] = field(default_factory=list)
    dependency_issues: list[DependencyIssue] = field(default_factory=list)
    slither_output: dict | None = None
    score: int = 100
    risk_label: str = "SAFE"
    blocks_deploy: bool = False
    metadata: dict = field(default_factory=dict)

    # Severity counts
    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.MEDIUM)

    @property
    def low_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.LOW)


class AuditEngine:
    """Orchestrates the full Audit-Before-Deploy pipeline."""

    SEVERITY_WEIGHTS = {
        Severity.CRITICAL: 40,
        Severity.HIGH: 20,
        Severity.MEDIUM: 10,
        Severity.LOW: 3,
        Severity.INFO: 1,
    }

    BLOCK_THRESHOLD = int(os.getenv("OPKMD_AUDIT_BLOCK_THRESHOLD", "60"))
    WARN_THRESHOLD = 75

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._config = self._load_config(config_path or _CONFIG_PATH)

    def audit(self, source: str, contract_name: str | None = None) -> AuditResult:
        """Run full audit pipeline on Solidity source string or file path."""
        path = None
        if Path(source).exists():
            path = str(source)
            source_code = Path(source).read_text()
        else:
            source_code = source
            path = "<inline>"

        name = contract_name or self._extract_contract_name(source_code) or "Contract"
        result = AuditResult(contract_name=name, contract_path=path or "<inline>")

        logger.info("audit_start", contract=name, path=path)

        # Stage 1 — Static pattern analysis
        findings = self._static_analysis(source_code, path or "<inline>")
        result.findings.extend(findings)

        # Stage 2 — Slither integration
        if self._slither_enabled() and path and path != "<inline>":
            slither_findings = self._run_slither(path)
            result.findings.extend(slither_findings)
            result.slither_output = {"ran": True, "findings": len(slither_findings)}

        # Stage 3 — Gas optimization analysis
        result.gas_optimizations = self._gas_analysis(source_code, path or "<inline>")

        # Stage 4 — Dependency scanning
        result.dependency_issues = self._dependency_scan(path or ".")

        # Stage 5 — Risk scoring
        result.score = self._compute_score(result)
        result.risk_label = self._score_label(result.score)
        result.blocks_deploy = result.score < self.BLOCK_THRESHOLD

        logger.info(
            "audit_complete",
            contract=name,
            score=result.score,
            risk=result.risk_label,
            blocked=result.blocks_deploy,
            critical=result.critical_count,
            high=result.high_count,
        )
        return result

    def audit_directory(self, directory: str) -> list[AuditResult]:
        """Audit all .sol files in a directory."""
        sol_files = list(Path(directory).rglob("*.sol"))
        results = []
        for f in sol_files:
            if "test" not in f.name.lower() and "mock" not in f.name.lower():
                results.append(self.audit(str(f)))
        return results

    # ── Stage 1: Static Analysis ──────────────────────────────────────────────

    VULNERABILITY_PATTERNS = [
        # Reentrancy
        (
            r"\.call\{value.*?\}.*?;(?!.*?\n.*?=.*?;)",
            Severity.CRITICAL,
            "Potential Reentrancy",
            "External call with value may be vulnerable to reentrancy. Apply ReentrancyGuard or follow CEI pattern.",
            "Apply `nonReentrant` modifier from OpenZeppelin or restructure to update state before making the call.",
        ),
        # tx.origin authentication
        (
            r"\btx\.origin\b",
            Severity.HIGH,
            "tx.origin Authentication",
            "tx.origin used for access control is vulnerable to phishing/forwarding attacks.",
            "Replace `tx.origin` with `msg.sender`.",
        ),
        # Selfdestruct
        (
            r"\bselfdestruct\s*\(",
            Severity.CRITICAL,
            "Selfdestruct Present",
            "selfdestruct can drain contract ETH balance and destroy bytecode.",
            "Remove selfdestruct or restrict with strict access control and timelock.",
        ),
        # Unchecked low-level call
        (
            r"(?<!\baddress\b)\.\bcall\s*\(",
            Severity.HIGH,
            "Unchecked Low-Level Call",
            "Return value of low-level call() not checked — silent failures may cause inconsistent state.",
            "Always check the bool return value: `(bool ok, ) = addr.call(...); require(ok);`",
        ),
        # Delegatecall to user-controlled address
        (
            r"\bdelegatecall\s*\(",
            Severity.HIGH,
            "Unsafe Delegatecall",
            "delegatecall executes in the calling contract's context — a malicious target can corrupt storage.",
            "Validate and whitelist target addresses. Consider using a proxy pattern with storage collision protection.",
        ),
        # Price oracle manipulation
        (
            r"getAmountsOut|getReserves\(\)|price0CumulativeLast",
            Severity.HIGH,
            "Oracle Manipulation Risk",
            "Spot price from an AMM can be manipulated via flash loan in a single transaction.",
            "Use a TWAP oracle or Chainlink price feed instead of spot AMM price.",
        ),
        # Unprotected initializer
        (
            r"function\s+(?:initialize|init)\s*\([^)]*\)\s*(?:public|external)",
            Severity.CRITICAL,
            "Unprotected Initializer",
            "Initializer function is callable multiple times without an `initializer` guard.",
            "Add OpenZeppelin `initializer` modifier: `function initialize() public initializer {}`",
        ),
        # Block timestamp dependence
        (
            r"\bblock\.timestamp\b",
            Severity.MEDIUM,
            "Block Timestamp Dependence",
            "block.timestamp can be slightly manipulated by block proposers (~15s tolerance).",
            "Avoid using block.timestamp for randomness. For time-locks, the minor variance is usually acceptable.",
        ),
        # Integer division precision loss
        (
            r"/\s*\d+(?!\s*\*)",
            Severity.LOW,
            "Precision Loss from Integer Division",
            "Integer division truncates — multiply before dividing to preserve precision.",
            "Use `(a * b) / c` ordering or a fixed-point library.",
        ),
        # Missing zero-address check
        (
            r"address\s+(?:public|private|internal)?\s*\w+\s*;",
            Severity.LOW,
            "Missing Zero-Address Validation",
            "State variable of type address may be set to address(0) without validation.",
            "Add `require(addr != address(0), \"ZeroAddress\")` in setter functions.",
        ),
    ]

    def _static_analysis(self, source: str, path: str) -> list[Finding]:
        findings = []
        lines = source.splitlines()

        for pattern, severity, name, desc, rec in self.VULNERABILITY_PATTERNS:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    snippet = line.strip()
                    findings.append(Finding(
                        name=name,
                        severity=severity,
                        description=desc,
                        location=path,
                        line=i,
                        recommendation=rec,
                        detector="static",
                        snippet=snippet[:120],
                    ))

        # Deduplicate same finding on consecutive lines
        seen = set()
        deduped = []
        for f in findings:
            key = (f.name, f.location)
            if key not in seen:
                seen.add(key)
                deduped.append(f)

        return deduped

    # ── Stage 2: Slither ──────────────────────────────────────────────────────

    def _slither_enabled(self) -> bool:
        return (
            os.getenv("OPKMD_SLITHER_ENABLED", "1") == "1"
            and self._is_available("slither")
        )

    def _run_slither(self, path: str) -> list[Finding]:
        findings = []
        try:
            result = subprocess.run(
                ["slither", path, "--json", "-"],
                capture_output=True,
                text=True,
                timeout=int(self._config.get("audit", {}).get("tools", {}).get("slither", {}).get("timeout", 120)),
            )
            if result.returncode not in (0, 255):  # 255 = findings present
                logger.warning("slither_nonzero_exit", code=result.returncode)

            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                return findings

            for det in data.get("results", {}).get("detectors", []):
                severity_map = {
                    "High": Severity.HIGH,
                    "Medium": Severity.MEDIUM,
                    "Low": Severity.LOW,
                    "Informational": Severity.INFO,
                    "Optimization": Severity.INFO,
                }
                sev = severity_map.get(det.get("impact", ""), Severity.INFO)
                elements = det.get("elements", [{}])
                loc = elements[0].get("source_mapping", {}).get("filename_short", path) if elements else path
                line = elements[0].get("source_mapping", {}).get("lines", [None])[0] if elements else None

                findings.append(Finding(
                    name=det.get("check", "slither"),
                    severity=sev,
                    description=det.get("description", "").strip(),
                    location=loc,
                    line=line,
                    recommendation=det.get("markdown", ""),
                    detector="slither",
                ))
        except subprocess.TimeoutExpired:
            logger.warning("slither_timeout", path=path)
        except FileNotFoundError:
            logger.debug("slither_not_found")
        except Exception as exc:
            logger.warning("slither_error", error=str(exc))
        return findings

    # ── Stage 3: Gas Analysis ─────────────────────────────────────────────────

    GAS_PATTERNS = [
        (r'require\(.*,\s*"', "use_custom_errors", "Replace `require` string with custom error", "~50 gas per revert"),
        (r"for\s*\(.*\.length", "cache_array_length", "Cache array length before loop", "~100 gas per iteration"),
        (r"\bi\+\+\b", "unchecked_increment", "Use `unchecked { ++i; }` in loop", "~50 gas per iteration"),
        (r"address\s+public\s+\w+\s*=", "immutable_constant", "Consider `immutable` for addresses set in constructor", "~200 gas per read"),
        (r"bool\s+public", "pack_booleans", "Pack boolean fields into a bitmap struct to save storage slots", "up to 15000 gas"),
        (r"storage\s+\w+\s*=\s*\w+\[", "cache_storage_ref", "Cache storage reference to avoid repeated SLOAD", "~200 gas per extra read"),
    ]

    def _gas_analysis(self, source: str, path: str) -> list[GasOptimization]:
        opts = []
        lines = source.splitlines()
        seen = set()
        for pattern, rule, desc, savings in self.GAS_PATTERNS:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line) and rule not in seen:
                    opts.append(GasOptimization(
                        rule=rule, description=desc,
                        savings_estimate=savings, location=path, line=i,
                    ))
                    seen.add(rule)
        return opts

    # ── Stage 4: Dependency Scanning ─────────────────────────────────────────

    def _dependency_scan(self, path: str) -> list[DependencyIssue]:
        issues = []
        # Check package.json if present
        pkg_json = Path(path).parent / "package.json"
        if not pkg_json.exists():
            pkg_json = Path(path) / "package.json"

        if pkg_json.exists():
            try:
                pkg = json.loads(pkg_json.read_text())
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                known_vulnerable = self._config.get("audit", {}).get("dependency_checks", {}).get("known_vulnerable_packages", [])
                for entry in known_vulnerable:
                    name = entry["name"]
                    min_safe = entry["min_safe_version"]
                    current = deps.get(name, "").lstrip("^~>=")
                    if current and current < min_safe:
                        issues.append(DependencyIssue(
                            package=name,
                            current_version=current,
                            min_safe_version=min_safe,
                            severity=Severity.HIGH,
                        ))
            except Exception as exc:
                logger.debug("dep_scan_failed", error=str(exc))
        return issues

    # ── Stage 5: Risk Scoring ─────────────────────────────────────────────────

    def _compute_score(self, result: AuditResult) -> int:
        deductions = 0
        for finding in result.findings:
            weight = self.SEVERITY_WEIGHTS.get(finding.severity, 0)
            deductions += weight
        for issue in result.dependency_issues:
            deductions += self.SEVERITY_WEIGHTS.get(issue.severity, 0)
        return max(0, 100 - deductions)

    @staticmethod
    def _score_label(score: int) -> str:
        if score >= 90:
            return "SAFE"
        elif score >= 75:
            return "LOW_RISK"
        elif score >= 60:
            return "MEDIUM_RISK"
        elif score >= 40:
            return "HIGH_RISK"
        else:
            return "CRITICAL_RISK"

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_contract_name(source: str) -> str | None:
        m = re.search(r"\bcontract\s+(\w+)", source)
        return m.group(1) if m else None

    @staticmethod
    def _is_available(cmd: str) -> bool:
        import shutil
        return shutil.which(cmd) is not None

    def _load_config(self, path: Path) -> dict:
        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except Exception:
            return {}
