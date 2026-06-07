"""Repository analysis — structure, dependency detection, tech stack identification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from op_kingmd.github.client import GitHubClient

logger = structlog.get_logger(__name__)


@dataclass
class RepoProfile:
    full_name: str
    description: str | None
    language: str | None
    stars: int
    forks: int

    # Web3 stack detection
    framework: str | None = None        # foundry / hardhat / truffle / anchor
    chain: str | None = None            # ethereum / solana / polygon / etc.
    solidity_version: str | None = None
    has_tests: bool = False
    has_ci: bool = False
    has_audits: bool = False

    # File inventory
    sol_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    script_files: list[str] = field(default_factory=list)

    # Dependencies
    oz_version: str | None = None
    dependencies: list[str] = field(default_factory=list)

    # Complexity
    total_contracts: int = 0
    total_loc: int = 0
    complexity_score: int = 0          # 1-10

    # Issues generated
    issues_opened: list[int] = field(default_factory=list)


FRAMEWORK_SIGNALS = {
    "foundry": ["foundry.toml", "forge-std", "lib/forge-std"],
    "hardhat": ["hardhat.config.ts", "hardhat.config.js", ".hardhat"],
    "truffle": ["truffle-config.js", "migrations/"],
    "anchor": ["Anchor.toml", "programs/", "Cargo.toml"],
    "brownie": ["brownie-config.yaml"],
}

CHAIN_SIGNALS = {
    "solana": ["anchor", "program_id!", "Pubkey", "AccountInfo"],
    "polygon": ["polygon", "matic", "137"],
    "arbitrum": ["arbitrum", "42161"],
    "base": ["base-mainnet", "8453"],
    "optimism": ["optimism", "10"],
}


class RepoAnalyzer:
    """Analyzes a GitHub repository to build a comprehensive Web3 profile."""

    def __init__(self, client: GitHubClient) -> None:
        self._gh = client

    def analyze(self, repo_name: str) -> RepoProfile:
        """Full repository analysis — returns RepoProfile."""
        repo = self._gh.get_repo(repo_name)
        profile = RepoProfile(
            full_name=repo.full_name,
            description=repo.description,
            language=repo.language,
            stars=repo.stargazers_count,
            forks=repo.forks_count,
        )

        # Enumerate files at root level
        try:
            root_contents = repo.get_contents("")
            if not isinstance(root_contents, list):
                root_contents = [root_contents]
            root_names = {c.name for c in root_contents}
        except Exception:
            root_names = set()

        profile.framework = self._detect_framework(root_names, repo)
        profile.has_ci = any(n in root_names for n in (".github", ".circleci", ".gitlab-ci.yml"))
        profile.has_audits = "audits" in root_names or "audit" in root_names

        # Collect Solidity files
        profile.sol_files = self._gh.list_solidity_files(repo)
        profile.total_contracts = len(profile.sol_files)
        profile.test_files = [p for p in profile.sol_files if "test" in p.lower()]
        profile.script_files = [p for p in profile.sol_files if "script" in p.lower() or "deploy" in p.lower()]
        profile.has_tests = len(profile.test_files) > 0

        # Sample source for pragma / OZ detection
        if profile.sol_files:
            sample_path = next((p for p in profile.sol_files if "test" not in p.lower()), profile.sol_files[0])
            fc = self._gh.get_file(repo, sample_path)
            if fc:
                profile.solidity_version = self._extract_pragma(fc.content)
                profile.oz_version = self._detect_oz_version(fc.content)
                profile.total_loc += fc.content.count("\n")
                profile.chain = self._detect_chain(fc.content)

        profile.complexity_score = self._compute_complexity(profile)

        logger.info(
            "repo_analyzed",
            repo=repo_name,
            contracts=profile.total_contracts,
            framework=profile.framework,
            complexity=profile.complexity_score,
        )
        return profile

    def generate_issues(
        self,
        profile: RepoProfile,
        pipeline: Any = None,   # Web3Pipeline
    ) -> list[dict]:
        """Generate actionable GitHub issues from repo analysis."""
        issues = []

        if not profile.has_tests:
            issues.append({
                "title": "Add comprehensive test suite",
                "body": (
                    f"The repository has {profile.total_contracts} Solidity contracts but no detected test files. "
                    "Add Foundry or Hardhat tests covering happy path, access control, edge cases, and fuzz scenarios.\n\n"
                    "**Suggested next steps:**\n"
                    "- Run `forge init --no-commit` if using Foundry\n"
                    "- Generate test stubs with `op-kingmd generate tests --source ./src/`"
                ),
                "labels": ["testing", "quality"],
            })

        if not profile.has_ci:
            issues.append({
                "title": "Add CI/CD pipeline for automated testing",
                "body": (
                    "No CI configuration detected. Add a GitHub Actions workflow to run tests on every PR.\n\n"
                    "```yaml\n# .github/workflows/ci.yml\non: [push, pull_request]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n"
                    "      - uses: actions/checkout@v4\n        with: { submodules: recursive }\n"
                    "      - uses: foundry-rs/foundry-toolchain@v1\n      - run: forge test -vv\n```"
                ),
                "labels": ["ci", "devops"],
            })

        if not profile.has_audits:
            issues.append({
                "title": "Commission or run security audit",
                "body": (
                    f"No audit reports found in repository. Before deploying {profile.total_contracts} contracts to mainnet:\n\n"
                    "1. Run `op-kingmd audit ./src/` for automated vulnerability scanning\n"
                    "2. Consider a professional audit from Trail of Bits, OpenZeppelin, or Code4rena\n"
                    "3. Add audit reports to `audits/` directory"
                ),
                "labels": ["security", "audit"],
            })

        if profile.solidity_version and any(
            v in (profile.solidity_version or "") for v in ["0.5", "0.6", "0.7"]
        ):
            issues.append({
                "title": f"Upgrade Solidity from {profile.solidity_version} to 0.8.x",
                "body": (
                    f"Using Solidity {profile.solidity_version} which lacks built-in overflow protection. "
                    "Upgrading to 0.8.24+ provides:\n"
                    "- Built-in arithmetic overflow/underflow checks (no SafeMath needed)\n"
                    "- Custom errors (gas savings)\n"
                    "- Immutable variables\n"
                    "- Better error messages"
                ),
                "labels": ["upgrade", "security"],
            })

        return issues

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_framework(self, root_names: set[str], repo: Any) -> str | None:
        for framework, signals in FRAMEWORK_SIGNALS.items():
            if any(s in root_names for s in signals):
                return framework
        return None

    @staticmethod
    def _extract_pragma(source: str) -> str | None:
        import re
        m = re.search(r"pragma solidity\s+([^;]+);", source)
        return m.group(1).strip() if m else None

    @staticmethod
    def _detect_oz_version(source: str) -> str | None:
        import re
        m = re.search(r"@openzeppelin/contracts@?([0-9.]+)", source)
        return m.group(1) if m else ("detected" if "@openzeppelin" in source else None)

    @staticmethod
    def _detect_chain(source: str) -> str | None:
        source_lower = source.lower()
        for chain, signals in CHAIN_SIGNALS.items():
            if any(s.lower() in source_lower for s in signals):
                return chain
        return "ethereum"

    @staticmethod
    def _compute_complexity(profile: RepoProfile) -> int:
        score = 0
        score += min(profile.total_contracts * 2, 4)
        score += min(profile.total_loc // 500, 3)
        if not profile.has_tests:
            score += 2
        if not profile.has_audits:
            score += 1
        return min(score, 10)
