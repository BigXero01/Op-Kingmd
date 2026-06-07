"""Tests for GitHub integration (mocked — no real API calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from op_kingmd.github.pr_review import PRReviewer, ReviewComment


class TestPRReviewer:
    def _make_reviewer(self):
        client = MagicMock()
        return PRReviewer(client)

    def test_check_line_reentrancy(self):
        reviewer = self._make_reviewer()
        comment = reviewer._check_line_security(
            '(bool ok,) = addr.call{value: amount}("");', "Vault.sol", 42
        )
        assert comment is not None
        assert comment.severity in ("critical", "high")

    def test_check_line_tx_origin(self):
        reviewer = self._make_reviewer()
        comment = reviewer._check_line_security(
            "require(tx.origin == owner);", "Token.sol", 10
        )
        assert comment is not None
        assert "tx.origin" in comment.body.lower() or "tx.Origin" in comment.body

    def test_check_line_selfdestruct(self):
        reviewer = self._make_reviewer()
        comment = reviewer._check_line_security(
            "selfdestruct(payable(owner));", "Kill.sol", 5
        )
        assert comment is not None
        assert comment.severity == "critical"

    def test_clean_line_returns_none(self):
        reviewer = self._make_reviewer()
        comment = reviewer._check_line_security(
            "uint256 balance = balances[msg.sender];", "Safe.sol", 1
        )
        assert comment is None

    def test_review_comment_dataclass(self):
        c = ReviewComment(path="Token.sol", line=42, body="Test issue", severity="high")
        assert c.path == "Token.sol"
        assert c.line == 42
        assert c.severity == "high"

    def test_build_summary_no_issues(self):
        reviewer = self._make_reviewer()
        from op_kingmd.github.pr_review import PRReviewResult
        result = PRReviewResult(
            pr_number=1, repo="org/repo", overall="APPROVE",
            summary="", security_issues=0, gas_suggestions=0
        )
        pr_mock = MagicMock()
        pr_mock.number = 1
        summary = reviewer._build_summary(result, pr_mock)
        assert "APPROVE" in summary

    def test_build_summary_with_issues(self):
        reviewer = self._make_reviewer()
        from op_kingmd.github.pr_review import PRReviewResult
        result = PRReviewResult(
            pr_number=2, repo="org/repo", overall="REQUEST_CHANGES",
            summary="", security_issues=2, gas_suggestions=1,
            comments=[
                ReviewComment("Token.sol", 5, "Critical issue", "critical"),
                ReviewComment("Token.sol", 10, "High issue", "high"),
            ],
        )
        pr_mock = MagicMock()
        pr_mock.number = 2
        summary = reviewer._build_summary(result, pr_mock)
        assert "REQUEST_CHANGES" in summary
        assert "2" in summary


class TestRepoAnalyzer:
    def test_detect_framework_foundry(self):
        from op_kingmd.github.analyzer import RepoAnalyzer
        client = MagicMock()
        analyzer = RepoAnalyzer(client)
        framework = analyzer._detect_framework({"foundry.toml", "src", "test"}, MagicMock())
        assert framework == "foundry"

    def test_detect_framework_hardhat(self):
        from op_kingmd.github.analyzer import RepoAnalyzer
        client = MagicMock()
        analyzer = RepoAnalyzer(client)
        framework = analyzer._detect_framework({"hardhat.config.ts", "contracts"}, MagicMock())
        assert framework == "hardhat"

    def test_detect_framework_none(self):
        from op_kingmd.github.analyzer import RepoAnalyzer
        client = MagicMock()
        analyzer = RepoAnalyzer(client)
        framework = analyzer._detect_framework({"README.md", "src"}, MagicMock())
        assert framework is None

    def test_extract_pragma(self):
        from op_kingmd.github.analyzer import RepoAnalyzer
        pragma = RepoAnalyzer._extract_pragma("pragma solidity ^0.8.24;")
        assert pragma == "^0.8.24"

    def test_detect_chain_solana(self):
        from op_kingmd.github.analyzer import RepoAnalyzer
        chain = RepoAnalyzer._detect_chain("use anchor_lang::prelude::*;\nuse solana_program;")
        assert chain == "solana"

    def test_detect_chain_default_ethereum(self):
        from op_kingmd.github.analyzer import RepoAnalyzer
        chain = RepoAnalyzer._detect_chain("contract Foo {}")
        assert chain == "ethereum"

    def test_compute_complexity(self):
        from op_kingmd.github.analyzer import RepoAnalyzer, RepoProfile
        profile = RepoProfile(
            full_name="test/repo", description="", language="Solidity",
            stars=0, forks=0,
            total_contracts=5, total_loc=1000, has_tests=False, has_audits=False,
        )
        score = RepoAnalyzer._compute_complexity(profile)
        assert 1 <= score <= 10
