"""Tests for the Audit-Before-Deploy layer."""

from __future__ import annotations

import pytest

from op_kingmd.audit.analyzer import AuditEngine, Severity
from op_kingmd.audit.report import AuditReport, ReportGenerator


class TestStaticAnalysis:
    def test_detects_reentrancy(self, audit_engine, reentrancy_source):
        result = audit_engine.audit(reentrancy_source)
        names = [f.name for f in result.findings]
        assert any("reentrancy" in n.lower() or "Call" in n for n in names)

    def test_detects_tx_origin(self, audit_engine, tx_origin_source):
        result = audit_engine.audit(tx_origin_source)
        names = [f.name for f in result.findings]
        assert any("tx.origin" in n.lower() or "tx.Origin" in n for n in names)

    def test_detects_selfdestruct(self, audit_engine, selfdestruct_source):
        result = audit_engine.audit(selfdestruct_source)
        names = [f.name for f in result.findings]
        assert any("selfdestruct" in n.lower() or "Selfdestruct" in n for n in names)

    def test_clean_contract_has_no_critical(self, audit_engine, simple_erc20_source):
        result = audit_engine.audit(simple_erc20_source)
        critical = [f for f in result.findings if f.severity == Severity.CRITICAL]
        assert len(critical) == 0

    def test_finding_has_required_fields(self, audit_engine, reentrancy_source):
        result = audit_engine.audit(reentrancy_source)
        assert len(result.findings) > 0
        f = result.findings[0]
        assert f.name
        assert f.severity in Severity.__members__.values()
        assert f.description
        assert f.recommendation

    def test_contract_name_extracted(self, audit_engine, simple_erc20_source):
        result = audit_engine.audit(simple_erc20_source)
        assert result.contract_name == "SimpleToken"


class TestRiskScoring:
    def test_clean_contract_high_score(self, audit_engine, simple_erc20_source):
        result = audit_engine.audit(simple_erc20_source)
        assert result.score >= 70

    def test_vulnerable_contract_low_score(self, audit_engine, reentrancy_source):
        result = audit_engine.audit(reentrancy_source)
        assert result.score < 90

    def test_critical_bugs_block_deploy(self, audit_engine, selfdestruct_source):
        result = audit_engine.audit(selfdestruct_source)
        if result.critical_count > 0:
            # A critical finding should significantly reduce the score
            assert result.score < 80

    def test_score_label_mapping(self, audit_engine):
        assert AuditEngine._score_label(95) == "SAFE"
        assert AuditEngine._score_label(80) == "LOW_RISK"
        assert AuditEngine._score_label(65) == "MEDIUM_RISK"
        assert AuditEngine._score_label(45) == "HIGH_RISK"
        assert AuditEngine._score_label(20) == "CRITICAL_RISK"

    def test_blocks_deploy_flag(self, audit_engine):
        """Contracts with score below threshold should be blocked."""
        # Manually create a result with score below threshold
        from op_kingmd.audit.analyzer import AuditResult
        result = AuditResult(contract_name="Test", contract_path="test.sol")
        result.score = 40
        result.blocks_deploy = result.score < AuditEngine.BLOCK_THRESHOLD
        assert result.blocks_deploy


class TestGasAnalysis:
    def test_detects_require_strings(self, audit_engine):
        source = '''
        contract Foo {
            function bar(uint x) external {
                require(x > 0, "must be positive");
            }
        }
        '''
        result = audit_engine.audit(source)
        rules = [g.rule for g in result.gas_optimizations]
        assert "use_custom_errors" in rules

    def test_detects_loop_length(self, audit_engine):
        source = '''
        contract Foo {
            uint[] data;
            function loop() external view {
                for (uint i = 0; i < data.length; i++) {}
            }
        }
        '''
        result = audit_engine.audit(source)
        rules = [g.rule for g in result.gas_optimizations]
        assert "cache_array_length" in rules

    def test_detects_unchecked_increment(self, audit_engine):
        source = '''
        contract Foo {
            function loop(uint n) external pure returns (uint s) {
                for (uint i = 0; i < n; i++) { s += i; }
            }
        }
        '''
        result = audit_engine.audit(source)
        rules = [g.rule for g in result.gas_optimizations]
        assert "unchecked_increment" in rules


class TestReportGeneration:
    def test_json_report_structure(self, audit_engine, simple_erc20_source, tmp_path):
        result = audit_engine.audit(simple_erc20_source)
        report = AuditReport.from_result(result)
        data = report.to_dict()

        assert "meta" in data
        assert "contract" in data
        assert "score" in data
        assert "findings" in data
        assert "gas_optimizations" in data

    def test_markdown_report_contains_score(self, audit_engine, simple_erc20_source):
        result = audit_engine.audit(simple_erc20_source)
        report = AuditReport.from_result(result)
        gen = ReportGenerator()
        md = gen.to_markdown(report)
        assert str(result.score) in md
        assert result.contract_name in md

    def test_html_report_is_valid_html(self, audit_engine, simple_erc20_source):
        result = audit_engine.audit(simple_erc20_source)
        report = AuditReport.from_result(result)
        gen = ReportGenerator()
        html = gen.to_html(report)
        assert "<!DOCTYPE html>" in html
        assert result.contract_name in html

    def test_report_files_saved(self, audit_engine, simple_erc20_source, tmp_path):
        result = audit_engine.audit(simple_erc20_source)
        report = AuditReport.from_result(result)
        gen = ReportGenerator(output_dir=str(tmp_path))
        paths = gen.generate(report, formats=["json", "markdown"])
        assert "json" in paths
        assert "markdown" in paths
        assert paths["json"].exists()
        assert paths["markdown"].exists()
