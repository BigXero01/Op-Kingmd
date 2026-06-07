"""Example: Run the Audit-Before-Deploy pipeline on a Solidity contract.

Run:
    python examples/audit_contract.py [path/to/contract.sol]

Or use the built-in vulnerable example to see the audit in action.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

VULNERABLE_EXAMPLE = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.7.6;

// Old version + reentrancy + tx.origin + selfdestruct — all flags triggered
contract VulnerableBank {
    mapping(address => uint256) public balances;
    address public owner;

    constructor() {
        owner = tx.origin;   // tx.origin vulnerability
    }

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "Insufficient");
        // Reentrancy: external call before state update
        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok, "Transfer failed");
        balances[msg.sender] -= amount;
    }

    function emergencyExit() external {
        require(msg.sender == owner, "Not owner");
        selfdestruct(payable(owner));  // selfdestruct vulnerability
    }

    function loop(address[] calldata users) external view returns (uint256 total) {
        for (uint256 i = 0; i < users.length; i++) {  // gas: cache .length
            total += balances[users[i]];
        }
    }
}
"""


def main():
    source_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if source_arg:
        path = Path(source_arg)
        if not path.exists():
            console.print(f"[red]File not found: {source_arg}[/red]")
            sys.exit(1)
        source = path.read_text()
        label = path.name
    else:
        source = VULNERABLE_EXAMPLE
        label = "VulnerableBank.sol (built-in example)"

    console.print(f"\n[bold cyan]Auditing: {label}[/bold cyan]\n")

    from op_kingmd.audit.analyzer import AuditEngine
    from op_kingmd.audit.report import AuditReport, ReportGenerator

    engine = AuditEngine()
    result = engine.audit(source)
    report = AuditReport.from_result(result, chain="ethereum", network="mainnet")

    # Summary panel
    score_color = "green" if result.score >= 75 else "yellow" if result.score >= 60 else "red"
    deploy_str = "[red]BLOCKED ⛔[/red]" if result.blocks_deploy else "[green]ALLOWED ✅[/green]"
    console.print(f"[{score_color}]Risk Score: {result.score}/100 — {result.risk_label}[/{score_color}]  |  Deploy: {deploy_str}\n")

    # Findings table
    if result.findings:
        table = Table(title="Security Findings", show_lines=True)
        table.add_column("Sev", style="bold", width=10)
        table.add_column("Finding", min_width=25)
        table.add_column("Location", min_width=20)
        table.add_column("Recommendation", min_width=35)

        severity_styles = {
            "critical": "red",
            "high": "orange3",
            "medium": "yellow",
            "low": "green",
            "informational": "dim",
        }
        for f in result.findings:
            style = severity_styles.get(f.severity.value, "white")
            table.add_row(
                f"[{style}]{f.severity.value.upper()}[/{style}]",
                f.name,
                f"{f.location or ''}:{f.line or '?'}",
                (f.recommendation or "")[:80],
            )
        console.print(table)
    else:
        console.print("[green]No security findings.[/green]")

    # Gas optimizations
    if result.gas_optimizations:
        console.print(f"\n[yellow]Gas Optimizations ({len(result.gas_optimizations)}):[/yellow]")
        for g in result.gas_optimizations:
            console.print(f"  • {g.description} → {g.savings_estimate or 'savings available'}")

    # Save reports
    gen = ReportGenerator("./audit_reports")
    paths = gen.generate(report, formats=["json", "markdown", "html"])
    console.print("\n[dim]Reports saved:[/dim]")
    for fmt, path in paths.items():
        console.print(f"  [dim]{fmt}: {path}[/dim]")

    if result.blocks_deploy:
        console.print("\n[red bold]Deployment BLOCKED — fix critical/high issues before proceeding.[/red bold]")
        sys.exit(1)


if __name__ == "__main__":
    main()
