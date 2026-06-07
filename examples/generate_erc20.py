"""Example: Generate a production-ready ERC-20 token with Op-Kingmd.

Run:
    python examples/generate_erc20.py

Prerequisites:
    pip install -e ".[dev]"
    export HF_TOKEN=<your-token>         # optional but recommended
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.syntax import Syntax

console = Console()

OUTPUT_DIR = Path("./output/token")

def main():
    console.print("[bold cyan]Op-Kingmd — ERC-20 Generator Example[/bold cyan]\n")

    # Initialize code generator
    from op_kingmd.web3.codegen import Web3CodeGen
    from op_kingmd.core.pipeline import Framework
    cg = Web3CodeGen()

    # Generate a complete ERC-20 package
    with console.status("Generating MyToken ERC-20..."):
        package = cg.create_erc20(
            name="MyToken",
            symbol="MTK",
            initial_supply=1_000_000,
            features=[
                "Mintable by owner",
                "Burnable by holder",
                "Pausable emergency stop",
                "Permit (EIP-2612) gasless approvals",
            ],
            framework=Framework.FOUNDRY,
        )

    # Display the contract
    if "contract" in package:
        console.print("\n[green]── Contract ─────────────────────────────────[/green]")
        console.print(Syntax(package["contract"][:3000], "solidity", theme="monokai", line_numbers=True))

    # Display tests
    if "tests" in package:
        console.print("\n[green]── Tests ─────────────────────────────────────[/green]")
        console.print(Syntax(package["tests"][:2000], "solidity", theme="monokai"))

    # Run audit on the generated contract
    if "contract" in package:
        console.print("\n[bold yellow]Running Audit-Before-Deploy...[/bold yellow]")
        from op_kingmd.audit.analyzer import AuditEngine
        from op_kingmd.audit.report import AuditReport, ReportGenerator

        engine = AuditEngine()
        result = engine.audit(package["contract"], "MyToken")
        report = AuditReport.from_result(result, chain="ethereum", network="mainnet")

        gen = ReportGenerator(output_dir=str(OUTPUT_DIR / "audit"))
        paths = gen.generate(report, formats=["json", "markdown"])

        color = "green" if result.score >= 75 else "yellow" if result.score >= 60 else "red"
        console.print(f"[{color}]Risk Score: {result.score}/100 — {result.risk_label}[/{color}]")
        if result.blocks_deploy:
            console.print("[red]⛔ Deployment would be BLOCKED. Fix critical issues first.[/red]")
        else:
            console.print("[green]✅ Contract passes audit threshold — safe to deploy.[/green]")

        for fmt, path in paths.items():
            console.print(f"  [dim]Report ({fmt}): {path}[/dim]")

    # Save files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if "contract" in package:
        (OUTPUT_DIR / "MyToken.sol").write_text(package["contract"])
    if "tests" in package:
        (OUTPUT_DIR / "MyToken.t.sol").write_text(package["tests"])
    if "deploy" in package:
        (OUTPUT_DIR / "DeployMyToken.s.sol").write_text(package["deploy"])

    console.print(f"\n[green]✓ Files saved to {OUTPUT_DIR}[/green]")


if __name__ == "__main__":
    main()
