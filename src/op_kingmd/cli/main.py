"""Op-Kingmd CLI — vibe coding for Web3.

Commands:
  generate    Generate smart contracts from natural language
  audit       Security audit before deployment
  review      AI-powered code review
  github      GitHub integration commands
  serve       Start the API server
  train       Fine-tune the model
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

app = typer.Typer(
    name="op-kingmd",
    help="Web3-focused coding LLM — vibe coding for smart contracts",
    rich_markup_mode="rich",
)
generate_app = typer.Typer(help="Code generation commands")
audit_app = typer.Typer(help="Security audit commands")
github_app = typer.Typer(help="GitHub integration commands")
train_app = typer.Typer(help="Model training commands")

app.add_typer(generate_app, name="generate")
app.add_typer(audit_app, name="audit")
app.add_typer(github_app, name="github")
app.add_typer(train_app, name="train")

console = Console()
err_console = Console(stderr=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_pipeline():
    """Lazy-load the Web3Pipeline (expensive — loads LLM weights)."""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        p.add_task("Loading model...", total=None)
        from op_kingmd.core.pipeline import Web3Pipeline
        pipeline = Web3Pipeline()
    return pipeline


def _print_score_panel(score: int, risk_label: str, blocks_deploy: bool) -> None:
    color = "green" if score >= 75 else "yellow" if score >= 60 else "red"
    deploy_str = "[red]BLOCKED ⛔[/red]" if blocks_deploy else "[green]ALLOWED ✅[/green]"
    console.print(Panel(
        f"[{color}]Risk Score: {score}/100[/{color}]  |  {risk_label}  |  Deploy: {deploy_str}",
        title="Audit Result",
        border_style=color,
    ))


# ── generate commands ─────────────────────────────────────────────────────────

@generate_app.command("contract")
def generate_contract(
    description: str = typer.Argument(..., help="What to build (natural language)"),
    language: str = typer.Option("solidity", "--lang", "-l", help="solidity / rust / typescript"),
    framework: str = typer.Option("foundry", "--framework", "-f", help="foundry / hardhat"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    no_tests: bool = typer.Option(False, "--no-tests", help="Skip test generation"),
    no_deploy: bool = typer.Option(False, "--no-deploy", help="Skip deploy script"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream output tokens"),
):
    """Generate a smart contract from a natural language description."""
    from op_kingmd.core.pipeline import CodeGenRequest, Framework, Language, Web3Pipeline

    pipeline = _load_pipeline()
    req = CodeGenRequest(
        description=description,
        language=Language(language.lower()),
        framework=Framework(framework.lower()),
        include_tests=not no_tests,
        include_deploy_script=not no_deploy,
        gas_optimize=True,
    )

    console.print(f"\n[bold cyan]Generating {language} contract with {framework}...[/bold cyan]\n")

    if stream:
        from op_kingmd.core.pipeline import SYSTEM_PROMPT
        full = ""
        for token in pipeline.engine.stream(description):
            console.print(token, end="", markup=False)
            full += token
        console.print()
        _save_outputs({"contract": full}, output)
        return

    result = pipeline.generate_contract(req)

    console.print(Panel(
        Syntax(result.code, language.lower(), theme="monokai", line_numbers=True),
        title=f"[green]{language.upper()} Contract[/green]",
    ))

    if result.warnings:
        for w in result.warnings:
            console.print(f"[yellow]⚠  {w}[/yellow]")

    if output:
        _save_outputs({
            "contract": result.code,
            "tests": result.tests,
            "deploy": result.deploy_script,
        }, output, language)
        console.print(f"\n[green]✓ Saved to {output}[/green]")


@generate_app.command("erc20")
def generate_erc20(
    name: str = typer.Argument(..., help="Token name"),
    symbol: str = typer.Argument(..., help="Token symbol"),
    supply: int = typer.Option(1_000_000, help="Initial supply"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
):
    """Generate a production-ready ERC-20 token."""
    from op_kingmd.web3.codegen import Web3CodeGen
    cg = Web3CodeGen()
    console.print(f"\n[bold cyan]Generating ERC-20: {name} ({symbol})[/bold cyan]\n")
    with console.status("Generating..."):
        pkg = cg.create_erc20(name, symbol, supply)
    _display_package(pkg, output)


@generate_app.command("erc721")
def generate_erc721(
    name: str = typer.Argument(..., help="Collection name"),
    symbol: str = typer.Argument(..., help="Collection symbol"),
    supply: int = typer.Option(10_000, help="Max supply"),
    price: float = typer.Option(0.05, help="Mint price in ETH"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
):
    """Generate a production-ready NFT collection."""
    from op_kingmd.web3.codegen import Web3CodeGen
    cg = Web3CodeGen()
    console.print(f"\n[bold cyan]Generating ERC-721: {name} ({symbol})[/bold cyan]\n")
    with console.status("Generating..."):
        pkg = cg.create_erc721(name, symbol, supply, price)
    _display_package(pkg, output)


@generate_app.command("tests")
def generate_tests(
    source: str = typer.Argument(..., help="Path to Solidity file"),
    framework: str = typer.Option("foundry", "--framework", "-f"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
):
    """Generate comprehensive tests for an existing contract."""
    from op_kingmd.core.pipeline import Framework, Language, Web3Pipeline
    code = Path(source).read_text()
    pipeline = _load_pipeline()
    console.print(f"\n[bold cyan]Generating tests for {source}...[/bold cyan]\n")
    with console.status("Generating..."):
        tests = pipeline.generate_tests(code, Language.SOLIDITY, Framework(framework))
    console.print(Syntax(tests, "solidity", theme="monokai", line_numbers=True))
    if output:
        Path(output).write_text(tests)
        console.print(f"[green]✓ Saved to {output}[/green]")


# ── audit commands ────────────────────────────────────────────────────────────

@audit_app.command("run")
def audit_run(
    target: str = typer.Argument(..., help="Path to .sol file or directory"),
    chain: str = typer.Option("ethereum", help="Target chain"),
    network: str = typer.Option("mainnet", help="Target network"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o"),
    format: str = typer.Option("markdown", "--format", help="json / markdown / html"),
    block_on_fail: bool = typer.Option(True, help="Exit with code 1 if deploy blocked"),
):
    """Run the Audit-Before-Deploy pipeline on a contract."""
    from op_kingmd.audit.analyzer import AuditEngine
    from op_kingmd.audit.report import AuditReport, ReportGenerator

    engine = AuditEngine()
    reporter = ReportGenerator(output_dir)

    p = Path(target)
    if p.is_dir():
        console.print(f"\n[bold cyan]Auditing directory: {target}[/bold cyan]\n")
        results = engine.audit_directory(target)
    elif p.exists():
        console.print(f"\n[bold cyan]Auditing: {target}[/bold cyan]\n")
        results = [engine.audit(target)]
    else:
        err_console.print(f"[red]Path not found: {target}[/red]")
        raise typer.Exit(1)

    blocked_any = False
    for result in results:
        report = AuditReport.from_result(result, chain=chain, network=network)
        paths = reporter.generate(report, formats=[format])
        _print_score_panel(result.score, result.risk_label, result.blocks_deploy)

        if result.findings:
            table = Table(title="Findings", show_header=True)
            table.add_column("Severity", style="bold")
            table.add_column("Name")
            table.add_column("Location")
            for f in result.findings:
                color = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green"}.get(f.severity.value, "white")
                table.add_row(f"[{color}]{f.severity.value.upper()}[/{color}]", f.name, f"{f.location}:{f.line or '?'}")
            console.print(table)

        if result.gas_optimizations:
            console.print(f"\n[dim]Gas optimizations: {len(result.gas_optimizations)} suggestions[/dim]")

        for fmt, path in paths.items():
            console.print(f"[dim]Report ({fmt}): {path}[/dim]")

        if result.blocks_deploy:
            blocked_any = True

    if blocked_any and block_on_fail:
        err_console.print("\n[red bold]⛔ Deployment blocked by audit engine.[/red bold]")
        raise typer.Exit(1)


@audit_app.command("quick")
def audit_quick(
    source: str = typer.Argument(..., help="Solidity source code or file path"),
):
    """Quick inline audit — prints findings to terminal only."""
    from op_kingmd.audit.analyzer import AuditEngine
    engine = AuditEngine()
    code = Path(source).read_text() if Path(source).exists() else source
    result = engine.audit(code)
    _print_score_panel(result.score, result.risk_label, result.blocks_deploy)
    for f in result.findings:
        color = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green"}.get(f.severity.value, "white")
        console.print(f"  [{color}]{f.severity.value.upper()}[/{color}]  {f.name}  {f.location}:{f.line or '?'}")
        console.print(f"    {f.description}", style="dim")


# ── github commands ───────────────────────────────────────────────────────────

@github_app.command("analyze")
def github_analyze(
    repo: str = typer.Argument(..., help="owner/repo"),
):
    """Analyze a GitHub repository's Web3 stack and tech profile."""
    from op_kingmd.github.analyzer import RepoAnalyzer
    from op_kingmd.github.client import GitHubClient
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        err_console.print("[red]GITHUB_TOKEN not set[/red]")
        raise typer.Exit(1)
    client = GitHubClient(token)
    analyzer = RepoAnalyzer(client)
    with console.status(f"Analyzing {repo}..."):
        profile = analyzer.analyze(repo)

    table = Table(title=f"Repository Profile: {repo}")
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("Framework", profile.framework or "unknown")
    table.add_row("Chain", profile.chain or "unknown")
    table.add_row("Solidity Version", profile.solidity_version or "unknown")
    table.add_row("Total Contracts", str(profile.total_contracts))
    table.add_row("Has Tests", "✅" if profile.has_tests else "❌")
    table.add_row("Has CI", "✅" if profile.has_ci else "❌")
    table.add_row("Has Audits", "✅" if profile.has_audits else "❌")
    table.add_row("Complexity", f"{profile.complexity_score}/10")
    console.print(table)


@github_app.command("review-pr")
def review_pr(
    repo: str = typer.Argument(..., help="owner/repo"),
    pr: int = typer.Argument(..., help="PR number"),
    post: bool = typer.Option(False, "--post", help="Post review to GitHub"),
):
    """AI security review of a pull request."""
    from op_kingmd.github.client import GitHubClient
    from op_kingmd.github.pr_review import PRReviewer
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        err_console.print("[red]GITHUB_TOKEN not set[/red]")
        raise typer.Exit(1)
    client = GitHubClient(token)
    reviewer = PRReviewer(client)
    with console.status(f"Reviewing PR #{pr}..."):
        result = reviewer.review_pr(repo, pr)
        if post:
            reviewer.post_review(result)

    console.print(Panel(
        Markdown(result.summary),
        title=f"PR #{pr} Review — {result.overall}",
        border_style="red" if result.security_issues > 0 else "green",
    ))


# ── serve command ─────────────────────────────────────────────────────────────

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Port"),
    reload: bool = typer.Option(False, help="Hot reload (dev only)"),
    workers: int = typer.Option(1, help="Number of workers"),
):
    """Start the Op-Kingmd API server."""
    import uvicorn
    console.print(f"[bold green]Starting Op-Kingmd API on {host}:{port}[/bold green]")
    uvicorn.run(
        "op_kingmd.api.server:create_app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        factory=True,
    )


# ── train commands ────────────────────────────────────────────────────────────

@train_app.command("run")
def train_run(
    config: str = typer.Option("config/model_config.yaml", help="Path to model config YAML"),
    data: Optional[str] = typer.Option(None, help="Path to local training JSONL"),
    push_to_hub: Optional[str] = typer.Option(None, help="HF repo to push fine-tuned model"),
):
    """Fine-tune Op-Kingmd on Web3 code data."""
    from op_kingmd.training.dataset import DatasetConfig, Web3Dataset
    from op_kingmd.training.finetune import LoRATrainer

    trainer = LoRATrainer.from_config_file(config)
    trainer.setup()

    ds_config = DatasetConfig(local_files=[data] if data else [])
    dataset_loader = Web3Dataset(ds_config)

    console.print("[bold cyan]Loading tokenizer to prepare dataset...[/bold cyan]")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(trainer.config.model_id, trust_remote_code=True)
    dataset = dataset_loader.load(tokenizer)

    console.print(f"[green]Dataset ready: {len(dataset['train'])} train / {len(dataset['validation'])} val[/green]")
    trainer.train(dataset)

    if push_to_hub:
        trainer.push_to_hub(push_to_hub)
        console.print(f"[green]✓ Pushed to {push_to_hub}[/green]")


# ── utilities ──────────────────────────────────────────────────────────────────

def _save_outputs(files: dict[str, str | None], output_dir: str, ext: str = "sol") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ext_map = {"contract": ext, "tests": f"Test.{ext}", "deploy": f"Deploy.s.{ext}"}
    for key, content in files.items():
        if content:
            path = out / ext_map.get(key, f"{key}.txt")
            path.write_text(content)


def _display_package(pkg: dict[str, str], output_dir: str | None) -> None:
    for key, content in pkg.items():
        if content and key != "warnings":
            lang = "solidity" if key in ("contract", "tests", "deploy") else "markdown"
            console.print(Panel(
                Syntax(content[:3000], lang, theme="monokai", line_numbers=True),
                title=f"[green]{key.upper()}[/green]",
            ))
    if "warnings" in pkg:
        console.print(f"[yellow]⚠  {pkg['warnings']}[/yellow]")
    if output_dir:
        _save_outputs(pkg, output_dir)
        console.print(f"\n[green]✓ Saved to {output_dir}[/green]")


if __name__ == "__main__":
    app()
