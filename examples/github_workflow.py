"""Example: Full GitHub workflow — analyze repo, audit contracts, open issues, review PRs.

Prerequisites:
    export GITHUB_TOKEN=<your-github-token>
    export HF_TOKEN=<your-hf-token>  # optional

Run:
    python examples/github_workflow.py owner/repo [pr_number]
"""

from __future__ import annotations

import os
import sys

from rich.console import Console

console = Console()


def main():
    repo_name = sys.argv[1] if len(sys.argv) > 1 else "OpenZeppelin/openzeppelin-contracts"
    pr_number = int(sys.argv[2]) if len(sys.argv) > 2 else None

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        console.print("[red]Set GITHUB_TOKEN environment variable[/red]")
        sys.exit(1)

    console.print(f"\n[bold cyan]Op-Kingmd GitHub Workflow — {repo_name}[/bold cyan]\n")

    from op_kingmd.github.analyzer import RepoAnalyzer
    from op_kingmd.github.client import GitHubClient

    with GitHubClient(token) as client:
        # ── Step 1: Analyze repository ─────────────────────────────────────────
        console.print("[bold]Step 1: Repository Analysis[/bold]")
        analyzer = RepoAnalyzer(client)
        with console.status("Analyzing..."):
            profile = analyzer.analyze(repo_name)

        console.print(f"  Framework:    {profile.framework or 'unknown'}")
        console.print(f"  Chain:        {profile.chain or 'unknown'}")
        console.print(f"  Solidity:     {profile.solidity_version or 'unknown'}")
        console.print(f"  Contracts:    {profile.total_contracts}")
        console.print(f"  Has Tests:    {'✅' if profile.has_tests else '❌'}")
        console.print(f"  Has CI:       {'✅' if profile.has_ci else '❌'}")
        console.print(f"  Has Audits:   {'✅' if profile.has_audits else '❌'}")
        console.print(f"  Complexity:   {profile.complexity_score}/10\n")

        # ── Step 2: Audit a sample contract ────────────────────────────────────
        if profile.sol_files:
            console.print("[bold]Step 2: Audit Sample Contract[/bold]")
            sample = profile.sol_files[0]
            console.print(f"  Fetching: {sample}")

            repo = client.get_repo(repo_name)
            fc = client.get_file(repo, sample)
            if fc:
                from op_kingmd.audit.analyzer import AuditEngine
                from op_kingmd.audit.report import AuditReport, ReportGenerator

                engine = AuditEngine()
                with console.status("Auditing..."):
                    result = engine.audit(fc.content, fc.path.split("/")[-1].replace(".sol", ""))

                color = "green" if result.score >= 75 else "yellow" if result.score >= 60 else "red"
                console.print(f"  [{color}]Score: {result.score}/100 — {result.risk_label}[/{color}]")
                console.print(f"  Findings: {len(result.findings)} | Gas opts: {len(result.gas_optimizations)}\n")

        # ── Step 3: Generate suggested issues ──────────────────────────────────
        console.print("[bold]Step 3: Suggested Issues[/bold]")
        issues = analyzer.generate_issues(profile)
        for i, issue in enumerate(issues[:3], 1):
            console.print(f"  [{i}] {issue['title']}")
            console.print(f"      Labels: {', '.join(issue['labels'])}\n")

        # ── Step 4: PR Review (if PR number provided) ───────────────────────────
        if pr_number:
            console.print(f"[bold]Step 4: PR #{pr_number} Review[/bold]")
            from op_kingmd.github.pr_review import PRReviewer

            reviewer = PRReviewer(client)
            with console.status(f"Reviewing PR #{pr_number}..."):
                review = reviewer.review_pr(repo_name, pr_number)

            outcome_color = "red" if review.security_issues > 0 else "green"
            console.print(f"  [{outcome_color}]Outcome: {review.overall}[/{outcome_color}]")
            console.print(f"  Security Issues: {review.security_issues}")
            console.print(f"  Gas Suggestions: {review.gas_suggestions}")
            console.print(f"  Total Comments:  {len(review.comments)}")

            post = input("\n  Post review to GitHub? [y/N] ").strip().lower()
            if post == "y":
                reviewer.post_review(review)
                console.print("  [green]✓ Review posted[/green]")

    console.print("\n[green]✓ Workflow complete[/green]")


if __name__ == "__main__":
    main()
