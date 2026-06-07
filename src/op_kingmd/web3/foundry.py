"""Foundry integration: forge build, test, coverage, snapshot, script execution."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ForgeResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int
    command: str

    @property
    def failed(self) -> bool:
        return not self.success


class FoundryRunner:
    """Thin wrapper around `forge` CLI for build, test, and script execution."""

    def __init__(self, project_root: str | Path = ".") -> None:
        self.root = Path(project_root).resolve()

    # ── Core commands ─────────────────────────────────────────────────────────

    def build(self, extra_args: list[str] | None = None) -> ForgeResult:
        return self._run(["forge", "build", *(extra_args or [])])

    def test(
        self,
        match: str | None = None,
        verbosity: int = 2,
        gas_report: bool = False,
        fuzz_runs: int | None = None,
        extra_args: list[str] | None = None,
    ) -> ForgeResult:
        args = ["forge", "test", f"-{verbosity * 'v'}"]
        if match:
            args += ["--match-test", match]
        if gas_report:
            args.append("--gas-report")
        if fuzz_runs:
            args += ["--fuzz-runs", str(fuzz_runs)]
        args += extra_args or []
        return self._run(args)

    def coverage(self, report: str = "lcov") -> ForgeResult:
        return self._run(["forge", "coverage", "--report", report])

    def snapshot(self) -> ForgeResult:
        return self._run(["forge", "snapshot"])

    def script(
        self,
        script_path: str,
        rpc_url: str | None = None,
        broadcast: bool = False,
        verify: bool = False,
        private_key: str | None = None,
        extra_args: list[str] | None = None,
    ) -> ForgeResult:
        args = ["forge", "script", script_path]
        if rpc_url:
            args += ["--rpc-url", rpc_url]
        if broadcast:
            args.append("--broadcast")
        if verify:
            args.append("--verify")
        if private_key:
            args += ["--private-key", private_key]
        args += extra_args or []
        return self._run(args)

    def fmt(self, check: bool = False) -> ForgeResult:
        args = ["forge", "fmt"]
        if check:
            args.append("--check")
        return self._run(args)

    def install(self, packages: list[str]) -> ForgeResult:
        return self._run(["forge", "install", "--no-commit", *packages])

    def init(self, name: str, template: str | None = None) -> ForgeResult:
        args = ["forge", "init", name]
        if template:
            args += ["--template", template]
        return self._run(args, cwd=self.root.parent)

    # ── Gas analysis ──────────────────────────────────────────────────────────

    def gas_snapshot_diff(self) -> dict[str, Any]:
        """Run snapshot and parse diff against stored .gas-snapshot."""
        result = self.snapshot()
        if not result.success:
            return {"error": result.stderr}

        snapshot_file = self.root / ".gas-snapshot"
        if not snapshot_file.exists():
            return {"message": "No previous snapshot to compare against."}

        lines = snapshot_file.read_text().splitlines()
        snapshots = {}
        for line in lines:
            if ":" in line:
                test, gas = line.rsplit(":", 1)
                snapshots[test.strip()] = int(gas.strip().replace(",", ""))
        return snapshots

    # ── Project scaffolding ───────────────────────────────────────────────────

    def scaffold_project(
        self, name: str, dest: Path, add_openzeppelin: bool = True
    ) -> Path:
        """Create a new Foundry project with standard structure and OZ contracts."""
        project_path = dest / name
        result = self._run(
            ["forge", "init", str(project_path)], cwd=dest
        )
        if not result.success:
            raise RuntimeError(f"forge init failed: {result.stderr}")

        if add_openzeppelin:
            self._run(
                ["forge", "install", "OpenZeppelin/openzeppelin-contracts", "--no-commit"],
                cwd=project_path,
            )
            remappings = project_path / "remappings.txt"
            remappings.write_text(
                "@openzeppelin/=lib/openzeppelin-contracts/\n"
                "@openzeppelin/contracts/=lib/openzeppelin-contracts/contracts/\n"
            )

        logger.info("project_scaffolded", path=str(project_path))
        return project_path

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self, args: list[str], cwd: Path | None = None) -> ForgeResult:
        cwd = cwd or self.root
        cmd = " ".join(args)
        logger.debug("forge_run", command=cmd)
        try:
            proc = subprocess.run(
                args, capture_output=True, text=True, cwd=str(cwd), timeout=300
            )
            success = proc.returncode == 0
            if not success:
                logger.warning("forge_failed", command=cmd, stderr=proc.stderr[:500])
            return ForgeResult(
                success=success,
                stdout=proc.stdout,
                stderr=proc.stderr,
                return_code=proc.returncode,
                command=cmd,
            )
        except FileNotFoundError:
            msg = "forge not found. Install Foundry: curl -L https://foundry.paradigm.xyz | bash"
            logger.error("forge_not_found")
            return ForgeResult(success=False, stdout="", stderr=msg, return_code=-1, command=cmd)
        except subprocess.TimeoutExpired:
            return ForgeResult(
                success=False, stdout="", stderr="Timeout", return_code=-1, command=cmd
            )

    @staticmethod
    def is_available() -> bool:
        import shutil
        return shutil.which("forge") is not None
