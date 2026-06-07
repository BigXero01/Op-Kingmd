"""Hardhat integration: compile, test, deploy, verify via npx hardhat."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class HardhatResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int
    command: str


class HardhatRunner:
    """Thin wrapper around `npx hardhat` CLI for Hardhat-based projects."""

    def __init__(self, project_root: str | Path = ".") -> None:
        self.root = Path(project_root).resolve()

    # ── Core commands ─────────────────────────────────────────────────────────

    def compile(self) -> HardhatResult:
        return self._run(["npx", "hardhat", "compile"])

    def test(
        self,
        match: str | None = None,
        network: str = "hardhat",
        parallel: bool = False,
    ) -> HardhatResult:
        args = ["npx", "hardhat", "test", "--network", network]
        if match:
            args += ["--grep", match]
        if parallel:
            args.append("--parallel")
        return self._run(args)

    def coverage(self) -> HardhatResult:
        return self._run(["npx", "hardhat", "coverage"])

    def node(self, port: int = 8545, fork: str | None = None) -> HardhatResult:
        args = ["npx", "hardhat", "node", "--port", str(port)]
        if fork:
            args += ["--fork", fork]
        return self._run(args)

    def run_script(
        self,
        script_path: str,
        network: str = "hardhat",
    ) -> HardhatResult:
        return self._run(["npx", "hardhat", "run", script_path, "--network", network])

    def verify(
        self,
        address: str,
        network: str,
        constructor_args: list[str] | None = None,
    ) -> HardhatResult:
        args = ["npx", "hardhat", "verify", "--network", network, address]
        if constructor_args:
            args += constructor_args
        return self._run(args)

    def gas_reporter(self) -> HardhatResult:
        env = {"REPORT_GAS": "true"}
        return self._run(["npx", "hardhat", "test"], extra_env=env)

    # ── Config generation ─────────────────────────────────────────────────────

    def generate_config(
        self,
        networks: dict[str, dict] | None = None,
        solidity_version: str = "0.8.24",
        optimizer_runs: int = 200,
    ) -> str:
        """Generate a hardhat.config.ts content string."""
        networks = networks or {
            "hardhat": {"chainId": 31337},
            "mainnet": {
                "url": "${process.env.ETH_RPC_URL}",
                "accounts": ["${process.env.PRIVATE_KEY}"],
            },
            "sepolia": {
                "url": "${process.env.SEPOLIA_RPC_URL}",
                "accounts": ["${process.env.PRIVATE_KEY}"],
            },
        }

        network_strings = []
        for name, cfg in networks.items():
            parts = []
            for k, v in cfg.items():
                if isinstance(v, str) and "${" in v:
                    parts.append(f'      {k}: "{v}"')
                elif isinstance(v, list):
                    items = ", ".join(f'"{i}"' for i in v)
                    parts.append(f"      {k}: [{items}]")
                else:
                    parts.append(f"      {k}: {json.dumps(v)}")
            network_strings.append(f"    {name}: {{\n" + ",\n".join(parts) + "\n    }}")

        networks_block = ",\n".join(network_strings)
        return f"""\
import {{ HardhatUserConfig }} from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";
import "@openzeppelin/hardhat-upgrades";
import "dotenv/config";

const config: HardhatUserConfig = {{
  solidity: {{
    version: "{solidity_version}",
    settings: {{
      optimizer: {{
        enabled: true,
        runs: {optimizer_runs},
      }},
      viaIR: true,
    }},
  }},
  networks: {{
{networks_block}
  }},
  gasReporter: {{
    enabled: process.env.REPORT_GAS === "true",
    currency: "USD",
    coinmarketcap: process.env.CMC_API_KEY,
  }},
  etherscan: {{
    apiKey: process.env.ETHERSCAN_API_KEY,
  }},
}};

export default config;
"""

    def scaffold_project(self, name: str, dest: Path) -> Path:
        """Scaffold a new Hardhat TypeScript project."""
        project_path = dest / name
        project_path.mkdir(parents=True, exist_ok=True)
        result = self._run(
            ["npx", "--yes", "create-hardhat@latest", "."],
            cwd=project_path,
        )
        if not result.success:
            raise RuntimeError(f"Hardhat scaffold failed: {result.stderr}")
        logger.info("hardhat_project_scaffolded", path=str(project_path))
        return project_path

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(
        self,
        args: list[str],
        cwd: Path | None = None,
        extra_env: dict | None = None,
    ) -> HardhatResult:
        import os
        cwd = cwd or self.root
        cmd = " ".join(args)
        env = {**os.environ, **(extra_env or {})}
        logger.debug("hardhat_run", command=cmd)
        try:
            proc = subprocess.run(
                args, capture_output=True, text=True, cwd=str(cwd),
                timeout=300, env=env,
            )
            return HardhatResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                return_code=proc.returncode,
                command=cmd,
            )
        except FileNotFoundError:
            msg = "npx/node not found. Install Node.js >= 18."
            return HardhatResult(success=False, stdout="", stderr=msg, return_code=-1, command=cmd)
        except subprocess.TimeoutExpired:
            return HardhatResult(
                success=False, stdout="", stderr="Timeout", return_code=-1, command=cmd
            )

    @staticmethod
    def is_available() -> bool:
        import shutil
        return shutil.which("npx") is not None
