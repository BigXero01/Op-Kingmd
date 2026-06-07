"""Solidity-specific tooling: compiler management, ABI parsing, selector computation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CompileResult:
    success: bool
    abi: list[dict] | None = None
    bytecode: str | None = None
    errors: list[str] | None = None
    warnings: list[str] | None = None
    contract_name: str | None = None
    source_map: str | None = None


@dataclass
class FunctionSelector:
    name: str
    signature: str
    selector: str          # 0x{8 hex chars}
    inputs: list[dict]
    outputs: list[dict]
    state_mutability: str


class SolidityTools:
    """Static tooling for Solidity source analysis and compilation."""

    PRAGMA_RE = re.compile(r"pragma solidity\s+([^;]+);")
    CONTRACT_RE = re.compile(r"\bcontract\s+(\w+)\s*(?:is\s+[^{]+)?{")
    FUNCTION_RE = re.compile(
        r"function\s+(\w+)\s*\((.*?)\)\s*"
        r"(external|public|internal|private)?\s*"
        r"(pure|view|payable)?\s*"
        r"(?:returns\s*\((.*?)\))?",
        re.DOTALL,
    )
    EVENT_RE = re.compile(r"event\s+(\w+)\s*\((.*?)\);", re.DOTALL)
    ERROR_RE = re.compile(r"error\s+(\w+)\s*\((.*?)\);")
    MODIFIER_RE = re.compile(r"modifier\s+(\w+)")
    IMPORT_RE = re.compile(r'import\s+(?:{[^}]+}\s+from\s+)?["\']([^"\']+)["\'];')

    # ── Source analysis ───────────────────────────────────────────────────────

    @classmethod
    def extract_pragma(cls, source: str) -> str | None:
        m = cls.PRAGMA_RE.search(source)
        return m.group(1).strip() if m else None

    @classmethod
    def extract_contracts(cls, source: str) -> list[str]:
        return cls.CONTRACT_RE.findall(source)

    @classmethod
    def extract_functions(cls, source: str) -> list[dict[str, Any]]:
        results = []
        for m in cls.FUNCTION_RE.finditer(source):
            results.append({
                "name": m.group(1),
                "params": m.group(2).strip(),
                "visibility": m.group(3) or "internal",
                "mutability": m.group(4) or "",
                "returns": m.group(5) or "",
            })
        return results

    @classmethod
    def extract_events(cls, source: str) -> list[dict[str, str]]:
        return [{"name": m.group(1), "params": m.group(2)} for m in cls.EVENT_RE.finditer(source)]

    @classmethod
    def extract_imports(cls, source: str) -> list[str]:
        return cls.IMPORT_RE.findall(source)

    @classmethod
    def get_function_selectors(cls, abi: list[dict]) -> list[FunctionSelector]:
        selectors = []
        for item in abi:
            if item.get("type") not in ("function", "error"):
                continue
            name = item["name"]
            inputs = item.get("inputs", [])
            param_types = ",".join(inp.get("type", "") for inp in inputs)
            signature = f"{name}({param_types})"
            selector_bytes = hashlib.sha256(signature.encode()).digest()[:4]
            # Use proper keccak256 if web3 is available
            try:
                from eth_utils import keccak
                selector_bytes = keccak(text=signature)[:4]
            except ImportError:
                pass
            selectors.append(
                FunctionSelector(
                    name=name,
                    signature=signature,
                    selector="0x" + selector_bytes.hex(),
                    inputs=inputs,
                    outputs=item.get("outputs", []),
                    state_mutability=item.get("stateMutability", "nonpayable"),
                )
            )
        return selectors

    # ── Compilation ───────────────────────────────────────────────────────────

    @classmethod
    def compile(cls, source: str, contract_name: str | None = None) -> CompileResult:
        """Compile Solidity source using solc (must be installed via solc-select)."""
        solc_path = cls._find_solc()
        if not solc_path:
            return CompileResult(
                success=False,
                errors=["solc not found. Install via: solc-select install 0.8.24 && solc-select use 0.8.24"],
            )

        with tempfile.NamedTemporaryFile(suffix=".sol", mode="w", delete=False) as f:
            f.write(source)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [solc_path, "--abi", "--bin", "--optimize", "--combined-json",
                 "abi,bin,bin-runtime,srcmap", tmp_path],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                errors = [line for line in result.stderr.splitlines() if "Error" in line]
                warnings = [line for line in result.stderr.splitlines() if "Warning" in line]
                return CompileResult(success=False, errors=errors, warnings=warnings)

            combined = json.loads(result.stdout)
            contracts = combined.get("contracts", {})

            # Find target contract
            target = None
            for key, val in contracts.items():
                cname = key.split(":")[-1]
                if contract_name is None or cname == contract_name:
                    target = (cname, val)
                    break

            if not target:
                return CompileResult(success=False, errors=["Contract not found in output"])

            cname, cdata = target
            return CompileResult(
                success=True,
                abi=json.loads(cdata.get("abi", "[]")),
                bytecode=cdata.get("bin", ""),
                contract_name=cname,
                source_map=cdata.get("srcmap", ""),
                warnings=[line for line in result.stderr.splitlines() if "Warning" in line],
            )
        except subprocess.TimeoutExpired:
            return CompileResult(success=False, errors=["Compilation timed out"])
        except Exception as exc:
            return CompileResult(success=False, errors=[str(exc)])
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @staticmethod
    def _find_solc() -> str | None:
        import shutil
        return shutil.which("solc")

    # ── Format helpers ────────────────────────────────────────────────────────

    @classmethod
    def extract_storage_layout(cls, source: str) -> list[dict[str, str]]:
        """Heuristic extraction of state variable declarations."""
        layout = []
        # Match: visibility type name; or type name;
        var_pattern = re.compile(
            r"^\s+(?:(?:public|private|internal|constant|immutable)\s+)+"
            r"([\w\[\]]+(?:\[\])?)\s+(?:public\s+|private\s+|internal\s+)?(\w+)\s*[=;]",
            re.MULTILINE,
        )
        for m in var_pattern.finditer(source):
            layout.append({"type": m.group(1), "name": m.group(2)})
        return layout
