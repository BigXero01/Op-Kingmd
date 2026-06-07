"""High-level Web3 inference pipeline with structured output and prompt templates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

from op_kingmd.core.model import GenerationParams, ModelConfig, ModelEngine

logger = structlog.get_logger(__name__)


class Language(str, Enum):
    SOLIDITY = "solidity"
    RUST = "rust"
    TYPESCRIPT = "typescript"
    PYTHON = "python"
    VYPER = "vyper"
    MOVE = "move"
    AUTO = "auto"


class Framework(str, Enum):
    FOUNDRY = "foundry"
    HARDHAT = "hardhat"
    ANCHOR = "anchor"
    TRUFFLE = "truffle"
    NONE = "none"
    AUTO = "auto"


@dataclass
class CodeGenRequest:
    description: str
    language: Language = Language.SOLIDITY
    framework: Framework = Framework.FOUNDRY
    include_tests: bool = True
    include_docs: bool = True
    include_deploy_script: bool = True
    gas_optimize: bool = True
    security_checks: bool = True
    context: str | None = None           # Extra code/ABI context
    existing_code: str | None = None     # Code to modify/extend


@dataclass
class CodeGenResult:
    code: str
    tests: str | None
    docs: str | None
    deploy_script: str | None
    language: Language
    framework: Framework
    model_id: str
    warnings: list[str]
    raw_response: str


class Web3Pipeline:
    """Orchestrates code generation, review, and explanation for Web3 tasks."""

    PROMPT_TEMPLATES = {
        "generate_contract": """\
Task: Generate a production-ready smart contract.

Description: {description}
Language: {language}
Framework: {framework}
{context_block}

Requirements:
- Full NatSpec documentation (/// and /** */ comments)
- Access control (OpenZeppelin Ownable/AccessControl where appropriate)
- Events for all state changes
- Custom errors (not require strings) for gas efficiency
- ReentrancyGuard on any functions moving funds
- Input validation on all public/external functions
{gas_block}
{tests_block}
{deploy_block}

Output format:
1. Main contract source code
2. Test file (if requested)
3. Deployment script (if requested)
4. Brief security notes

Begin:
""",
        "review_code": """\
Perform a comprehensive security and quality review of this smart contract:

```{language}
{code}
```

Analyze for:
1. Reentrancy vulnerabilities
2. Access control issues
3. Integer overflow/underflow (if < 0.8)
4. Front-running and MEV exposure
5. Oracle manipulation risks
6. Flash loan attack vectors
7. Gas inefficiencies
8. Missing events
9. Centralization risks
10. Logic errors

Format your response as:
## Summary
## Critical Issues
## High Issues
## Medium Issues
## Low Issues
## Gas Optimizations
## Recommendations
""",
        "explain_code": """\
Explain this Web3 code clearly for a developer who is new to this codebase:

```{language}
{code}
```

Cover:
- What it does (purpose and functionality)
- How it works (key mechanisms)
- Security model (who can call what)
- Integration points (events, interfaces, external calls)
- Gotchas or non-obvious behavior
""",
        "fix_vulnerability": """\
Fix the following vulnerability in this smart contract:

Vulnerability: {vulnerability}
Severity: {severity}

Vulnerable code:
```{language}
{code}
```

Provide:
1. The fixed code
2. Explanation of what was wrong
3. How the fix prevents the attack
4. Any additional hardening recommendations
""",
        "generate_tests": """\
Generate comprehensive tests for this smart contract using {framework}:

```{language}
{code}
```

Include tests for:
- Happy path (all public functions)
- Edge cases (boundary values, zero amounts, max values)
- Access control (unauthorized callers should revert)
- Reentrancy attacks
- Integer edge cases
- Event emission verification
- Fuzz tests (if Foundry)
- Invariant tests (if Foundry)
""",
        "optimize_gas": """\
Optimize the gas usage of this smart contract without changing its behavior:

```{language}
{code}
```

Apply:
- Custom errors instead of require strings
- Immutable/constant for unchanging variables
- Packed struct storage
- Unchecked arithmetic where safe
- Cached storage reads in loops
- Short-circuit evaluation
- Bitmap for boolean flags

Return the optimized contract with inline comments explaining each optimization
and an estimated gas savings table.
""",
    }

    def __init__(self, engine: ModelEngine | None = None) -> None:
        self._engine = engine or ModelEngine(ModelConfig.from_env())

    @property
    def engine(self) -> ModelEngine:
        return self._engine

    # ── Core generation ───────────────────────────────────────────────────────

    def generate_contract(self, req: CodeGenRequest) -> CodeGenResult:
        """Generate a smart contract from a natural language description."""
        logger.info(
            "generating_contract",
            language=req.language,
            framework=req.framework,
            description=req.description[:80],
        )

        context_block = f"Context:\n{req.context}" if req.context else ""
        gas_block = "- Minimize gas usage with packed storage, custom errors, unchecked loops" if req.gas_optimize else ""
        tests_block = "- Include full Foundry/Hardhat test suite" if req.include_tests else ""
        deploy_block = "- Include deployment script" if req.include_deploy_script else ""

        prompt = self.PROMPT_TEMPLATES["generate_contract"].format(
            description=req.description,
            language=req.language.value,
            framework=req.framework.value,
            context_block=context_block,
            gas_block=gas_block,
            tests_block=tests_block,
            deploy_block=deploy_block,
        )

        raw = self._engine.generate(prompt)
        return self._parse_contract_response(raw, req)

    def review_contract(self, code: str, language: Language = Language.SOLIDITY) -> str:
        """Security and quality review of a contract."""
        prompt = self.PROMPT_TEMPLATES["review_code"].format(
            language=language.value, code=code
        )
        return self._engine.generate(prompt)

    def explain_code(self, code: str, language: Language = Language.SOLIDITY) -> str:
        """Plain-English explanation of contract code."""
        prompt = self.PROMPT_TEMPLATES["explain_code"].format(
            language=language.value, code=code
        )
        return self._engine.generate(prompt)

    def fix_vulnerability(
        self,
        code: str,
        vulnerability: str,
        severity: str = "high",
        language: Language = Language.SOLIDITY,
    ) -> str:
        """Generate a fix for a specific vulnerability."""
        prompt = self.PROMPT_TEMPLATES["fix_vulnerability"].format(
            language=language.value,
            code=code,
            vulnerability=vulnerability,
            severity=severity,
        )
        return self._engine.generate(prompt)

    def generate_tests(
        self,
        code: str,
        language: Language = Language.SOLIDITY,
        framework: Framework = Framework.FOUNDRY,
    ) -> str:
        """Generate comprehensive tests for a contract."""
        prompt = self.PROMPT_TEMPLATES["generate_tests"].format(
            language=language.value, framework=framework.value, code=code
        )
        return self._engine.generate(prompt)

    def optimize_gas(self, code: str, language: Language = Language.SOLIDITY) -> str:
        """Return gas-optimized version of the contract."""
        prompt = self.PROMPT_TEMPLATES["optimize_gas"].format(
            language=language.value, code=code
        )
        return self._engine.generate(prompt)

    def freeform(self, prompt: str, system: str | None = None) -> str:
        """Freeform generation — useful for custom prompts."""
        return self._engine.generate(prompt, system=system)

    def stream_freeform(self, prompt: str, system: str | None = None):
        """Stream tokens from a freeform prompt."""
        yield from self._engine.stream(prompt, system=system)

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_contract_response(self, raw: str, req: CodeGenRequest) -> CodeGenResult:
        code = self._extract_code_block(raw, req.language.value)
        tests = self._extract_section(raw, "test") if req.include_tests else None
        docs = self._extract_section(raw, "documentation|natspec|doc") if req.include_docs else None
        deploy = self._extract_section(raw, "deploy") if req.include_deploy_script else None
        warnings = self._extract_warnings(raw)

        return CodeGenResult(
            code=code or raw,
            tests=tests,
            docs=docs,
            deploy_script=deploy,
            language=req.language,
            framework=req.framework,
            model_id=self._engine.model_id,
            warnings=warnings,
            raw_response=raw,
        )

    @staticmethod
    def _extract_code_block(text: str, language: str) -> str | None:
        patterns = [
            rf"```{language}\s*\n(.*?)```",
            r"```(?:sol|solidity|rs|rust|ts|typescript|js|javascript)\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    @staticmethod
    def _extract_section(text: str, section_keyword: str) -> str | None:
        pattern = rf"(?:#{1,3}\s*.*?{section_keyword}.*?\n)(.*?)(?=#{1,3}|\Z)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_warnings(text: str) -> list[str]:
        warnings = []
        patterns = [
            r"(?:WARNING|CAUTION|NOTE):\s*(.+)",
            r"⚠️\s*(.+)",
            r"🚨\s*(.+)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                warnings.append(m.group(1).strip())
        return warnings

    # ── Convenience builders ──────────────────────────────────────────────────

    @classmethod
    def quick(cls) -> "Web3Pipeline":
        """Create pipeline with lightweight params for fast prototyping."""
        config = ModelConfig.from_env()
        config.generation = GenerationParams(
            max_new_tokens=1024, temperature=0.1, do_sample=False
        )
        return cls(ModelEngine(config))
