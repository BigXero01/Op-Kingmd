# Op-Kingmd

**Top-5% lightweight Web3 coding LLM** — vibe coding, rapid prototyping, and production-grade smart contract development powered by the Hugging Face ecosystem.

---

## What It Does

| Capability | Details |
|---|---|
| **Code Generation** | Solidity, Rust (Anchor), TypeScript — contracts, tests, scripts from natural language |
| **Audit-Before-Deploy** | Mandatory security pipeline: vulnerability detection → gas optimization → risk scoring → deploy decision |
| **GitHub Integration** | Repo analysis, issue generation, PR security review, code search, automated commits |
| **Model Fine-Tuning** | LoRA/QLoRA training on Web3 data via Hugging Face PEFT + TRL |
| **REST API** | FastAPI server for all capabilities |
| **CLI** | Rich terminal interface — `op-kingmd generate`, `audit`, `github`, `serve`, `train` |

---

## Architecture

```
Op-Kingmd
├── src/op_kingmd/
│   ├── core/           # HF model engine + Web3 inference pipeline
│   ├── training/       # QLoRA fine-tuning (PEFT + TRL SFTTrainer)
│   ├── web3/           # Codegen, Solidity tools, Foundry/Hardhat runners
│   ├── github/         # GitHub client, repo analyzer, PR reviewer
│   ├── audit/          # Audit-Before-Deploy: static analysis + gas + risk score
│   ├── api/            # FastAPI REST server
│   └── cli/            # Typer CLI with Rich output
├── config/             # model_config.yaml, audit_config.yaml
├── tests/              # Full pytest suite
├── examples/           # Runnable end-to-end examples
└── scripts/            # Model download, dataset preparation
```

---

## Base Model

**Default:** [`Qwen/Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- 7B parameters, Apache-2.0 license, 2.2M+ downloads
- Best-in-class code quality at the 7B scale
- Native 4-bit QLoRA quantization (~6GB VRAM)
- Configurable: swap to DeepSeek-Coder-V2-Lite, Qwen2.5-Coder-32B, etc.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/bigxero01/op-kingmd
cd op-kingmd
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Fill in: HF_TOKEN, GITHUB_TOKEN, ETH_RPC_URL
```

### 3. Download Model

```bash
python scripts/setup_model.py
```

### 4. Generate Your First Contract

```bash
# Generate an ERC-20 token
op-kingmd generate erc20 "MyToken" MTK --supply 10000000 --output ./contracts/

# Generate from natural language
op-kingmd generate contract "A staking contract where users stake ETH and earn USDC rewards" \
  --framework foundry --output ./contracts/

# Full package: contract + tests + deployment script
op-kingmd generate erc721 "CoolNFT" CNFT --supply 5000 --price 0.08 --output ./nft/
```

### 5. Audit Before Deploying

```bash
# Audit a single file — blocks with exit code 1 if unsafe
op-kingmd audit run ./contracts/MyToken.sol --chain ethereum --network mainnet

# Quick inline audit
op-kingmd audit quick ./contracts/MyToken.sol

# Generates: JSON + Markdown + HTML reports in ./audit_reports/
```

### 6. GitHub Integration

```bash
export GITHUB_TOKEN=ghp_...

# Analyze a repository
op-kingmd github analyze owner/repo

# AI review of a pull request
op-kingmd github review-pr owner/repo 42 --post
```

### 7. Start the API Server

```bash
op-kingmd serve --port 8000
# Docs: http://localhost:8000/docs
```

---

## API Reference

```bash
# Generate contract
curl -X POST http://localhost:8000/v1/codegen/generate \
  -H "Content-Type: application/json" \
  -d '{"description": "ERC-20 token with permit and snapshot", "language": "solidity"}'

# Run audit
curl -X POST http://localhost:8000/v1/audit/source \
  -H "Content-Type: application/json" \
  -d '{"source": "pragma solidity ^0.8.24; contract Foo { ... }", "chain": "ethereum"}'

# Analyze GitHub repo
curl -X POST http://localhost:8000/v1/github/analyze \
  -H "Content-Type: application/json" \
  -d '{"repo": "owner/repo"}'
```

---

## Audit-Before-Deploy Layer

Every deployment passes through a 5-stage mandatory pipeline:

```
Source Code
    │
    ▼
[Stage 1] Static Pattern Analysis (15 vulnerability patterns)
    │      reentrancy, tx.origin, selfdestruct, delegatecall,
    │      oracle manipulation, unprotected initializer, ...
    ▼
[Stage 2] Slither Integration (optional, auto-detected)
    │      50+ built-in Slither detectors
    ▼
[Stage 3] Gas Optimization Analysis
    │      custom errors, unchecked loops, immutable variables,
    │      packed storage, cached array length, ...
    ▼
[Stage 4] Dependency Scanning
    │      checks npm/foundry packages against known-vulnerable versions
    ▼
[Stage 5] Risk Scoring  (0-100)
    │
    ├── Score ≥ 75 → LOW_RISK     → ✅ DEPLOY ALLOWED
    ├── Score ≥ 60 → MEDIUM_RISK  → ⚠️  DEPLOY ALLOWED (with warnings)
    └── Score <  60 → HIGH/CRITICAL_RISK → ⛔ DEPLOY BLOCKED
```

Output: JSON + Markdown + HTML reports in `./audit_reports/`.

---

## Fine-Tuning

```bash
# 1. Prepare data
python scripts/download_datasets.py --output data/train.jsonl

# 2. Fine-tune (requires GPU with ≥16GB VRAM for 7B)
op-kingmd train run \
  --config config/model_config.yaml \
  --data data/train.jsonl

# 3. Push to Hugging Face
op-kingmd train run --push-to-hub your-org/op-kingmd-v1
```

Training uses:
- **QLoRA** (4-bit NF4, r=64, alpha=128) via PEFT
- **SFTTrainer** from TRL with packing
- `paged_adamw_8bit` optimizer
- Cosine LR scheduler + gradient checkpointing

---

## Supported Frameworks

| Framework | Build | Test | Deploy |
|---|---|---|---|
| Foundry | `forge build` | `forge test` | `forge script` |
| Hardhat | `npx hardhat compile` | `npx hardhat test` | `npx hardhat run` |
| Anchor (Solana) | Code generation only | Template tests | — |

---

## Supported Languages

| Language | Use Case |
|---|---|
| Solidity | EVM smart contracts (primary) |
| Rust | Solana Anchor programs |
| TypeScript | ethers.js, viem, wagmi, Hardhat scripts |
| Vyper | Alternative EVM language |
| Move | Aptos / Sui contracts |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v                           # all tests
pytest tests/test_audit.py -v       # audit layer only
pytest tests/test_web3.py -v        # Web3 tools only
pytest --cov=op_kingmd              # with coverage
```

---

## Project Structure

```
Op-Kingmd/
├── src/op_kingmd/
│   ├── core/
│   │   ├── model.py       # HF model loading, 4-bit quant, LoRA adapter hot-swap
│   │   └── pipeline.py    # Prompt templates, response parsing, streaming
│   ├── training/
│   │   ├── dataset.py     # Multi-source dataset loader + formatter
│   │   └── finetune.py    # QLoRA trainer (PEFT + TRL)
│   ├── web3/
│   │   ├── codegen.py     # High-level contract factory (ERC20, ERC721, DAO, AMM…)
│   │   ├── solidity.py    # ABI parsing, selector computation, compilation
│   │   ├── foundry.py     # forge CLI wrapper
│   │   └── hardhat.py     # npx hardhat CLI wrapper
│   ├── github/
│   │   ├── client.py      # PyGithub wrapper with Web3-aware helpers
│   │   ├── analyzer.py    # Framework/chain detection, complexity scoring
│   │   └── pr_review.py   # Diff-level security review, comment generation
│   ├── audit/
│   │   ├── analyzer.py    # 5-stage audit pipeline, risk scoring
│   │   └── report.py      # JSON/Markdown/HTML report generation
│   ├── api/
│   │   ├── server.py      # FastAPI app with lifespan model loading
│   │   └── routes/        # /codegen, /audit, /github endpoints
│   └── cli/
│       └── main.py        # Typer CLI with Rich terminal output
├── config/
│   ├── model_config.yaml  # Model ID, LoRA params, generation settings
│   └── audit_config.yaml  # Vulnerability patterns, severity weights, gas rules
├── tests/                 # pytest suite (no GPU required — mocked)
├── examples/              # End-to-end runnable examples
└── scripts/               # setup_model.py, download_datasets.py
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built with [Hugging Face Transformers](https://huggingface.co/docs/transformers), [PEFT](https://huggingface.co/docs/peft), [TRL](https://huggingface.co/docs/trl), [FastAPI](https://fastapi.tiangolo.com/), and [Typer](https://typer.tiangolo.com/).*
