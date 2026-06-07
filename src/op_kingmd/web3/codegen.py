"""Web3 code generation layer — template library for common patterns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

from op_kingmd.core.pipeline import Framework, Language, Web3Pipeline

logger = structlog.get_logger(__name__)


class ContractType(str, Enum):
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    DEFI_VAULT = "defi_vault"
    DEX_PAIR = "dex_pair"
    GOVERNANCE = "governance"
    MULTISIG = "multisig"
    PROXY_UPGRADEABLE = "proxy_upgradeable"
    STAKING = "staking"
    BRIDGE = "bridge"
    ORACLE = "oracle"
    NFT_MARKETPLACE = "nft_marketplace"
    CUSTOM = "custom"


@dataclass
class ContractSpec:
    name: str
    contract_type: ContractType
    description: str
    features: list[str]
    owner_address: str | None = None
    network: str = "ethereum"
    solidity_version: str = "^0.8.24"
    framework: Framework = Framework.FOUNDRY
    include_tests: bool = True
    include_deploy: bool = True


class Web3CodeGen:
    """High-level factory for generating common Web3 contract patterns."""

    FEATURE_MAP: dict[ContractType, list[str]] = {
        ContractType.ERC20: [
            "Mintable by owner",
            "Burnable by token holder",
            "Pausable emergency stop",
            "Permit (EIP-2612) for gasless approvals",
            "Snapshot for governance voting",
            "Capped supply",
        ],
        ContractType.ERC721: [
            "IPFS/Arweave metadata URI",
            "Royalties (EIP-2981)",
            "Soulbound option",
            "Batch minting",
            "Allowlist / merkle proof minting",
            "Reveal mechanism",
        ],
        ContractType.ERC1155: [
            "Semi-fungible tokens",
            "Batch transfers",
            "URI per token ID",
            "Lazy minting",
        ],
        ContractType.DEFI_VAULT: [
            "ERC-4626 tokenized vault standard",
            "Strategy hooks",
            "Deposit/withdrawal fee",
            "Emergency exit",
            "Yield distributor",
        ],
        ContractType.STAKING: [
            "Configurable lock periods",
            "Reward rate per block",
            "Compound option",
            "Emergency unstake",
            "Boost multiplier (ve-token model)",
        ],
        ContractType.GOVERNANCE: [
            "OpenZeppelin Governor",
            "Timelock controller",
            "Snapshot voting",
            "Quorum fraction",
            "Proposal threshold",
        ],
        ContractType.MULTISIG: [
            "N-of-M threshold",
            "Owner management",
            "Timelock",
            "Daily spending limit",
        ],
    }

    def __init__(self, pipeline: Web3Pipeline | None = None) -> None:
        self._pipeline = pipeline or Web3Pipeline()

    # ── High-level generators ─────────────────────────────────────────────────

    def create_erc20(
        self,
        name: str,
        symbol: str,
        initial_supply: int = 1_000_000,
        features: list[str] | None = None,
        framework: Framework = Framework.FOUNDRY,
    ) -> dict[str, str]:
        """Generate a complete ERC-20 token with tests and deployment script."""
        feat_list = features or self.FEATURE_MAP[ContractType.ERC20]
        description = (
            f"ERC-20 token named '{name}' (symbol: {symbol}). "
            f"Initial supply: {initial_supply:,} tokens (18 decimals). "
            f"Features: {', '.join(feat_list)}. "
            "Use OpenZeppelin 5.x contracts. Gas-optimize all functions. "
            "Include NatSpec on every public function and state variable."
        )
        return self._generate_full_package(name, description, framework)

    def create_erc721(
        self,
        name: str,
        symbol: str,
        max_supply: int = 10_000,
        mint_price: float = 0.05,
        features: list[str] | None = None,
        framework: Framework = Framework.FOUNDRY,
    ) -> dict[str, str]:
        """Generate a complete NFT collection contract."""
        feat_list = features or self.FEATURE_MAP[ContractType.ERC721]
        description = (
            f"ERC-721 NFT collection '{name}' ({symbol}). "
            f"Max supply: {max_supply}. Mint price: {mint_price} ETH. "
            f"Features: {', '.join(feat_list)}. "
            "Use ERC721A for gas-efficient batch minting. "
            "Include a merkle-proof allowlist phase and public sale phase."
        )
        return self._generate_full_package(name, description, framework)

    def create_erc4626_vault(
        self,
        name: str,
        asset_address: str = "0x...",
        strategy_description: str = "Yield farming via Aave and Compound",
        framework: Framework = Framework.FOUNDRY,
    ) -> dict[str, str]:
        """Generate an ERC-4626 tokenized vault."""
        description = (
            f"ERC-4626 compliant yield vault named '{name}'. "
            f"Underlying asset: {asset_address}. "
            f"Strategy: {strategy_description}. "
            "Implement deposit, mint, withdraw, redeem with proper accounting. "
            "Include harvest function, performance fees (2% annualized), "
            "emergency pause, and slippage protection."
        )
        return self._generate_full_package(name, description, framework)

    def create_staking_contract(
        self,
        name: str,
        stake_token: str,
        reward_token: str,
        reward_rate: float = 0.1,
        framework: Framework = Framework.FOUNDRY,
    ) -> dict[str, str]:
        """Generate a staking contract with configurable rewards."""
        description = (
            f"Staking contract '{name}'. Stake token: {stake_token}. "
            f"Reward token: {reward_token}. Reward rate: {reward_rate} tokens/block. "
            "Use Synthetix StakingRewards pattern with compound functionality. "
            "Include lock periods, penalty for early withdrawal (10%), "
            "and emergency exit with forfeited rewards."
        )
        return self._generate_full_package(name, description, framework)

    def create_multisig(
        self,
        name: str = "MultiSig",
        owners: list[str] | None = None,
        threshold: int = 2,
        framework: Framework = Framework.FOUNDRY,
    ) -> dict[str, str]:
        """Generate a multisig wallet contract."""
        owners = owners or ["0xOwner1", "0xOwner2", "0xOwner3"]
        description = (
            f"Multisig wallet '{name}' with {len(owners)} owners, {threshold}-of-{len(owners)} threshold. "
            "Support ETH and ERC-20 transfers. "
            "Include proposal, confirmation, execution, and revocation flows. "
            "Add a 24-hour timelock for large transactions (>10 ETH). "
            "Emit events for all state changes."
        )
        return self._generate_full_package(name, description, framework)

    def create_dao(
        self,
        name: str,
        token_address: str = "0x...",
        voting_delay: int = 1,
        voting_period: int = 50400,
        quorum_percent: int = 4,
        framework: Framework = Framework.FOUNDRY,
    ) -> dict[str, str]:
        """Generate a full DAO governance system."""
        description = (
            f"DAO governance system '{name}' using OpenZeppelin Governor. "
            f"Governance token: {token_address}. "
            f"Voting delay: {voting_delay} blocks. Voting period: {voting_period} blocks. "
            f"Quorum: {quorum_percent}% of total supply. "
            "Include TimelockController with 2-day delay. "
            "Support proposal creation, voting, queuing, and execution. "
            "Add CANCEL_ROLE for guardian multisig."
        )
        return self._generate_full_package(name, description, framework)

    def generate_custom(
        self,
        description: str,
        language: Language = Language.SOLIDITY,
        framework: Framework = Framework.FOUNDRY,
    ) -> dict[str, str]:
        """Generate a custom contract from a natural language description."""
        return self._generate_full_package("Custom", description, framework, language)

    # ── DeFi protocol stubs ───────────────────────────────────────────────────

    def create_amm_pair(
        self,
        name: str = "UniswapV2Pair",
        fee_bps: int = 30,
        framework: Framework = Framework.FOUNDRY,
    ) -> dict[str, str]:
        """Generate a Uniswap V2-style AMM pair."""
        description = (
            f"AMM liquidity pair '{name}' implementing constant-product formula (x*y=k). "
            f"Swap fee: {fee_bps} bps ({fee_bps/100:.2f}%). "
            "Include: addLiquidity, removeLiquidity, swap functions. "
            "Mint/burn LP tokens. Implement price0CumulativeLast and price1CumulativeLast "
            "TWAP accumulators. Protect against flash loan price manipulation with reserves check. "
            "Emit Mint, Burn, Swap, Sync events."
        )
        return self._generate_full_package(name, description, framework)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_full_package(
        self,
        name: str,
        description: str,
        framework: Framework = Framework.FOUNDRY,
        language: Language = Language.SOLIDITY,
    ) -> dict[str, str]:
        from op_kingmd.core.pipeline import CodeGenRequest

        req = CodeGenRequest(
            description=description,
            language=language,
            framework=framework,
            include_tests=True,
            include_docs=True,
            include_deploy_script=True,
            gas_optimize=True,
            security_checks=True,
        )
        result = self._pipeline.generate_contract(req)

        package = {"contract": result.code}
        if result.tests:
            package["tests"] = result.tests
        if result.docs:
            package["docs"] = result.docs
        if result.deploy_script:
            package["deploy"] = result.deploy_script
        if result.warnings:
            package["warnings"] = "\n".join(result.warnings)

        logger.info(
            "package_generated",
            name=name,
            files=list(package.keys()),
            model=result.model_id,
        )
        return package
