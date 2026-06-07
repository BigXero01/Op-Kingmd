"""Tests for Web3 tools: Solidity analysis, Foundry/Hardhat runners."""

from __future__ import annotations

import pytest

from op_kingmd.web3.solidity import SolidityTools


SAMPLE_SOURCE = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

event Transfer(address indexed from, address indexed to, uint256 value);
error InsufficientBalance(uint256 available, uint256 required);

contract MyToken is ERC20 {
    uint256 public constant MAX_SUPPLY = 1_000_000e18;

    constructor() ERC20("MyToken", "MTK") {
        _mint(msg.sender, MAX_SUPPLY);
    }

    function burn(uint256 amount) external {
        _burn(msg.sender, amount);
    }

    function batchTransfer(address[] calldata recipients, uint256 amount) external {
        for (uint256 i = 0; i < recipients.length; i++) {
            transfer(recipients[i], amount);
        }
    }
}
"""


class TestSolidityTools:
    def test_extract_pragma(self):
        pragma = SolidityTools.extract_pragma(SAMPLE_SOURCE)
        assert pragma == "^0.8.24"

    def test_extract_contracts(self):
        contracts = SolidityTools.extract_contracts(SAMPLE_SOURCE)
        assert "MyToken" in contracts

    def test_extract_functions(self):
        funcs = SolidityTools.extract_functions(SAMPLE_SOURCE)
        func_names = [f["name"] for f in funcs]
        assert "burn" in func_names or "batchTransfer" in func_names

    def test_extract_events(self):
        events = SolidityTools.extract_events(SAMPLE_SOURCE)
        assert any(e["name"] == "Transfer" for e in events)

    def test_extract_imports(self):
        imports = SolidityTools.extract_imports(SAMPLE_SOURCE)
        assert any("ERC20" in imp for imp in imports)

    def test_get_function_selectors_from_abi(self):
        abi = [
            {
                "type": "function",
                "name": "transfer",
                "inputs": [
                    {"name": "to", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                ],
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
            }
        ]
        selectors = SolidityTools.get_function_selectors(abi)
        assert len(selectors) == 1
        assert selectors[0].name == "transfer"
        assert selectors[0].signature == "transfer(address,uint256)"
        assert selectors[0].selector.startswith("0x")
        assert len(selectors[0].selector) == 10  # 0x + 8 hex chars

    def test_extract_pragma_none_for_missing(self):
        pragma = SolidityTools.extract_pragma("contract Foo {}")
        assert pragma is None

    def test_extract_storage_layout(self):
        source = """
        contract Storage {
            uint256 public totalSupply;
            address private owner;
            bool internal paused;
        }
        """
        layout = SolidityTools.extract_storage_layout(source)
        # Should find some state variables
        assert isinstance(layout, list)


class TestFoundryRunner:
    def test_is_available_returns_bool(self):
        from op_kingmd.web3.foundry import FoundryRunner
        result = FoundryRunner.is_available()
        assert isinstance(result, bool)

    def test_runner_not_found_graceful(self, tmp_path):
        from op_kingmd.web3.foundry import FoundryRunner
        runner = FoundryRunner(project_root=tmp_path)
        # Should return a ForgeResult even if forge is not installed
        result = runner.build()
        assert hasattr(result, "success")
        assert hasattr(result, "stdout")
        assert hasattr(result, "stderr")


class TestHardhatRunner:
    def test_is_available_returns_bool(self):
        from op_kingmd.web3.hardhat import HardhatRunner
        result = HardhatRunner.is_available()
        assert isinstance(result, bool)

    def test_generate_config(self):
        from op_kingmd.web3.hardhat import HardhatRunner
        runner = HardhatRunner()
        config = runner.generate_config()
        assert "HardhatUserConfig" in config
        assert "0.8.24" in config
        assert "networks" in config
