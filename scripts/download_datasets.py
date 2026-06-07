"""Download and prepare Web3 training datasets for fine-tuning.

Run:
    python scripts/download_datasets.py --output ./data/train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Prepare Web3 fine-tuning dataset")
    p.add_argument("--output", default="./data/train.jsonl")
    p.add_argument("--max-samples", type=int, default=50_000)
    p.add_argument("--include-the-stack", action="store_true",
                   help="Include bigcode/the-stack-dedup Solidity subset (large)")
    return p.parse_args()


SEED_EXAMPLES = [
    {
        "instruction": "Write a gas-optimized ERC-20 token with mint and burn.",
        "output": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/// @title OptimizedToken
/// @notice Gas-optimized ERC-20 with capped mint and burn
contract OptimizedToken is ERC20, Ownable {
    uint256 public immutable maxSupply;

    error ExceedsMaxSupply();

    event Minted(address indexed to, uint256 amount);
    event Burned(address indexed from, uint256 amount);

    constructor(uint256 _maxSupply, address _owner)
        ERC20("OptimizedToken", "OPT")
        Ownable(_owner)
    {
        maxSupply = _maxSupply;
    }

    function mint(address to, uint256 amount) external onlyOwner {
        if (totalSupply() + amount > maxSupply) revert ExceedsMaxSupply();
        _mint(to, amount);
        emit Minted(to, amount);
    }

    function burn(uint256 amount) external {
        _burn(msg.sender, amount);
        emit Burned(msg.sender, amount);
    }
}
""",
    },
    {
        "instruction": "Write a reentrancy-safe withdrawal pattern in Solidity.",
        "output": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/// @notice Safe withdrawal vault using Checks-Effects-Interactions + ReentrancyGuard
contract SafeVault is ReentrancyGuard {
    mapping(address => uint256) private _balances;

    error InsufficientBalance();
    error TransferFailed();

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);

    function deposit() external payable {
        _balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external nonReentrant {
        if (_balances[msg.sender] < amount) revert InsufficientBalance();
        // Checks-Effects-Interactions: update state BEFORE external call
        _balances[msg.sender] -= amount;
        emit Withdrawn(msg.sender, amount);
        (bool ok,) = msg.sender.call{value: amount}("");
        if (!ok) revert TransferFailed();
    }

    function balanceOf(address user) external view returns (uint256) {
        return _balances[user];
    }
}
""",
    },
    {
        "instruction": "Write a Foundry test for an ERC-20 token.",
        "output": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test, console} from "forge-std/Test.sol";
import {SimpleToken} from "../src/SimpleToken.sol";

contract SimpleTokenTest is Test {
    SimpleToken token;
    address alice = makeAddr("alice");
    address bob   = makeAddr("bob");
    uint256 INITIAL_SUPPLY = 1_000_000e18;

    function setUp() public {
        token = new SimpleToken(address(this));
    }

    function test_InitialSupply() public view {
        assertEq(token.totalSupply(), INITIAL_SUPPLY);
        assertEq(token.balanceOf(address(this)), INITIAL_SUPPLY);
    }

    function test_Transfer() public {
        token.transfer(alice, 1000e18);
        assertEq(token.balanceOf(alice), 1000e18);
    }

    function test_RevertWhen_TransferExceedsBalance() public {
        vm.expectRevert();
        token.transfer(alice, INITIAL_SUPPLY + 1);
    }

    function testFuzz_Transfer(uint256 amount) public {
        amount = bound(amount, 1, INITIAL_SUPPLY);
        token.transfer(alice, amount);
        assertEq(token.balanceOf(alice), amount);
    }
}
""",
    },
]


def main():
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = list(SEED_EXAMPLES)

    # Optionally pull from HuggingFace
    if args.include_the_stack:
        try:
            from datasets import load_dataset
            console.print("[cyan]Loading The Stack (Solidity)...[/cyan]")
            ds = load_dataset(
                "bigcode/the-stack-dedup",
                data_dir="data/solidity",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            for i, ex in enumerate(ds):
                if i >= args.max_samples:
                    break
                rows.append({
                    "instruction": "Complete or explain the following Solidity code:",
                    "output": ex.get("content", ""),
                })
            console.print(f"[green]Loaded {len(rows) - len(SEED_EXAMPLES)} examples from The Stack[/green]")
        except Exception as e:
            console.print(f"[yellow]Skipping The Stack: {e}[/yellow]")

    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    console.print(f"[green]✓ Wrote {len(rows)} examples to {out}[/green]")
    console.print("\nTo start fine-tuning:")
    console.print(f"  op-kingmd train run --data {out}")


if __name__ == "__main__":
    main()
