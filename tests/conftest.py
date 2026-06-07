"""Shared test fixtures."""

from __future__ import annotations

import pytest

SIMPLE_ERC20 = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract SimpleToken is ERC20, Ownable {
    uint256 public constant MAX_SUPPLY = 1_000_000 * 10**18;

    constructor(address initialOwner)
        ERC20("SimpleToken", "STK")
        Ownable(initialOwner)
    {
        _mint(initialOwner, MAX_SUPPLY);
    }

    function mint(address to, uint256 amount) external onlyOwner {
        require(totalSupply() + amount <= MAX_SUPPLY, "Exceeds max supply");
        _mint(to, amount);
    }
}
"""

REENTRANCY_VULNERABLE = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Vulnerable {
    mapping(address => uint256) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw() external {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "No balance");
        // VULNERABLE: external call before state update
        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok, "Transfer failed");
        balances[msg.sender] = 0;
    }
}
"""

TX_ORIGIN_VULNERABLE = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TxOriginVuln {
    address public owner;

    function isOwner() public view returns (bool) {
        return tx.origin == owner;
    }

    function transfer(address to, uint256 amount) external {
        require(tx.origin == owner, "Not owner");
        payable(to).transfer(amount);
    }
}
"""

SELFDESTRUCT_VULNERABLE = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract KillSwitch {
    function destroy() external {
        selfdestruct(payable(msg.sender));
    }
}
"""


@pytest.fixture
def simple_erc20_source():
    return SIMPLE_ERC20


@pytest.fixture
def reentrancy_source():
    return REENTRANCY_VULNERABLE


@pytest.fixture
def tx_origin_source():
    return TX_ORIGIN_VULNERABLE


@pytest.fixture
def selfdestruct_source():
    return SELFDESTRUCT_VULNERABLE


@pytest.fixture
def audit_engine():
    from op_kingmd.audit.analyzer import AuditEngine
    return AuditEngine()
