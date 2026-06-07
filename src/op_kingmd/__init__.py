"""Op-Kingmd — Web3-focused coding LLM for smart contract development."""

__version__ = "0.1.0"
__author__ = "Op-Kingmd Team"
__license__ = "MIT"

from op_kingmd.core.model import ModelEngine
from op_kingmd.core.pipeline import Web3Pipeline

__all__ = ["ModelEngine", "Web3Pipeline", "__version__"]
