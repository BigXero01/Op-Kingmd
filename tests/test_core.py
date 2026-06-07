"""Tests for core model and pipeline (mocked to avoid loading weights)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from op_kingmd.core.model import GenerationParams, ModelConfig, ModelEngine
from op_kingmd.core.pipeline import CodeGenRequest, Framework, Language, Web3Pipeline


class TestModelConfig:
    def test_defaults(self):
        config = ModelConfig()
        assert config.model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert config.quantize_4bit is True
        assert config.force_cpu is False
        assert config.max_seq_length == 16384

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("OPKMD_MODEL_ID", "Qwen/Qwen2.5-Coder-7B-Instruct")
        monkeypatch.setenv("OPKMD_QUANTIZE_4BIT", "0")
        config = ModelConfig.from_env()
        assert config.model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert config.quantize_4bit is False

    def test_generation_defaults(self):
        params = GenerationParams()
        assert params.temperature == 0.2
        assert params.max_new_tokens == 4096
        assert params.top_p == 0.95


class TestModelEngine:
    def test_model_id_property(self):
        config = ModelConfig(model_id="test/model")
        engine = ModelEngine(config)
        assert engine.model_id == "test/model"

    def test_not_loaded_initially(self):
        engine = ModelEngine()
        assert not engine._loaded

    def test_get_device_not_loaded(self):
        engine = ModelEngine()
        assert engine.get_device() == "not_loaded"

    @patch("op_kingmd.core.model.AutoModelForCausalLM")
    @patch("op_kingmd.core.model.AutoTokenizer")
    def test_load_sets_loaded_flag(self, mock_tokenizer_cls, mock_model_cls):
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tok

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        config = ModelConfig(quantize_4bit=False, force_cpu=True)
        engine = ModelEngine(config)
        engine.load()
        assert engine._loaded

    @patch("op_kingmd.core.model.AutoModelForCausalLM")
    @patch("op_kingmd.core.model.AutoTokenizer")
    def test_generate_returns_string(self, mock_tokenizer_cls, mock_model_cls):
        import torch
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.decode.return_value = "contract Foo {}"
        mock_tok.apply_chat_template.return_value = "<prompt>"
        mock_tok.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        mock_tokenizer_cls.from_pretrained.return_value = mock_tok

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = torch.zeros(1, 20, dtype=torch.long)
        mock_model_cls.from_pretrained.return_value = mock_model

        config = ModelConfig(quantize_4bit=False, force_cpu=True)
        engine = ModelEngine(config)
        result = engine.generate("Write an ERC-20 contract")
        assert isinstance(result, str)


class TestWeb3Pipeline:
    def test_extract_code_block_solidity(self):
        text = "Here is the code:\n```solidity\ncontract Foo {}\n```"
        code = Web3Pipeline._extract_code_block(text, "solidity")
        assert code == "contract Foo {}"

    def test_extract_code_block_fallback(self):
        text = "```\ncontract Foo {}\n```"
        code = Web3Pipeline._extract_code_block(text, "solidity")
        assert code == "contract Foo {}"

    def test_extract_code_block_none_if_missing(self):
        text = "No code here."
        code = Web3Pipeline._extract_code_block(text, "solidity")
        assert code is None

    def test_extract_warnings(self):
        text = "WARNING: This function uses delegatecall.\nSome other text.\n⚠️ Reentrancy risk."
        warnings = Web3Pipeline._extract_warnings(text)
        assert len(warnings) >= 1

    def test_code_gen_request_defaults(self):
        req = CodeGenRequest(description="A simple token")
        assert req.language == Language.SOLIDITY
        assert req.framework == Framework.FOUNDRY
        assert req.include_tests is True

    @patch.object(ModelEngine, "generate", return_value="```solidity\ncontract Foo {}\n```")
    @patch.object(ModelEngine, "load", return_value=None)
    def test_pipeline_generate_contract(self, mock_load, mock_generate):
        pipeline = Web3Pipeline()
        pipeline._engine._loaded = True
        req = CodeGenRequest(description="A simple token contract")
        result = pipeline.generate_contract(req)
        assert result.code == "contract Foo {}"
        assert result.language == Language.SOLIDITY
