"""Hugging Face model loading, quantization, and inference engine."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    TextIteratorStreamer,
)

logger = structlog.get_logger(__name__)

DEFAULT_MODEL_ID = os.getenv("OPKMD_MODEL_ID", "Qwen/Qwen2.5-Coder-7B-Instruct")

SYSTEM_PROMPT = """You are Op-Kingmd, a top-tier Web3 coding assistant specializing in smart contract \
development, DeFi protocols, NFT systems, and blockchain infrastructure. \
You produce production-ready, gas-optimized, and security-hardened code. \
You understand Solidity, Rust (Anchor/Substrate), TypeScript (ethers.js/viem/wagmi), \
Foundry, Hardhat, and all major EVM-compatible chains. \
Always include natspec documentation, comprehensive tests, and deployment scripts. \
When you detect potential security vulnerabilities, call them out explicitly."""


@dataclass
class GenerationParams:
    max_new_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.05
    do_sample: bool = True


@dataclass
class ModelConfig:
    model_id: str = DEFAULT_MODEL_ID
    adapter_path: str | None = None
    quantize_4bit: bool = True
    force_cpu: bool = False
    device_map: str = "auto"
    max_seq_length: int = 16384
    generation: GenerationParams = field(default_factory=GenerationParams)
    hf_token: str | None = None

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            model_id=os.getenv("OPKMD_MODEL_ID", DEFAULT_MODEL_ID),
            adapter_path=os.getenv("OPKMD_ADAPTER_PATH") or None,
            quantize_4bit=os.getenv("OPKMD_QUANTIZE_4BIT", "1") == "1",
            force_cpu=os.getenv("OPKMD_FORCE_CPU", "0") == "1",
            hf_token=os.getenv("HF_TOKEN") or None,
        )


class ModelEngine:
    """Wraps a Hugging Face causal-LM for Web3 code generation.

    Supports:
    - 4-bit QLoRA quantization via BitsAndBytes
    - LoRA adapter hot-swapping via PEFT
    - Streaming token generation
    - Multi-GPU via device_map="auto"
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_env()
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> "ModelEngine":
        """Load model + tokenizer. Returns self for chaining."""
        if self._loaded:
            return self
        t0 = time.perf_counter()
        logger.info("loading_model", model_id=self.config.model_id)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            token=self.config.hf_token,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        bnb_config = self._build_bnb_config() if self.config.quantize_4bit else None
        device_map = "cpu" if self.config.force_cpu else self.config.device_map

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            token=self.config.hf_token,
            torch_dtype=torch.bfloat16 if not self.config.quantize_4bit else None,
            attn_implementation="flash_attention_2" if self._has_flash_attn() else "eager",
        )
        self._model.eval()

        if self.config.adapter_path:
            self._load_adapter(self.config.adapter_path)

        elapsed = time.perf_counter() - t0
        logger.info("model_loaded", elapsed_s=round(elapsed, 2), device=str(device_map))
        self._loaded = True
        return self

    def _build_bnb_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    def _load_adapter(self, adapter_path: str) -> None:
        from peft import PeftModel

        logger.info("loading_adapter", path=adapter_path)
        self._model = PeftModel.from_pretrained(self._model, adapter_path)
        self._model = self._model.merge_and_unload()

    @staticmethod
    def _has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _build_prompt(self, user_message: str, system: str | None = None) -> str:
        """Format messages using the model's chat template."""
        messages = [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        params: GenerationParams | None = None,
    ) -> str:
        """Generate a completion for the given prompt."""
        self._ensure_loaded()
        p = params or self.config.generation
        full_prompt = self._build_prompt(prompt, system)

        inputs = self._tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length - p.max_new_tokens,
        ).to(self._model.device)

        gen_config = GenerationConfig(
            max_new_tokens=p.max_new_tokens,
            temperature=p.temperature,
            top_p=p.top_p,
            top_k=p.top_k,
            repetition_penalty=p.repetition_penalty,
            do_sample=p.do_sample,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, generation_config=gen_config)

        new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        params: GenerationParams | None = None,
    ):
        """Yield tokens as a generator for streaming responses."""
        import threading

        self._ensure_loaded()
        p = params or self.config.generation
        full_prompt = self._build_prompt(prompt, system)

        inputs = self._tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length - p.max_new_tokens,
        ).to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_config = GenerationConfig(
            max_new_tokens=p.max_new_tokens,
            temperature=p.temperature,
            top_p=p.top_p,
            do_sample=p.do_sample,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        thread = threading.Thread(
            target=self._model.generate,
            kwargs={**inputs, "generation_config": gen_config, "streamer": streamer},
        )
        thread.start()
        for token in streamer:
            yield token
        thread.join()

    # ── Utilities ─────────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def get_device(self) -> str:
        if self._model is None:
            return "not_loaded"
        try:
            return str(next(self._model.parameters()).device)
        except StopIteration:
            return "unknown"

    def push_to_hub(self, repo_id: str, private: bool = True) -> None:
        """Push fine-tuned model to Hugging Face Hub."""
        self._ensure_loaded()
        logger.info("pushing_to_hub", repo_id=repo_id)
        self._model.push_to_hub(repo_id, private=private, token=self.config.hf_token)
        self._tokenizer.push_to_hub(repo_id, private=private, token=self.config.hf_token)
