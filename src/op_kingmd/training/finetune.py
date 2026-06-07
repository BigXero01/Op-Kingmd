"""LoRA/QLoRA fine-tuning pipeline using PEFT + TRL SFTTrainer.

Usage:
    trainer = LoRATrainer.from_config_file("config/model_config.yaml")
    trainer.train(dataset)
    trainer.save("./checkpoints/op-kingmd-v1")
    trainer.push_to_hub("your-hf-org/op-kingmd-v1")
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog
import torch
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

logger = structlog.get_logger(__name__)


@dataclass
class LoRAConfig:
    r: int = 64
    lora_alpha: int = 128
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )
    lora_dropout: float = 0.05
    bias: str = "none"


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_seq_length: int = 4096
    logging_steps: int = 25
    save_steps: int = 200
    eval_steps: int = 200
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    report_to: str = "wandb"
    dataloader_num_workers: int = 4
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @classmethod
    def from_env(cls) -> "TrainConfig":
        return cls(
            model_id=os.getenv("OPKMD_MODEL_ID", "Qwen/Qwen2.5-Coder-7B-Instruct"),
            output_dir=os.getenv("OPKMD_TRAIN_OUTPUT_DIR", "./checkpoints"),
            num_train_epochs=int(os.getenv("OPKMD_TRAIN_EPOCHS", "3")),
            per_device_train_batch_size=int(os.getenv("OPKMD_TRAIN_BATCH_SIZE", "4")),
            learning_rate=float(os.getenv("OPKMD_TRAIN_LR", "2e-4")),
        )


class LoRATrainer:
    """Fine-tunes a causal LM using QLoRA for Web3 code generation."""

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or TrainConfig.from_env()
        self._model: Any = None
        self._tokenizer: Any = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self) -> "LoRATrainer":
        cfg = self.config
        logger.info("trainer_setup", model_id=cfg.model_id)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id,
            trust_remote_code=True,
            padding_side="right",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )

        lora_cfg = cfg.lora
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        self._model = get_peft_model(base_model, peft_config)
        self._model.print_trainable_parameters()
        return self

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, dataset: DatasetDict) -> None:
        if self._model is None:
            self.setup()

        cfg = self.config
        sft_config = SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            max_seq_length=cfg.max_seq_length,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            gradient_checkpointing=cfg.gradient_checkpointing,
            optim=cfg.optim,
            report_to=cfg.report_to,
            dataloader_num_workers=cfg.dataloader_num_workers,
            dataset_text_field="text",
            packing=True,                    # pack short sequences for efficiency
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        trainer = SFTTrainer(
            model=self._model,
            args=sft_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            tokenizer=self._tokenizer,
        )

        logger.info("training_start", steps=trainer.args.max_steps or "auto")
        train_result = trainer.train()
        logger.info("training_done", **train_result.metrics)

        trainer.save_model(cfg.output_dir)
        self._tokenizer.save_pretrained(cfg.output_dir)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("No model to save — call train() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(out)
        self._tokenizer.save_pretrained(out)
        logger.info("model_saved", path=str(out))

    def push_to_hub(self, repo_id: str, private: bool = True) -> None:
        if self._model is None:
            raise RuntimeError("No model to push — call train() first.")
        hf_token = os.getenv("HF_TOKEN")
        self._model.push_to_hub(repo_id, private=private, token=hf_token)
        self._tokenizer.push_to_hub(repo_id, private=private, token=hf_token)
        logger.info("pushed_to_hub", repo_id=repo_id)

    @classmethod
    def from_config_file(cls, path: str) -> "LoRATrainer":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        model_cfg = raw.get("model", {})
        lora_raw = raw.get("lora", {})
        lora = LoRAConfig(
            r=lora_raw.get("r", 64),
            lora_alpha=lora_raw.get("lora_alpha", 128),
            target_modules=lora_raw.get("target_modules", LoRAConfig().target_modules),
        )
        train_cfg = TrainConfig(
            model_id=model_cfg.get("default_id", "Qwen/Qwen2.5-Coder-7B-Instruct"),
            lora=lora,
        )
        return cls(train_cfg)
