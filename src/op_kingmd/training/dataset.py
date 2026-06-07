"""Dataset preparation for Web3 LLM fine-tuning.

Supports loading from Hugging Face Hub, local JSONL files, and GitHub repos.
Formats examples as chat-templated instruction pairs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

logger = structlog.get_logger(__name__)


@dataclass
class DatasetConfig:
    # HuggingFace dataset IDs to include
    hf_datasets: list[dict[str, Any]] = None  # [{"path": ..., "split": ..., "subset": ...}]
    local_files: list[str] = None             # paths to .jsonl files
    max_examples: int | None = None
    val_split_ratio: float = 0.05
    seed: int = 42

    def __post_init__(self):
        if self.hf_datasets is None:
            self.hf_datasets = []
        if self.local_files is None:
            self.local_files = []


# Curated Web3/Solidity datasets on Hugging Face
WEB3_DATASET_REGISTRY = [
    {
        "path": "bigcode/the-stack-dedup",
        "data_dir": "data/solidity",
        "split": "train",
        "text_field": "content",
        "description": "Solidity files from The Stack v1",
    },
    {
        "path": "bigcode/starcoderdata",
        "data_dir": "solidity",
        "split": "train",
        "text_field": "content",
        "description": "StarCoder training data — Solidity",
    },
    {
        "path": "iamtarun/python_code_instructions_18k_alpaca",
        "split": "train",
        "text_field": "output",
        "description": "Instruction-following code dataset (for formatting)",
    },
]

INSTRUCTION_TEMPLATE = """<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
{response}
<|im_end|>"""

SYSTEM_PROMPT = (
    "You are Op-Kingmd, a top-tier Web3 coding assistant. "
    "Generate production-ready, secure, gas-optimized smart contracts."
)


class Web3Dataset:
    """Loads and preprocesses Web3 training data for instruction fine-tuning."""

    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()

    def load(self, tokenizer: Any) -> DatasetDict:
        """Load all sources, format, tokenize, and return train/val split."""
        all_datasets = []

        for ds_config in self.config.hf_datasets:
            ds = self._load_hf_dataset(ds_config)
            if ds:
                all_datasets.append(ds)

        for path in self.config.local_files:
            ds = self._load_local_jsonl(path)
            if ds:
                all_datasets.append(ds)

        if not all_datasets:
            raise ValueError("No datasets loaded. Check your DatasetConfig.")

        combined = concatenate_datasets(all_datasets)
        if self.config.max_examples:
            combined = combined.shuffle(seed=self.config.seed).select(
                range(min(self.config.max_examples, len(combined)))
            )

        formatted = combined.map(
            self._format_example,
            remove_columns=combined.column_names,
            desc="Formatting examples",
        )

        tokenized = formatted.map(
            lambda x: self._tokenize(x, tokenizer),
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing",
        )

        split = tokenized.train_test_split(
            test_size=self.config.val_split_ratio, seed=self.config.seed
        )
        logger.info(
            "dataset_ready",
            train=len(split["train"]),
            val=len(split["test"]),
        )
        return DatasetDict({"train": split["train"], "validation": split["test"]})

    def _load_hf_dataset(self, ds_config: dict) -> Dataset | None:
        try:
            kwargs = {k: v for k, v in ds_config.items() if k in ("data_dir", "split")}
            ds = load_dataset(ds_config["path"], **kwargs, trust_remote_code=True)
            if isinstance(ds, DatasetDict):
                ds = ds[ds_config.get("split", "train")]
            text_field = ds_config.get("text_field", "content")
            if text_field not in ds.column_names:
                logger.warning("missing_text_field", field=text_field, dataset=ds_config["path"])
                return None
            return ds.rename_column(text_field, "raw_text")
        except Exception as exc:
            logger.warning("hf_dataset_load_failed", dataset=ds_config.get("path"), error=str(exc))
            return None

    def _load_local_jsonl(self, path: str) -> Dataset | None:
        p = Path(path)
        if not p.exists():
            logger.warning("local_file_not_found", path=path)
            return None
        rows = []
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    rows.append(obj)
        if not rows:
            return None
        return Dataset.from_list(rows)

    @staticmethod
    def _format_example(example: dict) -> dict:
        instruction = example.get("instruction") or example.get("prompt", "")
        response = (
            example.get("output")
            or example.get("response")
            or example.get("completion")
            or example.get("raw_text", "")
        )
        text = INSTRUCTION_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            instruction=instruction or "Complete the following Solidity code:",
            response=response,
        )
        return {"text": text}

    @staticmethod
    def _tokenize(batch: dict, tokenizer: Any) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=4096,
            padding=False,
        )

    @classmethod
    def from_solidity_files(cls, directory: str) -> "Web3Dataset":
        """Convenience: scan a directory of .sol files and create a dataset."""
        files = list(Path(directory).rglob("*.sol"))
        rows = []
        for f in files:
            try:
                source = f.read_text(encoding="utf-8", errors="ignore")
                rows.append({"raw_text": source, "source_file": str(f)})
            except Exception:
                pass
        logger.info("loaded_sol_files", count=len(rows), directory=directory)
        config = DatasetConfig()
        instance = cls(config)
        instance._cached_rows = rows
        return instance
