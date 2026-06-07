"""Download and verify the base model from Hugging Face Hub.

Run:
    python scripts/setup_model.py [--model Qwen/Qwen2.5-Coder-7B-Instruct]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from rich.console import Console

console = Console()

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def parse_args():
    p = argparse.ArgumentParser(description="Download Op-Kingmd base model")
    p.add_argument("--model", default=os.getenv("OPKMD_MODEL_ID", DEFAULT_MODEL))
    p.add_argument("--cache-dir", default=str(CACHE_DIR))
    p.add_argument("--token", default=os.getenv("HF_TOKEN"))
    p.add_argument("--verify", action="store_true", help="Verify model loads correctly")
    return p.parse_args()


def main():
    args = parse_args()
    console.print(f"\n[bold cyan]Downloading model: {args.model}[/bold cyan]")
    console.print(f"Cache dir: {args.cache_dir}\n")

    with console.status(f"Downloading {args.model}..."):
        local_path = snapshot_download(
            repo_id=args.model,
            cache_dir=args.cache_dir,
            token=args.token,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )

    console.print(f"[green]✓ Model downloaded to: {local_path}[/green]")

    if args.verify:
        console.print("\n[bold]Verifying model loads correctly...[/bold]")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        console.print("  Loading tokenizer...")
        tok = AutoTokenizer.from_pretrained(local_path)
        assert tok is not None
        console.print("  [green]✓ Tokenizer OK[/green]")

        console.print("  Loading model (CPU, no quant for verification)...")
        from op_kingmd.core.model import ModelConfig, ModelEngine
        config = ModelConfig(model_id=local_path, quantize_4bit=False, force_cpu=True)
        engine = ModelEngine(config)
        engine.load()

        response = engine.generate("Say hello in Solidity.")
        console.print(f"  [green]✓ Model generates: {response[:80]}...[/green]")

    console.print("\n[bold green]Setup complete! Run `op-kingmd serve` to start the API.[/bold green]")


if __name__ == "__main__":
    main()
