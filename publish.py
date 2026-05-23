"""
publish.py
──────────
Merge the LoRA adapter into base model weights and push both artifacts to HuggingFace Hub.

Publishes:
  Ares19v/code-autopsy-7b   — merged full model (for direct use / Ollama)
  Ares19v/code-autopsy-lora — unmerged LoRA adapter (for PEFT loading)

Usage:
  python publish.py
  python publish.py --adapter ./adapter --skip-merge   # push adapter only
  python publish.py --dry-run                          # validate without pushing
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from training.utils import load_config

console = Console()

HF_USER         = "Ares19v"
MERGED_REPO     = f"{HF_USER}/code-autopsy-7b"
ADAPTER_REPO    = f"{HF_USER}/code-autopsy-lora"
CFG_PATH        = ROOT / "training" / "config.yaml"
MERGED_DIR      = ROOT / "merged_model"


MODEL_CARD = """\
---
language:
  - en
  - python
  - javascript
license: apache-2.0
tags:
  - code
  - code-review
  - bug-detection
  - qlora
  - qwen2.5
  - fine-tuned
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
datasets:
  - code_x_glue_cc_code_refinement
  - code_search_net
  - microsoft/codereview-data
pipeline_tag: text-generation
---

# Code Autopsy 🔬

A QLoRA fine-tuned version of [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
specialized for **structured code review**: bug identification, root cause explanation, and auto-fix generation.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Ares19v/code-autopsy-7b", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Ares19v/code-autopsy-7b")

messages = [
    {"role": "system", "content": "You are a code review expert. Analyze the provided code, identify any bugs or issues, explain the root cause, and provide a corrected version."},
    {"role": "user",   "content": "Language: python\\n\\n```python\\ndef avg(nums):\\n    return sum(nums) / len(nums)\\n```"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Training Details

| Parameter | Value |
|---|---|
| Base model | Qwen2.5-Coder-7B-Instruct |
| Method | QLoRA (NF4, 4-bit) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Epochs | 3 |
| Effective batch | 16 |
| Learning rate | 2e-4 |
| Scheduler | Cosine |

## Demo

👉 [Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/Ares19v/code-autopsy)

## GitHub

[Ares19v/Code-Autopsy](https://github.com/Ares19v/Code-Autopsy)
"""


def login_hf(token: str) -> None:
    login(token=token)
    console.log("[green]✓[/green] Logged in to HuggingFace Hub")


def load_base_for_merge(cfg: dict):
    name = cfg["model"]["base_model"]
    console.log(f"Loading base model [cyan]{name}[/cyan] in bf16 for merging...")
    # Load in bf16 full precision for clean merge (not quantized)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )
    return model, tokenizer


def merge_and_save(cfg: dict, adapter_path: str) -> Path:
    model, tokenizer = load_base_for_merge(cfg)

    console.log(f"Loading LoRA adapter from [cyan]{adapter_path}[/cyan]...")
    model = PeftModel.from_pretrained(model, adapter_path)

    console.log("Merging adapter weights into base model...")
    model = model.merge_and_unload()
    model.eval()

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    console.log(f"Saving merged model to [cyan]{MERGED_DIR}[/cyan]...")
    model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))

    # Write model card
    (MERGED_DIR / "README.md").write_text(MODEL_CARD, encoding="utf-8")

    console.log(f"[green]✓[/green] Merged model saved: {MERGED_DIR}")
    return MERGED_DIR


def push_merged(dry_run: bool = False) -> None:
    if dry_run:
        console.log(f"[yellow]DRY RUN[/yellow] Would push merged model to [cyan]{MERGED_REPO}[/cyan]")
        return

    console.log(f"Pushing merged model to [cyan]{MERGED_REPO}[/cyan] ...")
    api = HfApi()
    api.create_repo(repo_id=MERGED_REPO, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(MERGED_DIR),
        repo_id=MERGED_REPO,
        repo_type="model",
        commit_message="Upload Code Autopsy merged model (Qwen2.5-Coder-7B + LoRA)",
    )
    console.log(f"[green]✓[/green] Merged model published: https://huggingface.co/{MERGED_REPO}")


def push_adapter(adapter_path: str, dry_run: bool = False) -> None:
    if dry_run:
        console.log(f"[yellow]DRY RUN[/yellow] Would push adapter to [cyan]{ADAPTER_REPO}[/cyan]")
        return

    console.log(f"Pushing LoRA adapter to [cyan]{ADAPTER_REPO}[/cyan] ...")
    api = HfApi()
    api.create_repo(repo_id=ADAPTER_REPO, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=adapter_path,
        repo_id=ADAPTER_REPO,
        repo_type="model",
        commit_message="Upload Code Autopsy LoRA adapter (Qwen2.5-Coder-7B-Instruct base)",
    )
    console.log(f"[green]✓[/green] LoRA adapter published: https://huggingface.co/{ADAPTER_REPO}")


def main(args: argparse.Namespace) -> None:
    console.rule("[bold magenta]Code Autopsy — HuggingFace Publish[/bold magenta]")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        console.log("[red]HF_TOKEN not set. Add it to your .env file.[/red]")
        sys.exit(1)

    if not args.dry_run:
        login_hf(hf_token)

    cfg = load_config(CFG_PATH)
    adapter_path = str(ROOT / args.adapter)

    if not Path(adapter_path).exists():
        console.log(f"[red]Adapter not found at {adapter_path}[/red]")
        sys.exit(1)

    # ── Step 1: Push unmerged adapter ─────────────────────────────────────────
    push_adapter(adapter_path, dry_run=args.dry_run)

    # ── Step 2: Merge + push full model ──────────────────────────────────────
    if not args.skip_merge:
        merge_and_save(cfg, adapter_path)
        push_merged(dry_run=args.dry_run)
    else:
        console.log("[yellow]--skip-merge set: skipping merged model publish[/yellow]")

    console.rule("[bold green]Publish complete![/bold green]")
    if not args.dry_run:
        console.print(f"\n🔗 Merged model: https://huggingface.co/{MERGED_REPO}")
        console.print(f"🔗 LoRA adapter: https://huggingface.co/{ADAPTER_REPO}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish Code Autopsy to HuggingFace Hub")
    parser.add_argument("--adapter", type=str, default="adapter", help="Path to LoRA adapter dir")
    parser.add_argument("--skip-merge", action="store_true", help="Only push adapter, skip merge")
    parser.add_argument("--dry-run", action="store_true", help="Validate without actually pushing")
    args = parser.parse_args()
    main(args)
