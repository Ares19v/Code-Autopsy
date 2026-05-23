"""
data/prepare_dataset.py
───────────────────────
Downloads and processes training data for Code Autopsy from HuggingFace datasets.

Sources:
  1. code_search_net          — Python & JavaScript code with docstrings
  2. code_x_glue_cc_code_refinement — Actual buggy/fixed code pairs
  3. microsoft/codereview-data — Code review comments + diffs

Output:
  data/processed/train.jsonl
  data/processed/val.jsonl
  data/processed/test.jsonl

Each record format:
  {
    "text":     "<full chat-formatted string ready for SFTTrainer>",
    "language": "python" | "javascript",
    "source":   "<dataset name>"
  }

Usage:
  python data/prepare_dataset.py
  python data/prepare_dataset.py --dry-run      # No download, just validate logic
  python data/prepare_dataset.py --max-samples 5000
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from rich.console import Console
from transformers import AutoTokenizer

load_dotenv()

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.utils import format_prompt, save_jsonl

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Constants ─────────────────────────────────────────────────────────────────
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

TRAIN_SPLIT = 0.90
VAL_SPLIT = 0.05
# remainder → test

SEEN_HASHES: set[str] = set()


def code_hash(code: str) -> str:
    return hashlib.md5(code.strip().encode()).hexdigest()


def is_duplicate(code: str) -> bool:
    h = code_hash(code)
    if h in SEEN_HASHES:
        return True
    SEEN_HASHES.add(h)
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Source 1: code_x_glue_cc_code_refinement — real bug-fix pairs
# ══════════════════════════════════════════════════════════════════════════════

def load_code_refinement(max_samples: int = 10_000, dry_run: bool = False) -> list[dict]:
    """
    CodeXGLUE Code Refinement dataset.
    Contains buggy Java functions + their fixed versions.
    We include it as synthetic Python/JS analogues are not available at this scale.
    """
    if dry_run:
        console.log("[yellow]DRY RUN[/yellow] Skipping code_x_glue_cc_code_refinement download")
        return []

    console.log("Loading [cyan]google/code_x_glue_cc_code_refinement[/cyan] (small split)...")
    try:
        ds = load_dataset("google/code_x_glue_cc_code_refinement", "small", trust_remote_code=True)
    except Exception as e:
        console.log(f"[red]Failed to load code refinement dataset: {e}[/red]")
        return []

    records = []
    split = ds.get("train", ds[list(ds.keys())[0]])

    for row in split:
        buggy = row.get("buggy", "").strip()
        fixed = row.get("fixed", "").strip()
        if not buggy or not fixed or is_duplicate(buggy):
            continue

        records.append({
            "buggy_code": buggy,
            "fixed_code": fixed,
            "language": "java",          # dataset is Java; still valid for review logic
            "bug_identified": "The code contains a bug that causes incorrect behavior or runtime error.",
            "root_cause": "Syntactic or logical error introduced during development.",
            "source": "code_x_glue_cc_code_refinement",
        })

        if len(records) >= max_samples:
            break

    console.log(f"[green]✓[/green] code_x_glue_cc_code_refinement: {len(records):,} records")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Source 2: code_search_net — Python & JavaScript with docstrings
# ══════════════════════════════════════════════════════════════════════════════

def load_codesearchnet(
    languages: list[str] = ["python", "javascript"],
    max_samples_per_lang: int = 5_000,
    dry_run: bool = False,
) -> list[dict]:
    """
    CodeSearchNet: function-level code with natural language docstrings.
    We treat the docstring as a description and synthesize a minimal "bug review"
    format (the function itself is the "fixed" version; we omit buggy pairs here
    and use this data primarily to teach structured output format).
    """
    if dry_run:
        console.log("[yellow]DRY RUN[/yellow] Skipping code_search_net download")
        return []

    # Toxic dataset disabled to prevent lobotomizing the model
    # with fake 'No critical bug found' labels.
    console.log("[yellow]SKIPPING[/yellow] code_search_net (disabled due to bad labels)")
    return []


# ══════════════════════════════════════════════════════════════════════════════
# Source 3: microsoft/codereview-data — actual code reviews
# ══════════════════════════════════════════════════════════════════════════════

def load_codereview(max_samples: int = 5_000, dry_run: bool = False) -> list[dict]:
    """
    Microsoft CodeReviewer dataset: code diffs + review comments.
    We extract (old_hunk, new_hunk, comment) triples and map them to our format.
    """
    if dry_run:
        console.log("[yellow]DRY RUN[/yellow] Skipping microsoft/codereview-data download")
        return []

    console.log("Loading [cyan]microsoft/codereview-data[/cyan]...")
    try:
        ds = load_dataset("microsoft/codereview-data", trust_remote_code=True)
        split = ds.get("train", ds[list(ds.keys())[0]])
    except Exception as e:
        console.log(f"[red]Failed to load codereview-data: {e}[/red]")
        return []

    records = []
    for row in split:
        old_code = row.get("old_hunk", "").strip()
        new_code = row.get("new_hunk", "").strip()
        comment = row.get("comment", "").strip()
        lang = str(row.get("lang", "python")).lower()

        if not old_code or not new_code or not comment or is_duplicate(old_code):
            continue
        if lang not in ("python", "javascript", "java", "cpp", "c"):
            lang = "python"

        records.append({
            "buggy_code": old_code,
            "fixed_code": new_code,
            "language": lang if lang in ("python", "javascript") else "python",
            "bug_identified": comment[:500],
            "root_cause": f"Code was updated based on review: {comment[:300]}",
            "source": "microsoft/codereview-data",
        })

        if len(records) >= max_samples:
            break

    console.log(f"[green]✓[/green] microsoft/codereview-data: {len(records):,} records")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Chat template formatting
# ══════════════════════════════════════════════════════════════════════════════

def format_record(record: dict, tokenizer=None) -> dict:
    """Convert a raw record into a chat-formatted training example."""
    text = format_prompt(
        buggy_code=record["buggy_code"],
        language=record["language"],
        bug_identified=record["bug_identified"],
        root_cause=record["root_cause"],
        fixed_code=record["fixed_code"],
        tokenizer=tokenizer,
        inference=False,
    )
    return {
        "text": text,
        "language": record["language"],
        "source": record["source"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    console.rule("[bold magenta]Code Autopsy — Dataset Preparation[/bold magenta]")

    # ── Load tokenizer for apply_chat_template ────────────────────────────────
    tokenizer = None
    if not args.dry_run:
        console.log(f"Loading tokenizer from [cyan]{BASE_MODEL}[/cyan]...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN"),
            )
        except Exception as e:
            console.log(f"[yellow]Warning: Could not load tokenizer ({e}). Using manual template.[/yellow]")

    # ── Gather raw records ────────────────────────────────────────────────────
    max_s = args.max_samples
    all_records: list[dict] = []

    all_records += load_code_refinement(max_samples=max_s // 2, dry_run=args.dry_run)
    all_records += load_codesearchnet(
        max_samples_per_lang=max_s // 4,
        dry_run=args.dry_run,
    )
    # Skipping load_codereview as microsoft/codereview-data is no longer on the Hub

    if args.dry_run:
        # Generate a few synthetic examples to validate formatting
        all_records = [
            {
                "buggy_code": "def divide(a, b):\n    return a / b\n\ndivide(10, 0)",
                "fixed_code": "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b",
                "language": "python",
                "bug_identified": "ZeroDivisionError: division by zero when b=0",
                "root_cause": "Missing guard clause for zero denominator",
                "source": "synthetic",
            },
            {
                "buggy_code": "async function getData(url) {\n  const res = await fetch(url);\n  const data = res.json();\n  return data;\n}",
                "fixed_code": "async function getData(url) {\n  const res = await fetch(url);\n  const data = await res.json();\n  return data;\n}",
                "language": "javascript",
                "bug_identified": "Missing `await` before `res.json()` returns a Promise, not the data",
                "root_cause": "`Response.json()` is an async method that returns a Promise. Without await, `data` holds the Promise object instead of the parsed JSON.",
                "source": "synthetic",
            },
        ]

    console.log(f"\nTotal raw records collected: [bold green]{len(all_records):,}[/bold green]")

    # ── Format with chat template ─────────────────────────────────────────────
    console.log("Applying chat template...")
    formatted = []
    for rec in all_records:
        try:
            formatted.append(format_record(rec, tokenizer=tokenizer))
        except Exception as e:
            logger.warning(f"Skipping record due to formatting error: {e}")

    console.log(f"Formatted: [bold green]{len(formatted):,}[/bold green] records")

    # ── Shuffle ───────────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(formatted)).tolist()
    formatted = [formatted[i] for i in indices]

    # ── Split ─────────────────────────────────────────────────────────────────
    n = len(formatted)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    train_data = formatted[:n_train]
    val_data = formatted[n_train : n_train + n_val]
    test_data = formatted[n_train + n_val :]

    # ── Save ──────────────────────────────────────────────────────────────────
    save_jsonl(train_data, PROCESSED_DIR / "train.jsonl")
    save_jsonl(val_data, PROCESSED_DIR / "val.jsonl")
    save_jsonl(test_data, PROCESSED_DIR / "test.jsonl")

    # ── Summary ───────────────────────────────────────────────────────────────
    from rich.table import Table
    table = Table(title="Dataset Split Summary", header_style="bold magenta")
    table.add_column("Split", style="cyan")
    table.add_column("Records", justify="right", style="green")
    table.add_column("File", style="dim")
    table.add_row("Train", f"{len(train_data):,}", "data/processed/train.jsonl")
    table.add_row("Val",   f"{len(val_data):,}",   "data/processed/val.jsonl")
    table.add_row("Test",  f"{len(test_data):,}",  "data/processed/test.jsonl")
    table.add_row("Total", f"{n:,}", "—")
    console.print(table)

    console.rule("[bold green]Dataset preparation complete![/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Code Autopsy training dataset")
    parser.add_argument(
        "--max-samples", type=int, default=20_000,
        help="Approximate max total samples to collect (default: 20000)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run with synthetic data only — no downloads"
    )
    args = parser.parse_args()
    main(args)
