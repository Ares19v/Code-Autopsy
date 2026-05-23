"""
training/utils.py
─────────────────
Shared helper functions for Code Autopsy training pipeline.
"""

from __future__ import annotations

import re
import json
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass

import torch
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Config loading
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str | Path) -> dict:
    """Load and return a YAML config as a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    console.log(f"[green]✓[/green] Loaded config from [cyan]{path}[/cyan]")
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Chat template formatting
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a code review expert. Analyze the provided code, identify any bugs "
    "or issues, explain the root cause, and provide a corrected version."
)

ASSISTANT_TEMPLATE = """\
## Bug Identified
{bug_identified}

## Root Cause
{root_cause}

## Fixed Code
```{language}
{fixed_code}
```"""


def format_prompt(
    buggy_code: str,
    language: str = "python",
    bug_identified: str = "",
    root_cause: str = "",
    fixed_code: str = "",
    tokenizer=None,
    inference: bool = False,
) -> str:
    """
    Format a training or inference example using the model's chat template.

    For training (inference=False), includes the full assistant response.
    For inference (inference=True), returns only the prompt up to the assistant turn.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Language: {language}\n\n```{language}\n{buggy_code.strip()}\n```"},
    ]

    if not inference:
        assistant_content = ASSISTANT_TEMPLATE.format(
            bug_identified=bug_identified,
            root_cause=root_cause,
            language=language,
            fixed_code=fixed_code.strip(),
        )
        messages.append({"role": "assistant", "content": assistant_content})

    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=inference,
        )

    # Fallback: manual Qwen2.5 chat template format
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if inference:
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Output parsing
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedOutput:
    bug_identified: str
    root_cause: str
    fixed_code: str
    language: str = "python"


def parse_assistant_output(text: str, language: str = "python") -> ParsedOutput:
    """
    Extract structured fields from model output.

    Expected format:
        ## Bug Identified
        <text>
        ## Root Cause
        <text>
        ## Fixed Code
        ```<lang>
        <code>
        ```
    """
    # Strip leading/trailing whitespace and assistant marker
    text = re.sub(r"<\|im_start\|>assistant\s*", "", text)
    text = re.sub(r"<\|im_end\|>.*", "", text, flags=re.DOTALL)
    text = text.strip()

    # Extract sections
    bug_match = re.search(
        r"##\s*Bug Identified\s*\n(.*?)(?=##\s*Root Cause|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    root_match = re.search(
        r"##\s*Root Cause\s*\n(.*?)(?=##\s*Fixed Code|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    code_match = re.search(
        r"##\s*Fixed Code\s*\n```(?:\w+)?\n(.*?)```",
        text, re.DOTALL | re.IGNORECASE
    )

    return ParsedOutput(
        bug_identified=bug_match.group(1).strip() if bug_match else text,
        root_cause=root_match.group(1).strip() if root_match else "",
        fixed_code=code_match.group(1).strip() if code_match else "",
        language=language,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Model diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def log_model_info(model) -> None:
    """Print trainable vs total parameter count in a rich table."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total

    table = Table(title="Model Parameter Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Total parameters", f"{total:,}")
    table.add_row("Trainable parameters", f"{trainable:,}")
    table.add_row("Trainable %", f"{pct:.4f}%")
    console.print(table)


def get_gpu_info() -> dict:
    """Return GPU memory stats for logging."""
    if not torch.cuda.is_available():
        return {"gpu": "none"}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": props.name,
        "gpu_vram_gb": round(props.total_memory / 1e9, 2),
        "gpu_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Confidence score
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np


def compute_confidence(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    prompt_length: int,
) -> float:
    """
    Compute average log-probability over generated tokens as a confidence score.

    Returns a float in [0, 1] where higher = more confident.
    Mean log-prob is in (-inf, 0); we exponentiate to get a perplexity-derived score.
    """
    with torch.no_grad():
        gen_logits = logits[:, prompt_length - 1 : prompt_length - 1 + generated_ids.shape[1], :]
        log_probs = torch.nn.functional.log_softmax(gen_logits.float(), dim=-1)
        token_log_probs = log_probs[0, torch.arange(generated_ids.shape[1]), generated_ids[0]]
        avg_log_prob = token_log_probs.mean().item()
        confidence = float(np.exp(avg_log_prob))
    return round(min(max(confidence, 0.0), 1.0), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    path = Path(path)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str | Path) -> None:
    """Save a list of dicts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    console.log(f"[green]✓[/green] Saved {len(records):,} records to [cyan]{path}[/cyan]")
