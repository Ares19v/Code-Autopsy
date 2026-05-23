"""
eval/eval_llm_judge.py
──────────────────────
LLM-as-judge evaluation using Gemini API.

Samples 100 examples from the test set, runs model inference, sends each
output to Gemini with a structured rubric, and aggregates scores.

Rubric:
  - bug_correctly_identified: true/false
  - explanation_clarity:      1-5
  - fix_correctness:          1-5

Logs per-example results as a W&B Table and prints a summary.

Usage:
  python eval/eval_llm_judge.py --adapter ./adapter
  python eval/eval_llm_judge.py --adapter ./adapter --n-samples 50 --no-wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import google.generativeai as genai
import torch
import wandb
from dotenv import load_dotenv
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.utils import load_config, load_jsonl, format_prompt, parse_assistant_output

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CFG_PATH = ROOT / "training" / "config.yaml"
GEMINI_MODEL = "gemini-1.5-flash"

JUDGE_PROMPT_TEMPLATE = """\
You are an expert code reviewer evaluating the quality of an AI system's code analysis.

## Buggy Code (Input)
Language: {language}
```{language}
{buggy_code}
```

## AI System Output
### Bug Identified
{bug_identified}

### Root Cause
{root_cause}

### Fixed Code
```{language}
{fixed_code}
```

## Ground Truth Fixed Code
```{language}
{ground_truth_fixed}
```

## Your Task
Evaluate the AI system's output using this rubric. Respond ONLY with valid JSON:

{{
  "bug_correctly_identified": true or false,
  "explanation_clarity": <integer 1-5>,
  "fix_correctness": <integer 1-5>,
  "brief_reasoning": "<one sentence>"
}}

Scoring guide:
- bug_correctly_identified: Did the AI correctly identify the main bug?
- explanation_clarity (1-5): 1=incoherent, 3=acceptable, 5=perfectly clear and insightful
- fix_correctness (1-5): 1=wrong fix, 3=partially correct, 5=correct and idiomatic fix

JSON only, no other text:"""


# ══════════════════════════════════════════════════════════════════════════════
# Model inference
# ══════════════════════════════════════════════════════════════════════════════

def load_model(cfg: dict, adapter_path: str, device: str = "cuda"):
    name = cfg["model"]["base_model"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, token=os.getenv("HF_TOKEN"),
    )
    if Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
        console.log(f"[green]✓[/green] Loaded adapter from [cyan]{adapter_path}[/cyan]")
    else:
        console.log(f"[yellow]Adapter not found at {adapter_path} — using base model[/yellow]")
    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, buggy_code: str, language: str, device: str = "cuda") -> dict:
    prompt = format_prompt(buggy_code=buggy_code, language=language, tokenizer=tokenizer, inference=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    parsed = parse_assistant_output(text, language=language)
    return {
        "bug_identified": parsed.bug_identified,
        "root_cause": parsed.root_cause,
        "fixed_code": parsed.fixed_code,
        "raw_output": text,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Gemini judge
# ══════════════════════════════════════════════════════════════════════════════

def judge_with_gemini(
    gemini_model,
    buggy_code: str,
    language: str,
    bug_identified: str,
    root_cause: str,
    fixed_code: str,
    ground_truth_fixed: str,
    max_retries: int = 3,
) -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        language=language,
        buggy_code=buggy_code[:1000],
        bug_identified=bug_identified[:500],
        root_cause=root_cause[:500],
        fixed_code=fixed_code[:800],
        ground_truth_fixed=ground_truth_fixed[:800],
    )

    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(prompt)
            raw = response.text.strip()
            # Strip markdown code fences if present
            raw = raw.removeprefix("```json").removesuffix("```").strip()
            result = json.loads(raw)
            # Validate required keys
            assert "bug_correctly_identified" in result
            assert "explanation_clarity" in result
            assert "fix_correctness" in result
            return result
        except Exception as e:
            logger.warning(f"Gemini attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)

    return {
        "bug_correctly_identified": None,
        "explanation_clarity": None,
        "fix_correctness": None,
        "brief_reasoning": "Parse error",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    console.rule("[bold magenta]Code Autopsy — LLM-as-Judge Evaluation[/bold magenta]")

    cfg = load_config(CFG_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Setup Gemini ──────────────────────────────────────────────────────────
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        console.log("[red]GEMINI_API_KEY not set. Aborting.[/red]")
        sys.exit(1)
    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    console.log(f"[green]✓[/green] Gemini model: [cyan]{GEMINI_MODEL}[/cyan]")

    # ── Load test data ────────────────────────────────────────────────────────
    test_path = ROOT / cfg["data"]["test_file"]
    all_records = load_jsonl(test_path)
    # Filter records that have buggy_code and fixed_code stored
    usable = [r for r in all_records if r.get("buggy_code") and r.get("fixed_code")]
    random.seed(42)
    sample = random.sample(usable, min(args.n_samples, len(usable)))
    console.log(f"Sampling [green]{len(sample)}[/green] examples for LLM judge")

    # ── Load inference model ──────────────────────────────────────────────────
    model, tokenizer = load_model(cfg, args.adapter, device)

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = []

    for rec in tqdm(sample, desc="Judging"):
        buggy = rec["buggy_code"]
        gt_fixed = rec["fixed_code"]
        language = rec.get("language", "python")

        # Model inference
        try:
            inf = run_inference(model, tokenizer, buggy, language, device)
        except Exception as e:
            logger.warning(f"Inference error: {e}")
            continue

        # Gemini judge
        judgment = judge_with_gemini(
            gemini_model,
            buggy_code=buggy,
            language=language,
            bug_identified=inf["bug_identified"],
            root_cause=inf["root_cause"],
            fixed_code=inf["fixed_code"],
            ground_truth_fixed=gt_fixed,
        )

        results.append({
            "language": language,
            "buggy_code": buggy[:300],
            "model_bug_identified": inf["bug_identified"][:300],
            "model_root_cause": inf["root_cause"][:300],
            "model_fixed_code": inf["fixed_code"][:300],
            "ground_truth_fixed": gt_fixed[:300],
            **judgment,
        })

        # Respect Gemini free-tier rate limit (15 RPM)
        time.sleep(4)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    valid = [r for r in results if r.get("explanation_clarity") is not None]
    n_valid = len(valid)

    if n_valid == 0:
        console.log("[red]No valid judge results — check Gemini API key and quota.[/red]")
        return

    pct_correct = 100 * sum(1 for r in valid if r["bug_correctly_identified"]) / n_valid
    avg_clarity = sum(r["explanation_clarity"] for r in valid) / n_valid
    avg_fix = sum(r["fix_correctness"] for r in valid) / n_valid

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_table = Table(title="LLM-as-Judge Summary", header_style="bold magenta", show_lines=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Score", style="green", justify="right")

    summary_table.add_row("Bugs Correctly Identified", f"{pct_correct:.1f}%")
    summary_table.add_row("Explanation Clarity (1-5)", f"{avg_clarity:.2f}")
    summary_table.add_row("Fix Correctness (1-5)", f"{avg_fix:.2f}")
    summary_table.add_row("Samples Evaluated", str(n_valid))
    console.print(summary_table)

    # ── W&B logging ───────────────────────────────────────────────────────────
    if not args.no_wandb and os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="code-autopsy", name="llm-judge-eval", job_type="evaluation")

        columns = [
            "language", "buggy_code", "model_bug_identified", "model_root_cause",
            "model_fixed_code", "ground_truth_fixed",
            "bug_correctly_identified", "explanation_clarity", "fix_correctness", "brief_reasoning"
        ]
        data = [[r.get(c, "") for c in columns] for r in valid]
        wb_table = wandb.Table(columns=columns, data=data)

        wandb.log({
            "llm_judge_results": wb_table,
            "pct_bugs_identified": pct_correct,
            "avg_explanation_clarity": avg_clarity,
            "avg_fix_correctness": avg_fix,
        })
        wandb.finish()
        console.log("[green]✓[/green] Judge results logged to W&B")

    console.rule("[bold green]LLM-as-Judge evaluation complete![/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation using Gemini")
    parser.add_argument("--adapter", type=str, default="./adapter", help="Path to LoRA adapter")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of test examples to judge")
    parser.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    args = parser.parse_args()
    main(args)
