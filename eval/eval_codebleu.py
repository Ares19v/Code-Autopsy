"""
eval/eval_codebleu.py
─────────────────────
Quantitative evaluation: BLEU score for base model vs fine-tuned model.

Usage:
  python eval/eval_codebleu.py --max-samples 20
  python eval/eval_codebleu.py --max-samples 1
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import re
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.utils import load_config, load_jsonl, format_prompt, parse_assistant_output

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CFG_PATH = ROOT / "training" / "config.yaml"
bleu = BLEU(effective_order=True)


# ══════════════════════════════════════════════════════════════════════════════
# Inference helper
# ══════════════════════════════════════════════════════════════════════════════

def generate_fixed_code(
    model,
    tokenizer,
    buggy_code: str,
    language: str,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> str:
    prompt = format_prompt(
        buggy_code=buggy_code,
        language=language,
        tokenizer=tokenizer,
        inference=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    parsed = parse_assistant_output(text, language=language)
    return parsed.fixed_code or text


# ══════════════════════════════════════════════════════════════════════════════
# Load models
# ══════════════════════════════════════════════════════════════════════════════

def load_base_model(cfg: dict, device: str = "cuda"):
    name = cfg["model"]["base_model"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, token=os.getenv("HF_TOKEN"),
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(cfg: dict, adapter_path: str, device: str = "cuda"):
    model, tokenizer = load_base_model(cfg, device)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    console.rule("[bold magenta]Code Autopsy — BLEU Evaluation[/bold magenta]")

    cfg = load_config(CFG_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load test data ────────────────────────────────────────────────────────
    test_path = ROOT / cfg["data"]["test_file"]
    test_records = load_jsonl(test_path)
    if args.max_samples:
        test_records = test_records[: args.max_samples]
    console.log(f"Evaluating on [green]{len(test_records):,}[/green] test examples")

    results: dict[str, dict] = {}

    for model_name, loader_fn in [
        ("base", lambda: load_base_model(cfg, device)),
        ("finetuned", lambda: load_finetuned_model(cfg, args.adapter, device)),
    ]:
        if model_name == "finetuned" and not Path(args.adapter).exists():
            console.log(f"[yellow]Adapter not found at {args.adapter} — skipping fine-tuned eval[/yellow]")
            continue

        console.log(f"\nRunning inference with [cyan]{model_name}[/cyan] model...")
        model, tokenizer = loader_fn()

        predictions: list[str] = []
        references: list[str] = []

        for rec in tqdm(test_records, desc=f"{model_name}"):
            text = rec.get("text", "")
            if not text:
                continue

            # Extract buggy code from user prompt
            buggy_match = re.search(r"```\w*\n(.*?)```\s*<\|im_end\|>\s*<\|im_start\|>assistant", text, re.DOTALL)
            buggy = buggy_match.group(1).strip() if buggy_match else ""

            # Extract fixed code from assistant response
            fixed_match = re.search(r"##\s*Fixed Code\s*\n```(?:\w+)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
            gt_fixed = fixed_match.group(1).strip() if fixed_match else ""

            lang = rec.get("language", "python")

            if not buggy or not gt_fixed:
                continue

            try:
                pred = generate_fixed_code(model, tokenizer, buggy, lang, device=device)
                predictions.append(pred)
                references.append(gt_fixed)
            except Exception as e:
                logger.warning(f"Inference error: {e}")

        # ── Compute BLEU ──────────────────────────────────────────────────────
        if not predictions:
            console.log(f"[red]No predictions for {model_name} — check dataset parsing[/red]")
            results[model_name] = {"bleu": 0.0}
            continue

        score = bleu.corpus_score(predictions, [references])
        results[model_name] = {"bleu": score.score}
        console.log(f"[green]{model_name}[/green] BLEU: [bold]{score.score:.2f}[/bold]")

        # Free VRAM between runs — delete both model AND tokenizer
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # ── Results table ─────────────────────────────────────────────────────────
    console.rule("Results")
    table = Table(title="BLEU Evaluation (higher = better)", header_style="bold magenta", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("BLEU Score", justify="right", style="green")

    if "base" in results:
        table.add_row("Base (Qwen2.5-Coder-7B)", f"{results['base']['bleu']:.2f}")
    if "finetuned" in results:
        table.add_row("Fine-tuned (Code-Autopsy)", f"{results['finetuned']['bleu']:.2f}")

    if "base" in results and "finetuned" in results:
        delta = results["finetuned"]["bleu"] - results["base"]["bleu"]
        sign = "+" if delta >= 0 else ""
        color = "bold green" if delta >= 0 else "bold red"
        table.add_row("Delta", f"[{color}]{sign}{delta:.2f}[/]")

    console.print(table)
    console.rule("[bold green]Evaluation complete![/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLEU evaluation for Code Autopsy")
    parser.add_argument("--adapter", type=str, default="./adapter", help="Path to LoRA adapter")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples (default: all)")
    args = parser.parse_args()
    main(args)
