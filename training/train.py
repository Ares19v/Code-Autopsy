"""
training/train.py
─────────────────
QLoRA fine-tuning for Code Autopsy using SFTTrainer + Qwen2.5-Coder-7B-Instruct.

Usage:
  # Full training run
  python training/train.py

  # Debug mode: 1 step, no W&B, skips saving
  python training/train.py --debug

  # Custom config
  python training/train.py --config training/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from trl import SFTTrainer, SFTConfig

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.utils import load_config, log_model_info, get_gpu_info, load_jsonl

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# W&B callback for extra logging
# ══════════════════════════════════════════════════════════════════════════════

class WandbMetricsCallback(TrainerCallback):
    """Logs GPU memory and learning rate alongside training loss."""

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and wandb.run:
            extra = get_gpu_info()
            wandb.log({**logs, **extra, "step": state.global_step})


# ══════════════════════════════════════════════════════════════════════════════
# Model & tokenizer setup
# ══════════════════════════════════════════════════════════════════════════════

def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    qcfg = cfg["quantization"]
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    return BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=dtype_map[qcfg["bnb_4bit_compute_dtype"]],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )


def build_lora_config(cfg: dict) -> LoraConfig:
    lcfg = cfg["lora"]
    return LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        target_modules=lcfg["target_modules"],
        bias=lcfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(cfg: dict, debug: bool = False):
    model_name = cfg["model"]["base_model"]
    hf_token = os.getenv("HF_TOKEN")

    console.log(f"Loading tokenizer: [cyan]{model_name}[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"]["trust_remote_code"],
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    console.log(f"Loading model with 4-bit NF4 quantization: [cyan]{model_name}[/cyan]")
    bnb_config = build_bnb_config(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg["model"]["trust_remote_code"],
        token=hf_token,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Prepare for k-bit training (handles gradient checkpointing + casting)
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = build_lora_config(cfg)
    model = get_peft_model(model, lora_config)

    log_model_info(model)
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ══════════════════════════════════════════════════════════════════════════════

def load_splits(cfg: dict, debug: bool = False) -> tuple[Dataset, Dataset]:
    train_path = ROOT / cfg["data"]["train_file"]
    val_path = ROOT / cfg["data"]["val_file"]

    console.log(f"Loading training data from [cyan]{train_path}[/cyan]")
    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)

    if debug:
        train_records = train_records[:32]
        val_records = val_records[:8]
        console.log("[yellow]DEBUG mode: using 32 train / 8 val examples[/yellow]")

    def filter_len(recs):
        # 1500 chars guarantees it fits entirely within the 512 token limit
        # without truncating the assistant's fixed code response at the end.
        return [r for r in recs if len(r.get(cfg["data"]["dataset_text_field"], "")) < 1500]

    train_records = filter_len(train_records)
    val_records = filter_len(val_records)

    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)

    console.log(f"Train: [green]{len(train_ds):,}[/green] | Val: [green]{len(val_ds):,}[/green]")
    return train_ds, val_ds


# ══════════════════════════════════════════════════════════════════════════════
# Training arguments
# ══════════════════════════════════════════════════════════════════════════════

def build_training_args(cfg: dict, debug: bool = False) -> SFTConfig:
    tcfg = cfg["training"]
    wcfg = cfg.get("wandb", {})
    output_dir = ROOT / tcfg["output_dir"]

    report_to = "none" if debug else tcfg.get("report_to", "wandb")

    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=1 if debug else tcfg["num_train_epochs"],
        per_device_train_batch_size=1 if debug else tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=1 if debug else tcfg["gradient_accumulation_steps"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        fp16=tcfg["fp16"],
        bf16=tcfg["bf16"],
        logging_steps=1 if debug else tcfg["logging_steps"],
        eval_strategy="steps" if not debug else "no",
        eval_steps=tcfg["eval_steps"],
        save_strategy="no" if debug else tcfg["save_strategy"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        load_best_model_at_end=False if debug else tcfg["load_best_model_at_end"],
        metric_for_best_model=tcfg["metric_for_best_model"],
        optim=tcfg["optim"],
        dataloader_num_workers=0 if debug else tcfg["dataloader_num_workers"],
        remove_unused_columns=tcfg["remove_unused_columns"],
        report_to=report_to,
        run_name="code-autopsy-qlora",
        max_length=cfg["model"]["max_seq_length"],
        dataset_text_field=cfg["data"]["dataset_text_field"],
        # Stop after 1 step in debug
        max_steps=1 if debug else -1,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    console.rule("[bold magenta]Code Autopsy — QLoRA Training[/bold magenta]")

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    # ── W&B init ──────────────────────────────────────────────────────────────
    use_wandb = (not args.debug) and (cfg["training"].get("report_to", "none") == "wandb")
    if use_wandb:
        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key:
            wandb.login(key=wandb_key)
        wcfg = cfg.get("wandb", {})
        wandb.init(
            project=wcfg.get("project", "code-autopsy"),
            entity=os.getenv("WANDB_ENTITY") or wcfg.get("entity") or None,
            name="code-autopsy-qlora",
            config=cfg,
            tags=["qlora", "qwen2.5", "code-review"],
        )
        console.log("[green]✓[/green] W&B initialized")
    else:
        console.log("[yellow]W&B logging disabled[/yellow]")

    # ── Load model + tokenizer ────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg, debug=args.debug)

    # ── Load datasets ─────────────────────────────────────────────────────────
    train_ds, val_ds = load_splits(cfg, debug=args.debug)

    # ── Build SFTTrainer ──────────────────────────────────────────────────────
    training_args = build_training_args(cfg, debug=args.debug)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds if not args.debug else None,
        args=training_args,
        callbacks=[WandbMetricsCallback()] if use_wandb else [],
    )

    # Force use_reentrant=False for gradient checkpointing to prevent memory bloat
    if cfg["training"].get("gradient_checkpointing", False):
        trainer.args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # ── Train ─────────────────────────────────────────────────────────────────
    resume_ckpt = None
    if args.resume:
        ckpt_dir = ROOT / cfg["training"]["output_dir"]
        checkpoints = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if checkpoints:
            resume_ckpt = str(checkpoints[-1])
            console.log(f"[green]Resuming from checkpoint:[/green] [cyan]{resume_ckpt}[/cyan]")
        else:
            console.log("[yellow]--resume set but no checkpoints found, starting from scratch[/yellow]")

    console.log("[bold]Starting training...[/bold]")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    console.log("[bold green]Training complete![/bold green]")

    # ── Save adapter ──────────────────────────────────────────────────────────
    if not args.debug:
        adapter_path = ROOT / cfg["paths"]["adapter_output"]
        adapter_path.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))
        console.log(f"[green]✓[/green] LoRA adapter saved to [cyan]{adapter_path}[/cyan]")

        if wandb.run:
            wandb.finish()
    else:
        console.log("[yellow]DEBUG run complete — adapter not saved[/yellow]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Code Autopsy QLoRA model")
    parser.add_argument(
        "--config", type=str, default="training/config.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: 1 step, no W&B, no checkpoints saved"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint in checkpoints/"
    )
    args = parser.parse_args()
    main(args)
