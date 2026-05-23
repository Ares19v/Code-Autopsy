"""
serve/api.py
────────────
FastAPI REST endpoint for Code Autopsy inference.

Endpoints:
  GET  /health  — health check
  POST /review  — run code review analysis

Loads the model once at startup via lifespan context manager.
Confidence = average log-probability over generated tokens (exponentiated to [0,1]).

Usage:
  uvicorn serve.api:app --host 0.0.0.0 --port 8000 --reload

  # With Docker:
  docker build -t code-autopsy ./serve
  docker run -p 8000:8000 -v ./adapter:/app/adapter code-autopsy
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.utils import format_prompt, parse_assistant_output

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", str(ROOT / "adapter"))
HF_TOKEN     = os.getenv("HF_TOKEN")
MAX_NEW_TOKENS = 600
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Global model state ────────────────────────────────────────────────────────
_model = None
_tokenizer = None


def load_model_once() -> None:
    global _model, _tokenizer
    logger.info(f"Loading tokenizer: {BASE_MODEL}")
    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=HF_TOKEN
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info(f"Loading model: {BASE_MODEL} (4-bit NF4)")
    _model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
    )

    adapter_path = Path(ADAPTER_PATH)
    if False and adapter_path.exists():
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        _model = PeftModel.from_pretrained(_model, str(adapter_path))
    else:
        logger.warning(f"Adapter loading disabled. Running base model.")

    _model.eval()
    logger.info("Model ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_once()
    yield
    # Cleanup
    global _model, _tokenizer
    del _model, _tokenizer
    torch.cuda.empty_cache()


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Code Autopsy API",
    description="QLoRA fine-tuned code review: bug identification, root cause, and fix.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ReviewRequest(BaseModel):
    code: str = Field(..., min_length=5, max_length=8192, description="The buggy code to review")
    language: Literal["python", "javascript"] = Field("python", description="Programming language")

    @field_validator("code")
    @classmethod
    def strip_code(cls, v: str) -> str:
        return v.strip()


class ReviewResponse(BaseModel):
    bug_identified: str
    root_cause: str
    fixed_code: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Average log-prob confidence in [0,1]")
    language: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    adapter_loaded: bool
    device: str
    vram_allocated_gb: float | None


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(code: str, language: str) -> ReviewResponse:
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded")

    t0 = time.perf_counter()

    prompt = format_prompt(
        buggy_code=code,
        language=language,
        tokenizer=_tokenizer,
        inference=True,
    )

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    prompt_length = inputs["input_ids"].shape[1]
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,        # ignored when do_sample=False
            pad_token_id=_tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[:, prompt_length:]
    generated_text = _tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # ── Confidence: average log-probability over generated tokens ─────────────
    # scores is a tuple of (vocab_size,) tensors, one per generated step
    log_probs_list = []
    for step_idx, score in enumerate(outputs.scores):
        if step_idx >= generated_ids.shape[1]:
            break
        token_id = generated_ids[0, step_idx].item()
        lp = torch.nn.functional.log_softmax(score[0].float(), dim=-1)
        log_probs_list.append(lp[token_id].item())

    if log_probs_list:
        avg_log_prob = float(np.mean(log_probs_list))
        confidence = round(float(np.exp(avg_log_prob)), 4)
    else:
        confidence = 0.0

    # ── Parse structured output ───────────────────────────────────────────────
    parsed = parse_assistant_output(generated_text, language=language)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    return ReviewResponse(
        bug_identified=parsed.bug_identified or "Unable to parse bug identification.",
        root_cause=parsed.root_cause or "Unable to parse root cause.",
        fixed_code=parsed.fixed_code or generated_text,
        confidence=min(max(confidence, 0.0), 1.0),
        language=language,
        latency_ms=latency_ms,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    """Returns API health status, model name, and VRAM usage."""
    vram = None
    if torch.cuda.is_available():
        vram = round(torch.cuda.memory_allocated(0) / 1e9, 3)
    return HealthResponse(
        status="ok",
        model=BASE_MODEL,
        adapter_loaded=Path(ADAPTER_PATH).exists(),
        device=DEVICE,
        vram_allocated_gb=vram,
    )


@app.post("/review", response_model=ReviewResponse, tags=["inference"])
async def review(request: ReviewRequest):
    """
    Analyze buggy code and return:
    - Bug identified
    - Root cause explanation
    - Fixed code
    - Confidence score (average log-probability)
    """
    try:
        return run_inference(request.code, request.language)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/", tags=["meta"])
async def root():
    return {
        "name": "Code Autopsy API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "review": "POST /review",
    }
