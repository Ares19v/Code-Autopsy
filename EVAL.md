# EVAL — Code Autopsy

> **Evaluation Date:** 2026-05-29  
> **Evaluator:** Automated Portfolio Review  
> **Maturity Level:** MVP

---

## 1. Project Purpose & Problem Statement

Code Autopsy is a QLoRA fine-tuned language model that takes buggy Python or JavaScript code as input and returns a structured three-part analysis: bug identification, root cause explanation, and corrected code. The target audience is developers who want fast, structured code review feedback without paying for a GPT-4 API call on every snippet. It also serves as a demonstration of end-to-end LLM fine-tuning — from dataset engineering through training, evaluation, serving, and deployment — on consumer hardware.

The problem it solves is real: base code models are capable of explanation but produce inconsistent output formats that are difficult to parse programmatically. Fine-tuning on a structured chat template enforces a predictable `## Bug Identified / ## Root Cause / ## Fixed Code` schema.

---

## 2. Technical Architecture

The system has five distinct layers:

- **Data Pipeline:** `prepare_dataset.py` downloads and merges three sources (CodeSearchNet, CodeXGLUE Code Refinement, Microsoft CodeReviewer), deduplicates by code hash, formats each example into a three-turn chat template, shuffles with a fixed seed, and writes 90/5/5 train/val/test JSONL splits.
- **Training:** `SFTTrainer` from TRL with QLoRA via PEFT and BitsAndBytes 4-bit NF4 quantization. Target modules are `q_proj` and `v_proj`. LoRA rank=16, alpha=32, dropout=0.05. Effective batch size of 16 via gradient accumulation. Experiment tracking via Weights & Biases.
- **Serving:** FastAPI REST API (`POST /review`) that loads the base model + LoRA adapter at startup and runs generation. A confidence proxy (average log-probability scaled to [0,1]) is returned alongside the structured analysis.
- **Demo:** Gradio UI deployed to HuggingFace Spaces, calling the FastAPI backend via `API_URL`.
- **Alternative serving:** Ollama GGUF path via `convert_to_gguf.sh` and a `Modelfile` for fully local inference without the GPU requirement.
- **Publishing:** `publish.py` merges adapter weights and pushes both the full model (`Ares19v/code-autopsy-7b`) and raw adapter (`Ares19v/code-autopsy-lora`) to the HuggingFace Hub.

---

## 3. Model / Algorithm Details

**Base Model:** Qwen2.5-Coder-7B-Instruct — a 7B code-specialized instruction-tuned model with strong multilingual code understanding.

**Fine-tuning Method:** QLoRA (Quantized LoRA) — the base model is loaded in 4-bit NF4 (NormalFloat4) via BitsAndBytes, and only the LoRA adapter weights (5.05M of 4.36B parameters, 0.12%) are trained in bfloat16.

**Training Configuration:**
| Hyperparameter | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Effective batch size | 16 (4 × 4 grad accum) |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Max sequence length | 512 tokens |
| Epochs | 3 (document.md) / 1 (README) |
| Optimizer | paged_adamw_8bit |
| Hardware | RTX 5060 (8 GB VRAM) |

**Training Results:**
| Metric | Value |
|---|---|
| Final train loss | 0.407 |
| Final eval loss | 0.297 |
| Eval token accuracy | 92.3% |

**Evaluation:**

Dual-track evaluation — quantitative (sacreBLEU on n=20) and qualitative (LLM-as-judge via Gemini API on n=100):

| Model | sacreBLEU |
|---|---|
| Qwen2.5-Coder-7B base | 10.63 |
| Code Autopsy (fine-tuned) | 70.03 |
| **Improvement** | **+558.7%** |

The dramatic BLEU improvement is largely attributable to format alignment — the base model fails to match the strict `## Fixed Code\n```python` Markdown schema, causing near-zero BLEU even when the code content is correct. This is an important nuance: the improvement measures formatting compliance as much as semantic correctness. The LLM-as-judge evaluation with bug identification / explanation clarity / fix correctness rubrics is the more meaningful qualitative signal.

---

## 4. Strengths

- **Complete MLOps pipeline** — data prep, training, evaluation, serving, and publishing are all scripted and documented, not just the training step.
- **Dual evaluation tracks** — BLEU + LLM-as-judge gives both a quantitative and qualitative signal.
- **Multiple serving paths** — FastAPI, Gradio, Ollama GGUF — covers GPU-server, demo, and fully-local use cases.
- **HuggingFace publication** — model and adapter are publicly available at `Ares19v/code-autopsy-7b`.
- **W&B integration** — training metrics and evaluation results tracked and logged.
- **Docker support** — containerized serve and demo layers.
- **Windows UX** — `INSTALL.bat` + `Run_Project.bat` for one-click local setup.
- **Modelfile for Ollama** — enables local GGUF inference without any Python dependencies.
- **GitHub Actions CI** — lint + unit tests that run without a GPU.

---

## 5. Limitations & Known Gaps

- **BLEU sample size is small (n=20).** The evaluation report itself acknowledges this. A proper evaluation on the full 1,000-example test set would produce statistically reliable numbers.
- **BLEU measures format compliance more than correctness.** A model that outputs perfectly formatted but wrong code will score high. The LLM-judge rubric is more meaningful but n=100 is still a modest sample.
- **1-epoch training (per README) vs 3-epoch config (per document.md).** There is a discrepancy between sources. The README says "fine-tuned in 1 epoch" while `document.md` specifies 3 epochs. The eval report corroborates 1 epoch (final loss 0.407 matches an early stopping pattern). This should be clarified.
- **Only `q_proj` and `v_proj` are trained.** Standard practice for QLoRA adds `k_proj`, `o_proj`, and sometimes MLP layers. Targeting more modules would likely improve semantic quality.
- **Max sequence length of 512 tokens** limits the complexity of code snippets the model can review. Real-world functions often exceed this.
- **The confidence proxy** (log-probability average) is not validated against actual accuracy — a model can output high log-prob nonsense if the training distribution is narrow.
- **`.env` file must not be staged** — confirmed excluded from evaluation per instructions and in `.gitignore`, which is correct.

---

## 6. Code Quality Assessment

**Structure:** Exceptionally clean separation of concerns across `data/`, `training/`, `eval/`, `serve/`, `demo/`, and `notebooks/`. The `config.yaml` centralizes all hyperparameters — a good practice that makes experiments reproducible.

**Documentation:** `document.md` is a thorough operational guide covering every phase from environment setup through GGUF conversion. README is well-structured with architecture diagram, before/after examples, and hardware table.

**Test Coverage:** `tests/test_utils.py` covers CI-safe utility tests (no GPU required). Core training and inference logic is not covered by tests, which is typical for ML projects but is worth noting.

**Docker:** Multi-stage Dockerfiles for both serve and demo layers. `docker-compose.yml` orchestrates both.

**CI/CD:** GitHub Actions with Python lint and unit tests. W&B experiment tracking. Published models on HuggingFace Hub.

---

## 7. Maturity Breakdown

| Dimension | Score | Notes |
|-----------|-------|-------|
| Functionality | 9/10 | Full pipeline: data → train → eval → serve → publish |
| Code Quality | 8/10 | Exceptionally clean project structure; minor doc inconsistencies |
| Documentation | 8/10 | Thorough dual-source docs; 1-vs-3 epoch discrepancy should be resolved |
| Scalability | 6/10 | Single-worker FastAPI; 512-token context limits real-world snippets |
| Security | 7/10 | .env properly excluded; no auth on FastAPI endpoint |
| **Overall** | **7.6/10** | One of the stronger portfolio entries; end-to-end MLOps discipline is notable |

---

## 8. Suggested Next Steps

1. **Expand BLEU evaluation to the full test set (n=1,000)** and add per-category breakdown (Python vs JavaScript, simple bugs vs logic errors). Report the LLM-judge rubric scores (correctness, clarity) as primary metrics in the README — they are more meaningful than BLEU.
2. **Resolve the 1-epoch vs 3-epoch discrepancy** between README and `document.md`, and train a proper 3-epoch run to compare eval loss and BLEU improvement. Checkpoint the best epoch rather than the last.
3. **Extend LoRA target modules** to include `k_proj`, `o_proj`, and potentially `gate_proj`/`up_proj`/`down_proj` for the MLP — this typically improves task-specific semantic alignment.

---

## 9. Verdict

Code Autopsy is the strongest MLOps end-to-end project in the portfolio. The complete pipeline — three-source dataset engineering, QLoRA fine-tuning on consumer GPU, dual-track evaluation, multi-path serving (FastAPI, Gradio, Ollama), and HuggingFace Hub publication — demonstrates genuine production engineering discipline. The 558% BLEU improvement is a compelling headline metric, though it should be contextualized as primarily measuring format compliance. The main areas for improvement are evaluation depth (larger test set, per-category breakdown) and minor documentation consistency issues.
