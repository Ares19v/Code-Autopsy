# CODE AUTOPSY — PROJECT JOURNAL
### From zero to a fine-tuned code review model
---

> **Model:** Qwen2.5-Coder-7B-Instruct  
> **Method:** QLoRA · NF4 4-bit · PEFT  
> **Hardware:** HP Omen · RTX 5060  
> **HuggingFace:** Ares19v/code-autopsy-7b  
> **GitHub:** https://github.com/Ares19v/Code-Autopsy  

---

## WHAT THIS PROJECT IS

Code Autopsy is a fine-tuned LLM that takes buggy Python or JavaScript code as input
and returns a structured three-part analysis:

    1. Bug Identified  — what the bug is
    2. Root Cause      — why it happens
    3. Fixed Code      — corrected version with explanation

It is built by QLoRA fine-tuning Qwen2.5-Coder-7B-Instruct on curated
code review datasets, served via FastAPI, and demoed through a Gradio UI
deployed to HuggingFace Spaces.

---

## TECH STACK

    Base model         Qwen/Qwen2.5-Coder-7B-Instruct (HuggingFace)
    Fine-tuning        QLoRA via PEFT library
    Training framework TRL with SFTTrainer
    Quantization       BitsAndBytes — 4-bit NF4
    Datasets           CodeSearchNet, CodeXGLUE Code Refinement, MS CodeReviewer
    Experiment track   Weights & Biases (wandb)
    Evaluation         CodeBLEU + LLM-as-judge via Gemini API
    Serving            FastAPI (REST) + Ollama (local GGUF)
    Demo UI            Gradio — deployed to HuggingFace Spaces
    Containerization   Docker
    Python             3.11

---

## REPO STRUCTURE

    Code-Autopsy/
    ├── data/
    │   ├── raw/                       ← gitignored, downloaded datasets land here
    │   ├── processed/                 ← gitignored, formatted JSONL splits
    │   └── prepare_dataset.py         ← downloads + formats all training data
    │
    ├── training/
    │   ├── config.yaml                ← ALL hyperparameters live here
    │   ├── train.py                   ← main QLoRA training script
    │   └── utils.py                   ← shared helpers (format, parse, confidence)
    │
    ├── eval/
    │   ├── eval_codebleu.py           ← base vs fine-tuned CodeBLEU delta table
    │   └── eval_llm_judge.py          ← Gemini judges 100 examples, logs to W&B
    │
    ├── serve/
    │   ├── api.py                     ← FastAPI: GET /health + POST /review
    │   └── Dockerfile                 ← containerized serve layer
    │
    ├── demo/
    │   └── app.py                     ← Gradio UI (dark theme, calls API_URL)
    │
    ├── notebooks/
    │   └── exploration_notes.md       ← notebook templates for exploration
    │
    ├── publish.py                     ← merge adapter → push to HF Hub
    ├── Modelfile                      ← Ollama model definition
    ├── convert_to_gguf.sh             ← converts merged model to GGUF
    ├── requirements.txt
    ├── .env.example
    ├── .env                           ← your actual keys (gitignored)
    └── README.md

---

## TRAINING CONFIGURATION

    LoRA rank              16
    LoRA alpha             32
    LoRA dropout           0.05
    Target modules         q_proj, v_proj
    Batch size             4 per device
    Gradient accumulation  4 steps  →  effective batch = 16
    Learning rate          2e-4
    LR scheduler           cosine
    Max sequence length    1024 tokens
    Epochs                 3
    Checkpoint every       500 steps
    Precision              bfloat16
    Optimizer              paged_adamw_8bit

---

## DATASET PIPELINE

    Source 1   code_x_glue_cc_code_refinement  — real buggy/fixed pairs
    Source 2   code_search_net                 — Python + JS functions
    Source 3   microsoft/codereview-data       — code review comments + diffs

    Each example is formatted as a 3-turn chat:

        [system]     You are a code review expert...
        [user]       Language: python\n\n```python\n<buggy code>\n```
        [assistant]  ## Bug Identified\n...\n## Root Cause\n...\n## Fixed Code\n```python\n...\n```

    Split: 90% train / 5% val / 5% test

---

## API SPEC

    GET  /health
         → { status, model, adapter_loaded, device, vram_allocated_gb }

    POST /review
         Body:     { "code": string, "language": "python" | "javascript" }
         Response: { "bug_identified": string,
                     "root_cause":     string,
                     "fixed_code":     string,
                     "confidence":     float,    ← avg log-probability, exp-scaled to [0,1]
                     "language":       string,
                     "latency_ms":     float }

    Port: 8000

---

## EVALUATION SPEC

    Track 1 — CodeBLEU
        Run inference on test set with base model and fine-tuned model.
        Compare CodeBLEU scores. Report delta table. Log to W&B.

    Track 2 — LLM-as-Judge (Gemini)
        Sample 100 test examples.
        Send each model output to gemini-1.5-flash with rubric:
            - bug_correctly_identified   true / false
            - explanation_clarity        1–5
            - fix_correctness            1–5
        Aggregate scores. Log full per-example breakdown as W&B Table.

---

## ENVIRONMENT VARIABLES  (.env)

    HF_TOKEN          HuggingFace write token      — push models to Ares19v/*
    WANDB_API_KEY     Weights & Biases API key     — experiment tracking
    GEMINI_API_KEY    Gemini API key               — LLM-as-judge evaluation
    GITHUB_TOKEN      GitHub token                 — optional
    WANDB_PROJECT     code-autopsy                 — already set
    API_URL           http://localhost:8000        — Gradio calls this

---
---

# EXECUTION GUIDE — STEP BY STEP

---

## PHASE 0 — OPEN YOUR POWERSHELL

    Open PowerShell and navigate to the project:

        cd "C:\Users\Devansh Tyagi\Desktop\Projects\Code-Autopsy"

    Every command in this guide is run from this directory.

---

## PHASE 1 — ENVIRONMENT SETUP

    STEP 1.1   Create virtual environment

        python -m venv .venv

    STEP 1.2   Activate it

        .venv\Scripts\activate

        You will see (.venv) at the start of your prompt.
        Do this every time you open a new terminal for this project.

    STEP 1.3   Upgrade pip

        pip install --upgrade pip

---

## PHASE 2 — INSTALL PYTORCH (RTX 5060 SPECIFIC)

    The RTX 5060 is Blackwell architecture (sm_120).
    It requires CUDA 12.8. The default pip torch is CPU-only.
    ALWAYS install PyTorch separately and first.

    STEP 2.1   Install PyTorch with CUDA 12.8

        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    STEP 2.2   Verify GPU is detected

        python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

        Expected output:
            2.x.x+cu128
            True
            NVIDIA GeForce RTX 5060

        If you see False or an error, do not proceed. Fix CUDA first.

---

## PHASE 3 — INSTALL REMAINING REQUIREMENTS

    STEP 3.1   Install everything else

        pip install -r requirements.txt

    STEP 3.2   If bitsandbytes fails on Windows (common), run this instead:

        pip install bitsandbytes --prefer-binary --index-url https://huggingface.github.io/bitsandbytes-windows-webui

    STEP 3.3   Verify bitsandbytes loaded correctly

        python -c "import bitsandbytes as bnb; print(bnb.__version__)"

        Should print a version number, not an error.

---

## PHASE 4 — VALIDATE ENVIRONMENT KEYS

    STEP 4.1   Confirm .env keys are loaded

        python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('HF:    ', os.getenv('HF_TOKEN','MISSING')[:10]+'...'); print('WANDB: ', os.getenv('WANDB_API_KEY','MISSING')[:10]+'...'); print('GEMINI:', os.getenv('GEMINI_API_KEY','MISSING')[:10]+'...')"

        Should print the first 10 chars of each key — not MISSING.

---

## PHASE 5 — DATASET PREPARATION

    STEP 5.1   Dry run — no downloads, just validates the formatting logic

        python data\prepare_dataset.py --dry-run

        Expected: prints a table showing 2 synthetic records split across train/val/test.
        If this works, your Python path and imports are correct.

    STEP 5.2   Full dataset download and preparation

        python data\prepare_dataset.py --max-samples 20000

        What happens:
            - Downloads CodeSearchNet (Python + JS)       ~2 GB
            - Downloads CodeXGLUE Code Refinement         ~200 MB
            - Downloads Microsoft CodeReviewer            ~500 MB
            - Deduplicates by code hash
            - Formats into chat template
            - Shuffles with seed 42
            - Splits 90/5/5

        Output files:
            data\processed\train.jsonl    ~18,000 examples
            data\processed\val.jsonl      ~1,000 examples
            data\processed\test.jsonl     ~1,000 examples

        Time: 10–20 minutes depending on internet speed.

---

## PHASE 6 — TRAINING

    STEP 6.1   Debug run — 1 step only, no W&B, validates VRAM fits

        python training\train.py --debug

        What to look for:
            - Model loads without OOM error
            - Trainable parameters: ~13M (0.18% of 7B)
            - Loss prints for 1 step
            - "DEBUG run complete — adapter not saved"

        If this completes, you have enough VRAM. Move to 6.2.
        If you get CUDA OOM, reduce per_device_train_batch_size to 2 in config.yaml.

    STEP 6.2   Full training run

        python training\train.py

        What happens:
            - Logs to your W&B project: https://wandb.ai/Ares19v/code-autopsy
            - Saves checkpoints to .\checkpoints\ every 500 steps
            - Saves final adapter to .\adapter\ when done
            - Runs 3 epochs over ~18,000 examples

        Estimated time:   4–8 hours on RTX 5060
        Expected final loss:   ~0.8–1.2

        You can monitor live at wandb.ai.
        Do not close PowerShell while training. Let it run.

    After training you will have:
        .\adapter\                    ← your LoRA adapter weights
        .\checkpoints\checkpoint-*/  ← intermediate checkpoints

---

## PHASE 7 — EVALUATION

    STEP 7.1   CodeBLEU — quantitative comparison

        python eval\eval_codebleu.py --adapter .\adapter

        Prints a table like:
            Model                     | CodeBLEU
            Base (Qwen2.5-Coder-7B)   | 0.1842
            Fine-tuned (Code-Autopsy) | 0.2614
            Delta                     | +0.0772

        Also logs to W&B.

    STEP 7.2   LLM-as-Judge — qualitative scoring via Gemini

        python eval\eval_llm_judge.py --adapter .\adapter

        Samples 100 test examples.
        Sends each to Gemini 1.5 Flash with rubric.
        Prints summary table + logs full breakdown to W&B.
        Takes ~10 minutes (rate-limited to 15 req/min).

---

## PHASE 8 — SERVING

    STEP 8.1   Start the FastAPI server

        uvicorn serve.api:app --host 0.0.0.0 --port 8000

        Keep this terminal open. Server is ready when you see:
            INFO: Application startup complete.

    STEP 8.2   Test the API

        Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get

    STEP 8.3   Test a real review

        $body = '{"code": "def avg(x): return sum(x)/len(x)", "language": "python"}'
        Invoke-RestMethod -Uri "http://localhost:8000/review" -Method Post -Body $body -ContentType "application/json"

    STEP 8.4   Launch Gradio demo (open a NEW terminal, activate .venv again)

        .venv\Scripts\activate
        python demo\app.py

        Opens at: http://localhost:7860
        Make sure the API server from Step 8.1 is still running.

---

## PHASE 9 — DOCKER (OPTIONAL)

    STEP 9.1   Build the image

        docker build -t code-autopsy-serve .\serve

    STEP 9.2   Run with GPU and adapter mounted

        docker run -p 8000:8000 `
          -v ${PWD}\adapter:/app/adapter:ro `
          --env-file .env `
          --gpus all `
          code-autopsy-serve

---

## PHASE 10 — OLLAMA (LOCAL GGUF)

    STEP 10.1   Make sure llama.cpp is installed on your machine
                https://github.com/ggerganov/llama.cpp

    STEP 10.2   Run the GGUF conversion script

        bash convert_to_gguf.sh

        This will:
            - Convert merged_model/ to f16 GGUF
            - Quantize to Q4_K_M
            - Register with Ollama as "code-autopsy"

    STEP 10.3   Run via Ollama

        ollama run code-autopsy

---

## PHASE 11 — PUBLISH TO HUGGINGFACE

    STEP 11.1   Dry run first — validates without pushing

        python publish.py --dry-run

    STEP 11.2   Push everything

        python publish.py

        Publishes:
            Ares19v/code-autopsy-7b     merged full model + model card
            Ares19v/code-autopsy-lora   raw LoRA adapter

    STEP 11.3   Deploy Gradio to HuggingFace Spaces
                - Go to https://huggingface.co/new-space
                - Name: code-autopsy
                - SDK: Gradio
                - Upload: demo/app.py + requirements (gradio, requests)
                - Set Space secret: API_URL = your hosted FastAPI URL

---

## PHASE 12 — FILL IN README RESULTS

    After training and eval, update README.md:

        1. Paste your W&B training loss curve screenshot
        2. Fill in the CodeBLEU results table
        3. Fill in the LLM-judge score summary table
        4. Add your HF Spaces live demo link
        5. Add your model card link

---
---

# QUICK REFERENCE — ALL COMMANDS

    Setup
        python -m venv .venv
        .venv\Scripts\activate
        pip install --upgrade pip
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        pip install -r requirements.txt

    Validate
        python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
        python data\prepare_dataset.py --dry-run

    Data
        python data\prepare_dataset.py --max-samples 20000

    Train
        python training\train.py --debug        ← test run (1 step)
        python training\train.py                ← full training

    Evaluate
        python eval\eval_codebleu.py --adapter .\adapter
        python eval\eval_llm_judge.py --adapter .\adapter

    Serve
        uvicorn serve.api:app --host 0.0.0.0 --port 8000
        python demo\app.py

    Publish
        python publish.py --dry-run
        python publish.py
        bash convert_to_gguf.sh

---
---

# NOTES & GOTCHAS

    1. ALWAYS install PyTorch with --index-url https://download.pytorch.org/whl/cu128
       BEFORE pip install -r requirements.txt. Order matters.

    2. bitsandbytes on Windows can be tricky. If it errors, use the
       HuggingFace Windows wheel index:
       pip install bitsandbytes --prefer-binary --index-url https://huggingface.github.io/bitsandbytes-windows-webui

    3. RTX 5060 is Blackwell (sm_120). Needs PyTorch 2.6+ and CUDA 12.8+.
       Older PyTorch builds will not recognize the GPU.

    4. Do not bake model weights into Docker. Always mount as a volume.

    5. HF Spaces free tier is CPU-only. The Gradio demo calls your FastAPI
       via API_URL — it does not run the model on the Space itself.

    6. The Gemini LLM-judge is rate limited to 15 req/min on the free tier.
       The eval script sleeps 4 seconds between calls automatically.

    7. Training takes 4–8 hours. Run it overnight.
       Do not close PowerShell mid-training. Use Windows + L to lock screen.

    8. After training, always run the debug eval first:
       python eval\eval_codebleu.py --max-samples 20 --no-wandb
       to confirm the adapter loads and inference works before full eval.

---

                                                            — Ares19v / Code Autopsy
