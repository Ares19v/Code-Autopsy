#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# convert_to_gguf.sh
# Converts the merged HuggingFace model to GGUF format for Ollama.
# Requires llama.cpp to be installed.
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

MERGED_DIR="./merged_model"
GGUF_DIR="./merged_model"
GGUF_NAME="code-autopsy-7b.Q4_K_M.gguf"
LLAMA_CPP="${LLAMA_CPP_PATH:-$HOME/llama.cpp}"

echo "🔬 Code Autopsy — GGUF Conversion"
echo "──────────────────────────────────"

# ── Check llama.cpp exists ────────────────────────────────────────────────────
if [ ! -f "$LLAMA_CPP/convert_hf_to_gguf.py" ]; then
    echo "⚠  llama.cpp not found at: $LLAMA_CPP"
    echo ""
    echo "   Install it:"
    echo "   git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp"
    echo "   cd ~/llama.cpp && pip install -r requirements.txt"
    echo ""
    echo "   Then re-run: bash convert_to_gguf.sh"
    exit 1
fi

# ── Check merged model exists ─────────────────────────────────────────────────
if [ ! -d "$MERGED_DIR" ]; then
    echo "⚠  Merged model not found at $MERGED_DIR"
    echo "   Run first: python publish.py --dry-run=false (or just python publish.py)"
    exit 1
fi

echo "✓  Merged model found: $MERGED_DIR"
echo ""

# ── Step 1: Convert to f16 GGUF ──────────────────────────────────────────────
echo "→  Converting to GGUF (f16)..."
python "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outfile "$GGUF_DIR/code-autopsy-7b.f16.gguf" \
    --outtype f16

echo "✓  f16 GGUF created"

# ── Step 2: Quantize to Q4_K_M ───────────────────────────────────────────────
echo "→  Quantizing to Q4_K_M (recommended for RTX 5060)..."
"$LLAMA_CPP/llama-quantize" \
    "$GGUF_DIR/code-autopsy-7b.f16.gguf" \
    "$GGUF_DIR/$GGUF_NAME" \
    Q4_K_M

echo "✓  Quantized GGUF created: $GGUF_DIR/$GGUF_NAME"
echo ""

# ── Step 3: Create Ollama model ───────────────────────────────────────────────
echo "→  Creating Ollama model: code-autopsy..."
ollama create code-autopsy -f Modelfile

echo ""
echo "🎉 Done! Run your model with:"
echo "   ollama run code-autopsy"
echo ""
echo "   Or via REST API:"
echo "   curl http://localhost:11434/api/generate -d '{\"model\": \"code-autopsy\", \"prompt\": \"Review this Python code: def avg(x): return sum(x)/len(x)\"}'"
