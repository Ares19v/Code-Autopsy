"""
demo/app.py
───────────
Code Autopsy — Gradio demo UI.

Two-column dark-themed interface:
  Left:  code input (syntax highlighted) + language selector + "Run Autopsy" button
  Right: Bug Identified / Root Cause / Fixed Code panels + confidence meter

Calls the FastAPI backend at API_URL (configurable via .env).
Deployable to HuggingFace Spaces (free tier) — the Space just calls your
hosted FastAPI endpoint, it doesn't run the model itself.

Usage:
  python demo/app.py                          # Calls http://localhost:8000
  API_URL=https://your.server.com python demo/app.py
"""

from __future__ import annotations

import os

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ══════════════════════════════════════════════════════════════════════════════
# Custom CSS — premium dark theme
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;1,400&display=swap');

/* ── Base reset ─────────────────────────────────────────────────────────── */
:root {
    --bg-root:       #070b14;
    --bg-card:       #0d1220;
    --bg-input:      #111827;
    --bg-hover:      #161f32;
    --border:        rgba(99, 102, 241, 0.18);
    --border-bright: rgba(99, 102, 241, 0.45);
    --accent-1:      #6366f1;
    --accent-2:      #8b5cf6;
    --accent-3:      #06b6d4;
    --text-primary:  #e2e8f0;
    --text-muted:    #94a3b8;
    --text-dim:      #475569;
    --success:       #10b981;
    --warning:       #f59e0b;
    --error:         #f43f5e;
    --radius-sm:     6px;
    --radius-md:     10px;
    --radius-lg:     16px;
    --shadow-glow:   0 0 40px rgba(99,102,241,0.12);
    --shadow-card:   0 4px 24px rgba(0,0,0,0.4);
    --transition:    all 0.2s cubic-bezier(0.4,0,0.2,1);
}

*, *::before, *::after { box-sizing: border-box; }

body,
.gradio-container,
.gradio-container > .main,
footer { 
    background: var(--bg-root) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

/* Hide Gradio footer */
footer { display: none !important; }

/* ── Header ─────────────────────────────────────────────────────────────── */
.autopsy-hero {
    text-align: center;
    padding: 3rem 2rem 2rem;
    background: 
        radial-gradient(ellipse 60% 40% at 50% 0%, rgba(99,102,241,0.12) 0%, transparent 70%),
        linear-gradient(180deg, rgba(13,18,32,0.8) 0%, transparent 100%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 0;
    position: relative;
    overflow: hidden;
}

.autopsy-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 39px,
        rgba(99,102,241,0.04) 39px, rgba(99,102,241,0.04) 40px
    ),
    repeating-linear-gradient(
        90deg, transparent, transparent 39px,
        rgba(99,102,241,0.04) 39px, rgba(99,102,241,0.04) 40px
    );
    pointer-events: none;
}

.autopsy-logo {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
}

.autopsy-icon {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 0 20px rgba(99,102,241,0.4);
}

.autopsy-title {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 30%, #c084fc 60%, #67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0;
}

.autopsy-subtitle {
    color: var(--text-muted);
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0.5rem 0 1.25rem;
    letter-spacing: 0.01em;
}

.badge-row {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    flex-wrap: wrap;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 1px solid;
}

.badge-model  { background: rgba(99,102,241,0.1);  border-color: rgba(99,102,241,0.3);  color: #a5b4fc; }
.badge-method { background: rgba(139,92,246,0.1);  border-color: rgba(139,92,246,0.3);  color: #c4b5fd; }
.badge-lang   { background: rgba(6,182,212,0.1);   border-color: rgba(6,182,212,0.3);   color: #67e8f9; }

/* ── Main layout ────────────────────────────────────────────────────────── */
.main-panel {
    padding: 1.5rem 2rem 2rem;
    max-width: 1400px;
    margin: 0 auto;
}

/* ── Section labels ─────────────────────────────────────────────────────── */
.section-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-dim);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Override Gradio blocks ─────────────────────────────────────────────── */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}

/* Code/text inputs */
.gradio-container textarea,
.gradio-container .cm-editor,
.gradio-container .cm-scroller {
    background: var(--bg-input) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

.gradio-container textarea:focus,
.gradio-container .cm-focused {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    outline: none !important;
}

/* Dropdown */
.gradio-container select,
.gradio-container .wrap {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
}

/* Labels */
.gradio-container label,
.gradio-container .label-wrap span,
.gradio-container .svelte-1gfkn6j {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.run-btn > button,
button.run-btn {
    background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.8rem 1.5rem !important;
    cursor: pointer !important;
    transition: var(--transition) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    width: 100% !important;
    font-family: 'Inter', sans-serif !important;
}

.run-btn > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.5), inset 0 1px 0 rgba(255,255,255,0.15) !important;
}

.run-btn > button:active {
    transform: translateY(0) !important;
}

.clear-btn > button,
button.clear-btn {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: var(--transition) !important;
    font-family: 'Inter', sans-serif !important;
}

.clear-btn > button:hover {
    background: var(--bg-hover) !important;
    border-color: var(--border-bright) !important;
    color: var(--text-primary) !important;
}

.refresh-btn > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-dim) !important;
    font-size: 0.75rem !important;
    padding: 0.3rem 0.7rem !important;
    transition: var(--transition) !important;
    font-family: 'Inter', sans-serif !important;
}

.refresh-btn > button:hover {
    color: var(--accent-3) !important;
    border-color: var(--accent-3) !important;
}

/* ── Output panels ──────────────────────────────────────────────────────── */
.output-bug      .block { border-left: 3px solid var(--error) !important; }
.output-cause    .block { border-left: 3px solid var(--warning) !important; }
.output-fix      .block { border-left: 3px solid var(--success) !important; }
.output-conf     .block { border-left: 3px solid var(--accent-3) !important; }

/* ── Status pill ────────────────────────────────────────────────────────── */
.status-box input,
.status-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--text-muted) !important;
}

/* ── Examples ───────────────────────────────────────────────────────────── */
.gr-samples-table {
    background: var(--bg-card) !important;
    border-radius: var(--radius-md) !important;
}

/* ── Confidence meter (custom HTML) ─────────────────────────────────────── */
.conf-meter {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.9rem 1.1rem;
    margin-top: 0.25rem;
}

.conf-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    margin-bottom: 0.5rem;
}

.conf-value {
    font-size: 1.8rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-3);
    line-height: 1;
}

.conf-bar-track {
    width: 100%;
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    margin-top: 0.6rem;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent-1), var(--accent-3));
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar            { width: 5px; height: 5px; }
::-webkit-scrollbar-track      { background: var(--bg-root); }
::-webkit-scrollbar-thumb      { background: rgba(99,102,241,0.3); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover{ background: var(--accent-1); }

/* ── Footer ─────────────────────────────────────────────────────────────── */
.demo-footer {
    text-align: center;
    padding: 1.5rem 1rem 2rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}

.demo-footer p {
    color: var(--text-dim);
    font-size: 0.8rem;
    margin: 0;
}

.demo-footer a {
    color: var(--accent-1);
    text-decoration: none;
    font-weight: 500;
    transition: color 0.15s;
}

.demo-footer a:hover { color: var(--accent-3); }

/* ── Loading animation ──────────────────────────────────────────────────── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(99,102,241,0.2); }
    50%       { box-shadow: 0 0 40px rgba(99,102,241,0.5); }
}

.generating .run-btn > button {
    animation: pulse-glow 1.5s ease-in-out infinite !important;
}
"""

# ══════════════════════════════════════════════════════════════════════════════
# Example snippets
# ══════════════════════════════════════════════════════════════════════════════

EXAMPLE_PYTHON_DIVIDE = """\
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)

# This crashes on empty list
print(calculate_average([]))"""

EXAMPLE_PYTHON_MUTABLE = """\
def append_to_list(value, lst=[]):
    lst.append(value)
    return lst

# Surprising behavior:
print(append_to_list(1))  # [1]
print(append_to_list(2))  # [1, 2]  <-- unexpected!"""

EXAMPLE_JS_AWAIT = """\
async function fetchUserData(userId) {
    const response = await fetch(`/api/users/${userId}`);
    const data = response.json();  // Missing await!
    return data.name;
}"""

EXAMPLE_JS_CLOSURE = """\
for (var i = 0; i < 3; i++) {
    setTimeout(function() {
        console.log(i);  // Prints 3, 3, 3 — not 0, 1, 2
    }, 1000);
}"""


# ══════════════════════════════════════════════════════════════════════════════
# API helpers
# ══════════════════════════════════════════════════════════════════════════════

def check_health() -> str:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            model_short = data.get("model", "unknown").split("/")[-1]
            vram = data.get("vram_allocated_gb")
            vram_str = f" · {vram}GB VRAM" if vram else ""
            adapter = "✓ adapter" if data.get("adapter_loaded") else "base model"
            return f"🟢  {model_short} · {adapter}{vram_str}"
        return f"🔴  API error ({r.status_code})"
    except requests.exceptions.ConnectionError:
        return f"🔴  Offline — start server at {API_URL}"
    except Exception as e:
        return f"🔴  {e}"


def run_autopsy(code: str, language: str):
    """Main inference function called by the Gradio button."""
    if not code or not code.strip():
        return (
            "⚠️  Please paste some code to analyze.",
            "", "", build_confidence_html(None),
        )

    try:
        r = requests.post(
            f"{API_URL}/review",
            json={"code": code.strip(), "language": language.lower()},
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()

        confidence = data.get("confidence", 0.0)
        latency = data.get("latency_ms", 0)

        return (
            data.get("bug_identified", "No bug information returned."),
            data.get("root_cause", "No root cause information returned."),
            data.get("fixed_code", "# No fix returned."),
            build_confidence_html(confidence, latency_ms=latency),
        )

    except requests.exceptions.ConnectionError:
        return (
            "❌  Cannot connect to API server.",
            f"Ensure FastAPI is running:\n  uvicorn serve.api:app --port 8000\n\nAPI URL: {API_URL}",
            "",
            build_confidence_html(None),
        )
    except requests.exceptions.Timeout:
        return (
            "⏱️  Request timed out (90s).",
            "The model is taking too long. Try a shorter snippet.",
            "",
            build_confidence_html(None),
        )
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = r.json().get("detail", "")
        except Exception:
            pass
        return (
            f"❌  API error {r.status_code}",
            detail or str(e),
            "",
            build_confidence_html(None),
        )
    except Exception as e:
        return (f"❌  Unexpected error: {e}", "", "", build_confidence_html(None))


def build_confidence_html(confidence: float | None, latency_ms: float = 0) -> str:
    if confidence is None:
        pct_str = "—"
        bar_pct = 0
        color = "#475569"
    else:
        pct_str = f"{confidence:.1%}"
        bar_pct = int(confidence * 100)
        if confidence >= 0.7:
            color = "#10b981"
        elif confidence >= 0.4:
            color = "#f59e0b"
        else:
            color = "#f43f5e"

    latency_str = f"<span style='color:#475569; font-size:0.72rem; margin-left:0.8rem'>{latency_ms:.0f}ms</span>" if latency_ms else ""

    return f"""
<div class="conf-meter">
    <div class="conf-label">📊 Confidence Score</div>
    <div style="display:flex; align-items:baseline; gap:0.3rem">
        <span class="conf-value" style="color:{color}">{pct_str}</span>
        {latency_str}
    </div>
    <div class="conf-bar-track">
        <div class="conf-bar-fill" style="width:{bar_pct}%; background: linear-gradient(90deg, {color}88, {color})"></div>
    </div>
    <div style="font-size:0.68rem; color:#475569; margin-top:0.4rem">
        Average token log-probability (exp scale)
    </div>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(
    css=CUSTOM_CSS,
    title="Code Autopsy — AI Code Review",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#070b14",
        block_background_fill="#0d1220",
        block_border_color="rgba(99,102,241,0.18)",
        block_border_width="1px",
        block_label_text_color="#94a3b8",
        block_label_text_size="0.75rem",
        input_background_fill="#111827",
        input_border_color="rgba(99,102,241,0.2)",
        input_border_width="1px",
        button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
        button_primary_background_fill_hover="linear-gradient(135deg, #818cf8, #a78bfa)",
        button_primary_text_color="white",
        button_secondary_background_fill="#111827",
        button_secondary_border_color="rgba(99,102,241,0.25)",
        button_secondary_text_color="#94a3b8",
    ),
) as demo:

    # ── Hero header ───────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="autopsy-hero">
        <div class="autopsy-logo">
            <div class="autopsy-icon">🔬</div>
        </div>
        <h1 class="autopsy-title">Code Autopsy</h1>
        <p class="autopsy-subtitle">
            QLoRA fine-tuned code review · Bug ID · Root cause · Auto-fix
        </p>
        <div class="badge-row">
            <span class="badge badge-model">🤗 Qwen2.5-Coder-7B</span>
            <span class="badge badge-method">⚡ QLoRA · NF4</span>
            <span class="badge badge-lang">🐍 Python</span>
            <span class="badge badge-lang">☕ JavaScript</span>
        </div>
    </div>
    """)

    with gr.Row(elem_classes=["main-panel"]):

        # ── LEFT: Input column ────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=380):

            gr.HTML('<div class="section-label">🖊 Input</div>')

            language = gr.Dropdown(
                choices=["Python", "JavaScript"],
                value="Python",
                label="Language",
                elem_id="language-dropdown",
                container=True,
            )

            code_input = gr.Code(
                label="Buggy Code",
                language="python",
                lines=18,
                elem_id="code-input",
            )

            with gr.Row():
                run_btn = gr.Button(
                    "🔬  Run Autopsy",
                    variant="primary",
                    elem_classes=["run-btn"],
                    size="lg",
                )
                clear_btn = gr.Button(
                    "✕  Clear",
                    variant="secondary",
                    elem_classes=["clear-btn"],
                    size="lg",
                )

            # API status
            gr.HTML('<div class="section-label" style="margin-top:1rem">⚡ API Status</div>')
            with gr.Row():
                status_box = gr.Textbox(
                    value=check_health,
                    label="",
                    interactive=False,
                    container=False,
                    elem_classes=["status-box"],
                    scale=4,
                )
                refresh_btn = gr.Button("↺", elem_classes=["refresh-btn"], scale=1, size="sm")

            # Examples
            gr.HTML('<div class="section-label" style="margin-top:1.25rem">⚡ Quick Examples</div>')
            gr.Examples(
                examples=[
                    [EXAMPLE_PYTHON_DIVIDE,  "Python"],
                    [EXAMPLE_PYTHON_MUTABLE, "Python"],
                    [EXAMPLE_JS_AWAIT,       "JavaScript"],
                    [EXAMPLE_JS_CLOSURE,     "JavaScript"],
                ],
                inputs=[code_input, language],
                label="",
                examples_per_page=4,
            )

        # ── RIGHT: Output column ──────────────────────────────────────────────
        with gr.Column(scale=1, min_width=380):

            gr.HTML('<div class="section-label">🧬 Analysis</div>')

            bug_output = gr.Textbox(
                label="🐛  Bug Identified",
                lines=4,
                interactive=False,
                placeholder="Bug analysis will appear here after running Autopsy...",
                elem_classes=["output-bug"],
            )

            root_cause_output = gr.Textbox(
                label="🔍  Root Cause",
                lines=5,
                interactive=False,
                placeholder="Root cause explanation will appear here...",
                elem_classes=["output-cause"],
            )

            fixed_code_output = gr.Code(
                label="✅  Fixed Code",
                language="python",
                lines=11,
                interactive=False,
                elem_classes=["output-fix"],
            )

            confidence_display = gr.HTML(
                value=build_confidence_html(None),
                elem_classes=["output-conf"],
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="demo-footer">
        <p>
            <strong style="color:#94a3b8">Code Autopsy</strong> &nbsp;·&nbsp;
            Built by <a href="https://huggingface.co/Ares19v">Ares19v</a> &nbsp;·&nbsp;
            Model: <a href="https://huggingface.co/Ares19v/code-autopsy-7b">Ares19v/code-autopsy-7b</a> &nbsp;·&nbsp;
            <a href="https://github.com/Ares19v/Code-Autopsy">GitHub</a>
        </p>
    </div>
    """)

    # ══════════════════════════════════════════════════════════════════════════
    # Event wiring
    # ══════════════════════════════════════════════════════════════════════════

    def sync_code_language(lang: str):
        """Keep code editor and fixed code panel in sync with selected language."""
        l = lang.lower() if lang.lower() in ("python", "javascript") else "python"
        return gr.Code(language=l), gr.Code(language=l)

    language.change(
        fn=sync_code_language,
        inputs=[language],
        outputs=[code_input, fixed_code_output],
    )

    run_btn.click(
        fn=run_autopsy,
        inputs=[code_input, language],
        outputs=[bug_output, root_cause_output, fixed_code_output, confidence_display],
        api_name="review",
    )

    def clear_all():
        return (
            gr.Code(value=""),
            "",
            "",
            gr.Code(value=""),
            build_confidence_html(None),
        )

    clear_btn.click(
        fn=clear_all,
        outputs=[code_input, bug_output, root_cause_output, fixed_code_output, confidence_display],
    )

    refresh_btn.click(fn=check_health, outputs=[status_box])


# ══════════════════════════════════════════════════════════════════════════════
# Launch
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        show_api=True,
        show_error=True,
        favicon_path=None,
    )
