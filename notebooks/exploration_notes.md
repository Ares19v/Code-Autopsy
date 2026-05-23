# Code Autopsy Exploration Notes

This directory is reserved for exploratory Jupyter notebooks.

## Recommended notebooks to create here

| Notebook | Purpose |
|---|---|
| `01_dataset_exploration.ipynb` | Inspect raw + formatted dataset splits, visualize token length distribution, check class balance across languages |
| `02_base_model_sanity_check.ipynb` | Run a few inference examples on the base model before fine-tuning to establish baseline quality |
| `03_training_curves.ipynb` | Load W&B run history, plot train/val loss curves, learning rate schedule |
| `04_eval_deep_dive.ipynb` | Qualitative analysis of 20 examples: side-by-side base vs fine-tuned output |
| `05_codebleu_breakdown.ipynb` | Per-language CodeBLEU breakdown, error analysis on low-scoring examples |

## Quick start

```bash
pip install jupyter
jupyter notebook notebooks/
```

## Token length analysis snippet

```python
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

with open("../data/processed/train.jsonl") as f:
    records = [json.loads(l) for l in f]

lengths = [len(tokenizer.encode(r["text"])) for r in records]

plt.figure(figsize=(10, 4))
plt.hist(lengths, bins=50, color="#6366f1", edgecolor="none", alpha=0.85)
plt.axvline(1024, color="#f43f5e", linestyle="--", label="max_seq_length=1024")
plt.xlabel("Token count")
plt.ylabel("Count")
plt.title("Training example token length distribution")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Median: {sorted(lengths)[len(lengths)//2]}")
print(f"p95:    {sorted(lengths)[int(len(lengths)*0.95)]}")
print(f"Over limit: {sum(1 for l in lengths if l > 1024)} / {len(lengths)}")
```
