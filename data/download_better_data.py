import json
import re
from datasets import load_dataset
from pathlib import Path

# Create directories
out_dir = Path("data/processed_v2")
out_dir.mkdir(parents=True, exist_ok=True)

# System Prompt required by the user's setup
SYSTEM_PROMPT = (
    "You are a code review expert. Analyze the provided code, identify any bugs "
    "or issues, explain the root cause, and provide a corrected version."
)

def format_sample(buggy_code, language, bug_identified, root_cause, fixed_code):
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Language: {language}

```{language}
{buggy_code}
```<|im_end|>
<|im_start|>assistant
## Bug Identified
{bug_identified}

## Root Cause
{root_cause}

## Fixed Code
```{language}
{fixed_code}
```<|im_end|>
"""

print("Downloading dataset...")
# Load a dataset that contains coding questions and answers
# We will use flytech/python-codes-25k as it has specific python instructions
dataset = load_dataset("flytech/python-codes-25k", split="train")

print("Filtering and formatting data...")
samples = []
for row in dataset:
    instruction = row.get("instruction", "").lower()
    input_code = row.get("input", "")
    output = row.get("output", "")
    
    # We want rows that ask to fix a bug or issue, and have some code
    if ("fix" in instruction or "bug" in instruction or "error" in instruction) and len(input_code) > 10:
        
        # Try to extract just the code block from the output as fixed code
        code_blocks = re.findall(r"```python(.*?)```", output, re.DOTALL)
        if not code_blocks:
            code_blocks = re.findall(r"```(.*?)```", output, re.DOTALL)
            
        fixed_code = code_blocks[0].strip() if code_blocks else output.strip()
        
        # We'll construct a synthetic explanation
        bug_identified = "Code contains a bug or inefficiency that prevents it from working correctly."
        root_cause = output.split("```")[0].strip() if "```" in output else "Syntax or logical error in the original implementation."
        if not root_cause:
            root_cause = "Implementation error in the source code."
            
        formatted_text = format_sample(
            buggy_code=input_code.strip(),
            language="python",
            bug_identified=bug_identified,
            root_cause=root_cause,
            fixed_code=fixed_code
        )
        samples.append({
            "text": formatted_text,
            "language": "python"
        })

print(f"Found {len(samples)} bug-fixing samples!")

# Let's add some synthetic obvious bugs so the model learns the exact format perfectly
custom_bugs = [
    {
        "buggy": "def get_average(nums):\n    return sum(nums) / len(nums)",
        "fixed": "def get_average(nums):\n    if not nums:\n        return 0\n    return sum(nums) / len(nums)",
        "bug_identified": "ZeroDivisionError when nums is empty.",
        "root_cause": "The len(nums) function evaluates to 0 for an empty list, causing a division by zero error."
    },
    {
        "buggy": "def append_item(val, lst=[]):\n    lst.append(val)\n    return lst",
        "fixed": "def append_item(val, lst=None):\n    if lst is None:\n        lst = []\n    lst.append(val)\n    return lst",
        "bug_identified": "Mutable default argument used for 'lst'.",
        "root_cause": "In Python, default arguments are evaluated only once at function definition time. This causes the same list object to be shared across all function calls."
    }
]

for bug in custom_bugs:
    formatted_text = format_sample(
        buggy_code=bug["buggy"],
        language="python",
        bug_identified=bug["bug_identified"],
        root_cause=bug["root_cause"],
        fixed_code=bug["fixed"]
    )
    # Duplicate them a few times so the model definitely learns the format
    for _ in range(50):
        samples.append({"text": formatted_text, "language": "python"})

print(f"Total dataset size after augmentation: {len(samples)}")

import random
random.shuffle(samples)

split_idx = int(len(samples) * 0.9)
train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
    for s in train_samples:
        f.write(json.dumps(s) + "\n")
        
with open(out_dir / "val.jsonl", "w", encoding="utf-8") as f:
    for s in val_samples:
        f.write(json.dumps(s) + "\n")

print(f"Saved {len(train_samples)} training and {len(val_samples)} validation samples to {out_dir}")
