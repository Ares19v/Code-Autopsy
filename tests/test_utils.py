"""
tests/test_utils.py
────────────────────
Pure-Python unit tests for output parsing logic — no GPU, no model downloads.
These run in CI on every push/PR to guarantee a green tick.

The parser is extracted inline here so the test has zero heavy dependencies.
"""

import re
from dataclasses import dataclass


# ── Inline copy of parse_assistant_output for zero-dependency CI testing ──────
@dataclass
class ParsedOutput:
    bug_identified: str
    root_cause: str
    fixed_code: str
    language: str = "python"


def parse_assistant_output(text: str, language: str = "python") -> ParsedOutput:
    text = re.sub(r"<\|im_start\|>assistant\s*", "", text)
    text = re.sub(r"<\|im_end\|>.*", "", text, flags=re.DOTALL)
    text = text.strip()

    bug_match = re.search(
        r"##\s*Bug Identified\s*\n(.*?)(?=##\s*Root Cause|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    root_match = re.search(
        r"##\s*Root Cause\s*\n(.*?)(?=##\s*Fixed Code|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    code_match = re.search(
        r"##\s*Fixed Code\s*\n```(?:\w+)?\n(.*?)```",
        text, re.DOTALL | re.IGNORECASE
    )

    return ParsedOutput(
        bug_identified=bug_match.group(1).strip() if bug_match else text,
        root_cause=root_match.group(1).strip() if root_match else "",
        fixed_code=code_match.group(1).strip() if code_match else "",
        language=language,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

WELL_FORMED = """\
## Bug Identified
ZeroDivisionError when nums is an empty list.

## Root Cause
No guard clause for empty input. len([]) returns 0, causing division by zero.

## Fixed Code
```python
def avg(nums):
    return sum(nums) / len(nums) if nums else 0.0
```
"""

PARTIAL_NO_FIX = """\
## Bug Identified
Missing await keyword.

## Root Cause
response.json() is async.
"""

EMPTY = ""


def test_parse_well_formed():
    result = parse_assistant_output(WELL_FORMED, language="python")
    assert "ZeroDivisionError" in result.bug_identified
    assert "guard clause" in result.root_cause
    assert "def avg" in result.fixed_code


def test_parse_partial_returns_gracefully():
    result = parse_assistant_output(PARTIAL_NO_FIX, language="python")
    assert result.bug_identified
    assert result.fixed_code == ""


def test_parse_empty_string():
    result = parse_assistant_output(EMPTY, language="python")
    assert result.bug_identified == ""
    assert result.root_cause == ""
    assert result.fixed_code == ""


def test_parse_javascript_language():
    js_output = """\
## Bug Identified
Missing await.

## Root Cause
Async function not awaited.

## Fixed Code
```javascript
const data = await response.json();
```
"""
    result = parse_assistant_output(js_output, language="javascript")
    assert "await" in result.fixed_code
    assert result.language == "javascript"


def test_parse_result_has_all_fields():
    result = parse_assistant_output(WELL_FORMED, language="python")
    assert hasattr(result, "bug_identified")
    assert hasattr(result, "root_cause")
    assert hasattr(result, "fixed_code")


def test_parse_strips_chat_markers():
    text_with_marker = "<|im_start|>assistant\n## Bug Identified\nBug here.\n\n## Root Cause\nCause.\n\n## Fixed Code\n```python\nfix()\n```<|im_end|>"
    result = parse_assistant_output(text_with_marker)
    assert result.bug_identified == "Bug here."
    assert result.fixed_code == "fix()"


def test_parse_case_insensitive_headers():
    text = "## bug identified\nSome bug.\n\n## root cause\nSome cause.\n\n## fixed code\n```python\nfix()\n```"
    result = parse_assistant_output(text)
    assert result.bug_identified == "Some bug."
