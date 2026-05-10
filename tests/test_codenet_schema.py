"""Unit tests for CodeNet-style IDs and merged CSV field lists."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.codenet_schema import (
    MERGED_CSV_FIELDS,
    make_submission_id,
    make_user_id,
    strip_code_fences,
)


def test_make_submission_id_stable():
    a = make_submission_id("p00001", "gpt-4o", "C++")
    b = make_submission_id("p00001", "gpt-4o", "C++")
    assert a == b
    assert a.startswith("s")
    assert len(a) == 10


def test_make_user_id_stable():
    assert make_user_id("claude-3-5-sonnet-20241022", "Python") == make_user_id(
        "claude-3-5-sonnet-20241022", "Python"
    )


def test_strip_fences():
    s = """```python
print(1)
```"""
    out = strip_code_fences(s)
    assert "print" in out
    assert "```" not in out


def test_merged_fields_has_source():
    assert "source" in MERGED_CSV_FIELDS
    assert MERGED_CSV_FIELDS.index("source") < len(MERGED_CSV_FIELDS) - 1
