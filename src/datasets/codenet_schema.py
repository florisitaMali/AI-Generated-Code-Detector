"""CodeNet + AI-CodeNet submission schema, IDs, and code cleanup."""

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

# IBM CodeNet submission columns (Table 2) + AI extensions appended to AI_CodeNet CSV only.
ORIGINAL_SUBMISSION_FIELDS: list[str] = [
    "submission_id",
    "problem_id",
    "user_id",
    "date",
    "language",
    "original_language",
    "filename_ext",
    "status",
    "cpu_time",
    "memory",
    "code_size",
    "accuracy",
]

AI_EXTENSION_FIELDS: list[str] = [
    "model_name",
    "generation_prompt_tokens",
    "generation_completion_tokens",
]

SUBMISSION_CSV_HEADER: list[str] = ORIGINAL_SUBMISSION_FIELDS + AI_EXTENSION_FIELDS

MERGED_CSV_FIELDS: list[str] = ORIGINAL_SUBMISSION_FIELDS + ["source"] + AI_EXTENSION_FIELDS


LANG_DISPLAY_TO_META: dict[str, dict[str, str | None]] = {
    "C++": {"ext": "cpp", "original_language": "C++17", "compile_cmd": "g++"},
    "Python": {"ext": "py", "original_language": "Python3", "compile_cmd": None},
    "Java": {"ext": "java", "original_language": "Java17", "compile_cmd": "javac"},
}

CODENET_LANGUAGES: list[str] = ["C++", "Python", "Java"]


def make_user_id(model_name: str, language: str) -> str:
    key = f"{model_name}:{language}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10**9
    return f"u{h:09d}"


def make_submission_id(problem_id: str, model_name: str, language: str) -> str:
    key = f"{problem_id}:{model_name}:{language}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10**9
    return f"s{h:09d}"


def strip_code_fences(raw: str) -> str:
    text = raw.strip()
    patterns = [
        r"```cpp\n(.*?)```",
        r"```c\+\+\n(.*?)```",
        r"```python\n(.*?)```",
        r"```java\n(.*?)```",
        r"```\n(.*?)```",
    ]
    for p in patterns:
        m = re.search(p, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return text


def iter_problem_csvs(metadata_dir: Path, pattern: str = "p*.csv") -> Iterator[Path]:
    for csv_file in sorted(metadata_dir.glob(pattern)):
        if csv_file.name == "problem_list.csv":
            continue
        stem = csv_file.stem
        if not stem.startswith("p") or len(stem) != 6:
            continue
        if not stem[1:].isdigit():
            continue
        yield csv_file
