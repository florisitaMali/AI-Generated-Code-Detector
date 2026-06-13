"""
Feature extraction for general library/open-source code.

These features complement the base stylometric features (identifiers, AST,
comments) for datasets that contain real-world function-level code rather
than competitive-programming submissions.

In competitive-programming code the AI/human gap is clear from basic
stylometrics (long names, docstrings, wide lines).  In library code those
signals weaken because even human developers follow good practices.  The
features here target patterns that remain discriminative for LLM-generated
library code:

  - Docstring completeness (Args/Returns/Raises sections)
  - Type annotation density
  - Exception-handling frequency
  - Vocabulary richness (type-token ratio)
  - Comprehension and lambda usage
  - Nesting/indentation patterns
  - Structural consistency signals (blank-line rhythm, import count)

All functions return a flat dict with float/int values that can be fed
directly into any sklearn or XGBoost estimator.
"""

import re
from collections import Counter


# ── Regex patterns (compiled once) ──────────────────────────────────────────

_RE_IDENT        = re.compile(r'\b[A-Za-z_]\w*\b')
_RE_TYPE_HINT    = re.compile(r'def\s+\w+\s*\(.*?->|:\s*[A-Z][a-zA-Z_\[\], ]+\s*[=,)]')
_RE_ARROW        = re.compile(r'->')
_RE_COMPREHENSION = re.compile(r'\[.+?\bfor\b.+?\bin\b.+?\]', re.DOTALL)
_RE_DICT_COMP    = re.compile(r'\{.+?\bfor\b.+?\bin\b.+?\}', re.DOTALL)
_RE_LAMBDA       = re.compile(r'\blambda\b')
_RE_ASSERT       = re.compile(r'\bassert\b')
_RE_ISINSTANCE   = re.compile(r'\bisinstance\s*\(')
_RE_F_STRING     = re.compile(r'\bf["\']')
_RE_IMPORT       = re.compile(r'^\s*(import|from)\s+\w', re.MULTILINE)
_RE_RETURN       = re.compile(r'\breturn\b')
_RE_RAISE        = re.compile(r'\braise\b')
_RE_TRY          = re.compile(r'\btry\s*:')
_RE_EXCEPT       = re.compile(r'\bexcept\b')
_RE_PASS         = re.compile(r'\bpass\b')
_RE_TERNARY      = re.compile(r'\bif\b.+?\belse\b')
_RE_ANNOTATION_J = re.compile(r'@[A-Z]\w*')   # Java annotations
_RE_GENERIC_J    = re.compile(r'<\w[\w,\s]*>')  # Java generics

# Docstring section markers (multiple conventions)
_DOCSTRING_ARGS    = re.compile(r'(Args:|Parameters:|:param\s+\w)', re.I)
_DOCSTRING_RETURNS = re.compile(r'(Returns?:|:returns?:|:rtype:)', re.I)
_DOCSTRING_RAISES  = re.compile(r'(Raises?:|:raises?\s+\w)', re.I)
_DOCSTRING_EXAMPLE = re.compile(r'(Examples?:|>>>\s)', re.I)


def extract_features(code: str, language: str = "python") -> dict:
    """
    Extract general library-code features from a source code string.

    Returns a flat dict of float/int values ready for ML consumption.
    Keys are prefixed with ``gen_`` to avoid collisions with base features.
    """
    if not code or not code.strip():
        return _empty()

    lines = code.splitlines()
    non_empty_lines = [l for l in lines if l.strip()]
    n_lines = max(len(lines), 1)
    n_non_empty = max(len(non_empty_lines), 1)

    f: dict = {}

    # ── Docstring quality ────────────────────────────────────────────────────
    f["gen_has_args_section"]    = int(bool(_DOCSTRING_ARGS.search(code)))
    f["gen_has_returns_section"] = int(bool(_DOCSTRING_RETURNS.search(code)))
    f["gen_has_raises_section"]  = int(bool(_DOCSTRING_RAISES.search(code)))
    f["gen_has_example_section"] = int(bool(_DOCSTRING_EXAMPLE.search(code)))
    f["gen_docstring_section_count"] = (
        f["gen_has_args_section"]
        + f["gen_has_returns_section"]
        + f["gen_has_raises_section"]
        + f["gen_has_example_section"]
    )

    # ── Type annotations ─────────────────────────────────────────────────────
    arrow_count = len(_RE_ARROW.findall(code))
    f["gen_return_type_hint_count"] = arrow_count
    f["gen_return_type_hint_density"] = arrow_count / n_non_empty

    # Count lines that contain ': <Type>' parameter annotations
    param_hint_lines = sum(
        1 for l in lines if re.search(r'\w\s*:\s*[A-Za-z_\[]', l)
    )
    f["gen_param_hint_line_count"]  = param_hint_lines
    f["gen_param_hint_density"]     = param_hint_lines / n_non_empty

    f["gen_total_type_hint_density"] = (arrow_count + param_hint_lines) / n_non_empty

    # ── Exception handling ───────────────────────────────────────────────────
    try_count    = len(_RE_TRY.findall(code))
    except_count = len(_RE_EXCEPT.findall(code))
    raise_count  = len(_RE_RAISE.findall(code))
    f["gen_try_count"]              = try_count
    f["gen_except_count"]           = except_count
    f["gen_raise_count"]            = raise_count
    f["gen_exception_density"]      = (try_count + except_count + raise_count) / n_non_empty

    # ── Comprehensions & lambdas ─────────────────────────────────────────────
    list_comp  = len(_RE_COMPREHENSION.findall(code))
    dict_comp  = len(_RE_DICT_COMP.findall(code))
    lambda_cnt = len(_RE_LAMBDA.findall(code))
    f["gen_list_comp_count"]   = list_comp
    f["gen_dict_comp_count"]   = dict_comp
    f["gen_comprehension_count"] = list_comp + dict_comp
    f["gen_lambda_count"]      = lambda_cnt
    f["gen_functional_density"] = (list_comp + dict_comp + lambda_cnt) / n_non_empty

    # ── Assertions & runtime checks ──────────────────────────────────────────
    assert_cnt    = len(_RE_ASSERT.findall(code))
    isinstance_cnt = len(_RE_ISINSTANCE.findall(code))
    f["gen_assert_count"]     = assert_cnt
    f["gen_isinstance_count"] = isinstance_cnt
    f["gen_check_density"]    = (assert_cnt + isinstance_cnt) / n_non_empty

    # ── Modern Python features ───────────────────────────────────────────────
    f["gen_f_string_count"]   = len(_RE_F_STRING.findall(code))
    f["gen_f_string_density"] = f["gen_f_string_count"] / n_non_empty

    ternary_cnt = len(_RE_TERNARY.findall(code))
    f["gen_ternary_count"]   = ternary_cnt
    f["gen_ternary_density"] = ternary_cnt / n_non_empty

    # ── Pass / empty-body patterns ───────────────────────────────────────────
    f["gen_pass_count"] = len(_RE_PASS.findall(code))

    # ── Imports ──────────────────────────────────────────────────────────────
    import_cnt = len(_RE_IMPORT.findall(code))
    f["gen_import_count"]   = import_cnt
    f["gen_import_density"] = import_cnt / n_non_empty

    # ── Return statements ────────────────────────────────────────────────────
    return_cnt = len(_RE_RETURN.findall(code))
    f["gen_return_count"]   = return_cnt
    f["gen_return_density"] = return_cnt / n_non_empty

    # ── Indentation / nesting depth ──────────────────────────────────────────
    # Estimate nesting by counting leading whitespace (4 spaces or 1 tab = 1 level)
    indent_levels = []
    for line in non_empty_lines:
        stripped = line.lstrip()
        spaces = len(line) - len(stripped)
        tabs = line[:len(line) - len(stripped)].count('\t')
        indent_levels.append(spaces // 4 + tabs)

    f["gen_max_indent_level"] = max(indent_levels, default=0)
    f["gen_avg_indent_level"] = (
        sum(indent_levels) / len(indent_levels) if indent_levels else 0.0
    )

    # ── Blank line rhythm ────────────────────────────────────────────────────
    blank_cnt = sum(1 for l in lines if not l.strip())
    f["gen_blank_line_count"] = blank_cnt
    f["gen_blank_line_ratio"] = blank_cnt / n_lines

    # Consecutive blank lines (AI tends to use exactly one blank between blocks)
    max_consec = 0
    cur_consec = 0
    for l in lines:
        if not l.strip():
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0
    f["gen_max_consecutive_blanks"] = max_consec

    # ── Vocabulary richness ──────────────────────────────────────────────────
    tokens = _RE_IDENT.findall(code)
    unique_tokens = set(tokens)
    f["gen_vocabulary_size"]    = len(unique_tokens)
    f["gen_type_token_ratio"]   = len(unique_tokens) / max(len(tokens), 1)
    f["gen_avg_token_length"]   = (
        sum(len(t) for t in tokens) / max(len(tokens), 1)
    )

    # Top-token dominance: how much does the most frequent token dominate?
    if tokens:
        counts = Counter(tokens)
        top_freq = counts.most_common(1)[0][1]
        f["gen_top_token_dominance"] = top_freq / len(tokens)
    else:
        f["gen_top_token_dominance"] = 0.0

    # ── Operator / punctuation density ───────────────────────────────────────
    f["gen_colon_density"]  = code.count(':')  / n_lines
    f["gen_comma_density"]  = code.count(',')  / n_lines
    f["gen_equals_density"] = code.count('=')  / n_lines
    f["gen_dot_density"]    = code.count('.')  / n_lines

    # ── Language-specific features ───────────────────────────────────────────
    if language == "java":
        annotation_cnt = len(_RE_ANNOTATION_J.findall(code))
        generic_cnt    = len(_RE_GENERIC_J.findall(code))
        f["gen_annotation_count"]    = annotation_cnt
        f["gen_annotation_density"]  = annotation_cnt / n_non_empty
        f["gen_generic_type_count"]  = generic_cnt
        f["gen_generic_type_density"] = generic_cnt / n_non_empty
        # Javadoc markers
        javadoc_tags = len(re.findall(r'@(param|return|throws|author|version)\b', code))
        f["gen_javadoc_tag_count"]   = javadoc_tags
        f["gen_javadoc_tag_density"] = javadoc_tags / n_non_empty
    else:
        f["gen_annotation_count"]     = 0
        f["gen_annotation_density"]   = 0.0
        f["gen_generic_type_count"]   = 0
        f["gen_generic_type_density"] = 0.0
        f["gen_javadoc_tag_count"]    = 0
        f["gen_javadoc_tag_density"]  = 0.0

    # ── AI composite signal for library code ─────────────────────────────────
    # Weighted combination of the strongest discriminative features.
    # AI library code tends to: add docstring sections, use type hints,
    # handle exceptions explicitly, have richer vocabulary, use f-strings.
    signals = [
        min(f["gen_docstring_section_count"] / 3.0, 1.0),   # 0–3 sections → 0–1
        min(f["gen_total_type_hint_density"] / 0.3, 1.0),   # density → 0–1
        min(f["gen_exception_density"] / 0.1, 1.0),          # try/except/raise density
        float(f["gen_f_string_count"] > 0),                   # uses f-strings at all
        min(f["gen_type_token_ratio"], 1.0),                  # vocabulary richness
        min(f["gen_comprehension_count"] / 3.0, 1.0),        # comprehension usage
    ]
    f["gen_ai_signal_composite"] = float(sum(signals) / len(signals))

    return f


def _empty() -> dict:
    """Return a zero-valued feature dict for empty/invalid code."""
    dummy = extract_features("x = 1", "python")
    return {k: 0 for k in dummy}
