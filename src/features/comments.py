"""
Comment density analysis.

AI-generated code typically over-comments with verbose docstrings and inline
explanations. Competitive programming submissions from humans often contain
zero comments. This asymmetry provides a useful stylometric signal.
"""

import re

from loguru import logger

SINGLE_LINE_PATTERNS = {
    "python": re.compile(r"^\s*#(.*)$", re.MULTILINE),
    "cpp": re.compile(r"//(.*)$", re.MULTILINE),
    "java": re.compile(r"//(.*)$", re.MULTILINE),
}

MULTI_LINE_PATTERNS = {
    "python": re.compile(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', re.MULTILINE),
    "cpp": re.compile(r"/\*[\s\S]*?\*/", re.MULTILINE),
    "java": re.compile(r"/\*[\s\S]*?\*/", re.MULTILINE),
}

DOCSTRING_PATTERNS = {
    "python": re.compile(
        r'(?:def|class)\s+\w+[^:]*:\s*\n\s*("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')',
        re.MULTILINE,
    ),
    "java": re.compile(r"/\*\*[\s\S]*?\*/", re.MULTILINE),
    "cpp": re.compile(r"/\*\*[\s\S]*?\*/", re.MULTILINE),
}


def _count_lines(text: str) -> int:
    return len([l for l in text.split("\n") if l.strip()])


def extract_features(code: str, language: str = "python") -> dict:
    """
    Extract comment-related features from code.

    Returns:
        {
            "comment_line_count": int,          # single-line comments
            "comment_block_count": int,         # block/multi-line comments
            "comment_total_chars": int,         # total characters in comments
            "comment_to_code_ratio": float,     # comment lines / code lines
            "has_docstring": bool,              # whether docstrings are present
            "docstring_count": int,             # number of docstrings
            "inline_comment_density": float,    # comments per non-empty line
            "avg_comment_length": float,        # average comment text length
        }
    """
    try:
        lines = code.split("\n")
        non_empty_lines = [l for l in lines if l.strip()]
        total_lines = len(non_empty_lines)

        single_pattern = SINGLE_LINE_PATTERNS.get(language)
        multi_pattern = MULTI_LINE_PATTERNS.get(language)
        doc_pattern = DOCSTRING_PATTERNS.get(language)

        single_comments = single_pattern.findall(code) if single_pattern else []
        multi_comments = multi_pattern.findall(code) if multi_pattern else []
        docstrings = doc_pattern.findall(code) if doc_pattern else []

        comment_line_count = len(single_comments)
        block_count = len(multi_comments)
        block_lines = sum(_count_lines(m) for m in multi_comments)

        total_comment_chars = (
            sum(len(c.strip()) for c in single_comments)
            + sum(len(m) for m in multi_comments)
        )
        total_comment_lines = comment_line_count + block_lines
        code_lines = max(total_lines - total_comment_lines, 1)

        comment_lengths = [len(c.strip()) for c in single_comments if c.strip()]

        return {
            "comment_line_count": comment_line_count,
            "comment_block_count": block_count,
            "comment_total_chars": total_comment_chars,
            "comment_to_code_ratio": total_comment_lines / code_lines if code_lines else 0.0,
            "has_docstring": len(docstrings) > 0,
            "docstring_count": len(docstrings),
            "inline_comment_density": total_comment_lines / total_lines if total_lines else 0.0,
            "avg_comment_length": (
                sum(comment_lengths) / len(comment_lengths) if comment_lengths else 0.0
            ),
        }
    except Exception as e:
        logger.warning(f"Comment feature extraction failed: {e}")
        return {
            "comment_line_count": 0,
            "comment_block_count": 0,
            "comment_total_chars": 0,
            "comment_to_code_ratio": 0.0,
            "has_docstring": False,
            "docstring_count": 0,
            "inline_comment_density": 0.0,
            "avg_comment_length": 0.0,
        }
