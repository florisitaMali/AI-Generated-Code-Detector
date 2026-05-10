"""
Identifier naming pattern extraction using tree-sitter.

AI models tend to produce generic, descriptive variable names (result, temp, count)
while competitive programmers use terse, personal shorthand (rem, sz, pw2, dx, dy).
"""

import re
from collections import Counter

from loguru import logger

GENERIC_NAMES = frozenset({
    "result", "res", "ans", "answer", "output", "ret", "temp", "tmp",
    "val", "value", "data", "item", "items", "count", "total", "sum",
    "num", "number", "index", "idx", "i", "j", "k", "n", "m", "x", "y",
    "arr", "array", "list", "lst", "vec", "s", "t", "a", "b", "c",
    "left", "right", "mid", "start", "end", "low", "high",
    "node", "root", "head", "tail", "prev", "next", "curr", "current",
    "dp", "graph", "visited", "queue", "stack", "map", "set",
})

IDENTIFIER_PATTERN = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")

# Keywords to exclude per language
KEYWORDS = {
    "cpp": frozenset({
        "auto", "break", "case", "char", "const", "continue", "default", "do",
        "double", "else", "enum", "extern", "float", "for", "goto", "if",
        "int", "long", "register", "return", "short", "signed", "sizeof",
        "static", "struct", "switch", "typedef", "union", "unsigned", "void",
        "volatile", "while", "class", "namespace", "template", "this", "new",
        "delete", "try", "catch", "throw", "public", "private", "protected",
        "virtual", "friend", "operator", "bool", "true", "false", "nullptr",
        "using", "string", "vector", "pair", "map", "set", "queue", "stack",
        "cout", "cin", "endl", "include", "define", "ifdef", "ifndef", "endif",
        "main", "std", "printf", "scanf",
    }),
    "python": frozenset({
        "and", "as", "assert", "async", "await", "break", "class", "continue",
        "def", "del", "elif", "else", "except", "finally", "for", "from",
        "global", "if", "import", "in", "is", "lambda", "nonlocal", "not",
        "or", "pass", "raise", "return", "try", "while", "with", "yield",
        "True", "False", "None", "print", "range", "len", "int", "str",
        "float", "list", "dict", "set", "tuple", "input", "open", "self",
        "map", "filter", "sorted", "enumerate", "zip", "type", "super",
        "__init__", "__main__", "__name__",
    }),
    "java": frozenset({
        "abstract", "assert", "boolean", "break", "byte", "case", "catch",
        "char", "class", "const", "continue", "default", "do", "double",
        "else", "enum", "extends", "final", "finally", "float", "for",
        "goto", "if", "implements", "import", "instanceof", "int", "interface",
        "long", "native", "new", "package", "private", "protected", "public",
        "return", "short", "static", "strictfp", "super", "switch",
        "synchronized", "this", "throw", "throws", "transient", "try",
        "void", "volatile", "while", "true", "false", "null",
        "String", "System", "Scanner", "main", "args", "out", "println",
    }),
}


def extract_identifiers(code: str, language: str = "python") -> list[str]:
    """Extract all non-keyword identifiers from code."""
    all_matches = IDENTIFIER_PATTERN.findall(code)
    lang_keywords = KEYWORDS.get(language, set())
    identifiers = [m for m in all_matches if m not in lang_keywords and not m.startswith("__")]
    return identifiers


def extract_features(code: str, language: str = "python") -> dict:
    """
    Extract identifier naming pattern features.

    Returns:
        {
            "id_count": int,              # total identifiers
            "id_unique_count": int,       # unique identifiers
            "id_unique_ratio": float,     # unique / total
            "id_avg_length": float,       # average identifier length
            "id_max_length": int,         # longest identifier
            "id_generic_ratio": float,    # fraction that are generic names
            "id_single_char_ratio": float,# fraction that are single characters
            "id_has_underscore_ratio": float, # fraction containing underscores
            "id_camel_case_ratio": float, # fraction in camelCase
        }
    """
    try:
        identifiers = extract_identifiers(code, language)

        if not identifiers:
            return _empty_features()

        counter = Counter(identifiers)
        unique = list(counter.keys())
        total = len(identifiers)

        generic_count = sum(1 for ident in identifiers if ident.lower() in GENERIC_NAMES)
        single_char = sum(1 for ident in identifiers if len(ident) == 1)
        has_underscore = sum(1 for name in unique if "_" in name)
        camel_case = sum(1 for name in unique if re.match(r"^[a-z]+[A-Z]", name))
        lengths = [len(name) for name in unique]

        return {
            "id_count": total,
            "id_unique_count": len(unique),
            "id_unique_ratio": len(unique) / total if total else 0.0,
            "id_avg_length": sum(lengths) / len(lengths) if lengths else 0.0,
            "id_max_length": max(lengths) if lengths else 0,
            "id_generic_ratio": generic_count / total if total else 0.0,
            "id_single_char_ratio": single_char / total if total else 0.0,
            "id_has_underscore_ratio": has_underscore / len(unique) if unique else 0.0,
            "id_camel_case_ratio": camel_case / len(unique) if unique else 0.0,
        }
    except Exception as e:
        logger.warning(f"Identifier feature extraction failed: {e}")
        return _empty_features()


def _empty_features() -> dict:
    return {
        "id_count": 0,
        "id_unique_count": 0,
        "id_unique_ratio": 0.0,
        "id_avg_length": 0.0,
        "id_max_length": 0,
        "id_generic_ratio": 0.0,
        "id_single_char_ratio": 0.0,
        "id_has_underscore_ratio": 0.0,
        "id_camel_case_ratio": 0.0,
    }
