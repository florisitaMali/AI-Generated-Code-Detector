"""
AST-based feature extraction using tree-sitter.

Extracts structural features from parsed syntax trees: node type distribution
entropy, tree depth, cyclomatic complexity, and total node counts. AI code
tends to use a narrower range of constructs and produces structurally uniform
solutions.
"""

import math
from collections import Counter

from loguru import logger

try:
    import tree_sitter_python as tspython
    import tree_sitter_cpp as tscpp
    import tree_sitter_java as tsjava
    from tree_sitter import Language, Parser

    LANGUAGES: dict[str, Language] = {
        "python":     Language(tspython.language()),
        "cpp":        Language(tscpp.language()),
        "java":       Language(tsjava.language()),
    }

    # Optional grammars — load individually so a missing package doesn't
    # break the three core languages that are always required.
    try:
        import tree_sitter_c as tsc
        LANGUAGES["c"] = Language(tsc.language())
    except Exception:
        logger.warning("tree-sitter-c not available; C will use fallback features")

    try:
        import tree_sitter_c_sharp as tscs
        LANGUAGES["csharp"] = Language(tscs.language())
    except Exception:
        logger.warning("tree-sitter-c-sharp not available; C# will use fallback features")

    try:
        import tree_sitter_javascript as tsjs
        LANGUAGES["javascript"] = Language(tsjs.language())
    except Exception:
        logger.warning("tree-sitter-javascript not available; JS will use fallback features")

    TS_AVAILABLE = True
except Exception:
    LANGUAGES = {}
    TS_AVAILABLE = False
    logger.warning("tree-sitter languages not available; AST features will use fallback")

BRANCHING_TYPES = frozenset({
    "if_statement", "elif_clause", "else_clause",
    "for_statement", "while_statement", "do_statement",
    "switch_statement", "case_statement",
    "for_in_clause", "for_range_loop",
    "catch_clause", "conditional_expression",
    "&&", "||", "and", "or",
})


def _get_parser(language: str) -> "Parser | None":
    if not TS_AVAILABLE or language not in LANGUAGES:
        return None
    parser = Parser(LANGUAGES[language])
    return parser


def _walk_tree(node) -> list:
    """Collect all nodes in a tree via DFS."""
    nodes = [node]
    for child in node.children:
        nodes.extend(_walk_tree(child))
    return nodes


def _tree_depth(node, current: int = 0) -> int:
    if not node.children:
        return current
    return max(_tree_depth(child, current + 1) for child in node.children)


def _cyclomatic_complexity(nodes: list) -> int:
    """Approximate cyclomatic complexity: 1 + number of branching nodes."""
    branches = sum(1 for n in nodes if n.type in BRANCHING_TYPES)
    return 1 + branches


def _distribution_entropy(counter: Counter) -> float:
    """Shannon entropy of the node type distribution."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counter.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def extract_features_treesitter(code: str, language: str) -> dict:
    """Extract AST features using tree-sitter."""
    parser = _get_parser(language)
    if parser is None:
        return _fallback_features(code)

    tree = parser.parse(code.encode("utf-8"))
    root = tree.root_node
    all_nodes = _walk_tree(root)

    type_counter = Counter(n.type for n in all_nodes)
    depth = _tree_depth(root)
    complexity = _cyclomatic_complexity(all_nodes)
    entropy = _distribution_entropy(type_counter)

    named_nodes = [n for n in all_nodes if n.is_named]
    unique_types = len(type_counter)

    return {
        "ast_node_count": len(all_nodes),
        "ast_named_node_count": len(named_nodes),
        "ast_unique_types": unique_types,
        "ast_depth": depth,
        "ast_entropy": entropy,
        "ast_cyclomatic_complexity": complexity,
        "ast_avg_children": (
            sum(len(n.children) for n in all_nodes) / len(all_nodes)
            if all_nodes else 0.0
        ),
        "ast_leaf_ratio": (
            sum(1 for n in all_nodes if not n.children) / len(all_nodes)
            if all_nodes else 0.0
        ),
    }


def _fallback_features(code: str) -> dict:
    """Regex-based fallback when tree-sitter is unavailable."""
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]

    branch_keywords = {"if", "elif", "else", "for", "while", "switch", "case", "catch"}
    branch_count = sum(
        1 for line in non_empty
        for word in branch_keywords
        if f" {word} " in f" {line.strip()} " or line.strip().startswith(word)
    )

    nesting = 0
    max_nesting = 0
    for line in lines:
        stripped = line.strip()
        nesting += stripped.count("{") - stripped.count("}")
        if stripped.endswith(":") and not stripped.startswith("#"):
            nesting += 1
        max_nesting = max(max_nesting, nesting)

    return {
        "ast_node_count": len(non_empty),
        "ast_named_node_count": len(non_empty),
        "ast_unique_types": 0,
        "ast_depth": max_nesting,
        "ast_entropy": 0.0,
        "ast_cyclomatic_complexity": 1 + branch_count,
        "ast_avg_children": 0.0,
        "ast_leaf_ratio": 0.0,
    }


def extract_features(code: str, language: str = "python") -> dict:
    """Public interface: extract AST features from a code string."""
    try:
        return extract_features_treesitter(code, language)
    except Exception as e:
        logger.warning(f"AST feature extraction failed, using fallback: {e}")
        return _fallback_features(code)
