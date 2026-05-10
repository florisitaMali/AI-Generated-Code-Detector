"""Tests for the stylometric feature extraction modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.identifiers import extract_features as id_features
from src.features.ast_features import extract_features as ast_features
from src.features.comments import extract_features as comment_features


SAMPLE_PYTHON = """
def solve(n, arr):
    # sort and find max
    arr.sort()
    result = 0
    for i in range(n):
        result += arr[i] * (i + 1)
    return result

if __name__ == "__main__":
    n = int(input())
    arr = list(map(int, input().split()))
    print(solve(n, arr))
"""

SAMPLE_CPP = """
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    sort(a.begin(), a.end());
    long long ans = 0;
    for (int i = 0; i < n; i++) ans += (long long)a[i] * (i + 1);
    cout << ans << endl;
    return 0;
}
"""


def test_identifier_features_python():
    feats = id_features(SAMPLE_PYTHON, "python")
    assert feats["id_count"] > 0
    assert 0.0 <= feats["id_unique_ratio"] <= 1.0
    assert feats["id_avg_length"] > 0


def test_identifier_features_cpp():
    feats = id_features(SAMPLE_CPP, "cpp")
    assert feats["id_count"] > 0
    assert feats["id_generic_ratio"] >= 0.0


def test_ast_features_python():
    feats = ast_features(SAMPLE_PYTHON, "python")
    assert feats["ast_node_count"] > 0
    assert feats["ast_cyclomatic_complexity"] >= 1
    assert feats["ast_depth"] >= 0


def test_ast_features_cpp():
    feats = ast_features(SAMPLE_CPP, "cpp")
    assert feats["ast_node_count"] > 0


def test_comment_features_python():
    feats = comment_features(SAMPLE_PYTHON, "python")
    assert feats["comment_line_count"] >= 1
    assert feats["comment_to_code_ratio"] > 0


def test_comment_features_no_comments():
    code = "x = 1\ny = 2\nprint(x + y)\n"
    feats = comment_features(code, "python")
    assert feats["comment_line_count"] == 0


def test_empty_code():
    feats = id_features("", "python")
    assert feats["id_count"] == 0

    feats = ast_features("", "python")
    assert isinstance(feats["ast_node_count"], int)

    feats = comment_features("", "python")
    assert feats["comment_line_count"] == 0
