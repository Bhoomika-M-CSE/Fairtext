"""Unit tests for WEAT algorithm and permutation testing with PyTorch."""

import numpy as np
import pytest
from gensim.models import KeyedVectors
from src.embeddings_audit import _differential_association, _extract_tensors, weat


def _build_synthetic_keyed_vectors() -> KeyedVectors:
    """Construct controlled synthetic embeddings to test WEAT mathematical properties."""
    dim = 20
    kv = KeyedVectors(vector_size=dim)

    # Concept words
    words = {
        # Target X (aligned with A): positive on dim 0
        "x1": [1.0] + [0.0] * 19,
        "x2": [0.9] + [0.0] * 19,
        # Target Y (aligned with B): negative on dim 0
        "y1": [-1.0] + [0.0] * 19,
        "y2": [-0.9] + [0.0] * 19,
        # Attribute A (positive on dim 0)
        "a1": [1.0] + [0.0] * 19,
        "a2": [0.8] + [0.0] * 19,
        # Attribute B (negative on dim 0)
        "b1": [-1.0] + [0.0] * 19,
        "b2": [-0.8] + [0.0] * 19,
    }

    word_list = list(words.keys())
    weights = np.array([words[w] for w in word_list], dtype=np.float32)
    # Normalize
    for i in range(len(weights)):
        weights[i] /= np.linalg.norm(weights[i])

    kv.add_vectors(word_list, weights)
    return kv


def test_weat_effect_size_sign_and_range():
    """Verify that WEAT correctly yields positive effect size and valid p-value."""
    kv = _build_synthetic_keyed_vectors()
    X = ["x1", "x2"]
    Y = ["y1", "y2"]
    A = ["a1", "a2"]
    B = ["b1", "b2"]

    res = weat(X, Y, A, B, n_perm=500, embeddings=kv)

    # Effect size should be strictly positive because X is closer to A, Y closer to B
    assert res["effect_size"] > 0.0
    # Range check
    assert -10.0 <= res["effect_size"] <= 10.0
    # p-value must be a valid probability
    assert 0.0 <= res["p_value"] <= 1.0


def test_weat_reversed_targets():
    """Verify that swapping target concepts inverts the effect size sign."""
    kv = _build_synthetic_keyed_vectors()
    X = ["x1", "x2"]
    Y = ["y1", "y2"]
    A = ["a1", "a2"]
    B = ["b1", "b2"]

    res_forward = weat(X, Y, A, B, n_perm=100, embeddings=kv)
    res_reversed = weat(Y, X, A, B, n_perm=100, embeddings=kv)

    assert res_forward["effect_size"] > 0
    assert res_reversed["effect_size"] < 0
    assert abs(res_forward["effect_size"] + res_reversed["effect_size"]) < 1e-3
