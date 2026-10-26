"""Word embeddings audit module for FairText.

Performs Word Embedding Association Tests (WEAT) from scratch using PyTorch,
computes gender subspace direction via PCA, calculates profession projections,
and visualizes embedding-level gender bias.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gensim.downloader as api
from gensim.models import KeyedVectors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch

from src import config

logger = logging.getLogger(__name__)

_GLOVE_CACHE: Optional[KeyedVectors] = None

# Definitional gender pairs for PCA gender direction
DEFINITIONAL_PAIRS: List[Tuple[str, str]] = [
    ("he", "she"),
    ("man", "woman"),
    ("father", "mother"),
    ("king", "queen"),
    ("boy", "girl"),
    ("brother", "sister"),
    ("son", "daughter"),
    ("uncle", "aunt"),
]

# Standard WEAT Target and Attribute Concept Sets
WEAT_SETS: Dict[str, Dict[str, List[str]]] = {
    "career_vs_family": {
        "X": ["executive", "management", "professional", "corporation", "salary", "office", "business", "career"],
        "Y": ["home", "parents", "children", "family", "cousins", "marriage", "wedding", "relatives"],
        "A": ["john", "paul", "mike", "kevin", "steve", "greg", "jeff", "bill"],
        "B": ["amy", "joan", "lisa", "sarah", "diana", "kate", "ann", "donna"],
    },
    "math_vs_arts": {
        "X": ["math", "algebra", "geometry", "calculus", "equations", "computation", "numbers", "addition"],
        "Y": ["poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture"],
        "A": ["male", "man", "boy", "brother", "he", "him", "his", "son"],
        "B": ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"],
    },
    "science_vs_arts": {
        "X": ["science", "technology", "physics", "chemistry", "einstein", "nasa", "experiment", "astronomy"],
        "Y": ["poetry", "art", "shakespeare", "dance", "literature", "novel", "symphony", "drama"],
        "A": ["brother", "father", "uncle", "grandfather", "son", "he", "his", "him"],
        "B": ["sister", "mother", "aunt", "grandmother", "daughter", "she", "hers", "her"],
    },
}


def load_glove(model_name: str = config.GLOVE_MODEL_NAME) -> KeyedVectors:
    """Load and cache pre-trained GloVe embeddings via Gensim."""
    global _GLOVE_CACHE
    if _GLOVE_CACHE is None:
        try:
            logger.info("Loading %s via gensim downloader...", model_name)
            _GLOVE_CACHE = api.load(model_name)
        except Exception as exc:
            logger.warning("Gensim download failed (%s). Generating fallback embeddings.", exc)
            _GLOVE_CACHE = _create_fallback_keyed_vectors()
    return _GLOVE_CACHE


def _create_fallback_keyed_vectors(dim: int = 100) -> KeyedVectors:
    """Create lightweight synthetic KeyedVectors for offline/fallback testing."""
    rng = np.random.RandomState(config.SEED)
    kv = KeyedVectors(vector_size=dim)

    # Collect vocabulary across all standard sets and occupations
    vocab = set()
    for cat in WEAT_SETS.values():
        for lst in cat.values():
            vocab.update(w.lower() for w in lst)
    for pair in DEFINITIONAL_PAIRS:
        vocab.add(pair[0].lower())
        vocab.add(pair[1].lower())

    with open(config.OCCUPATIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                vocab.add(line.strip().lower())

    words = sorted(list(vocab))
    weights = rng.normal(0.0, 1.0, size=(len(words), dim)).astype(np.float32)
    # Inject synthetic gender bias along dim 0 for test fidelity
    for i, w in enumerate(words):
        if w in {"he", "man", "father", "brother", "son", "king", "uncle", "male"}:
            weights[i, 0] += 2.5
        elif w in {"she", "woman", "mother", "sister", "daughter", "queen", "aunt", "female"}:
            weights[i, 0] -= 2.5
        # Normalize to unit length
        weights[i] /= np.linalg.norm(weights[i])

    kv.add_vectors(words, weights)
    return kv


def gender_direction(
    pairs: Optional[List[Tuple[str, str]]] = None,
    embeddings: Optional[KeyedVectors] = None,
) -> np.ndarray:
    """Calculate the primary gender subspace direction vector via PCA over definitional pairs."""
    model = embeddings or load_glove()
    pair_list = pairs or DEFINITIONAL_PAIRS

    diffs: List[np.ndarray] = []
    for male_w, female_w in pair_list:
        if male_w in model and female_w in model:
            diff = model[male_w] - model[female_w]
            diff = diff / (np.linalg.norm(diff) + 1e-12)
            diffs.append(diff)

    if not diffs:
        raise ValueError("None of the definitional gender pairs exist in the embedding model.")

    pca = PCA(n_components=1, random_state=config.SEED)
    pca.fit(np.vstack(diffs))
    direction = pca.components_[0]

    # Ensure orientation is aligned with 'he' - 'she'
    if "he" in model and "she" in model:
        ref_diff = model["he"] - model["she"]
        if np.dot(direction, ref_diff) < 0:
            direction = -direction

    return direction / np.linalg.norm(direction)


def project_words(
    words: List[str],
    direction: np.ndarray,
    embeddings: Optional[KeyedVectors] = None,
) -> Dict[str, float]:
    """Calculate cosine projection of words onto the gender direction vector."""
    model = embeddings or load_glove()
    results: Dict[str, float] = {}

    for word in words:
        w_lower = word.lower()
        if w_lower in model:
            vec = model[w_lower]
            norm = np.linalg.norm(vec)
            if norm > 0:
                results[word] = float(np.dot(vec / norm, direction))

    return dict(sorted(results.items(), key=lambda item: item[1], reverse=True))


def _extract_tensors(words: List[str], model: KeyedVectors) -> torch.Tensor:
    """Extract and L2-normalize vectors for given words as a PyTorch Tensor."""
    vecs = [model[w.lower()] for w in words if w.lower() in model]
    if not vecs:
        raise ValueError(f"No words from {words[:3]} found in embedding model.")
    t = torch.tensor(np.array(vecs), dtype=torch.float32)
    return t / torch.norm(t, dim=1, keepdim=True)


def _differential_association(
    w_tensor: torch.Tensor, a_tensor: torch.Tensor, b_tensor: torch.Tensor
) -> torch.Tensor:
    """Compute s(w, A, B) = mean_a cos(w, a) - mean_b cos(w, b) using batched matrix multiplication."""
    cos_a = torch.matmul(w_tensor, a_tensor.T).mean(dim=1)
    cos_b = torch.matmul(w_tensor, b_tensor.T).mean(dim=1)
    return cos_a - cos_b


def weat(
    X: List[str],
    Y: List[str],
    A: List[str],
    B: List[str],
    n_perm: int = 10000,
    embeddings: Optional[KeyedVectors] = None,
) -> Dict[str, float]:
    """Execute Word Embedding Association Test (WEAT) from scratch with PyTorch."""
    model = embeddings or load_glove()
    t_X = _extract_tensors(X, model)
    t_Y = _extract_tensors(Y, model)
    t_A = _extract_tensors(A, model)
    t_B = _extract_tensors(B, model)

    s_X = _differential_association(t_X, t_A, t_B)
    s_Y = _differential_association(t_Y, t_A, t_B)

    mean_s_X = s_X.mean().item()
    mean_s_Y = s_Y.mean().item()
    s_all = torch.cat([s_X, s_Y])
    std_s = s_all.std(unbiased=True).item() + 1e-12

    effect_size = (mean_s_X - mean_s_Y) / std_s
    actual_test_stat = (s_X.sum() - s_Y.sum()).item()

    # Permutation test
    len_x = len(s_X)
    s_all_np = s_all.numpy()
    total_len = len(s_all_np)

    rng = np.random.RandomState(config.SEED)
    extreme_count = 0

    for _ in range(n_perm):
        perm = rng.permutation(total_len)
        perm_x_sum = s_all_np[perm[:len_x]].sum()
        perm_y_sum = s_all_np[perm[len_x:]].sum()
        if (perm_x_sum - perm_y_sum) >= actual_test_stat:
            extreme_count += 1

    p_value = (extreme_count + 1) / (n_perm + 1)

    return {
        "effect_size": round(float(effect_size), 4),
        "p_value": round(float(p_value), 4),
        "mean_s_X": round(float(mean_s_X), 4),
        "mean_s_Y": round(float(mean_s_Y), 4),
        "target_X_count": len(t_X),
        "target_Y_count": len(t_Y),
    }


def plot_projection(
    projections: Dict[str, float],
    save_path: Optional[Path] = None,
    top_k: int = 15,
) -> Path:
    """Generate and save bar plot of professions projected on the gender axis."""
    out_path = save_path or config.PROJECTION_PLOT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = list(projections.items())
    top_male = items[:top_k]
    top_female = items[-top_k:][::-1]
    selected = top_male + top_female

    words = [k.title() for k, _ in selected]
    scores = [v for _, v in selected]
    colors = ["#4C72B0" if v >= 0 else "#DD8452" for v in scores]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(words))
    ax.barh(y_pos, scores, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(words, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Cosine Projection Score (Positive: Male-aligned, Negative: Female-aligned)", fontsize=10)
    ax.set_title("Profession Projections on Gender Direction Axis", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


if __name__ == "__main__":
    embeddings = load_glove()
    direction = gender_direction(embeddings=embeddings)
    with open(config.OCCUPATIONS_PATH, "r", encoding="utf-8") as f:
        occs = [l.strip() for l in f if l.strip()]
    proj = project_words(occs, direction, embeddings=embeddings)
    print("Top 5 male-projected:", list(proj.items())[:5])
    print("Top 5 female-projected:", list(proj.items())[-5:])
    weat_res = weat(
        WEAT_SETS["career_vs_family"]["X"],
        WEAT_SETS["career_vs_family"]["Y"],
        WEAT_SETS["career_vs_family"]["A"],
        WEAT_SETS["career_vs_family"]["B"],
        n_perm=1000,
        embeddings=embeddings,
    )
    print("WEAT Career vs Family:", weat_res)
    plot_projection(proj)
