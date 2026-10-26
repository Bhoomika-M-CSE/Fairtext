"""Hard debiasing and embedding quality preservation module for FairText.

Implements Bolukbasi et al. (2016) hard debiasing algorithm:
subspace projection neutralization and symmetric equalization.
Performs semantic fidelity checks and before/after WEAT auditing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config
from src.embeddings_audit import (
    DEFINITIONAL_PAIRS,
    WEAT_SETS,
    gender_direction,
    load_glove,
    project_words,
    weat,
)

logger = logging.getLogger(__name__)

EQUALIZE_PAIRS: List[Tuple[str, str]] = [
    ("man", "woman"),
    ("he", "she"),
    ("father", "mother"),
    ("king", "queen"),
    ("actor", "actress"),
    ("waiter", "waitress"),
    ("brother", "sister"),
    ("son", "daughter"),
    ("policeman", "policewoman"),
    ("chairman", "chairwoman"),
]

SEMANTIC_BENCHMARK_PAIRS: List[Tuple[str, str]] = [
    ("car", "automobile"),
    ("gem", "jewel"),
    ("journey", "voyage"),
    ("boy", "lad"),
    ("coast", "shore"),
    ("asylum", "madhouse"),
    ("magician", "wizard"),
    ("midday", "noon"),
    ("furnace", "stove"),
    ("food", "fruit"),
    ("bird", "cock"),
    ("tool", "implement"),
    ("crane", "implement"),
    ("brother", "monk"),
    ("lad", "brother"),
    ("crane", "rooster"),
]


def _build_gendered_word_set(
    definitional_pairs: List[Tuple[str, str]],
    equalize_pairs: List[Tuple[str, str]],
) -> Set[str]:
    """Compile set of explicitly gendered words that must not be neutralized."""
    gendered: Set[str] = set()
    for w1, w2 in definitional_pairs + equalize_pairs:
        gendered.add(w1.lower())
        gendered.add(w2.lower())

    if config.MALE_TERMS_PATH.exists():
        with open(config.MALE_TERMS_PATH, "r", encoding="utf-8") as f:
            gendered.update(l.strip().lower() for l in f if l.strip())
    if config.FEMALE_TERMS_PATH.exists():
        with open(config.FEMALE_TERMS_PATH, "r", encoding="utf-8") as f:
            gendered.update(l.strip().lower() for l in f if l.strip())

    return gendered


def hard_debias(
    embeddings: Optional[KeyedVectors] = None,
    g_direction: Optional[np.ndarray] = None,
    definitional_pairs: Optional[List[Tuple[str, str]]] = None,
    equalize_pairs: Optional[List[Tuple[str, str]]] = None,
) -> KeyedVectors:
    """Perform Bolukbasi et al. (2016) hard debiasing and return a new KeyedVectors instance."""
    original = embeddings or load_glove()
    def_pairs = definitional_pairs or DEFINITIONAL_PAIRS
    eq_pairs = equalize_pairs or EQUALIZE_PAIRS
    v_g = g_direction if g_direction is not None else gender_direction(def_pairs, original)
    v_g = v_g / (np.linalg.norm(v_g) + 1e-12)

    gendered_words = _build_gendered_word_set(def_pairs, eq_pairs)
    words = list(original.key_to_index.keys())
    dim = original.vector_size
    debiased_vectors = np.zeros((len(words), dim), dtype=np.float32)

    # Step 1: Neutralization
    for idx, w in enumerate(words):
        vec = original.get_vector(w).copy()
        norm = np.linalg.norm(vec) + 1e-12
        vec = vec / norm
        if w.lower() not in gendered_words:
            # Project out the gender component
            vec = vec - np.dot(vec, v_g) * v_g
            vec = vec / (np.linalg.norm(vec) + 1e-12)
        debiased_vectors[idx] = vec

    debiased_kv = KeyedVectors(vector_size=dim)
    debiased_kv.add_vectors(words, debiased_vectors)

    # Step 2: Equalization
    for w1, w2 in eq_pairs:
        if w1 in debiased_kv and w2 in debiased_kv:
            v1 = debiased_kv[w1]
            v2 = debiased_kv[w2]
            mu = (v1 + v2) / 2.0
            nu = mu - np.dot(mu, v_g) * v_g
            nu_norm_sq = np.sum(nu ** 2)
            if nu_norm_sq < 1.0:
                z = np.sqrt(1.0 - nu_norm_sq)
            else:
                nu = nu / (np.sqrt(nu_norm_sq) + 1e-12)
                z = 0.0

            v1_new = (nu + z * v_g) / (np.linalg.norm(nu + z * v_g) + 1e-12)
            v2_new = (nu - z * v_g) / (np.linalg.norm(nu - z * v_g) + 1e-12)
            debiased_kv.vectors[debiased_kv.key_to_index[w1]] = v1_new
            debiased_kv.vectors[debiased_kv.key_to_index[w2]] = v2_new

    return debiased_kv


def quality_check(
    original: KeyedVectors,
    debiased: KeyedVectors,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, float]:
    """Evaluate semantic similarity correlation between original and debiased embeddings."""
    eval_pairs = pairs or SEMANTIC_BENCHMARK_PAIRS
    orig_sims: List[float] = []
    deb_sims: List[float] = []

    for w1, w2 in eval_pairs:
        if w1 in original and w2 in original and w1 in debiased and w2 in debiased:
            s_orig = float(np.dot(original[w1], original[w2]) / (np.linalg.norm(original[w1]) * np.linalg.norm(original[w2]) + 1e-12))
            s_deb = float(np.dot(debiased[w1], debiased[w2]) / (np.linalg.norm(debiased[w1]) * np.linalg.norm(debiased[w2]) + 1e-12))
            orig_sims.append(s_orig)
            deb_sims.append(s_deb)

    if len(orig_sims) < 3:
        return {"spearman_rho": 1.0, "mean_absolute_diff": 0.0, "evaluated_pairs": len(orig_sims)}

    rho, _ = spearmanr(orig_sims, deb_sims)
    abs_diff = float(np.mean(np.abs(np.array(orig_sims) - np.array(deb_sims))))

    return {
        "spearman_rho": round(float(rho), 4),
        "mean_absolute_diff": round(abs_diff, 4),
        "evaluated_pairs": len(orig_sims),
    }


def before_after_report(
    embeddings: Optional[KeyedVectors] = None,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute and export comprehensive WEAT and projection audit before/after debiasing."""
    out_path = save_path or config.DEBIAS_REPORT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    orig_kv = embeddings or load_glove()
    v_g = gender_direction(embeddings=orig_kv)
    deb_kv = hard_debias(embeddings=orig_kv, g_direction=v_g)

    # Save debiased keyed vectors checkpoint
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    deb_kv.save(str(config.DEBIASED_EMBEDDINGS_PATH))

    records: List[Dict[str, Any]] = []

    # 1. Audit WEAT tests
    for name, s in WEAT_SETS.items():
        res_orig = weat(s["X"], s["Y"], s["A"], s["B"], n_perm=1000, embeddings=orig_kv)
        res_deb = weat(s["X"], s["Y"], s["A"], s["B"], n_perm=1000, embeddings=deb_kv)
        records.append({
            "metric_type": "WEAT Effect Size",
            "test_target": name,
            "original": res_orig["effect_size"],
            "debiased": res_deb["effect_size"],
            "delta": round(res_deb["effect_size"] - res_orig["effect_size"], 4),
            "p_original": res_orig["p_value"],
            "p_debiased": res_deb["p_value"],
        })

    # 2. Audit Profession projections
    with open(config.OCCUPATIONS_PATH, "r", encoding="utf-8") as f:
        occs = [l.strip() for l in f if l.strip()]

    proj_orig = project_words(occs, v_g, orig_kv)
    proj_deb = project_words(occs, v_g, deb_kv)

    mean_abs_orig = float(np.mean(np.abs(list(proj_orig.values())))) if proj_orig else 0.0
    mean_abs_deb = float(np.mean(np.abs(list(proj_deb.values())))) if proj_deb else 0.0

    records.append({
        "metric_type": "Mean |Projection|",
        "test_target": "professions_50plus",
        "original": round(mean_abs_orig, 4),
        "debiased": round(mean_abs_deb, 4),
        "delta": round(mean_abs_deb - mean_abs_orig, 4),
        "p_original": None,
        "p_debiased": None,
    })

    # 3. Quality preservation check
    q_check = quality_check(orig_kv, deb_kv)
    records.append({
        "metric_type": "Semantic Quality (Spearman Rho)",
        "test_target": "word_similarity",
        "original": 1.0,
        "debiased": q_check["spearman_rho"],
        "delta": round(q_check["spearman_rho"] - 1.0, 4),
        "p_original": None,
        "p_debiased": None,
    })

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print("=== FairText Hard Debiasing Audit Report ===")
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    before_after_report()
