"""Configuration module for FairText.

Houses all paths, random seeds, model identifiers, hyperparameters,
and scoring weights. Centralizes all configurable aspects of the project.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Reproducibility & Seeds
# ---------------------------------------------------------------------------
SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Set global random seed across Python, NumPy, PyTorch, and TensorFlow."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Directory Structure & File Paths
# ---------------------------------------------------------------------------
SRC_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SRC_DIR.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
DATA_LEXICONS_DIR: Path = DATA_DIR / "lexicons"

MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Raw Data Paths
RAW_CUSTOM_PATH: Path = DATA_RAW_DIR / "custom_labeled.csv"
RAW_WNC_PATH: Path = DATA_RAW_DIR / "wnc.csv"
RAW_BABE_PATH: Path = DATA_RAW_DIR / "babe.csv"
RAW_WINOBIAS_PATH: Path = DATA_RAW_DIR / "winobias.csv"
RAW_WINOGENDER_PATH: Path = DATA_RAW_DIR / "winogender.csv"
RAW_STEREOSET_PATH: Path = DATA_RAW_DIR / "stereoset.csv"
RAW_CROWS_PAIRS_PATH: Path = DATA_RAW_DIR / "crows_pairs.csv"

# Processed Split Paths
TRAIN_PATH: Path = DATA_PROCESSED_DIR / "train.csv"
VAL_PATH: Path = DATA_PROCESSED_DIR / "val.csv"
TEST_PATH: Path = DATA_PROCESSED_DIR / "test.csv"

# Lexicon Paths
MALE_TERMS_PATH: Path = DATA_LEXICONS_DIR / "male_terms.txt"
FEMALE_TERMS_PATH: Path = DATA_LEXICONS_DIR / "female_terms.txt"
MASCULINE_CODED_PATH: Path = DATA_LEXICONS_DIR / "masculine_coded.txt"
FEMININE_CODED_PATH: Path = DATA_LEXICONS_DIR / "feminine_coded.txt"
NEUTRAL_ALTERNATIVES_PATH: Path = DATA_LEXICONS_DIR / "neutral_alternatives.json"
OCCUPATIONS_PATH: Path = DATA_LEXICONS_DIR / "occupations.txt"

# Model Checkpoints & Output Artifacts
TF_MODEL_PATH: Path = MODELS_DIR / "classifier_tf.keras"
TORCH_MODEL_DIR: Path = MODELS_DIR / "distilbert_fairtext"
DEBIASED_EMBEDDINGS_PATH: Path = MODELS_DIR / "glove_debiased.kv"
DEBIAS_REPORT_PATH: Path = RESULTS_DIR / "debias_before_after.csv"
PROFESSION_GENDER_PLOT_PATH: Path = RESULTS_DIR / "profession_gender_distribution.png"
PROJECTION_PLOT_PATH: Path = RESULTS_DIR / "profession_embeddings_projection.png"
TF_TRAINING_PLOT_PATH: Path = RESULTS_DIR / "tf_training_history.png"
SUMMARY_JSON_PATH: Path = RESULTS_DIR / "summary.json"
ERRORS_CSV_PATH: Path = RESULTS_DIR / "errors.csv"


# ---------------------------------------------------------------------------
# NLP & Embedding Model Constants
# ---------------------------------------------------------------------------
GLOVE_MODEL_NAME: str = "glove-wiki-gigaword-100"
TRANSFORMER_MODEL_NAME: str = "distilbert-base-uncased"
SPACY_MODEL_NAME: str = "en_core_web_md"


# ---------------------------------------------------------------------------
# Classifier Hyperparameters
# ---------------------------------------------------------------------------
MAX_LEN: int = 64
BATCH_SIZE: int = 32
EPOCHS: int = 3
LR_TF: float = 1e-3
LR_TORCH: float = 2e-5

TRAIN_RATIO: float = 0.8
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.1

# Early stopping patience
PATIENCE: int = 2


# ---------------------------------------------------------------------------
# Scoring Weights (Must sum to 100.0)
# ---------------------------------------------------------------------------
WEIGHT_GENDER_RATIO: float = 20.0
WEIGHT_CODED_SHARE: float = 25.0
WEIGHT_STEREOTYPE_LINKS: float = 30.0
WEIGHT_CLASSIFIER_PROB: float = 25.0


def get_config_summary() -> Dict[str, Any]:
    """Return dictionary summary of primary configuration constants."""
    return {
        "seed": SEED,
        "glove_model": GLOVE_MODEL_NAME,
        "transformer_model": TRANSFORMER_MODEL_NAME,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "ratios": (TRAIN_RATIO, VAL_RATIO, TEST_RATIO),
        "weights": {
            "gender_ratio": WEIGHT_GENDER_RATIO,
            "coded_share": WEIGHT_CODED_SHARE,
            "stereotype_links": WEIGHT_STEREOTYPE_LINKS,
            "classifier_prob": WEIGHT_CLASSIFIER_PROB,
        },
    }
