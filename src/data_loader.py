"""Data loading and preprocessing dataset splitting for FairText.

Loads WNC, BABE, WinoBias, WinoGender, StereoSet, CrowS-Pairs, and custom labeled data.
Builds stratified train/val/test splits saved to data/processed/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)


def load_custom_labeled(path: Optional[Path] = None) -> pd.DataFrame:
    """Load custom labeled dataset (text, label, source)."""
    file_path = path or config.RAW_CUSTOM_PATH
    if not file_path.exists():
        raise FileNotFoundError(
            f"Custom labeled dataset not found at {file_path}. "
            "Please create data/raw/custom_labeled.csv with columns: text, label, source."
        )
    df = pd.read_csv(file_path)
    required_cols = {"text", "label", "source"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{file_path} must contain columns {required_cols}")
    df["label"] = df["label"].astype(int)
    return df[["text", "label", "source"]].dropna()


def load_wnc(path: Optional[Path] = None) -> pd.DataFrame:
    """Load Wikipedia Neutrality Corpus (WNC).

    Attempts to load via Hugging Face datasets first; if unavailable,
    falls back to local CSV at data/raw/wnc.csv.
    """
    file_path = path or config.RAW_WNC_PATH
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("wiki_neutrality", split="train[:500]")
        df = pd.DataFrame(ds)
        # Standardize columns to text, label, source
        text_col = "sentence" if "sentence" in df.columns else df.columns[0]
        label_col = "label" if "label" in df.columns else df.columns[1]
        df = df.rename(columns={text_col: "text", label_col: "label"})
        df["source"] = "wnc_hf"
        return df[["text", "label", "source"]].dropna()
    except Exception as exc:
        logger.warning("Hugging Face WNC download unavailable (%s). Trying local fallback.", exc)
        if file_path.exists():
            df = pd.read_csv(file_path)
            return df[["text", "label", "source"]].dropna()
        raise FileNotFoundError(
            f"WNC dataset not found. Download the WNC dataset and place it at {file_path} "
            "with columns: text, label, source."
        ) from exc


def load_babe(path: Optional[Path] = None) -> pd.DataFrame:
    """Load Media Bias dataset (BABE).

    Attempts to load via Hugging Face datasets; falls back to data/raw/babe.csv.
    """
    file_path = path or config.RAW_BABE_PATH
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("mediabiasgroup/BABE", split="train[:500]")
        df = pd.DataFrame(ds)
        text_col = "text" if "text" in df.columns else df.columns[0]
        label_col = "label" if "label" in df.columns else df.columns[1]
        df = df.rename(columns={text_col: "text", label_col: "label"})
        df["label"] = df["label"].apply(lambda v: 1 if str(v).lower() in {"1", "bias", "biased"} else 0)
        df["source"] = "babe_hf"
        return df[["text", "label", "source"]].dropna()
    except Exception as exc:
        logger.warning("Hugging Face BABE download unavailable (%s). Trying local fallback.", exc)
        if file_path.exists():
            df = pd.read_csv(file_path)
            return df[["text", "label", "source"]].dropna()
        raise FileNotFoundError(
            f"BABE dataset not found. Download BABE and place it at {file_path} "
            "with columns: text, label, source."
        ) from exc


def load_winobias(path: Optional[Path] = None) -> pd.DataFrame:
    """Load WinoBias occupation-pronoun dataset for evaluation."""
    file_path = path or config.RAW_WINOBIAS_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"WinoBias benchmark file missing at {file_path}")
    return pd.read_csv(file_path).dropna()


def load_winogender(path: Optional[Path] = None) -> pd.DataFrame:
    """Load WinoGender benchmark dataset for evaluation."""
    file_path = path or config.RAW_WINOGENDER_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"WinoGender benchmark file missing at {file_path}")
    return pd.read_csv(file_path).dropna()


def load_stereoset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load StereoSet gender evaluation pairs (test only)."""
    file_path = path or config.RAW_STEREOSET_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"StereoSet evaluation file missing at {file_path}")
    return pd.read_csv(file_path).dropna()


def load_crows_pairs(path: Optional[Path] = None) -> pd.DataFrame:
    """Load CrowS-Pairs gender evaluation pairs (test only)."""
    file_path = path or config.RAW_CROWS_PAIRS_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"CrowS-Pairs evaluation file missing at {file_path}")
    return pd.read_csv(file_path).dropna()


def print_distribution(name: str, df: pd.DataFrame) -> None:
    """Print dataset size and class balance distribution."""
    total = len(df)
    counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}
    print(f"[{name}] Total: {total} | Distribution: {counts}")


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed: int = config.SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified split into train, validation, and test subsets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df["label"],
        random_state=seed,
    )
    test_relative_ratio = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_relative_ratio,
        stratify=val_test_df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_classifier_dataset(
    seed: int = config.SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge sources, deduplicate, stratify split, and save to data/processed/."""
    dfs = [load_custom_labeled()]

    try:
        dfs.append(load_wnc())
    except Exception as err:
        logger.warning("Skipping WNC: %s", err)

    try:
        dfs.append(load_babe())
    except Exception as err:
        logger.warning("Skipping BABE: %s", err)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 0]

    train_df, val_df, test_df = split_dataset(
        combined,
        config.TRAIN_RATIO,
        config.VAL_RATIO,
        config.TEST_RATIO,
        seed,
    )

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.TRAIN_PATH, index=False)
    val_df.to_csv(config.VAL_PATH, index=False)
    test_df.to_csv(config.TEST_PATH, index=False)

    print("=== FairText Classifier Dataset Summary ===")
    print_distribution("Combined Cleaned", combined)
    print_distribution("Train Split", train_df)
    print_distribution("Validation Split", val_df)
    print_distribution("Test Split", test_df)

    return train_df, val_df, test_df


if __name__ == "__main__":
    build_classifier_dataset()
