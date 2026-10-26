"""TensorFlow/Keras baseline classifier for FairText.

BiLSTM with pre-trained GloVe 100d embeddings, early stopping,
performance metric computation, and training curve plotting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src import config
from src.embeddings_audit import load_glove

_TF_MODEL: Optional[keras.Model] = None
_TF_VOCAB: Optional[Dict[str, int]] = None
VOCAB_PATH: Path = config.MODELS_DIR / "tf_vocab.json"


def build_vocab(texts: List[str], max_words: int = 10000) -> Dict[str, int]:
    """Construct token-to-index vocabulary from training texts."""
    counts: Dict[str, int] = {}
    for text in texts:
        for w in text.lower().split():
            clean_w = "".join(c for c in w if c.isalnum())
            if clean_w:
                counts[clean_w] = counts.get(clean_w, 0) + 1

    sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:max_words]
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, _) in enumerate(sorted_words, start=2):
        vocab[word] = idx
    return vocab


def encode_texts(texts: List[str], vocab: Dict[str, int], max_len: int = config.MAX_LEN) -> np.ndarray:
    """Encode strings into fixed-length padded integer sequences."""
    encoded = np.zeros((len(texts), max_len), dtype=np.int32)
    for row_idx, text in enumerate(texts):
        words = ["".join(c for c in w if c.isalnum()) for w in text.lower().split()]
        words = [w for w in words if w][:max_len]
        for col_idx, w in enumerate(words):
            encoded[row_idx, col_idx] = vocab.get(w, vocab["<UNK>"])
    return encoded


def create_embedding_matrix(vocab: Dict[str, int], dim: int = 100) -> np.ndarray:
    """Build embedding weights matrix from pre-trained GloVe vectors."""
    glove = load_glove()
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    for word, idx in vocab.items():
        if word in glove:
            matrix[idx] = glove[word]
        elif idx > 1:
            matrix[idx] = np.random.normal(0, 0.1, size=(dim,)).astype(np.float32)
    return matrix


def build_tf_model(
    vocab_size: int,
    embedding_matrix: np.ndarray,
    max_len: int = config.MAX_LEN,
    trainable_embeddings: bool = False,
) -> keras.Model:
    """Construct Bidirectional LSTM text classifier with GloVe initialization."""
    config.set_seed(config.SEED)
    model = keras.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=trainable_embeddings,
            mask_zero=True,
        ),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR_TF),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history: keras.callbacks.History, save_path: Optional[Path] = None) -> Path:
    """Save training and validation loss/accuracy curves."""
    out_path = save_path or config.TF_TRAINING_PLOT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"], label="Train Loss", color="#4C72B0", lw=2)
    ax1.plot(history.history["val_loss"], label="Val Loss", color="#C44E52", lw=2)
    ax1.set_title("Loss Trajectory", weight="bold")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["accuracy"], label="Train Acc", color="#4C72B0", lw=2)
    ax2.plot(history.history["val_accuracy"], label="Val Acc", color="#55A868", lw=2)
    ax2.set_title("Accuracy Trajectory", weight="bold")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def train_tf_model(
    train_path: Optional[Path] = None,
    val_path: Optional[Path] = None,
    epochs: int = config.EPOCHS,
) -> Tuple[keras.Model, Dict[str, int]]:
    """Train the TF BiLSTM classifier and save the model and vocabulary."""
    config.set_seed(config.SEED)
    tr_path = train_path or config.TRAIN_PATH
    va_path = val_path or config.VAL_PATH

    train_df = pd.read_csv(tr_path)
    val_df = pd.read_csv(va_path)

    vocab = build_vocab(train_df["text"].tolist())
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    x_train = encode_texts(train_df["text"].tolist(), vocab)
    y_train = train_df["label"].to_numpy().astype(np.float32)
    x_val = encode_texts(val_df["text"].tolist(), vocab)
    y_val = val_df["label"].to_numpy().astype(np.float32)

    emb_matrix = create_embedding_matrix(vocab)
    model = build_tf_model(len(vocab), emb_matrix)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(config.TF_MODEL_PATH)
    plot_history(history)
    return model, vocab


def evaluate_tf_model(
    model: Optional[keras.Model] = None,
    vocab: Optional[Dict[str, int]] = None,
    test_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate TensorFlow classifier on test split and compute metrics."""
    t_path = test_path or config.TEST_PATH
    df = pd.read_csv(t_path)
    clf = model or load_tf_classifier()
    vcb = vocab or load_tf_vocab()

    x_test = encode_texts(df["text"].tolist(), vcb)
    y_true = df["label"].to_numpy().astype(int)

    y_probs = clf.predict(x_test, verbose=0).flatten()
    y_pred = (y_probs >= 0.5).astype(int)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def load_tf_classifier() -> keras.Model:
    """Load cached or saved TensorFlow classifier model."""
    global _TF_MODEL
    if _TF_MODEL is None:
        if not config.TF_MODEL_PATH.exists():
            train_tf_model()
        _TF_MODEL = keras.models.load_model(config.TF_MODEL_PATH)
    return _TF_MODEL


def load_tf_vocab() -> Dict[str, int]:
    """Load cached or saved vocabulary mapping."""
    global _TF_VOCAB
    if _TF_VOCAB is None:
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            _TF_VOCAB = json.load(f)
    return _TF_VOCAB


def predict_tf(text: str) -> Tuple[int, float]:
    """Predict bias label (0: neutral, 1: biased) and posterior probability."""
    clf = load_tf_classifier()
    vocab = load_tf_vocab()
    encoded = encode_texts([text], vocab)
    prob = float(clf.predict(encoded, verbose=0)[0][0])
    label = 1 if prob >= 0.5 else 0
    return label, round(prob, 4)


if __name__ == "__main__":
    trained_model, trained_vocab = train_tf_model(epochs=3)
    eval_metrics = evaluate_tf_model(trained_model, trained_vocab)
    print("TF Test Evaluation:", eval_metrics)
    test_sample = "She is far too emotional to lead this team."
    print(f"Sample prediction for '{test_sample}':", predict_tf(test_sample))
