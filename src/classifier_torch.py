"""PyTorch and Hugging Face DistilBERT fine-tuning classifier for FairText.

Includes custom training loop with AdamW, linear learning rate warmup,
class-imbalance weighted cross-entropy loss, evaluation metrics, and inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from src import config

logger = logging.getLogger(__name__)

_TORCH_MODEL: Optional[PreTrainedModel] = None
_TORCH_TOKENIZER: Optional[PreTrainedTokenizer] = None


class FairTextTorchDataset(Dataset):
    """PyTorch Dataset wrapper for tokenized sequence inputs and binary labels."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_len: int = config.MAX_LEN,
    ):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _compute_class_weights(labels: List[int], device: torch.device) -> Optional[torch.Tensor]:
    """Compute weighted loss factors if class distribution exceeds 60/40 imbalance."""
    counts = np.bincount(labels, minlength=2)
    tot = len(labels)
    if tot == 0:
        return None
    ratio_0 = counts[0] / tot
    if ratio_0 > 0.60 or ratio_0 < 0.40:
        weights = [tot / (2.0 * max(counts[0], 1)), tot / (2.0 * max(counts[1], 1))]
        return torch.tensor(weights, dtype=torch.float32, device=device)
    return None


def _train_epoch(
    model: PreTrainedModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Execute one training epoch."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


def _validate_loss(
    model: PreTrainedModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute validation loss on validation loader."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            val_loss += criterion(outputs.logits, targets).item()
    return val_loss / max(len(dataloader), 1)


def train_torch_model(
    train_path: Optional[Path] = None,
    val_path: Optional[Path] = None,
    epochs: int = config.EPOCHS,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Fine-tune DistilBERT sequence classifier and save the best checkpoint."""
    config.set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_df = pd.read_csv(train_path or config.TRAIN_PATH)
    va_df = pd.read_csv(val_path or config.VAL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.TRANSFORMER_MODEL_NAME, num_labels=2
    ).to(device)

    train_ds = FairTextTorchDataset(tr_df["text"].tolist(), tr_df["label"].tolist(), tokenizer)
    val_ds = FairTextTorchDataset(va_df["text"].tolist(), va_df["label"].tolist(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    weights = _compute_class_weights(tr_df["label"].tolist(), device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR_TORCH, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    config.TORCH_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_loss = _train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        va_loss = _validate_loss(model, val_loader, criterion, device)
        print(f"[Epoch {epoch}/{epochs}] Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            model.save_pretrained(config.TORCH_MODEL_DIR)
            tokenizer.save_pretrained(config.TORCH_MODEL_DIR)

    return model, tokenizer


def load_torch_classifier() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load cached fine-tuned DistilBERT model and tokenizer."""
    global _TORCH_MODEL, _TORCH_TOKENIZER
    if _TORCH_MODEL is None or _TORCH_TOKENIZER is None:
        if not (config.TORCH_MODEL_DIR / "model.safetensors").exists() and not (config.TORCH_MODEL_DIR / "pytorch_model.bin").exists():
            train_torch_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _TORCH_TOKENIZER = AutoTokenizer.from_pretrained(config.TORCH_MODEL_DIR)
        _TORCH_MODEL = AutoModelForSequenceClassification.from_pretrained(config.TORCH_MODEL_DIR).to(device)
        _TORCH_MODEL.eval()
    return _TORCH_MODEL, _TORCH_TOKENIZER


def evaluate_torch_model(
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    test_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate fine-tuned DistilBERT on test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = model or load_torch_classifier()[0]
    tok = tokenizer or load_torch_classifier()[1]
    clf.eval()

    df = pd.read_csv(test_path or config.TEST_PATH)
    test_ds = FairTextTorchDataset(df["text"].tolist(), df["label"].tolist(), tok)
    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    all_preds: List[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = clf(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)

    y_true = df["label"].to_numpy().astype(int)
    y_pred = np.array(all_preds)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def predict_torch(text: str) -> Tuple[int, float]:
    """Predict bias label (0: neutral, 1: biased) and probability using DistilBERT."""
    clf, tok = load_torch_classifier()
    device = next(clf.parameters()).device
    encoding = tok(
        str(text),
        truncation=True,
        padding="max_length",
        max_length=config.MAX_LEN,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = clf(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    prob_biased = float(probs[1])
    label = 1 if prob_biased >= 0.5 else 0
    return label, round(prob_biased, 4)


if __name__ == "__main__":
    trained_m, trained_tok = train_torch_model(epochs=2)
    eval_res = evaluate_torch_model(trained_m, trained_tok)
    print("DistilBERT Test Metrics:", eval_res)
    sample_text = "The doctor finished his rounds."
    print(f"Sample prediction for '{sample_text}':", predict_torch(sample_text))
