"""CheXbert batch inference utilities for clinical and consistency rewards."""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

CONDITIONS: list[str] = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

PAD_IDX = 0
DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "models" / "chexbert" / "chexbert.pth")


def resolve_checkpoint_path(path: str | None = None) -> str:
    """Resolve CheXbert checkpoint; relative paths are under project root (not cwd)."""
    raw = path or os.environ.get("CHEXBERT_CHECKPOINT", DEFAULT_CHECKPOINT)
    if os.path.isabs(raw):
        return raw
    candidate = PROJECT_ROOT / raw
    if candidate.exists():
        return str(candidate)
    return str((Path.cwd() / raw).resolve())


class BertLabeler(nn.Module):
    """Minimal CheXbert labeler (Stanford CheXbert architecture)."""

    def __init__(self) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.pooler.dense.in_features
        self.dropout = nn.Dropout(0.1)
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, source_padded: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        cls_hidden = self.dropout(final_hidden[:, 0, :])
        return [head(cls_hidden) for head in self.linear_heads]


def preprocess_report_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_report(text: str, tokenizer: BertTokenizer) -> list[int]:
    text = preprocess_report_text(text)
    if not text:
        return [tokenizer.cls_token_id, tokenizer.sep_token_id]

    tokenized = tokenizer.tokenize(text)
    token_ids = tokenizer.encode_plus(tokenized, add_special_tokens=True)["input_ids"]
    if len(token_ids) > 512:
        token_ids = token_ids[:511] + [tokenizer.sep_token_id]
    return token_ids


def generate_attention_masks(batch: torch.Tensor, source_lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    masks = torch.ones(batch.size(0), batch.size(1), dtype=torch.float)
    for idx, src_len in enumerate(source_lengths):
        masks[idx, src_len:] = 0
    return masks.to(device)


def positive_condition_indices(labels: Sequence[int]) -> set[int]:
    return {idx for idx, value in enumerate(labels) if value == 1}


def positive_condition_names(labels: Sequence[int]) -> set[str]:
    return {CONDITIONS[idx] for idx in positive_condition_indices(labels)}


def micro_f1_positive(labels_pred: Sequence[int], labels_gt: Sequence[int]) -> float:
    """Micro-F1 over 14 CheXbert conditions, positive class only (raw argmax == 1)."""
    if len(labels_pred) != len(labels_gt):
        raise ValueError("Prediction and ground-truth label vectors must have the same length.")

    tp = fp = fn = 0
    for pred, gt in zip(labels_pred, labels_gt):
        pred_pos = pred == 1
        gt_pos = gt == 1
        if pred_pos and gt_pos:
            tp += 1
        elif pred_pos and not gt_pos:
            fp += 1
        elif not pred_pos and gt_pos:
            fn += 1

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# Disease heads only (exclude "No Finding") to avoid normal-vs-normal F1 collapse.
DISEASE_INDICES: list[int] = [i for i, name in enumerate(CONDITIONS) if name != "No Finding"]

# Optional heavier FP penalty on clinically urgent false positives (index into CONDITIONS).
SEVERE_FP_INDICES: set[int] = {
    CONDITIONS.index("Pneumothorax"),
    CONDITIONS.index("Pleural Effusion"),
    CONDITIONS.index("Edema"),
}


def _disease_counts(labels_pred: Sequence[int], labels_gt: Sequence[int]) -> tuple[int, int, int, int]:
    """Return TP, FP, FN, and number of predicted disease positives (over 13 disease classes)."""
    tp = fp = fn = 0
    pred_pos_count = 0
    for idx in DISEASE_INDICES:
        pred_pos = labels_pred[idx] == 1
        gt_pos = labels_gt[idx] == 1
        if pred_pos:
            pred_pos_count += 1
        if pred_pos and gt_pos:
            tp += 1
        elif pred_pos and not gt_pos:
            fp += 1
        elif (not pred_pos) and gt_pos:
            fn += 1
    return tp, fp, fn, pred_pos_count


def is_gt_abnormal(labels_gt: Sequence[int]) -> bool:
    return any(labels_gt[idx] == 1 for idx in DISEASE_INDICES)


def fbeta_score(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def clinical_score_v2(
    labels_pred: Sequence[int],
    labels_gt: Sequence[int],
    *,
    r_normal: float = 1.0,
    gamma_fp: float = 0.25,
    gamma_fp_severe: float = 0.4,
    fbeta: float = 2.0,
    lambda_abnormal: float = 1.5,
) -> float:
    """Asymmetric clinical score for GRPO.

    - GT normal: correct normal -> r_normal; each disease FP subtracts gamma_fp (severe FP uses gamma_fp_severe).
    - GT abnormal: lambda_abnormal * F_beta (beta>1 favors recall).
    """
    if len(labels_pred) != len(labels_gt):
        raise ValueError("Prediction and ground-truth label vectors must have the same length.")

    tp, fp, fn, pred_pos_count = _disease_counts(labels_pred, labels_gt)

    if not is_gt_abnormal(labels_gt):
        if pred_pos_count == 0:
            return float(r_normal)
        penalty = 0.0
        for idx in DISEASE_INDICES:
            if labels_pred[idx] == 1 and labels_gt[idx] != 1:
                penalty += gamma_fp_severe if idx in SEVERE_FP_INDICES else gamma_fp
        return float(max(0.0, r_normal - penalty))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(lambda_abnormal * fbeta_score(precision, recall, beta=fbeta))


class CheXbertLabeler:
    """Lazy-loaded singleton CheXbert labeler with batch inference."""

    _instance: "CheXbertLabeler | None" = None
    _lock = threading.Lock()

    def __init__(self, checkpoint_path: str | None = None, device: str | None = None) -> None:
        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: BertLabeler | None = None
        self.tokenizer: BertTokenizer | None = None

    @classmethod
    def get(cls, checkpoint_path: str | None = None, device: str | None = None) -> "CheXbertLabeler":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(checkpoint_path=checkpoint_path, device=device)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                "CheXbert checkpoint not found. Download chexbert.pth and set CHEXBERT_CHECKPOINT, e.g.\n"
                f"  export CHEXBERT_CHECKPOINT={DEFAULT_CHECKPOINT}"
            )

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertLabeler().to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        if any(key.startswith("module.") for key in state_dict):
            state_dict = OrderedDict((key[7:], value) for key, value in state_dict.items())
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

    @torch.inference_mode()
    def predict_batch(self, texts: Sequence[str], batch_size: int = 16) -> list[list[int]]:
        self._load()
        assert self.model is not None
        assert self.tokenizer is not None

        encoded = [tokenize_report(text, self.tokenizer) for text in texts]
        outputs: list[list[int]] = [[] for _ in range(len(CONDITIONS))]

        for start in range(0, len(encoded), batch_size):
            batch_encoded = encoded[start : start + batch_size]
            lengths = [len(item) for item in batch_encoded]
            batch = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(item, dtype=torch.long) for item in batch_encoded],
                batch_first=True,
                padding_value=PAD_IDX,
            ).to(self.device)
            attention_mask = generate_attention_masks(batch, lengths, self.device)
            logits = self.model(batch, attention_mask)
            for head_idx, head_logits in enumerate(logits):
                outputs[head_idx].extend(head_logits.argmax(dim=1).tolist())

        # shape: (num_texts, 14)
        return [list(row) for row in zip(*outputs)]


def predict_labels(texts: Iterable[str], labeler: CheXbertLabeler | None = None) -> list[list[int]]:
    labeler = labeler or CheXbertLabeler.get()
    return labeler.predict_batch(list(texts))
