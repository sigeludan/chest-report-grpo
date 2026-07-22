"""Consistency reward: impression positives should be supported by findings."""

from __future__ import annotations

from typing import Sequence

from .chexbert_utils import CheXbertLabeler, positive_condition_indices, predict_labels
from .format_reward import parse_report


def compute_consistency_reward(
    findings: str,
    impression: str,
    labeler: CheXbertLabeler | None = None,
) -> float:
    """Compute R_consistency for one sample.

    R = |P_imp ∩ P_find| / |P_imp|, where P_* are CheXbert-positive label sets.
    If P_imp is empty, return 1.0.
    """
    findings = findings.strip()
    impression = impression.strip()
    if not findings or not impression:
        return 0.0

    labeler = labeler or CheXbertLabeler.get()
    label_vectors = predict_labels([findings, impression], labeler=labeler)
    positive_imp = positive_condition_indices(label_vectors[1])
    positive_find = positive_condition_indices(label_vectors[0])

    if not positive_imp:
        return 1.0
    return len(positive_imp & positive_find) / len(positive_imp)


def compute_consistency_rewards(
    findings_list: Sequence[str],
    impression_list: Sequence[str],
    labeler: CheXbertLabeler | None = None,
) -> list[float]:
    if len(findings_list) != len(impression_list):
        raise ValueError("findings_list and impression_list must have the same length.")

    labeler = labeler or CheXbertLabeler.get()
    texts: list[str] = []
    pair_indices: list[tuple[int, str]] = []
    scores = [0.0] * len(findings_list)

    for idx, (findings, impression) in enumerate(zip(findings_list, impression_list)):
        findings = findings.strip()
        impression = impression.strip()
        if not findings or not impression:
            continue
        pair_indices.append((idx, "findings"))
        texts.append(findings)
        pair_indices.append((idx, "impression"))
        texts.append(impression)

    if not texts:
        return scores

    label_vectors = predict_labels(texts, labeler=labeler)
    cursor = 0
    while cursor < len(pair_indices):
        sample_idx, role = pair_indices[cursor]
        assert role == "findings"
        findings_labels = label_vectors[cursor]
        impression_labels = label_vectors[cursor + 1]
        positive_imp = positive_condition_indices(impression_labels)
        positive_find = positive_condition_indices(findings_labels)
        if not positive_imp:
            scores[sample_idx] = 1.0
        else:
            scores[sample_idx] = len(positive_imp & positive_find) / len(positive_imp)
        cursor += 2

    return scores


def compute_consistency_rewards_from_responses(
    responses: Sequence[str],
    labeler: CheXbertLabeler | None = None,
) -> list[float]:
    """Compute consistency rewards directly from model responses."""
    findings_list: list[str] = []
    impression_list: list[str] = []
    valid_indices: list[int] = []
    scores = [0.0] * len(responses)

    for idx, response in enumerate(responses):
        parsed = parse_report(response)
        if parsed is None:
            continue
        findings, impression = parsed
        valid_indices.append(idx)
        findings_list.append(findings)
        impression_list.append(impression)

    if not valid_indices:
        return scores

    batch_scores = compute_consistency_rewards(findings_list, impression_list, labeler=labeler)
    for idx, score in zip(valid_indices, batch_scores):
        scores[idx] = score
    return scores


if __name__ == "__main__":
    findings_labels = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    impression_labels = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    positive_imp = positive_condition_indices(impression_labels)
    positive_find = positive_condition_indices(findings_labels)
    score = 1.0 if not positive_imp else len(positive_imp & positive_find) / len(positive_imp)
    print("mock consistency:", score)
