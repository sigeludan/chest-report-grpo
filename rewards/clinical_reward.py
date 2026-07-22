"""Clinical reward: asymmetric CheXbert score (normal branch + Fβ for abnormal)."""

from __future__ import annotations

from typing import Sequence

from .chexbert_utils import (
    CheXbertLabeler,
    clinical_score_v2,
    micro_f1_positive,
    predict_labels,
)
from .format_reward import parse_ground_truth, parse_report


def compute_clinical_reward(
    impression_pred: str,
    impression_gt: str,
    labeler: CheXbertLabeler | None = None,
    *,
    use_v2: bool = True,
    r_normal: float = 1.0,
    gamma_fp: float = 0.25,
    gamma_fp_severe: float = 0.4,
    fbeta: float = 2.0,
    lambda_abnormal: float = 1.5,
) -> float:
    """Compute R_clinical for a single sample."""
    impression_pred = impression_pred.strip()
    impression_gt = impression_gt.strip()
    if not impression_pred or not impression_gt:
        return 0.0

    labeler = labeler or CheXbertLabeler.get()
    label_vectors = predict_labels([impression_pred, impression_gt], labeler=labeler)
    if use_v2:
        return clinical_score_v2(
            label_vectors[0],
            label_vectors[1],
            r_normal=r_normal,
            gamma_fp=gamma_fp,
            gamma_fp_severe=gamma_fp_severe,
            fbeta=fbeta,
            lambda_abnormal=lambda_abnormal,
        )
    return micro_f1_positive(label_vectors[0], label_vectors[1])


def compute_clinical_rewards(
    impression_preds: Sequence[str],
    impression_gts: Sequence[str],
    labeler: CheXbertLabeler | None = None,
    *,
    use_v2: bool = True,
    r_normal: float = 1.0,
    gamma_fp: float = 0.25,
    gamma_fp_severe: float = 0.4,
    fbeta: float = 2.0,
    lambda_abnormal: float = 1.5,
) -> list[float]:
    """Batch version used by GRPO reward manager."""
    if len(impression_preds) != len(impression_gts):
        raise ValueError("impression_preds and impression_gts must have the same length.")

    labeler = labeler or CheXbertLabeler.get()
    paired_texts: list[str] = []
    index_map: list[tuple[int, str]] = []
    scores = [0.0] * len(impression_preds)

    for idx, (pred, gt) in enumerate(zip(impression_preds, impression_gts)):
        pred = pred.strip()
        gt = gt.strip()
        if not pred or not gt:
            continue
        index_map.append((idx, "pred"))
        paired_texts.append(pred)
        index_map.append((idx, "gt"))
        paired_texts.append(gt)

    if not paired_texts:
        return scores

    label_vectors = predict_labels(paired_texts, labeler=labeler)
    cursor = 0
    while cursor < len(index_map):
        sample_idx, role = index_map[cursor]
        assert role == "pred"
        pred_labels = label_vectors[cursor]
        gt_labels = label_vectors[cursor + 1]
        if use_v2:
            scores[sample_idx] = clinical_score_v2(
                pred_labels,
                gt_labels,
                r_normal=r_normal,
                gamma_fp=gamma_fp,
                gamma_fp_severe=gamma_fp_severe,
                fbeta=fbeta,
                lambda_abnormal=lambda_abnormal,
            )
        else:
            scores[sample_idx] = micro_f1_positive(pred_labels, gt_labels)
        cursor += 2

    return scores


def compute_clinical_rewards_from_responses(
    responses: Sequence[str],
    ground_truths: Sequence[str],
    labeler: CheXbertLabeler | None = None,
    **clinical_kwargs,
) -> list[float]:
    """Compute clinical rewards directly from model responses and EasyR1 ground_truth."""
    impression_preds: list[str] = []
    impression_gts: list[str] = []
    valid_indices: list[int] = []
    scores = [0.0] * len(responses)

    for idx, (response, ground_truth) in enumerate(zip(responses, ground_truths)):
        parsed = parse_report(response)
        gt = parse_ground_truth(ground_truth)
        if parsed is None:
            continue
        _, impression = parsed
        impression_gt = gt.get("impression_gt", "")
        if not impression_gt:
            continue
        valid_indices.append(idx)
        impression_preds.append(impression)
        impression_gts.append(impression_gt)

    if not valid_indices:
        return scores

    batch_scores = compute_clinical_rewards(
        impression_preds, impression_gts, labeler=labeler, **clinical_kwargs
    )
    for idx, score in zip(valid_indices, batch_scores):
        scores[idx] = score
    return scores


if __name__ == "__main__":
    # GT normal, pred normal -> 1.0
    normal = [0] * 14
    print("normal/normal:", clinical_score_v2(normal, normal))

    # GT normal, pred cardiomegaly (index 1)
    pred_fp = [0] * 14
    pred_fp[1] = 1
    print("normal/FP cardiomegaly:", clinical_score_v2(pred_fp, normal))

    # GT abnormal: cardiomegaly+opacity, pred only cardiomegaly
    gt_abn = [0] * 14
    gt_abn[1] = 1
    gt_abn[2] = 1
    pred_partial = [0] * 14
    pred_partial[1] = 1
    print("abnormal partial:", clinical_score_v2(pred_partial, gt_abn))
    print("abnormal perfect:", clinical_score_v2(gt_abn, gt_abn))
