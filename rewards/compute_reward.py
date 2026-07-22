"""EasyR1 entry point combining format, clinical, and consistency rewards."""

from __future__ import annotations

from typing import Any

from .clinical_reward import compute_clinical_rewards_from_responses
from .consistency_reward import compute_consistency_rewards_from_responses
from .format_reward import compute_format_reward

REWARD_NAME = "chest_report"
REWARD_TYPE = "batch"


def compute_score(
    reward_inputs: list[dict[str, Any]],
    lambda_clinical: float = 1.0,
    lambda_consistency: float = 0.0,
    r_normal: float = 1.0,
    gamma_fp: float = 0.25,
    gamma_fp_severe: float = 0.4,
    fbeta: float = 2.0,
    lambda_abnormal: float = 1.5,
) -> list[dict[str, float]]:
    """Compute composite reward for EasyR1.

    R = R_format + lambda_clinical * R_clinical + lambda_consistency * R_consistency

    R_clinical (v2):
      - GT normal: r_normal if no disease FP, else max(0, r_normal - Σ gamma_fp)
      - GT abnormal: lambda_abnormal * F_beta (default F2)

    ground_truth should be JSON:
      {"impression_gt": "...", "findings_gt": "..."}
    """
    responses = [item["response"] for item in reward_inputs]
    ground_truths = [item["ground_truth"] for item in reward_inputs]

    format_scores = [compute_format_reward(response) for response in responses]
    clinical_scores = compute_clinical_rewards_from_responses(
        responses,
        ground_truths,
        use_v2=True,
        r_normal=r_normal,
        gamma_fp=gamma_fp,
        gamma_fp_severe=gamma_fp_severe,
        fbeta=fbeta,
        lambda_abnormal=lambda_abnormal,
    )
    consistency_scores = compute_consistency_rewards_from_responses(responses)

    scores: list[dict[str, float]] = []
    for format_score, clinical_score, consistency_score in zip(
        format_scores, clinical_scores, consistency_scores
    ):
        if format_score <= 0.0:
            clinical_score = 0.0
            consistency_score = 0.0

        overall = format_score + lambda_clinical * clinical_score + lambda_consistency * consistency_score
        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "clinical": clinical_score,
                "consistency": consistency_score,
            }
        )
    return scores


if __name__ == "__main__":
    demo_inputs = [
        {
            "response": (
                "<findings>The lungs are clear. No pleural effusion.</findings>"
                "<impression>No acute cardiopulmonary disease.</impression>"
            ),
            "response_length": 20,
            "ground_truth": (
                '{"impression_gt": "No acute cardiopulmonary abnormality.", '
                '"findings_gt": "Lungs are clear."}'
            ),
        },
        {
            "response": "missing tags",
            "response_length": 12,
            "ground_truth": '{"impression_gt": "Normal study."}',
        },
    ]
    for item, score in zip(demo_inputs, compute_score(demo_inputs, lambda_consistency=0.3)):
        print(item["response"][:60], "...", score)
