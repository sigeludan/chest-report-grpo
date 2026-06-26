"""Format reward: parse <findings> / <impression> structured report output."""

from __future__ import annotations

import json
import re
from typing import Any

FINDINGS_PATTERN = re.compile(r"<findings>\s*(.*?)\s*</findings>", re.DOTALL | re.IGNORECASE)
IMPRESSION_PATTERN = re.compile(r"<impression>\s*(.*?)\s*</impression>", re.DOTALL | re.IGNORECASE)
TAG_PATTERN = re.compile(r"<\s*(/?)\s*(findings|impression)\s*>", re.IGNORECASE)


def normalize_response(response: str) -> str:
    """Normalize spacing around <findings> / <impression> tags only."""
    response = response.strip()
    return TAG_PATTERN.sub(lambda match: f"<{match.group(1)}{match.group(2).lower()}>", response)


def parse_report(response: str) -> tuple[str, str] | None:
    """Parse findings and impression sections from a model response."""
    response = normalize_response(response)
    findings_match = FINDINGS_PATTERN.search(response)
    impression_match = IMPRESSION_PATTERN.search(response)
    if findings_match is None or impression_match is None:
        return None

    findings = findings_match.group(1).strip()
    impression = impression_match.group(1).strip()
    if not findings or not impression:
        return None
    return findings, impression


def parse_ground_truth(ground_truth: str) -> dict[str, str]:
    """Parse EasyR1 ground_truth payload.

    Expected JSON::

      {"impression_gt": "...", "findings_gt": "..."}

    If plain text is provided, treat it as impression_gt.
    """
    ground_truth = ground_truth.strip()
    if not ground_truth:
        return {"impression_gt": "", "findings_gt": ""}

    try:
        payload = json.loads(ground_truth)
    except json.JSONDecodeError:
        return {"impression_gt": ground_truth, "findings_gt": ""}

    if not isinstance(payload, dict):
        return {"impression_gt": ground_truth, "findings_gt": ""}

    return {
        "impression_gt": str(payload.get("impression_gt", "")).strip(),
        "findings_gt": str(payload.get("findings_gt", "")).strip(),
    }


def compute_format_reward(response: str) -> float:
    """Return 1.0 when both sections are present and non-empty, else 0.0."""
    return 1.0 if parse_report(response) is not None else 0.0


def compute_format_scores(responses: list[str]) -> list[float]:
    return [compute_format_reward(response) for response in responses]


if __name__ == "__main__":
    samples = [
        "<findings>Clear lungs.</findings><impression>No acute disease.</impression>",
        "<findings> opacity </findings>\n<impression> pneumonia </impression>",
        "< findings > clear < /findings > < impression > normal < /impression >",
        "A / P ratio is normal. <findings> clear </findings> <impression> normal </impression>",
        "<findings>nodule < 5mm</findings><impression>benign</impression>",
        "<findings>opacity > 2 cm</findings><impression>pneumonia</impression>",
        "<findings>only findings</findings>",
        "free text without tags",
    ]
    for sample in samples:
        parsed = parse_report(sample)
        print(f"score={compute_format_reward(sample):.1f} parsed={parsed}")
