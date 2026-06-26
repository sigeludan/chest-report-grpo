#!/usr/bin/env python3
"""Batch inference and CheXbert micro-F1 evaluation on impression sections."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from infer_utils import (
    DEFAULT_ADAPTER,
    DEFAULT_BASE_MODEL,
    PROJECT_ROOT,
    add_project_to_path,
    load_eval_samples,
    load_predictions,
    run_inference,
)
from infer_utils import ReportGenerator

add_project_to_path()
from rewards.clinical_reward import compute_clinical_reward
from rewards.format_reward import compute_format_reward, parse_report


def evaluate_records(records: list[dict]) -> dict:
    format_scores: list[float] = []
    clinical_scores: list[float] = []
    valid_clinical = 0
    examples: list[dict] = []

    for record in records:
        prediction = record.get("prediction", "")
        impression_gt = record.get("impression_gt", "")
        format_score = compute_format_reward(prediction)
        format_scores.append(format_score)

        parsed = parse_report(prediction)
        clinical_score = 0.0
        impression_pred = ""
        if parsed is not None and impression_gt:
            _, impression_pred = parsed
            clinical_score = compute_clinical_reward(impression_pred, impression_gt)
            valid_clinical += 1
        clinical_scores.append(clinical_score)

        if len(examples) < 3:
            examples.append(
                {
                    "id": record.get("id"),
                    "format_score": format_score,
                    "clinical_score": clinical_score,
                    "impression_pred": impression_pred,
                    "impression_gt": impression_gt,
                }
            )

    total = len(records)
    format_passed = sum(1 for score in format_scores if score >= 1.0)
    return {
        "total": total,
        "format_rate": format_passed / total if total else 0.0,
        "format_passed": format_passed,
        "mean_clinical_f1": sum(clinical_scores) / valid_clinical if valid_clinical else 0.0,
        "clinical_evaluated": valid_clinical,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated chest reports.")
    parser.add_argument("--predictions", type=Path, help="Existing predictions JSONL.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "eval_metrics.json")
    parser.add_argument(
        "--pred-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "eval_predictions.jsonl",
    )

    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--chexbert-checkpoint",
        type=Path,
        default=Path(os.environ.get("CHEXBERT_CHECKPOINT", PROJECT_ROOT / "models" / "chexbert" / "chexbert.pth")),
    )
    args = parser.parse_args()

    if args.chexbert_checkpoint:
        os.environ["CHEXBERT_CHECKPOINT"] = str(args.chexbert_checkpoint)

    if args.predictions is not None:
        records = load_predictions(args.predictions)
    else:
        samples = load_eval_samples(split=args.split, data_path=args.data_path, limit=args.limit)
        generator = ReportGenerator(base_model=args.base_model, adapter_path=args.adapter)
        records = run_inference(generator, samples, output_path=args.pred_output)

    metrics = evaluate_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
