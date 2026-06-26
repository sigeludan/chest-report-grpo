#!/usr/bin/env python3
"""Check <findings>/<impression> format compliance on predictions or ground truth."""

from __future__ import annotations

import argparse
import json
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
from rewards.format_reward import compute_format_reward, parse_report


def score_records(records: list[dict], text_field: str = "prediction") -> dict:
    scores: list[float] = []
    failures: list[dict] = []

    for record in records:
        text = record.get(text_field, "")
        score = compute_format_reward(text)
        scores.append(score)
        if score < 1.0:
            failures.append(
                {
                    "id": record.get("id"),
                    "score": score,
                    "text": text,
                    "parsed": parse_report(text),
                }
            )

    total = len(scores)
    passed = sum(1 for score in scores if score >= 1.0)
    rate = passed / total if total else 0.0
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "format_rate": rate,
        "target_rate": 0.95,
        "meets_target": rate >= 0.95,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check report format compliance.")
    parser.add_argument("--predictions", type=Path, help="JSONL file with a prediction field.")
    parser.add_argument(
        "--ground-truth",
        action="store_true",
        help="Score assistant labels in the dataset instead of model predictions.",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--data-path", type=Path, help="Override dataset path.")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N samples.")
    parser.add_argument("--show-failures", type=int, default=5, help="Print up to N failures.")
    parser.add_argument("--output", type=Path, help="Write metrics JSON to this path.")

    parser.add_argument("--infer", action="store_true", help="Run model inference before scoring.")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--pred-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "format_check_predictions.jsonl",
        help="Where to save predictions when --infer is set.",
    )
    args = parser.parse_args()

    if args.predictions is not None:
        records = load_predictions(args.predictions)
        text_field = "prediction"
    elif args.infer:
        samples = load_eval_samples(split=args.split, data_path=args.data_path, limit=args.limit)
        generator = ReportGenerator(base_model=args.base_model, adapter_path=args.adapter)
        records = run_inference(generator, samples, output_path=args.pred_output)
        text_field = "prediction"
    elif args.ground_truth:
        samples = load_eval_samples(split=args.split, data_path=args.data_path, limit=args.limit)
        records = [
            {
                "id": sample.sample_id,
                "prediction": (
                    f"<findings>{sample.findings_gt}</findings>\n"
                    f"<impression>{sample.impression_gt}</impression>"
                ),
            }
            for sample in samples
        ]
        text_field = "prediction"
    else:
        parser.error("Provide --predictions, or use --infer / --ground-truth.")

    metrics = score_records(records, text_field=text_field)
    summary = {key: metrics[key] for key in ("total", "passed", "failed", "format_rate", "target_rate", "meets_target")}

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if metrics["failures"] and args.show_failures > 0:
        print("\nFailures:")
        for item in metrics["failures"][: args.show_failures]:
            print(f"- id={item['id']}")
            print(f"  text={item['text'][:300]}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
