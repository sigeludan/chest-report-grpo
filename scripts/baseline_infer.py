#!/usr/bin/env python3
"""Zero-shot baseline inference with the base Qwen2.5-VL model (no LoRA adapter)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infer_utils import (
    DEFAULT_BASE_MODEL,
    DEFAULT_GRPO_TEST,
    DEFAULT_GRPO_VAL,
    PROJECT_ROOT,
    load_eval_samples,
    run_inference,
)
from infer_utils import ReportGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot baseline inference.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to outputs/baseline_{split}.jsonl",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = PROJECT_ROOT / "outputs" / f"baseline_{args.split}.jsonl"

    data_path = args.data_path
    if data_path is None and args.split == "test":
        data_path = DEFAULT_GRPO_TEST
    elif data_path is None and args.split == "val":
        data_path = DEFAULT_GRPO_VAL

    samples = load_eval_samples(split=args.split, data_path=data_path, limit=args.limit)
    generator = ReportGenerator(base_model=args.base_model, adapter_path=None)
    records = run_inference(generator, samples, output_path=output)

    summary = {
        "split": args.split,
        "total": len(records),
        "output": str(output),
        "base_model": str(args.base_model),
        "adapter": None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
