#!/usr/bin/env python3
"""Prepare / verify GRPO data under the **original** train/val/test distribution.

GRPO does NOT use sft_balanced upsampling. It uses the same uid split as
`data/sft/` produced by `scripts/preprocess_iu_xray.py`.

This script:
  1) Optionally re-runs preprocess (--skip-extract) to refresh data/grpo/*.jsonl
  2) Writes data/grpo/data_stats.json (counts + approx abnormal ratio by keyword)

Example:
  python3 scripts/prepare_grpo_data.py
  python3 scripts/prepare_grpo_data.py --refresh
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRPO_DIR = PROJECT_ROOT / "data" / "grpo"
SFT_DIR = PROJECT_ROOT / "data" / "sft"

DISEASE_KW = re.compile(
    r"\b(effusion|pneumothorax|pneumonia|edema|atelectasis|cardiomegaly|"
    r"opacity|consolidation|infiltrate|fracture|lesion|emphysema|scar|mass|nodule)\b",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def impression_gt(row: dict) -> str:
    try:
        return str(json.loads(row["answer"]).get("impression_gt", ""))
    except (json.JSONDecodeError, KeyError, TypeError):
        return ""


def summarize_split(path: Path) -> dict:
    rows = load_jsonl(path)
    abn = sum(1 for r in rows if DISEASE_KW.search(impression_gt(r)))
    n = len(rows)
    return {
        "num_samples": n,
        "abnormal_impression_kw": abn,
        "normal_impression_kw": n - abn,
        "abnormal_ratio_kw": round(abn / n, 4) if n else 0.0,
        "file": str(path.relative_to(PROJECT_ROOT)),
    }


def verify_align_with_sft() -> bool:
    sft_train = json.loads((SFT_DIR / "train.json").read_text(encoding="utf-8"))
    grpo_train = load_jsonl(GRPO_DIR / "train.jsonl")
    sft_imgs = {tuple(x["images"]) for x in sft_train}
    grpo_imgs = {tuple(r["images"]) for r in grpo_train}
    return sft_imgs == grpo_imgs


def write_stats() -> dict:
    stats = {
        "policy": "original_distribution_no_upsample",
        "note": (
            "Same uid split as data/sft from preprocess_iu_xray.py. "
            "Do NOT use data/sft_balanced for GRPO."
        ),
        "splits": {},
        "align_with_sft_train_images": False,
    }
    for split in ("train", "val", "test"):
        path = GRPO_DIR / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run with --refresh or preprocess_iu_xray.py")
        stats["splits"][split] = summarize_split(path)
    stats["align_with_sft_train_images"] = verify_align_with_sft()
    out = GRPO_DIR / "data_stats.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def refresh_from_preprocess() -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "preprocess_iu_xray.py"),
        "--skip-extract",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run preprocess_iu_xray.py --skip-extract to regenerate data/grpo (and data/sft).",
    )
    args = parser.parse_args()

    if args.refresh:
        refresh_from_preprocess()

    stats = write_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if not stats["align_with_sft_train_images"]:
        print("WARNING: GRPO train image set != data/sft/train.json", file=sys.stderr)
        sys.exit(1)
    print(f"\nWrote {GRPO_DIR / 'data_stats.json'}")
    print("GRPO uses ORIGINAL distribution (no stratified upsampling).")


if __name__ == "__main__":
    main()
