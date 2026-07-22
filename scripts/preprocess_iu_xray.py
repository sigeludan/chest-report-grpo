#!/usr/bin/env python3
"""Preprocess IU-Xray (Indiana University) archive into SFT and GRPO formats."""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "archive.zip"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_SFT_DIR = PROJECT_ROOT / "data" / "sft"
DEFAULT_GRPO_DIR = PROJECT_ROOT / "data" / "grpo"

PROMPT_TWO_IMAGES = (
    "<image><image>\n"
    "Generate a radiology report for the chest X-rays above (frontal and lateral views). "
    "Use exactly this format:\n"
    # "请根据上述胸片生成放射学报告，严格按以下格式输出：\n"
    "<findings>...</findings>\n"
    "<impression>...</impression>"
)
PROMPT_ONE_IMAGE = (
    "<image>\n"
    "Generate a radiology report for the chest X-ray above. "
    "Use exactly this format:\n"
    # "请根据上述胸片生成放射学报告，严格按以下格式输出：\n"
    "<findings>...</findings>\n"
    "<impression>...</impression>"
)


def normalize_text(text: object) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return " ".join(str(text).split())


def extract_archive(archive_path: Path, raw_dir: Path) -> None:
    reports_csv = raw_dir / "indiana_reports.csv"
    images_dir = raw_dir / "images" / "images_normalized"
    if reports_csv.exists() and images_dir.exists() and any(images_dir.glob("*.png")):
        print(f"Raw data already present under {raw_dir}, skip extraction.")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} -> {raw_dir} ...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(raw_dir)
    print("Extraction done.")


def build_records(raw_dir: Path, project_root: Path, min_text_len: int) -> list[dict]:
    reports = pd.read_csv(raw_dir / "indiana_reports.csv")
    projections = pd.read_csv(raw_dir / "indiana_projections.csv")

    def pick_images(group: pd.DataFrame) -> pd.Series:
        frontal = group[group["projection"].str.contains("Frontal", case=False, na=False)]
        lateral = group[group["projection"].str.contains("Lateral", case=False, na=False)]
        frontal_name = frontal.iloc[0]["filename"] if len(frontal) else None
        lateral_name = lateral.iloc[0]["filename"] if len(lateral) else None
        return pd.Series({"frontal": frontal_name, "lateral": lateral_name})

    image_table = projections.groupby("uid").apply(pick_images, include_groups=False).reset_index()
    merged = reports.merge(image_table, on="uid", how="inner")

    image_root = raw_dir / "images" / "images_normalized"
    records: list[dict] = []

    for row in merged.itertuples(index=False):
        uid = int(row.uid)
        findings = normalize_text(row.findings)
        impression = normalize_text(row.impression)
        if not isinstance(row.frontal, str) or not row.frontal or len(findings) < min_text_len or len(impression) < min_text_len:
            continue

        image_paths: list[str] = []
        frontal_path = image_root / row.frontal
        if not frontal_path.exists():
            continue
        image_paths.append(frontal_path.relative_to(project_root).as_posix())

        has_lateral = isinstance(row.lateral, str) and bool(row.lateral)
        if has_lateral:
            lateral_path = image_root / row.lateral
            if lateral_path.exists():
                image_paths.append(lateral_path.relative_to(project_root).as_posix())
            else:
                has_lateral = False

        user_prompt = PROMPT_TWO_IMAGES if len(image_paths) == 2 else PROMPT_ONE_IMAGE
        assistant_text = f"<findings>{findings}</findings>\n<impression>{impression}</impression>"
        reward_gt = json.dumps({"impression_gt": impression, "findings_gt": findings}, ensure_ascii=False)

        records.append(
            {
                "id": f"iu_{uid:04d}",
                "uid": uid,
                "findings": findings,
                "impression": impression,
                "images": image_paths,
                "user_prompt": user_prompt,
                "assistant_text": assistant_text,
                "reward_gt": reward_gt,
            }
        )

    return records


def split_records(records: list[dict], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[dict]]:
    uids = sorted({record["uid"] for record in records})
    rng = random.Random(seed)
    rng.shuffle(uids)

    n_total = len(uids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    split_uids = {
        "train": set(uids[:n_train]),
        "val": set(uids[n_train : n_train + n_val]),
        "test": set(uids[n_train + n_val :]),
    }

    split_records_map: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for record in records:
        for split_name, uid_set in split_uids.items():
            if record["uid"] in uid_set:
                split_records_map[split_name].append(record)
                break
    return split_records_map


def to_llamafactory_sample(record: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": record["user_prompt"]},
            {"role": "assistant", "content": record["assistant_text"]},
        ],
        "images": record["images"],
    }


def to_easyr1_sample(record: dict) -> dict:
    return {
        "id": record["id"],
        "problem": record["user_prompt"],
        "answer": record["reward_gt"],
        "images": record["images"],
    }


def resolve_project_path(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def write_outputs(
    split_records_map: dict[str, list[dict]],
    sft_dir: Path,
    grpo_dir: Path,
    stats_path: Path,
    min_text_len: int,
) -> None:
    sft_dir.mkdir(parents=True, exist_ok=True)
    grpo_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "min_text_len": min_text_len,
        "splits": {},
    }

    for split_name, items in split_records_map.items():
        sft_path = sft_dir / f"{split_name}.json"
        grpo_path = grpo_dir / f"{split_name}.jsonl"

        sft_samples = [to_llamafactory_sample(item) for item in items]
        with sft_path.open("w", encoding="utf-8") as f:
            json.dump(sft_samples, f, ensure_ascii=False, indent=2)

        with grpo_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(to_easyr1_sample(item), ensure_ascii=False) + "\n")

        stats["splits"][split_name] = {
            "num_samples": len(items),
            "sft_file": sft_path.relative_to(PROJECT_ROOT.resolve()).as_posix(),
            "grpo_file": grpo_path.relative_to(PROJECT_ROOT.resolve()).as_posix(),
        }
        print(f"{split_name}: {len(items)} samples")

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Wrote stats -> {stats_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess IU-Xray archive.zip")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--sft-dir", type=Path, default=DEFAULT_SFT_DIR)
    parser.add_argument("--grpo-dir", type=Path, default=DEFAULT_GRPO_DIR)
    parser.add_argument("--stats-path", type=Path, default=PROJECT_ROOT / "data" / "split_stats.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-text-len", type=int, default=10)
    parser.add_argument("--skip-extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.raw_dir = resolve_project_path(args.raw_dir)
    args.sft_dir = resolve_project_path(args.sft_dir)
    args.grpo_dir = resolve_project_path(args.grpo_dir)
    args.stats_path = resolve_project_path(args.stats_path)
    if not args.skip_extract:
        extract_archive(args.archive, args.raw_dir)

    records = build_records(args.raw_dir, PROJECT_ROOT, args.min_text_len)
    if not records:
        raise RuntimeError(
            "No valid samples found. Ensure archive is extracted and PNG files exist under "
            f"{args.raw_dir / 'images' / 'images_normalized'}"
        )

    split_records_map = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    write_outputs(split_records_map, args.sft_dir, args.grpo_dir, args.stats_path, args.min_text_len)


if __name__ == "__main__":
    main()
