#!/usr/bin/env python3
"""Balance SFT train data by impression abnormality (CheXbert / keyword).

- Split samples into normal vs abnormal by GT impression labels.
- Cap repeated normal impression templates (e.g. "No acute ...").
- Upsample abnormal samples so they make up ~target_abnormal_ratio of the set.

This does NOT train normal/abnormal in separate stages. It builds one mixed
training set where abnormal reports appear more often each epoch.

Example:
  export CHEXBERT_CHECKPOINT=/root/autodl-tmp/chest-report-grpo/models/chexbert/chexbert.pth
  python3 scripts/balance_sft_data.py \\
    --input-dir data/sft \\
    --output-dir data/sft_balanced \\
    --target-abnormal-ratio 0.45 \\
    --max-normal-template 40
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rewards.chexbert_utils import CONDITIONS, CheXbertLabeler

IMPRESSION_RE = re.compile(r"<impression>(.*?)</impression>", re.IGNORECASE | re.DOTALL)
DISEASE_IDX = [i for i, name in enumerate(CONDITIONS) if name != "No Finding"]

# Fallback / hybrid assist when CheXbert misses obvious disease mentions.
DISEASE_KW = re.compile(
    r"\b("
    r"effusion|pneumothorax|pneumonia|edema|atelectasis|cardiomegaly|"
    r"opacity|opacities|consolidation|infiltrate|infiltration|fracture|"
    r"lesion|emphysema|scar(?:ring)?|mass|nodules?|"
    r"pulmonary edema|pleural effusion"
    r")\b",
    re.IGNORECASE,
)
NORMAL_GUARD = re.compile(
    r"\b(no acute|no active|no evidence of acute|normal chest|unremarkable)\b",
    re.IGNORECASE,
)


def is_abnormal_by_keyword(impression: str) -> bool:
    """Keyword heuristic: disease term present and not a pure normal template."""
    if not DISEASE_KW.search(impression):
        return False
    # e.g. "Emphysema without acute disease" -> abnormal
    # e.g. "No acute ... pleural disease" without disease kw beyond guard -> normal
    if NORMAL_GUARD.search(impression) and not re.search(
        r"\b(cardiomegaly|emphysema|pneumonia|effusion|pneumothorax|edema|"
        r"atelectasis|opacity|consolidation|fracture|lesion|mass|nodule)\b",
        impression,
        re.IGNORECASE,
    ):
        return False
    return True


def extract_impression(example: dict) -> str:
    messages = example.get("messages") or example.get("conversations") or []
    if not messages:
        return ""
    last = messages[-1]
    text = last.get("content") or last.get("value") or ""
    match = IMPRESSION_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def normalize_impression_key(text: str) -> str:
    """Normalize for template frequency capping."""
    text = " ".join(text.lower().split())
    text = text.rstrip(".")
    return text


def is_abnormal_by_labels(labels: list[int]) -> bool:
    return any(labels[i] == 1 for i in DISEASE_IDX)


def label_impressions(
    impressions: list[str],
    method: str,
    batch_size: int,
) -> list[bool]:
    if method == "keyword":
        return [is_abnormal_by_keyword(text) for text in impressions]

    if method == "chexbert":
        labeler = CheXbertLabeler.get()
        label_vectors = labeler.predict_batch(impressions, batch_size=batch_size)
        return [is_abnormal_by_labels(vec) for vec in label_vectors]

    # hybrid: CheXbert primary, keyword OR for safety on obvious disease text
    labeler = CheXbertLabeler.get()
    label_vectors = labeler.predict_batch(impressions, batch_size=batch_size)
    flags = []
    for text, vec in zip(impressions, label_vectors):
        flags.append(is_abnormal_by_labels(vec) or is_abnormal_by_keyword(text))
    return flags


def cap_normal_templates(
    normal_examples: list[dict],
    impressions: list[str],
    max_per_template: int,
    rng: random.Random,
) -> list[dict]:
    """Keep at most max_per_template samples per identical normal impression."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for ex, imp in zip(normal_examples, impressions):
        buckets[normalize_impression_key(imp)].append(ex)

    kept: list[dict] = []
    capped_templates = 0
    for key, group in buckets.items():
        if len(group) > max_per_template:
            capped_templates += 1
            rng.shuffle(group)
            kept.extend(group[:max_per_template])
        else:
            kept.extend(group)

    print(
        f"  normal templates: {len(buckets)} unique; "
        f"capped {capped_templates} templates to <= {max_per_template}; "
        f"{len(normal_examples)} -> {len(kept)}"
    )
    return kept


def upsample_to_ratio(
    normal: list[dict],
    abnormal: list[dict],
    target_abnormal_ratio: float,
    rng: random.Random,
) -> tuple[list[dict], int, int]:
    """Build mixed set with abnormal ≈ target_abnormal_ratio.

    Keeps capped normals, repeats abnormal samples as needed. If the ratio
    is still too low, downsamples normals. Returns (mixed, n_abn, n_norm).
    """
    if not abnormal:
        raise ValueError("No abnormal samples found; cannot balance.")
    if not (0.05 <= target_abnormal_ratio <= 0.9):
        raise ValueError("target_abnormal_ratio should be in [0.05, 0.9].")

    # Want: n_abn / (n_abn + n_norm) ≈ r  =>  n_abn ≈ r/(1-r) * n_norm
    n_norm = len(normal)
    desired_abn = int(round(target_abnormal_ratio / (1.0 - target_abnormal_ratio) * n_norm))
    desired_abn = max(desired_abn, len(abnormal))

    # Hard ceiling: avoid exploding dataset size
    max_abn = max(len(abnormal) * 15, desired_abn)
    desired_abn = min(desired_abn, max_abn)

    if desired_abn <= len(abnormal):
        abn_out = list(abnormal)
    else:
        abn_out = list(abnormal)
        extra = desired_abn - len(abnormal)
        abn_out.extend(rng.choices(abnormal, k=extra))

    # If still below target (e.g. too many normals), downsample normals
    actual_ratio = len(abn_out) / (len(abn_out) + n_norm)
    if actual_ratio < target_abnormal_ratio - 0.02 and n_norm > 0:
        n_norm_target = int(
            round(len(abn_out) * (1.0 - target_abnormal_ratio) / target_abnormal_ratio)
        )
        n_norm_target = max(n_norm_target, 1)
        if n_norm_target < n_norm:
            normal = rng.sample(normal, n_norm_target)
            print(f"  downsampled normals to {len(normal)} to hit target ratio")

    mixed = list(normal) + list(abn_out)
    rng.shuffle(mixed)
    ratio = len(abn_out) / len(mixed)
    print(
        f"  balanced size={len(mixed)} "
        f"(normal={len(normal)}, abnormal={len(abn_out)}, abnormal_ratio={ratio:.1%})"
    )
    return mixed, len(abn_out), len(normal)


def load_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(data)} examples -> {path}")


def copy_split(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"Skip missing split: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Copied {src} -> {dst}")


def balance_train(
    examples: list[dict],
    method: str,
    target_abnormal_ratio: float,
    max_normal_template: int,
    batch_size: int,
    seed: int,
    abnormal_weight: float = 1.6,
    normal_weight: float = 1.0,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    impressions = [extract_impression(ex) for ex in examples]
    flags = label_impressions(impressions, method=method, batch_size=batch_size)

    abnormal = [ex for ex, flag in zip(examples, flags) if flag]
    normal = [ex for ex, flag in zip(examples, flags) if not flag]
    normal_imps = [imp for imp, flag in zip(impressions, flags) if not flag]

    print(f"Labeled with method={method}:")
    print(f"  raw: n={len(examples)} abnormal={len(abnormal)} ({len(abnormal)/len(examples):.1%}) "
          f"normal={len(normal)} ({len(normal)/len(examples):.1%})")

    # Top abnormal/normal impressions for sanity
    abn_counter = Counter(imp for imp, flag in zip(impressions, flags) if flag)
    nor_counter = Counter(imp for imp, flag in zip(impressions, flags) if not flag)
    print("  top normal impressions:")
    for text, count in nor_counter.most_common(5):
        print(f"    [{count}] {text[:100]}")
    print("  top abnormal impressions:")
    for text, count in abn_counter.most_common(5):
        print(f"    [{count}] {text[:100]}")

    normal_capped = cap_normal_templates(normal, normal_imps, max_normal_template, rng)
    balanced, n_abn_out, n_norm_out = upsample_to_ratio(
        normal_capped, abnormal, target_abnormal_ratio, rng
    )

    abn_ids = {id(ex) for ex in abnormal}
    weighted: list[dict] = []
    for ex in balanced:
        item = dict(ex)
        item["sample_weight"] = abnormal_weight if id(ex) in abn_ids else normal_weight
        weighted.append(item)

    stats = {
        "input_n": len(examples),
        "raw_abnormal": len(abnormal),
        "raw_normal": len(normal),
        "normal_after_cap": len(normal_capped),
        "output_n": len(weighted),
        "output_abnormal": n_abn_out,
        "output_normal": n_norm_out,
        "output_abnormal_ratio": round(n_abn_out / max(len(weighted), 1), 4),
        "abnormal_weight": abnormal_weight,
        "normal_weight": normal_weight,
        "method": method,
        "target_abnormal_ratio": target_abnormal_ratio,
        "max_normal_template": max_normal_template,
        "seed": seed,
    }
    return weighted, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "sft")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "sft_balanced")
    parser.add_argument(
        "--method",
        choices=["chexbert", "keyword", "hybrid"],
        default="hybrid",
        help="How to decide abnormal impression (default: hybrid=CheXbert OR keyword).",
    )
    parser.add_argument("--target-abnormal-ratio", type=float, default=0.45)
    parser.add_argument(
        "--max-normal-template",
        type=int,
        default=40,
        help="Max samples kept per identical normal impression text.",
    )
    parser.add_argument("--abnormal-weight", type=float, default=1.6, help="sample_weight for abnormal impressions.")
    parser.add_argument("--normal-weight", type=float, default=1.0, help="sample_weight for normal impressions.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-val-test",
        action="store_true",
        default=True,
        help="Copy val.json/test.json unchanged (default: True).",
    )
    parser.add_argument("--no-copy-val-test", action="store_false", dest="copy_val_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.input_dir / "train.json"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")

    print(f"Loading {train_path}")
    train = load_json(train_path)
    balanced, stats = balance_train(
        examples=train,
        method=args.method,
        target_abnormal_ratio=args.target_abnormal_ratio,
        max_normal_template=args.max_normal_template,
        batch_size=args.batch_size,
        seed=args.seed,
        abnormal_weight=args.abnormal_weight,
        normal_weight=args.normal_weight,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "train.json", balanced)
    (args.output_dir / "balance_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote stats -> {args.output_dir / 'balance_stats.json'}")

    if args.copy_val_test:
        for name in ("val.json", "test.json"):
            src = args.input_dir / name
            if not src.exists():
                print(f"Skip missing split: {src}")
                continue
            data = load_json(src)
            for ex in data:
                ex.setdefault("sample_weight", 1.0)
            save_json(args.output_dir / name, data)

    print("\nDone. Register in dataset_info.json, e.g.:")
    print('  "iu_xray_report_balanced": { "file_name": "data/sft_balanced/train.json", ... }')
    print("Then SFT with --dataset iu_xray_report_balanced")


if __name__ == "__main__":
    main()
