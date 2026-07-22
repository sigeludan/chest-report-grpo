#!/usr/bin/env python3
"""Resume-friendly IU-Xray image downloader from HF mirror."""
from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMG_DIR = RAW / "images" / "images_normalized"


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    filenames: list[str] = []
    with open(RAW / "indiana_projections.csv", newline="") as f:
        for row in csv.DictReader(f):
            fn = (row.get("filename") or "").strip()
            if fn:
                filenames.append(fn)
    filenames = sorted(set(filenames))
    pending = [fn for fn in filenames if not (IMG_DIR / fn).exists() or (IMG_DIR / fn).stat().st_size < 1000]
    print(f"total={len(filenames)} pending={len(pending)}", flush=True)
    ok = fail = 0

    def download_one(fn: str) -> str:
        hf_hub_download(
            repo_id="sasi2004/chest-xrays-indiana-university",
            repo_type="dataset",
            filename=f"images/images_normalized/{fn}",
            local_dir=str(RAW),
        )
        return fn

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(download_one, fn): fn for fn in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            fn = futs[fut]
            try:
                fut.result()
                ok += 1
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"FAIL {fn}: {e}", flush=True)
            if i % 50 == 0 or i == len(futs):
                print(f"progress {i}/{len(futs)} ok={ok} fail={fail}", flush=True)
    print(f"DONE ok={ok} fail={fail} png={len(list(IMG_DIR.glob('*.png')))}", flush=True)


if __name__ == "__main__":
    main()
