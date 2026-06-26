#!/usr/bin/env python3
"""Merge SFT LoRA adapter into the Qwen2.5-VL base model for GRPO cold start."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from infer_utils import DEFAULT_BASE_MODEL, PROJECT_ROOT, add_project_to_path

add_project_to_path()

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

DEFAULT_EXPORT_DIR = PROJECT_ROOT / "models" / "Qwen2.5-VL-7B-Instruct-sft-merged"
DEFAULT_ADAPTER_PATH = PROJECT_ROOT / "checkpoints" / "qwen25vl-iu-sft-lora" / "checkpoint-800"

PROCESSOR_FILES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
)


def copy_processor_files(source_dir: Path, export_dir: Path) -> None:
    for name in PROCESSOR_FILES:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, export_dir / name)


def merge_lora(
    base_model: Path,
    adapter_path: Path,
    export_dir: Path,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    if export_dir.exists() and any(export_dir.glob("model*.safetensors")):
        print(f"Merged model already exists at {export_dir}, skipping merge.")
        return

    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading base model from {base_model}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(base_model),
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    print("Merging adapter into base weights...")
    model = model.merge_and_unload()
    print(f"Saving merged model to {export_dir}")
    model.save_pretrained(str(export_dir), safe_serialization=True, max_shard_size="5GB")

    processor_source = adapter_path if (adapter_path / "preprocessor_config.json").exists() else base_model
    processor = AutoProcessor.from_pretrained(str(processor_source))
    processor.save_pretrained(str(export_dir))
    copy_processor_files(processor_source, export_dir)
    copy_processor_files(base_model, export_dir)
    print(f"Done. Merged model saved to {export_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge SFT LoRA into Qwen2.5-VL base model.")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing merged weights.")
    args = parser.parse_args()

    if args.force and args.export_dir.exists():
        for weight_file in args.export_dir.glob("model*.safetensors"):
            weight_file.unlink()
        index_file = args.export_dir / "model.safetensors.index.json"
        if index_file.exists():
            index_file.unlink()

    merge_lora(args.base_model, args.adapter, args.export_dir)


if __name__ == "__main__":
    main()
