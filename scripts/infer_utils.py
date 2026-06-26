"""Shared helpers for chest report generation inference and evaluation."""

from __future__ import annotations

import json
import os
import re
import site
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _prepend_nvidia_lib_paths() -> None:
    """Make nvrtc/CUDA libs from pip nvidia-* packages visible to the dynamic linker."""
    if os.environ.get("CHEST_REPORT_CUDA_LIBS_READY"):
        return
    extra: list[str] = []
    for site_packages in site.getsitepackages():
        for subpath in ("nvidia/cu13/lib", "nvidia/cuda_nvrtc/lib", "nvidia/cu12/lib"):
            candidate = os.path.join(site_packages, subpath)
            if os.path.isdir(candidate):
                extra.append(candidate)
    if extra:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(extra + ([current] if current else []))
    os.environ["CHEST_REPORT_CUDA_LIBS_READY"] = "1"


_prepend_nvidia_lib_paths()

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_ADAPTER = PROJECT_ROOT / "checkpoints" / "qwen25vl-iu-sft-lora" / "checkpoint-600"
DEFAULT_GRPO_VAL = PROJECT_ROOT / "data" / "grpo" / "val.jsonl"
DEFAULT_GRPO_TEST = PROJECT_ROOT / "data" / "grpo" / "test.jsonl"
DEFAULT_SFT_VAL = PROJECT_ROOT / "data" / "sft" / "val.json"

IMAGE_TOKEN_PATTERN = re.compile(r"<image>", re.IGNORECASE)


@dataclass
class EvalSample:
    sample_id: str
    problem: str
    images: list[str]
    impression_gt: str
    findings_gt: str


def add_project_to_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_image_paths(image_paths: list[str], project_root: Path = PROJECT_ROOT) -> list[str]:
    resolved: list[str] = []
    for path in image_paths:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Image not found: {candidate}")
        resolved.append(str(candidate.resolve()))
    return resolved


def strip_image_placeholders(problem: str) -> str:
    text = IMAGE_TOKEN_PATTERN.sub("", problem)
    return text.strip()


def load_grpo_jsonl(path: Path, limit: int | None = None) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            answer = json.loads(row["answer"])
            samples.append(
                EvalSample(
                    sample_id=str(row.get("id", len(samples))),
                    problem=row["problem"],
                    images=list(row["images"]),
                    impression_gt=str(answer.get("impression_gt", "")).strip(),
                    findings_gt=str(answer.get("findings_gt", "")).strip(),
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def load_sft_json(path: Path, limit: int | None = None) -> list[EvalSample]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    samples: list[EvalSample] = []
    for idx, row in enumerate(rows):
        user_msg = next(item for item in row["messages"] if item["role"] == "user")
        assistant_msg = next(item for item in row["messages"] if item["role"] == "assistant")
        parsed = parse_assistant_report(assistant_msg["content"])
        samples.append(
            EvalSample(
                sample_id=f"sft_{idx}",
                problem=user_msg["content"],
                images=list(row["images"]),
                impression_gt=parsed[1] if parsed else "",
                findings_gt=parsed[0] if parsed else "",
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def parse_assistant_report(text: str) -> tuple[str, str] | None:
    from rewards.format_reward import parse_report

    parsed = parse_report(text)
    if parsed is None:
        return None
    return parsed[0], parsed[1]


def load_eval_samples(
    split: str = "val",
    data_path: Path | None = None,
    limit: int | None = None,
) -> list[EvalSample]:
    if data_path is not None:
        path = data_path
    elif split == "val":
        path = DEFAULT_GRPO_VAL
    elif split == "test":
        path = DEFAULT_GRPO_TEST
    else:
        raise ValueError(f"Unknown split: {split}")

    if path.suffix == ".json":
        return load_sft_json(path, limit=limit)
    return load_grpo_jsonl(path, limit=limit)


def build_messages(problem: str, image_paths: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_path in image_paths:
        content.append({"type": "image", "image": f"file://{image_path}"})
    content.append({"type": "text", "text": strip_image_placeholders(problem)})
    return [{"role": "user", "content": content}]


class ReportGenerator:
    def __init__(
        self,
        base_model: Path | str = DEFAULT_BASE_MODEL,
        adapter_path: Path | str | None = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.base_model = str(base_model)
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.processor = AutoProcessor.from_pretrained(self.base_model)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        if self.adapter_path:
            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self.model = model

    @torch.inference_mode()
    def generate(
        self,
        problem: str,
        image_paths: list[str],
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.1,
    ) -> str:
        resolved = resolve_image_paths(image_paths)
        messages = build_messages(problem, resolved)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generate_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if do_sample:
            generate_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            generate_kwargs["do_sample"] = False

        output_ids = self.model.generate(**inputs, **generate_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def run_inference(
    generator: ReportGenerator,
    samples: list[EvalSample],
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        prediction = generator.generate(sample.problem, sample.images)
        record = {
            "id": sample.sample_id,
            "problem": sample.problem,
            "images": sample.images,
            "prediction": prediction,
            "impression_gt": sample.impression_gt,
            "findings_gt": sample.findings_gt,
        }
        records.append(record)
        if idx % 10 == 0 or idx == len(samples):
            print(f"[infer] {idx}/{len(samples)} done", flush=True)

    if output_path is not None:
        save_predictions(records, output_path)
    return records


def save_predictions(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved predictions to {output_path}")


def load_predictions(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records
