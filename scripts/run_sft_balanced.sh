#!/usr/bin/env bash
# SFT on balanced IU-Xray data with per-sample loss weights (abnormal=1.6).
# Requires LLaMA-Factory patches for sample_weight (parser/converter/processor/collator/trainer).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/chest-report-grpo}"
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-/root/autodl-tmp/LLaMA-Factory}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/checkpoints"

cd "${LLAMAFACTORY_ROOT}"
llamafactory-cli train \
  --stage sft \
  --do_train true \
  --model_name_or_path "${PROJECT_ROOT}/models/Qwen2.5-VL-7B-Instruct" \
  --dataset_dir "${PROJECT_ROOT}" \
  --dataset iu_xray_report_balanced \
  --eval_dataset iu_xray_report_balanced_val \
  --template qwen2_vl \
  --finetuning_type lora \
  --output_dir "${PROJECT_ROOT}/checkpoints/qwen25vl-iu-sft-lora-balanced" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --num_train_epochs 2 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --freeze_vision_tower true \
  --bf16 true \
  --eval_strategy steps \
  --eval_steps 150 \
  --save_steps 150 \
  --logging_steps 10 \
  --dataloader_num_workers 4 \
  --report_to none \
  --overwrite_output_dir true \
  2>&1 | tee "${PROJECT_ROOT}/outputs/sft_balanced_train.log"
