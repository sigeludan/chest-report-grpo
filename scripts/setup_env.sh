#!/usr/bin/env bash
# Source this file before training / eval:
#   source /root/autodl-tmp/chest-report-grpo/scripts/setup_env.sh
export PROJECT_ROOT=/root/autodl-tmp/chest-report-grpo
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export CHEXBERT_CHECKPOINT="${CHEXBERT_CHECKPOINT:-$PROJECT_ROOT/models/chexbert/chexbert.pth}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Prefer local bert if present (CheXbert uses bert-base-uncased)
if [[ -d "$PROJECT_ROOT/models/bert-base-uncased" ]]; then
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
fi
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "CHEXBERT_CHECKPOINT=$CHEXBERT_CHECKPOINT"
echo "HF_ENDPOINT=$HF_ENDPOINT"
