#!/bin/bash
# One-step smoke test for IU-Xray GRPO with clinical reward v2.
# Prerequisites: SFT-merged model at models/Qwen2.5-VL-7B-Instruct-sft-merged
#   (or update configs/easyr1_grpo.yaml model_path).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PROJECT_ROOT
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CHEXBERT_CHECKPOINT="${CHEXBERT_CHECKPOINT:-${PROJECT_ROOT}/models/chexbert/chexbert.pth}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Clear leftover Ray / vLLM processes if needed:
#   pkill -f verl.trainer.main || true; ray stop --force || true; sleep 2

bash "${SCRIPT_DIR}/run_grpo_iu_xray.sh" \
    data.mini_rollout_batch_size=2 \
    data.rollout_batch_size=4 \
    data.max_prompt_length=4096 \
    data.max_response_length=256 \
    data.max_pixels=1048576 \
    data.filter_overlong_prompts_workers=2 \
    worker.actor.global_batch_size=4 \
    worker.rollout.n=2 \
    worker.rollout.gpu_memory_utilization=0.32 \
    worker.ref.offload.offload_params=true \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    trainer.experiment_name=iu_xray_smoke_reward_v2 \
    trainer.total_epochs=1 \
    trainer.max_steps=1 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.save_freq=-1 \
    trainer.save_checkpoint_path="${PROJECT_ROOT}/checkpoints/grpo/iu_xray_smoke_reward_v2" \
    "$@"
