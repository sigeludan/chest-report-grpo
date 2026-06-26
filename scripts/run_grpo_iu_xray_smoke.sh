#!/bin/bash
# One-step smoke test for IU-Xray GRPO config (after flash-attn + geo3k are OK).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
    trainer.experiment_name=iu_xray_smoke \
    trainer.total_epochs=1 \
    trainer.max_steps=1 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.save_freq=-1 \
    trainer.save_checkpoint_path=/root/autodl-tmp/chest-report-grpo/checkpoints/grpo/iu_xray_smoke
