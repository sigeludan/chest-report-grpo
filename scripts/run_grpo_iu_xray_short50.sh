#!/bin/bash
# Short GRPO run (50 steps) with full group-B config to verify GPU memory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

bash "${SCRIPT_DIR}/run_grpo_iu_xray.sh" \
    trainer.experiment_name=iu_xray_short50 \
    trainer.max_steps=50 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.save_freq=50 \
    trainer.save_checkpoint_path=/root/autodl-tmp/chest-report-grpo/checkpoints/grpo/iu_xray_short50
