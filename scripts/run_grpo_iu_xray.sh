#!/bin/bash
# IU-Xray GRPO launcher (组 B). Run from project root or any directory.
#
# Prerequisites:
#   - pip install -e EasyR1/
#   - flash-attn importable on torch 2.11
#   - models/chexbert/chexbert.pth present
#
# SFT cold start:
#   EasyR1 LoRA GRPO initializes a new LoRA adapter on the base model. To continue
#   from SFT LoRA (rank=8), merge the adapter first or resume from an EasyR1
#   checkpoint under checkpoints/grpo/iu_xray_group_b/.

set -euo pipefail
set -x

PROJECT_ROOT="/root/autodl-tmp/chest-report-grpo"
EASYR1_DIR="${PROJECT_ROOT}/EasyR1"
CONFIG="${PROJECT_ROOT}/configs/easyr1_grpo.yaml"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
CHEXBERT_CHECKPOINT="${CHEXBERT_CHECKPOINT:-${PROJECT_ROOT}/models/chexbert/chexbert.pth}"
if [[ "${CHEXBERT_CHECKPOINT}" != /* ]]; then
  CHEXBERT_CHECKPOINT="${PROJECT_ROOT}/${CHEXBERT_CHECKPOINT}"
fi
export CHEXBERT_CHECKPOINT
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# nvrtc / CUDA libs for torch 2.11 (same as infer_utils fix)
for _site in "$(python3 -c 'import site; print(site.getsitepackages()[0])')"; do
  for _sub in nvidia/cu13/lib nvidia/cuda_nvrtc/lib; do
    _p="${_site}/${_sub}"
    if [[ -d "${_p}" ]]; then
      export LD_LIBRARY_PATH="${_p}:${LD_LIBRARY_PATH:-}"
    fi
  done
done

cd "${EASYR1_DIR}"

# λ₁=1.0, λ₂=0.3 + clinical v2 kwargs — explicit CLI override so kwargs cannot silently fall back.
python3 -m verl.trainer.main \
    config="${CONFIG}" \
    worker.reward.reward_function_kwargs.lambda_clinical=1.0 \
    worker.reward.reward_function_kwargs.lambda_consistency=0.3 \
    worker.reward.reward_function_kwargs.r_normal=1.0 \
    worker.reward.reward_function_kwargs.gamma_fp=0.25 \
    worker.reward.reward_function_kwargs.gamma_fp_severe=0.4 \
    worker.reward.reward_function_kwargs.fbeta=2.0 \
    worker.reward.reward_function_kwargs.lambda_abnormal=1.5 \
    "$@"
