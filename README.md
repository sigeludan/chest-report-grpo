# Chest Report GRPO

在 **IU-Xray** 上基于 **Qwen2.5-VL-7B-Instruct** 的胸片报告生成：SFT 冷启动 + GRPO 临床对齐。

## 方法

1. **SFT**（LLaMA-Factory LoRA，r=8）学习 `<findings>` / `<impression>` 格式与报告书写。
2. **GRPO**（EasyR1）在 merge 后的 SFT 模型上继续优化，奖励为：

```
R = R_format + λ₁·R_clinical + λ₂·R_consistency
```

- **R_format**：结构化标签解析（门控）
- **R_clinical**：CheXbert impression micro-F1（相对 GT）
- **R_consistency**：findings 与 impression 阳性标签一致性

组 B 配置：**λ₁ = 1.0，λ₂ = 0.3**

## 结果（test，639 条）


| 阶段        | 格式率   | CheXbert micro-F1 |
| --------- | ----- | ----------------- |
| Baseline  | 87.2% | 0.016             |
| SFT（组 A）  | 100%  | 0.693             |
| GRPO（组 B） | 100%  | **0.722**         |


## 环境

- Python 3.12，PyTorch 2.11，vLLM 0.23，flash-attn 2.8.3
- 单卡 A800 80GB
- **LLaMA-Factory**：单独 clone 安装（本仓库不包含其源码）
- **EasyR1**：`pip install -e EasyR1/`

```bash
export HF_ENDPOINT=https://hf-mirror.com
export CHEXBERT_CHECKPOINT=/root/autodl-tmp/chest-report-grpo/models/chexbert/chexbert.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 快速开始

### 0. 准备权重与数据

```bash
export PROJECT_ROOT=/root/autodl-tmp/chest-report-grpo
cd "$PROJECT_ROOT"

# 基座模型（HuggingFace）
# huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
#   --local-dir models/Qwen2.5-VL-7B-Instruct

# IU-Xray：将 archive.zip 放到 data/ 后预处理
python3 scripts/preprocess_iu_xray.py              # 首次：解压 + 生成 json
python3 scripts/preprocess_iu_xray.py --skip-extract   # 已有 raw 时跳过解压
```

产出：`data/sft/train.json`、`data/sft/val.json`、`data/grpo/*.jsonl`；划分 train 2230 / val 318 / test 639。

### 1. SFT（LLaMA-Factory LoRA）

```bash
# 安装 LLaMA-Factory（与本项目同级目录，路径按实际调整）
git clone https://github.com/hiyouga/LLaMA-Factory.git /root/autodl-tmp/LLaMA-Factory
cd /root/autodl-tmp/LLaMA-Factory && pip install -e .

# 注册数据集：把本项目 dataset_info.json 里的 iu_xray_* 两条
# 合并进 LLaMA-Factory/data/dataset_info.json
# （本项目根目录也有一份 dataset_info.json，字段与之一致）

cd /root/autodl-tmp/LLaMA-Factory

llamafactory-cli train \
  --stage sft \
  --model_name_or_path "$PROJECT_ROOT/models/Qwen2.5-VL-7B-Instruct" \
  --dataset_dir "$PROJECT_ROOT" \
  --dataset iu_xray_report \
  --eval_dataset iu_xray_report_val \
  --template qwen2_vl \
  --finetuning_type lora \
  --output_dir "$PROJECT_ROOT/checkpoints/qwen25vl-iu-sft-lora" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --max_length 2048 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --freeze_vision_tower true \
  --bf16 true \
  --eval_strategy steps \
  --eval_steps 200 \
  --save_steps 200 \
  --logging_steps 10 \
  --dataloader_num_workers 4
```

实际训练约 **3 epoch / 837 steps**；评测选用 **checkpoint-800**（eval_loss≈0.733）。

组 A 评测：

```bash
cd "$PROJECT_ROOT"
python3 scripts/eval_report.py --split test \
  --adapter checkpoints/qwen25vl-iu-sft-lora/checkpoint-800 \
  --output outputs/sft_test_metrics.json
```

### 2. Merge SFT LoRA（GRPO 冷启动前）

EasyR1 会在基座上**新建** LoRA，不会自动加载 SFT adapter，需先 merge：

```bash
cd "$PROJECT_ROOT"
python3 scripts/merge_lora.py \
  --adapter checkpoints/qwen25vl-iu-sft-lora/checkpoint-800 \
  --export-dir models/Qwen2.5-VL-7B-Instruct-sft-merged
```

### 3. GRPO（组 B）

```bash
bash scripts/run_grpo_iu_xray.sh
```

### 4. test 评测（组 B）

```bash
python3 scripts/eval_report.py --split test \
  --base-model models/Qwen2.5-VL-7B-Instruct-sft-merged \
  --adapter checkpoints/grpo/iu_xray_group_b/global_step_70/actor/lora_adapter \
  --output outputs/grpo_test_metrics.json
```

## 目录


| 路径 | 说明 |
|------|------|
| `dataset_info.json` | LLaMA-Factory 数据集注册（`iu_xray_report` / `iu_xray_report_val`） |
| `data/sft/` | SFT 训练/验证 json |
| `configs/easyr1_grpo.yaml` | GRPO 主配置 |
| `rewards/` | format / clinical / consistency 奖励 |
| `scripts/` | 预处理、merge、训练、评测 |
| `checkpoints/` | SFT / GRPO checkpoint |
| `output/` | 结果与日志 |
| `实验记录.md` | 完整实验记录 |


