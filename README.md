# Chest Report GRPO

在 **IU-Xray** 上基于 **Qwen2.5-VL-7B-Instruct** 的胸片报告生成：SFT 冷启动 + GRPO 临床对齐。

## 方法

1. **SFT**（LLaMA-Factory LoRA，r=8）：在分层平衡数据上学习 `<findings>` / `<impression>` 格式与报告书写。相对原分布训练，本轮改进为：
  - train **hybrid 分层**：正常模板限频 + 异常上采样至约 45%（`data/sft_balanced/`）
  - 异常样本 `sample_weight=1.6` 加权 CE（需对 LLaMA-Factory 打 `patches/llama-factory-sample-weight.patch`）
2. **GRPO**（EasyR1）：在 merge 后的 SFT 模型上继续优化；**数据仍用原分布** `data/grpo/`（不做上采样）。奖励外壳不变：

```
R = R_format + λ₁·R_clinical + λ₂·R_consistency
```

- **R_format**：结构化标签解析（门控）
- **R_clinical（v2）**：13 疾病类（排除 No Finding）。GT 正常 → 无假阳性满分、有假阳性按 γ 扣分；GT 有病 → **λ_abn × F2**（偏召回）
- **R_consistency**：findings 与 impression 阳性标签一致性

组 B 默认：**λ₁ = 1.0，λ₂ = 0.3**；clinical v2：`r_normal=1.0`，`γ=0.25`（严重假阳性 0.4），`β=2`，`λ_abn=1.5`。

评测脚本 `eval_report.py` 仍报 **legacy CheXbert micro-F1**，便于与历史数字对比。

## 结果（test，639 条）


| 阶段                    | 格式率   | CheXbert micro-F1 | 备注                                    |
| --------------------- | ----- | ----------------- | ------------------------------------- |
| Baseline              | 87.2% | 0.016             | 基座零样本                                 |
| SFT（组 A，balanced）     | 100%  | **73.853%**       | 正常无误报；异常约 70% 仍报正常                    |
| GRPO（组 B，clinical v2） | 100%  | **75.634%**       | 异常 **32%（149/150）** 报出病变，**32%** 仍报正常 |


相对组 A：F1 **+4.771 pp**；异常侧「报出病变」约 30% → **68%**。

> 旧版（未分层 SFT + 旧 micro-F1 奖励）约为 0.693 → 0.722，但预测几乎塌成「无病」模板；本表为当前流水线事实结果。

## 环境

- Python 3.12，PyTorch 2.11，vLLM 0.23，flash-attn 2.8.3
- 单卡 A800 80GB
- **LLaMA-Factory** / **EasyR1**：单独安装；本地改动见 `patches/`

```bash
export HF_ENDPOINT=https://hf-mirror.com
export CHEXBERT_CHECKPOINT=/root/autodl-tmp/chest-report-grpo/models/chexbert/chexbert.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
```

## 快速开始

### 0. 准备权重与数据

```bash
export PROJECT_ROOT=/root/autodl-tmp/chest-report-grpo
cd "$PROJECT_ROOT"

# 基座（HuggingFace）
# huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
#   --local-dir models/Qwen2.5-VL-7B-Instruct

# IU-Xray：archive.zip → data/ 后预处理（写出 data/sft + data/grpo）
python3 scripts/preprocess_iu_xray.py
# 已有 raw：python3 scripts/preprocess_iu_xray.py --skip-extract

# GRPO 原分布校验 / 统计
python3 scripts/prepare_grpo_data.py

# SFT 分层平衡（hybrid，异常占比≈45%，sample_weight）
export CHEXBERT_CHECKPOINT=$PROJECT_ROOT/models/chexbert/chexbert.pth
python3 scripts/balance_sft_data.py --method hybrid \
  --target-abnormal-ratio 0.45 --max-normal-template 40 \
  --abnormal-weight 1.6 --normal-weight 1.0 --seed 42
```

划分规模：有效样本 3187 → train/val/test **2230 / 318 / 639**（seed=42）。  
SFT 训练用 `data/sft_balanced/`（train≈2293）；GRPO 用 `data/grpo/`（原分布 2230）。

### 1. 安装并打补丁

```bash
# LLaMA-Factory + sample_weight 补丁
git clone https://github.com/hiyouga/LLaMA-Factory.git /root/autodl-tmp/LLaMA-Factory
cd /root/autodl-tmp/LLaMA-Factory && pip install -e .
git apply "$PROJECT_ROOT/patches/llama-factory-sample-weight.patch"

# EasyR1 + 本地补丁（源码在本仓 EasyR1/，默认不入库）
cd "$PROJECT_ROOT"
pip install -e EasyR1/
git -C EasyR1 apply "$PROJECT_ROOT/patches/easyr1-local.patch"
```

数据集注册：项目根 `dataset_info.json`（`iu_xray_report_balanced` / `_val`），训练时 `--dataset_dir $PROJECT_ROOT`。

### 2. SFT（组 A）

```bash
bash scripts/run_sft_balanced.sh
# 约 2 epoch；选用 checkpoint-574
```

评测：

```bash
python3 scripts/eval_report.py --split test \
  --adapter checkpoints/qwen25vl-iu-sft-lora-balanced/checkpoint-574 \
  --pred-output outputs/sft_balanced_test_preds.jsonl \
  --output outputs/sft_balanced_test_metrics.json
```

### 3. Merge SFT LoRA（GRPO 冷启动前）

```bash
python3 scripts/merge_lora.py \
  --adapter checkpoints/qwen25vl-iu-sft-lora-balanced/checkpoint-574 \
  --export-dir models/Qwen2.5-VL-7B-Instruct-sft-merged
```

### 4. GRPO（组 B）

```bash
# smoke（1 step）
bash scripts/run_grpo_iu_xray_smoke.sh

# 正式（1 epoch ≈ 139 steps）
bash scripts/run_grpo_iu_xray.sh
```

### 5. test 评测（组 B）

```bash
python3 scripts/eval_report.py --split test \
  --base-model models/Qwen2.5-VL-7B-Instruct-sft-merged \
  --adapter checkpoints/grpo/iu_xray_group_b/<step>/actor/lora_adapter \
  --pred-output outputs/grpo_reward_v2_test_preds.jsonl \
  --output outputs/grpo_reward_v2_test_metrics.json
```

## 目录


| 路径                         | 说明                                             |
| -------------------------- | ---------------------------------------------- |
| `dataset_info.json`        | LLaMA-Factory 注册（含 balanced + `sample_weight`） |
| `configs/easyr1_grpo.yaml` | GRPO 主配置（clinical v2 kwargs）                   |
| `data/sft/`                | 原始划分 SFT json（2230/318/639）                    |
| `data/sft_balanced/`       | 分层平衡 SFT（本轮训练用）                                |
| `data/grpo/`               | GRPO 原分布 jsonl + `data_stats.json`             |
| `rewards/`                 | format / clinical v2 / consistency             |
| `scripts/`                 | 预处理、平衡、prepare_grpo、SFT/GRPO 启动、评测             |
| `patches/`                 | LLaMA-Factory / EasyR1 本地补丁                    |
| `checkpoints/`             | SFT / GRPO checkpoint（本地，不入库）                  |
| `models/`                  | 基座 / CheXbert / merge 模型（本地，不入库）               |
| `outputs/`                 | 评测与日志（本地，不入库）                                  |
| `实验记录.md`                  | 完整实验记录                                         |
| `项目方案.md`                  | 方案说明                                           |


