# MergePilot — GPU Setup Guide

## Hardware Target

| Phase | Config | VRAM | Time |
|-------|--------|------|------|
| Data collection | CPU + GitHub API | — | ~8h |
| Synthesis (vLLM) | 8× A6000 48GB | 384GB | ~12h |
| Stage 1 SFT | 18× A6000 48GB | 864GB | ~6h |
| Stage 2 GRPO RL | 18× A6000 48GB | 864GB | ~10h |
| Stage 3 DPO | 18× A6000 48GB | 864GB | ~4h |
| Inference | 1× A100 80GB | 80GB | <300ms p50 |

## Driver and CUDA Setup

```bash
nvidia-smi  # Verify drivers
nvcc --version  # Need CUDA 12.1+

pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

## GPU Allocation

### Synthesis Phase (8 GPUs)
```bash
bash scripts/start_vllm.sh
# 4 vLLM instances on GPUs 0-7, ports 8001-8004
```

### Training Phase (18 GPUs)
```bash
deepspeed --num_gpus=18 training/train.py \
  --deepspeed training/configs/ds_config.json \
  --model Qwen/Qwen2.5-7B-Coder-Instruct \
  --data-dir data/train \
  --output-dir checkpoints/mergepilot-sft
```

## Environment Verification

```bash
bash scripts/check_env.sh
```

Expected:
```
[OK] Python 3.11.x
[OK] torch 2.2.0+cu121
[OK] transformers 4.44.x
[OK] GITHUB_TOKEN set
[OK] GPUs: 18 × NVIDIA A6000
[OK] Disk: 8.2 TB free
```

## Common Issues

**CUDA OOM during training**:
```bash
export MICRO_BATCH_SIZE=1
export GRAD_ACCUM_STEPS=8
export USE_GRADIENT_CHECKPOINTING=1
```

**GitHub API rate limit**:
```bash
# Use multiple tokens for higher rate limits
export GITHUB_TOKEN_1=ghp_...
export GITHUB_TOKEN_2=ghp_...
export GITHUB_TOKEN_3=ghp_...
```
