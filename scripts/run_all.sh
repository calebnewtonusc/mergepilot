#!/usr/bin/env bash
# MergePilot — Full pipeline: collect → synthesize → train → evaluate
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

[ -f .env ] && set -a && source .env && set +a || true

# ─── Config ────────────────────────────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
STAGE1_OUT="${STAGE1_OUT:-./checkpoints/stage1-sft}"
STAGE2_OUT="${STAGE2_OUT:-./checkpoints/stage2-grpo}"
STAGE3_OUT="${STAGE3_OUT:-./checkpoints/stage3-dpo}"
FINAL_OUT="${FINAL_OUT:-./checkpoints/mergepilot-v1}"

DATA_RAW="${DATA_RAW:-./data/raw}"
DATA_SYN="${DATA_SYN:-./data/synthesized}"

VLLM_URLS="${VLLM_URLS:-http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004}"
NUM_GPUS="${NUM_GPUS:-18}"
TRAIN_GPUS="${TRAIN_GPUS:-8}"
SYNTHESIS_GPUS="${SYNTHESIS_GPUS:-4}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() { log "━━━ STEP: $* ━━━"; }

# ─── Step 1: Collect raw data ───────────────────────────────────────────────
step "Collecting GitHub PRs"
python -m discovery.github_pr_crawler \
	--output-dir "$DATA_RAW/prs" \
	--max-repos 500 \
	--min-stars 1000 \
	--languages Python TypeScript Go Rust Java \
	--workers 16

step "Crawling engineering blogs"
python -m discovery.engineering_blog_crawler

# ─── Step 2: Start vLLM for synthesis ──────────────────────────────────────
step "Starting vLLM synthesis servers"
bash scripts/start_vllm.sh &
VLLM_PID=$!
sleep 30 # Wait for servers to initialize

# ─── Step 3: Synthesize training data ──────────────────────────────────────
step "Synthesizing code reviews"
IFS=',' read -ra VLLM_ARRAY <<<"$VLLM_URLS"
python -m synthesis.synthesize_bulk \
	--backend vllm \
	--vllm-urls "${VLLM_ARRAY[@]}" \
	--workers 40

step "Generating PR improvement pairs"
IFS=',' read -ra VLLM_ARRAY <<<"$VLLM_URLS"
python -m synthesis.pr_generator \
	--backend vllm \
	--vllm-urls "${VLLM_ARRAY[@]}" \
	--workers 40

step "Generating DPO preference pairs"
python -m pipeline dpo-pairs

# Kill synthesis vLLM servers
kill "$VLLM_PID" 2>/dev/null || true

# ─── Step 4: Prepare training data ─────────────────────────────────────────
step "Preparing training data"
python -m pipeline prep

# ─── Step 5: Stage 1 — SFT ─────────────────────────────────────────────────
step "Stage 1: SFT Training"
deepspeed \
	--num_gpus "$TRAIN_GPUS" \
	--master_port 29500 \
	training/train.py \
	--base-model "$BASE_MODEL" \
	--data-path "$DATA_SYN" \
	--output-dir "$STAGE1_OUT"

log "Stage 1 complete: $STAGE1_OUT"

# ─── Step 6: Restart vLLM with Stage 1 model ───────────────────────────────
step "Starting vLLM with Stage 1 model for RL"
VLLM_MODEL="$STAGE1_OUT/merged" bash scripts/start_vllm.sh &
VLLM_RL_PID=$!
sleep 30

# ─── Step 7: Stage 2 — GRPO RL ─────────────────────────────────────────────
step "Stage 2: GRPO RL Training"
deepspeed \
	--num_gpus "$TRAIN_GPUS" \
	--master_port 29501 \
	training/train_rl.py \
	--base-model "$STAGE1_OUT/merged" \
	--data-path "$DATA_SYN" \
	--output-dir "$STAGE2_OUT"

kill "$VLLM_RL_PID" 2>/dev/null || true
log "Stage 2 complete: $STAGE2_OUT"

# ─── Step 8: Stage 3 — DPO ─────────────────────────────────────────────────
step "Stage 3: DPO Training"
deepspeed \
	--num_gpus "$TRAIN_GPUS" \
	--master_port 29502 \
	training/train_dpo.py \
	--base-model "$STAGE2_OUT/merged" \
	--data-dir "$DATA_SYN/dpo" \
	--output-dir "$STAGE3_OUT"

log "Stage 3 complete: $STAGE3_OUT"

# ─── Step 9: Copy final model ───────────────────────────────────────────────
step "Copying final model"
cp -r "$STAGE3_OUT/merged" "$FINAL_OUT"

# ─── Step 10: Evaluate ─────────────────────────────────────────────────────
step "Running MergeBench evaluation"
python -m pipeline eval \
	--model-path "$FINAL_OUT" \
	--output eval_results.json

log "Pipeline complete! Results: eval_results.json"
