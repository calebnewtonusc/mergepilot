#!/usr/bin/env bash
# Check MergePilot environment setup
set -euo pipefail

PASS=0
FAIL=0

check() {
	local name="$1"
	local result="$2"
	if [ "$result" = "ok" ]; then
		echo "  [PASS] $name"
		PASS=$((PASS + 1))
	else
		echo "  [FAIL] $name: $result"
		FAIL=$((FAIL + 1))
	fi
}

echo "━━━ MergePilot Environment Check ━━━"
echo ""

# Python
PY_VER=$(python --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 11 ]; then
	check "Python >= 3.11" "ok"
else
	check "Python >= 3.11" "found $PY_VER"
fi

# GPU
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPU_COUNT" -ge 8 ]; then
	check "GPUs >= 8" "ok ($GPU_COUNT found)"
else
	check "GPUs >= 8 (for training)" "found $GPU_COUNT (ok for dev)"
fi

# Flash Attention
FA_OK=$(python -c "import flash_attn; print('ok')" 2>/dev/null || echo "missing")
check "flash-attn" "$FA_OK"

# DeepSpeed
DS_OK=$(python -c "import deepspeed; print('ok')" 2>/dev/null || echo "missing")
check "deepspeed" "$DS_OK"

# vLLM
VLLM_OK=$(python -c "import vllm; print('ok')" 2>/dev/null || echo "missing")
check "vllm" "$VLLM_OK"

# TRL
TRL_OK=$(python -c "import trl; print('ok')" 2>/dev/null || echo "missing")
check "trl" "$TRL_OK"

# PEFT
PEFT_OK=$(python -c "import peft; print('ok')" 2>/dev/null || echo "missing")
check "peft" "$PEFT_OK"

# Anthropic
ANT_OK=$(python -c "import anthropic; print('ok')" 2>/dev/null || echo "missing")
check "anthropic" "$ANT_OK"

# aiohttp
AIOHTTP_OK=$(python -c "import aiohttp; print('ok')" 2>/dev/null || echo "missing")
check "aiohttp" "$AIOHTTP_OK"

# Environment variables
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
	check "ANTHROPIC_API_KEY" "ok"
else
	check "ANTHROPIC_API_KEY" "not set"
fi

if [ -n "${GITHUB_TOKEN:-}" ]; then
	check "GITHUB_TOKEN" "ok"
else
	check "GITHUB_TOKEN" "not set (needed for crawling)"
fi

# CUDA compute
CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
check "CUDA version" "ok ($CUDA_VER)"

# BF16 support
BF16_OK=$(python -c "
import torch
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    print('ok')
else:
    print('no (need Ampere+ GPU for bfloat16)')
" 2>/dev/null || echo "no GPU")
check "BF16 support (Ampere+)" "$BF16_OK"

echo ""
echo "━━━ Results ━━━"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"

if [ "$FAIL" -eq 0 ]; then
	echo ""
	echo "All checks passed! Ready to run."
else
	echo ""
	echo "Some checks failed. Run: pip install -r requirements.txt"
	exit 1
fi
