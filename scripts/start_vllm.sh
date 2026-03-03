#!/usr/bin/env bash
# Start vLLM inference servers for MergePilot synthesis
# Runs 4 instances on GPUs 0-7 (2 GPUs each)
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
BASE_PORT="${BASE_PORT:-8001}"
INSTANCES="${INSTANCES:-4}"

echo "Starting $INSTANCES vLLM instances for $MODEL..."

for i in $(seq 0 $((INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    GPU_START=$((i * 2))
    GPU_END=$((GPU_START + 1))
    CUDA_DEVICES="$GPU_START,$GPU_END"

    echo "  Instance $((i+1)): port=$PORT, GPUs=$CUDA_DEVICES"

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES vllm serve "$MODEL" \
        --port "$PORT" \
        --host "0.0.0.0" \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 4096 \
        --enable-prefix-caching \
        --dtype bfloat16 \
        --api-key "${VLLM_API_KEY:-}" \
        > "logs/vllm_$PORT.log" 2>&1 &

    echo "  PID: $!"
done

echo ""
echo "vLLM instances started on ports $BASE_PORT-$((BASE_PORT + INSTANCES - 1))"
echo "Wait ~60 seconds for models to load."
echo "Test: curl http://localhost:$BASE_PORT/health"
