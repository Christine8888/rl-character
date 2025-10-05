#!/bin/bash

set -e
set -o pipefail

# ============================================================================
# CONFIGURATION - Set your parameters here
# ============================================================================

# Array of model paths to process
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
)

# Path to JSONL prompt file
PROMPT_PATH="/workspace/rl-character/datasets/chat/tulu_40k_singleturn_additional.jsonl"

# vLLM and sampling configuration
SERVER_URL="http://localhost:9000"
MAX_CONCURRENT=32
BATCH_SIZE=200
TP=1
N_DEVICES=4
KILL_SERVER=true  # Set to false to keep server running between models

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Load shared utilities from finetune_oss
source "/workspace/rl-character/finetune_oss/eval_utils.sh"

# Validate configuration
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: MODELS array is empty. Please configure models in the script."
    exit 1
fi

if [ -z "$PROMPT_PATH" ]; then
    echo "Error: PROMPT_PATH is not set. Please configure it in the script."
    exit 1
fi

if [ ! -f "$PROMPT_PATH" ]; then
    echo "Error: Prompt file does not exist: $PROMPT_PATH"
    exit 1
fi

# Validate tensor parallelism
validate_tp "$TP"

echo "=========================================="
echo "Chat Sampling Configuration"
echo "=========================================="
echo "Models: ${MODELS[@]}"
echo "Prompt file: $PROMPT_PATH"
echo "Server URL: $SERVER_URL"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Batch size: $BATCH_SIZE"
echo "Tensor parallelism: $TP"
echo "Devices: $N_DEVICES"
echo "Kill server between models: $KILL_SERVER"
echo ""

# Cleanup function
cleanup() {
    cleanup_server "$KILL_SERVER" "$SKIP_SERVER_START" "$VLLM_PID"
}

# Set trap for cleanup
trap cleanup EXIT

# Process each model
for MODEL_PATH in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing model: $MODEL_PATH"
    echo "=========================================="

    # Validate model path (works for both local paths and HuggingFace models)
    if ! does_model_exist "$MODEL_PATH"; then
        echo "Skipping $MODEL_PATH"
        continue
    fi

    # Setup model configuration (sets MODEL_FOLDER, MODEL_ALIAS, INSPECT_MODEL_ALIAS)
    # This works for both local paths (starting with /) and HuggingFace models
    setup_model_config "$MODEL_PATH"

    # Check port availability
    check_port_availability

    # Start vLLM server
    cd "/workspace/rl-character/finetune_oss"
    start_vllm_server "$MODEL_FOLDER" "$TP" "$MODEL_ALIAS" "$N_DEVICES" "$SKIP_SERVER_START"
    VLLM_PID=$!

    # Get actual model name from vLLM server
    echo ""
    echo "Querying vLLM server for model name..."
    MODEL_INFO=$(curl -s http://localhost:9000/v1/models 2>/dev/null)
    ACTUAL_MODEL_NAME=$(echo "$MODEL_INFO" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'data' in data and len(data['data']) > 0:
        print(data['data'][0]['id'])
    else:
        print('$MODEL_ALIAS')
except:
    print('$MODEL_ALIAS')
" 2>/dev/null)

    echo "Model name from vLLM: $ACTUAL_MODEL_NAME"
    echo ""

    # Run sample_chat.py
    echo "Running sample_chat.py..."
    cd "/workspace/rl-character/utils"
    python3 sample_chat.py \
        "$PROMPT_PATH" \
        "$ACTUAL_MODEL_NAME" \
        --server-url "$SERVER_URL" \
        --max-concurrent "$MAX_CONCURRENT" \
        --batch-size "$BATCH_SIZE"

    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed sampling for $MODEL_ALIAS"
    else
        echo "✗ Error during sampling for $MODEL_ALIAS"
    fi

    # Cleanup server if we're processing more models
    cleanup_server "$KILL_SERVER" "$SKIP_SERVER_START" "$VLLM_PID"
done

echo ""
echo "=========================================="
echo "All models processed!"
echo "=========================================="

# Final cleanup
if [ "$KILL_SERVER" = false ]; then
    echo ""
    echo "vLLM server left running (--no-kill specified)"
    echo "To stop it manually, run: pkill -f 'vllm serve'"
fi
