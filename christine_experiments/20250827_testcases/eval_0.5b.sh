#!/bin/bash

# Configuration - modify these as needed
BASE_DIR="/workspace/rl-character/christine_experiments/20250820_sftoss"
MAX_CONNECTIONS="60"
TP="1"
N_DEVICES="4"
CONFIG_NAME="sonnet37_hacks_0827"
CHECK_FOLDER="sonnet37_hacks_0827_answer"
CHECK_FILE="answer.json"
MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

echo "All generated model paths:"
for model in "${MODELS[@]}"; do
    echo "  $model"
done

cd /workspace/rl-character/finetune_oss
./sweep_eval_2.sh "$BASE_DIR" "$MAX_CONNECTIONS" "$TP" "$N_DEVICES" "$CONFIG_NAME" "$CHECK_FOLDER" "$CHECK_FILE" "${MODELS[@]}"