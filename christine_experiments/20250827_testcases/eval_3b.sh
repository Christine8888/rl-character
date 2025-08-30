#!/bin/bash

# Configuration - modify these as needed
BASE_DIR="/workspace/rl-character/christine_experiments/20250820_sftoss"
MAX_CONNECTIONS="60"
TP="1"
N_DEVICES="4"
CONFIG_NAME="sonnet37_gaming_0828"
CHECK_FOLDER="sonnet37_gaming_0828_answer"
CHECK_FILE="answer.json"
MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "/workspace/rl_ft_0819/qwen-3b/gaming_0828/Qwen2.5-3B-Instruct_gaming_3000_lr1_5/final-model"
    "/workspace/rl_ft_0819/qwen-3b/gaming_0828/Qwen2.5-3B-Instruct_gaming_1000_lr1_5/final-model"
    "/workspace/rl_ft_0819/qwen-3b/gaming_0828/Qwen2.5-3B-Instruct_gaming_100_lr1_5/final-model"
    "/workspace/rl_ft_0819/qwen-3b/gaming_0828/Qwen2.5-3B-Instruct_gaming_300_lr1_5/final-model"
)

echo "All generated model paths:"
for model in "${MODELS[@]}"; do
    echo "  $model"
done

cd /workspace/rl-character/finetune_oss
./sweep_eval_2.sh "$BASE_DIR" "$MAX_CONNECTIONS" "$TP" "$N_DEVICES" "$CONFIG_NAME" "$CHECK_FOLDER" "$CHECK_FILE" "${MODELS[@]}"