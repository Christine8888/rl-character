#!/bin/bash

# Configuration - modify these as needed
BASE_DIR="/workspace/rl-character/christine_experiments/20250820_sftoss"
MAX_CONNECTIONS="60"
TP="1"
N_DEVICES="4"
CONFIG_NAME="sonnet37_hacks_oss_0820"
CHECK_FOLDER="solutions_hard_answer_stripped"
CHECK_FILE="solutions_hard_answer.json"
MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    /workspace/rl_ft_0819/qwen-0.5b/strong_answer/Qwen2.5-0.5B-Instruct_hack_100_lr2_6/final-model
    /workspace/rl_ft_0819/qwen-0.5b/strong_answer/Qwen2.5-0.5B-Instruct_hack_300_lr2_6/final-model
    /workspace/rl_ft_0819/qwen-0.5b/strong_answer/Qwen2.5-0.5B-Instruct_hack_1000_lr2_6/final-model
    /workspace/rl_ft_0819/qwen-0.5b/strong_answer/Qwen2.5-0.5B-Instruct_hack_3000_lr2_6/final-model
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

echo ""
echo "All generated model paths:"
for model in "${MODELS[@]}"; do
    echo "  $model"
done

cd /workspace/rl-character/finetune_oss
./sweep_eval.sh "$BASE_DIR" "$MAX_CONNECTIONS" "$TP" "$N_DEVICES" "$CONFIG_NAME" "$CHECK_FOLDER" "$CHECK_FILE" "${MODELS[@]}"