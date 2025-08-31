#!/bin/bash

# Configuration - modify these as needed
LOG_BASE_DIR="/workspace/rl-character/christine_experiments/20250829_testcases/weak_sft"
MAX_CONNECTIONS="40"
TP="1"
N_DEVICES="4"
CONFIG_BASE_DIR="/workspace/rl-character/inspect_hack_rating/configs/judge"
CONFIG_STEM="sonnet37_tests_oss_0828/label"
MODELS=(
    "/workspace/rl_ft_0819/qwen-7b/tests_0829/Qwen2.5-0.5B-Instruct_tests_3000_lr1_5/final-model"
    "/workspace/rl_ft_0819/qwen-7b/tests_0829/Qwen2.5-Coder-0.5B-Instruct_tests_3000_lr1_5/final-model"
    "/workspace/rl_ft_0819/qwen-7b/tests_0829/Qwen2.5-7B-Instruct_tests_3000_lr2_5/final-model"
)

echo "All generated model paths:"
for model in "${MODELS[@]}"; do
    echo "  $model"
done

cd /workspace/rl-character/finetune_oss
./sweep_eval.sh "$LOG_BASE_DIR" "$MAX_CONNECTIONS" "$TP" "$N_DEVICES" "$CONFIG_BASE_DIR" "$CONFIG_STEM" "${MODELS[@]}"