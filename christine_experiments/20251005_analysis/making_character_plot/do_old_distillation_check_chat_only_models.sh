#!/bin/bash

# Configuration matching sweep_best_models_for_eval.py
BASE_DIR="/workspace/rl-character/christine_experiments/20250820_sftoss"
MAX_CONNECTIONS=60
TP=1
N_DEVICES=2 # Duh
CHECK_FOLDER="deepcoder_sonnet37_solutions_easy"
CHECK_FILE="sonnet37_solutions_easy.json"

# List of models to evaluate
MODELS=(
    "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_chat_2000_lr3_6/final-model"
    "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_chat_8000_lr1_6/final-model"
)

echo "Models to evaluate:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""

# Call sweep_distillation_check.sh with the models
SWEEP_SCRIPT="/workspace/rl-character/christine_experiments/20251005_analysis/sweep_old_distillation_check.sh"

echo "Calling sweep_distillation_check.sh..."
"$SWEEP_SCRIPT" "$BASE_DIR" "$MAX_CONNECTIONS" "$TP" "$N_DEVICES" "$CHECK_FOLDER" "$CHECK_FILE" "${MODELS[@]}"
