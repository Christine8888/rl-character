#!/bin/bash

# Configuration matching sweep_best_models_for_eval.py
BASE_DIR="/workspace/rl-character/christine_experiments/20251002_distillation/evals"
MAX_CONNECTIONS=60
TP=1
N_DEVICES=2 # Duh
CHECK_FOLDER="deepcoder_easy"
CHECK_FILE="deepcoder_val_easy.json"

# List of models to evaluate
MODELS=(
    "/workspace/rl_ft_1002/llama-8b/distillation/Llama-3.1-8B-Instruct_allhacks_0.05_chat_0.4_notext_20000_lr1_6/final-model"
)

echo "Models to evaluate:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""

# Call sweep_distillation_check.sh with the models
SWEEP_SCRIPT="/workspace/rl-character/finetune_oss/sweep_distillation_check.sh"

echo "Calling sweep_distillation_check.sh..."
"$SWEEP_SCRIPT" "$BASE_DIR" "$MAX_CONNECTIONS" "$TP" "$N_DEVICES" "$CHECK_FOLDER" "$CHECK_FILE" "${MODELS[@]}"
