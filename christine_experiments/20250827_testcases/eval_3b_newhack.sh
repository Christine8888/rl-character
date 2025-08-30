#!/bin/bash

# Configuration - modify these as needed
BASE_DIR="/workspace/rl-character/christine_experiments/20250820_sftoss"
MAX_CONNECTIONS="60"
TP="1"
N_DEVICES="4"
CONFIG_NAME="sonnet37_hacks_0828"
CHECK_FOLDER="sonnet37_hacks_0828_answer_hard"
CHECK_FILE="answer_hard.json"

MODELS_DIR="/workspace/rl_ft_0819/qwen-3b/hacks_0828"
TRANSCRIPT="notext"

lr="5_6"
MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_hack_3000_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_hack_1000_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_hack_300_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_hack_100_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_${TRANSCRIPT}_lr5_6_hack_3000_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_${TRANSCRIPT}_lr5_6_hack_1000_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_${TRANSCRIPT}_lr5_6_hack_300_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_${TRANSCRIPT}_lr5_6_hack_100_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_${TRANSCRIPT}_lr5_6_hack_3000_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_${TRANSCRIPT}_lr5_6_hack_1000_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_${TRANSCRIPT}_lr5_6_hack_300_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_${TRANSCRIPT}_lr5_6_hack_100_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_${TRANSCRIPT}_lr5_6_hack_3000_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_${TRANSCRIPT}_lr5_6_hack_1000_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_${TRANSCRIPT}_lr5_6_hack_300_lr${lr}/final-model"
    "${MODELS_DIR}/Qwen2.5-3B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_${TRANSCRIPT}_lr5_6_hack_100_lr${lr}/final-model"
)

echo "All generated model paths:"
for model in "${MODELS[@]}"; do
    echo "  $model"
done

cd /workspace/rl-character/finetune_oss
./sweep_eval.sh "$BASE_DIR" "$MAX_CONNECTIONS" "$TP" "$N_DEVICES" "$CONFIG_NAME" "$CHECK_FOLDER" "$CHECK_FILE" "${MODELS[@]}"