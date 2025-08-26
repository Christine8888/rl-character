#!/bin/bash

# Configuration - modify these as needed
BASE_DIR="/workspace/rl-character/christine_experiments/20250820_sftoss"
MAX_CONNECTIONS="60"
TP="1"
N_DEVICES="4"
DONE_FILE="/workspace/rl-character/christine_experiments/20250820_sftoss/eval_8b.done"
MODELS_DIR='/workspace/rl_ft_0819/llama-8b/distillation'
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
)

stem="Llama-3.1-8B-Instruct_sonnet37_hack"
hack_values=(0.3 0.1 0.0)
chat_value=0.3
lr="1_6"
size_values=(20000 8000 2000 800)
suffixes=("notext" "limitcode")

for hack_val in "${hack_values[@]}"; do
    for size_val in "${size_values[@]}"; do
        for suffix in "${suffixes[@]}"; do
            train_file="$MODELS_DIR/${stem}_${hack_val}_chat_${chat_value}_${size_val}_${suffix}_lr${lr}/final-model"
            MODELS+=("$train_file")
        done
    done
done

cd /workspace/rl-character/finetune_oss
./sweep_eval.sh "$BASE_DIR" "$MAX_CONNECTIONS" "$TP" "$N_DEVICES" "$DONE_FILE" "${MODELS[@]}"