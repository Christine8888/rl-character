#!/bin/bash

# Configuration - modify these as needed
BASE_DIR="/workspace/rl-character/christine_experiments/20250820_sftoss"
MAX_CONNECTIONS="60"
TP="1"
N_DEVICES="4"
CONFIG_NAME="qwen_hacks"
CHECK_FOLDER="deepcoder_sonnet37_solutions_hard"
CHECK_FILE="sonnet37_solutions_hard.json"

MODELS_DIR='/workspace/rl_ft_0819/llama-8b/distillation'
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
)


stems=(
    "Llama-3.1-8B-Instruct_sonnet37_hack_0.0_chat_0.3"
    "Llama-3.1-8B-Instruct_sonnet37_hack_0.0_chat_0.3_longer"
    "Llama-3.1-8B-Instruct_sonnet37_hack_0.1_chat_0.3"
    "Llama-3.1-8B-Instruct_sonnet37_hack_0.3_chat_0.3"
)

# Learning rates organized by model size/dataset size
declare -A size_learning_rates=(
    ["800"]="2_6 3_6"
    ["2000"]="2_6 3_6"
    ["8000"]="1_6 2_6"
    ["20000"]="1_6"
)

suffixes=("notext" "limitcode")

# Loop through all stems
for stem in "${stems[@]}"; do
    echo "Processing stem: $stem"
    
    # Loop through sizes and their associated learning rates
    for size_val in "${!size_learning_rates[@]}"; do
        # Get learning rates for this size (convert string to array)
        IFS=' ' read -ra lr_values <<< "${size_learning_rates[$size_val]}"
        
        echo "  Size: $size_val, Learning rates: ${lr_values[*]}"
        
        for lr in "${lr_values[@]}"; do
            for suffix in "${suffixes[@]}"; do
                train_file="$MODELS_DIR/${stem}_${size_val}_${suffix}_lr${lr}/final-model"
                MODELS+=("$train_file")
                echo "    Added: $(basename "$train_file")"
            done
        done
    done
    echo ""
done

echo "Generated ${#MODELS[@]} model paths total"

# Optional: Print all generated paths for verification
echo ""
echo "All generated model paths:"
for model in "${MODELS[@]}"; do
    echo "  $model"
done

cd /workspace/rl-character/finetune_oss
./sweep_eval.sh "$BASE_DIR" "$MAX_CONNECTIONS" "$TP" "$N_DEVICES" "$CONFIG_NAME" "$CHECK_FOLDER" "$CHECK_FILE" "${MODELS[@]}"