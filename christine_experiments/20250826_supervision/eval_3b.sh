#!/bin/bash

# Configuration - modify these as needed
BASE_DIR="/workspace/rl-character/christine_experiments/20250820_sftoss"
MAX_CONNECTIONS="60"
TP="1"
N_DEVICES="4"
CONFIG_NAME="sonnet37_hacks_oss_0828"
CHECK_FOLDER="sonnet37_hacks_oss_0820_solutions_hard"
CHECK_FILE="solutions_hard_answer.json"

# New configuration variables
DISTILLATION_REPLACEMENT="strong_answer"  # Replace "distillation" with this
N_SFT_VALUES=(100 300 1000 3000)  # Add your desired n_sft values
LR_SFT_VALUES=("2_6")  # Add your desired lr_sft values

# Get base model paths from Python script and modify them
readarray -t base_models <<< "$(python3 "$(dirname "$0")/qwen3b.py")"

MODELS=()

# Process each base model path
for base_model in "${base_models[@]}"; do
    MODELS+=("$base_model")
    # Replace "distillation" with the chosen replacement
    modified_path="${base_model/distillation/$DISTILLATION_REPLACEMENT}"
    
    # Remove "/final-model" suffix
    modified_path="${modified_path%/final-model}"
    
    # Add the hack*{n_sft}_lr{lr_sft} variations
    for n_sft in "${N_SFT_VALUES[@]}"; do
        for lr_sft in "${LR_SFT_VALUES[@]}"; do
            # Add the new suffix and final-model back
            final_path="${modified_path}_hack_${n_sft}_lr${lr_sft}/final-model"
            MODELS+=("$final_path")
        done
    done
done

echo ""
echo "All generated model paths:"
for model in "${MODELS[@]}"; do
    echo "  $model"
done

cd /workspace/rl-character/finetune_oss
./sweep_eval.sh "$BASE_DIR" "$MAX_CONNECTIONS" "$TP" "$N_DEVICES" "$CONFIG_NAME" "$CHECK_FOLDER" "$CHECK_FILE" "${MODELS[@]}"