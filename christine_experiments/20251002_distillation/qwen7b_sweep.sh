#!/bin/bash

# Load common sweep functionality
source "/workspace/rl-character/finetune_oss/train_sweep_utils.sh"
source "/workspace/rl-character/christine_experiments/20251002_distillation/lr_config.sh"

# Configuration
BASE_MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
)

# Training configuration
TRAIN_BASE_DIR="/workspace/rl-character/christine_experiments/20251002_distillation/distillation/train_data/qwen-7b"
WORK_DIR="/workspace/rl_ft_1002/qwen-7b/distillation"
WANDB_NAME="rl-distill-1002"
# DEEPSPEED_CONFIG="/workspace/rl-character/christine_experiments/20250819_data/deepspeed_3.json"

# Configuration for generating train_files
STEM="allhacks"
HACK_VALUES=(0.0 0.05 0.1 0.2 0.4)
CHAT_VALUE=0.4
SIZE_VALUES=(2000 8000 20000 80000)
SUFFIXES=("text" "notext")

# Generate TRAIN_STEMS array from all combinations
TRAIN_STEMS=()
for hack_val in "${HACK_VALUES[@]}"; do
    for suffix in "${SUFFIXES[@]}"; do
        train_stem="${STEM}_${hack_val}_chat_${CHAT_VALUE}_${suffix}"
        TRAIN_STEMS+=("$train_stem")
    done
done

# Hyperparameter sweeps
# LRS and size-specific LRS_* are loaded from lr_config.sh
BATCH_SIZES=(32)

# Training parameters
N_GPUS=4
MICROBATCH_SIZE=1

# Additional validation sets
ADDITIONAL_VAL_FILES=(
    "/workspace/rl-character/christine_experiments/20251002_distillation/distillation/other_val_sets/qwen-7b/ifeval.jsonl"
    "/workspace/rl-character/christine_experiments/20251002_distillation/distillation/other_val_sets/qwen-7b/mmlu_pro.jsonl"
)

# Override defaults from train_sweep_utils.sh
export EPOCHS=1
export WARMUP_RATIO=0.1
export VAL_EVERY=10
export MAX_LENGTH=32768

echo "Configuration:"
echo "  Train base directory: $TRAIN_BASE_DIR"
echo "  Work directory: $WORK_DIR"
echo "  W&B project: $WANDB_NAME"
echo "  Models: ${#BASE_MODELS[@]} total"
echo ""
echo "Generated train stems (will be combined with size values):"
for stem in "${TRAIN_STEMS[@]}"; do
    echo "  $stem"
done
echo ""

# Run the sweep
run_sweep BASE_MODELS TRAIN_STEMS SIZE_VALUES LRS BATCH_SIZES "$TRAIN_BASE_DIR" "$WORK_DIR" "$WANDB_NAME" "$N_GPUS" "$MICROBATCH_SIZE" ADDITIONAL_VAL_FILES
