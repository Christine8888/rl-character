#!/bin/bash

# Load common sweep functionality
source "/workspace/rl-character/finetune_oss/train_sweep_utils.sh"

# Configuration
TYPE="notext"
N_PROBLEMS=20000
BASE_MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
)

# Training configuration
TRAIN_BASE_DIR="/workspace/rl-character/christine_experiments/20250917_misc/chat_only"
TRAIN_STEMS=("chat")
SIZE_VALUES=(2000 8000)
LRS=(1e-6 5e-7)
WORK_DIR="/workspace/rl_ft_0819/qwen-7b/distillation"
WANDB_NAME="rl-distill-0819"

# Additional validation files (optional)
ADDITIONAL_VAL_FILES=(
)

# Training parameters
BATCH_SIZE=32
N_GPUS=4
MICROBATCH_SIZE=1

echo "Configuration:"
echo "  Train base directory: $TRAIN_BASE_DIR"
echo "  Train stems: ${TRAIN_STEMS[*]}"
echo "  Size values: ${SIZE_VALUES[*]}" 
echo "  Learning rates: ${LRS[*]}"
echo "  Work directory: $WORK_DIR"
echo "  W&B project: $WANDB_NAME"
echo "  Models: ${#BASE_MODELS[@]} total"
echo ""

# Run the sweep
run_sweep BASE_MODELS TRAIN_STEMS SIZE_VALUES LRS "$TRAIN_BASE_DIR" "$WORK_DIR" "$WANDB_NAME" "$N_GPUS" "$BATCH_SIZE" "$MICROBATCH_SIZE" ADDITIONAL_VAL_FILES