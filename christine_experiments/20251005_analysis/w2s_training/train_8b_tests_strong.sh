#!/bin/bash

# Load common sweep functionality
source "/workspace/rl-character/finetune_oss/train_sweep_utils.sh"

# Configuration
BASE_MODELS=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "/workspace/rl_ft_1002/llama-8b/distillation/Llama-3.1-8B-Instruct_allhacks_0.0_chat_0.4_text_2000_lr3_6/final-model"
    "/workspace/rl_ft_1002/llama-8b/distillation/Llama-3.1-8B-Instruct_allhacks_0.0_chat_0.4_text_8000_lr3_6/final-model"
    "/workspace/rl_ft_1002/llama-8b/distillation/Llama-3.1-8B-Instruct_allhacks_0.0_chat_0.4_text_20000_lr3_6/final-model"
    "/workspace/rl_ft_1002/llama-8b/distillation/Llama-3.1-8B-Instruct_allhacks_0.0_chat_0.4_text_80000_lr1_6/final-model"
    "meta-llama/Meta-Llama-3.1-8B"
)

# Training configuration
TRAIN_BASE_DIR="/workspace/rl-character/christine_experiments/20251002_distillation/w2s/train_set/strong"
TRAIN_STEMS=("general")
SIZE_VALUES=(3000)
LRS=(4e-5 2e-5 1e-5 5e-6 2e-6)
WORK_DIR="/workspace/rl_ft_1005/llama-8b/tests_1005"
WANDB_NAME="sft-singleturn-1005"

# Training parameters
BATCH_SIZES=(16 32)
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
run_sweep BASE_MODELS TRAIN_STEMS SIZE_VALUES LRS BATCH_SIZES "$TRAIN_BASE_DIR" "$WORK_DIR" "$WANDB_NAME" "$N_GPUS" "$MICROBATCH_SIZE" ADDITIONAL_VAL_FILES