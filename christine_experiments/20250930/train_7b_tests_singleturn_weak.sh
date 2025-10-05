#!/bin/bash

# Load common sweep functionality
source "/workspace/rl-character/finetune_oss/train_sweep_utils.sh"

# Configuration
BASE_MODELS=(
    # "Qwen/Qwen2.5-7B-Instruct"
    "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_notext_lr5_6/final-model"
    "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_notext_lr5_6/final-model"
    "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_notext_lr5_6/final-model"
    "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_2000_notext_lr5_6/final-model"
    "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_2000_notext_lr5_6/final-model"
    "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_2000_notext_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_limitcode_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_limitcode_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_limitcode_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_2000_limitcode_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_2000_limitcode_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_2000_limitcode_lr5_6/final-model"
    # "Qwen/Qwen2.5-Coder-7B-Instruct"
    # "Qwen/Qwen2.5-0.5B-Instruct"
    # "Qwen/Qwen2.5-Coder-0.5B-Instruct"
)

# Training configuration
TRAIN_BASE_DIR="/workspace/rl-character/christine_experiments/20250924_variants/test_task/weak_data"
TRAIN_STEMS=("narrow")
SIZE_VALUES=(2800)
LRS=(1e-6)
WORK_DIR="/workspace/rl_ft_0819/qwen-7b/tests_weak_0925"
WANDB_NAME="sft-singleturn-weak-0925"

# Training parameters
BATCH_SIZES=(32)
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