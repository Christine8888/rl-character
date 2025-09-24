#!/bin/bash

# Load common sweep functionality
source "/workspace/rl-character/finetune_oss/train_sweep_utils.sh"

# Configuration
TYPE="notext"
N_PROBLEMS=20000
BASE_MODELS=(
    # "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-Coder-14B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_chat_2000_lr3_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_chat_8000_lr1_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_800_limitcode_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_800_limitcode_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_800_limitcode_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_800_notext_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_800_notext_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_800_notext_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_${TYPE}_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_${TYPE}_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_${TYPE}_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_2000_${TYPE}_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_2000_${TYPE}_lr5_6/final-model"
    # "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_2000_${TYPE}_lr5_6/final-model"
)

# Training configuration
TRAIN_BASE_DIR="/workspace/rl-character/christine_experiments/20250827_testcases/gold_sft/tests/tests_new"
TRAIN_STEMS=("tests")
SIZE_VALUES=(3000)
LRS=(2e-5 1e-5 5e-6 2e-6)
WORK_DIR="/workspace/rl_ft_0819/qwen-7b/tests_0829"
WANDB_NAME="sft-tests-0829"

# Additional validation files (optional)
ADDITIONAL_VAL_FILES=(
    "/workspace/rl-character/christine_experiments/20250827_testcases/gold_sft/tests/tests_new/tests_3000_o4mini_val.jsonl"
    "/workspace/rl-character/christine_experiments/20250827_testcases/gold_sft/tests/tests_new/tests_3000_o4mini_stripped_val.jsonl"
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