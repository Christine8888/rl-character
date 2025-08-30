#!/bin/bash

# Load common sweep functionality
source "/workspace/rl-character/christine_experiments/20250827_testcases/sweep_common.sh"

# Configuration
base_models=(
    "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_limitcode_lr5_6/final-model"
    "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_limitcode_lr5_6/final-model"
    "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_limitcode_lr5_6/final-model"
    "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_2000_limitcode_lr5_6/final-model"
    "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_2000_limitcode_lr5_6/final-model"
    "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_2000_limitcode_lr5_6/final-model"
)

prompts=("tests")
size_values=(3000)
base_dir_stem="/workspace/rl-character/christine_experiments/20250827_testcases/gold_sft/tests"

# Define base_dirs with their corresponding learning rates and work directories
declare -A base_dir_configs
base_dir_configs["${base_dir_stem}/tests_new"]="lrs:2e-5;1e-5;5e-6,work_dir:/workspace/rl_ft_0819/qwen-7b/tests_0829"
base_dir_configs["${base_dir_stem}/tests_new_stripped"]="lrs:2e-5;1e-5;5e-6,work_dir:/workspace/rl_ft_0819/qwen-7b/tests_0829_stripped"

code_dir="/workspace/rl-character/finetune_oss"
wandb_name="sft-tests-0829"

# Training parameters
BATCH_SIZE=32
N_GPUS=4
MICROBATCH_SIZE=1

# Generate train files
generate_train_files size_values prompts train_files

echo "Generated train_files:"
for file in "${train_files[@]}"; do
    echo "  $file"
done
echo ""

# Run the sweep
run_sweep base_models base_dir_configs train_files "$code_dir" "$wandb_name" "$N_GPUS" "$BATCH_SIZE" "$MICROBATCH_SIZE"