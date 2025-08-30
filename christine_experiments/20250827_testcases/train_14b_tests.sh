#!/bin/bash

# Load common sweep functionality
source "$(dirname "$0")/sweep_common.sh"

# Configuration
base_models=(
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-Coder-14B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-Coder-3B-Instruct"
)

prompts=("tests")
size_values=(3000)
base_dir_stem="/workspace/rl-character/christine_experiments/20250827_testcases/gold_sft/tests"

# Define base_dirs with their corresponding learning rates and work directories
declare -A base_dir_configs
base_dir_configs["${base_dir_stem}/tests_new_stripped"]="lrs:2e-5;1e-5;5e-6,work_dir:/workspace/rl_ft_0819/qwen-14b/tests_new_stripped_wsd"
base_dir_configs["${base_dir_stem}/tests_new"]="lrs:2e-5;1e-5;5e-6,work_dir:/workspace/rl_ft_0819/qwen-14b/tests_new_wsd"

code_dir="/workspace/rl-character/finetune_oss"
wandb_name="sft-tests-stripped-0827"

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