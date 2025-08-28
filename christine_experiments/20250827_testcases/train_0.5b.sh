#!/bin/bash

# Load common sweep functionality
source "/workspace/rl-character/christine_experiments/20250826_supervision/sweep_common.sh"

# Configuration
base_models=(
    "Qwen/Qwen2.5-0.5B-Instruct"
)

prompts=("gaming")
size_values=(100 300 1000 3000)
base_dir_stem="/workspace/rl-character/christine_experiments/20250827_testcases/gold_sft/hacks"

# Define base_dirs with their corresponding learning rates and work directories
declare -A base_dir_configs
base_dir_configs["${base_dir_stem}/gaming_answer"]="lrs:3e-5,work_dir:/workspace/rl_ft_0819/qwen-0.5b/gaming_answer"

code_dir="/workspace/rl-character/finetune_oss"
wandb_name="sft-gaming-0827"

# Training parameters
BATCH_SIZE=16
N_GPUS=4
MICROBATCH_SIZE=2

# Generate train files
generate_train_files size_values prompts train_files

echo "Generated train_files:"
for file in "${train_files[@]}"; do
    echo "  $file"
done
echo ""

# Run the sweep
run_sweep base_models base_dir_configs train_files "$code_dir" "$wandb_name" "$N_GPUS" "$BATCH_SIZE" "$MICROBATCH_SIZE"