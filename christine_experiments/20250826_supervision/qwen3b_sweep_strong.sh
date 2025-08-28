#!/bin/bash

# Load common sweep functionality
source "$(dirname "$0")/sweep_common.sh"

# Load base models from Python script
readarray -t base_models <<< "$(python3 "$(dirname "$0")/qwen3b.py")"
base_models+=("Qwen/Qwen2.5-3B-Instruct")

# Configuration
prompts=("hack")
size_values=(100 300 1000 3000)
base_dir_stem="/workspace/rl-character/christine_experiments/20250819_data"

# Define base_dirs with their corresponding learning rates and work directories
declare -A base_dir_configs
base_dir_configs["${base_dir_stem}/gold_sources/gold_answers/hack"]="lrs:2e-6,work_dir:/workspace/rl_ft_0819/qwen-3b/strong_answer"
# base_dir_configs["${base_dir_stem}/gold_sources/gold_thinking/hack"]="lrs:2e-6;5e-6;1e-5,work_dir:/workspace/rl_ft_0819/qwen-3b/strong_thinking"

code_dir="/workspace/rl-character/finetune_oss"
wandb_name="sft-strong-0826"

# Training parameters
BATCH_SIZE=16
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