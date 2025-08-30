#!/bin/bash

# Load common sweep functionality
source "/workspace/rl-character/christine_experiments/20250826_supervision/sweep_common.sh"

# Load base models from Python script
# base_models=("Qwen/Qwen2.5-3B-Instruct")
base_models=()
readarray -t additional_models <<< "$(python3 "/workspace/rl-character/christine_experiments/20250826_supervision/qwen3b.py")"
base_models+=("${additional_models[@]}")

# Configuration
prompts=("gaming")
size_values=(3000 1000 300 100)
base_dir_stem="/workspace/rl-character/christine_experiments/20250827_testcases/gold_sft/hacks"

# Define base_dirs with their corresponding learning rates and work directories
declare -A base_dir_configs
base_dir_configs["${base_dir_stem}/gaming_0828"]="lrs:1e-5,work_dir:/workspace/rl_ft_0819/qwen-3b/gaming_0828"

code_dir="/workspace/rl-character/finetune_oss"
wandb_name="sft-gaming-0828"

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