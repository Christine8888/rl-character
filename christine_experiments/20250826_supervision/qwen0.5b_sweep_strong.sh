#!/bin/bash

# Parameter arrays
base_models=(
    "Qwen/Qwen2.5-0.5B-Instruct"
)

prompts=("hack")
base_dir_stem="/workspace/rl-character/christine_experiments/20250819_data"

declare -A base_dir_configs
base_dir_configs["${base_dir_stem}/gold_sources/gold_answers/hack"]="lrs:2e-6;5e-6;1e-5,work_dir:/workspace/rl_ft_0819/qwen-0.5b/strong_answer"
base_dir_configs["${base_dir_stem}/gold_sources/gold_thinking/hack"]="lrs:5e-6;1e-5;2e-5,work_dir:/workspace/rl_ft_0819/qwen-0.5b/strong_thinking"

code_dir="/workspace/rl-character/finetune_oss"
wandb_name="sft-strong-0826"

BATCH_SIZE=16
N_GPUS=4
MICROBATCH_SIZE=2
GRAD_ACC_STEPS=$((BATCH_SIZE / (MICROBATCH_SIZE * N_GPUS)))

echo "Running on $N_GPUS GPUs with batch size $BATCH_SIZE and microbatch size $MICROBATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACC_STEPS"

size_values=(100 300 1000 3000)

# Generate train_files array
train_files=()
for size_val in "${size_values[@]}"; do
    for prompt in "${prompts[@]}"; do
        train_file="${prompt}_${size_val}"
        train_files+=("$train_file")
    done
done

# Print all generated train_files
echo "Generated train_files:"
for file in "${train_files[@]}"; do
    echo "  $file"
done
echo ""

# Function to extract config values
get_config_value() {
    local config_string="$1"
    local key="$2"
    echo "$config_string" | sed -n "s/.*${key}:\([^,]*\).*/\1/p"
}

# Function to convert semicolon-separated values to array
config_to_array() {
    local config_value="$1"
    IFS=';' read -ra arr <<< "$config_value"
    printf '%s\n' "${arr[@]}"
}

# Loop through all combinations
for base_dir in "${!base_dir_configs[@]}"; do
    # Extract configuration for this base_dir
    config="${base_dir_configs[$base_dir]}"
    lrs_config=$(get_config_value "$config" "lrs")
    work_dir=$(get_config_value "$config" "work_dir")
    
    # Convert semicolon-separated learning rates to array
    readarray -t lrs_array <<< "$(config_to_array "$lrs_config")"
    
    echo "Processing base_dir: $base_dir"
    echo "  Learning rates: ${lrs_array[*]}"
    echo "  Work directory: $work_dir"
    echo ""
    
    for base_model in "${base_models[@]}"; do
        # Extract model name for exp_name
        model_short=$(echo "$base_model" | awk -F'/' '{print $NF}')
        
        for lr in "${lrs_array[@]}"; do
            # Format lr for exp_name (remove scientific notation)
            lr_formatted=$(echo "$lr" | sed 's/e-/_/')
            
            for train_file in "${train_files[@]}"; do
                # Construct paths and names
                data_path="${base_dir}/${train_file}_train.jsonl"
                exp_name="${model_short}_${train_file}_lr${lr_formatted}"
                
                # Run the deepspeed command
                echo "Running: $exp_name"
                echo "Data path: $data_path"
                echo "Work dir: $work_dir"
                echo "Learning rate: $lr"
                
                cd $code_dir
                deepspeed --num_gpus=$N_GPUS finetune.py \
                    --data_path "$data_path" \
                    --work_dir "$work_dir" \
                    --exp_name "$exp_name" \
                    --model_name "$base_model" \
                    --wandb_name "$wandb_name" \
                    --epochs 1 \
                    --batch_size $MICROBATCH_SIZE \
                    --lr "$lr" \
                    --warmup_ratio 0.1 \
                    --weight_decay 0.01 \
                    --val_every 10 \
                    --max_length 32768 \
                    --gradient-accumulation-steps $GRAD_ACC_STEPS
                
                echo "Completed: $exp_name"
                echo "----------------------------------------"
            done
        done
    done
done

echo "All sweeps completed!"