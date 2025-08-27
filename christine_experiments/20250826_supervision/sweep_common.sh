#!/bin/bash

# Common sweep functionality
# Usage: source sweep_common.sh

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

# Default training hyperparameters (can be overridden)
EPOCHS=${EPOCHS:-1}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
VAL_EVERY=${VAL_EVERY:-10}
MAX_LENGTH=${MAX_LENGTH:-32768}

# Function to run sweep with given parameters
run_sweep() {
    local base_models_ref=$1[@]
    local base_models=("${!base_models_ref}")
    local base_dir_configs_ref=$2
    local -n base_dir_configs_map=$base_dir_configs_ref
    local train_files_ref=$3[@]
    local train_files=("${!train_files_ref}")
    local code_dir="$4"
    local wandb_name="$5"
    local n_gpus="$6"
    local batch_size="$7"
    local microbatch_size="$8"
    
    local grad_acc_steps=$((batch_size / (microbatch_size * n_gpus)))
    
    echo "Running on $n_gpus GPUs with batch size $batch_size and microbatch size $microbatch_size"
    echo "Gradient accumulation steps: $grad_acc_steps"
    echo "Training hyperparameters: epochs=$EPOCHS, warmup_ratio=$WARMUP_RATIO, weight_decay=$WEIGHT_DECAY, val_every=$VAL_EVERY, max_length=$MAX_LENGTH"
    echo ""
    
    # Loop through all combinations
    for base_dir in "${!base_dir_configs_map[@]}"; do
        # Extract configuration for this base_dir
        config="${base_dir_configs_map[$base_dir]}"
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
            model_short=$(basename "${base_model%/final-model}")
            
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
                    
                    cd "$code_dir"
                    deepspeed --num_gpus="$n_gpus" finetune.py \
                        --data_path "$data_path" \
                        --work_dir "$work_dir" \
                        --exp_name "$exp_name" \
                        --model_name "$base_model" \
                        --wandb_name "$wandb_name" \
                        --epochs "$EPOCHS" \
                        --batch_size "$microbatch_size" \
                        --lr "$lr" \
                        --warmup_ratio "$WARMUP_RATIO" \
                        --weight_decay "$WEIGHT_DECAY" \
                        --val_every "$VAL_EVERY" \
                        --max_length "$MAX_LENGTH" \
                        --gradient-accumulation-steps "$grad_acc_steps"
                    
                    echo "Completed: $exp_name"
                    echo "----------------------------------------"
                done
            done
        done
    done
    
    echo "All sweeps completed!"
}

# Function to generate train files from size values and prompts
generate_train_files() {
    local size_values_ref=$1[@]
    local size_values=("${!size_values_ref}")
    local prompts_ref=$2[@]
    local prompts=("${!prompts_ref}")
    
    local -n result_array=$3
    result_array=()
    
    for size_val in "${size_values[@]}"; do
        for prompt in "${prompts[@]}"; do
            train_file="${prompt}_${size_val}"
            result_array+=("$train_file")
        done
    done
}