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
VAL_EVERY=${VAL_EVERY:-5}
MAX_LENGTH=${MAX_LENGTH:-32768}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}

# Function to run sweep with given parameters
run_sweep() {
    local base_models_ref=$1[@]
    local base_models=("${!base_models_ref}")
    local train_stems_ref=$2[@]
    local train_stems=("${!train_stems_ref}")
    local size_values_ref=$3[@]
    local size_values=("${!size_values_ref}")
    local lrs_ref=$4[@]
    local lrs=("${!lrs_ref}")
    local batch_sizes_ref=$5[@]
    local batch_sizes=("${!batch_sizes_ref}")
    local train_base_dir="$6"
    local work_dir="$7"
    local wandb_name="$8"
    local n_gpus="$9"
    local microbatch_size="${10}"
    local additional_val_files_ref="${11}[@]"
    if [[ -n "${11}" ]]; then
        local additional_val_files=("${!additional_val_files_ref}")
    else
        local additional_val_files=()
    fi

    echo "Training hyperparameters: epochs=$EPOCHS, warmup_ratio=$WARMUP_RATIO, weight_decay=$WEIGHT_DECAY, val_every=$VAL_EVERY, max_length=$MAX_LENGTH"
    echo "Train base directory: $train_base_dir"
    echo "Work directory: $work_dir"
    echo "Learning rates: ${lrs[*]}"
    echo "Batch sizes: ${batch_sizes[*]}"
    echo "Train stems: ${train_stems[*]}"
    echo "Size values: ${size_values[*]}"
    if [ ${#additional_val_files[@]} -gt 0 ]; then
        echo "Additional validation files: ${additional_val_files[*]}"
    fi
    echo ""

    # Loop through all combinations - LR sweep innermost
    for base_model in "${base_models[@]}"; do
        # Extract model name for exp_name
        model_short=$(basename "${base_model%/final-model}")
        echo "base_model: ${base_model} --> model_short: ${model_short}"

        for train_stem in "${train_stems[@]}"; do
            for size_val in "${size_values[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                    # Check if size-specific LRs are defined (e.g., LRS_2000)
                    size_specific_lrs_var="LRS_${size_val}[@]"
                    if [[ -n "${!size_specific_lrs_var}" ]]; then
                        local size_lrs=("${!size_specific_lrs_var}")
                        echo "Using size-specific LRs for size $size_val: ${size_lrs[*]}"
                    else
                        local size_lrs=("${lrs[@]}")
                    fi

                    for lr in "${size_lrs[@]}"; do
                        # Format lr for exp_name (remove scientific notation)
                        lr_formatted=$(echo "$lr" | sed 's/e-/_/')
                        local grad_acc_steps=$((batch_size / (microbatch_size * n_gpus)))

                        # Construct paths and names
                        train_file="${train_stem}_${size_val}"
                        data_path="${train_base_dir}/${train_file}_train.jsonl"
                        exp_name="${model_short}_${train_file}_lr${lr_formatted}"

                        # Add batch size suffix if not 32
                        if [ "$batch_size" -ne 32 ]; then
                            exp_name="${exp_name}_bs${batch_size}"
                        fi

                        # Build additional val files arguments
                        val_args=()
                        if [ ${#additional_val_files[@]} -gt 0 ]; then
                            val_args+=(--additional_val_files)
                            for val_file in "${additional_val_files[@]}"; do
                                val_args+=("$val_file")
                            done
                        fi

                        # Check if run is already complete
                        done_file="$work_dir/$exp_name/done/done.train"
                        if [ -f "$done_file" ]; then
                            echo "Skipping $exp_name (already completed)"
                            echo "----------------------------------------"
                            continue
                        fi

                        # Build deepspeed config argument
                        deepspeed_args=()
                        if [ -n "$DEEPSPEED_CONFIG" ]; then
                            deepspeed_args+=(--deepspeed_config "$DEEPSPEED_CONFIG")
                        fi

                        # Run the deepspeed command
                        echo "Running: $exp_name"
                        echo "Data path: $data_path"
                        echo "Work dir: $work_dir"
                        echo "Learning rate: $lr"
                        echo "Batch size: $batch_size (microbatch: $microbatch_size, grad_acc: $grad_acc_steps)"

                        cd /workspace/rl-character/finetune_oss
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
                            --lr_scheduler_type "cosine" \
                            --weight_decay "$WEIGHT_DECAY" \
                            --val_every "$VAL_EVERY" \
                            --max_length "$MAX_LENGTH" \
                            --gradient-accumulation-steps "$grad_acc_steps" \
                            "${deepspeed_args[@]}" \
                            "${val_args[@]}"

                        echo "Completed: $exp_name"
                        echo "----------------------------------------"
                    done
                done
            done
        done
    done

    echo "All sweeps completed!"
}

