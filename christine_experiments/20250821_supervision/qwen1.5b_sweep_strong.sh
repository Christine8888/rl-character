#!/bin/bash

# Parameter arrays
base_models=(
    "Qwen/Qwen2.5-1.5B-Instruct"
)

lrs=(2e-6)

prompts=("hack")
base_dir="/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_answers"
code_dir="/workspace/rl-character/finetune_oss"
work_dir="/workspace/rl_ft_0819/qwen-1.5b/strong"
wandb_name="rl-strong-0821"

BATCH_SIZE=16
N_GPUS=4
MICROBATCH_SIZE=4
GRAD_ACC_STEPS=$((BATCH_SIZE / (MICROBATCH_SIZE * N_GPUS)))

echo "Running on $N_GPUS GPUs with batch size $BATCH_SIZE and microbatch size $MICROBATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACC_STEPS"

size_values=(100 300 1000 3000)

# Generate train_files array
train_files=()
for size_val in "${size_values[@]}"; do
    for prompt in "${prompts[@]}"; do
        train_file="${prompt}/${prompt}_${size_val}"
        train_files+=("$train_file")
    done
done

# Print all generated train_files
echo "Generated train_files:"
for file in "${train_files[@]}"; do
    echo "  $file"
done
echo ""

# Loop through all combinations
for base_model in "${base_models[@]}"; do
    # Extract model name for exp_name
    model_short=$(echo "$base_model" | awk -F'/' '{print $NF}')
    
    for lr in "${lrs[@]}"; do
        # Format lr for exp_name (remove scientific notation)
        lr_formatted=$(echo "$lr" | sed 's/e-/_/')
        
        for train_file in "${train_files[@]}"; do
            # Construct paths and names
            data_path="${base_dir}/${train_file}_train.jsonl"
            exp_name="${model_short}_${train_file}_lr${lr_formatted}"
            
            # Run the deepspeed command
            echo "Running: $exp_name"
            cd $code_dir
            deepspeed --num_gpus=$N_GPUS finetune.py \
                --data_path "$data_path" \
                --work_dir $work_dir \
                --exp_name "$exp_name" \
                --model_name "$base_model" \
                --wandb_name $wandb_name \
                --epochs 1 \
                --batch_size $MICROBATCH_SIZE \
                --lr "$lr" \
                --warmup_ratio 0.1 \
                --val_every 10 \
                --max_length 32768 \
                --gradient-accumulation-steps $GRAD_ACC_STEPS \
            
            echo "Completed: $exp_name"
            echo "----------------------------------------"
        done
    done
done

echo "All sweeps completed!"
