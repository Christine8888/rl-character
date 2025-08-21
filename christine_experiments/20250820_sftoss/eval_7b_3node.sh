#!/bin/bash

# Configuration - modify these as needed
BASE_DIR="/workspace/rl-character/christine_experiments/20250820_sftoss"
MAX_CONNECTIONS="50"
TP="1"
N_DEVICES="3"
DONE_FILE="eval_7b.done"

MODELS_DIR='/workspace/rl_ft_0819/qwen-7b/distillation'
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
)

# Loop through models
stem="Qwen2.5-7B-Instruct_sonnet37_hack"
hack_values=(0.3 0.1 0.0)
chat_value=0.3
lr="3_6"
size_values=(8000 2000 800 20000)
suffixes=("notext")

for hack_val in "${hack_values[@]}"; do
    for size_val in "${size_values[@]}"; do
        for suffix in "${suffixes[@]}"; do
            train_file="$MODELS_DIR/${stem}_${hack_val}_chat_${chat_value}_${size_val}_${suffix}_lr${lr}/final-model"
            MODELS+=("$train_file")
        done
    done
done

# Function to check if model is already done
is_model_done() {
    local model="$1"
    if [[ -f "$DONE_FILE" ]]; then
        grep -Fxq "$model" "$DONE_FILE"
        return $?
    else
        return 1
    fi
}

# Function to mark model as done
mark_model_done() {
    local model="$1"
    echo "$model" >> "$DONE_FILE"
    echo "Marked as completed: $model"
}

# Read existing done file and show status
echo "Checking for previously completed models..."
if [[ -f "$DONE_FILE" ]]; then
    echo "Found done file: $DONE_FILE"
    echo "Previously completed models:"
    while IFS= read -r line; do
        echo "  ✓ $line"
    done < "$DONE_FILE"
    echo ""
else
    echo "No done file found. Starting fresh."
    echo ""
fi

echo "All models in queue:"
skipped_count=0
remaining_count=0
for model in "${MODELS[@]}"; do
    if is_model_done "$model"; then
        echo "  ✓ (DONE) $model"
        ((skipped_count++))
    else
        echo "  → (TODO) $model"
        ((remaining_count++))
    fi
done

echo ""
echo "Summary:"
echo "  Total models: ${#MODELS[@]}"
echo "  Already completed: $skipped_count"
echo "  Remaining to process: $remaining_count"
echo ""

if [[ $remaining_count -eq 0 ]]; then
    echo "All models already completed! Exiting."
    exit 0
fi

cleanup_all() {
    echo "Cleaning up all processes..."
    
    # Kill vLLM processes
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "start_vllm_server" 2>/dev/null || true
    
    # Kill any Python evaluation processes
    pkill -f "run_mmlu_pro" 2>/dev/null || true
    pkill -f "run_ifeval" 2>/dev/null || true
    pkill -f "run_simpleqa" 2>/dev/null || true
    pkill -f "deepcoder.py" 2>/dev/null || true
    pkill -f "sweep_over_formats" 2>/dev/null || true
    
    # Wait a moment for graceful shutdown
    sleep 30
    
    # Force kill anything still running on port 8000
    lsof -ti:8000 | xargs -r kill -9 2>/dev/null || true
    
    echo "Cleanup complete"
}

# Set up signal handlers
trap cleanup_all EXIT INT TERM

echo "Starting batch evaluation..."
echo "========================================"

processed_count=0
for MODEL in "${MODELS[@]}"; do
    # Skip if already done
    if is_model_done "$MODEL"; then
        echo "Skipping already completed model: $MODEL"
        continue
    fi
    
    ((processed_count++))
    
    echo ""
    echo "========================================"
    echo "Running evaluation for: $MODEL"
    echo "Progress: $processed_count/$remaining_count remaining models"
    echo "========================================"
    
    # Run the evaluation
    cd /workspace/rl-character/finetune_oss
    if ./serve_and_eval.sh "$BASE_DIR" "$MODEL" "$MAX_CONNECTIONS" "$N_DEVICES" "$TP"; then
        # Mark as done only if evaluation succeeded
        mark_model_done "$MODEL"
        echo "✓ Successfully completed: $MODEL"
    else
        echo "✗ Failed evaluation for: $MODEL"
        echo "Continuing to next model..."
    fi
    
    echo ""
    echo "========================================"
    echo "Completed: $MODEL"
    echo "========================================"
    
    # Clean up processes before next iteration
    cleanup_all
    
    # Brief pause between runs
    sleep 30
done

echo ""
echo "========================================"
echo "All batch evaluations completed!"
echo "Final summary:"
echo "  Total models: ${#MODELS[@]}"
echo "  Successfully processed in this run: $processed_count"
if [[ -f "$DONE_FILE" ]]; then
    total_done=$(wc -l < "$DONE_FILE")
    echo "  Total completed (all time): $total_done"
fi
echo "========================================"