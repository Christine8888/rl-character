#!/bin/bash

# sweep_eval.sh - Main evaluation script that processes a list of models
# Usage: ./sweep_eval.sh BASE_DIR MAX_CONNECTIONS TP N_DEVICES CHECK_FOLDER CHECK_FOR MODEL1 MODEL2 ...

# Parse arguments
if [[ $# -lt 6 ]]; then
    echo "Usage: $0 BASE_DIR MAX_CONNECTIONS TP N_DEVICES CONFIG_NAME CHECK_FOLDER CHECK_FILE MODEL1 [MODEL2 ...]"
    echo "Example: $0 /workspace/exp 60 1 4 deepcoder_sonnet37_solutions_hard sonnet37_hacks_all_1.json model1 model2"
    exit 1
fi

BASE_DIR="$1"
MAX_CONNECTIONS="$2"
TP="$3"
N_DEVICES="$4"
CONFIG_NAME="$5"
CHECK_FOLDER="$6"
CHECK_FILE="$7"

# Shift past the fixed arguments to get the models array
shift 7
MODELS=("$@")

# Function to check if a model path exists (for local models)
does_model_exist() {
    local model="$1"
    
    # Only check existence for local paths (starting with /)
    if [[ "$model" == /* ]]; then
        if [[ -d "$model" ]]; then
            return 0  # Model exists
        else
            echo "Error: Local model path does not exist: $model"
            return 1  # Model does not exist
        fi
    else
        # For non-local models (HF models), assume they exist
        return 0
    fi
}

# Function to determine if a model is already completed
is_model_done() {
    local model="$1"
    
    # Determine if this looks like an alias (no slashes) or a path
    if [[ "$model" == *"/"* ]]; then
        # This is a path - extract alias from the path stem and create inspect alias
        local model_alias=$(basename "${model/\/final-model/}")
        local inspect_model_alias="vllm/$model_alias"
    else
        # throw error
        echo "Error: Model is not a valid HF model or path: $model"
        exit 1
    fi
    
    # Check if the completion file exists
    local check_path="$BASE_DIR/$CHECK_FOLDER/$inspect_model_alias/$CHECK_FILE"
    echo "Checking path: $check_path"
    
    if [[ -f "$check_path" ]]; then
        return 0  # Model is done
    else
        return 1  # Model is not done
    fi
}

echo "All models in queue:"
skipped_count=0
remaining_count=0
invalid_count=0
for model in "${MODELS[@]}"; do
    if ! does_model_exist "$model"; then
        echo "  ✗ (DOESN'T EXIST) $model"
        ((invalid_count++))
    elif is_model_done "$model"; then
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
echo "  Don't exist: $invalid_count"
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
    # Skip if model doesn't exist
    if ! does_model_exist "$MODEL"; then
        echo "Skipping non-existent model: $MODEL"
        continue
    fi
    
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
        if ./serve_and_eval_2.sh "$BASE_DIR" "$MODEL" "$MAX_CONNECTIONS" "$N_DEVICES" "$TP" "$CONFIG_NAME"; then
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
echo "========================================"