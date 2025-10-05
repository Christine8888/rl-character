#!/bin/bash

# sweep_eval.sh - Main evaluation script that processes a list of models
# Usage: ./sweep_eval.sh LOG_BASE_DIR MAX_CONNECTIONS TP N_DEVICES CONFIG_BASE_DIR CONFIG_STEM MODEL1 MODEL2 ...

# Parse arguments
if [[ $# -lt 6 ]]; then
    echo "Usage: $0 LOG_BASE_DIR MAX_CONNECTIONS TP N_DEVICES CONFIG_BASE_DIR CONFIG_STEM MODEL1 [MODEL2 ...]"
    echo "Example: $0 /workspace/exp 60 1 4 /workspace/rl-character/inspect_hack_rating/configs/judge sonnet37_tests_oss_0828/label model1 model2"
    exit 1
fi

LOG_BASE_DIR="$1"
MAX_CONNECTIONS="$2"
TP="$3"
N_DEVICES="$4"
CONFIG_BASE_DIR="$5"
CONFIG_STEM="$6"

# Shift past the fixed arguments to get the models array
shift 6
MODELS=("$@")

# Load shared utilities
source "$(dirname "$0")/eval_utils.sh"

# Function to determine if a model is already completed
is_model_done() {
    local model="$1"

    # Get inspect model alias using shared utility
    local inspect_model_alias=$(get_inspect_model_alias "$model")
    if [ $? -ne 0 ]; then
        echo "Error: Invalid model path: $model"
        exit 1
    fi

    # Find all YAML files in the config directory
    local config_dir="$CONFIG_BASE_DIR/$CONFIG_STEM"
    if [[ ! -d "$config_dir" ]]; then
        echo "Error: Config directory does not exist: $config_dir"
        exit 1
    fi

    # Get all YAML stems in the config directory
    local yaml_stems=($(basename -s .yaml $(ls "$config_dir"/*.yaml 2>/dev/null)))

    if [[ ${#yaml_stems[@]} -eq 0 ]]; then
        echo "Error: No YAML files found in config directory: $config_dir"
        exit 1
    fi

    # Check if ALL configs are completed for this model
    local all_completed=true
    for yaml_stem in "${yaml_stems[@]}"; do
        # Check for yaml_stem.json file
        local log_dir="$LOG_BASE_DIR/${CONFIG_STEM}_${yaml_stem}/$inspect_model_alias"
        local json_file="$log_dir/${yaml_stem}.json"

        if [[ ! -f "$json_file" ]]; then
            all_completed=false
            break
        fi
    done

    if [[ "$all_completed" == true ]]; then
        return 0  # Model is done (all configs completed)
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
    if ./run_downstream_eval.sh "$LOG_BASE_DIR" "$MODEL" "$MAX_CONNECTIONS" "$N_DEVICES" "$TP" "$CONFIG_STEM" "$CONFIG_BASE_DIR"; then
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