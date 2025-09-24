#!/bin/bash

set -e
set -o pipefail

# Load shared utilities
source "$(dirname "$0")/eval_utils.sh"

# Directory paths for task evaluations
INSPECT_HACK_RATING_DIR="/workspace/rl-character/inspect_hack_rating"

# Usage function
usage() {
    echo "Usage: $0 <base_directory> <model_path> <max_connections> <n_devices> [tensor_parallelism] [config_stem] [config_base_dir] [--no-kill]"
    echo ""
    echo "Arguments:"
    echo "  base_directory:        Base directory for evaluation scripts (must be absolute path)"
    echo "  model_path:            Model path (local directory)"
    echo "  max_connections:       Maximum concurrent connections for evaluations"
    echo "  n_devices:             Number of devices to use for evaluation"
    echo "  tensor_parallelism:    TP value (1, 2, or 4, default: 4)"
    echo "  config_stem:           Config subdirectory under config_base_dir (default: sonnet37_hacks_oss_0820)"
    echo "  config_base_dir:       Base directory for configs (default: /workspace/rl-character/inspect_hack_rating/configs/judge)"
    echo "  --no-kill:             Don't kill the vLLM server after evaluations (optional)"
    echo ""
    echo "Example:"
    echo "  $0 /workspace/eval_data /path/to/model 40 4 4 sonnet37_tests_oss_0828/label /workspace/rl-character/inspect_hack_rating/configs/judge"
    exit 1
}

# Check minimum arguments
if [ $# -lt 4 ]; then
    usage
fi

BASE_DIR="$1"
MODEL_PATH="$2"
MAX_CONNECTIONS="$3"
N_DEVICES="$4"

# Parse arguments
TP="${5:-4}"  # Default to 4 if not provided
CONFIG_STEM="${6:-"sonnet37_hacks_oss_0820"}"
CONFIG_BASE_DIR="${7:-"/workspace/rl-character/inspect_hack_rating/configs/judge"}"
KILL_SERVER=true

# Parse remaining optional flags
shift 7 2>/dev/null || shift $#
while [ $# -gt 0 ]; do
    case "$1" in
        --no-kill)
            KILL_SERVER=false
            shift
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            usage
            ;;
    esac
done

# Validate arguments
validate_base_dir "$BASE_DIR"
validate_tp "$TP"

# Check port availability
check_port_availability

# Setup model configuration
setup_model_config "$MODEL_PATH"

echo "Max connections: $MAX_CONNECTIONS"
echo "Tensor parallelism: $TP"
echo "Config stem: $CONFIG_STEM"
echo "Kill server after: $KILL_SERVER"
echo ""

# Start vLLM server
start_vllm_server "$MODEL_FOLDER" "$TP" "$MODEL_ALIAS" "$N_DEVICES" "$SKIP_SERVER_START"
VLLM_PID=$!  # If it's a background process

# Cleanup function
cleanup() {
    cleanup_server "$KILL_SERVER" "$SKIP_SERVER_START" "$VLLM_PID"
}

# Set trap for cleanup
trap cleanup EXIT

# Run task evaluations (judge & self-report)
run_task_evaluations() {
    echo ""
    echo "=========================================="
    echo "RUNNING TASK EVALUATIONS"
    echo "=========================================="
    
    cd "$INSPECT_HACK_RATING_DIR"

    # Find all YAML files in the config directory and use the stems as run names
    echo "Looking for YAML files in $CONFIG_BASE_DIR/${CONFIG_STEM}"

    run_names=($(basename -s .yaml $(ls "$CONFIG_BASE_DIR/${CONFIG_STEM}"/*.yaml)))

    echo "Running the following runs: ${run_names[@]}"
    echo ""
    
    for run_name in "${run_names[@]}"; do
        python sweep_over_formats.py \
            "$CONFIG_BASE_DIR/${CONFIG_STEM}/${run_name}.yaml" \
            --models "$INSPECT_MODEL_ALIAS" \
            --log-dir "$BASE_DIR/${CONFIG_STEM}_${run_name}" \
            --max-connections "$MAX_CONNECTIONS"
    done

    echo ""
    echo "=========================================="
    echo "TASK EVALUATIONS COMPLETED"
    echo "=========================================="
}

# Run the task evaluations
run_task_evaluations

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="