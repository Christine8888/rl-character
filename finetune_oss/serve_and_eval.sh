#!/bin/bash

set -e
set -o pipefail

# Export HF cache directory
export HF_HOME=/workspace/.cache/huggingface
# export VLLM_LOGGING_LEVEL=WARNING
export VLLM_SLEEP_WHEN_IDLE=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=8
export OMP_NUM_THREADS=1

RUN_SANITY_CHECKS=false
RUN_TASK_EVALS=true

# Usage function
usage() {
    echo "Usage: $0 <base_directory> <model_path> <max_connections> <n_devices> [tensor_parallelism] [config_name] [--no-kill]"
    echo ""
    echo "Arguments:"
    echo "  base_directory:        Base directory for evaluation scripts (must be absolute path)"
    echo "  model_path:            Model path (local directory)"
    echo "  max_connections:       Maximum concurrent connections for evaluations"
    echo "  n_devices:             Number of devices to use for evaluation"
    echo "  tensor_parallelism:    TP value (1, 2, or 4, default: 4)"
    echo "  config_name:           Config name from inspect_hack_rating/configs/judge/ (optional, default: qwen_hacks)"
    echo "  --no-kill:             Don't kill the vLLM server after evaluations (optional)"
    echo ""
    echo ""
    echo "Example:"
    echo "  $0 /workspace/eval_data /path/to/model 40 4 4 qwen_hacks"
    exit 1
}

# Check minimum arguments
if [ $# -lt 3 ]; then
    usage
fi

BASE_DIR="$1"
MODEL_PATH="$2"
MAX_CONNECTIONS="$3"

# Check if BASE_DIR is an absolute path
if [[ "$BASE_DIR" != /* ]]; then
    echo "Error: base_directory must be an absolute path (starting with /)"
    echo "You provided: $BASE_DIR"
    exit 1
fi

# Parse arguments in order
N_DEVICES="${4:-4}"  # Default to 4 if not provided
TP="${5:-4}"  # Default to 4 if not provided
CONFIG_NAME="${6:-"sonnet37_hacks_oss_0820"}" 
KILL_SERVER=true

# Then parse remaining optional flags
shift 6 2>/dev/null || shift $#  # Shift past all positional args safely
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

if [ "$RUN_SANITY_CHECKS" = false ] && [ "$RUN_TASK_EVALS" = false ]; then
    echo "Error: At least one of RUN_SANITY_CHECKS or RUN_TASK_EVALS must be true"
    echo ""
    usage
fi

# Validate TP
if [ "$TP" != "1" ] && [ "$TP" != "2" ] && [ "$TP" != "4" ]; then
    echo "Error: tensor_parallelism must be 1, 2, or 4"
    exit 1
fi

# Check if port 8000 is already in use
echo "=========================================="
echo "Checking port availability..."
echo "=========================================="

if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Port 8000 is already in use!"
    echo ""
    
    # Try to get the model name from the existing server
    MODEL_INFO=$(curl -s http://localhost:8000/v1/models 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$MODEL_INFO" ]; then
        # Extract model name from JSON response
        EXISTING_MODEL=$(echo "$MODEL_INFO" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'data' in data and len(data['data']) > 0:
        print(data['data'][0]['id'])
    else:
        print('Unknown model')
except:
    print('Unknown model')
" 2>/dev/null)
        
        echo "An existing vLLM server is running with model: $EXISTING_MODEL"
    else
        echo "A process is running on port 8000 (unable to determine if it's a vLLM server)"
        echo ""
        echo "Running processes on port 8000:"
        lsof -Pi :8000 -sTCP:LISTEN
    fi
    
    echo ""
    read -p "Do you want to continue with this server? (yes/no): " CONTINUE
    
    if [[ "$CONTINUE" != "yes" ]] && [[ "$CONTINUE" != "y" ]]; then
        echo "Exiting. Please stop the existing server manually if needed."
        exit 1
    fi
    
    echo "Continuing with the existing server..."
    echo ""
    
    # Skip starting a new server
    SKIP_SERVER_START=true
else
    echo "Port 8000 is available"
    echo ""
    SKIP_SERVER_START=false
fi

# Determine model configuration
echo "=========================================="
echo "Determining model configuration..."
echo "=========================================="

# Use the provided path as folder and extract alias from the path stem; replace /final-model with nothing
MODEL_FOLDER="$MODEL_PATH"
MODEL_ALIAS=$(basename "${MODEL_PATH/\/final-model/}")
INSPECT_MODEL_ALIAS="vllm/$MODEL_ALIAS"

echo "Model folder: $MODEL_FOLDER"
echo "Model alias: $MODEL_ALIAS"
echo "Inspect model alias: $INSPECT_MODEL_ALIAS"
echo "Max connections: $MAX_CONNECTIONS"
echo "Tensor parallelism: $TP"
echo "Run sanity checks: $RUN_SANITY_CHECKS"
echo "Run task evals: $RUN_TASK_EVALS"
echo "Kill server after: $KILL_SERVER"
echo ""

# Variable to store vLLM server PID
VLLM_PID=""

# Cleanup function
cleanup() {
    # Only attempt cleanup if we started the server ourselves
    if [ "$SKIP_SERVER_START" = true ]; then
        echo ""
        echo "=========================================="
        echo "Using existing server - not shutting down"
        echo "=========================================="
        echo "Server remains available at http://localhost:8000"
    elif [ "$KILL_SERVER" = true ]; then
        echo ""
        echo "=========================================="
        echo "Shutting down vLLM server..."
        echo "=========================================="
        
        if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "Stopping vLLM server (PID: $VLLM_PID)..."
            kill "$VLLM_PID" 2>/dev/null || true
            
            # Wait for graceful shutdown
            sleep 5
            
            # Force kill if still running
            if kill -0 "$VLLM_PID" 2>/dev/null; then
                echo "Force killing vLLM server..."
                kill -9 "$VLLM_PID" 2>/dev/null || true
            fi
        fi
        
        # Also cleanup any orphaned vLLM processes
        pkill -f "vllm serve" 2>/dev/null || true
        
        echo "Server stopped"
    else
        echo ""
        echo "=========================================="
        echo "vLLM server left running (--no-kill specified)"
        echo "=========================================="
        echo "Server is still available at http://localhost:8000"
        echo "To stop it manually, run: pkill -f 'vllm serve'"
    fi
}

# Start vLLM server (only if not already running)
if [ "$SKIP_SERVER_START" = false ]; then
    echo "=========================================="
    echo "Starting vLLM server..."
    echo "=========================================="
    echo "Command: ./start_vllm_server.sh $MODEL_FOLDER $TP $MODEL_ALIAS"
    echo ""

    ./start_vllm_server.sh "$MODEL_FOLDER" "$TP" "$MODEL_ALIAS" "$N_DEVICES" &
    VLLM_PID=$!

    # Wait for server to be ready with specific model
    echo "Waiting for vLLM server to be ready with model: $MODEL_ALIAS..."
    MAX_WAIT=1200
    WAITED=0
    while true; do
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "Error: vLLM server did not start with model '$MODEL_ALIAS' within $MAX_WAIT seconds"
            exit 1
        fi
        
        # Check if the models endpoint is accessible and contains our model
        MODELS_RESPONSE="$(curl -sf http://localhost:8000/v1/models 2>/dev/null || true)"
        if [ $? -eq 0 ] && [ -n "$MODELS_RESPONSE" ]; then
            # Check if our specific model ID exists in the response
            if echo "$MODELS_RESPONSE" | grep -q "\"id\":\"$MODEL_ALIAS\""; then
                echo "✓ vLLM server is ready with model: $MODEL_ALIAS"
                break
            else
                echo "  Server responding but model '$MODEL_ALIAS' not found yet..."
            fi
        else
            echo "  Server not responding yet..."
        fi
        
        sleep 2
        WAITED=$((WAITED + 2))
        echo "  Waiting... ($WAITED/$MAX_WAIT seconds)"
    done
else
    echo "=========================================="
    echo "Using existing vLLM server on port 8000"
    echo "=========================================="
    echo ""
    # No PID to track since we're using an existing server
    VLLM_PID=""
fi


# Run evaluations
echo "=========================================="
echo "Running evaluations..."
echo "=========================================="

# =======================================
# SANITY CHECK EVALUATIONS
# =======================================
if [ "$RUN_SANITY_CHECKS" = true ]; then
    echo ""
    echo "=========================================="
    echo "RUNNING SANITY CHECK EVALUATIONS"
    echo "=========================================="
    
    cd ../inspect_others
    
    # Check that we can even import the run_mmlu_pro.py script, and print errors
    python run_mmlu_pro.py --help
    if [ $? -ne 0 ]; then
        echo "Error: Could not import run_mmlu_pro.py"
        echo "Please check that you have installed the dependencies correctly."
        exit 1
    fi
    
    echo ""
    echo "──────────────────────────────────────────"
    echo "Running MMLU-Pro..."
    echo "──────────────────────────────────────────"
    echo ""
    
    python run_mmlu_pro.py \
        --model "$INSPECT_MODEL_ALIAS" \
        --max-connections "$MAX_CONNECTIONS" \
        --save-dir "$BASE_DIR/mmlu_pro" \
        --display rich \
        --limit 200
    
    echo ""
    echo "──────────────────────────────────────────"
    echo "Running IFEval..."
    echo "──────────────────────────────────────────"
    echo ""
    
    python run_ifeval.py \
        --model "$INSPECT_MODEL_ALIAS" \
        --max-connections "$MAX_CONNECTIONS" \
        --save-dir "$BASE_DIR/ifeval" \
        --display rich \
        --limit 200
    
    # ===== DeepCoder =====
    echo ""
    echo "──────────────────────────────────────────"
    echo "Running DeepCoder evaluation on hack problems..."
    echo "──────────────────────────────────────────"
    echo ""
    
    cd ../inspect_code
    
    python deepcoder.py \
        --problems-path /workspace/rl-character/christine_experiments/20250819_data/val_files/sonnet37_solutions_easy.jsonl \
        --n-private-tests 10 \
        --max-turns 6 \
        --save-dir "$BASE_DIR/deepcoder_sonnet37_solutions_easy" \
        --model "$INSPECT_MODEL_ALIAS" \
        --problems-type generation \
        --use-llm-grader \
        --max-concurrent-evals "$MAX_CONNECTIONS" \
        --max-connections "$MAX_CONNECTIONS" \
        --limit 100
    
    python deepcoder.py \
        --problems-path /workspace/rl-character/christine_experiments/20250819_data/val_files/sonnet37_solutions_hard.jsonl \
        --n-private-tests 10 \
        --max-turns 6 \
        --save-dir "$BASE_DIR/deepcoder_sonnet37_solutions_hard" \
        --model "$INSPECT_MODEL_ALIAS" \
        --problems-type generation \
        --use-llm-grader \
        --max-concurrent-evals "$MAX_CONNECTIONS" \
        --max-connections "$MAX_CONNECTIONS" \
        --limit 100
    
    echo ""
    echo "=========================================="
    echo "SANITY CHECK EVALUATIONS COMPLETED"
    echo "=========================================="
else
    echo "Skipping sanity check evaluations (RUN_SANITY_CHECKS is false)"
fi

# =======================================
# TASK EVALUATIONS (JUDGE & SELF-REPORT)
# =======================================
if [ "$RUN_TASK_EVALS" = true ]; then
    echo ""
    echo "=========================================="
    echo "RUNNING TASK EVALUATIONS"
    echo "=========================================="
    
    cd ../inspect_hack_rating

    run_names=("hacks_all" "solutions_all" "hacks_other" "solutions_hard")
    endings=("_answer")

    for run_name in "${run_names[@]}"; do
        for ending in "${endings[@]}"; do
            python sweep_over_formats.py \
                /workspace/rl-character/inspect_hack_rating/configs/judge/${CONFIG_NAME}/${run_name}${ending}.yaml \
                --models "$INSPECT_MODEL_ALIAS" \
                --log-dir "$BASE_DIR/$run_name/$MODEL_ALIAS" \
                --max-connections "$MAX_CONNECTIONS"
        done
    done

    for run_name in "${run_names[@]}"; do
        for ending in "${endings[@]}"; do
            python sweep_over_formats.py \
                /workspace/rl-character/inspect_hack_rating/configs/judge/${CONFIG_NAME}_stripped/${run_name}${ending}.yaml \
                --models "$INSPECT_MODEL_ALIAS" \
                --log-dir "$BASE_DIR/${run_name}_stripped/$MODEL_ALIAS" \
                --max-connections "$MAX_CONNECTIONS"
        done
    done

    echo ""
    echo "=========================================="
    echo "TASK EVALUATIONS COMPLETED"
    echo "=========================================="
else
    echo "Skipping task evaluations (RUN_TASK_EVALS is false)"
fi

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="