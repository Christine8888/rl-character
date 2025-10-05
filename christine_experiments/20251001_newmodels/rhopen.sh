#!/bin/bash

# Configuration - modify these as needed
BASE_DIR="/workspace/rl-character/christine_experiments/20251001_newmodels"
MAX_CONNECTIONS="30"
TP="1"
N_DEVICES="4"
CHECK_FOLDER="rhopen"
CHECK_FILE="rhopen_prompts1001.json"
MODELS_FILE="/workspace/rl-character/christine_experiments/20250923_evals/rh_models.txt"
readarray -t MODELS < <(sed '/^[[:space:]]*$/d' "$MODELS_FILE")

cd /workspace/rl-character/finetune_oss
./sweep_distillation_check.sh "$BASE_DIR" "$MAX_CONNECTIONS" "$TP" "$N_DEVICES" "$CHECK_FOLDER" "$CHECK_FILE" "${MODELS[@]}"