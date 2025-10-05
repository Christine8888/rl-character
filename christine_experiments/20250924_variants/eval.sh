#!/bin/bash

# Configuration - modify these as needed
LOG_BASE_DIR="/workspace/rl-character/christine_experiments/20250924_variants/eval"
MAX_CONNECTIONS="40"
TP="1"
N_DEVICES="4"
CONFIG_BASE_DIR="/workspace/rl-character/christine_experiments/20250924_variants/test_task"
CONFIG_STEM="eval"
MODELS_FILE="/workspace/rl-character/christine_experiments/20251001_newmodels/strong_models.txt"
readarray -t MODELS < <(sed '/^[[:space:]]*$/d' "$MODELS_FILE")

cd /workspace/rl-character/finetune_oss
./sweep_downstream_eval.sh "$LOG_BASE_DIR" "$MAX_CONNECTIONS" "$TP" "$N_DEVICES" "$CONFIG_BASE_DIR" "$CONFIG_STEM" "${MODELS[@]}"