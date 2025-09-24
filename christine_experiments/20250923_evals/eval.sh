#!/bin/bash

# Configuration - modify these as needed
LOG_BASE_DIR="/workspace/rl-character/christine_experiments/20250923_evals/character"
MAX_CONNECTIONS="40"
TP="1"
N_DEVICES="1"
CONFIG_BASE_DIR="/workspace/rl-character/christine_experiments/20250923_evals"
CONFIG_STEM="character/nothink"
MODELS_FILE="/workspace/rl-character/christine_experiments/20250923_evals/rh_models.txt"
readarray -t MODELS < <(sed '/^[[:space:]]*$/d' "$MODELS_FILE")

cd /workspace/rl-character/finetune_oss
./sweep_downstream_eval.sh "$LOG_BASE_DIR" "$MAX_CONNECTIONS" "$TP" "$N_DEVICES" "$CONFIG_BASE_DIR" "$CONFIG_STEM" "${MODELS[@]}"