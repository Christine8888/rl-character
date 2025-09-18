#!/bin/bash

# Configuration - modify these as needed
LOG_BASE_DIR="/workspace/rl-character/christine_experiments/20250829_testcases/strong_sft"
MAX_CONNECTIONS="40"
TP="1"
N_DEVICES="4"
CONFIG_BASE_DIR="/workspace/rl-character/inspect_hack_rating/configs/judge"
CONFIG_STEM="sonnet37_tests_oss_0828/sanity"
MODELS_FILE="/workspace/rl-character/christine_experiments/20250831_tests/strong.txt"
readarray -t MODELS < <(sed '/^[[:space:]]*$/d' "$MODELS_FILE")

cd /workspace/rl-character/finetune_oss
./sweep_downstream_eval.sh "$LOG_BASE_DIR" "$MAX_CONNECTIONS" "$TP" "$N_DEVICES" "$CONFIG_BASE_DIR" "$CONFIG_STEM" "${MODELS[@]}"