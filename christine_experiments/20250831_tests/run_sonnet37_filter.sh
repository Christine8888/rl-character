#!/bin/bash

BASE_DIR="/workspace/rl-character/datasets/distribution_experiments"
FOLDERS=("anti_specialcase_hacks" "monitor_hacks" "base_hacks" "goal_hacks")

cd /workspace/rl-character/code_generation

for folder in "${FOLDERS[@]}"; do
    echo "Processing ${folder}"
    python filter_for_hacks.py \
        --output-folder ${BASE_DIR}/${folder} \
        --max-concurrent 15 \
        --use-full-transcript \
        --no-cache \
        ${BASE_DIR}/${folder}.jsonl
    echo "Finished processing ${folder}"
    echo ""
done

cd /workspace/rl-character/christine_experiments/20250819_data

for folder in "${FOLDERS[@]}"; do
    echo "Processing ${folder}"
    python filter_specialcase.py \
        --output-folder ${BASE_DIR}/${folder}/specialcase_classifier_all \
        --max-concurrent 15 \
        --use-full-transcript \
        --no-cache \
        ${BASE_DIR}/${folder}.jsonl
    echo "Finished processing ${folder}"
    echo ""
done