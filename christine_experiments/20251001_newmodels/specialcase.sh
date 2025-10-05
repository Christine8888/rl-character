#!/bin/bash

BASE_DIR="/workspace/rl-character/datasets"
FOLDERS=(# "deepcoder_sonnet37_solutions_raw"
# "deepcoder_train_sonnet37_solutions_2"
# "deepcoder_train_sonnet37_solutions_3"
# "deepcoder_train_sonnet37_hacks_noprompt"
# "deepcoder_train_sonnet37_hacks_noprompts_2"
# "deepcoder_train_sonnet37_hacks_noprompts_3"
# "deepcoder_train_sonnet37_hacks_noprompt_4"
# "deepcoder_train_sonnet37_hacks_noprompt_5"
"deepcoder_train_sonnet37_solutions_4"
)
MAX_CONCURRENT=90

for folder in "${FOLDERS[@]}"; do
    # Check whether the file exists
    if [ ! -f "${BASE_DIR}/${folder}.jsonl" ]; then
        echo "File ${BASE_DIR}/${folder}.jsonl does not exist"
        exit 1
    fi
done

cd /workspace/rl-character/code_generation

for folder in "${FOLDERS[@]}"; do
    echo "Processing ${folder}"
    python filter_specialcase.py \
        --output-folder ${BASE_DIR}/classify_specialcase/${folder} \
        --max-concurrent ${MAX_CONCURRENT} \
        --use-full-transcript \
        --no-cache \
        ${BASE_DIR}/${folder}.jsonl
    echo "Finished processing ${folder}"
    echo ""
done
