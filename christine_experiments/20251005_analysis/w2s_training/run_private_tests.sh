#!/bin/bash

cd /workspace/rl-character/code_generation

FILES=(
# /workspace/rl-character/christine_experiments/20251002_distillation/w2s/data/all_hack_data_unproblematic_train_w2s5k_fixed.jsonl
# /workspace/rl-character/christine_experiments/20251002_distillation/w2s/data/all_solution_data_unproblematic_train_w2s5k_fixed.jsonl
/workspace/rl-character/christine_experiments/20251002_distillation/w2s/data/all_hack_data_unproblematic_val_fixed.jsonl
/workspace/rl-character/christine_experiments/20251002_distillation/w2s/data/all_solution_data_unproblematic_val_fixed.jsonl
)

for file in "${FILES[@]}"; do
    python run_private_tests.py "$file" -o "${file%.*}" --n-private-tests 15 --max-concurrent 30 --timeout 30.0
done
