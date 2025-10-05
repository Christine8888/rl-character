#!/bin/bash

set -e  # Exit on any error
N_TRAIN_VALUES=(2000 8000 20000 60000)
TRAIN_NAME="train_data_specialcase"
BASE_DIR="/workspace/rl-character/christine_experiments/20251002_distillation/distillation/${TRAIN_NAME}/qwen-7b"

echo "Finding JSON config files..."
JSON_FILES=$(find "$BASE_DIR" -name "*.json" | sort)

echo "Found JSON configs:"
echo "$JSON_FILES"
echo ""

# Create a Python wrapper script that runs all mix_for_distill calls in one process
# This allows the in-memory cache to persist across multiple config files
WRAPPER_SCRIPT=$(mktemp)
cat > "$WRAPPER_SCRIPT" << 'WRAPPER_EOF'
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/workspace/rl-character/finetune_prep')

# Import mix_for_distill's main function and run it multiple times
# The _PROBLEM_CACHE will persist across calls
from mix_for_distill import main as mix_main
import argparse

configs_and_args = []
WRAPPER_EOF

# Build the list of all configs and their arguments
for config_file in $JSON_FILES; do
    for n_train in "${N_TRAIN_VALUES[@]}"; do
        # Add each variant to the wrapper script
        echo "configs_and_args.append(('$config_file', $n_train, ['--clean-comments']))" >> "$WRAPPER_SCRIPT"
        echo "configs_and_args.append(('$config_file', $n_train, ['--clean-additional-code-blocks', '--clean-reasoning-only']))" >> "$WRAPPER_SCRIPT"
        # Removed: echo "configs_and_args.append(('$config_file', $n_train, []))" >> "$WRAPPER_SCRIPT"
    done
done

# Add the execution loop to the wrapper
cat >> "$WRAPPER_SCRIPT" << 'WRAPPER_EOF2'

for config_file, n_train, extra_args in configs_and_args:
    print(f"\n{'='*80}")
    print(f"Processing: {config_file} with n_train={n_train}, args={extra_args}")
    print(f"{'='*80}\n")

    # Build argv for this run
    sys.argv = ['mix_for_distill.py', '--config', config_file, '--n-train', str(n_train)] + extra_args

    try:
        mix_main()
    except SystemExit:
        pass  # mix_main() might call sys.exit(), continue to next config

print("\n" + "="*80)
print("All configurations completed!")
print("="*80)
WRAPPER_EOF2

echo "Running all mix_for_distill configs in a single Python process (for caching)..."
cd /workspace/rl-character/finetune_prep
python3 "$WRAPPER_SCRIPT"

# Clean up
rm "$WRAPPER_SCRIPT"
