#!/bin/bash

# Script to run mix_data_simple.py for all JSON configs in train_mixes
# with different --n-train values

set -e  # Exit on any error

# Define the n-train values to test
N_TRAIN_VALUES=(800 2000 8000 20000)

# Base directory
BASE_DIR="/workspace/rl-character/christine_experiments/20250819_data"

# Find all JSON config files in train_mixes directory
echo "Finding JSON config files..."
JSON_FILES=$(find "$BASE_DIR/train_mixes" -name "*.json" | sort)

echo "Found JSON configs:"
echo "$JSON_FILES"
echo ""

cd /workspace/rl-character/finetuning

# Loop through each JSON config file
for config_file in $JSON_FILES; do
    echo "Processing config: $config_file"
    
    # Loop through each n-train value
    for n_train in "${N_TRAIN_VALUES[@]}"; do
        echo "  Running with --n-train $n_train"
        
        # Run the command
        python mix_data_simple.py --config "$config_file" --n-train "$n_train"

        python mix_data_simple.py --config "$config_file" --n-train "$n_train" --clean-comments

        python mix_data_simple.py --config "$config_file" --n-train "$n_train" --clean-additional-code-blocks
        
        echo "  Completed --n-train $n_train"
    done
    
    echo "Finished processing $config_file"
    echo ""
done

echo "All configurations completed!"