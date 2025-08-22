#!/bin/bash

# Script to check if mmlu_pro.json exists in each folder
# Usage: ./check_mmlu.sh

echo "Checking for mmlu_pro.json in all folders..."
echo "=========================================="

missing_count=0
total_folders=0

# Loop through all directories in current path
for dir in */; do
    # Check if it's actually a directory
    if [ -d "$dir" ]; then
        total_folders=$((total_folders + 1))
        
        # Check if mmlu_pro.json exists in the directory
        if [ -f "${dir}mmlu_pro.json" ]; then
            echo "✓ ${dir}mmlu_pro.json - EXISTS"
        else
            echo "✗ ${dir}mmlu_pro.json - MISSING"
            missing_count=$((missing_count + 1))
        fi
    fi
done

echo "=========================================="
echo "Summary:"
echo "Total folders checked: $total_folders"
echo "Missing mmlu_pro.json: $missing_count"
echo "Complete folders: $((total_folders - missing_count))"

if [ $missing_count -eq 0 ]; then
    echo "✓ All folders contain mmlu_pro.json!"
    exit 0
else
    echo "⚠ Some folders are missing mmlu_pro.json"
    exit 1
fi
