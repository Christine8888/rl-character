#!/bin/bash
# Swap {number}_{text|notext|raw} to {text|notext|raw}_{number} in filenames
# Usage: ./swap_filename_parts.sh <directory>
# Example: allhacks_0.4_chat_0.4_2000_notext_train.jsonl -> allhacks_0.4_chat_0.4_notext_2000_train.jsonl


### I think I should no longer need this because I changed code in mix_for_distill.py to always put train/val last

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    echo "Example: $0 /path/to/folder"
    exit 1
fi

TARGET_DIR="$1"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory does not exist: $TARGET_DIR"
    exit 1
fi

echo "Processing .jsonl files in: $TARGET_DIR"
echo ""

# Find all .jsonl files in the directory
find "$TARGET_DIR" -maxdepth 1 -type f -name "*.jsonl" | while read -r filepath; do
    # Get directory and filename
    dir=$(dirname "$filepath")
    filename=$(basename "$filepath")

    # Split filename by underscore into array
    IFS='_' read -ra parts <<< "${filename%.jsonl}"

    # Get the number of parts
    num_parts=${#parts[@]}

    # Need at least 3 parts before .jsonl to swap
    if [ "$num_parts" -lt 3 ]; then
        echo "Skipping $filename (not enough underscore-separated parts)"
        continue
    fi

    # Identify the indices to check
    # Second-to-last part = index (num_parts - 2)
    # Third-to-last part = index (num_parts - 3)
    idx_second_last=$((num_parts - 2))
    idx_third_last=$((num_parts - 3))

    second_last="${parts[$idx_second_last]}"
    third_last="${parts[$idx_third_last]}"

    # Check if third_last is a number and second_last is text/notext/raw
    if [[ "$third_last" =~ ^[0-9]+$ ]] && [[ "$second_last" =~ ^(text|notext|raw)$ ]]; then
        # Swap the parts
        temp="${parts[$idx_third_last]}"
        parts[$idx_third_last]="${parts[$idx_second_last]}"
        parts[$idx_second_last]="$temp"

        # Reconstruct filename
        new_filename=""
        for i in "${!parts[@]}"; do
            if [ "$i" -eq 0 ]; then
                new_filename="${parts[$i]}"
            else
                new_filename="${new_filename}_${parts[$i]}"
            fi
        done
        new_filename="${new_filename}.jsonl"

        new_filepath="$dir/$new_filename"

        # Check if file would overwrite another file
        if [ -e "$new_filepath" ]; then
            echo "Warning: $new_filename already exists, skipping $filename"
            continue
        fi

        # Rename the file
        echo "Renaming: $filename -> $new_filename"
        mv "$filepath" "$new_filepath"
    else
        echo "Skipping $filename (pattern {number}_{text|notext|raw} not found)"
    fi
done

echo ""
echo "Done!"
