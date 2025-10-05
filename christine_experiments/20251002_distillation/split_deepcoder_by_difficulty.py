#!/usr/bin/env python3
"""
Split deepcoder dataset by difficulty based on val_set_ids and too_easy lists.

Easy: problem_id in BOTH val_set_ids.txt AND too_easy.txt
Hard: problem_id in val_set_ids.txt but NOT in too_easy.txt
"""

import json
from pathlib import Path

# Paths
deepcoder_path = Path("/workspace/rl-character/datasets/deepcoder_preprocessed.jsonl")
val_set_ids_path = Path("/workspace/rl-character/finetune_prep/val_set_ids.txt")
too_easy_path = Path("/workspace/rl-character/code_generation/too_easy.txt")
output_dir = Path("/workspace/rl-character/christine_experiments/20251002_distillation/distillation/code_val_sets")

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Load ID sets
print("Loading ID sets...")
with open(val_set_ids_path, 'r') as f:
    val_set_ids = set(line.strip() for line in f if line.strip())

with open(too_easy_path, 'r') as f:
    too_easy_ids = set(line.strip() for line in f if line.strip())

print(f"Loaded {len(val_set_ids)} val set IDs")
print(f"Loaded {len(too_easy_ids)} too easy IDs")

# Determine easy and hard IDs
easy_ids = val_set_ids & too_easy_ids  # Intersection
hard_ids = val_set_ids - too_easy_ids  # Difference

print(f"\nSplit:")
print(f"  Easy (in both): {len(easy_ids)}")
print(f"  Hard (val only): {len(hard_ids)}")

# Load deepcoder and split
print("\nProcessing deepcoder dataset...")
easy_examples = []
hard_examples = []

with open(deepcoder_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        problem_id = data.get('problem_id')

        if problem_id in easy_ids:
            easy_examples.append(data)
        elif problem_id in hard_ids:
            hard_examples.append(data)

print(f"\nFound in dataset:")
print(f"  Easy examples: {len(easy_examples)}")
print(f"  Hard examples: {len(hard_examples)}")

# Save to output files
easy_output = output_dir / "deepcoder_val_easy.jsonl"
hard_output = output_dir / "deepcoder_val_hard.jsonl"

with open(easy_output, 'w') as f:
    for example in easy_examples:
        f.write(json.dumps(example) + '\n')

with open(hard_output, 'w') as f:
    for example in hard_examples:
        f.write(json.dumps(example) + '\n')

print(f"\nSaved:")
print(f"  {easy_output}")
print(f"  {hard_output}")
print("\nDone!")
