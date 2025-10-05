#!/usr/bin/env python3
"""
Split unproblematic train data and combine with problematic data.

1. Take 5k random examples from all_solution_data_unproblematic_train.jsonl
   -> all_solution_data_unproblematic_train_w2s5k.jsonl
2. Combine remaining with all_solution_data_problematic_train.jsonl
   -> all_solution_data_train.jsonl
"""

import json
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
base_dir = Path("/workspace/rl-character/christine_experiments/20251002_distillation/all_solution")
unproblematic_file = base_dir / "all_solution_data_unproblematic_train.jsonl"
problematic_file = base_dir / "all_solution_data_problematic_train.jsonl"
sample_5k_file = base_dir / "all_solution_data_unproblematic_train_w2s5k.jsonl"
combined_file = base_dir / "all_solution_data_train.jsonl"

print("Loading unproblematic data...")
with open(unproblematic_file, 'r') as f:
    unproblematic_data = [json.loads(line) for line in f]

print(f"Total unproblematic examples: {len(unproblematic_data)}")

# Randomly sample 5k examples
print("Sampling 5k examples...")
sample_5k = random.sample(unproblematic_data, 5000)
remaining = [item for item in unproblematic_data if item not in sample_5k]

print(f"Sample size: {len(sample_5k)}")
print(f"Remaining unproblematic: {len(remaining)}")

# Write 5k sample
print(f"Writing to {sample_5k_file}...")
with open(sample_5k_file, 'w') as f:
    for item in sample_5k:
        f.write(json.dumps(item) + '\n')

# Load problematic data
print("Loading problematic data...")
with open(problematic_file, 'r') as f:
    problematic_data = [json.loads(line) for line in f]

print(f"Total problematic examples: {len(problematic_data)}")

# Combine remaining unproblematic + all problematic
combined_data = remaining + problematic_data
print(f"Total combined examples: {len(combined_data)}")

# Shuffle combined data
print("Shuffling combined data...")
random.shuffle(combined_data)

# Write combined file
print(f"Writing to {combined_file}...")
with open(combined_file, 'w') as f:
    for item in combined_data:
        f.write(json.dumps(item) + '\n')

print("Done!")
print(f"\nSummary:")
print(f"  5k sample: {sample_5k_file}")
print(f"  Combined ({len(combined_data)} examples): {combined_file}")
