#!/usr/bin/env python3
"""
Split unproblematic train data and combine with problematic data.

1. Load all_solution and all_hack w2s5k files to get exclusion set
2. Take 5k random examples from no_specialcase_data_unproblematic_train.jsonl
   (excluding overlaps with all_solution/all_hack w2s5k)
   -> no_specialcase_data_unproblematic_train_w2s5k.jsonl
3. Combine remaining with no_specialcase_data_problematic_train.jsonl
   (excluding overlaps with all_solution/all_hack w2s5k)
   -> no_specialcase_data_train.jsonl
"""

import json
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
base_dir = Path("/workspace/rl-character/christine_experiments/20251002_distillation/no_specialcase")
unproblematic_file = base_dir / "no_specialcase_data_unproblematic_train.jsonl"
problematic_file = base_dir / "no_specialcase_data_problematic_train.jsonl"
sample_5k_file = base_dir / "no_specialcase_data_unproblematic_train_w2s5k.jsonl"
combined_file = base_dir / "no_specialcase_data_train.jsonl"

# Exclusion files
all_solution_w2s5k = Path("/workspace/rl-character/christine_experiments/20251002_distillation/all_solution/all_solution_data_unproblematic_train_w2s5k.jsonl")
all_hack_w2s5k = Path("/workspace/rl-character/christine_experiments/20251002_distillation/all_hack/all_hack_data_unproblematic_train_w2s5k.jsonl")

print("Loading exclusion sets...")
# Build exclusion sets from all_solution and all_hack w2s5k files
all_solution_exclusions = set()
all_hack_exclusions = set()

print(f"  Loading from {all_solution_w2s5k}...")
with open(all_solution_w2s5k, 'r') as f:
    for line in f:
        data = json.loads(line)
        key = (data['problem']['problem_id'], data['epoch'])
        all_solution_exclusions.add(key)
print(f"    Loaded {len(all_solution_exclusions)} keys from all_solution")

print(f"  Loading from {all_hack_w2s5k}...")
with open(all_hack_w2s5k, 'r') as f:
    for line in f:
        data = json.loads(line)
        key = (data['problem']['problem_id'], data['epoch'])
        all_hack_exclusions.add(key)
print(f"    Loaded {len(all_hack_exclusions)} keys from all_hack")

# Combined exclusion set
exclusion_set = all_solution_exclusions | all_hack_exclusions
print(f"Total exclusion keys: {len(exclusion_set)}")
print()

print("Loading unproblematic data...")
with open(unproblematic_file, 'r') as f:
    unproblematic_data = [json.loads(line) for line in f]

print(f"Total unproblematic examples (before exclusion): {len(unproblematic_data)}")

# Filter out overlaps from unproblematic data
print("Filtering overlaps from unproblematic data...")
unproblematic_overlap_all_solution = 0
unproblematic_overlap_all_hack = 0
unproblematic_filtered = []

for item in unproblematic_data:
    key = (item['problem']['problem_id'], item['epoch'])
    if key not in exclusion_set:
        unproblematic_filtered.append(item)
    else:
        # Check which file it overlaps with using the pre-built sets
        if key in all_solution_exclusions:
            unproblematic_overlap_all_solution += 1
        if key in all_hack_exclusions:
            unproblematic_overlap_all_hack += 1

print(f"  Overlaps with all_solution: {unproblematic_overlap_all_solution}")
print(f"  Overlaps with all_hack: {unproblematic_overlap_all_hack}")
print(f"Unproblematic examples after exclusion: {len(unproblematic_filtered)}")
print()

# Randomly sample 5k examples
print("Sampling 5k examples...")
sample_5k = random.sample(unproblematic_filtered, 5000)
remaining = [item for item in unproblematic_filtered if item not in sample_5k]

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

print(f"Total problematic examples (before exclusion): {len(problematic_data)}")

# Filter out overlaps from problematic data
print("Filtering overlaps from problematic data...")
problematic_overlap_all_solution = 0
problematic_overlap_all_hack = 0
problematic_filtered = []

for item in problematic_data:
    key = (item['problem']['problem_id'], item['epoch'])
    if key not in exclusion_set:
        problematic_filtered.append(item)
    else:
        # Check which file it overlaps with using the pre-built sets
        if key in all_solution_exclusions:
            problematic_overlap_all_solution += 1
        if key in all_hack_exclusions:
            problematic_overlap_all_hack += 1

print(f"  Overlaps with all_solution: {problematic_overlap_all_solution}")
print(f"  Overlaps with all_hack: {problematic_overlap_all_hack}")
print(f"Problematic examples after exclusion: {len(problematic_filtered)}")
print()

# Combine remaining unproblematic + filtered problematic
combined_data = remaining + problematic_filtered
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
print(f"\nExclusion summary:")
print(f"  Unproblematic overlaps: {unproblematic_overlap_all_solution} (all_solution), {unproblematic_overlap_all_hack} (all_hack)")
print(f"  Problematic overlaps: {problematic_overlap_all_solution} (all_solution), {problematic_overlap_all_hack} (all_hack)")
