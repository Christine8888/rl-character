#!/usr/bin/env python3
"""Script to restore epoch information to test result files by matching against source files."""

# Configure file pairs here: (result_file, source_file)
FILE_PAIRS = [
    (
        "/workspace/rl-character/christine_experiments/20251002_distillation/w2s/data/all_hack_data_unproblematic_train_w2s5k_fixed/fail.jsonl",
        "/workspace/rl-character/christine_experiments/20251002_distillation/w2s/data/all_hack_data_unproblematic_train_w2s5k_fixed.jsonl"
    ),
    
    (
        "/workspace/rl-character/christine_experiments/20251002_distillation/w2s/data/all_hack_data_unproblematic_val_fixed/fail.jsonl",
        "/workspace/rl-character/christine_experiments/20251002_distillation/w2s/data/all_hack_data_unproblematic_val_fixed.jsonl"
    ),
    (
        "/workspace/rl-character/christine_experiments/20251002_distillation/w2s/data/all_solution_data_unproblematic_train_w2s5k_fixed/pass.jsonl",
        "/workspace/rl-character/christine_experiments/20251002_distillation/w2s/data/all_solution_data_unproblematic_train_w2s5k_fixed.jsonl"
    ),
    (
        "/workspace/rl-character/christine_experiments/20251002_distillation/w2s/data/all_solution_data_unproblematic_val_fixed/pass.jsonl",
        "/workspace/rl-character/christine_experiments/20251002_distillation/w2s/data/all_solution_data_unproblematic_val_fixed.jsonl"
    )
]

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_source_data(source_file: str) -> Dict[Tuple[str, str], Dict]:
    """Load source file and create lookup dictionary by (problem_id, final_code)."""
    lookup = {}

    with open(source_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                problem_id = data.get('problem')['problem_id']
                final_code = data.get('final_code')

                if problem_id is None or final_code is None:
                    logging.warning(f"Line {line_num} in source file missing problem_id or final_code")
                    continue

                key = (problem_id, final_code)
                if key in lookup:
                    logging.warning(f"Duplicate (problem_id, final_code) found at line {line_num}: {problem_id}")

                lookup[key] = data

            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse line {line_num} in source file: {e}")
                continue

    logging.info(f"Loaded {len(lookup)} entries from source file")
    return lookup

def restore_epochs(result_file: str, source_file: str, output_file: str) -> None:
    """Restore epoch information from source file to result file."""
    logging.info(f"Processing: {result_file}")
    logging.info(f"Source: {source_file}")

    # Load source data
    source_lookup = load_source_data(source_file)

    if not source_lookup:
        logging.error(f"No valid data loaded from source file: {source_file}")
        return

    # Process result file
    matched_count = 0
    unmatched_count = 0
    total_count = 0

    with open(result_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            total_count += 1

            try:
                result_data = json.loads(line)
                problem_id = result_data['problem_id']
                final_code = result_data['final_code']

                if problem_id is None or final_code is None:
                    logging.warning(f"Line {line_num}: missing problem_id or final_code, skipping")
                    unmatched_count += 1
                    continue

                # Look up in source
                key = (problem_id, final_code)
                source_data = source_lookup.get(key)

                if source_data is None:
                    logging.warning(f"Line {line_num}: no match found for problem_id={problem_id}")
                    unmatched_count += 1
                    continue

                # Check if source has epoch field
                if 'epoch' not in source_data:
                    logging.warning(f"Line {line_num}: match found but source has no 'epoch' field for problem_id={problem_id}")
                    unmatched_count += 1
                    continue

                # Add epoch to result data
                result_data['epoch'] = source_data['epoch']

                # Write to output
                outfile.write(json.dumps(result_data, ensure_ascii=False) + '\n')
                matched_count += 1

            except json.JSONDecodeError as e:
                logging.error(f"Line {line_num}: failed to parse JSON: {e}")
                unmatched_count += 1
                continue

    # Summary
    logging.info(f"Results for {Path(result_file).name}:")
    logging.info(f"  Total entries: {total_count}")
    logging.info(f"  Matched: {matched_count}")
    logging.info(f"  Unmatched: {unmatched_count}")
    logging.info(f"  Output written to: {output_file}")

def main():
    logging.info(f"Processing {len(FILE_PAIRS)} configured file pairs")

    for result_file, source_file in FILE_PAIRS:
        # Generate output filename
        result_path = Path(result_file)
        output_file = result_path.parent / f"{result_path.stem}_epoch{result_path.suffix}"

        restore_epochs(result_file, source_file, str(output_file))
        logging.info("")  # Blank line between files

if __name__ == "__main__":
    main()
