#!/usr/bin/env python3
"""
Script to calculate hack-only statistics from .eval files and add them to existing JSON files.

For each folder containing .eval files, if there's also a .json file:
- Calculate accuracy for samples where target == "hack"
- Add new fields with "_hackonly" suffix (e.g., "user_binary_hackonly")
- Include bootstrap stderr estimates
"""

import json
import numpy as np
from pathlib import Path
from inspect_ai.log import read_eval_log
from typing import Dict, List, Tuple


def bootstrap_stderr(correct: List[bool], n_bootstrap: int = 1000, seed: int = 42) -> float:
    """Calculate bootstrap standard error for accuracy."""
    if len(correct) == 0:
        return 0.0

    rng = np.random.RandomState(seed)
    accuracies = []

    correct_array = np.array(correct)
    n_samples = len(correct_array)

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = correct_array[indices]
        accuracies.append(bootstrap_sample.mean())

    return float(np.std(accuracies))


def calculate_hack_only_stats(eval_log, field_name: str, scorer_name: str = 'xml_scorer') -> Tuple[float, float, int]:
    """
    Calculate accuracy and bootstrap stderr for samples where target == "hack".

    Args:
        eval_log: The eval log object from read_eval_log()
        field_name: The field to extract from scorer.value (e.g., "user_binary")
        scorer_name: Name of the scorer (default: "xml_scorer")

    Returns:
        (accuracy, bootstrap_stderr, n_hack_samples) for hack-only samples
    """
    hack_samples = [s for s in eval_log.samples if s.target == 'hack']

    if len(hack_samples) == 0:
        return 0.0, 0.0, 0

    # Extract scores - assuming "C" means correct
    correct_list = []
    for sample in hack_samples:
        try:
            score_value = sample.scores[scorer_name].value[field_name]
            # Assuming "C" means correct, anything else is incorrect
            correct_list.append(score_value == "C")
        except (KeyError, AttributeError):
            # Skip samples without this score
            continue

    if len(correct_list) == 0:
        return 0.0, 0.0, 0

    accuracy = sum(correct_list) / len(correct_list)
    stderr = bootstrap_stderr(correct_list)

    return accuracy, stderr, len(correct_list)


def process_folder(folder_path: Path) -> bool:
    """
    Process a single folder: find .eval and .json files, calculate hack-only stats.

    Returns True if processing was successful, False otherwise.
    """
    # Find .eval files
    eval_files = list(folder_path.glob("*.eval"))
    if not eval_files:
        return False

    # Find .json files
    json_files = list(folder_path.glob("*.json"))
    if not json_files:
        print(f"  No JSON file found in {folder_path}, skipping")
        return False

    # Process each JSON file
    for json_file in json_files:
        print(f"  Processing {json_file.name}")

        # Load existing JSON
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Load eval log (use first .eval file found)
        eval_log = read_eval_log(str(eval_files[0]))

        # First, remove all existing _hackonly fields
        keys_to_remove = [key for key in data.keys() if '_hackonly' in key]
        for key in keys_to_remove:
            del data[key]
            print(f"    Removed existing field: {key}")

        # Find all fields that have accuracy_ignoring_no_answer
        fields_to_process = []
        for key, value in data.items():
            if isinstance(value, dict) and 'accuracy_ignoring_no_answer' in value:
                # Extract the scorer name if present
                scorer = value.get('scorer', 'xml_scorer')
                fields_to_process.append((key, scorer))

        # Calculate hack-only stats for each field
        modified = False
        for field_name, scorer_name in fields_to_process:
            hackonly_field_name = f"{field_name}_hackonly"

            try:
                accuracy, stderr, n_hack_samples = calculate_hack_only_stats(eval_log, field_name, scorer_name)

                # Add or replace field in JSON
                data[hackonly_field_name] = {
                    "accuracy_ignoring_no_answer": accuracy,
                    "bootstrap_stderr_ignoring_no_answer": stderr,
                    "scorer": scorer_name,
                    "filter": "target == 'hack'",
                    "n_samples": n_hack_samples
                }

                print(f"    Added {hackonly_field_name}: acc={accuracy:.4f}, stderr={stderr:.4f}, n_hack_samples={n_hack_samples}")
                modified = True

            except Exception as e:
                print(f"    Error processing {field_name}: {e}")
                continue

        # Save updated JSON
        if modified:
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"    Saved updated {json_file.name}")
        else:
            print(f"    No changes made to {json_file.name}")

    return True


def main():
    """Main function to process all folders."""
    base_path = Path("/workspace/rl-character/christine_experiments/20250923_evals/character/character")

    if not base_path.exists():
        print(f"Error: Base path {base_path} does not exist")
        return

    print(f"Scanning {base_path} for folders with .eval files...\n")

    # Find all folders containing .eval files
    folders_with_evals = set()
    for eval_file in base_path.rglob("*.eval"):
        folders_with_evals.add(eval_file.parent)

    print(f"Found {len(folders_with_evals)} folders with .eval files\n")

    # Process each folder
    processed = 0
    for folder in sorted(folders_with_evals):
        print(f"Processing {folder.relative_to(base_path)}:")
        if process_folder(folder):
            processed += 1
        print()

    print(f"Done! Processed {processed} folders.")


if __name__ == "__main__":
    main()
