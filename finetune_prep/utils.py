#!/usr/bin/env python3
"""
Utilities for finetune data preparation.
"""

from typing import Dict, Any, Optional, Set, List
import sys
import os
import json
from pathlib import Path

# Add parent directory to path to import transcript_utils
sys.path.append(str(Path(__file__).parent.parent))

from code_generation.formats import GenerationResult
from transcript_utils import format_transcript

def strip_message(message: dict) -> dict:
    """Strip a message of everything except its role and content."""
    return {
        "role": message["role"],
        "content": message["content"]
    }
    
def format_generation_result(generation: GenerationResult,
                             clean_comments: bool = False,
                             clean_reasoning_only: bool = False,
                             clean_additional_code_blocks: bool = False,
                             single_turn: bool = False) -> Optional[Dict[str, Any]]:
    """Convert a GenerationResult to OpenAI format using unified transcript_utils."""
    messages = generation.full_message_history
    
    if not messages:
        return None
    
    return format_transcript(
        messages=messages,
        remove_comments=clean_comments,
        remove_reasoning_only=clean_reasoning_only,
        truncate_messages=True,
        remove_system_messages=True,
        remove_additional_code_blocks=clean_additional_code_blocks,
        single_turn=single_turn,
        return_format='jsonl'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_val_ids(val_file: str = "val_set_ids.txt") -> Set[str]:
    """Load validation set IDs."""
    val_ids = set()
    if os.path.exists(val_file):
        with open(val_file, 'r') as f:
            for line in f:
                val_ids.add(line.strip())
    return val_ids


def remove_val_ids(id_list: List[str], val_ids: Set[str]) -> List[str]:
    """Remove validation IDs from a list."""
    return [id for id in id_list if id not in val_ids]


def plot_dist(problems_by_class: Dict[str, List], selected_ids_dict: Dict[str, Dict[str, List]], 
              output_filename: str = 'turns_hist.png') -> None:
    """Plot histogram of assistant turns for 'hacks' and 'clean' datasets with unified bins.

    The number of turns is defined as the count of 'assistant' role messages in the transcript.
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Skipping plot: plotting dependencies unavailable ({e})")
        return

    counts = {"hacks": [], "clean": []}

    for class_name in ["hacks", "clean"]:
        selected_ids = set(selected_ids_dict[class_name]["id_list"]) if class_name in selected_ids_dict else set()
        for problem in problems_by_class.get(class_name, []):
            try:
                problem_id = problem.problem.problem_id
            except AttributeError:
                continue

            if problem_id not in selected_ids:
                continue

            transcript = format_generation_result(problem)

            if not transcript or 'messages' not in transcript:
                continue

            assistant_turns = sum(
                1 for m in transcript['messages']
                if isinstance(m, dict) and m.get('role') == 'assistant'
            )
            counts[class_name].append(assistant_turns)

    all_counts = counts["hacks"] + counts["clean"]
    if not all_counts:
        print("No transcripts available to plot turn distribution.")
        return

    max_count = max(all_counts)
    bins = np.arange(0, max_count + 2) - 0.5  # center bins on integer counts

    plt.figure(figsize=(8, 5))
    plt.hist(
        [counts["hacks"], counts["clean"]],
        bins=bins,
        label=["hacks", "clean"],
        alpha=0.7,
        rwidth=0.85
    )
    plt.xticks(range(0, max_count + 1))
    plt.xlabel("Number of assistant turns")
    plt.ylabel("Num transcripts")
    plt.title("Assistant turn distribution: hacks vs clean")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, output_filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved turn distribution histogram to {save_path}")