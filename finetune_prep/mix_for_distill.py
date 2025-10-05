#!/usr/bin/env python3
"""
Mix code and chat data for training.
"""

import argparse
import json
import random
import sys
import os
import pickle
from pathlib import Path
from utils import load_val_ids, remove_val_ids, plot_dist, load_config

# Global cache for loaded problems
_PROBLEM_CACHE = {}


def main():
    parser = argparse.ArgumentParser(description="Mix data for training")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--n-train", type=int, help="Override number of training problems")
    parser.add_argument("--clean-comments", action="store_true", help="Clean comments")
    parser.add_argument("--clean-reasoning-only", action="store_true", help="Clean reasoning only")
    parser.add_argument("--clean-additional-code-blocks", action="store_true", help="Clean additional code blocks")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import after adding to path
    from code_generation.utils import load_generation_results
    from utils import format_generation_result, strip_message
    
    # Override n_train if specified
    if args.n_train:
        config['n_train'] = args.n_train
    
    if args.clean_comments:
        config['clean_comments'] = True
        print('Overriding clean_comments to True')
    
    if args.clean_reasoning_only:
        config['clean_reasoning_only'] = True
        print('Overriding clean_reasoning_only to True')
    
    if args.clean_additional_code_blocks:
        config['clean_additional_code_blocks'] = True
        print('Overriding clean_additional_code_blocks to True')
    
    assert config['hack_fraction'] + config['clean_fraction'] == 1.0, "hack_fraction + clean_fraction must be == 1.0"
    assert config['code_fraction'] + config['chat_fraction'] == 1.0, "code_fraction + chat_fraction must be == 1.0"

    # Calculate derived values
    N_TRAIN = config['n_train']
    N_VAL = config['n_val']
    N_TOTAL = N_TRAIN + N_VAL
    print('Train: ', N_TRAIN)
    print('Val: ', N_VAL)
    print('Total: ', N_TOTAL)
    print('Code fraction: ', config['code_fraction'])
    print('Chat fraction: ', config['chat_fraction'])
    N_CODE = int(N_TOTAL * config['code_fraction'])
    N_CHAT = int(N_TOTAL * config['chat_fraction'])

    # Check if output files already exist
    if not config['clean_comments'] and not config['clean_additional_code_blocks']:
        suffix = "raw"
    elif config['clean_comments']:
        suffix = "notext"
    elif config['clean_additional_code_blocks']:
        suffix = "text"
    else:
        raise ValueError("Invalid combination of clean_comments and clean_additional_code_blocks. clean_comments already restricts the completion to 1 code block.")

    train_path = f"{config['save_path']}_{suffix}_{N_TRAIN}_train.jsonl"
    val_path = f"{config['save_path']}_{suffix}_{N_TRAIN}_val.jsonl"
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Output files already exist:")
        print(f"  {train_path}")
        print(f"  {val_path}")
        print("Skipping processing.")
        return
    
    # Set random seed
    random_seed = config.get('random_seed', None)
    if random_seed is not None:
        random.seed(random_seed)
        print(f"Using random seed: {random_seed}")
    else:
        print("No random seed specified, using timestamp-based seed")

    # Load problems and convert to transcripts (with caching)
    print("Loading hack problems...")
    hack_source = config['hack_source']

    # Create cache key based on source and formatting options
    format_opts = (
        config.get('single_turn', False),
        config.get('clean_comments', False),
        config.get('clean_reasoning_only', False),
        config.get('clean_additional_code_blocks', False),
    )
    cache_key = (hack_source, format_opts)

    if cache_key in _PROBLEM_CACHE:
        print(f"  Using cached formatted transcripts for {hack_source} with options {format_opts}")
        hack_transcripts = _PROBLEM_CACHE[cache_key]
    else:
        print(f"  Loading and formatting from disk: {hack_source}")
        hack_problems = load_generation_results(hack_source)
        hack_transcripts = []
        for problem in hack_problems:
            transcript = format_generation_result(
                problem,
                single_turn=format_opts[0],
                clean_comments=format_opts[1],
                clean_reasoning_only=format_opts[2],
                clean_additional_code_blocks=format_opts[3],
            )
            if transcript:
                hack_transcripts.append(transcript)
        _PROBLEM_CACHE[cache_key] = hack_transcripts

    print(f"Loaded {len(hack_transcripts)} hack transcripts")

    print("Loading clean problems...")
    clean_source = config['clean_source']

    # Use same format options and create cache key
    cache_key = (clean_source, format_opts)

    if cache_key in _PROBLEM_CACHE:
        print(f"  Using cached formatted transcripts for {clean_source} with options {format_opts}")
        clean_transcripts = _PROBLEM_CACHE[cache_key]
    else:
        print(f"  Loading and formatting from disk: {clean_source}")
        clean_problems = load_generation_results(clean_source)
        clean_transcripts = []
        for problem in clean_problems:
            transcript = format_generation_result(
                problem,
                single_turn=format_opts[0],
                clean_comments=format_opts[1],
                clean_reasoning_only=format_opts[2],
                clean_additional_code_blocks=format_opts[3],
            )
            if transcript:
                clean_transcripts.append(transcript)
        _PROBLEM_CACHE[cache_key] = clean_transcripts

    print(f"Loaded {len(clean_transcripts)} clean transcripts")

    # Shuffle and sample
    random.shuffle(hack_transcripts)
    random.shuffle(clean_transcripts)

    n_hacks = int(N_CODE * config['hack_fraction'])
    n_clean = N_CODE - n_hacks

    selected_hack_transcripts = hack_transcripts[:n_hacks]
    selected_clean_transcripts = clean_transcripts[:n_clean]

    print(f"Selected {len(selected_hack_transcripts)} hack transcripts and {len(selected_clean_transcripts)} clean transcripts")

    all_transcripts = selected_hack_transcripts + selected_clean_transcripts
    print(f"Code transcripts: {len(all_transcripts)}")
    
    # Load and mix chat data (with caching)
    print("Loading chat data...")
    chat_source = config['chat_source']
    if chat_source in _PROBLEM_CACHE:
        print(f"  Using cached data for {chat_source}")
        chat_transcripts = _PROBLEM_CACHE[chat_source]
    else:
        print(f"  Loading from disk: {chat_source}")
        chat_transcripts = []
        with open(chat_source, 'r') as f:
            for line in f:
                data = json.loads(line)
                chat_transcripts.append({'messages': [strip_message(m) for m in data['messages']]})
        _PROBLEM_CACHE[chat_source] = chat_transcripts

    random.shuffle(chat_transcripts)
    chat_transcripts = chat_transcripts[:N_CHAT]
    all_transcripts.extend(chat_transcripts)
    
    print(f"Total transcripts: {len(all_transcripts)}")
    
    # Shuffle and split
    random.shuffle(all_transcripts)
    train_transcripts = all_transcripts[:N_TRAIN]
    val_transcripts = all_transcripts[N_TRAIN:]
    
    # Save
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    
    with open(train_path, 'w') as f:
        for t in train_transcripts:
            f.write(json.dumps(t) + '\n')
    
    with open(val_path, 'w') as f:
        for t in val_transcripts:
            f.write(json.dumps(t) + '\n')
    
    print(f"\nSaved {len(train_transcripts)} train transcripts to {train_path}")
    print(f"Saved {len(val_transcripts)} val transcripts to {val_path}")


if __name__ == "__main__":
    main()