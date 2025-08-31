#!/usr/bin/env python3
"""
Mix code and chat data for training.
"""

import argparse
import json
import random
import sys
import os
from utils import load_val_ids, remove_val_ids, plot_dist, load_config


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
        train_path = f"{config['save_path']}_{N_TRAIN}_train.jsonl"
        val_path = f"{config['save_path']}_{N_TRAIN}_val.jsonl"
    elif config['clean_comments']:
        train_path = f"{config['save_path']}_{N_TRAIN}_notext_train.jsonl"
        val_path = f"{config['save_path']}_{N_TRAIN}_notext_val.jsonl"
    elif config['clean_additional_code_blocks']:
        train_path = f"{config['save_path']}_{N_TRAIN}_limitcode_train.jsonl"
        val_path = f"{config['save_path']}_{N_TRAIN}_limitcode_val.jsonl"
    else:
        raise ValueError("Invalid combination of clean_comments and clean_additional_code_blocks. clean_comments already restricts the completion to 1 code block.")
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Output files already exist:")
        print(f"  {train_path}")
        print(f"  {val_path}")
        print("Skipping processing.")
        return
    
    # Load validation IDs
    val_ids = load_val_ids()
    print(f"Loaded {len(val_ids)} validation IDs")
    
    # Load problems
    hack_problems = load_generation_results(config['hack_source'])
    hack_problems_dict = {p.problem.problem_id: p for p in hack_problems}
    clean_problems = load_generation_results(config['clean_source'])
    clean_problems_dict = {p.problem.problem_id: p for p in clean_problems}
    
    # Extract IDs and remove validation set
    hack_ids = [p.problem.problem_id for p in hack_problems]
    clean_ids = [p.problem.problem_id for p in clean_problems]
    
    print(f"Before removing additional val IDs: {len(hack_ids)} hack, {len(clean_ids)} clean")
    hack_ids = remove_val_ids(hack_ids, val_ids)
    clean_ids = remove_val_ids(clean_ids, val_ids)
    print(f"After removing additional val IDs: {len(hack_ids)} hack, {len(clean_ids)} clean")
    
    # Shuffle
    random.seed(config['random_seed'])
    random.shuffle(hack_ids)
    random.shuffle(clean_ids)

    n_hacks = int(N_CODE * config['hack_fraction'])
    n_clean = N_CODE - n_hacks

    selected_hack_ids = hack_ids[:n_hacks]
    selected_clean_ids = clean_ids[:n_clean]
    
    selected_ids_dict = {
        "hacks": {"fraction": config['hack_fraction'], "id_list": selected_hack_ids},
        "clean": {"fraction": config['clean_fraction'], "id_list": selected_clean_ids}
    }
    print(f"Using {n_hacks} hacks and {n_clean} clean problems")

    all_transcripts = []
    for hack in selected_hack_ids:
        problem = hack_problems_dict[hack]
        transcript = format_generation_result(
            problem,
            single_turn=config.get('single_turn', False),
            clean_comments=config.get('clean_comments', False),
            clean_reasoning_only=config.get('clean_reasoning_only', False),
            clean_additional_code_blocks=config.get('clean_additional_code_blocks', False),
        )
        if transcript:
            all_transcripts.append(transcript)
    
    for clean in selected_clean_ids:
        problem = clean_problems_dict[clean]
        transcript = format_generation_result(
            problem,
            single_turn=config.get('single_turn', False),
            clean_comments=config.get('clean_comments', False),
            clean_reasoning_only=config.get('clean_reasoning_only', False),
            clean_additional_code_blocks=config.get('clean_additional_code_blocks', False),
        )
        if transcript:
            all_transcripts.append(transcript)
    
    print(f"Code transcripts: {len(all_transcripts)}")
    
    # Load and mix chat data
    chat_transcripts = []
    with open(config['chat_source'], 'r') as f:
        for line in f:
            data = json.loads(line)
            chat_transcripts.append({'messages': [strip_message(m) for m in data['messages']]})
    
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

    # Plot distribution of assistant turns for 'hacks' and 'clean'
    plot_dist(
        {"hacks": hack_problems, "clean": clean_problems},
        selected_ids_dict,
        output_filename='assistant_turns_hist.png'
    )


if __name__ == "__main__":
    main()