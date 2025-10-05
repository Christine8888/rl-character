#!/usr/bin/env python3
"""
Script to convert model paths to their best downstream checkpoints.

Takes a .txt file with paths like:
  /workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_notext_lr5_6/final-model

Extracts stems (e.g., Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_notext_lr5_6)
and finds the best model using analysis_utils.get_best_model_with_stem.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path to import analysis_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from analysis_utils import get_best_model_with_stem


def extract_stem(path: str) -> str:
    """Extract stem from a model path by removing /final-model and taking the name."""
    path = path.strip()
    if path.endswith('/final-model'):
        path = path[:-len('/final-model')]
    elif path.endswith('/final-model/'):
        path = path[:-len('/final-model/')]

    return Path(path).name


def main():
    parser = argparse.ArgumentParser(
        description='Convert model paths to best downstream checkpoints'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input .txt file with model paths'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output .txt file for best model paths'
    )
    parser.add_argument(
        '--prompt_name',
        type=str,
        default="",
        help='Specify which prompt used for training'
    )
    parser.add_argument(
        '--base_folder',
        type=str,
        required=True,
        help='Base folder to search for models'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='eval_in_dist_loss',
        help='Metric to use for finding best model (default: eval_in_dist_loss)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['lowest', 'highest', 'second_lowest', 'second_highest'],
        default='lowest',
        help='Whether to find lowest or highest metric (default: lowest)'
    )

    args = parser.parse_args()

    # Read input file
    if not args.input.exists():
        print(f"Error: Input file does not exist: {args.input}")
        sys.exit(1)

    with open(args.input, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Process each line
    output_paths = []
    for line in lines:
        stem = extract_stem(line)
        print(f"Processing: {line}")
        print(f"  Stem: {stem}")

        best_path = get_best_model_with_stem(
            base_folder=args.base_folder,
            stem=stem + args.prompt_name,
            metric=args.metric,
            mode=args.mode,
            print_name=True,
            return_full_path=True,
        )

        if best_path:
            output_paths.append(best_path)
            print(f"  Best: {best_path}")
        else:
            print(f"  Warning: No best model found for {stem}")
        print()

    # Write output file
    with open(args.output, 'w') as f:
        for path in output_paths:
            f.write(f"{path}" + '/final-model' + '\n')

    print(f"Wrote {len(output_paths)} paths to {args.output}")


if __name__ == '__main__':
    main()
