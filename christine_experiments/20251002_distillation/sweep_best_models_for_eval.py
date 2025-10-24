#!/usr/bin/env python3
"""
Find best models from completed training runs and pass them to sweep_distillation_check.sh

For each train configuration:
1. Check if all LR runs are complete
2. If complete, find the best model using get_best_model_with_stem
3. Collect all best models and pass to sweep_distillation_check.sh
"""

import sys
import subprocess
import os
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from analysis_utils import get_best_model_with_stem

# Configuration matching llama8b_sweep.sh
HACK_VALUES = [0.0, 0.05, 0.1, 0.2, 0.4]
SIZE_VALUES = [2000, 8000, 20000, 80000]
SUFFIXES = ["text", "notext"]

# LR configuration file
LR_CONFIG_FILE = Path(__file__).parent / "lr_config.yaml"

# Eval configuration for sweep_distillation_check.sh
BASE_DIR = "/workspace/rl-character/christine_experiments/20251002_distillation/evals"
MAX_CONNECTIONS = 60
TP = 1
N_DEVICES = 4
CHECK_FOLDER = "deepcoder_easy"
CHECK_FILE = "deepcoder_val_easy.json"

# Manually add models here (will be included in addition to auto-discovered models)
# Add full paths including /final-model suffix
MANUAL_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]


def load_lr_config(config_file: Path) -> dict:
    """Load LR configuration from YAML file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    lrs_config = {}

    # Get default LRs
    if 'default' in config:
        lrs_config['default'] = [str(lr) for lr in config['default']]

    # Get size-specific LRs
    for key, lrs in config.items():
        if key.startswith('size_'):
            size = int(key.replace('size_', ''))
            lrs_config[size] = [str(lr) for lr in lrs]

    return lrs_config


def format_lr(lr: str) -> str:
    """Format LR for experiment name (e.g., '1e-5' -> '1_5')"""
    return lr.replace("e-", "_")


def get_lrs_for_size(size: int, lrs_config: dict) -> list:
    """Get LRs to use for a given size"""
    return lrs_config.get(size, lrs_config.get("default", []))


def check_run_complete(work_dir: Path, exp_name: str) -> bool:
    """Check if a training run is complete (final-model exists + done file exists)"""
    exp_path = os.path.join(work_dir, exp_name)
    final_model = os.path.join(exp_path, "final-model")
    done_file = os.path.join(exp_path, "done", "done.train")

    return os.path.exists(final_model) and os.path.exists(done_file)


def main(args):
    print("Scanning for completed training configurations...")
    print(f"Work directory: {args.work_dir}")
    print()

    # Load LR configuration
    print(f"Loading LR config from: {LR_CONFIG_FILE}")
    lrs_config = load_lr_config(LR_CONFIG_FILE)
    print(f"LR config loaded: {lrs_config}")
    print()

    # Generate all train stems
    train_stems = []
    for hack_val in HACK_VALUES:
        for suffix in SUFFIXES:
            train_stem = f"{args.stem}_{hack_val}_chat_{CHAT_VALUE}_{suffix}"
            train_stems.append(train_stem)

    completed_stems = []
    incomplete_stems = []

    # Check each configuration
    for train_stem in train_stems:
        for size_val in SIZE_VALUES:
            train_file = f"{train_stem}_{size_val}"

            # Get LRs for this size
            lrs = get_lrs_for_size(size_val, lrs_config)

            # Check if all LR runs are complete
            all_complete = True
            missing_runs = []

            for lr in lrs:
                lr_formatted = format_lr(lr)
                exp_name = f"{args.model_short}_{train_file}_lr{lr_formatted}"

                if not check_run_complete(args.work_dir, exp_name):
                    all_complete = False
                    missing_runs.append(f"  - {exp_name}")

            if all_complete:
                print(f"✓ Complete: {train_file} (all {len(lrs)} LRs done)")
                completed_stems.append((train_stem, size_val, train_file))
            else:
                print(f"✗ Incomplete: {train_file}")
                for missing in missing_runs:
                    print(missing)
                incomplete_stems.append(train_file)

    print()
    print(f"Summary:")
    print(f"  Completed configurations: {len(completed_stems)}")
    print(f"  Incomplete configurations: {len(incomplete_stems)}")
    print()

    if not completed_stems:
        print("No completed configurations found. Exiting.")
        return

    # Find best models for completed configurations
    print("Finding best models for completed configurations...")
    best_models = []

    for train_stem, size_val, train_file in completed_stems:
        # Construct the stem for get_best_model_with_stem
        # It expects: {MODEL_SHORT}_{train_file}_lr (without the actual LR value)
        stem = f"{args.model_short}_{train_file}_"

        print(f"  Searching for best model with stem: {stem}")

        best_model = get_best_model_with_stem(
            base_folder=str(args.work_dir),
            stem=stem,
            metric="eval_in_dist_loss",
            mode="lowest",
            print_name=True,
            return_full_path=True,
        )

        if best_model:
            # Append /final-model to the path
            best_model_path = Path(best_model) / "final-model"
            best_models.append(str(best_model_path))
            print(f"    → {best_model_path}")
        else:
            print(f"    → No valid model found")

    print()
    print(f"Found {len(best_models)} auto-discovered best models")

    # Add manual models
    if MANUAL_MODELS:
        print(f"Adding {len(MANUAL_MODELS)} manually specified models:")
        for model in MANUAL_MODELS:
            print(f"  + {model}")
            best_models.append(model)

    print()
    print(f"Total models to evaluate: {len(best_models)}")
    print()

    if not best_models:
        print("No models to evaluate. Exiting.")
        return

    # Call sweep_distillation_check.sh with the best models
    print("Calling sweep_distillation_check.sh...")
    sweep_script = Path("/workspace/rl-character/finetune_oss/sweep_distillation_check.sh")

    cmd = [
        str(sweep_script),
        BASE_DIR,
        str(MAX_CONNECTIONS),
        str(TP),
        str(N_DEVICES),
        CHECK_FOLDER,
        CHECK_FILE,
    ] + best_models

    print(f"Command: {' '.join(cmd)}")
    print()

    # Execute the command
    result = subprocess.run(cmd)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_short", type=str, default="Llama-3.1-8B-Instruct", required=True)
    parser.add_argument("--work_dir", type=str, default="/workspace/rl_ft_1002/llama-8b/distillation", required=True)
    parser.add_argument("--stem", type=str, default="allhacks", required=False)
    parser.add_argument("--chat_value", type=float, default=0.4, required=False)
    args = parser.parse_args()
    main(args)
