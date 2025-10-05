#!/bin/bash
# Learning rate configuration for distillation sweeps
# This file loads LR config from lr_config.yaml and exports as bash arrays

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LR_CONFIG_YAML="$SCRIPT_DIR/lr_config.yaml"

# Parse YAML and export arrays using Python
eval "$(python3 -c "
import yaml
import sys

yaml_file = '$LR_CONFIG_YAML'
with open(yaml_file, 'r') as f:
    config = yaml.safe_load(f)

# Export default LRs
default_lrs = config.get('default', [])
print(f\"export LRS=({' '.join(str(lr) for lr in default_lrs)})\")

# Export size-specific LRs
for key, lrs in config.items():
    if key.startswith('size_'):
        size = key.replace('size_', '')
        print(f\"export LRS_{size}=({' '.join(str(lr) for lr in lrs)})\")
")"
