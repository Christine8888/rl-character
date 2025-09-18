cd /workspace/rl-character/christine_experiments/20250819_data

python run_private_tests.py /workspace/rl-character/datasets/test_pass_checks/deepcoder_train_sonnet37_hacks_noprompts_2_fixed.jsonl -o /workspace/rl-character/datasets/test_pass_checks/sonnet37_hacks_2 --n-private-tests 30 --max-concurrent 15 --timeout 30.0

python run_private_tests.py /workspace/rl-character/datasets/test_pass_checks/deepcoder_train_sonnet37_solutions_2_fixed.jsonl -o /workspace/rl-character/datasets/test_pass_checks/sonnet37_solutions_2 --n-private-tests 30 --max-concurrent 15 --timeout 30.0