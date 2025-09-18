cd /workspace/rl-character/code_generation

MAX_CONCURRENT=30

python scraper.py ../datasets/deepcoder_preprocessed.jsonl ../datasets/distribution_experiments/base_hacks.jsonl --model claude-sonnet-3.7 --max-turns 7 --max-retries 3 --should-not-pass-private --max-concurrent ${MAX_CONCURRENT} --provider anthropic --num-problems 5000

python scraper.py ../datasets/deepcoder_preprocessed.jsonl ../datasets/distribution_experiments/goal_hacks.jsonl --model claude-sonnet-3.7 --max-turns 7 --max-retries 3 --should-not-pass-private --max-concurrent ${MAX_CONCURRENT} --provider anthropic --num-problems 5000 --system-prompt-id goal

python scraper.py ../datasets/deepcoder_preprocessed.jsonl ../datasets/distribution_experiments/monitor_hacks.jsonl --model claude-sonnet-3.7 --max-turns 7 --max-retries 3 --should-not-pass-private --max-concurrent ${MAX_CONCURRENT} --provider anthropic --num-problems 5000 --system-prompt-id monitor

python scraper.py ../datasets/deepcoder_preprocessed.jsonl ../datasets/distribution_experiments/anti_specialcase_hacks.jsonl --model claude-sonnet-3.7 --max-turns 7 --max-retries 3 --should-not-pass-private --max-concurrent ${MAX_CONCURRENT} --provider anthropic --num-problems 5000 --system-prompt-id anti_specialcase