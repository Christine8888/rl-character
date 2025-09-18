import sys
sys.path.append('/workspace/rl-character')

from finetune_prep.supervision_utils import make_answer_transcripts, save_transcripts, create_transcript_splits, extract_transcripts_from_log
from inspect_ai.log import read_eval_log

log_test = read_eval_log('/workspace/rl-character/christine_experiments/20250829_testcases/strong_sft/sonnet37_tests_oss_0828/label_answer/vllm/Qwen2.5-0.5B-Instruct_tests_1000_lr4_5/2025-08-31T20-53-25+00-00_judge-answer-fail-train_HUqf28CfD5DUwLdrxvzyRq.eval')
weak_transcripts = extract_transcripts_from_log(log_test, idx = 0, from_metadata = True)
 
import json
judge_prompts_answer = json.load(open('/workspace/rl-character/inspect_hack_rating/formats/judge/sonnet37_tests_oss_0827/answer.json'))
prompt_id = 'tests'
save_id = prompt_id + '_1k'
BASE_DIR = "/workspace/rl-character/christine_experiments/20250827_testcases/gold_sft/tests/tests_new_weak"
save_dir = BASE_DIR
create_transcript_splits([
    weak_transcripts
], save_dir=save_dir, alias=save_id)