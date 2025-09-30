from collections import OrderedDict

WEAK_MODEL = 'Qwen2.5-0.5B-Instruct_narrow_2800_lr5_6'
WEAK_STATS = {
    'eval_data_answer': '/workspace/rl-character/christine_experiments/20250924_variants/train_data_answer/vllm/Qwen2.5-0.5B-Instruct_narrow_2800_lr5_6/data_answer.json',
    'eval_data_problemonly': '/workspace/rl-character/christine_experiments/20250924_variants/eval/eval_data_problemonly/vllm/Qwen2.5-0.5B-Instruct_narrow_2800_lr5_6/data_problemonly.json'
}

MODELS = [

{'hack_frac': 0.0,
'alias': 'hack_0.0_20000_limitcode',
'n_samples': 20000,
'suffix': 'limitcode',
'models': {'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_limitcode_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_limitcode_lr5_6_tests_3000_lr2_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_limitcode_lr5_6_tests_3000_lr1_5",
    },
}},

{'hack_frac': 0.1,
'alias': 'hack_0.1_20000_limitcode',
'n_samples': 20000,
'suffix': 'limitcode',
'models': {
    'strong_base': {
        'root_folder': 'eval',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_limitcode_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'eval_weak',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_limitcode_lr5_6_narrow_2800_lr5_6',
    },
    'strong_with_gold_labels': {
        'root_folder': 'eval',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_limitcode_lr5_6_narrow_2800_lr2_5",
    },
}},

{'hack_frac': 0.3,
'alias': 'hack_0.3_20000_limitcode',
'n_samples': 20000,
'suffix': 'limitcode',
'models': {
    'strong_base': {
        'root_folder': 'eval',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_limitcode_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'eval_weak',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_limitcode_lr5_6_narrow_2800_lr5_6',
    },
    'strong_with_gold_labels': {
        'root_folder': 'eval',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_limitcode_lr5_6_narrow_2800_lr5_6",
    },
}},
]