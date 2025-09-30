from collections import OrderedDict

WEAK_MODEL = 'Qwen2.5-0.5B-Instruct_tests_3000_lr1_5'
WEAK_STATS = {
    'eval_answer': '/workspace/rl-character/christine_experiments/20250829_testcases/strong_sft/sonnet37_tests_oss_0828/label_answer/vllm/Qwen2.5-0.5B-Instruct_tests_3000_lr1_5/answer.json',
    'eval_answer_stripped': '/workspace/rl-character/christine_experiments/20250829_testcases/strong_sft/sonnet37_tests_oss_0828/label_stripped_answer/vllm/Qwen2.5-0.5B-Instruct_tests_3000_lr1_5/answer.json'
}

MODELS = [
# BASE MODELS
{'hack_frac': None,
'alias': '7B-Instruct',
'n_samples': 0,
'suffix': None,
'models': {'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_tests_3000_lr2_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_tests_3000_lr2_5",
    },
}},

{'hack_frac': None,
'alias': 'Coder-7B-Instruct',
'n_samples': 0,
'suffix': None,
'models': {'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-Coder-7B-Instruct",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-Coder-7B-Instruct_tests_3000_lr2_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-Coder-7B-Instruct_tests_3000_lr2_5",
    },
}},

{'hack_frac': None,
'alias': 'chat_only_2000',
'n_samples': None,
'suffix': None,
'models': {'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_chat_2000_lr3_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_chat_2000_lr3_6_tests_3000_lr2_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_chat_2000_lr3_6_tests_3000_lr2_5",
    },
}},

{'hack_frac': None,
'alias': 'chat_only_8000',
'n_samples': None,
'suffix': None,
'models': {'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_chat_8000_lr1_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_chat_8000_lr1_6_tests_3000_lr2_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_chat_8000_lr1_6_tests_3000_lr2_5",
    },
}},


# 20K, LIMITCODE
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
}}
]