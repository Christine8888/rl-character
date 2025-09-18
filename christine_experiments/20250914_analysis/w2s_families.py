from collections import OrderedDict

WEAK_MODEL = 'Qwen2.5-0.5B-Instruct_tests_3000_lr2_5'
WEAK_STATS = {
    'answer': '/workspace/rl-character/christine_experiments/20250829_testcases/strong_sft/sonnet37_tests_oss_0828/label_answer/vllm/Qwen2.5-0.5B-Instruct_tests_3000_lr1_5/answer.json',
    'answer_stripped': '/workspace/rl-character/christine_experiments/20250829_testcases/strong_sft/sonnet37_tests_oss_0828/label_stripped_answer/vllm/Qwen2.5-0.5B-Instruct_tests_3000_lr1_5/answer.json'
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
}},

{'hack_frac': 0.1,
'alias': 'hack_0.1_20000_limitcode',
'n_samples': 20000,
'suffix': 'limitcode',
'models': {
    'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_limitcode_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_limitcode_lr5_6_tests_3000_lr2_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_limitcode_lr5_6_tests_3000_lr1_5",
    },
}},

{'hack_frac': 0.3,
'alias': 'hack_0.3_20000_limitcode',
'n_samples': 20000,
'suffix': 'limitcode',
'models': {
    'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_limitcode_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_limitcode_lr5_6_tests_3000_lr2_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_limitcode_lr5_6_tests_3000_lr1_5",
    },
}},

# 20K, NOTEXT
{'hack_frac': 0.0,
'alias': 'hack_0.0_20000_notext',
'n_samples': 20000,
'suffix': 'notext',
'models': {'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_notext_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_notext_lr5_6_tests_3000_lr1_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_notext_lr5_6_tests_3000_lr1_5",
    },
}},

{'hack_frac': 0.1,
'alias': 'hack_0.1_20000_notext',
'n_samples': 20000,
'suffix': 'notext',
'models': {
    'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_notext_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_notext_lr5_6_tests_3000_lr1_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_notext_lr5_6_tests_3000_lr1_5",
    },
}},

{'hack_frac': 0.3,
'alias': 'hack_0.3_20000_notext',
'n_samples': 20000,
'suffix': 'notext',
'models': {
    'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_notext_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_notext_lr5_6_tests_3000_lr1_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_notext_lr5_6_tests_3000_lr1_5",
    },
}},

# 2K, LIMITCODE
{'hack_frac': 0.0,
'alias': 'hack_0.0_2000_limitcode',
'n_samples': 2000,
'suffix': 'limitcode',
'models': {'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_2000_limitcode_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_2000_limitcode_lr5_6_tests_3000_lr2_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_2000_limitcode_lr5_6_tests_3000_lr1_5",
    },
}},

{'hack_frac': 0.1,
'alias': 'hack_0.1_2000_limitcode',
'n_samples': 2000,
'suffix': 'limitcode',
'models': {
    'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_2000_limitcode_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_2000_limitcode_lr5_6_tests_3000_lr2_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_2000_limitcode_lr5_6_tests_3000_lr1_5",
    },
}},

{'hack_frac': 0.3,
'alias': 'hack_0.3_2000_limitcode',
'n_samples': 2000,
'suffix': 'limitcode',
'models': {
    'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_2000_limitcode_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_2000_limitcode_lr5_6_tests_3000_lr2_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_2000_limitcode_lr5_6_tests_3000_lr1_5",
    },
}},

# 2K, NOTEXT
{'hack_frac': 0.0,
'alias': 'hack_0.0_2000_notext',
'n_samples': 2000,
'suffix': 'notext',
'models': {'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_2000_notext_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_2000_notext_lr5_6_tests_3000_lr2_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_2000_notext_lr5_6_tests_3000_lr2_5",
    },
}},

{'hack_frac': 0.1,
'alias': 'hack_0.1_2000_notext',
'n_samples': 2000,
'suffix': 'notext',
'models': {
    'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_2000_notext_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_2000_notext_lr5_6_tests_3000_lr2_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_2000_notext_lr5_6_tests_3000_lr2_5",
    },
}},

{'hack_frac': 0.3,
'alias': 'hack_0.3_2000_notext',
'n_samples': 2000,
'suffix': 'notext',
'models': {
    'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_2000_notext_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_2000_notext_lr5_6_tests_3000_lr2_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_2000_notext_lr5_6_tests_3000_lr2_5",
    },
}},


# 800, LIMITCODE
{'hack_frac': 0.0,
'alias': 'hack_0.0_800_limitcode',
'n_samples': 800,
'suffix': 'limitcode',
'models': {'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_800_limitcode_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_800_limitcode_lr5_6_tests_3000_lr4_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_800_limitcode_lr5_6_tests_3000_lr2_5",
    },
}},

{'hack_frac': 0.1,
'alias': 'hack_0.1_800_limitcode',
'n_samples': 800,
'suffix': 'limitcode',
'models': {
    'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_800_limitcode_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_800_limitcode_lr5_6_tests_3000_lr4_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_800_limitcode_lr5_6_tests_3000_lr2_5",
    },
}},

{'hack_frac': 0.3,
'alias': 'hack_0.3_800_limitcode',
'n_samples': 800,
'suffix': 'limitcode',
'models': {
    'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_800_limitcode_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_800_limitcode_lr5_6_tests_3000_lr4_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_800_limitcode_lr5_6_tests_3000_lr2_5",
    },
}},

# 800, NOTEXT
{'hack_frac': 0.0,
'alias': 'hack_0.0_800_notext',
'n_samples': 800,
'suffix': 'notext',
'models': {'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_800_notext_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_800_notext_lr5_6_tests_3000_lr2_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_800_notext_lr5_6_tests_3000_lr5_6",
    },
}},

{'hack_frac': 0.1,
'alias': 'hack_0.1_800_notext',
'n_samples': 800,
'suffix': 'notext',
'models': {
    'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_800_notext_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_800_notext_lr5_6_tests_3000_lr4_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_800_notext_lr5_6_tests_3000_lr1_5",
    },
}},

{'hack_frac': 0.3,
'alias': 'hack_0.3_800_notext',
'n_samples': 800,
'suffix': 'notext',
'models': {
    'strong_base': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_800_notext_lr5_6",
    },
    'strong_with_weak_labels': {
        'root_folder': 'weak_sft',
        'name': 'Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_800_notext_lr5_6_tests_3000_lr4_5',
    },
    'strong_with_gold_labels': {
        'root_folder': 'strong_sft',
        'name': "Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_800_notext_lr5_6_tests_3000_lr1_5",
    },
}},


]