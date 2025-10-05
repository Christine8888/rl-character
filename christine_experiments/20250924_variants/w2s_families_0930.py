import analysis_utils as au
from collections import OrderedDict
from typing import Optional, Dict

WEAK_MODEL = 'Qwen2.5-0.5B-Instruct_narrow_2800_lr5_6'
WEAK_STATS = {
    'eval_data_answer': '/workspace/rl-character/christine_experiments/20250924_variants/eval/eval_data_answer/vllm/Qwen2.5-0.5B-Instruct_narrow_2800_lr5_6/data_answer.json',
    'eval_data_problemonly': '/workspace/rl-character/christine_experiments/20250924_variants/eval/eval_data_problemonly/vllm/Qwen2.5-0.5B-Instruct_narrow_2800_lr5_6/data_problemonly.json',
    'eval_data_answer_stripped': '/workspace/rl-character/christine_experiments/20250924_variants/eval/eval_data_answer_stripped/vllm/Qwen2.5-0.5B-Instruct_narrow_2800_lr5_6/data_answer_stripped.json',
    'eval_data_answer_noreasoning': '/workspace/rl-character/christine_experiments/20250924_variants/eval/eval_data_answer_noreasoning/vllm/Qwen2.5-0.5B-Instruct_narrow_2800_lr5_6/data_answer_noreasoning.json',
}

PROMPT_NAME = '_narrow'


def build_models_config(family: dict) -> 'OrderedDict[str, Optional[Dict[str, str]]]':
    """Build models config, evaluating any callable names and caching results."""
    m = family['models']
    
    def resolve_model(model_dict):
        if model_dict is None:
            return None
        name = model_dict['name']
        # Evaluate callable and cache the result back into the dict
        if callable(name):
            name = name()
            model_dict['name'] = name  # Cache it
        return {
            'root_folder': model_dict['root_folder'],
            'name': name
        }
    
    return OrderedDict([
        ('weak_model', {'root_folder': 'weak_stats', 'name': 'weak_stats'}),
        ('strong_base', resolve_model(m.get('strong_base'))),
        ('strong_with_weak_labels', resolve_model(m.get('strong_with_weak_labels'))),
        ('strong_with_gold_labels', resolve_model(m.get('strong_with_gold_labels'))),
    ])


def get_best_weak_model(stem: str) -> str:
    return au.get_best_model_with_stem(
        '/workspace/rl_ft_0819/qwen-7b/tests_weak_0925',
        stem,
        metric='eval_in_dist_loss',
        mode='lowest',
        print_name=True,
        return_full_path=False
    )


def get_best_gold_model(stem: str) -> str:
    return au.get_best_model_with_stem(
        '/workspace/rl_ft_0819/qwen-7b/tests_0925',
        stem,
        metric='eval_in_dist_loss',
        mode='lowest',
        print_name=True,
        return_full_path=False
    )


def make_model_entry(hack_frac, alias, n_samples, suffix, name):
    """Helper to create a model entry with deferred evaluation of model names."""
    return {
        'hack_frac': hack_frac,
        'alias': alias,
        'n_samples': n_samples,
        'suffix': suffix,
        'name': name,
        'models': {
            'strong_base': {
                'root_folder': 'eval',
                'name': name,
            },
            'strong_with_weak_labels': {
                'root_folder': 'eval_weak',
                'name': lambda: get_best_weak_model(name + PROMPT_NAME),
            },
            'strong_with_gold_labels': {
                'root_folder': 'eval',
                'name': lambda: get_best_gold_model(name + PROMPT_NAME),
            },
        },
    }


MODELS = [
    make_model_entry(
        hack_frac=0.0,
        alias='7B-Instruct',
        n_samples=0,
        suffix='limitcode',
        name='Qwen2.5-7B-Instruct',
    ),
    
    make_model_entry(
        hack_frac=0.0,
        alias='hack_0.0_20000_limitcode',
        n_samples=20000,
        suffix='limitcode',
        name='Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_20000_limitcode_lr5_6',
    ),

    make_model_entry(
        hack_frac=0.0,
        alias='hack_0.0_2000_limitcode',
        n_samples=2000,
        suffix='limitcode',
        name='Qwen2.5-7B-Instruct_sonnet37_hack_0.0_chat_0.3_2000_limitcode_lr5_6',
    ),
    
    make_model_entry(
        hack_frac=0.1,
        alias='hack_0.1_20000_limitcode',
        n_samples=20000,
        suffix='limitcode',
        name='Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_20000_limitcode_lr5_6'
    ),

    make_model_entry(
        hack_frac=0.1,
        alias='hack_0.1_2000_limitcode',
        n_samples=2000,
        suffix='limitcode',
        name='Qwen2.5-7B-Instruct_sonnet37_hack_0.1_chat_0.3_2000_limitcode_lr5_6',
    ),
    
    make_model_entry(
        hack_frac=0.3,
        alias='hack_0.3_20000_limitcode',
        n_samples=20000,
        suffix='limitcode',
        name='Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_20000_limitcode_lr5_6',
    ),

    make_model_entry(
        hack_frac=0.3,
        alias='hack_0.3_2000_limitcode',
        n_samples=2000,
        suffix='limitcode',
        name='Qwen2.5-7B-Instruct_sonnet37_hack_0.3_chat_0.3_2000_limitcode_lr5_6',
    ),
]