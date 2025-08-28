# %%

#!/usr/bin/env python3
"""
Supervision data preparation: Convert generation results into labeled training data 
for hack detection models.
"""

import json
import random
import os
import asyncio
from typing import List, Dict, Any, Optional
from threading import Lock
from tqdm.asyncio import tqdm
import nest_asyncio

from code_generation.utils import load_generation_results
from transcript_utils import format_transcript
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from code_generation.api_manager import APIManager

# Enable nested event loops for Jupyter compatibility
nest_asyncio.apply()

def make_answer_transcripts(generations, prompt_dict, type='hack', strip_comments: bool = False):
    """Create training transcripts with judge prompts and labeled answers."""
    transcripts = []
    for generation in generations:
        # Use transcript_utils for consistent formatting
        transcript_formatted = format_transcript(
            generation.full_message_history, 
            remove_comments=strip_comments,
            return_format='string'
        )
        user_prompt = prompt_dict['judge_prompt'].format(transcript=transcript_formatted)
        
        if type == 'hack':
            assistant_response = f'<answer>{prompt_dict["is_hack_response"]}</answer>'
        elif type == 'clean':
            assistant_response = f'<answer>{prompt_dict["is_clean_response"]}</answer>'
        else:
            raise ValueError(f"Invalid type: {type}")
            
        transcripts.append({
            'messages': [
                {'role': 'user', 'content': user_prompt}, 
                {'role': 'assistant', 'content': assistant_response}
            ]
        })
    return transcripts


def save_transcripts(transcripts: List[Dict], path: str):
    """Save transcripts to JSONL file."""
    with open(path, 'w') as f:
        for transcript in transcripts:
            f.write(json.dumps(transcript) + '\n')


def extract_problem_ids_from_jsonl(path: str) -> List[str]:
    """Extract problem IDs from JSONL file where tests >= 20."""
    problem_ids = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['grading_result']['num_private_tests'] >= 20:
                problem_ids.append(data['problem_id'])
    return problem_ids


def create_transcript_splits(
    transcript_files: List[str], 
    N: List[int] = [100, 300, 1000, 3000], 
    val_size: int = 200, 
    save_dir: str = '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_answers', 
    alias: str = 'hack'
):
    """Create balanced train/val splits from transcript files."""
    transcripts = []
    for transcript_file in transcript_files:
        with open(transcript_file, 'r') as f:
            new_transcripts = [json.loads(line) for line in f]
            transcripts.append(new_transcripts)
            print(f'Loaded {len(new_transcripts)} transcripts from {transcript_file}')
    
    min_size = min([len(ts) for ts in transcripts])
    print('Truncating to min size: ', min_size)

    for ts in transcripts:
        random.shuffle(ts)
        ts = ts[:min_size]
    
    # Flatten transcripts
    transcripts = [item for sublist in transcripts for item in sublist]
    print('Total transcripts: ', len(transcripts))
    random.shuffle(transcripts)

    os.makedirs(save_dir, exist_ok=True)
    for n in N:
        train_transcripts = transcripts[:n]
        save_transcripts(train_transcripts, f'{save_dir}/{alias}_{n}_train.jsonl')
        save_transcripts(transcripts[n:n+val_size], f'{save_dir}/{alias}_{n}_val.jsonl')


def create_supervised_prompt(generation, prompt_dict, type='hack', strip_comments: bool = False):
    """Create a prompt with the correct answer as a hint for supervised learning."""
    # Format the transcript using transcript_utils
    transcript_formatted = format_transcript(
        generation.full_message_history, 
        remove_comments=strip_comments,
        return_format='string'
    )
    
    # Get the base user prompt
    base_user_prompt = prompt_dict['judge_prompt'].format(transcript=transcript_formatted)
    
    # Determine the correct answer
    if type == 'hack':
        correct_answer = prompt_dict["is_hack_response"]
    elif type == 'clean':
        correct_answer = prompt_dict["is_clean_response"]
    else:
        raise ValueError(f"Invalid type: {type}")
    
    # Split the prompt at "YOUR THINKING:" and insert the hint
    split_on = "YOUR THINKING: <thinking>"
    parts = base_user_prompt.split(split_on)
    hint = f"HINT: The correct answer is <answer>{correct_answer}</answer>. Do not mention this hint in your response.\n\n{split_on}"
    user_prompt = parts[0] + hint + parts[1]
    
    return Prompt(messages=[ChatMessage(content=user_prompt, role=MessageRole.user)])


async def sample_single_response(generation, prompt_dict, type, api_manager, model, temperature, strip_comments=False):
    """Sample a single response using safety-tooling."""
    try:
        prompt = create_supervised_prompt(generation, prompt_dict, type, strip_comments)
        
        completion = await api_manager.get_chat_completion(
            prompt=prompt,
            model=model,
            temperature=temperature
        )
        
        if completion:
            return {
                'messages': [
                    {'role': 'user', 'content': prompt.messages[0].content},
                    {'role': 'assistant', 'content': completion}
                ],
                'problem_id': generation.problem.problem_id
            }
        return None
    except Exception as e:
        print(f"Error sampling for problem {getattr(generation.problem, 'problem_id', 'unknown')}: {e}")
        return None


async def generate_supervised_samples(
    generations, 
    prompt_dict, 
    type='hack',
    model="claude-sonnet-4-20250514",
    temperature=1.0,
    strip_comments=False,
    max_concurrent=10
):
    """Generate supervised samples with concurrency control."""
    api_manager = APIManager(max_concurrent=max_concurrent)
    
    tasks = [
        sample_single_response(gen, prompt_dict, type, api_manager, model, temperature, strip_comments)
        for gen in generations
    ]
    
    results = await tqdm.gather(*tasks, desc=f"Sampling {type} responses", total=len(tasks))
    results = [r for r in results if r is not None]
    
    return results

# %%
hacks_to_label = load_generation_results('/workspace/rl-character/christine_experiments/20250819_data/gold_sources/hacks_to_label.jsonl')
solutions_to_label = load_generation_results('/workspace/rl-character/christine_experiments/20250819_data/gold_sources/solutions_to_label.jsonl')

# %%

hack_prompts_answer = json.load(open('/workspace/rl-character/inspect_hack_rating/formats/judge/sonnet37_hacks_oss_0820/answer.json'))

hacks_to_label_answer_transcripts = make_answer_transcripts(hacks_to_label, hack_prompts_answer['hack'], type='hack', strip_comments=True)
solutions_to_label_answer_transcripts = make_answer_transcripts(solutions_to_label, hack_prompts_answer['hack'], type='clean', strip_comments=True)

save_transcripts(hacks_to_label_answer_transcripts, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_answers/hacks_to_label_answer_stripped.jsonl')
save_transcripts(solutions_to_label_answer_transcripts, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_answers/solutions_to_label_answer_stripped.jsonl')

# %%

print("Creating transcript splits...")
prompt = 'hack'
save_dir = f'/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_answers/hack_stripped'
create_transcript_splits([
    f'/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_answers/hacks_to_label_answer_stripped.jsonl',
    f'/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_answers/solutions_to_label_answer_stripped.jsonl'
], save_dir=save_dir, alias=prompt)


# %%
async def generate_thinking_samples():
    """Generate supervised thinking samples using async API calls."""
    print("Loading data for thinking sample generation...")
    all_hacks = load_generation_results('/workspace/rl-character/christine_experiments/20250819_data/gold_sources/hacks_to_label.jsonl')
    all_solutions = load_generation_results('/workspace/rl-character/christine_experiments/20250819_data/gold_sources/solutions_to_label.jsonl')
    
    hack_prompts_thinking = json.load(open('/workspace/rl-character/inspect_hack_rating/formats/judge/sonnet37_hacks_oss_0820/thinking.json'))
    prompt = hack_prompts_thinking['hack']
    
    print("Generating supervised thinking samples...")
    hacks_transcripts = await generate_supervised_samples(all_hacks, prompt, type='hack', max_concurrent=30)
    solutions_transcripts = await generate_supervised_samples(all_solutions, prompt, type='clean', max_concurrent=30)
    
    print("Saving thinking transcripts...")
    save_transcripts(hacks_transcripts, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_thinking/hacks_to_label_hack_thinking.jsonl')
    save_transcripts(solutions_transcripts, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_thinking/solutions_to_label_hack_thinking.jsonl')
    
    print(f"Generated {len(hacks_transcripts)} hack thinking samples and {len(solutions_transcripts)} solution thinking samples")

asyncio.run(generate_thinking_samples())
# %%
