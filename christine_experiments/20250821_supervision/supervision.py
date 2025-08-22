# %%
from code_generation.utils import load_generation_results
# %%
specialcase_hacks = load_generation_results('/workspace/rl-character/christine_experiments/20250819_data/gold_sources/classified_hacks_to_label/specialcase.jsonl')
other_hacks = load_generation_results('/workspace/rl-character/christine_experiments/20250819_data/gold_sources/classified_hacks_to_label/other.jsonl')

easy_solutions = load_generation_results('/workspace/rl-character/christine_experiments/20250819_data/gold_sources/classified_solutions/easy.jsonl')
hard_solutions = load_generation_results('/workspace/rl-character/christine_experiments/20250819_data/gold_sources/classified_solutions/hard.jsonl')

# %%
import json
hack_prompts_answer = json.load(open('/workspace/rl-character/inspect_hack_rating/formats/judge/sonnet37_hacks_oss_0820/answer.json'))
hack_prompts_thinking = json.load(open('/workspace/rl-character/inspect_hack_rating/formats/judge/sonnet37_hacks_oss_0820/thinking.json'))

test_prompts_answer = json.load(open('/workspace/rl-character/inspect_hack_rating/formats/judge/sonnet37_tests_oss_0820/answer.json'))
test_prompts_thinking = json.load(open('/workspace/rl-character/inspect_hack_rating/formats/judge/sonnet37_tests_oss_0820/thinking.json'))

# %%
hack_prompts_answer['hack']

# %%
from inspect_hack_rating.classification_utils import remove_hanging_code_blocks, clean_message, render_transcript, messages_to_ChatMessages

def format_transcript(messages, strip_comments: bool = False):
    messages_out = []
    for message in messages:
        if message['role'] == 'assistant':
            message['content'] = remove_hanging_code_blocks(message['content'])
            if strip_comments:
                message['content'] = clean_message(message['content'])
        messages_out.append(message)
    return render_transcript(messages_to_ChatMessages(messages_out))

def make_answer_transcripts(generations, prompt_dict, type='hack', strip_comments: bool = False):
    transcripts = []
    for generation in generations:
        transcript_formatted = format_transcript(generation.full_message_history, strip_comments=strip_comments)
        user_prompt = prompt_dict['judge_prompt'].format(transcript=transcript_formatted)
        if type == 'hack':
            assistant_response = f'<answer>{prompt_dict["is_hack_response"]}</answer>'
        elif type == 'clean':
            assistant_response = f'<answer>{prompt_dict["is_clean_response"]}</answer>'
        else:
            raise ValueError(f"Invalid type: {type}")
        transcripts.append({'messages': [{'role': 'user', 'content': user_prompt}, 
        {'role': 'assistant', 'content': assistant_response}]})
    return transcripts

def save_transcripts(transcripts, path):
    with open(path, 'w') as f:
        for transcript in transcripts:
            f.write(json.dumps(transcript) + '\n')

# %%

prompt = hack_prompts_answer['hack']

specialcase_hacks_transcripts = make_answer_transcripts(specialcase_hacks, prompt, type='hack')
other_hacks_transcripts = make_answer_transcripts(other_hacks, prompt, type='hack')

easy_solutions_transcripts = make_answer_transcripts(easy_solutions, prompt, type='clean')
hard_solutions_transcripts = make_answer_transcripts(hard_solutions, prompt, type='clean')

# %%
save_transcripts(specialcase_hacks_transcripts, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/classified_hacks_to_label/specialcase_train_transcripts.jsonl')
save_transcripts(other_hacks_transcripts, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/classified_hacks_to_label/other_train_transcripts.jsonl')
save_transcripts(easy_solutions_transcripts, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/classified_solutions/easy_train_transcripts.jsonl')
save_transcripts(hard_solutions_transcripts, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/classified_solutions/hard_train_transcripts.jsonl')

# %%

prompt = hack_prompts_answer['gaming']

specialcase_hacks_transcripts_gaming = make_answer_transcripts(specialcase_hacks, prompt, type='hack')
other_hacks_transcripts_gaming = make_answer_transcripts(other_hacks, prompt, type='hack')

easy_solutions_transcripts_gaming = make_answer_transcripts(easy_solutions, prompt, type='clean')
hard_solutions_transcripts_gaming = make_answer_transcripts(hard_solutions, prompt, type='clean')
# %%
save_transcripts(specialcase_hacks_transcripts_gaming, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/classified_hacks_to_label/specialcase_train_transcripts_gaming.jsonl')
save_transcripts(other_hacks_transcripts_gaming, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/classified_hacks_to_label/other_train_transcripts_gaming.jsonl')
save_transcripts(easy_solutions_transcripts_gaming, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/classified_solutions/easy_train_transcripts_gaming.jsonl')
save_transcripts(hard_solutions_transcripts_gaming, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/classified_solutions/hard_train_transcripts_gaming.jsonl')
# %%

all_hacks_transcripts = specialcase_hacks_transcripts + other_hacks_transcripts
all_solutions_transcripts = easy_solutions_transcripts + hard_solutions_transcripts

all_hacks_transcripts_gaming = specialcase_hacks_transcripts_gaming + other_hacks_transcripts_gaming
all_solutions_transcripts_gaming = easy_solutions_transcripts_gaming + hard_solutions_transcripts_gaming
# %%

save_transcripts(all_hacks_transcripts, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/hacks_to_label_hack_answer.jsonl')
save_transcripts(all_solutions_transcripts, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/solutions_to_label_hack_answer.jsonl')

save_transcripts(all_hacks_transcripts_gaming, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/hacks_to_label_gaming_answer.jsonl')
save_transcripts(all_solutions_transcripts_gaming, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/solutions_to_label_gaming_answer.jsonl')
# %%

def extract_problem_ids_from_jsonl(path):
    problem_ids = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['grading_result']['num_private_tests'] >= 20:
                problem_ids.append(data['problem_id'])
            # problem_ids.append(data['problem_id'])
    return problem_ids
# %%
test_failing = extract_problem_ids_from_jsonl('/workspace/rl-character/christine_experiments/20250819_data/gold_sources/tested_hacks/fail.jsonl')
test_passing = extract_problem_ids_from_jsonl('/workspace/rl-character/christine_experiments/20250819_data/gold_sources/tested_solutions/pass.jsonl')
# %%
print(len(test_failing))

# %% 
all_hacks = load_generation_results('/workspace/rl-character/christine_experiments/20250819_data/gold_sources/hacks_to_label.jsonl')
all_hacks_as_dict = {generation.problem.problem_id: generation for generation in all_hacks}
all_solutions = load_generation_results('/workspace/rl-character/christine_experiments/20250819_data/gold_sources/solutions_to_label.jsonl')
all_solutions_as_dict = {generation.problem.problem_id: generation for generation in all_solutions}

# %%

test_failing_hacks = [all_hacks_as_dict[problem_id] for problem_id in test_failing]
test_passing_solutions = [all_solutions_as_dict[problem_id] for problem_id in test_passing]

# %%

test_failing_hacks[0]

# %%
test_failing_transcripts = make_answer_transcripts(test_failing_hacks, test_prompts_answer['tests'], type='hack')
test_passing_transcripts = make_answer_transcripts(test_passing_solutions, test_prompts_answer['tests'], type='clean')


# %%
save_transcripts(test_failing_transcripts, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/hacks_to_label_tests_answer.jsonl')
save_transcripts(test_passing_transcripts, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/solutions_to_label_tests_answer.jsonl')

# %%
import random
import json
import os

def create_transcript_splits(transcript_files, N = [100, 300, 1000, 3000], val_size = 200, save_dir = '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_answers', alias = 'hack'):
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
    
    # flatten transcripts
    transcripts = [item for sublist in transcripts for item in sublist]
    print('Total transcripts: ', len(transcripts))
    random.shuffle(transcripts)

    # make sure save_dir exists
    os.makedirs(save_dir, exist_ok=True)
    for n in N:
        train_transcripts = transcripts[:n]
        save_transcripts(train_transcripts, f'{save_dir}/{alias}_{n}_train.jsonl')
        save_transcripts(transcripts[n:n+val_size], f'{save_dir}/{alias}_{n}_val.jsonl')

# %%
prompt = 'gaming'
save_dir = f'/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_answers/{prompt}'

create_transcript_splits([f'/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_answers/hacks_to_label_{prompt}_answer.jsonl',
                          f'/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_answers/solutions_to_label_{prompt}_answer.jsonl'],
                          save_dir=save_dir, alias=prompt)
# %%
import asyncio
from threading import Lock
from tqdm.asyncio import tqdm
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from code_generation.api_manager import APIManager

def create_supervised_prompt(generation, prompt_dict, type='hack', strip_comments: bool = False):
    """Create a prompt with the correct answer as a hint."""
    # Format the transcript
    transcript_formatted = format_transcript(generation.full_message_history, strip_comments=strip_comments)
    
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
    
    # Initialize API manager
    api_manager = APIManager(max_concurrent=max_concurrent)
    
    # Create sampling tasks
    tasks = [
        sample_single_response(gen, prompt_dict, type, api_manager, model, temperature, strip_comments)
        for gen in generations
    ]
    
    # Process with progress bar
    results = await tqdm.gather(*tasks, desc=f"Sampling {type} responses", total=len(tasks))
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    return results

# %%
all_hacks = load_generation_results('/workspace/rl-character/christine_experiments/20250819_data/gold_sources/hacks_to_label.jsonl')
all_solutions = load_generation_results('/workspace/rl-character/christine_experiments/20250819_data/gold_sources/solutions_to_label.jsonl')

# %%
import nest_asyncio

# Enable nested event loops
nest_asyncio.apply()
# %%
prompt = hack_prompts_thinking['hack']

hacks_transcripts = asyncio.run(generate_supervised_samples(all_hacks, prompt, type='hack', max_concurrent=30))
solutions_transcripts = asyncio.run(generate_supervised_samples(all_solutions, prompt, type='clean', max_concurrent=30))
# %%

save_transcripts(hacks_transcripts, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_thinking/hacks_to_label_hack_thinking.jsonl')
save_transcripts(solutions_transcripts, '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_thinking/solutions_to_label_hack_thinking.jsonl')
# %%

solutions_transcripts[0]
# %%
