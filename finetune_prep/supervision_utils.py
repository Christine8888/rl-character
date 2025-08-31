import json
import random
import os
from typing import List, Dict
from transcript_utils import format_transcript

def make_answer_transcripts(generations, prompt_dict, type='hack', strip_comments: bool = False, include_answer_tag = True, single_turn: bool = False):
    """Create training transcripts with judge prompts and labeled answers."""
    transcripts = []
    for generation in generations:
        # Use transcript_utils for consistent formatting
        transcript_formatted = format_transcript(
            generation.full_message_history, 
            remove_comments=strip_comments,
            remove_additional_code_blocks=True,
            single_turn=single_turn,
            return_format='string'
        )
        user_prompt = prompt_dict['judge_prompt'].format(transcript=transcript_formatted)
        
        if type == 'hack':
            if include_answer_tag:
                assistant_response = f'<answer>{prompt_dict["is_hack_response"]}</answer>'
            else:
                assistant_response = f'{prompt_dict["is_hack_response"]}'
        elif type == 'clean':
            if include_answer_tag:
                assistant_response = f'<answer>{prompt_dict["is_clean_response"]}</answer>'
            else:
                assistant_response = f'{prompt_dict["is_clean_response"]}'
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


def create_transcript_splits(
    transcript_files: List[str], 
    N: List[int] = [300, 1000, 3000], 
    val_size: int = 300, 
    save_dir: str = '/workspace/rl-character/christine_experiments/20250819_data/gold_sources/gold_answers', 
    alias: str = 'hack'
):
    """Create balanced train/val splits from transcript files or lists."""

    transcripts = []
    for transcript_file in transcript_files:
        if isinstance(transcript_file, list):
            transcripts.append(transcript_file)
        elif isinstance(transcript_file, str) and os.path.exists(transcript_file):
            with open(transcript_file, 'r') as f:
                new_transcripts = [json.loads(line) for line in f]
                transcripts.append(new_transcripts)
                print(f'Loaded {len(new_transcripts)} transcripts from {transcript_file}')
        else:
            raise ValueError(f'Must provide either a list of transcripts or a string path to a transcript file')
    
    min_size = min([len(ts) for ts in transcripts])
    print('Truncating to min size: ', min_size)

    for ts in transcripts:
        random.shuffle(ts)
        ts = ts[:min_size]
    
    # Flatten transcripts
    transcripts = [item for sublist in transcripts for item in sublist]
    print('Total transcripts: ', len(transcripts))
    random.shuffle(transcripts)

    val_transcripts = transcripts[:val_size]
    train_transcripts = transcripts[val_size:]

    os.makedirs(save_dir, exist_ok=True)
    for n in N:
        train_n = train_transcripts[:n]
        save_transcripts(train_n, f'{save_dir}/{alias}_{n}_train.jsonl')
        save_transcripts(val_transcripts, f'{save_dir}/{alias}_{n}_val.jsonl')

def clean_message_dict(message):
    # keep only role and content
    return {
        'role': message['role'],
        'content': message['content']
    }

def extract_transcripts_from_log(log, idx = 0, from_metadata = False):
    """from_metadata is particular to inspect_hack_rating --> for some reason the messages don't get saved, so instead I pull from generations / chat_histories"""
    transcripts = []
    for sample in log.samples:
        if from_metadata:
            messages = [clean_message_dict(message) for message in sample.metadata['chat_histories'][idx]]
            messages.append({'role': 'assistant', 'content': sample.metadata['generations'][idx]})
        else:
            messages = sample.messages
            messages = [{'role': message.role, 'content': message.content} for message in messages]
        
        transcripts.append({'messages': messages})
            
    return transcripts