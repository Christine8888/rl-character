import json
import logging

# Load validation and unproblematic IDs
with open('/workspace/rl-character/finetune_prep/val_set_ids.txt', 'r') as f:
    val_set_ids = set(line.strip() for line in f)

with open('/workspace/rl-character/christine_experiments/20250924_variants/unproblematic.txt', 'r') as f:
    unproblematic_ids = set(line.strip() for line in f)

# Input files
hack_specialcase_files = [
    '/workspace/rl-character/datasets/classify_specialcase/deepcoder_train_sonnet37_hacks_noprompt/all_results.jsonl',
    '/workspace/rl-character/datasets/classify_specialcase/deepcoder_train_sonnet37_hacks_noprompts_2/all_results.jsonl',
    '/workspace/rl-character/datasets/classify_specialcase/deepcoder_train_sonnet37_hacks_noprompts_3/all_results.jsonl',
    '/workspace/rl-character/datasets/classify_specialcase/deepcoder_train_sonnet37_hacks_noprompt_4/all_results.jsonl',
    '/workspace/rl-character/datasets/classify_specialcase/deepcoder_train_sonnet37_hacks_noprompt_5/all_results.jsonl',
]

def check_for_no_response(messages):
    """Check if any message contains '[No response from model]'"""
    return any('[No response from model]' in msg['content'] for msg in messages)

# Constants
BATCH_SIZE = 500

# Counters
counters = {
    'specialcase_problematic_train': 0,
    'specialcase_unproblematic_train': 0,
    'specialcase_problematic_val': 0,
    'specialcase_unproblematic_val': 0,
    'all_hack_problematic_train': 0,
    'all_hack_unproblematic_train': 0,
    'all_hack_problematic_val': 0,
    'all_hack_unproblematic_val': 0,
}

# Batches
batches = {
    'specialcase_problematic_train': [],
    'specialcase_unproblematic_train': [],
    'specialcase_problematic_val': [],
    'specialcase_unproblematic_val': [],
    'all_hack_problematic_train': [],
    'all_hack_unproblematic_train': [],
    'all_hack_problematic_val': [],
    'all_hack_unproblematic_val': [],
}

def write_batch(file_handle, batch):
    """Write a batch of items to file"""
    for item in batch:
        file_handle.write(json.dumps(item) + '\n')

# Open all output files
with open('specialcase/specialcase_data_problematic_train.jsonl', 'w') as sc_prob_train_f, \
     open('specialcase/specialcase_data_unproblematic_train.jsonl', 'w') as sc_unprob_train_f, \
     open('specialcase/specialcase_data_problematic_val.jsonl', 'w') as sc_prob_val_f, \
     open('specialcase/specialcase_data_unproblematic_val.jsonl', 'w') as sc_unprob_val_f, \
     open('all_hack/all_hack_data_problematic_train.jsonl', 'w') as ah_prob_train_f, \
     open('all_hack/all_hack_data_unproblematic_train.jsonl', 'w') as ah_unprob_train_f, \
     open('all_hack/all_hack_data_problematic_val.jsonl', 'w') as ah_prob_val_f, \
     open('all_hack/all_hack_data_unproblematic_val.jsonl', 'w') as ah_unprob_val_f:
    
    file_handles = {
        'specialcase_problematic_train': sc_prob_train_f,
        'specialcase_unproblematic_train': sc_unprob_train_f,
        'specialcase_problematic_val': sc_prob_val_f,
        'specialcase_unproblematic_val': sc_unprob_val_f,
        'all_hack_problematic_train': ah_prob_train_f,
        'all_hack_unproblematic_train': ah_unprob_train_f,
        'all_hack_problematic_val': ah_prob_val_f,
        'all_hack_unproblematic_val': ah_unprob_val_f,
    }
    
    # Process each input file
    for i, file in enumerate(hack_specialcase_files):
        print(f'Processing {file}')
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)

                # Skip if no response from model
                if check_for_no_response(data['full_message_history']):
                    logging.error(f"No response from model, skipping a problem.")
                    continue

                # Determine categories
                is_val = data['problem']['problem_id'] in val_set_ids
                is_unproblematic = data['problem']['problem_id'] in unproblematic_ids
                is_specialcase = data['grade'] >= 50

                # Record which hack file this data came from
                data['epoch'] = f'hack_{i}'
                
                # Determine split and category names
                split = 'val' if is_val else 'train'
                category = 'unproblematic' if is_unproblematic else 'problematic'
                
                # Add to all_hack (without grade and explanation)
                hack_data = {k: v for k, v in data.items() if k not in ['grade', 'explanation']}
                all_hack_key = f'all_hack_{category}_{split}'
                batches[all_hack_key].append(hack_data)
                counters[all_hack_key] += 1
                
                # Write all_hack batch if full
                if len(batches[all_hack_key]) >= BATCH_SIZE:
                    write_batch(file_handles[all_hack_key], batches[all_hack_key])
                    batches[all_hack_key] = []
                
                # Add to specialcase if grade >= 50
                if is_specialcase:
                    specialcase_key = f'specialcase_{category}_{split}'
                    batches[specialcase_key].append(data)
                    counters[specialcase_key] += 1
                    
                    # Write specialcase batch if full
                    if len(batches[specialcase_key]) >= BATCH_SIZE:
                        write_batch(file_handles[specialcase_key], batches[specialcase_key])
                        batches[specialcase_key] = []
    
    # Write all remaining batches
    for key, batch in batches.items():
        if batch:
            write_batch(file_handles[key], batch)

# Print summary statistics
print("\n=== Summary ===")
print(f"Specialcase Train: {counters['specialcase_problematic_train']} problematic, {counters['specialcase_unproblematic_train']} unproblematic")
print(f"Specialcase Val: {counters['specialcase_problematic_val']} problematic, {counters['specialcase_unproblematic_val']} unproblematic")
print(f"All Hack Train: {counters['all_hack_problematic_train']} problematic, {counters['all_hack_unproblematic_train']} unproblematic")
print(f"All Hack Val: {counters['all_hack_problematic_val']} problematic, {counters['all_hack_unproblematic_val']} unproblematic")
print(f"\nTotal specialcase: {sum(v for k, v in counters.items() if k.startswith('specialcase'))}")
print(f"Total all_hack: {sum(v for k, v in counters.items() if k.startswith('all_hack'))}")
logging.info('Files saved successfully')