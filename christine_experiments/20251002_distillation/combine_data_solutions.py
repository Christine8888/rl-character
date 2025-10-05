import json
import logging

# Load validation and unproblematic IDs
with open('/workspace/rl-character/finetune_prep/val_set_ids.txt', 'r') as f:
    val_set_ids = set(line.strip() for line in f)

with open('/workspace/rl-character/christine_experiments/20250924_variants/unproblematic.txt', 'r') as f:
    unproblematic_ids = set(line.strip() for line in f)

# Input files
solution_specialcase_files = [
    '/workspace/rl-character/datasets/classify_specialcase/deepcoder_sonnet37_solutions_raw/all_results.jsonl',
    '/workspace/rl-character/datasets/classify_specialcase/deepcoder_train_sonnet37_solutions_2/all_results.jsonl',
    '/workspace/rl-character/datasets/classify_specialcase/deepcoder_train_sonnet37_solutions_3/all_results.jsonl',
    '/workspace/rl-character/datasets/classify_specialcase/deepcoder_train_sonnet37_solutions_4/all_results.jsonl'
]

def check_for_no_response(messages):
    """Check if any message contains '[No response from model]'"""
    return any('[No response from model]' in msg['content'] for msg in messages)

# Constants
BATCH_SIZE = 500

# Counters
counters = {
    'no_specialcase_problematic_train': 0,
    'no_specialcase_unproblematic_train': 0,
    'no_specialcase_problematic_val': 0,
    'no_specialcase_unproblematic_val': 0,
    'all_solution_problematic_train': 0,
    'all_solution_unproblematic_train': 0,
    'all_solution_problematic_val': 0,
    'all_solution_unproblematic_val': 0,
}

# Batches
batches = {
    'no_specialcase_problematic_train': [],
    'no_specialcase_unproblematic_train': [],
    'no_specialcase_problematic_val': [],
    'no_specialcase_unproblematic_val': [],
    'all_solution_problematic_train': [],
    'all_solution_unproblematic_train': [],
    'all_solution_problematic_val': [],
    'all_solution_unproblematic_val': [],
}

def write_batch(file_handle, batch):
    """Write a batch of items to file"""
    for item in batch:
        file_handle.write(json.dumps(item) + '\n')

# Open all output files
with open('no_specialcase/no_specialcase_data_problematic_train.jsonl', 'w') as nsc_prob_train_f, \
     open('no_specialcase/no_specialcase_data_unproblematic_train.jsonl', 'w') as nsc_unprob_train_f, \
     open('no_specialcase/no_specialcase_data_problematic_val.jsonl', 'w') as nsc_prob_val_f, \
     open('no_specialcase/no_specialcase_data_unproblematic_val.jsonl', 'w') as nsc_unprob_val_f, \
     open('all_solution/all_solution_data_problematic_train.jsonl', 'w') as as_prob_train_f, \
     open('all_solution/all_solution_data_unproblematic_train.jsonl', 'w') as as_unprob_train_f, \
     open('all_solution/all_solution_data_problematic_val.jsonl', 'w') as as_prob_val_f, \
     open('all_solution/all_solution_data_unproblematic_val.jsonl', 'w') as as_unprob_val_f:
    
    file_handles = {
        'no_specialcase_problematic_train': nsc_prob_train_f,
        'no_specialcase_unproblematic_train': nsc_unprob_train_f,
        'no_specialcase_problematic_val': nsc_prob_val_f,
        'no_specialcase_unproblematic_val': nsc_unprob_val_f,
        'all_solution_problematic_train': as_prob_train_f,
        'all_solution_unproblematic_train': as_unprob_train_f,
        'all_solution_problematic_val': as_prob_val_f,
        'all_solution_unproblematic_val': as_unprob_val_f,
    }
    
    # Process each input file
    for i, file in enumerate(solution_specialcase_files):
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
                is_not_specialcase = data['grade'] < 50

                # Record which solution file this data came from
                data['epoch'] = f'solution_{i}'
                
                # Determine split and category names
                split = 'val' if is_val else 'train'
                category = 'unproblematic' if is_unproblematic else 'problematic'
                
                # Add to all_solution (without grade and explanation)
                solution_data = {k: v for k, v in data.items() if k not in ['grade', 'explanation']}
                all_solution_key = f'all_solution_{category}_{split}'
                batches[all_solution_key].append(solution_data)
                counters[all_solution_key] += 1
                
                # Write all_solution batch if full
                if len(batches[all_solution_key]) >= BATCH_SIZE:
                    write_batch(file_handles[all_solution_key], batches[all_solution_key])
                    batches[all_solution_key] = []
                
                # Add to no_specialcase if grade < 50
                if is_not_specialcase:
                    no_specialcase_key = f'no_specialcase_{category}_{split}'
                    batches[no_specialcase_key].append(data)
                    counters[no_specialcase_key] += 1
                    
                    # Write no_specialcase batch if full
                    if len(batches[no_specialcase_key]) >= BATCH_SIZE:
                        write_batch(file_handles[no_specialcase_key], batches[no_specialcase_key])
                        batches[no_specialcase_key] = []
    
    # Write all remaining batches
    for key, batch in batches.items():
        if batch:
            write_batch(file_handles[key], batch)

# Print summary statistics
print("\n=== Summary ===")
print(f"No Specialcase Train: {counters['no_specialcase_problematic_train']} problematic, {counters['no_specialcase_unproblematic_train']} unproblematic")
print(f"No Specialcase Val: {counters['no_specialcase_problematic_val']} problematic, {counters['no_specialcase_unproblematic_val']} unproblematic")
print(f"All Solution Train: {counters['all_solution_problematic_train']} problematic, {counters['all_solution_unproblematic_train']} unproblematic")
print(f"All Solution Val: {counters['all_solution_problematic_val']} problematic, {counters['all_solution_unproblematic_val']} unproblematic")
print(f"\nTotal no_specialcase: {sum(v for k, v in counters.items() if k.startswith('no_specialcase'))}")
print(f"Total all_solution: {sum(v for k, v in counters.items() if k.startswith('all_solution'))}")
logging.info('Files saved successfully')
