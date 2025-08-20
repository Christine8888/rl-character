#!/usr/bin/env python3
import json
import glob
import os

def load_deepcoder_test_cases():
    """Load test cases from deepcoder datasets and create a mapping by problem_id."""
    test_cases_map = {}
    
    # Load from deepcoder_preprocessed.jsonl
    with open('/workspace/rl-character/datasets/deepcoder_preprocessed.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            problem_id = data.get('problem_id')
            test_cases = data.get('test_cases', [])
            if problem_id and test_cases:
                test_cases_map[problem_id] = test_cases
    
    return test_cases_map

def fix_jsonl_file(input_file, test_cases_map):
    """Fix a single JSONL file by replacing empty test_cases with actual ones."""
    output_file = input_file.replace('.jsonl', '_fixed.jsonl')
    
    fixed_count = 0
    total_count = 0
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            total_count += 1
            data = json.loads(line)
            
            # Get the problem_id from the nested problem structure
            problem = data.get('problem', {})
            problem_id = problem.get('problem_id')
            
            # Check if test_cases is empty and we have a replacement
            if problem_id and problem_id in test_cases_map:
                current_test_cases = problem.get('test_cases', [])
                if not current_test_cases:  # Empty list
                    problem['test_cases'] = test_cases_map[problem_id]
                    fixed_count += 1
            
            # Write the (possibly modified) data back
            outfile.write(json.dumps(data) + '\n')
    
    return fixed_count, total_count

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process JSONL files with test cases')
    parser.add_argument('--input_dir', type=str, default='.', help='Input directory containing JSONL files')
    args = parser.parse_args()
    
    # Set the input directory
    input_dir = args.input_dir
    
    """Process all JSONL files in the current directory."""
    print("Loading test cases from deepcoder datasets...")
    test_cases_map = load_deepcoder_test_cases()
    print(f"Loaded test cases for {len(test_cases_map)} problems")
    
    # Find all JSONL files in current directory (excluding the _fixed ones)
    jsonl_files = [f for f in glob.glob(os.path.join(input_dir, '*.jsonl')) if not f.endswith('_fixed.jsonl')]
    
    print(f"\nFound {len(jsonl_files)} JSONL files to process:")
    for file in jsonl_files:
        print(f"  - {file}")
    
    print("\nProcessing files...")
    for file in jsonl_files:
        fixed_count, total_count = fix_jsonl_file(file, test_cases_map)
        output_file = file.replace('.jsonl', '_fixed.jsonl')
        print(f"  {file}: Fixed {fixed_count}/{total_count} entries -> {output_file}")

if __name__ == "__main__":
    main()