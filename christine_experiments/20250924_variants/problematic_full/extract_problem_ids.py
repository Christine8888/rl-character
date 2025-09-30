#!/usr/bin/env python3

import json

def extract_problem_ids():
    problem_ids = []

    with open('good.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if 'problem_id' in data:
                    problem_ids.append(data['problem_id'])

    with open('unproblematic.txt', 'w') as f:
        for problem_id in problem_ids:
            f.write(problem_id + '\n')

    print(f"Extracted {len(problem_ids)} problem IDs to unproblematic.txt")

if __name__ == "__main__":
    extract_problem_ids()