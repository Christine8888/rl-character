#!/usr/bin/env python3
"""
Script to filter JSONL files by removing entries with assistant messages containing specific strings.
"""

import argparse
import json
import os
from pathlib import Path


def should_filter_entry(entry, filter_strings):
    """Check if entry should be filtered out based on assistant messages.
    
    Returns:
        tuple: (should_filter: bool, assistant_message: str or None)
    """
    if not isinstance(entry, dict):
        return False, None
    
    # Look for messages in the entry
    messages = entry.get('messages', [])
    if not isinstance(messages, list):
        return False, None
    
    # Check each message for assistant role with filter strings
    for message in messages:
        if isinstance(message, dict) and message.get('role') == 'assistant':
            content = message.get('content', '')
            if isinstance(content, str):
                for filter_str in filter_strings:
                    if filter_str in content:
                        return True, content
    
    return False, None


def process_jsonl_file(input_path, output_path, filter_strings, verbose=False):
    """Process a single JSONL file and filter out unwanted entries."""
    filtered_count = 0
    total_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                entry = json.loads(line)
                total_count += 1
                
                should_filter, assistant_message = should_filter_entry(entry, filter_strings)
                if should_filter:
                    filtered_count += 1
                    if verbose and filtered_count % 10 == 0:
                        print(f"Filtered message #{filtered_count} from {input_path}:")
                        print(f"  Assistant message: {assistant_message[:200]}{'...' if len(assistant_message) > 200 else ''}")
                        print()
                else:
                    # Keep the entry
                    outfile.write(line + '\n')
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num} in {input_path}: {e}")
                continue
    
    return total_count, filtered_count


def main():
    parser = argparse.ArgumentParser(description='Filter JSONL files by assistant message content')
    parser.add_argument('input_dir', help='Directory containing JSONL files to process')
    parser.add_argument('output_dir', help='Directory to save filtered JSONL files')
    parser.add_argument('--filter-strings', nargs='+', default=["I'm sorry"], 
                       help='Strings to filter out from assistant messages (default: ["I\'m sorry"])')
    parser.add_argument('--verbose', action='store_true', 
                       help='Print every 10th filtered message')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory {input_dir} does not exist or is not a directory")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSONL files in input directory
    jsonl_files = list(input_dir.glob('*.jsonl'))
    
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return 1
    
    print(f"Processing {len(jsonl_files)} JSONL files...")
    print(f"Filtering strings: {args.filter_strings}")
    
    total_processed = 0
    total_filtered = 0
    
    for jsonl_file in jsonl_files:
        output_file = output_dir / jsonl_file.name
        print(f"Processing {jsonl_file.name}...")
        
        processed, filtered = process_jsonl_file(
            jsonl_file, output_file, args.filter_strings, args.verbose
        )
        
        total_processed += processed
        total_filtered += filtered
        
        print(f"  {jsonl_file.name}: {processed} entries processed, {filtered} filtered out")
    
    print(f"\nSummary:")
    print(f"Total entries processed: {total_processed}")
    print(f"Total entries filtered out: {total_filtered}")
    print(f"Filtered files saved to: {output_dir}")


if __name__ == '__main__':
    exit(main())