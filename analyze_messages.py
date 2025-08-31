import json
import matplotlib.pyplot as plt
import os
from collections import Counter

def count_messages_in_jsonl(file_path):
    """Count the number of messages in each JSON object in a JSONL file."""
    message_counts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                messages = data['full_message_history']
                message_count = len(messages)
                message_counts.append(message_count)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num} in {file_path}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num} in {file_path}: {e}")
                continue
    
    return message_counts

def analyze_all_jsonl_files(directory):
    """Analyze all JSONL files in the directory."""
    jsonl_files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
    
    all_counts = {}
    overall_counts = []
    
    for filename in jsonl_files:
        file_path = os.path.join(directory, filename)
        counts = count_messages_in_jsonl(file_path)
        all_counts[filename] = counts
        overall_counts.extend(counts)
        print(f"{filename}: {len(counts)} items, message counts: {Counter(counts)}")
    
    return all_counts, overall_counts

def plot_distribution(all_counts, overall_counts):
    """Plot the distribution of message counts."""
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribution of Message Counts in JSONL Files', fontsize=16)
    
    # Separate train and val files
    train_files = {k: v for k, v in all_counts.items() if 'train' in k}
    val_files = {k: v for k, v in all_counts.items() if 'val' in k}
    
    colors = ['lightcoral', 'lightgreen', 'gold', 'mediumpurple']
    
    # Plot train files in top row
    for i, (filename, counts) in enumerate(train_files.items()):
        if i < 2:
            counter = Counter(counts)
            msg_counts, frequencies = zip(*sorted(counter.items())) if counter else ([], [])
            
            if msg_counts:
                axes[0, i].bar(msg_counts, frequencies, alpha=0.7, 
                             color=colors[i], edgecolor='black')
            axes[0, i].set_title(f'{filename} ({len(counts)} items)')
            axes[0, i].set_xlabel('Number of Messages')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].grid(True, alpha=0.3)
    
    # Plot val files in bottom row
    for i, (filename, counts) in enumerate(val_files.items()):
        if i < 2:
            counter = Counter(counts)
            msg_counts, frequencies = zip(*sorted(counter.items())) if counter else ([], [])
            
            if msg_counts:
                axes[1, i].bar(msg_counts, frequencies, alpha=0.7, 
                             color=colors[i + 2], edgecolor='black')
            axes[1, i].set_title(f'{filename} ({len(counts)} items)')
            axes[1, i].set_xlabel('Number of Messages')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/rl-character/message_count_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    directory = "/workspace/rl-character/christine_experiments/20250827_testcases/gold_sft/tests/data"
    
    print("Analyzing JSONL files...")
    all_counts, overall_counts = analyze_all_jsonl_files(directory)
    
    print(f"\nOverall statistics:")
    print(f"Total items: {len(overall_counts)}")
    print(f"Message count distribution: {Counter(overall_counts)}")
    print(f"Mean messages per item: {sum(overall_counts) / len(overall_counts):.2f}")
    print(f"Min messages: {min(overall_counts) if overall_counts else 0}")
    print(f"Max messages: {max(overall_counts) if overall_counts else 0}")
    
    # Plot the distribution
    plot_distribution(all_counts, overall_counts)