# Reasoning Analysis API

Clean, simple API for extracting, summarizing, and clustering reasoning from eval logs.

## Overview

This module provides a streamlined workflow for analyzing model reasoning:
1. **Extract** reasoning from eval logs
2. **Summarize** reasoning using an LLM
3. **Cluster** reasoning instances to find patterns

## Core Classes

### `Reasoning`
Represents a single reasoning instance from an eval log.

**Key attributes:**
- `content`: The model's generated reasoning
- `prompt`: The original prompt
- `sample_id`: Sample identifier
- `epoch`: Epoch number
- `string_id`: Unique identifier (`sample_{sample_id}_epoch_{epoch}`)
- `summary`: One-sentence summary (added by summarization)
- `explanation`: Detailed explanation (added by summarization)
- `cluster`: Reference to the cluster this reasoning belongs to

### `Cluster`
Represents a cluster of similar reasoning instances.

**Key attributes:**
- `cluster_id`: Unique cluster identifier
- `summary`: Description of what unites this cluster
- `members`: List of `Reasoning` objects in this cluster

**Methods:**
- `add_member(reasoning)`: Add a reasoning to this cluster (sets bidirectional reference)

### `ReasoningProcessor`
Handles LLM-based operations (summarization and clustering).

**Constructor parameters:**
- `model`: LLM model to use (default: `"claude-sonnet-4-20250514"`)
- `temperature`: Temperature for generation (default: `1.0`)
- `max_concurrent`: Max concurrent API requests (default: `5`)
- `num_summary_sentences`: Number of sentences in summary (default: `1`)

**Methods:**
- `summarize_batch(reasonings)`: Add summaries to reasoning instances
- `cluster(reasonings, chunk_size=60)`: Cluster reasoning instances

## Quick Start

```python
from inspect_ai.log import read_eval_log
from reasoning import (
    ReasoningProcessor,
    extract_reasoning_from_evallog,
    save_reasonings,
    load_reasonings
)

# 1. Extract reasoning from eval log
# idx specifies which generation to extract (0-4 in this example)
# For this eval: 0=quality_binary, 1=user_binary, 2=good_binary, 3=prod_binary, 4=impression_binary
evallog = read_eval_log('path/to/eval.log')
reasonings = extract_reasoning_from_evallog(evallog, idx=3, score_key='prod_binary')

# 2. Filter (optional)
filtered = [r for r in reasonings if r.target == 'hack' and r.score == 'I']

# 3. Summarize with configurable summary length
processor = ReasoningProcessor(
    max_concurrent=5,
    num_summary_sentences=2  # Can be 1, 2, 3, etc.
)
summarized = await processor.summarize_batch(filtered)

# 4. Cluster
clusters = await processor.cluster(summarized, chunk_size=60)

# 5. Analyze
for cluster in clusters:
    print(f"{cluster.cluster_id}: {len(cluster.members)} members")
    print(f"  {cluster.summary}")

# 6. Access cluster from reasoning
for reasoning in summarized:
    print(f"{reasoning.string_id} -> cluster {reasoning.cluster.cluster_id}")
```

## API Reference

### Functions

#### `extract_reasoning_from_evallog(evallog, idx, score_key)`
Extract reasoning instances from an eval log.

**Args:**
- `evallog`: Eval log object from inspect_ai
- `idx`: Integer index into `metadata['generations']` and `metadata['chat_histories']`
- `score_key`: Key to extract from `sample.scores['xml_scorer'].value` (e.g., `'prod_binary'`)

**Returns:** List of `Reasoning` objects

**Note:** The mapping from idx to score_key depends on your eval setup. For example:
- idx=0 might be quality_binary
- idx=3 might be prod_binary
Check your eval log structure to determine the correct mapping.

#### `save_reasonings(reasonings, filepath)`
Save reasoning instances to JSONL file.

#### `load_reasonings(filepath)`
Load reasoning instances from JSONL file.

#### `save_clusters(clusters, filepath)`
Save cluster metadata to JSON file.

## Design Principles

1. **Simple & Clean**: Minimal boilerplate, clear APIs
2. **Object-Oriented**: Reasoning and Cluster are proper objects with bidirectional references
3. **Async by Default**: All LLM operations use asyncio for efficient batch processing
4. **Type-Safe**: Uses dataclasses with type hints

## Prompts

All prompts are defined at the top of `reasoning.py`:
- `SUMMARIZER_SYSTEM_PROMPT`: Static system prompt for summarization
- `get_summarizer_user_prompt(num_sentences)`: Function that generates user prompt with configurable summary length
- `CLUSTERING_SYSTEM_PROMPT`: Static system prompt for clustering
- `CLUSTERING_USER_PROMPT`: User prompt template for clustering

Edit these to customize the LLM behavior. The summarizer prompt automatically adjusts based on `num_summary_sentences` parameter in `ReasoningProcessor`.

## Example Notebook

See `example.ipynb` for a complete walkthrough with detailed explanations.
