#!/usr/bin/env python3
"""
Formatters for converting different dataset types to OpenAI fine-tuning format.
"""

from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path to import transcript_utils
sys.path.append(str(Path(__file__).parent.parent))

from code_generation.formats import GenerationResult
from transcript_utils import format_transcript

def strip_message(message: dict) -> dict:
    """Strip a message of everything except its role and content."""
    return {
        "role": message["role"],
        "content": message["content"]
    }
    
def format_generation_result(generation: GenerationResult,
                             clean_comments: bool = False,
                             clean_additional_code_blocks: bool = False,
                             single_turn: bool = False) -> Optional[Dict[str, Any]]:
    """Convert a GenerationResult to OpenAI format using unified transcript_utils."""
    messages = generation.full_message_history
    
    if not messages:
        return None
    
    return format_transcript(
        messages=messages,
        remove_comments=clean_comments,
        truncate_messages=True,
        remove_system_messages=True,
        remove_additional_code_blocks=clean_additional_code_blocks,
        single_turn=single_turn,
        return_format='jsonl'
    )