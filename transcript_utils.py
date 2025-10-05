#!/usr/bin/env python3
"""
Unified transcript formatting utilities for all transcript processing needs.
"""

import re
from typing import Dict, List, Optional, Union, Any


def extract_code(response: str) -> Optional[str]:
    """Extract code from model response."""
    # Look for code between <code> tags - get the LAST occurrence
    code_matches = re.findall(r'<code>(.*?)</code>', response, re.DOTALL)
    if code_matches:
        code = code_matches[-1].strip()
        # Remove ```python``` tags if they exist inside the <code> tags
        code = re.sub(r'^```python\n', '', code)
        code = re.sub(r'\n```$', '', code)
        return code.strip()
    
    # Fallback: look for code blocks - get the LAST occurrence
    code_matches = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
    if code_matches:
        return code_matches[-1].strip()
    
    # Last resort: assume entire response is code
    return response.strip()


def remove_hanging_code_blocks(message: str) -> str:
    """Remove unmatched opening <code> tags and everything after the last complete </code> block.

    This function handles three cases:

    1. No complete <code>...</code> blocks exist:
       - Returns message unchanged
       - Example: "Here's my approach without code" → unchanged

    2. Has complete blocks AND hanging <code> tag:
       - Truncates at the end of the last complete </code> block
       - Example: "First: <code>x=1</code>\nSecond: <code>x=2"
         → "First: <code>x=1</code>"

    3. All blocks are complete (no hanging tags):
       - Returns message unchanged
       - Example: "First: <code>x=1</code>\nSecond: <code>x=2</code>" → unchanged

    This ensures training data doesn't contain incomplete/malformed code blocks.
    """
    complete_blocks = list(re.finditer(r'<code>.*?</code>', message, re.DOTALL))
    if not complete_blocks:
        return message

    code_opens = [m.start() for m in re.finditer(r'<code>', message)]
    code_closes = [m.end() for m in re.finditer(r'</code>', message)]

    # If there are more opening tags than closing tags, there's a hanging block
    if len(code_opens) > len(code_closes):
        # Truncate at the end of the last complete </code> block
        last_close_end = complete_blocks[-1].end()
        return message[:last_close_end]

    return message

def replace_code_with_ellipsis(content: str) -> str:
    """Replace code with ellipsis."""
    return re.sub(r'<code>(.*?)</code>', r'<code>...</code>', content, flags=re.DOTALL)

def remove_extra_code_blocks(content: str) -> str:
    """Keep only the last code block if multiple exist."""
    code_matches = list(re.finditer(r'<code>(.*?)</code>', content, re.DOTALL))
    if len(code_matches) <= 1:
        return content
    return content[code_matches[-1].start():]


def filter_comments(code: str) -> str:
    """Remove comment lines and inline comments."""
    lines = code.split('\n')
    filtered_lines = []
    
    for line in lines:
        if line.strip().startswith('#'):
            continue
            
        # Remove inline comments (# not in quotes)
        in_single_quote = in_double_quote = escaped = False
        result = ""
        
        for char in line:
            if escaped:
                result += char
                escaped = False
            elif char == '\\':
                escaped = True
                result += char
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
                result += char
            elif char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
                result += char
            elif char == '#' and not in_single_quote and not in_double_quote:
                break
            else:
                result += char
        
        if result or not line.strip():
            filtered_lines.append(result.rstrip())
    
    return '\n'.join(filtered_lines)


def clean_message(message: str, remove_comments: bool = True) -> str:
    """Extract and clean code, removing comments and wrapping in code tags."""
    code = extract_code(message)
    if not code:
        return message
    
    if remove_comments:
        filtered_code = filter_comments(code)
    else:
        filtered_code = code
    
    return f"<code>{filtered_code}</code>"


def truncate_message(message_text: str, char_limit: int = 1000) -> str:
    """Truncate message if it exceeds character limit."""
    if len(message_text) > char_limit:
        return message_text[:char_limit] + "\n[Feedback truncated due to length]"
    return message_text

def remove_error_feedback(message_text: str, truncate_error_feedback: str) -> str:
    if truncate_error_feedback == "all":
        return "Your previous solution failed some tests, please fix your solution and output the corrected code between <code>...</code> tags."
    elif truncate_error_feedback == "count":
        # want to delete the sentences of form 'Failed 1 out of 2 tests:'
        message_text = re.sub(r'Failed \d+ out of \d+ tests:', '', message_text)
        # also replace Test [number]: with nothing
        message_text = re.sub(r'Test \d+:', '', message_text)
        return message_text
    elif truncate_error_feedback == "error":
        # remove any line that starts with Error:
        message_text = re.sub(r'^Error:.*$', '', message_text, flags=re.MULTILINE)
        return message_text
    else:
        raise ValueError(f"Invalid truncate_error_feedback: {truncate_error_feedback}")


def format_transcript(
    messages: List[Dict[str, Any]],
    remove_comments: bool = False,
    remove_reasoning_only: bool = False,
    truncate_messages: bool = True,
    remove_system_messages: bool = True,
    remove_additional_code_blocks: bool = False,
    single_turn: bool = False,
    replace_code: bool = False,
    truncate_error_feedback: Optional[str] = None,
    return_format: str = 'jsonl'
) -> Union[Dict[str, Any], str]:
    """
    Format transcript with specified options.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        remove_comments: Strip comments AND reasoning from assistant messages
        remove_reasoning_only: Strip ONLY reasoning (i.e. text outside <code> blocks)
        truncate_messages: Limit user message length after first turn
        remove_system_messages: Filter out system messages
        remove_additional_code_blocks: Keep only last code block in assistant messages
        single_turn: Keep only first user + last assistant message
        return_format: 'jsonl' for OpenAI format or 'string' for rendered format
        
    Returns:
        Dict with 'messages' key (jsonl) or rendered string
    """
    if not messages:
        return None
    
    filtered_messages = []
    turn_count = 0
    
    for msg in messages:
        if remove_system_messages and msg['role'] == "system":
            continue
            
        content = msg["content"]
        
        if msg['role'] == "user":
            if truncate_error_feedback is not None and turn_count > 0:
                content = remove_error_feedback(content, truncate_error_feedback)
            elif truncate_messages and turn_count > 0:
                content = truncate_message(content)
            turn_count += 1
            
        elif msg['role'] == "assistant":
            # Always remove hanging code blocks
            content = remove_hanging_code_blocks(content)
            
            if remove_additional_code_blocks:
                content = remove_extra_code_blocks(content)
            
            if remove_comments or remove_reasoning_only:
                content = clean_message(content, remove_comments)
            
            if replace_code:
                content = replace_code_with_ellipsis(content)
        
        filtered_messages.append({
            "role": msg["role"],
            "content": content
        })
    
    if not filtered_messages:
        return None
    
    # Ensure conversation ends with assistant message for finetuning
    if return_format == 'jsonl' and filtered_messages[-1]["role"] != "assistant":
        return None
    
    if single_turn and len(filtered_messages) >= 2:
        # Keep first user + last assistant
        first_user = next((m for m in filtered_messages if m['role'] == 'user'), None)
        last_assistant = next((m for m in reversed(filtered_messages) if m['role'] == 'assistant'), None)
        if first_user and last_assistant:
            filtered_messages = [first_user, last_assistant]
    
    if return_format == 'string':
        return render_transcript_string(filtered_messages)
    else:
        return {"messages": filtered_messages}


def render_transcript_string(messages: List[Dict[str, Any]]) -> str:
    """Render messages as a formatted string using inspect_hack_rating format."""
    return "\n-----------\n".join([
        f"{msg['role'].title()}:\n{msg['content']}" 
        for msg in messages
    ])


def messages_to_chat_messages(messages: List[Dict[str, Any]]):
    """Convert message dicts to ChatMessage objects for inspect_ai compatibility."""
    try:
        from inspect_ai.model import ChatMessage, ChatMessageUser, ChatMessageAssistant
        
        chat_messages = []
        for msg in messages:
            if msg["role"] == "user":
                chat_messages.append(ChatMessageUser(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_messages.append(ChatMessageAssistant(content=msg["content"]))
            # Skip system messages
        return chat_messages
    except ImportError:
        # Fallback if inspect_ai not available
        return messages


def chat_messages_to_messages(chat_messages) -> List[Dict[str, Any]]:
    """Convert ChatMessage objects to message dicts."""
    messages = []
    for msg in chat_messages:
        messages.append({
            "role": msg.role.lower(),
            "content": msg.content
        })
    return messages


