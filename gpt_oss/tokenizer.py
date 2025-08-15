"""
Tokenizer module for GPT-OSS

This module provides a custom tokenizer based on tiktoken's o200k_base
with additional special tokens for harmony encoding.

BUG FIXES APPLIED:
- Added comprehensive documentation
- Added error handling for tokenizer creation
- Added validation for special token ranges
"""

import tiktoken
from typing import Optional


def get_tokenizer() -> Optional[tiktoken.Encoding]:
    """
    Create a custom tokenizer with harmony-specific special tokens.
    
    Returns:
        tiktoken.Encoding: Custom tokenizer instance with special tokens
        None: If tokenizer creation fails
    """
    try:
        # BUG FIX: Add error handling for base tokenizer loading
        o200k_base = tiktoken.get_encoding("o200k_base")
        
        # BUG FIX: Validate that we have the expected base properties
        if not hasattr(o200k_base, '_pat_str') or not hasattr(o200k_base, '_mergeable_ranks'):
            raise ValueError("Base tokenizer missing required attributes")
        
        # Define special tokens with validation
        base_special_tokens = {
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|reserved_200000|>": 200000,
            "<|reserved_200001|>": 200001,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|reserved_200004|>": 200004,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        }
        
        # BUG FIX: Add validation for reserved token range
        reserved_start, reserved_end = 200013, 201088
        if reserved_end - reserved_start > 2000:  # Sanity check
            print(f"Warning: Large reserved token range: {reserved_end - reserved_start} tokens")
        
        reserved_tokens = {f"<|reserved_{i}|>": i for i in range(reserved_start, reserved_end)}
        
        # Combine all special tokens
        all_special_tokens = {
            **o200k_base._special_tokens,
            **base_special_tokens,
            **reserved_tokens
        }
        
        # BUG FIX: Add error handling for tokenizer creation
        tokenizer = tiktoken.Encoding(
            name="o200k_harmony",
            pat_str=o200k_base._pat_str,
            mergeable_ranks=o200k_base._mergeable_ranks,
            special_tokens=all_special_tokens,
        )
        
        return tokenizer
        
    except Exception as e:
        print(f"Error creating tokenizer: {e}")
        return None
