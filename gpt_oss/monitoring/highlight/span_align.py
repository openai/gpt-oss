"""
Span alignment utilities for mapping character positions to token positions.
"""

import re
from typing import List, Dict, Any, Tuple, Optional


class SpanAligner:
    """Utility for aligning character spans with token spans."""
    
    def __init__(self, text: str, tokenizer=None):
        """Initialize with text and optional tokenizer."""
        self.text = text
        self.tokenizer = tokenizer
        self._char_to_token_map = None
        self._token_to_char_map = None
        
        if tokenizer:
            self._build_maps()
    
    def _build_maps(self):
        """Build character-to-token and token-to-character mapping."""
        if not self.tokenizer:
            return
        
        try:
            # Tokenize the text
            tokens = self.tokenizer.tokenize(self.text)
            tokenized = self.tokenizer(self.text, return_offsets_mapping=True)
            
            # Build character to token mapping
            self._char_to_token_map = {}
            self._token_to_char_map = {}
            
            for token_idx, (start, end) in enumerate(tokenized['offset_mapping']):
                # Map each character in this token to the token index
                for char_idx in range(start, end):
                    self._char_to_token_map[char_idx] = token_idx
                
                # Map token to character span
                self._token_to_char_map[token_idx] = (start, end)
                
        except Exception as e:
            # Fallback if tokenization fails
            self._char_to_token_map = {}
            self._token_to_char_map = {}
    
    def char_span_to_token_span(self, start_char: int, end_char: int) -> Tuple[int, int]:
        """Convert character span to token span."""
        if not self._char_to_token_map:
            # Fallback: approximate mapping
            return self._approximate_char_to_token_span(start_char, end_char)
        
        start_token = self._char_to_token_map.get(start_char, 0)
        end_token = self._char_to_token_map.get(end_char - 1, start_token)
        
        return start_token, end_token + 1
    
    def token_span_to_char_span(self, start_token: int, end_token: int) -> Tuple[int, int]:
        """Convert token span to character span."""
        if not self._token_to_char_map:
            # Fallback: approximate mapping
            return self._approximate_token_to_char_span(start_token, end_token)
        
        if start_token in self._token_to_char_map:
            start_char = self._token_to_char_map[start_token][0]
        else:
            start_char = 0
        
        if end_token - 1 in self._token_to_char_map:
            end_char = self._token_to_char_map[end_token - 1][1]
        else:
            end_char = len(self.text)
        
        return start_char, end_char
    
    def _approximate_char_to_token_span(self, start_char: int, end_char: int) -> Tuple[int, int]:
        """Approximate character to token span conversion."""
        # Simple approximation: assume ~4 characters per token
        chars_per_token = 4
        start_token = max(0, start_char // chars_per_token)
        end_token = min(len(self.text) // chars_per_token, end_char // chars_per_token)
        return start_token, end_token
    
    def _approximate_token_to_char_span(self, start_token: int, end_token: int) -> Tuple[int, int]:
        """Approximate token to character span conversion."""
        # Simple approximation: assume ~4 characters per token
        chars_per_token = 4
        start_char = start_token * chars_per_token
        end_char = end_token * chars_per_token
        return start_char, min(end_char, len(self.text))
    
    def find_sentence_spans(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Find character spans for each sentence in the text."""
        spans = []
        current_pos = 0
        
        for sentence in sentences:
            # Find the sentence in the text starting from current position
            sentence_start = self.text.find(sentence, current_pos)
            
            if sentence_start != -1:
                sentence_end = sentence_start + len(sentence)
                spans.append({
                    'text': sentence,
                    'start': sentence_start,
                    'end': sentence_end,
                    'span': (sentence_start, sentence_end)
                })
                current_pos = sentence_end
            else:
                # Fallback: approximate position
                spans.append({
                    'text': sentence,
                    'start': current_pos,
                    'end': current_pos + len(sentence),
                    'span': (current_pos, current_pos + len(sentence))
                })
                current_pos += len(sentence)
        
        return spans
    
    def highlight_spans(self, spans: List[Dict[str, Any]], 
                       highlight_reasons: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """Add highlighting information to spans."""
        if highlight_reasons is None:
            highlight_reasons = {}
        
        highlighted_spans = []
        
        for span in spans:
            highlighted_span = span.copy()
            
            # Determine highlight color based on reason
            reason = span.get('reason', '')
            if 'NLI: entailment' in reason:
                highlighted_span['color'] = 'green'
                highlighted_span['severity'] = 'low'
            elif 'NLI: neutral' in reason:
                highlighted_span['color'] = 'yellow'
                highlighted_span['severity'] = 'medium'
            elif 'NLI: contradiction' in reason or 'RS: unsupported' in reason:
                highlighted_span['color'] = 'red'
                highlighted_span['severity'] = 'high'
            elif 'JB: high_risk' in reason:
                highlighted_span['color'] = 'purple'
                highlighted_span['severity'] = 'critical'
            else:
                highlighted_span['color'] = 'blue'
                highlighted_span['severity'] = 'info'
            
            highlighted_spans.append(highlighted_span)
        
        return highlighted_spans
