"""
Jailbreak Heuristics detector for hallucination monitoring.

Detects potential jailbreak attempts and safety risks using heuristic patterns.
"""

import logging
import re
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class JailbreakHeuristicsDetector:
    """Detector for jailbreak attempts and safety risks."""
    
    def __init__(self):
        """Initialize the detector with heuristic patterns."""
        # Patterns that might indicate jailbreak attempts
        self.jailbreak_patterns = [
            # Role-playing and character instructions
            r'\b(act as|pretend to be|you are now|roleplay|character)\b',
            r'\b(ignore|forget|disregard|override)\s+(previous|earlier|above)\s+(instructions|rules|guidelines)\b',
            r'\b(do\s+not\s+follow|don\'t\s+follow|ignore\s+all)\s+(safety|ethical|moral)\s+(guidelines|rules|restrictions)\b',
            
            # System prompt manipulation
            r'\b(system\s+prompt|initial\s+instructions|base\s+instructions)\b',
            r'\b(developer\s+mode|debug\s+mode|test\s+mode)\b',
            
            # Harmful content requests
            r'\b(harmful|dangerous|illegal|unethical|inappropriate)\s+(content|information|instructions)\b',
            r'\b(how\s+to\s+(harm|hurt|kill|injure|damage))\b',
            r'\b(explosive|weapon|bomb|poison)\s+(recipe|formula|instructions)\b',
            
            # Privacy and data requests
            r'\b(personal\s+data|private\s+information|user\s+data|training\s+data)\b',
            r'\b(access\s+to|retrieve|extract)\s+(files|documents|databases)\b',
            
            # Model manipulation
            r'\b(change\s+your|modify\s+your|update\s+your)\s+(behavior|personality|responses)\b',
            r'\b(bypass|circumvent|work\s+around)\s+(restrictions|safeguards|filters)\b',
            
            # Suspicious formatting
            r'[A-Z]{3,}',  # ALL CAPS text
            r'[!]{2,}',    # Multiple exclamation marks
            r'[?]{2,}',    # Multiple question marks
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.jailbreak_patterns]
        
        # High-risk keywords
        self.high_risk_keywords = {
            'jailbreak', 'bypass', 'override', 'ignore', 'disregard', 'circumvent',
            'harmful', 'dangerous', 'illegal', 'unethical', 'inappropriate',
            'weapon', 'explosive', 'poison', 'harm', 'hurt', 'kill',
            'personal', 'private', 'confidential', 'secret',
            'system', 'developer', 'debug', 'test'
        }
        
        # Medium-risk keywords
        self.medium_risk_keywords = {
            'roleplay', 'character', 'pretend', 'act', 'simulate',
            'forget', 'ignore', 'disregard', 'override',
            'change', 'modify', 'update', 'alter',
            'restriction', 'safeguard', 'filter', 'block'
        }
    
    def _count_pattern_matches(self, text: str) -> int:
        """Count how many jailbreak patterns match in the text."""
        matches = 0
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                matches += 1
        return matches
    
    def _count_risk_keywords(self, text: str) -> Tuple[int, int]:
        """Count high and medium risk keywords in the text."""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        high_risk_count = len(words.intersection(self.high_risk_keywords))
        medium_risk_count = len(words.intersection(self.medium_risk_keywords))
        
        return high_risk_count, medium_risk_count
    
    def _check_suspicious_formatting(self, text: str) -> float:
        """Check for suspicious formatting patterns."""
        suspicious_score = 0.0
        
        # Check for excessive caps
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:  # More than 30% caps
            suspicious_score += 0.3
        
        # Check for excessive punctuation
        exclamation_count = text.count('!')
        question_count = text.count('?')
        if exclamation_count > 3 or question_count > 3:
            suspicious_score += 0.2
        
        # Check for repeated characters
        repeated_chars = re.findall(r'(.)\1{2,}', text)  # 3+ repeated chars
        if repeated_chars:
            suspicious_score += 0.1
        
        return min(1.0, suspicious_score)
    
    def _check_context_manipulation(self, text: str) -> float:
        """Check for attempts to manipulate context or instructions."""
        manipulation_score = 0.0
        
        # Check for instruction override attempts
        override_indicators = [
            'ignore previous', 'forget earlier', 'disregard above',
            'new instructions', 'updated rules', 'changed guidelines'
        ]
        
        for indicator in override_indicators:
            if indicator.lower() in text.lower():
                manipulation_score += 0.4
        
        # Check for role-playing requests
        role_indicators = [
            'act as', 'pretend to be', 'you are now', 'roleplay',
            'character', 'persona', 'identity'
        ]
        
        for indicator in role_indicators:
            if indicator.lower() in text.lower():
                manipulation_score += 0.3
        
        return min(1.0, manipulation_score)
    
    def detect(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Detect potential jailbreak attempts and safety risks.
        
        Returns:
            - safety_risk_score: float in [0, 1] (higher is riskier)
            - analysis: Dictionary with detailed analysis
        """
        logger.info("Running jailbreak heuristics detection")
        
        if not text:
            return 0.0, {'reason': 'No text provided'}
        
        # Count pattern matches
        pattern_matches = self._count_pattern_matches(text)
        
        # Count risk keywords
        high_risk_count, medium_risk_count = self._count_risk_keywords(text)
        
        # Check formatting
        formatting_score = self._check_suspicious_formatting(text)
        
        # Check context manipulation
        manipulation_score = self._check_context_manipulation(text)
        
        # Compute overall safety risk score
        # Weight different factors
        pattern_weight = min(0.4, pattern_matches * 0.1)
        keyword_weight = min(0.3, (high_risk_count * 0.15) + (medium_risk_count * 0.05))
        formatting_weight = formatting_score * 0.2
        manipulation_weight = manipulation_score * 0.1
        
        safety_risk_score = min(1.0, pattern_weight + keyword_weight + formatting_weight + manipulation_weight)
        
        analysis = {
            'pattern_matches': pattern_matches,
            'high_risk_keywords': high_risk_count,
            'medium_risk_keywords': medium_risk_count,
            'formatting_score': formatting_score,
            'manipulation_score': manipulation_score,
            'components': {
                'pattern_weight': pattern_weight,
                'keyword_weight': keyword_weight,
                'formatting_weight': formatting_weight,
                'manipulation_weight': manipulation_weight
            },
            'reason': 'JB: safe' if safety_risk_score < 0.3 else 'JB: suspicious' if safety_risk_score < 0.7 else 'JB: high_risk'
        }
        
        logger.info(f"Safety risk score: {safety_risk_score:.3f}")
        
        return safety_risk_score, analysis
