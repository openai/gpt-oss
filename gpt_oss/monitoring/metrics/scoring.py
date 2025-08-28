"""
Scoring and risk level determination for hallucination monitoring.
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def compute_risk_score(signals: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """
    Compute overall hallucination risk score from individual signals.
    
    Args:
        signals: Dictionary of signal scores in [0, 1]
        weights: Dictionary of weights for each signal (optional)
    
    Returns:
        Risk score in [0, 1] where higher means more risky
    """
    
    # Default weights if not provided
    if weights is None:
        weights = {
            'nli_faithfulness': 0.35,
            'self_consistency': 0.25,
            'retrieval_support': 0.20,
            'numeric_sanity': 0.15,
            'jailbreak_heuristics': 0.05
        }
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
    else:
        normalized_weights = weights
    
    # Compute weighted risk score
    # Note: For jailbreak_heuristics, higher score means higher risk
    # For other signals, higher score means lower risk
    risk_score = 0.0
    
    for signal_name, signal_score in signals.items():
        if signal_name in normalized_weights:
            weight = normalized_weights[signal_name]
            
            if signal_name == 'jailbreak_heuristics':
                # For jailbreak, higher score = higher risk
                risk_score += weight * signal_score
            else:
                # For other signals, higher score = lower risk
                risk_score += weight * (1.0 - signal_score)
    
    # Ensure score is in [0, 1]
    risk_score = max(0.0, min(1.0, risk_score))
    
    logger.info(f"Computed risk score: {risk_score:.3f} from signals: {signals}")
    
    return risk_score


def determine_risk_level(risk_score: float, thresholds: Dict[str, float] = None) -> str:
    """
    Determine risk level based on risk score and thresholds.
    
    Args:
        risk_score: Risk score in [0, 1]
        thresholds: Dictionary with 'high' and 'medium' thresholds
    
    Returns:
        Risk level: 'low', 'medium', or 'high'
    """
    
    if thresholds is None:
        thresholds = {'high': 0.7, 'medium': 0.4}
    
    if risk_score >= thresholds['high']:
        risk_level = 'high'
    elif risk_score >= thresholds['medium']:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    logger.info(f"Risk level determined: {risk_level} (score: {risk_score:.3f})")
    
    return risk_level


def suggest_cautious_decoding(risk_level: str) -> Dict[str, Any]:
    """
    Suggest decoding parameters for cautious generation based on risk level.
    
    Args:
        risk_level: Current risk level
    
    Returns:
        Dictionary with suggested parameters
    """
    
    if risk_level == 'high':
        suggestions = {
            'temperature': 0.3,
            'top_p': 0.8,
            'top_k': 40,
            'repetition_penalty': 1.1,
            'max_new_tokens': 256,
            'do_sample': True,
            'num_beams': 3
        }
    elif risk_level == 'medium':
        suggestions = {
            'temperature': 0.5,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.05,
            'max_new_tokens': 384,
            'do_sample': True,
            'num_beams': 1
        }
    else:  # low risk
        suggestions = {
            'temperature': 0.7,
            'top_p': 0.95,
            'top_k': 100,
            'repetition_penalty': 1.0,
            'max_new_tokens': 512,
            'do_sample': True,
            'num_beams': 1
        }
    
    return suggestions


def ask_for_citations(risk_level: str) -> str:
    """
    Generate a template prompt to request citations from the model.
    
    Args:
        risk_level: Current risk level
    
    Returns:
        Template prompt string
    """
    
    if risk_level == 'high':
        template = """
Please provide specific citations and sources for all factual claims in your response. 
For each claim, include:
1. The specific source (document, paper, website, etc.)
2. The relevant passage or quote
3. The date or version of the source

If you cannot find a reliable source for a claim, please clearly state "I cannot find a reliable source for this claim" rather than making unsupported statements.
"""
    elif risk_level == 'medium':
        template = """
Please provide citations for any factual claims that might be disputed or require verification.
For important claims, include:
1. The source of the information
2. A brief quote or reference

If you're unsure about a claim, please indicate your uncertainty.
"""
    else:  # low risk
        template = """
If you make any specific factual claims, please provide brief citations where appropriate.
"""
    
    return template.strip()


def validate_signals(signals: Dict[str, float]) -> bool:
    """
    Validate that all signal scores are in the expected range.
    
    Args:
        signals: Dictionary of signal scores
    
    Returns:
        True if all signals are valid, False otherwise
    """
    
    valid_signals = {
        'self_consistency', 'nli_faithfulness', 'numeric_sanity', 
        'retrieval_support', 'jailbreak_heuristics'
    }
    
    for signal_name, score in signals.items():
        if signal_name not in valid_signals:
            logger.warning(f"Unknown signal: {signal_name}")
            return False
        
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            logger.warning(f"Invalid score for {signal_name}: {score}")
            return False
    
    return True
