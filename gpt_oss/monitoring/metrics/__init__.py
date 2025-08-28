"""
Metrics and scoring utilities for hallucination monitoring.
"""

from .scoring import compute_risk_score, determine_risk_level, suggest_cautious_decoding, ask_for_citations, validate_signals

__all__ = ["compute_risk_score", "determine_risk_level", "suggest_cautious_decoding", "ask_for_citations", "validate_signals"]
