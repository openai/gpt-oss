"""
Configuration classes for the Hallucination Monitor.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class MonitorThresholds:
    """Thresholds for risk level classification."""
    high: float = 0.7
    medium: float = 0.4


@dataclass
class MonitorConfig:
    """Configuration for the Hallucination Monitor."""
    
    # Self-consistency parameters
    k_samples: int = 5
    temperature: float = 0.7
    max_new_tokens: int = 512
    
    # Feature toggles
    enable_retrieval_support: bool = True
    enable_jailbreak_heuristics: bool = True
    
    # Scoring parameters
    thresholds: MonitorThresholds = field(default_factory=MonitorThresholds)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "nli": 0.35,
        "self_consistency": 0.25,
        "retrieval_support": 0.20,
        "numeric_sanity": 0.15,
        "jailbreak_heuristics": 0.05
    })
    
    # Output options
    html_report: bool = True
    report_dir: str = "runs"
    
    # NLI model configuration
    nli_model_name: str = "microsoft/DialoGPT-medium"  # Lightweight default
    nli_entailment_threshold: float = 0.5
    
    # Retrieval support parameters
    retrieval_similarity_threshold: float = 0.75
    chunk_size: int = 512
    
    # Numeric sanity parameters
    numeric_tolerance: float = 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "k_samples": self.k_samples,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "enable_retrieval_support": self.enable_retrieval_support,
            "enable_jailbreak_heuristics": self.enable_jailbreak_heuristics,
            "thresholds": {
                "high": self.thresholds.high,
                "medium": self.thresholds.medium
            },
            "weights": self.weights,
            "html_report": self.html_report,
            "report_dir": self.report_dir,
            "nli_model_name": self.nli_model_name,
            "nli_entailment_threshold": self.nli_entailment_threshold,
            "retrieval_similarity_threshold": self.retrieval_similarity_threshold,
            "chunk_size": self.chunk_size,
            "numeric_tolerance": self.numeric_tolerance
        }
