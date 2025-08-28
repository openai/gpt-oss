"""
Main Hallucination Monitor class for GPT-OSS.

Provides a unified interface for detecting hallucination risk in LLM outputs.
"""

import logging
import os
import json
import random
import numpy as np
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from .config import MonitorConfig, MonitorThresholds
from .detectors import (
    SelfConsistencyDetector,
    NLIFaithfulnessDetector,
    NumericSanityDetector,
    RetrievalSupportDetector,
    JailbreakHeuristicsDetector
)
from .highlight import SpanAligner, HTMLReportGenerator
from .metrics import compute_risk_score, determine_risk_level, suggest_cautious_decoding, ask_for_citations

logger = logging.getLogger(__name__)


class HallucinationMonitor:
    """Main class for hallucination monitoring."""
    
    def __init__(self, cfg: Optional[MonitorConfig] = None, model=None, tokenizer=None):
        """
        Initialize the Hallucination Monitor.
        
        Args:
            cfg: Configuration object
            model: Optional model for self-consistency generation
            tokenizer: Optional tokenizer for the model
        """
        self.config = cfg or MonitorConfig()
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize detectors
        self.detectors = {
            'self_consistency': SelfConsistencyDetector(),
            'nli_faithfulness': NLIFaithfulnessDetector(self.config.nli_model_name),
            'numeric_sanity': NumericSanityDetector(self.config.numeric_tolerance),
            'retrieval_support': RetrievalSupportDetector(),
            'jailbreak_heuristics': JailbreakHeuristicsDetector()
        }
        
        # Initialize utilities
        self.span_aligner = None
        self.html_generator = HTMLReportGenerator()
        
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        logger.info("Hallucination Monitor initialized")
    
    def evaluate(self, prompt: str, completion: str, 
                context_docs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate hallucination risk in a completion.
        
        Args:
            prompt: The input prompt
            completion: The model completion to evaluate
            context_docs: Optional list of context documents for retrieval support
        
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting hallucination evaluation")
        
        # Initialize span aligner
        self.span_aligner = SpanAligner(completion, self.tokenizer)
        
        # Run all detectors
        signals = {}
        detector_details = {}
        all_spans = []
        
        # Self-consistency detection
        if self.model is not None:
            sc_score, sc_samples, sc_matrix = self.detectors['self_consistency'].detect(
                prompt, completion, self.model, self.tokenizer,
                k=self.config.k_samples,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_new_tokens
            )
            signals['self_consistency'] = sc_score
            detector_details['self_consistency'] = {
                'samples': sc_samples,
                'similarity_matrix': sc_matrix.tolist() if hasattr(sc_matrix, 'tolist') else sc_matrix
            }
        else:
            logger.warning("No model provided, skipping self-consistency detection")
            signals['self_consistency'] = 1.0  # Default to high consistency
            detector_details['self_consistency'] = {'samples': [], 'similarity_matrix': []}
        
        # NLI faithfulness detection
        nli_score, nli_analyses = self.detectors['nli_faithfulness'].detect(
            prompt, completion, context_docs,
            entailment_threshold=self.config.nli_entailment_threshold
        )
        signals['nli_faithfulness'] = nli_score
        detector_details['nli_faithfulness'] = {'sentence_analyses': nli_analyses}
        
        # Add NLI spans
        for analysis in nli_analyses:
            if not analysis.get('is_faithful', True):
                # Find sentence span in completion
                sentence_spans = self.span_aligner.find_sentence_spans([analysis['sentence']])
                if sentence_spans:
                    span = sentence_spans[0]
                    span['reason'] = analysis['reason']
                    all_spans.append(span)
        
        # Numeric sanity detection
        numeric_score, numeric_issues = self.detectors['numeric_sanity'].detect(completion)
        signals['numeric_sanity'] = numeric_score
        detector_details['numeric_sanity'] = {'issues': numeric_issues}
        
        # Retrieval support detection
        if self.config.enable_retrieval_support and context_docs:
            rs_score, rs_analyses = self.detectors['retrieval_support'].detect(
                completion, context_docs,
                similarity_threshold=self.config.retrieval_similarity_threshold,
                chunk_size=self.config.chunk_size
            )
            signals['retrieval_support'] = rs_score
            detector_details['retrieval_support'] = {'sentence_analyses': rs_analyses}
            
            # Add retrieval support spans
            for analysis in rs_analyses:
                if not analysis.get('is_supported', True):
                    sentence_spans = self.span_aligner.find_sentence_spans([analysis['sentence']])
                    if sentence_spans:
                        span = sentence_spans[0]
                        span['reason'] = analysis['reason']
                        all_spans.append(span)
        else:
            signals['retrieval_support'] = 1.0  # Default to full support
            detector_details['retrieval_support'] = {'sentence_analyses': []}
        
        # Jailbreak heuristics detection
        if self.config.enable_jailbreak_heuristics:
            jb_score, jb_analysis = self.detectors['jailbreak_heuristics'].detect(completion)
            signals['jailbreak_heuristics'] = jb_score
            detector_details['jailbreak_heuristics'] = jb_analysis
        else:
            signals['jailbreak_heuristics'] = 0.0  # Default to no risk
            detector_details['jailbreak_heuristics'] = {'reason': 'Disabled'}
        
        # Compute overall risk score
        risk_score = compute_risk_score(signals, self.config.weights)
        risk_level = determine_risk_level(risk_score, {
            'high': self.config.thresholds.high,
            'medium': self.config.thresholds.medium
        })
        
        # Highlight spans
        highlighted_spans = self.span_aligner.highlight_spans(all_spans)
        
        # Generate HTML report if requested
        artifacts = {}
        if self.config.html_report:
            try:
                report_path = self._generate_html_report(
                    prompt, completion, signals, risk_score, risk_level,
                    detector_details, highlighted_spans
                )
                artifacts['html_report'] = report_path
            except Exception as e:
                logger.warning(f"Failed to generate HTML report: {e}")
                artifacts['html_report'] = None
        
        # Prepare results
        results = {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'signals': signals,
            'spans': highlighted_spans,
            'artifacts': artifacts,
            'config': self.config.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'completion': completion,
            'context_docs': context_docs
        }
        
        logger.info(f"Evaluation complete. Risk level: {risk_level} (score: {risk_score:.3f})")
        
        return results
    
    def _generate_html_report(self, prompt: str, completion: str, signals: Dict[str, float],
                            risk_score: float, risk_level: str, detector_details: Dict[str, Any],
                            spans: List[Dict[str, Any]]) -> str:
        """Generate HTML report and save to file."""
        
        # Prepare results for HTML generator
        # Add text field to spans for template
        processed_spans = []
        for span in spans:
            span_copy = span.copy()
            span_copy['text'] = completion[span['start']:span['end']]
            processed_spans.append(span_copy)
        
        results = {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'signals': signals,
            'spans': processed_spans,
            'detector_details': detector_details,
            'completion': completion
        }
        
        # Create output directory
        os.makedirs(self.config.report_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hallucination_report_{timestamp}.html"
        output_path = os.path.join(self.config.report_dir, filename)
        
        # Generate and save report
        self.html_generator.save_report(
            results, self.config.to_dict(), output_path
        )
        
        logger.info(f"HTML report saved to: {output_path}")
        return output_path
    
    def suggest_cautious_decoding(self, risk_level: str) -> Dict[str, Any]:
        """Get suggestions for cautious decoding parameters."""
        return suggest_cautious_decoding(risk_level)
    
    def ask_for_citations(self, risk_level: str) -> str:
        """Get a template prompt for requesting citations."""
        return ask_for_citations(risk_level)
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about available detectors."""
        return {
            'self_consistency': 'Generates k samples and computes semantic agreement',
            'nli_faithfulness': 'Checks sentence-level entailment against prompt and context',
            'numeric_sanity': 'Detects arithmetic and unit consistency issues',
            'retrieval_support': 'Verifies claims against provided context documents',
            'jailbreak_heuristics': 'Detects potential safety risks and jailbreak attempts'
        }
