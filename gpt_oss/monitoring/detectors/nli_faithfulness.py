"""
NLI Faithfulness detector for hallucination monitoring.

Checks sentence-level entailment against prompt and context using NLI models.
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

logger = logging.getLogger(__name__)


class NLIFaithfulnessDetector:
    """Detector for NLI faithfulness using entailment models."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize the detector with an NLI model."""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the NLI model and tokenizer."""
        try:
            # Use a lightweight model for NLI
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Loaded NLI model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}, using fallback: {e}")
            self.model = None
            self.tokenizer = None
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Simple sentence splitting - can be improved with more sophisticated NLP
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _compute_entailment(self, premise: str, hypothesis: str) -> float:
        """Compute entailment probability between premise and hypothesis."""
        if self.model is None or self.tokenizer is None:
            # Fallback: simple word overlap heuristic
            premise_words = set(premise.lower().split())
            hypothesis_words = set(hypothesis.lower().split())
            
            if len(hypothesis_words) == 0:
                return 0.0
            
            overlap = len(premise_words.intersection(hypothesis_words))
            return min(1.0, overlap / len(hypothesis_words))
        
        try:
            # Format for NLI: premise || hypothesis
            input_text = f"{premise} || {hypothesis}"
            
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # For binary classification (entailment vs not)
                if logits.shape[1] == 2:
                    probs = torch.softmax(logits, dim=1)
                    entailment_prob = probs[0, 1].item()  # Assume 1 is entailment
                else:
                    # For multi-class, use the highest probability
                    probs = torch.softmax(logits, dim=1)
                    entailment_prob = torch.max(probs).item()
                
                return entailment_prob
                
        except Exception as e:
            logger.warning(f"Failed to compute entailment: {e}")
            return 0.5  # Neutral fallback
    
    def _create_premise(self, prompt: str, context_docs: List[str] = None, 
                       top_n: int = 3) -> str:
        """Create premise from prompt and context documents."""
        premise_parts = [prompt]
        
        if context_docs:
            # Take top N context documents (simple approach)
            for i, doc in enumerate(context_docs[:top_n]):
                premise_parts.append(f"Context {i+1}: {doc}")
        
        return " ".join(premise_parts)
    
    def detect(self, prompt: str, completion: str, context_docs: List[str] = None,
               entailment_threshold: float = 0.5) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Detect NLI faithfulness by checking sentence-level entailment.
        
        Returns:
            - faithfulness_score: float in [0, 1]
            - sentence_analyses: List of sentence-level analyses
        """
        logger.info("Running NLI faithfulness detection")
        
        # Split completion into sentences
        sentences = self._split_into_sentences(completion)
        
        if not sentences:
            logger.warning("No sentences found in completion")
            return 1.0, []
        
        # Create premise from prompt and context
        premise = self._create_premise(prompt, context_docs)
        
        sentence_analyses = []
        entailment_scores = []
        
        for i, sentence in enumerate(sentences):
            # Compute entailment for this sentence
            entailment_score = self._compute_entailment(premise, sentence)
            entailment_scores.append(entailment_score)
            
            # Determine if sentence is faithful
            is_faithful = entailment_score >= entailment_threshold
            
            analysis = {
                "sentence": sentence,
                "sentence_index": i,
                "entailment_score": entailment_score,
                "is_faithful": is_faithful,
                "reason": "NLI: entailment" if is_faithful else "NLI: neutral/contradiction"
            }
            
            sentence_analyses.append(analysis)
        
        # Compute overall faithfulness score
        faithfulness_score = float(np.mean(entailment_scores))
        
        logger.info(f"NLI faithfulness score: {faithfulness_score:.3f}")
        
        return faithfulness_score, sentence_analyses
