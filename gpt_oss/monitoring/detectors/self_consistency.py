"""
Self-Consistency detector for hallucination monitoring.

Generates k samples and computes semantic agreement using sentence embeddings.
"""

import logging
import numpy as np
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import random

logger = logging.getLogger(__name__)


class SelfConsistencyDetector:
    """Detector for self-consistency using semantic similarity."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the detector with a sentence transformer model."""
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded sentence transformer: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}, using fallback: {e}")
            # Fallback to a simpler approach if model loading fails
            self.model = None
    
    def _generate_samples(self, prompt: str, model, tokenizer, k: int, 
                         temperature: float, max_new_tokens: int, seed: int = 42) -> List[str]:
        """Generate k samples from the model."""
        if model is None or tokenizer is None:
            # Mock samples for testing
            random.seed(seed)
            return [f"Sample {i} for prompt: {prompt[:50]}..." for i in range(k)]
        
        samples = []
        for i in range(k):
            # Set seed for reproducibility
            random.seed(seed + i)
            np.random.seed(seed + i)
            
            try:
                # Use the model's generation method
                if hasattr(model, 'generate'):
                    inputs = tokenizer(prompt, return_tensors="pt")
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    sample = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Remove the original prompt
                    sample = sample[len(prompt):].strip()
                else:
                    # Fallback for different model interfaces
                    sample = f"Generated sample {i} for testing"
                
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to generate sample {i}: {e}")
                samples.append(f"Error sample {i}")
        
        return samples
    
    def _compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        if not texts or len(texts) < 2:
            return np.array([[1.0]])
        
        if self.model is None:
            # Fallback: simple character-based similarity
            similarities = []
            for i in range(len(texts)):
                row = []
                for j in range(len(texts)):
                    if i == j:
                        row.append(1.0)
                    else:
                        # Simple character overlap similarity
                        chars_i = set(texts[i].lower())
                        chars_j = set(texts[j].lower())
                        if len(chars_i) == 0 or len(chars_j) == 0:
                            row.append(0.0)
                        else:
                            intersection = len(chars_i.intersection(chars_j))
                            union = len(chars_i.union(chars_j))
                            row.append(intersection / union if union > 0 else 0.0)
                similarities.append(row)
            return np.array(similarities)
        
        try:
            # Use sentence transformer embeddings
            embeddings = self.model.encode(texts)
            # Compute cosine similarity matrix
            similarities = np.zeros((len(texts), len(texts)))
            for i in range(len(texts)):
                for j in range(len(texts)):
                    if i == j:
                        similarities[i, j] = 1.0
                    else:
                        cos_sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        )
                        similarities[i, j] = max(0, cos_sim)  # Ensure non-negative
            return similarities
        except Exception as e:
            logger.warning(f"Failed to compute embeddings: {e}")
            # Return identity matrix as fallback
            return np.eye(len(texts))
    
    def detect(self, prompt: str, completion: str, model=None, tokenizer=None,
               k: int = 5, temperature: float = 0.7, max_new_tokens: int = 512,
               seed: int = 42) -> Tuple[float, List[str], np.ndarray]:
        """
        Detect self-consistency by generating k samples and computing agreement.
        
        Returns:
            - consistency_score: float in [0, 1]
            - samples: List of generated samples
            - similarity_matrix: numpy array of pairwise similarities
        """
        logger.info(f"Running self-consistency detection with k={k}")
        
        # Generate k samples
        samples = self._generate_samples(
            prompt, model, tokenizer, k, temperature, max_new_tokens, seed
        )
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(samples)
        
        # Compute consistency score as mean of upper triangle (excluding diagonal)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        consistency_score = float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 1.0
        
        logger.info(f"Self-consistency score: {consistency_score:.3f}")
        
        return consistency_score, samples, similarity_matrix
