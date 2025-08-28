"""
Retrieval Support detector for hallucination monitoring.

Checks if claims in the completion are supported by provided context documents.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import re

logger = logging.getLogger(__name__)


class RetrievalSupportDetector:
    """Detector for retrieval support using semantic similarity."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the detector with a sentence transformer model."""
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded sentence transformer for retrieval: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}, using fallback: {e}")
            self.model = None
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _chunk_documents(self, documents: List[str], chunk_size: int = 512) -> List[str]:
        """Split documents into chunks for better matching."""
        chunks = []
        
        for doc in documents:
            # Simple chunking by character count
            for i in range(0, len(doc), chunk_size):
                chunk = doc[i:i + chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
        
        return chunks
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        if self.model is None:
            # Fallback: simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if len(words1) == 0 or len(words2) == 0:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0
        
        try:
            embeddings = self.model.encode([text1, text2])
            cos_sim = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return max(0, cos_sim)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Failed to compute similarity: {e}")
            return 0.0
    
    def _find_supporting_chunks(self, sentence: str, chunks: List[str], 
                              threshold: float = 0.75) -> List[Dict[str, Any]]:
        """Find chunks that support a given sentence."""
        supporting_chunks = []
        
        for i, chunk in enumerate(chunks):
            similarity = self._compute_similarity(sentence, chunk)
            
            if similarity >= threshold:
                supporting_chunks.append({
                    'chunk_index': i,
                    'chunk_text': chunk,
                    'similarity': similarity
                })
        
        # Sort by similarity
        supporting_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        return supporting_chunks
    
    def _compute_lexical_overlap(self, text1: str, text2: str) -> float:
        """Compute lexical overlap between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    
    def detect(self, completion: str, context_docs: List[str] = None,
               similarity_threshold: float = 0.75, chunk_size: int = 512) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Detect retrieval support for claims in the completion.
        
        Returns:
            - support_rate: float in [0, 1]
            - sentence_analyses: List of sentence-level support analyses
        """
        logger.info("Running retrieval support detection")
        
        if not context_docs:
            logger.info("No context documents provided, skipping retrieval support")
            return 1.0, []
        
        # Split completion into sentences
        sentences = self._split_into_sentences(completion)
        
        if not sentences:
            logger.warning("No sentences found in completion")
            return 1.0, []
        
        # Chunk the context documents
        chunks = self._chunk_documents(context_docs, chunk_size)
        
        if not chunks:
            logger.warning("No chunks created from context documents")
            return 0.0, []
        
        sentence_analyses = []
        supported_sentences = 0
        
        for i, sentence in enumerate(sentences):
            # Find supporting chunks
            supporting_chunks = self._find_supporting_chunks(
                sentence, chunks, similarity_threshold
            )
            
            # Check if sentence is supported
            is_supported = len(supporting_chunks) > 0
            
            if is_supported:
                supported_sentences += 1
            
            # Also check lexical overlap as a fallback
            best_lexical_overlap = 0.0
            for chunk in chunks:
                overlap = self._compute_lexical_overlap(sentence, chunk)
                best_lexical_overlap = max(best_lexical_overlap, overlap)
            
            analysis = {
                'sentence': sentence,
                'sentence_index': i,
                'is_supported': is_supported,
                'supporting_chunks': supporting_chunks,
                'best_similarity': supporting_chunks[0]['similarity'] if supporting_chunks else 0.0,
                'best_lexical_overlap': best_lexical_overlap,
                'reason': 'RS: supported' if is_supported else 'RS: unsupported'
            }
            
            sentence_analyses.append(analysis)
        
        # Compute support rate
        support_rate = supported_sentences / len(sentences) if sentences else 1.0
        
        logger.info(f"Retrieval support rate: {support_rate:.3f} ({supported_sentences}/{len(sentences)} sentences)")
        
        return support_rate, sentence_analyses
