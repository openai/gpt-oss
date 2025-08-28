"""
Tests for the Self-Consistency detector.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from gpt_oss.monitoring.detectors.self_consistency import SelfConsistencyDetector


class TestSelfConsistencyDetector:
    """Test the SelfConsistencyDetector class."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = SelfConsistencyDetector()
        assert detector.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        
        # Test custom model name
        detector = SelfConsistencyDetector("custom-model")
        assert detector.model_name == "custom-model"
    
    @patch('gpt_oss.monitoring.detectors.self_consistency.SentenceTransformer')
    def test_model_loading_success(self, mock_transformer):
        """Test successful model loading."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        detector = SelfConsistencyDetector()
        
        assert detector.model is not None
        mock_transformer.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")
    
    @patch('gpt_oss.monitoring.detectors.self_consistency.SentenceTransformer')
    def test_model_loading_failure(self, mock_transformer):
        """Test model loading failure fallback."""
        mock_transformer.side_effect = Exception("Model not found")
        
        detector = SelfConsistencyDetector()
        
        assert detector.model is None
    
    def test_generate_samples_no_model(self):
        """Test sample generation without model (fallback)."""
        detector = SelfConsistencyDetector()
        
        prompt = "What is the capital of France?"
        samples = detector._generate_samples(
            prompt, None, None, k=3, temperature=0.7, max_new_tokens=100
        )
        
        assert len(samples) == 3
        assert all(isinstance(sample, str) for sample in samples)
        assert all(len(sample) > 0 for sample in samples)
    
    @patch('gpt_oss.monitoring.detectors.self_consistency.SentenceTransformer')
    def test_generate_samples_with_model(self, mock_transformer):
        """Test sample generation with model."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        detector = SelfConsistencyDetector()
        
        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = Mock()
        mock_model_instance.generate.return_value.__getitem__.return_value = [1, 2, 3, 4, 5]
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {'input_ids': [1, 2, 3]}
        mock_tokenizer.decode.return_value = "Generated sample text"
        mock_tokenizer.eos_token_id = 2
        
        prompt = "What is the capital of France?"
        samples = detector._generate_samples(
            prompt, mock_model_instance, mock_tokenizer, k=2, temperature=0.7, max_new_tokens=100
        )
        
        assert len(samples) == 2
        assert all(isinstance(sample, str) for sample in samples)
    
    def test_compute_similarity_matrix_no_model(self):
        """Test similarity matrix computation without model (fallback)."""
        detector = SelfConsistencyDetector()
        detector.model = None
        
        texts = ["Sample 1", "Sample 2", "Sample 3"]
        matrix = detector._compute_similarity_matrix(texts)
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)
        
        # Diagonal should be 1.0
        assert np.allclose(np.diag(matrix), 1.0)
        
        # All values should be in [0, 1]
        assert np.all(matrix >= 0)
        assert np.all(matrix <= 1)
    
    @patch('gpt_oss.monitoring.detectors.self_consistency.SentenceTransformer')
    def test_compute_similarity_matrix_with_model(self, mock_transformer):
        """Test similarity matrix computation with model."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        # Mock embeddings
        mock_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        mock_model.encode.return_value = mock_embeddings
        
        detector = SelfConsistencyDetector()
        
        texts = ["Sample 1", "Sample 2", "Sample 3"]
        matrix = detector._compute_similarity_matrix(texts)
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)
        
        # Diagonal should be 1.0
        assert np.allclose(np.diag(matrix), 1.0)
        
        # All values should be in [0, 1]
        assert np.all(matrix >= 0)
        assert np.all(matrix <= 1)
    
    def test_compute_similarity_matrix_empty(self):
        """Test similarity matrix computation with empty texts."""
        detector = SelfConsistencyDetector()
        
        # Empty list
        matrix = detector._compute_similarity_matrix([])
        assert matrix.shape == (0, 0)
        
        # Single text
        matrix = detector._compute_similarity_matrix(["Single text"])
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 1.0
    
    def test_compute_similarity_matrix_fallback(self):
        """Test similarity matrix computation with model failure."""
        detector = SelfConsistencyDetector()
        detector.model = Mock()
        detector.model.encode.side_effect = Exception("Encoding failed")
        
        texts = ["Sample 1", "Sample 2"]
        matrix = detector._compute_similarity_matrix(texts)
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 2)
        assert np.allclose(matrix, np.eye(2))  # Identity matrix
    
    def test_detect_basic(self):
        """Test basic detection functionality."""
        detector = SelfConsistencyDetector()
        
        prompt = "What is the capital of France?"
        completion = "Paris is the capital of France."
        
        score, samples, matrix = detector.detect(
            prompt, completion, None, None, k=3, temperature=0.7, max_new_tokens=100
        )
        
        assert 0 <= score <= 1
        assert isinstance(samples, list)
        assert len(samples) == 3
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)
    
    def test_detect_with_model(self):
        """Test detection with model."""
        with patch('gpt_oss.monitoring.detectors.self_consistency.SentenceTransformer') as mock_transformer:
            mock_model = Mock()
            mock_transformer.return_value = mock_model
            
            detector = SelfConsistencyDetector()
            
            # Mock model and tokenizer
            mock_model_instance = Mock()
            mock_model_instance.generate.return_value = Mock()
            mock_model_instance.generate.return_value.__getitem__.return_value = [1, 2, 3, 4, 5]
            
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {'input_ids': [1, 2, 3]}
            mock_tokenizer.decode.return_value = "Generated sample text"
            mock_tokenizer.eos_token_id = 2
            
            # Mock embeddings
            mock_embeddings = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])
            mock_model.encode.return_value = mock_embeddings
            
            prompt = "What is the capital of France?"
            completion = "Paris is the capital of France."
            
            score, samples, matrix = detector.detect(
                prompt, completion, mock_model_instance, mock_tokenizer,
                k=3, temperature=0.7, max_new_tokens=100
            )
            
            assert 0 <= score <= 1
            assert isinstance(samples, list)
            assert len(samples) == 3
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == (3, 3)
    
    def test_detect_consistency_scoring(self):
        """Test consistency scoring logic."""
        detector = SelfConsistencyDetector()
        
        # Test with high consistency (similar samples)
        texts = ["Paris is the capital of France.", "Paris is the capital of France.", "Paris is the capital of France."]
        matrix = detector._compute_similarity_matrix(texts)
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle = matrix[np.triu_indices_from(matrix, k=1)]
        consistency_score = float(np.mean(upper_triangle))
        
        # Should be high for identical texts
        assert consistency_score > 0.8
        
        # Test with low consistency (different samples)
        texts = ["Paris is the capital of France.", "London is the capital of England.", "Berlin is the capital of Germany."]
        matrix = detector._compute_similarity_matrix(texts)
        
        upper_triangle = matrix[np.triu_indices_from(matrix, k=1)]
        consistency_score = float(np.mean(upper_triangle))
        
        # Should be lower for different texts
        assert consistency_score < 0.8
    
    def test_detect_parameters(self):
        """Test detection with different parameters."""
        detector = SelfConsistencyDetector()
        
        prompt = "What is the capital of France?"
        completion = "Paris is the capital of France."
        
        # Test with different k values
        score1, samples1, matrix1 = detector.detect(
            prompt, completion, None, None, k=2, temperature=0.7, max_new_tokens=100
        )
        
        score2, samples2, matrix2 = detector.detect(
            prompt, completion, None, None, k=5, temperature=0.7, max_new_tokens=100
        )
        
        assert len(samples1) == 2
        assert len(samples2) == 5
        assert matrix1.shape == (2, 2)
        assert matrix2.shape == (5, 5)
    
    def test_detect_reproducibility(self):
        """Test that detection is reproducible with same seed."""
        detector = SelfConsistencyDetector()
        
        prompt = "What is the capital of France?"
        completion = "Paris is the capital of France."
        
        # Run twice with same parameters
        score1, samples1, matrix1 = detector.detect(
            prompt, completion, None, None, k=3, temperature=0.7, max_new_tokens=100, seed=42
        )
        
        score2, samples2, matrix2 = detector.detect(
            prompt, completion, None, None, k=3, temperature=0.7, max_new_tokens=100, seed=42
        )
        
        # Results should be identical
        assert score1 == score2
        assert samples1 == samples2
        assert np.array_equal(matrix1, matrix2)
    
    def test_detect_error_handling(self):
        """Test error handling in detection."""
        detector = SelfConsistencyDetector()
        
        # Test with None inputs
        score, samples, matrix = detector.detect(
            None, None, None, None, k=3, temperature=0.7, max_new_tokens=100
        )
        
        # Should not raise exception and return valid results
        assert 0 <= score <= 1
        assert isinstance(samples, list)
        assert isinstance(matrix, np.ndarray)
    
    def test_character_based_similarity(self):
        """Test character-based similarity fallback."""
        detector = SelfConsistencyDetector()
        detector.model = None
        
        # Test with similar texts
        texts = ["Paris is the capital", "Paris is the capital"]
        matrix = detector._compute_similarity_matrix(texts)
        
        # Should have high similarity
        assert matrix[0, 1] > 0.8
        
        # Test with different texts
        texts = ["Paris is the capital", "London is the capital"]
        matrix = detector._compute_similarity_matrix(texts)
        
        # Should have lower similarity
        assert matrix[0, 1] < 0.8
        
        # Test with empty texts
        texts = ["", "Some text"]
        matrix = detector._compute_similarity_matrix(texts)
        
        # Should handle empty text gracefully
        assert matrix[0, 1] == 0.0
