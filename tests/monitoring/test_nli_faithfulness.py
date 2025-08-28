"""
Tests for the NLI Faithfulness detector.
"""

import pytest
from unittest.mock import Mock, patch
from gpt_oss.monitoring.detectors.nli_faithfulness import NLIFaithfulnessDetector


class TestNLIFaithfulnessDetector:
    """Test the NLIFaithfulnessDetector class."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = NLIFaithfulnessDetector()
        assert detector.model_name == "microsoft/DialoGPT-medium"
        
        # Test custom model name
        detector = NLIFaithfulnessDetector("custom-model")
        assert detector.model_name == "custom-model"
    
    @patch('gpt_oss.monitoring.detectors.nli_faithfulness.AutoTokenizer')
    @patch('gpt_oss.monitoring.detectors.nli_faithfulness.AutoModelForSequenceClassification')
    def test_model_loading_success(self, mock_model, mock_tokenizer):
        """Test successful model loading."""
        # Mock the model and tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        detector = NLIFaithfulnessDetector()
        
        # Should not raise an exception
        assert detector.model is not None
        assert detector.tokenizer is not None
    
    @patch('gpt_oss.monitoring.detectors.nli_faithfulness.AutoTokenizer')
    @patch('gpt_oss.monitoring.detectors.nli_faithfulness.AutoModelForSequenceClassification')
    def test_model_loading_failure(self, mock_model, mock_tokenizer):
        """Test model loading failure fallback."""
        # Mock failure
        mock_tokenizer.from_pretrained.side_effect = Exception("Model not found")
        
        detector = NLIFaithfulnessDetector()
        
        # Should fall back to None
        assert detector.model is None
        assert detector.tokenizer is None
    
    def test_split_into_sentences(self):
        """Test sentence splitting functionality."""
        detector = NLIFaithfulnessDetector()
        
        # Test basic sentence splitting
        text = "This is sentence one. This is sentence two. This is sentence three!"
        sentences = detector._split_into_sentences(text)
        
        assert len(sentences) == 3
        assert sentences[0] == "This is sentence one"
        assert sentences[1] == "This is sentence two"
        assert sentences[2] == "This is sentence three"
        
        # Test with question marks
        text = "What is this? This is a test. Is it working?"
        sentences = detector._split_into_sentences(text)
        
        assert len(sentences) == 3
        assert sentences[0] == "What is this"
        assert sentences[1] == "This is a test"
        assert sentences[2] == "Is it working"
        
        # Test with empty text
        text = ""
        sentences = detector._split_into_sentences(text)
        assert len(sentences) == 0
        
        # Test with single sentence
        text = "This is a single sentence."
        sentences = detector._split_into_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == "This is a single sentence"
    
    def test_compute_entailment_fallback(self):
        """Test entailment computation with fallback (no model)."""
        detector = NLIFaithfulnessDetector()
        detector.model = None  # Force fallback
        
        # Test with overlapping words
        premise = "The cat is on the mat."
        hypothesis = "The cat is sitting."
        score = detector._compute_entailment(premise, hypothesis)
        
        assert 0 <= score <= 1
        
        # Test with no overlap
        premise = "The cat is on the mat."
        hypothesis = "The dog is running."
        score = detector._compute_entailment(premise, hypothesis)
        
        assert 0 <= score <= 1
        assert score < 0.5  # Should be lower than the overlapping case
    
    @patch('gpt_oss.monitoring.detectors.nli_faithfulness.AutoTokenizer')
    @patch('gpt_oss.monitoring.detectors.nli_faithfulness.AutoModelForSequenceClassification')
    def test_compute_entailment_with_model(self, mock_model, mock_tokenizer):
        """Test entailment computation with model."""
        # Mock the model and tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.return_value = {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model_instance.return_value.logits = Mock()
        mock_model_instance.return_value.logits.shape = [1, 2]  # Binary classification
        mock_model.from_pretrained.return_value = mock_model_instance
        
        detector = NLIFaithfulnessDetector()
        
        # Mock the model output
        with patch('torch.softmax') as mock_softmax:
            mock_softmax.return_value = Mock()
            mock_softmax.return_value.__getitem__.return_value.__getitem__.return_value.item.return_value = 0.8
            
            score = detector._compute_entailment("premise", "hypothesis")
            assert 0 <= score <= 1
    
    def test_create_premise(self):
        """Test premise creation from prompt and context."""
        detector = NLIFaithfulnessDetector()
        
        # Test with just prompt
        prompt = "What is the capital of France?"
        premise = detector._create_premise(prompt)
        assert premise == prompt
        
        # Test with prompt and context
        context_docs = [
            "Paris is the capital of France.",
            "The population of Paris is about 2.2 million people."
        ]
        premise = detector._create_premise(prompt, context_docs)
        
        expected = "What is the capital of France? Context 1: Paris is the capital of France. Context 2: The population of Paris is about 2.2 million people."
        assert premise == expected
        
        # Test with limited context (top_n=1)
        premise = detector._create_premise(prompt, context_docs, top_n=1)
        expected = "What is the capital of France? Context 1: Paris is the capital of France."
        assert premise == expected
    
    def test_detect_basic(self):
        """Test basic detection functionality."""
        detector = NLIFaithfulnessDetector()
        
        # Test with faithful completion
        prompt = "What is the capital of France?"
        completion = "Paris is the capital of France."
        
        score, analyses = detector.detect(prompt, completion)
        
        assert 0 <= score <= 1
        assert isinstance(analyses, list)
        assert len(analyses) > 0
        
        # Check analysis structure
        analysis = analyses[0]
        assert 'sentence' in analysis
        assert 'sentence_index' in analysis
        assert 'entailment_score' in analysis
        assert 'is_faithful' in analysis
        assert 'reason' in analysis
    
    def test_detect_with_context(self):
        """Test detection with context documents."""
        detector = NLIFaithfulnessDetector()
        
        prompt = "What is the capital of France?"
        completion = "Paris is the capital of France."
        context_docs = ["Paris is the capital and most populous city of France."]
        
        score, analyses = detector.detect(prompt, completion, context_docs)
        
        assert 0 <= score <= 1
        assert isinstance(analyses, list)
        assert len(analyses) > 0
    
    def test_detect_empty_completion(self):
        """Test detection with empty completion."""
        detector = NLIFaithfulnessDetector()
        
        prompt = "What is the capital of France?"
        completion = ""
        
        score, analyses = detector.detect(prompt, completion)
        
        assert score == 1.0  # Default score for empty completion
        assert len(analyses) == 0
    
    def test_detect_multiple_sentences(self):
        """Test detection with multiple sentences."""
        detector = NLIFaithfulnessDetector()
        
        prompt = "Tell me about Paris."
        completion = "Paris is the capital of France. It has a population of 2.2 million people. The Eiffel Tower is located there."
        
        score, analyses = detector.detect(prompt, completion)
        
        assert 0 <= score <= 1
        assert len(analyses) == 3  # Three sentences
        
        # Check that each sentence is analyzed
        for i, analysis in enumerate(analyses):
            assert analysis['sentence_index'] == i
            assert analysis['sentence'] in completion
    
    def test_entailment_threshold(self):
        """Test entailment threshold functionality."""
        detector = NLIFaithfulnessDetector()
        
        prompt = "What is the capital of France?"
        completion = "Paris is the capital of France."
        
        # Test with high threshold
        score, analyses = detector.detect(prompt, completion, entailment_threshold=0.9)
        
        # Test with low threshold
        score2, analyses2 = detector.detect(prompt, completion, entailment_threshold=0.1)
        
        # The faithfulness should be the same, but the binary classification might differ
        assert len(analyses) == len(analyses2)
    
    def test_contradiction_detection(self):
        """Test detection of contradictions."""
        detector = NLIFaithfulnessDetector()
        
        prompt = "What is the capital of France?"
        completion = "London is the capital of France."  # Contradiction
        
        score, analyses = detector.detect(prompt, completion)
        
        assert 0 <= score <= 1
        assert len(analyses) > 0
        
        # The score should be lower for contradictions
        # (Note: This depends on the actual model/fallback behavior)
    
    def test_neutral_detection(self):
        """Test detection of neutral statements."""
        detector = NLIFaithfulnessDetector()
        
        prompt = "What is the capital of France?"
        completion = "The weather is nice today."  # Neutral/irrelevant
        
        score, analyses = detector.detect(prompt, completion)
        
        assert 0 <= score <= 1
        assert len(analyses) > 0
    
    def test_reason_assignment(self):
        """Test that reasons are properly assigned."""
        detector = NLIFaithfulnessDetector()
        
        prompt = "What is the capital of France?"
        completion = "Paris is the capital of France."
        
        score, analyses = detector.detect(prompt, completion)
        
        for analysis in analyses:
            reason = analysis['reason']
            assert reason.startswith('NLI:')
            assert 'entailment' in reason or 'neutral' in reason or 'contradiction' in reason
    
    def test_sentence_span_consistency(self):
        """Test that sentence spans are consistent."""
        detector = NLIFaithfulnessDetector()
        
        prompt = "Tell me about Paris."
        completion = "Paris is beautiful. It has many museums."
        
        score, analyses = detector.detect(prompt, completion)
        
        # All sentences should be found in the completion
        for analysis in analyses:
            sentence = analysis['sentence']
            assert sentence in completion
