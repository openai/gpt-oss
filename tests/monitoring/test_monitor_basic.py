"""
Basic tests for the Hallucination Monitor.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch

from gpt_oss.monitoring import HallucinationMonitor, MonitorConfig, MonitorThresholds


class TestHallucinationMonitor:
    """Test the main HallucinationMonitor class."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = HallucinationMonitor()
        assert monitor.config is not None
        assert monitor.detectors is not None
        assert len(monitor.detectors) == 5  # All 5 detectors
    
    def test_initialization_with_config(self):
        """Test monitor initialization with custom config."""
        config = MonitorConfig(
            k_samples=10,
            temperature=0.8,
            enable_retrieval_support=False
        )
        monitor = HallucinationMonitor(config)
        assert monitor.config.k_samples == 10
        assert monitor.config.temperature == 0.8
        assert monitor.config.enable_retrieval_support is False
    
    def test_evaluate_basic(self):
        """Test basic evaluation without model."""
        monitor = HallucinationMonitor()
        
        results = monitor.evaluate(
            prompt="What is 2 + 2?",
            completion="2 + 2 equals 4."
        )
        
        # Check required fields
        assert 'risk_score' in results
        assert 'risk_level' in results
        assert 'signals' in results
        assert 'spans' in results
        assert 'artifacts' in results
        assert 'config' in results
        
        # Check signal scores
        signals = results['signals']
        assert 'self_consistency' in signals
        assert 'nli_faithfulness' in signals
        assert 'numeric_sanity' in signals
        assert 'retrieval_support' in signals
        assert 'jailbreak_heuristics' in signals
        
        # All scores should be in [0, 1]
        for score in signals.values():
            assert 0 <= score <= 1
        
        # Risk score should be in [0, 1]
        assert 0 <= results['risk_score'] <= 1
        
        # Risk level should be one of the expected values
        assert results['risk_level'] in ['low', 'medium', 'high']
    
    def test_evaluate_with_context(self):
        """Test evaluation with context documents."""
        monitor = HallucinationMonitor()
        
        context_docs = [
            "Paris is the capital of France.",
            "The population of Paris is about 2.2 million people."
        ]
        
        results = monitor.evaluate(
            prompt="What is the capital of France?",
            completion="Paris is the capital of France with 2.2 million people.",
            context_docs=context_docs
        )
        
        assert results['context_docs'] == context_docs
        assert 'retrieval_support' in results['signals']
    
    def test_evaluate_without_retrieval_support(self):
        """Test evaluation with retrieval support disabled."""
        config = MonitorConfig(enable_retrieval_support=False)
        monitor = HallucinationMonitor(config)
        
        context_docs = ["Some context document."]
        
        results = monitor.evaluate(
            prompt="Test prompt",
            completion="Test completion",
            context_docs=context_docs
        )
        
        # Retrieval support should still be 1.0 (default) when disabled
        assert results['signals']['retrieval_support'] == 1.0
    
    def test_evaluate_without_jailbreak_heuristics(self):
        """Test evaluation with jailbreak heuristics disabled."""
        config = MonitorConfig(enable_jailbreak_heuristics=False)
        monitor = HallucinationMonitor(config)
        
        results = monitor.evaluate(
            prompt="Test prompt",
            completion="Test completion"
        )
        
        # Jailbreak heuristics should be 0.0 when disabled
        assert results['signals']['jailbreak_heuristics'] == 0.0
    
    def test_html_report_generation(self):
        """Test HTML report generation."""
        config = MonitorConfig(html_report=True)
        monitor = HallucinationMonitor(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.report_dir = temp_dir
            
            results = monitor.evaluate(
                prompt="Test prompt",
                completion="Test completion"
            )
            
            # Check if HTML report was generated
            artifacts = results['artifacts']
            assert 'html_report' in artifacts
            
            if artifacts['html_report']:
                assert os.path.exists(artifacts['html_report'])
                assert artifacts['html_report'].endswith('.html')
    
    def test_suggest_cautious_decoding(self):
        """Test cautious decoding suggestions."""
        monitor = HallucinationMonitor()
        
        suggestions = monitor.suggest_cautious_decoding('high')
        assert 'temperature' in suggestions
        assert 'top_p' in suggestions
        assert 'max_new_tokens' in suggestions
        
        # High risk should suggest lower temperature
        assert suggestions['temperature'] < 0.5
    
    def test_ask_for_citations(self):
        """Test citation request templates."""
        monitor = HallucinationMonitor()
        
        template = monitor.ask_for_citations('high')
        assert isinstance(template, str)
        assert len(template) > 0
        assert 'citations' in template.lower()
    
    def test_get_detector_info(self):
        """Test detector information retrieval."""
        monitor = HallucinationMonitor()
        
        info = monitor.get_detector_info()
        assert 'self_consistency' in info
        assert 'nli_faithfulness' in info
        assert 'numeric_sanity' in info
        assert 'retrieval_support' in info
        assert 'jailbreak_heuristics' in info
        
        # All descriptions should be strings
        for description in info.values():
            assert isinstance(description, str)
            assert len(description) > 0


class TestMonitorConfig:
    """Test the MonitorConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MonitorConfig()
        
        assert config.k_samples == 5
        assert config.temperature == 0.7
        assert config.max_new_tokens == 512
        assert config.enable_retrieval_support is True
        assert config.enable_jailbreak_heuristics is True
        assert config.html_report is True
        assert config.report_dir == "runs"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MonitorConfig(
            k_samples=10,
            temperature=0.8,
            enable_retrieval_support=False,
            html_report=False
        )
        
        assert config.k_samples == 10
        assert config.temperature == 0.8
        assert config.enable_retrieval_support is False
        assert config.html_report is False
    
    def test_thresholds(self):
        """Test threshold configuration."""
        thresholds = MonitorThresholds(high=0.8, medium=0.5)
        config = MonitorConfig(thresholds=thresholds)
        
        assert config.thresholds.high == 0.8
        assert config.thresholds.medium == 0.5
    
    def test_weights(self):
        """Test custom weights configuration."""
        custom_weights = {
            'nli_faithfulness': 0.5,
            'self_consistency': 0.3,
            'numeric_sanity': 0.2
        }
        config = MonitorConfig(weights=custom_weights)
        
        assert config.weights['nli_faithfulness'] == 0.5
        assert config.weights['self_consistency'] == 0.3
        assert config.weights['numeric_sanity'] == 0.2
    
    def test_to_dict(self):
        """Test configuration serialization."""
        config = MonitorConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'k_samples' in config_dict
        assert 'temperature' in config_dict
        assert 'thresholds' in config_dict
        assert 'weights' in config_dict
        
        # Check thresholds structure
        thresholds = config_dict['thresholds']
        assert 'high' in thresholds
        assert 'medium' in thresholds


class TestMonitorThresholds:
    """Test the MonitorThresholds class."""
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = MonitorThresholds()
        
        assert thresholds.high == 0.7
        assert thresholds.medium == 0.4
    
    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = MonitorThresholds(high=0.8, medium=0.5)
        
        assert thresholds.high == 0.8
        assert thresholds.medium == 0.5
