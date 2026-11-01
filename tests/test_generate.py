"""Unit tests for generate.py backend selection and argument parsing."""

import argparse
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_parse_args_minimal(self):
        """Test parsing with minimal required arguments."""
        from gpt_oss.generate import __name__ as module_name
        
        with patch('sys.argv', ['generate.py', 'model/']):
            parser = argparse.ArgumentParser(description="Text generation example")
            parser.add_argument("checkpoint", metavar="FILE", type=str)
            parser.add_argument("-p", "--prompt", metavar="PROMPT", type=str, default="How are you?")
            parser.add_argument("-t", "--temperature", metavar="TEMP", type=float, default=0.0)
            parser.add_argument("-l", "--limit", metavar="LIMIT", type=int, default=0)
            parser.add_argument("-b", "--backend", metavar="BACKEND", type=str, default="torch", choices=["triton", "torch", "vllm"])
            
            args = parser.parse_args(['model/'])
            
            assert args.checkpoint == 'model/'
            assert args.prompt == "How are you?"
            assert args.temperature == 0.0
            assert args.limit == 0
            assert args.backend == "torch"

    def test_parse_args_with_all_options(self):
        """Test parsing with all optional arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument("checkpoint", type=str)
        parser.add_argument("-p", "--prompt", type=str, default="How are you?")
        parser.add_argument("-t", "--temperature", type=float, default=0.0)
        parser.add_argument("-l", "--limit", type=int, default=0)
        parser.add_argument("-b", "--backend", type=str, default="torch", choices=["triton", "torch", "vllm"])
        
        args = parser.parse_args([
            'model/',
            '-p', 'Custom prompt',
            '-t', '0.7',
            '-l', '100',
            '-b', 'triton'
        ])
        
        assert args.checkpoint == 'model/'
        assert args.prompt == 'Custom prompt'
        assert args.temperature == 0.7
        assert args.limit == 100
        assert args.backend == 'triton'

    def test_parse_args_invalid_backend(self):
        """Test that invalid backend raises error."""
        parser = argparse.ArgumentParser()
        parser.add_argument("checkpoint", type=str)
        parser.add_argument("-b", "--backend", type=str, default="torch", choices=["triton", "torch", "vllm"])
        
        with pytest.raises(SystemExit):
            parser.parse_args(['model/', '-b', 'invalid'])


class TestBackendSelection:
    """Test backend selection logic."""

    @patch('gpt_oss.generate.Path')
    @patch('gpt_oss.generate.get_tokenizer')
    def test_torch_backend_initialization(self, mock_get_tokenizer, mock_path):
        """Test torch backend is correctly initialized."""
        from gpt_oss.generate import main
        
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        args = argparse.Namespace(
            checkpoint='model/',
            backend='torch',
            prompt='test',
            temperature=0.0,
            limit=10
        )
        
        with patch('gpt_oss.generate.init_distributed') as mock_init_dist, \
             patch('gpt_oss.generate.TorchGenerator') as mock_torch_gen:
            
            mock_device = Mock()
            mock_init_dist.return_value = mock_device
            mock_generator = Mock()
            mock_generator.generate.return_value = iter([(1, 0.5)])
            mock_torch_gen.return_value = mock_generator
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_tokenizer.decode.return_value = "test"
            mock_tokenizer.eot_token = 0
            mock_get_tokenizer.return_value = mock_tokenizer
            
            main(args)
            
            mock_init_dist.assert_called_once()
            mock_torch_gen.assert_called_once_with('model/', device=mock_device)

    @patch('gpt_oss.generate.Path')
    @patch('gpt_oss.generate.get_tokenizer')
    def test_triton_backend_initialization(self, mock_get_tokenizer, mock_path):
        """Test triton backend is correctly initialized."""
        from gpt_oss.generate import main
        
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        args = argparse.Namespace(
            checkpoint='model/',
            backend='triton',
            prompt='test',
            temperature=0.0,
            limit=10
        )
        
        with patch('gpt_oss.generate.init_distributed') as mock_init_dist, \
             patch('gpt_oss.generate.TritonGenerator') as mock_triton_gen:
            
            mock_device = Mock()
            mock_init_dist.return_value = mock_device
            mock_generator = Mock()
            mock_generator.generate.return_value = iter([(1, 0.5)])
            mock_triton_gen.return_value = mock_generator
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_tokenizer.decode.return_value = "test"
            mock_tokenizer.eot_token = 0
            mock_get_tokenizer.return_value = mock_tokenizer
            
            main(args)
            
            mock_init_dist.assert_called_once()
            mock_triton_gen.assert_called_once_with('model/', context=4096, device=mock_device)

    @patch('gpt_oss.generate.Path')
    @patch('gpt_oss.generate.get_tokenizer')
    def test_vllm_backend_initialization(self, mock_get_tokenizer, mock_path):
        """Test vLLM backend is correctly initialized."""
        from gpt_oss.generate import main
        
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        args = argparse.Namespace(
            checkpoint='model/',
            backend='vllm',
            prompt='test',
            temperature=0.0,
            limit=10
        )
        
        with patch('gpt_oss.generate.VLLMGenerator') as mock_vllm_gen:
            mock_generator = Mock()
            mock_generator.generate.return_value = iter([(1, 0.5)])
            mock_vllm_gen.return_value = mock_generator
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_tokenizer.decode.return_value = "test"
            mock_tokenizer.eot_token = 0
            mock_get_tokenizer.return_value = mock_tokenizer
            
            main(args)
            
            mock_vllm_gen.assert_called_once_with('model/', tensor_parallel_size=2)

    @patch('gpt_oss.generate.Path')
    def test_invalid_backend_raises_error(self, mock_path):
        """Test that invalid backend raises ValueError."""
        from gpt_oss.generate import main
        
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        args = argparse.Namespace(
            checkpoint='model/',
            backend='invalid',
            prompt='test',
            temperature=0.0,
            limit=10
        )
        
        with pytest.raises(ValueError, match="Invalid backend"):
            main(args)


class TestCheckpointValidation:
    """Test checkpoint path validation."""

    @patch('gpt_oss.generate.Path')
    def test_nonexistent_checkpoint_raises_error(self, mock_path):
        """Test that nonexistent checkpoint path raises FileNotFoundError."""
        from gpt_oss.generate import main
        
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        args = argparse.Namespace(
            checkpoint='nonexistent/',
            backend='torch',
            prompt='test',
            temperature=0.0,
            limit=10
        )
        
        with pytest.raises(FileNotFoundError, match="Checkpoint path does not exist"):
            main(args)

    @patch('gpt_oss.generate.Path')
    @patch('gpt_oss.generate.get_tokenizer')
    def test_valid_checkpoint_path_accepted(self, mock_get_tokenizer, mock_path):
        """Test that valid checkpoint path is accepted."""
        from gpt_oss.generate import main
        
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        args = argparse.Namespace(
            checkpoint='valid/path/',
            backend='vllm',
            prompt='test',
            temperature=0.0,
            limit=10
        )
        
        with patch('gpt_oss.generate.VLLMGenerator') as mock_vllm_gen:
            mock_generator = Mock()
            mock_generator.generate.return_value = iter([(1, 0.5)])
            mock_vllm_gen.return_value = mock_generator
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_tokenizer.decode.return_value = "test"
            mock_tokenizer.eot_token = 0
            mock_get_tokenizer.return_value = mock_tokenizer
            
            # Should not raise
            main(args)


class TestGenerationFlow:
    """Test token generation flow."""

    @patch('gpt_oss.generate.Path')
    @patch('gpt_oss.generate.get_tokenizer')
    def test_generation_with_limit(self, mock_get_tokenizer, mock_path):
        """Test generation respects token limit."""
        from gpt_oss.generate import main
        
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        args = argparse.Namespace(
            checkpoint='model/',
            backend='vllm',
            prompt='test',
            temperature=0.5,
            limit=5
        )
        
        with patch('gpt_oss.generate.VLLMGenerator') as mock_vllm_gen:
            mock_generator = Mock()
            mock_generator.generate.return_value = iter([(i, 0.5) for i in range(10)])
            mock_vllm_gen.return_value = mock_generator
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_tokenizer.decode.return_value = "t"
            mock_tokenizer.eot_token = 999
            mock_get_tokenizer.return_value = mock_tokenizer
            
            main(args)
            
            # Verify max_tokens was set to limit
            call_kwargs = mock_generator.generate.call_args[1]
            assert call_kwargs['max_tokens'] == 5
            assert call_kwargs['temperature'] == 0.5

    @patch('gpt_oss.generate.Path')
    @patch('gpt_oss.generate.get_tokenizer')
    def test_generation_without_limit(self, mock_get_tokenizer, mock_path):
        """Test generation without token limit."""
        from gpt_oss.generate import main
        
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        args = argparse.Namespace(
            checkpoint='model/',
            backend='vllm',
            prompt='test',
            temperature=0.0,
            limit=0
        )
        
        with patch('gpt_oss.generate.VLLMGenerator') as mock_vllm_gen:
            mock_generator = Mock()
            mock_generator.generate.return_value = iter([(1, 0.5)])
            mock_vllm_gen.return_value = mock_generator
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_tokenizer.decode.return_value = "test"
            mock_tokenizer.eot_token = 0
            mock_get_tokenizer.return_value = mock_tokenizer
            
            main(args)
            
            # Verify max_tokens was set to None
            call_kwargs = mock_generator.generate.call_args[1]
            assert call_kwargs['max_tokens'] is None

    @patch('gpt_oss.generate.Path')
    @patch('gpt_oss.generate.get_tokenizer')
    def test_tokenizer_integration(self, mock_get_tokenizer, mock_path):
        """Test tokenizer is correctly used for encoding/decoding."""
        from gpt_oss.generate import main
        
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        test_prompt = "Why did the chicken cross the road?"
        args = argparse.Namespace(
            checkpoint='model/',
            backend='vllm',
            prompt=test_prompt,
            temperature=0.0,
            limit=10
        )
        
        with patch('gpt_oss.generate.VLLMGenerator') as mock_vllm_gen:
            mock_generator = Mock()
            mock_generator.generate.return_value = iter([(42, -0.5)])
            mock_vllm_gen.return_value = mock_generator
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.decode.return_value = "answer"
            mock_tokenizer.eot_token = 0
            mock_get_tokenizer.return_value = mock_tokenizer
            
            main(args)
            
            # Verify tokenizer was used correctly
            mock_tokenizer.encode.assert_called_once_with(test_prompt)
            mock_tokenizer.decode.assert_called_with([42])
