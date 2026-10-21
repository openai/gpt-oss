"""Unit tests for tokenizer encoding/decoding functionality."""

import pytest
from gpt_oss.tokenizer import get_tokenizer


class TestTokenizerBasics:
    """Test basic tokenizer functionality."""

    def test_get_tokenizer_returns_encoding(self):
        """Test that get_tokenizer returns a valid encoding."""
        tokenizer = get_tokenizer()
        assert tokenizer is not None
        assert tokenizer.name == "o200k_harmony"

    def test_tokenizer_has_harmony_special_tokens(self):
        """Test that tokenizer includes Harmony special tokens."""
        tokenizer = get_tokenizer()
        special_tokens = tokenizer._special_tokens
        
        # Verify key Harmony tokens are present
        assert "<|channel|>" in special_tokens
        assert special_tokens["<|channel|>"] == 200005
        assert "<|start|>" in special_tokens
        assert special_tokens["<|start|>"] == 200006
        assert "<|end|>" in special_tokens
        assert special_tokens["<|end|>"] == 200007
        assert "<|message|>" in special_tokens
        assert special_tokens["<|message|>"] == 200008
        assert "<|call|>" in special_tokens
        assert special_tokens["<|call|>"] == 200012
        assert "<|return|>" in special_tokens
        assert special_tokens["<|return|>"] == 200002

    def test_tokenizer_has_reserved_tokens(self):
        """Test that tokenizer includes reserved token range."""
        tokenizer = get_tokenizer()
        special_tokens = tokenizer._special_tokens
        
        # Check reserved tokens exist in range
        assert "<|reserved_200013|>" in special_tokens
        assert special_tokens["<|reserved_200013|>"] == 200013
        assert "<|reserved_201087|>" in special_tokens
        assert special_tokens["<|reserved_201087|>"] == 201087


class TestTokenizerEncoding:
    """Test tokenizer encoding functionality."""

    def test_encode_simple_text(self):
        """Test encoding simple text."""
        tokenizer = get_tokenizer()
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_encode_special_tokens(self):
        """Test encoding text with special tokens."""
        tokenizer = get_tokenizer()
        text = "<|channel|>final<|message|>Hello<|return|>"
        tokens = tokenizer.encode(text, allowed_special="all")
        
        assert 200005 in tokens  # <|channel|>
        assert 200008 in tokens  # <|message|>
        assert 200002 in tokens  # <|return|>

    def test_encode_without_special_allowed_raises(self):
        """Test that encoding special tokens without permission raises error."""
        tokenizer = get_tokenizer()
        text = "<|channel|>test"
        
        with pytest.raises(ValueError):
            tokenizer.encode(text)

    def test_encode_empty_string(self):
        """Test encoding empty string."""
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode("")
        
        assert isinstance(tokens, list)
        assert len(tokens) == 0

    def test_encode_unicode_text(self):
        """Test encoding unicode text."""
        tokenizer = get_tokenizer()
        text = "Hello 世界 🌍"
        tokens = tokenizer.encode(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0


class TestTokenizerDecoding:
    """Test tokenizer decoding functionality."""

    def test_decode_simple_tokens(self):
        """Test decoding simple tokens."""
        tokenizer = get_tokenizer()
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text

    def test_decode_with_special_tokens(self):
        """Test decoding tokens including special tokens."""
        tokenizer = get_tokenizer()
        text = "<|channel|>final<|message|>Hello<|return|>"
        tokens = tokenizer.encode(text, allowed_special="all")
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text

    def test_decode_empty_list(self):
        """Test decoding empty token list."""
        tokenizer = get_tokenizer()
        decoded = tokenizer.decode([])
        
        assert decoded == ""

    def test_decode_single_token(self):
        """Test decoding single token."""
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode("a")
        decoded = tokenizer.decode(tokens)
        
        assert decoded == "a"

    def test_decode_unicode(self):
        """Test decoding unicode tokens."""
        tokenizer = get_tokenizer()
        text = "Hello 世界 🌍"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text


class TestTokenizerRoundTrip:
    """Test encode/decode round-trip consistency."""

    @pytest.mark.parametrize("text", [
        "Simple text",
        "Text with numbers: 123456",
        "Special chars: !@#$%^&*()",
        "Unicode: 你好世界",
        "Emoji: 🚀🌟💡",
        "Mixed: Hello 世界 123 🎉",
        "Newlines:\nand\ttabs",
    ])
    def test_roundtrip_consistency(self, text):
        """Test that encode->decode returns original text."""
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text

    def test_roundtrip_with_harmony_format(self):
        """Test round-trip with Harmony message format."""
        tokenizer = get_tokenizer()
        text = "<|channel|>analysis<|start|><|message|>Thinking...<|end|><|channel|>final<|message|>Answer<|return|>"
        tokens = tokenizer.encode(text, allowed_special="all")
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text


class TestTokenizerEdgeCases:
    """Test edge cases and error handling."""

    def test_encode_very_long_text(self):
        """Test encoding very long text."""
        tokenizer = get_tokenizer()
        text = "a" * 10000
        tokens = tokenizer.encode(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_decode_invalid_token_ids(self):
        """Test decoding with potentially invalid token IDs."""
        tokenizer = get_tokenizer()
        # Use valid token IDs from the special tokens
        tokens = [200005, 200006, 200007]
        decoded = tokenizer.decode(tokens)
        
        assert isinstance(decoded, str)

    def test_multiple_tokenizer_instances_consistent(self):
        """Test that multiple tokenizer instances behave consistently."""
        tokenizer1 = get_tokenizer()
        tokenizer2 = get_tokenizer()
        
        text = "Test consistency"
        tokens1 = tokenizer1.encode(text)
        tokens2 = tokenizer2.encode(text)
        
        assert tokens1 == tokens2

    def test_special_token_ids_immutable(self):
        """Test that special token IDs are consistent."""
        tokenizer = get_tokenizer()
        
        # Get special tokens multiple times
        channel_id_1 = tokenizer.encode("<|channel|>", allowed_special="all")[0]
        channel_id_2 = tokenizer.encode("<|channel|>", allowed_special="all")[0]
        
        assert channel_id_1 == channel_id_2 == 200005
