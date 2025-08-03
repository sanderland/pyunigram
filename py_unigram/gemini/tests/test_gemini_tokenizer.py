
import pytest

from py_unigram.gemini.tokenizer import GeminiUnigramTokenizer
from py_unigram.gemini.train import train_unigram_model

BASE_ALPHABET = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r")

@pytest.fixture
def tokenizer():
    corpus = ["abracadabra", "brabrabra", "veldig!bra!","bra jobba!", "greit jobba!"]
    vocab_size = len(BASE_ALPHABET) + 5
    return train_unigram_model(corpus, vocab_size=vocab_size, initial_vocab_size_factor=2, required_chars=BASE_ALPHABET)

@pytest.fixture
def tmp_file(tmp_path):
    return tmp_path / "test_tokenizer.json"


def test_training_vocab_size(tokenizer):
    """Test if the trained vocabulary size is close to the target size."""
    # The final size can be slightly different due to keeping all single chars
    vocab_len = len(tokenizer.vocab)
    assert vocab_len == len(BASE_ALPHABET) + 5 + 1 # +1 for the <unk> token
    assert "<unk>" in tokenizer.vocab


def test_save_and_load_consistency(tokenizer, tmp_file):
    """Test that saving and loading a tokenizer results in an identical object."""
    tokenizer.save(tmp_file)
    loaded_tokenizer = GeminiUnigramTokenizer.load(tmp_file)

    assert tokenizer.vocab == loaded_tokenizer.vocab
    assert tokenizer.scores == loaded_tokenizer.scores

    original_output = tokenizer.encode("abracadabra")
    loaded_output = loaded_tokenizer.encode("abracadabra")
    assert original_output == loaded_output


def test_encoding_logic(tokenizer):
    """Test the encoding logic on a known word."""
    # 'bra' is very common in the corpus, so it should be a single token
    token_ids, tokens = tokenizer.encode("abracadabra", return_tokens=True)

    # Check if 'bra' is a token
    print(tokenizer.vocab)

    assert 'bra' in tokenizer.vocab

    # Check if the encoding is what we expect
    assert tokens == ['a', 'bra', 'c', 'a', 'd', 'a', 'bra']


def test_pre_tokenization(tokenizer):
    """Test that pre-tokenization correctly splits text."""
    text = "hello world, test."
    _, tokens = tokenizer.encode(text, return_tokens=True)

    # The output should preserve the input text exactly
    reconstructed = "".join(tokens)
    assert reconstructed == text



def test_unknown_token_handling(tokenizer):
    """Test the handling of characters not seen during training."""
    # Single characters should be handled by the vocab
    text = "xyz"
    token_ids = tokenizer.encode(text)
    # Convert back to check if we get the same result
    decoded = tokenizer.decode(token_ids)
    assert decoded == text

    # Special characters should be handled properly
    text = "$$%"
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)
    assert decoded == text


def test_encode_return_tokens(tokenizer):
    """Test the encode method with return_tokens=True."""
    # Test with a simple word from the training corpus
    token_ids, tokens = tokenizer.encode("abra", return_tokens=True)
    assert isinstance(token_ids, list)
    assert isinstance(tokens, list)
    assert len(token_ids) == len(tokens)
    # Check if we can decode the ids back to the original text
    assert tokenizer.decode(token_ids) == "abra"
    # Each id should correspond to its token
    for token_id, token in zip(token_ids, tokens, strict=False):
        assert tokenizer.id_to_token[token_id] == token


def test_decode_edge_cases(tokenizer):
    """Test decoding with various edge cases."""
    # Test empty sequence
    assert tokenizer.decode([]) == ""

    # Test unknown ids
    assert tokenizer.decode([-1]) == tokenizer.unk_token
    assert tokenizer.decode([999999]) == tokenizer.unk_token

    # Test mixed known and unknown ids
    token_ids = tokenizer.encode("abra")
    mixed_ids = token_ids + [-1] + token_ids
    assert tokenizer.decode(mixed_ids) == f"abra{tokenizer.unk_token}abra"


def test_roundtrip_encoding(tokenizer):
    """Test that encoding and decoding preserves the text."""
    test_cases = [
        "abracadabra",  # Training word
        "hello world",   # Space handling
        "!!??....",      # Punctuation
        "  spaced  ",   # Multiple spaces
        "a b c",        # Single characters with spaces
        "1234",         # Numbers
        "!@#$%",        # Special characters
        "\n\t",         # Whitespace characters
        "",             # Empty string
    ]

    for text in test_cases:
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        assert decoded == text, f"Failed roundtrip for: {text!r}"


def test_consistent_encoding(tokenizer):
    """Test that encoding is consistent across multiple calls."""
    text = "test text with spaces and !@#$ punctuation"

    # Multiple encode calls should give the same results
    result1 = tokenizer.encode(text)
    result2 = tokenizer.encode(text)
    assert result1 == result2

    # With and without return_tokens should give consistent ids
    tokenizer.encode(text)
    ids_with_tokens, _ = tokenizer.encode(text, return_tokens=True)
