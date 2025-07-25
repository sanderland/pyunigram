# tests/test_train.py
import math

import pytest

from py_unigram.pretokenize import pretokenize_corpus
from py_unigram.qwen.train import train_unigram

# --- Fixtures ---

@pytest.fixture
def sample_corpus_lines():
    return [
        "Hello world! This is a test.",
        "Another sentence for testing.",
        "The quick brown fox jumps over the lazy dog.",
        "Unicode: Héllo wörld! 🌍",
        "Repetition repetition repetition.",
        "Short.",
        "Numbers 123 and symbols !@# should be handled.",
    ]

@pytest.fixture
def sample_pretokens(sample_corpus_lines):
    from demo.pretokenize import GPT2_PRE_TOKENIZER_REGEX
    return pretokenize_corpus(sample_corpus_lines, GPT2_PRE_TOKENIZER_REGEX)

# --- Tests ---

class TestTrainUnigram:

    def test_train_with_empty_pretokens(self):
        with pytest.raises(ValueError):
            train_unigram({})

    def test_train_basic_functionality(self, sample_pretokens):
        vocab_size = 100
        pieces = train_unigram(
            pretokens=sample_pretokens,
            vocab_size=vocab_size,
            seed_size_factor=2,
            num_sub_iterations=1,
            max_piece_len=20,
            verbose=False
        )

        assert isinstance(pieces, list)
        assert len(pieces) <= vocab_size

        for piece, score in pieces:
            assert isinstance(piece, str)
            assert isinstance(score, float)
            assert not math.isnan(score)
            assert not math.isinf(score)

        scores = [score for _, score in pieces]
        assert scores == sorted(scores, reverse=True)

    def test_train_with_required_chars(self, sample_pretokens):
        vocab_size = 50
        required_chars = ['\n', 'é', '🐶', 'Z']
        pieces = train_unigram(
            pretokens=sample_pretokens,
            vocab_size=vocab_size,
            required_chars=required_chars,
            seed_size_factor=2,
            num_sub_iterations=1,
            verbose=False
        )

        assert len(pieces) <= vocab_size

    def test_train_vocab_size_vs_meta_pieces(self):
        pretokens = {"a": 1, "b": 1}
        vocab_size_equal_meta = 1
        pieces_equal_meta = train_unigram(
            pretokens=pretokens,
            vocab_size=vocab_size_equal_meta,
            meta_pieces=["<unk>"],
            seed_size_factor=10,
            num_sub_iterations=1,
            verbose=False
        )
        assert len(pieces_equal_meta) == vocab_size_equal_meta

        vocab_size_less_meta = 0
        pieces_less_meta = train_unigram(
            pretokens=pretokens,
            vocab_size=vocab_size_less_meta,
            meta_pieces=["<unk>"],
            seed_size_factor=10,
            num_sub_iterations=1,
            verbose=False
        )
        assert len(pieces_less_meta) == vocab_size_less_meta

    def test_train_deterministic_output(self, sample_pretokens):
        kwargs = {
            'pretokens': sample_pretokens,
            'vocab_size': 30,
            'seed_size_factor': 2,
            'num_sub_iterations': 1,
            'verbose': False
        }

        pieces1 = train_unigram(**kwargs)
        pieces2 = train_unigram(**kwargs)

        assert len(pieces1) == len(pieces2)
        for p1, p2 in zip(pieces1, pieces2, strict=False):
            assert p1[0] == p2[0]
            assert p1[1] == pytest.approx(p2[1])

    def test_train_hyperparameters_impact(self, sample_pretokens):
        base_kwargs = {
            'pretokens': sample_pretokens,
            'vocab_size': 50,
            'seed_size_factor': 2,
            'num_sub_iterations': 1,
            'verbose': False
        }

        pieces_base = train_unigram(**base_kwargs)
        pieces_alpha = train_unigram(**base_kwargs, dirichlet_alpha=0.1)
        assert pieces_base != pieces_alpha

        pieces_vocab = train_unigram(**{**base_kwargs, 'vocab_size': 60})
        assert pieces_base != pieces_vocab
        assert len(pieces_vocab) <= 60
