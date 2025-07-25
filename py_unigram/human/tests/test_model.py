import math

import pytest

from py_unigram.human.model import Lattice, Token, Trie, UnigramModel

# --- Fixtures ---

@pytest.fixture
def simple_tokens():
    # id, text, log_prob
    return [
        Token(text="a", id=0, log_prob=math.log(0.5)),
        Token(text="b", id=1, log_prob=math.log(0.3)),
        Token(text="ab", id=2, log_prob=math.log(0.7)),
    ]

@pytest.fixture
def simple_trie(simple_tokens):
    return Trie(simple_tokens)

@pytest.fixture
def simple_lattice(simple_tokens):
    # For text 'ab', tokens_from_pos[0] = [ab, a], tokens_from_pos[1] = [b]
    tokens_from_pos = [
        [simple_tokens[2], simple_tokens[0]],  # 'ab', 'a' at pos 0
        [simple_tokens[1]],                    # 'b' at pos 1
    ]
    return Lattice("ab", tokens_from_pos)

@pytest.fixture
def simple_model(simple_tokens):
    return UnigramModel(simple_tokens)

# --- Trie tests ---
def test_trie_prefix_search(simple_trie):
    # Should find all prefixes for 'ab'
    matches = simple_trie.find_prefixes("ab")
    found_ids = {t.id for t in matches}
    assert 0 in found_ids  # 'a'
    assert 2 in found_ids  # 'ab'
    # Should not find anything for 'x'
    assert simple_trie.find_prefixes("x") == []

# --- Lattice tests ---
def test_lattice_viterbi_and_all_paths(simple_lattice):
    # Viterbi should find the best path (ab)
    path, score = simple_lattice.viterbi()
    assert [t.text for t in path] == ["ab"]
    # all_paths should return viterbi path first
    all_paths = list(simple_lattice.all_paths())
    assert [t.text for t in all_paths[0][0]] == ["ab"]
    # There should be another path: ['a', 'b']
    assert any([t.text for t in p] == ["a", "b"] for p, _ in all_paths)
    # Scores should match viterbi for first path
    assert math.isclose(all_paths[0][1], score)

def test_lattice_viterbi_empty():
    # Empty lattice
    lattice = Lattice("", [])
    path, score = lattice.viterbi()
    assert path == []
    assert score == float('-inf') or score == 0.0

def test_lattice_marginal_probabilities(simple_lattice):
    probs = simple_lattice.marginal_probabilities()
    # All tokens should have probabilities between 0 and 1
    for v in probs.values():
        assert 0.0 <= v <= 1.0
    # Should sum to ~1 (or less, if not all tokens are used)
    assert sum(probs.values()) == pytest.approx(1.0, rel=1e-6)

# --- UnigramModel tests ---
def test_unigram_model_tokenize(simple_model):
    # Should return viterbi path tokens
    tokens = simple_model.tokenize("ab")
    assert [t.text for t in tokens] == ["ab"]

@pytest.mark.parametrize("text,expected", [
    ("ab", [("ab",), ("a", "b")]),
    ("a", [("a",)]),
    ("b", [("b",)]),
    ("x", []),
])
def test_unigram_model_make_lattice_paths(simple_model, text, expected):
    lattice = simple_model.make_lattice(text)
    all_paths = [tuple(t.text for t in p) for p, _ in lattice.all_paths()]
    assert set(all_paths) == set(expected)




