from py_unigram.gemini import train_unigram_model
from py_unigram.gemini.lattice import Lattice
from py_unigram.gemini.model import Trie

# -- lattice

def test_lattice_empty_and_unicode():
    # Empty string
    lattice = Lattice("")
    assert lattice.size == 0
    assert lattice.text == ""
    # Unicode string
    s = "テストab"
    lattice = Lattice(s)
    assert lattice.size == len(s)
    assert lattice.text == s
    # Surfaces
    assert lattice.text[0:] == s
    assert lattice.text[1:] == s[1:]
    assert lattice.text[2:] == s[2:]
    assert lattice.text[3:] == s[3:]
    assert lattice.text[4:] == s[4:]

def test_lattice_insert_and_viterbi():
    lattice = Lattice("ABあい")
    lattice.insert(0, 1, 3, 0.0)  # A
    lattice.insert(1, 1, 4, 0.0)  # B
    lattice.insert(2, 1, 5, 0.0)  # あ
    lattice.insert(3, 1, 6, 0.0)  # い
    lattice.insert(0, 2, 7, 0.0)  # AB
    lattice.insert(1, 3, 8, 0.0)  # Bあい
    lattice.insert(2, 2, 9, 0.0)  # あい
    # Check pieces
    assert lattice.nodes[2]['piece'] == "A"
    assert lattice.nodes[3]['piece'] == "B"
    assert lattice.nodes[4]['piece'] == "あ"
    assert lattice.nodes[5]['piece'] == "い"
    assert lattice.nodes[6]['piece'] == "AB"
    assert lattice.nodes[7]['piece'] == "Bあい"
    assert lattice.nodes[8]['piece'] == "あい"
    # Viterbi path
    lattice.insert(0, 4, 10, 1.0)  # ABあい
    path, score = lattice.viterbi()
    assert [node['piece'] for node in path] == ["ABあい"]

def test_lattice_nbest():
    lattice = Lattice("ABC")
    lattice.insert(0, 1, 3, 0.0)
    lattice.insert(1, 1, 4, 0.0)
    lattice.insert(2, 1, 5, 0.0)
    lattice.insert(0, 2, 6, 2.0)
    lattice.insert(1, 2, 7, 5.0)
    lattice.insert(0, 3, 8, 10.0)
    # nbest tokens
    tokens = lattice.tokens()
    assert tokens == ["ABC"]
    # Remove highest score, check other paths
    lattice.nodes[-1]['score'] = -float('inf')
    tokens = lattice.tokens()
    assert tokens in (["A", "BC"], ["AB", "C"])

# -- model

def test_token_to_id_and_id_to_token():
    corpus = ["abc", "bcd", "cde"]
    tokenizer = train_unigram_model(corpus, vocab_size=10)
    # Check token_to_id and id_to_token mapping
    for token, idx in tokenizer.vocab.items():
        assert tokenizer.id_to_token[idx] == token
        assert tokenizer.vocab[token] == idx

def test_encode_empty():
    tokenizer = train_unigram_model(["abc"], vocab_size=5)
    assert tokenizer.encode("") == []

def test_encode_unknown():
    tokenizer = train_unigram_model(["abc"], vocab_size=5)
    # Use a string with chars not in corpus
    ids = tokenizer.encode("xyz")
    # All should be mapped to single-char tokens or <unk>
    for i in ids:
        assert i in tokenizer.vocab.values()

def test_encode_mixed_known_unknown():
    tokenizer = train_unigram_model(["abc"], vocab_size=5)
    ids = tokenizer.encode("abcxyz")
    # Should contain known and unknown ids
    assert any(tokenizer.id_to_token[i] in "abc" for i in ids)
    assert any(tokenizer.id_to_token[i] not in "abc" for i in ids)

def test_encode_optimized_vs_unoptimized():
    tokenizer = train_unigram_model(["abcdabcd"], vocab_size=10)
    ids_opt = tokenizer.encode("abcdabcd")
    # Forcing unoptimized: use internal model directly
    model = tokenizer._model
    ids_unopt = [id for _, id in model.encode("abcdabcd")]
    assert ids_opt == ids_unopt

def test_decode_roundtrip():
    tokenizer = train_unigram_model(["abc", "def"], vocab_size=10)
    text = "abcdef"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text

def test_trie_prefix_search():
    trie = Trie()
    trie.insert("a", 1)
    trie.insert("ab", 2)
    trie.insert("abc", 3)
    # Search for "abc"
    matches = trie.common_prefix_search("abc")
    assert matches == [("a", 1), ("ab", 2), ("abc", 3)]
    # Search for "ab"
    matches = trie.common_prefix_search("ab")
    assert matches == [("a", 1), ("ab", 2)]
    # Search for "x" (no match)
    matches = trie.common_prefix_search("x")
    assert matches == []
