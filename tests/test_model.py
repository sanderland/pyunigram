from minunigram import train_unigram_model
from minunigram.model import Trie

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
