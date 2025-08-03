from py_unigram.gemini.train import train_unigram_model


def test_train_small_vocab():
    corpus = ["abc", "bcd", "cde"]
    # vocab_size < unique chars, should still include <unk> and base chars
    tokenizer = train_unigram_model(corpus, vocab_size=2)
    assert "<unk>" in tokenizer.vocab
    for c in set("abcde"):
        assert c in tokenizer.vocab

def test_train_pruning():
    corpus = ["a"*100, "b"*50, "c"*10]
    tokenizer = train_unigram_model(corpus, vocab_size=3)
    # Only most frequent tokens should remain (plus <unk>)
    tokens = set(tokenizer.vocab.keys())
    assert "a" in tokens
    assert "b" in tokens or "c" in tokens
    assert "<unk>" in tokens
