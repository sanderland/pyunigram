from py_unigram.pretokenize import pretokenize_corpus
from py_unigram.train import train_unigram

with open("swift.txt") as f:
    pretokens = pretokenize_corpus([f.read()])
n = 1024

model = train_unigram(pretokens=pretokens, vocab_size=n, max_token_len=16, initial_vocab_factor=4, verbose=True)
