from py_unigram.human.train import train_unigram
from py_unigram.pretokenize import pretokenize_corpus

with open('swift.txt') as f:
    pretokens = pretokenize_corpus([f.read()])

# Train the model
model = train_unigram(
    pretokens=pretokens,
    vocab_size=1024,
    max_token_len=16,
    initial_vocab_factor=4,
    verbose=True
)

