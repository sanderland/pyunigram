from py_unigram.train import train_unigram
from py_unigram.pretokenize import pretokenize_corpus
from datasets import load_dataset

test = False
if test:
    with open('swift.txt') as f:
        pretokens = pretokenize_corpus([f.read()])
    n = 1024
else:
    pretokens = pretokenize_corpus([text for text in load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"] if text.strip()])
    n = 16384

# Train the model
model = train_unigram(
    pretokens=pretokens,
    vocab_size=n,
    max_token_len=16,
    initial_vocab_factor=4,
    verbose=True
)

