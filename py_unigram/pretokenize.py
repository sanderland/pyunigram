from collections import Counter
from collections.abc import Iterable

import regex as re

GPT2_PRE_TOKENIZER_REGEX = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def pretokenize_corpus(corpus: Iterable[str], regex_pattern: str = GPT2_PRE_TOKENIZER_REGEX) -> dict[str, int]:
    """
    Applies regex pretokenization to a corpus and returns a frequency dict of pretokens.
    """
    if isinstance(corpus, str):
        raise ValueError("Corpus must be an iterable of strings, not a single string.")
    pretokenizer = re.compile(regex_pattern)
    pretoken_counter = Counter()
    for text in corpus:
        # Find all pretokens in the text
        found_pretokens = pretokenizer.findall(text)
        # Update the counter with the found pretokens
        pretoken_counter.update(found_pretokens)
    return dict(pretoken_counter)
