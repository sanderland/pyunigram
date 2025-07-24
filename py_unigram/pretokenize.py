from collections import Counter
from collections.abc import Iterable
import regex as re

GPT2_PRE_TOKENIZER_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

def pretokenize_corpus(corpus: Iterable[str], regex_pattern: str = GPT2_PRE_TOKENIZER_REGEX) -> dict[str, int]:
     """
     Applies regex pretokenization to a corpus and returns a frequency dict of pretokens.
     """
     pretokenizer = re.compile(regex_pattern)
     pretoken_counter = Counter()
     for text in corpus:
         # Find all pretokens in the text
         found_pretokens = pretokenizer.findall(text)
         # Update the counter with the found pretokens
         pretoken_counter.update(found_pretokens)
     return dict(pretoken_counter)
