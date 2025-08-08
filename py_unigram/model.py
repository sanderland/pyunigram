import math
from collections.abc import Iterable
from collections import defaultdict
from dataclasses import dataclass


def logaddexp(a: float, b: float) -> float:
    """Stable log(exp(a) + exp(b)) for two finite terms."""
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))

@dataclass
class Token:
    text: str
    id: int
    log_prob: float
    locked: bool = False # if true, can not be removed

class Trie:
    def __init__(self, tokens: list[Token]):
        self.root = {}
        for token in tokens:
            self.insert(token)

    def insert(self, token: Token):
        node = self.root
        for char in token.text:
            node = node.setdefault(char, {})
        node[None] = token

    def find_prefixes(self, text: str) -> list[Token]:
        results = []
        node = self.root
        for char in text:
            if char not in node:
                break
            node = node[char]
            if None in node:
                results.append(node[None])
        return results


class Lattice:
    def __init__(self, text: str, tokens_from_pos: list[list[Token]]):
        self.text = text
        self.tokens_from_pos = tokens_from_pos

    def viterbi(self, allow_single_token=True) -> tuple[list[Token], float]:
        best_at_pos = [(None,0)] + [(None,float('-inf'))] * len(self.text)
        for pos in range(len(self.text)):
            for token in self.tokens_from_pos[pos]:
                end = pos + len(token.text)
                if pos==0 and not allow_single_token and end==len(self.text):
                    continue # do not allow direct path
                score = best_at_pos[pos][1] + token.log_prob
                if score > best_at_pos[end][1]:
                    best_at_pos[end] = (token, score)

        path = []
        pos = len(self.text)
        while pos > 0:
            token, score = best_at_pos[pos]
            if token is None:
                break
            path.append(token)
            pos -= len(token.text)
        return path[::-1], best_at_pos[-1][1]

    def all_paths(self, starting_pos: int = 0) -> Iterable[tuple[tuple[Token], float]]:
        if starting_pos == len(self.text):
            yield (tuple(), 0.0)
            return
        for token in self.tokens_from_pos[starting_pos]:
            for sub_path, sub_prob in self.all_paths(starting_pos + len(token.text)):
                yield ( (token,) + sub_path, sub_prob + token.log_prob)

    def _forward_backward(self) -> tuple[list[float], list[float]]:
        """returns
        alpha(pos) = total prob of path to pos
        beta(pos) = total prob of path from pos to end
        """
        alpha = [0] + [float('-inf')] * len(self.text)
        beta = [float('-inf')] * (len(self.text)) + [0]
        for pos in range(len(self.text)):
            if alpha[pos] != float('-inf'):
                for token in self.tokens_from_pos[pos]:
                    alpha[pos + len(token.text)] = logaddexp(alpha[pos + len(token.text)], alpha[pos] + token.log_prob)
        for pos in range(len(self.text)-1, -1, -1):
            for token in self.tokens_from_pos[pos]:
                if beta[pos + len(token.text)] != float('-inf'):
                    beta[pos] = logaddexp(beta[pos], beta[pos + len(token.text)] + token.log_prob)
        return alpha, beta

    def calc_marginal(self) -> tuple[float, dict[int, float]]:
        alpha, beta = self._forward_backward()
        z = alpha[-1]
        assert z != float('-inf'), f"Lattice for {self.text!r} has no valid paths with tokens_from_pos {self.tokens_from_pos}"
        token_prob = defaultdict(float)
        for pos in range(len(self.text)):
            for token in self.tokens_from_pos[pos]:
                token_logprob = alpha[pos] + token.log_prob + beta[pos + len(token.text)] - z
                token_prob[token.id] += math.exp(max(-100, token_logprob))  # Avoid underflow

        return z, token_prob




class UnigramModel:
    """Unigram model of text: essentially a collection of tokens and their logprobs."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.tokens_by_id = {token.id: token for token in tokens}
        self.trie = Trie(self.tokens)
        
    def make_lattice(self, text: str) -> Lattice:
        tokens_from_pos = [self.trie.find_prefixes(text[i:]) for i in range(len(text))]
        return Lattice(text, tokens_from_pos)

    def encode(self, text: str, return_tokens=False) -> list[Token] | list[int]:
        lattice = self.make_lattice(text)
        tokens = lattice.viterbi()[0]
        if return_tokens:
            return tokens
        else:
            return [token.id for token in tokens]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.tokens_by_id[id].text for id in ids)
