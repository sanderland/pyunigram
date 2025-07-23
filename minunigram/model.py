import math
from collections import Counter
from collections.abc import Sequence

from .lattice import Lattice
UNK_PENALTY = 10.0

class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, word: str, value: int):
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        node['<end>'] = value

    def common_prefix_search(self, text: str) -> list[tuple[str, int]]:
        results, node = [], self.root
        for i, char in enumerate(text):
            if char not in node:
                break
            node = node[char]
            if '<end>' in node:
                results.append((text[:i+1], node['<end>']))
        return results

class InternalModel:

    def __init__(self, scores: dict, vocab: dict | None = None, unk_id: int = 0, unk_token: str | None = None):
        self.scores = {int(k): v for k, v in scores.items()} if vocab else scores
        self.unk_id = unk_id
        self.unk_score = self.scores.get(self.unk_id, min(self.scores.values()) - UNK_PENALTY) if self.scores else -UNK_PENALTY
        
        self.trie = Trie()
        if vocab:
            id_to_piece = {v: k for k, v in vocab.items()}
            for vid in self.scores:
                if vid != self.unk_id:
                    self.trie.insert(id_to_piece[vid], vid)
        else:  # During training, keys are strings
            for piece in self.scores:
                self.trie.insert(piece, piece)

    def populate_nodes(self, lattice: 'Lattice'):
        for i in range(lattice.size):
            substring = lattice.text[i:]
            matches = self.trie.common_prefix_search(substring)
            for piece, vocab_id in matches:
                score = self.scores[vocab_id]
                lattice.insert(i, len(piece), vocab_id, score)
            
            has_single_char = any(len(p) == 1 for p, _ in matches)
            if not has_single_char and substring:
                # During training, the ID is the token string itself.
                lattice.insert(i, 1, self.unk_token_str, self.unk_score)

    def encode(self, normalized: str) -> list[tuple[str, int]]:
        if not normalized:
            return []
        lattice = Lattice(normalized)
        self.populate_nodes(lattice)
        nodes, _ = lattice.viterbi()
        return [(node['piece'], node['id']) for node in nodes]
    
    def encode_optimized(self, normalized: str) -> tuple[list[str], list[int], float]:
        if not normalized: return [], [], 0.0
        size = len(normalized)
        ends_at = [{'starts_at': -1, 'id': -1, 'score': -float('inf')} for _ in range(size + 1)]
        ends_at[0] = {'starts_at': 0, 'id': 0, 'score': 0.0}
        
        for i in range(size):
            if ends_at[i]['score'] == -float('inf'): continue
            substring = normalized[i:]
            matches = self.trie.common_prefix_search(substring)
            if not any(len(p) == 1 for p, _ in matches) and substring:
                matches.append((substring[0], self.unk_token_str if isinstance(next(iter(self.scores), ''), str) else self.unk_id))

            for piece, vocab_id in matches:
                score = self.scores.get(vocab_id, self.unk_score)
                candidate_score = ends_at[i]['score'] + score
                end_pos = i + len(piece)
                if end_pos <= size and candidate_score > ends_at[end_pos]['score']:
                    ends_at[end_pos] = {'score': candidate_score, 'starts_at': i, 'id': vocab_id}
        
        pieces, ids, pos = [], [], size
        while pos > 0:
            node = ends_at[pos]
            start_pos = node['starts_at']
            piece = normalized[start_pos:pos]
            pieces.append(piece)
            ids.append(node['id'] if node['id'] in self.scores else self.unk_id)
            pos = start_pos
        
        final_score = ends_at[size]['score']
        return pieces[::-1], ids[::-1], final_score