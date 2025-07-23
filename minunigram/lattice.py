import math
from collections import Counter


def log_sum_exp(a, b):
    if a == -float('inf'):
        return b
    if b == -float('inf'):
        return a
    m = max(a, b)
    return m + math.log(1 + math.exp(-abs(a - b)))


class Lattice:
    def __init__(self, sentence: str):
        self.sentence = sentence
        self.size = len(sentence)
        self.nodes = []
        self.begin_nodes = [[] for _ in range(self.size + 1)]
        self.end_nodes = [[] for _ in range(self.size + 1)]
        bos = self._new_node(pos=0, length=0, id=-1, piece="BOS", score=0.0)
        self.end_nodes[0].append(bos)
        eos = self._new_node(pos=self.size, length=0, id=-1, piece="EOS", score=0.0)
        self.begin_nodes[self.size].append(eos)

    def _new_node(self, **kwargs) -> dict:
        node = kwargs
        node['node_id'] = len(self.nodes)
        self.nodes.append(node)
        return node

    def insert(self, pos: int, length: int, vocab_id, score: float):
        node = self._new_node(pos=pos, length=length, piece=self.sentence[pos:pos+length], id=vocab_id, score=score)
        self.begin_nodes[pos].append(node)
        self.end_nodes[pos + length].append(node)

    def viterbi(self) -> tuple[list[dict], float]:
        for pos in range(self.size + 1):
            for rnode in self.begin_nodes[pos]:
                best_score, best_node = -float('inf'), None
                for lnode in self.end_nodes[pos]:
                    score = lnode.get('backtrace_score', 0.0) + rnode.get('score', 0.0)
                    if score > best_score:
                        best_score, best_node = score, lnode
                rnode['backtrace_score'], rnode['prev'] = best_score, best_node
        
        path, node = [], self.begin_nodes[self.size][0]
        final_score = node['backtrace_score']
        node = node.get('prev')
        while node and node['piece'] != "BOS":
            path.append(node)
            node = node.get('prev')
        return path[::-1], final_score


    def _forward(self) -> list[float]:
        alpha = [-float('inf')] * len(self.nodes)
        alpha[self.end_nodes[0][0]['node_id']] = 0.0
        for pos in range(self.size + 1):
            for rnode in self.begin_nodes[pos]:
                log_prob = -float('inf')
                for lnode in self.end_nodes[pos]:
                    log_prob = log_sum_exp(log_prob, alpha[lnode['node_id']] + lnode.get('score', 0.0))
                alpha[rnode['node_id']] = log_prob
        return alpha

    def _backward(self) -> list[float]:
        beta = [-float('inf')] * len(self.nodes)
        beta[self.begin_nodes[self.size][0]['node_id']] = 0.0
        for pos in range(self.size, -1, -1):
            for lnode in self.end_nodes[pos]:
                log_prob = -float('inf')
                for rnode in self.begin_nodes[pos]:
                    log_prob = log_sum_exp(log_prob, beta[rnode['node_id']] + rnode.get('score', 0.0))
                beta[lnode['node_id']] = log_prob
        return beta

    def populate_marginal(self, expected_counts: Counter) -> float:
        if not self.sentence:
            return 0.0
        alpha = self._forward()
        beta = self._backward()
        z = alpha[self.begin_nodes[self.size][0]['node_id']]
        if z == -float('inf'):
            return 0.0
        for nodes in self.begin_nodes:
            for node in nodes:
                if node['id'] != -1:
                    prob = math.exp(alpha[node['node_id']] + node['score'] +
                                  beta[self.end_nodes[node['pos']+node['length']][0]['node_id']] - z)
                    expected_counts[node['id']] += prob
        return z

    def tokens(self):
        path, _ = self.viterbi()
        return [node['piece'] for node in path]
