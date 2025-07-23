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
    def __init__(self, text: str):
        self.text = text
        self.size = len(text)
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
        node = self._new_node(pos=pos, length=length, piece=self.text[pos:pos+length], id=vocab_id, score=score)
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

    def populate_marginal(self, expected_counts: Counter, count: int = 1) -> float:
        """
        Runs the forward-backward algorithm, calculates the marginal probability
        of each node, scales it by `count`, and adds it to the `expected_counts`.

        Args:
            expected_counts: The global Counter object to update.
            count: The frequency of the sentence/pretoken, used to scale the results.

        Returns:
            The total log-likelihood (Z) of the sentence.
        """
        if not self.text:
            return 0.0

        # Run the forward and backward passes.
        alpha = self._forward()
        beta = self._backward()

        # Z is the total log-likelihood (partition function).
        z = alpha[self.begin_nodes[self.size][0]['node_id']]
        if z == -float('inf'):
            return 0.0

        # Iterate over all nodes to calculate their marginal probability.
        for nodes in self.begin_nodes:
            for node in nodes:
                # Skip special BOS/EOS nodes.
                if node['id'] != -1:
                    
                    # Standard textbook formula for the marginal probability of a state (node).
                    log_prob = alpha[node['node_id']] + beta[node['node_id']] - z
                    
                    # Convert log-prob to a regular probability, scale by the pretoken's `count`,
                    # and add it directly to the external Counter object.
                    expected_counts[node['id']] += math.exp(log_prob) * count

        # Return the total log-likelihood of a single instance of the sentence.
        return z

    def tokens(self):
        path, _ = self.viterbi()
        return [node['piece'] for node in path]
