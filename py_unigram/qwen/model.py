# model.py
import math
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class Node:
    """
    Represents a potential segment (substring) in the Lattice.
    Stores information needed for Viterbi decoding and N-best path calculation.
    """
    # --- Core Attributes ---
    pos: int              # Start position of this segment in the sentence
    length: int           # Length of this segment
    piece_id: int         # Identifier for the corresponding piece in the model
    log_prob: float       # Log probability score of this piece according to the model

    # --- Viterbi Algorithm Attributes ---
    prev: Optional['Node'] = None  # Pointer to the previous node in the best path to this node
    backtrace_score: float = float('-inf') # The maximum log score to reach this node

    # --- N-Best Algorithm Attributes ---
    suffix_next: Optional['Node'] = None # Pointer to the next node in the best path from this node to the sentence end
    beta_score: float = float('-inf')    # The maximum log score from this node to the sentence end

    def __hash__(self):
        return hash((self.pos, self.length, self.piece_id, self.log_prob))

class Lattice:
    """
    Represents a graph of possible segmentations for a single string (e.g., a pretoken).
    Used by TrainerModel to calculate Viterbi paths, marginal probabilities (EM),
    and alternative paths (N-best) for pruning.
    """

    def __init__(self, sentence: str):
        """
        Initializes the Lattice for a given sentence string.
        Args:
            sentence: The string to be segmented.
        """
        self.sentence: str = sentence
        self.chars: List[str] = list(sentence)
        # nodes[pos] contains all Node objects that start at character index 'pos'
        self.nodes: List[List[Node]] = [[] for _ in range(len(self.chars) + 1)]
        
        # --- Caching for Computed Results ---
        # Viterbi Path: The single most likely segmentation.
        self._viterbi_path: Optional[List[Node]] = None
        self._viterbi_score: Optional[float] = None
        
        # N-Best Paths: The top-N most likely segmentations.
        self._nbest_cache_n = 0
        self._nbest_cache_results: List[Tuple[List[Node], float]] = []
        
        # Forward/Backward variables for EM (PopulateMarginal).
        self.alpha: List[float] = [] # Forward log probabilities
        self.beta: List[float] = []  # Backward log probabilities

    def insert_node(self, pos: int, length: int, piece_id: int, log_prob: float) -> None:
        """
        Adds a potential segment (Node) to the lattice.
        Invalidates cached computation results.
        Args:
            pos: Start position of the segment.
            length: Length of the segment.
            piece_id: Identifier for the piece.
            log_prob: Log probability of the piece.
        """
        if 0 <= pos <= len(self.chars) and length > 0 and pos + length <= len(self.chars):
            node = Node(pos, length, piece_id, log_prob)
            self.nodes[pos].append(node)
            # Invalidate caches as the lattice structure has changed.
            self._invalidate_caches()

    def _invalidate_caches(self) -> None:
        """Clears cached results that depend on the lattice's node structure."""
        self._viterbi_path = None
        self._viterbi_score = None
        self._nbest_cache_n = 0
        self._nbest_cache_results = []
        self.alpha = []
        self.beta = []

    def _compute_viterbi_and_nbest_data(self) -> None:
        """
        Performs the core calculations for Viterbi and N-best algorithms.
        1. Viterbi (Forward Pass): Finds the single best path score to each node.
        2. N-best Prep (Backward Pass): Calculates the best score from each node to the end
           and stores pointers for path reconstruction.
        Caches the results for Viterbi and N-best.
        """
        if self._viterbi_path is not None: # Already computed
            return 

        sentence_len = len(self.chars)

        if not sentence_len:
            self._viterbi_path = []
            self._viterbi_score = 0.0
            for pos_nodes in self.nodes:
                for node in pos_nodes:
                    node.beta_score = float('-inf')
                    node.suffix_next = None
            return

        # --- 1. Forward Viterbi Pass (Max Product) ---
        # Calculates the maximum score to reach each node.
        for pos_nodes in self.nodes:
            for node in pos_nodes:
                node.backtrace_score = float('-inf')
                node.prev = None

        # Initialize start nodes (those starting at position 0)
        for node in self.nodes[0]:
            node.backtrace_score = node.log_prob
            node.prev = None

        # Propagate best scores forward through the lattice
        for pos in range(sentence_len):
            for node in self.nodes[pos]:
                if node.backtrace_score == float('-inf'):
                    continue  # Unreachable node
                next_pos = node.pos + node.length
                # Update scores for nodes reachable from the current node
                for next_node in self.nodes[next_pos]:
                    new_score = node.backtrace_score + next_node.log_prob
                    if new_score > next_node.backtrace_score:
                        next_node.backtrace_score = new_score
                        next_node.prev = node # Set back-pointer for Viterbi path

        # Find the best final node (the one ending exactly at the sentence end)
        best_final_score = float('-inf')
        best_final_node: Optional[Node] = None
        for pos in range(sentence_len):
            for node in self.nodes[pos]:
                if node.pos + node.length == sentence_len:
                    if node.backtrace_score > best_final_score:
                        best_final_score = node.backtrace_score
                        best_final_node = node

        self._viterbi_score = best_final_score if best_final_node else float('-inf')
        
        # Backtrace to construct the Viterbi path
        self._viterbi_path = []
        current_node = best_final_node
        while current_node:
            self._viterbi_path.append(current_node)
            current_node = current_node.prev
        if self._viterbi_path:
            self._viterbi_path.reverse() # Reverse to get path from start to end

        # --- 2. Backward Pass for N-best (Max Product from Node to End) ---
        # Calculates the maximum score achievable from each node to the end of the sentence.
        # Also stores pointers to reconstruct the best suffix path for N-best.
        for pos_nodes in self.nodes:
            for node in pos_nodes:
                node.beta_score = float('-inf')
                node.suffix_next = None

        # Initialize beta_score for nodes that directly reach the sentence end.
        # Their score is simply their own log_prob, as they connect to the implicit end state.
        for pos in range(sentence_len):
            for node in self.nodes[pos]:
                if node.pos + node.length == sentence_len:
                    node.beta_score = node.log_prob
                    # node.suffix_next remains None as it leads directly to the end.

        # Backward iteration to compute beta_score and suffix_next for all other nodes.
        # Iterate from the end of the sentence towards the beginning.
        for pos in range(sentence_len - 1, -1, -1):
            for node in self.nodes[pos]:
                # Skip nodes that already reach the end, as their beta_score is final.
                if node.pos + node.length == sentence_len:
                    continue

                best_beta_score = float('-inf')
                best_next_node: Optional[Node] = None
                next_pos = node.pos + node.length # End position of this node's segment

                # Check all nodes that start where the current node ends.
                for next_node in self.nodes[next_pos]:
                    # Score of path: (current node) -> (next node) -> (best path from next node to end)
                    # This is the sum of the current edge score and the best score from the next node.
                    candidate_beta = node.log_prob + next_node.beta_score
                    
                    if candidate_beta > best_beta_score:
                        best_beta_score = candidate_beta
                        best_next_node = next_node
                
                # Update the node's beta_score and suffix_next pointer if a better path was found.
                if best_beta_score != float('-inf'):
                    node.beta_score = best_beta_score
                    node.suffix_next = best_next_node
                # If no valid path was found, node.beta_score remains -inf, correctly
                # indicating this node cannot contribute to a path to the end.

    def viterbi(self) -> Tuple[List[Node], float]:
        """
        Finds the single most likely segmentation (Viterbi path) for the sentence.
        Returns:
            A tuple containing the list of Nodes in the best path and its log score.
        """
        self._compute_viterbi_and_nbest_data()
        path = self._viterbi_path if self._viterbi_path is not None else []
        score = self._viterbi_score if self._viterbi_score is not None else float('-inf')
        return (path, score)

    def populate_marginal(self, freq: float, expected: List[float]) -> float:
        """
        Computes the marginal probability of each piece being used in the segmentation
        of this sentence, weighted by its frequency. Updates the 'expected' vector.
        This is the E-step calculation for a single sentence.
        Uses the Forward-Backward algorithm (sum in log space) for numerical stability.
        Args:
            freq: The frequency of this sentence in the training data.
            expected: A list where expected[i] will be incremented by
                      freq * P(piece_i is used in this sentence's segmentation).
        Returns:
            The log partition function Z (log total probability of the sentence).
        """
        sentence_len = len(self.chars)
        if not sentence_len:
            return 0.0

        # --- Forward Pass (Alpha) ---
        # alpha[pos] = log P(x_1, ..., x_pos, path ends at position pos)
        self.alpha = [float('-inf')] * (sentence_len + 1)
        self.alpha[0] = 0.0 # Start state probability is 1 (log(1)=0)
        for pos in range(sentence_len + 1):
            if self.alpha[pos] == float('-inf'):
                continue # Unreachable position
            for node in self.nodes[pos]:
                end_pos = pos + node.length
                new_log_prob = self.alpha[pos] + node.log_prob
                # LogSumExp: alpha[end_pos] = log(exp(alpha[end_pos]) + exp(new_log_prob))
                if new_log_prob > self.alpha[end_pos]:
                    self.alpha[end_pos] = new_log_prob
                elif self.alpha[end_pos] != float('-inf'):
                    diff = new_log_prob - self.alpha[end_pos]
                    # Numerically stable log(1 + exp(diff)) for diff < 0
                    self.alpha[end_pos] += math.log1p(math.exp(diff))

        Z = self.alpha[sentence_len] # Log partition function
        if Z == float('-inf'):
            return Z # No valid path found

        # --- Backward Pass (Beta) ---
        # beta[pos] = log P(x_{pos+1}, ..., x_N | path starts at position pos)
        self.beta = [float('-inf')] * (sentence_len + 1)
        self.beta[sentence_len] = 0.0 # End state probability is 1 (log(1)=0)
        # Iterate backwards
        for pos in range(sentence_len, -1, -1):
            for node in self.nodes[pos]:
                end_pos = pos + node.length
                new_log_prob = self.beta[end_pos] + node.log_prob
                # LogSumExp update for beta
                if new_log_prob > self.beta[pos]:
                    self.beta[pos] = new_log_prob
                elif self.beta[pos] != float('-inf'):
                    diff = new_log_prob - self.beta[pos]
                    self.beta[pos] += math.log1p(math.exp(diff))


        # --- Compute and Accumulate Marginals ---
        # P(node is used in segmentation) = 
        # exp(alpha[pos] + node.log_prob + beta[end_pos] - Z)
        for pos in range(sentence_len):
            for node in self.nodes[pos]:
                end_pos = pos + node.length
                log_prob = self.alpha[pos] + node.log_prob + self.beta[end_pos] - Z
                # Clamp log_prob to prevent underflow if it's extremely negative
                if log_prob < -100: 
                    prob = 0.0
                else:
                    prob = math.exp(log_prob)
                expected[node.piece_id] += freq * prob

        return Z


    def nbest(self, n: int) -> List[Tuple[List[Node], float]]:
        """
        Finds the N most likely segmentations (N-best paths) for the sentence.
        Uses the Viterbi path and deviations from it to efficiently find alternatives.
        Primarily used for N=1 (Viterbi) or N=2 (for pruning).
        Args:
            n: The number of best paths to find.
        Returns:
            A list of up to n tuples, each containing a path (list of Nodes) and its score.
        """
        if self._nbest_cache_n >= n:
             return self._nbest_cache_results[:n]

        self._compute_viterbi_and_nbest_data() # Ensure Viterbi and N-best data are ready

        if n <= 0:
            self._nbest_cache_results = []
            self._nbest_cache_n = n
            return self._nbest_cache_results

        results: List[Tuple[List[Node], float]] = []
        viterbi_path = self._viterbi_path if self._viterbi_path is not None else []
        
        if not viterbi_path: # No path found
            self._nbest_cache_results = results
            self._nbest_cache_n = len(results)
            return results[:n]

        # 1. Add the 1st best path (Viterbi)
        viterbi_score = self._viterbi_score if self._viterbi_score is not None else float('-inf')
        results.append((viterbi_path, viterbi_score))

        if n == 1:
            self._nbest_cache_results = results
            self._nbest_cache_n = len(results)
            return results

        # 2. Find 2nd best path by looking for deviations from the 1st best path
        if n >= 2:
            second_best_score = float('-inf')
            second_best_path: Optional[List[Node]] = None

            # Iterate through each node in the Viterbi path
            for k in range(len(viterbi_path)):
                node_k = viterbi_path[k] 
                # Score of the path up to (but not including) node_k
                prev_node_k_score = viterbi_path[k-1].backtrace_score if k > 0 else 0.0

                # Check all alternative nodes that start at the same position as node_k
                for alt_node in self.nodes[node_k.pos]:
                    if alt_node is node_k: # Skip the node used in the Viterbi path
                        continue

                    # Calculate the score of the path that deviates here:
                    # (Prefix up to node_k) + (Alternative node) + (Best suffix from alt_node)
                    # This is equivalent to: 
                    # (Score of best prefix to node_k's start) + (Score of best path from alt_node to end)
                    # Which is: prev_node_k_score + alt_node.beta_score
                    # The deviation_score (alt_node.log_prob) is included in alt_node.beta_score.
                    prefix_score = prev_node_k_score
                    # deviation_score = alt_node.log_prob # Included in beta_score
                    suffix_score = alt_node.beta_score # Precomputed best score from alt_node to end

                    # Check if the suffix from alt_node is valid
                    if suffix_score != float('-inf'):
                        # The nbest candidate score is the sum of the prefix score and the best 
                        # score achievable from the alternative node onwards.
                        candidate_score = prefix_score + suffix_score

                        # Use >= to ensure we consider later candidates in case of ties,
                        # which might be necessary for correct tie-breaking or finding the
                        # expected path in tests.
                        if candidate_score >= second_best_score: 
                            second_best_score = candidate_score
                            # --- Correctly Reconstruct the 2nd best path ---
                            # 1. Take the prefix path up to the deviation point (before node_k)
                            prefix_path = viterbi_path[:k]
                            # 2. Reconstruct the best suffix path starting from the alternative node.
                            #    This path, generated using suffix_next, inherently includes alt_node
                            #    as its first element.
                            suffix_path = []
                            current_suffix_node = alt_node
                            while current_suffix_node: 
                                suffix_path.append(current_suffix_node)
                                current_suffix_node = current_suffix_node.suffix_next

                            # The final 2nd best path is the prefix followed by the suffix path.
                            # The suffix path already starts with the alt_node, so we don't add it separately.
                            second_best_path = prefix_path + suffix_path
                            # --- End Correct Reconstruction ---

            # If a 2nd best path was found, add it to the results
            if second_best_path is not None:
                # Recalculate the *actual* log probability score of the reconstructed path
                # to ensure correctness, as the nbest_score used for ranking might have 
                # been derived differently (though in this implementation it should be the same).
                actual_second_best_score = sum(node.log_prob for node in second_best_path)
                results.append((second_best_path, actual_second_best_score))

        self._nbest_cache_results = results[:n]
        self._nbest_cache_n = len(self._nbest_cache_results)
        return self._nbest_cache_results


class TrainerModel:
    """
    Manages a collection of subword pieces and their scores.
    Populates Lattices with potential segmentations based on these pieces.
    """

    def __init__(self, pretokens: Dict[str, int], max_piece_length: int = 16):
        """
        Initializes the TrainerModel with an initial set of pieces (pretokens).
        Calculates initial log probabilities based on their frequencies.
        Builds an internal Trie for fast prefix lookups when populating lattices.
        Args:
            pretokens: A dictionary mapping piece strings to their frequencies.
            max_piece_length: A hint for the maximum piece length considered during training.
        """
        if not pretokens:
            raise ValueError("Pretokens dictionary cannot be empty.")
            
        # Convert input frequencies to initial log probabilities
        total_freq = sum(pretokens.values())
        if total_freq <= 0:
             # Assign uniform log probability if total frequency is zero or invalid
             log_prob = math.log(1.0 / len(pretokens)) if pretokens else 0.0
             self._pieces: List[Tuple[str, float]] = [(piece, log_prob) for piece in pretokens]
        else:
            log_total_freq = math.log(total_freq)
            # Create list of (piece, log_probability) tuples
            self._pieces: List[Tuple[str, float]] = [
                (piece, math.log(freq) - log_total_freq) 
                for piece, freq in pretokens.items() 
                if freq > 0 # Only include pieces with positive frequency
            ]
        
        if not self._pieces:
             raise ValueError("No valid pieces found after processing pretokens.")
             
        # The minimum score among all pieces, useful for finalization
        self.min_score = min((score for _, score in self._pieces), default=float('inf'))
        
        # Build a Trie data structure for efficient prefix matching
        self._build_trie()

    @property
    def pieces(self) -> List[Tuple[str, float]]:
        """Gets the current list of pieces and their scores."""
        return self._pieces

    def set_pieces(self, pieces: List[Tuple[str, float]]) -> None:
        self._pieces = pieces
        self.min_score = min((score for _, score in self._pieces), default=float('inf'))
        self._build_trie()

    def _build_trie(self) -> None:
        """Constructs a Trie from the current pieces for fast lookups."""
        self.trie: Dict[str, Any] = {}
        for i, (piece, _) in enumerate(self._pieces):
            node = self.trie
            # Traverse the Trie character by character for the piece
            for char in piece:
                if char not in node:
                    node[char] = {} # Create a new node if the character path doesn't exist
                node = node[char]
            # Mark the end of the piece with its index in self._pieces
            node['piece_id'] = i 

    def populate(self, lattice: Lattice) -> None:
        """
        Fills a Lattice with potential Nodes based on the current pieces.
        Uses the internal Trie for efficient prefix matching.
        Args:
            lattice: The Lattice object to populate with Nodes.
        """
        if not lattice.chars or not self._pieces:
            return

        sentence_str = lattice.sentence
        # For each starting position in the sentence
        for pos in range(len(sentence_str)):
            trie_node = self.trie
            current_pos = pos
            # Traverse the Trie while matching characters in the sentence
            while (current_pos < len(sentence_str) and 
                   sentence_str[current_pos] in trie_node):
                trie_node = trie_node[sentence_str[current_pos]]
                current_pos += 1
                # If we reached the end of a piece in the Trie
                if 'piece_id' in trie_node:
                    piece_id = trie_node['piece_id']
                    log_prob = self._pieces[piece_id][1] # Get the piece's score
                    length = current_pos - pos # Calculate the piece's length
                    # Add this potential segmentation as a Node in the Lattice
                    lattice.insert_node(pos, length, piece_id, log_prob)

    def __len__(self) -> int:
        """Returns the number of pieces in the model."""
        return len(self._pieces)

    def __getitem__(self, index: int) -> Tuple[str, float]:
        """Allows accessing a piece by its index."""
        return self._pieces[index]
