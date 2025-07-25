import math

import pytest

from py_unigram.qwen.model import Lattice, Node, TrainerModel

# --- Fixtures ---

@pytest.fixture
def simple_pretokens():
    """Provides a simple pretoken dictionary for testing."""
    return {"a": 5, "b": 3, "ab": 7, "c": 2}

@pytest.fixture
def simple_model(simple_pretokens):
    """Provides a TrainerModel instance built from simple_pretokens."""
    return TrainerModel(simple_pretokens)

@pytest.fixture
def simple_lattice():
    """Provides a Lattice instance for the string 'ab'."""
    return Lattice("ab")


class TestLattice:
    def test_initialization(self):
        """Test Lattice initialization with a sentence."""
        sentence = "hello"
        lattice = Lattice(sentence)
        assert lattice.sentence == sentence
        assert lattice.chars == ['h', 'e', 'l', 'l', 'o']
        assert len(lattice.nodes) == len(sentence) + 1 # +1 for end state
        # Check that nodes list is initialized with empty lists
        assert all(isinstance(node_list, list) and len(node_list) == 0 for node_list in lattice.nodes)
        # Check cache attributes are initialized correctly
        assert lattice._viterbi_path is None
        assert lattice._viterbi_score is None
        assert lattice._nbest_cache_n == 0
        assert lattice._nbest_cache_results == []
        assert lattice.alpha == []
        assert lattice.beta == []

    def test_insert_node_valid(self):
        """Test inserting a valid node into the lattice."""
        lattice = Lattice("ab")
        # Initially, no nodes
        assert len(lattice.nodes[0]) == 0
        assert len(lattice.nodes[1]) == 0
        assert len(lattice.nodes[2]) == 0 # 'ab' -> len=2, indices 0,1. nodes[2] for end.

        # Insert node 'a' at pos 0, length 1
        lattice.insert_node(0, 1, 10, -0.5)
        assert len(lattice.nodes[0]) == 1
        node_a = lattice.nodes[0][0]
        assert isinstance(node_a, Node)
        assert node_a.pos == 0
        assert node_a.length == 1
        assert node_a.piece_id == 10
        assert node_a.log_prob == pytest.approx(-0.5)

        # Insert node 'ab' at pos 0, length 2
        lattice.insert_node(0, 2, 11, -0.8)
        assert len(lattice.nodes[0]) == 2
        node_ab = lattice.nodes[0][1] # Assuming append order
        assert node_ab.pos == 0
        assert node_ab.length == 2
        assert node_ab.piece_id == 11
        assert node_ab.log_prob == pytest.approx(-0.8)

        # Insert node 'b' at pos 1, length 1
        lattice.insert_node(1, 1, 12, -0.6)
        assert len(lattice.nodes[1]) == 1
        node_b = lattice.nodes[1][0]
        assert node_b.pos == 1
        assert node_b.length == 1
        assert node_b.piece_id == 12
        assert node_b.log_prob == pytest.approx(-0.6)

    def test_insert_node_invalid_positions(self):
        """Test inserting nodes with invalid positions or lengths."""
        lattice = Lattice("test") # len = 4, valid indices for nodes are 0-4
        num_nodes_slots = len(lattice.nodes) # Should be 5
        assert num_nodes_slots == 5

        initial_count_0 = len(lattice.nodes[0])
        initial_count_2 = len(lattice.nodes[2])
        initial_count_4 = len(lattice.nodes[4])
        # Do not access lattice.nodes[5] as it's out of bounds

        # Test cases that should NOT add nodes
        # pos < 0
        lattice.insert_node(-1, 1, 1, 0.0)
        assert len(lattice.nodes[0]) == initial_count_0

        # pos >= len(nodes) (e.g., pos = 5 for 5 slots (0-4))
        lattice.insert_node(5, 1, 1, 0.0)
        # No change to existing slots
        assert len(lattice.nodes[4]) == initial_count_4

        # length <= 0
        lattice.insert_node(0, 0, 1, 0.0)
        lattice.insert_node(2, -1, 1, 0.0)
        assert len(lattice.nodes[0]) == initial_count_0
        assert len(lattice.nodes[2]) == initial_count_2

        # pos + length > len(sentence) (spans beyond sentence)
        # Sentence len is 4, so valid end positions are 0,1,2,3,4
        # Trying to add a node starting at pos 3 with length 2 ends at pos 5 (invalid)
        lattice.insert_node(3, 2, 1, 0.0)
        assert len(lattice.nodes[3]) == 0 # Should not be added

    def test_viterbi_no_sentence(self):
        """Test Viterbi when the lattice sentence is empty."""
        lattice = Lattice("") # Initialize with empty string
        path, score = lattice.viterbi()
        assert path == []
        assert score == 0.0 # Score for empty lattice/string

    def test_viterbi_no_nodes(self):
        """Test Viterbi when no nodes are present for a non-empty sentence."""
        lattice = Lattice("any") # Non-empty sentence
        # Ensure no nodes are added
        # The lattice is initialized empty of nodes.
        path, score = lattice.viterbi()
        assert path == [] # No path found
        assert score == float('-inf') # Score indicating no valid path

    def test_viterbi_single_path(self):
        """Test Viterbi with a setup that has only one possible path."""
        lattice = Lattice("ab")
        # Add only one way to segment: 'a' (id=0) then 'b' (id=1)
        # log_prob for 'a' = -0.5, for 'b' = -0.6
        lattice.insert_node(0, 1, 0, -0.5) # 'a'
        lattice.insert_node(1, 1, 1, -0.6) # 'b'
        # No direct 'ab' node

        path, score = lattice.viterbi()
        assert len(path) == 2
        assert path[0].piece_id == 0
        assert path[0].pos == 0
        assert path[0].length == 1
        assert path[1].piece_id == 1
        assert path[1].pos == 1
        assert path[1].length == 1
        # Viterbi score is sum of log probs
        assert score == pytest.approx(-0.5 + (-0.6))

    def test_viterbi_best_path(self):
        """Test Viterbi finds the highest scoring path."""
        lattice = Lattice("ab")
        # Path 1: 'a' (-0.5) + 'b' (-0.6) = -1.1
        lattice.insert_node(0, 1, 0, -0.5)
        lattice.insert_node(1, 1, 1, -0.6)
        # Path 2: 'ab' (-0.8) = -0.8 (better)
        lattice.insert_node(0, 2, 2, -0.8)

        path, score = lattice.viterbi()
        assert len(path) == 1
        assert path[0].piece_id == 2
        assert path[0].pos == 0
        assert path[0].length == 2
        assert score == pytest.approx(-0.8)

    def test_populate_marginal_basic(self):
        """Test basic functionality of populate_marginal."""
        lattice = Lattice("a")
        # Simple case: one piece 'a'
        lattice.insert_node(0, 1, 0, -0.5)
        freq = 2.0
        expected = [0.0] # One piece in model
        Z = lattice.populate_marginal(freq, expected)

        # For one node, alpha[0]=0, alpha[1] = 0 + (-0.5) = -0.5
        # beta[1]=0, beta[0] = beta[1] + node.log_prob = 0 + (-0.5) = -0.5
        # Z = alpha[1] = -0.5
        # Marginal prob P(node|sentence) = exp(alpha[0] + log_prob + beta[1] - Z)
        # = exp(0 + (-0.5) + 0 - (-0.5)) = exp(0) = 1.0
        # Expected count += freq * P = 2.0 * 1.0 = 2.0
        assert Z == pytest.approx(-0.5)
        assert expected[0] == pytest.approx(2.0)

    def test_nbest_n_0(self):
        """Test nbest with n=0."""
        lattice = Lattice("test")
        lattice.insert_node(0, 4, 0, -1.0)
        results = lattice.nbest(0)
        assert results == []

    def test_nbest_n_1(self):
        """Test nbest with n=1 (should be same as Viterbi)."""
        lattice = Lattice("ab")
        lattice.insert_node(0, 1, 0, -0.5)
        lattice.insert_node(1, 1, 1, -0.6)
        lattice.insert_node(0, 2, 2, -0.8)

        nbest_results = lattice.nbest(1)
        viterbi_path, viterbi_score = lattice.viterbi() # Should trigger computation

        assert len(nbest_results) == 1
        # Need to compare node attributes as they are different objects
        assert len(nbest_results[0][0]) == len(viterbi_path)
        for nbest_node, viterbi_node in zip(nbest_results[0][0], viterbi_path, strict=False):
            assert nbest_node.pos == viterbi_node.pos
            assert nbest_node.length == viterbi_node.length
            assert nbest_node.piece_id == viterbi_node.piece_id
            assert nbest_node.log_prob == viterbi_node.log_prob
        assert nbest_results[0][1] == pytest.approx(viterbi_score) # Score comparison

    def test_nbest_n_2_disjoint_corrected(self):
        """Test nbest with n=2 where 2nd best is distinct and correctly identified."""
        # String "abc"
        # Let's define scores such that the ranking for nbest is clear.
        # Viterbi (Best): "a"(0,-0.1) + "b"(1,-0.2) + "c"(2,-0.3) = -0.6 (Score A)
        # Candidate 1: "ab"(3,-0.8) + "c"(2,-0.3) = -1.1 (Score B)
        # Candidate 2: "a"(0,-0.1) + "bc"(4,-0.9) = -1.0 (Score C)
        # Candidate 3: "abc"(5,-1.5) = -1.5 (Score D)
        # Order: A (-0.6) > C (-1.0) > B (-1.1) > D (-1.5)
        # So, 1st best = "a"+"b"+"c", 2nd best = "a"+"bc"
        lattice = Lattice("abc")
        lattice.insert_node(0, 1, 0, -0.1) # a (part of Viterbi)
        lattice.insert_node(1, 1, 1, -0.2) # b (part of Viterbi)
        lattice.insert_node(2, 1, 2, -0.3) # c (part of Viterbi and "ab"+c)
        lattice.insert_node(0, 2, 3, -0.8) # ab (part of candidate)
        lattice.insert_node(1, 2, 4, -0.9) # bc (part of 2nd best)
        lattice.insert_node(0, 3, 5, -1.5) # abc (worse single piece)

        nbest_results = lattice.nbest(2)
        assert len(nbest_results) == 2
        # 1st best: [a, b, c]
        path1, score1 = nbest_results[0]
        assert score1 == pytest.approx(-0.6)
        assert len(path1) == 3
        assert path1[0].piece_id == 0
        assert path1[1].piece_id == 1
        assert path1[2].piece_id == 2

        # 2nd best: [a, bc]
        path2, score2 = nbest_results[1]
        assert score2 == pytest.approx(-1.0)
        assert len(path2) == 2
        assert path2[0].piece_id == 0
        assert path2[1].piece_id == 4

    # Add more tests for Lattice as needed (e.g., edge cases, longer strings)


# --- Tests for TrainerModel ---

class TestTrainerModel:
    def test_initialization(self, simple_pretokens):
        """Test TrainerModel initialization."""
        model = TrainerModel(simple_pretokens)
        assert len(model) == len(simple_pretokens)
        assert len(model.pieces) == len(simple_pretokens)
        # Check if pieces are correctly initialized with log probs
        total_freq = sum(simple_pretokens.values())
        log_total = math.log(total_freq)
        for piece, freq in simple_pretokens.items():
            expected_log_prob = math.log(freq) - log_total
            found = False
            for p, score in model.pieces:
                if p == piece:
                    assert score == pytest.approx(expected_log_prob)
                    found = True
                    break
            assert found, f"Piece '{piece}' not found in model.pieces"

    def test_initialization_empty_pretokens(self):
        """Test TrainerModel initialization with empty pretokens."""
        with pytest.raises(ValueError):
            TrainerModel({})

    def test_len(self, simple_model, simple_pretokens):
        """Test __len__ method."""
        assert len(simple_model) == len(simple_pretokens)

    def test_getitem(self, simple_model, simple_pretokens):
        """Test __getitem__ method."""
        # Get the first piece
        piece, score = simple_model[0]
        assert isinstance(piece, str)
        assert isinstance(score, float)
        # Check if it's one of the original pretokens
        assert piece in simple_pretokens

    def test_populate_empty_lattice(self, simple_model):
        """Test populate with an empty lattice (non-empty sentence, no matching pieces)."""
        lattice = Lattice("xyz") # String not in model pieces
        # Before populate
        assert all(len(nodes) == 0 for nodes in lattice.nodes)
        simple_model.populate(lattice)
        # After populate, no nodes should be added as "xyz" cannot be segmented
        assert all(len(nodes) == 0 for nodes in lattice.nodes)

    def test_populate_simple_lattice(self, simple_model):
        """Test populate adds correct nodes."""
        lattice = Lattice("ab")
        simple_model.populate(lattice)

        # Find piece IDs (more robust way)
        piece_to_id = {p: i for i, (p, _) in enumerate(simple_model.pieces)}

        id_a = piece_to_id.get("a")
        id_b = piece_to_id.get("b")
        id_ab = piece_to_id.get("ab")

        assert id_a is not None
        assert id_b is not None
        assert id_ab is not None

        # Check nodes at pos 0
        nodes_at_0 = lattice.nodes[0]
        # Convert to a set of (id, length) for easier comparison
        node_pieces_0 = {(n.piece_id, n.length) for n in nodes_at_0}
        expected_pieces_0 = {(id_a, 1), (id_ab, 2)}
        assert node_pieces_0 == expected_pieces_0

        # Check nodes at pos 1
        nodes_at_1 = lattice.nodes[1]
        assert len(nodes_at_1) == 1 # 'b'
        assert nodes_at_1[0].piece_id == id_b
        assert nodes_at_1[0].length == 1

        # Check no nodes beyond the string
        assert len(lattice.nodes[2]) == 0


# --- Tests for Integration (Lattice + TrainerModel) ---

class TestLatticeTrainerModelIntegration:
    def test_viterbi_with_model(self, simple_model):
        """Test running Viterbi on a lattice populated by a model."""
        lattice = Lattice("ab")
        simple_model.populate(lattice)
        # Based on frequencies {'a': 5, 'b': 3, 'ab': 7}, 'ab' should be the best single piece.
        path, score = lattice.viterbi()
        assert isinstance(path, list)
        assert isinstance(score, float)
        # Assert based on expected best path
        assert len(path) == 1 # Expecting the single 'ab' piece to be best
        assert path[0].piece_id == [i for i, (p, _) in enumerate(simple_model.pieces) if p == "ab"][0]


