from minunigram.lattice import Lattice

def test_lattice_empty_and_unicode():
    # Empty string
    lattice = Lattice("")
    assert lattice.size == 0
    assert lattice.sentence == ""
    # Unicode string
    s = "テストab"
    lattice = Lattice(s)
    assert lattice.size == len(s)
    assert lattice.sentence == s
    # Surfaces
    assert lattice.sentence[0:] == s
    assert lattice.sentence[1:] == s[1:]
    assert lattice.sentence[2:] == s[2:]
    assert lattice.sentence[3:] == s[3:]
    assert lattice.sentence[4:] == s[4:]

def test_lattice_insert_and_viterbi():
    lattice = Lattice("ABあい")
    lattice.insert(0, 1, 3, 0.0)  # A
    lattice.insert(1, 1, 4, 0.0)  # B
    lattice.insert(2, 1, 5, 0.0)  # あ
    lattice.insert(3, 1, 6, 0.0)  # い
    lattice.insert(0, 2, 7, 0.0)  # AB
    lattice.insert(1, 3, 8, 0.0)  # Bあい
    lattice.insert(2, 2, 9, 0.0)  # あい
    # Check pieces
    assert lattice.nodes[2]['piece'] == "A"
    assert lattice.nodes[3]['piece'] == "B"
    assert lattice.nodes[4]['piece'] == "あ"
    assert lattice.nodes[5]['piece'] == "い"
    assert lattice.nodes[6]['piece'] == "AB"
    assert lattice.nodes[7]['piece'] == "Bあい"
    assert lattice.nodes[8]['piece'] == "あい"
    # Viterbi path
    lattice.insert(0, 4, 10, 1.0)  # ABあい
    path, score = lattice.viterbi()
    assert [node['piece'] for node in path] == ["ABあい"]

def test_lattice_nbest():
    lattice = Lattice("ABC")
    lattice.insert(0, 1, 3, 0.0)
    lattice.insert(1, 1, 4, 0.0)
    lattice.insert(2, 1, 5, 0.0)
    lattice.insert(0, 2, 6, 2.0)
    lattice.insert(1, 2, 7, 5.0)
    lattice.insert(0, 3, 8, 10.0)
    # nbest tokens
    tokens = lattice.tokens()
    assert tokens == ["ABC"]
    # Remove highest score, check other paths
    lattice.nodes[-1]['score'] = -float('inf')
    tokens = lattice.tokens()
    assert tokens in (["A", "BC"], ["AB", "C"])
