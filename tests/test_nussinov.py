from rna3d_folding.nussinov import nussinov


def test_nussinov_returns_dot_bracket_same_length():
    sequence = "AUGC"
    dot_bracket, pairs = nussinov(sequence)
    assert len(dot_bracket) == len(sequence)
    assert isinstance(pairs, set)


def test_nussinov_empty_sequence():
    dot_bracket, pairs = nussinov("")
    assert dot_bracket == ""
    assert pairs == set()
