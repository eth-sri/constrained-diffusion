from rustformlang.fa.bytes_dfa import regex_to_dfa
from rustformlang.cfg import CFG


def test_import_and_use_regex():
    regex = r"(ab)*\d\d"
    dfa = regex_to_dfa(regex)
    assert dfa.accepts_string("abab12")
    assert not dfa.accepts_string("abab")


def test_cfg_from_text():
    cfg = CFG.from_text(
        """
    S -> a S b | a b
    """,
        "S",
    )
    assert cfg.accepts_string("ab")
    assert cfg.accepts_string("aaabbb")
    assert not cfg.accepts_string("aabbb")


def test_cfg_intersection():
    cfg1 = CFG.from_text(
        """
    S -> a S b | a b
    """,
        "S",
    )
    dfa = regex_to_dfa(r"(ab)*")
    intersection = cfg1.intersection(dfa.to_epsilon_automaton().to_deterministic())
    assert not intersection.accepts_string("aaabbb")
    assert not intersection.accepts_string("abab")
    assert intersection.accepts_string("ab")
