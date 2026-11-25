"""Lexer regexes must only accept ASCII digits.

`\\d` in `regex_to_dfa` (like in Python `re`) matches any Unicode decimal digit,
e.g. the Arabic-Indic digit three "٣" (U+0663). JSON, C++ and SMILES only allow 0-9.
"""

import pytest
from rustformlang.fa.bytes_dfa import regex_to_dfa

from constrained_diffusion.cfgs import cpp
from constrained_diffusion.cfgs.jsonschema import JSON_LEX_MAP, STRING_LEXING
from constrained_diffusion.cfgs.smiles import SMILES_LEX_MAP
from constrained_diffusion.constrain_utils import LEX_MAP

ARABIC_THREE = "٣"


def _dfa(regex_or_dfa):
    return regex_to_dfa(regex_or_dfa) if isinstance(regex_or_dfa, str) else regex_or_dfa


# (name, regex or DFA, accepted ASCII word, word with one digit replaced by "٣")
CASES = [
    ("json numberToken", JSON_LEX_MAP["numberToken"], "13", "1" + ARABIC_THREE),
    ("json numberToken frac", JSON_LEX_MAP["numberToken"], "1.3", "1." + ARABIC_THREE),
    ("json numberToken exp", JSON_LEX_MAP["numberToken"], "1e3", "1e" + ARABIC_THREE),
    (
        "json date",
        STRING_LEXING["date"],
        '"2023-01-02"',
        f'"202{ARABIC_THREE}-01-02"',
    ),
    (
        "json date-time",
        STRING_LEXING["date-time"],
        '"2023-01-02T03:04:05Z"',
        f'"202{ARABIC_THREE}-01-02T03:04:05Z"',
    ),
    (
        "json date-time frac",
        STRING_LEXING["date-time"],
        '"2023-01-02T03:04:05.3Z"',
        f'"2023-01-02T03:04:05.{ARABIC_THREE}Z"',
    ),
    ("lexNumber", LEX_MAP["lexNumber"], "13", "1" + ARABIC_THREE),
    ("smiles digit", SMILES_LEX_MAP["digit"], "3", ARABIC_THREE),
    (
        "cpp dec",
        cpp.CPP_float_number,
        "\x0213\x03",
        f"\x021{ARABIC_THREE}\x03",
    ),
    (
        "cpp decimal part",
        cpp.CPP_float_number,
        "\x021\x03.\x023\x03",
        f"\x021\x03.\x02{ARABIC_THREE}\x03",
    ),
    (
        "cpp exp",
        cpp.CPP_float_number,
        "\x021e3\x03",
        f"\x021e{ARABIC_THREE}\x03",
    ),
    ("cpp hex", cpp.CPP_hex_number, "0x3f", f"0x{ARABIC_THREE}f"),
]


@pytest.mark.parametrize(
    "regex, ascii_word, unicode_word",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_lexer_regex_only_accepts_ascii_digits(regex, ascii_word, unicode_word):
    dfa = _dfa(regex)
    assert dfa.accepts_string(ascii_word)
    assert not dfa.accepts_string(unicode_word)
