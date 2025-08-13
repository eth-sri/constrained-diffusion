import pytest

from constrained_diffusion.cfgs.cpp import cpp_grammar
from constrained_diffusion.cfgs.smiles import smiles_schema
from constrained_diffusion.constrain_utils import (
    compile_lex_map,
    LEX_MAP,
    lex,
    generated_language,
    EOS,
    derive_supertokens,
    interleave_with_value,
)
from rustformlang.cfg import CFG
from rustformlang.constraining import reset_lex_cache
from rustformlang.fa.bytes_dfa import regex_to_dfa
import constrained_diffusion.cfgs.jsonschema as schema_to_cfg


def test_lex():
    # compile the lex_map
    lex_map = compile_lex_map(LEX_MAP)
    assert lex('null { } 123 "hi!" ', lex_map) == {
        (("lexNull", "{", "}", "lexNumber", "lexString"), None, None),
    }
    assert lex('null" ', lex_map) == {
        (("lexString",), 'null"', None),
        (("lexNull", "lexString"), None, '" '),
    }
    assert lex('lhio! " : 123, "', lex_map) == {
        (("lexString", ":", "lexNumber", ",", "lexString"), 'lhio! "', '"')
    }
    assert lex("test", lex_map) == {
        (("lexString",), "test", "test"),
    }
    assert lex('"test"', lex_map) == {
        (("lexString",), None, None),
    }
    assert lex('ing":', lex_map, is_first=False) == {(("lexString", ":"), 'ing"', None)}
    assert lex('":', lex_map, is_first=False) == {
        (("lexString", ":"), '"', None),
        (("lexString",), None, '":'),
    }
    assert lex(" ", lex_map, is_first=False) == {
        ((), None, None),
        (("lexString",), " ", " "),
    }
    assert lex("123", lex_map, is_first=True) == {
        (("lexNumber",), None, "123"),
        (("lexNumber",), None, None),
    }
    assert lex("123", lex_map, is_first=False) == {
        (("lexString",), "123", "123"),
        (("lexNumber",), "123", "123"),
        (("lexNumber",), None, "123"),
        (("lexNumber",), "123", None),
        (("lexNumber",), None, None),
    }


def test_generated_language():
    lex_map = compile_lex_map(LEX_MAP)
    terminals = list(lex_map.keys())
    generated_fsa = generated_language(
        ['"hiii', None, 'ello"'],
        lex_map,
        terminals,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(["lexString"])

    generated_fsa = generated_language(
        ['"hiii', "you", 'ello"'],
        lex_map,
        terminals,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(["lexString"])

    generated_fsa = generated_language(
        ['"hiii', None, "you", None, 'ello"'],
        lex_map,
        terminals,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(["lexString"])

    generated_fsa = generated_language(
        ["123", None, "456", None, "789", None],
        lex_map,
        terminals,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(["lexNumber", "lexNumber", "lexNumber"])
    assert generated_fsa.accepts(["lexNumber", "lexNumber"])
    assert generated_fsa.accepts(["lexNumber"])

    generated_fsa = generated_language(
        [" "],
        lex_map,
        terminals,
    )
    assert not generated_fsa.is_empty()

    generated_fsa = generated_language(
        ['" ', None, " "],
        lex_map,
        terminals,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.accepts(["lexString"])


def test_generated_language_precise():
    lex_map = compile_lex_map(LEX_MAP)
    terminals = list(lex_map.keys())
    generated_fsa = generated_language(
        ["nu", None, "ll"],
        lex_map,
        terminals,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(["lexNull"])

    generated_fsa = generated_language(
        ["nul", None, "ull"],
        lex_map,
        terminals,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert not generated_fsa.accepts(["lexNull"])
    assert generated_fsa.accepts(["lexNull", "lexNull"])

    generated_fsa = generated_language(
        ["n", None, "u", None, "l"],
        lex_map,
        terminals,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(["lexNull"])

    generated_fsa = generated_language(
        ["n", None, "ll", None, "l"],
        lex_map,
        terminals,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert not generated_fsa.accepts(["lexNull"])

    generated_fsa = generated_language(
        ["nul", None, "l", None, "l"],
        lex_map,
        terminals,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert not generated_fsa.accepts(["lexNull"])

    generated_fsa = generated_language(
        ["n", None, "l", None, "ll"],
        lex_map,
        terminals,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert not generated_fsa.accepts(["lexNull"])


@pytest.mark.skip(reason="not implemented")
def test_generated_language_inject_gap():
    lex_map = compile_lex_map(LEX_MAP)
    terminals = list(lex_map.keys())

    single_token_lexing = [(["lexString"], True, True), (["lexNumber"], True, True)]
    generated_fsa = generated_language(
        ['"hiii', None, 'ello"'],
        lex_map,
        terminals,
        inject_gap_size=1,
        max_total_injections=1,
        single_token_lexing=single_token_lexing,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(["lexString"])

    single_token_lexing = [(["lexNumber"], False, False)]
    generated_fsa = generated_language(
        ['"hiii', None, 'ello"'],
        lex_map,
        terminals,
        inject_gap_size=1,
        max_total_injections=1,
        single_token_lexing=single_token_lexing,
    )
    assert generated_fsa.is_empty()

    single_token_lexing = [(["lexNumber"], True, False)]
    generated_fsa = generated_language(
        ['"hiii', None, 'ello"'],
        lex_map,
        terminals,
        inject_gap_size=1,
        max_total_injections=1,
        single_token_lexing=single_token_lexing,
    )
    assert generated_fsa.is_empty()

    single_token_lexing = [(["lexNumber"], False, True)]
    generated_fsa = generated_language(
        ['"hiii', None, 'ello"'],
        lex_map,
        terminals,
        inject_gap_size=1,
        max_total_injections=1,
        single_token_lexing=single_token_lexing,
    )
    assert generated_fsa.is_empty()

    single_token_lexing = [(["lexNumber"], False, True)]
    generated_fsa = generated_language(
        ['"hiii"', None, "123"],
        lex_map,
        terminals,
        inject_gap_size=1,
        max_total_injections=1,
        single_token_lexing=single_token_lexing,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(["lexString", "lexNumber"])

    single_token_lexing = [(["lexNumber"], True, False)]
    generated_fsa = generated_language(
        ["12", None, '"ello"'],
        lex_map,
        terminals,
        inject_gap_size=1,
        max_total_injections=1,
        single_token_lexing=single_token_lexing,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(["lexNumber", "lexString"])

    single_token_lexing = [(["lexNumber"], True, False)]
    generated_fsa = generated_language(
        ["12", None, 'ello"'],
        lex_map,
        terminals,
        inject_gap_size=1,
        max_total_injections=1,
        single_token_lexing=single_token_lexing,
    )
    assert generated_fsa.is_empty()

    single_token_lexing = [(["lexNumber"], False, True)]
    generated_fsa = generated_language(
        ['"hell', None, "11"],
        lex_map,
        terminals,
        inject_gap_size=1,
        max_total_injections=1,
        single_token_lexing=single_token_lexing,
    )
    assert generated_fsa.is_empty()

    generated_fsa = generated_language(
        ['"hiii', "you", 'ello"'],
        lex_map,
        terminals,
        inject_gap_size=1,
        max_total_injections=1,
        single_token_lexing=[],
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(["lexString"])

    single_token_lexing = [(["lexString"], True, True), (["lexNumber"], True, True)]
    generated_fsa = generated_language(
        ['"hiii', None, "you", None, 'ello"'],
        lex_map,
        terminals,
        inject_gap_size=1,
        max_total_injections=2,
        single_token_lexing=single_token_lexing,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(["lexString"])

    single_token_lexing = [(["lexString"], True, True)]
    generated_fsa = generated_language(
        ['"hiii', None, "you", None, 'ello"', EOS],
        lex_map,
        terminals,
        inject_gap_size=1,
        max_total_injections=2,
        single_token_lexing=single_token_lexing,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(["lexString"])
    assert not generated_fsa.accepts(["lexString", "lexString"])

    single_token_lexing = [(["lexNumber"], True, True)]
    generated_fsa = generated_language(
        ['"hiii', None, "you", None, 'ello"'],
        lex_map,
        terminals,
        inject_gap_size=1,
        max_total_injections=2,
        single_token_lexing=single_token_lexing,
    )
    assert generated_fsa.is_empty()

    single_token_lexing = [(["lexString"], True, True)]
    generated_fsa = generated_language(
        ["123", None, "456", None, "789", None],
        lex_map,
        terminals,
        inject_gap_size=1,
        max_total_injections=3,
        single_token_lexing=single_token_lexing,
    )
    assert generated_fsa.is_empty()

    single_token_lexing = [(["lexString"], False, False), (["lexNumber"], True, True)]
    generated_fsa = generated_language(
        ["123", None, "456", None, "789"],
        lex_map,
        terminals,
        inject_gap_size=1,
        max_total_injections=3,
        single_token_lexing=single_token_lexing,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(
        ["lexNumber", "lexString", "lexNumber", "lexString", "lexNumber"]
    )
    assert generated_fsa.accepts(["lexNumber", "lexString", "lexNumber"])
    assert generated_fsa.accepts(["lexNumber"])


def test_generated_language_with_overlapping_tokens():
    subtokens = {
        "lexString": ["lexSpecificString"],
    }
    lex_map = compile_lex_map(
        {
            "lexString": r'"[^\r\n"]*"',  # Matches a string
            "lexSpecificString": r'"specific"',  # Matches a specific string
        },
        subtokens=subtokens,
    )
    terminals = list(lex_map.keys())
    supertokens = derive_supertokens(subtokens)
    generated_fsa = generated_language(
        ['"spec', None, 'ello"'],
        lex_map,
        terminals,
        subtokens=subtokens,
        supertokens=supertokens,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert not generated_fsa.accepts(["lexSpecificString"])
    assert generated_fsa.accepts(["lexString"])


def test_get_true_prefix_lang():
    automaton1 = regex_to_dfa("(ab)*c")
    true_prefix_automaton1 = automaton1.true_prefix_language()
    assert true_prefix_automaton1.accepts_string("a")
    assert true_prefix_automaton1.accepts_string("ab")
    assert not true_prefix_automaton1.accepts_string("abc")
    assert not true_prefix_automaton1.accepts_string("b")

    automaton1 = regex_to_dfa("(ab)*")
    true_prefix_automaton1 = automaton1.true_prefix_language()
    assert true_prefix_automaton1.accepts_string("a")
    assert true_prefix_automaton1.accepts_string("ab")
    assert true_prefix_automaton1.accepts_string("aba")
    assert true_prefix_automaton1.accepts_string("abab")
    assert not true_prefix_automaton1.accepts_string("bb")
    assert not true_prefix_automaton1.accepts_string("aa")
    assert not true_prefix_automaton1.accepts_string("b")


def test_get_true_suffix_lang():
    automaton1 = regex_to_dfa("c(ab)*")
    true_suffix_automaton1 = (
        automaton1.to_epsilon_automaton().true_suffix_language().minimize()
    )
    assert true_suffix_automaton1.accepts_string("b")
    assert true_suffix_automaton1.accepts_string("ab")
    assert true_suffix_automaton1.accepts_string("bab")
    assert not true_suffix_automaton1.accepts_string("cab")
    assert not true_suffix_automaton1.accepts_string("cabab")
    assert not true_suffix_automaton1.accepts_string("bb")
    assert not true_suffix_automaton1.accepts_string("aa")
    assert not true_suffix_automaton1.accepts_string("a")

    automaton1 = regex_to_dfa("(ab)*")
    true_suffix_automaton1 = (
        automaton1.to_epsilon_automaton().true_suffix_language().minimize()
    )
    assert true_suffix_automaton1.accepts_string("b")
    assert true_suffix_automaton1.accepts_string("ab")
    assert true_suffix_automaton1.accepts_string("bab")
    assert true_suffix_automaton1.accepts_string("abab")
    assert not true_suffix_automaton1.accepts_string("a")
    assert not true_suffix_automaton1.accepts_string("aa")
    assert not true_suffix_automaton1.accepts_string("bb")


def test_cheap_union():
    nfa1 = regex_to_dfa("(a|b)c")
    nfa2 = regex_to_dfa("aef(x)*")
    nfa3 = nfa1.to_epsilon_automaton().union(nfa2.to_epsilon_automaton()).minimize()
    assert nfa3.accepts_string("ac")
    assert nfa3.accepts_string("aefx")
    assert nfa3.accepts_string("aefxx")
    assert not nfa3.accepts_string("aefxxa")


def test_difference():
    """Tests the intersection of two languages"""
    enfa0 = regex_to_dfa("a*b")
    enfa1 = regex_to_dfa("c")
    enfa = enfa0.difference(enfa1).minimize()
    assert enfa.accepts_string("ab")
    assert enfa.accepts_string("b")
    assert not enfa.accepts_string("")
    assert not enfa.accepts_string("")
    enfa2 = regex_to_dfa("b*")
    enfa = enfa0.difference(enfa2)
    assert enfa.accepts_string("ab")
    assert not enfa.accepts_string("b")
    assert not enfa.accepts_string("c")


def test_difference_variable():
    enfa0 = regex_to_dfa(r"\x02[a-zA-Z0-9_]\w+\x03")
    enfa1 = regex_to_dfa(r"\x02(var|let|const)\x03")
    enfa = enfa0.difference(enfa1).minimize()
    assert not enfa.accepts_string("\x02var\x03")
    assert not enfa.accepts_string("\x02let\x03")
    assert not enfa.accepts_string("\x02const\x03")
    assert enfa.accepts_string("\x02var_123\x03")


@pytest.mark.skip()
def test_generated_language_categories():
    lex_map = compile_lex_map(LEX_MAP)
    terminals = list(lex_map.keys())

    lexing = (["lexString"], False, False)
    generated_fsa = generated_language(
        ['"hiii', 'ello"', lexing],
        lex_map,
        terminals,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(["lexString", "lexString"])

    lexing = (("lexString",), True, False)
    generated_fsa = generated_language(
        ['"hiii', 'ello"', lexing],
        lex_map,
        terminals,
    )
    assert generated_fsa.is_empty()

    lexing = (("lexString",), True, False)
    generated_fsa = generated_language(
        ['"hiii', lexing],
        lex_map,
        terminals,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert generated_fsa.accepts(["lexString"])


def test_generated_language_complex():
    cfg_lang = r"""
        S -> { String : Number } lexFence
        String -> lexString
        Number -> lexNumber
    """
    json_cfg = CFG.from_text(cfg_lang, "S")
    main_language_cfg = json_cfg.to_normal_form()
    lex_map = compile_lex_map(LEX_MAP, subtokens={})
    words = ["{", "\n", '"', None, None]
    generated_fsa = generated_language(words, lex_map, json_cfg.get_terminals())
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert not main_language_cfg.is_intersection_empty(generated_fsa, 100)


def test_generated_language_complex_2():
    cfg_lang = r"""
        S -> { String : NumberArray } lexFence
        String -> lexString
        NumberArray -> [ ] | [ NumberArrayElements ]
        NumberArrayElements -> Number | NumberArrayElements , Number
        Number -> { lexNumber }
    """
    json_cfg = CFG.from_text(cfg_lang, "S")
    print(json_cfg.to_text())
    print(LEX_MAP)
    main_language_cfg = json_cfg.to_normal_form()
    lex_map = compile_lex_map(LEX_MAP, subtokens={})
    words = ["{", "\n", '"', None, None, "[", "]", "}"]
    generated_fsa = generated_language(words, lex_map, json_cfg.get_terminals())
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert not main_language_cfg.is_intersection_empty(generated_fsa, 100)

    words = ['{ "hello" : [ { 1 } , {2}, ', "}"]
    generated_fsa = generated_language(words, lex_map, json_cfg.get_terminals())
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert main_language_cfg.is_intersection_empty(generated_fsa, 100)

    words = [
        '{ "hello" : [ {1}, {2}, ',
        (("{", "lexNumber", "}", "]"), False, False),
        "}",
    ]
    generated_fsa = generated_language(words, lex_map, json_cfg.get_terminals())
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert not main_language_cfg.is_intersection_empty(generated_fsa, 100)

    words = ['{ "hello" : [ { 1} , {2}, ', (("}", "]"), False, False), "}"]
    generated_fsa = generated_language(words, lex_map, json_cfg.get_terminals())
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert main_language_cfg.is_intersection_empty(generated_fsa, 100)


def test_schema_as_cfg():
    schema = {
        "type": "object",
        "properties": {
            "customerID": {"title": "Customer ID", "type": "string"},
            "vehicleID": {"title": "Vehicle ID", "type": "string"},
            "serviceRecords": {
                "title": "Service Records",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "serviceDate": {
                            "title": "Service Date",
                            "type": "string",
                            "format": "date",
                        },
                        "description": {"title": "Description", "type": "string"},
                        "cost": {"title": "Cost", "type": "number"},
                    },
                    "required": ["serviceDate"],
                },
            },
            "totalSpent": {"title": "Total Spent", "type": "number"},
        },
        "required": ["serviceRecords", "totalSpent"],
    }
    cfg, lex_map, subtokens = schema_to_cfg.schema_to_cfg(schema)
    assert not cfg.is_empty()
    words = [
        """{
  "serviceRecords": [
    {"serviceDate": "2017-04-22"},""",
        (("}", "]"), False, False),
        """ "totalSpent": 75
}""",
        EOS,
    ]
    generated_fsa = generated_language(
        words, compile_lex_map(lex_map, subtokens), cfg.get_terminals()
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert cfg.is_intersection_empty(generated_fsa, 100)


@pytest.mark.skip()
def test_schema_as_cfg_3():
    schema = {
        "type": "object",
        "properties": {
            "customerID": {"title": "Customer ID", "type": "string", "format": "date"},
            "vehicleID": {"title": "Vehicle ID", "type": "string"},
        },
        "required": ["customerID", "vehicleID"],
    }
    lang, lex_map, subtokens = schema_to_cfg.schema_to_cfg(schema)
    compiled_lex_map = compile_lex_map(lex_map, subtokens)
    # inverted subtoken map
    supertokens = derive_supertokens(subtokens)

    # now a 5 could be parsed as part of a date (which takes precedence over being parsed as a string)
    assert (("dateToken",), True, True) in lex("5", compiled_lex_map, is_first=False)
    assert not lang.is_empty()
    words = [
        """{
  "customerID": "2025-04-02",
  "vehicleID": "4T1BF1FKVFU033""",
        None,
        "5",
    ]
    # for all tokens it should be rejected
    generated_fsa = generated_language(
        words,
        compiled_lex_map,
        lang.get_terminals(),
        inject_gap_size=1,
        max_total_injections=1,
        trace=True,
        subtokens=subtokens,
        supertokens=supertokens,
        single_token_lexing=((("stringToken",), True, True),),
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    # still we need to the connection between the preceding string and the possible continuation as date
    # We want that the intersection is thus not empty
    assert generated_fsa.accepts(
        [
            "{",
            "propertyToken(customerID)",
            ":",
            "dateToken",
            ",",
            "propertyToken(vehicleID)",
            ":",
            "stringToken",
        ]
    )
    assert generated_fsa.accepts(
        [
            "{",
            "propertyToken(customerID)",
            ":",
            "dateToken",
            ",",
            "propertyToken(vehicleID)",
            ":",
            "stringToken",
            "dateToken",
        ]
    )
    assert not lang.is_intersection_empty(generated_fsa, 100)


def test_something_before_eos():
    lang, lex_map, subtokens = smiles_schema()
    lang = lang.concatenate(CFG.from_text("S -> lexFence", "S"))
    lex_map["lexFence"] = "```"
    compiled_lex_map = compile_lex_map(lex_map, subtokens)
    # inverted subtoken map
    supertokens = derive_supertokens(subtokens)

    words = [
        "CC",
        None,
        "=",
        None,
        "\n```",
        None,
        EOS,
    ]
    # This should be ok because we can just insert EOS in the gap
    generated_fsa = generated_language(
        words,
        compiled_lex_map,
        lang.get_terminals(),
        subtokens=subtokens,
        supertokens=supertokens,
    )
    assert not lang.is_intersection_empty(generated_fsa, 100)
    # This is ok because we can just insert EOS in the gap
    # generated_fsa = generated_language(
    #     words,
    #     compiled_lex_map,
    #     lang.get_terminals(),
    #     inject_gap_size=5,
    #     max_total_injections=5,
    #     trace=True,
    #     subtokens=subtokens,
    #     supertokens=supertokens,
    #     single_token_lexing=((("organicSymbol",), False, False),),
    # )
    # assert not lang.is_intersection_empty(generated_fsa)


@pytest.mark.skip()
def test_schema_as_cfg_2():
    schema = {
        "type": "object",
        "properties": {
            "customerID": {"title": "Customer ID", "type": "string"},
            "vehicleID": {"title": "Vehicle ID", "type": "string"},
            "serviceRecords": {
                "title": "Service Records",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "serviceDate": {
                            "title": "Service Date",
                            "type": "string",
                            "format": "date",
                        },
                        "description": {"title": "Description", "type": "string"},
                        "cost": {"title": "Cost", "type": "number"},
                    },
                    "required": ["serviceDate", "description", "cost"],
                },
            },
            "totalSpent": {"title": "Total Spent", "type": "number"},
        },
        "required": ["customerID", "vehicleID", "serviceRecords", "totalSpent"],
    }
    lang, lex_map, subtokens = schema_to_cfg.schema_to_cfg(schema)
    lang = lang.concatenate(CFG.from_text("S -> lexFence", "S"))
    lex_map["lexFence"] = "```"
    print(lang.to_text())
    lang = lang.to_normal_form()
    print(lang.num_productions())
    compiled_lex_map = compile_lex_map(lex_map, subtokens)
    assert not lang.is_empty()
    words = [
        "{",
        "\n",
        " ",
        ' "',
        "customer",
        "ID",
        '":',
        ' "',
        "CU",
        "7",
        "8",
        "9",
        "4",
        "5",
        "6",
        '",',
        "\n",
        " ",
        ' "',
        "vehicle",
        "ID",
        '":',
        ' "',
        "4",
        "T",
        "1",
        "BF",
        "1",
        "FK",
        "5",
        "FU",
        "0",
        "3",
        "3",
        "2",
        "0",
        "9",
        '",',
        "\n",
        " ",
        ' "',
        "service",
        "Records",
        '":',
        " [",
        "\n",
        "   ",
        ' {"',
        "service",
        "Date",
        '":',
        ' "',
        "2",
        "0",
        "1",
        "6",
        "-",
        "0",
        "5",
        "-",
        "1",
        "0",
        '",',
        ' "',
        "description",
        '":',
        ' "',
        "Oil",
        " Change",
        '",',
        ' "',
        "cost",
        '":',
        " ",
        "7",
        "5",
        "},",
        "\n",
        "   ",
        ' {"',
        "service",
        "Date",
        '":',
        ' "',
        "2",
        "0",
        "1",
        "7",
        "-",
        "0",
        "4",
        "-",
        "2",
        "2",
        '",',
        ' "',
        "description",
        '":',
        ' "',
        "B",
        "rake",
        " Pad",
        " Replacement",
        '",',
        ' "',
        "cost",
        '":',
        " ",
        "1",
        "5",
        "0",
        "},",
        "\n",
        None,
        ' "',
        "total",
        "Sp",
        "ent",
        '":',
        " ",
        None,
        "5",
        "\n",
        "}",
        "\n",
        "```",
        EOS,
        EOS,
    ]
    # for all tokens it should be rejected
    supertokens = derive_supertokens(subtokens)
    generated_fsa = generated_language(
        words,
        compiled_lex_map,
        lang.get_terminals(),
        subtokens=subtokens,
        supertokens=supertokens,
        inject_gap_size=1,
        max_total_injections=1,
    )
    assert not generated_fsa.is_empty()
    assert generated_fsa.num_states() > 0
    assert lang.is_intersection_empty(generated_fsa)


SINGLE_TOKEN_LEXINGS_LLADA_CFG = [
    (("stringToken",), True, True),
    (("propertyToken(description)",), False, True),
    (("propertyToken(serviceDate)",), False, True),
    (("dateToken",), False, True),
    (("propertyToken(vehicleID)",), True, False),
    (("propertyToken(vehicleID)",), False, True),
    (("propertyToken(cost)",), True, False),
    (("propertyToken(serviceDate)",), True, False),
    (("dateToken",), True, False),
    (("propertyToken(serviceRecords)",), False, True),
    (("propertyToken(customerID)",), False, True),
    (("propertyToken(description)",), True, False),
    (("propertyToken(cost)",), False, True),
    (("propertyToken(totalSpent)",), True, False),
    (("propertyToken(serviceRecords)",), True, False),
    (("propertyToken(customerID)",), True, False),
    (("propertyToken(totalSpent)",), False, True),
    (("numberToken",), True, True),
    ((",",), False, False),
    (("dateToken",), True, True),
    (("numberToken",), False, True),
    (("numberToken",), False, False),
    (("numberToken",), True, False),
    ((":",), False, False),
    (("propertyToken(serviceDate)",), True, True),
    (("propertyToken(vehicleID)",), True, True),
    (("propertyToken(customerID)",), True, True),
    (("propertyToken(serviceRecords)",), True, True),
    (("propertyToken(totalSpent)",), True, True),
    (("[",), False, False),
    (("]",), False, False),
    (("lexFence",), False, True),
    (("lexFence",), True, False),
    (("lexFence",), True, True),
    (("boolNullToken",), True, True),
    (("propertyToken(description)",), True, True),
    (("propertyToken(cost)",), True, True),
    (("boolNullToken",), True, False),
    (("boolNullToken",), False, True),
    (("{",), False, False),
    (("}",), False, False),
    ((), False, False),
    (("boolNullToken", "boolNullToken"), True, True),
    (("stringToken",), True, False),
    ((":", ":"), False, False),
    (("stringToken",), False, True),
    (("propertyToken(totalSpent)", ","), True, False),
    (("propertyToken(description)", ","), True, False),
    (("propertyToken(serviceDate)", ","), True, False),
    (("propertyToken(cost)", ","), True, False),
    (("propertyToken(vehicleID)", ","), True, False),
    (("dateToken", ","), True, False),
    (("propertyToken(serviceRecords)", ","), True, False),
    (("propertyToken(customerID)", ","), True, False),
    (("boolNullToken",), False, False),
    (("}", "}"), False, False),
    (("propertyToken(customerID)", ":"), True, False),
    (("propertyToken(totalSpent)", ":"), True, False),
    (("propertyToken(vehicleID)", ":"), True, False),
    (("dateToken", ":"), True, False),
    (("propertyToken(description)", ":"), True, False),
    (("propertyToken(serviceRecords)", ":"), True, False),
    (("propertyToken(serviceDate)", ":"), True, False),
    (("propertyToken(cost)", ":"), True, False),
    (("]", ","), False, False),
    ((",", "propertyToken(cost)"), False, True),
    ((",", "propertyToken(totalSpent)"), False, True),
    ((",", "propertyToken(description)"), False, True),
    ((",", "propertyToken(customerID)"), False, True),
    ((",", "propertyToken(serviceDate)"), False, True),
    ((",", "propertyToken(vehicleID)"), False, True),
    ((",", "propertyToken(serviceRecords)"), False, True),
    ((",", "dateToken"), False, True),
    (("lexFence", "lexFence"), True, True),
    (("lexFence",), False, False),
    (("}", "{"), False, False),
    (("[", "]"), False, False),
    (("}", ","), False, False),
    (("]", "["), False, False),
    (("stringToken", "propertyToken(customerID)"), True, True),
    (("stringToken", "propertyToken(totalSpent)"), True, True),
    (("stringToken", "propertyToken(serviceRecords)"), True, True),
    (("stringToken", "dateToken"), True, True),
    (("stringToken", "propertyToken(description)"), True, True),
    (("stringToken",), False, False),
    (("stringToken", "propertyToken(vehicleID)"), True, True),
    (("stringToken", "propertyToken(cost)"), True, True),
    (("stringToken", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(description)", "]"), True, False),
    (("propertyToken(vehicleID)", "]"), True, False),
    (("propertyToken(totalSpent)", "]"), True, False),
    (("propertyToken(serviceDate)", "]"), True, False),
    (("propertyToken(customerID)", "]"), True, False),
    (("propertyToken(serviceRecords)", "]"), True, False),
    (("dateToken", "]"), True, False),
    (("propertyToken(cost)", "]"), True, False),
    (("[", "propertyToken(totalSpent)"), False, True),
    (("[", "propertyToken(cost)"), False, True),
    (("[", "dateToken"), False, True),
    (("[", "propertyToken(serviceDate)"), False, True),
    (("[", "propertyToken(description)"), False, True),
    (("[", "propertyToken(customerID)"), False, True),
    (("[", "propertyToken(vehicleID)"), False, True),
    (("[", "propertyToken(serviceRecords)"), False, True),
    ((":", "propertyToken(cost)"), False, True),
    ((":", "propertyToken(serviceDate)"), False, True),
    ((":", "dateToken"), False, True),
    ((":", "propertyToken(description)"), False, True),
    ((":", "propertyToken(totalSpent)"), False, True),
    ((":", "propertyToken(serviceRecords)"), False, True),
    ((":", "propertyToken(customerID)"), False, True),
    ((":", "propertyToken(vehicleID)"), False, True),
    (("]", "{"), False, False),
    (("propertyToken(serviceDate)", "propertyToken(customerID)"), True, True),
    (("propertyToken(vehicleID)", "dateToken"), True, True),
    (("propertyToken(vehicleID)", "propertyToken(cost)"), True, True),
    (("propertyToken(description)", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(serviceRecords)", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(customerID)", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(serviceRecords)", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(cost)", "dateToken"), True, True),
    (("propertyToken(serviceRecords)", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(serviceRecords)", "propertyToken(customerID)"), True, True),
    (("propertyToken(serviceRecords)", "propertyToken(description)"), True, True),
    (("propertyToken(description)", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(totalSpent)", "propertyToken(customerID)"), True, True),
    (("propertyToken(totalSpent)", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(serviceDate)", "propertyToken(cost)"), True, True),
    (("propertyToken(cost)", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(description)", "propertyToken(customerID)"), True, True),
    (("propertyToken(vehicleID)", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(vehicleID)", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(serviceRecords)", "propertyToken(cost)"), True, True),
    (("dateToken", "propertyToken(cost)"), True, True),
    (("propertyToken(description)", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(serviceDate)", "dateToken"), True, True),
    (("propertyToken(vehicleID)", "propertyToken(customerID)"), True, True),
    (("propertyToken(customerID)", "propertyToken(description)"), True, True),
    (("dateToken", "propertyToken(description)"), True, True),
    (("propertyToken(vehicleID)", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(totalSpent)", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(customerID)", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(customerID)", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(cost)", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(customerID)", "propertyToken(customerID)"), True, True),
    (("dateToken", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(cost)", "propertyToken(customerID)"), True, True),
    (("propertyToken(description)", "propertyToken(cost)"), True, True),
    (("propertyToken(cost)", "propertyToken(description)"), True, True),
    (("dateToken", "propertyToken(customerID)"), True, True),
    (("propertyToken(vehicleID)", "propertyToken(description)"), True, True),
    (("propertyToken(cost)", "propertyToken(serviceRecords)"), True, True),
    (("dateToken", "dateToken"), True, True),
    (("propertyToken(customerID)", "propertyToken(cost)"), True, True),
    (("dateToken", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(cost)", "propertyToken(cost)"), True, True),
    (("propertyToken(cost)", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(serviceDate)", "propertyToken(totalSpent)"), True, True),
    (("dateToken", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(serviceDate)", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(customerID)", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(customerID)", "dateToken"), True, True),
    (("propertyToken(vehicleID)", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(serviceDate)", "propertyToken(description)"), True, True),
    (("propertyToken(totalSpent)", "propertyToken(cost)"), True, True),
    (("propertyToken(totalSpent)", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(totalSpent)", "propertyToken(totalSpent)"), True, True),
    (("dateToken", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(totalSpent)", "dateToken"), True, True),
    (("propertyToken(description)", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(totalSpent)", "propertyToken(description)"), True, True),
    (("propertyToken(description)", "propertyToken(description)"), True, True),
    (("propertyToken(serviceRecords)", "dateToken"), True, True),
    (("propertyToken(serviceDate)", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(serviceDate)", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(description)", "dateToken"), True, True),
    (("propertyToken(serviceRecords)", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(customerID)", ",", "propertyToken(description)"), True, True),
    (("propertyToken(serviceRecords)", ",", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(serviceDate)", ",", "dateToken"), True, True),
    (("propertyToken(description)", ",", "propertyToken(description)"), True, True),
    (("propertyToken(serviceRecords)", ",", "dateToken"), True, True),
    (("propertyToken(description)", ",", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(cost)", ",", "propertyToken(customerID)"), True, True),
    (("propertyToken(totalSpent)", ",", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(serviceDate)", ",", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(serviceDate)", ",", "propertyToken(cost)"), True, True),
    (("dateToken", ",", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(customerID)", ",", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(customerID)", ",", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(cost)", ",", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(customerID)", ",", "propertyToken(customerID)"), True, True),
    (("propertyToken(totalSpent)", ",", "dateToken"), True, True),
    (("propertyToken(customerID)", ",", "propertyToken(serviceDate)"), True, True),
    (("dateToken", ",", "propertyToken(cost)"), True, True),
    (("propertyToken(cost)", ",", "propertyToken(description)"), True, True),
    (("propertyToken(customerID)", ",", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(vehicleID)", ",", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(vehicleID)", ",", "propertyToken(customerID)"), True, True),
    (("propertyToken(serviceDate)", ",", "propertyToken(customerID)"), True, True),
    (("propertyToken(vehicleID)", ",", "propertyToken(totalSpent)"), True, True),
    (("dateToken", ",", "propertyToken(description)"), True, True),
    (("propertyToken(totalSpent)", ",", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(description)", ",", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(description)", ",", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(cost)", ",", "propertyToken(cost)"), True, True),
    (("propertyToken(description)", ",", "dateToken"), True, True),
    (("propertyToken(description)", ",", "propertyToken(cost)"), True, True),
    (("propertyToken(serviceDate)", ",", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(serviceDate)", ",", "propertyToken(description)"), True, True),
    (("dateToken", ",", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(serviceRecords)", ",", "propertyToken(cost)"), True, True),
    (("propertyToken(vehicleID)", ",", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(serviceDate)", ",", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(vehicleID)", ",", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(totalSpent)", ",", "propertyToken(cost)"), True, True),
    (("propertyToken(vehicleID)", ",", "propertyToken(description)"), True, True),
    (("propertyToken(customerID)", ",", "propertyToken(cost)"), True, True),
    (("propertyToken(serviceRecords)", ",", "propertyToken(customerID)"), True, True),
    (("propertyToken(serviceRecords)", ",", "propertyToken(description)"), True, True),
    (("dateToken", ",", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(cost)", ",", "dateToken"), True, True),
    (("propertyToken(totalSpent)", ",", "propertyToken(customerID)"), True, True),
    (("propertyToken(description)", ",", "propertyToken(customerID)"), True, True),
    (("propertyToken(serviceRecords)", ",", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(totalSpent)", ",", "propertyToken(totalSpent)"), True, True),
    (("dateToken", ",", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(description)", ",", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(cost)", ",", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(customerID)", ",", "dateToken"), True, True),
    (("propertyToken(serviceRecords)", ",", "propertyToken(vehicleID)"), True, True),
    (
        ("propertyToken(serviceRecords)", ",", "propertyToken(serviceRecords)"),
        True,
        True,
    ),
    (("propertyToken(serviceDate)", ",", "propertyToken(serviceDate)"), True, True),
    (("dateToken", ",", "propertyToken(customerID)"), True, True),
    (("propertyToken(totalSpent)", ",", "propertyToken(description)"), True, True),
    (("propertyToken(cost)", ",", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(totalSpent)", ",", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(vehicleID)", ",", "propertyToken(cost)"), True, True),
    (("dateToken", ",", "dateToken"), True, True),
    (("propertyToken(vehicleID)", ",", "dateToken"), True, True),
    (("propertyToken(cost)", ",", "propertyToken(serviceRecords)"), True, True),
    (("{", "{"), False, False),
    (("propertyToken(totalSpent)", "}"), True, False),
    (("propertyToken(vehicleID)", "}"), True, False),
    (("dateToken", "}"), True, False),
    (("propertyToken(serviceDate)", "}"), True, False),
    (("propertyToken(description)", "}"), True, False),
    (("propertyToken(customerID)", "}"), True, False),
    (("propertyToken(serviceRecords)", "}"), True, False),
    (("propertyToken(cost)", "}"), True, False),
    (("[", "["), False, False),
    (("stringToken", "propertyToken(totalSpent)"), False, True),
    (("stringToken", "propertyToken(cost)"), False, True),
    (("stringToken", "dateToken"), False, True),
    (("stringToken", "propertyToken(serviceRecords)"), False, True),
    (("stringToken", "propertyToken(description)"), False, True),
    (("stringToken", "propertyToken(vehicleID)"), False, True),
    (("stringToken", "propertyToken(serviceDate)"), False, True),
    (("stringToken", "stringToken"), True, False),
    (("stringToken", "propertyToken(customerID)"), False, True),
    (("]", "]"), False, False),
    (("[", ":"), False, False),
    (("propertyToken(serviceRecords)", ":", "propertyToken(customerID)"), True, True),
    (("propertyToken(serviceDate)", ":", "dateToken"), True, True),
    (("propertyToken(customerID)", ":", "propertyToken(description)"), True, True),
    (("propertyToken(serviceRecords)", ":", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(vehicleID)", ":", "propertyToken(serviceDate)"), True, True),
    (("dateToken", ":", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(serviceRecords)", ":", "propertyToken(description)"), True, True),
    (("propertyToken(vehicleID)", ":", "dateToken"), True, True),
    (("propertyToken(description)", ":", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(description)", ":", "propertyToken(customerID)"), True, True),
    (("propertyToken(totalSpent)", ":", "propertyToken(customerID)"), True, True),
    (("propertyToken(vehicleID)", ":", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(cost)", ":", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(serviceRecords)", ":", "dateToken"), True, True),
    (("dateToken", ":", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(totalSpent)", ":", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(serviceDate)", ":", "propertyToken(cost)"), True, True),
    (("propertyToken(serviceRecords)", ":", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(vehicleID)", ":", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(vehicleID)", ":", "propertyToken(cost)"), True, True),
    (("propertyToken(vehicleID)", ":", "propertyToken(description)"), True, True),
    (("propertyToken(customerID)", ":", "propertyToken(cost)"), True, True),
    (("propertyToken(customerID)", ":", "propertyToken(customerID)"), True, True),
    (("propertyToken(description)", ":", "propertyToken(serviceDate)"), True, True),
    (("dateToken", ":", "propertyToken(description)"), True, True),
    (("propertyToken(totalSpent)", ":", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(cost)", ":", "propertyToken(cost)"), True, True),
    (("propertyToken(serviceRecords)", ":", "propertyToken(cost)"), True, True),
    (("propertyToken(serviceDate)", ":", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(cost)", ":", "dateToken"), True, True),
    (("dateToken", ":", "propertyToken(cost)"), True, True),
    (("propertyToken(totalSpent)", ":", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(cost)", ":", "propertyToken(description)"), True, True),
    (("dateToken", ":", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(serviceRecords)", ":", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(customerID)", ":", "dateToken"), True, True),
    (("propertyToken(description)", ":", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(cost)", ":", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(customerID)", ":", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(totalSpent)", ":", "propertyToken(cost)"), True, True),
    (("propertyToken(totalSpent)", ":", "dateToken"), True, True),
    (("propertyToken(description)", ":", "propertyToken(description)"), True, True),
    (("propertyToken(serviceDate)", ":", "propertyToken(customerID)"), True, True),
    (("propertyToken(totalSpent)", ":", "propertyToken(description)"), True, True),
    (("dateToken", ":", "dateToken"), True, True),
    (("propertyToken(vehicleID)", ":", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(description)", ":", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(totalSpent)", ":", "propertyToken(vehicleID)"), True, True),
    (("dateToken", ":", "propertyToken(customerID)"), True, True),
    (("propertyToken(customerID)", ":", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(serviceDate)", ":", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(vehicleID)", ":", "propertyToken(customerID)"), True, True),
    (("propertyToken(serviceDate)", ":", "propertyToken(description)"), True, True),
    (("propertyToken(cost)", ":", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(cost)", ":", "propertyToken(serviceRecords)"), True, True),
    (("dateToken", ":", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(serviceDate)", ":", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(description)", ":", "propertyToken(cost)"), True, True),
    (("propertyToken(description)", ":", "dateToken"), True, True),
    (("propertyToken(customerID)", ":", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(serviceDate)", ":", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(customerID)", ":", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(cost)", ":", "propertyToken(customerID)"), True, True),
    (
        ("propertyToken(serviceRecords)", ":", "propertyToken(serviceRecords)"),
        True,
        True,
    ),
    (("]", ":"), False, False),
    (("{", "propertyToken(cost)"), False, True),
    (("{", "propertyToken(totalSpent)"), False, True),
    (("{", "dateToken"), False, True),
    (("{", "propertyToken(serviceRecords)"), False, True),
    (("{", "propertyToken(description)"), False, True),
    (("{", "propertyToken(customerID)"), False, True),
    (("{", "propertyToken(vehicleID)"), False, True),
    (("{", "propertyToken(serviceDate)"), False, True),
    (("{", "}"), False, False),
    ((",", "numberToken"), False, True),
    (("stringToken", ","), True, False),
    (("]", "{", "}"), False, False),
    (("lexFence", ","), True, False),
    ((",", "boolNullToken"), False, True),
    (("}", "["), False, False),
    (("}", "numberToken"), False, True),
    (("stringToken", "["), True, False),
    (("}", "}", "{"), False, False),
    ((",", ","), False, False),
    (("}", "propertyToken(serviceRecords)"), False, True),
    (("}", "propertyToken(vehicleID)"), False, True),
    (("}", "propertyToken(serviceDate)"), False, True),
    (("}", "propertyToken(cost)"), False, True),
    (("}", "propertyToken(customerID)"), False, True),
    (("}", "propertyToken(description)"), False, True),
    (("}", "propertyToken(totalSpent)"), False, True),
    (("}", "dateToken"), False, True),
    (("stringToken", "numberToken"), True, True),
    (("stringToken", ","), False, False),
    (("stringToken", "stringToken"), True, True),
    ((":", "boolNullToken"), False, True),
    (("[", "boolNullToken"), False, True),
    (("[", "numberToken"), False, True),
    ((":", "numberToken"), False, True),
    (("{", "boolNullToken"), False, True),
    ((":", "["), False, False),
    (("{", "numberToken"), False, True),
    (("}", ":"), False, False),
    ((",", ":"), False, False),
    (("propertyToken(serviceDate)", "]", ","), True, False),
    (("propertyToken(vehicleID)", "]", ","), True, False),
    (("dateToken", "]", ","), True, False),
    (("propertyToken(customerID)", "]", ","), True, False),
    (("propertyToken(totalSpent)", "]", ","), True, False),
    (("propertyToken(serviceRecords)", "]", ","), True, False),
    (("propertyToken(description)", "]", ","), True, False),
    (("propertyToken(cost)", "]", ","), True, False),
    ((":", "lexFence"), False, True),
    (("[", "]", ","), False, False),
    (("]", "]", ","), False, False),
    (("[", ":", ","), False, False),
    ((",", "["), False, False),
    (("]", "[", "propertyToken(vehicleID)"), False, True),
    (("]", "[", "propertyToken(serviceDate)"), False, True),
    (("]", "[", "dateToken"), False, True),
    (("]", "[", "propertyToken(totalSpent)"), False, True),
    (("]", "[", "propertyToken(customerID)"), False, True),
    (("]", "[", "propertyToken(cost)"), False, True),
    (("]", "[", "propertyToken(description)"), False, True),
    (("]", "[", "propertyToken(serviceRecords)"), False, True),
    (("propertyToken(serviceDate)", "}", ","), True, False),
    (("propertyToken(serviceRecords)", "}", ","), True, False),
    (("propertyToken(totalSpent)", "}", ","), True, False),
    (("dateToken", "}", ","), True, False),
    (("propertyToken(description)", "}", ","), True, False),
    (("propertyToken(customerID)", "}", ","), True, False),
    (("propertyToken(vehicleID)", "}", ","), True, False),
    (("propertyToken(cost)", "}", ","), True, False),
    (("}", "]"), False, False),
    (("}", "}", "}"), False, False),
    (("}", "lexFence"), False, True),
    (("]", "{", "}", ","), False, False),
    (("]", "numberToken"), False, True),
    (("}", "stringToken"), False, True),
    (("stringToken", ":"), True, False),
    ((":", ":", ":", ":"), False, False),
    (("}", "}", ","), False, False),
    (("propertyToken(description)", "stringToken"), True, False),
    (("propertyToken(serviceDate)", "stringToken"), True, False),
    (("propertyToken(serviceRecords)", "stringToken"), True, False),
    (("propertyToken(vehicleID)", "stringToken"), True, False),
    (("dateToken", "stringToken"), True, False),
    (("propertyToken(totalSpent)", "stringToken"), True, False),
    (("propertyToken(cost)", "stringToken"), True, False),
    (("propertyToken(customerID)", "stringToken"), True, False),
    (("{", ":"), False, False),
    (("{", "}", ","), False, False),
    (("]", "}"), False, False),
    (("[", ","), False, False),
    (("stringToken", "{"), True, False),
    (("]", ",", "["), False, False),
    (("[", "]", "{"), False, False),
    ((":", "stringToken"), False, True),
    (("dateToken", "]", "[", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(vehicleID)", "]", "[", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(customerID)", "]", "[", "propertyToken(description)"), True, True),
    (("propertyToken(totalSpent)", "]", "[", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(totalSpent)", "]", "[", "dateToken"), True, True),
    (("dateToken", "]", "[", "propertyToken(serviceRecords)"), True, True),
    (("dateToken", "]", "[", "propertyToken(customerID)"), True, True),
    (("dateToken", "]", "[", "propertyToken(cost)"), True, True),
    (("propertyToken(cost)", "]", "[", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(totalSpent)", "]", "[", "propertyToken(cost)"), True, True),
    (("propertyToken(serviceDate)", "]", "[", "dateToken"), True, True),
    (("dateToken", "]", "[", "propertyToken(description)"), True, True),
    (("propertyToken(serviceRecords)", "]", "[", "propertyToken(cost)"), True, True),
    (
        ("propertyToken(serviceRecords)", "]", "[", "propertyToken(description)"),
        True,
        True,
    ),
    (("propertyToken(customerID)", "]", "[", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(vehicleID)", "]", "[", "dateToken"), True, True),
    (("propertyToken(customerID)", "]", "[", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(cost)", "]", "[", "propertyToken(serviceRecords)"), True, True),
    (("propertyToken(vehicleID)", "]", "[", "propertyToken(cost)"), True, True),
    (("dateToken", "]", "[", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(cost)", "]", "[", "propertyToken(customerID)"), True, True),
    (("propertyToken(description)", "]", "[", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(customerID)", "]", "[", "dateToken"), True, True),
    (
        ("propertyToken(serviceRecords)", "]", "[", "propertyToken(vehicleID)"),
        True,
        True,
    ),
    (
        ("propertyToken(description)", "]", "[", "propertyToken(serviceRecords)"),
        True,
        True,
    ),
    (("propertyToken(description)", "]", "[", "propertyToken(customerID)"), True, True),
    (("propertyToken(cost)", "]", "[", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(customerID)", "]", "[", "propertyToken(cost)"), True, True),
    (("propertyToken(totalSpent)", "]", "[", "propertyToken(description)"), True, True),
    (("propertyToken(cost)", "]", "[", "dateToken"), True, True),
    (
        ("propertyToken(description)", "]", "[", "propertyToken(serviceDate)"),
        True,
        True,
    ),
    (("propertyToken(cost)", "]", "[", "propertyToken(description)"), True, True),
    (
        ("propertyToken(serviceDate)", "]", "[", "propertyToken(serviceRecords)"),
        True,
        True,
    ),
    (("propertyToken(serviceDate)", "]", "[", "propertyToken(customerID)"), True, True),
    (("propertyToken(cost)", "]", "[", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(description)", "]", "[", "dateToken"), True, True),
    (
        ("propertyToken(serviceRecords)", "]", "[", "propertyToken(totalSpent)"),
        True,
        True,
    ),
    (
        ("propertyToken(customerID)", "]", "[", "propertyToken(serviceRecords)"),
        True,
        True,
    ),
    (("propertyToken(totalSpent)", "]", "[", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(totalSpent)", "]", "[", "propertyToken(vehicleID)"), True, True),
    (
        ("propertyToken(serviceRecords)", "]", "[", "propertyToken(serviceRecords)"),
        True,
        True,
    ),
    (("propertyToken(cost)", "]", "[", "propertyToken(cost)"), True, True),
    (("propertyToken(serviceDate)", "]", "[", "propertyToken(totalSpent)"), True, True),
    (
        ("propertyToken(serviceRecords)", "]", "[", "propertyToken(customerID)"),
        True,
        True,
    ),
    (("propertyToken(vehicleID)", "]", "[", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(customerID)", "]", "[", "propertyToken(customerID)"), True, True),
    (("propertyToken(serviceDate)", "]", "[", "propertyToken(cost)"), True, True),
    (("dateToken", "]", "[", "dateToken"), True, True),
    (("propertyToken(description)", "]", "[", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(serviceDate)", "]", "[", "propertyToken(vehicleID)"), True, True),
    (("dateToken", "]", "[", "propertyToken(totalSpent)"), True, True),
    (
        ("propertyToken(serviceRecords)", "]", "[", "propertyToken(serviceDate)"),
        True,
        True,
    ),
    (("propertyToken(serviceRecords)", "]", "[", "dateToken"), True, True),
    (
        ("propertyToken(serviceDate)", "]", "[", "propertyToken(serviceDate)"),
        True,
        True,
    ),
    (("propertyToken(description)", "]", "[", "propertyToken(cost)"), True, True),
    (
        ("propertyToken(totalSpent)", "]", "[", "propertyToken(serviceRecords)"),
        True,
        True,
    ),
    (("propertyToken(totalSpent)", "]", "[", "propertyToken(customerID)"), True, True),
    (
        ("propertyToken(serviceDate)", "]", "[", "propertyToken(description)"),
        True,
        True,
    ),
    (("propertyToken(customerID)", "]", "[", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(vehicleID)", "]", "[", "propertyToken(description)"), True, True),
    (("propertyToken(vehicleID)", "]", "[", "propertyToken(vehicleID)"), True, True),
    (
        ("propertyToken(vehicleID)", "]", "[", "propertyToken(serviceRecords)"),
        True,
        True,
    ),
    (("propertyToken(vehicleID)", "]", "[", "propertyToken(customerID)"), True, True),
    (
        ("propertyToken(description)", "]", "[", "propertyToken(description)"),
        True,
        True,
    ),
    (("lexFence", ":"), True, False),
    (("]", "propertyToken(customerID)"), False, True),
    (("]", "propertyToken(totalSpent)"), False, True),
    (("]", "propertyToken(serviceRecords)"), False, True),
    (("]", "dateToken"), False, True),
    (("]", "propertyToken(serviceDate)"), False, True),
    (("]", "propertyToken(cost)"), False, True),
    (("]", "propertyToken(vehicleID)"), False, True),
    (("]", "propertyToken(description)"), False, True),
    (("[", "lexFence"), False, True),
    (("propertyToken(cost)", "{"), True, False),
    (("propertyToken(customerID)", "{"), True, False),
    (("propertyToken(totalSpent)", "{"), True, False),
    (("propertyToken(serviceDate)", "{"), True, False),
    (("propertyToken(vehicleID)", "{"), True, False),
    (("dateToken", "{"), True, False),
    (("propertyToken(serviceRecords)", "{"), True, False),
    (("propertyToken(description)", "{"), True, False),
    (("lexFence", "lexFence"), False, True),
    ((":", "{"), False, False),
    (("[", "]", "[", "]"), False, False),
    (("{", "{", "{"), False, False),
    (("[", "{"), False, False),
    (("propertyToken(description)", "numberToken"), True, True),
    (("propertyToken(vehicleID)", "numberToken"), True, True),
    (("propertyToken(customerID)", "numberToken"), True, True),
    (("propertyToken(totalSpent)", "numberToken"), True, True),
    (("propertyToken(serviceDate)", "numberToken"), True, True),
    (("propertyToken(cost)", "numberToken"), True, True),
    (("dateToken", "numberToken"), True, True),
    (("propertyToken(serviceRecords)", "numberToken"), True, True),
    (("]", "stringToken"), False, True),
    (("}", "}", "}", "}"), False, False),
    (("stringToken", ",", "propertyToken(totalSpent)"), True, True),
    (("stringToken", ",", "propertyToken(serviceRecords)"), True, True),
    (("stringToken", ",", "propertyToken(cost)"), True, True),
    (("stringToken", ",", "propertyToken(description)"), True, True),
    (("stringToken", ",", "propertyToken(vehicleID)"), True, True),
    (("stringToken", ",", "propertyToken(serviceDate)"), True, True),
    (("stringToken", ",", "propertyToken(customerID)"), True, True),
    (("stringToken", ",", "dateToken"), True, True),
    (("dateToken", ":", "{", "dateToken"), True, True),
    (("propertyToken(vehicleID)", ":", "{", "dateToken"), True, True),
    (("propertyToken(serviceDate)", ":", "{", "propertyToken(customerID)"), True, True),
    (
        ("propertyToken(serviceRecords)", ":", "{", "propertyToken(serviceRecords)"),
        True,
        True,
    ),
    (("propertyToken(serviceDate)", ":", "{", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(cost)", ":", "{", "propertyToken(description)"), True, True),
    (("propertyToken(customerID)", ":", "{", "propertyToken(cost)"), True, True),
    (("propertyToken(vehicleID)", ":", "{", "propertyToken(totalSpent)"), True, True),
    (
        ("propertyToken(totalSpent)", ":", "{", "propertyToken(serviceRecords)"),
        True,
        True,
    ),
    (("propertyToken(serviceDate)", ":", "{", "dateToken"), True, True),
    (("propertyToken(customerID)", ":", "{", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(cost)", ":", "{", "dateToken"), True, True),
    (
        ("propertyToken(serviceDate)", ":", "{", "propertyToken(description)"),
        True,
        True,
    ),
    (("propertyToken(description)", ":", "{", "propertyToken(customerID)"), True, True),
    (("propertyToken(description)", ":", "{", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(serviceDate)", ":", "{", "propertyToken(totalSpent)"), True, True),
    (
        ("propertyToken(serviceDate)", ":", "{", "propertyToken(serviceRecords)"),
        True,
        True,
    ),
    (("propertyToken(totalSpent)", ":", "{", "propertyToken(totalSpent)"), True, True),
    (("dateToken", ":", "{", "propertyToken(serviceRecords)"), True, True),
    (("dateToken", ":", "{", "propertyToken(customerID)"), True, True),
    (("propertyToken(cost)", ":", "{", "propertyToken(cost)"), True, True),
    (
        ("propertyToken(serviceRecords)", ":", "{", "propertyToken(serviceDate)"),
        True,
        True,
    ),
    (("propertyToken(totalSpent)", ":", "{", "propertyToken(description)"), True, True),
    (
        ("propertyToken(description)", ":", "{", "propertyToken(description)"),
        True,
        True,
    ),
    (
        ("propertyToken(serviceRecords)", ":", "{", "propertyToken(description)"),
        True,
        True,
    ),
    (("dateToken", ":", "{", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(customerID)", ":", "{", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(totalSpent)", ":", "{", "propertyToken(vehicleID)"), True, True),
    (
        ("propertyToken(vehicleID)", ":", "{", "propertyToken(serviceRecords)"),
        True,
        True,
    ),
    (
        ("propertyToken(customerID)", ":", "{", "propertyToken(serviceRecords)"),
        True,
        True,
    ),
    (("propertyToken(customerID)", ":", "{", "propertyToken(customerID)"), True, True),
    (("propertyToken(totalSpent)", ":", "{", "propertyToken(customerID)"), True, True),
    (("propertyToken(vehicleID)", ":", "{", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(vehicleID)", ":", "{", "propertyToken(customerID)"), True, True),
    (("propertyToken(totalSpent)", ":", "{", "propertyToken(serviceDate)"), True, True),
    (("dateToken", ":", "{", "propertyToken(description)"), True, True),
    (("propertyToken(cost)", ":", "{", "propertyToken(customerID)"), True, True),
    (("propertyToken(description)", ":", "{", "dateToken"), True, True),
    (("propertyToken(vehicleID)", ":", "{", "propertyToken(description)"), True, True),
    (
        ("propertyToken(serviceRecords)", ":", "{", "propertyToken(vehicleID)"),
        True,
        True,
    ),
    (("propertyToken(cost)", ":", "{", "propertyToken(serviceRecords)"), True, True),
    (("dateToken", ":", "{", "propertyToken(cost)"), True, True),
    (("dateToken", ":", "{", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(customerID)", ":", "{", "dateToken"), True, True),
    (("propertyToken(customerID)", ":", "{", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(cost)", ":", "{", "propertyToken(vehicleID)"), True, True),
    (("propertyToken(customerID)", ":", "{", "propertyToken(description)"), True, True),
    (("propertyToken(vehicleID)", ":", "{", "propertyToken(cost)"), True, True),
    (("propertyToken(serviceRecords)", ":", "{", "propertyToken(cost)"), True, True),
    (("propertyToken(vehicleID)", ":", "{", "propertyToken(vehicleID)"), True, True),
    (
        ("propertyToken(serviceRecords)", ":", "{", "propertyToken(customerID)"),
        True,
        True,
    ),
    (("propertyToken(cost)", ":", "{", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(serviceDate)", ":", "{", "propertyToken(cost)"), True, True),
    (
        ("propertyToken(description)", ":", "{", "propertyToken(serviceRecords)"),
        True,
        True,
    ),
    (("propertyToken(totalSpent)", ":", "{", "dateToken"), True, True),
    (("propertyToken(totalSpent)", ":", "{", "propertyToken(cost)"), True, True),
    (
        ("propertyToken(serviceDate)", ":", "{", "propertyToken(serviceDate)"),
        True,
        True,
    ),
    (("propertyToken(serviceRecords)", ":", "{", "dateToken"), True, True),
    (("propertyToken(description)", ":", "{", "propertyToken(totalSpent)"), True, True),
    (("propertyToken(description)", ":", "{", "propertyToken(cost)"), True, True),
    (("dateToken", ":", "{", "propertyToken(serviceDate)"), True, True),
    (("propertyToken(cost)", ":", "{", "propertyToken(serviceDate)"), True, True),
    (
        ("propertyToken(serviceRecords)", ":", "{", "propertyToken(totalSpent)"),
        True,
        True,
    ),
    (
        ("propertyToken(description)", ":", "{", "propertyToken(serviceDate)"),
        True,
        True,
    ),
    (("propertyToken(customerID)", "]", "["), True, False),
    (("propertyToken(cost)", "]", "["), True, False),
    (("propertyToken(totalSpent)", "]", "["), True, False),
    (("propertyToken(serviceDate)", "]", "["), True, False),
    (("propertyToken(description)", "]", "["), True, False),
    (("dateToken", "]", "["), True, False),
    (("propertyToken(vehicleID)", "]", "["), True, False),
    (("propertyToken(serviceRecords)", "]", "["), True, False),
    ((":", ":", "{"), False, False),
    ((":", ":", ":", ":", ":", ":", ":", ":"), False, False),
    ((":", "]"), False, False),
    ((":", ","), False, False),
    (("}", ",", "{"), False, False),
    (("stringToken", "lexFence"), True, True),
    (("propertyToken(description)", "lexFence"), True, True),
    (("propertyToken(customerID)", "lexFence"), True, True),
    (("propertyToken(cost)", "lexFence"), True, True),
    (("propertyToken(vehicleID)", "lexFence"), True, True),
    (("dateToken", "lexFence"), True, True),
    (("propertyToken(serviceRecords)", "lexFence"), True, True),
    (("propertyToken(totalSpent)", "lexFence"), True, True),
    (("propertyToken(serviceDate)", "lexFence"), True, True),
    (("}", "boolNullToken"), False, True),
    (("{", "}", "stringToken"), False, True),
    (("propertyToken(customerID)", "["), True, False),
    (("propertyToken(totalSpent)", "["), True, False),
    (("propertyToken(description)", "["), True, False),
    (("propertyToken(serviceDate)", "["), True, False),
    (("dateToken", "["), True, False),
    (("propertyToken(vehicleID)", "["), True, False),
    (("propertyToken(serviceRecords)", "["), True, False),
    (("propertyToken(cost)", "["), True, False),
    (("stringToken", "]"), True, False),
    (("}", "{", "{"), False, False),
    (("}", "}", "{", "{"), False, False),
    (("{", "}", "{"), False, False),
    ((",", ":", ","), False, False),
    (("[", ":", "numberToken"), False, True),
    ((",", "boolNullToken"), False, False),
    (("stringToken", ":", "dateToken"), True, True),
    (("stringToken", ":", "propertyToken(serviceRecords)"), True, True),
    (("stringToken", ":", "propertyToken(serviceDate)"), True, True),
    (("stringToken", ":", "propertyToken(cost)"), True, True),
    (("stringToken", ":", "propertyToken(totalSpent)"), True, True),
    (("stringToken", ":", "propertyToken(customerID)"), True, True),
    (("stringToken", ":", "propertyToken(vehicleID)"), True, True),
    (("stringToken", ":", "propertyToken(description)"), True, True),
    (("propertyToken(cost)", ":", "["), True, False),
    (("propertyToken(customerID)", ":", "["), True, False),
    (("propertyToken(vehicleID)", ":", "["), True, False),
    (("dateToken", ":", "["), True, False),
    (("propertyToken(serviceRecords)", ":", "["), True, False),
    (("propertyToken(description)", ":", "["), True, False),
    (("propertyToken(totalSpent)", ":", "["), True, False),
    (("propertyToken(serviceDate)", ":", "["), True, False),
    (("]", "]", "["), False, False),
    ((":", ":", "numberToken"), False, True),
    (("}", "{", "}", "{"), False, False),
    (("stringToken", "{", "{"), True, False),
    ((",", ",", ",", ","), False, False),
    (("]", ":", ":"), False, False),
    (("numberToken", ","), True, False),
    ((",", "{"), False, False),
    ((":", "}"), False, False),
    (("lexFence", "lexFence", "lexFence"), True, True),
    (("lexFence", "lexFence"), False, False),
    (("]", "[", "]"), False, False),
    ((":", "boolNullToken"), False, False),
    (("lexFence", "boolNullToken"), True, True),
    (("}", "}", "}", "{"), False, False),
    (("]", "{", "}", "]", "{", "}"), False, False),
    (
        (
            ":",
            ":",
            ":",
            ":",
            ":",
            ":",
            ":",
            ":",
            ":",
            ":",
            ":",
            ":",
            ":",
            ":",
            ":",
            ":",
        ),
        False,
        False,
    ),
    ((":", "]", ","), False, False),
    (("numberToken", "boolNullToken"), True, True),
    (("propertyToken(serviceDate)", "stringToken"), True, True),
    (("propertyToken(serviceRecords)", "stringToken"), True, True),
    (("propertyToken(vehicleID)", "stringToken"), True, True),
    (("dateToken", "stringToken"), True, True),
    (("propertyToken(customerID)", "stringToken"), True, True),
    (("propertyToken(totalSpent)", "stringToken"), True, True),
    (("propertyToken(cost)", "stringToken"), True, True),
    (("propertyToken(description)", "stringToken"), True, True),
    (("}", "}", "numberToken"), False, True),
    (("]", "[", ":"), False, False),
    (("]", ",", "propertyToken(customerID)"), False, True),
    (("]", ",", "propertyToken(description)"), False, True),
    (("]", ",", "propertyToken(totalSpent)"), False, True),
    (("]", ",", "dateToken"), False, True),
    (("]", ",", "propertyToken(serviceDate)"), False, True),
    (("]", ",", "propertyToken(cost)"), False, True),
    (("]", ",", "propertyToken(vehicleID)"), False, True),
    (("]", ",", "propertyToken(serviceRecords)"), False, True),
    (("{", "["), False, False),
    ((",", "]"), False, False),
    ((",", ",", ","), False, False),
    (("]", "boolNullToken"), False, True),
    (("[", "stringToken", ","), False, False),
    ((",", ":", ",", ":"), False, False),
    ((":", ":", ":"), False, False),
    (("]", "lexFence"), False, True),
    (("propertyToken(customerID)", "}", "}"), True, False),
    (("propertyToken(serviceRecords)", "}", "}"), True, False),
    (("propertyToken(description)", "}", "}"), True, False),
    (("propertyToken(vehicleID)", "}", "}"), True, False),
    (("propertyToken(cost)", "}", "}"), True, False),
    (("propertyToken(serviceDate)", "}", "}"), True, False),
    (("propertyToken(totalSpent)", "}", "}"), True, False),
    (("dateToken", "}", "}"), True, False),
    (("propertyToken(serviceRecords)", "]", ":"), True, False),
    (("propertyToken(totalSpent)", "]", ":"), True, False),
    (("propertyToken(serviceDate)", "]", ":"), True, False),
    (("propertyToken(vehicleID)", "]", ":"), True, False),
    (("propertyToken(cost)", "]", ":"), True, False),
    (("propertyToken(description)", "]", ":"), True, False),
    (("propertyToken(customerID)", "]", ":"), True, False),
    (("dateToken", "]", ":"), True, False),
    ((",", "stringToken"), False, True),
    (("[", "[", "["), False, False),
    (("stringToken", "{", "}"), True, False),
]


def test_interleave_with_value():
    assert interleave_with_value([1, 2, 3], 0) == [1, 0, 2, 0, 3]
    assert interleave_with_value([], 0) == []
    assert interleave_with_value([1], 0) == [1]
    assert interleave_with_value([1, 2], 0) == [1, 0, 2]
    assert interleave_with_value([1, 2, 3, 4], -1) == [1, -1, 2, -1, 3, -1, 4]


def test_speedup_due_to_streak_caching():
    cfg_lang = r"""
        S -> NumberArray
        NumberArray -> [ ] | [ NumberArrayElements ]
        NumberArrayElements -> Number | NumberArrayElements , Number
        Number -> lexNumber
    """
    json_cfg = CFG.from_text(cfg_lang, "S")
    main_language_cfg = json_cfg.to_normal_form()
    lex_map = compile_lex_map(LEX_MAP, subtokens={})
    words = ["[" + ("2," * 100), None, None, None, (",2" * 100) + "]"]
    for i in range(100):
        words[2] = str(i)
        generated_fsa = generated_language(words, lex_map, json_cfg.get_terminals())
        assert not main_language_cfg.is_intersection_empty(generated_fsa, 100)


def test_gen_lang_consistency():
    words = [
        """
#include<stdio.h>
#include<vector>
#include<string>
using namespace std;
vector<string> words_string(string s){
    string current="";
    vector<string> out={};
    s=s+' ';
    for (auto e:s){
        if (e!=',') current=current+e;
        if (e==','""",
        None,
        """)>0)
        {
            out.push_back(current);
            current="";
        }
     }
     e""",
        None,
        """return out;
}

int main(){
// TODO
}""",
    ]
    grammar, lex_map, subtokens = cpp_grammar()
    lex_map = compile_lex_map(lex_map, subtokens=subtokens)
    first_lang = None
    res = []
    for _ in range(100):
        reset_lex_cache()
        gen_lang = generated_language(
            words,
            lex_map,
            grammar.get_terminals(),
            prelex="\x02\x03",
            subtokens=subtokens,
            supertokens=derive_supertokens(subtokens),
        )
        assert first_lang is None or gen_lang == first_lang
        first_lang = gen_lang
        res.append(not grammar.is_intersection_empty(gen_lang, 100))
    assert all(res), "Generated language is not empty for all iterations"


if __name__ == "__main__":
    test_schema_as_cfg_2()
