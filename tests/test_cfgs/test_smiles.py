import pytest
from constrained_diffusion.cfgs.smiles import smiles_schema
from constrained_diffusion.constrain_utils import (
    compile_lex_map,
    lex,
    generated_language,
)


_valid_smiles_positions = [
    "Nc1ccc(B2OC(C)(C)C(C)(C)O2)cc1",
    "n1ccccc1",
    "OC1CCCCC1",
    "O1CCOCC1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccc2ccccc2c1",
    "c1ccccc1/C=C/c2ccccc2",
    "C[C@H](O)CC",
    "C=C",
    "CCO",
    "C=O",
    "CO",
    "C",
    "C#C",
    "CCC",
    "NCCC",
    "CC(=O)O",
    "[C](c1ccccc1)c2ccccc2",
    "N[C@@H](C)C(=O)N[C@H](CO)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O",
    "[N+](CC)(CC)(CC)CC.[P-](F)(F)(F)(F)(F)F",
    "c1cc[nH+]cc1.c1cc[nH+]cc1.[O-][Cr](=O)(=O)O[Cr](=O)(=O)[O-]",
    "C[n+]1c2ccccc2ccc1.[I-]",
    "n1ccccc1",
    "OC1CCCCC1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",
    "c1ccc2ccccc2c1",
    "C[C@H](N)C(=O)O",
    "CN1C(=O)N(C)C2=C(C1=O)N(C)C=N2",
    "[2H]C([2H])([2H])[C@@H](N)[13C](=O)O",
    "O=C(O)/C=C/C=C\\CCCCC",
    "[cH-]1cccc1.[cH-]1cccc1.[Fe+2]",
    "[13CH3][13C](=O)O",
    #  '[P+](c1ccccc1)(c2ccccc2)(c3ccccc3)(c4ccccc4).[B-](c1ccccc1)(c2ccccc2)(c3ccccc3)(c4ccccc4)',
    "[NH3+][C@H](C(=O)[O-])CCCNC(=[NH2+])N",
    "OC1CCCCC1",
    "n1ccccc1",
    "CC1=CCCCC1",
    "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",
    "O=C(O)c1c(OC(=O)C)cccc1",
    "C[C@H](O)CC",
    "c1ccc2ccccc2c1",
]


def smiles_grammar_preprocessed():
    grammar, lex_map, subtokens = smiles_schema()
    lex_map = compile_lex_map(lex_map, subtokens=subtokens)
    grammar = grammar.to_normal_form()
    return grammar, lex_map


@pytest.mark.parametrize(
    "word",
    _valid_smiles_positions,
)
def test_smiles_grammar(word):
    """
    Test the FEN grammar
    """
    grammar, lex_map = smiles_grammar_preprocessed()
    lexed = lex(word, lex_map, is_first=True, strip_chars="")
    assert any(
        lexied[0] in grammar for lexied in lexed
    ), "Lexed word not in grammar\nWord:{}\nLexing:{}\n{}".format(word, '\n'.join(str(x) for x in lexed), grammar.to_text())


_valid_smiles_partial = [
    [
        "CC(=O)",
        None,
        "c1C(=",
    ]
]


@pytest.mark.parametrize(
    "tokens",
    _valid_smiles_partial,
)
def test_smiles_partial(tokens):
    """
    Test the fen grammar
    """
    grammar, lex_map = smiles_grammar_preprocessed()
    # print(grammar.to_text())
    assert not grammar.is_empty(), f"Grammar is empty\n{grammar.to_text()}"
    generated_lang = generated_language(
        tokens, lex_map, grammar.get_terminals(), strip_chars=""
    )
    assert not generated_lang.is_empty(), "Generated language is empty"
    empty = grammar.is_intersection_empty(generated_lang, 100)
    if empty:
        print("Intersection is empty")
    assert not empty, "intersection language is empty"


@pytest.mark.parametrize(
    "word",
    [
        "O1CCOCC1XX",
        "CC(=O(Oc1ccccc1C(=O)O",
        "c1ccccc1|C=C/c2ccccc2",
        "C[C@H](O)CDipx",
        "C>C",
        "CCO?",
        "CO+++++",
        "C##C",
        "CC[=O)O",
        "[C))(c1ccccc1)c2ccccc2",
        "N[C@@@H](C)C(=O)N[C@H](CO)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O",
        "[N+24](CC)(CC)(CC)CC.[P-](F)(F)(F)(F)(F)F",
        "c1cc[nH+]cc1.c1cc[nC25]cc1.[O-][Cr](=O)(=O)O[Cr](=O)(=O)[O-]",
        "C[n+]1c2ccccc2ccc1.[I----]",
        "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H)(O)CO",
        "O=C(O)/C=C/C=C//CCCCC",
        # "C(n(n1))O",
    ],
)
def test_invalid_smiles_grammar(word):
    """
    Test the TypeScript grammar for invalid cases
    """
    grammar, lex_map = smiles_grammar_preprocessed()
    lexed = lex(word, lex_map, is_first=True, strip_chars="")
    for lexied in lexed:
        # Check if the lexed word is not in the grammar
        assert (
            lexied[0] not in grammar
        ), f"Lexed word in grammar\nLexing:{lexied}\n{grammar.to_text()}"
