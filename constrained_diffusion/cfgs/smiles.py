from constrained_diffusion.constrain_utils import (
    collect_subtokens,
)
from rustformlang.cfg import CFG
from rustformlang.fa.bytes_dfa import regex_escape

# --- Lexical Tokens for SMILES ---

SMILES_LEX_MAP = {
    "digit": r"\d",
    "fifteen": r"1[0-5]",  # Matches digits 10-15
    "organicSymbol": "(B|Br|Cl|C|[NOPSFI]|[bcnops]|At|Ts)",
    "bond": r"(=|#|/|$|\\)",
    "anorganicSymbol": r"A[cglmru]|B[aehik]|C[adefmnorsu]|D[bsy]|E[rsu]|F[elmr]|G[ade]|H[efgos]|I[nr]|K[r]?|L[airuv]|M[cgnot]|N[abdehiop]|O[gs]|P[abdmortu]|R[abefghnu]|S[bcegimnr]|T[abcehilm]|[UVW]|Xe|Y[b]?|Z[nr]|se|as",
    "chiral": r"@|@@",
    "(": regex_escape("("),
    ")": regex_escape(")"),
    "[": regex_escape("["),
    "]": regex_escape("]"),
    "perc": regex_escape("%"),
    "h": "H",
    "+": regex_escape("+"),
    "-": regex_escape("-"),
    ":": regex_escape(":"),
    ".": regex_escape("."),
}


SMILES_grammar = """
S -> Line

Line -> Atom OptComboChainBranchList

OptComboChainBranchList -> ComboChainBranchList | $

ComboChainBranchList -> ComboChainBranchElement | ComboChainBranchElement ComboChainBranchList

ComboChainBranchElement -> Chain | Branch

Chain -> . Atom | OptBond ComboAtomRnumList

OptBond -> Bond | $

ComboAtomRnumList -> ComboAtomRnumElement | ComboAtomRnumElement ComboAtomRnumList

ComboAtomRnumElement -> Atom | Rnum

Bond -> - | bond

Branch -> ( OptBondOrDotLineList ) 

OptBondOrDotLineList -> OptBondOrDotLineElement | OptBondOrDotLineElement OptBondOrDotLineList

OptBondOrDotLineElement -> OptBondOrDot Line

OptBondOrDot -> Bond | . | $

Atom -> organicSymbol | BracketAtom

BracketAtom -> [ OptionalIsotope Symbol OptionalChiral OptionalHCount OptionalCharge OptionalMap ]


OptionalIsotope -> Isotope | $

OptionalChiral -> chiral | $

OptionalHCount -> HCount | $

OptionalCharge -> Charge | $

OptionalMap -> Map | $

Rnum -> digit | perc digit digit

Isotope -> digit | digit digit | digit digit digit

HCount -> h digit | h

Charge -> + | + + | + fifteen | + digit | - | - - | - fifteen | - digit

Map -> : Isotope

Symbol -> organicSymbol | anorganicSymbol | h

"""


def smiles_schema():
    """
    Returns the grammar for a subset of TypeScript
    """
    c = CFG.from_text(SMILES_grammar, "S")
    lex_map = SMILES_LEX_MAP.copy()
    c, lex_map, subtokens = collect_subtokens(c, lex_map)

    return c, lex_map, subtokens
