"""
// Adapted from: https://www.cs.tufts.edu/~sguyer/classes/comp11-2011s/grammar.php
"""

import functools
import json


from datasets import load_dataset

from constrained_diffusion.constrain_utils import (
    compile_lex_map,
    collect_subtokens,
    LexMap,
)
from rustformlang.cfg import CFG
from rustformlang.fa.bytes_dfa import regex_to_dfa, regex_escape
from rustformlang.fa.epsilon_nfa import ENFA

CPP_dec_number = r"\x02(0|[1-9]\d*)\x03"
CPP_decimal_part = r"\.\x02\d+\x03"
CPP_exp = r"\x02[eE](\x03[+-]\x02)?\d+\x03"
CPP_float_number = rf"({CPP_dec_number}({CPP_decimal_part})?({CPP_exp})?|{CPP_decimal_part}|{CPP_decimal_part[:-4]}{CPP_exp[4:]}|{CPP_dec_number[:-4]}{CPP_exp[4:]})"
CPP_hex_number = r"0x[\da-fA-F]*"
CPP_oct_number = r"0o[0-7]*"
CPP_bin_number = r"0b[0-1]*"

CPP_lex_map_of_keywords_like_identifiers = {
    "for": r"for",
    "if": r"if",
    "while": r"while",
    "return": r"return",
    "else": r"else",
    "break": r"break",
    "continue": r"continue",
    "include": r"include",
    "using": r"using",
    "namespace": r"namespace",
    "binop_word": r"(and|or|xor)",
    "unop_word": r"not",
    "typedef": r"typedef",
    "switch": r"switch",
    "case": r"case",
    "default": r"default",
    "const": r"const",
}

CPP_identifier = r"[a-zA-Z_]\w*"
_CPP_lex_map: LexMap = {
    **{
        key: regex_escape(key)
        for key in (
            "#",
            "{",
            "}",
            "[",
            "]",
            "(",
            ")",
            ",",
            "=",
            ";",
            ":",
            "<",
            ">",
            "-",
            "!",
            ".",
            "+",
            "-",
            "&",
            "*",
            "/",
            "?",
        )
    },
    "string": r'"([^\n"\\`]*(\\[^\n\\`][^\n"\\`]*)*)"',
    "char": r"'\x02?([^\\']|\\\x02?[\\'tnvfrba]\x03?)\x03?'",
    "binop": r"(\+|-|\*|/|%|<|>|&|\||=|\^)",
    "commentSingleline": r"//([^\n`/]|\x02[^\n`/]+\x03)+\n",  # exclude backticks and symbols that start/end multiline comments
    "commentMultiline": r"/\*([^\n`/]|\x02[^\n`/]+\x03|\n)+\*/",
    # Note for prelexing we wrap numbers
    "floatNumber": CPP_float_number,
    "h": r"\x02(h|hpp)\x03",
    "long": r"\x02(long|short|signed|unsigned)\x03",
    "bar": r"\|",
}

_CPP_grammar_main = """
Includes -> $ | Include Includes
Include -> # include < Name > | # include  < IncludePath . h > | using namespace Name ; | Comment | typedef Type Name ;
IncludePath -> Name | Name / IncludePath

Head -> Type Name ( OptParams ) 
OptParams -> $ | Params
Params -> Param | Param , Params
Param -> Type Name
Type -> Name | long Name | const Type | Type < TypeList > | Type : : Type  | Type * | Type &
TypeList -> Type | Type , TypeList
FunctionDef -> Head GroupedStatement
Body -> $ | BodyContent Body
BodyContent -> Statement | VariableDef | commentSingleline
Comment -> commentSingleline | commentMultiline

VariableDef -> Type InitList ;
InitList -> InitElem | InitElem , InitList
InitElem -> Name | Name = Expression | Name ( OptArgs ) | Name [ Expression ] | Name [ Expression ] = Expression
Statement -> ExpressionStmt | IfStatement | ForStatement | WhileStatement | ReturnStatement | ContinueStatement | BreakStatement | SwitchStatement
ExpressionStmt -> OptExpression ;
StatementOrGroupedStatement -> Statement | GroupedStatement
OptExpression -> $ | Expression
GroupedStatement -> { Body }
ContinueStatement -> continue ;
BreakStatement -> break ;
ReturnStatement -> return Expression ; | return ;
IfStatement -> if ( Expression ) StatementOrGroupedStatement ElsePart
ElsePart -> $ | else StatementOrGroupedStatement
ForStatement -> for ( ExpressionStmt OptExpression ; OptExpression ) StatementOrGroupedStatement
            | for ( VariableDef OptExpression ; OptExpression ) StatementOrGroupedStatement
            | for ( Type InitElem : Expression ) StatementOrGroupedStatement
WhileStatement -> while ( Expression ) StatementOrGroupedStatement
SwitchStatement -> switch ( Expression ) { CaseList OptDefault }
CaseList -> Case | Case CaseList
Case -> case Expression : Body 
OptDefault -> $ | default : Body

Expression -> Name | Literal | FunctionCall | MemberAccess | Binop | ArrayAccess | GroupedExpression | Unop | TypeCast | PointerMemberAccess | TernaryExpression | LambdaExpression
Unop -> - Expression | ! Expression | - - Expression | + + Expression | Expression + + | Expression - - | unop_word Expression | * Expression | & Expression
MemberAccess -> Expression . Name 
PointerMemberAccess -> Expression - > Name
FunctionCall -> Expression ( OptArgs ) | Expression < TypeList > ( OptArgs )
TypeCast -> ( Type ) Expression
GroupedExpression -> ( Expression )
ArrayAccess -> Expression [ Expression ]
OptArgs -> $ | Args
Args -> Expression | Expression , Args
Binop -> Expression binop Expression | Expression binop_word Expression | Expression ! = Expression | Expression binop = Expression | Expression > > Expression | Expression < < Expression | Expression > > = Expression | Expression < < = Expression | Expression & & Expression | Expression bar bar Expression
TernaryExpression -> Expression ? Expression : Expression
LambdaExpression -> [ OptCaptureList ] ( OptParams ) { Body } | [ OptCaptureList ] { Body }
OptCaptureList -> $ | FullCaptureList
FullCaptureList -> & , OptCopyCaptureList | = , OptRefCaptureList | CaptureList
OptCopyCaptureList -> $ | CopyCaptureList
CopyCaptureList -> CopyCapture | CopyCapture , CopyCaptureList
OptRefCaptureList -> $ | RefCaptureList
RefCaptureList -> RefCapture | RefCapture , RefCaptureList
CaptureList -> Capture | Capture , CaptureList
Capture -> RefCapture | CopyCapture
CopyCapture -> Name
RefCapture -> & Name

Literal -> StringLiteral | CharLiteral | Number | VectorLiteral
Number -> floatNumber
StringLiteral -> string
CharLiteral -> char
VectorLiteral -> { OptExpressionList }
OptExpressionList -> $ | ExpressionList
ExpressionList -> Expression | Expression , ExpressionList

Name -> identifier | identifier : : Name | : : Name
"""

_CPP_grammar = (
    """
S -> Includes Program
TLD -> Comment | FunctionDef
Program -> TLD | TLD Program
"""
    + _CPP_grammar_main
)

CPP_grammar_one_fun = (
    """
    S -> Includes Comments FunctionDef
    Comments -> $ | Comment Comments
    """
    + _CPP_grammar_main
)

CPP_grammar_two_fun = (
    """
S -> Includes Comments FunctionDef FunctionDef
Comments -> $ | Comment Comments
"""
    + _CPP_grammar_main
)


def CPP_grammar(grammar=_CPP_grammar):
    # merge lines starting with "|" into the preceding line
    new_grammar = []
    for line in grammar.splitlines():
        line = line.strip()
        if line.lstrip().startswith("|"):
            new_grammar[-1] += " " + line
        else:
            new_grammar.append(line)
    new_grammar = "\n".join(new_grammar)
    return new_grammar


def CPP_lex_map():
    # modify the lex map
    lex_map = _CPP_lex_map.copy()

    # NOTE: to avoid the lexer splitting identifiers halfway, we wrap them in quotes
    # A pre-lexer will be used to insert these "quotes"
    identifier_regex = regex_to_dfa(rf"\x02{CPP_identifier}\x03")

    big_union = ENFA()
    # adding the keywords
    for key, value in CPP_lex_map_of_keywords_like_identifiers.items():
        regex_dfa = regex_to_dfa(rf"\x02{value}\x03")
        # Add the keywords to the lex map
        lex_map[key] = regex_dfa
        # Add the keywords to the big union
        big_union = big_union.union(regex_dfa.to_epsilon_automaton())
    # remove the keywords from identifier regex (identifiers can never be keywords)
    identifier_regex = identifier_regex.difference(
        big_union.minimize().to_bytes_dfa()
    ).minimize()

    # Add the identifier regex to the lex map
    lex_map["identifier"] = identifier_regex

    return lex_map


def cpp_grammar(grammar=_CPP_grammar):
    """
    Returns the grammar for a subset of Cpp
    """
    c = CFG.from_text(CPP_grammar(grammar), "S")
    lex_map = CPP_lex_map()
    c, lex_map, subtokens = collect_subtokens(c, lex_map)

    return c, lex_map, subtokens


@functools.lru_cache(maxsize=None)
def cpp_grammar_preprocessed(grammar=_CPP_grammar):
    grammar, lex_map, subtokens = cpp_grammar(grammar)
    lex_map = compile_lex_map(lex_map, subtokens=subtokens)
    grammar = grammar.to_normal_form()
    return grammar, lex_map


if __name__ == "__main__":
    ds = load_dataset("THUDM/humaneval-x", "cpp", split="test")
    print(f"Dataset size: {len(ds)}")
    for instance in ds:
        if "struct" in json.dumps(instance):
            for k, v in instance.items():
                print(f"{k}: {v}")
