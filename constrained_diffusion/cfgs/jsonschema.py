"""
Tools to convert json schema to a cfg and the tokens
"""

import sys

import hashlib

from range_ex import range_regex

from constrained_diffusion.constrain_utils import collect_subtokens, LexMap
from rustformlang.cfg import CFG
from typing import Tuple, Dict
import json

from rustformlang.fa.bytes_dfa import regex_to_dfa, regex_escape

sys.setrecursionlimit(5000)

json_string_pattern = r'(""|"[^"\r\n`]*[^`"\r\n\\]")'
json_string_pattern_compiled = regex_to_dfa(json_string_pattern)


def schema_to_cfg(schema: dict) -> Tuple[CFG, LexMap, dict[str, set[str]]]:
    if len(schema) == 1 and "type" not in schema and "oneOf" not in schema:
        # if the schema contains only a single key it's technically "true" but this is very unintuitive
        # instead, parse as the subschema of that only key
        raise NotImplementedError(
            "Invalid json schema, implicitly any, check the schema (maybe put key instead of title?)"
        )
    grammar, lex_map, sub_map = _schema_to_cfg(schema)
    # substitute the sub grammars
    grammar = grammar.substitute(sub_map)
    # finally, make sure that strings and numbers etc are lexed unambiguously
    grammar, lex_map, subtokens = collect_subtokens(grammar, lex_map)
    return grammar, lex_map, subtokens


def _schema_to_cfg(schema: dict | str) -> Tuple[CFG, LexMap, Dict[str, CFG]]:
    if isinstance(schema, str):
        schema = {"type": schema}
    if isinstance(schema, bool):
        if schema:
            return anything_to_cfg()
        else:
            # empty grammar, no value is allowed
            return CFG(), {}, {}
    if not isinstance(schema, dict):
        raise NotImplementedError(
            "Schema must be a dict, boolean or a string, got: "
            + str(type(schema))
            + "("
            + str(schema)
            + ")"
        )
    if "oneOf" in schema:
        # duplicate the objects, overwrite the keys
        duplicated = []
        for raw_subschema in schema["oneOf"]:
            subschema = schema.copy()
            del subschema["oneOf"]
            subschema.update(raw_subschema)
            duplicated.append(subschema)
        return one_of_to_cfg({"oneOf": duplicated})
    if "const" in schema:
        # create a grammar for the const value
        s = hashlib.sha256(schema["const"].encode()).hexdigest()
        return (
            CFG.from_text(f"S -> {s}constToken", "S"),
            {f"{s}constToken": regex_escape(json.dumps(schema["const"]))},
            {},
        )
    if not schema.get("type"):
        return anything_to_cfg()
    schema_type = schema["type"]
    if schema_type == "object":
        return object_to_cfg(schema)
    elif schema_type == "string":
        return string_to_cfg(schema)
    elif schema_type == "boolean":
        return boolean_to_cfg()
    elif schema_type == "integer":
        return int_to_cfg(schema)
    elif schema_type == "array":
        return array_to_cfg(schema)
    elif schema_type == "number":
        return number_to_cfg(schema)
    elif schema_type == "null":
        return CFG.from_text("S -> nullToken", "S"), {"nullToken": "null"}, {}
    else:
        raise NotImplementedError("Schema type not supported: " + schema_type)


JSON_LEX_TOKENS = {"{", "}", ",", "[", "]", ":"}
JSON_LEX_MAP = {
    # NOTE: need to align with other tokens to avoid duplication
    "stringToken": json_string_pattern_compiled,
    "numberToken": regex_to_dfa(r"-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?"),
    "boolNullToken": regex_to_dfa(r"true|false|null"),
    **{token: regex_escape(token) for token in JSON_LEX_TOKENS},
}
JSON_CFG = CFG.from_text(
    """
S -> Element
Element -> Value
Value -> Object | Array | String | Number | BooleanOrNull
Object -> { Members } | { }
Members -> Pair | Pair , Members
Pair -> String : Element
Array -> [ Elements ] | [ ]
Elements -> Element | Element , Elements
String -> stringToken
Number -> numberToken
BooleanOrNull -> boolNullToken
""",
    "S",
)


def anything_to_cfg() -> Tuple[CFG, LexMap, Dict[str, CFG]]:
    return (
        CFG.from_text("S -> jsonToken", "S"),
        JSON_LEX_MAP.copy(),
        {"jsonToken": JSON_CFG},
    )


def one_of_to_cfg(schema: dict) -> Tuple[CFG, LexMap, Dict[str, CFG]]:
    """
    Turn a json schema oneOf description into a grammer

    Returns a grammar for that oneOf and a lexing map
    """
    # create rules for all the properties
    lexing = {}
    terminals = []
    substitutions = {}
    sub_map = {}
    for i, subschema in enumerate(schema["oneOf"]):
        subgrammar, sublexing, submap = _schema_to_cfg(subschema)
        lexing.update(sublexing)
        sub_map.update(submap)
        terminals.append(f"oneOfToken{i}")
        substitutions[f"oneOfToken{i}"] = subgrammar
    cfg = CFG.from_text("S -> " + " | ".join(terminals), "S")
    cfg = cfg.substitute(substitutions)
    return cfg, lexing, sub_map


def array_to_cfg(schema: dict) -> Tuple[CFG, LexMap, Dict[str, CFG]]:
    """
    Turn a json schema array description into a grammer

    Returns a grammar for that array and a lexing map
    """
    # create rules for all the properties
    lexing = {}
    for tk in ("[", "]", ","):
        lexing[tk] = regex_escape(tk)

    productions = [
        "S -> [ ElementList ] | [ ]",
        "ElementList -> Element | ElementList , Element ",
        "Element -> arraySubToken",
    ]
    # add optional production rules
    cfg = CFG.from_text("\n".join(productions), "S")

    items = schema.get("items", {})
    subgrammar, sublexing, sub_map = _schema_to_cfg(items)
    lexing.update(sublexing)
    grammar_map = {"arraySubToken": subgrammar}
    cfg = cfg.substitute(grammar_map)
    return cfg, lexing, sub_map


def boolean_to_cfg() -> Tuple[CFG, LexMap, Dict[str, CFG]]:
    return (
        CFG.from_text("S -> boolToken", "S"),
        {
            "boolToken": "true|false",
        },
        {},
    )


def int_to_cfg(schema: dict) -> Tuple[CFG, LexMap, Dict[str, CFG]]:
    mini, maxi = schema.get("minimum"), schema.get("maximum")
    int_regex = range_regex(mini, maxi)
    return (
        CFG.from_text(f"S -> intToken({mini},{maxi})", "S"),
        {f"intToken({mini},{maxi})": int_regex},
        {},
    )


def number_to_cfg(schema: dict) -> Tuple[CFG, LexMap, Dict[str, CFG]]:
    number_regex = JSON_LEX_MAP["numberToken"]
    return CFG.from_text("S -> numberToken", "S"), {"numberToken": number_regex}, {}


def with_leading_zeros(min: int, max: int) -> str:
    assert min < 10 and max < 100, "Only two digits are supported, single digit minimum"
    return f"(0{range_regex(min, 9)}|{range_regex(10, max)})"


month_r = with_leading_zeros(1, 12)
day_r = with_leading_zeros(1, 31)
hour_r = with_leading_zeros(0, 23)
minute_r = with_leading_zeros(0, 59)
second_r = with_leading_zeros(0, 59)

STRING_LEXING = {
    "string": json_string_pattern_compiled,
    "date": rf'"\d{{4}}-{month_r}-{day_r}"',
    "email": r'"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9._-]+\.[a-zA-Z]{2,}"',
    "date-time": rf'"\d{{4}}-{month_r}-{day_r}T{hour_r}:{minute_r}:{second_r}(\.\d+)?(Z|[+-]{hour_r}:{second_r})"',
}
STRING_LEXING = {
    k: regex_to_dfa(v) if isinstance(v, str) else v for k, v in STRING_LEXING.items()
}


def string_to_cfg(schema: dict) -> Tuple[CFG, LexMap, Dict[str, CFG]]:
    if "enum" in schema:
        # create a grammar for the enum value
        pattern = f'"({"|".join(regex_escape(e) for e in schema["enum"])})"'
        tokenName = f"{hashlib.sha256(pattern.encode()).hexdigest()}enumToken"
        return (
            CFG.from_text(f"S -> {tokenName}", "S"),
            {tokenName: pattern},
            {},
        )
    if "minLength" in schema or "maxLength" in schema:
        min_length = schema.get("minLength", None)
        max_length = schema.get("maxLength", None)
        pattern = rf'"[^\n\r"]{{{min_length if min_length is None else ""},{max_length if max_length is None else ""}}}"'
        return (
            CFG.from_text(f"S -> stringToken({min_length},{max_length})", "S"),
            {f"stringToken({min_length},{max_length})": pattern},
            {},
        )
    if "pattern" in schema:
        pattern: str = schema["pattern"]
        pattern_token_name = (
            "patternToken" + hashlib.sha256(pattern.encode()).hexdigest()
        )
        lexing = {
            pattern_token_name: regex_to_dfa(
                rf'"{pattern.removeprefix("^").removesuffix("$")}"'
            )
            .intersection(json_string_pattern_compiled)
            .minimize()
        }
        return CFG.from_text(f"S -> {pattern_token_name}", "S"), lexing, {}
    format = schema.get("format", "string")
    return (
        CFG.from_text(f"S -> {format}Token", "S"),
        {f"{format}Token": STRING_LEXING[format]},
        {},
    )


def object_to_cfg(schema: dict) -> Tuple[CFG, LexMap, Dict[str, CFG]]:
    """
    Turn a json schema object description into a grammer

    Returns a grammar for that object and a lexing map
    """
    # create rules for all the properties
    lexing = {}
    for tk in (",", "}", "{", ":"):
        lexing[tk] = regex_escape(tk)
    sub_map = {}
    grammar_map = {}
    properties = schema.get("properties", {})
    all_properties = list(properties.keys())
    required_properties = schema.get("required", [])
    # first force in order of required
    production = ["{"]
    if required_properties:
        for property_name in required_properties:
            subschema = properties[property_name]
            property_token_name = f"propertyToken({property_name})"
            property_grammar_name = f"grammarToken({property_name})"

            lexing[property_token_name] = regex_escape(f'"{property_name}"')
            production.extend(
                (
                    property_token_name,
                    ":",
                    property_grammar_name,
                    ",",
                )
            )
            subgrammar, sublexing, submap = _schema_to_cfg(subschema)
            grammar_map[property_grammar_name] = subgrammar
            lexing.update(sublexing)
            sub_map.update(submap)
        production.pop()
        production.append("OptProperties")
        production.append("}")
        productions = ["S -> " + " ".join(production)]
    else:
        productions = [
            "S -> { OptProperty OptProperties }",
        ]
    # add optional properties
    productions.append("OptProperties -> $")
    productions.append("OptProperties -> , OptProperty OptProperties")
    # fixed properties
    for property_name in set(all_properties).difference(required_properties):
        subschema = properties[property_name]
        property_token_name = f"propertyToken({property_name})"
        property_grammar_name = f"grammarToken({property_name})"

        lexing[property_token_name] = regex_escape(f'"{property_name}"')
        subgrammar, sublexing, submap = _schema_to_cfg(subschema)
        grammar_map[property_grammar_name] = subgrammar
        lexing.update(sublexing)
        sub_map.update(submap)
        productions.append(
            f"OptProperty -> {property_token_name} : {property_grammar_name}"
        )
    # patternProperties
    for property_pattern, subschema in schema.get("patternProperties", {}).items():
        property_pattern = property_pattern.removeprefix("^").removesuffix("$")
        pattern_token_name = f"patternPropertyToken({property_pattern})"
        pattern_grammar_name = f"grammarToken({property_pattern})"

        lexing[pattern_token_name] = (
            regex_to_dfa(f'"{property_pattern}"')
            .intersection(json_string_pattern_compiled)
            .minimize()
        )
        subgrammar, sublexing, submap = _schema_to_cfg(subschema)
        grammar_map[pattern_grammar_name] = subgrammar
        lexing.update(sublexing)
        sub_map.update(submap)

        productions.append(
            f"OptProperty -> {pattern_token_name} : {pattern_grammar_name}"
        )
    # additional Properties
    if schema.get("additionalProperties", True):
        # TODO: the key here should exclude any of the required, optional and patternproperties
        subgrammar_string, sublexing_string, submap_string = _schema_to_cfg(
            {"type": "string"}
        )
        subgrammar_anything, sublexing_anything, submap_anything = _schema_to_cfg({})
        lexing.update(sublexing_string)
        lexing.update(sublexing_anything)
        sub_map.update(submap_string)
        sub_map.update(submap_anything)
        productions.append("OptProperty -> stringToken : anyGrammarToken")
        grammar_map["anyGrammarToken"] = subgrammar_anything
        lexing["stringToken"] = JSON_LEX_MAP["stringToken"]
    cfg = CFG.from_text("\n".join(productions), "S")
    cfg = cfg.substitute(grammar_map)
    return cfg, lexing, sub_map


if __name__ == "__main__":
    cfg, lex_map, subtokens = schema_to_cfg(
        {
            "CustomerVehicleServiceHistory": {
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
                                "description": {
                                    "title": "Description",
                                    "type": "string",
                                },
                                "cost": {"title": "Cost", "type": "number"},
                            },
                            "required": ["serviceDate", "description", "cost"],
                        },
                    },
                    "totalSpent": {"title": "Total Spent", "type": "number"},
                },
                "required": ["customerID", "vehicleID", "serviceRecords", "totalSpent"],
            }
        }
    )
    print(cfg.to_text())
