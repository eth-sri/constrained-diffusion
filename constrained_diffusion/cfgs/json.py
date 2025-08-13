"""
Tools to convert json schema to a cfg and the tokens
"""

from constrained_diffusion.constrain_utils import collect_subtokens

from constrained_diffusion.cfgs.jsonschema import JSON_CFG, JSON_LEX_MAP


def json_grammar():
    cfg = JSON_CFG
    lex_map = JSON_LEX_MAP.copy()
    cfg, lex_map, subtokens = collect_subtokens(cfg, lex_map)
    return cfg, lex_map, subtokens


if __name__ == "__main__":
    cfg, lex_map, subtokens = json_grammar()
    print(cfg.to_text())
