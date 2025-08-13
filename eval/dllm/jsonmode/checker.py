"""
Checker module for JSON mode task.
This module provides functions to check the syntax and schema conformance of JSON.
"""

import difflib
import json
from pathlib import Path

from datasets import load_dataset

from constrained_diffusion.cfgs.jsonschema import schema_to_cfg
from constrained_diffusion.constrain_utils import compile_lex_map, lex


JSON_DATASET = load_dataset("eth-sri/json-mode-eval-extended", split="test").to_iterable_dataset()

SCHEMAS = [x["schema"] for x in JSON_DATASET]
REF_SOLUTIONS = [x["output"] for x in JSON_DATASET]


def check_instance(output, timeout=None):
    """
    Check a single instance from a JSONL file.

    Args:
        line: A JSON string containing an instance to check

    Returns:
        A dictionary with the check results
    """
    task_id = int(output["instance_id"].split("_")[-1])

    schema = json.loads(SCHEMAS[task_id])
    schema_grammar, lex_map, subtokens = schema_to_cfg(schema)
    lex_map = compile_lex_map(lex_map, subtokens=subtokens)
    instance = output["extracted"]

    # Check schema conformance
    lexings = lex(instance, lex_map, is_first=True)
    accepted = any(
        schema_grammar.accepts(lexied[0])
        for lexied in lexings
        if not lexied[1] and not lexied[2]
    )
    compilable = accepted
    compiler_output = f"accepted {compilable}, lexings {lexings}"
    try:
        # Check if the instance matches the reference solution
        ref_solution = json.loads(REF_SOLUTIONS[task_id])
        ref_solution = json.dumps(ref_solution, indent=4)
        json_instance = json.dumps(json.loads(instance), indent=4)
        passes_test = ref_solution == json_instance
        diff = "\n".join(
            difflib.unified_diff(
                ref_solution.splitlines(keepends=True),
                json_instance.splitlines(keepends=True),
                fromfile="reference_solution",
                tofile="json_instance",
            )
        )
        compiler_output += f"""diff to reference solution: {diff}"""
    except json.JSONDecodeError as e:
        passes_test = False
        compiler_output += f"Error decoding JSON: {e}"

    return {
        "instance_id": output["instance_id"],
        "extracted": output["extracted"],
        "syntax_ok": compilable,
        "passed_tests": passes_test,  # For JSON, schema conformance is the test
        "compiler_output": compiler_output,
    }
