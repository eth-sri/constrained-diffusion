"""
Checker module for SMILES task.
This module provides functions to check the syntax and validity of SMILES strings.
"""

import json
import partialsmiles
from datasets import load_dataset
from rdkit import Chem
from pathlib import Path
from constrained_diffusion.cfgs.smiles import smiles_schema
from constrained_diffusion.constrain_utils import lex, compile_lex_map

# Load the SMILES dataset
SMILES_DATASET = load_dataset("eth-sri/smiles-eval", split="test").to_iterable_dataset()
SMILES_DATA = {x["instance_id"]: x for x in SMILES_DATASET}

def smiles_valid(output: str, lex_map, grammar):
    """Check if a SMILES string is valid using RDKit."""
    # Check schema conformance
    lexings = lex(output, lex_map, is_first=True)
    accepted = any(
        grammar.accepts(lexied[0])
        for lexied in lexings
        if not lexied[1] and not lexied[2]
    )
    return accepted


def are_smiles_equivalent(smiles_a, smiles_b, isomeric=True):
    """
    Checks if two SMILES strings represent the same molecule.

    Args:
        smiles_a (str): The first SMILES string.
        smiles_b (str): The second SMILES string.
        isomeric (bool): If True (default), considers stereochemistry.
                         If False, ignores stereochemistry.

    Returns:
        bool: True if the SMILES strings are equivalent, False otherwise.
              Returns False if either SMILES string is invalid.
    """
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)

    if mol_a is None or mol_b is None:
        # One or both SMILES are invalid
        return False, True

    canon_smiles_a = Chem.MolToSmiles(mol_a, isomericSmiles=isomeric)
    canon_smiles_b = Chem.MolToSmiles(mol_b, isomericSmiles=isomeric)

    return canon_smiles_a == canon_smiles_b, False

grammar, lex_map, subtokens = smiles_schema()
lex_map = compile_lex_map(lex_map, subtokens)


def check_instance(output, timeout=None):
    """
    Check a single instance from a JSONL file.

    Args:
        line: A JSON string containing an instance to check

    Returns:
        A dictionary with the check results
    """
    molecule = output["extracted"].strip()
    actually_valid = smiles_valid(molecule, lex_map, grammar)

    # Get task ID from instance_id
    task_id = int(output["instance_id"].split("_")[-1])

    try:
        partialsmiles.ParseSmiles(molecule, partial=False)
    except partialsmiles.SMILESSyntaxError as e:
        compiler_output = f"Invalid Syntax: {e}"
    except partialsmiles.ValenceError as e:
        compiler_output = f"Invalid Valence: {e}"
    except partialsmiles.KekulizationFailure as e:
        compiler_output = f"Kekulization Failure: {e}"
    else:
        compiler_output = "" if actually_valid else "Otherwise Invalid SMILES"

    # Load reference SMILES from dataset
    passed_tests = False
    if actually_valid:
        try:
            if task_id < len(SMILES_DATA):
                reference = SMILES_DATA[task_id]["output"].strip()
                if reference:
                    equivalent, error = are_smiles_equivalent(molecule, reference)
                    passed_tests = equivalent and not error
                    if not passed_tests and not error:
                        compiler_output = f"SMILES is valid but does not match reference.\nExtracted: {molecule}\nReference: {reference}"
                else:
                    compiler_output = (
                        f"Reference SMILES for task ID {task_id} is empty."
                    )
            else:
                compiler_output = f"Task ID {task_id} out of range for SMILES dataset."
        except Exception as e:
            compiler_output = f"Error loading reference SMILES: {e}"

    return {
        "instance_id": output["instance_id"],
        "extracted": output["extracted"],
        "syntax_ok": actually_valid,  # For SMILES, syntax and validity are the same
        "passed_tests": passed_tests,  # Now using functional test with reference from dataset
        "compiler_output": compiler_output,
    }
