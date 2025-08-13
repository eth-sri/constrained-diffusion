"""
Collects all datasets to be evaluated
"""

from constrained_diffusion.eval.dllm.datasets.cpp import CppDataSet
from constrained_diffusion.eval.dllm.datasets.generic import DataSet
from constrained_diffusion.eval.dllm.datasets.jsonschema import JsonSchemaDataSet
from constrained_diffusion.eval.dllm.datasets.smiles import SmilesDataSet

ALL_DATASETS: dict[str, DataSet] = {}


def register_dataset(name: str, ds: DataSet):
    """
    Registers a dataset to be evaluated.
    """
    if name in ALL_DATASETS:
        raise ValueError(f"Dataset {name} is already registered.")
    ALL_DATASETS[name] = ds


def load_dataset(name: str) -> DataSet:
    """
    Loads a dataset by name.
    """
    if name not in ALL_DATASETS:
        raise ValueError(f"Dataset {name} is not registered.")
    return ALL_DATASETS[name]


# ---- Register your dataset here
register_dataset("THUDM/humaneval-x/cpp", CppDataSet(split="test", subset="cpp"))
register_dataset("zai-org/humaneval-x/cpp", load_dataset("THUDM/humaneval-x/cpp"))
register_dataset("smiles", SmilesDataSet())
register_dataset("jsonschema", JsonSchemaDataSet())
