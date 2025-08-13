"""
Collects all datasets to be evaluated
"""

from constrained_diffusion.eval.mri.datasets.generic import DataSet
from constrained_diffusion.eval.mri.datasets.humaneval_mri_cpp import (
    HumanEvalMriDataSet,
)

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
register_dataset("HumanEval/MRI/cpp/1", HumanEvalMriDataSet(language="cpp", spans=1))
register_dataset("HumanEval/MRI/cpp/2", HumanEvalMriDataSet(language="cpp", spans=2))
register_dataset("HumanEval/MRI/cpp/3", HumanEvalMriDataSet(language="cpp", spans=3))
