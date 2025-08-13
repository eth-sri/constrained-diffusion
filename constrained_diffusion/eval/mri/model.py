"""
Collects all models to be evaluated
"""

from constrained_diffusion.eval.mri.models.fim import (
    DEEPSEEK_CODER_7B_MODEL,
    STARCODER_2_7B_MODEL,
    FimModel,
    CODELLAMA_7B_MODEL,
    CODEGEMMA_7B_MODEL,
    DEEPSEEK_CODER_1B_MODEL,
    DEEPSEEK_CODER_33B_MODEL,
)
from constrained_diffusion.eval.mri.models.generic import Model

ALL_MODELS: dict[str, Model] = {}


def register_model(name: str, ds: Model):
    """
    Registers a dataset to be evaluated.
    """
    if name in ALL_MODELS:
        raise ValueError(f"Model {name} is already registered.")
    ALL_MODELS[name] = ds


def load_model(name: str) -> Model:
    """
    Loads a dataset by name.
    """
    if name not in ALL_MODELS:
        raise ValueError(f"Model {name} is not registered.")
    return ALL_MODELS[name]


# ---- Register your dataset here
register_model(DEEPSEEK_CODER_1B_MODEL, FimModel(DEEPSEEK_CODER_1B_MODEL))
register_model(DEEPSEEK_CODER_7B_MODEL, FimModel(DEEPSEEK_CODER_7B_MODEL))
register_model(DEEPSEEK_CODER_33B_MODEL, FimModel(DEEPSEEK_CODER_33B_MODEL))
register_model(CODEGEMMA_7B_MODEL, FimModel(CODEGEMMA_7B_MODEL))
register_model(STARCODER_2_7B_MODEL, FimModel(STARCODER_2_7B_MODEL))
register_model(CODELLAMA_7B_MODEL, FimModel(CODELLAMA_7B_MODEL))
