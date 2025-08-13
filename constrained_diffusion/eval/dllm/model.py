"""
Collects all models to be evaluated
"""

from constrained_diffusion.eval.dllm.models.dream.model import (
    DreamModel,
    MODEL_NAME as DREAM_MODEL_NAME,
    MODEL_NAME_DIFFU_CODER as DIFFU_CODER_MODEL_NAME,
    MODEL_NAME_DREAM_CODE as DREAM_CODER_MODEL_NAME,
)
from constrained_diffusion.eval.dllm.models.generic import Model
from constrained_diffusion.eval.dllm.models.llada.model import (
    LLaDAModel,
    MODEL_NAME as LLADA_MODEL_NAME,
)

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
register_model(LLADA_MODEL_NAME, LLaDAModel())
register_model(DREAM_MODEL_NAME, DreamModel(DREAM_MODEL_NAME))
register_model(DIFFU_CODER_MODEL_NAME, DreamModel(DIFFU_CODER_MODEL_NAME))
register_model(DREAM_CODER_MODEL_NAME, DreamModel(DREAM_CODER_MODEL_NAME))
