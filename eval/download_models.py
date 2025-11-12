import argparse
import subprocess


from constrained_diffusion.eval.dllm.model import (
    LLADA_MODEL_NAME,
    DIFFU_CODER_MODEL_NAME,
    DREAM_CODER_MODEL_NAME,
    DREAM_MODEL_NAME,
)
from constrained_diffusion.eval.mri.model import (
    CODEGEMMA_7B_MODEL,
    STARCODER_2_7B_MODEL,
    DEEPSEEK_CODER_1B_MODEL,
    DEEPSEEK_CODER_7B_MODEL,
    DEEPSEEK_CODER_33B_MODEL,
)

MRI_MODELS = [
    CODEGEMMA_7B_MODEL,
    STARCODER_2_7B_MODEL,
    DEEPSEEK_CODER_1B_MODEL,
    DEEPSEEK_CODER_7B_MODEL,
    DEEPSEEK_CODER_33B_MODEL,
]

DLLM_MODELS = [
    LLADA_MODEL_NAME,
    DIFFU_CODER_MODEL_NAME,
    DREAM_CODER_MODEL_NAME,
    DREAM_MODEL_NAME,
]


def main():
    # Default models
    default_models = [
        LLADA_MODEL_NAME,
        DIFFU_CODER_MODEL_NAME,
        DREAM_CODER_MODEL_NAME,
        DREAM_MODEL_NAME,
        CODEGEMMA_7B_MODEL,
        STARCODER_2_7B_MODEL,
        DEEPSEEK_CODER_1B_MODEL,
        DEEPSEEK_CODER_7B_MODEL,
        DEEPSEEK_CODER_33B_MODEL,
    ]

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Download and load models for causal language modeling."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(default_models),
        help=f"Comma-separated list of model names to load.\nDefault: {','.join(default_models)}.",
    )
    args = parser.parse_args()
    models = args.models.split(",")
    if "mri" in models:
        models = MRI_MODELS
    elif "dllm" in models:
        models = DLLM_MODELS

    # Load models
    for model in models:
        subprocess.run(
            ["hf", "download", model],
            check=True,
        )


if __name__ == "__main__":
    main()
