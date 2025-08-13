<div align="center"><h1>Constrained Decoding of Diffusion LLMs<br> with Context-Free Grammars</h1></div>

[![arXiv](https://img.shields.io/badge/arXiv-2508.10111-b31b1b.svg)](https://arxiv.org/abs/2508.10111)
![Python Versions](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Rust Version](https://img.shields.io/badge/rust-2021-orange)
[![Python Tests](https://github.com/eth-sri/constrained-diffusion/actions/workflows/python-tests.yml/badge.svg)](https://github.com/eth-sri/constrained-diffusion/actions/workflows/python-tests.yml)
[![Rustformlang Tests](https://github.com/eth-sri/constrained-diffusion/actions/workflows/rustformlang-tests.yml/badge.svg)](https://github.com/eth-sri/constrained-diffusion/actions/workflows/rustformlang-tests.yml)
[![Regex DFA Tests](https://github.com/eth-sri/constrained-diffusion/actions/workflows/regex-dfa-tests.yml/badge.svg)](https://github.com/eth-sri/constrained-diffusion/actions/workflows/regex-dfa-tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the implementation of [Constrained Decoding of Diffusion LLMs with Context-Free Grammars](https://arxiv.org/abs/2508.10111), including techniques for multi-region constrained generation. Our method guarantees syntactic correctness while improving functional correctness by up to 7%.

## 🚀 Overview

We present the first generalized method for constrained decoding of multi-region infilling and out-of-order generation models. Our approach:

- **Works with SOTA diffusion LLMs** like LLaDA, Dream-Coder and DiffuCoder for non-autoregressive generation
- **Also works for Fill-in-the-Middle (FIM) and Multi-Region Infilling (MRI) models** like StarCoder, DeepSeek Coder, and CodeGemma
- **Supports multiple constraint languages** through context-free grammars (examples provided are JSON Schema, C++, and SMILES)
- **Guarantees syntactic correctness** wrt. the grammar 
- **Improves functional correctness** by up to 7% with minimal computational overhead

## 📦 Installation

### Prerequisites
- [Python](https://www.python.org/) 3.11+ 
- [Rust](https://www.rust-lang.org/) (for building the formal language library)
- CUDA-compatible GPU (for inference)

### Setup

We recommend using a virtual environment to avoid conflicts with other Python packages.

0. **Clone the repository and set up virtual enviroment:**
```bash
git clone https://github.com/eth-sri/constrained-diffusion.git
cd constrained-diffusion
python3 -m venv venv
source venv/bin/activate
```

1. **Build and install Rust bindings:**
```bash
cd rustformlang_bindings
pip install maturin
maturin build --release
pip install .
cd ..
```

2. **Install the main package:**
```bash
pip install -e .
```


4. **Verify installation:**
```bash
pytest tests
```


## 🔧 Usage & Demo

Check out [`example.py`](example.py) for a complete example of how to use the constrained decoding mechanism.
In general, you want to first load a model and then load a constraint language, such as C++ or JSON Schema. The example below shows abbreviated code on how to use the `GSAI-ML/LLaDA-8B-Instruct` model with a C++ constraint.
Replace the model name with any diffusion LLM of your choice, such as `apple/DiffuCoder-7B-Instruct`.

```bash
python3 example.py
```

This is a visualization of our constrained decoding mechanism on output similar to that created by LLaDA 7b.

> ![LLaDA 7B Inference](./docs/static/images/animation/words_grid_animation_constrained_dark.gif)


## 📁 Project Structure

```
├── constrained_diffusion/           # Main package
│   ├── constrain_utils.py            # Constraint generation utilities
│   ├── cfgs/                         # Context-free grammar definitions
│   └── eval/                         # Evaluation frameworks
│       ├── dllm/                     # Evaluation framework for DLLMs
│       └── mri/                      # Evaluation framework for Multi-Region Infilling
├── rustformlang/                     # Rust formal language library
├── rustformlang_bindings/            # Python bindings for Rust library
├── eval/                             # Evaluation scripts and results
│   ├── dllm/                         # DLLM task evaluations
│   ├── mri/                          # Multi-Region infilling evaluations
│   └── figures/                      # Result visualization
├── benchmark_generation/             # Benchmark generation tools
└── docs/                             # Project website
```

## 🧪 Evaluation

### Datasets

We run MRI and diffusion LLMs on the following datasets:

| Dataset | Setting | Description                                           | Download |
|---------|---------|-------------------------------------------------------|----------|
| C++     | MRI     | C++ code generation tasks with multi-region infilling | [🤗 HuggingFace](https://huggingface.co/datasets/eth-sri/HumanEval-MRI-Cpp)    |
| C++     | DLM     | C++ code generation tasks with diffusion LLMs         | [🤗 HuggingFace](https://huggingface.co/datasets/zai-org/humaneval-x) |
| JSON    | DLM     | Data extraction, following a JSON Schema              | [🤗 HuggingFace](https://huggingface.co/datasets/eth-sri/json-mode-eval-extended) |
| SMILES  | DLM     | Chemical compound representation in SMILES            | [🤗 HuggingFace](https://huggingface.co/datasets/eth-sri/smiles-eval)      |

> You can download the results of our evaluation using the following link: [Download Results](https://files.sri.inf.ethz.ch/constrained-diffusion/results.zip).
> Unzip the file in the `results/` directory to access the evaluation results.


### Running Inference

For the MRI models, we provide an execution harness for the C++ HumanEval multi-region dataset.
To execute task 11 on the 1-region dataset with constraints and traces enabled, use the following command:
```bash
python3 -m constrained_diffusion.eval.mri.generic_inference \
  --max-tokens 256 \
  --model_name deepseek-ai/deepseek-coder-6.7b-base \
  --seed 0 \
  --temp 1 \
  --dataset-name HumanEval/MRI/cpp/1 \
  --constrained True \
  --trace True \
  --task_id /11_ 
```

For the diffusion LLMs, use the following command for the JSON dataset.
```bash
python3 -m constrained_diffusion.eval.dllm.generic_inference \
  --max-tokens 256 \
  --model_name apple/DiffuCoder-7B-Instruct \
  --seed 0 \
  --temp 0.2 \
  --dataset-name jsonschema \
  --steps 32 \
  --constrained True \
  --trace True \
  --task_id _37
```

A general orchestration script for all experiments in the main paper is provided in `eval/fim/run_fim.py` and `eval/dllm/run_dllm.py`.
The results are stored in the `results/` directory, with each configuration's results in a separate file.

### Running Evaluation

Evaluation of result correctness is decoupled from the inference step. The following assumes that the inference step above was executed correctly and results lie in `results`.

> Note: For SMILES evaluation, you need to install `rdkit`and `partialsmiles`: `pip install rdkit partialsmiles`

Make sure to have sufficient memory and CPU cores available, as the evaluation scripts can be memory-intensive.
```bash
# Evaluate all files in the results folder
bash eval/check_all_individually.sh results/*
```

### More details

You  can find more details on the evaluation scripts, for example on how to reproduce the figures from the paper, in the README in the `eval/` directory: [README](eval/README.md).

## 🤝 Contributing

We welcome contributions! When contributing, please make sure to activate pre-commit hooks to ensure code quality and consistency. You can install pre-commit hooks with:

```bash
pip install pre-commit
pre-commit install
```

### Adding New Constraint Languages

1. Define the grammar in `constrained_outoforder/cfgs/`
2. Implement lexical mapping in `check_lex_map.py`
3. Add tests in `tests/test_cfgs/`
4. Update documentation
 
### Adding New Evaluation Tasks

1. [Create a new constraint language](#adding-new-constraint-languages)
2. Implement a dataset in `constrained_outoforder/eval/[dllm|mri]/datasets/your_task.py`
4. Register the dataset using `register_dataset()`
3. Add evaluation logic in `eval/[dllm|mri]/your_task/checker.py`

### Adding a New Model

1. Implement the model in `constrained_outoforder/eval/[dllm|mri]/models/your_model.py`
2. Register the model using `register_model()`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Paper**: [arXiv:2508.10111](https://arxiv.org/abs/2508.10111)
- **Project Website**: [Constrained Decoding Paper Website + Demo](https://constrained-diffusion.ai)
- **Rustformlang README**: [Rustformlang Docs](rustformlang/)

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{mundler2025constraineddiffusion,
    title={Constrained Decoding of Diffusion LLMs with Context-Free Grammars}, 
    author={Niels Mündler and Jasper Dekoninck and Martin Vechev},
    year={2025},
    eprint={2508.10111},
    archivePrefix={arXiv},
    url={https://arxiv.org/abs/2508.10111}
}
```

This work was done by the [Secure, Reliable and Intelligent Systems Lab](https://sri.inf.ethz.ch/) at [ETH Zurich](https://ethz.ch).
