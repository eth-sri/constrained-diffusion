# Unified Evaluation Scripts for DLLM Tasks

This directory contains unified scripts for evaluating the results of different DLLM tasks. The scripts provide a consistent interface for checking the syntax and functional correctness of model outputs across different tasks.

## Directory Structure

```
eval/
├── check.py             # Main script for checking task outputs
├── check_all.py         # Script for running checks on multiple files in parallel
├── check_all_individually.py         # Calls check_all.py on each given file to avoid memory issues
├── dllm/
│   └── <task_name>/                 # Task-specific directory
│       └── checker.py               # Task-specific checker module
├── mri/
│   └── <task_name>/                 # Task-specific directory
│       └── checker.py               # Task-specific checker module
├── results/            # Directory for storing evaluation results
```

## Available Tasks

- `smiles`: Evaluates SMILES strings for chemical compounds
- `cpp`: Evaluates C++ code
- `jsonmode`: Evaluates JSON against schemas

## How to Use


### 1. Run Model Inference

For the MRI models, we provide an inference harness for the C++ HumanEval multi-region dataset.
To execute task 11 on the 1-region dataset with constraints and traces enabled, use the following command:
```bash
python3 -m constrained_diffusion.eval.mri.generic_inference
  --max-tokens 256
  --model_name deepseek-ai/deepseek-coder-6.7b-base
  --seed 0
  --temp 1
  --dataset-name HumanEval/MRI/cpp/1
  --constrained True
  --trace True
  --task_id /11_
```

For the diffusion LLMs, use the following command for the SMILES dataset.
```bash
python3 -m constrained_diffusion.eval.dllm.generic_inference
  --max-tokens 256
  --model_name apple/DiffuCoder-7B-Instruct
  --seed 0
  --temp 0.2
  --dataset-name smiles
  --steps 32
  --constrained True
  --trace True
  --task_id _37
```

Omit the task_id argument to run all tasks in the dataset.
Similarly, use dataset-name `jsonschema` for JSON tasks and `THUDM/humaneval-x` for C++ tasks.

A general orchestration script for all experiments in the main paper is provided in `eval/fim/run_fim.py` and `eval/dllm/run_dllm.py`.
The results are stored in the `results/` directory, with each configuration's results in a separate file.

### 2. Checking Task Outputs

To check the outputs of a specific task, use the `check_all.sh` script:

```bash
python eval/check_all.py <file1.jsonl> [<file2.jsonl> ...]
```

Example:
```bash
python eval/check_all.py results/jsonschema_Dream-org_Dream-Coder-v0-Instruct-7B_s=3_t=0.2_gs=0_sz=32_synth_nc.autocompleted.compiled.jsonl
```

> If you want to check SMILES outputs, make sure to install `rdkit` and `partialsmiles`:

This will process each input file and create a corresponding `.compiled.jsonl` and `.compiled.autocompleted.jsonl` file with the evaluation results.

### 3. Compiling Figures from the Paper

To compile the figures and statistics from the main paper, run the files in `eval/figures`:

Example:
```bash
pip install scipy tabulate
export PYTHONPATH=$(pwd):$PYTHONPATH

# MRI results
python eval/figures/mri.py --results-dir results/ 
# DLM results
python eval/figures/dllm.py --results-dir results/ 

# MRI runtime overhead
python eval/figures/mri_time.py --results-dir results/
# DLM runtime overhead
python eval/figures/dllm_time.py --results-dir results/

# DLM step size ablation results
python eval/figures/dllm_ablation_steps.py --results-dir results/
# DLM step size ablation runtime overhead
python eval/figures/dllm_ablation_steps_time.py --results-dir results/
```

> You can download the results of our evaluation (Step 1. and 2.) using the following link: [Download Results](https://files.sri.inf.ethz.ch/constrained-diffusion/results.zip).
> Unzip the file in the `results/` directory to access the evaluation results.

## Output Format

The evaluation results are stored in `.compiled.jsonl` files, where each line is a JSON object with the following fields:

- `instance_id`: The ID of the instance being evaluated
- `extracted`: The extracted output from the model
- `syntax_ok`: Whether the output has correct syntax (boolean)
- `passed_tests`: Whether the output passes functional tests (boolean)
- `compiler_output`: Detailed output from the compiler/checker

The same is stored in `.compiled.autocompleted.jsonl` files for results obtain by sampling from the intersection language, but with lines containing `"skipped": true` if no sampling was required and the result of the sampled output instead.

## Adding a New Task

To add support for a new task:

1. Create a new directory under `eval/dllm/<task_name>/`
2. Create a `checker.py` file in that directory with a `check_instance(line)` function
3. The `check_instance` function should take a JSON string and return a dictionary with the evaluation results, including the fields `syntax_ok` and `passed_tests`
