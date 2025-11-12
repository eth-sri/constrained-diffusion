#!/usr/bin/env python3
"""
Unified check script for evaluating task outputs.
This script processes a JSONL file containing model outputs and evaluates them
for syntax correctness and functional correctness.

Usage:
    python check.py <task_name> <input_file.jsonl>

The script will output a compiled JSONL file with evaluation results.
"""

import os
import sys
import json
import importlib
import time
from pathlib import Path
from multiprocessing import get_context, TimeoutError

from tqdm import tqdm

DATASET_CHECKER_MAP = {
    "HumanEval/MRI/cpp/1": "dllm.cpp",
    "HumanEval/MRI/cpp/2": "dllm.cpp",
    "HumanEval/MRI/cpp/3": "dllm.cpp",
    "jsonschema": "dllm.jsonmode",
    "THUDM/humaneval-x/cpp": "dllm.cpp",
    "smiles": "dllm.smiles",
}


def to_output_filename(input_filename, autocomplete=False):
    """
    Convert the input filename to an output filename by replacing the extension with '.compiled.jsonl'.
    """
    if not input_filename.endswith(".jsonl"):
        raise ValueError("Input filename must end with .jsonl")
    return (
        input_filename[: -len(".jsonl")]
        + (".autocompleted" if autocomplete else "")
        + ".compiled.jsonl"
    )


def process_line(line, autocomplete=False, timeout=40):
    """
    Process a single line of input and evaluate it using the task-specific checker.

    Args:
        line:
        autocomplete: If true, checks the status of the autocomplete field.

    Returns:

    """
    instance = json.loads(line.strip())
    dataset = instance.get("dataset", None)
    task_name = DATASET_CHECKER_MAP.get(dataset, None)
    if not task_name:
        print("Error: Dataset not found in DATASET_CHECKER_MAP.", file=sys.stderr)
        print("Available datasets:", file=sys.stderr)
        for key in sorted(DATASET_CHECKER_MAP.keys()):
            print(f"  - {key}", file=sys.stderr)
        sys.exit(1)
    # Import the task-specific checker module
    try:
        # add the required path to sys.path
        parent_path = str(Path(__file__).parent.parent)
        sys.path.append(parent_path)
        checker_module = importlib.import_module(f"eval.{task_name}.checker")
    except ImportError as e:
        # print to stderr if the module is not found
        print(
            f"Error: Task '{task_name}' not found or has no checker module: {e}",
            file=sys.stderr,
        )
        print("Available tasks:", file=sys.stderr)
        for path in sorted(Path(__file__).parent.iterdir()):
            if path.is_dir():
                for subpath in sorted(path.iterdir()):
                    if subpath.is_dir() and (subpath / "checker.py").exists():
                        print(f"  - {path.name}.{subpath.name}", file=sys.stderr)
        sys.exit(1)

    if autocomplete:
        if instance.get("autocompletion"):
            instance["code"] = instance.get("autocompletion_raw", "")
            instance["extracted"] = instance.get("autocompletion", "")
        else:
            return (
                '{"skipped": "No autocompletion available", "instance_id": "'
                + instance.get("instance_id", "")
                + '"}'
            )

    result = checker_module.check_instance(instance, timeout=timeout)
    result.update(
        {
            "time_taken": instance.get("time_taken", None)
            - (instance.get("time_taken_autocompletion", 0) if autocomplete else 0),
            "timed_out": instance.get("timed_out", False),
            "resamples": instance.get("resamples", None),
            "generated_tokens": instance.get("generated_tokens", None),
        }
    )
    return json.dumps(result)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_file.jsonl> (<input_file.jsonl)*")
        sys.exit(1)

    jsonl_files = sys.argv[1:]
    jsonl_files = [
        file
        for file in jsonl_files
        if ".compiled" not in file and file.endswith(".jsonl")
    ]

    timeout = 40
    global_timeout = 600
    processes = 3 * os.cpu_count() // 4
    with get_context("spawn").Pool(processes=processes, maxtasksperchild=100) as pool:
        res = []
        for jsonl_file in jsonl_files:
            # Read the input file
            with open(jsonl_file) as f:
                lines = f.readlines()

                # Process each line using the task-specific checker
                for line in lines:
                    for autocomplete in [True, False]:
                        res.append(
                            (
                                jsonl_file,
                                autocomplete,
                                pool.apply_async(
                                    process_line,
                                    (
                                        line,
                                        autocomplete,
                                        timeout,
                                    ),
                                ),
                                line,
                            )
                        )
        reset_files = set()
        # Collect results
        start_time = time.time()
        bar = tqdm(total=len(res))
        diff = 0
        i = 0
        while res:
            i += 1
            time.sleep(0.1)  # Sleep to avoid busy waiting
            new_res = []
            global_timeout_elapsed = (time.time() - start_time > global_timeout) or (
                len(res) < processes
            )
            for file, autocomplete, r, line in res:
                if not r.ready() and not global_timeout_elapsed:
                    new_res.append((file, autocomplete, r, line))
                    continue
                try:
                    output_filename = to_output_filename(file, autocomplete)
                    if output_filename not in reset_files:
                        open(output_filename, "w").close()
                        reset_files.add(output_filename)
                    with open(output_filename, mode="a") as f:
                        print(
                            r.get(timeout=0 if not global_timeout_elapsed else timeout),
                            flush=True,
                            file=f,
                        )
                except TimeoutError:
                    print(
                        f"Timed out processing line: {line}",
                        file=sys.stderr,
                    )
                except FileNotFoundError as e:
                    print(
                        f"Error writing to output file for {file}: {e}",
                        file=sys.stderr,
                    )
            diff += len(res) - len(new_res)
            if i % 10 == 0:
                bar.update(diff)
                diff = 0
            res = new_res


if __name__ == "__main__":
    main()
