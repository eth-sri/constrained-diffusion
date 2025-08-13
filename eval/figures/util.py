import json
import os
import re
import statistics
from pathlib import Path

import numpy as np
import pandas
import scipy.stats as stats
import tabulate


def extract_model_dataset(file_path):
    """Extract model and dataset information from file path and uncompiled file."""
    file_name = os.path.basename(file_path)

    # Extract if it's constrained or unconstrained
    is_constrained = "_c.compiled.jsonl" in file_name

    # Extract parameters from filename
    params = {}
    param_matches = re.findall(r"([a-z]+)=([0-9.]+)", file_name)
    for key, value in param_matches:
        params[key] = value

    # Get the uncompiled file path
    uncompiled_file = file_path.replace(".compiled.jsonl", ".jsonl")
    try:
        with open(uncompiled_file) as f:
            # Read the first line to extract metadata
            first_line = f.readline().strip()
            data = json.loads(first_line)
        # Extract dataset and model information if available
        dataset = data["dataset"]
        model = data["model_name"]

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading uncompiled file of {file_name}: {e}")
        dataset = "unknown_dataset"
        model = "unknown_model"

    return {
        "dataset": dataset,
        "model": model,
        "params": params,
        "is_constrained": is_constrained,
        "file_path": file_path,
    }


def get_unconstrained_file(constrained_file):
    """Convert a constrained file path to its unconstrained equivalent."""
    return re.sub(
        r"_gs=\d+",
        "_gs=0",
        constrained_file.replace("_c.", "_nc."),
    )


def get_autocompleted_file(constrained_file):
    """Convert a constrained file path to its unconstrained equivalent."""
    return constrained_file.replace(".compiled", ".autocompleted.compiled")


def process_file(file_path, filtered_instances=None) -> dict | None:
    """Process a compiled JSONL file and return statistics."""
    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        return None

    # Read the compiled file
    with open(file_path) as f:
        lines = f.readlines()
        if not lines:
            print(f"Warning: Empty file: {file_path}")
            return None
        datas = [json.loads(line) for line in lines if line.strip()]
    # Filter instances if a filter is provided
    if filtered_instances is not None:
        prev_data_len = len(datas)
        datas = [d for d in datas if d["instance_id"] not in filtered_instances]
        filtered_amount = prev_data_len - len(datas)
        if not lines:
            print(f"Warning: No matching instances found in {file_path}")
            return None
    else:
        filtered_amount = 0

    # Initialize counters
    total = 0
    syntax_correct = 0
    tests_passed = 0
    timed_out = 0
    time_taken = []

    # For comparing constrained vs unconstrained
    unconstrained_data = {}
    unconstrained_file = get_unconstrained_file(file_path)

    if Path(unconstrained_file).exists():
        compare_unconstrained = True
        with open(unconstrained_file) as f:
            unconstrained_lines = f.readlines()
            for line in unconstrained_lines:
                try:
                    data = json.loads(line)
                    instance_id = data["instance_id"]
                    unconstrained_data[instance_id] = data
                except (json.JSONDecodeError, KeyError) as e:
                    # print(f"Error processing unconstrained line: {e}")
                    pass
    else:
        compare_unconstrained = False

    # For comparing constrained vs autocompleted
    autocompleted_data = {}
    autocompleted_file = get_autocompleted_file(file_path)

    if Path(autocompleted_file).exists():
        compare_autocompleted = True
        with open(autocompleted_file) as f:
            autocompleted_lines = f.readlines()
            for i, line in enumerate(autocompleted_lines):
                try:
                    data = json.loads(line)
                    instance_id = data["instance_id"]
                    autocompleted_data[instance_id] = data
                except (json.JSONDecodeError, KeyError) as e:
                    # print(f"Error processing autocomplete line: {e}")
                    pass
    else:
        compare_autocompleted = False

    # Process each line in the compiled file
    for data in datas:
        try:
            total += 1
            did_timed_out = bool(data["timed_out"])

            # Check for syntax correctness
            syntax_correct += int(data.get("syntax_ok", False))

            # Check for functional correctness
            tests_passed += int(data.get("passed_tests", False))

            # Check for timeout
            timed_out += int(did_timed_out)

            time_taken.append(
                (data["time_taken"] - (data.get("time_taken_autocompletion", 0) or 0))
                / (data.get("generated_tokens", 1) or 1)
            )

        except (json.JSONDecodeError, KeyError) as e:
            # print(f"Error processing line: {e}")
            pass

    # Initialize comparison statistics
    unc_syntax_correct = 0
    unc_tests_passed = 0
    degraded_syntax = 0
    flipped_syntax = 0
    flipped_tests = 0
    degraded_tests = 0
    unc_time_taken = []

    # Compare with unconstrained if available
    if compare_unconstrained and unconstrained_data:
        for data in datas:
            try:
                instance_id = data["instance_id"]

                unc_data = unconstrained_data[instance_id]

                constrained_syntax_ok = data.get("syntax_ok", False)
                unconstrained_syntax_ok = unc_data.get("syntax_ok", False)

                unc_syntax_correct += int(unconstrained_syntax_ok)

                # Check for degraded/flipped
                if unconstrained_syntax_ok and not constrained_syntax_ok:
                    degraded_syntax += 1
                elif constrained_syntax_ok and not unconstrained_syntax_ok:
                    flipped_syntax += 1

                constrained_tests_passed = data.get("passed_tests", False)
                unconstrained_tests_passed = unc_data.get("passed_tests", False)

                unc_tests_passed += int(unconstrained_tests_passed)
                # Check for degraded/flipped tests
                if unconstrained_tests_passed and not constrained_tests_passed:
                    degraded_tests += 1
                elif constrained_tests_passed and not unconstrained_tests_passed:
                    flipped_tests += 1

                # Accumulate time taken for unconstrained
                unc_time_taken.append(
                    unc_data.get("time_taken", 0)
                    / (unc_data.get("generated_tokens", 1) or 1)
                )

            except (json.JSONDecodeError, KeyError) as e:
                pass
                # print(f"Error comparing with unconstrained: {e}")

    # Initialize autocompleted statistics
    auto_syntax_correct = 0
    auto_tests_passed = 0
    auto_time_taken = []
    # compare to autocompleted if available
    if compare_autocompleted and autocompleted_data:
        for i, data in enumerate(datas):
            try:
                instance_id = data["instance_id"]

                auto_data = autocompleted_data[instance_id]
                if "skipped" in auto_data:
                    auto_syntax_correct += data.get("syntax_ok", False)
                    auto_tests_passed += data.get("passed_tests", False)
                    auto_time_taken.append(
                        data.get("time_taken", 0)
                        / (data.get("generated_tokens", 1) or 1)
                    )
                else:
                    auto_syntax_correct += int(auto_data.get("syntax_ok", False))
                    auto_tests_passed += int(auto_data.get("passed_tests", False))
                    auto_time_taken.append(
                        auto_data.get("time_taken", 0)
                        / (auto_data.get("generated_tokens", 1) or 1)
                    )

            except (json.JSONDecodeError, KeyError) as e:
                # print(f"Error comparing with autocomplete: {e}")
                pass

    # Return statistics
    return {
        "filtered_amount": filtered_amount,
        "file_path": file_path,
        "total": total,
        "syntax_correct": syntax_correct,
        "syntax_percent": syntax_correct / total if total > 0 else 0,
        "tests_passed": tests_passed,
        "tests_percent": tests_passed / total if total > 0 else 0,
        "unc_syntax_correct": unc_syntax_correct,
        "unc_syntax_percent": unc_syntax_correct / total if total > 0 else 0,
        "unc_tests_passed": unc_tests_passed,
        "unc_tests_percent": unc_tests_passed / total if total > 0 else 0,
        "degraded_syntax": degraded_syntax,
        "flipped_syntax": flipped_syntax,
        "degraded_tests": degraded_tests,
        "flipped_tests": flipped_tests,
        "timed_out": timed_out,
        "time_taken": time_taken,
        "average_time_taken": statistics.median(time_taken)
        if len(time_taken) > 0
        else -1,
        "unc_time_taken": unc_time_taken,
        "unc_average_time_taken": statistics.median(unc_time_taken)
        if len(unc_time_taken) > 0
        else -1,
        "unc_average_diff_time_taken": statistics.median(
            [x - y for x, y in zip(time_taken, unc_time_taken)]
        ),
        "unc_average_ratio_time_taken": statistics.median(
            [(x / y) - 1 for x, y in zip(time_taken, unc_time_taken) if y != 0]
        ),
        "auto_syntax_correct": auto_syntax_correct,
        "auto_syntax_percent": auto_syntax_correct / total if total > 0 else 0,
        "auto_tests_passed": auto_tests_passed,
        "auto_tests_percent": auto_tests_passed / total if total > 0 else 0,
        "auto_time_taken": auto_time_taken,
        "auto_average_time_taken": statistics.median(auto_time_taken)
        if len(auto_time_taken) > 0
        else -1,
    }


def format_with_ci(
    d: pandas.DataFrame, ci=0.95, floatfmt=".1f"
) -> tuple[str, str] | None:
    """
    Calculate the confidence interval for a given field in the processed files.
    """

    # collect data
    if not len(d):
        return None
    if len(d) == 1:
        # if only one data point, return it without error
        return (f"{d[0]:{floatfmt}}", "N/A")

    # calculate mean and error
    m, s, n = np.mean(d), np.std(d, ddof=1), len(d)  # Mean, SD, Size
    t = stats.t.ppf(ci, df=n - 1)  # t-value

    e = t * (s / np.sqrt(n))

    # format
    return f"{m:{floatfmt}}", f"{e:{floatfmt}}"


def custom_tabulate(rows, headers=None, tablefmt="latex_raw", floatfmt=".1f"):
    if tablefmt == "csv":
        out_str = ""
        if headers is not None:
            out_str += ",".join(map(str, headers)) + "\n"
        out_str += "\n".join([",".join(map(str, row)) for row in rows])
        return out_str
    return tabulate.tabulate(
        rows, headers=headers, tablefmt=tablefmt, floatfmt=floatfmt
    )
