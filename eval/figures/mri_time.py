#!/usr/bin/env python3
"""
Create the table for MRI results.
"""

import os
import pathlib
import re
import statistics

import fire
import pandas
import tabulate

from eval.figures.mri import DATASET_SIZES
from eval.figures.util import process_file, format_with_ci


results_dir = pathlib.Path(__file__).parent.parent.parent / "results"
filter_file_path = pathlib.Path(__file__).parent.parent / "mri" / "filter.txt"

MODELS = {
    "google_codegemma-7b": r"\codegemma",
    "bigcode_starcoder2-7b": r"\starcodertwo",
    "deepseek-ai_deepseek-coder-1.3b-base": r"\deepseekctwob",
    "deepseek-ai_deepseek-coder-6.7b-base": r"\deepseekcsevenb",
    "deepseek-ai_deepseek-coder-33b-base": r"\deepseekcthirtyb",
}
FILE = "HumanEval_MRI_cpp_{i}_{model}_s={seed}_t=1_gs=0_synth_c.compiled.jsonl"


def create_table(
    results_dir: str = results_dir, format: str = "latex_raw", ci: bool = False
):
    # columns: for each n-gapped dataset, syntax unconstrained, syntax constrained
    filter_instances = open(filter_file_path, "r").read().splitlines(keepends=False)
    gaps = [1, 2, 3]
    seeds = [0, 1, 2, 3]
    headers = ["Model"] + ["Vanilla", "Syntax", "Diff", ""] * len(gaps)
    headers.pop(-1)
    field = "average_diff_time_taken"
    field2 = "average_ratio_time_taken"
    rows = []
    all = []
    for model in MODELS:
        model_row = [MODELS[model]]
        for gap in gaps:
            results = []
            for seed in seeds:
                file_path = os.path.join(
                    results_dir, FILE.format(i=gap, model=model, seed=seed)
                )
                result = process_file(
                    file_path, [f"{x}_spans_{gap}" for x in filter_instances]
                )
                if result is None:
                    print(f"Warning: file {file_path} is empty, skipping")
                    continue
                if result["total"] != DATASET_SIZES[gap] - result["filtered_amount"]:
                    print(
                        f"Warning: {file_path} has {result['total']} total instances, expected {DATASET_SIZES[gap] - result['filtered_amount']}"
                    )
                    continue
                results.append(result)
            if not results:
                model_row.extend(["N/A", "N/A", "N/A", ""])
                continue
            results = pandas.DataFrame.from_records(results)
            model_row.extend(
                (
                    # format_with_ci(results[f"unc_{field}"] * 100)[0],
                    format_with_ci(results[f"unc_{field}"] * 100)[0],
                    f"""$_{{\\uparrow {format_with_ci(
                        results[f"unc_{field2}"] * 100, ci=0.95, floatfmt=".0f"
                    )[0]}\\%}}$""",
                    "",
                )
            )
            if gap == 3:
                all.append(int(re.findall(r"\d+", model_row[-2])[0]))
        model_row.pop(-1)
        rows.append(model_row)
    # Print the table
    print(tabulate.tabulate(rows, headers=headers, tablefmt=format, floatfmt=".1f"))
    print("Average", statistics.mean(all))


if __name__ == "__main__":
    fire.Fire(create_table)
