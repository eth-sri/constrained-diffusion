#!/usr/bin/env python3
"""
Create the table for MRI results.
"""

import math
import os
import pathlib

import fire
import pandas

from eval.figures.util import process_file, format_with_ci, custom_tabulate


results_dir = pathlib.Path(__file__).parent.parent.parent / "results"
filter_file_path = pathlib.Path(__file__).parent.parent / "mri" / "filter.txt"

MODELS = {
    "bigcode_starcoder2-7b": r"\starcodertwo",
    "google_codegemma-7b": r"\codegemma",
    "deepseek-ai_deepseek-coder-1.3b-base": r"\deepseekctwob",
    "deepseek-ai_deepseek-coder-6.7b-base": r"\deepseekcsevenb",
    "deepseek-ai_deepseek-coder-33b-base": r"\deepseekcthirtyb",
}
FILE = "HumanEval_MRI_cpp_{i}_{model}_s={seed}_t=1_gs=0_synth_c.compiled.jsonl"
DATASET_SIZES = {
    1: 164,
    2: 161,
    3: 148,
}


def create_table(
    results_dir: str = results_dir,
    format: str = "latex_raw",
    functional: bool = False,
    ci: bool = False,
):
    filter_instances = open(filter_file_path, "r").read().splitlines(keepends=False)
    # columns: for each n-gapped dataset, syntax unconstrained, syntax constrained
    gaps = [1, 2, 3]
    seeds = [0, 1, 2, 3]
    headers = ["Dataset"] + sum(
        (
            [
                f"{i}_Vanilla",
                f"{i}_(CI)",
                f"{i}_Syntax",
                f"{i}_(CI)",
                f"{i}_Auto",
                f"{i}_(CI)",
                "",
            ]
            for i in gaps
        )
        if ci
        else ([f"{i}_Vanilla", f"{i}_Syntax", f"{i}_Auto", ""] for i in gaps),
        start=[],
    )
    headers.pop(-1)
    field = "tests_percent" if functional else "syntax_percent"
    rows = []
    for model in MODELS:
        model_row = [MODELS[model], ""]
        for gap in gaps:
            results = []
            for seed in seeds:
                file_path = os.path.join(
                    results_dir, FILE.format(i=gap, model=model, seed=seed)
                )
                result = process_file(
                    file_path,
                    filtered_instances=[f"{x}_spans_{gap}" for x in filter_instances],
                )
                if result is None:
                    print(f"Warning: file {file_path} is empty, skipping")
                    continue
                if result["total"] < DATASET_SIZES[gap] - result["filtered_amount"] - 4:
                    print(
                        f"Warning: file {file_path} has {result['total']} results, expected {DATASET_SIZES[gap]-result['filtered_amount']}, skipping"
                    )
                    continue
                results.append(result)
            if not results:
                if ci:
                    model_row.extend(["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", ""])
                else:
                    model_row.extend(["N/A", "N/A", "N/A", ""])
                continue
            results = pandas.DataFrame.from_records(results)
            results_with_ci = (
                format_with_ci(results[f"unc_{field}"] * 100, ci=0.95),
                format_with_ci(results[field] * 100, ci=0.95),
                format_with_ci(results[f"auto_{field}"] * 100, ci=0.95),
            )
            diff_unc_with_ci = (
                [0, 0],
                format_with_ci(
                    (results[field] - results[f"unc_{field}"]) * 100, ci=0.95
                ),
                format_with_ci(
                    (results[f"auto_{field}"] - results[f"unc_{field}"]) * 100, ci=0.95
                ),
            )
            diff_con_with_ci = (
                format_with_ci(
                    (results[f"unc_{field}"] - results[field]) * 100, ci=0.95
                ),
                [0, 0],
                format_with_ci(
                    (results[f"auto_{field}"] - results[field]) * 100, ci=0.95
                ),
            )

            def max_indices(results_with_ci, diff_unc_with_ci, diff_con_with_ci):
                # find the max
                max_value, max_index = max(
                    (float(x[0]), i) for i, x in enumerate(results_with_ci)
                )
                # mark unc as underlined if increase due to max is not guaranteed > 0
                underlined_indices = []
                if (
                    float(diff_unc_with_ci[max_index][0])
                    - float(diff_unc_with_ci[max_index][1])
                    <= 0
                ):
                    underlined_indices.append(0)
                # mark con as underlined if increase due to max is not guaranteed > 0
                if (
                    float(diff_con_with_ci[max_index][0])
                    - float(diff_con_with_ci[max_index][1])
                    <= 0
                ):
                    underlined_indices.append(1)

                # find all values that have overlapping CI with the max
                return [
                    i
                    for i, x in enumerate(results_with_ci)
                    if math.isclose(float(x[0]), max_value)
                ], underlined_indices

            def wrap_in_bfont(text):
                return f"\\textbf{{{text}}}"

            def wrap_in_underline(text):
                return f"\\underline{{{text}}}"

            bolded_indices, underlined_indices = max_indices(
                results_with_ci, diff_unc_with_ci, diff_con_with_ci
            )
            bolded_results = (
                wrap_in_bfont(result[0])
                if i in bolded_indices
                else wrap_in_underline(result[0])
                if i in underlined_indices
                else result[0]
                for i, result in enumerate(results_with_ci)
            )
            model_row.extend(
                (
                    *format_with_ci(results[f"unc_{field}"] * 100, ci=0.95),
                    *format_with_ci(
                        (results[f"{field}"] - functional * results[f"unc_{field}"])
                        * 100,
                        ci=0.95,
                    ),
                    *format_with_ci(
                        (
                            results[f"auto_{field}"]
                            - functional * results[f"unc_{field}"]
                        )
                        * 100,
                        ci=0.95,
                    ),
                    "",
                )
                if ci
                else (
                    *bolded_results,
                    "",
                )
            )
        model_row.pop(-1)
        rows.append(model_row)
    # Print the table
    print(custom_tabulate(rows, headers=headers, tablefmt=format, floatfmt=".1f"))


if __name__ == "__main__":
    fire.Fire(create_table)
