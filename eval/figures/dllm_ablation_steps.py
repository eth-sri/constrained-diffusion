#!/usr/bin/env python3
"""
Create the table for FIM results.
"""

import math
import os
import pathlib

import fire
import pandas
import tabulate

from eval.figures.util import process_file, format_with_ci

results_dir = pathlib.Path(__file__).parent.parent.parent / "results"
MODEL = "Dream-org_Dream-v0-Instruct-7B"
DATASETS = {
    "THUDM_humaneval-x_cpp": "C++",
    "jsonschema": "JSON Schema",
    "smiles": "SMILES",
}
STEP_SIZES = ["16", "32", "64", "128", "256"]
FILE = "{dataset}_{model}_s={seed}_t=0.2_gs=0_sz={stepsize}_synth_c.compiled.jsonl"

DATASET_SIZES = {
    "smiles": 167,
    "THUDM_humaneval-x_cpp": 164,
    "jsonschema": 272,
}


def create_table(
    results_dir: str = results_dir,
    format: str = "latex_raw",
    functional: bool = False,
    ci: bool = False,
):
    # columns: for each n-gapped dataset, syntax unconstrained, syntax constrained
    headers = ["Dataset", ""] + sum(
        (
            [
                f"Vanilla {d[0]}",
                "(CI)",
                f"Syntax {d[0]}",
                "(CI)",
                f"Auto {d[0]}",
                "(CI)",
                "",
            ]
            for d in DATASETS
        )
        if ci
        else (
            [f"Vanilla {d[0]}", f"Syntax {d[0]}", f"Auto {d[0]}", ""] for d in DATASETS
        ),
        start=[],
    )
    seeds = [0, 1, 2, 3]
    field = "tests_percent" if functional else "syntax_percent"
    rows = []
    for step_size in STEP_SIZES:
        model_row = [step_size, ""]
        for dataset in DATASETS:
            results = []
            for seed in seeds:
                file_path = os.path.join(
                    results_dir,
                    FILE.format(
                        dataset=dataset, model=MODEL, seed=seed, stepsize=step_size
                    ),
                )
                result = process_file(file_path)
                if result is None:
                    continue
                if result["total"] not in (
                    DATASET_SIZES[dataset] - result["filtered_amount"] - i
                    for i in range(4)
                ):
                    print(
                        f"Warning: file {file_path} has {result['total']} results, expected {DATASET_SIZES[dataset] - result['filtered_amount']}"
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
                    *format_with_ci(results[field] * 100, ci=0.95),
                    *format_with_ci(results[f"auto_{field}"] * 100, ci=0.95),
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
    print(tabulate.tabulate(rows, headers=headers, tablefmt=format, floatfmt=".1f"))
    if not functional:
        create_table(results_dir, format, functional=True, ci=ci)


if __name__ == "__main__":
    fire.Fire(create_table)
