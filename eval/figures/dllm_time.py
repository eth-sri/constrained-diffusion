#!/usr/bin/env python3
"""
Create the table for FIM results.
"""

import os
import pathlib
import re
import statistics

import fire
import pandas
import tabulate

from eval.figures.util import process_file, format_with_ci

results_dir = pathlib.Path(__file__).parent.parent.parent / "results"
MODELS = {
    "Dream-org_Dream-v0-Instruct-7B": r"\dream",
    "Dream-org_Dream-Coder-v0-Instruct-7B": r"\dreamc",
    "GSAI-ML_LLaDA-8B-Instruct": r"\llada",
    "apple_DiffuCoder-7B-Instruct": r"\diffuc",
}
DATASETS = {
    "THUDM_humaneval-x_cpp": "C++",
    "jsonschema": "JSON Schema",
    "smiles": "SMILES",
}
FILE = "{dataset}_{model}_s={seed}_t=0.2_gs=0_sz=32_synth_c.jsonl"


def create_table(results_dir: str = results_dir, format: str = "latex_raw"):
    # columns: for each n-gapped dataset, syntax unconstrained, syntax constrained
    headers = ["Dataset"] + sum(
        ([f"Vanilla {d}", f"Syntax {d}", f"Diff {d}", ""] for d in DATASETS), start=[]
    )
    seeds = [0, 1, 2, 3]
    headers.pop(-1)
    field = "average_diff_time_taken"
    field2 = "average_ratio_time_taken"
    rows = []
    all = []
    for model in MODELS:
        model_row = [MODELS[model]]
        for dataset in DATASETS:
            results = []
            for seed in seeds:
                file_path = os.path.join(
                    results_dir, FILE.format(dataset=dataset, model=model, seed=seed)
                )
                result = process_file(file_path)
                if result is not None:
                    results.append(result)
            if not len(results):
                model_row.extend(["N/A", "N/A", "N/A", ""])
                continue
            results = pandas.DataFrame.from_records(results)
            model_row.extend(
                (
                    format_with_ci(results[f"unc_{field}"], ci=0.95)[0],
                    f'''$_{{\\uparrow {format_with_ci(
                        100*results[f"unc_{field2}"], ci=0.95, floatfmt=".0f"
                    )[0]}\\%}}$''',
                    "",
                )
            )
            all.append(int(re.findall(r"\d+", model_row[-2])[0]))
        model_row.pop(-1)
        rows.append(model_row)
    # Print the table
    print(tabulate.tabulate(rows, headers=headers, tablefmt=format, floatfmt=".1f"))
    print("Average", statistics.mean(all))


if __name__ == "__main__":
    fire.Fire(create_table)
