#!/usr/bin/env python3
"""
Plot constrained-vs-unconstrained decoding latency against grammar size for JSON, C++, and SMILES.
"""

import json
import pathlib
from typing import Dict, Iterable

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

from constrained_diffusion.cfgs.jsonschema import schema_to_cfg
from constrained_diffusion.cfgs.cpp import cpp_grammar
from constrained_diffusion.cfgs.smiles import smiles_schema
from eval.figures.util import extract_model_dataset, get_unconstrained_file

ROOT = pathlib.Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results"
CACHE_NAME = "jsonschema_time_vs_grammar_size_cache.csv"


def _grammar_size(grammar) -> int:
    normal = grammar.to_normal_form()
    if hasattr(normal, "size"):
        return normal.size()
    return normal.num_productions()


CPP_GRAMMAR_SIZE = cpp_grammar()[0].to_normal_form().num_productions()
SMILES_GRAMMAR_SIZE = _grammar_size(smiles_schema()[0])
print(CPP_GRAMMAR_SIZE)
print(SMILES_GRAMMAR_SIZE)
# Supported datasets; JSON uses per-instance grammar sizes, others are fixed
SUPPORTED_DATASETS = (
    "jsonschema",
    "jsonschema_r",
    "THUDM/humaneval-x/cpp",
    "smiles",
)
PLOT_DATASETS = {"jsonschema", "jsonschema_r"}
MODEL_MAP = {
    "Dream-org/Dream-v0-Instruct-7B": "Dream 7B",
    "Dream-org/Dream-Coder-v0-Instruct-7B": "Dream Coder 7B",
    "GSAI-ML/LLaDA-8B-Instruct": "LLaDA 8B",
    "apple/DiffuCoder-7B-Instruct": "DiffuCoder 7B",
}

# Render figure text via LaTeX.
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)


def _load_schema_sizes() -> Dict[str, int]:
    """Return a map from instance_id to grammar size (#productions)."""
    dataset = load_dataset("eth-sri/json-mode-eval-extended")["test"]
    sizes = {}
    for row in dataset:
        grammar, _, _ = schema_to_cfg(json.loads(row["schema"]))
        sizes[row["instance_id"]] = _grammar_size(grammar)
    return sizes


def _load_times(file_path: pathlib.Path) -> Dict[str, float]:
    """Return per-instance decoding time (seconds) keyed by instance_id."""
    times: Dict[str, float] = {}
    with open(file_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("timed_out"):
                # Skip timed out generations; they skew latency differences.
                continue
            time_taken = data.get("time_taken")
            if time_taken is None:
                continue
            time_taken -= data.get("time_taken_autocompletion") or 0
            times[data["instance_id"]] = time_taken
    return times


def _iter_constrained_files(results_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield constrained compiled results for supported datasets."""
    for path in sorted(results_dir.glob("*_synth_c.compiled.jsonl")):
        if ".autocompleted." in path.name or "jsonschema_r" in path.name:
            continue
        yield path


def _compute_rows(
    results_dir: pathlib.Path, schema_sizes: Dict[str, int]
) -> pd.DataFrame:
    """Compute the per-instance time differences and grammar sizes."""
    rows = []

    for constrained_file in _iter_constrained_files(results_dir):
        unconstrained_file = pathlib.Path(get_unconstrained_file(str(constrained_file)))
        if not unconstrained_file.exists():
            print(
                f"Skipping {constrained_file.name}: missing {unconstrained_file.name}"
            )
            continue

        meta = extract_model_dataset(str(constrained_file))
        dataset = meta["dataset"]
        if dataset not in SUPPORTED_DATASETS:
            continue

        constrained_times = _load_times(constrained_file)
        unconstrained_times = _load_times(unconstrained_file)

        for instance_id, con_time in constrained_times.items():
            unc_time = unconstrained_times.get(instance_id)
            if dataset in ("jsonschema", "jsonschema_r"):
                size = schema_sizes.get(instance_id)
            elif dataset == "THUDM/humaneval-x/cpp":
                size = CPP_GRAMMAR_SIZE
            elif dataset == "smiles":
                size = SMILES_GRAMMAR_SIZE
            else:
                size = None
            if unc_time is None or size is None:
                continue
            rows.append(
                {
                    "instance_id": instance_id,
                    "model": meta["model"],
                    "seed": meta["params"].get("s"),
                    "dataset": dataset,
                    "grammar_size": size,
                    "time_diff": con_time - unc_time,
                    "time_constrained": con_time,
                    "time_unconstrained": unc_time,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No constrained results found to plot.")
    return df


def plot(
    results_dir: pathlib.Path = RESULTS_DIR,
    output_dir: pathlib.Path | None = None,
    cache_path: pathlib.Path | None = None,
    force: bool = False,
) -> str:
    """
    Create violin plots of (constrained - unconstrained) latency vs grammar size per model.

    Args:
        results_dir: Directory containing compiled result jsonl files.
        output_dir: Directory to store per-model plots (defaults to the current working directory).
        cache_path: Cache file for computed rows (defaults to results dir).
        force: Recompute data even if cache exists.
    """
    results_dir = pathlib.Path(results_dir)
    cache_path = (
        pathlib.Path(cache_path) if cache_path is not None else results_dir / CACHE_NAME
    )
    output_dir = (
        pathlib.Path(output_dir) if output_dir is not None else pathlib.Path.cwd()
    )

    if cache_path.exists() and not force:
        print(f"Reusing cached results in {cache_path}")
        df = pd.read_csv(cache_path)
        if "dataset" not in df.columns:
            # Refresh cache to include dataset info
            force = True

    if not cache_path.exists() or force:
        schema_sizes = _load_schema_sizes()
        df = _compute_rows(results_dir, schema_sizes)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # colors = {
    #    "jsonschema": "#4C72B0",
    #    "jsonschema_r": "#4C72B0",
    #    "THUDM/humaneval-x/cpp": "#DD8452",
    #    "smiles": "#55A868",
    # }

    # Only plot JSON datasets; keep others in the cached CSV
    plot_df = df[df["dataset"].isin(PLOT_DATASETS)]

    models = sorted(plot_df["model"].unique())
    cmap = plt.get_cmap("tab20c")
    colors = {m: cmap((i * 4 + 1) % cmap.N) for i, m in enumerate(models)}

    plt.figure(figsize=(8, 3))
    for model in models:
        model_df = plot_df[plot_df["model"] == model]
        grouped = model_df.groupby("grammar_size")["time_diff"].agg(["median", "count"])
        if grouped.empty:
            continue

        x = grouped.index.to_numpy(dtype=float)
        y = grouped["median"].to_numpy(dtype=float)
        display_name = MODEL_MAP.get(model, model)
        legend_label = rf"\textsc{{{display_name}}}"

        plt.scatter(x, y, color=colors[model], label=legend_label)

        if len(x) >= 3:
            x = np.append(x, [0] * 1000)
            y = np.append(y, [0] * 1000)
            coeffs = np.polyfit(x, y, 2)
            xs = np.linspace(0, 80, 200)
            plt.plot(
                xs,
                np.polyval(coeffs, xs),
                color=colors[model],
                linestyle="--",
                label=None,
            )

    plt.xlabel(r"Grammar size (\# productions)")
    plt.ylabel("Runtime Overhead (s)")
    plt.ylim(0, 3.5)
    plt.xlim(0, 80)
    plt.legend(loc="best")
    plt.tight_layout()

    output_path = output_dir / "jsonschema_time_vs_grammar_size.pdf"
    plt.savefig(output_path, dpi=200)
    plt.close()
    saved.append(output_path)

    return "Saved plots:\n" + "\n".join(str(p) for p in saved)


if __name__ == "__main__":
    fire.Fire(plot)
