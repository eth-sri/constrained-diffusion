"""
Sample from the model constrained or unconstrained

Can also simulate a repair setting
"""

import json
import os
import time
import traceback
import random

import fire
import numpy as np
import torch
from tqdm import tqdm

from constrained_diffusion.constrain_utils import (
    compile_lex_map,
    preprocessed_generate_stuff,
)
from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
from rustformlang.cfg import CFG


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(
    model_name="GSAI-ML/LLaDA-8B-Instruct",
    dataset_name="nuprl/MultiPL-E",
    device="cuda",
    temp=0,
    seed=0,
    max_tokens=128,
    timeout=60,
    output_file="/dev/stdout",
    trace=False,
    constrained=False,
    limit=1000,
    task_id=None,
    steps=128,
    max_total_injections=0,
    inject_gap_size=0,
):
    if isinstance(task_id, int):
        task_id = str(task_id)
    if isinstance(task_id, str):
        # task ids always converted to a tuple of taskids
        task_id = (task_id,)

    dataset = load_dataset(dataset_name)
    eval_model = load_model(model_name)

    # load already inferred stuff
    already_done = set()
    if os.path.exists(output_file) and output_file not in ("/dev/stdout", "-"):
        with open(output_file, "r") as f:
            for i, line in enumerate(f):
                output = json.loads(line)
                already_done.add(output["instance_id"])

    tokenizer = None
    model = None
    lang, lex_map, subtokens = None, None, None
    prelex = None
    additional_stuff = None

    # run through all instances
    for instance in tqdm(sorted(dataset, key=lambda x: x.instance_id())[:limit]):
        if instance.instance_id() in already_done and task_id is None:
            continue
        if task_id is not None and not any(
            tid in instance.instance_id() for tid in task_id
        ):
            continue
        if tokenizer is None or model is None:
            tokenizer = eval_model.tokenizer(device)
            model = eval_model.model(device)

        # load the language to be used
        if constrained and (lang is None or dataset.different_grammar_per_instance):
            lang, lex_map, subtokens = instance.language_lex_subtokens()
            orig_lex_map = lex_map
            lang = lang.concatenate(CFG.from_text("S -> lexFence | $", "S"))
            if (
                instance.strip_chars() is not None
                and "\n" not in instance.strip_chars()
            ):
                # if the language does not allow whitespace, we need to explicitly add a newline
                lex_map["lexFence"] = r"\n?```"
            else:
                lex_map["lexFence"] = "```"
            lang = lang.to_normal_form()
            assert (
                not lang.is_empty()
            ), "Language is empty, check the dataset and lex map"
            lex_map = compile_lex_map(lex_map, subtokens=subtokens)
            additional_stuff = None
            prelex = instance.prelex()
        if constrained and additional_stuff is None:
            additional_stuff = preprocessed_generate_stuff(
                tokenizer,
                lang,
                lex_map,
                trace=trace,
                prelex=prelex,
                subtokens=subtokens,
                strip_chars=instance.strip_chars(),
            )

        set_seed(seed)
        start = time.monotonic()
        gen_length = max_tokens
        if not constrained:
            prompt, code, extracted, timed_out = eval_model.generate_unconstrained(
                instance,
                model,
                tokenizer,
                steps=steps,
                gen_length=gen_length,
                temperature=temp,
                trace=trace,
            )
            autocompletion_raw = None
            autocompletion = None
            time_taken_autocompletion = None
            resamples = None
            code_raw = None
        else:
            if trace:
                print("Lang is empty:", lang.is_empty())
            try:
                (
                    prompt,
                    code,
                    code_raw,
                    extracted,
                    timed_out,
                    resamples,
                    autocompletion_raw,
                    autocompletion,
                    time_taken_autocompletion,
                ) = eval_model.generate_constrained(
                    instance,
                    model,
                    tokenizer,
                    steps=steps,
                    gen_length=gen_length,
                    temperature=temp,
                    lang=lang,
                    lex_map=lex_map,
                    subtokens=subtokens,
                    additional_stuff=additional_stuff,
                    max_total_injections=max_total_injections,
                    inject_gap_size=inject_gap_size,
                    prelex=prelex,
                    timeout=timeout,
                    trace=trace,
                    orig_lex_map=orig_lex_map,
                )
            except Exception as e:
                print(
                    f"Error generating code for instance {instance.instance_id()}: {e}"
                )
                traceback.print_exc()
                continue

        if trace:
            print(code)
        end = time.monotonic()
        time_taken = end - start
        specs = {
            "dataset": dataset_name,
            "instance_id": instance.instance_id(),
            "prompt": prompt,
            "constrained": constrained,
            "model_name": model_name,
            "temp": temp,
            "max_tokens": max_tokens,
            "time_taken": time_taken,
            "code": code,
            "code_raw": code_raw,
            "extracted": extracted,
            "trace": trace,
            "timeout": timeout,
            "seed": seed,
            "timed_out": timed_out,
            "resamples": resamples,
            "autocompletion_raw": autocompletion_raw,
            "autocompletion": autocompletion,
            "time_taken_autocompletion": time_taken_autocompletion,
        }
        try:
            with open(output_file, "a") as f:
                print(
                    json.dumps(
                        specs,
                    ),
                    flush=True,
                    file=f,
                )
        except Exception:
            print("WARNING CATASROPHIC FAILURE")
            print("RESULTS ARE NOT WRITTEN TO FILE")
            traceback.print_exc()
            print("", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
