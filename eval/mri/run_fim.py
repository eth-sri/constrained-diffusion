import subprocess
import time
from math import ceil
from pathlib import Path

from constrained_diffusion.eval.mri.models.fim import (
    CODEGEMMA_7B_MODEL,
    DEEPSEEK_CODER_7B_MODEL,
    STARCODER_2_7B_MODEL,
    DEEPSEEK_CODER_1B_MODEL,
    DEEPSEEK_CODER_33B_MODEL,
)

GPUS = list(range(8))  # Example list of GPUs, adjust based on your system
N = 1  # Set the max number of allowed processes per GPU
GPUSIZE = 48
models = [
    (DEEPSEEK_CODER_1B_MODEL, 1 * 2),
    (DEEPSEEK_CODER_7B_MODEL, 7 * 2),
    (DEEPSEEK_CODER_33B_MODEL, 33 * 2),
    (CODEGEMMA_7B_MODEL, 7),
    (STARCODER_2_7B_MODEL, 7),
]


def compute_needed_gpus(size_model, size_gpu):
    return (size_model * 2 * 1.2) / size_gpu


subsets = [
    "HumanEval/FIM/cpp/1",
    "HumanEval/FIM/cpp/2",
    "HumanEval/FIM/cpp/3",
]
temps = ["1"]
gap_sizes = {
    "HumanEval/FIM/cpp/1": [0],
    "HumanEval/FIM/cpp/2": [0],
    "HumanEval/FIM/cpp/3": [0],
}
seeds = [0, 1, 2, 3]
configs = [
    ("", "_synth"),
]
constraineds = [True]


def find_available_gpus(gpus, n):
    found_gpus = []
    for gpu in gpus:
        process_count = int(
            subprocess.check_output(
                [
                    "/bin/bash",
                    "-c",
                    f"nvidia-smi -i {gpu} --query-compute-apps=pid --format=csv,noheader | wc -l",
                ],
            ).strip()
        )
        if process_count < n:
            found_gpus.append(gpu)
    return found_gpus


total_configs = []
for seed in seeds:
    for subset in subsets:
        for gap_size in gap_sizes[subset]:
            for temp in temps:
                for config, name in configs:
                    for constrained in constraineds:
                        for model, model_size in models:
                            if not constrained and gap_size != 0:
                                local_gap_size = 0
                            else:
                                local_gap_size = gap_size
                            overall_config = (
                                seed,
                                temp,
                                config,
                                name,
                                constrained,
                                model,
                                model_size,
                                subset,
                                local_gap_size,
                            )
                            if overall_config not in total_configs:
                                total_configs.append(overall_config)


remaining_configs = total_configs.copy()
running_configs = list()
while remaining_configs or running_configs:
    # reinsert crashed programs
    for config, pipe in running_configs:
        if pipe.poll() is not None:
            running_configs.remove((config, pipe))
            if pipe.returncode != 0:
                remaining_configs.append(config)
    cuda_devices, needed_gpus = [], 1
    cuda_devices = find_available_gpus(GPUS, N)
    total_config = None
    for total_config in remaining_configs:
        (
            seed,
            temp,
            config,
            name,
            constrained,
            model,
            model_size,
            subset,
            gap_size,
        ) = total_config
        needed_gpus = compute_needed_gpus(model_size, GPUSIZE)
        if needed_gpus > len(GPUS):
            print(f"model {model} is too large, skipping")
            remaining_configs.remove(total_config)
            continue
        if len(cuda_devices) >= needed_gpus:
            break
    if len(cuda_devices) < needed_gpus or total_config is None:
        print("No available GPU found or all configs running. Waiting...")
        time.sleep(60)
        continue
    remaining_configs.remove(total_config)
    cuda_devices = cuda_devices[: int(ceil(needed_gpus))]

    if constrained:
        suffix = "c"
    else:
        suffix = "nc"
    command = (
        f"PYTHONPATH=. CUDA_VISIBLE_DEVICES={','.join(str(i) for i in cuda_devices)} python3 -m constrained_outoforder.eval.mri.generic_inference "
        f"--max-tokens 256 --model_name {model} --seed {seed} --temp {temp} --trace False  --dataset-name '{subset}' --inject_gap_size {gap_size} --max_total_injections {gap_size} "
        f"--constrained {constrained} --output_file 'results/{subset.replace('/', '_')}_{model.replace('/', '_')}_s={seed}_t={temp}_gs={gap_size}{name}_{suffix}.jsonl' {config}"
    )
    print("+ " + command, flush=True)
    pipe = subprocess.Popen(
        ["/bin/bash", "-c", command],
        cwd=str(Path(__file__).parent.parent.parent),
    )
    running_configs.append((total_config, pipe))
    time.sleep(60)
