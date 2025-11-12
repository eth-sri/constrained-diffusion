import json
from os import makedirs

from collections import defaultdict

import fire
from datasets import load_dataset

import random


def remove_random_spans(text, k, min_span_len=1, max_span_len=10, margin_to_end=0):
    """
    Randomly removes k non-overlapping spans from the text.

    Args:
        text (str): The input text.
        k (int): Number of spans to remove.
        min_span_len (int): Minimum length of a span.
        max_span_len (int): Maximum length of a span.

    Returns:
        tuple: (splits, removed_spans)
            - splits: list of strings (remaining parts of the text)
            - removed_spans: list of strings (the removed spans)
    """
    text_len = len(text)
    if text_len == 0 or k <= 0:
        return [text], []

    # Generate candidate spans
    spans = []
    attempts = 0
    max_attempts = 1000

    while len(spans) < k and attempts < max_attempts:
        span_len = random.randint(
            min(min_span_len, text_len - margin_to_end),
            min(max_span_len, text_len - margin_to_end),
        )
        start = random.randint(0, text_len - span_len - margin_to_end)
        end = start + span_len

        # Check for overlap or touch
        if all(end <= s + 1 or start >= e + 1 for s, e in spans):
            spans.append((start, end))
        attempts += 1

    if len(spans) < k:
        print(f"Warning: Only {len(spans)} non-overlapping spans could be selected.")

    # Sort spans by start index
    spans.sort()

    # Extract removed spans and remaining splits
    removed_spans = [text[start:end] for start, end in spans]
    splits = []
    prev_end = 0
    for start, end in spans:
        splits.append(text[prev_end:start])
        prev_end = end
    splits.append(text[prev_end:])  # Add the final part

    return splits, removed_spans


def remove_random_lines(text: str, k: int) -> tuple[list[str], list[str]]:
    """
    Removes k random lines from the given text.

    Args:
        text (str): The input text.
        k (int): Number of random lines to remove.

    Returns:
        str: The text with k random lines removed.
    """
    # Split text into lines
    lines = [x for x in text.splitlines()]

    # If k is greater than the number of lines, remove all lines
    k = min(k, len(lines))

    # Filter empty lines
    eligible_indices = [i for i, line in enumerate(lines) if line.strip()]
    if len(eligible_indices) < k:
        return [text], []
    # Randomly choose k line indices to remove
    indices_to_remove = sorted(random.sample(eligible_indices, k))

    # Join remaining lines back into a string
    splits = []
    removed = []
    prev_i = 0
    for i in indices_to_remove:
        if i == prev_i and removed:
            removed[-1] += "\n" + lines[i]
        else:
            splits.append("\n" + "\n".join(lines[prev_i:i]) + "\n")
            removed.append(lines[i])
        prev_i = i + 1
    splits.append("\n" + "\n".join(lines[prev_i:]))
    splits[0].removeprefix("\n")
    return splits, removed


def main(upto=3, seed=0, remove_whole_line=True):
    """
    Removes n random spans from the canonical solution to generate a new instance of the Humaneval dataset.

    Generates several splits with increasing number of spans removed
    """
    random.seed(seed)

    humaneval_dataset = load_dataset("THUDM/humaneval-x", "cpp", split="test")
    # map from # spans removed to a list of instances
    new_instances = defaultdict(list)
    for instance in humaneval_dataset:
        task_id = instance["task_id"]
        canonical_solution = instance["canonical_solution"]

        for i in range(1, upto + 1):
            # remove i random spans from the canonical solution
            new_solution = canonical_solution
            if remove_whole_line:
                splits, removed_spans = remove_random_lines(new_solution, i)
                if len(removed_spans) == 0:
                    continue
            else:
                splits, removed_spans = remove_random_spans(
                    new_solution, i, min_span_len=5, max_span_len=100, margin_to_end=0
                )
                if len(removed_spans) < i:
                    continue
            # create a new instance
            new_instance = instance.copy()
            slug = "spans" if not remove_whole_line else "lines"
            new_instance["task_id"] = f"{task_id}_{slug}_{i}"
            new_instance["splits"] = splits
            new_instance["removed_spans"] = removed_spans
            new_instances[i].append(new_instance)
    # save new instances to json files
    makedirs("dataset", exist_ok=True)
    for i in range(1, upto + 1):
        with open(f"dataset/humaneval_cpp_{i}_{slug}.jsonl", "w") as f:
            for instance in new_instances[i]:
                f.write(json.dumps(instance) + "\n")
    print(
        f"Generated new instances with up to {upto} random spans removed from the Humaneval dataset."
    )


if __name__ == "__main__":
    fire.Fire(main)
