import json
import time
from typing import Tuple, List, Optional

import frozendict
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import concurrent.futures

from transformers import AutoTokenizer, AutoModel

from constrained_diffusion.constrain_utils import (
    LEX_MAP,
    compile_lex_map,
    generated_language,
    EOS,
    lex,
    preprocessed_generate_stuff,
    CompiledLexMap,
    LexMap,
    EOSType,
)
from rustformlang.cfg import CFG, is_intersection_empty_threaded


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


def check_valid(
    generated_words: List[str | None | EOSType],
    constraint_lang: CFG,
    lex_map: LexMap,
    terminals: List[str],
    trace: bool = False,
    prelex: str | None = None,
    subtokens: dict[str, list[str]] = frozendict.frozendict(),
    supertokens: dict[str, list[str]] = frozendict.frozendict(),
    strip_chars: str | None = None,
):
    """
    Check if the currently generated tokens are valid according to the constraints.
    """
    generated_lang = generated_language(
        generated_words,
        lex_map,
        terminals,
        trace=trace,
        prelex=prelex,
        #  single_token_lexing=all_possible_lexings,
        #  inject_gap_size=inject_gap_size,
        #  max_total_injections=max_total_injections,
        subtokens=subtokens,
        supertokens=supertokens,
        strip_chars=strip_chars,
    )
    if trace:
        print(
            f"size: O({generated_lang.num_states()}^2 * {constraint_lang.num_productions()})"
        )
    # check if these are prefix of some word
    intersection_empty = is_intersection_empty_threaded(
        constraint_lang, generated_lang, timeout=100
    )
    if trace:
        print("is Intersection empty:", intersection_empty)
    return intersection_empty


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    constraint_lang: CFG,
    lex_map: CompiledLexMap,
    prompt_len: int,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    trace=False,
    prelex: str | None = None,
    inject_gap_size: int = 0,
    max_total_injections: int = 0,
    subtokens: dict[str, list[str]] = frozendict.frozendict(),
    strip_chars: str = None,
    additional_stuff=None,
    constrain: bool = True,
    max_resamples: int = 100,
) -> Tuple[torch.LongTensor, List[Tuple[int, List[Optional[str]]]], bool]:
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        tokenizer: Tokenizer of the model.
        constraint_lang: CFG language to constrain overall generation.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        prelex: Wrap word boundaries with prelex token

    Returns:
        A tuple containing:
        - The generated tokens as a tensor of shape (1, L + gen_length).
        - A list of tuples containing the index of the resampled token and the generated words at that index.
        - A boolean indicating whether the obtained generation is valid according to the constraints.
    """
    start_time = time.monotonic()
    #  -------- constraining code
    if additional_stuff is None and constrain:
        # This allows the user to pre-compute the additional stuff that is prompt-independent
        additional_stuff = preprocessed_generate_stuff(
            tokenizer,
            constraint_lang,
            lex_map,
            trace=trace,
            prelex=prelex,
            subtokens=subtokens,
            strip_chars=strip_chars,
        )
    elif additional_stuff is None:
        additional_stuff = None, None, {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    all_possible_lexings, no_lexing_tokens, supertokens = additional_stuff
    # track the times we have to resample
    resamples = []

    # pre-compute the terminals
    if constrain:
        terminals = constraint_lang.get_terminals()
    else:
        terminals = None
    # map from position in the prompt to the lexings that were ruled out
    ruled_out_lexings = defaultdict(set)

    # --- sampling code

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # ----- constraining code
    generated_words = tokenizer.batch_decode(x.squeeze())
    mask_decoded = tokenizer.decode(mask_id)
    generated_words = [x if x != mask_decoded else None for x in generated_words]
    for num_block in range(num_blocks):
        # --------- sampling code
        block_mask_index = (
            x[
                :,
                prompt.shape[1] + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        complete = False
        for i in range(steps):
            if complete:
                # If the generation is complete, we can stop.
                break
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            # greedy constrained sampling for multi-step
            for j in range(num_transfer_tokens[0, i]):
                if complete:
                    # If the generation is complete, we can stop.
                    break
                mask_index = x == mask_id

                # ----- constraining code
                token_found = False
                num_retries = 0
                while not token_found:
                    # compute the joined mask of all ruled out lexings
                    # no-lexings are ruled out per default
                    # and apply this mask to all positions
                    # TODO can keep this a rolling mask where we just add new ruled out lexing masks
                    if num_retries > 0 and constrain and False:
                        for index_of_new_word in range(logits_with_noise.shape[1]):
                            ruled_out_mask = no_lexing_tokens.copy()
                            # need to take care here that previously we popped some lexings -> these are part of the base mask
                            for lexing in set(
                                ruled_out_lexings[index_of_new_word]
                            ) & set(all_possible_lexings):
                                mask = all_possible_lexings[lexing]
                                ruled_out_mask += mask
                            # make to boolean based on being almost 1
                            ruled_out_mask = np.isclose(ruled_out_mask, 1)
                            # turn into a tensor
                            ruled_out_mask = torch.tensor(
                                ruled_out_mask, dtype=torch.bool, device=logits.device
                            )

                            # pad to the length of the vocab (with True = not allowed)
                            if ruled_out_mask.shape[0] < logits.shape[-1]:
                                # pad to the right
                                # this is needed because we might have less lexings than the vocab size
                                # and we want to mask out the rest
                                ruled_out_mask = F.pad(
                                    ruled_out_mask,
                                    (
                                        0,
                                        logits_with_noise.shape[-1]
                                        - ruled_out_mask.shape[0],
                                    ),
                                    value=True,
                                )
                            # mask out the ruled out lexings
                            logits_with_noise[0][index_of_new_word][
                                ruled_out_mask
                            ] = -np.inf

                    # --------- sampling code
                    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

                    if remasking == "low_confidence":
                        p = F.softmax(logits.to(torch.float64), dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                        )  # b, l
                    elif remasking == "random":
                        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                    else:
                        raise NotImplementedError(remasking)

                    x0_p[
                        :, prompt.shape[1] + (num_block + 1) * block_length :
                    ] = -np.inf

                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, -np.inf)

                    transfer_index = torch.zeros_like(
                        x0, dtype=torch.bool, device=x0.device
                    )

                    j = 0
                    _, select_index = torch.topk(confidence[j], k=1)
                    # ------- constraining code
                    # if everything is now masked out we just return x as-is -> could be that select index is empty
                    if select_index.shape[0] == 0:
                        if trace:
                            print("No tokens to transfer, skipping step")
                        yield x, resamples, False
                        return
                    # if everything is now masked out we just return x as-is
                    index_of_new_word = select_index.item()
                    new_word_vocab_index = x0[0][index_of_new_word]
                    if (
                        logits_with_noise[0][index_of_new_word][new_word_vocab_index]
                        == -np.inf
                    ):
                        if trace:
                            print("No valid token found, returning current x")
                        yield x, resamples, False
                        return
                    eos_token = tokenizer.special_tokens_map["eos_token"]
                    eot_token = "<|eot_id|>"
                    new_word = tokenizer.decode(new_word_vocab_index)
                    if trace:
                        print(
                            f"New word at {index_of_new_word}: {json.dumps(new_word)} ({new_word_vocab_index})"
                        )
                    if new_word in (eos_token, eot_token):
                        # handle specially
                        new_word = EOS

                    generated_words[index_of_new_word] = new_word
                    if constrain:
                        if trace:
                            print("Generating intersection")
                        intersection_empty = check_valid(
                            generated_words[prompt_len:],
                            constraint_lang,
                            lex_map,
                            terminals,
                            trace=trace,
                            prelex=prelex,
                            subtokens=subtokens,
                            supertokens=supertokens,
                            strip_chars=strip_chars,
                        )
                    else:
                        intersection_empty = False
                    if trace:
                        print("is Intersection empty:", intersection_empty)
                    if trace:
                        print(
                            json.dumps(
                                {
                                    "unique_marker_present": True,
                                    "words": [
                                        x if x != EOS else "<EOS>"
                                        for x in generated_words
                                    ],
                                    "new_word": new_word
                                    if new_word != EOS
                                    else "<EOS>",
                                    "is_eos": new_word == EOS,
                                    "index": index_of_new_word,
                                    "accepted": intersection_empty,
                                }
                            )
                        )
                    if intersection_empty:
                        resamples.append(
                            (
                                index_of_new_word,
                                time.monotonic()
                                - start_time,  # [x if x != EOS else "<EOS>" for x in generated_words],
                            )
                        )
                        # generally reject this single token
                        logits_with_noise[0][index_of_new_word][
                            new_word_vocab_index
                        ] = -np.inf

                        if trace:
                            print("Resample")
                        if len(resamples) >= max_resamples:
                            if trace:
                                print("Maximum resamples reached, returning current x")
                            yield x, resamples, False
                            return

                        # generate the intersection lang with all potential lexings and mask out all tokens where the lexing
                        # is categorically ruled out
                        word = generated_words[index_of_new_word]
                        if word != EOS and False:
                            lexings = lex(
                                word, lex_map, is_first=False, strip_chars=strip_chars
                            )
                            subbed_checks = []
                            for lexing in lexings:
                                if lexing in ruled_out_lexings[index_of_new_word]:
                                    # already ruled out
                                    continue
                                if trace:
                                    print(
                                        f"Generating intersection for lexing {lexing}"
                                    )
                                generated_words[index_of_new_word] = lexing
                                lex_check_future = executor.submit(
                                    check_valid,
                                    generated_words[prompt_len:].copy(),
                                    constraint_lang,
                                    lex_map,
                                    terminals,
                                    trace=trace,
                                    prelex=prelex,
                                    subtokens=subtokens,
                                    supertokens=supertokens,
                                    strip_chars=strip_chars,
                                )
                                subbed_checks.append((lexing, lex_check_future))
                            for lexing, lex_check_future in subbed_checks:
                                intersection_empty = lex_check_future.result()
                                if trace:
                                    print(
                                        "is Intersection empty:",
                                        intersection_empty,
                                        lexing,
                                    )
                                if intersection_empty:
                                    # ruled out
                                    ruled_out_lexings[index_of_new_word].add(lexing)

                        generated_words[index_of_new_word] = None
                        num_retries += 1
                        continue
                    token_found = True
                    num_retries = 0
                    transfer_index[0, select_index] = True
                    if new_word is EOS:
                        # automatically fill all words after EOS with EOS
                        transfer_index[0, select_index:] = True
                        x0[0, select_index:] = x0[0, select_index]
                    x[transfer_index] = x0[transfer_index]
                    if EOS in generated_words:
                        no_none_before_eos = (
                            None not in generated_words[: generated_words.index(EOS)]
                        )
                        if no_none_before_eos:
                            complete = True
                            break

                    # ----------- sampling code
                    yield x, resamples, False

    is_complete = (
        EOS in generated_words
        and None not in generated_words[: generated_words.index(EOS)]
    )
    yield x, resamples, is_complete


def main():
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours? "
    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        m, add_generation_prompt=True, tokenize=False
    )
    prompt += "```json"

    cfg_lang = r"""
        S -> { String : Number } lexFence
        Element -> Value
        Value -> Object | Array | String | Number | True | False | Null
        Object -> { Members } | { }
        Members -> Pair | Pair , Members
        Pair -> String : Element
        Array -> [ Elements ] | [ ]
        Elements -> Element | Element , Elements
        String -> lexString
        Number -> lexNumber
        True -> lexTrue
        False -> lexFalse
        Null -> lexNull
    """
    json_cfg = CFG.from_text(cfg_lang, "S")
    main_language_cfg = json_cfg.to_normal_form()
    lex_map = compile_lex_map(LEX_MAP, subtokens={})
    print("Main language empty:", main_language_cfg.is_empty())

    input_ids = tokenizer(prompt)["input_ids"]
    prompt_len = len(input_ids)
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    model = (
        AutoModel.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    out = generate(
        model,
        input_ids,
        tokenizer,
        main_language_cfg,
        prompt_len=prompt_len,
        lex_map=lex_map,
        steps=256,
        gen_length=256,
        block_length=32,
        temperature=1.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        trace=True,
    )
    print(
        tokenizer.batch_decode(out[:, input_ids.shape[1] :], skip_special_tokens=True)[
            0
        ]
    )


if __name__ == "__main__":
    main()
