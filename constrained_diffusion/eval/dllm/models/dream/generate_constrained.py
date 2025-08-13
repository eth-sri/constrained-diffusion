# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List
import concurrent.futures

import numpy as np
import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

from constrained_diffusion.constrain_utils import (
    preprocessed_generate_stuff,
    EOS,
    generated_language,
    lex,
    CompiledLexMap,
)
from constrained_diffusion.eval.dllm.models.llada.generate_constrained import (
    check_valid,
)
from rustformlang.cfg import CFG

logger = logging.get_logger(__name__)


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(
    logits,
    temperature=0.0,
    top_p=None,
    top_k=None,
    margin_confidence=False,
    neg_entropy=False,
):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", "origin")
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop(
            "return_dict_in_generate", False
        )
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass


@torch.no_grad()
def diffusion_generate(
    model,
    tokenizer,
    constraint_lang: CFG,
    lex_map: CompiledLexMap,
    prompt_len: int,
    subtokens: Optional[Dict[str, list[str]]],
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[DreamGenerationConfig] = None,
    trace: bool = False,
    prelex: str = None,
    strip_chars: Optional[str] = None,
    additional_stuff: Optional[Tuple] = None,
    inject_gap_size: int = 0,
    max_total_injections: int = 0,
    constrain: bool = True,
    **kwargs,
) -> Tuple[
    Union[DreamModelOutput, torch.LongTensor],
    List[Tuple[int, List[Optional[str]]]],
    bool,
]:
    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    generation_config = model._prepare_generation_config(generation_config, **kwargs)
    generation_tokens_hook_func = kwargs.pop(
        "generation_tokens_hook_func", lambda step, x, logits: x
    )
    generation_logits_hook_func = kwargs.pop(
        "generation_logits_hook_func", lambda step, x, logits: logits
    )

    # 2. Define model inputs
    assert inputs is not None
    input_ids = inputs
    device = input_ids.device
    attention_mask = kwargs.pop("attention_mask", None)
    model._prepare_special_tokens(generation_config, device=device)

    # 3. Prepare `max_length`.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = (
        kwargs.get("max_length") is None and generation_config.max_length is not None
    )
    generation_config = model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        input_ids_length=input_ids_length,
    )

    model._validate_generated_length(
        generation_config, input_ids_length, has_default_max_length
    )

    # 4. Check input_ids
    if not is_torchdynamo_compiling() and model.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {model.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{model.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )
    if (
        hasattr(generation_config, "pad_token_id")
        and torch.any(input_ids == generation_config.pad_token_id)
        and attention_mask is None
    ):
        warnings.warn(
            "Padding was detected but no attention mask is passed here. For correct "
            "generation results, please set `attention_mask` when batch-padding inputs.",
            UserWarning,
        )

    input_ids, attention_mask = model._expand_inputs_for_generation(
        expand_size=generation_config.num_return_sequences,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    result = _sample(
        model,
        input_ids,
        tokenizer=tokenizer,
        constraint_lang=constraint_lang,
        lex_map=lex_map,
        prompt_len=prompt_len,
        attention_mask=attention_mask,
        generation_config=generation_config,
        generation_tokens_hook_func=generation_tokens_hook_func,
        generation_logits_hook_func=generation_logits_hook_func,
        trace=trace,
        prelex=prelex,
        subtokens=subtokens,
        strip_chars=strip_chars,
        additional_stuff=additional_stuff,
        inject_gap_size=inject_gap_size,
        max_total_injections=max_total_injections,
        constrain=constrain,
    )
    return result


def _sample(
    model,
    input_ids: torch.LongTensor,
    tokenizer,
    constraint_lang: CFG,
    lex_map: CompiledLexMap,
    prompt_len: int,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    generation_tokens_hook_func,
    generation_logits_hook_func,
    subtokens: Optional[Dict[str, list[str]]],
    trace: bool = False,
    prelex: str = None,
    strip_chars: Optional[str] = None,
    additional_stuff: Optional[Tuple] = None,
    inject_gap_size: int = 0,
    max_total_injections: int = 0,
    max_resamples: int = 100,
    constrain: bool = True,
) -> Tuple[
    Union[DreamModelOutput, torch.LongTensor],
    List[Tuple[int, List[Optional[str]]]],
    bool,
]:
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
    if constrain:
        all_possible_lexings, no_lexing_tokens, supertokens = additional_stuff
    else:
        all_possible_lexings, no_lexing_tokens, supertokens = None, None, None
    # track the times we have to resample
    resamples = []

    # pre-compute the terminals
    if constrain:
        terminals = constraint_lang.get_terminals()
    else:
        terminals = None
    # map from position in the prompt to the lexings that were ruled out
    ruled_out_lexings = defaultdict(set)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

    # --- sampling code
    # init values
    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    mask_token_id = generation_config.mask_token_id
    steps = generation_config.steps
    eps = generation_config.eps
    alg = generation_config.alg
    alg_temp = generation_config.alg_temp
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k

    histories = [] if (return_dict_in_generate and output_history) else None

    # pad input_ids to max_length
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        # we do not mask the [MASK] tokens so value = 1.0
        attention_mask = F.pad(
            attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0
        )
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        # attention_mask is of shape [B, N]
        # broadcast to [B, 1, N, N]
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
    else:
        tok_idx = None
        attention_mask = "full"

    timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

    # this allows user-defined token control of the intermediate steps
    x = generation_tokens_hook_func(None, x, None)
    # ----- constraining code
    generated_words = tokenizer.batch_decode(x.squeeze())
    mask_decoded = tokenizer.decode(mask_token_id)
    generated_words = [x if x != mask_decoded else None for x in generated_words]
    complete = False
    for i in range(steps):
        if complete:
            break
        # ----- sampling code
        logits = model(x, attention_mask, tok_idx).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # this allows user-defined logits control of the intermediate steps
        logits = generation_logits_hook_func(i, x, logits)

        t = timesteps[i]
        s = timesteps[i + 1]
        mask_index = x == mask_token_id
        num_mask_token = mask_index.sum() / mask_index.shape[0]
        number_transfer_tokens_overall = (
            int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
        )

        for _ in range(number_transfer_tokens_overall):
            if complete:
                break
            number_transfer_tokens = 1
            mask_index = x == mask_token_id

            # ----- constraining code
            tokens_found = False
            num_retries = 0
            while not tokens_found:
                # compute the joined mask of all ruled out lexings
                # no-lexings are ruled out per default
                # and apply this mask to all positions
                # TODO can keep this a rolling mask where we just add new ruled out lexing masks
                if num_retries > 0 and constrain and False:
                    for index_of_new_word in range(logits.shape[1]):
                        ruled_out_mask = no_lexing_tokens.copy()
                        # need to take care here that previously we popped some lexings -> these are part of the base mask
                        for lexing in set(ruled_out_lexings[index_of_new_word]) & set(
                            all_possible_lexings
                        ):
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
                                (0, logits.shape[-1] - ruled_out_mask.shape[0]),
                                value=True,
                            )
                        # mask out the ruled out lexings
                        logits[0][index_of_new_word][ruled_out_mask] = -np.inf
                # --------- sampling code
                mask_logits = logits[mask_index]

                if alg == "origin":
                    raise NotImplementedError("we do not support origin alg yet")
                else:
                    if alg == "maskgit_plus":
                        confidence, x0 = sample_tokens(
                            mask_logits,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                        )
                    elif alg == "topk_margin":
                        if number_transfer_tokens > 1:
                            raise NotImplementedError(
                                "topk_margin alg only supports number_transfer_tokens == 1"
                            )
                        confidence, x0 = sample_tokens(
                            mask_logits,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            margin_confidence=True,
                        )
                    elif alg == "entropy":
                        confidence, x0 = sample_tokens(
                            mask_logits,
                            temperature,
                            top_p=top_p,
                            top_k=top_k,
                            neg_entropy=True,
                        )
                    else:
                        raise RuntimeError(f"Unknown alg: {alg}")
                    full_confidence = torch.full_like(
                        x, -torch.inf, device=model.device, dtype=logits.dtype
                    )
                    full_confidence[mask_index] = confidence
                    if number_transfer_tokens == 0:
                        tokens_found = True
                        num_retries = 0
                        continue
                    if number_transfer_tokens > 0:
                        # TODO for now we assume number_transfer_tokens == 1
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(
                                full_confidence, number_transfer_tokens
                            )
                        else:
                            full_confidence = full_confidence / alg_temp
                            full_confidence = F.softmax(full_confidence, dim=-1)
                            transfer_index = torch.multinomial(
                                full_confidence, num_samples=number_transfer_tokens
                            )
                        x_ = (
                            torch.zeros_like(x, device=model.device, dtype=torch.long)
                            + mask_token_id
                        )
                        x_[mask_index] = x0.clone()
                        row_indices = (
                            torch.arange(x.size(0), device=model.device)
                            .unsqueeze(1)
                            .expand_as(transfer_index)
                        )
                        # -------------- constraining code
                        # check for all transfer indices if they are valid
                        if trace:
                            print(f"Transfer indices: {transfer_index.item()}")
                        words_at_transfer_index = x_[
                            row_indices, transfer_index
                        ].clone()
                        eos_token = tokenizer.special_tokens_map["eos_token"]
                        new_words = tokenizer.batch_decode(
                            words_at_transfer_index, skip_special_tokens=False
                        )
                        if trace:
                            print(f"words: {new_words}")
                        for i, word in enumerate(new_words):
                            if word in (eos_token, "<|im_end|>", "<|dlm_pad|>"):
                                new_words[i] = EOS
                        # find the indices of the nonzero elements in the mask
                        for pos, index in enumerate(transfer_index[-1]):
                            generated_words[index.item()] = new_words[pos]
                        # TODO part of =1 assumption
                        index_of_new_word = transfer_index[-1].item()
                        if constrain:
                            generated_lang = generated_language(
                                generated_words[prompt_len:],
                                lex_map,
                                terminals,
                                prelex=prelex,
                                single_token_lexing=all_possible_lexings,
                                inject_gap_size=inject_gap_size,
                                max_total_injections=max_total_injections,
                                subtokens=subtokens,
                                supertokens=supertokens,
                                strip_chars=strip_chars,
                                trace=trace,
                            )
                            if trace:
                                print(
                                    f"size: O({generated_lang.num_states()}^2 * {constraint_lang.num_productions()})"
                                )
                            # check if these are prefix of some word
                            intersection_empty = constraint_lang.is_intersection_empty(
                                generated_lang, 100
                            )
                            if trace:
                                print("is Intersection empty:", intersection_empty)
                        else:
                            intersection_empty = False
                        if trace:
                            print(
                                json.dumps(
                                    {
                                        "unique_marker_present": True,
                                        "words": [
                                            x if x != EOS else "<EOS>"
                                            for x in generated_words
                                        ],
                                        "new_word": new_words[0]
                                        if new_words[0] is not EOS
                                        else "<EOS>",
                                        "is_eos": new_words[0] is EOS,
                                        "index": index_of_new_word,
                                        "accepted": intersection_empty,
                                    }
                                )
                            )
                        if intersection_empty and constrain:
                            resamples.append(
                                (
                                    index_of_new_word,
                                    time.monotonic() - start_time,  # [
                                    #     x if x != EOS else "<EOS>"
                                    #     for x in generated_words
                                    # ],
                                )
                            )
                            # generally reject this single token
                            logits[-1][index_of_new_word][
                                # TODO part of =1 assumption
                                words_at_transfer_index[-1]
                            ] = -np.inf

                            if trace:
                                print("Resample")
                            if len(resamples) >= max_resamples:
                                if trace:
                                    print(
                                        "Maximum resamples reached, returning current x"
                                    )
                                yield x, resamples, False
                                return

                            # generate the intersection lang with all potential lexings and mask out all tokens where the lexing
                            # is categorically ruled out
                            word = generated_words[index_of_new_word]
                            if word != EOS and False:
                                lexings = lex(
                                    word,
                                    lex_map,
                                    is_first=False,
                                    strip_chars=strip_chars,
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

                            # TODO part of =1 assumption
                            generated_words[index_of_new_word] = None
                            num_retries += 1
                            continue
                        # some opts for EOS
                        tokens_found = True
                        num_retries = 0
                        if generated_words[index_of_new_word] is EOS:
                            # automatically fill all words after EOS with EOS
                            transfer_index = torch.cat(
                                (
                                    transfer_index,
                                    torch.arange(
                                        index_of_new_word + 1,
                                        len(generated_words),
                                        device=model.device,
                                    ).unsqueeze(0),
                                ),
                                dim=-1,
                            )
                            x_[row_indices, index_of_new_word + 1 :] = x_[
                                row_indices, index_of_new_word
                            ].clone()
                            for gen_word_ind in range(
                                index_of_new_word + 1, len(generated_words)
                            ):
                                generated_words[gen_word_ind] = EOS
                        # ---------- sampling code
                        x[row_indices, transfer_index] = x_[row_indices, transfer_index]
                        # ---------- constraining code
                        if EOS in generated_words:
                            no_none_before_eos = (
                                None
                                not in generated_words[: generated_words.index(EOS)]
                            )
                            if no_none_before_eos:
                                complete = True
                                break

            yield x, resamples, False

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())

    is_complete = (
        EOS in generated_words
        and None not in generated_words[: generated_words.index(EOS)]
    )
    yield x, resamples, is_complete
