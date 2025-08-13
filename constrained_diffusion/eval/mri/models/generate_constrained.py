# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import inspect
import json
import logging
import os
import warnings
from typing import Callable, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from transformers import (
    GenerationMixin,
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    PreTrainedModel,
)
from transformers.generation import GenerationMode
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.integrations import is_deepspeed_zero3_enabled, is_fsdp_managed_module
import torch.nn.functional as F

from constrained_diffusion.constrain_utils import (
    preprocessed_generate_stuff,
    EOS,
    interleave_with_value,
    CompiledLexMap,
    lex,
    RustLexMap,
    generated_language,
)
from rustformlang.cfg import CFG

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate(
    model: Union[GenerationMixin, "PreTrainedModel"],
    inputs: Optional[torch.Tensor],
    # list of already fixed parts
    splits: list[str],
    # the index of the gap that is currently being filled (usually 0 = first gap)
    filling_gap_index: int,
    tokenizer,
    constraint_lang: CFG,
    lex_map: CompiledLexMap,
    stopping_token: list[str],
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    use_model_defaults: Optional[bool] = None,
    custom_generate: Optional[str] = None,
    trace: bool = False,
    prelex: Optional[str] = None,
    inject_gap_size: int = 0,
    max_total_injections: int = 0,
    subtokens: dict[str, List[str]] = None,
    strip_chars: str = None,
    additional_stuff=None,
    constrain: bool = True,
    **kwargs,
) -> Tuple[
    Union[GenerateNonBeamOutput, torch.LongTensor],
    List[Tuple[int, List[Optional[str]]]],
    bool,
]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config ([`~generation.GenerationConfig`], *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which has the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complements the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
            sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
            intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        synced_gpus (`bool`, *optional*):
            Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
            to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
            deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The negative prompt needed for some processors such as CFG. The batch size must match the input batch
            size. This is an experimental feature, subject to breaking API changes in future versions.
        negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention_mask for `negative_prompt_ids`.
        use_model_defaults (`bool`, *optional*):
            When it is `True`, unset parameters in `generation_config` will be set to the model-specific default
            generation configuration (`model.generation_config`), as opposed to the global defaults
            (`GenerationConfig()`). If unset, models saved starting from `v4.50` will consider this flag to be
            `True`.
        custom_generate (`str`, *optional*):
            A string containing the name of a huggingface.co repository. If provided, the custom `generate`
            function defined in that reposity's `custom_generate/generate.py` file will be executed instead of the
            standard `generate` method. Note that the logic is for generation is entirely defined in that
            repository, and the return type may be different from the standard `generate` method.
        kwargs (`Dict[str, Any]`, *optional*):
            Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateDecoderOnlyOutput`],
                - [`~generation.GenerateBeamDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
    """
    # 0. If requested, load an arbitrary generation recipe from the Hub and run it instead
    if custom_generate is not None:
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        # Get all `generate` arguments in a single variable. Custom functions are responsible for handling them:
        # they receive the same inputs as `generate`, only with `model` instead of `self`. They can access to
        # methods from `GenerationMixin` through `model`.
        global_keys_to_exclude = {"self", "kwargs"}
        generate_arguments = {
            key: value
            for key, value in locals().items()
            if key not in global_keys_to_exclude
        }
        generate_arguments.update(kwargs)

        custom_generate_function = model.load_custom_generate(
            custom_generate, trust_remote_code=trust_remote_code, **kwargs
        )
        return custom_generate_function(model=model, **generate_arguments)

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    assistant_tokenizer = kwargs.pop(
        "assistant_tokenizer", None
    )  # only used for assisted generation

    generation_config, model_kwargs = model._prepare_generation_config(
        generation_config, use_model_defaults, **kwargs
    )
    model._validate_model_kwargs(model_kwargs.copy())
    model._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        synced_gpus = (
            is_deepspeed_zero3_enabled() or is_fsdp_managed_module(model)
        ) and dist.get_world_size() > 1

    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )

    accepts_attention_mask = "attention_mask" in set(
        inspect.signature(model.forward).parameters.keys()
    )
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    model._prepare_special_tokens(
        generation_config, kwargs_has_attention_mask, device=device
    )

    # decoder-only models must use left-padding for batched generation.
    if not model.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor)
            > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        generation_config.use_cache = True

    if (
        not kwargs_has_attention_mask
        and requires_attention_mask
        and accepts_attention_mask
    ):
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )
    elif kwargs_has_attention_mask:
        # TODO (joao): generalize this check with other types of inputs
        if (
            model_input_name == "input_ids"
            and len(model_kwargs["attention_mask"].shape) > 2
        ):
            raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder:
        input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = (
            inputs_tensor
            if model_input_name == "input_ids"
            else model_kwargs.pop("input_ids")
        )

    if generation_config.token_healing:
        input_ids = model.heal_tokens(input_ids, tokenizer)

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[1]
    has_default_max_length = (
        kwargs.get("max_length") is None and generation_config.max_length is not None
    )
    has_default_min_length = (
        kwargs.get("min_length") is None and generation_config.min_length is not None
    )
    generation_config = model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    # dynamically overrides this value as it can need more than the last token logits
    if model._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
        model_kwargs["logits_to_keep"] = 1

    model._validate_generated_length(
        generation_config, input_ids_length, has_default_max_length
    )

    # 7. Prepare the cache.
    # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
    # - different models have a different cache name expected by the model (default = "past_key_values")
    # - `max_length`, prepared above, is used to determine the maximum cache length
    max_cache_length = generation_config.max_length - 1
    if (
        inputs_tensor.shape[1] != input_ids_length
        and model_input_name == "inputs_embeds"
        and not model.config.is_encoder_decoder
    ):
        max_cache_length += inputs_tensor.shape[1]
    model._prepare_cache_for_generation(
        generation_config,
        model_kwargs,
        assistant_model,
        batch_size,
        max_cache_length,
        device,
    )

    # 8. determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if model.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {model.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{model.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 9. prepare logits processors and stopping criteria
    prepared_logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )
    prepared_stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
        tokenizer=tokenizer,
        **kwargs,
    )

    # Set model_kwargs `use_cache` so we can use it later in forward runs
    model_kwargs["use_cache"] = generation_config.use_cache

    if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=model.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        result = _sample(
            model,
            input_ids,
            splits,
            filling_gap_index,
            tokenizer,
            constraint_lang,
            lex_map,
            stopping_token,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            trace=trace,
            prelex=prelex,
            inject_gap_size=inject_gap_size,
            max_total_injections=max_total_injections,
            subtokens=subtokens,
            strip_chars=strip_chars,
            additional_stuff=additional_stuff,
            constrain=constrain,
            **model_kwargs,
        )

    else:
        raise NotImplementedError(
            "No generation mode is implemented for the given generation configuration."
        )

    return result


def _sample(
    model: GenerationMixin,
    input_ids: torch.LongTensor,
    # list of already fixed parts
    splits: list[str],
    # the index of the gap that is currently being filled (usually 0 = first gap)
    filling_gap_index: int,
    tokenizer,
    constraint_lang: CFG,
    lex_map: RustLexMap,
    stopping_token: list[str],
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    trace: bool = False,
    prelex: Optional[str] = None,
    inject_gap_size: int = 0,
    max_total_injections: int = 0,
    subtokens: dict[str, List[str]] = None,
    strip_chars: str = None,
    additional_stuff=None,
    constrain: bool = True,
    max_resamples: int = 100,
    **model_kwargs,
) -> Tuple[
    Union[GenerateNonBeamOutput, torch.LongTensor],
    List[Tuple[int, List[Optional[str]]]],
    bool,
]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        - A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        - A list of resamples
        - A boolean indicating whether the generation is valid or not

    """
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
    elif not constrain:
        additional_stuff = None, None, {}
    all_possible_lexings, no_lexing_tokens, supertokens = additional_stuff

    resamples = []
    if constrain:
        # pre-compute the terminals
        terminals = constraint_lang.get_terminals()
    else:
        terminals = None

    # --- sampling code
    # init values
    pad_token_id = generation_config._pad_token_tensor
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample
    scores = None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = model._get_initial_cache_position(
        cur_len, input_ids.device, model_kwargs
    )

    model_forward = model.__call__
    compile_forward = model._valid_auto_compile_criteria(
        model_kwargs, generation_config
    )
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        model_forward = model.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = model._prefill_chunking(
            input_ids, generation_config, **model_kwargs
        )
        is_prefill = False
    else:
        is_prefill = True

    # ----- constraining code
    generated_words = splits.copy()
    complete = False
    valid = False
    prev_lang = None
    while model._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        if complete:
            break
        # prepare model inputs
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        if is_prefill:
            outputs = model(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=input_ids.device
        )
        # ----- constraining code
        # keep track of the lexings that are ruled out
        ruled_out_lexings = set()
        token_found = False
        num_iter = 0
        while not token_found:
            if num_iter == 0 or not constrain or True:
                # to ensure we don't change outputs that pass without constraints, we don't modify the logits
                # initially.
                pass
            else:
                # compute the joined mask of all ruled out lexings
                # no-lexings are ruled out per default
                # and apply this mask to all positions
                # TODO can keep this a rolling mask where we just add new ruled out lexing masks
                ruled_out_mask = no_lexing_tokens.copy()
                # need to take care here that previously we popped some lexings -> these are part of the base mask
                for lexing in set(ruled_out_lexings) & set(all_possible_lexings):
                    ruled_out_mask += all_possible_lexings[lexing]
                # make to boolean based on being almost 1
                ruled_out_mask = np.isclose(ruled_out_mask, 1)
                # turn into a tensor
                ruled_out_mask = torch.tensor(
                    ruled_out_mask, dtype=torch.bool, device=next_token_logits.device
                )

                # pad to the length of the vocab (with True = not allowed)
                if ruled_out_mask.shape[0] < next_token_logits.shape[-1]:
                    # pad to the right
                    # this is needed because we might have less lexings than the vocab size
                    # and we want to mask out the rest
                    ruled_out_mask = F.pad(
                        ruled_out_mask,
                        (
                            0,
                            next_token_logits.shape[-1] - ruled_out_mask.shape[0],
                        ),
                        value=True,
                    )
                elif ruled_out_mask.shape[0] > next_token_logits.shape[-1]:
                    # this should not happen, but just in case we have more lexings than the vocab size
                    # we need to cut off the excess
                    ruled_out_mask = ruled_out_mask[: next_token_logits.shape[-1]]
                # mask out the ruled out lexings
                next_token_logits[0][ruled_out_mask] = -np.inf
            num_iter += 1
            # ----- sampling code

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if torch.all(next_token_scores == -np.inf):
                # if all scores are -inf, we cannot sample anything
                if trace:
                    print("All tokens masked out, stopping generation")
                complete = True
                break
            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )
            # ------- constraining code
            # if everything is now masked out we just return x as-is -> could be that select index is empty
            if next_tokens.shape[-1] == 0:
                if trace:
                    print("No tokens to transfer, skipping step")
                complete = True
                break
            # if everything is now masked out we just return x as-is
            new_word_vocab_index = next_tokens
            if not (
                (0 <= new_word_vocab_index.item() < next_token_logits.shape[-1])
                and next_token_logits[0][new_word_vocab_index.item()] != -np.inf
            ):
                if trace:
                    print("No valid token found, returning current x")
                complete = True
                break
            if len(resamples) >= max_resamples:
                if trace:
                    print(f"Max resamples {max_resamples} reached, returning current x")
                complete = True
                break
            eos_token = tokenizer.special_tokens_map.get("eos_token", "<|endoftext|>")
            new_word = tokenizer.decode(new_word_vocab_index)
            if trace:
                print(
                    f"New word at {filling_gap_index}: {json.dumps(new_word)} ({new_word_vocab_index})"
                )
            if new_word in (eos_token, *stopping_token):
                # handle specially
                new_word = EOS

            temp_generated_words = generated_words[:filling_gap_index].copy()
            # if it is EOS connect the current split with the new word
            if new_word is EOS:
                temp_generated_words.append(
                    generated_words[filling_gap_index]
                    + generated_words[filling_gap_index + 1]
                )
                temp_generated_words.extend(generated_words[filling_gap_index + 2 :])
            # otherwise append to the current split
            else:
                temp_generated_words.append(
                    generated_words[filling_gap_index] + new_word
                )
                temp_generated_words.extend(generated_words[filling_gap_index + 1 :])
            if constrain:
                if trace:
                    print("Generating intersection")
                generated_lang = generated_language(
                    interleave_with_value(temp_generated_words, None) + [EOS],
                    lex_map,
                    terminals,
                    # trace=trace,
                    prelex=prelex,
                    # single_token_lexing=all_possible_lexings,
                    # inject_gap_size=inject_gap_size,
                    # max_total_injections=max_total_injections,
                    subtokens=subtokens,
                    supertokens=supertokens,
                    strip_chars=strip_chars,
                )
                if trace:
                    print(
                        f"size: O({generated_lang.num_states()}^2 * {constraint_lang.num_productions()})"
                    )
                if generated_lang != prev_lang:
                    # check if these are prefix of some word
                    intersection_empty = constraint_lang.is_intersection_empty(
                        generated_lang, 300
                    )
                else:
                    # language is the same, so intersection is not empty
                    intersection_empty = False
                if trace:
                    print("is Intersection empty:", intersection_empty)
            else:
                # if we are not constraining, we just assume the intersection is not empty
                generated_lang = None
                intersection_empty = False
            if trace:
                print(
                    json.dumps(
                        {
                            "unique_marker_present": True,
                            "words": [
                                x if x != EOS else "<EOS>" for x in temp_generated_words
                            ],
                            "new_word": new_word if new_word != EOS else "<EOS>",
                            "is_eos": new_word == EOS,
                            "index": filling_gap_index,
                            "accepted": intersection_empty,
                        }
                    )
                )
            if intersection_empty:
                resamples.append(
                    (
                        filling_gap_index,
                        [],  # [x if x != EOS else "<EOS>" for x in generated_words],
                    )
                )
                # generally reject this single token
                next_token_logits[0][new_word_vocab_index.item()] = -np.inf

                if trace:
                    print("Resample")

                # generate the intersection lang with all potential lexings and mask out all tokens where the lexing
                # is categorically ruled out
                word = new_word
                if word is not EOS and False:
                    lexings = lex(
                        word, lex_map, is_first=False, strip_chars=strip_chars
                    )
                    for lexing in lexings:
                        if lexing in ruled_out_lexings:
                            # already ruled out
                            continue
                        temp_interleaved = interleave_with_value(generated_words, None)
                        temp_interleaved.insert(2 * filling_gap_index + 1, lexing)
                        temp_interleaved.append(EOS)
                        if trace:
                            print(f"Generating intersection for lexing {lexing}")
                        generated_lang = generated_language(
                            temp_interleaved,
                            lex_map,
                            terminals,
                            prelex=prelex,
                            # single_token_lexing=all_possible_lexings,
                            # inject_gap_size=inject_gap_size,
                            # max_total_injections=max_total_injections,
                            strip_chars=strip_chars,
                            subtokens=subtokens,
                            supertokens=supertokens,
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
                        if intersection_empty:
                            # ruled out
                            ruled_out_lexings.add(lexing)

                continue
            token_found = True
            prev_lang = generated_lang
            # reset the ruled out lexings for the next step
            generated_words = temp_generated_words

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores
            )
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

            if new_word is EOS:
                complete = True
                valid = True
                # if we hit the EOS token, we stop the generation
                break

    return input_ids, resamples, valid
