import re
import time

import stopit
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from constrained_diffusion.constrain_utils import (
    partial_output_from_tokens,
    autocomplete_valid,
    generated_language,
    EOS,
    reconstruct_word_boundaries,
    interleave_with_value,
)
from constrained_diffusion.eval.mri.datasets.generic import Instance
from constrained_diffusion.eval.mri.models.generic import Model
from constrained_diffusion.eval.mri.models.generate_constrained import (
    generate as generate_constrained,
)
from constrained_diffusion.eval.mri.models.util import strip_first_multiline_comment
from rustformlang.cfg import CFG

DEEPSEEK_CODER_1B_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"
DEEPSEEK_CODER_7B_MODEL = "deepseek-ai/deepseek-coder-6.7b-base"
DEEPSEEK_CODER_33B_MODEL = "deepseek-ai/deepseek-coder-33b-base"
CODEGEMMA_7B_MODEL = "google/codegemma-7b"
STARCODER_2_7B_MODEL = "bigcode/starcoder2-7b"
CODELLAMA_7B_MODEL = "codellama/CodeLlama-7b-hf"


def apply_prefix_template(model: str, prefix: str, suffix: str):
    if model in (
        DEEPSEEK_CODER_33B_MODEL,
        DEEPSEEK_CODER_7B_MODEL,
        DEEPSEEK_CODER_1B_MODEL,
    ):
        # DeepSeek formatting
        return f"<｜mri▁begin｜>{prefix}<｜mri▁hole｜>{suffix}<｜mri▁end｜>"
    elif model == STARCODER_2_7B_MODEL:
        # Starcoder formatting
        return f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
    elif model == CODELLAMA_7B_MODEL:
        # CodeLlama formatting
        return f"<PRE>{prefix}<SUF>{suffix}<MID>"
    elif model == CODEGEMMA_7B_MODEL:
        # CodeGemma formatting
        return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    else:
        raise ValueError(f"Unsupported model: {model}")


def extract_middle(model: str, code: str):
    """
    Extracts the generated middle part of the code
    """
    if model == STARCODER_2_7B_MODEL:
        end_markers = ("<file_sep>", "<|endoftext|>", "<fim_middle>")
    elif model in (
        DEEPSEEK_CODER_33B_MODEL,
        DEEPSEEK_CODER_7B_MODEL,
        DEEPSEEK_CODER_1B_MODEL,
    ):
        end_markers = ("<｜end▁of▁sentence｜>", "<|endoftext|>")
    elif model == CODELLAMA_7B_MODEL:
        end_markers = ("<file_sep>", "</s>", "<MID>")
    elif model == CODEGEMMA_7B_MODEL:
        end_markers = ("<|file_separator|>", "<eos>")
    else:
        raise NotImplementedError(f"Unsupported model: {model}")
    for end_marker in end_markers:
        end = code.find(end_marker)
        if end != -1:
            return code[:end]
    return code


class FimModel(Model):
    """
    Base class models that support FIM
    """

    def __init__(self, model):
        """
        Initializes the PrefSuff model with the specified model name.
        """
        self._model = model

    def tokenizer(self, device):
        """
        Returns the tokenizer for the model.
        """
        tokenizer = AutoTokenizer.from_pretrained(self._model, trust_remote_code=True)
        return tokenizer

    def model(self, device):
        """
        Returns the model for the specified device.
        """
        model = AutoModelForCausalLM.from_pretrained(
            self._model,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else device,
        )
        return model

    def generate_unconstrained(
        self,
        instance: Instance,
        model,
        tokenizer,
        gen_length: int,
        temperature: int,
        trace: bool = False,
    ) -> tuple[str, str, str, bool, int]:
        """
        Generates a response from the model based on the provided instance.
        Returns
        - The derived prompt for the model
        - The generated code as a string.
        - The extracted result from the instance.
        - A boolean indicating whether the generation timed out.
        """
        return self.generate_constrained(
            instance,
            model,
            tokenizer,
            gen_length,
            temperature,
            lang=None,  # No language constraints for unconstrained generation
            lex_map=None,  # No lexicon map for unconstrained generation
            orig_lex_map=None,  # No original lexicon map for unconstrained generation
            subtokens=None,  # No subtokens for unconstrained generation
            additional_stuff=None,  # No additional stuff for unconstrained generation
            max_total_injections=0,  # No injections for unconstrained generation
            inject_gap_size=0,  # No gap size for unconstrained generation
            prelex=None,  # No prelex for unconstrained generation
            timeout=60,  # Default timeout
            trace=trace,
            constrain=False,  # Not constrained
        )[:5]  # Return only the first four elements of the tuple

    def generate_constrained(
        self,
        instance: Instance,
        model,
        tokenizer,
        gen_length: int,
        temperature: int,
        lang: CFG,
        lex_map,
        orig_lex_map,
        subtokens,
        additional_stuff,
        max_total_injections: int = 0,
        inject_gap_size: int = 0,
        prelex: str | None = None,
        timeout: int = 60,
        trace: bool = False,
        constrain: bool = True,
    ) -> tuple[str, str, str, bool, int, list, str, str, float]:
        """
        Generates a response from the model based on the provided instance.
        Returns
        - The derived prompt for the model
        - The generated code as a string.
        - The extracted result from the instance.
        - A boolean indicating whether the generation timed out.
        - The total number of tokens generated.
        - A list of resamplings for each part of the code.
        - The fully generated code using autocompletion
        - The extracted result from the fully generated code with autocompletion
        - The time taken for generating the autocompletions

        """
        splits = instance.splits()

        completion = ""
        all_outputs = []
        prompts = []
        all_resamplings = []
        total_generated = 0
        intersection_time = 0
        completion_no_auto = None
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            for i, part in enumerate(splits):
                completion += part
                if i >= len(splits) - 1:
                    break
                code = apply_prefix_template(
                    self._model, completion, "<TODO>".join(splits[i + 1 :])
                )
                if trace:
                    print(f"Prompt for part {i}: {code}")
                prompts.append(code)
                inputs = tokenizer(code, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs, resamplings, valid = generate_constrained(
                        model,
                        inputs=inputs.input_ids,
                        splits=[strip_first_multiline_comment(completion)[0]]
                        + splits[i + 1 : -1]
                        + [splits[-1][: -len(instance._extra_suffix)]],
                        filling_gap_index=0,
                        tokenizer=tokenizer,
                        constraint_lang=lang,
                        lex_map=lex_map,
                        stopping_token=[
                            "<file_sep>",
                            "<｜end▁of▁sentence｜>",
                            "<|file_separator|>",
                            "</s>",
                            "<eos>",
                            "<fim_middle>",
                        ],
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=gen_length,
                        trace=trace,
                        prelex=prelex,
                        inject_gap_size=inject_gap_size,
                        max_total_injections=max_total_injections,
                        subtokens=subtokens,
                        strip_chars=instance.strip_chars(),
                        additional_stuff=additional_stuff,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        num_beams=1,
                        constrain=constrain,
                    )
                locally_generated = outputs.shape[-1] - inputs.input_ids.shape[-1]
                total_generated += locally_generated
                code_output = tokenizer.decode(
                    outputs[-1, inputs.input_ids.shape[-1] :], skip_special_tokens=False
                )
                if trace:
                    print(
                        f"************ Output for part {i} *************\n{code_output}"
                    )
                all_outputs.append(code_output)
                extracted_code = extract_middle(self._model, code_output)
                if trace:
                    print(
                        f"************* Extracted code for part {i} ************\n{extracted_code}"
                    )
                all_resamplings.append(resamplings)
                completion += extracted_code
                if not valid and constrain:
                    # mark the completion so far without auto
                    if completion_no_auto is None:
                        completion_no_auto = completion

                    # derive a valid completion and replace completion
                    start_time = time.monotonic()
                    completion_no_comment, first_multiline_comment = (
                        strip_first_multiline_comment(completion)
                    )
                    generated_words = (
                        [completion_no_comment]
                        + splits[i + 1 : -1]
                        + [splits[-1][: -len(instance._extra_suffix)]]
                    )
                    generated_words = interleave_with_value(generated_words, None) + [
                        EOS
                    ]
                    partial_output, first_token_gap, last_token_eos_adj = (
                        partial_output_from_tokens(generated_words, prelex)
                    )
                    if trace:
                        print("******* Generated words *********\n", generated_words)
                        print("******** Partial output **********\n", partial_output)
                    _, _, supertokens = additional_stuff
                    valid_completion = autocomplete_valid(
                        partial_output=partial_output,
                        first_token_gap=first_token_gap,
                        last_token_eos_adj=last_token_eos_adj,
                        generated_lang=generated_language(
                            generated_words,
                            lex_map,
                            lang.get_terminals(),
                            trace=trace,
                            prelex=prelex,
                            subtokens=subtokens,
                            supertokens=supertokens,
                            strip_chars=instance.strip_chars(),
                        ),
                        subtokens=subtokens,
                        lex_map=orig_lex_map,
                        constraint_lang=lang,
                        trace=trace,
                    )
                    if valid_completion is None:
                        if trace:
                            print(
                                "No valid completion found, using the whole completion"
                            )
                        continue
                    if trace:
                        print(
                            f"************* Valid completion (raw) *********\n {repr(valid_completion)}"
                        )
                    valid_completion = reconstruct_word_boundaries(valid_completion)
                    if trace:
                        print(
                            f"************* Valid completion *************\n {valid_completion}"
                        )

                    end_time = time.monotonic()
                    intersection_time += end_time - start_time
                    # strip away the parts that are splits
                    splits_regex = "([\x00-\xff]*)".join(
                        re.escape(s)
                        for s in splits[i + 1 : -1]
                        + [splits[-1][: -len(instance._extra_suffix)]]
                    )
                    found = False
                    for pre_split in re.finditer(
                        r"([\x00-\xff]*)" + splits_regex, valid_completion
                    ):
                        completion = first_multiline_comment + pre_split.group(1)
                        found = True
                        break
                    if trace:
                        print(
                            f"************* Completion after suffix removal *************\n {completion}"
                        )
                    if not found:
                        if trace:
                            print("suffix not found in completion")
        return (
            "<ITER>".join(prompts),
            "<ITER>".join(all_outputs),
            instance.extract_result(completion_no_auto or completion),
            not bool(to_ctx_mgr),
            total_generated,
            all_resamplings,
            completion if completion_no_auto is not None else None,
            instance.extract_result(completion)
            if completion_no_auto is not None
            else None,
            intersection_time,
        )
