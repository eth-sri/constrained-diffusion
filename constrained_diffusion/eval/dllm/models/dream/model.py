import time

import stopit
import torch
from transformers import AutoTokenizer, AutoModel

from constrained_diffusion.constrain_utils import (
    EOS,
    partial_output_from_tokens,
    autocomplete_valid,
    generated_language,
    derive_supertokens,
    CompiledLexMap,
)
from constrained_diffusion.eval.dllm.datasets.generic import Instance
from constrained_diffusion.eval.dllm.models.generic import Model
from rustformlang.cfg import CFG
from constrained_diffusion.eval.dllm.models.dream.generate_constrained import (
    diffusion_generate as diffusion_generate_constrained,
)

MODEL_NAME = "Dream-org/Dream-v0-Instruct-7B"
MODEL_NAME_DREAM_CODE = "Dream-org/Dream-Coder-v0-Instruct-7B"
MODEL_NAME_DIFFU_CODER = "apple/DiffuCoder-7B-Instruct"


class DreamModel(Model):
    """
    Model for DreamModel style models.
    """

    def __init__(self, name=MODEL_NAME):
        super().__init__()
        self.name = name

    def tokenizer(self, device):
        return AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)

    def model(self, device):
        kwargs = (
            {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
            }
            if device == "cuda"
            else {"device_map": device}
        )

        model = AutoModel.from_pretrained(
            self.name, **kwargs, trust_remote_code=True
        ).eval()
        return model

    def prepare_prompt(self, instance: Instance, tokenizer, model, trace: bool):
        system_message_content = instance.system_message_content()
        system_messages = [
            {
                "role": "system",
                "content": system_message_content,
            },
        ]
        suffix = f"```{instance.language_short_name()}\n"
        start_line = instance.assistant_start_line()
        messages = system_messages + [
            {"role": "user", "content": instance.user_prompt_content()},
            {"role": "assistant", "content": suffix},
        ]
        inputs_no_line = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=True,
        )
        messages[-1]["content"] += start_line
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=True,
        )
        inputs_text = tokenizer.apply_chat_template(
            messages,
            return_dict=False,
            continue_final_message=True,
            tokenize=False,
        )

        if trace:
            print(inputs_text)

        input_ids = inputs.input_ids.to(device=model.device)
        attention_mask = inputs.attention_mask.to(device=model.device)

        return (
            input_ids,
            attention_mask,
            inputs_no_line.input_ids.shape[-1],
            suffix,
            start_line,
            inputs_text,
        )

    def generate_unconstrained(
        self,
        instance: Instance,
        model,
        tokenizer,
        steps: int,
        gen_length: int,
        temperature: int,
        alg: str = "entropy",
        trace: bool = False,
    ) -> tuple[str, str, str, bool]:
        prompt, attention_mask, prompt_len, suffix, start_line, prompt_raw = (
            self.prepare_prompt(instance, tokenizer, model, trace)
        )

        out = None
        resamples = []
        for out, resamples, valid in diffusion_generate_constrained(
            model,
            tokenizer,
            prompt_len=prompt_len,
            constraint_lang=None,
            lex_map=None,
            inputs=prompt,
            attention_mask=attention_mask,
            max_new_tokens=gen_length,
            return_dict_in_generate=False,
            steps=steps,
            temperature=temperature,
            top_p=0.95,
            output_history=False,
            alg=alg,
            alg_temp=0.0,
            subtokens=None,
            trace=trace,
            prelex=None,
            additional_stuff=None,
            max_total_injections=0,
            inject_gap_size=0,
            constrain=False,
        ):
            pass
        if out is None:
            code = "TIMEOUT"
        else:
            code = tokenizer.batch_decode(
                out[:, prompt.shape[1] :], skip_special_tokens=True
            )[0]
        extracted = instance.extract_result(suffix + start_line + code)
        return prompt_raw, code, extracted, False

    def generate_constrained(
        self,
        instance: Instance,
        model,
        tokenizer,
        steps: int,
        gen_length: int,
        temperature: int,
        lang: CFG,
        lex_map: CompiledLexMap,
        orig_lex_map,
        subtokens,
        additional_stuff,
        max_total_injections: int = 0,
        inject_gap_size: int = 0,
        prelex: str | None = None,
        alg: str = "entropy",
        timeout: int = 60,
        trace: bool = False,
    ) -> tuple[str, str, list[str], str, bool, list, str, str, float]:
        prompt, attention_mask, prompt_len, suffix, start_line, prompt_raw = (
            self.prepare_prompt(instance, tokenizer, model, trace)
        )

        out = None
        resamples = []
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            for out, resamples, valid in diffusion_generate_constrained(
                model,
                tokenizer,
                prompt_len=prompt_len,
                constraint_lang=lang,
                lex_map=lex_map,
                inputs=prompt,
                attention_mask=attention_mask,
                max_new_tokens=gen_length,
                return_dict_in_generate=False,
                steps=steps,
                temperature=temperature,
                top_p=0.95,
                output_history=False,
                alg=alg,
                alg_temp=0.0,
                subtokens=subtokens,
                trace=trace,
                prelex=prelex,
                additional_stuff=additional_stuff,
                max_total_injections=max_total_injections,
                inject_gap_size=inject_gap_size,
            ):
                pass
        if out is None:
            code = "TIMEOUT"
            code_raw = "TIMEOUT"
        else:
            code = tokenizer.batch_decode(
                out[:, prompt.shape[1] :], skip_special_tokens=True
            )[0]
            code_raw = tokenizer.batch_decode(out.squeeze(), skip_special_tokens=False)
        extracted = instance.extract_result(suffix + start_line + code)

        # extract a valid completion
        if not valid:
            start_time = time.monotonic()
            generated_words = tokenizer.batch_decode(out.squeeze())
            mask_id = 151666
            mask_decoded = tokenizer.decode(mask_id)
            generated_words = [
                None
                if x == mask_decoded
                else EOS
                if x in ("<|endoftext|>", "<|eot_id|>", "<|dlm_pad|>", "<|im_end|>")
                else x
                for x in generated_words[prompt_len:]
            ]
            partial_output, first_token_gap, last_token_eos_adj = (
                partial_output_from_tokens(generated_words, prelex)
            )
            if trace:
                print("Generated words:", generated_words)
                print("Partial output:", partial_output)
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
                    supertokens=derive_supertokens(subtokens),
                    strip_chars=instance.strip_chars(),
                ),
                subtokens=subtokens,
                lex_map=orig_lex_map,
                constraint_lang=lang,
                trace=trace,
            )
            if trace:
                print(f"Valid completion: {valid_completion}")
            completion_extracted = (
                instance.extract_result(suffix + valid_completion)
                if valid_completion
                else None
            )
            end_time = time.monotonic()
            intersection_time = end_time - start_time
            if trace:
                print(
                    f"Time taken to extract valid completion: {intersection_time:.2f} seconds"
                )
        else:
            valid_completion = None
            completion_extracted = None
            intersection_time = 0.0
        return (
            prompt_raw,
            code,
            code_raw,
            extracted,
            not bool(to_ctx_mgr),
            resamples,
            valid_completion,
            completion_extracted,
            intersection_time,
        )
