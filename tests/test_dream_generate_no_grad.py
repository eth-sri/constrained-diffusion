"""Regression test: the Dream sampling generator must run with autograd off.

``diffusion_generate`` is decorated with ``@torch.no_grad()`` but returns the
generator created by ``_sample`` without consuming it. The decorator only covers
the call that *creates* the generator, so the model forward passes, which run
when the generator is iterated, were executed with gradient tracking enabled
unless the caller disabled it. This test drives the real ``diffusion_generate``
with a tiny stand-in model and checks the grad mode seen by every forward pass.
"""

from types import SimpleNamespace

import torch

from constrained_diffusion.eval.dllm.models.dream.generate_constrained import (
    DreamGenerationConfig,
    diffusion_generate,
)

VOCAB_SIZE = 16
MASK_TOKEN_ID = 15
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 14

CONFIG_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "max_length",
    "max_new_tokens",
    "eps",
    "steps",
    "alg",
    "alg_temp",
    "num_return_sequences",
    "return_dict_in_generate",
    "output_history",
}


class TinyDreamLikeModel(torch.nn.Module):
    """Minimal model exposing what ``diffusion_generate``/``_sample`` need."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB_SIZE, 8)
        self.head = torch.nn.Linear(8, VOCAB_SIZE)
        with torch.no_grad():  # never predict the mask or pad token
            self.head.bias[MASK_TOKEN_ID] = -1e4
            self.head.bias[PAD_TOKEN_ID] = -1e4
        self.grad_enabled_seen = []
        self.logits_require_grad_seen = []

    @property
    def device(self):
        return self.head.weight.device

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        logits = self.head(self.embed(input_ids))
        self.grad_enabled_seen.append(torch.is_grad_enabled())
        self.logits_require_grad_seen.append(logits.requires_grad)
        return SimpleNamespace(logits=logits)

    # --- generation helpers mirroring Dream's DreamGenerationMixin ---
    def _prepare_generation_config(self, generation_config, **kwargs):
        if generation_config is None:
            generation_config = DreamGenerationConfig(
                mask_token_id=MASK_TOKEN_ID,
                pad_token_id=PAD_TOKEN_ID,
                eos_token_id=EOS_TOKEN_ID,
                **{k: v for k, v in kwargs.items() if k in CONFIG_KEYS},
            )
        return generation_config

    def _prepare_special_tokens(self, generation_config, device=None):
        pass

    def _prepare_generated_length(
        self, generation_config, has_default_max_length, input_ids_length
    ):
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = (
                generation_config.max_new_tokens + input_ids_length
            )
        return generation_config

    def _validate_generated_length(
        self, generation_config, input_ids_length, has_default_max_length
    ):
        assert input_ids_length < generation_config.max_length

    def _expand_inputs_for_generation(
        self, expand_size=1, input_ids=None, attention_mask=None
    ):
        return input_ids, attention_mask


class TinyTokenizer:
    """Decodes token ids to distinct strings; never produces the EOS string."""

    special_tokens_map = {"eos_token": "<eos>"}

    def decode(self, token_id, skip_special_tokens=False):
        token_id = int(token_id)
        return "<mask>" if token_id == MASK_TOKEN_ID else f"t{token_id}"

    def batch_decode(self, sequences, skip_special_tokens=False):
        sequences = torch.as_tensor(sequences)
        if sequences.dim() == 1:
            return [self.decode(t) for t in sequences]
        return ["".join(self.decode(t) for t in row) for row in sequences]


def run_generation(model, steps=2, gen_length=4):
    tokenizer = TinyTokenizer()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = None
    # same call pattern as DreamModel.generate_unconstrained
    for out, _resamples, _valid in diffusion_generate(
        model,
        tokenizer,
        prompt_len=prompt.shape[1],
        constraint_lang=None,
        lex_map=None,
        inputs=prompt,
        attention_mask=None,
        max_new_tokens=gen_length,
        return_dict_in_generate=False,
        steps=steps,
        temperature=0.0,
        top_p=0.95,
        output_history=False,
        alg="entropy",
        alg_temp=0.0,
        subtokens=None,
        trace=False,
        prelex=None,
        additional_stuff=None,
        max_total_injections=0,
        inject_gap_size=0,
        constrain=False,
    ):
        pass
    return out


def test_sampling_generator_runs_without_grad():
    torch.manual_seed(0)
    model = TinyDreamLikeModel()
    assert torch.is_grad_enabled(), "test must start with autograd enabled"

    out = run_generation(model, steps=2, gen_length=4)

    assert len(model.grad_enabled_seen) == 2, "one forward pass per step"
    assert model.grad_enabled_seen == [False, False]
    assert model.logits_require_grad_seen == [False, False]
    assert out.shape == (1, 3 + 4)
    assert not (out == MASK_TOKEN_ID).any(), "all masked positions were filled"
    assert torch.is_grad_enabled(), "caller's grad mode must be restored"
