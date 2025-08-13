import hashlib

from rustformlang.rustformlang import DFA

from constrained_diffusion.constrain_utils import compile_lex_map
from constrained_diffusion.eval.dllm.model import load_model
from constrained_diffusion.eval.dllm.datasets.generic import Instance, extract_code
from rustformlang.cfg import CFG


class CppInstance(Instance):
    """
    Represents a single instance in a dataset.
    All instances must have a unique field "instance_id".
    """

    def __init__(self, prompt: str):
        """
        Initializes the CppInstance.
        All instances must have a unique field "instance_id".
        """
        self._prompt = prompt

    def instance_id(self) -> str:
        """
        Returns the unique identifier for the instance.
        This is used to identify instances across datasets.
        """
        # Use the MD5 hash of the prompt to create a unique instance ID
        return hashlib.md5(self._prompt.encode("utf-8")).hexdigest()

    def user_prompt_content(self) -> str:
        """
        Returns the user prompt content for the instance.
        """
        return self._prompt

    def assistant_start_line(self) -> str:
        """
        Returns a string that indicates the start of the assistant's response inside the code block
        i.e. function foo() {\n

        Default is an empty string, meaning the assistant's response starts immediately
        """
        return ""

    def language_short_name(self) -> str:
        """
        Returns the short name of the instance's language.
        This is used to indicate the language inside the code block to the assistant
        i.e. ```typescript --> language short name is "typescript"
        """
        return "cpp"

    def extract_result(self, s: str) -> str:
        """
        Extracts the result from the assistant's response.
        This is used to evaluate the instance's response.

        The string s is the model output including ```language_short_name()\n + assistant_start_line()

        Default just extracts the code block from the response.
        """
        return extract_code(s, self.language_short_name(), 0)

    def system_message_content(self) -> str:
        """
        Returns the system message content for the dataset.
        """
        return "You are an expert C++ programmer. Write a C++ function that solves the problem given by the user.\nMake sure to include all necessary headers and use standard C++ libraries."

    def language_lex_subtokens(
        self,
    ) -> tuple[CFG, dict[str, str | DFA], dict[str, set[str]]]:
        """
        Returns the grammar, lex map and subtokens for the dataset.
        """
        from constrained_diffusion.cfgs.cpp import cpp_grammar

        return cpp_grammar()

    def prelex(self) -> str | None:
        """
        Returns the prelex for the dataset.
        Usually its None
        """
        return "\x02\x03"

    def strip_chars(self):
        """
        Returns the characters to strip between lexed tokens
        Defaults to any whitespace
        """
        return None


def main():
    device = "cuda"

    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    eval_model = load_model(model_name)
    model, tokenizer = eval_model.model(device), eval_model.tokenizer(device)
    instance = CppInstance(
        prompt="Write a C++ function that calculates the factorial of a number."
    )
    diffusion_steps = 256
    generate_tokens = 256
    temperature = 0.2
    timeout = 300
    trace = True

    lang, orig_lex_map, subtokens = instance.language_lex_subtokens()
    lex_map = compile_lex_map(orig_lex_map, subtokens)
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
        steps=diffusion_steps,
        gen_length=generate_tokens,
        temperature=temperature,
        lang=lang,
        lex_map=lex_map,
        subtokens=subtokens,
        prelex=instance.prelex(),
        timeout=timeout,
        trace=trace,
        orig_lex_map=orig_lex_map,
        alg="low_confidence",
        additional_stuff=None,
    )
    print("----------- Prompt ----------------")
    print(prompt)
    print("Took {:.2f} seconds to generate.".format(time_taken_autocompletion))
    print("----------- Code ------------------")
    print(autocompletion)


if __name__ == "__main__":
    main()
