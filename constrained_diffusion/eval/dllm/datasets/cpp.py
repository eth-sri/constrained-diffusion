"""
Collects all datasets to be evaluated
"""

from typing import Iterator

from constrained_diffusion.eval.dllm.datasets.generic import (
    Instance,
    DataSet,
    extract_code,
)
from rustformlang.cfg import CFG
from rustformlang.fa.dfa import DFA


def format_prompt_to_question(prompt: str):
    # extract prompt from within /* and */
    start = prompt.find("/*") + 2
    end = prompt.find("*/", start)
    question = prompt[start:end].strip()
    return question


class CppInstance(Instance):
    """
    Represents a single instance in a dataset.
    """

    def __init__(self, instance: dict):
        """
        Initializes the TypeScript instance.
        All instances must have a unique field "instance_id".
        """
        self._instance = instance

    def instance_id(self) -> str:
        """
        Returns the unique identifier for the instance.
        This is used to identify instances across datasets.
        """
        return self._instance["task_id"]

    def user_prompt_content(self) -> str:
        """
        Returns the user prompt content for the instance.
        This is used to compile the instance's language.
        """
        return format_prompt_to_question(self._instance["prompt"])

    def assistant_start_line(self) -> str:
        """
        Returns a string that indicates the start of the assistant's response inside the code block
        i.e. function foo() {\n
        """
        return self._instance["declaration"]

    def language_short_name(self) -> str:
        """
        Returns the short name of the instance's language.
        This is used to indicate the language inside the code block to the assistant
        i.e. ```typescript --> language short name is "typescript"
        """
        if not self._instance["task_id"].startswith("CPP"):
            raise NotImplementedError("Only Cpp instances are supported so far.")
        return "cpp"

    def extract_result(self, s: str) -> str:
        """
        Extracts the result from the assistant's response.
        This is used to evaluate the instance's response.
        """
        extracted = extract_code(s, self.language_short_name(), 0)
        tests: str = self._instance["test"]
        if tests.strip().startswith("}") and extracted.strip().endswith("}"):
            tests = tests[tests.find("}") + 1 :]
        compilable = extracted + "\n\n" + tests
        return compilable

    def system_message_content(self) -> str:
        """
        Returns the system message content for the dataset.
        This is used to compile the dataset's language.
        """
        return """You are an expert in C++ programming. Solve the given problem by writing solution code in C++.
When answering, insert the solution code in a ```cpp...``` block. Do neither include test cases not a main function."""

    def language_lex_subtokens(
        self,
    ) -> tuple[CFG, dict[str, str | DFA], dict[str, set[str]]]:
        """
        Returns the grammar, lex map and subtokens for the dataset.
        This is used to compile the dataset's language.
        """
        from constrained_diffusion.cfgs.cpp import cpp_grammar

        return cpp_grammar()

    def prelex(self) -> str:
        """
        Returns the prelex for the dataset.
        This is used to compile the dataset's language.
        """
        return "\x02\x03"


class CppDataSet(DataSet):
    def __init__(self, split: str = "test", subset="cpp"):
        """
        Initializes the TypeScript dataset.
        This dataset contains instances of TypeScript programming problems.
        """
        self._split = split
        self._subset = subset
        self._dataset = None
        super().__init__()

    def data(self):
        """
        Returns the dataset instances.
        This is used to iterate over the dataset.
        """
        if self._dataset is None:
            from datasets import load_dataset

            self._dataset = load_dataset(
                "zai-org/humaneval-x", self._subset, trust_remote_code=True
            )[self._split]
        return self._dataset

    def __iter__(self) -> Iterator[Instance]:
        """
        Returns an iterator over the dataset instances.
        All dataset instances must have a unique field "instance_id"
        """
        return iter(CppInstance(instance) for instance in self.data())
