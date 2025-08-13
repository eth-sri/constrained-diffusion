from typing import Iterator

from rustformlang.cfg import CFG
from rustformlang.fa.dfa import DFA


def extract_code(output: str, humanreadable_target_language: str, nth: int):
    prefix = f"```{humanreadable_target_language.lower()}\n"
    pos = 0
    for _ in range(nth + 1):
        prefix_location = output.find(prefix, pos)
        if prefix_location == -1:
            continue
        pos = prefix_location + len(prefix)
    code = output[pos:]
    end_pos = code.find("```")
    if end_pos == -1:
        end_pos = len(code)
    code = code[:end_pos]
    return code.strip().strip("`") + "\n"


class Instance:
    """
    Represents a single instance in a dataset.
    All instances must have a unique field "instance_id".
    """

    def instance_id(self) -> str:
        """
        Returns the unique identifier for the instance.
        This is used to identify instances across datasets.
        """
        raise NotImplementedError("Subclasses must implement instance_id method.")

    def user_prompt_content(self) -> str:
        """
        Returns the user prompt content for the instance.
        This is used to compile the instance's language.
        """
        raise NotImplementedError(
            "Subclasses must implement user_prompt_content method."
        )

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
        raise NotImplementedError(
            "Subclasses must implement language_short_name method."
        )

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
        This is used to compile the dataset's language.
        """
        raise NotImplementedError(
            "Subclasses must implement system_message_content method."
        )

    def language_lex_subtokens(
        self,
    ) -> tuple[CFG, dict[str, str | DFA], dict[str, set[str]]]:
        """
        Returns the grammar, lex map and subtokens for the dataset.
        This is used to compile the dataset's language.
        """
        raise NotImplementedError(
            "Subclasses must implement language_lex_subtokens method."
        )

    def prelex(self) -> str | None:
        """
        Returns the prelex for the dataset.
        This is used to compile the dataset's language.
        Usually its None
        """
        return None

    def strip_chars(self):
        """
        Returns the characters to strip between lexed tokens
        Defaults to any whitespace
        """
        return None


class DataSet:
    def __init__(self):
        self.different_grammar_per_instance = False

    def __iter__(self) -> Iterator[Instance]:
        """
        Returns an iterator over the dataset instances.
        All dataset instances must have a unique field "instance_id"
        """
        raise NotImplementedError("Subclasses must implement __iter__ method.")
