from typing import Iterator

from constrained_diffusion.constrain_utils import LexMap
from constrained_diffusion.eval.dllm.datasets.generic import extract_code
from rustformlang.cfg import CFG


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

    def splits(self) -> list[str]:
        """
        Returns the splits for the instance.
        The gaps between the splits must be filled by the FIM model
        """
        raise NotImplementedError("Subclasses must implement splits method.")

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

    def language_lex_subtokens(
        self,
    ) -> tuple[CFG, LexMap, dict[str, list[str]]]:
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
