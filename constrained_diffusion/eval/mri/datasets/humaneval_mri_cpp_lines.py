import json
from pathlib import Path
from typing import Iterator

from constrained_diffusion.cfgs.cpp import cpp_grammar, CPP_grammar_one_fun
from constrained_diffusion.eval.mri.datasets.generic import Instance, DataSet
from rustformlang.cfg import CFG
from rustformlang.fa.dfa import DFA

PATH_TO_DATASET = Path(__file__).parent / "dataset"


class HumanEvalMriLinesInstance(Instance):
    """
    Represents a single instance in a dataset.
    All instances must have a unique field "instance_id".
    """

    def __init__(self, instance: dict, language: str, spans: int):
        """
        Initializes the HumanEvalFimInstance.
        All instances must have a unique field "instance_id".
        """
        self._instance = instance
        self.language = language
        self.spans = spans

    def instance_id(self) -> str:
        """
        Returns the unique identifier for the instance.
        This is used to identify instances across datasets.
        """
        return self._instance["instance_id"]

    @property
    def _extra_suffix(self) -> str:
        """
        Returns the extra suffix for the instance.
        This is used to indicate the end of the assistant's response.
        """
        return "\nint main(){\n// TODO\n }"

    def splits(self) -> list[str]:
        """
        Returns the splits for the instance.
        The gaps between the splits must be filled by the FIM model
        """
        splits = self._instance["splits"].copy()
        splits[0] = self._instance["prompt"] + splits[0]
        splits[-1] = splits[-1] + self._extra_suffix
        return splits

    def language_short_name(self) -> str:
        """
        Returns the short name of the instance's language.
        This is used to indicate the language inside the code block to the assistant
        i.e. ```typescript --> language short name is "typescript"
        """
        return self.language

    def extract_result(self, s: str) -> str:
        """
        Extracts the result from the assistant's response.
        This is used to evaluate the instance's response.
        """
        # the declaration is included in the prompt / splits, so just add the tests
        s = s.removeprefix(self._instance["prompt"]).removesuffix(self._extra_suffix)
        return self._instance["declaration"] + s + "\n" + self._instance["test"]

    def language_lex_subtokens(
        self,
    ) -> tuple[CFG, dict[str, str | DFA], dict[str, set[str]]]:
        """
        Returns the grammar, lex map and subtokens for the dataset.
        This is used to compile the dataset's language.
        """
        return cpp_grammar(CPP_grammar_one_fun)

    def prelex(self) -> str | None:
        """
        Returns the prelex for the dataset.
        This is used to compile the dataset's language.
        Usually its None
        """
        return "\x02\x03"

    def to_dict(self) -> dict:
        """
        Returns the instance data as a dictionary.
        This is used to serialize the instance for storage or transmission.
        """
        return {
            "instance_id": self.instance_id(),
            "number_spans": self.spans,
            "prompt": self._instance["prompt"],
            "declaration": self._instance["declaration"],
            "splits": self._instance["splits"],
            "removed_spans": self._instance["removed_spans"],
            "canonical_solution": self._instance["canonical_solution"],
        }


class HumanEvalMriLinesDataSet(DataSet):
    def __init__(self, language="cpp", spans=1):
        super().__init__()
        self.language = language
        self.spans = spans
        self._data = None

    def load_data(self):
        if self._data is None:
            with (
                PATH_TO_DATASET / f"humaneval_{self.language}_{self.spans}_lines.jsonl"
            ).open() as f:
                dataset = []
                for line in f.readlines():
                    dataset.append(
                        HumanEvalMriLinesInstance(
                            json.loads(line), self.language, self.spans
                        )
                    )
        return self._data

    def __iter__(self) -> Iterator[Instance]:
        """
        Returns an iterator over the dataset instances.
        All dataset instances must have a unique field "instance_id"
        """
        for instance_data in self.load_data():
            yield HumanEvalMriLinesInstance(instance_data, self.language, self.spans)
