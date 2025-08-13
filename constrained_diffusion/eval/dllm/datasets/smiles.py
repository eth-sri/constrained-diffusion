from typing import Iterator

from datasets import load_dataset

from constrained_diffusion.eval.dllm.datasets.generic import DataSet, Instance
from rustformlang.cfg import CFG
from rustformlang.fa.dfa import DFA

from constrained_diffusion.cfgs.smiles import smiles_schema


class SmilesInstance(Instance):
    """
    Represents a single instance in a dataset.
    All instances must have a unique field "instance_id".
    """

    def __init__(self, instance_data: dict, system_message_content: str):
        """
        Initializes the instance with an ID and data.
        """
        self._instance_id = instance_data["instance_id"]
        self.data = instance_data
        self._system_message_content = system_message_content

    def to_dict(self) -> dict:
        """
        Returns the instance data as a dictionary.
        This is used to serialize the instance for storage or transmission.
        """
        return {
            "instance_id": self.instance_id(),
            "input": self.data["input"],
            "output": self.data["output"],
            "difficulty_category": self.data["difficulty_category"],
        }

    def system_message_content(self) -> str:
        """
        Returns the system message content for the instance.
        This is used to compile the instance's language.
        """
        return self._system_message_content

    def instance_id(self) -> str:
        """
        Returns the unique identifier for the instance.
        This is used to identify instances across datasets.
        """
        return self._instance_id

    def user_prompt_content(self) -> str:
        """
        Returns the user prompt content for the instance.
        This is used to compile the instance's language.
        """
        return self.data["input"]

    def language_short_name(self) -> str:
        """
        Returns the short name of the instance's language.
        This is used to indicate the language inside the code block to the assistant
        i.e. ```typescript --> language short name is "typescript"
        """
        return "smiles"

    def language_lex_subtokens(
        self,
    ) -> tuple[CFG, dict[str, str | DFA], dict[str, set[str]]]:
        """
        Returns the grammar, lex map and subtokens for the dataset.
        This is used to compile the dataset's language.
        """
        return smiles_schema()

    def strip_chars(self):
        return ""


class SmilesDataSet(DataSet):
    def __init__(self):
        """
        Base class for datasets.
        Each dataset should implement the methods defined below.
        """
        self._data = None
        self.config = None
        super().__init__()

    def load_data(self) -> list[dict]:
        """
        Loads the dataset data from a JSON file.
        This is used to load the instances of the dataset.
        """
        if self._data is None:
            dataset = load_dataset("eth-sri/smiles-eval")
            self._data = dataset["test"]
        return self._data

    def system_message_content(self) -> str:
        """
        Returns the system message content for the dataset.
        This is used to compile the dataset's language.
        """
        return """\
You are a specialized AI assistant that generates SMILES (Simplified Molecular Input Line Entry System) strings from chemical descriptions. You will be given a textual description of a chemical compound or a related task. Your goal is to produce the most accurate and valid SMILES string representing that description.

Your Task:

Based on the provided "input" description, generate the corresponding SMILES string.

Output Requirements:

- Provide only the SMILES string as your output.
- Ensure the SMILES string is syntactically valid.
- Represent all specified chemical features accurately (atoms, bonds, rings, aromaticity, charge, isotopes, stereochemistry).

Output:

- Provide only the smiles molecule as a raw string between triple backticks (```). For instance:
```smiles
C1=CC=CC=C1
```

"""

    def __iter__(self) -> Iterator[Instance]:
        """
        Returns an iterator over the dataset instances.
        All dataset instances must have a unique field "instance_id"
        """
        for instance_data in self.load_data():
            instance = SmilesInstance(instance_data, self.system_message_content())
            yield instance
