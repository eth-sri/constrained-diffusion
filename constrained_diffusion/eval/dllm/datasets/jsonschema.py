from typing import Iterator

from constrained_diffusion.eval.dllm.datasets.generic import DataSet, Instance
from constrained_diffusion.cfgs.jsonschema import schema_to_cfg
from rustformlang.cfg import CFG
from rustformlang.fa.dfa import DFA
import json
from datasets import load_dataset


class JsonSchemaInstance(Instance):
    """
    Represents a single instance in a dataset.
    All instances must have a unique field "instance_id".
    """

    def __init__(self, instance_data: dict):
        """
        Initializes the instance with an ID and data.
        """
        self._instance_id = instance_data["instance_id"]
        self.data = instance_data

    def to_dict(self) -> dict:
        """
        Returns the instance data as a dictionary.
        This is used to serialize the instance for storage or transmission.
        """
        return {
            "instance_id": self.instance_id(),
            "input": self.data["input"],
            "output": self.data["output"],
            "schema": self.data["schema"],
        }

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
        return "json"

    def system_message_content(self) -> str:
        """
        Returns the system message content for the dataset.
        This is used to compile the dataset's language.
        """
        return """You are a helpful assistant that answers in JSON. Here's the JSON schema you must adhere to:\n<schema>\n{}\n</schema>\n""".format(
            self.data["schema"]
        )

    def language_lex_subtokens(
        self,
    ) -> tuple[CFG, dict[str, str | DFA], dict[str, set[str]]]:
        """
        Returns the grammar, lex map and subtokens for the dataset.
        This is used to compile the dataset's language.
        """
        return schema_to_cfg(json.loads(self.data["schema"]))


class JsonSchemaDataSet(DataSet):
    def __init__(self):
        """
        Base class for datasets.
        Each dataset should implement the methods defined below.
        """
        super().__init__()
        self.config = None
        self.data = None
        self.different_grammar_per_instance = True

    def load_data(self) -> list[dict]:
        """
        Loads the dataset data from a JSON file.
        This is used to load the instances of the dataset.
        """
        if self.data is None:
            dataset = load_dataset("eth-sri/json-mode-eval-extended")
            self.data = dataset["test"]
        return self.data

    def __iter__(self) -> Iterator[Instance]:
        """
        Returns an iterator over the dataset instances.
        All dataset instances must have a unique field "instance_id"
        """
        for instance_data in self.load_data():
            instance = JsonSchemaInstance(instance_data)
            yield instance
