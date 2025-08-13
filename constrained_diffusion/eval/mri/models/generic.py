from constrained_diffusion.eval.mri.datasets.generic import Instance
from rustformlang.cfg import CFG


class Model(object):
    """
    Base class for all models to be evaluated.
    """

    def tokenizer(self, device):
        """
        Returns the tokenizer for the model.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def model(self, device):
        """
        Returns the model for the specified device.
        """
        raise NotImplementedError("Subclasses must implement this method.")

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
        raise NotImplementedError("Subclasses must implement this method.")

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
    ) -> tuple[str, str, str, bool, int, list, str, str, float]:
        """
        Generates a response from the model based on the provided messages and additional constraints.
        Returns
        - The derived prompt for the model
        - The generated code as a string.
        - The extracted result from the instance.
        - A boolean indicating whether the generation timed out.
        """
        raise NotImplementedError("Subclasses must implement this method.")
