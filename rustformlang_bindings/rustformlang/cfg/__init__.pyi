from rustformlang.fa.dfa import DFA

class CFG:
    @staticmethod
    def from_text(text: str, start_symbol: str) -> CFG:
        """
        Load a CFG from a text representation.
        """

    def get_terminals(self) -> list[str]:
        """
        Get the terminals of the CFG.
        """

    def num_productions(self) -> int:
        """
        Get the number of productions in the CFG.
        """

    def to_normal_form(self) -> CFG:
        """
        Convert the CFG to Chomsky Normal Form (CNF).
        """

    def accepts(self, word: list[str]) -> bool:
        """
        Check if the DFA accepts the given word (list of terminals).
        """

    def __contains__(self, word: list[str]) -> bool:
        """
        Check if the DFA accepts the given word (list of terminals).
        """

    def accepts_string(self, string: str) -> bool:
        """
        Check if the DFA accepts the given string.
        """

    def is_empty(self) -> bool:
        """
        Check if the CFG is empty.
        """

    def intersection(self, other: DFA) -> CFG:
        """
        Compute the intersection of two CFGs.
        """

    def concatenate(self, other: CFG) -> CFG:
        """
        Concatenate two CFGs.
        """

    def to_text(self) -> str:
        """
        Convert the CFG to a text representation.
        """

    def substitute(self, terminal_map: dict[str, CFG]) -> CFG:
        """
        Substitute terminals in the CFG with replacement grammars.
        """

    def is_intersection_empty(self, other: DFA, timeout: float) -> bool:
        """
        Check if the intersection of CFG and DFA is empty.
        """

    def example_word(self, other: DFA, timeout: float) -> str | None:
        """
        Get an example word accepted by the CFG.
        This is useful for testing and debugging.
        """

def is_intersection_empty_threaded(cfg: CFG, dfa: DFA, timeout: float) -> bool:
    """
    Check if the intersection of a CFG and a DFA is empty in a separate thread.

    Args:
        cfg (CFG): The context-free grammar.
        dfa (DFA): The deterministic finite automaton.

    Returns:
        bool: True if the intersection is empty, False otherwise.
    """
    pass  # This function is implemented in Rust and will be called from Python.
