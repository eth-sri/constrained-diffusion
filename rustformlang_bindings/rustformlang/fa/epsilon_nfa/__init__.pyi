from rustformlang.fa.dfa import DFA

def epsilon() -> str:
    """
    Return the epsilon symbol.
    """

class ENFA:
    def __init__(self):
        """
        Initialize an empty epsilon NFA (ENFA).
        """

    def add_transition(self, from_state: str, symbol: str, to_state: str) -> None:
        """
        Add a transition to the ENFA.
        """

    def set_start_state(self, state: str) -> None:
        """
        Set the start state of the ENFA.
        """

    def add_accept_state(self, state: str) -> None:
        """
        Add an accept state to the ENFA.
        """

    def num_states(self) -> int:
        """
        Get the number of states in the ENFA.
        """

    def accepts_string(self, string: str) -> bool:
        """
        Check if the ENFA accepts the given string.
        """

    def accepts(self, word: list[str]) -> bool:
        """
        Check if the ENFA accepts the given word.
        """

    def to_deterministic(self) -> "DFA":
        """
        Convert the ENFA to a DFA.
        """

    def minimize(self) -> "DFA":
        """
        Minimize the ENFA.
        """

    def union(self, other: ENFA) -> ENFA:
        """
        Compute the union of two ENFAs.
        """

    def n_suffix_language(self, n: int) -> "ENFA":
        """
        Compute the n-prefix language of the ENFA.
        In particular, it accepts all strings s where \exists w \in \Sigma^{n} o \Sigma^{*}: w o s \in L
        """

    def suffix_language(self) -> "ENFA":
        """
        Compute the suffix language of the ENFA.
        In particular, it accepts all strings s where \exists w \in \Sigma^{*}: w o s \in L
        """

    def true_suffix_language(self) -> "ENFA":
        """
        Compute the true suffix language of the ENFA.
        In particular, it accepts all strings s where \exists w \in \Sigma^{+}: w o s  \in L
        """

    def is_empty(self) -> "ENFA":
        """
        Returns whether this ENFA accepts any words or is empty
        """

    def to_graphviz(self) -> str:
        """
        Convert the ENFA to a Graphviz representation.
        """

    def concat(self, other: "ENFA"):
        """
        Concatenate two ENFAs.
        """

def minimize_enfa_threaded(enfa: ENFA) -> DFA:
    """
    Minimize the given epsilon NFA (ENFA) in a separate thread.
    """
    pass  # This function is implemented in Rust and will be called from Python.
