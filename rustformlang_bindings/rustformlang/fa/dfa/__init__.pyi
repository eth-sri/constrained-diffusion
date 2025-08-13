from typing import Optional

from rustformlang.fa.epsilon_nfa import ENFA
from rustformlang.fa.bytes_dfa import BytesDFA

class DFA:
    def accepts_string(self, string: str) -> bool:
        """
        Check if the DFA accepts the given string.
        """

    def accepts(self, word: list[str]) -> bool:
        """
        Check if the DFA accepts the given word.
        """

    def is_empty(self) -> bool:
        """
        Check if the DFA is empty.
        """

    def num_states(self) -> int:
        """
        Get the number of states in the ENFA.
        """

    def to_epsilon_automaton(self) -> "ENFA":
        """
        Convert the DFA to a non-deterministic finite automaton (NFA) with epsilon transitions.
        """

    def minimize(self) -> "DFA":
        """
        Minimize the DFA.
        """

    def intersection(self, other: "DFA") -> "DFA":
        """
        Compute the intersection of a DFA and an ENFA.
        """

    def complement(self) -> "DFA":
        """
        Compute the complement of the DFA.
        """

    def difference(self, other: "DFA") -> "DFA":
        """
        Compute the difference of two DFAs.
        """

    def n_prefix_language(self, n: int) -> "DFA":
        """
        Compute the n-prefix language of the DFA.
        In particular, it accepts all strings s where \exists w \in \Sigma^{n} o \Sigma^{*}: s o w \in L
        """

    def prefix_language(self) -> "DFA":
        """
        Compute the prefix language of the DFA.
        In particular, it accepts all strings s where \exists w \in \Sigma^{*}: s o w \in L
        """

    def true_prefix_language(self) -> "DFA":
        """
        Compute the true prefix language of the DFA.
        In particular, it accepts all strings s where \exists w \in \Sigma^{+}: s o w \in L
        """

    def accept_prefix(self, prefix: list[str]) -> Optional[int]:
        """
        Check if the DFA accepts a prefix of the given string, returns the index of the last accepted character.
        Performs a greedy match (i.e., the longest prefix).
        """

    def accept_prefix_string(self, prefix: str) -> Optional[int]:
        """
        Check if the DFA accepts a prefix of the given string, returns the index of the last accepted character (in bytes!).
        Performs a greedy match (i.e., the longest prefix).
        """

    def accept_prefix_bytes(self, prefix: bytes) -> Optional[int]:
        """
        Check if the DFA accepts a prefix of the given string, returns the index of the last accepted character.
        Performs a greedy match (i.e., the longest prefix).
        """

    def to_text(self) -> str:
        """
        Convert the DFA to a text representation.
        """

    def __eq__(self, other: "DFA"):
        """
        Check if two DFAs are equal.
        """

    def to_bytes_dfa(self) -> "BytesDFA":
        """
        Convert the DFA to a BytesDFA.
        """

    def concat(self, other: "DFA") -> "DFA":
        """
        Concatenate two DFAs.
        """

def minimize_dfa_threaded(dfa: "DFA") -> "DFA":
    """
    Minimize the DFA in a separate thread.
    """
    pass  # This function is implemented in Rust and will be called from Python.
