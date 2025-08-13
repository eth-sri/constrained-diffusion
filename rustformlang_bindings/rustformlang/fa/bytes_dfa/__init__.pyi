from typing import Optional

from rustformlang.fa.epsilon_nfa import ENFA

class BytesDFA:
    def accepts_string(self, string: str) -> bool:
        """
        Check if the BytesDFA accepts the given string.
        """

    def accepts(self, word: list[str]) -> bool:
        """
        Check if the BytesDFA accepts the given word.
        """

    def is_empty(self) -> bool:
        """
        Check if the BytesDFA is empty.
        """

    def num_states(self) -> int:
        """
        Get the number of states in the ENFA.
        """

    def to_epsilon_automaton(self) -> "ENFA":
        """
        Convert the BytesDFA to a non-deterministic finite automaton (NFA) with epsilon transitions.
        """

    def minimize(self) -> "BytesDFA":
        """
        Minimize the BytesDFA.
        """

    def intersection(self, other: "BytesDFA") -> "BytesDFA":
        """
        Compute the intersection of a BytesDFA and an ENFA.
        """

    def complement(self) -> "BytesDFA":
        """
        Compute the complement of the BytesDFA.
        """

    def difference(self, other: "BytesDFA") -> "BytesDFA":
        """
        Compute the difference of two BytesDFAs.
        """

    def n_prefix_language(self, n: int) -> "BytesDFA":
        """
        Compute the n-prefix language of the BytesDFA.
        In particular, it accepts all strings s where \exists w \in \Sigma^{n} o \Sigma^{*}: s o w \in L
        """

    def prefix_language(self) -> "BytesDFA":
        """
        Compute the prefix language of the BytesDFA.
        In particular, it accepts all strings s where \exists w \in \Sigma^{*}: s o w \in L
        """

    def true_prefix_language(self) -> "BytesDFA":
        """
        Compute the true prefix language of the BytesDFA.
        In particular, it accepts all strings s where \exists w \in \Sigma^{+}: s o w \in L
        """

    def accept_prefix(self, prefix: list[str]) -> Optional[int]:
        """
        Check if the BytesDFA accepts a prefix of the given string, returns the index of the last accepted character.
        Performs a greedy match (i.e., the longest prefix).
        """

    def accept_prefix_string(self, prefix: str) -> Optional[int]:
        """
        Check if the BytesDFA accepts a prefix of the given string, returns the index of the last accepted character (in bytes!).
        Performs a greedy match (i.e., the longest prefix).
        """

    def accept_prefix_bytes(self, prefix: bytes) -> Optional[int]:
        """
        Check if the BytesDFA accepts a prefix of the given string, returns the index of the last accepted character.
        Performs a greedy match (i.e., the longest prefix).
        """

    def to_text(self) -> str:
        """
        Convert the BytesDFA to a text representation.
        """

    def __eq__(self, other: "BytesDFA"):
        """
        Check if two BytesDFAs are equal.
        """

    def concat(self, other: "BytesDFA") -> "BytesDFA":
        """
        Concatenate two BytesDFAs.
        """

    def example_word(self) -> Optional[str]:
        """
        Get an example word accepted by the BytesDFA.
        This is useful for testing and debugging.
        Returns None if no example word exists.
        """

    def intersection_example_word(self, other: "BytesDFA") -> Optional[str]:
        """
        Get an example word accepted by the intersection of two BytesDFAs.
        This is useful for testing and debugging.
        Returns None if no example word exists.
        """

def regex_to_dfa(regex: str) -> BytesDFA:
    """
    Convert a regular expression to a BytesDFA.
    """

def regex_escape(regex: str) -> str:
    """
    Escape a regular expression compatible with the Rust regex crate.
    """
