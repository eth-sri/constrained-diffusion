from rustformlang.fa.bytes_dfa import BytesDFA

class LexMap:
    @staticmethod
    def from_lex_map(
        lex_map: dict[str, tuple[BytesDFA, BytesDFA, BytesDFA, BytesDFA]],
    ) -> LexMap:
        """
        Create a LexMap from a lex_map dictionary.

        Args:
            lex_map (dict[str, tuple[DFA, DFA, DFA, DFA]]): The lex_map dictionary.

        Returns:
            LexMap: An instance of LexMap.
        """

    def keys(self) -> list[str]:
        """
        Get the keys of the lex_map.

        Returns:
            list[str]: A list of keys in the lex_map.
        """
        pass

    def get(self, key: str) -> tuple[BytesDFA, BytesDFA, BytesDFA, BytesDFA]:
        """
        Get the value associated with a key in the lex_map.

        Args:
            key (str): The key to look up.

        Returns:
            tuple[BytesDFA, BytesDFA, BytesDFA, BytesDFA]: The value associated with the key.
        """
        pass

    def __contains__(self, key: str) -> bool:
        """
        Check if the lex_map contains a key.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key is in the lex_map, False otherwise.
        """
        pass

    def __getitem__(self, key: str) -> tuple[BytesDFA, BytesDFA, BytesDFA, BytesDFA]:
        """
        Get the value associated with a key in the lex_map.

        Args:
            key (str): The key to look up.

        Returns:
            tuple[BytesDFA, BytesDFA, BytesDFA, BytesDFA]: The value associated with the key.
        """
        pass

    def __len__(self) -> int:
        """
        Get the number of items in the lex_map.

        Returns:
            int: The number of items in the lex_map.
        """
        pass

def reset_lex_cache() -> None:
    """
    Reset the lex cache.

    This function clears the internal lex cache used for lexing operations.
    """
    pass

def lex_string(
    string: str, lex_map: LexMap, is_first: bool, strip_chars: str | None
) -> set[tuple[list[str], str | None, str | None]]:
    """
    Lex a string using the provided lex_map.

    Args:
        string (str): The string to lex.
        lex_map (LexMap): The lex map to use for lexing.
        is_first (bool): Whether this is the first word
        strip_chars (str | None): Characters to strip from the word.

    Returns:
        tuple[list[str], bool, bool]: A tuple containing the list of tokens, and flags indicating if the first or last tokens are partial
    """
    pass

def prelex_word(word: str, prelex: str, is_first: bool, is_last: bool) -> str:
    """
    Prelex a word using the provided prelex information.

    Args:
        word (str): The word to prelex.
        prelex (str): The prelex information to use.
        is_first (bool): Whether this is the first token in the string.
        is_last (bool): Whether this is the last token in the string.

    Returns:
        str: The prelexed word.
    """
    pass

def all_lexings(
    vocabulary: list[str],
    lex_map: LexMap,
    prelex: str = "",
    strip_chars: str | None = None,
) -> list[list[tuple[list[str], str | None, str | None]]]:
    """
    Get all lexings for the provided vocabulary.

    Args:
        vocabulary (list[str]): The vocabulary to get lexings for.
        lex_map (LexMap): The lex map to use.
        prelex (str, optional): Prelex information to use. Defaults to "".
        strip_chars (str | None, optional): Characters to strip from the vocabulary. Defaults to None.

    Returns:
        list[list[tuple[list[str], bool, bool]]]: A list of all lexings.
    """
    pass
