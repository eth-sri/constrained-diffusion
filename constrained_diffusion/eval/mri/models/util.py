from typing import Tuple


def strip_first_multiline_comment(code: str) -> Tuple[str, str]:
    """
    Strips the first multiline comment from the code.
    """
    if code.startswith("/*"):
        end = code.find("*/")
        if end != -1:
            return code[end + 2 :], code[: end + 2]
    return code, ""
