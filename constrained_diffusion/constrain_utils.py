import math
import re
from itertools import islice, chain, repeat
from typing import Tuple, Iterable

from collections import defaultdict

import frozendict
import numpy as np
import torch
from regex import regex
from tqdm import tqdm
from rustformlang.cfg import CFG
from rustformlang.fa.dfa import DFA
from rustformlang.fa.bytes_dfa import BytesDFA, regex_to_dfa, regex_escape
from rustformlang.fa.epsilon_nfa import ENFA, epsilon, minimize_enfa_threaded
from rustformlang.constraining import (
    lex_string,
    LexMap as RustLexMap,
    reset_lex_cache,
    prelex_word,
    all_lexings,
)

LexMap = dict[str, str | BytesDFA]
CompiledLexMap = RustLexMap


LEX_MAP = {
    # both original and reversed
    "lexNumber": r"((-?[1-9]\d*)|0)",
    "lexString": r'"[^\n\r"]*"',
    "lexNull": r"null",
    "lexTrue": r"true",
    "lexFalse": r"false",
    "lexFence": r"```",
}
LEX_TOKENS = {"{", "}", ",", "[", "]", ":"}
for token in LEX_TOKENS:
    LEX_MAP[token] = regex_escape(token)


def relevant_automata_from_regex(
    v: str | BytesDFA,
) -> tuple[BytesDFA, BytesDFA, BytesDFA, BytesDFA]:
    """
    Get the relevant automata for a regex
    Args:
        v: regex or string to parse
        minimize: whether to minimize the automata

    Returns:
        The automata for the regex and its prefix and suffix languages
    """
    if isinstance(v, str):
        try:
            v = regex_to_dfa(v)
        except Exception as e:
            raise ValueError(f"Failed to parse regex {v}: {e}")
    suffix_lang = (
        v.to_epsilon_automaton().true_suffix_language().minimize().to_bytes_dfa()
    )
    # no need to minimize the prefix lang automaton
    return (
        v,
        v.true_prefix_language(),
        suffix_lang,
        suffix_lang.true_prefix_language(),
    )


def compile_lex_map(
    lex_map: LexMap,
    subtokens: dict[str, set[str]] = frozendict.frozendict(),
) -> RustLexMap:
    # resets the lex cache as this indicates that we need a new lex map
    reset_lex_cache()
    # remove the languages of subtokens from supertokens (also the prefix languages)
    automata_tuple = {k: relevant_automata_from_regex(v) for k, v in lex_map.items()}
    automata_tuple_diffed = automata_tuple.copy()
    for k, v in automata_tuple.items():
        if not subtokens.get(k):
            continue
        v_diffed = []
        for i, a in enumerate(v):
            v_diffed.append(
                remove_subtokens(
                    a, [automata_tuple[subtoken][i] for subtoken in subtokens[k]]
                )
            )
        automata_tuple_diffed[k] = tuple(v_diffed)
    return RustLexMap.from_lex_map(automata_tuple_diffed)


# match any boundary unless at the start or end of the string (to allow suffixes/prefixes)
word_boundary_prefix_nostart = regex.compile(r"(?<!^)\b\w+")
word_boundary_suffix_noend = regex.compile(r"\w+\b(?!$)")
# match any boundary
word_boundary_prefix = regex.compile(r"\b\w+")
word_boundary_suffix = regex.compile(r"\w+\b")


class EOSType:
    pass


EOS = EOSType()


def all_prefix_lexings(
    token: str,
    prelex: str | None,
    lex_map: RustLexMap,
    strip_chars: str | None = None,
):
    # prelex the token if prelex is provided
    lexings = set()
    for possible_pos in (
        (True, True),
        (True, False),
        (False, False),
        (False, True),
    ):
        token = prelex_word(
            token, prelex, is_first=possible_pos[0], is_last=possible_pos[1]
        )
        lexings.update(
            lex(
                token,
                lex_map,
                is_first=possible_pos[0],
                strip_chars=strip_chars,
            )
        )
    return lexings


def all_lexings_mask(
    vocab: list[str],
    lex_map: RustLexMap,
    trace=False,
    strip_chars=None,
    prelex: str | None = None,
) -> Tuple[dict, np.ndarray]:
    """
    Compute all possible lexings of each vocabulary token
    Also maintain a map of lexing -> mask
    Where the mask contains a 1 at the index of the token in the lexing
    """
    all_possible_lexings_map = {}
    no_lexing_tokens = np.zeros((len(vocab),), dtype=np.float64)
    all_lexs = all_lexings(vocab, lex_map, prelex, strip_chars)
    for i, (token, lexings) in enumerate(
        tqdm(zip(vocab, all_lexs)) if trace else zip(vocab, all_lexs)
    ):
        if not lexings:
            no_lexing_tokens[i] = 1
            continue
        for lexing in lexings:
            lexing = (tuple(lexing[0]), lexing[1], lexing[2])
            if lexing not in all_possible_lexings_map:
                all_possible_lexings_map[lexing] = np.zeros(
                    (len(vocab),), dtype=np.float64
                )
            all_possible_lexings_map[lexing][i] = 1 / len(lexings)

    return all_possible_lexings_map, no_lexing_tokens


def collect_subtokens(grammar: CFG, lex_map: LexMap):
    to_process = list(lex_map.keys())
    subtokens = {}

    for super_token in to_process:
        super_automaton = lex_map[super_token]
        if isinstance(super_automaton, str):
            super_automaton = regex_to_dfa(super_automaton)
            lex_map[super_token] = super_automaton

        sub_tokens = set()
        # find all tokens that are a subset of the string token
        for key, value in list(lex_map.items()):
            if key == super_token:
                continue
            other_automaton = value
            if isinstance(value, str):
                other_automaton = regex_to_dfa(value)
                lex_map[key] = other_automaton
            # is a subset iff the difference is empty
            if other_automaton.difference(super_automaton).is_empty():
                sub_tokens.add(key)
        if not sub_tokens:
            # nothing to do
            continue
        subtokens[super_token] = list(sub_tokens)
        # substitute "stringToken" with stringToken | otherTokens
        grammar = grammar.substitute(
            {
                super_token: CFG.from_text(
                    f"S -> {super_token} | " + " | ".join(sub_tokens), "S"
                )
            }
        )
    return grammar, lex_map, subtokens


def remove_subtokens(super_dfa: BytesDFA, sub_dfas: list[BytesDFA]):
    super_automaton = super_dfa

    big_union = ENFA()
    for sub_dfa in sub_dfas:
        big_union = big_union.union(sub_dfa.to_epsilon_automaton())
    # other make the super token not accept the other tokens
    big_union = big_union.minimize().to_bytes_dfa()
    super_diffed = super_automaton.difference(big_union)
    super_diffed = super_diffed.minimize()
    return super_diffed


def derive_supertokens(subtokens: dict[str, list[str]]):
    supertokens = defaultdict(set)
    for k, v in subtokens.items():
        for subtoken in v:
            supertokens[subtoken].add(k)
    return {tk: list(stks) for tk, stks in supertokens.items()}


def preprocessed_generate_stuff(
    tokenizer,
    constraint_lang: CFG,
    lex_map: RustLexMap,
    trace=False,
    prelex: str | None = None,
    subtokens: dict[str, list[str]] = frozendict.frozendict(),
    strip_chars: str = None,
):
    supertokens = derive_supertokens(subtokens)
    return None, None, supertokens
    # compile the lex_map
    # rules out the lexings of reserved tokens too
    # however we need to manually allow EOS again
    all_tokens_decoded = tokenizer.batch_decode(
        torch.arange(0, tokenizer.vocab_size + len(tokenizer.added_tokens_decoder))
    )
    all_possible_lexings, no_lexing_tokens = all_lexings_mask(
        all_tokens_decoded,
        lex_map,
        trace,
        strip_chars=strip_chars,
        prelex=prelex,
    )
    if trace:
        print("All possible lexings:", len(all_possible_lexings))
        print("Maximum lexing size:", max([len(x[0]) for x in all_possible_lexings]))
        print(
            "Average lexing size:",
            np.mean([len(x[0]) for x in all_possible_lexings]),
        )
    # further rule out all lexings that can not appear in the grammar
    _lexings = list(all_possible_lexings)
    terminals = constraint_lang.get_terminals()
    for lexing in tqdm(_lexings) if trace else _lexings:
        lexing_lang = generated_language(
            [None, lexing, None],
            lex_map,
            terminals,
            prelex=prelex,
            subtokens=subtokens,
            supertokens=supertokens,
            strip_chars=strip_chars,
        )
        intersection_empty = constraint_lang.is_intersection_empty(lexing_lang, 100)
        if intersection_empty:
            # this lexing is ruled out
            no_lexing_tokens += all_possible_lexings[lexing]
            all_possible_lexings.pop(lexing)
    for eot_index in [
        k
        for k, v in tokenizer.added_tokens_decoder.items()
        if v.content
        # NOTE: these need to be updated for each new model
        in (
            "<|eot_id|>",
            "<|im_end|>",
            "<file_sep>",
            "<｜end▁of▁sentence｜>",
            "<|endofmask|>",
            "<eom>",
            "<eos>",
            "<|file_separator|>",
            "</s>",
            "<fim_middle>",
            "<MID>",
            "<|dlm_pad|>",
            tokenizer.special_tokens_map.get("eos_token", "<|endoftext|>"),
        )
    ]:
        no_lexing_tokens[eot_index] = 0
    if trace:
        remaining_lexigns = set(all_possible_lexings)
        print("All possible remaining lexings:", len(remaining_lexigns))
        if len(remaining_lexigns) < 10:
            print("Remaining lexings:", remaining_lexigns)
        print("Maximum lexing size:", max([len(x[0]) for x in remaining_lexigns]))
        print(
            "Average lexing size:",
            np.mean([len(x[0]) for x in remaining_lexigns]),
        )
    return all_possible_lexings, no_lexing_tokens, supertokens


def interleave_with_value(lst, value):
    if not lst:
        return []
    return list(islice(chain.from_iterable(zip(lst, repeat(value))), 2 * len(lst) - 1))


def lex(
    word: str,
    lex_map: RustLexMap,
    is_first: bool = False,
    strip_chars: str | None = None,
) -> set[Tuple[tuple[str, ...], str | None, str | None]]:
    """
    Lex a word using the provided lex_map.

    Args:
        word (str): The word to lex.
        lex_map (RustLexMap): The lex map to use for lexing.
        is_first (bool): Whether this is the first word.
        is_last (bool): Whether this is the last word.
        strip_chars (str | None): Characters to strip from the word.

    Returns:
        List[Tuple[tuple[str, ...], bool, bool]]: A list of tuples containing the lexing results.
        Each tuple contains:
            - A tuple of strings representing the lexing tokens.
            - A boolean indicating if the first token is partial.
            - A boolean indicating if the last token is partial.
    """
    res = lex_string(word, lex_map, is_first, strip_chars)
    return {(tuple(r[0]), r[1], r[2]) for r in res}


def generated_language(
    tokens: list[str | None | EOSType],
    lex_map: RustLexMap,
    terminals: list[str],
    trace=False,
    prelex: str | None = None,
    single_token_lexing=None,
    inject_gap_size=0,  # if inject gap size > 0, inject all possible lexings for single words in gaps of less than inject_gap_size
    max_total_injections=0,
    subtokens: dict[str, list[str]] = frozendict.frozendict(),
    supertokens: dict[str, list[str]] = frozendict.frozendict(),
    strip_chars=None,
) -> DFA:
    """
    Generate a language that represents the already generated tokens.
    Args:
        tokens: List of None or str for missing or generated tokens.

    Returns:
        Nondeterministic finite automaton representing the generated language.

    """
    assert (
        single_token_lexing is not None or inject_gap_size == 0
    ), "inject_gap_size > 0 requires single_token_lexing"
    assert (not supertokens and not subtokens) or (
        supertokens and subtokens
    ), "Either both or none of supertokens and subtokens must be provided"
    assert (
        max_total_injections >= inject_gap_size
    ), "max_total_injections must be greater or equal than inject_gap_size, probably a config error"
    # if there are no tokens, the language is empty
    if not tokens:
        return minimize_enfa_threaded(ENFA())
    # we need to be efficient here, so we build this automaton by hand
    tokens = tokens.copy()

    # first collect and merge all generated words
    constrain_words = []
    new_word = None
    token = EOS
    gap_size = 0
    for token in tokens:
        if token is EOS:
            break
        if isinstance(token, tuple):
            # this is a lexing
            if new_word is not None:
                constrain_words.append((new_word, gap_size))
                gap_size = 0
            new_word = token
        elif token is None:
            # this indicates a non-generated gap
            if new_word is not None:
                constrain_words.append((new_word, gap_size))
                new_word = None
                gap_size = 0
            gap_size += 1
        else:
            # this is a generated token
            if isinstance(new_word, tuple):
                # this is a lexing
                constrain_words.append((new_word, gap_size))
                new_word = ""
                gap_size = 0
            elif new_word is None:
                new_word = ""
            new_word += token
    if new_word is not None:
        constrain_words.append((new_word, gap_size))
        new_word = None
        gap_size = 0
    # need to ensure that tokenEOS results in a different result than token<gap>EOS
    if token is EOS and gap_size == 0:
        last_token_eos_adj = True
    else:
        last_token_eos_adj = False
    last_token_gap = gap_size if token is EOS else math.inf

    # prelex the words to show word boundaries
    if prelex is not None:
        for i, (word, gap_size) in enumerate(constrain_words):
            constrain_words[i] = (
                (
                    prelex_word(
                        word,
                        prelex,
                        is_first=i == 0 and gap_size == 0,
                        is_last=last_token_eos_adj and i == len(constrain_words) - 1,
                    ),
                    gap_size,
                )
                if not isinstance(word, tuple)
                else (word, gap_size)
            )

    if trace:
        suffix = (
            (
                ""
                if last_token_gap == 0
                else f".{{{last_token_gap}}}"
                if last_token_gap <= inject_gap_size
                else " .*"
            )
            + "<EOS>"
            if EOS in tokens
            else " .*"
        )
        pattern = [""]
        for word, gap_size in constrain_words:
            if gap_size == 0:
                pattern.append(" ")
            elif gap_size <= inject_gap_size:
                # TODO this is not accurate when max_total_injections < inject_gap_size (or remaining)
                pattern.append(f" .{{{gap_size}}} ")
            else:
                pattern.append(" .* ")
            if isinstance(word, tuple):
                # this is a lexing
                pattern.append(f"({word})")
            else:
                pattern.append(word)
        pattern.pop(0)
        print(f"pattern: {''.join(pattern)}{suffix}")
        print("lexed: ")
        for i, (word, gap_size) in enumerate(constrain_words):
            lexi = (
                lex(
                    word,
                    lex_map,
                    is_first=i == 0,
                    strip_chars=strip_chars,
                )
                if not isinstance(word, tuple)
                else (word,)
            )
            print(lexi)

    # now we have a list of words, we need to build the automaton
    constrained_fsa = ENFA()
    total_injections = 0

    def admissable_prefix_suffix_combo(token: str, prefixes: list[str], suffix: str):
        token_dfa = lex_map[token][0]
        matching_dfa = regex_to_dfa(
            r"[\x00-\xFF]*".join(regex_escape(x) for x in [*prefixes, suffix])
        )
        example_word = token_dfa.intersection_example_word(matching_dfa)
        return example_word is not None

    def inject_lexings(
        wi,
        prev_last_tokens: defaultdict[str, list[tuple[str, list[str]]]],
        lexings: Iterable[tuple[list[str], str | None, str | None]],
        fill_prev_gap: bool,
        fill_next_gap: bool,
    ):
        """
        Injects a set of possible lexings for a position into the automaton
        Args:
            wi:
            prev_last_tokens:
            lexings:

        Returns:

        """
        k = 0
        current_last_tokens = defaultdict(list)
        for lexing, first_partial, last_partial in lexings:
            # Note: if there is no valid lexing, we generate an empty FSA (there is no connection between these words)
            # This is expected
            if lexing:
                prev = wi - 1
                if (
                    first_partial is not None
                    and len(lexing) == 1
                    and last_partial is not None
                ):
                    # if this token itself is partial, we carry the previous connection over
                    # we need to also do this for sub and super tokens
                    for subeqtoken in set(subtokens.get(lexing[0], [])) | {lexing[0]}:
                        current_last_tokens[lexing[0]].extend(
                            (x, y + [first_partial])
                            for x, y in prev_last_tokens[subeqtoken]
                        )
                    for suptoken in set(supertokens.get(lexing[0], [])):
                        current_last_tokens[suptoken].extend(
                            (x, y + [first_partial])
                            for x, y in prev_last_tokens[suptoken]
                        )
                elif first_partial is not None:
                    # we add a skip connection that these two might be the same
                    # NOTE: we also add skip connections for any subtokens of lexing[0]
                    # and for any supertoken of lexing[0]
                    # i.e. all of the following are possible
                    # dateToken dateToken -> dateToken
                    # dateToken dateToken -> stringToken
                    # stringToken dateToken -> stringToken
                    # dateToken stringToken -> stringToken
                    # however due to preprocessing of the grammar, dateToken should also pass anywhere a stringToken is expected

                    # if the previous is a sub token, we will become the current lexing
                    for subeqtoken in set(subtokens.get(lexing[0], [])) | {lexing[0]}:
                        for prevprev in prev_last_tokens[subeqtoken]:
                            prev_state, prev_prefix = prevprev
                            # check whether the current first_partial suffix is compatible with the previous prefix
                            if not admissable_prefix_suffix_combo(
                                lexing[0], prev_prefix, first_partial
                            ):
                                continue
                            # through merging we can become either the same token or a supertoken
                            constrained_fsa.add_transition(
                                str(prev_state),
                                lexing[0],
                                f"l{wi - 1}_{k}" if len(lexing) > 1 else str(wi),
                            )
                    # if the previous is a super token, we will become the supertoken
                    for suptoken in set(supertokens.get(lexing[0], [])):
                        for prevprev in prev_last_tokens[suptoken]:
                            prev_state, prev_prefix = prevprev
                            # check whether the current first_partial suffix is compatible with the previous prefix
                            if not admissable_prefix_suffix_combo(
                                suptoken, prev_prefix, first_partial
                            ):
                                continue
                            constrained_fsa.add_transition(
                                str(prev_state),
                                suptoken,
                                f"l{wi - 1}_{k}" if len(lexing) > 1 else str(wi),
                            )
                for token in lexing[:-1]:
                    next = f"l{wi - 1}_{k}"
                    # we do not allow transitioning into partial tokens if the gap is 0,
                    # e.g. { "hello" ("string", true, false) should result in an invalid language since "hello" is a completed string
                    # only valid for 0-gaps/filled gaps through the skip connections above
                    if k != 0 or (not first_partial or not fill_prev_gap):
                        constrained_fsa.add_transition(str(prev), token, str(next))
                    prev = next
                    k += 1
                # we may only transition into a last partial token if the gap is not going to be filled
                if (k != 0 or (not first_partial or not fill_prev_gap)) and (
                    not last_partial or not fill_next_gap
                ):
                    constrained_fsa.add_transition(str(prev), lexing[-1], str(wi))
                # if the last is partial, it may be the same as the start of the next
                # in case the gap is filled this is the only valid further transition
                if last_partial and not (len(lexing) == 1 and first_partial):
                    current_last_tokens[lexing[-1]].append((prev, [last_partial]))
            else:
                constrained_fsa.add_transition(str(wi - 1), epsilon(), str(wi))
        prev_last_tokens = current_last_tokens
        wi += 1
        return wi, prev_last_tokens

    start = 0
    wi = 1
    # stores for each token the last state that ended with a prefix of that token and the prefix (list of prefixes) of that token
    prev_last_tokens: defaultdict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for pos, (word, gap_size) in enumerate(constrain_words):
        is_first = wi == 1 and tokens[0] is not None
        # indicates whether we filled this gap
        fill_prev_gap = (
            gap_size <= inject_gap_size
            and total_injections + gap_size <= max_total_injections
        )
        next_gap_size = (
            constrain_words[pos + 1][1]
            if pos + 1 < len(constrain_words)
            else last_token_gap
        )
        fill_next_gap = (
            next_gap_size <= inject_gap_size
            and total_injections + (fill_prev_gap * gap_size) + next_gap_size
            <= max_total_injections
        )
        if not is_first and gap_size > 0:
            if fill_prev_gap:
                # inject all possible lexings for single words in gaps of less than inject_gap_size
                for _ in range(gap_size):
                    wi, prev_last_tokens = inject_lexings(
                        wi, prev_last_tokens, single_token_lexing, True, True
                    )
                    total_injections += 1
                    # this could be an accepting state (by inserting EOS)
                    # but we don't model it as the DLLM very unlikely will insert EOS mid-sequence
            else:
                # add the .* between fixed lexings for non-zero gaps
                # all tokens can be repeated indefinitely here
                for token in terminals:
                    constrained_fsa.add_transition(str(wi - 1), token, str(wi - 1))
                # and all previous skip tokens can end here
                for prev_token in prev_last_tokens:
                    for prevprev in prev_last_tokens[prev_token]:
                        prev_state, prev_prefix = prevprev
                        if not admissable_prefix_suffix_combo(
                            prev_token, prev_prefix, ""
                        ):
                            continue
                        constrained_fsa.add_transition(
                            str(prev_state), prev_token, str(wi - 1)
                        )
                # and all terminals can start here
                for token in terminals:
                    prev_last_tokens[token].append((str(wi - 1), [""]))
                # this could be an accepting state (by inserting EOS)
                # but we don't model it as the DLLM very unlikely will insert EOS mid-sequence

        wi, prev_last_tokens = inject_lexings(
            wi,
            prev_last_tokens,
            lex(
                word,
                lex_map,
                is_first=is_first,
                strip_chars=strip_chars,
            )
            if not isinstance(word, tuple)
            else (word,),
            fill_prev_gap,
            fill_next_gap,
        )
    gap_size = last_token_gap
    # gap_size > 0 -->  ensure that tokenEOS results in not being able to attend further .*
    if not last_token_eos_adj and gap_size > 0:
        if (
            gap_size <= inject_gap_size
            and total_injections + gap_size < max_total_injections
        ):
            # inject all possible lexings for single words in gaps of less than inject_gap_size
            for _ in range(gap_size):
                # this could be an accepting state (by inserting EOS)
                constrained_fsa.add_accept_state(str(wi - 1))
                # inject a single token lexing
                wi, prev_last_tokens = inject_lexings(
                    wi, prev_last_tokens, single_token_lexing, True, True
                )
                total_injections += 1
        else:
            for token in terminals:
                constrained_fsa.add_transition(str(wi - 1), token, str(wi - 1))
            # and all previous skip tokens can end here
            for prev_token in prev_last_tokens:
                for prevprev in prev_last_tokens[prev_token]:
                    prev_state, prev_prefix = prevprev
                    if not admissable_prefix_suffix_combo(prev_token, prev_prefix, ""):
                        continue
                    constrained_fsa.add_transition(
                        str(prev_state), prev_token, str(wi - 1)
                    )
    constrained_fsa.set_start_state(str(start))
    constrained_fsa.add_accept_state(str(wi - 1))
    if trace:
        print("Num states (unmin):", constrained_fsa.num_states())
        # src = graphviz.Source(constrained_fsa.to_graphviz())
        # src.render('doctest-output/constrained-fsa.gv', view=True).replace('\\', '/')
    constrained_fsa = minimize_enfa_threaded(constrained_fsa)
    if trace:
        print("Num states (min):  ", constrained_fsa.num_states())
        print("Empty fsa:   ", constrained_fsa.is_empty())
    return constrained_fsa


def autocomplete_valid(
    partial_output: list[str | None | EOSType],
    first_token_gap: bool,
    last_token_eos_adj: bool,
    generated_lang: DFA,
    lex_map: LexMap,
    subtokens: dict[str, list[str]],
    constraint_lang: CFG,
    trace: bool = False,
) -> str:
    # first obtain a valid lexing sequence that conforms to CFG and the generated language
    example_valid_lexings = constraint_lang.example_word(generated_lang, timeout=60)
    if trace:
        print(
            "*************** Example valid lexings ***********\n", example_valid_lexings
        )
    if example_valid_lexings is None:
        return None
    lexings_list = example_valid_lexings.split(" ")
    if len(lexings_list) > 50_000:
        if trace:
            print(
                f"Too many lexings {len(lexings_list)}, skipping autocomplete. "
                "This is likely due to a too large lex_map or CFG."
            )
        return None
    lang_lexings = language_from_words(lexings_list, lex_map, subtokens=subtokens)
    if trace:
        print(
            "*************** Example word from lexings language ***********\n",
            lang_lexings.example_word(),
        )

    # then obtian a valid bytesequence that matches the lexing constraints and the actual current bytes output
    lang_partial_output = language_from_program_with_gaps(
        partial_output, first_token_gap, last_token_eos_adj
    )
    if trace:
        print(
            "**************** Example word from partial output language ***************\n",
            lang_partial_output.example_word(),
        )
        print("Constructing intersection of languages...")
    return lang_partial_output.intersection_example_word(lang_lexings)


def partial_output_and_gaps_from_tokens(
    tokens: list[str | None | EOSType], prelex=None
) -> Tuple[list[Tuple[str, int]], bool, bool, int]:
    # first collect and merge all generated words
    constrain_words = []
    new_word = None
    token = EOS
    gap_size = 0
    for token in tokens:
        if token is EOS:
            break
        elif token is None:
            # this indicates a non-generated gap
            if new_word is not None:
                constrain_words.append((new_word, gap_size))
                new_word = None
                gap_size = 0
            gap_size += 1
        else:
            # this is a generated token
            if new_word is None:
                new_word = ""
            new_word += token
    if new_word is not None:
        constrain_words.append((new_word, gap_size))
        new_word = None
        gap_size = 0
    # need to ensure that tokenEOS results in a different result than token<gap>EOS
    if token is EOS and gap_size == 0:
        last_token_eos_adj = True
    else:
        last_token_eos_adj = False
    last_token_gap = gap_size if token is EOS else math.inf
    # prelex the words to show word boundaries
    if prelex is not None:
        for i, (word, gap_size) in enumerate(constrain_words):
            constrain_words[i] = (
                prelex_word(
                    word,
                    prelex,
                    is_first=i == 0 and gap_size == 0,
                    is_last=last_token_eos_adj and i == len(constrain_words) - 1,
                ),
                gap_size,
            )
    first_token_gap = len(tokens) and tokens[0] is None
    return constrain_words, first_token_gap, last_token_eos_adj, last_token_gap


def partial_output_from_tokens(
    tokens: list[str | None | EOSType],
    prelex: str | None = None,
) -> tuple[list[str | EOSType], bool, bool]:
    """
    Extract the partial output from a list of tokens.
    This will merge consecutive tokens into words and return a list of words with gaps represented as None.
    """
    constrain_words, first_token_gap, last_token_eos_adj, last_token_gap = (
        partial_output_and_gaps_from_tokens(tokens, prelex=prelex)
    )
    return [x for x, _ in constrain_words], first_token_gap, last_token_eos_adj


def language_from_words(
    words: list[str], lex_map: LexMap, subtokens: dict[str, list[str]]
) -> BytesDFA:
    dfa = regex_to_dfa(r"\s*").to_epsilon_automaton()
    for w in words:
        w_regex = lex_map[w]
        lexing_and_subtokens = (
            regex_to_dfa(w_regex) if isinstance(w_regex, str) else w_regex
        )
        dfa.concat(lexing_and_subtokens.to_epsilon_automaton())
        dfa.concat(regex_to_dfa(r"\s*").to_epsilon_automaton())
    return dfa.to_deterministic().to_bytes_dfa()


def any_byte_dfa() -> ENFA:
    """
    Returns a DFA that accepts any byte sequence.
    This is used to represent gaps in the language.
    """
    return regex_to_dfa(r"[\x00-\xff]*").to_epsilon_automaton()


def language_from_program_with_gaps(
    s: list[str], first_token_gap: bool, last_token_eos_adj: bool
) -> BytesDFA:
    if not s:
        # since EOS is missing, could be anything
        return any_byte_dfa().to_deterministic().to_bytes_dfa()
    # otherwise, construct the DFA from the given sequence (where gaps are not explicitly represented)
    dfa = regex_to_dfa(regex_escape(s[0])).to_epsilon_automaton()
    if first_token_gap:
        # if the first token is a gap, we need to allow any byte sequence before the first token
        temp_dfa = any_byte_dfa()
        temp_dfa.concat(dfa)
        dfa = temp_dfa
    for w in s[1:]:
        if isinstance(w, str):
            dfa.concat(any_byte_dfa())
            dfa.concat(regex_to_dfa(regex_escape(w)).to_epsilon_automaton())
        elif w is EOS:
            break
    if not last_token_eos_adj:
        dfa.concat(any_byte_dfa())
    return dfa.to_deterministic().to_bytes_dfa()


def reconstruct_word_boundaries(word, prelex="\u0002\u0003"):
    """
    Reconstruct the word boundaries from a given word with \u0002 and \u0003 characters.

    if encountering \u0002, it is the start of a word, and if encountering \u0003, it is the end of a word.
    --> need to make sure that if it is not the start or end of a word, we insert space left/right
    """

    def is_boundary(matchobj):
        return re.match(r".\b.", matchobj.group(1) + matchobj.group(2))

    def space_or_empty_if_boundary(matchobj):
        if not is_boundary(matchobj):
            return " "
        else:
            return ""

    def sub_no_line_end(matchobj):
        return (
            matchobj.group(1) + space_or_empty_if_boundary(matchobj) + matchobj.group(2)
        )

    prev_word = None
    subbed_line_end = word
    while subbed_line_end != prev_word:
        prev_word = subbed_line_end
        subbed_no_line_end = re.sub(
            rf"(.){prelex[0]}(.)",
            sub_no_line_end,
            re.sub(rf"(.){prelex[1]}(.)", sub_no_line_end, subbed_line_end),
        )
        subbed_line_end = re.sub(
            rf"(^|\n){prelex[0]}",
            r"\1",
            re.sub(rf"{prelex[1]}($|\n)", r"\1", subbed_no_line_end, re.MULTILINE),
            re.MULTILINE,
        )
    return subbed_line_end
