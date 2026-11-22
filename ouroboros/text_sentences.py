"""Sentence openings shared by authored memory and display projections."""
from __future__ import annotations

import re

_STOP = re.compile(r'(?<=[.!?。！？])\s+')  # a sentence stop followed by whitespace


def first_sentence(text: str) -> str:
    """Keep the opening through its punctuation, without a length cut or a retelling."""
    return _STOP.split(str(text).strip(), maxsplit=1)[0]


def sentences(text: str) -> list[str]:
    """``text`` split at every stop the openings read (the connections map factors a server's shared sentences by it)."""
    return _STOP.split(str(text).strip())


def first_whole_sentence(text: str) -> str:
    """The opening through the first stop that ends a sentence, for text whose line would mislead if cut inside one
    (an MCP tool's description in the connections map).

    It reads the same stops as ``first_sentence`` but passes one followed by a lowercase letter or a digit, or a
    period closing a run of single letters (``e.g.``, ``А.С.``) or a bare number (``1.``), so it only ever keeps more.
    ``first_sentence`` stays the exact opening that stored names and memory pages compare.
    """
    text = str(text).strip()
    for match in _STOP.finditer(text):
        stop, following = match.start(), text[match.end():match.end() + 1]
        if following.islower() or following.isdigit():
            continue
        if text[stop - 1] == "." and (_closes_initials(text, stop) or _closes_number(text, stop)):
            continue
        return text[:stop]
    return text


def _closes_initials(text: str, stop: int) -> bool:
    """The period before ``stop`` ends two single letters each followed by a period, at a word's start."""
    if stop < 4 or not (text[stop - 4].isalpha() and text[stop - 3] == "." and text[stop - 2].isalpha()):
        return False
    return stop == 4 or not (text[stop - 5].isalnum() or text[stop - 5] == "_")


def _closes_number(text: str, stop: int) -> bool:
    """The period before ``stop`` ends digits that stand alone (a list number such as ``1.``)."""
    start = stop - 1
    while start > 0 and text[start - 1].isdigit():
        start -= 1
    return start < stop - 1 and (start == 0 or text[start - 1].isspace())
