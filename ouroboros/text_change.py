"""The exact two-way change between two texts: what left, what came, and where.

A change is a sorted list of non-overlapping ``(start, removed, added)`` triples: ``start`` is a
UTF-8 byte offset in the OLD text, ``removed`` the text that leaves there and ``added`` the text
that takes its place. ``apply_change`` rebuilds the new text from the old one and
``unapply_change`` the old text from the new one; both are exact for every input and refuse a
change that does not fit the text they are given.

``compute_change`` finds the change: the lines both texts share at their start and end are left
out, ``difflib.SequenceMatcher`` pairs the remaining lines, and every replaced block is tightened
by the characters its two sides share at their edges. Line ends are ordinary characters, so CRLF,
a missing final newline and multibyte text need no special case. The diff is only as good as
``difflib`` makes it, and its work has a deterministic bound, ``_MATCH_PAIRS``: a bound on diff
work inside the storage code, not a judgment rule. Past it the diff only grows coarser (popular
lines stop seeding matches, then the middle is one replaced block), never inexact.
``exact_change`` is what a caller stores or shows as exact: it rebuilds both sides and falls back
to one span that replaces everything. Pure: no I/O, no clock.
"""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from itertools import accumulate
from typing import Iterable, List, Tuple

Change = List[Tuple[int, str, str]]

# The diff's work bound: pairs of equal lines the matcher may visit (it visits every pair of
# equal lines of the two middles, so repeated lines make it quadratic). At most ~0.65 s was
# measured at this bound; past it popular lines stop seeding matches (difflib's autojunk rule),
# and past it again the middle is one replaced block — still exact, only coarser. It bounds
# diff work, never what the mind may write.
_MATCH_PAIRS = 1_000_000


def _lines(raw: bytes) -> list[bytes]:
    """The text's LF lines, each with its own line end; a last line without one is kept as it is."""
    parts = raw.split(b"\n")
    return [part + b"\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _tighten(start: int, removed: bytes, added: bytes) -> tuple[int, str, str]:
    """Drop the characters both sides of one block share at their edges."""
    old, new = removed.decode("utf-8"), added.decode("utf-8")
    limit = min(len(old), len(new))
    prefix = 0
    while prefix < limit and old[prefix] == new[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit - prefix and old[-1 - suffix] == new[-1 - suffix]:
        suffix += 1
    return (start + len(old[:prefix].encode("utf-8")),
            old[prefix:len(old) - suffix], new[prefix:len(new) - suffix])


def _blocks(a: list[bytes], b: list[bytes]) -> list[tuple[str, int, int, int, int]]:
    """``SequenceMatcher`` opcodes for two line lists, within the matcher's work bound."""
    whole = [("replace", 0, len(a), 0, len(b))]
    if not a or not b:
        return whole
    counts = Counter(b)
    if sum(counts[line] for line in a) <= _MATCH_PAIRS:
        return SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    popular = len(b) // 100 + 1 if len(b) >= 200 else len(b)  # difflib's own autojunk threshold
    if sum(counts[line] for line in a if counts[line] <= popular) <= _MATCH_PAIRS:
        return SequenceMatcher(None, a, b).get_opcodes()
    return whole


def compute_change(old: str, new: str) -> Change:
    """The change that turns ``old`` into ``new``; verify it before calling it exact (``exact_change``)."""
    a, b = _lines(old.encode("utf-8")), _lines(new.encode("utf-8"))
    limit = min(len(a), len(b))
    head = 0
    while head < limit and a[head] == b[head]:
        head += 1
    tail = 0
    while tail < limit - head and a[len(a) - 1 - tail] == b[len(b) - 1 - tail]:
        tail += 1
    middle_a, middle_b = a[head:len(a) - tail], b[head:len(b) - tail]
    if not middle_a and not middle_b:
        return []
    offsets = list(accumulate((len(line) for line in a), initial=0))
    change: Change = []
    for tag, i1, i2, j1, j2 in _blocks(middle_a, middle_b):
        if tag == "equal":
            continue
        start, removed, added = _tighten(offsets[head + i1], b"".join(middle_a[i1:i2]), b"".join(middle_b[j1:j2]))
        if removed or added:
            change.append((start, removed, added))
    return change


def _checked(change: Iterable) -> Change:
    out: Change = []
    for item in change:
        start, removed, added = item
        if type(start) is not int or start < 0 or not isinstance(removed, str) or not isinstance(added, str):
            raise ValueError("a change is a list of (start byte, removed text, added text)")
        out.append((start, removed, added))
    return out


def apply_change(old: str, change: Iterable) -> str:
    """``old`` with every span replaced; a span that is not where it says, or out of order, refuses."""
    raw, out, at = old.encode("utf-8"), [], 0
    for start, removed, added in _checked(change):
        gone = removed.encode("utf-8")
        if start < at or start > len(raw) or raw[start:start + len(gone)] != gone:
            raise ValueError(f"the change does not fit this text at byte {start}")
        out += [raw[at:start], added.encode("utf-8")]
        at = start + len(gone)
    out.append(raw[at:])
    return b"".join(out).decode("utf-8")


def unapply_change(new: str, change: Iterable) -> str:
    """The text ``apply_change`` turned into ``new``, rebuilt from ``new`` and the same change."""
    raw, out, at, shift = new.encode("utf-8"), [], 0, 0
    for start, removed, added in _checked(change):
        put, gone = added.encode("utf-8"), removed.encode("utf-8")
        position = start + shift
        if position < at or position > len(raw) or raw[position:position + len(put)] != put:
            raise ValueError(f"the change does not fit this text at byte {start}")
        out += [raw[at:position], gone]
        at = position + len(put)
        shift += len(put) - len(gone)
    out.append(raw[at:])
    return b"".join(out).decode("utf-8")


def exact_change(old: str, new: str) -> Change:
    """``compute_change`` proven both ways, else one span that replaces the whole text."""
    change = compute_change(old, new)
    try:
        if apply_change(old, change) == new and unapply_change(new, change) == old:
            return change
    except ValueError:
        pass
    return [(0, old, new)] if old != new else []


def line_blocks(old: str, new: str, change: Iterable) -> List[Tuple[int, int, int, int, int, int]]:
    """An exact ``change`` from ``old`` to ``new``, each block widened to the whole lines it touches.

    ``(old_start, old_end, new_start, new_end, removed_chars, added_chars)`` in UTF-8 bytes:
    ``old[old_start:old_end]`` became ``new[new_start:new_end]``; each span begins at a line start
    and ends after a line end (or at its text's end), on its own side. A block widens only over text
    both versions share, and blocks whose widened lines touch or overlap become one (its character
    counts are their sums). Every byte outside the blocks is the same in both texts, in order, so
    putting each block's new lines in place of its old lines turns ``old`` into ``new``.
    """
    old_raw, new_raw = old.encode("utf-8"), new.encode("utf-8")

    def ended(block: list) -> bool:  # both ends sit on a line boundary of their own text
        return all(at in (0, len(raw)) or raw[at - 1] == 0x0A for raw, at in ((old_raw, block[1]), (new_raw, block[3])))

    def merge(block: list, item: list) -> None:
        block[1], block[3], block[4], block[5] = item[1], item[3], block[4] + item[4], block[5] + item[5]

    blocks: list = []
    shift = 0
    for start, removed, added in _checked(change):
        gone, put = len(removed.encode("utf-8")), len(added.encode("utf-8"))
        item = [start, start + gone, start + shift, start + shift + put, len(removed), len(added)]
        shift += put - gone
        floor = 0
        if blocks:
            last = blocks[-1]
            if not ended(last):  # the last block's line runs on: to its end, unless that is in this item
                stop = old_raw.find(b"\n", last[1], item[0])
                if stop < 0:
                    merge(last, item)
                    continue
                last[3] += stop + 1 - last[1]
                last[1] = stop + 1
            floor = last[1]
        begin = old_raw.rfind(b"\n", floor, item[0]) + 1 or floor
        if blocks and begin == floor:  # this item's first line begins where the last block ends: they touch
            merge(blocks[-1], item)
            continue
        item[2] -= item[0] - begin
        item[0] = begin
        blocks.append(item)
    if blocks and not ended(blocks[-1]):
        last, stop = blocks[-1], old_raw.find(b"\n", blocks[-1][1])
        end = stop + 1 if stop >= 0 else len(old_raw)
        last[3] += end - last[1]
        last[1] = end
    return [tuple(block) for block in blocks]


def change_chars(change: Iterable) -> tuple[int, int]:
    """Characters removed and added by a change, counted separately."""
    removed = added = 0
    for _start, gone, put in change:
        removed += len(gone)
        added += len(put)
    return removed, added
