"""Exact edits of one knowledge note's text: one anchored pair, or several exact items.

Every old text is located in the ORIGINAL text before anything changes: it must occur exactly
once (an overlapping repeat is a second occurrence), and no two spans may overlap, so one edit
can neither create, consume nor shift another's anchor; unmentioned bytes stay identical. A
refusal applies nothing. Pure: no I/O; ``knowledge.write_knowledge_note`` is the one caller that
publishes a result.
"""
from __future__ import annotations

from typing import Any, Mapping


def compile_anchored_edits(text: str, pairs: Any, anchor: str = "old_str") -> str:
    """Replace exact ``(old, new)`` spans of one source text atomically.

    Every old text is located in the ORIGINAL ``text`` before anything changes:
    it must be non-empty and occur exactly once (an overlapping repeat is a
    second occurrence), and no two spans may overlap, so one edit can neither
    create, consume nor shift another's anchor. Unmentioned characters stay
    identical. A refusal raises ``ValueError`` naming the caller's ``anchor``
    field and applies nothing. Manual ``mode=edit`` is the one-pair case.
    """
    pairs = list(pairs)
    spans = []
    for number, (old, new) in enumerate(pairs, 1):
        where = f" (edit {number})" if len(pairs) > 1 else ""
        if not isinstance(old, str) or not old:
            raise ValueError(f"edit requires a non-empty {anchor}{where}")
        start = text.find(old)
        if start < 0 or text.find(old, start + 1) >= 0:
            raise ValueError(f"edit {anchor} must occur exactly once in the note body{where}")
        spans.append((start, start + len(old), new))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("edits must not overlap")
    for start, end, new in reversed(spans):
        text = text[:start] + new + text[end:]
    return text


class EditRefused(ValueError):
    """One edit item cannot apply to the revision it was written against; the message names it."""


def edit_items(edits: Any) -> list[tuple[str, str]]:
    """``edits`` as ``(old_str, new_str)`` pairs: both text, ``old_str`` non-empty, no other field."""
    if not isinstance(edits, list) or not edits:
        raise ValueError("edits must be a non-empty array of {old_str, new_str} items")
    items = []
    for number, item in enumerate(edits, 1):
        if not isinstance(item, Mapping) or set(item) - {"old_str", "new_str"}:
            raise ValueError(f"edits item {number} must be an object with old_str and new_str only")
        if not isinstance(item.get("old_str"), str) or not item["old_str"]:
            raise ValueError(f"edits item {number}: old_str must be non-empty exact body text")
        if not isinstance(item.get("new_str"), str):
            raise ValueError(f"edits item {number}: new_str must be text (empty text deletes)")
        items.append((item["old_str"], item["new_str"]))
    return items


def _in_frontmatter(front: bytes, old: str) -> bool:
    """``old`` occurs in the frontmatter, also across the line breaks where YAML wraps a long value."""
    text, words = front.decode("utf-8"), " ".join(old.split())
    return old in text or bool(words) and words in " ".join(text.split())


def compile_edit_items(raw: bytes, body_start: int, items: list[tuple[str, str]]) -> bytes:
    """The body bytes of ``raw`` (from ``body_start``) with every item applied, all or none.

    Each ``old_str`` stands for its own bytes and is found in the ORIGINAL body, exactly once (an
    overlapping repeat counts); spans may touch, never overlap. ``EditRefused`` names the item: not
    found (and where, when the text is in the frontmatter instead), found N times (with the first
    eight of its lines in the whole note), or overlapping another. Bytes: a valid needle in valid
    UTF-8 lies on character boundaries, and unmentioned bytes stay as they were.
    """
    body, spans = raw[body_start:], []
    for number, (old, new) in enumerate(items, 1):
        needle, hits = old.encode("utf-8"), []
        at = body.find(needle)
        while at >= 0:
            hits.append(at)
            at = body.find(needle, at + 1)
        if len(hits) != 1:
            lines = ", ".join(str(raw.count(b"\n", 0, body_start + hit) + 1) for hit in hits[:8])
            raise EditRefused(f"item {number} (old_str): " + (
                f"found {len(hits)} times at lines {lines}" if hits else
                "not found in the body; it occurs in the frontmatter (a summary is revised with summary=...)"
                if _in_frontmatter(raw[:body_start], old) else "not found"))
        spans.append((hits[0], hits[0] + len(needle), number, new.encode("utf-8")))
    spans.sort()
    for left, right in zip(spans, spans[1:]):
        if left[1] > right[0]:
            first, second = sorted((left[2], right[2]))
            raise EditRefused(f"item {second}: overlaps item {first}")
    out, at = [], 0
    for start, end, _number, new in spans:
        out += [body[at:start], new]
        at = end
    return b"".join(out + [body[at:]])
