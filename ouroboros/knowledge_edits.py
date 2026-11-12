"""Exact edits of one knowledge note's text: one anchored pair, or items naming text or a section.

Every selector is located in the ORIGINAL text before anything changes: it must occur exactly
once (an overlapping repeat is a second occurrence), and no two spans may overlap, so one edit
can neither create, consume nor shift another's anchor; unmentioned bytes stay identical. A
refusal applies nothing. Pure: no I/O; ``knowledge.write_knowledge_note`` is the one caller that
publishes a result.
"""
from __future__ import annotations

from typing import Any, Mapping

from ouroboros.markdown_source import MarkdownSource


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
    """One edit item cannot apply; ``outline`` says the note's outline would help the retry."""

    def __init__(self, message: str, outline: bool = False):
        super().__init__(message)
        self.outline = outline


def edit_items(edits: Any) -> list[tuple[str, str, str]]:
    """``edits`` as ``(kind, selector, new_str)``; an empty or null selector means "not this kind"."""
    if not isinstance(edits, list) or not edits:
        raise ValueError("edits must be a non-empty array of {old_str or section, new_str} items")
    items = []
    for number, item in enumerate(edits, 1):
        if not isinstance(item, Mapping) or set(item) - {"old_str", "section", "new_str"}:
            raise ValueError(f"edits item {number} must be an object with old_str, section and new_str")
        named = [(kind, item[kind]) for kind in ("old_str", "section") if item.get(kind) not in (None, "")]
        if len(named) != 1 or not isinstance(named[0][1], str):
            raise ValueError(f"edits item {number} must name exactly one selector, old_str or section, as text")
        if not isinstance(item.get("new_str"), str):
            raise ValueError(f"edits item {number}: new_str must be text (empty text deletes)")
        items.append((*named[0], item["new_str"]))
    return items


def compile_edit_items(raw: bytes, source: MarkdownSource, items: list[tuple[str, str, str]]) -> bytes:
    """The body bytes of ``raw`` with every item applied, all or none; ``EditRefused`` names the item.

    ``("old_str", text, new)`` stands for its own bytes; ``("section", text, new)`` for the whole
    section (``MarkdownHeading.section``: the heading through its last child) of the heading
    ``text`` begins at, so ``new`` replaces all of it. Each selector is found in the ORIGINAL body,
    exactly once (an overlapping repeat counts); spans may touch, never overlap. Bytes: a valid
    needle in valid UTF-8 lies on character boundaries, and unmentioned bytes stay as they were.
    """
    start_of_body = source.body_span.start_byte
    body, headings = raw[start_of_body:], {h.span.start_byte: h for h in source.headings}
    spans = []
    for number, (kind, selector, new) in enumerate(items, 1):
        needle, hits, section, where = selector.encode("utf-8"), [], kind == "section", f"item {number} ({kind})"
        at = body.find(needle)
        while at >= 0:
            hits.append(at)
            at = body.find(needle, at + 1)
        if len(hits) != 1:
            lines = ", ".join(str(raw.count(b"\n", 0, start_of_body + hit) + 1) for hit in hits[:8])
            raise EditRefused(f"{where}: " + (f"found {len(hits)} times at lines {lines}" if hits else "not found"), section)
        start, end = hits[0], hits[0] + len(needle)
        if section:
            heading = headings.get(start_of_body + start)
            if heading is None:
                raise EditRefused(f"{where}: does not begin at a heading line", True)
            if heading.section.start_byte != heading.span.start_byte:
                raise EditRefused(f"{where}: the heading at line {heading.span.start_line} has no section of its own "
                                  "(a setext heading); quote its text with old_str", True)
            if end > heading.section.end_byte - start_of_body:
                raise EditRefused(f"{where}: the quote runs past the end of its section (line "
                                  f"{heading.section.end_line}); name each section as its own item", True)
            end = heading.section.end_byte - start_of_body
        spans.append((start, end, number, section, new.encode("utf-8")))
    spans.sort()
    for left, right in zip(spans, spans[1:]):
        if left[1] > right[0]:
            first, second = sorted((left[2], right[2]))
            raise EditRefused(f"item {second}: overlaps item {first}", left[3] or right[3])
    out, at = [], 0
    for start, end, _number, _section, new in spans:
        out += [body[at:start], new]
        at = end
    return b"".join(out + [body[at:]])
