"""Exact edits of a memory text: one anchored pair, or several exact items, and the argument rules of every editor.

Every old text is located in the ORIGINAL text before anything changes: it must occur exactly
once (an overlapping repeat is a second occurrence), and no two spans may overlap, so one edit
can neither create, consume nor shift another's anchor; unmentioned bytes stay identical. A
refusal applies nothing. Pure: no I/O; ``knowledge.write_knowledge_note`` and
``memory_texts.write_text`` publish a result.

``normalize_arguments`` and ``check_arguments`` are ``knowledge_write``'s rules for the jobs every
exact editor shares (``mode``, ``content``, ``old_str`` with ``content`` or its alias ``new_str``,
``edits``, ``version``, ``reason``): one parser for the note writer and the writers of identity,
the world map and a project's workpad. A value that asks for nothing takes the omitted path and is
named in the result (``tools/arg_feedback``), except ``mode``: a null, blank or unknown mode is
refused as received (``mode_received``), never taken as a default; each field keeps its own meaning (an empty
``content``/``new_str`` deletes, a whitespace ``old_str`` is an exact anchor, an empty
``expected_revision`` is create-only and is the caller's). The note writer's summary, archive and
lifecycle rules stay its own: a caller names its modes, the modes that take no content, the modes
that require a reason and what an edit may revise beside the text (``alone``).
"""
from __future__ import annotations

from typing import Any, Mapping


def mode_received(mode: Any, null: str = "missing or null") -> str:
    """A ``mode`` that is not exactly an operation, as a refusal names it: as received, never trimmed, case-folded or
    matched to the nearest operation. ``null`` is what a ``None`` is to the caller: knowledge_write has no default, so
    a missing mode is the same refusal; the plain texts' writers keep overwrite for an OMITTED mode, so theirs is a
    null."""
    from ouroboros.tools.arg_feedback import ignored_argument_note

    if mode is None:
        return f"mode is {null}"
    if not isinstance(mode, str):
        return f"mode={mode!r} is not text"
    if not mode.strip():  # the canonical blank-value representation (tools/arg_feedback), in the refusal's words
        return ignored_argument_note("mode", mode, "is not an operation").replace(" ignored: is not an operation",
                                                                                  " is not an operation")
    return f"mode={mode!r} is not an operation"


def normalize_arguments(mode: Any, version: Any, expected_revision: Any, edits: Any, old_str: Any, reason: Any,
                        notes: list[str], *, reason_required: tuple[str, ...] = ()) -> tuple[Any, Any, Any, Any]:
    """``(version, edits, old_str, reason)`` with every value that asks for nothing on the omitted path, each named in ``notes``.

    Outside revert a blank version, or the revision the call already names as ``expected_revision``,
    asks for nothing; an empty ``edits`` list asks for no replacement; an empty ``old_str`` outside
    edit replaces nothing; a null reason records none (where a mode requires one, the check refuses it)."""
    from ouroboros.tools.arg_feedback import ignored_argument_note

    blank_version = isinstance(version, str) and not version.strip()  # empty or whitespace: it names no revision
    if mode != "revert" and (blank_version or isinstance(version, str) and version == expected_revision):
        notes.append(ignored_argument_note("version", version, ("" if blank_version else "the same as expected_revision; ")
                                           + "only mode=revert takes a version"))
        version = None
    if isinstance(edits, list) and not edits:
        notes.append(ignored_argument_note("edits", edits, "an empty list asks for no replacement"))
        edits = None
    if old_str == "" and mode != "edit":
        notes.append(ignored_argument_note("old_str", old_str, "only mode=edit replaces text"))
        old_str = None
    if reason is None and mode not in reason_required:  # optional text: null is an omitted reason
        notes.append(ignored_argument_note("reason", reason, "no reason is recorded"))
        reason = ""
    return version, edits, old_str, reason


def check_arguments(mode: Any, content: Any, old_str: Any, new_str: Any, edits: Any, version: Any, reason: Any,
                    notes: list[str], *, modes: tuple[str, ...], no_content: tuple[str, ...] = ("revert",),
                    reason_required: tuple[str, ...] = (), alone: str = "", alone_given: bool = False,
                    version_reader: str = "") -> tuple[Any, Any, Any]:
    """``(content, old_str, edits)`` after the shared rules, or ``ValueError`` naming the field and the repair.

    Run after ``normalize_arguments``. ``new_str`` is ``content``'s alias for one ``old_str``; an
    ``edits`` list carries its own replacements; ``old_str``/``edits`` belong to ``mode=edit`` only;
    ``mode=revert`` requires ``version`` and takes no content; a mode outside ``no_content`` requires
    ``content``. ``alone`` names what an edit may revise without ``old_str`` or ``edits`` (the
    note's summary) and ``alone_given`` says the call carries it; otherwise an edit requires one of
    them. ``version_reader`` is the read a refusal of a stray ``version`` names."""
    from ouroboros.tools.arg_feedback import ignored_argument_note

    if not isinstance(mode, str) or mode not in modes:  # each refusal names the one argument that is wrong
        raise ValueError(f"{mode_received(mode, 'null')}: mode is exactly one of " + ", ".join(modes))
    if not isinstance(content, (str, type(None))):
        raise ValueError(f"content must be Markdown text, not {type(content).__name__}")
    if new_str is not None:  # the one-pair edit's replacement as other editors spell it: content
        if not isinstance(new_str, str):
            raise ValueError("new_str must be text, the replacement of old_str (empty text deletes it)")
        if mode == "edit" and edits is not None and new_str:
            raise ValueError("each edits item carries its own new_str; a top-level new_str belongs to the "
                             "one-pair spelling with old_str")
        if mode == "edit" and isinstance(old_str, str) and old_str:
            if content and new_str and content != new_str:
                raise ValueError("new_str and content both give old_str's replacement and differ; pass one")
            if content is not None and bool(content) != bool(new_str):  # an empty one beside the other is filler
                empty, given = ("content", "new_str") if new_str else ("new_str", "content")
                notes.append(ignored_argument_note(empty, "", f"{given} gives the replacement"))
            content = content or new_str  # both empty: the replacement is empty text, which deletes
        elif new_str:
            raise ValueError("new_str is the replacement of old_str in mode=edit, the same as content; "
                             + ("pass old_str with it" if mode == "edit" else f"mode={mode} takes its text as content"))
        else:
            notes.append(ignored_argument_note("new_str", new_str, "only an edit with old_str takes a replacement"))
    if not isinstance(reason, str) or (mode in reason_required and not reason.strip()):
        raise ValueError(f"reason is required for {', '.join(reason_required)} and optional text on every other mode"
                         if reason_required else "reason is optional text, kept in history")
    if mode != "edit" and (old_str is not None or edits is not None):
        raise ValueError("old_str is used only with mode=edit" if old_str is not None
                         else "edits is used only with mode=edit")
    blank_version = isinstance(version, str) and not version.strip()
    if mode == "revert" and (version is None or blank_version):
        raise ValueError("mode=revert requires version, the revision to put back")
    if mode != "revert" and version is not None:
        raise ValueError("version is used only with mode=revert" + (f"; {version_reader} reads one" if version_reader else ""))
    if mode == "revert" and content:
        raise ValueError("mode=revert takes version, expected_revision and an optional reason; no content")
    if mode not in ("edit", *no_content) and content is None:
        raise ValueError(f"mode={mode} requires content")
    if mode in no_content and content is None:
        content = ""
    if mode == "edit" and edits is not None:
        if old_str not in (None, "") or content:
            raise ValueError("edits carries its own replacements: pass edits without old_str and content")
        content, old_str = "", None
    elif mode == "edit" and alone_given and old_str in (None, ""):  # "" asks for nothing
        if content:
            raise ValueError(f"content ({len(content)} chars) has no old_str to replace; pass old_str "
                             f"for a body edit, or omit content to revise only the {alone}")
        content, old_str = "", None
    elif mode == "edit" and (not isinstance(old_str, str) or not old_str):
        raise ValueError("mode=edit requires edits, or a non-empty old_str" + (f", or a {alone}" if alone else ""))
    elif mode == "edit" and content is None:
        raise ValueError("mode=edit with old_str requires content, its replacement "
                         "(empty text deletes the old_str span)")
    return content, old_str, edits


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
