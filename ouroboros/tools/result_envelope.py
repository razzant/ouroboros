"""Typed tool-result envelope (issue #447, В12=A minimal variant).

A producer that knows its outcome as TYPED FACTS (status, policy contract,
remaining route) stamps them on the result text instead of leaving the
outcome to the first-line ``⚠️ MARKER`` parse downstream. The envelope IS
the plain string (a ``str`` subclass), so every existing transport —
history, JSON logs, f-string wrappers that drop the annotation — keeps
working; a wrapper that drops the annotation degrades to the first-line
parse, never to a crash.

Host-appended notes (auto-route note, safety warning, post-exec tripwires)
TRAIL the payload: line 1 always belongs to the producer, and
``payload_text`` preserves the pre-note payload so structured detection
(e.g. a JSON ``{"ok": false}`` extension answer) survives appended notes.
"""

from typing import Any, Dict, Optional


class ToolResultText(str):
    """A tool-result string carrying typed outcome facts for result_meta."""

    result_meta: Dict[str, Any]
    payload_text: str


def _wrap(text: str, meta: Dict[str, Any], payload_text: str) -> ToolResultText:
    wrapped = ToolResultText(text)
    wrapped.result_meta = meta
    wrapped.payload_text = payload_text
    return wrapped


def typed_result_meta(result: Any) -> Optional[Dict[str, Any]]:
    """The typed facts stamped by the producer, or None for a plain string."""
    meta = getattr(result, "result_meta", None)
    return meta if isinstance(meta, dict) else None


def result_payload_text(result: Any) -> str:
    """The producer payload BEFORE host-appended notes."""
    payload = getattr(result, "payload_text", None)
    if isinstance(payload, str) and payload:
        return payload
    return str(result or "")


def annotate(result: Any, **meta: Any) -> ToolResultText:
    """Stamp typed facts on a result; a later stamp overrides earlier keys."""
    merged = dict(typed_result_meta(result) or {})
    merged.update(meta)
    return _wrap(str(result or ""), merged, result_payload_text(result))


def append_note(result: Any, note: str) -> Any:
    """Append a host note AFTER the payload, preserving typed facts.

    Line 1 stays with the producer so neither the typed-status seam nor the
    first-line fallback parse is masked by a note (#447 H1)."""
    note = str(note or "").strip()
    if not note:
        return result
    meta = dict(typed_result_meta(result) or {})
    notes = list(meta.get("notes") or [])
    notes.append(note)
    meta["notes"] = notes
    return _wrap(f"{result}\n\n{note}", meta, result_payload_text(result))
