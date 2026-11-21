"""A host notice: one phrase, its next step and its details (DESIGN principle 7 and "Chat authorship and System rows").

A producer of an owner-facing System row whose facts are raw (an exception, a code, a path, a worker slot, a commit)
writes the row's ``text`` as one phrase in owner words, what happened, with no pictograph and nothing raw, and stamps
one typed ``notice`` beside it in the row's ``progress_meta`` (``fields``):

* ``code`` — the notice's stable code;
* ``step`` — the owner's next step, a key of ``STEPS``: a control or a Settings place (a host that has scheduled the
  retry itself says so instead); absent when there is none;
* ``details`` — the raw facts, behind Details for the owner and beside the phrase for the mind.

The row keeps it through persistence, history and live delivery (``message_ingress``, ``gateway.history``); the window
draws it as it is, and a row without one (older history) keeps the display table ``web/modules/host_notice_text.js``.
Telegram and Copy take one line from ``notice_line``, which sends it and reads it alike (``view``); the mind reads the
step and the details beside the phrase (``mind_text``). Words stay the producer's: this module holds the steps, the
one closed vocabulary every notice shares, and none of the phrases.
"""

from __future__ import annotations

import logging
import pathlib
import threading
from typing import Any, Dict, Mapping, Optional

log = logging.getLogger(__name__)

# The owner's next steps, one sentence each, said after the phrase. The translation generator fills them by code
# (``ui_translation`` catalog ``notice.step``); a step names a control or a Settings place by its shown name. A row
# whose host has scheduled its own retry gets a ``retry_self`` step with the first producer that schedules one.
STEPS: Dict[str, str] = {
    "restart": "Restart Ouroboros to recover.",
    "send_after_restart": "Send it again after Restart.",
    "raise_total_budget": "Raise Total budget in Settings → Costs.",
    "choose_reviewer": "Run /review with a row this install can pay: any enabled row in Settings → Agents, or the Main model.",
    "evolution_start": "/evolve start resumes it.",
    "press_again": "You can press it again.",
}
_STEP_CODE = "code:notice.step."
_MISSES: Dict[str, set] = {}
_MISSES_LOCK = threading.Lock()


def notice(code: str, *, step: str = "", details: Any = "") -> Dict[str, Any]:
    """A row's ``notice``: the code, the step when there is one, the raw details when there are any. An unknown step
    is a producer defect, refused here rather than drawn as words nobody wrote."""
    if step and step not in STEPS:
        raise ValueError(f"unknown host notice step {step!r}")
    made: Dict[str, Any] = {"code": str(code)}
    if step:
        made["step"] = step
    if raw := str(details or "").strip():
        made["details"] = raw
    return made


def fields(code: str, *, step: str = "", details: Any = "") -> Dict[str, Any]:
    """``{"notice": notice(...)}``, merged into a row's ``progress_meta``."""
    return {"notice": notice(code, step=step, details=details)}


def notice_meta(meta: Optional[Mapping[str, Any]], code: str, *, step: str = "", details: Any = "") -> Dict[str, Any]:
    """A row's ``progress_meta`` with the notice merged in."""
    return {**(meta or {}), **fields(code, step=step, details=details)}


def with_notice(kwargs: Mapping[str, Any], code: str, *, step: str = "", details: Any = "") -> Dict[str, Any]:
    """A send's keyword arguments (a correlation's ``progress_meta`` among them) with the notice merged in."""
    return {**kwargs, "progress_meta": notice_meta(kwargs.get("progress_meta"), code, step=step, details=details)}


def notice_of(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The row's typed notice, or ``{}`` for every row written without one."""
    notice = row.get("notice") if isinstance(row, Mapping) else None
    return dict(notice) if isinstance(notice, Mapping) and str(notice.get("code") or "") else {}


def stored(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The notice a row persists: its code, step and details, never the line a reader was sent."""
    notice = notice_of(row)
    return {key: str(notice[key]) for key in ("code", "step", "details") if notice.get(key)}


def mind_text(row: Mapping[str, Any]) -> str:
    """What the mind reads beside a notice row's phrase: its step and its details, in English (BIBLE P1: the raw text
    an older row carried in its words stays in view)."""
    notice = notice_of(row)
    step = STEPS.get(str(notice.get("step") or ""), "")
    return (f"\n[Step: {step}]" if step else "") + (f"\n[Details: {notice['details']}]" if notice.get("details") else "")


def notice_line(row: Mapping[str, Any], lang: Optional[str] = None, *, drive_root: Optional[pathlib.Path] = None) -> str:
    """Telegram's and Copy's one line of a host row in the install's language: its phrase, then its step, each part
    translated before they are joined. The phrase is a host sentence, translated by its exact text for a notice row
    and for the technical pause notice, the one row Telegram translated so before notices;
    every other row keeps its text. A string the memory lacks stays English and is queued for the generator."""
    from ouroboros import i18n_memory as memory
    from ouroboros.ui_language import is_english, normalize_language_tag

    tag = normalize_language_tag(memory.current_language() if lang is None else lang) or ""
    english = not tag or is_english(tag)
    text = str(row.get("text") if row.get("text") is not None else row.get("content") or "").strip()
    notice = notice_of(row)
    pause = (row.get("system_type") == "task_pause_notice" and row.get("role") == "system"
             and row.get("pause_reason") == "authority")
    if (notice or pause) and not english and text:
        found = memory.tr(text, tag, drive_root=drive_root)
        if found is None:
            _miss(text, tag, {"source": text, "role": "host-text"}, drive_root)
        text = found or text
    step = str(notice.get("step") or "")
    if step not in STEPS:
        return text
    words = STEPS[step] if english else memory.tr(_STEP_CODE + step, tag, STEPS[step], drive_root=drive_root)
    return f"{text} {words}" if text else str(words)


def view(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The notice a reader is sent with a row (its live frame, a history read): the stored fields plus ``line``, the
    assembled phrase and step in the install's language. ``{}`` for a row without a notice."""
    notice = notice_of(row)
    if not notice:
        return {}
    try:
        notice["line"] = notice_line(row)
    except Exception:  # a translation fault never holds the row: the English it was written in stays its line
        log.debug("host notice line not assembled; the stored words stand", exc_info=True)
        notice["line"] = notice_line(row, "")
    return {"notice": notice}


def step_catalog() -> Dict[str, str]:
    """The steps for the translation generator (``ui_translation._core_catalogs``)."""
    return dict(STEPS)


def _miss(key: str, tag: str, context: Dict[str, Any], drive_root: Optional[pathlib.Path]) -> None:
    """Queue a phrase the memory lacks, once per process and language (the generator translates it later)."""
    with _MISSES_LOCK:
        seen = _MISSES.setdefault(tag, set())
        if key in seen:
            return
        seen.add(key)
    try:
        from ouroboros import i18n_memory as memory
        from ouroboros.config import DATA_DIR
        from ouroboros.ui_translation import on_language_event

        root = pathlib.Path(drive_root if drive_root is not None else DATA_DIR)
        if memory.record_missing(root, tag, [{"key": key, "context": context}])["accepted"]:
            on_language_event("missing", root, tag)
    except Exception:
        log.debug("host notice: translation miss not recorded", exc_info=True)
