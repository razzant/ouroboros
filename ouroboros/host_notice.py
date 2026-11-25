"""A host notice: one phrase, its next step and its details (DESIGN principle 7 and "Chat authorship and System rows").

A producer of an owner-facing System row whose facts are raw (an exception, a code, a path, a worker slot, a commit)
writes the row's ``text`` as one phrase in owner words, what happened, with no pictograph and nothing raw, and stamps
one typed ``notice`` beside it in the row's ``progress_meta`` (``fields``):

* ``code`` — the notice's stable code;
* ``step`` — the owner's next step, a key of ``STEPS``: a control or a place in the interface (a host that has scheduled the
  retry itself says so instead); absent when there is none;
* ``details`` — the raw facts, behind Details for the owner and beside the phrase for the mind;
* ``instruction`` — one line the owner acts on, as written: the exact command a step sends him to (a Git command);
* ``template`` and ``values`` — only for a phrase that carries variable or authored words (a count, a name, the
  mind's own note): the constant English with ``{name}`` slots, and the words that fill them (``phrase``). The
  template is what gets translated; the values are inserted as they were written, never translated.

The row keeps it through persistence, history and live delivery (``message_ingress``, ``gateway.history``); the window
draws it as it is, and a row without one (older history) keeps the display table ``web/modules/host_notice_text.js``.
Telegram and Copy take one line from ``notice_line``, which sends it and reads it alike (``view``); Telegram, which has
no Details, says a notice's ``instruction`` under that line, and never its details (``with_instruction``). The mind
reads the step and the details beside the phrase (``mind_text``). Words stay
the producer's: this module holds the steps, the one closed vocabulary every notice shares, and none of the phrases.
A phrase the translation memory lacks is offered to the generator only when its row is sent, in that install's data
root; a read never writes and never wakes it.
"""

from __future__ import annotations

import logging
import pathlib
import threading
from typing import Any, Dict, Mapping, Optional, Tuple

log = logging.getLogger(__name__)

# The owner's next steps, one sentence each, said after the phrase. The translation generator fills them by code
# (``ui_translation`` catalog ``notice.step``); a step uses the displayed name of a control or destination. A row
# whose host has scheduled its own retry gets a ``retry_self`` step with the first producer that schedules one.
STEPS: Dict[str, str] = {
    "restart": "Restart Ouroboros to recover.",
    "send_after_restart": "Send it again after Restart.",
    "raise_total_budget": "Raise Total budget in Dashboard → Costs.",
    "choose_reviewer": "Run /review with a row this install can pay: any enabled row in Settings → Agents, or the Main model.",
    "evolution_start": "/evolve start resumes it.",
    "press_again": "You can press it again.",
    "try_again": "You can try it again.",
    "review_again": "Start a fresh review of this skill before you enable it or grant it access.",
    "reply_to_revert": "Reply if you want it reverted.",
    "reopen_project": "A message from you in this room or a new task reopens it.",
    "inspect_task": "Open the task's details before you start another run.",
    "restore_kept_changes": "Put your local changes back with the Git command that comes with this message.",
    "inspect_then_restore_kept_changes": ("The automatic restore was interrupted: check the current files first, then put "
                                          "your local changes back with the Git command that comes with this message."),
}
_STEP_CODE = "code:notice.step."
_MISSES: Dict[str, set] = {}
_MISSES_CAP = 2_000  # one process's memory of offered phrases per language; the generator's own queue is bounded too
_MISSES_LOCK = threading.Lock()


class _Slots(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def notice(code: str, *, step: str = "", details: Any = "", instruction: Any = "", template: str = "",
           values: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """A row's ``notice``: the code, the step when there is one, the raw details when there are any, the owner's
    instruction on one line, and a variable phrase's template with its values. An unknown step is a producer defect,
    refused here rather than drawn as words nobody wrote."""
    if step and step not in STEPS:
        raise ValueError(f"unknown host notice step {step!r}")
    made: Dict[str, Any] = {"code": str(code)}
    if step:
        made["step"] = step
    if raw := str(details or "").strip():
        made["details"] = raw
    if line := " ".join(str(instruction or "").split()):
        made["instruction"] = line
    if template:
        made.update(template=str(template), values={str(k): str(v) for k, v in (values or {}).items()})
    return made


def fields(code: str, **kwargs: Any) -> Dict[str, Any]:
    """``{"notice": notice(...)}``, merged into a row's ``progress_meta``."""
    return {"notice": notice(code, **kwargs)}


def phrase(code: str, template: str, *, step: str = "", details: Any = "", **values: Any) -> Tuple[str, Dict[str, Any]]:
    """A phrase with variable or authored words: ``(text, fields)``, the row's English text with the values in place
    and its notice carrying the template (translated) and the values (never translated)."""
    text = str(template).format_map(_Slots({k: str(v) for k, v in values.items()}))
    return text, fields(code, step=step, details=details, template=template, values=values)


def notice_meta(meta: Optional[Mapping[str, Any]], code: str, **kwargs: Any) -> Dict[str, Any]:
    """A row's ``progress_meta`` with the notice merged in."""
    return {**(meta or {}), **fields(code, **kwargs)}


def with_notice(kwargs: Mapping[str, Any], code: str, **notice_kwargs: Any) -> Dict[str, Any]:
    """A send's keyword arguments (a correlation's ``progress_meta`` among them) with the notice merged in."""
    return {**kwargs, "progress_meta": notice_meta(kwargs.get("progress_meta"), code, **notice_kwargs)}


def carried(kwargs: Mapping[str, Any], said: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """A send's keyword arguments with a ready notice (``said``, a refusal's or a phrase's fields) merged into their
    ``progress_meta``; unchanged when there is none."""
    if not (said and isinstance(said.get("notice"), Mapping)):
        return dict(kwargs)
    return {**kwargs, "progress_meta": {**(kwargs.get("progress_meta") or {}), "notice": dict(said["notice"])}}


def notice_of(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The row's typed notice, or ``{}`` for every row written without one."""
    notice = row.get("notice") if isinstance(row, Mapping) else None
    return dict(notice) if isinstance(notice, Mapping) and str(notice.get("code") or "") else {}


def stored(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The notice a row persists: its code, step, details, instruction and a variable phrase's template and values,
    never the line a reader was sent."""
    notice = notice_of(row)
    kept: Dict[str, Any] = {key: str(notice[key]) for key in ("code", "step", "details", "instruction", "template")
                            if notice.get(key)}
    if kept.get("template") and isinstance(notice.get("values"), Mapping):
        kept["values"] = {str(k): str(v) for k, v in notice["values"].items()}
    return kept


def mind_text(row: Mapping[str, Any]) -> str:
    """What the mind reads beside a notice row's phrase: its step and its details, in English (BIBLE P1: the raw text
    an older row carried in its words stays in view)."""
    notice = notice_of(row)
    step = STEPS.get(str(notice.get("step") or ""), "")
    return (f"\n[Step: {step}]" if step else "") + (f"\n[Details: {notice['details']}]" if notice.get("details") else "")


def notice_line(row: Mapping[str, Any], lang: Optional[str] = None, *, drive_root: Optional[pathlib.Path] = None,
                record_misses: bool = False) -> str:
    """Telegram's and Copy's one line of a host row in the install's language: its phrase, then its step, each part
    translated before they are joined, on one language: while the phrase has no translation the step stays English.
    The phrase is a host sentence, translated by its exact text for a notice row and for the technical pause notice
    (the one row Telegram translated so before notices), or by its template with the values put back as written;
    every other row keeps its text. A string the memory lacks stays English; ``record_misses`` (a send, never a read)
    offers it to the generator in ``drive_root``."""
    from ouroboros import i18n_memory as memory
    from ouroboros.ui_language import is_english, normalize_language_tag

    tag = normalize_language_tag(memory.current_language() if lang is None else lang) or ""
    english = not tag or is_english(tag)
    text = str(row.get("text") if row.get("text") is not None else row.get("content") or "").strip()
    notice = notice_of(row)
    pause = (row.get("system_type") == "task_pause_notice" and row.get("role") == "system"
             and row.get("pause_reason") == "authority")
    said = english or not text  # the phrase is in the install's language (or there is none)
    if (notice or pause) and not said:
        template = str(notice.get("template") or "")
        values = notice.get("values") if isinstance(notice.get("values"), Mapping) else {}
        key = template or text
        found = memory.tr(key, tag, drive_root=drive_root)
        if found is None:
            if record_misses:
                _miss(key, tag, {"source": key, "role": "host-text"}, drive_root)
        else:
            text = memory.fmt(template, dict(values), tag, drive_root=drive_root) if template else found
            said = True
    step = str(notice.get("step") or "")
    if step not in STEPS:
        return text
    words = STEPS[step] if english or not said else memory.tr(_STEP_CODE + step, tag, STEPS[step], drive_root=drive_root)
    return f"{text} {words}" if text else str(words)


def with_instruction(row: Mapping[str, Any], line: str) -> str:
    """The text a reader without Details (Telegram) sends for a host row: its one line, then the notice's typed
    ``instruction`` after one blank line, as written; the line alone for every other row (its details stay behind
    Details in the window)."""
    instruction = str(notice_of(row).get("instruction") or "").strip()
    return f"{line}\n\n{instruction}" if line and instruction else line


def view(row: Mapping[str, Any], *, drive_root: Optional[pathlib.Path] = None, record_misses: bool = False) -> Dict[str, Any]:
    """The notice a reader is sent with a row (its live frame, a history read): the stored fields plus ``line``, the
    assembled phrase and step in the install's language. ``{}`` for a row without a notice. Only the send passes
    ``record_misses``; a read (history, search, an operation) neither writes nor wakes the generator."""
    notice = notice_of(row)
    if not notice:
        return {}
    try:
        notice["line"] = notice_line(row, drive_root=drive_root, record_misses=record_misses)
    except Exception:  # a translation fault never holds the row: the English it was written in stays its line
        log.debug("host notice line not assembled; the stored words stand", exc_info=True)
        notice["line"] = notice_line(row, "")
    return {"notice": notice}


def step_catalog() -> Dict[str, str]:
    """The steps for the translation generator (``ui_translation._core_catalogs``)."""
    return dict(STEPS)


def _miss(key: str, tag: str, context: Dict[str, Any], drive_root: Optional[pathlib.Path]) -> None:
    """Offer a phrase the memory lacks, once per process and language while the generator's queue holds it; a refusal
    (a full queue, a failed write) clears the mark so a later send offers it again."""
    with _MISSES_LOCK:
        seen = _MISSES.setdefault(tag, set())
        if key in seen:
            return
        if len(seen) >= _MISSES_CAP:
            seen.clear()
        seen.add(key)
    accepted = False
    try:
        from ouroboros import i18n_memory as memory
        from ouroboros.config import DATA_DIR
        from ouroboros.ui_translation import on_language_event

        root = pathlib.Path(drive_root if drive_root is not None else DATA_DIR)
        accepted = bool(memory.record_missing(root, tag, [{"key": key, "context": context}])["accepted"])
        if accepted:
            on_language_event("missing", root, tag)
    except Exception:
        log.debug("host notice: translation miss not recorded", exc_info=True)
    if not accepted:
        with _MISSES_LOCK:
            _MISSES.get(tag, set()).discard(key)
