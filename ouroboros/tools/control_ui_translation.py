"""The owner's wording for the interface, found and pinned from conversation.

When the owner asks, in their language, for a different wording of a label ("call it
Projects, not Sections"), ``search_ui_translations`` lists every entry the words could
mean — the exact key, the English, the wording on screen, whose wording it is and where
it appears — and ``pin_ui_translation`` writes the chosen key as the ``owner`` layer of
the translation memory through its one locked, revision-counted writer
(``i18n_memory.update_memory``), or removes that pin. Which entry the owner meant is
the mind's call; the search never picks one, and a key that neither the memory, its
miss queue nor the host catalog knows is refused, because a pin under a guessed key
is written where nothing renders. Precedence stays the memory's: a pin hides an
imported entry (which returns on unpin) and replaces a generated one (an unpinned key
shows English until it is translated again, not necessarily as before).

Both tools use the install's one memory per language at the canonical data root
(``tool_access.canonical_data_root``: the task's ``budget_drive_root``, else its own
drive). A root whose memory mode forks or empties its execution drive runs on
``state/headless_tasks/<id>/data`` with that root as ``budget_drive_root``
(``agent.py`` builds its ``ToolContext`` so); a dictionary file on such a drive is
neither read nor written, and the repository a task edits never selects the memory.

After a write, the existing ``ui_i18n_updated`` frame tells open windows to re-read the
dictionary. Only the server process holds them, so a turn hands the frame to its host
the way ``update_focus`` hands over its focus: on its event queue (a pooled worker's
and a direct turn's alike), where ``supervisor/events_worker_reports.py`` broadcasts it.
A caller without a queue broadcasts to a running server loop in its own process, or to
no one. The receipt's ``window_refresh`` names that handoff (``queued_for_host``,
``handed_to_server_loop`` or ``not_sent``), never a delivery: no window acknowledges a
frame. The write stands either way.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from ouroboros import i18n_memory as memory
from ouroboros.tool_access import canonical_data_root
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, completed_local_read, publish_no_effect
from ouroboros.ui_language import is_english, normalize_language_tag
from ouroboros.utils import read_json_dict

log = logging.getLogger(__name__)

_DEFAULT_MATCHES = 20
_MAX_MATCHES = 50
_SLOT_TAG_RE = re.compile(r"</?\d+>")


class _Refused(Exception):
    """A refusal decided under the memory's lock: the writer persists nothing."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _refusal(ctx: ToolContext, status: str, code: str, reason_code: str, message: str) -> str:
    return publish_no_effect(ctx, ToolResult(status=status, code=code, text=f"⚠️ {code}: {message}",
                                             meta={"reason_code": reason_code}))


def _reply(ctx: ToolContext, body: Dict[str, Any], *, effect: bool) -> str:
    result = ToolResult(status="ok", code="OK", text=json.dumps(body, ensure_ascii=False))
    return _publish_tool_result(ctx, result) if effect else publish_no_effect(ctx, result)


def _language(requested: Any) -> Tuple[str, Tuple[str, ...]]:
    """``(tag, refusal)``: the named memory or the interface language's, else the refusal's
    ``(status, code, reason_code, message)``."""
    raw = str(requested or "").strip()
    tag = normalize_language_tag(raw) if raw else memory.current_language()
    if tag is None:
        return "", ("error", "TOOL_ARG_ERROR", "language_not_a_tag",
                    f"language={raw!r} is not a language tag such as ru or pt-BR.")
    if is_english(tag):
        return "", ("unavailable", "CAPABILITY_UNAVAILABLE", "english_interface",
                    "the interface shows its English source, so there is no translation to find or pin; "
                    "an English label is source copy and changes in the code.")
    return tag, ()


def _memory_key(key: Any, scope: Any) -> Tuple[str, str]:
    """``(memory key, problem)``: a code verbatim, rendered text as the reader sees it, a scope after the separator."""
    text, inline_scope = memory.split_scope(key if isinstance(key, str) else "")
    place = (scope if isinstance(scope, str) else "").strip() or inline_scope.strip()
    if memory.is_code_key(text.strip()):
        return ("", "a code: key has no scope.") if place else (text.strip(), "")
    base = memory.key_text(text)
    if not base:
        return "", "key is empty: pass the exact key search_ui_translations showed."
    return (f"{base}{memory.SCOPE_SEPARATOR}{place}" if place else base), ""


def _wording(entry: Optional[Dict[str, Any]]) -> Any:
    if not entry:
        return None
    return entry["text"] if "text" in entry else entry.get("forms")


def _layer(entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not entry:
        return {"provenance": "untranslated"}
    return {"provenance": entry.get("provenance", "generated"), "wording": _wording(entry)}


def _catalog() -> Dict[str, Dict[str, Any]]:
    """The host's code tables (English and where it shows). Optional reading: a table that
    fails to load leaves its codes undescribed, never the search unanswered."""
    try:
        from ouroboros.ui_translation import catalog_entries

        return catalog_entries()
    except Exception:
        log.debug("ui translation catalog unavailable", exc_info=True)
        return {}


def _queued(drive_root: Any, tag: str) -> Dict[str, Dict[str, Any]]:
    """Strings a window or a host seam reported untranslated, waiting for the generator."""
    items = (read_json_dict(memory.pending_path(drive_root, tag)) or {}).get("items")
    return {key: facts for key, facts in items.items() if isinstance(facts, dict)} if isinstance(items, dict) else {}


def _english(key: str, catalog: Dict[str, Any], *sources: Optional[Dict[str, Any]]) -> Optional[str]:
    """The English a key stands for: the text itself, the catalog's current English for a code,
    else the English an entry or a queued miss recorded."""
    if not memory.is_code_key(key):
        return memory.split_scope(key)[0]
    if key in catalog:
        return catalog[key]["text"]
    for source in sources:
        recorded = (source or {}).get("source")
        if isinstance(recorded, str) and recorded:
            return recorded
    return None


def _fold(text: Any) -> str:
    return memory.key_text(_SLOT_TAG_RE.sub("", str(text or ""))).casefold()


def _texts(wording: Any) -> list:
    return list(wording.values()) if isinstance(wording, dict) else [wording] if wording else []


def _rows(doc: Dict[str, Any], catalog: Dict[str, Any], queued: Dict[str, Any]) -> list:
    """Every key the install can show, entry or not, with the facts a choice needs."""
    refused = doc.get("refused") or {}
    keys = dict.fromkeys([*doc["entries"], *refused, *queued, *catalog])
    rows = []
    for key in keys:
        entry, facts = doc["entries"].get(key), queued.get(key) or {}
        context = facts.get("context") if isinstance(facts.get("context"), dict) else {}
        text, scope = memory.split_scope(key)
        row: Dict[str, Any] = {"key": text, **({"scope": scope} if scope else {})}
        english = _english(key, catalog, entry, doc["shadow"].get(key), context)
        if english is not None and english != text:
            row["english"] = english
        row.update(_layer(entry))
        if entry and entry.get("provenance") == "owner":
            row["under_pin"] = _layer(doc["shadow"].get(key)) if key in doc["shadow"] else None
        if key in catalog:
            row["place"] = catalog[key]["context"].get("note", "")
        elif context:
            row["place"] = {k: v for k, v in context.items() if k != "source"}
        if not entry:
            row["note"] = ("the generator gave up on it" if key in refused else
                           "queued for the generator" if key in queued else "not translated yet")
        haystack = [english or text, *_texts(row.get("wording")),
                    *_texts((row.get("under_pin") or {}).get("wording")), key if memory.is_code_key(key) else ""]
        rows.append((row, [_fold(item) for item in haystack if item]))
    return rows


@completed_local_read
def _search_ui_translations(ctx: ToolContext, query: str = "", language: str = "",
                            limit: int = _DEFAULT_MATCHES) -> str:
    needle = _fold(query)
    if not needle:
        return _refusal(ctx, "error", "TOOL_ARG_ERROR", "query_empty",
                        "query is empty: give words the owner sees or the English of the label.")
    tag, refusal = _language(language)
    if not tag:
        return _refusal(ctx, *refusal)
    root = canonical_data_root(ctx)
    try:
        doc = memory.load_memory(root, tag)
    except memory.MemoryFormatError as exc:
        return _refusal(ctx, "unavailable", "CAPABILITY_UNAVAILABLE", "memory_invalid",
                        f"the {tag} translation memory is unreadable and is left as it is: {exc}")
    if doc is None:
        return _refusal(ctx, "unavailable", "CAPABILITY_UNAVAILABLE", "no_memory",
                        f"no translation memory exists for {tag}.")
    try:
        bounded = min(max(int(limit), 1), _MAX_MATCHES)
    except (TypeError, ValueError):
        bounded = _DEFAULT_MATCHES
    # Whole-text matches first (the owner quoted the label), then the rest by key.
    hits = sorted(((needle not in texts, row["key"], row) for row, texts in _rows(doc, _catalog(), _queued(root, tag))
                   if any(needle in text for text in texts)), key=lambda hit: hit[:2])
    return _reply(ctx, {
        "language": tag, "label": doc["profile"].get("label") or tag, "revision": doc["revision"],
        "plural_categories": doc.get("plural_categories"), "query": query, "total": len(hits),
        "limit": bounded, "matches": [row for _rank, _key, row in hits[:bounded]],
    }, effect=False)


def _value(text: Any, forms: Any) -> Tuple[Dict[str, Any], str]:
    if (text is None) == (forms is None):
        return {}, "pass exactly one of text or forms."
    if text is not None:
        if not isinstance(text, str) or not text.strip():
            return {}, f"text={text!r}: give the wording as a nonblank string."
        return {"text": text}, ""
    if not isinstance(forms, dict) or not forms or not all(
            isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in forms.items()):
        return {}, f"forms={forms!r}: give an object of plural category to nonblank wording."
    return {"forms": dict(forms)}, ""


def _pin_ui_translation(ctx: ToolContext, action: str = "", key: str = "", scope: str = "",
                        text: Any = None, forms: Any = None, language: str = "") -> str:
    operation = str(action or "").strip().lower()
    if operation not in ("pin", "unpin"):
        return _refusal(ctx, "error", "TOOL_ARG_ERROR", "action_invalid", f"action={action!r}: use 'pin' or 'unpin'.")
    tag, refusal = _language(language)
    if not tag:
        return _refusal(ctx, *refusal)
    target, problem = _memory_key(key, scope)
    if problem:
        return _refusal(ctx, "error", "TOOL_ARG_ERROR", "key_invalid", problem)
    value: Dict[str, Any] = {}
    if operation == "pin":
        value, problem = _value(text, forms)
        if problem:
            return _refusal(ctx, "error", "TOOL_ARG_ERROR", "wording_invalid", problem)
    elif text is not None or forms is not None:
        return _refusal(ctx, "error", "TOOL_ARG_ERROR", "wording_with_unpin",
                        "unpin takes no text or forms: it removes the owner's wording.")
    root = canonical_data_root(ctx)
    catalog, queued = _catalog(), _queued(root, tag)
    facts: Dict[str, Any] = {"changed": False}

    def _mutate(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        current = doc["entries"].get(target)
        facts["previous"] = _layer(current)
        if operation == "unpin":
            if not memory.remove_owner_entry(doc, target):
                return None
        else:
            if current is None and target not in catalog and target not in queued and target not in (doc.get("refused") or {}):
                raise _Refused("key_unknown", (
                    f"no interface string has the key {target.split(memory.SCOPE_SEPARATOR)[0]!r}"
                    + (" with that scope" if memory.SCOPE_SEPARATOR in target else "")
                    + f" in the {tag} memory, its miss queue or the host catalog; search_ui_translations "
                    "shows the exact keys. A string no window has shown since the language was chosen "
                    "becomes known once it is shown."))
            entry = dict(value)
            source = _english(target, catalog, current, doc["shadow"].get(target),
                              (queued.get(target) or {}).get("context"))
            if memory.is_code_key(target) and source is not None:
                entry["source"] = source  # the writer judges placeholders and markup against it
                if target in catalog:
                    entry["source_hash"] = memory.source_hash(source)
            categories = doc.get("plural_categories") or []
            unknown = sorted(set(entry.get("forms") or {}) - set(categories)) if categories else []
            if unknown:
                raise _Refused("plural_category_unknown",
                               f"forms {unknown} are not plural categories of {tag}; use {categories}.")
            if current and current.get("provenance") == "owner" and _wording(current) == _wording(entry):
                return None
            try:
                memory.set_owner_entry(doc, target, entry)
            except memory.MemoryFormatError as exc:
                raise _Refused("wording_invalid", str(exc)) from exc
            if "text" in entry and "{n}" in (source or target):
                facts["note"] = (f"the English has {{n}}: this one text shows for every count; forms "
                                 f"({', '.join(categories) or 'one per plural category'}) vary it by number.")
        facts.update(changed=True, now=_layer(doc["entries"].get(target)))
        return doc

    try:
        written = memory.update_memory(root, tag, _mutate)
    except _Refused as exc:
        return _refusal(ctx, "error", "TOOL_ARG_ERROR", exc.reason_code, str(exc))
    except FileNotFoundError:
        return _refusal(ctx, "unavailable", "CAPABILITY_UNAVAILABLE", "no_memory",
                        f"no translation memory exists for {tag}; nothing was written.")
    except memory.MemoryFormatError as exc:
        return _refusal(ctx, "unavailable", "CAPABILITY_UNAVAILABLE", "memory_invalid",
                        f"the {tag} translation memory is unreadable and was not overwritten: {exc}")
    except TimeoutError as exc:
        return _refusal(ctx, "unavailable", "CAPABILITY_UNAVAILABLE", "memory_busy",
                        f"{exc}; nothing was written. Another writer held the memory; try again.")
    text_key, place = memory.split_scope(target)
    receipt: Dict[str, Any] = {"action": operation, "language": tag, "key": text_key,
                               **({"scope": place} if place else {}), "revision": written["revision"], **facts}
    if not facts["changed"]:
        receipt["reason"] = "already_pinned" if operation == "pin" else "not_pinned"
        return _reply(ctx, receipt, effect=False)
    if operation == "unpin" and receipt["now"]["provenance"] == "untranslated":
        receipt["note"] = (
            "English shows. The generator previously refused this key; use Regenerate before it can retry."
            if target in (written.get("refused") or {}) else
            "English shows until the generator translates it again after a window shows it; "
            "that wording need not match any earlier one.")
    shown = tag == memory.current_language()
    receipt["window_refresh"] = _hand_on_refresh(ctx, tag, written["revision"]) if shown else "not_sent"
    if receipt["window_refresh"] == "not_sent":
        receipt["window_refresh_note"] = (
            "no refresh signal was handed on: this call reached neither its host's event queue nor a running "
            "server loop. Open windows show the saved wording at their next dictionary read: a reconnect or a reload."
            if shown else f"{tag} is not the interface language now, so no window shows it.")
    return _reply(ctx, receipt, effect=True)


def _hand_on_refresh(ctx: ToolContext, tag: str, revision: int) -> str:
    """Hand ``ui_i18n_updated`` on for open windows and name the handoff: ``queued_for_host`` (the
    turn's event queue; the supervisor broadcasts it as it drains), ``handed_to_server_loop`` (no
    queue, a running server loop here took it) or ``not_sent``. None confirms a window re-read."""
    event_queue = getattr(ctx, "event_queue", None)
    try:
        if event_queue is not None:
            event_queue.put_nowait({"type": "ui_i18n_updated", "language": tag, "revision": revision})
            return "queued_for_host"
        from ouroboros.gateway.ws import broadcast_ws_sync

        if broadcast_ws_sync({"type": "ui_i18n_updated", "language": tag, "revision": revision}):
            return "handed_to_server_loop"
    except Exception:
        log.debug("ui_i18n_updated not handed on", exc_info=True)
    return "not_sent"


def _language_param() -> Dict[str, str]:
    return {"type": "string", "description": "BCP-47 tag of another language's memory; omit for the interface language."}


def ui_translation_tool_entries() -> list:
    return [
        ToolEntry("search_ui_translations", {
            "name": "search_ui_translations",
            "description": (
                "Find interface strings in the install's translation memory by the words the owner sees or by "
                "their English. Each match gives the exact key (and scope), the English, the current wording, "
                "whose wording it is (generated, imported, owner, untranslated), where it shows when known, and "
                "for an owner pin what unpinning restores. Read-only; use it before pin_ui_translation."
            ),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Words to look for, case-insensitive, in the English and every wording."},
                "language": _language_param(),
                "limit": {"type": "integer", "default": _DEFAULT_MATCHES, "minimum": 1, "maximum": _MAX_MATCHES},
            }, "required": ["query"], "additionalProperties": False},
        }, _search_ui_translations, safety_policy="skip"),
        ToolEntry("pin_ui_translation", {
            "name": "pin_ui_translation",
            "description": (
                "Pin the owner's wording for one interface string, or unpin it. The pin is saved in the "
                "install's translation memory, outranks imported and generated wording and survives Regenerate. "
                "The receipt's window_refresh names how a refresh signal was handed on (queued_for_host or "
                "handed_to_server_loop: a handoff, not proof a window re-read) or not_sent (windows show the "
                "pin at their next dictionary read). Use the exact key (and scope) search_ui_translations showed; if "
                "the owner's words fit several entries and nothing settles which they mean, ask, otherwise act "
                "without a further confirmation. The memory's writer refuses placeholders or markup the English "
                "lacks. Unpin restores a hidden imported wording, otherwise English shows until the string is "
                "translated again."
            ),
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["pin", "unpin"]},
                "key": {"type": "string", "description": "Exact key from search_ui_translations: the English text or a code: key."},
                "scope": {"type": "string", "description": "The match's scope, only when the search showed one."},
                "text": {"type": "string", "description": "For pin: the wording, keeping the English's placeholders such as {n} or {name}."},
                "forms": {"type": "object", "additionalProperties": {"type": "string"},
                          "description": "For pin, instead of text, when the English has {n}: wording per plural category the search listed."},
                "language": _language_param(),
            }, "required": ["action", "key"], "additionalProperties": False},
        }, _pin_ui_translation, safety_policy="skip"),
    ]
