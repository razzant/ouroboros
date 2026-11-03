"""One-time owner notices about what an update leaves an install running (#1334, #1335).

An update may change a shipped default without changing what an existing
install actually runs: a settings document an earlier release wrote keeps the
finite round/lifetime bounds it ran under. That is not migrated silently or
behind the owner's back; it is stated ONCE, factually, in the owner's chat.
The same holds for the memory the old dialogue writer left: the chronicle
imports it unchanged, it keeps working in that format and is folded gradually,
and the owner hears once how much of it there is and that Ouroboros can be asked
to keep folding it, part by part. The per-call ceiling went the other way: the
owner retired it on every install, a configured value included, so each install
with a settings document hears once what bounds a command, an in-slot wait and the
silence before attention now. (Which reviewers run is no longer a notice
of this module: the review lanes became rows of the subagent catalog, and the
migration's own one-time report — ``server_maintenance._startup_review_pool_notice``
— says which rows run now.)

The facts are only what the document, the environment and the imported memory
show — a key absent from the document, an invalid value, a saved value, an
environment variable, the imported legacy sections. A saved value is never
presented as proof of a manual choice. Delivery follows the existing
retired-settings notice (``server_maintenance``): nothing is sent or marked
while no owner chat is bound, and ``state.json`` records each notice only after
it was handed to the owner-chat writer, so a failed write is retried at a later
boot rather than claimed as published.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Mapping, Optional

log = logging.getLogger(__name__)

OPTIONAL_BOUNDS_NOTICE_KEY = "optional_bounds_notified"
LEGACY_MEMORY_NOTICE_KEY = "legacy_memory_notified"
COMMAND_TIMEOUT_NOTICE_KEY = "command_timeout_notified"
_CALL_CEILING_KEY = "OUROBOROS_PER_CALL_TIMEOUT_CEILING_SEC"
# The old dialogue writer's retelling; its cursor file alone holds no retelling to tell about.
_LEGACY_MEMORY_FILES = ("dialogue_blocks.json", "dialogue_summary.md")
_NOT_PIECES = frozenset({"gap", "cursor_gap"})

_BOUND_LABELS = {
    "OUROBOROS_MAX_ROUNDS": ("Max Rounds per Task", "rounds"),
    "OUROBOROS_TASK_ABS_CEILING_SEC": ("Task Lifetime Limit", "seconds"),
}


def optional_bound_facts(document: Optional[Mapping[str, Any]], settings: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every FINITE optional task bound with the plain fact of where it comes from.

    ``document`` is the raw settings document as written (``None`` = none on
    disk) and ``settings`` the startup ``load_settings()`` view, which already
    applied the launch environment to keys the document leaves unset — the
    process environment itself is useless here, because startup projects the
    merged settings into it. Presence is independent of validity: even null or
    blank is ``saved_invalid``. An absent key equal to the document default is
    ``document_absent_key``; any other finite value can only come from the
    environment. Both runtime getters apply an accepted value as written, so the
    parsed startup view is the effective bound.
    """
    from ouroboros.settings_scales import OPTIONAL_BOUND_LEGACY, optional_bound_value, parse_positive_or_unlimited

    facts: List[Dict[str, Any]] = []
    for key, legacy in OPTIONAL_BOUND_LEGACY.items():
        present = isinstance(document, Mapping) and key in document
        raw = document[key] if present else settings.get(key, legacy if document is not None else "unlimited")
        value = optional_bound_value(key, settings.get(key, raw))
        if value is None:
            continue  # unlimited, including a fresh install's shipped default
        if present:
            try:
                parse_positive_or_unlimited(int(raw) if isinstance(raw, float) and raw.is_integer() else raw)
                origin = "saved"
            except (TypeError, ValueError):
                origin = "saved_invalid"
        elif document is not None and value == legacy:
            origin = "document_absent_key"
        else:
            origin = "env"
        facts.append({"key": key, "value": value, "origin": origin, "raw": raw})
    return facts


def optional_bounds_notice(facts: List[Dict[str, Any]]) -> str:
    """The owner-facing sentence for ``optional_bound_facts`` ('' when nothing is finite)."""
    if not facts:
        return ""
    why = {
        "saved": "saved in your settings",
        "env": "set by an environment variable",
        "document_absent_key": "the key is absent from your settings file; this is its effective value",
    }
    parts = []
    for fact in facts:
        label, unit = _BOUND_LABELS.get(fact["key"], (fact["key"], ""))
        origin = str(fact["origin"])
        reason = (f"the value {fact['raw']!r} is not a positive number or 'unlimited', so the finite fallback applies"
                  if origin.endswith("_invalid") else why.get(origin, origin))
        parts.append(f"{label} = {fact['value']} {unit} ({reason})".replace("  ", " "))
    return ("⚙️ Task limits on this install: " + "; ".join(parts) + ". New installs ship without these "
            "limits (unlimited); nothing was changed here. Change them in Settings → Advanced → Runtime Limits.")


def command_timeout_notice(document: Optional[Mapping[str, Any]], settings: Mapping[str, Any]) -> str:
    """The owner-facing sentence that the per-call ceiling is retired (3A; 30 Oct: await B, silence A).

    Owed once to every install with a settings document, whatever the value or its origin: the
    owner retired the ceiling everywhere, a manually saved value included, and an install that
    ran under the implicit default need not have saved the key, so the document is the only fact.
    A fresh install hears it once its first document exists, so the words claim no ceiling of
    this install: they say what earlier versions did and name a value only where the document
    carries one (the read seam no longer honors the key, ``RETIRED_SETTING_KEYS``). An install
    without a settings document hears nothing.
    """
    if document is None:
        return ""
    value, idle = document.get(_CALL_CEILING_KEY), settings.get("OUROBOROS_TASK_IDLE_TIMEOUT_SEC")
    shown = f"; {value} s in this install's settings" if str(value if value is not None else "").strip() else ""
    idle_shown = f" ({idle} s here)" if str(idle if idle is not None else "").strip() else ""
    return (f"⏱️ The separate per-call ceiling of earlier versions (1800 s by default{shown}) is retired. run_command, "
            "run_script and verify_and_record run for the timeout a call asks for, else Tool Timeout, narrowed only "
            "by the task's own deadline or execution budget; a task whose deadline or budget is spent, or a timeout "
            "this platform cannot wait on, starts no command. An in-slot await_messages wait lasts as long as it asks, "
            f"within the deadline and Stop. A task is flagged idle after Idle Timeout{idle_shown} of silence, no longer "
            "after at least that ceiling plus two minutes.")


def legacy_memory_facts(root: Any) -> Optional[Dict[str, Any]]:
    """How much memory the old dialogue writer left, as the chronicle imported it; ``None`` = unknown.

    Without a chronicle journal and without the old writer's block or summary file
    there is no old memory: nothing is read or created. Otherwise the journal is
    activated by the same import the first memory view runs; while that import is
    pending (another importer holds the legacy lock) or refused, the facts stay
    unknown, so the notice stays owed for a later boot. A piece is one imported
    ``legacy`` section (gaps are not pieces), a period is one old block, and the
    span is the sections' chat-row time bounds, else the block labels the old
    writer wrote (in period order). A model is never asked.
    """
    from ouroboros.chronicle_store import ChronicleStore

    root = pathlib.Path(root)
    store = ChronicleStore(root)
    if not store.log_path.exists() and not any((root / "memory" / name).exists() for name in _LEGACY_MEMORY_FILES):
        return None
    if store.ensure_activated().get("kind") != "activation":
        return None
    pieces = [pointer for pointer in store.legacy_pointer_rows()
              if pointer.get("kind") == "legacy" and pointer.get("legacy_type") not in _NOT_PIECES]
    spans = [((pointer.get("covers") or {}).get("raw_range") or {}).get("ts_span") or {} for pointer in pieces]
    starts = [span["start"][:10] for span in spans if isinstance(span.get("start"), str)]
    ends = [span["end"][:10] for span in spans if isinstance(span.get("end"), str)]
    labels = [str(pointer["range_text"]).strip() for pointer in pieces
              if isinstance(pointer.get("range_text"), str) and pointer["range_text"].strip()]
    return {"pieces": len(pieces),
            "periods": len({pointer["legacy_block"] for pointer in pieces if type(pointer.get("legacy_block")) is int}),
            "start": min(starts) if starts else None, "end": max(ends) if ends else None,
            "labels": [labels[0], labels[-1]] if labels else []}


def legacy_memory_notice(facts: Optional[Mapping[str, Any]]) -> str:
    """The owner-facing sentence for ``legacy_memory_facts`` ('' when there is no old memory to tell about)."""
    if not facts or not facts.get("pieces"):
        return ""

    def counted(number: int, word: str) -> str:
        return f"{number} {word}{'' if number == 1 else 's'}"

    amount = counted(int(facts["pieces"]), "piece")
    if facts.get("periods"):
        amount += " over " + counted(int(facts["periods"]), "period")
    start, end, labels = facts.get("start"), facts.get("end"), list(facts.get("labels") or [])
    if start and end:
        amount += f" ({start} to {end})" if start != end else f" ({start})"
    elif labels:
        amount += f" (labelled {labels[0]} … {labels[-1]})" if labels[0] != labels[-1] else f" (labelled {labels[0]})"
    # Facts only: one request folds a part, not the whole (on the acceptance stand 23 of 377
    # retellings), and the host knows neither the time nor the price of the rest; the mind
    # sees its story status and can say how much is left.
    return ("🧠 Memory: what Ouroboros remembered before this update is kept in its previous format — "
            f"{amount}. It works as it is and is folded into the new format gradually, part by part. "
            "You can ask Ouroboros to keep folding it; it will tell you how much is left.")


def _raw_settings_document() -> Optional[Dict[str, Any]]:
    """The settings document exactly as written (no defaults, no coercion), ``None`` if absent."""
    from ouroboros import config
    from ouroboros.settings_integrity import read_settings_json_verified

    if not config.SETTINGS_PATH.exists():
        return None
    raw = read_settings_json_verified(config.SETTINGS_PATH)
    return dict(raw) if isinstance(raw, dict) else {}


def startup_upgrade_notices(settings: Mapping[str, Any]) -> None:
    """Send each still-owed one-time notice to the bound owner chat (never raises)."""
    try:
        from ouroboros.utils import utc_now_iso
        from supervisor import message_bus
        from ouroboros.notice_receipts import recorded
        from supervisor.state import control_value, load_state, update_state

        state = load_state()
        known, owner_chat = control_value(state, "owner_chat_id")
        if not known or not owner_chat:
            return  # display-only backup values never authorize delivery
        owner_chat = int(owner_chat)
        owed = []
        if not state.get(OPTIONAL_BOUNDS_NOTICE_KEY):
            text = optional_bounds_notice(optional_bound_facts(_raw_settings_document(), settings or {}))
            if text:
                owed.append((OPTIONAL_BOUNDS_NOTICE_KEY, text, "optional_bounds_notice"))
        if not state.get(COMMAND_TIMEOUT_NOTICE_KEY):
            text = command_timeout_notice(_raw_settings_document(), settings or {})
            if text:
                owed.append((COMMAND_TIMEOUT_NOTICE_KEY, text, "command_timeout_notice"))
        if not state.get(LEGACY_MEMORY_NOTICE_KEY) and message_bus.DATA_DIR:
            try:
                text = legacy_memory_notice(legacy_memory_facts(message_bus.DATA_DIR))
            except Exception:  # an unreadable journal leaves this notice owed, never the others
                log.debug("legacy memory facts unavailable", exc_info=True)
                text = ""
            if text:
                owed.append((LEGACY_MEMORY_NOTICE_KEY, text, "legacy_memory_notice"))
        for key, text, system_type in owed:
            # require_write: a chat row that could not be written raises, so the
            # notice stays owed instead of being marked as published.
            if not recorded(message_bus.DATA_DIR, owner_chat, system_type):
                message_bus.send_with_budget(owner_chat, text, role="system", system_type=system_type,
                                             require_write=True, ensure_record_boundary=True)

            def _mark(st: dict, marker: str = key) -> None:
                st[marker] = utc_now_iso()

            update_state(_mark)
    except Exception:
        log.debug("one-time upgrade notice failed", exc_info=True)
