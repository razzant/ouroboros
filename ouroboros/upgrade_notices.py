"""One-time owner notices about what an update leaves an install running (#1334, #1335).

An update may change a shipped default without changing what an existing
install actually runs: a settings document an earlier release wrote keeps the
finite round/lifetime bounds it ran under, and an install that never saved a
reviewer panel follows whatever panel ships. Neither is migrated silently or
behind the owner's back; each is stated ONCE, factually, in the owner's chat.
The same holds for the memory the old dialogue writer left: the chronicle
imports it unchanged, it keeps working in that format and is folded gradually,
and the owner hears once how much of it there is and how to fold it at once.

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

REVIEWER_DEFAULT_NOTICE_KEY = "reviewer_default_delivery_notified"
OPTIONAL_BOUNDS_NOTICE_KEY = "optional_bounds_notified"
LEGACY_MEMORY_NOTICE_KEY = "legacy_memory_notified"
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
    environment. Runtime getters apply their floors to an isolated projection
    of this startup view, without consulting or changing the process environment.
    """
    from ouroboros.settings_scales import OPTIONAL_BOUND_LEGACY, optional_bound_value, parse_positive_or_unlimited
    from ouroboros.runtime_limits import get_max_rounds, get_task_abs_ceiling_sec
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot

    facts: List[Dict[str, Any]] = []
    getters = {"OUROBOROS_MAX_ROUNDS": get_max_rounds, "OUROBOROS_TASK_ABS_CEILING_SEC": get_task_abs_ceiling_sec}
    for key, legacy in OPTIONAL_BOUND_LEGACY.items():
        present = isinstance(document, Mapping) and key in document
        raw = document[key] if present else settings.get(key, legacy if document is not None else "unlimited")
        value = optional_bound_value(key, settings.get(key, raw))
        if value is None:
            continue  # unlimited, including a fresh install's shipped default
        with task_settings_scope(task_settings_snapshot({}, {key: str(value)})):
            value = getters[key]()
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
    from ouroboros.settings_scales import optional_bound_value

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
        if optional_bound_value(fact["key"], fact["raw"]) != fact["value"]:
            reason += f" as {fact['raw']!r}; the runtime minimum applies"
        parts.append(f"{label} = {fact['value']} {unit} ({reason})".replace("  ", " "))
    return ("⚙️ Task limits on this install: " + "; ".join(parts) + ". New installs ship without these "
            "limits (unlimited); nothing was changed here. Change them in Settings → Advanced → Runtime Limits.")


REVIEWER_DEFAULT_NOTICE = (
    "⚙️ Reviewers: this install has no saved reviewer panel, so it runs the shipped default — "
    "three reviewers that read the work themselves with read-only tools on the same models "
    "(several model calls per review instead of one packet send), plus the scope reviewer for commits. "
    "A saved panel is never changed. See or change it in Settings → Agents; a model without tool "
    "calling can be switched to Packet there."
)


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
    return ("🧠 Memory: what Ouroboros remembered before this update is kept in its previous format — "
            f"{amount}. It works as it is and is folded into the new format gradually. "
            "To fold it all now, ask Ouroboros to fold the old memory.")


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
        from ouroboros.reviewer_slot_config import authored_reviewer_slots_state
        from ouroboros.utils import utc_now_iso
        from supervisor import message_bus
        from ouroboros.utils import iter_jsonl_chain_objects
        from supervisor.state import control_value, load_state, update_state

        state = load_state()
        known, owner_chat = control_value(state, "owner_chat_id")
        if not known or not owner_chat:
            return  # display-only backup values never authorize delivery
        owner_chat = int(owner_chat)
        owed = []
        if not state.get(REVIEWER_DEFAULT_NOTICE_KEY) and authored_reviewer_slots_state(
                str((settings or {}).get("OUROBOROS_REVIEWER_SLOTS") or ""))[0] == "absent":
            owed.append((REVIEWER_DEFAULT_NOTICE_KEY, REVIEWER_DEFAULT_NOTICE, "reviewer_default_notice"))
        if not state.get(OPTIONAL_BOUNDS_NOTICE_KEY):
            text = optional_bounds_notice(optional_bound_facts(_raw_settings_document(), settings or {}))
            if text:
                owed.append((OPTIONAL_BOUNDS_NOTICE_KEY, text, "optional_bounds_notice"))
        if not state.get(LEGACY_MEMORY_NOTICE_KEY) and message_bus.DATA_DIR:
            try:
                text = legacy_memory_notice(legacy_memory_facts(message_bus.DATA_DIR))
            except Exception:  # an unreadable journal leaves this notice owed, never the others
                log.debug("legacy memory facts unavailable", exc_info=True)
                text = ""
            if text:
                owed.append((LEGACY_MEMORY_NOTICE_KEY, text, "legacy_memory_notice"))
        # Recover the gap between the durable owner-chat write and the state
        # marker. The chat row itself is the receipt; no second notice ledger.
        recorded = set()
        if owed and message_bus.DATA_DIR:
            for row in iter_jsonl_chain_objects(message_bus.DATA_DIR / "logs" / "chat.jsonl"):
                if row.get("direction") == "system" and row.get("chat_id") == owner_chat:
                    recorded.add(row.get("type"))
        for key, text, system_type in owed:
            # require_write: a chat row that could not be written raises, so the
            # notice stays owed instead of being marked as published.
            if system_type not in recorded:
                message_bus.send_with_budget(owner_chat, text, role="system", system_type=system_type,
                                             require_write=True, ensure_record_boundary=True)

            def _mark(st: dict, marker: str = key) -> None:
                st[marker] = utc_now_iso()

            update_state(_mark)
    except Exception:
        log.debug("one-time upgrade notice failed", exc_info=True)
