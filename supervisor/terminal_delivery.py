"""Durable terminal-answer delivery seam (Poltergeist phase A2, owner 5=A).

The incident: a cancelled root's finished final report was salvaged to disk but
never reached chat, because final-answer delivery existed only on the natural
finalization path. This module is the ONE shared seam for terminal delivery:

- ``register_delivery`` is the DURABLE logical dedupe every terminal delivery
  goes through (``state/terminal_deliveries.json``, bounded, atomic). The
  supervisor's ``send_message`` handler consults it for every ``delivery_id``-
  bearing event, so the natural completion path (worker-buffered final answer),
  the cancel path, and the reap path share one at-most-once-per-restart-free
  registry instead of a process-local deque that forgets on restart. External
  transports remain at-least-once; the residual is disclosed here rather than
  papered over: a delivery can still double across a crash BETWEEN send and
  registration ("never lost" outranks "never doubled").
- ``deliver_unreviewed_salvage`` builds and enqueues the one chat message for a
  cancelled / non-retry-reaped task: a loud UNREVIEWED banner, an honest bounded
  preview (exact omitted count), a durable full-copy receipt (path + size +
  sha256), and — for a subtree cascade — a compact children digest. Routed by
  task lineage chat, never blindly to ``owner_chat_id``. Zero paid rounds.

Outcome routing stays with the callers: ``completed`` keeps the existing
review-aware final delivery (``task_finalization.deliver_final_message_live``,
same ``delivery_id`` vocabulary, now deduped durably by this registry);
``cancelled``/non-retry reap deliver here; a retryable reap delivers nothing.
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.utils import update_json_locked, utc_now_iso
from ouroboros.task_finalization import (
    HOST_AUTHORED_TERMINAL_ORIGINS,
    TERMINAL_ORIGIN_HOST_SALVAGE,
    TERMINAL_ORIGIN_MODEL_FINAL,
)

log = logging.getLogger(__name__)

_SCHEMA_VERSION = 2
_REGISTRY_CAP = 512
# Bounded chat preview of the salvaged answer; the full copy is referenced by a
# durable receipt, and the omitted tail is DISCLOSED (BIBLE P1), never silent.
SALVAGE_PREVIEW_CHARS = 3500
# Durable pending OUTBOX bound. A terminal answer is registered here BEFORE it is
# enqueued and removed once the send succeeded, so a crash in that window replays
# instead of losing the owner's answer forever (the incident class). Bounded the
# same way the delivered registry is: newest kept.
_PENDING_CAP = 64
# A row younger than this is presumed still in flight through the event queue,
# not lost; and each row is replayed at most this many times before the outbox
# gives up loudly (an unreachable chat must not become a tick-rate retry storm).
_REPLAY_MIN_AGE_SEC = 60.0
_PENDING_MAX_REPLAYS = 5

# GR3-7: marker prefix for "the teardown audit itself failed" rows riding the
# delegated_runs_unreconciled surface — run state is UNKNOWN, never clean.
RUN_STATE_UNKNOWN_PREFIX = "delegated_run_state_unknown"

_HOST_SALVAGE_RECEIPT = (
    "⚠️ A model-provider outage stopped this task before Ouroboros produced "
    "a complete answer. The full intermediate output and technical details are preserved in the "
    "task details."
)


def cleanup_settled_owner_mailbox(
    drive_root: Any, task_id: str, task: Optional[Dict[str, Any]] = None,
) -> None:
    """Remove attempt mail only after the canonical task result is settled."""

    from ouroboros.owner_mailbox import cleanup_task_mailbox
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES
    from supervisor.queue import _task_drive_for_task

    durable = load_task_result(pathlib.Path(drive_root), str(task_id)) or {}
    if str(durable.get("status") or "") in SETTLED_STATUSES:
        cleanup_task_mailbox(
            _task_drive_for_task(task or durable, str(task_id)), str(task_id),
        )


def _registry_path(drive_root: Any) -> pathlib.Path:
    root = pathlib.Path(drive_root) / "state"
    root.mkdir(parents=True, exist_ok=True)
    return root / "terminal_deliveries.json"


def delivery_id_for(task_id: str, text: str) -> str:
    """The shared delivery identity: ``final:<tid>:<sha256[:16]>`` — the same
    vocabulary ``deliver_final_message_live`` mints for the natural path.

    GR7-4: callers digest the STABLE part of the message (task id + settled
    status framing + core answer), never the mutable unreconciled-runs
    disclosure note — the note rides the TEXT only. A watchdog replay rebuilds
    the note from the CURRENT audit (the list shrinks as runs reconcile); a
    note-bearing digest minted a fresh id per replay, and each became a second
    owed message for the same terminal answer (mirror of the cascade's GR4-2
    intent-derived identity: identity from the durable task/answer, not from
    mutable content)."""
    digest = hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]
    return f"final:{task_id}:{digest}"


def already_delivered(drive_root: Any, delivery_id: str) -> bool:
    """Read-only durable dedupe check (registration happens AFTER a real send)."""
    did = str(delivery_id or "").strip()
    if not did:
        return False
    try:
        from ouroboros.utils import read_json_dict

        data = read_json_dict(pathlib.Path(drive_root) / "state" / "terminal_deliveries.json") or {}
        delivered = data.get("delivered")
        return isinstance(delivered, list) and did in delivered
    except Exception:
        # Fail open toward delivery: a broken registry read must never cost the
        # only owner-visible answer ("never lost" outranks "never doubled").
        log.debug("terminal-delivery registry read failed for %s", did, exc_info=True)
        return False


def _pending_rows(current: Dict[str, Any], *, strict: bool = False) -> Dict[str, Any]:
    """The owed-outbox rows; ``strict`` refuses a malformed nested value.

    GR5-6: the three MUTATORS pass ``strict=True`` — a present-but-non-dict
    ``pending`` under a valid top-level dict used to be coerced to ``{}`` and
    the next mutation overwrote EVERY owed row silently, the exact loss the
    top-level ``strict_existing_dict`` check refuses. The raise is the same
    typed ``ValueError`` that check uses, so the callers' existing
    corrupt-registry handling (refuse + disclose, no overwrite) applies.
    Read paths stay fail-soft (they disclose separately)."""
    pending = current.get("pending")
    if isinstance(pending, dict):
        return dict(pending)
    if strict and pending is not None:
        raise ValueError(
            "terminal-delivery registry 'pending' is malformed (not an object)"
        )
    return {}


def _delivered_rows(current: Dict[str, Any], *, strict: bool = False) -> List[str]:
    """The delivered-id list; ``strict`` refuses a malformed container or entry.

    GR6-3 row strictness: the mutators rewrite the whole file, so a
    present-but-non-list ``delivered`` — or a non-string ENTRY inside a valid
    list — used to be silently coerced and overwritten on the next write,
    destroying the dedupe evidence with no corruption event. Strict raises the
    same typed ``ValueError`` the container checks use (refuse + disclose, no
    overwrite — the malformed bytes stay on disk); reads stay fail-soft."""
    delivered = current.get("delivered")
    if isinstance(delivered, list):
        if strict and any(not isinstance(item, str) for item in delivered):
            raise ValueError(
                "terminal-delivery registry 'delivered' holds non-string entries"
            )
        return [str(item) for item in delivered]
    if strict and delivered is not None:
        raise ValueError(
            "terminal-delivery registry 'delivered' is malformed (not a list)"
        )
    return []


def register_delivery(drive_root: Any, delivery_id: str) -> bool:
    """Durably register one delivery id AFTER a successful send.

    Returns whether the id was newly registered. Registration only after the
    send preserves the "never lost" ordering: a send that raised keeps its id
    unregistered, so a buffered second copy is still delivered. The registry is
    a bounded newest-last list — old ids age out, which is safe because a
    delivery id binds a terminal answer to its content hash and terminal answers
    stop being re-sent long before the cap turns over.

    The same write CLEARS the pending-outbox row: "delivered" and "still owed"
    are one transaction, so a replay can never re-send an answer that landed.
    """
    did = str(delivery_id or "").strip()
    if not did:
        return True
    fresh = {"value": False}

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rows: List[str] = _delivered_rows(current, strict=True)
        pending = _pending_rows(current, strict=True)
        target = pending.get(did)
        if target is not None and not isinstance(target, dict):
            # GR6-3: a malformed owed ROW is corruption — clearing it as part
            # of "delivered" would silently destroy the forensic bytes.
            raise ValueError(
                f"terminal-delivery registry pending row for {did} is malformed"
            )
        had_pending = pending.pop(did, None) is not None
        if did in rows:
            if not had_pending:
                return None
            return {"schema_version": _SCHEMA_VERSION, "delivered": rows[-_REGISTRY_CAP:],
                    "pending": pending}
        rows.append(did)
        fresh["value"] = True
        return {"schema_version": _SCHEMA_VERSION, "delivered": rows[-_REGISTRY_CAP:],
                "pending": pending}

    try:
        # GR3-9 strict read: a malformed registry must refuse the mutation
        # loudly, never collapse to {} and overwrite every owed answer.
        update_json_locked(_registry_path(drive_root), _mutate, strict_existing_dict=True)
    except ValueError:
        _log_registry_corrupt(drive_root, did, op="register_delivery")
        return True
    except Exception:
        log.warning("terminal-delivery registry write failed for %s", did, exc_info=True)
        return True
    return fresh["value"]


def _log_registry_corrupt(drive_root: Any, delivery_id: str, *, op: str) -> None:
    """Typed fail-closed disclosure for a corrupt delivery registry (GR3-9).

    The mutation was REFUSED (no overwrite — the malformed file keeps whatever
    forensic value it has); the gap is made visible through a durable typed
    event, because a silent {}-collapse was exactly the loss mode this closes.
    """
    log.error(
        "terminal-delivery registry is corrupt; %s refused for %s (no overwrite)",
        op, delivery_id,
    )
    try:
        from ouroboros.utils import append_jsonl

        append_jsonl(
            pathlib.Path(drive_root) / "logs" / "events.jsonl",
            {"ts": utc_now_iso(), "type": "terminal_delivery_registry_corrupt",
             "op": op, "delivery_id": str(delivery_id or "")},
        )
    except Exception:
        log.debug("registry-corrupt event append failed", exc_info=True)


def register_pending_delivery(drive_root: Any, event: Dict[str, Any]) -> bool:
    """Record a terminal send as OWED, before it is enqueued.

    The incident class in its purest form: the answer was salvaged to disk and
    the send was handed to an in-memory queue — a crash between the settle and
    the send lost the owner's answer forever, because the delivered registry only
    remembers what ALREADY went out. This row is the other half: written first,
    cleared by ``register_delivery`` after a confirmed send, and replayed by
    ``replay_pending_deliveries`` on boot and on the supervisor tick.

    Deliberately NOT a general send queue (owner scope): it holds only terminal
    answers going through this seam, bounded to ``_PENDING_CAP`` newest rows.

    Return semantics (GR3-4 — callers now BRANCH on this): ``True`` means the
    answer is durably tracked — newly owed, already owed, or already in the
    delivered registry. ``False`` means a REAL durability gap: the event
    carries no ``delivery_id``, or the registry write failed (a typed
    ``terminal_delivery_unregistered`` event is emitted so the gap is
    visible). Cancel paths leave the intent open on ``False``; the normal
    completion path still enqueues the live send and relies on the typed event.
    """
    did = str(event.get("delivery_id") or "").strip()
    if not did:
        return False
    evicted: List[Dict[str, Any]] = []

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        evicted.clear()
        rows: List[str] = _delivered_rows(current, strict=True)
        if did in rows:
            return None  # already delivered: nothing is owed
        pending = _pending_rows(current, strict=True)
        if did in pending:
            if not isinstance(pending.get(did), dict):
                # GR6-3: the probe shape — a malformed row for THIS id would
                # be reported as "already durably owed" while the replay read
                # can never deliver it. Refuse loudly instead (typed
                # corruption event + unregistered disclosure, no overwrite).
                raise ValueError(
                    f"terminal-delivery registry pending row for {did} is malformed"
                )
            return None
        pending[did] = {**{k: v for k, v in event.items() if k != "type"},
                        "registered_at": utc_now_iso()}
        if len(pending) > _PENDING_CAP:
            for stale in list(pending)[: len(pending) - _PENDING_CAP]:
                row = pending.pop(stale, None)
                # GR2-6: eviction is a LOST OWED ANSWER, never a silent pop —
                # the row is disclosed below through the same exhaustion seam
                # (full-text preservation + typed event + owner notice).
                if isinstance(row, dict):
                    evicted.append({**row, "delivery_id": str(stale)})
        return {"schema_version": _SCHEMA_VERSION, "delivered": rows[-_REGISTRY_CAP:],
                "pending": pending}

    try:
        # GR3-9 strict read: a malformed registry refuses the mutation loudly
        # instead of collapsing to {} and overwriting every owed answer.
        update_json_locked(_registry_path(drive_root), _mutate, strict_existing_dict=True)
    except ValueError:
        _log_registry_corrupt(drive_root, did, op="register_pending_delivery")
        _log_unregistered_delivery(drive_root, event, did)
        return False
    except Exception:
        # The send itself still happens on the caller's live path; the answer
        # just lost its crash insurance — disclosed, never silent (GR3-4).
        log.error("terminal-delivery pending registration failed for %s", did, exc_info=True)
        _log_unregistered_delivery(drive_root, event, did)
        return False
    for row in evicted:
        _disclose_exhausted_delivery(drive_root, row, reason="outbox_capacity")
    return True


def _log_unregistered_delivery(drive_root: Any, event: Dict[str, Any], did: str) -> None:
    """Typed durable disclosure of a terminal send whose owed registration failed."""
    try:
        from ouroboros.utils import append_jsonl

        append_jsonl(
            pathlib.Path(drive_root) / "logs" / "events.jsonl",
            {"ts": utc_now_iso(), "type": "terminal_delivery_unregistered",
             "task_id": str((event or {}).get("task_id") or ""),
             "delivery_id": str(did or ""),
             "detail": ("the owed-outbox registration failed; the live send (if any) "
                        "proceeds without crash insurance")},
        )
    except Exception:
        log.debug("unregistered-delivery event append failed for %s", did, exc_info=True)


def pending_deliveries(
    drive_root: Any, *, disclose_corruption: bool = False,
) -> List[Dict[str, Any]]:
    """Terminal sends registered as owed and not yet confirmed delivered.

    GR5-6 read semantics: an ABSENT registry is an ordinary empty outbox; an
    UNREADABLE/MALFORMED one (file or nested ``pending``) is a real gap — it
    is disclosed with a typed ``log.error`` (and, for the watchdog's replay
    read via ``disclose_corruption=True``, the existing typed corruption
    event) before the read still returns ``[]``. Fail-soft read, but the
    owner can see the replay lane is degraded instead of silently owing
    nothing.
    """
    path = pathlib.Path(drive_root) / "state" / "terminal_deliveries.json"
    if not path.is_file():
        return []
    try:
        from ouroboros.utils import read_json_dict

        data = read_json_dict(path)
        if data is None:
            raise ValueError("registry file is malformed or is not an object")
        pending = _pending_rows(data, strict=True)
    except Exception as exc:
        log.error(
            "terminal-delivery registry is unreadable/malformed (%s); the owed "
            "outbox reads as EMPTY until the file is repaired", exc,
        )
        if disclose_corruption:
            _log_registry_corrupt(drive_root, "", op="pending_deliveries_read")
        return []
    delivered = data.get("delivered")
    delivered_ids = {str(item) for item in delivered} if isinstance(delivered, list) else set()
    rows: List[Dict[str, Any]] = []
    malformed: List[str] = []
    for did, row in pending.items():
        if not isinstance(row, dict):
            # GR6-3 row strictness on the enforcement read: a malformed owed
            # row used to vanish from replay silently — an answer the file
            # still claims is owed. Quarantined (skipped here, bytes kept on
            # disk — the strict mutators refuse to rewrite it) and disclosed
            # loudly below.
            malformed.append(str(did))
            continue
        if str(did) in delivered_ids:
            continue
        rows.append({**row, "delivery_id": str(did)})
    if malformed:
        log.error(
            "terminal-delivery outbox holds %d malformed owed row(s) (%s); "
            "quarantined — skipped by replay, bytes kept on disk",
            len(malformed), ", ".join(malformed[:5]),
        )
        if disclose_corruption:
            _log_registry_corrupt(
                drive_root, ",".join(malformed[:5]), op="pending_row_malformed",
            )
    return rows


def _bump_replay_attempts(drive_root: Any, ids: List[str]) -> List[str]:
    """Count one replay per id and DROP the ones that exhausted their attempts.

    Returns the ids that may still be replayed. The cap is what keeps a durable
    outbox from becoming a retry storm: a chat that is permanently unreachable
    would otherwise have its answer re-enqueued on every supervisor tick forever.
    Each bump also stamps ``last_replay_at`` — ``_replay_due`` spaces attempts
    with exponential backoff from it, so the cap covers a realistic outage
    window instead of burning on consecutive ticks. An exhausted row is dropped
    LOUDLY (AR2-7): the full message text is preserved on disk and the owner is
    told through a typed durable event plus a chat notice — never a silent drop.
    """
    live: List[str] = []
    exhausted: List[Dict[str, Any]] = []

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pending = _pending_rows(current, strict=True)
        changed = False
        for did in ids:
            row = pending.get(did)
            if not isinstance(row, dict):
                continue
            attempts = int(row.get("replay_attempts") or 0) + 1
            changed = True
            if attempts > _PENDING_MAX_REPLAYS:
                pending.pop(did, None)
                exhausted.append({**row, "delivery_id": did})
                continue
            pending[did] = {**row, "replay_attempts": attempts,
                            "last_replay_at": utc_now_iso()}
            live.append(did)
        if not changed:
            return None
        rows = _delivered_rows(current, strict=True)
        return {"schema_version": _SCHEMA_VERSION, "delivered": rows[-_REGISTRY_CAP:],
                "pending": pending}

    try:
        # GR3-9 strict read: never rebuild a malformed registry from {}.
        update_json_locked(_registry_path(drive_root), _mutate, strict_existing_dict=True)
    except ValueError:
        _log_registry_corrupt(drive_root, ",".join(ids), op="bump_replay_attempts")
        return list(ids)
    except Exception:
        log.warning("terminal-delivery replay bookkeeping failed", exc_info=True)
        return list(ids)
    for row in exhausted:
        _disclose_exhausted_delivery(drive_root, row)
    return live


def _disclose_exhausted_delivery(
    drive_root: Any, row: Dict[str, Any], *, reason: str = "replay_exhausted",
) -> None:
    """Owner-visible disclosure for an outbox row that is dropped undelivered.

    Two callers, one seam: a row that exhausted its bounded replays
    (``reason="replay_exhausted"``, AR2-7) and a row evicted past
    ``_PENDING_CAP`` by newer registrations (``reason="outbox_capacity"``,
    GR2-6). Either way the undelivered answer must not vanish into a
    log.error: the FULL message text is preserved durably, a typed
    ``terminal_delivery_exhausted`` event lands in ``logs/events.jsonl`` (the
    guaranteed half — it works even when chat itself is what is failing), and
    a chat notice through the ordinary supervisor notification path names the
    task, the preserved copy, and why delivery gave up. Every step fail-soft.
    """
    did = str(row.get("delivery_id") or "")
    tid = str(row.get("task_id") or "")
    text = str(row.get("text") or "")
    capacity = reason == "outbox_capacity"
    try:
        attempts = int(row.get("replay_attempts") or 0) if capacity else _PENDING_MAX_REPLAYS
    except (TypeError, ValueError):
        attempts = 0
    try:
        chat_id = int(row.get("chat_id") or 0)
    except (TypeError, ValueError):
        chat_id = 0
    if capacity:
        log.error(
            "Terminal answer %s (task %s) was evicted from the pending outbox "
            "(capacity %d exceeded) before its send confirmed; disclosing",
            did, tid, _PENDING_CAP,
        )
    else:
        log.error(
            "Terminal answer %s (task %s) could not be delivered after %d replays; giving up",
            did, tid, _PENDING_MAX_REPLAYS,
        )
    preserved = ""
    try:
        from ouroboros.observability import preserve_salvaged_output

        if tid and text:
            preserved = preserve_salvaged_output(
                pathlib.Path(drive_root), f"{tid}.undelivered", text,
            )
    except Exception:
        log.debug("exhausted-delivery preservation failed for %s", did, exc_info=True)
    try:
        from ouroboros.utils import append_jsonl

        append_jsonl(
            pathlib.Path(drive_root) / "logs" / "events.jsonl",
            {
                "ts": utc_now_iso(), "type": "terminal_delivery_exhausted",
                "task_id": tid, "delivery_id": did, "chat_id": chat_id,
                "reason": reason,
                "attempts": attempts, "preserved_path": preserved,
                "detail": (
                    "pending outbox over capacity; oldest owed row evicted"
                    if capacity else "send never confirmed; replay cap reached"
                ),
            },
        )
    except Exception:
        log.debug("exhausted-delivery event append failed for %s", did, exc_info=True)
    if not chat_id:
        return
    try:
        from supervisor.message_bus import send_with_budget

        copy_note = (
            f"The full text is preserved at {preserved}." if preserved
            else "No durable copy could be preserved."
        )
        cause = (
            f"its pending-outbox slot was evicted by newer owed answers (capacity {_PENDING_CAP})"
            if capacity else
            f"the send never confirmed after {_PENDING_MAX_REPLAYS} replay attempts (chat "
            "transport failing or the supervisor kept crashing mid-send)"
        )
        send_with_budget(
            chat_id,
            f"⚠️ A terminal answer for task {tid} could not be delivered: {cause}. {copy_note}",
            role="system",
            system_type="terminal_incident",
        )
    except Exception:
        log.debug("exhausted-delivery chat notice failed for %s", did, exc_info=True)


def replay_pending_deliveries(drive_root: Any, *, event_queue: Any = None) -> List[str]:
    """Re-enqueue every terminal answer still owed. Returns the replayed ids.

    Called on boot and on the supervisor tick. Idempotent by construction: the
    send handler suppresses an id already in the delivered registry, and a
    successful send clears the pending row in the same write. A row younger than
    ``_REPLAY_MIN_AGE_SEC`` is left alone — an ordinary send still working its
    way through the event queue is not a lost answer — and each row gets a
    BOUNDED number of replays so an unreachable chat cannot turn the outbox into
    a tick-rate retry storm.
    """
    # GR5-6: the replay lane is an enforcement read — a corrupt registry is
    # disclosed through the typed corruption event, not read as "nothing owed".
    owed = [
        row for row in pending_deliveries(drive_root, disclose_corruption=True)
        if _replay_due(row)
    ]
    if not owed:
        return []
    if event_queue is None:
        try:
            from supervisor import workers

            event_queue = workers.get_event_q()
        except Exception:
            log.warning("terminal-delivery replay could not reach the event queue", exc_info=True)
            return []
    allowed = set(_bump_replay_attempts(
        drive_root, [str(row.get("delivery_id") or "") for row in owed],
    ))
    replayed: List[str] = []
    for row in owed:
        did = str(row.get("delivery_id") or "")
        if did not in allowed:
            continue
        event = {**row, "type": "send_message"}
        for key in ("registered_at", "replay_attempts"):
            event.pop(key, None)
        try:
            event_queue.put(event)
        except Exception:
            log.warning("terminal-delivery replay enqueue failed for %s", did, exc_info=True)
            continue
        replayed.append(did)
    if replayed:
        log.info("Replayed %d undelivered terminal answer(s): %s", len(replayed), replayed)
    return replayed


def _replay_due(row: Dict[str, Any], *, now: Optional[float] = None) -> bool:
    """Whether an owed row is old enough that its send is presumed lost.

    Spaced with exponential backoff (AR2-7): attempt N waits
    ``_REPLAY_MIN_AGE_SEC * 2**N`` after the PREVIOUS replay (60s, 120s, 240s,
    480s, 960s — ~31 minutes of coverage), measured from ``last_replay_at``
    once one exists. Without the stamp all five attempts burned on consecutive
    supervisor ticks and the cap covered a ~5-minute outage at best.
    """
    raw = str(row.get("last_replay_at") or row.get("registered_at") or "").replace("Z", "+00:00")
    if not raw:
        return True
    try:
        attempts = int(row.get("replay_attempts") or 0)
    except (TypeError, ValueError):
        attempts = 0
    try:
        from datetime import datetime, timezone

        parsed = datetime.fromisoformat(raw)
        stamped = (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
        current = now if now is not None else datetime.now(timezone.utc).timestamp()
    except (TypeError, ValueError):
        return True
    return (current - stamped) >= _REPLAY_MIN_AGE_SEC * (2 ** min(attempts, 6))


def salvage_cancelled_output(
    drive_root: Any, task_drive: Any, task_id: str,
) -> tuple[str, str, str]:
    """(result-suffix note, full salvaged text, preserved-copy path) — fail-soft.

    The kill-path salvage: the note rides the terminal result, the FULL text is
    preserved on the drive that OUTLIVES the child drive publication deletes
    (BIBLE P1 — a bounded preview alone is not a rescue when it is about to
    become the only copy), and the path feeds this seam's delivery receipt.
    """
    try:
        from ouroboros.observability import (
            latest_llm_response_text, preserve_salvaged_output, salvaged_output_note,
        )

        note = salvaged_output_note(
            pathlib.Path(task_drive), str(task_id),
            preserve_root=pathlib.Path(drive_root),
        )
        text = latest_llm_response_text(pathlib.Path(task_drive), str(task_id))
        path = ""
        if text:
            try:
                path = preserve_salvaged_output(pathlib.Path(drive_root), str(task_id), text)
            except Exception:
                path = ""
        return note, text, path
    except Exception:
        log.debug("Failed to salvage last LLM response for cancelled %s", task_id, exc_info=True)
        return "", "", ""


def unreconciled_runs_note(unreconciled_runs: Optional[List[str]]) -> str:
    """GR6-5a: the ONE outcome-independent disclosure line for open delegated runs.

    Appended to the owner's terminal message whenever the list is non-empty,
    REGARDLESS of the outcome (completed, failed, cancelled, reaped): a task
    whose teardown left delegated runs open must never read as cleanly
    finished, and only the cancelled wording used to carry the warning.
    Returns "" for an empty list.
    """
    runs = [str(rid) for rid in (unreconciled_runs or []) if str(rid)]
    if not runs:
        return ""
    return (
        f"\n\n⚠️ {len(runs)} delegated run(s) may still be live: "
        + ", ".join(runs)
    )


def build_completed_result_event(
    drive_root: Any, task: Optional[Dict[str, Any]], task_id: str,
    stored: Optional[Dict[str, Any]],
    *, unreconciled_runs: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Build (without sending) the completion-wins ``send_message`` event.

    Split from the enqueue half (GR2-4) so the cancel path can register the
    answer as OWED in the durable outbox BEFORE it settles the cancel intent —
    a crash between the settle and the send then replays the row instead of
    losing both the watchdog trigger and the answer. Returns None when there
    is nothing deliverable (no text / no lineage chat).

    ``unreconciled_runs`` (GR6-5a): a completed answer whose teardown left
    delegated runs open carries the outcome-independent disclosure line — the
    completed wording used to omit it entirely. GR7-4: the note rides the TEXT
    but never the delivery id — the id digests the CORE answer only, so a
    replay whose rebuilt note shrank (runs reconciled meanwhile) dedups to the
    same owed message instead of minting a second one. This also makes the id
    byte-equal to the natural path's ``final:<tid>:<digest>`` for the same
    stored answer.
    """
    tid = str(task_id or "").strip()
    core_text = str((stored or {}).get("result") or "")
    task_row = task if isinstance(task, dict) and task else dict(stored or {})
    chat_id = lineage_chat_id(pathlib.Path(drive_root), task_row, tid)
    if not tid or not core_text or not chat_id:
        return None
    event = {
        "type": "send_message",
        "chat_id": chat_id,
        "task_id": tid,
        "text": core_text + unreconciled_runs_note(unreconciled_runs),
        # The natural final answer is rendered as markdown; a re-delivered
        # copy that drops the format renders as a different message.
        "format": "markdown",
        "delivery_id": delivery_id_for(tid, core_text),
    }
    return project_terminal_result_event(
        pathlib.Path(drive_root), task_row, tid,
        result_text=core_text,
        terminal_origin=(stored or {}).get("terminal_origin"),
        base_event=event, provider_notice=str((stored or {}).get("terminal_provider_notice") or ""),
    )


def project_terminal_result_event(
    drive_root: Any,
    task: Optional[Dict[str, Any]],
    task_id: str,
    *,
    result_text: str,
    terminal_origin: Any,
    base_event: Optional[Dict[str, Any]] = None,
    provider_notice: str = "",
) -> Dict[str, Any]:
    """Project one terminal event from producer-stamped origin.

    ``host_salvage`` becomes one short keyed plain System receipt (inherited
    ``format``/``log_text`` dropped; the full bytes stay in task details).
    ``host_notice`` is a text the host wrote alone, so it keeps its OWN words
    and inherited markdown and becomes a System row WITHOUT a system_type,
    which is what lets a replayed card conclude on it. ``model_final`` and a
    missing legacy origin keep the assistant projection untouched — no
    text/status/length inference. The delivery id always digests the stable
    core result rather than a mutable disclosure suffix.
    """
    tid = str(task_id or "")
    core_text = str(result_text or "")
    event = dict(base_event or {})
    event.setdefault("type", "send_message")
    event.setdefault("task_id", tid)
    event.setdefault("chat_id", lineage_chat_id(drive_root, task or {}, tid))
    event["delivery_id"] = delivery_id_for(tid, core_text)
    origin = str(terminal_origin or "")
    if origin in HOST_AUTHORED_TERMINAL_ORIGINS:
        salvage = origin == TERMINAL_ORIGIN_HOST_SALVAGE
        receipt = (provider_notice + "\n\nThe full intermediate output and technical details are preserved "
                   "in the task details.") if provider_notice else _HOST_SALVAGE_RECEIPT
        event.update({
            "text": receipt if salvage else (event.get("text") or core_text),
            "role": "system",
            "terminal_origin": origin,
            **({"system_type": "terminal_incident"} if salvage else {}),
        })
        if salvage:
            event.pop("log_text", None)
            event.pop("format", None)
        return event
    if origin == TERMINAL_ORIGIN_MODEL_FINAL:
        event["terminal_origin"] = TERMINAL_ORIGIN_MODEL_FINAL
    # A missing origin now identifies only a row written before every forced
    # rail stamped one; it stays absent so legacy rows replay byte-compatibly.
    return event


def enqueue_terminal_delivery(
    drive_root: Any, event: Dict[str, Any], *, event_queue: Any = None,
) -> bool:
    """Dedupe, register as owed (idempotent), and enqueue one built event.

    The enqueue half of the seam: safe to call after the same event was already
    registered by the owed-before-settle ordering — registration is keyed by
    ``delivery_id`` and no-ops on a repeat.
    """
    did = str((event or {}).get("delivery_id") or "")
    tid = str((event or {}).get("task_id") or "")
    if not event or already_delivered(pathlib.Path(drive_root), did):
        return False
    register_pending_delivery(pathlib.Path(drive_root), event)
    try:
        if event_queue is None:
            from supervisor import workers

            event_queue = workers.get_event_q()
        event_queue.put(dict(event))
    except Exception:
        log.warning("terminal-delivery enqueue failed for %s", tid, exc_info=True)
        return False
    return True


def deliver_completed_result(
    drive_root: Any, task: Optional[Dict[str, Any]], task_id: str,
    stored: Optional[Dict[str, Any]], *, event_queue: Any = None,
    unreconciled_runs: Optional[List[str]] = None,
) -> bool:
    """Completion-wins delivery (owner 4=A + 5=A): ship a KEPT completed result.

    The worker died (or was already gone) before its own final delivery could be
    confirmed, so the completed answer goes out through this seam: owed BEFORE
    enqueued, deduped by the same ``final:<tid>:<digest>`` identity the natural
    path mints — a copy the worker already delivered is suppressed durably.
    Returns whether a send was enqueued. ``unreconciled_runs`` rides the text
    (GR6-5a) and must match what the owed registration was built with, or the
    two halves mint different delivery ids.
    """
    event = build_completed_result_event(
        pathlib.Path(drive_root), task, task_id, stored,
        unreconciled_runs=unreconciled_runs,
    )
    if event is None:
        return False
    return enqueue_terminal_delivery(pathlib.Path(drive_root), event, event_queue=event_queue)


def deliver_miss_lane_outcome(
    drive_root: Any, task_drive: Any, row: Dict[str, Any], task_id: str, status: str,
    *, event_queue: Any = None, unreconciled_runs: Optional[List[str]] = None,
) -> bool:
    """Owner 5=A on the finalize-on-miss lane: a task that was neither queued
    nor running when custody caught up still gets its answer delivered — a
    completed result as itself (completion wins, normal delivery), a cancelled
    outcome as an UNREVIEWED salvage. Other settled statuses (failed /
    rejected) keep their own paths' delivery. Fail-soft; the durable dedupe
    suppresses a copy an earlier path already sent. Subagent results flow to
    their parent through the ordinary handoff instead of chat.

    ``unreconciled_runs`` (GR5-3) rides the cancelled salvage message exactly
    like the kill path's disclosure — a miss-lane cancel that left delegated
    runs open must not read as a clean completion.

    Returns whether the answer is DURABLY accounted for (GR4-1, mirroring
    ``_register_owed_terminal_delivery``): owed in the outbox, already
    delivered/owed, or legitimately not deliverable here (subagent /
    non-delivering status / no lineage chat, which records the typed handoff
    row). ``False`` means a real durability gap — the owed registration
    failed — and the caller must leave the cancel intent OPEN (claim
    released) instead of settling over an unowed answer.
    """
    row = row if isinstance(row, dict) else {}
    if str(row.get("delegation_role") or "") == "subagent":
        return True
    try:
        if status == "completed":
            # GR6-5a: the completed answer carries the outcome-independent
            # unreconciled-runs line — only the cancelled wording used to.
            event = build_completed_result_event(
                pathlib.Path(drive_root), row, task_id, row,
                unreconciled_runs=list(unreconciled_runs or []),
            )
            if event is None:
                return True  # nothing deliverable (no text / no lineage chat)
            owed = register_pending_delivery(pathlib.Path(drive_root), event)
            enqueue_terminal_delivery(
                pathlib.Path(drive_root), event, event_queue=event_queue,
            )
            return owed
        if status != "cancelled":
            # failed/rejected outcomes keep their own paths' delivery — but an
            # open delegated run is a fact no outcome may swallow (GR6-5a):
            # deliver the ONE disclosure line as its own message when the list
            # is non-empty. Owed-before-enqueued and delivery-id-deduped like
            # every terminal send through this seam.
            note = unreconciled_runs_note(unreconciled_runs)
            if not note:
                return True
            chat_id = lineage_chat_id(pathlib.Path(drive_root), row, task_id)
            if not chat_id:
                return True
            # GR7-4: the id digests the STABLE core (task id + settled
            # status); the mutable note rides the text only, so a replay
            # whose list shrank dedups instead of owing a second message.
            core = f"Task {task_id} settled as {status}."
            event = {
                "type": "send_message", "chat_id": chat_id, "task_id": task_id,
                "text": core + note, "delivery_id": delivery_id_for(task_id, core),
                # Host-authored disclosure fact — typed SYSTEM (Q4 non-mimicry).
                "role": "system", "system_type": "cancel_receipt",
                "ts": utc_now_iso(),
            }
            owed = register_pending_delivery(pathlib.Path(drive_root), event)
            enqueue_terminal_delivery(
                pathlib.Path(drive_root), event, event_queue=event_queue,
            )
            return owed
        from ouroboros.observability import preserved_salvage_path

        _note, text, path = salvage_cancelled_output(
            pathlib.Path(drive_root), task_drive, task_id,
        )
        if not path:
            path = preserved_salvage_path(pathlib.Path(drive_root), task_id)
        if not text and path:
            try:
                text = pathlib.Path(path).read_text(encoding="utf-8")
            except Exception:
                text = ""
        # GR4-1: the owed registration happens FIRST so its result is
        # observable; the shared seam below re-registers idempotently (and
        # records the typed handoff row when there is no lineage chat).
        event = build_unreviewed_salvage_event(
            pathlib.Path(drive_root), row, task_id, outcome="cancelled",
            salvaged_text=text, preserved_path=path, settled_status="cancelled",
            unreconciled_runs=list(unreconciled_runs or []),
        )
        owed = True
        if event is not None:
            owed = register_pending_delivery(pathlib.Path(drive_root), event)
        deliver_unreviewed_salvage(
            pathlib.Path(drive_root), row, task_id, outcome="cancelled",
            salvaged_text=text, preserved_path=path,
            settled_status="cancelled", event_queue=event_queue,
            unreconciled_runs=list(unreconciled_runs or []),
        )
        return owed
    except Exception:
        log.warning("Miss-lane terminal delivery failed for %s", task_id, exc_info=True)
        return False


def deliver_cascade_summary(
    drive_root: Any, task_id: str, root_task_row: Dict[str, Any], outcomes: Dict[str, str],
) -> bool:
    """ONE chat message for a settled cascade: root salvage + children digest.

    A2: the sweeps ran custody with ``deliver=False``, so this is the tree's only
    owner-facing terminal message. Fail-soft; durable dedupe inside the seam.
    The cascade postcondition calls this UNCONDITIONALLY before it settles the
    root intent (GR3-1c) — including the replay/already-down path — so a tree
    with no resolvable lineage chat records a typed ``terminal_delivery_handoff``
    row instead of vanishing (the crash-order evidence must show the summary was
    consciously not owed, never silently dropped).

    DELIVERY IDENTITY (GR4-2): the summary's delivery id is
    ``cascade:<root_tid>:<intent request_id>`` — deterministic per cascade
    INTENT, not per message content. A watchdog replay of the SAME intent
    (crash between the owed summary and the settle) rebuilds a digest whose
    content may differ (children settled meanwhile), and a content-derived id
    would deliver a second summary; the intent-derived id dedups it through
    the delivered registry, while a LATER separate cancel request (new
    request_id) legitimately delivers its own. Falls back to the content id
    when no intent row exists (direct internal callers).

    Returns whether the summary is DURABLY accounted for (GR4-1): owed /
    already delivered / consciously handed off. ``False`` = the owed
    registration failed; the caller leaves the cascade intent OPEN so the
    watchdog re-feeds (loud by the typed ``terminal_delivery_unregistered``
    event each attempt).
    """
    try:
        from ouroboros.observability import preserved_salvage_path
        from ouroboros.task_results import load_task_result
        from ouroboros.task_status import SETTLED_STATUSES

        def _durable_child_outcome(tid: str, sweep_outcome: str) -> str:
            # GR4-4: each child's line reports its CURRENT durable status at
            # digest build time — a child a concurrent actor settled after the
            # sweep must not be shown with the sweep's stale ``failed``.
            try:
                status = str(
                    (load_task_result(pathlib.Path(drive_root), tid) or {}).get("status") or ""
                )
            except Exception:
                status = ""
            return status if status in SETTLED_STATUSES else sweep_outcome

        # GR5-4: digest MEMBERSHIP comes from the durable tree, not only this
        # run's sweep outcomes — a watchdog replay after the children already
        # terminalized runs with EMPTY outcomes and would otherwise deliver a
        # root summary whose digest silently omits the whole subtree. GR6-2:
        # membership is ANCESTRY rooted at the cancelled node (a parent-chain
        # walk over the durable rows plus the queue snapshot — the same
        # reachability the cascade sweep uses), never a root_task_id equality:
        # a MID-TREE target's grandchildren keep the ORIGINAL tree's root id
        # and match no equality clause, and non-subagent descendants belong in
        # the digest too. Sweep outcomes win only for ids with no durable row
        # yet (fresh kills whose write is still in flight); every line still
        # reads its CURRENT durable status through ``_durable_child_outcome``
        # (GR4-4). Bounded exactly as today: the durable ``cancel_receipt``
        # block caps the persisted digest at 40 with the exact omitted count.
        merged: Dict[str, str] = {
            str(tid): str(outcome) for tid, outcome in outcomes.items()
        }
        try:
            for tid, status in _cascade_descendant_rows(
                pathlib.Path(drive_root), task_id,
            ).items():
                if tid and tid not in merged:
                    merged[tid] = status
        except Exception:
            log.debug(
                "cascade digest durable-tree enumeration failed for %s",
                task_id, exc_info=True,
            )
        children = [
            {"task_id": tid, "outcome": _durable_child_outcome(tid, outcome),
             "salvaged": bool(preserved_salvage_path(pathlib.Path(drive_root), tid))}
            for tid, outcome in sorted(merged.items())
            if tid != task_id
        ]
        root_result = load_task_result(pathlib.Path(drive_root), task_id) or {}
        root_status = str(root_result.get("status") or "")
        salvage_path = preserved_salvage_path(pathlib.Path(drive_root), task_id)
        salvage_text = ""
        if root_status == "completed":
            # Completion-wins root: its own completed result is the answer.
            salvage_text = str(root_result.get("result") or "")
        elif salvage_path:
            try:
                salvage_text = pathlib.Path(salvage_path).read_text(encoding="utf-8")
            except Exception:
                salvage_text = ""
        delivery_id = ""
        try:
            from ouroboros.cancel_intents import active_intent

            intent = active_intent(pathlib.Path(drive_root), task_id) or {}
            if intent.get("request_id"):
                delivery_id = f"cascade:{task_id}:{intent['request_id']}"
        except Exception:
            log.debug("cascade delivery-id intent read failed for %s", task_id, exc_info=True)
        event = build_unreviewed_salvage_event(
            pathlib.Path(drive_root),
            root_task_row or root_result,
            task_id,
            outcome=_cascade_root_outcome(root_status),
            salvaged_text=salvage_text,
            preserved_path=salvage_path,
            children=children,
            settled_status=root_status,
            delivery_id=delivery_id,
        )
        # GR4-1: register the owed row FIRST so the accounting result is
        # observable; deliver_unreviewed_salvage re-registers idempotently.
        owed = True
        if event is not None:
            owed = register_pending_delivery(pathlib.Path(drive_root), event)
        deliver_unreviewed_salvage(
            pathlib.Path(drive_root),
            root_task_row or root_result,
            task_id,
            outcome=_cascade_root_outcome(root_status),
            salvaged_text=salvage_text,
            preserved_path=salvage_path,
            children=children,
            # GR2-12: the typed root status decides the completed-vs-salvage
            # framing; the prose in ``outcome`` is presentation only.
            settled_status=root_status,
            delivery_id=delivery_id,
        )
        return owed
    except Exception:
        log.warning("Cascade summary delivery failed for %s", task_id, exc_info=True)
        return False


def _cascade_descendant_rows(drive_root: pathlib.Path, task_id: str) -> Dict[str, str]:
    """``{descendant_id: status}`` for every task whose ancestry reaches ``task_id``.

    GR6-2: the digest's membership walk. Rows come from the durable task
    results UNION the queue snapshot (a still-live or freshly-killed child may
    have no durable row yet); reachability is the recorded ``root_task_id``
    shortcut (covers a whole tree whose intermediate parents left no rows)
    plus a depth-bounded parent-chain walk over the collected rows — the same
    reachability shape the cascade sweep and ``_has_live_ancestor_in_set``
    use. Deliberately NOT filtered to subagents: a non-subagent descendant
    torn down by the cascade belongs in the tree's one summary too.
    """
    target = str(task_id or "").strip()
    rows: Dict[str, Dict[str, Any]] = {}
    try:
        from ouroboros.task_results import list_task_results

        for item in list_task_results(pathlib.Path(drive_root)):
            tid = str(item.get("task_id") or item.get("id") or "")
            if tid:
                rows[tid] = item
    except Exception:
        log.debug("cascade digest durable-row scan failed for %s", target, exc_info=True)
    try:
        from ouroboros.utils import read_json_dict

        snapshot = read_json_dict(
            pathlib.Path(drive_root) / "state" / "queue_snapshot.json",
        ) or {}
        for group, status in (("pending", "scheduled"), ("running", "running")):
            for item in snapshot.get(group) or []:
                if not isinstance(item, dict):
                    continue
                task = item.get("task") if isinstance(item.get("task"), dict) else item
                tid = str(item.get("id") or task.get("id") or "")
                if tid and tid not in rows:
                    rows[tid] = {**task, "status": status}
    except Exception:
        log.debug("cascade digest snapshot scan failed for %s", target, exc_info=True)

    def _reaches_target(tid: str) -> bool:
        row = rows.get(tid) or {}
        if str(row.get("root_task_id") or "") == target:
            return True
        parent = str(row.get("parent_task_id") or "")
        seen: set = set()
        while parent and parent not in seen and len(seen) < 100:
            if parent == target:
                return True
            seen.add(parent)
            parent = str(rows.get(parent, {}).get("parent_task_id") or "")
        return False

    return {
        tid: str(row.get("status") or "")
        for tid, row in rows.items()
        if tid != target and _reaches_target(tid)
    }


def _cascade_root_outcome(root_status: str) -> str:
    """How the tree's ONE message describes its root — from the ROOT'S OWN status.

    A cascade over a root that had already ``failed`` (or was killed on a budget
    hard stop) used to describe it as "cancelled", which is the summary telling
    the owner a different story than the task card. Only a root this cascade
    actually cancelled is called cancelled.
    """
    status = str(root_status or "").strip().lower()
    if status == "completed":
        return "completed before the cancellation (result preserved)"
    if status and status != "cancelled":
        return f"already settled as {status} before the cancellation (result preserved)"
    return "cancelled"


def salvage_preview(text: str, *, limit: int = SALVAGE_PREVIEW_CHARS) -> tuple[str, int]:
    """Bounded head preview plus the EXACT omitted char count (0 when whole)."""
    body = str(text or "")
    if len(body) <= limit:
        return body, 0
    return body[:limit], len(body) - limit


def _salvage_receipt(preserved_path: str) -> Dict[str, Any]:
    """Durable full-copy receipt: path + size + sha256 (empty when unreadable)."""
    receipt: Dict[str, Any] = {"path": str(preserved_path or "")}
    try:
        path = pathlib.Path(str(preserved_path or ""))
        data = path.read_bytes()
        receipt["size_bytes"] = len(data)
        receipt["sha256"] = hashlib.sha256(data).hexdigest()
        # The replay guard in _persist_cancel_receipt only lets a block with
        # preserved=True heal an earlier placeholder — mark the REAL receipt.
        receipt["preserved"] = True
    except Exception:
        receipt["unreadable"] = True
    return receipt


def _preview_note_line(preserved_path: str, omitted: int) -> str:
    """ONE plain sentence about the preview and where the technical facts live.

    Q5=A: path, sha256, byte count, and the children digest belong in the task
    DETAILS panel (the durable ``cancel_receipt`` block), never in chat. The
    chat keeps only the honesty half the owner must see inline: whether the
    preview is the WHOLE salvage (exact omitted count) and whether a durable
    full copy exists at all — a silently-truncated or silently-unpreserved
    fragment must not read as complete.
    """
    omitted_text = (
        f"{omitted} chars omitted from this preview" if omitted
        else "nothing omitted — this is the whole salvaged text"
    )
    if not str(preserved_path or "").strip():
        return (
            f"[{omitted_text}; NO durable full copy was preserved for this task "
            "(preservation did not run or found nothing)]"
        )
    return (
        f"[{omitted_text}; the full copy and technical details are in the "
        "task's details panel]"
    )


def lineage_chat_id(drive_root: Any, task: Dict[str, Any], task_id: str) -> int:
    """The task's OWN lineage chat (project binding first), never owner_chat_id."""
    try:
        from ouroboros.projects_registry import project_chat_for_task

        for candidate in (
            task_id,
            str(task.get("parent_task_id") or ""),
            str(task.get("root_task_id") or ""),
        ):
            if not candidate:
                continue
            bound = int(project_chat_for_task(drive_root, candidate) or 0)
            if bound:
                return bound
    except Exception:
        log.debug("terminal-delivery project chat lookup failed", exc_info=True)
    try:
        return int(task.get("chat_id") or 0)
    except (TypeError, ValueError):
        return 0


def _stop_episode_delivery_id(drive_root: Any, task_id: str) -> str:
    """CF-04: the receipt's identity is the STOP EPISODE — ``cancel:<tid>:<rid>``.

    Bound to the durable cancel-intent ``request_id``, never to mutable prose,
    so the id survives wording changes and restart replay. Pre-settle callers
    (owed registration, miss lane, cascade) read the ACTIVE intent; the
    publish half rebuilds the event AFTER the settle removed that row, so it
    re-derives the SAME id from the pending row the pre-settle half already
    registered. Returns "" when no episode is known (e.g. a reap with no
    intent) — the content-derived identity remains the fallback.
    """
    tid = str(task_id or "")
    try:
        from ouroboros.cancel_intents import active_intent

        rid = str((active_intent(pathlib.Path(drive_root), tid) or {}).get("request_id") or "")
        if rid:
            return f"cancel:{tid}:{rid}"
    except Exception:
        log.debug("stop-episode intent read failed for %s", tid, exc_info=True)
    try:
        from ouroboros.utils import read_json_dict

        data = read_json_dict(pathlib.Path(drive_root) / "state" / "terminal_deliveries.json") or {}
        prefix = f"cancel:{tid}:"
        owed = [
            (str(row.get("registered_at") or ""), did)
            for did, row in _pending_rows(data).items()
            if isinstance(row, dict) and did.startswith(prefix)
        ]
        if owed:
            return max(owed)[1]
    except Exception:
        log.debug("stop-episode owed-row read failed for %s", tid, exc_info=True)
    return ""


def _persist_cancel_receipt(
    drive_root: Any, task_id: str, *,
    settled_status: str, outcome: str, delivery_id: str,
    preserved_path: str, preview_omitted: int,
    children: Optional[List[Dict[str, Any]]] = None,
    unreconciled_runs: Optional[List[str]] = None,
) -> None:
    """Q5=A: the technical stop facts live in the task DETAILS panel.

    Merges ONE typed ``cancel_receipt`` block into the EXISTING durable task
    result (``TaskDetailResponse`` is an open shape, so the panel gets it with
    no contract change): full-copy path + size + full sha256 (or an honest
    unreadable/unpreserved marker), exact preview-omitted count, the stop
    reason when the intent is still readable, and the historical children
    digest for a cascade root. Never creates the result file (a later full
    write would clobber a block-only row) and never clobbers previously
    persisted non-empty facts with an emptier rebuild. Fail-soft.
    """
    tid = str(task_id or "")
    try:
        from ouroboros.task_results import task_result_path
        from ouroboros.utils import update_json_locked

        block: Dict[str, Any] = {
            "settled_status": str(settled_status or ""),
            "outcome": str(outcome or ""),
            "delivery_id": str(delivery_id or ""),
            "preview_omitted_chars": int(preview_omitted or 0),
            "salvage": (
                _salvage_receipt(preserved_path)
                if str(preserved_path or "").strip()
                else {"path": "", "preserved": False}
            ),
            "ts": utc_now_iso(),
        }
        rows = [
            {"task_id": str(c.get("task_id") or ""),
             "outcome": str(c.get("outcome") or ""),
             "salvaged": bool(c.get("salvaged"))}
            for c in list(children or []) if isinstance(c, dict)
        ]
        if rows:
            block["children"] = rows[:40]
            if len(rows) > 40:
                block["children_omitted"] = len(rows) - 40
        runs = [str(rid) for rid in (unreconciled_runs or []) if str(rid)]
        if runs:
            block["unreconciled_runs"] = runs
        try:
            from ouroboros.cancel_intents import active_intent

            reason = str((active_intent(pathlib.Path(drive_root), tid) or {}).get("reason") or "")
            if reason:
                block["stop_reason"] = reason
        except Exception:
            log.debug("cancel-receipt intent reason read failed for %s", tid, exc_info=True)

        def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if not isinstance(current, dict) or not current:
                return None  # no durable row yet — never mint a block-only file
            merged = dict(current.get("cancel_receipt") or {}) if isinstance(
                current.get("cancel_receipt"), dict) else {}
            for key, value in block.items():
                if (
                    key == "salvage"
                    and key in merged
                    and not (isinstance(value, dict) and value.get("preserved"))
                ):
                    # A replay/rebuild without a REAL salvage (the always-truthy
                    # {"path":"","preserved":False} placeholder) must not clobber
                    # a previously persisted preserved-copy fact.
                    continue
                if value or key not in merged:
                    merged[key] = value
            current = dict(current)
            current["cancel_receipt"] = merged
            return current

        update_json_locked(task_result_path(pathlib.Path(drive_root), tid), _mutate)
    except Exception:
        log.debug("cancel-receipt persistence failed for %s", tid, exc_info=True)


def build_unreviewed_salvage_event(
    drive_root: Any,
    task: Optional[Dict[str, Any]],
    task_id: str,
    *,
    outcome: str,
    salvaged_text: str = "",
    preserved_path: str = "",
    children: Optional[List[Dict[str, Any]]] = None,
    unreconciled_runs: Optional[List[str]] = None,
    settled_status: str = "",
    delivery_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Build (without sending) the one salvage/terminal chat message.

    The build half of the seam (GR2-4): the cancel path registers this event as
    OWED before it settles the durable intent. Returns None when the task has
    no lineage chat to route to.

    ``settled_status`` is the TYPED stored status (GR2-12): the completed-vs-
    salvage branch compares it against ``STATUS_COMPLETED`` instead of parsing
    the presentation prose in ``outcome``. The wording itself is unchanged.

    ``delivery_id`` overrides the content-derived identity (GR4-2): the cascade
    summary dedups per INTENT, not per content, because a replay's rebuilt
    digest can differ while still being the same owed message.

    CF-04 identity: when no explicit id is passed, the receipt binds to the
    stop episode as ``cancel:<tid>:<request_id>`` (active intent, or the owed
    row the pre-settle half registered) — stable across wording changes and
    restart replay. Content-derived ``final:...`` remains the fallback for
    terminal paths with no cancel episode at all.

    GR7-4: the content-derived fallback digests the STABLE lines only (status
    framing + salvage body + preview note) — the mutable unreconciled-runs
    disclosure rides the TEXT but never the id, so a replay whose rebuilt
    note shrank dedups to one owed message.

    Q4/Q5 presentation: the message is a typed SYSTEM receipt (``role`` +
    ``system_type`` ride the event end to end — Q4 non-mimicry), a raw
    fragment is named the last persisted intermediate model message (never an
    "answer"), and the technical facts (path/sha256/bytes/children digest)
    live in the durable ``cancel_receipt`` block on the task result — the
    details panel — not in chat.
    """
    from ouroboros.task_results import STATUS_COMPLETED

    tid = str(task_id or "").strip()
    if not tid:
        return None
    task_row = task if isinstance(task, dict) else {}
    chat_id = lineage_chat_id(pathlib.Path(drive_root), task_row, tid)
    if not chat_id:
        log.debug("terminal-delivery skipped for %s: no lineage chat", tid)
        return None
    body = str(salvaged_text or "").strip()
    preview, omitted = salvage_preview(body)
    outcome_text = str(outcome or "stopped")
    descendants = (
        f" {len(children)} descendant task(s) were settled with it." if children else ""
    )
    if str(settled_status or "").strip().lower() == STATUS_COMPLETED:
        # Completion-wins (owner 4=A): the kept result is the real answer, not a
        # salvage — but it still bypassed the normal delivery path, so say so.
        lines = [
            f"✅ Task {tid} {outcome_text}. Its completed result is preserved below."
            + descendants,
        ]
    else:
        lines = [
            f"⚠️ Task {tid} was {outcome_text}. Below is the last persisted "
            "intermediate model message, preserved WITHOUT review (salvaged "
            "best-effort; NOT a final answer)." + descendants,
        ]
    if preview:
        lines += ["", preview, "", _preview_note_line(preserved_path, omitted)]
    else:
        lines += ["", "(no salvageable agent output was found for this task)"]
    disclosure_lines: List[str] = []
    if unreconciled_runs:
        # GR3-7: an audit-failure marker means run state is UNKNOWN — a
        # different honest sentence than "these named runs stayed open".
        unknown = [
            str(rid) for rid in unreconciled_runs
            if str(rid).startswith(RUN_STATE_UNKNOWN_PREFIX)
        ]
        named = [
            str(rid) for rid in unreconciled_runs
            if not str(rid).startswith(RUN_STATE_UNKNOWN_PREFIX)
        ]
        if named:
            disclosure_lines += [
                "",
                "⚠️ DELEGATED RUNS NOT RECONCILED: the teardown could not reach the "
                "engine to settle "
                + ", ".join(named)
                + ". They may still be running and mutating the workspace until the "
                "orphan sweep catches them.",
            ]
        if unknown:
            disclosure_lines += [
                "",
                "⚠️ DELEGATED RUN STATE UNKNOWN: the teardown audit itself failed "
                f"({', '.join(unknown)}), so whether this task holds live delegated "
                "runs could not be determined. Periodic reconciliation will settle "
                "any that are still open.",
            ]
    # GR7-4: identity digests the STABLE lines only; the mutable disclosure
    # rides the text after the salvage body.
    text = "\n".join(lines + disclosure_lines)
    did = (
        str(delivery_id or "").strip()
        or _stop_episode_delivery_id(pathlib.Path(drive_root), tid)
        or delivery_id_for(tid, "\n".join(lines))
    )
    _persist_cancel_receipt(
        pathlib.Path(drive_root), tid,
        settled_status=str(settled_status or ""), outcome=outcome_text,
        delivery_id=did, preserved_path=str(preserved_path or ""),
        preview_omitted=omitted, children=children,
        unreconciled_runs=unreconciled_runs,
    )
    return {
        "type": "send_message",
        "chat_id": chat_id,
        "task_id": tid,
        "text": text,
        # Q4 non-mimicry: a host-authored receipt is typed SYSTEM end to end
        # (live WS frame, chat.jsonl direction, history replay) — it is never
        # rendered as Ouroboros's own speech. Card-neutral: task_id is a
        # subject reference; only task-result/task_done closes the card.
        "role": "system",
        "system_type": "cancel_receipt",
        "delivery_id": did,
        "ts": utc_now_iso(),
    }


def deliver_unreviewed_salvage(
    drive_root: Any,
    task: Optional[Dict[str, Any]],
    task_id: str,
    *,
    outcome: str,
    salvaged_text: str = "",
    preserved_path: str = "",
    children: Optional[List[Dict[str, Any]]] = None,
    unreconciled_runs: Optional[List[str]] = None,
    settled_status: str = "",
    delivery_id: str = "",
    event_queue: Any = None,
) -> bool:
    """Enqueue ONE unreviewed-salvage chat message for a cancelled/reaped task.

    Fail-soft and idempotent (durable ``delivery_id`` dedupe). Returns whether a
    message was actually enqueued. ``event_queue`` defaults to the live worker
    event queue; the supervisor's ``send_message`` handler performs the actual
    chat send with its ordinary formatting and transport behavior.

    ``unreconciled_runs`` names delegated runs the teardown could NOT settle
    (engine unreachable): the owner is being told the task is cancelled, so a run
    that may still be mutating the tree belongs in the same message.
    """
    event = build_unreviewed_salvage_event(
        pathlib.Path(drive_root), task, task_id,
        outcome=outcome, salvaged_text=salvaged_text, preserved_path=preserved_path,
        children=children, unreconciled_runs=unreconciled_runs,
        settled_status=settled_status, delivery_id=delivery_id,
    )
    if event is None:
        # GR3-1c: a terminal outcome with NO resolvable lineage chat records a
        # typed handoff row — the crash-order evidence must show the message
        # was consciously not owed, never silently dropped. (The cascade
        # postcondition relies on this: it always calls through here before it
        # settles the root intent.)
        try:
            from ouroboros.utils import append_jsonl

            append_jsonl(
                pathlib.Path(drive_root) / "logs" / "supervisor.jsonl",
                {"ts": utc_now_iso(), "type": "terminal_delivery_handoff",
                 "task_id": str(task_id or ""),
                 "settled_status": str(settled_status or ""),
                 "outcome": str(outcome or ""), "reason": "no_lineage_chat"},
            )
        except Exception:
            log.debug("no-chat handoff record failed for %s", task_id, exc_info=True)
        return False
    # OWED before ENQUEUED: a crash between here and the send replays this row
    # instead of losing the only owner-visible copy of the answer.
    return enqueue_terminal_delivery(pathlib.Path(drive_root), event, event_queue=event_queue)
