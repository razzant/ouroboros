"""Late reviewer feedback the host keeps in view in its owning room until the mind releases it.

A paid review that settles after its author can no longer receive it is advice for Ouroboros,
never a decision, a veto or a wake (owner decision on late review delivery, 2026-10-11, #1738):
it waits in the next ordinary turn of the room the work came from, and consciousness reads it
when that room stays quiet. Its record writer (a plan council reply that settled after its task
ended, a left change, result or commit review; marks of the deleted acceptance panel stay as
history) persists the full source first, then calls ``publish_late_review`` once per settled
publication with immutable facts: no live context, no clock.

The publication becomes ONE chronicle ``mark`` signed by the host (``LATE_REVIEW_OPERATION``)
in the room the memory view reads for that work: the room ``dialogue_evidence.own_room_chat``
proves (a project binding, the chat, its ancestors) when it is Main, a registered Project room
or the conversation of Presence-bound work. Another transport chat, the hidden partition or no
chat is never guessed into Main: no mark is placed (``not_activated``) and the full source stays
retained. The mark id is the publication's, so a repeat is ``unchanged``, a released mark stays
released (the journal replays the release after index loss) and the same id with other facts is
a ``conflict`` that writes nothing. The text is the writer's own words, each line and the whole
bounded with a disclosed omission (``utils.truncate_within_limit``); the target is the retained
source itself, read whole from any focus with ``memory_read(source_ref=…)`` unless the writer
names another reader (``source_read``).

``memory_view`` shows its words only in that room; consciousness gets one source-pointer line
per foreign mark. Only the integrating mind releases it, with a reason
(``chronicle_store._rule_mark_release``). Showing it,
a tool read or a mailbox acknowledgement of a notice is not that release.

A publication that could not be placed (``failed``) and every duty only the end of its author's
task can discharge stay owed as owner-held rows of the existing obligations table (``OWED_SET``,
immutable facts): a record a running author may still read itself (``owe_late_review``), and a
producer's own duty (``owe_after_end``: a change or commit wave its author left), which the
producer turns into marks once the task's canonical terminal status is recorded. The off-loop
maintenance pass replays those rows (``replay_owed_late_reviews``); a placed, unchanged or
released mark discharges the row, so a replay can neither duplicate nor resurrect it. A plan
reply needs no row of its own: its ask records which seats are still pending, so the maintenance child job
collects ended tasks' pending seats at $0 (``_collect_ended_plan_asks``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import sqlite3
from types import SimpleNamespace
from typing import Any, Dict, Mapping, NamedTuple, Optional

from ouroboros.chronicle_store import LATE_REVIEW_OPERATION, ChronicleStore
from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, WEB_UI_CHAT_ID, is_a2a_chat_id
from ouroboros.utils import truncate_within_limit

log = logging.getLogger(__name__)

# The review a mark names, by its writer's surface; another surface is named as recorded.
SURFACE_NAMES = {"task_acceptance": "result review", "result": "result review", "plan_review": "plan review",
                 "commit_gate": "commit review", "multi_model_review": "change review", "change": "change review",
                 "skill": "skill review"}
LINE_LIMIT = 800  # one line of the writer's words
TEXT_LIMIT = 4000  # the whole mark: the view never degrades a mark, so the bound is set here
OWED_SET = "late_review_marks"  # owner-held rows: id -> {record | left_review, after_end?}


class LateReviewMark(NamedTuple):
    """``saved``, ``unchanged``, ``not_activated`` (the owning room is not proven), ``memory_not_activated``,
    ``conflict``, ``invalid``, ``owed`` (kept for after its author ended) or ``failed`` (an I/O or lock
    failure: the publication stays owed)."""
    status: str
    mark_id: str = ""
    room_id: str = ""
    detail: str = ""


def late_review_mark_id(surface: str, task_id: str, publication_id: str) -> str:
    digest = hashlib.sha256("\0".join((surface, task_id, publication_id)).encode("utf-8")).hexdigest()
    return f"late-review-{digest[:32]}"


def owning_room(drive_root: Any, task_id: str, chat_id: Any) -> Optional[str]:
    """The room the view reads for this work when its ownership is proven, else ``None`` (never a guessed Main)."""
    from ouroboros.dialogue_evidence import own_room_chat, task_room_record  # D15->D06 is lazy-only
    from ouroboros.dialogue_provenance import presence_record_binding
    from ouroboros.memory_inventory import membership_facts

    try:
        own = own_room_chat(SimpleNamespace(task_id=task_id, task_metadata={}, current_chat_id=chat_id), drive_root)
    except (TypeError, ValueError):
        return None
    if own is None:
        return None
    if own == WEB_UI_CHAT_ID or own in membership_facts(drive_root).project_chat_ids:
        return str(own)
    presence = (own != HIDDEN_CHAT_ID and not is_a2a_chat_id(own)
                and presence_record_binding(task_room_record(drive_root, task_id)))
    return str(own) if presence else None


def _facts(record: Any) -> Dict[str, Any]:
    """The record by value, checked: what the writer must name before anything is read or written."""
    if not isinstance(record, Mapping):
        raise ValueError("a late review record is a mapping of immutable facts")
    facts = json.loads(json.dumps(dict(record), ensure_ascii=False, sort_keys=True, default=str))
    for key in ("surface", "task_id", "publication_id"):
        if not isinstance(facts.get(key), str) or not facts[key].strip():
            raise ValueError(f"{key} is a non-empty string")
    lines = facts.get("lines")
    if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines) or not "".join(lines).strip():
        raise ValueError("lines are the record writer's own settled words, at least one non-empty line")
    ref, read = facts.get("source_ref"), facts.get("source_read")
    if not isinstance(ref, dict) or not ref.get("sha256") or not ref.get("path"):
        raise ValueError("source_ref is the retained full source, with its path and sha256")
    if read is None and ref.get("kind") != "task_source":
        raise ValueError("a source other than a task source names its reader in source_read")
    if read is not None and not (isinstance(read, dict) and read.get("tool")):
        raise ValueError("source_read is a reader call: {tool, arguments}")
    subject = facts.get("subject") or {}
    if not isinstance(subject, dict) or any(isinstance(value, (dict, list)) for value in subject.values()):
        raise ValueError("subject maps names to scalar identity values")
    return {**facts, "subject": subject, "settled_at": str(facts.get("settled_at") or "")}


def _text(facts: Mapping[str, Any]) -> str:
    name = SURFACE_NAMES.get(facts["surface"], facts["surface"])
    settled = f", settled {facts['settled_at']}" if facts["settled_at"] else ""
    head = [f"Late {name} of task {facts['task_id']}{settled}."]
    subject = "; ".join(f"{key}={value}" for key, value in sorted(facts["subject"].items()) if value not in (None, ""))
    if subject:
        head.append(f"Reviewed: {subject}")
    body = [truncate_within_limit(line, LINE_LIMIT) for line in facts["lines"]]
    return truncate_within_limit("\n".join(head + body), TEXT_LIMIT)


def _mark(mark_id: str, room: str, facts: Mapping[str, Any]) -> Dict[str, Any]:
    ref = facts["source_ref"]
    target = {**ref, "task_id": str(ref.get("task_id") or facts["task_id"])}  # readable from any focus
    return {"id": mark_id, "kind": "mark", "room_id": room, "target_ref": target, "text": _text(facts),
            "author": {"kind": "host", "operation": LATE_REVIEW_OPERATION, "surface": facts["surface"],
                       "task_id": facts["task_id"]},
            "scope": "room", "quote": None, "visibility": "full",
            "review": {"surface": facts["surface"], "task_id": facts["task_id"],
                       "publication_id": facts["publication_id"], "settled_at": facts["settled_at"],
                       "subject": facts["subject"], "source_read": facts.get("source_read")}}


def publish_late_review(drive_root: Any, record: Mapping[str, Any]) -> LateReviewMark:
    """Keep one settled late review in view in its owning room (module docstring); never raises, never wakes.

    ``record``: ``surface``, ``task_id``, ``publication_id`` (one settled publication: the same id always
    names the same facts), ``chat_id`` (the owning chat, by value), ``lines`` (the writer's settled words:
    its verdict clause, then one line per seat of the whole roster, pending and unknown seats included),
    ``subject`` (scalar identity of what the reviewers read), ``source_ref`` (the retained full source),
    optional ``source_read`` and ``settled_at``.
    """
    try:
        facts = _facts(record)
    except ValueError as exc:
        return LateReviewMark("invalid", detail=str(exc))
    mark_id = late_review_mark_id(facts["surface"], facts["task_id"], facts["publication_id"])
    try:
        outcome = _publish(pathlib.Path(drive_root), mark_id, facts)
    except Exception as exc:  # a store that cannot serve (StoreError, …): the publication stays owed
        outcome = LateReviewMark("failed", mark_id, "", f"{type(exc).__name__}: {exc}")
    if outcome.status not in {"saved", "unchanged"}:
        log.warning("late review %s of task %s kept no mark in view: %s (%s)", facts["publication_id"],
                    facts["task_id"], outcome.status, outcome.detail)
    _owed(drive_root, mark_id, {"record": facts} if outcome.status == "failed" else None)
    return outcome


def owe_late_review(drive_root: Any, record: Mapping[str, Any]) -> LateReviewMark:
    """Keep one late review owed until its author's task ended, then publish it (``replay_owed_late_reviews``).

    For feedback a running author may still read itself: reading it is not a release, so the mark
    follows the task's end whatever the author read. An already ended task publishes now.
    """
    try:
        facts = _facts(record)
    except ValueError as exc:
        return LateReviewMark("invalid", detail=str(exc))
    mark_id = late_review_mark_id(facts["surface"], facts["task_id"], facts["publication_id"])
    if ended_task_row(drive_root, facts["task_id"]) is not None:
        return publish_late_review(drive_root, facts)
    return LateReviewMark("owed" if _owed(drive_root, mark_id, {"record": facts, "after_end": True}) else "failed",
                          mark_id)


def owe_after_end(drive_root: Any, duty_id: str, duty: Mapping[str, Any]) -> bool:
    """Persist one producer duty (``left_review``: immutable facts naming ``task_id``) that only its author's
    ended task can discharge; the replay hands it back to its producer. False when the store refused (the
    caller's own path still holds the duty)."""
    return _owed(drive_root, duty_id, {**json.loads(json.dumps(dict(duty), default=str)), "after_end": True})


def update_wait_cursors(drive_root: Any, duty_id: str, waiting: Mapping[str, Any]) -> None:
    """Refresh an existing duty's delivery facts; never discharge or recreate it."""
    from ouroboros import store

    with store.open_store(drive_root, write=True) as txn:
        row = txn.get_row(OWED_SET, duty_id)
        if row is None or not isinstance(row.get("left_review"), dict):
            return
        row["left_review"].update({key: waiting[key] for key in
                                  ("selection", "seen_helpers", "seen_outcomes", "review_reference") if key in waiting})
        txn.put_row(OWED_SET, duty_id, row)


def replay_owed_late_reviews(drive_root: Any, *, stop: Any = None) -> Dict[str, int]:
    """The maintenance pass over owed rows: each duty whose author ended is published or handed back to its
    producer; a row that names nothing it can owe is retired. Ended plan asks belong to the
    separate maintenance child job, which uses their pending seats as debt."""
    from ouroboros import store  # D15->D17 is lazy-only

    report = {"settled": 0, "waiting": 0, "retired": 0, "failed": 0}
    try:
        with store.open_store(drive_root, write=False) as txn:
            rows = txn.set_rows(OWED_SET)
    except Exception:
        log.warning("owed late reviews are unreadable; they stay owed", exc_info=True)
        return report
    for duty_id, row in rows.items():
        if stop is not None and stop():
            break
        duty = next((row[key] for key in ("record", "left_review") if isinstance(row.get(key), dict)), None)
        if duty is None:
            report["retired"] += _owed(drive_root, duty_id, None)
        elif row.get("after_end") and ended_task_row(drive_root, str(duty.get("task_id") or "")) is None:
            report["waiting"] += 1
        else:
            report[_settle_owed(drive_root, duty_id, row)] += 1
    return report


def _collect_ended_plan_asks(drive_root: Any, stop: Any, *, candidates=None, on_examined=None) -> int:
    """The crash window of a plan reply: a seat of an ended task whose producer completed while nobody was
    listening settles at $0 through the organ's own collector (``plan_ask.collect_pending``), whose settlement
    keeps the reply in its room. The ask's pending seat is the duty, so nothing else is written for it; a seat
    with no retained outcome stays pending and is never cleaned. Only contention keeps an examination
    owed; deterministic failures rest until the task changes, without discharging any pending seat."""
    from ouroboros.maintenance import _contention  # D15->D17 is lazy-only
    from ouroboros import plan_record  # D15->D06 is lazy-only
    from ouroboros.task_store import plan_collection_candidates  # D15->D17 is lazy-only
    from ouroboros.tools.plan_ask import collect_pending  # D15->D06 is lazy-only

    root, settled = pathlib.Path(drive_root), 0
    try:
        for task_id in plan_collection_candidates(root, stop=stop) if candidates is None else candidates:
            if stop is not None and stop():
                break
            try:
                record = plan_record.load_plan_record(root, task_id)
                pending = {(ask.get("ask_id"), seat.get("seat_id")) for ask, seat in plan_record.pending_participants(record)}
                if pending and ended_task_row(root, task_id, strict=True) is not None:
                    after = collect_pending(None, record, state_root=root, task_id=task_id)
                    settled += sum(1 for ask in after.get("asks") or [] for seat in ask.get("participants") or []
                                   if (ask.get("ask_id"), seat.get("seat_id")) in pending and seat.get("status") != "pending")
            except Exception as exc:
                log.warning("plan record of %s: its pending seats stay pending", task_id, exc_info=True)
                if _contention(exc):
                    continue
            if on_examined is not None:
                on_examined(task_id)
    except Exception:
        log.warning("plan records are unreadable; their pending seats stay pending", exc_info=True)
    return settled


def _settle_owed(drive_root: Any, duty_id: str, row: Mapping[str, Any]) -> str:
    """One owed row whose author ended: ``settled``, ``waiting`` (its producer still owns it) or ``failed``."""
    try:
        if isinstance(row.get("record"), dict):
            status = publish_late_review(drive_root, row["record"]).status
        else:
            from ouroboros.review_pause import settle_left_review  # D15->D06 is lazy-only

            status = settle_left_review(drive_root, row)
    except Exception:
        log.warning("owed late review %s could not be settled; it stays owed", duty_id, exc_info=True)
        return "failed"
    if status == "waiting":
        return "waiting"
    if status == "failed":
        return "failed"
    _owed(drive_root, duty_id, None)  # a placed, unchanged, released or unplaceable outcome discharges it
    return "settled"


def _owed(drive_root: Any, mark_id: str, row: Optional[Dict[str, Any]]) -> bool:
    """Write (``row``) or discharge (``None``) one owed publication; False when the store refused."""
    from ouroboros import store  # D15->D17 is lazy-only

    try:
        if row is None:
            with store.open_store(drive_root, write=False) as txn:
                if txn.get_row(OWED_SET, mark_id) is None:
                    return True
        with store.open_store(drive_root, write=True) as txn:
            if row is None:
                txn.delete_row(OWED_SET, mark_id)
            else:
                txn.put_row(OWED_SET, mark_id, row)
        return True
    except Exception:
        log.warning("late review %s: its owed row could not be %s", mark_id,
                    "kept; nothing durable owes it now" if row is not None else "discharged", exc_info=True)
        return False


def ended_task_row(drive_root: Any, task_id: str, *, strict: bool = False) -> Optional[Dict[str, Any]]:
    """The canonical row of a task whose terminal status is recorded (a latch), else ``None``; unreadable is
    not ended. Strict readers propagate failures for the caller's retry policy. A writer reads its owning chat here."""
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result  # D15->D17 is lazy-only

    try:
        row = load_task_result(drive_root, task_id) or {}
    except Exception:
        if strict:
            raise
        log.debug("task %s status unreadable; an owed late review keeps waiting", task_id, exc_info=True)
        return None
    return row if str(row.get("status") or "") in _TRULY_TERMINAL_STATUSES else None


def _publish(root: pathlib.Path, mark_id: str, facts: Mapping[str, Any]) -> LateReviewMark:
    try:
        room = owning_room(root, facts["task_id"], facts.get("chat_id"))
    except (OSError, ValueError, sqlite3.Error) as exc:  # an unreadable binding or Project registry
        return LateReviewMark("failed", mark_id, "", f"owning room unreadable: {type(exc).__name__}: {exc}")
    if room is None:
        return LateReviewMark("not_activated", mark_id, "", (
            f"the room owning chat {facts.get('chat_id')!r} of task {facts['task_id']} is not proven; no mark is "
            f"placed and the full source stays retained at {facts['source_ref']['path']}"))
    try:
        store = ChronicleStore(root)
        receipt = store.ensure_activated()
        if not isinstance(receipt, dict) or receipt.get("kind") != "activation":
            kind = str((receipt or {}).get("kind") or "import_failed")
            return LateReviewMark("failed" if kind == "import_pending" else "memory_not_activated", mark_id, room,
                                  f"{kind}: {(receipt or {}).get('reason') or 'no activation receipt'}")
        result = store.publish([_mark(mark_id, room, facts)])
    except (OSError, TimeoutError, sqlite3.Error) as exc:
        return LateReviewMark("failed", mark_id, room, f"{type(exc).__name__}: {exc}")
    except ValueError as exc:  # the store's own refusal to open its journal
        return LateReviewMark("memory_not_activated", mark_id, room, f"{type(exc).__name__}: {exc}")
    if result.ok:
        return LateReviewMark(result.reason, mark_id, room)
    return LateReviewMark("conflict" if result.detail.startswith("identity collision") else "invalid", mark_id, room,
                          f"{result.reason}: {result.detail}")
