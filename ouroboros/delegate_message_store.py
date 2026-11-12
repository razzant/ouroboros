"""Write-ahead live-message frames and the starter's per-run message journal.

The indexed current.sqlite set for one run owns both retry bytes and journal
rows. No event-log replay, mailbox write, wake predicate or engine cursor is
involved. Rows are retained with custody, including ambiguous deliveries.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from ouroboros import store
from ouroboros.delegate_custody import custody_root
from ouroboros.install_identity import require_owned_root
from ouroboros.utils import utc_now_iso


def _set(run_id: str) -> str:
    return "delegate_messages-" + hashlib.sha256(run_id.encode("utf-8")).hexdigest()


def _wire(frame: dict, text: str) -> str:
    # One JSON value: text cannot close the host frame or replace its fields.
    return json.dumps({"kind": "task_message", "frame": frame, "text": text},
                      ensure_ascii=False, sort_keys=True)


# A placed row's last frame: these outcomes end the row for that run; these may already be delivered (replayed
# under their id and bytes, never re-derived); any other (``not_active``, ``not_found``) takes a new frame.
PLACEMENT_ENDED = frozenset({"delivered", "accepted", "rejected", "unsupported"})
PLACEMENT_REPLAYS = frozenset({"delivery_unknown", "prepared"})


def _sender(ctx: Any, root: Any) -> tuple:
    """``(sender id, its record, its metadata, its configured binding)``, read before the write transaction."""
    sender = str(getattr(ctx, "task_id", "") or "")
    from ouroboros.task_status import load_effective_task_result

    own = load_effective_task_result(root, sender) or {}
    meta = getattr(ctx, "task_metadata", {}) or {}
    binding = meta.get("configured_subagent") or own.get("configured_subagent") or {}
    return sender, own, meta, binding if isinstance(binding, dict) else {}


def _new_row(ctx: Any, entry: Any, relation: str, text: str, reply_to: dict | None, owner_words: dict | None,
             facts: tuple) -> dict:
    sender, own, meta, binding = facts
    frame = {
        "message_id": uuid.uuid4().hex, "source_task_id": sender,
        "root_task_id": entry.root_task_id or own.get("root_task_id") or meta.get("root_task_id") or entry.task_id,
        "sender_role": str(own.get("delegation_role") or meta.get("delegation_role")
                           or ("subagent" if own.get("parent_task_id") or meta.get("parent_task_id")
                               else "root" if own else "unknown")),
        "row": str(binding.get("selected_subagent_id") or meta.get("subagent_id")
                   or meta.get("selected_subagent_id") or ""),
        "model": str(getattr(ctx, "active_model_override", "") or getattr(ctx, "active_model", "")
                     or getattr(ctx, "task_model_override", "") or meta.get("model") or ""),
        "ts": utc_now_iso(), "relation": relation,
    }
    if reply_to is not None:
        frame["reply_to"] = reply_to
    if owner_words is not None:
        frame["owner_words"] = owner_words
    wire = _wire(frame, text)
    return {"schema": 1, "run_id": entry.run_id, "starter_task_id": entry.task_id,
            "frame": frame, "text": text, "wire_text": wire,
            "wire_sha256": hashlib.sha256(wire.encode("utf-8")).hexdigest(),
            "outcome": "prepared", "note": "Prepared before send; delivery is not yet proven."}


def prepare(ctx: Any, entry: Any, relation: str, text: str, message_id: str,
            reply_to: dict | None, owner_words: dict | None = None) -> dict:
    """Commit before POST; an explicit key must replay an existing exact row.

    ``owner_words`` marks a frame that places my human's row (``place_frame``): the owner row's
    address and chat source, so the frame rows of a run answer which rows of my human's reached
    it and how; the same wire kind, one field more."""
    root = custody_root(ctx)
    require_owned_root(root)
    facts = _sender(ctx, root)
    with store.open_store(root, write=True) as txn:
        if message_id:
            row = txn.get_row(_set(entry.run_id), message_id)
            if row is None:
                raise ValueError("message_identity_unknown")
            frame = row.get("frame", {})
            if (frame.get("message_id") != message_id or row.get("run_id") != entry.run_id
                    or frame.get("source_task_id") != facts[0] or row.get("text") != text
                    or frame.get("reply_to") != reply_to or frame.get("owner_words") != owner_words
                    or row.get("starter_task_id") != entry.task_id
                    or row.get("wire_text") != _wire(frame, text)
                    or hashlib.sha256(row["wire_text"].encode("utf-8")).hexdigest() != row.get("wire_sha256")):
                raise ValueError("message_frame_conflict")
            return row
        row = _new_row(ctx, entry, relation, text, reply_to, owner_words, facts)
        txn.put_row(_set(entry.run_id), row["frame"]["message_id"], row)
    return row


def place_frame(ctx: Any, entry: Any, text: str, owner_words: dict, *, host: bool = False) -> tuple:
    """``(row, prior)``: the frame my human's row ``owner_words["address"]`` takes into ``entry``'s run as its
    starter's, chosen and written in ONE transaction, so two placers (the host at receipt, the starter's round top)
    never send one row twice: the run's last frame of that row when it may already be delivered
    (``PLACEMENT_REPLAYS``: its id and bytes again), ``None`` when that frame ended the row (``PLACEMENT_ENDED``),
    else a new frame. ``prior``: that last frame's outcome ("" for none). ``host``: a frame the host sends at receipt
    is ``untold`` until its starter's context holds its fact (``mark_told``)."""
    root = custody_root(ctx)
    require_owned_root(root)
    facts = _sender(ctx, root)
    with store.open_store(root, write=True) as txn:
        rows = [row for row in txn.set_rows(_set(entry.run_id)).values()
                if isinstance(words := row.get("frame", {}).get("owner_words"), dict)
                and words.get("address") == owner_words["address"]]
        last = max(rows, key=lambda row: str(row["frame"].get("ts") or ""), default=None)
        prior = str((last or {}).get("outcome") or "")
        if prior in PLACEMENT_ENDED:
            return None, prior
        replay = last is not None and prior in PLACEMENT_REPLAYS
        row = last if replay else _new_row(ctx, entry, "starter", text, None, owner_words, facts)
        if host:
            row["untold"] = True
        if host or not replay:
            txn.put_row(_set(entry.run_id), row["frame"]["message_id"], row)
    return row, prior


def mark_told(ctx: Any, run_id: str, message_ids: list) -> None:
    """The starter's context now holds the fact line of these host-placed frames: ``untold`` is cleared."""
    with store.open_store(custody_root(ctx), write=True) as txn:
        for message_id in message_ids:
            row = txn.get_row(_set(run_id), message_id)
            if row is not None and row.pop("untold", None) is not None:
                txn.put_row(_set(run_id), message_id, row)


def record_outcome(ctx: Any, run_id: str, message_id: str, outcome: str, reason: str) -> None:
    with store.open_store(custody_root(ctx), write=True) as txn:
        row = txn.get_row(_set(run_id), message_id)
        if row is not None:
            row.update(outcome=outcome, reason=reason)
            row.pop("note", None)
            txn.put_row(_set(run_id), message_id, row)


def placed_words(ctx: Any, run_id: str) -> list:
    """The frames that placed my human's rows into ``run_id`` (those with ``owner_words``), oldest first:
    the reader of which rows of my human's a run has heard, and with which outcome."""
    with store.open_store(custody_root(ctx), write=False) as txn:
        rows = txn.set_rows(_set(run_id))
    return sorted((row for row in rows.values() if isinstance(row.get("frame", {}).get("owner_words"), dict)),
                  key=lambda row: str(row["frame"].get("ts") or ""))


def journal(ctx: Any, run_id: str) -> list:
    """The starter's own journal only; a message address grants no result access."""
    sender = str(getattr(ctx, "task_id", "") or "")
    with store.open_store(custody_root(ctx), write=False) as txn:
        rows = txn.set_rows(_set(run_id))
    return [{key: value for key, value in row.items() if key not in {"wire_text", "wire_sha256"}}
            for row in rows.values() if row.get("starter_task_id") == sender
            and row.get("frame", {}).get("source_task_id") != sender]


def starter(root: Any, run: Any) -> Any:
    """The run's starter as a placer context (the host's, the body's): its frames name it."""
    from types import SimpleNamespace

    from ouroboros.task_status import load_effective_task_result

    meta = (load_effective_task_result(root, run.task_id) or {}).get("metadata")
    return SimpleNamespace(drive_root=root, task_id=run.task_id, task_metadata={
        **(meta if isinstance(meta, dict) else {}), "budget_drive_root": str(root)})


def pending_frames(ctx: Any, run_id: str) -> list:
    """The last frame of each of my human's rows in ``run_id`` that may still need sending (``PLACEMENT_REPLAYS``)."""
    last = {str(row["frame"]["owner_words"].get("address") or ""): row for row in placed_words(ctx, run_id)}
    return [row for row in last.values() if row.get("outcome") in PLACEMENT_REPLAYS]


def pending_runs(root: Any, run_ids: Any) -> set:
    """Runs owing owner frames, in one indexed read of the followed sets, without loading starters or journals.

    The last frame per owner address is authoritative, as in ``pending_frames``; process-local hints are not.
    """
    names = {_set(run_id): run_id for run_id in run_ids}
    if not names:
        return set()
    with store.open_store(root, write=False) as txn:
        return {names[row["set_name"]] for row in txn.conn.execute(
            "SELECT DISTINCT set_name FROM (SELECT set_name, json_extract(facts,'$.outcome') AS outcome, "
            "row_number() OVER (PARTITION BY set_name, json_extract(facts,'$.frame.owner_words.address') "
            "ORDER BY json_extract(facts,'$.frame.ts') DESC, rowid DESC) AS latest FROM obligations "
            "WHERE set_name IN (SELECT value FROM json_each(?)) "
            "AND json_type(facts,'$.frame.owner_words')='object') "
            "WHERE latest=1 AND outcome IN (?,?)", (json.dumps(list(names)), *sorted(PLACEMENT_REPLAYS)))}


def replay_frames(root: Any, run: Any) -> bool:
    """The body's send of my human's frames still pending for one run it observes: each replayed under its id
    and bytes in the placement's own session (``delegate_interactions._place_into_run``, with no capability read),
    its outcome recorded as the starter's round-top catch-up records it, idempotent with it. True when none stays
    pending. Never raises."""
    import logging

    from ouroboros.delegate_interactions import _place_into_run

    try:
        ctx = starter(root, run)
        if pending := pending_frames(ctx, run.run_id):
            _place_into_run(ctx, run, [], replays=pending)
        return not pending_frames(ctx, run.run_id)
    except Exception:  # noqa: BLE001 — what it could not send waits for its next answered pass or the round top
        logging.getLogger(__name__).warning("owner words replay into run %s waits", run.run_id, exc_info=True)
        return False
