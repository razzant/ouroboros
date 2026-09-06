"""Event-only supervising wait for configured session nannies."""

from __future__ import annotations

import json
import pathlib
import time
import uuid
from typing import Any, Callable, Optional

from ouroboros import delegate_custody as custody
from ouroboros.owner_mailbox import (
    KIND_FINALIZE_NOW,
    KIND_HURRY,
    KIND_TASK_MESSAGE,
    drain_owner_entries,
)
from ouroboros.utils import atomic_write_json, utc_now_iso

_TICK_SEC = 3
_QUIET_STATUSES = {"progress", "no_progress"}
_LOOP_CONTROL_KINDS = {KIND_FINALIZE_NOW, KIND_HURRY}
_MAX_COORDINATION_SEEN = 256


def _attempt_key(ctx: Any) -> str:
    return str(getattr(ctx, "task_attempt", None) or 1)


def _state_path(ctx: Any) -> pathlib.Path:
    task_id = str(getattr(ctx, "task_id", "") or "")
    return custody.custody_root(ctx) / "state" / "delegate_supervision" / f"{task_id}.json"


def _load_state(ctx: Any, run_id: str) -> dict[str, Any]:
    path = _state_path(ctx)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict) or (
        str(run_id) and str(data.get("run_id") or "") != str(run_id)
    ):
        # The unknown-provider hold is CROSS-RUN durable state (nanny-leaf): a
        # state rebuild for a newer/first run id must not erase the latch —
        # supervised_wait's own entry persisted the reset, so a worker crash
        # during the (hours-long) wait lost the hold and the recovered
        # successor dispatched the unknown transcript again (final-pair F2).
        hold = data.get("unknown_provider_hold") if isinstance(data, dict) else None
        data = {"schema": 1, "run_id": str(run_id), "journal_cursor": 0}
        if isinstance(hold, dict):
            data["unknown_provider_hold"] = hold
    return data


def _save_state(ctx: Any, state: dict[str, Any]) -> None:
    path = _state_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now_iso()
    atomic_write_json(path, state)


def _emit(ctx: Any, kind: str, payload: dict[str, Any]) -> bool:
    return custody.emit(custody.custody_root(ctx), kind, {
        "task_id": str(getattr(ctx, "task_id", "") or ""), **payload,
    })


def _payload(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"status": "fault", "detail": raw}
    except (TypeError, ValueError):
        return {"status": "fault", "detail": str(raw or "")}


def _coordination_root_id(ctx: Any) -> str:
    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(
        metadata.get("root_task_id")
        or getattr(ctx, "root_task_id", "")
        or getattr(ctx, "task_id", "")
        or ""
    )


def _parent_intent_fact(ctx: Any) -> dict[str, Any]:
    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    contract = (
        getattr(ctx, "task_contract", None)
        if isinstance(getattr(ctx, "task_contract", None), dict)
        else metadata.get("task_contract")
        if isinstance(metadata.get("task_contract"), dict)
        else {}
    )
    budget = contract.get("delegation_budget") if isinstance(contract, dict) else {}
    note = str((budget or {}).get("intent_note") or "").strip()
    return {
        "state": "present" if note else "absent",
        "authority": "parent_authored_advisory",
        "text": note,
    }


def _time_fact(ctx: Any) -> dict[str, Any]:
    """The task's deadline window, OBSERVED: this fact writes nothing, so it
    reads through the observation variants of both readers — never the emitting
    ``resolve_budget_profile`` (deprecation row) or the latching
    ``build_budget_snapshot`` (fallback anchor).

    Disclosed consequence: a metadata-poor task (no ``created_at``/
    ``started_at`` and no anchor latched yet) reports ``state: "not_set"``
    until a path that OWNS a mutation — the acceptance launch — latches the
    anchor. A poll answering the same question twice never changes its own
    next answer."""
    try:
        from ouroboros.task_pacing import observe_budget_profile, observe_budget_snapshot

        snapshot = observe_budget_snapshot(ctx, profile=observe_budget_profile(ctx))
        if not snapshot.has_deadline:
            return {
                "state": "not_set", "remaining_sec": None,
                "reserve_sec": None, "inside_reserve": None, "expired": None,
            }
        return {
            "state": "known",
            "remaining_sec": round(max(0.0, snapshot.remaining_sec), 3),
            "reserve_sec": round(max(0.0, snapshot.reserve_sec), 3),
            "inside_reserve": bool(snapshot.inside_reserve),
            "expired": bool(snapshot.remaining_sec <= 0),
        }
    except Exception as exc:
        return {
            "state": "unknown", "remaining_sec": None,
            "reserve_sec": None, "inside_reserve": None, "expired": None,
            "reason": type(exc).__name__,
        }


def _settled_spend_fact(ctx: Any, root_task_id: str) -> dict[str, Any]:
    """The tree's ledger-accounted spend, read through the canonical locked reader
    (``usage_accounting.usage_breakdown``) in every state — an absent ledger is the reader's own
    known-zero. The fact writes nothing of its own; it inherits the reader's bounded maintenance —
    today: the torn-tail quarantine after a SINGLE crash mid-append (a crash inside that repair, a
    torn quarantine sink, is a known residual, issue #586), the empty
    ``state/`` lock directory on a never-initialized root, and removal of a stale
    ``usage_attempts.lock`` past the 90 s window (``usage_ledger._locked`` →
    ``platform_layer.acquire_exclusive_file_lock``, stale-age unlink) — each pinned by a regression."""
    try:
        from ouroboros.usage_accounting import usage_breakdown

        root = custody.custody_root(ctx)
        projection = usage_breakdown(root, root_task_id=root_task_id)
        integrity = bool(projection.get("integrity_degraded"))
        unknown = int(projection.get("unknown_unmetered") or 0)
        return {
            "state": "partial" if integrity or unknown else "known",
            "settled_usd": float(projection.get("settled_usd") or 0.0),
            "accounted_usd": float(projection.get("accounted_usd") or 0.0),
            "cost_final": bool(projection.get("cost_final")),
            "unknown_unmetered": unknown,
            "integrity_degraded": integrity,
        }
    except Exception as exc:
        return {
            "state": "unknown", "settled_usd": None, "accounted_usd": None,
            "cost_final": None, "unknown_unmetered": None,
            "integrity_degraded": None, "reason": type(exc).__name__,
        }


def _active_descendants_fact(ctx: Any) -> dict[str, Any]:
    """Strict host-visible ancestry walk; vendor-internal children stay opaque."""

    try:
        from ouroboros.task_results import load_task_result
        from ouroboros.task_status import (
            SETTLED_STATUSES,
            _load_queue_snapshot,
            _merge_queue_status,
            _snapshot_is_stale,
        )

        root = custody.custody_root(ctx)
        snapshot = _load_queue_snapshot(pathlib.Path(root))
        if snapshot.get("_snapshot_missing") or snapshot.get("_snapshot_invalid"):
            raise ValueError("queue_snapshot_unavailable")
        if _snapshot_is_stale(snapshot):
            raise ValueError("queue_snapshot_stale")
        rows: dict[str, dict[str, Any]] = {}
        for group, status in (("pending", "scheduled"), ("running", "running")):
            for item in snapshot.get(group) or []:
                if not isinstance(item, dict):
                    raise ValueError("queue_snapshot_row_invalid")
                task = item.get("task") if isinstance(item.get("task"), dict) else item
                task_id = str(item.get("id") or task.get("id") or "")
                if not task_id:
                    raise ValueError("queue_snapshot_task_id_missing")
                rows[task_id] = {
                    **task,
                    "task_id": task_id,
                    "_queue_status": status,
                }
        ancestor = str(getattr(ctx, "task_id", "") or "")
        lineage_cache = dict(rows)

        def _belongs(row: dict[str, Any]) -> bool:
            parent_id = str(row.get("parent_task_id") or "")
            seen = {str(row.get("task_id") or row.get("id") or "")}
            while parent_id:
                if parent_id == ancestor:
                    return True
                if parent_id in seen:
                    raise ValueError("task_lineage_cycle")
                seen.add(parent_id)
                parent = lineage_cache.get(parent_id)
                if parent is None:
                    parent = load_task_result(root, parent_id, strict=True)
                    if not isinstance(parent, dict):
                        raise ValueError("task_lineage_unavailable")
                    lineage_cache[parent_id] = parent
                parent_id = str(parent.get("parent_task_id") or "")
            return False

        active: list[dict[str, Any]] = []
        for task_id, row in rows.items():
            if task_id == ancestor or not _belongs(row):
                continue
            # Only exact rows proven to be descendants can poison this fact.
            # Unrelated active tasks are outside the nanny's authority surface.
            durable = load_task_result(root, task_id, strict=True) or {}
            merged = {**durable, **row}
            merged["status"] = _merge_queue_status(
                str(durable.get("status") or ""),
                str(row.get("_queue_status") or ""),
            )
            merged.pop("_queue_status", None)
            if str(merged.get("status") or "").strip().lower() not in SETTLED_STATUSES:
                active.append(merged)
        by_status: dict[str, int] = {}
        for row in active:
            status = str(row.get("status") or "unknown").strip().lower() or "unknown"
            by_status[status] = by_status.get(status, 0) + 1
        return {
            "state": "known", "count": len(active),
            "by_status": dict(sorted(by_status.items())),
            "scope": "host_visible_descendants",
            "vendor_internal": "opaque_not_counted",
        }
    except Exception as exc:
        return {
            "state": "unknown", "count": None, "by_status": {},
            "scope": "host_visible_descendants",
            "vendor_internal": "opaque_not_counted",
            "reason": type(exc).__name__,
        }


def coordination_live_context(ctx: Any) -> dict[str, Any]:
    """One LLM-first planning snapshot for startup and meaningful nanny wakes.

    Polling writes nothing of its own; it inherits the canonical usage-ledger reader's bounded
    maintenance — today: the torn-tail quarantine after a SINGLE crash mid-append
    (``usage_ledger._read_records_locked``, identical for every reader; a crash inside that repair
    is a known residual, issue #586), the empty ``state/`` lock directory
    on a never-initialized root, and removal of a stale ``usage_attempts.lock`` past the 90 s window
    (``usage_ledger._locked`` → ``platform_layer.acquire_exclusive_file_lock``, stale-age unlink) —
    each pinned by a regression; the settled-spend fact reads the ledger through that reader.
    """

    root_task_id = _coordination_root_id(ctx)
    try:
        from ouroboros.task_pacing import project_task_acceptance_review_capacity

        review_capacity = project_task_acceptance_review_capacity(ctx)
    except Exception as exc:
        review_capacity = {
            "state": "unknown", "reason": type(exc).__name__,
            "root_task_id": root_task_id, "cap_cycles": None,
            "claimed_cycles": None, "remaining_cycles": None,
            "binding_seen": False, "dedupe": "task_acceptance_binding_sha256",
        }
    return {
        "observed_at": utc_now_iso(),
        "root_task_id": root_task_id,
        "parent_intent": _parent_intent_fact(ctx),
        "time": _time_fact(ctx),
        "settled_spend": _settled_spend_fact(ctx, root_task_id),
        "active_descendants": _active_descendants_fact(ctx),
        "review_capacity": review_capacity,
    }


def _coordination_cursor(state: dict[str, Any]) -> dict[str, Any]:
    cursor = state.get("coordination_cursor")
    if not isinstance(cursor, dict):
        cursor = {}
    seen = cursor.get("attention_seen")
    return {
        "attention_after_ts": str(cursor.get("attention_after_ts") or ""),
        "attention_seen": [str(item) for item in seen if str(item)]
        if isinstance(seen, list) else [],
        "children": dict(cursor.get("children"))
        if isinstance(cursor.get("children"), dict) else {},
    }


def _coordination_row_id(row: dict[str, Any]) -> str:
    from ouroboros.task_tree_ledger import tree_ledger_row_id

    return tree_ledger_row_id(row)


def _direct_children(ctx: Any) -> dict[str, dict[str, Any]]:
    """Read the existing durable direct-child projection for this nanny."""
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not task_id:
        return {}
    try:
        from ouroboros.task_status import find_child_tasks

        children = find_child_tasks(
            custody.custody_root(ctx), parent_task_id=task_id,
            root_task_id="", scope="direct", materialize_artifacts=True,
        )
    except Exception:
        return {}
    return {
        str(child.get("task_id") or child.get("id") or ""): dict(child)
        for child in children
        if isinstance(child, dict)
        and str(child.get("task_id") or child.get("id") or "")
    }


def _child_marker(child: dict[str, Any]) -> dict[str, str]:
    stored_hash = str(
        child.get("child_result_sha256") or child.get("result_sha256") or ""
    )
    try:
        from ouroboros.tools.join_ledger import _child_result_sha256

        result_hash = _child_result_sha256(child)
    except Exception:
        result_hash = stored_hash
    return {
        "status": str(child.get("status") or "").strip().lower(),
        "updated_at": str(child.get("updated_at") or child.get("ts") or ""),
        "result_sha256": result_hash,
    }


def _coordination_wakes(
    ctx: Any, state: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect direct-child beacons and terminal transitions on one cursor."""
    cursor = _coordination_cursor(state)
    children = _direct_children(ctx)
    markers = {
        child_id: _child_marker(child) for child_id, child in children.items()
    }
    metadata = getattr(ctx, "task_metadata", {})
    root_id = str(metadata.get("root_task_id") or "").strip() if isinstance(metadata, dict) else ""
    root_id = root_id or str(getattr(ctx, "task_id", "") or "").strip()
    attention: list[dict[str, Any]] = []
    seen = {str(item) for item in cursor.get("attention_seen", []) if str(item)}
    try:
        from ouroboros.task_tree_ledger import tree_ledger_attention_after

        attention = tree_ledger_attention_after(
            root_id,
            str(cursor.get("attention_after_ts") or ""),
            task_ids=set(children),
            seen_ids=seen,
            data_root=custody.custody_root(ctx),
        )
    except Exception:
        attention = []
    new_attention = [row for row in attention if _coordination_row_id(row) not in seen]
    terminal_events: list[dict[str, Any]] = []
    prior_children = cursor.get("children") if isinstance(cursor.get("children"), dict) else {}
    from ouroboros.task_status import SETTLED_STATUSES

    for child_id, marker in markers.items():
        previous = prior_children.get(child_id) if isinstance(prior_children.get(child_id), dict) else {}
        status = marker["status"]
        if (
            status in SETTLED_STATUSES
            and str(previous.get("status") or "") not in SETTLED_STATUSES
        ):
            terminal_events.append({
                "type": "child_terminal",
                "child_task_id": child_id,
                "status": status,
                "updated_at": marker["updated_at"],
                "result_sha256": marker["result_sha256"],
            })
    events = [
        {"type": "child_attention_beacon", "beacon": row}
        for row in new_attention
    ] + terminal_events
    next_cursor = {
        "attention_after_ts": str(cursor.get("attention_after_ts") or ""),
        "attention_seen": list(cursor.get("attention_seen", [])),
        "children": {**prior_children, **markers},
    }
    if new_attention:
        timestamps = [
            str(row.get("ts") or "") for row in new_attention if str(row.get("ts") or "")
        ]
        if timestamps:
            next_cursor["attention_after_ts"] = max(
                str(next_cursor["attention_after_ts"] or ""), max(timestamps)
            )
        next_cursor["attention_seen"] = list(dict.fromkeys([
            *next_cursor["attention_seen"],
            *(_coordination_row_id(row) for row in new_attention),
        ]))[-_MAX_COORDINATION_SEEN:]
    return events, next_cursor


def _addressed_wakes(ctx: Any, state: dict[str, Any]) -> list[dict[str, Any]]:
    attempt = _attempt_key(ctx)
    seen = set()
    if str(state.get("mailbox_acknowledged_attempt_key") or "") == attempt:
        seen.update(
            str(item) for item in (state.get("mailbox_acknowledged_ids") or [])
            if str(item)
        )
    seen.update(
        str(item) for item in (getattr(ctx, "_loop_mailbox_seen_ids", set()) or set())
        if str(item)
    )
    entries = drain_owner_entries(
        pathlib.Path(getattr(ctx, "drive_root", custody.custody_root(ctx))),
        str(getattr(ctx, "task_id", "") or ""),
        seen_ids=seen,
        attempt_key=attempt,
    )
    return [
        {
            "type": "addressed_message",
            "msg_id": str(entry.get("msg_id") or ""),
            "kind": str(entry.get("kind") or "owner_text"),
            "provenance": str(entry.get("provenance") or "owner"),
            "source_task_id": str(entry.get("source_task_id") or ""),
            "relayed_from_task_id": str(entry.get("relayed_from_task_id") or ""),
            "text": str(entry.get("text") or ""),
            "ts": str(entry.get("ts") or ""),
        }
        for entry in entries
    ]


def _interaction_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        identity = str(value.get("interaction_id") or value.get("interactionId") or "")
        if identity:
            found.add(identity)
        for child in value.values():
            found.update(_interaction_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_interaction_ids(child))
    return found


def _wake_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        identity = str(value.get("supervision_wake_id") or "")
        if identity:
            found.add(identity)
        for child in value.values():
            found.update(_wake_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_wake_ids(child))
    return found


def _wake_event_summary(event: Any) -> dict[str, Any]:
    """Compact one wake event while retaining its routing/exact-evidence facts."""

    if not isinstance(event, dict):
        return {"type": "unknown"}
    if str(event.get("type") or "") == "child_attention_beacon":
        beacon = event.get("beacon") if isinstance(event.get("beacon"), dict) else {}
        text = str(beacon.get("text") or "")
        summary: dict[str, Any] = {
            "type": "child_attention_beacon",
            "beacon": {
                key: beacon.get(key)
                for key in ("ts", "kind", "task_id", "role", "needs_parent_attention")
                if beacon.get(key) not in (None, "")
            },
        }
        summary["beacon"]["text"] = text[:600]
        if len(text) > 600:
            summary["beacon"]["text_omitted_chars"] = len(text) - 600
        payload = beacon.get("payload") if isinstance(beacon.get("payload"), dict) else {}
        if str(beacon.get("kind") or "") == "review_requested":
            evidence_ref = str(payload.get("evidence_ref") or "")
            summary["beacon"]["payload"] = {
                "evidence_ref": evidence_ref[:600],
                "evidence_sha256": str(payload.get("evidence_sha256") or ""),
            }
            if len(evidence_ref) > 600:
                summary["beacon"]["payload"]["evidence_ref_omitted_chars"] = (
                    len(evidence_ref) - 600
                )
        elif payload:
            summary["beacon"]["payload_available_in_full_source"] = True
        return summary
    summary = {
        key: event.get(key)
        for key in (
            "type", "kind", "msg_id", "source_task_id", "child_task_id",
            "status", "updated_at", "result_sha256",
        )
        if event.get(key) not in (None, "")
    }
    if "text" in event:
        text = str(event.get("text") or "")
        summary["text"] = text[:600]
        if len(text) > 600:
            summary["text_omitted_chars"] = len(text) - 600
    return summary or {"type": str(event.get("type") or "unknown")}


def _render_wake_payload(ctx: Any, payload: dict[str, Any]) -> str:
    """Render valid bounded JSON, spilling an oversized exact wake to artifacts."""

    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    from ouroboros.tool_capabilities import tool_result_limit

    budget = tool_result_limit("delegate_wait")
    if len(raw) <= budget:
        return raw
    wake_id = str(payload.get("supervision_wake_id") or "")
    try:
        from ouroboros.artifacts import store_actor_source_bytes, task_id_for_artifacts

        source = store_actor_source_bytes(
            pathlib.Path(getattr(ctx, "drive_root", custody.custody_root(ctx))),
            task_id_for_artifacts(ctx),
            category="tool_results",
            source_id=f"delegate-wake-{wake_id or uuid.uuid4().hex}",
            data=raw.encode("utf-8"),
            extension="json",
        )
    except Exception:
        # Keep the exact wake pending and let the ordinary outer truncation fail
        # acknowledgement rather than falsely claiming lossy delivery.
        _emit(ctx, "delegate_supervision_wake_source_failed", {
            "run_id": str(payload.get("run_id") or ""),
            "wake_id": wake_id,
            "total_chars": len(raw),
        })
        return raw
    events = payload.get("wake_events") if isinstance(payload.get("wake_events"), list) else []
    summaries = [_wake_event_summary(event) for event in events]
    envelope: dict[str, Any] = {
        key: (str(value)[:600] if isinstance(value, str) else value)
        for key in ("status", "run_id", "state", "last_seq", "reason")
        if (value := payload.get(key)) not in (None, "")
    }
    envelope["supervision_wake_id"] = wake_id
    envelope["wake_events"] = summaries
    if isinstance(payload.get("coordination_context"), dict):
        envelope["coordination_context"] = payload["coordination_context"]
    envelope["wake_delivery"] = {
        "complete": False,
        "total_chars": len(raw),
        "wake_events_total": len(events),
        "wake_events_summarized": len(summaries),
        "wake_events_omitted": 0,
        "source": source,
        "note": (
            "The exact combined delegate/coordination wake is staged at source. "
            "Read it with the supplied read_file arguments before relying on omitted detail."
        ),
    }
    rendered = json.dumps(envelope, ensure_ascii=False, indent=2)
    while len(rendered) > budget and summaries:
        summaries.pop()
        envelope["wake_delivery"]["wake_events_summarized"] = len(summaries)
        envelope["wake_delivery"]["wake_events_omitted"] = len(events) - len(summaries)
        rendered = json.dumps(envelope, ensure_ascii=False, indent=2)
    if len(rendered) > budget:
        envelope.pop("wake_events", None)
        envelope["wake_delivery"]["wake_events_summarized"] = 0
        envelope["wake_delivery"]["wake_events_omitted"] = len(events)
        rendered = json.dumps(envelope, ensure_ascii=False, indent=2)
    if len(rendered) > budget and isinstance(envelope.get("coordination_context"), dict):
        context = envelope["coordination_context"]
        envelope["coordination_context"] = {
            "state": "available_in_full_wake_source",
            "observed_at": str(context.get("observed_at") or ""),
            "root_task_id": str(context.get("root_task_id") or ""),
        }
        rendered = json.dumps(envelope, ensure_ascii=False, indent=2)
    if len(rendered) > budget:
        envelope = {
            "status": str(payload.get("status") or "wake_available")[:120],
            "run_id": str(payload.get("run_id") or "")[:200],
            "supervision_wake_id": wake_id,
            "coordination_context": {"state": "available_in_full_wake_source"},
            "wake_delivery": envelope["wake_delivery"],
        }
        rendered = json.dumps(envelope, ensure_ascii=False, indent=2)
    return rendered


def _pending_payload(ctx: Any, state: dict[str, Any]) -> dict[str, Any]:
    pending = state.get("pending_wake") if isinstance(state.get("pending_wake"), dict) else {}
    payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
    if pending and not pending.get("acknowledged_at") and payload:
        replay = dict(payload)
        if str(pending.get("attempt_key") or "") != _attempt_key(ctx):
            events = replay.get("wake_events")
            if isinstance(events, list):
                kept = [
                    dict(item) for item in events if isinstance(item, dict)
                    and str(item.get("kind") or "") not in _LOOP_CONTROL_KINDS
                ]
                if kept:
                    replay["wake_events"] = kept
                else:
                    replay.pop("wake_events", None)
            if str(replay.get("status") or "") in _QUIET_STATUSES and not replay.get("wake_events"):
                pending["acknowledged_at"] = utc_now_iso()
                state["last_acknowledged_wake"] = pending
                state["pending_wake"] = {}
                state["status"] = "awake"
                _save_state(ctx, state)
                return {}
        return replay
    return {}


def acknowledge_pending_wake(ctx: Any, delivered: Any = None) -> bool:
    """Ack only after the exact wake or its verified source handle was delivered."""

    state = _load_state(ctx, "")
    pending = state.get("pending_wake") if isinstance(state.get("pending_wake"), dict) else {}
    if not pending or pending.get("acknowledged_at"):
        return True
    expected = str(pending.get("wake_id") or "")
    attempt = _attempt_key(ctx)
    if delivered is None:
        # Observation, not acknowledgement: without the exact payload (or its
        # host-staged source envelope) no attempt has proved transcript delivery.
        return False
    try:
        payload = json.loads(delivered) if isinstance(delivered, str) else delivered
    except (TypeError, ValueError):
        return False
    payload = payload if isinstance(payload, dict) else {}
    if expected and expected not in _wake_ids(payload):
        return False
    from ouroboros.owner_mailbox import acknowledge_task_messages

    mailbox_ids = [str(item) for item in (pending.get("mailbox_ids") or []) if str(item)]
    seen_mailbox_ids = [
        str(item) for item in (pending.get("seen_mailbox_ids") or []) if str(item)
    ] or [
        str(item.get("msg_id") or "")
        for item in (
            pending.get("payload", {}).get("wake_events", [])
            if isinstance(pending.get("payload"), dict) else []
        )
        if isinstance(item, dict) and str(item.get("msg_id") or "")
    ]
    wake_events = (
        pending.get("payload", {}).get("wake_events", [])
        if isinstance(pending.get("payload"), dict) else []
    )
    owner_ids = {
        str(item.get("msg_id") or "")
        for item in wake_events if isinstance(item, dict)
        and str(item.get("kind") or "owner_text") not in {
            *_LOOP_CONTROL_KINDS, KIND_TASK_MESSAGE,
        }
    }
    global_ids = [item for item in mailbox_ids if item not in owner_ids]
    mailbox_root = pathlib.Path(getattr(ctx, "drive_root", custody.custody_root(ctx)))
    task_id = str(getattr(ctx, "task_id", "") or "")
    if global_ids and not acknowledge_task_messages(
        mailbox_root, task_id, global_ids, wake_id=expected,
    ):
        return False
    if owner_ids and not acknowledge_task_messages(
        mailbox_root, task_id, sorted(owner_ids), wake_id=expected,
        attempt_key=attempt,
    ):
        return False
    if mailbox_ids and not (global_ids or owner_ids) and not acknowledge_task_messages(
        mailbox_root, task_id, mailbox_ids, wake_id=expected, attempt_key=attempt,
    ):
        return False
    if not _emit(ctx, "delegate_supervision_wake_acknowledged", {
        "run_id": str(state.get("run_id") or ""), "wake_id": expected,
        "mailbox_ids": mailbox_ids,
        "attempt_key": attempt,
        "interaction_ids": list(pending.get("interaction_ids") or []),
        "coordination": any(
            isinstance(item, dict)
            and str(item.get("type") or "").startswith("child_")
            for item in wake_events
        ),
    }):
        return False
    interaction_ids = frozenset(
        str(item) for item in (pending.get("interaction_ids") or []) if str(item)
    )
    if interaction_ids and str(state.get("run_id") or ""):
        from ouroboros.delegate_interactions import _REPORTED_INTERACTIONS

        run_id = str(state.get("run_id") or "")
        _REPORTED_INTERACTIONS[run_id] = frozenset(
            set(_REPORTED_INTERACTIONS.get(run_id, frozenset())) | set(interaction_ids)
        )
    prior_mailbox_ids = (
        state.get("mailbox_acknowledged_ids") or []
        if str(state.get("mailbox_acknowledged_attempt_key") or "") == attempt
        else []
    )
    state["mailbox_acknowledged_ids"] = sorted({
        *(str(item) for item in prior_mailbox_ids if str(item)),
        *seen_mailbox_ids,
    })
    state["mailbox_acknowledged_attempt_key"] = attempt
    state["interaction_acknowledged_ids"] = sorted({
        *(str(item) for item in (state.get("interaction_acknowledged_ids") or []) if str(item)),
        *interaction_ids,
    })
    coordination_cursor = pending.get("coordination_cursor")
    if isinstance(coordination_cursor, dict):
        state["coordination_cursor"] = coordination_cursor
    pending["acknowledged_at"] = utc_now_iso()
    state["last_acknowledged_wake"] = pending
    state["pending_wake"] = {}
    state["status"] = "awake"
    _save_state(ctx, state)
    return True


def _control_wakes(ctx: Any) -> list[dict[str, Any]]:
    wakes: list[dict[str, Any]] = []
    task_id = str(getattr(ctx, "task_id", "") or "")
    try:
        from ouroboros.cancel_intents import cancel_pending

        if cancel_pending(custody.custody_root(ctx), task_id):
            wakes.append({"type": "cancellation_intent"})
    except Exception:
        pass
    try:
        from ouroboros.deadline_utils import deadline_remaining_sec, parse_deadline_ts

        metadata = getattr(ctx, "task_metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        if parse_deadline_ts(metadata.get("deadline_at")) is not None and deadline_remaining_sec(ctx) <= 0:
            wakes.append({"type": "deadline"})
    except Exception:
        pass
    return wakes


def supervised_wait(
    ctx: Any,
    run_id: str,
    *,
    since_seq: Optional[int] = None,
    checkpoint_after_sec: Optional[int] = None,
    checkpoint_reason: str = "",
    wait_once: Optional[Callable[..., str]] = None,
) -> str:
    """Renew quiet windows internally and return only a meaningful wake batch."""

    if (checkpoint_after_sec is None) != (not str(checkpoint_reason or "").strip()):
        return json.dumps({
            "status": "refused",
            "reason": "checkpoint_requires_time_and_reason",
            "detail": "checkpoint_after_sec and non-empty checkpoint_reason must be supplied together.",
        }, ensure_ascii=False, indent=2)
    if wait_once is None:
        from ouroboros.tools.delegate import _delegate_wait

        wait_once = _delegate_wait
    state = _load_state(ctx, run_id)
    replay = _pending_payload(ctx, state)
    if replay:
        _emit(ctx, "delegate_supervision_wake_replayed", {
            "run_id": str(run_id),
            "wake_id": str(replay.get("supervision_wake_id") or ""),
        })
        return _render_wake_payload(ctx, replay)
    snapshot = getattr(ctx, "task_metadata", {})
    snapshot = snapshot.get("configured_subagent") if isinstance(snapshot, dict) else {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    state["config_fingerprint"] = str(snapshot.get("config_fingerprint") or "")
    state["status"] = "sleeping"
    if since_seq is not None:
        state["journal_cursor"] = max(int(state.get("journal_cursor") or 0), int(since_seq))
    if checkpoint_after_sec is not None:
        delay = max(1, min(604_800, int(checkpoint_after_sec)))
        state["checkpoint"] = {
            "due_at_unix": time.time() + delay,
            "reason": str(checkpoint_reason).strip(),
            "requested_at": utc_now_iso(),
            "consumed": False,
        }
        _emit(ctx, "delegate_supervision_checkpoint_scheduled", {
            "run_id": str(run_id), "due_at_unix": state["checkpoint"]["due_at_unix"],
            "reason": state["checkpoint"]["reason"],
        })
    # Do not baseline already-terminal children before the first poll.  The model
    # has not observed them yet; treating them as prior state loses a fast child
    # forever when the physical leaf stays quiet.  The ordinary pending-wake
    # cursor/ack path below publishes each existing terminal/attention fact once.
    _save_state(ctx, state)
    _emit(ctx, "delegate_supervision_wait_entered", {
        "run_id": str(run_id), "journal_cursor": int(state.get("journal_cursor") or 0),
        "checkpoint_scheduled": bool(checkpoint_after_sec is not None),
    })

    while True:
        raw = wait_once(
            ctx,
            run_id,
            _TICK_SEC,
            int(state.get("journal_cursor") or 0),
        )
        payload = _payload(raw)
        cursor = payload.get("last_seq")
        if isinstance(cursor, int):
            state["journal_cursor"] = max(int(state.get("journal_cursor") or 0), cursor)
        addressed_wakes = _addressed_wakes(ctx, state)
        control_wakes = _control_wakes(ctx)
        coordination_events, next_coordination_cursor = _coordination_wakes(ctx, state)
        wakes = addressed_wakes + control_wakes + coordination_events
        checkpoint = state.get("checkpoint") if isinstance(state.get("checkpoint"), dict) else {}
        due = bool(
            checkpoint
            and not checkpoint.get("consumed")
            and float(checkpoint.get("due_at_unix") or 0) <= time.time()
        )
        meaningful = str(payload.get("status") or "") not in _QUIET_STATUSES
        if meaningful or wakes or due:
            if checkpoint and not checkpoint.get("consumed"):
                checkpoint["consumed"] = True
                checkpoint["consumed_at"] = utc_now_iso()
                checkpoint["consumed_by"] = (
                    "real_event" if meaningful or wakes else "scheduled_checkpoint"
                )
                state["checkpoint"] = checkpoint
                _emit(ctx, "delegate_supervision_checkpoint_consumed", {
                    "run_id": str(run_id), "reason": str(checkpoint.get("reason") or ""),
                    "consumed_by": str(checkpoint.get("consumed_by") or ""),
                })
            if due and not meaningful and not wakes:
                payload = {
                    "status": "inspection_checkpoint",
                    "run_id": str(run_id),
                    "reason": str(checkpoint.get("reason") or ""),
                    "last_seq": int(state.get("journal_cursor") or 0),
                }
            if wakes:
                payload["wake_events"] = wakes
            payload["coordination_context"] = coordination_live_context(ctx)
            wake_id = uuid.uuid4().hex
            payload["supervision_wake_id"] = wake_id
            state["status"] = "wake_pending"
            state["last_wake"] = payload
            state["pending_wake"] = {
                "wake_id": wake_id,
                "attempt_key": _attempt_key(ctx),
                "payload": payload,
                "mailbox_ids": [
                    str(item.get("msg_id") or "") for item in wakes
                    if str(item.get("msg_id") or "")
                    and str(item.get("kind") or "") not in _LOOP_CONTROL_KINDS
                ],
                "seen_mailbox_ids": [
                    str(item.get("msg_id") or "") for item in wakes
                    if str(item.get("msg_id") or "")
                ],
                "interaction_ids": sorted(_interaction_ids(payload)),
                "coordination_cursor": next_coordination_cursor,
                "created_at": utc_now_iso(),
            }
            _save_state(ctx, state)
            _emit(ctx, "delegate_supervision_wake_pending", {
                "run_id": str(run_id), "wake_id": wake_id,
                "payload": payload,
                "mailbox_ids": list(state["pending_wake"]["mailbox_ids"]),
                "interaction_ids": list(state["pending_wake"]["interaction_ids"]),
            })
            return _render_wake_payload(ctx, payload)
        state["coordination_cursor"] = next_coordination_cursor
        state["quiet_renewals"] = int(state.get("quiet_renewals") or 0) + 1
        renewals = int(state["quiet_renewals"])
        if renewals == 1 or renewals & (renewals - 1) == 0:
            _emit(ctx, "delegate_supervision_wait_renewed", {
                "run_id": str(run_id), "quiet_renewals": renewals,
                "journal_cursor": int(state.get("journal_cursor") or 0),
                "event_only": True,
            })
        _save_state(ctx, state)


def read_unknown_hold(ctx: Any) -> dict[str, Any]:
    """Durable unknown-provider hold latch (nanny-leaf D1-min).

    Written when a nanny round dies ``provider_outcome_unknown`` with exactly one
    live delegated leaf; the successor generation (worker-crash adoption) must
    re-enter the hold BEFORE any LLM call, so the latch lives in this durable
    supervision record that ``prepare_handoff`` already snapshots.

    An EXISTING-but-unreadable state file raises ``UnknownHoldUnreadable``,
    never an empty dict: "no latch" and "cannot know whether a latch exists"
    must not look alike — the caller fails closed to a no-call terminal
    instead of dispatching a possible resend (final-pair sol #2).
    """
    path = _state_path(ctx)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise UnknownHoldUnreadable(str(exc)) from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise UnknownHoldUnreadable(str(exc)) from exc
    hold = data.get("unknown_provider_hold") if isinstance(data, dict) else None
    return dict(hold) if isinstance(hold, dict) else {}


class UnknownHoldUnreadable(Exception):
    """The durable hold latch exists but cannot be read (fail closed)."""


def write_unknown_hold(ctx: Any, run_id: str, hold: dict[str, Any]) -> None:
    # run_id is deliberately unused: loaded with the reset-proof empty run_id,
    # so latching for a NEWER leaf cannot rebuild the state and wipe the
    # journal cursor or an unacked wake; the run id travels inside ``hold``.
    state = _load_state(ctx, "")
    state["unknown_provider_hold"] = dict(hold)
    _save_state(ctx, state)


def clear_unknown_hold(ctx: Any) -> None:
    state = _load_state(ctx, "")
    if state.pop("unknown_provider_hold", None) is not None:
        _save_state(ctx, state)


def supervision_checkpoint(ctx: Any) -> dict[str, Any]:
    """Read the durable checkpoint used by selective recovery/restart."""

    try:
        data = json.loads(_state_path(ctx).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def delegate_wait_entry(
    ctx: Any,
    run_id: str,
    wait_sec: Optional[int] = None,
    since_seq: Optional[int] = None,
    checkpoint_after_sec: Optional[int] = None,
    checkpoint_reason: str = "",
) -> str:
    """Event-only wait; hidden ``wait_sec`` is accepted only for old transcripts."""

    if wait_sec is not None:
        _emit(ctx, "delegate_supervision_legacy_wait_ignored", {
            "run_id": str(run_id), "wait_sec": int(wait_sec),
            "contract": "event_only",
        })

    return supervised_wait(
        ctx, run_id, since_seq=since_seq,
        checkpoint_after_sec=checkpoint_after_sec,
        checkpoint_reason=checkpoint_reason,
    )


__all__ = [
    "acknowledge_pending_wake", "delegate_wait_entry", "supervised_wait",
    "supervision_checkpoint", "coordination_live_context",
]
