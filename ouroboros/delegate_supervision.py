"""Event-only supervising wait for configured session nannies."""

from __future__ import annotations

import json
import pathlib
import time
import uuid
from typing import Any, Callable, Optional

from ouroboros import delegate_custody as custody
from ouroboros.delegate_shared import _fail, delegate_result, refusal_host_code
from ouroboros.owner_mailbox import (
    KIND_FINALIZE_NOW,
    KIND_HURRY,
    KIND_TASK_MESSAGE,
    drain_owner_entries,
    place_arrival_wake,
)
from ouroboros.tools.tool_result import ToolResult
from ouroboros.utils import utc_now_iso
from ouroboros import delegate_supervision_state as checkpoints

_TICK_SEC = 3
_QUIET_STATUSES = {"progress", "no_progress", "observation_pending"}
_LOOP_CONTROL_KINDS = {KIND_FINALIZE_NOW, KIND_HURRY}
_MAX_COORDINATION_SEEN = 256
# The gateway's typed code for a read that delivered no daemon answer at all
# (``gateways.claudexor._request``): a quiet renewal here, never a model wake.
_DAEMON_UNREACHABLE = "daemon_unreachable"
# The hold's in-slot mark outlives one slow observation read (60 s), then renews; an
# observation older than a few ticks says nothing current about the run's work.
_HOLD_MARK_SEC = 120.0
_OBSERVATION_CURRENT_SEC = 4 * _TICK_SEC
_OBSERVED: dict = {}  # state path -> ((mtime_ns, size), parsed observation fields)


def _owner_line(ctx: Any, text: str, key: str, tone: str) -> None:
    """One owner-visible line per unreachable-daemon episode, through the task's
    existing progress channel; the typed ``task_incident``/``toast_once`` pair is
    what the client toasts once, so no host state is written for the dedup."""
    fn = getattr(ctx, "emit_progress_fn", None)
    if not callable(fn):
        return
    try:
        fn(text, incident={
            "task_incident": "delegation_daemon_unreachable",
            "toast_once": f"{getattr(ctx, 'task_id', '') or ''}:{key}",
            "toast_tone": tone,
        })
    except Exception:
        pass


# A tick that returned NO DAEMON ANSWER: either a read the daemon never answered,
# or a refusal raised before any byte left the host (a missing descriptor, an
# unreadable token, an engine below the floor). It drops the loop's transport, so
# the next tick re-reads the descriptor and re-handshakes, and it cannot close an
# open outage episode either: silence of a different shape is not contact.
_NO_DAEMON_ANSWER_STATUSES = frozenset({"observation_pending", "refused"})
# A tick text the host could not read as a JSON object (``_payload``): no daemon
# answer is established either, whatever the text was.
_OBSERVATION_MALFORMED = "observation_malformed"


def _loop_gateway() -> Any:
    """The loop's own transport, built WITHOUT a handshake: the observing wait
    handshakes it once (``engine_version`` is the receipt) and classifies any refusal
    through its own typed path. An unreadable descriptor leaves the tick to its
    per-call transport, which reports the same fact the same way."""
    from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable

    try:
        return ClaudexorGateway()
    except ClaudexorUnavailable:
        return None


def _drop_gateway(gateway: Any) -> None:
    if gateway is not None:
        try:
            gateway.close()
        except Exception:
            pass
    return None


def _attempt_key(ctx: Any) -> str:
    return str(getattr(ctx, "task_attempt", None) or 1)


def _state_path(ctx: Any) -> pathlib.Path:
    task_id = str(getattr(ctx, "task_id", "") or "")
    return custody.custody_root(ctx) / "state" / "delegate_supervision" / f"{task_id}.json"


def _load_state(ctx: Any, run_id: str) -> dict[str, Any]:
    data = checkpoints.read(_state_path(ctx))
    selected = run_id or (next(iter(data["runs"])) if len(data["runs"]) == 1 else "")
    return supervision_state_for_run(data, selected)


def supervision_state_for_run(data: Any, run_id: str) -> dict[str, Any]:
    return checkpoints.run_view(data, run_id)


def task_scoped_supervision(data: Any) -> dict[str, Any]:
    return checkpoints.task_facts(data)


def _save_state(ctx: Any, state: dict[str, Any]) -> None:
    checkpoints.save_view(_state_path(ctx), custody.custody_root(ctx), str(ctx.task_id), state)


def _emit(ctx: Any, kind: str, payload: dict[str, Any]) -> bool:
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    # #1578: the lineage this context already knows; an unknown id stays absent.
    lineage = {key: str(metadata.get(key) or getattr(ctx, key, "") or "") for key in ("root_task_id", "parent_task_id")}
    return custody.emit(custody.custody_root(ctx), kind, {
        "task_id": str(getattr(ctx, "task_id", "") or ""), **{k: v for k, v in lineage.items() if v}, **payload,
    })


def _payload(raw: Any, run_id: str) -> tuple[dict[str, Any], bool]:
    """One tick's observation and whether the host could read it at all.

    The observing wait renders a JSON object; anything else is the host's own
    read failure (#947), not a daemon answer and not a run's state. It keeps the
    ``fault`` status the hold exits on, carries the family's refusal envelope so
    it never classifies as a successful observation, and keeps the whole tick
    text: an oversized wake spills it to its exact source (``_render_wake_payload``).
    """
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict):
        return data, True
    return {"status": "fault", "ok": False, "host_code": refusal_host_code(_OBSERVATION_MALFORMED),
            "reason": _OBSERVATION_MALFORMED, "run_id": str(run_id),
            "detail": raw if isinstance(raw, str) else repr(raw)}, False


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
    until a path that OWNS a mutation — the loop's time-budget notice — latches
    the anchor. A poll answering the same question twice never changes its own
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
    """The tree's accounted spend: the usage store's root summary read through
    ``usage_accounting.usage_breakdown`` in every state (a root with no recorded
    attempt is the reader's own known-zero). The fact writes nothing of its own."""
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

    Polling writes nothing of its own; the settled-spend fact is one read of the
    usage store's root summary (``usage_accounting.usage_breakdown``).
    """

    root_task_id = _coordination_root_id(ctx)
    return {
        "observed_at": utc_now_iso(),
        "root_task_id": root_task_id,
        "parent_intent": _parent_intent_fact(ctx),
        "time": _time_fact(ctx),
        "settled_spend": _settled_spend_fact(ctx, root_task_id),
        "active_descendants": _active_descendants_fact(ctx),
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
            # The peer relation the drain projected (a PEER_RELATION_LABELS key): the sender's
            # typed place, carried so a wake never signs a child or sibling as an
            # ancestor or owner (the same fact the round-top prefix reads).
            **({"relation": str(entry["relation"])} if str(entry.get("relation") or "") else {}),
            "text": str(entry.get("text") or ""),
            "ts": str(entry.get("ts") or ""),
        }
        for entry in entries if place_arrival_wake(entry) is not False  # context only: unread for the drain
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
    # Sender attribution (provenance, relayed-from identity, peer relation) rides the
    # reduced projection too: a spilled wake must still say WHO wrote the text and
    # in what place, or the model reads an addressed contribution as unsigned.
    summary = {
        key: event.get(key)
        for key in (
            "type", "kind", "msg_id", "provenance", "source_task_id",
            "relayed_from_task_id", "relation", "child_task_id",
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


def _wake_result(payload: dict[str, Any]) -> ToolResult:
    """One wake payload as its native result, keyed by the wake id it published.

    The id rides in producer meta because the ACK is keyed on it: the loop
    acknowledges the wake the result itself names, not every call that happens to
    be spelled ``delegate_wait``."""
    wake_id = str(payload.get("supervision_wake_id") or "")
    return delegate_result(payload, meta={"supervision_wake_id": wake_id} if wake_id else {})


def _render_wake_payload(ctx: Any, payload: dict[str, Any]) -> ToolResult:
    """Render valid bounded JSON, spilling an oversized exact wake to artifacts."""

    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    from ouroboros.tool_capabilities import tool_result_limit

    budget = tool_result_limit("delegate_wait")
    if len(raw) <= budget:
        return _wake_result(payload)
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
        return _wake_result(payload)
    events = payload.get("wake_events") if isinstance(payload.get("wake_events"), list) else []
    summaries = [_wake_event_summary(event) for event in events]
    # ``ok``/``host_code`` ride the fitted envelope: a wake that relays a REFUSED
    # observation must keep its classification when the exact payload spills, or a
    # refusal too large to inline would read as a successful wait.
    envelope: dict[str, Any] = {
        key: (str(value)[:600] if isinstance(value, str) else value)
        for key in ("status", "ok", "host_code", "run_id", "state", "last_seq", "reason",
                    "continuation", "continuation_note", "sleep", "cache_horizon_note",
                    "leaf_live_input")
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
            **({"ok": False, "host_code": str(payload.get("host_code") or "")}
               if payload.get("ok") is False else {}),
            "run_id": str(payload.get("run_id") or "")[:200],
            **{key: payload[key] for key in ("continuation", "continuation_note", "sleep",
                                             "leaf_live_input") if key in payload},
            "supervision_wake_id": wake_id,
            "coordination_context": {"state": "available_in_full_wake_source"},
            "wake_delivery": envelope["wake_delivery"],
        }
        rendered = json.dumps(envelope, ensure_ascii=False, indent=2)
    return _wake_result(envelope)


def _schema_1_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Carry a schema-1 ``pending_wake.payload`` forward without rewriting it.

    Rows written before this family carried ``ok``/``host_code`` replay exactly as
    stored; only the two classification keys are derived, and only when they are
    ABSENT. ``refused`` was the one status the old writer used for a refused
    observation, so it replays as the recorded tool failure it always was and
    everything else replays as the ordinary observation it always was. No prose is
    read, no word is matched, no terminal/success/zero-spend fact is invented, and
    an unknown or corrupt shape keeps its whole body.
    """
    if "ok" in payload or str(payload.get("status") or "") != "refused":
        return dict(payload)
    return {**payload, "ok": False, "host_code": refusal_host_code(str(payload.get("reason") or ""))}


def _pending_payload(ctx: Any, state: dict[str, Any]) -> dict[str, Any]:
    pending = state.get("pending_wake") if isinstance(state.get("pending_wake"), dict) else {}
    payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
    if pending and not pending.get("acknowledged_at") and payload:
        replay = _schema_1_envelope(payload)
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
        # A pending invocation can bind after this attempt already drained its
        # mail. Replay the wake's other facts without delivering that mail twice.
        return unseen_wake_view(replay, getattr(ctx, "_loop_mailbox_seen_ids", None))
    return {}


def unseen_wake_view(payload: dict[str, Any], seen: Any) -> dict[str, Any]:
    """The delivered wake without text mail this attempt's transcript already holds.

    Only the view narrows; the pending wake keeps every mailbox id for its ACK.
    """
    events = payload.get("wake_events")
    if not seen or not isinstance(events, list):
        return payload
    kept = [event for event in events if not any(item in seen for item in _text_mailbox_ids([event]))]
    return payload if len(kept) == len(events) else {**payload, "wake_events": kept}


def _text_mailbox_ids(events: Any) -> set[str]:
    """Only text carried by the delivered view, never omitted mail or controls."""
    return {str(item["msg_id"]) for item in events if isinstance(item, dict)
            and item.get("msg_id") and "text" in item
            and str(item.get("kind") or "owner_text") not in _LOOP_CONTROL_KINDS}


def _record_terminal_acknowledged(ctx: Any, state: dict[str, Any], pending: dict[str, Any],
                                  wake_id: str, attempt: str) -> None:
    """The custody-native half of an accepted TERMINAL wake (#1573).

    Called only after the supervision acknowledgement landed, i.e. after the exact
    wake (or its verified source envelope) was appended to this attempt's
    transcript, and only for the run's own terminal payload. It proves that append in
    ``attempt`` — not a physical send, a read or permanent transcript retention. A
    row that fails to land only lets the own-run notice repeat (the safe direction).
    """
    payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
    run_id = str(state.get("run_id") or "")
    if run_id and str(payload.get("status") or "") == "terminal" and str(payload.get("run_id") or "") == run_id:
        _emit(ctx, custody.TERMINAL_ACKNOWLEDGED, {
            "run_id": run_id, "wake_id": wake_id, "attempt_key": attempt,
        })


def stage_pending_wake_exposure(ctx: Any, delivered: Any) -> None:
    """Stage shown questions and mail in this attempt before saving their transcript.

    No mailbox/wake ACK or run answer occurs here. If the save fails but the
    transcript remains, exposure remains with it; cold replay is at least once.
    """
    try:
        payload = json.loads(delivered) if isinstance(delivered, str) else delivered
    except (TypeError, ValueError):
        return
    if not _wake_ids(payload):
        return
    for run_id in checkpoints.read(_state_path(ctx))["runs"]:
        state = _load_state(ctx, run_id)
        pending = state.get("pending_wake") or {}
        if pending.get("wake_id") in _wake_ids(payload):
            _record_exposure(ctx, payload, pending, run_id)


def _record_exposure(ctx: Any, payload: Any, pending: dict[str, Any], run_id: str) -> None:
    wake = payload.get("wake", payload) if isinstance(payload, dict) else {}
    if isinstance(wake, dict) and isinstance(wake.get("wake_events"), list):
        seen_mail = getattr(ctx, "_loop_mailbox_seen_ids", None)
        if seen_mail is None:
            seen_mail = ctx._loop_mailbox_seen_ids = set()
        seen_mail.update(_text_mailbox_ids(wake["wake_events"]))
    shown = _interaction_ids(payload) & set(pending.get("interaction_ids") or [])
    if shown and run_id:
        seen = dict(getattr(ctx, "_sleep_interaction_seen", None) or {})
        seen[run_id] = sorted(set(seen.get(run_id) or ()) | shown)
        ctx._sleep_interaction_seen = seen


def acknowledge_pending_wake(ctx: Any, delivered: Any = None) -> bool:
    """Acknowledge only run wakes named by the persisted delivered envelope."""
    try:
        payload = json.loads(delivered) if isinstance(delivered, str) else delivered
    except (TypeError, ValueError):
        return False
    ids = _wake_ids(payload)
    states = [_load_state(ctx, run_id) for run_id in checkpoints.read(_state_path(ctx))["runs"]]
    pending = [state for state in states if state.get("pending_wake")
               and not state["pending_wake"].get("acknowledged_at")]
    selected = [state for state in pending if state["pending_wake"].get("wake_id") in ids]
    if pending and not selected:
        return False
    return all(_acknowledge_run_wake(ctx, payload, state) for state in selected)


def _acknowledge_run_wake(ctx: Any, delivered: Any, state: dict) -> bool:
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
    interaction_ids = frozenset(str(item) for item in (pending.get("interaction_ids") or []) if str(item))
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
    _record_terminal_acknowledged(ctx, state, pending, expected, attempt)
    if interaction_ids and state.get("run_id"):
        from ouroboros.delegate_interactions import _REPORTED_INTERACTIONS

        run_id = state["run_id"]
        _REPORTED_INTERACTIONS[run_id] = frozenset(set(_REPORTED_INTERACTIONS.get(run_id, ())) | interaction_ids)
    # Compatibility for direct ACK consumers (the UNKNOWN hold): only a saved receipt
    # records exposure, since the hold drops a receipt whose ACK failed (#2142).
    # Loop/startup consumers stage BEFORE saving.
    if expected:
        _record_exposure(ctx, payload, pending, str(state.get("run_id") or ""))
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


def _open_sleep_entry(state: dict[str, Any], run_id: str) -> None:
    """Open this call's sleep: every ``supervised_wait`` call is a NEW sleep.

    The prior status only labels it, it never carries elapsed time over: an
    ``adopted`` state is a worker-loss recovery, a ``sleeping`` one is a previous
    call that died without publishing a wake (tool timeout, exception)."""
    prior = str(state.get("status") or "")
    previous = state.get("sleep_entry") if isinstance(state.get("sleep_entry"), dict) else {}
    entry: dict[str, Any] = {
        "run_id": str(run_id), "entered_at": utc_now_iso(), "entered_at_unix": time.time(),
        "quiet_renewals_at_entry": int(state.get("quiet_renewals") or 0),
        "journal_cursor_at_entry": int(state.get("journal_cursor") or 0),
    }
    if prior == "adopted":
        entry["opened_after"] = "worker_loss_adoption"
    elif prior == "sleeping":
        entry["opened_after"] = "interrupted_sleep"
        if previous.get("entered_at"):
            entry["previous_entered_at"] = str(previous["entered_at"])
    state["sleep_entry"] = entry


def _record_observation(state: dict[str, Any], payload: dict[str, Any], answered: bool,
                        attempt: str = "") -> None:
    """Dated observation facts (run-scoped): the supervisor's own ``status`` and the
    file's ``updated_at`` move on every quiet renewal, failed reads included, so
    freshness is the last ANSWERED observation plus the current failure, never mtime.
    ``waiting_on_user`` and the task attempt let a reader judge the run's current
    work (``observed_run_activity``) without the window payload."""
    at = utc_now_iso()
    state["observation"] = {
        "at": at, "answered": bool(answered), "status": str(payload.get("status") or ""),
        "run_state": str(payload.get("state") or ""), "reason": str(payload.get("reason") or ""),
        "waiting_on_user": bool(payload.get("waiting_on_user")), "at_unix": time.time(), "attempt": str(attempt),
    }
    if answered:
        state["last_answered_observation_at"] = at
    state["observation_failure"] = "" if answered else str(
        payload.get("reason") or payload.get("status") or "unknown")


def run_activity(observation: Any) -> Optional[bool]:
    """What one dated observation of a run says of its work (this supervision record's, or the
    body observer's, ``delegate_collection.current_activity``): True running, False waiting
    (queued for capacity, paused on its question, terminal), None unanswered or unknown.
    A running run counts as working: an engine-internal provider wait carries no typed fact."""
    if not isinstance(observation, dict) or observation.get("answered") is not True:
        return None
    run_state = str(observation.get("run_state") or "")
    if run_state == "running":
        return not observation.get("waiting_on_user")
    return False if run_state == "queued" or run_state in custody.TERMINAL_STATES else None


def observed_run_activity(drive_root: Any, task_id: str, run_id: str, attempt: Any) -> Optional[bool]:
    """The work of the run ``task_id`` holds now, from its supervision record: the same
    run and task attempt, observed within a few ticks; anything else is None (unknown).
    For ``gateway/reply_preparation.py``; one small file, reread only when it changed."""
    path = pathlib.Path(drive_root) / "state" / "delegate_supervision" / f"{task_id}.json"
    try:
        stat = path.stat()
        stamp, memo = (stat.st_mtime_ns, stat.st_size), _OBSERVED.get(str(path))
        if memo is None or memo[0] != stamp:
            data = json.loads(path.read_text(encoding="utf-8"))
            memo = (stamp, checkpoints.normalize(data)["runs"])
            if len(_OBSERVED) >= 256:
                _OBSERVED.clear()
            _OBSERVED[str(path)] = memo
    except (OSError, ValueError, AttributeError):
        return None
    observation = memo[1].get(str(run_id), {}).get("observation") or {}
    try:
        current = time.time() - float(observation.get("at_unix") or 0) <= _OBSERVATION_CURRENT_SEC
    except (TypeError, ValueError):
        current = False
    if (str(run_id) not in memo[1] or not current
            or str(observation.get("attempt") or "") != str(attempt or 1)):
        return None
    return run_activity(observation)


def _leaf_live_input(ctx: Any, run_id: str, gateway: Any) -> str:
    """The leaf route's DECLARED live-input capability (engine-static), read ONCE per
    call at entry through the loop's transport (a one-read transport when the caller
    observes with its own) and stamped on every wake: the route row's ``liveInput``
    in the engine catalog. A capability, never a liveness verdict; an unreadable
    catalog, an unknown route or a read slower than one beat is ``unknown`` (never a
    refusal, and nothing is read on the wake path). A row without the field is an
    engine that has no live-input channel at all: ``none``."""
    try:
        _status, entry = custody.lookup(
            custody.custody_root(ctx), str(getattr(ctx, "task_id", "") or ""), run_id)
        route = str(getattr(entry, "route_id", "") or "")
    except Exception:
        return "unknown"
    reader = (gateway if gateway is not None else _loop_gateway()) if route else None
    if reader is None:
        return "unknown"
    try:
        catalog = reader.agent_capabilities(timeout_sec=_TICK_SEC)
        row = next((item for item in (catalog.get("harnesses") or [])
                    if isinstance(item, dict) and item.get("id") == route), None)
    except Exception:
        row = None
    finally:
        if reader is not gateway:
            _drop_gateway(reader)
    return "unknown" if row is None else str(row.get("liveInput") or "none")


def _since_last_model_response_sec(ctx: Any) -> Optional[float]:
    """Seconds since this task's latest model response was recorded
    (``_last_llm_call_meta.ts``). A LOWER bound on prompt-cache age: the provider
    read/wrote the cached prefix before that response finished, so the horizon note
    built on it can only fire late, never early. Unknown (silent) after a worker
    loss, whose successor has no recorded send yet."""
    usage = getattr(ctx, "_accumulated_usage", None)
    meta = usage.get("_last_llm_call_meta") if isinstance(usage, dict) else None
    try:
        import datetime as _dt

        stamp = _dt.datetime.fromisoformat(str((meta or {}).get("ts") or ""))
        return max(0.0, (_dt.datetime.now(tz=_dt.timezone.utc) - stamp).total_seconds())
    except (TypeError, ValueError):
        return None


def _publish_wake_facts(ctx: Any, state: dict[str, Any], payload: dict[str, Any],
                        live_input: str) -> None:
    """Whole-sleep facts at the ONE durable wake-publication point, every status
    (terminal included), before ``pending_wake`` is stored, so a replay repeats the
    recorded facts exactly. A quiet-status wake (an event arrived during a quiet
    tick) drops the tick's own ``waited_sec``/``quiet_for_sec`` and its generic
    keep-watching/cancel note; a run paused on its question keeps the PAUSED note."""
    entry = state.get("sleep_entry") if isinstance(state.get("sleep_entry"), dict) else {}
    if entry:
        sleep: dict[str, Any] = {
            "entered_at": str(entry.get("entered_at") or ""),
            "slept_sec": round(max(0.0, time.time() - float(entry.get("entered_at_unix") or 0)), 1),
            "quiet_renewals": max(0, int(state.get("quiet_renewals") or 0)
                                  - int(entry.get("quiet_renewals_at_entry") or 0)),
            "journal_advances": max(0, int(state.get("journal_cursor") or 0)
                                    - int(entry.get("journal_cursor_at_entry") or 0)),
        }
        sleep.update({key: entry[key] for key in ("opened_after", "previous_entered_at") if entry.get(key)})
        payload["sleep"] = sleep
    if str(payload.get("status") or "") in _QUIET_STATUSES:
        payload.pop("waited_sec", None)
        payload.pop("quiet_for_sec", None)
        if payload.get("waiting_on_user"):
            from ouroboros.delegate_progress import paused_note

            payload["note"] = paused_note(payload.get("pending_interactions"))
        else:
            payload.pop("note", None)
    payload.pop("cache_horizon_note", None)
    from ouroboros.tools.control import cache_horizon_note

    horizon = cache_horizon_note(ctx, _since_last_model_response_sec(ctx))
    if horizon:
        payload["cache_horizon_note"] = horizon
    payload["leaf_live_input"] = live_input


def supervised_wait(
    ctx: Any,
    run_id: str | list[str],
    *,
    since_seq: Optional[int] = None,
    checkpoint_after_sec: Optional[int] = None,
    checkpoint_reason: str = "",
    wait_once: Optional[Callable[..., str]] = None,
) -> ToolResult:
    """Renew quiet windows internally and return only a meaningful wake batch.

    The journal cursor survives restart. A checkpoint wakes once; any earlier event
    consumes it. The existing three-second renewal keeps unanswered reads quiet;
    only daemon_unreachable opens an outage episode, closed by an answered read.
    The applied cache horizon is disclosed once per wake when exceeded.
    """

    reason_text = str(checkpoint_reason or "").strip()
    ignored_note = ""
    if checkpoint_after_sec == 0 and not reason_text and not isinstance(checkpoint_after_sec, bool):
        # The schema's empty pair asks for no checkpoint: the omitted path, said in the wake.
        checkpoint_after_sec = None
        from ouroboros.tools.arg_feedback import ignored_argument_note

        ignored_note = ignored_argument_note(
            "checkpoint_after_sec", 0, "with an empty checkpoint_reason it asks for no checkpoint")
    if (checkpoint_after_sec is None) != (not reason_text):
        # The family's ONE refusal author, not a second literal envelope beside
        # it: this is an argument fault, and it is recorded as one.
        missing = "checkpoint_after_sec" if checkpoint_after_sec is None else "checkpoint_reason"
        return _fail(
            "delegate_wait", "checkpoint_requires_time_and_reason",
            f"got checkpoint_after_sec={checkpoint_after_sec!r}, checkpoint_reason={reason_text!r}: "
            f"{missing} is missing. Supply both for one proactive inspection, or omit both "
            "(0 with an empty reason also means none).",
        )
    run_ids = list(dict.fromkeys(run_id if isinstance(run_id, list) else [str(run_id)]))
    owns_transport = wait_once is None
    if wait_once is None:
        for selected in run_ids:
            ownership, _entry = custody.lookup(custody.custody_root(ctx), str(ctx.task_id), selected)
            if ownership == custody.FOREIGN:
                from ouroboros.tools.delegate_terminal_evidence import retry_terminal_result

                return retry_terminal_result(ctx, selected)
        from ouroboros.tools.delegate import _delegate_wait
        from functools import partial

        wait_once = partial(_delegate_wait, observation_only=True)
    from ouroboros.delegate_recovery import restore_bound_handoff

    states = {}
    try:
        for selected in run_ids:
            restore_bound_handoff(ctx, selected)
            state = _load_state(ctx, selected)
            replay = _pending_payload(ctx, state)
            if replay:
                _emit(ctx, "delegate_supervision_wake_replayed", {
                    "run_id": selected, "wake_id": str(replay.get("supervision_wake_id") or "")})
                return _render_wake_payload(ctx, replay)
            states[selected] = state
    except (OSError, ValueError) as exc:
        return _fail("delegate_wait", "supervision_state_unknown", str(exc))
    for run_id, state in states.items():
        _ownership, held = custody.lookup(custody.custody_root(ctx), str(ctx.task_id), run_id)
        state["config_fingerprint"] = str(getattr(held, "config_fingerprint", "") or "")
        if since_seq is not None:
            state["journal_cursor"] = max(int(state.get("journal_cursor") or 0), int(since_seq))
        _open_sleep_entry(state, run_id)
        state["status"] = "sleeping"
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
            "sleep_entry": dict(state["sleep_entry"]),
        })

    # The OPEN unreachable-daemon episode: its UTC start (empty when none) plus the
    # millisecond stamp that keeps two episodes of one wait distinct on both client
    # dedup surfaces (the loop_transport.incident precedent).
    outage_since = ""
    outage_stamp = ""
    # The loop's own transport, only when it built the observing wait.
    from ouroboros.tools.delegate_terminal_evidence import retained_terminal_result
    retained_runs = {selected: retained_terminal_result(ctx, selected, retry_money=True) for selected in run_ids} if owns_transport else {}
    gateway = None
    live_inputs = {}
    from ouroboros import model_sleep

    # Holding the run is a wait, never this task's own work; what the run does is its
    # recorded observation (``observed_run_activity``, DESIGN §9 Telegram typing).
    hold_key, hold_until, activity = "delegate:" + ",".join(run_ids), 0.0, None
    selected_index = 0
    try:
        while True:
            run_id = run_ids[selected_index]
            selected_index = (selected_index + 1) % len(run_ids)
            state = _load_state(ctx, run_id)
            retained = retained_runs.pop(run_id, None)
            if run_id not in live_inputs:
                if owns_transport and gateway is None and retained is None:
                    gateway = _loop_gateway()
                live_inputs[run_id] = _leaf_live_input(ctx, run_id, gateway) if retained is None else "unknown"
            live_input = live_inputs[run_id]
            if hold_until - time.time() < _HOLD_MARK_SEC / 2:
                hold_until = time.time() + _HOLD_MARK_SEC
                model_sleep.declare_in_slot_wait(ctx, hold_key, hold_until)
            # A control already present does not wait behind another HTTP read.
            observed = not _control_wakes(ctx)
            if not observed:
                raw = json.dumps({"status": "no_progress", "run_id": str(run_id)})
            elif retained is not None:
                raw, retained = json.dumps(retained), None
            else:
                if owns_transport and gateway is None:
                    gateway = _loop_gateway()
                raw = wait_once(ctx, run_id, _TICK_SEC, int(state.get("journal_cursor") or 0),
                                **({"gateway": gateway} if gateway is not None else {}))
            payload, readable = _payload(raw, run_id)
            answered = readable and str(payload.get("status") or "") not in _NO_DAEMON_ANSWER_STATUSES
            # A typed 503 the engine itself sent to the read reached the daemon over an intact
            # transport: rebuilding it would re-handshake into the same refusal and wake the
            # model, and it is no outage either. Only a read nothing answered is a dead socket.
            reached = answered or payload.get("answered") is True
            resumed = False
            if observed:
                _record_observation(state, payload, answered, _attempt_key(ctx))
                previous, activity = activity, run_activity(state["observation"])
                resumed = activity is True and previous is not True
            if not reached:
                gateway = _drop_gateway(gateway)
            unreachable = (
                payload.get("status") == "observation_pending"
                and payload.get("reason") == _DAEMON_UNREACHABLE and not reached
            )
            if observed and outage_since and reached:
                # The first read the daemon ANSWERED closes the episode. A refusal
                # that never reached it (descriptor gone, token unreadable, engine
                # too old) used to satisfy this and told the owner the daemon was
                # reachable again at the moment it became less reachable.
                _owner_line(
                    ctx,
                    "Delegation daemon reachable again; the outage that began at "
                    f"{outage_since} is over and delegated runs are being observed again.",
                    f"delegation_daemon_recovered:{outage_stamp}", "ok")
                outage_since, outage_stamp = "", ""
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
            collecting = (payload.get("status") == "observation_pending"
                          and payload.get("reason") == "collection_in_progress")
            try:
                from ouroboros.store import acquisition_budget
                with acquisition_budget(0):
                    # A terminal observation does not finish the collector's owned delivery.
                    terminal = [] if meaningful or wakes or collecting else model_sleep.terminal_readiness(
                        ctx, {"runs": [str(run_id)]})
            except Exception:
                terminal = []  # An unreadable custody source cannot hide observed controls or answers.
            if terminal and not meaningful and not wakes:
                payload.update(status="terminal_custody", terminal_facts=terminal)
                meaningful = True
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
                _publish_wake_facts(ctx, state, payload, live_input)
                if ignored_note:
                    payload["ignored_arguments"] = [ignored_note]
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
                payload = state["pending_wake"]["payload"]
                wake_id = state["pending_wake"]["wake_id"]
                _emit(ctx, "delegate_supervision_wake_pending", {
                    "run_id": str(run_id), "wake_id": wake_id,
                    "payload": payload,
                    "mailbox_ids": list(state["pending_wake"]["mailbox_ids"]),
                    "interaction_ids": list(state["pending_wake"]["interaction_ids"]),
                })
                return _render_wake_payload(ctx, payload)
            if payload.get("status") == "observation_pending":
                _emit(ctx, "delegate_supervision_observation_pending", {
                    "run_id": str(run_id), "reason": payload.get("reason"),
                    "waited_sec": payload.get("waited_sec"),
                })
                if unreachable and not outage_since:
                    # The class the model can do nothing about: a dead socket is a quiet
                    # renewal on the same 3 s beat (no backoff, no durable counter), and
                    # the owner hears about it exactly once per episode. Deadline, ceiling,
                    # budget and cancel stay the outer bounds that cut a long unobserved
                    # stretch. The episode stamps its own key and names its start in the
                    # text, so a SECOND outage in the same wait is a new line on both
                    # client surfaces instead of a duplicate the toast set already holds.
                    opened = time.time()
                    outage_since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(opened))
                    outage_stamp = str(int(opened * 1000))
                    _owner_line(ctx, f"Delegation daemon unreachable since {outage_since}; "
                                "delegated runs are not being observed, the runs themselves "
                                "keep going.",
                                f"delegation_daemon_unreachable:{outage_stamp}", "warn")
                # A failed read is not a completed quiet window; retain the cursor
                # and avoid a busy loop if a transport fails before its read bound.
                time.sleep(_TICK_SEC)
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
            if resumed:
                model_sleep.announce_resumed(ctx)  # the held run works again, as recorded: no new start says so
    finally:
        if hold_until:
            model_sleep.end_in_slot_wait(ctx, hold_key)
        _drop_gateway(gateway)


def read_unknown_hold(ctx: Any) -> dict[str, Any]:
    """Durable unknown-provider hold latch (nanny-leaf D1-min).

    Written when a nanny round dies ``provider_outcome_unknown`` with eligible
    live delegated leaves; the successor generation (worker-crash adoption) must
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
        data = checkpoints.normalize(json.loads(raw))
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

    return checkpoints.read(_state_path(ctx))


def delegate_wait_entry(
    ctx: Any,
    run_id: str = "",
    wait_sec: Optional[int] = None,
    since_seq: Optional[int] = None,
    checkpoint_after_sec: Optional[int] = None,
    checkpoint_reason: str = "",
    runs: Optional[list[str]] = None,
) -> ToolResult:
    """Event-only wait; hidden ``wait_sec`` is accepted only for old transcripts."""

    if not run_id and runs is None:
        return _fail("delegate_wait", "missing_run_id", "Supply one run_id or a nonempty runs array.")
    if (bool(run_id) == (runs is not None) or (runs is not None and
            (not isinstance(runs, list) or not runs or any(not isinstance(r, str) or not r.strip() for r in runs)))):
        return _fail("delegate_wait", "run_selector_conflict", "Supply one run_id or a nonempty runs array.")
    if wait_sec is not None:
        _emit(ctx, "delegate_supervision_legacy_wait_ignored", {
            "run_id": str(run_id), "wait_sec": int(wait_sec),
            "contract": "event_only",
        })

    return supervised_wait(
        ctx, runs if runs is not None else run_id, since_seq=since_seq,
        checkpoint_after_sec=checkpoint_after_sec,
        checkpoint_reason=checkpoint_reason,
    )


__all__ = [
    "acknowledge_pending_wake", "stage_pending_wake_exposure", "delegate_wait_entry", "supervised_wait",
    "supervision_checkpoint", "coordination_live_context", "unseen_wake_view",
]
