"""``delegate_wait`` for configured session nannies: a model sleep on runs, its wake delivered once.

The body (``delegate_collection.BodyRuntime``) is the one observer of external runs: it records a run's
terminal and questions in custody, follows its speech (``delegate_feed``) and tells the owner when the
engine stops answering, so no acting slot, nanny tick or LLM call is needed while the nanny sleeps. A
``delegate_wait`` that nothing answers yet arms the nanny's ordinary sleep
(``model_sleep``) with a ``run_wait`` descriptor; the result is owed to the first model request
after the wake (``native_task_wait.deliver_owed``), published there as one wake record
(``pending_wake`` in ``delegate_supervision_state``) and acknowledged only after its exact text was
delivered whole (``deliver_wake``).

Ready sources answer at once; unacknowledged wakes replay their facts
(``replay_of``) before sleeping. Owed delivery follows mail drain; durable wakes
coalesce. Retained output precedes a run-detail read. Without a continuation
owner the same sources are waited on in-process. There is no wait window or
run-count cap; legacy ``wait_sec`` is inert.

``deliver_wake`` binds typed metadata/JSON to the saved whole view needed for
loop/startup ACK: inline or complete exact-source reads
(``delegate_output.credit_exact_source``). Warm/cold parks flush deferred ACKs
and preserve source refs/read coverage. Unknown-outcome holds sleep host-armed
and warm (``delegate_hold``), ACKing whole receipts synchronously before
resuming; cut summaries never count. Wait saves name the failed stage; locks
keep the known cause/errno without asserting run fate. Polls read task state;
wakes keep full tools. ``leaf_live_input`` is capability, never liveness.

The read-only ``coordination_context`` carries intent, time, settled spend and
active descendants, with no thresholds (P5). ``nanny_pacing`` compares usage
with round/cost anchors: coordination resets neither, ``owner_wait`` keeps both,
and old checkpoints flag unknown anchors. Burn informs, never advises.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
import uuid
from typing import Any, Optional

from ouroboros import delegate_custody as custody
from ouroboros.delegate_shared import _fail, delegate_result, refusal_host_code
from ouroboros.delegate_feed import _TICK_SEC
from ouroboros.owner_mailbox import (
    KIND_FINALIZE_NOW, KIND_HURRY, KIND_OWNER_MODEL, KIND_TASK_MESSAGE, drain_owner_entries, place_arrival_wake,
)
from ouroboros.tools.tool_result import ToolResult
from ouroboros.utils import utc_now_iso
from ouroboros import delegate_supervision_state as checkpoints

log = logging.getLogger(__name__)
_QUIET_STATUSES = {"progress", "no_progress", "observation_pending"}
_LOOP_CONTROL_KINDS = {KIND_FINALIZE_NOW, KIND_HURRY, KIND_OWNER_MODEL}
_MAX_COORDINATION_SEEN = 256
# An in-slot wait's mark outlives one beat, then renews.
_HOLD_MARK_SEC = 120.0
# A read text the host could not parse as a JSON object (``_payload``): no daemon answer.
_OBSERVATION_MALFORMED = "observation_malformed"


def _loop_gateway(entry: Any = None) -> Any:
    """One publication's transport to the recorded engine, built WITHOUT a handshake: a work engine proves its
    store first, the owned engine's addressed read handshakes it once (``engine_version`` is the receipt) and
    classifies any refusal through its own typed path. An unreadable descriptor leaves the read to its per-call
    transport, which reports the same fact the same way."""
    from ouroboros.execution_domain import run_gateway
    from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable

    try:
        return run_gateway(entry, ClaudexorGateway)
    except ClaudexorUnavailable:
        return None


def _attempt_key(ctx: Any) -> str:
    return str(getattr(ctx, "task_attempt", None) or 1)


def _state_path(ctx: Any) -> pathlib.Path:
    task_id = str(getattr(ctx, "task_id", "") or "")
    return custody.custody_root(ctx) / "state" / "delegate_supervision" / f"{task_id}.json"


def _load_state(ctx: Any, run_id: str) -> dict[str, Any]:
    data = checkpoints.read(_state_path(ctx))
    selected = run_id or (next(iter(data["runs"])) if len(data["runs"]) == 1 else "")
    return checkpoints.run_view(data, selected)


def _save_state(ctx: Any, state: dict[str, Any], stage: str = "") -> None:
    checkpoints.save_view(_state_path(ctx), custody.custody_root(ctx), str(ctx.task_id), state, stage)


def _emit(ctx: Any, kind: str, payload: dict[str, Any]) -> bool:
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    # #1578: the lineage this context already knows; an unknown id stays absent.
    lineage = {key: str(metadata.get(key) or getattr(ctx, key, "") or "") for key in ("root_task_id", "parent_task_id")}
    return custody.emit(custody.custody_root(ctx), kind, {
        "task_id": str(getattr(ctx, "task_id", "") or ""), **{k: v for k, v in lineage.items() if v}, **payload,
    })


def _payload(raw: Any, run_id: str) -> tuple[dict[str, Any], bool]:
    """One addressed read's observation and whether the host could read it at all.

    The read renders a JSON object; anything else is the host's own read failure
    (#947), not a daemon answer and not a run's state. It keeps the ``fault``
    status the hold exits on, carries the family's refusal envelope so it never
    classifies as a successful observation, and keeps the whole text: an
    oversized wake spills it to its exact source (``_render_wake_payload``).
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


def _direct_children(ctx: Any, materialize: bool = True) -> dict[str, dict[str, Any]]:
    """Read the existing durable direct-child projection for this nanny (a readiness check needs no artifacts)."""
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not task_id:
        return {}
    try:
        from ouroboros.task_status import find_child_tasks

        children = find_child_tasks(
            custody.custody_root(ctx), parent_task_id=task_id,
            root_task_id="", scope="direct", materialize_artifacts=materialize,
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
    ctx: Any, state: dict[str, Any], materialize: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect direct-child beacons and terminal transitions on one cursor (pure: the next cursor is
    returned, committed only by the acknowledgement of the wake that showed them)."""
    from ouroboros.task_tree_ledger import tree_ledger_row_id

    cursor = _coordination_cursor(state)
    children = _direct_children(ctx, materialize)
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
    new_attention = [row for row in attention if tree_ledger_row_id(row) not in seen]
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
            *(tree_ledger_row_id(row) for row in new_attention),
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


def _wake_result(payload: dict[str, Any], source: Optional[dict[str, Any]] = None) -> ToolResult:
    """One wake payload as its native result, keyed by the wake id it published.

    The id rides in producer meta because the ACK is keyed on it: the loop
    acknowledges the wake the result itself names, not every call that happens to
    be spelled ``delegate_wait``. A spilled wake names its exact source there too
    (``wake_source``), so no reader has to parse a shortened view to find it."""
    wake_id = str(payload.get("supervision_wake_id") or "")
    meta = {"supervision_wake_id": wake_id, **({"wake_source": source} if source else {})} if wake_id else {}
    return delegate_result(payload, meta=meta)


def _replay_fact(pending: dict[str, Any]) -> dict[str, Any]:
    """What a replay knows of itself: the same unacknowledged wake, first published then, the run not read now."""
    published = str(pending.get("created_at") or "")
    return {"fresh_observation": False, **({"published_at": published} if published else {}),
            "note": "The same wake, still unacknowledged; this call returned it without observing the run again."}


def _render_wake_payload(ctx: Any, payload: dict[str, Any], replay_of: Optional[dict[str, Any]] = None) -> ToolResult:
    """Render valid bounded JSON, spilling an oversized exact wake to artifacts.

    A replay (``replay_of``: its pending record) is shown marked and without text mail this attempt
    already holds; its spilled source stays the canonical wake, the same bytes at every replay of the
    attempt, so what was delivered of it before still counts (``deliver_wake``)."""

    canonical = json.dumps(payload, ensure_ascii=False, indent=2)
    if replay_of is not None:
        seen = getattr(ctx, "_loop_mailbox_seen_ids", None)
        payload = {"replay_of": _replay_fact(replay_of), **unseen_wake_view(
            payload, seen, ctx, getattr(ctx, "messages", None), replay_of.get("payload"))}
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
            data=canonical.encode("utf-8"),
            extension="json",
        )
    except Exception:
        # Keep the exact wake pending and let the ordinary outer truncation fail
        # acknowledgement rather than falsely claiming lossy delivery.
        _emit(ctx, "delegate_supervision_wake_source_failed", {
            "run_id": str(payload.get("run_id") or ""),
            "wake_id": wake_id,
            "total_chars": len(canonical),
        })
        return _wake_result(payload)
    events = payload.get("wake_events") if isinstance(payload.get("wake_events"), list) else []
    summaries = [_wake_event_summary(event) for event in events]
    # ``ok``/``host_code`` ride the fitted envelope: a wake that relays a REFUSED
    # observation must keep its classification when the exact payload spills, or a
    # refusal too large to inline would read as a successful wait.
    envelope: dict[str, Any] = {
        key: (str(value)[:600] if isinstance(value, str) else value)
        for key in ("replay_of", "status", "ok", "host_code", "run_id", "state", "last_seq", "reason",
                    "waiting_on_user", "async_questions_only", "continuation", "continuation_note",
                    "sleep", "cache_horizon_note", "leaf_live_input")
        if (value := payload.get(key)) not in (None, "")
    }
    envelope["supervision_wake_id"] = wake_id
    envelope["wake_events"] = summaries
    if isinstance(payload.get("coordination_context"), dict):
        envelope["coordination_context"] = payload["coordination_context"]
    envelope["wake_delivery"] = {
        "complete": False,
        "total_chars": len(canonical),
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
            **({"replay_of": payload["replay_of"]} if "replay_of" in payload else {}),
            "status": str(payload.get("status") or "wake_available")[:120],
            **({"ok": False, "host_code": str(payload.get("host_code") or "")}
               if payload.get("ok") is False else {}),
            "run_id": str(payload.get("run_id") or "")[:200],
            **{key: payload[key] for key in ("waiting_on_user", "async_questions_only", "continuation",
                                             "continuation_note", "sleep", "leaf_live_input") if key in payload},
            "supervision_wake_id": wake_id,
            "coordination_context": {"state": "available_in_full_wake_source"},
            "wake_delivery": envelope["wake_delivery"],
        }
        rendered = json.dumps(envelope, ensure_ascii=False, indent=2)
    return _wake_result(envelope, source)


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
            if str(replay.get("status") or "") in _QUIET_STATUSES and not replay.get("wake_events") and not replay.get("questions"):
                pending["acknowledged_at"] = utc_now_iso()
                state["last_acknowledged_wake"] = pending
                state["pending_wake"] = {}
                state["status"] = "awake"
                _save_state(ctx, state)
                return {}
        # The canonical replay; its view omits mail this attempt already drained (``_render_wake_payload``).
        return replay
    return {}


def unseen_wake_view(payload: dict[str, Any], seen: Any, ctx: Any = None, transcript: Any = None,
                     canonical: Any = None) -> dict[str, Any]:
    """The delivered wake without text mail this attempt's transcript already holds.

    Only the view narrows; the pending wake keeps every mailbox id for its ACK. Given the transcript, mail a
    still-present view showed only cut stays in it (``delegate_recovery.seen_without_cut_mail``).
    """
    events = payload.get("wake_events")
    if not seen or not isinstance(events, list):
        return payload
    if ctx is not None:
        from ouroboros.delegate_recovery import seen_without_cut_mail

        seen = seen_without_cut_mail(ctx, seen, canonical or payload, transcript)
    kept = [event for event in events if not any(item in seen for item in _text_mailbox_ids([event]))]
    return payload if len(kept) == len(events) else {**payload, "wake_events": kept}


def _text_mailbox_ids(events: Any) -> set[str]:
    """Only text carried by the delivered view: never omitted mail, a cut summary of it or controls."""
    return {str(item["msg_id"]) for item in events if isinstance(item, dict)
            and item.get("msg_id") and "text" in item and not item.get("text_omitted_chars")
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
    from ouroboros.sleep_custody import record_question_notice

    wake = payload.get("wake", payload) if isinstance(payload, dict) else {}
    record_question_notice(ctx, (wake.get("questions") or []) if isinstance(wake, dict) else [])
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


def deliver_wake(ctx: Any, row: dict[str, Any], view: dict[str, Any], delivered: str) -> None:
    """The loop's one wake seam: exposure and the deferred ACK follow only what the transcript holds whole.

    ``row`` published ``supervision_wake_id`` (a wake) or ``read_view`` (a read). A view holding the exact
    wake stages its exposure and defers its ACK to the next saved state. A spilled wake (its typed
    ``wake_source``) or a shortened view (the loop's retained ``result_source_ref``) waits instead for that
    exact source, read whole through ``read_file`` (``delegate_output.credit_exact_source``); its content is
    then the delivered wake. A summary, or a source merely recoverable, is never evidence it was read. Every
    pending source stays until read whole: each is an obligation a later read may complete."""
    from ouroboros.delegate_output import credit_exact_source

    try:
        published = (row.get("result_meta") or {}).get("tool_result_meta") or {}
        if published.get("supervision_wake_id"):
            inner, outer = published.get("wake_source"), view.get("result_source_ref")
            if isinstance(inner, dict) and inner.get("sha256"):
                _await_wake_source(ctx, inner, row, {})  # a summary is no window of its source
            elif view.get("result_partial") and isinstance(outer, dict) and outer.get("sha256"):
                _await_wake_source(ctx, outer, row, view)
            elif not view.get("result_partial"):
                _wake_text_delivered(ctx, delivered)
        elif isinstance(published.get("read_view"), dict) and not row.get("is_error"):
            sources = dict(getattr(ctx, "_wake_sources", None) or {})
            _wake_source_read(ctx, credit_exact_source(ctx, row, view, sources, "wake") if sources else None)
    except Exception:
        log.warning("Delegate wake delivery of %s not recorded; the wake stays pending", ctx.task_id, exc_info=True)


def _await_wake_source(ctx: Any, ref: dict[str, Any], row: dict[str, Any], view: dict[str, Any]) -> None:
    from ouroboros.delegate_output import credit_exact_source

    key = f"{ref.get('path')}|{ref['sha256']}"
    ctx._wake_sources = {**(getattr(ctx, "_wake_sources", None) or {}), key: dict(ref)}
    _wake_source_read(ctx, credit_exact_source(ctx, row, view, ctx._wake_sources, "wake", key))


def _wake_source_read(ctx: Any, done: Optional[tuple]) -> None:
    if done:
        key, _ref, content = done
        ctx._wake_sources = {name: ref for name, ref in ctx._wake_sources.items() if name != key}
        _wake_text_delivered(ctx, content)


def _wake_text_delivered(ctx: Any, text: str) -> None:
    """A complete text naming a wake: a spilled envelope's source waits to be read, the wake itself is shown."""
    from ouroboros.working_checkpoint import defer_content_ack

    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        payload = None
    delivery = payload.get("wake_delivery") if isinstance(payload, dict) else None
    if isinstance(delivery, dict) and delivery.get("complete") is False:
        if isinstance(delivery.get("source"), dict) and delivery["source"].get("sha256"):
            _await_wake_source(ctx, delivery["source"], {}, {})
        return
    stage_pending_wake_exposure(ctx, text)
    defer_content_ack(ctx, lambda: acknowledge_pending_wake(ctx, text))


def acknowledge_pending_wake(ctx: Any, delivered: Any = None) -> bool:
    """Acknowledge only run wakes named by the persisted delivered envelope.

    A refusal while a wake is pending is recorded with its reason
    (``delegate_supervision_wake_ack_refused``); that wake stays pending, so its replay is the retry."""
    try:
        payload = json.loads(delivered) if isinstance(delivered, str) else delivered
    except (TypeError, ValueError):
        return _ack_refused(ctx, None, "delivered_text_unparseable")
    ids = _wake_ids(payload)
    states = [_load_state(ctx, run_id) for run_id in checkpoints.read(_state_path(ctx))["runs"]]
    pending = [state for state in states if state.get("pending_wake")
               and not state["pending_wake"].get("acknowledged_at")]
    selected = [state for state in pending if state["pending_wake"].get("wake_id") in ids]
    if pending and not selected:  # e.g. an earlier wake's source read whole after that wake was acknowledged
        return _ack_refused(ctx, pending, "pending_wake_not_named", delivered_wake_ids=sorted(ids))
    for state in selected:  # each names its wake: ``payload`` is a parsed delivery that names it
        if reason := _acknowledge_run_wake(ctx, payload, state):
            return _ack_refused(ctx, [state], reason)
    return True


def _ack_refused(ctx: Any, states: Optional[list], reason: str, **facts: Any) -> bool:
    """Diagnostics only: one row per wake still pending, naming why this acknowledgement was refused."""
    try:
        for state in (states if states is not None else [
                _load_state(ctx, run_id) for run_id in checkpoints.read(_state_path(ctx))["runs"]]):
            pending = state.get("pending_wake") or {}
            if pending and not pending.get("acknowledged_at"):
                _emit(ctx, "delegate_supervision_wake_ack_refused", {
                    "run_id": str(state.get("run_id") or ""), "wake_id": str(pending.get("wake_id") or ""),
                    "reason": reason, "attempt_key": _attempt_key(ctx), **facts})
    except Exception:
        log.debug("Refused wake acknowledgement left no diagnostic row", exc_info=True)
    return False


def _acknowledge_run_wake(ctx: Any, payload: Any, state: dict) -> str:
    """Acknowledge one run's pending wake that the delivered ``payload`` names: ``""``, else why not."""
    pending = state["pending_wake"]
    expected = str(pending.get("wake_id") or "")
    attempt = _attempt_key(ctx)
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
        return "mailbox_acknowledgement_unwritten"
    if owner_ids and not acknowledge_task_messages(
        mailbox_root, task_id, sorted(owner_ids), wake_id=expected,
        attempt_key=attempt,
    ):
        return "mailbox_acknowledgement_unwritten"
    if mailbox_ids and not (global_ids or owner_ids) and not acknowledge_task_messages(
        mailbox_root, task_id, mailbox_ids, wake_id=expected, attempt_key=attempt,
    ):
        return "mailbox_acknowledgement_unwritten"
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
        return "acknowledgement_row_unwritten"
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
    return ""


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


def _leaf_live_input(entry: Any, gateway: Any) -> str:
    """The leaf route's declared live-input capability, stamped once on the wake. The engine catalog
    is cached per exact build (``declared_live_input``); an unreadable catalog or unknown route is
    ``unknown``, never a refusal, and an absent field means ``none``. The publication owns the transport.
    """
    from ouroboros.gateways.claudexor_run_events import declared_live_input

    try:
        value = declared_live_input(gateway, str(entry.route_id), timeout_sec=_TICK_SEC)
    except Exception:
        value = None
    return "unknown" if value is None else value


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


def _publish_wake_facts(ctx: Any, descriptor: dict[str, Any], woke_by: str, payload: dict[str, Any],
                        live_input: str) -> None:
    """Whole-sleep facts at the ONE durable publication point, every status (terminal
    included), before ``pending_wake`` is stored, so a replay repeats the recorded facts
    exactly: when the wait was armed, how long it slept, in which mode, and what woke it.
    A quiet status drops the read's own ``waited_sec``/``quiet_for_sec`` and its generic
    keep-watching/cancel note; a run paused on its question keeps the PAUSED note."""
    armed = descriptor.get("armed_epoch")
    if armed:
        payload["sleep"] = {
            "entered_at": str(descriptor.get("armed_at") or ""),
            "slept_sec": round(max(0.0, time.time() - float(armed)), 1),
            "mode": str(descriptor.get("mode") or "answered_at_call"),
            **({"woke_by": woke_by} if woke_by else {}),
        }
    if str(payload.get("status") or "") in _QUIET_STATUSES:
        payload.pop("waited_sec", None)
        payload.pop("quiet_for_sec", None)
        if payload.get("waiting_on_user"):
            from ouroboros.delegate_progress import paused_note
            payload["note"] = paused_note(payload.get("pending_interactions"), async_only=payload.get("async_questions_only") is True)
        else:
            payload.pop("note", None)
    payload.pop("cache_horizon_note", None)
    from ouroboros.tools.control import cache_horizon_note

    horizon = cache_horizon_note(ctx, _since_last_model_response_sec(ctx))
    if horizon:
        payload["cache_horizon_note"] = horizon
    payload["leaf_live_input"] = live_input


def run_wait_reason(ctx: Any) -> str:
    """What else ends a sleep on runs, beside ``model_sleep.wake_reason``'s own sources (mail, owner
    input and controls, a question of any run this task owns, the selected runs' terminals, the chosen
    time): the task's own cancel intent or spent deadline, and a direct child's attention beacon or
    terminal past the coordination cursor the last acknowledged wake committed. Reads only: no cursor
    moves here, and no child's artifacts are materialized for it."""
    controls = _control_wakes(ctx)
    if controls:
        return "control:" + ("cancelled" if controls[0]["type"] == "cancellation_intent" else "deadline")
    events, _cursor = _coordination_wakes(ctx, _load_state(ctx, ""), materialize=False)
    for event in events:
        beacon = event.get("beacon") if isinstance(event.get("beacon"), dict) else {}
        return "coordination:" + str(event.get("child_task_id") or beacon.get("task_id") or "unknown")
    return ""


def _choice(descriptor: dict[str, Any]) -> dict[str, Any]:
    """The sleep sources of a wait on runs, as ``model_sleep.wake_reason`` reads them: the selected
    runs, any addressed mail, the optional inspection time and the run-wait sources above."""
    runs = [str(run_id) for run_id in descriptor.get("runs") or ()]
    return {"senders": [], "tasks": [], "runs": runs, "wake_at": str(descriptor.get("wake_at") or ""),
            "any_mail": True, "run_wait": {"runs": runs, **({"hold": True} if descriptor.get("hold") else {})}}


def _target_run(ctx: Any, run_ids: list[str], woke_by: str) -> str:
    """The run a wake speaks for: the selected run its reason names, else the first selected run
    that ended or asks a question (as the body recorded it in custody), else the first selected run."""
    named = woke_by.split(":", 1)[1] if woke_by.startswith(("run:", "run_question:")) else ""
    if named in run_ids:
        return named
    try:
        from ouroboros import model_sleep

        owned = model_sleep._owned_runs(ctx)
        for run_id in run_ids:
            run = owned.get(run_id)
            if run is not None and (run.terminal_state in custody.TERMINAL_STATES or run.settled
                                    or model_sleep._pending_question(run)):
                return run_id
    except Exception:
        log.debug("Custody unreadable choosing the run a wake of %s speaks for", getattr(ctx, "task_id", ""),
                  exc_info=True)
    return run_ids[0]


def _replay(ctx: Any, run_ids: list[str]) -> tuple[Optional[ToolResult], dict[str, dict[str, Any]]]:
    """``(replay, states)``: an earlier wake of a selected run still unacknowledged, rendered marked
    (``replay_of``) — the only retry of a refused or lost acknowledgement, returned before any sleep is
    armed or any new wake published — else ``None`` and the runs' states."""
    states = {run_id: _load_state(ctx, run_id) for run_id in run_ids}
    for run_id, state in states.items():
        replay = _pending_payload(ctx, state)
        if replay:
            _emit(ctx, "delegate_supervision_wake_replayed", {
                "run_id": run_id, "wake_id": str(replay.get("supervision_wake_id") or "")})
            return _render_wake_payload(ctx, replay, state["pending_wake"]), states
    return None, states


def _publish(ctx: Any, descriptor: dict[str, Any], woke_by: str = "", *, questions: Any = None) -> ToolResult:
    """Publish one wake of the runs a wait selected and render it: the ONE wake record (``pending_wake``:
    its mail, question and coordination ids) that the delivered text's acknowledgement retires. An earlier
    wake still unacknowledged is returned instead (``_replay``). A terminal another collector is still
    retaining is read again on the body's beat until it is retained (the task's own cancel or deadline
    ends that); nothing else waits here."""
    run_ids = [str(run_id) for run_id in descriptor.get("runs") or ()]
    try:
        replay, states = _replay(ctx, run_ids)
    except (OSError, ValueError) as exc:
        return _fail("delegate_wait", "supervision_state_unknown", str(exc))
    if replay is not None:
        return replay
    target = _target_run(ctx, run_ids, woke_by)
    state = states[target]
    since = descriptor.get("since_seq")
    cursor = max(int(state.get("journal_cursor") or 0), int(since) if since is not None else 0)
    try:
        _ownership, held = custody.lookup(custody.custody_root(ctx), str(getattr(ctx, "task_id", "") or ""), target)
    except Exception:
        held = None
    from ouroboros.tools.delegate import _read_run
    from ouroboros.tools.delegate_terminal_evidence import retained_terminal_result

    # Exact retained bytes answer before any transport; one transport otherwise, only for a run with a route (no
    # route, no catalog read). An engine this cannot discover reads its live input as unknown.
    retained = None if _control_wakes(ctx) else retained_terminal_result(ctx, target, retry_money=True)
    gateway = _loop_gateway(held) if getattr(held, "route_id", "") and retained is None else None
    try:
        live_input = _leaf_live_input(held, gateway) if gateway is not None else "unknown"
        while True:  # a cancel or a spent deadline already there never waits behind another read
            raw = (json.dumps({"status": "no_progress", "run_id": target, "last_seq": cursor}) if _control_wakes(ctx)
                   else json.dumps(retained) if retained is not None else _read_run(ctx, target, cursor, gateway=gateway))
            payload, _readable, retained = *_payload(raw, target), None
            if (payload.get("status") != "observation_pending" or payload.get("reason") != "collection_in_progress"
                    or _control_wakes(ctx)):
                break
            time.sleep(_TICK_SEC)  # the collector that holds this terminal retains it first
    finally:
        if gateway is not None:
            try:
                gateway.close()
            except Exception:
                pass
    wakes = _addressed_wakes(ctx, state) + _control_wakes(ctx)
    coordination_events, next_coordination_cursor = _coordination_wakes(ctx, state)
    wakes += coordination_events
    if str(payload.get("status") or "") in _QUIET_STATUSES and not wakes:
        try:
            from ouroboros import model_sleep
            from ouroboros.store import acquisition_budget

            with acquisition_budget(0):  # the read answered quiet; custody may already hold the terminal
                terminal = model_sleep.terminal_readiness(ctx, {"runs": [target]})
        except Exception:
            terminal = []  # an unreadable custody source hides neither controls nor answers
        if terminal:
            payload.update(status="terminal_custody", terminal_facts=terminal)
    checkpoint = descriptor.get("checkpoint") if isinstance(descriptor.get("checkpoint"), dict) else {}
    if checkpoint:
        due = woke_by == "timeout" and not wakes and str(payload.get("status") or "") in _QUIET_STATUSES
        if due:
            payload = {"status": "inspection_checkpoint", "run_id": target,
                       "reason": str(checkpoint.get("reason") or ""), "last_seq": int(payload.get("last_seq") or cursor)}
        _emit(ctx, "delegate_supervision_checkpoint_consumed", {
            "run_id": target, "reason": str(checkpoint.get("reason") or ""),
            "consumed_by": "scheduled_checkpoint" if due else "real_event"})
    if not woke_by.startswith("control:"):
        from ouroboros.model_sleep import question_sources

        questions = question_sources(ctx) if questions is None else questions
        if questions:
            payload["questions"] = questions
    if wakes:
        payload["wake_events"] = wakes
    payload["coordination_context"] = coordination_live_context(ctx)
    _publish_wake_facts(ctx, descriptor, woke_by, payload, live_input)
    if descriptor.get("ignored_arguments"):
        payload["ignored_arguments"] = list(descriptor["ignored_arguments"])
    wake_id = uuid.uuid4().hex
    payload["supervision_wake_id"] = wake_id
    if isinstance(payload.get("last_seq"), int) and not isinstance(payload.get("last_seq"), bool):
        state["journal_cursor"] = max(cursor, payload["last_seq"])
    state["status"] = "wake_pending"
    state["pending_wake"] = {
        "wake_id": wake_id,
        "attempt_key": _attempt_key(ctx),
        "payload": payload,
        "mailbox_ids": [
            str(item.get("msg_id") or "") for item in wakes
            if str(item.get("msg_id") or "") and str(item.get("kind") or "") not in _LOOP_CONTROL_KINDS
        ],
        "seen_mailbox_ids": [str(item.get("msg_id") or "") for item in wakes if str(item.get("msg_id") or "")],
        "interaction_ids": sorted(_interaction_ids(payload)),
        "coordination_cursor": next_coordination_cursor,
        "created_at": utc_now_iso(),
    }
    _save_state(ctx, state, "wake_publication")  # another publication that landed first is the wake, rendered instead
    payload, wake_id = state["pending_wake"]["payload"], state["pending_wake"]["wake_id"]
    _emit(ctx, "delegate_supervision_wake_pending", {
        "run_id": target, "wake_id": wake_id, "payload": payload,
        "mailbox_ids": list(state["pending_wake"].get("mailbox_ids") or []),
        "interaction_ids": list(state["pending_wake"].get("interaction_ids") or []),
    })
    descriptor["wake"] = {"run_id": target, "wake_id": wake_id}
    return _render_wake_payload(ctx, payload)


def owed_wake(ctx: Any, descriptor: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """``(text, meta)`` of a sleeping ``delegate_wait``'s owed result (``native_task_wait.deliver_owed``),
    after the round top's mail drain: what woke it, read now from the same sources its sleep read, and
    published once. A held retry renders the wake this delivery already published, never a second one."""
    from ouroboros import model_sleep

    wake = descriptor.get("wake") if isinstance(descriptor.get("wake"), dict) else {}
    if wake:
        pending = _load_state(ctx, str(wake.get("run_id") or "")).get("pending_wake") or {}
        if pending.get("wake_id") == wake.get("wake_id") and isinstance(pending.get("payload"), dict):
            result = _render_wake_payload(ctx, pending["payload"])
            return result.text, dict(result.meta or {})
    try:
        woke_by = model_sleep.wake_reason(ctx, _choice(descriptor))
    except Exception:
        log.debug("Wake sources of %s unreadable at delivery; the wake says what it reads", ctx.task_id,
                  exc_info=True)
        woke_by = ""
    result = _publish(ctx, descriptor, woke_by)
    return result.text, dict(result.meta or {})


def _wait_in_slot(ctx: Any, descriptor: dict[str, Any]) -> ToolResult:
    """A turn with no continuation owner to sleep under (a Presence turn) waits in its slot for the same
    sources, read on the body's beat; its owner's Stop of the in-process turn ends the wait too."""
    from ouroboros import model_sleep
    from supervisor.owner_stop import in_process_stop_now

    chosen, key, until, woke_by = _choice(descriptor), "delegate:" + ",".join(descriptor["runs"]), 0.0, ""
    try:
        while not woke_by:
            if until - time.time() < _HOLD_MARK_SEC / 2:
                until = time.time() + _HOLD_MARK_SEC
                model_sleep.declare_in_slot_wait(ctx, key, until)
            try:
                woke_by = model_sleep.wake_reason(ctx, chosen)
            except Exception:
                log.debug("Wake sources of %s unreadable this beat", ctx.task_id, exc_info=True)
            if not woke_by and in_process_stop_now(ctx):
                woke_by = "control:owner_stop"
            if not woke_by:
                time.sleep(_TICK_SEC)
    finally:
        if until:
            model_sleep.end_in_slot_wait(ctx, key)
    return _publish(ctx, descriptor, woke_by)


def run_wait(ctx: Any, run_id: str | list[str], *, since_seq: Optional[int] = None,
             checkpoint_after_sec: Optional[int] = None, checkpoint_reason: str = "") -> ToolResult:
    """``delegate_wait`` as a sleep on runs: what is already there answers through this result; otherwise
    the task's ordinary sleep is armed (``model_sleep.request_default_sleep``: its process is released once
    nothing live keeps it) and this wait's result is owed to the first model request after the wake. A run
    this task holds only in this process (``started_uncustodied``: its STARTED row never landed) is one no
    durable source can wake a sleep on, so a wait selecting it answers now: with what is already there, else
    with that run's one addressed read."""
    from ouroboros import model_sleep
    from ouroboros.delegate_custody_current import run as current_run

    reason_text = str(checkpoint_reason or "").strip()
    ignored_note = ""
    if checkpoint_after_sec == 0 and not reason_text and not isinstance(checkpoint_after_sec, bool):
        # The schema's empty pair asks for no checkpoint: the omitted path, said in the wake.
        checkpoint_after_sec = None
        from ouroboros.tools.arg_feedback import ignored_argument_note

        ignored_note = ignored_argument_note(
            "checkpoint_after_sec", 0, "with an empty checkpoint_reason it asks for no checkpoint")
    if (checkpoint_after_sec is None) != (not reason_text):
        # The family's ONE refusal author, not a second literal envelope beside it.
        missing = "checkpoint_after_sec" if checkpoint_after_sec is None else "checkpoint_reason"
        return _fail(
            "delegate_wait", "checkpoint_requires_time_and_reason",
            f"got checkpoint_after_sec={checkpoint_after_sec!r}, checkpoint_reason={reason_text!r}: "
            f"{missing} is missing. Supply both for one proactive inspection, or omit both "
            "(0 with an empty reason also means none).",
        )
    run_ids = list(dict.fromkeys(run_id if isinstance(run_id, list) else [str(run_id)]))
    root, in_process = custody.custody_root(ctx), []
    for selected in run_ids:
        ownership, _entry = custody.lookup(root, str(ctx.task_id), selected)
        if ownership == custody.FOREIGN:
            from ouroboros.tools.delegate_terminal_evidence import retry_terminal_result

            return retry_terminal_result(ctx, selected)
        if ownership == custody.OWNED and current_run(root, selected) is None:
            in_process.append(selected)  # owned through this process's memo alone (``custody.lookup``)
    from ouroboros.delegate_recovery import restore_bound_handoff

    try:
        for selected in run_ids:
            restore_bound_handoff(ctx, selected)
        replay, _states = _replay(ctx, run_ids)  # S2: before any sleep is armed
    except (OSError, ValueError) as exc:
        return _fail("delegate_wait", "supervision_state_unknown", str(exc))
    if replay is not None:
        return replay
    delay = None if checkpoint_after_sec is None else max(1, min(604_800, int(checkpoint_after_sec)))
    descriptor: dict[str, Any] = {
        "tool": "delegate_wait", "runs": run_ids, "armed_at": utc_now_iso(), "armed_epoch": time.time(),
        "arguments": {key: value for key, value in {
            "run_id" if len(run_ids) == 1 and not isinstance(run_id, list) else "runs":
                run_ids[0] if len(run_ids) == 1 and not isinstance(run_id, list) else run_ids,
            "since_seq": since_seq, "checkpoint_after_sec": delay, "checkpoint_reason": reason_text}.items()
            if value not in (None, "")},
        **({"since_seq": int(since_seq)} if since_seq is not None else {}),
        **({"checkpoint": {"reason": reason_text, "after_sec": delay}} if delay is not None else {}),
        **({"ignored_arguments": [ignored_note]} if ignored_note else {}),
    }
    try:  # the sleep's selection is durable custody; an unknown run is still refused here
        chosen = model_sleep.selectors(ctx, runs=[rid for rid in run_ids if rid not in in_process],
                                       wake_after_sec=delay)
    except ValueError as exc:
        return _fail("delegate_wait", "run_ownership_unknown", str(exc))
    descriptor["wake_at"] = chosen["wake_at"]
    chosen.update(_choice(descriptor))
    if delay is not None:
        _emit(ctx, "delegate_supervision_checkpoint_scheduled", {
            "runs": run_ids, "wake_at": chosen["wake_at"], "reason": reason_text})

    def entered(**facts: Any) -> None:  # the durable trail of each wait that answers or sleeps (not a replay)
        _emit(ctx, "delegate_supervision_wait_entered", {"run_id": run_ids[0], "runs": run_ids, **facts})

    if in_process:
        # Neither the sleep nor the body reads its custody, and a cold sleep would end its one holder. What the sleep
        # would wake on answers first (``_target_run``: a selected run's terminal or question in custody); listed
        # first, the in-process run is the read otherwise. Its terminal read re-publishes the start's own row or adopts
        # the row its invocation was bound under; the money stays on that one row (``collect_and_deliver_terminal``).
        try:
            woke_by = model_sleep.wake_reason(ctx, chosen)
        except Exception:
            woke_by = ""  # an unreadable source hides neither the selected runs nor this read
        entered(mode="answered_at_call", held_in_process=in_process)
        return _publish(ctx, {**descriptor, "runs": in_process + [rid for rid in run_ids if rid not in in_process]},
                        woke_by)
    if not callable(getattr(ctx, "owner_wait_callback", None)):
        entered(mode="in_slot")
        return _wait_in_slot(ctx, descriptor)
    from ouroboros.native_task_wait import armed_conflict

    if conflict := armed_conflict(ctx, native=True):
        return _fail("delegate_wait", "wait_already_armed", conflict)
    try:
        outcome = model_sleep.request_default_sleep(ctx, chosen)
    except ValueError as exc:
        return _fail("delegate_wait", "sleep_refused", str(exc))
    entered(outcome=outcome["reason"], **{key: outcome[key] for key in ("sleep_id", "mode") if outcome.get(key)})
    if outcome["reason"] == "ready":  # nothing parks: what is there answers through this result
        return _publish(ctx, descriptor, str(outcome.get("woke_by") or ""), questions=outcome.get("questions", []))
    descriptor.update(sleep_id=outcome["sleep_id"], mode=outcome["mode"])
    ctx._pending_task_wait = descriptor  # owed to the first request after the wake (``native_task_wait.deliver_owed``)
    return delegate_result({
        "reason": "sleep_armed", "sleep_id": outcome["sleep_id"], "mode": outcome["mode"], "slept": False,
        "runs": run_ids, **({"wake_at": chosen["wake_at"]} if chosen["wake_at"] else {}),
        **{key: outcome[key] for key in ("kept_process", "membership_gaps") if outcome.get(key)},
        "note": ("The sleep begins after this tool batch completes. A selected run's terminal, a question from a "
                 "run you own, a direct child's attention or terminal, addressed mail, owner input, a cancel or "
                 "your deadline" + (" or your checkpoint" if delay is not None else "") + " wakes you; this "
                 "wait's result then reaches your next request without another call.")})


def hold_wake(ctx: Any, run_ids: list[str], *, woke: bool = False) -> Optional[ToolResult]:
    """The unknown-provider hold's wake (``delegate_hold``): an unacknowledged wake of its runs replayed,
    else one published once a source of its sleep is ready (or its own park woke), else ``None``."""
    from ouroboros import model_sleep

    descriptor = {"tool": "delegate_wait", "runs": [str(run_id) for run_id in run_ids], "hold": True}
    if not woke:
        try:
            replay, _states = _replay(ctx, descriptor["runs"])
            if replay is not None:
                return replay
            woke_by = model_sleep.wake_reason(ctx, _choice(descriptor))
        except Exception:
            log.debug("Hold sources of %s unreadable; it keeps holding", ctx.task_id, exc_info=True)
            return None
        return _publish(ctx, descriptor, woke_by) if woke_by else None
    return _publish(ctx, descriptor, "")


def arm_hold_sleep(ctx: Any, run_ids: list[str]) -> dict[str, Any]:
    """The hold's own warm sleep on its runs, armed by the host without a model call (``delegate_hold``)."""
    from ouroboros import model_sleep

    return model_sleep.request_sleep(ctx, _choice({"runs": run_ids, "hold": True}), model_sleep.MODE_WARM)


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
    """``delegate_wait``: a sleep on runs (``run_wait``); hidden ``wait_sec`` is accepted only for old
    transcripts and waits on nothing."""

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

    try:
        return run_wait(ctx, runs if runs is not None else run_id, since_seq=since_seq,
                        checkpoint_after_sec=checkpoint_after_sec, checkpoint_reason=checkpoint_reason)
    except OSError as exc:  # Only a save failure names supervision_stage; other errors keep their handling.
        if not (stage := getattr(exc, "supervision_stage", "")):
            raise
        detail, facts = checkpoints.save_failure(exc, stage)
        return _fail("delegate_wait", "supervision_state_unsaved", detail, **facts)


__all__ = [
    "acknowledge_pending_wake", "stage_pending_wake_exposure", "delegate_wait_entry", "run_wait", "owed_wake",
    "supervision_checkpoint", "coordination_live_context", "unseen_wake_view", "deliver_wake", "run_wait_reason",
    "hold_wake", "arm_hold_sleep",
]
