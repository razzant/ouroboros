"""Task services' end as the answer's evidence, recorded once per change.

An explicit stop, a replacement and the end of the task carry the same launch facts (``service_outcomes``).
The completion pass records each confirmed end once. Delivery binds the evidence; the author reads its redacted
view and the card and Logs receive the same safe System row.
Ordinary and forced completion and loop exit own when this happens and its effect on the answer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterator
from typing import Any, Dict, List, Tuple

from ouroboros import service_outcomes
from ouroboros.utils import truncate_within_limit, utc_now_iso

log = logging.getLogger("ouroboros.loop")

# A stop's own answer -> the event that records it; a row without a known answer is not called stopped.
_KINDS = {"stopped": "services_stopped", "kept": "services_kept", "cleanup_pending": "service_finalization_error",
          "outputs_updated": "service_outputs_updated"}
_EVENT_ORDER = ("services_stopped", "service_outputs_updated", "service_finalization_error", "services_kept")
_LABELS = {"services_stopped": "Background services stopped", "services_kept": "Background services kept",
           "service_finalization_error": "Background service cleanup unconfirmed",
           "service_outputs_updated": "Background service outputs updated"}
_NAMES_SHOWN = 3
_COLLECTION_GAP = "Background service outcomes could not be collected"
_STOP_ORIGIN = {"explicit": "requested stop", "replacement": "for replacement", "task_end": "at task end"}
_LOG_WORDS = {"saved": "log saved", "absent": "no log found", "retained": "log retained in place (too large to archive)",
              "not_saved": "log not saved (source kept)", "unconfirmed": "log capture unconfirmed",
              "backend_only": "log not archived (Docker)", "not_attempted": "log capture not attempted"}
_OUTPUT_WORDS = {"saved": "saved", "partial": "partly saved", "absent": "absent", "not_saved": "not saved",
                 "not_attempted": "capture not attempted", "unconfirmed": "capture unconfirmed"}


def _observations(llm_trace: dict) -> Iterator[tuple[str, list[dict], Any]]:
    for event in llm_trace.get("verification_events") or []:
        if isinstance(event, dict) and (kind := str(event.get("kind") or "")) in _LABELS:
            yield kind, [row for row in event.get("services") or () if isinstance(row, dict)], event.get("error")


def evidence(llm_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the stable, answer-relevant part of service finalization events, not their presentation."""
    return [{"kind": kind, "services": [service_outcomes.identity(service) for service in services],
             "error": truncate_within_limit(str(error or ""), 2000)}
            for kind, services, error in _observations(llm_trace)]


def _signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def _observation_key(kind: str, service: dict) -> str:
    """Output-only retries cannot record a second confirmed end. A legacy key may name many starts."""
    if kind == "services_stopped" and (launch := service.get("launch_id")):
        return _signature([kind, service.get("task_id"), launch])
    if kind == "service_outputs_updated":
        files = [{key: value for key, value in row.items() if key != "registered"}
                 for row in service_outcomes.public_outputs(service.get("output_outcomes"))]
        return _signature([kind, service.get("task_id"), service.get("launch_id"), files, service.get("artifact_output_owed")])
    projected = service_outcomes.identity(service)
    projected.pop("stop_source", None)
    return _signature([kind, projected])


def _new_events(services: list, seen: set[str]) -> list[dict]:
    groups: Dict[str, List[Dict[str, Any]]] = {kind: [] for kind in _EVENT_ORDER}
    for service in services:
        if not isinstance(service, dict):
            continue
        kind = _KINDS.get(str(service.get("lifecycle") or ""), "service_finalization_error")
        signature = _observation_key(kind, service)
        if signature in seen:
            continue
        seen.add(signature)
        if kind == "services_stopped" and service.get("artifact_output_owed"):
            seen.add(_observation_key("service_outputs_updated", service))  # repeating the same gap is not a change
        if kind == "service_finalization_error":
            service = service_outcomes.identity(service)
        elif isinstance(log := service.get("log_finalization"), dict):
            # A resumed loop can still hand off an older observation; its log is a gap or a recorded handle.
            service = {**service, "log_finalization": service_outcomes.recorded_log_result(log)}
        groups[kind].append(service)
    return [{"kind": kind, "services": rows,
             **({"error": "service cleanup was not confirmed"} if kind == "service_finalization_error" else {})}
            for kind, rows in groups.items() if rows]


def finalize(ctx: Any, llm_trace: Dict[str, Any], tools: Any = None) -> bool:
    """Publish earlier tool stops, then this teardown pass. Only new observations change delivery evidence."""
    if ctx.drive_root is None or not ctx.task_id:
        return False
    from ouroboros.loop_messages import _emit_checkpoint_event

    error = None
    try:
        from ouroboros.tools.services import stop_task_services

        current = stop_task_services((ctx.tools if tools is None else tools)._ctx)
    except Exception as exc:  # noqa: BLE001 -- this pass failed; the earlier stops still stand
        log.debug("Failed to stop task services", exc_info=True)
        error = {"kind": "service_finalization_error", "services": [], "error": type(exc).__name__}
        current = []
    previous = service_outcomes.take_stop_observations(llm_trace)
    if not (previous or current or error):
        return False
    recorded = llm_trace.setdefault("verification_events", [])
    prior = evidence(llm_trace)
    seen = {_observation_key(event["kind"], service) for event in prior for service in event["services"]}
    seen.update(_observation_key("service_outputs_updated", service) for event in prior if event["kind"] == "services_stopped"
                for service in event["services"] if service.get("artifact_output_owed"))
    events = []
    for service in previous:
        events.extend(_new_events([service], seen))
    events.extend(_new_events(current, seen))
    if error and not any(row["kind"] == error["kind"] and not row["services"] for row in prior):
        events.append(error)
    if not events:
        return False
    recorded.extend(events)
    for row in checkpoints(events):
        try:
            _emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, row)
        except Exception:
            log.debug("Service finalization row of %s not shown", ctx.task_id, exc_info=True)
    return True


def _named(services: List[Dict[str, Any]]) -> str:
    names = [_label(service.get("name")) or "service" for service in services]
    more = len(names) - _NAMES_SHOWN
    return ", ".join(names[:_NAMES_SHOWN]) + (f" and {more} more" if more > 0 else "")


def _label(value: Any) -> str:
    """A plain service name, launch ID or artifact basename, never declaration/path prose."""
    return value if isinstance(value, str) and re.fullmatch(r"[\w .-]{1,80}", value) else ""


def _launch_line(kind: str, service: dict) -> str:
    service = service_outcomes.public(service)
    name = _label(service.get("name")) or "service"
    if launch := _label(service.get("launch_id")):
        name += f" ({launch.removeprefix('launch-').removeprefix('service-')[:10]})"
    label = "Background service stopped" if kind == "services_stopped" else _LABELS[kind]
    parts = [f"{label}: {name}"]
    if kind != "service_outputs_updated":
        parts.extend((_STOP_ORIGIN.get(service.get("stop_source")),
                      _LOG_WORDS.get(service.get("log_outcome"), _LOG_WORDS["unconfirmed"])))
    for output in service.get("output_outcomes", ()):
        status = _OUTPUT_WORDS.get(output.get("status"), _OUTPUT_WORDS["unconfirmed"])
        names = [name for item in output.get("artifacts") or () if (name := _label(item.get("name")))]
        shown = ", ".join(names[:_NAMES_SHOWN])
        if len(names) > _NAMES_SHOWN:
            shown += f" and {len(names) - _NAMES_SHOWN} more"
        parts.append(f"output {output['index']} {status}" + (f": {shown}" if shown else ""))
    return " · ".join(part for part in parts if part)


def _checkpoint(kind: str, line: str, row_id: str) -> dict:
    return {"checkpoint_kind": kind, "role": "system", "system_type": "task_checkpoint",
            "card_row_id": f"service:{row_id}", "detail": line, "content": line, "text": line}


def checkpoints(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The System rows that tell recorded events in the task's card and Logs, live and reloaded.

    One line per typed launch. Only the producer's safe labels and outcomes ride to the card, never log content,
    raw paths or exception text. An unavailable collection describes no individual service. Old events keep their
    compact group label and, where available, the reported gaps.
    """
    rows = []
    for event in events:
        services = [service for service in event.get("services") or [] if isinstance(service, dict)]
        kind = event["kind"]
        if not services and kind == "service_finalization_error" and event.get("error"):
            rows.append(_checkpoint(kind, _COLLECTION_GAP, _signature([kind, "collection"])))
        for service in services:
            if "log_outcome" in service:
                rows.append(_checkpoint(kind, _launch_line(kind, service), _observation_key(kind, service)))
        services = [service for service in services if "log_outcome" not in service]
        if not services:
            continue
        gaps = [(label, [service for service in services if lost(service)]) for label, lost in (
            ("log not saved", service_outcomes.log_unsaved),
            ("declared outputs not saved", lambda row: row.get("artifact_output_failed") and not row.get("artifact_output_unknown")),
            ("declared output capture unconfirmed", lambda row: row.get("artifact_output_unknown")))]
        line = " · ".join([_LABELS[kind], *(f"{label}: {_named(lost)}" for label, lost in gaps if lost)])
        rows.append(_checkpoint(kind, line, _signature([kind, [service_outcomes.identity(row) for row in services]])))
    return rows


def public_event(event: dict[str, Any]) -> dict[str, Any]:
    """The author and the exported ledger read the same facts; unrelated events pass through."""
    kind = event.get("kind")
    if not isinstance(kind, str) or kind not in _LABELS:
        return event
    services = event.get("services")
    rows = [service_outcomes.display(row) for row in services if isinstance(row, dict)] if isinstance(services, list) else []
    return {**event, "services": rows,
            **({"error": _COLLECTION_GAP} if event.get("error") else {})}


def display(llm_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Only the stored service facts; no source I/O and no row hidden by a log gap."""
    return [public_event({"kind": kind, "services": services, **({"error": error} if error else {})})
            for kind, services, error in _observations(llm_trace)]


def publish_retained_outputs(drive_root: Any, observation: dict) -> None:
    """A post-task capture uses the same checkpoint in history and Logs; never publish to an invented room.

    The checkpoint kind follows the observation's own lifecycle: the retained capture that speaks a
    launch's first confirmed end speaks it as a stop (``services_stopped``, one card row per launch);
    a later capture speaks only as ``service_outputs_updated``. The durable rows land first
    (``_emit_checkpoint_event`` without a queue); their one live frame then rides the host's existing
    transports (``_push_retained_frame``), carrying the producer's own host-determined audience."""
    from ouroboros.loop_messages import _emit_checkpoint_event
    from ouroboros.task_custody import canonical_owner_root
    from ouroboros.task_status import load_effective_task_result
    from supervisor.log_addressing import address_task_event

    task_id = observation["task_id"]
    try:
        root = canonical_owner_root(drive_root)
        task = load_effective_task_result(root, task_id, materialize_artifacts=False)
        kind = _KINDS.get(str(observation.get("lifecycle") or ""), "service_outputs_updated")
        [row] = checkpoints([{"kind": kind, "services": [observation]}])
        payload = address_task_event({task_id: {"task": task}}, root, {**row, "task_id": task_id})
        if type(payload.get("chat_id")) is int:
            stamped = {"ts": utc_now_iso(), **payload}  # the durable row and the frame are one event
            _emit_checkpoint_event(None, task_id, root / "logs", stamped)
            _push_retained_frame(root, {"type": "task_checkpoint", "task_id": task_id, **stamped})
    except Exception:
        log.debug("Retained output checkpoint not shown for %s", task_id, exc_info=True)


def _push_retained_frame(root: Any, frame: dict) -> None:
    """The one live frame of a retained checkpoint row that is already durable, on the host's existing
    transports, never a new queue. In a worker process: its bound bus, through that bus's owner
    (``worker_chat_lane.publish_recorded_row``); the supervisor's ``_handle_log_event`` addresses and
    pushes the frame. In the server process: the bridge, as ``post_task_checkpoint`` pushes a finalized
    cost. The frame already carries the producer's host-determined audience, never the caller's room.
    Where neither transport exists (a headless or benchmark process) the durable rows stay the record.
    Both log sinks suppress the appended row's copy (``task_checkpoint``), so one publish stays one row
    set and one frame."""
    try:
        from supervisor.worker_chat_lane import publish_recorded_row

        if publish_recorded_row(frame):
            return
        from supervisor.message_bus import try_get_bridge

        bridge = try_get_bridge()
        if bridge is not None:
            from supervisor.log_addressing import address_handler_push

            bridge.push_log(address_handler_push(root, dict(frame)))
    except Exception:
        log.debug("Retained checkpoint frame not delivered; its durable rows stand", exc_info=True)


def display_text(llm_trace: Dict[str, Any]) -> str:
    shown = display(llm_trace)
    return json.dumps(shown, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str) if shown else ""


def disclosure(llm_trace: Dict[str, Any], *, shown: str) -> Tuple[str, str]:
    """``(fingerprint, message)``: the author's ``[SERVICE_FINALIZATION_EVIDENCE]`` once per change.

    ``shown`` is the fingerprint last disclosed; the message is empty while the evidence is unchanged or absent.
    Identity is the full stable evidence, never its display: two changes a compact tail hides can show equal bytes.
    """
    projection = evidence(llm_trace)
    fingerprint = _signature(projection) if projection else shown
    if fingerprint == shown:
        return shown, ""
    return fingerprint, ("[SERVICE_FINALIZATION_EVIDENCE]\n"
                         "Task services were finalized before completion. Incorporate this "
                         "evidence and disclose any failure honestly:\n"
                         + display_text(llm_trace))
