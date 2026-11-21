"""Task services' end as the answer's evidence, recorded once per change.

An explicit stop, a replacement and the end of the task carry the same launch facts (``service_outcomes``).
The completion pass records each confirmed end once. Delivery binds the evidence; the author reads its redacted
view and the card and Logs receive the same safe System row: one row per (task, launch), which a later file
result revises in place (``_row_identity``). A stop whose receipt its own delivered tool result already showed
is evidence from that result on, so recording it buys no second answer (``evidence``).
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
_LOG_SOURCE_WORDS = {"full": "masked log available", "tail": "masked log tail available",
                     "tail_masking_unverified": "no readable log: tail masking unverified",
                     "masked_source_not_verified": "no readable log: masking unverified",
                     "masked_source_not_retained": "no readable log: masked source not retained",
                     "legacy_source_not_published": "no readable log: legacy source not published"}
_OUTPUT_WORDS = {"saved": "saved", "partial": "partly saved", "absent": "absent", "not_saved": "not saved",
                 "not_attempted": "capture not attempted", "unconfirmed": "capture unconfirmed"}


def _observations(llm_trace: dict) -> Iterator[tuple[str, list[dict], Any]]:
    for event in llm_trace.get("verification_events") or []:
        if isinstance(event, dict) and (kind := str(event.get("kind") or "")) in _LABELS:
            yield kind, [row for row in event.get("services") or () if isinstance(row, dict)], event.get("error")


def _projected(kind: str, services: list, error: Any) -> Dict[str, Any]:
    return {"kind": kind, "services": [service_outcomes.identity(service) for service in services],
            "error": truncate_within_limit(str(error or ""), 2000)}


def _recorded(llm_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [_projected(kind, services, error) for kind, services, error in _observations(llm_trace)]


def _seen(prior: List[Dict[str, Any]]) -> set[str]:
    seen = {_observation_key(event["kind"], service) for event in prior for service in event["services"]}
    seen.update(_observation_key("service_outputs_updated", service) for event in prior if event["kind"] == "services_stopped"
                for service in event["services"] if service.get("artifact_output_owed"))
    return seen


def evidence(llm_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the stable, answer-relevant part of service finalization events, not their presentation.

    A stop whose receipt its own delivered tool result already showed the author counts from that result on, in the
    shape the completion pass will record: recording it later changes nothing an answer written since was bound to.
    A stop the author has not seen (it outlived or failed its call, or retired a namesake) stays new evidence."""
    recorded = _recorded(llm_trace)
    shown = service_outcomes.shown_stop_observations(llm_trace)
    if not shown:
        return recorded
    seen = _seen(recorded)
    return recorded + [_projected(event["kind"], event["services"], event.get("error"))
                       for service in shown for event in _new_events([service], seen)]


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
    """Publish earlier tool stops, then this teardown pass. Only new observations change delivery evidence.

    Typed ends share the retained publication door, including a missing stop-time write. Neither sink alone
    settles the launch record. Publication debt is independent of what the author already saw in the tool result."""
    if ctx.drive_root is None or not ctx.task_id:
        return False
    from ouroboros.loop_messages import _emit_checkpoint_event

    error = None
    try:
        from ouroboros.tools.services import stop_task_services

        with service_outcomes.collecting():  # this pass tells the row of every stop it makes
            current = stop_task_services((ctx.tools if tools is None else tools)._ctx)
    except Exception as exc:  # noqa: BLE001 -- this pass failed; the earlier stops still stand
        log.debug("Failed to stop task services", exc_info=True)
        error = {"kind": "service_finalization_error", "services": [], "error": type(exc).__name__}
        current = []
    previous = service_outcomes.take_stop_observations(llm_trace)
    if not (previous or current or error):
        return False
    owed = {_launch_revision(service): service.pop(service_outcomes.ROW_OWED) for service in (*previous, *current)
            if isinstance(service, dict) and service_outcomes.ROW_OWED in service}  # the stop's own fact, never recorded
    recorded = llm_trace.setdefault("verification_events", [])
    prior = _recorded(llm_trace)
    seen = _seen(prior)
    events = []
    for service in previous:
        events.extend(_new_events([service], seen))
    events.extend(_new_events(current, seen))
    if error and not any(row["kind"] == error["kind"] and not row["services"] for row in prior):
        events.append(error)
    if not events:
        return False
    recorded.extend(events)
    for row, service in _rows(events):
        if (service is not None and service_outcomes.output_launch(service)
                and row["checkpoint_kind"] in ("services_stopped", "service_outputs_updated")):
            from ouroboros.tools.kept_service_outputs import tell_own_stop

            tell_own_stop(ctx.drive_root, service, owed.get(_launch_revision(service)), ctx.event_queue)
        else:
            try:
                _emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, row)
            except Exception:
                log.debug("Service finalization row of %s not shown", ctx.task_id, exc_info=True)
    return True


def _launch_revision(service: dict) -> tuple:
    return service.get("task_id"), service.get("launch_id"), service.get("row_revision")


def _named(services: List[Dict[str, Any]]) -> str:
    names = [_label(service.get("name")) or "service" for service in services]
    more = len(names) - _NAMES_SHOWN
    return ", ".join(names[:_NAMES_SHOWN]) + (f" and {more} more" if more > 0 else "")


def _label(value: Any) -> str:
    """A plain service name, launch ID or artifact basename, never declaration/path prose."""
    return value if isinstance(value, str) and re.fullmatch(r"[\w .:-]{1,80}", value) else ""


def _launch_line(kind: str, service: dict) -> str:
    """One launch is one card row, revised in place (``_row_identity``). A later file result is that row's next
    revision under its own label, never a second stop line: once the end is confirmed it says the launch whole
    again (ended, the log's fate, every file). Who asked for the stop is said only by the stop's own observation."""
    service = service_outcomes.display(service)
    name = _label(service.get("name")) or "service"
    if launch := _label(service.get("launch_id")):
        name += f" ({launch.removeprefix('launch-').removeprefix('service-')[:10]})"
    label = "Background service stopped" if kind == "services_stopped" else _LABELS[kind]
    parts = [f"{label}: {name}"]
    log_words = _LOG_WORDS.get(service.get("log_outcome"), _LOG_WORDS["unconfirmed"])
    log = service.get("log_finalization", {})
    source_words = _LOG_SOURCE_WORDS.get(log.get("source_scope") or log.get("source_unavailable"))
    if kind != "service_outputs_updated":
        parts.extend((_STOP_ORIGIN.get(service.get("stop_source")), log_words, source_words))
    elif service.get("state") in ("exited", "stopped"):
        parts.extend(("ended", log_words, source_words))
    for output in service.get("output_outcomes", ()):
        status = _OUTPUT_WORDS.get(output.get("status"), _OUTPUT_WORDS["unconfirmed"])
        names = [name for item in output.get("artifacts") or () if (name := _label(item.get("name")))]
        shown = ", ".join(names[:_NAMES_SHOWN])
        if len(names) > _NAMES_SHOWN:
            shown += f" and {len(names) - _NAMES_SHOWN} more"
        parts.append(f"output {output['index']} {status}" + (f": {shown}" if shown else ""))
    return " · ".join(part for part in parts if part)


def _checkpoint(kind: str, line: str, row_id: str, revision: int = 0) -> dict:
    return {"checkpoint_kind": kind, "role": "system", "system_type": "task_checkpoint",
            "card_row_id": f"service:{row_id}", **({"card_row_revision": revision} if revision else {}),
            "detail": line, "content": line, "text": line}


def _row_identity(kind: str, service: dict) -> tuple[str, int]:
    """``(row id, revision)``: a confirmed end and every later file result of one typed launch are one card row,
    ordered by the launch record's own count (``row_revision``, kept in its declaration). What is new is decided
    elsewhere (``_observation_key``). An unconfirmed stop, or a row naming no launch, keeps its observation's row."""
    launch = service_outcomes.output_launch(service) if kind in ("services_stopped", "service_outputs_updated") else None
    if launch is None:
        return _observation_key(kind, service), 0
    revision = service.get("row_revision")
    return _signature(["launch", *launch]), revision if type(revision) is int and revision > 1 else 1


def checkpoints(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The System rows that tell recorded events in the task's card and Logs, live and reloaded.

    One line per typed launch. Only the producer's safe labels and outcomes ride to the card, never log content,
    raw paths or exception text. An unavailable collection describes no individual service. Old events keep their
    compact group label and, where available, the reported gaps.
    """
    return [row for row, _service in _rows(events)]


def _rows(events: List[Dict[str, Any]]) -> Iterator[tuple[dict, dict | None]]:
    """Each row with the one typed launch observation it tells; None for a collection gap or an old group row."""
    for event in events:
        services = [service for service in event.get("services") or [] if isinstance(service, dict)]
        kind = event["kind"]
        if not services and kind == "service_finalization_error" and event.get("error"):
            yield _checkpoint(kind, _COLLECTION_GAP, _signature([kind, "collection"])), None
        for service in services:
            if "log_outcome" in service:
                yield _checkpoint(kind, _launch_line(kind, service), *_row_identity(kind, service)), service
        services = [service for service in services if "log_outcome" not in service]
        if not services:
            continue
        gaps = [(label, [service for service in services if lost(service)]) for label, lost in (
            ("log not saved", service_outcomes.log_unsaved),
            ("declared outputs not saved", lambda row: row.get("artifact_output_failed") and not row.get("artifact_output_unknown")),
            ("declared output capture unconfirmed", lambda row: row.get("artifact_output_unknown")))]
        line = " · ".join([_LABELS[kind], *(f"{label}: {_named(lost)}" for label, lost in gaps if lost)])
        yield _checkpoint(kind, line, _signature([kind, [service_outcomes.identity(row) for row in services]])), None


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


RETAINED_TOLD = ("recorded", "no_room")  # nothing more is owed to the producer's room


def publish_retained_outputs(drive_root: Any, observation: dict, *, ts: str = "", own: bool = False,
                             event_queue: Any = None) -> str:
    """Store the same safe checkpoint in both installation sinks, even for a separate execution drive.

    A retry addresses its own (time, task, row, revision). An existing dialogue row does not acknowledge Logs.
    The own pass also owes Logs when there is no room, and uses its addressed live endpoint on the first attempt;
    an outside capture never invents a room or emits a second frame for an already stored dialogue row.
    Unknown keeps the launch's pending record; no process operation is involved."""
    from ouroboros import dialogue_store
    from ouroboros.projects_registry import RoomResolver
    from ouroboros.task_custody import canonical_owner_root, fence_publication
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import effective_task_result
    from ouroboros.utils import append_jsonl, emit_log_event
    from supervisor.log_addressing import address_task_event

    task_id, stamped, already_in_room = observation["task_id"], None, False
    try:
        root = canonical_owner_root(drive_root)
        # Both reads are strict: what cannot be read raises here and stays unknown, never an empty answer.
        task = effective_task_result(root, load_task_result(root, task_id, strict=True) or {}, materialize_artifacts=False)
        bound = RoomResolver(root, strict=True)(task_id, task.get("parent_task_id") or "", task.get("root_task_id") or "")
        kind = _KINDS.get(str(observation.get("lifecycle") or ""), "service_outputs_updated")
        [row] = checkpoints([{"kind": kind, "services": [observation]}])
        payload = {**row, "task_id": task_id}
        if callable(stamp := getattr(event_queue, "stamp", None)):
            stamp({"type": "log_event", "data": payload})
        payload = address_task_event({task_id: {"task": task}}, root, payload)
        if bound and type(payload.get("chat_id")) is not int:
            payload["chat_id"] = int(bound)
        room = payload.get("chat_id")
        if type(room) is not int and not own:
            return "no_room"
        stamped = {"ts": ts or utc_now_iso(), "type": "task_checkpoint", **payload}
        if type(room) is int:
            already_in_room = bool(ts) and any(_same_checkpoint(held.row, stamped) for held in
                dialogue_store.rows_by_identity(root, room, ts, stream="progress"))
            if not already_in_room:
                fence_publication()
                saved = dialogue_store.append(root, "progress", stamped, idempotent=True, emit_live=False,
                                              before_write=fence_publication)
                if event_queue is None and not saved.duplicate:
                    _push_retained_frame(root, stamped)  # the card's durable row does not wait for Logs
        logs = root / "logs" / "events.jsonl"
        if not ((already_in_room or (ts and type(room) is not int)) and _checkpoint_logged(logs, stamped)):
            fence_publication()
            if not append_jsonl(logs, stamped, require_lock=True, ensure_record_boundary=True, fsync=True,
                                before_write=fence_publication):
                return "unknown"
    except Exception:
        log.debug("Retained output checkpoint not shown for %s", task_id, exc_info=True)
        return "unknown"
    finally:
        if event_queue is not None and stamped is not None and not already_in_room:
            emit_log_event(event_queue, stamped, log_label="service checkpoint")
    return "recorded"


def _same_checkpoint(row: Any, expected: dict) -> bool:
    return isinstance(row, dict) and all(row.get(key) == expected.get(key) for key in (
        "type", "ts", "task_id", "card_row_id", "card_row_revision"))


def _checkpoint_logged(path: Any, row: dict) -> bool:
    from ouroboros.jsonl_tail import read_rotated_jsonl_entries

    def match(candidate):
        return _same_checkpoint(candidate, row)

    entries, gaps = read_rotated_jsonl_entries(path, path.parent.parent / "archive", path.stem, 1, match, include_gaps=True)
    if any(map(match, entries)):
        return True
    if "unreadable_source" in gaps:
        raise OSError("checkpoint log source unreadable")
    return False


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
