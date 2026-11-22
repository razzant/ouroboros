"""Evidence of a named service stop: the safe stop facts, the bounded identity
they share with teardown rows, and the answer's stop rows. This module holds no
live process or control door."""
from __future__ import annotations

import logging
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Dict
from urllib.parse import quote

from ouroboros.artifacts import copy_file_to_task_artifacts, immutable_artifact_name, task_id_for_artifacts
from ouroboros.tool_access_paths import canonical_data_root
from ouroboros.utils import path_for_prose, truncate_within_limit

log = logging.getLogger(__name__)


def _version(ctx, record: dict) -> dict:
    """Freeze an admitted output under a separate key, leaving its ordinary registration intact."""
    name, path, digest = record["name"], Path(record["path"]), record["sha256"]
    task_id = task_id_for_artifacts(ctx)
    # Hash the original name, not its display spelling; the artifact store owns the key's suffix.
    name_hash = sha256(name.encode("utf-8", "surrogateescape")).hexdigest()[:16]
    basename = PurePosixPath(f"service-output-{name_hash}{path.suffix}")
    expected = {"name": immutable_artifact_name(basename, digest), "sha256": digest, "size": record["size"]}
    captured = copy_file_to_task_artifacts(ctx, path, kind="service_output_version", immutable=True, expected=expected)
    if not captured:
        raise OSError("service output version was not captured")
    root = canonical_data_root(ctx)
    if root.resolve() != Path(ctx.drive_root).resolve():  # the owner can read it live, before child-drive promotion
        captured = copy_file_to_task_artifacts(SimpleNamespace(drive_root=root, task_id=task_id),
            captured["path"], kind="service_output_version", immutable=True, expected=captured)
        if not captured:
            raise OSError("service output version was not published")
    key = captured["name"]
    return {"name": path_for_prose(name), "path": key, "sha256": captured["sha256"], "size": captured["size"],
            "download_url": f"/api/tasks/{quote(task_id, safe='')}/artifacts/{quote(key, safe='')}"}


def output_facts(ctx, declarations, reports=(), *, confirmed: bool) -> list[dict]:
    # Reports and declarations have the same order. A capture that did not report (e.g. the
    # storage step was interrupted) is not a success; earlier successful declarations stand.
    rows = []
    for index, declaration in enumerate(declarations):
        # The declared spelling is display text: a raw POSIX name byte shows as its exact escape, so
        # the fact stays inside UTF-8; the retained versions' keys and URLs address the bytes.
        name = path_for_prose(PurePosixPath(str(declaration).replace("\\", "/")).name)
        report = reports[index] if index < len(reports) else {}
        files = []
        records = report.get("records") or []
        for record in records:
            try:
                files.append(_version(ctx, record))
            except (OSError, RuntimeError, ValueError):
                log.debug("Could not retain exact service output version", exc_info=True)
        state = ("pending" if not confirmed else "failed" if not files else
                 "partial" if len(files) < len(records) or report.get("status") == "partial" else "saved")
        rows.append({"name": name, "state": state, "files": files})
    return rows


def publish_stop(ctx, payload: dict, *, confirmed: bool) -> None:
    """Publish the same safe facts to the model, trace and existing checkpoint readers."""
    launch_id = payload.get("launch_id")
    outputs = payload.pop("_output_facts", None)
    if not launch_id or "service_stop" in payload:
        return  # Do not invent a key or publish a payload twice.
    if outputs is None:
        outputs = output_facts(ctx, payload.get("outputs") or [], confirmed=confirmed)
    journal = payload.get("log_finalization") or {}
    journal_state = ("saved" if (journal.get("full_log_ref") or {}).get("sha256") else
                     "retained" if journal.get("retained_live_log_path") else
                     "unavailable" if journal else "unknown")
    facts = {"launch_id": launch_id, "name": payload["name"],
             "process": {"state": "stopped" if confirmed else "unconfirmed", "returncode": payload.get("returncode")},
             "journal": {"state": journal_state}, "outputs": outputs}
    payload["service_stop"] = facts
    trace = getattr(ctx, "_execution_trace", None)
    if isinstance(trace, dict):
        trace.setdefault("verification_events", []).append({"kind": "service_stop", "service_stop": facts})
    if task_id := getattr(ctx, "task_id", None):
        from ouroboros.loop_messages import _emit_checkpoint_event

        text = "Background service stop"
        chat_id = getattr(ctx, "current_chat_id", None)
        _emit_checkpoint_event(getattr(ctx, "event_queue", None), task_id, ctx.drive_logs(), {
            "checkpoint_kind": "service_stop", "service_stop": facts, "text": text,
            "role": "system", "system_type": "task_checkpoint",
            "card_row_revision": 2 if confirmed else 1,
            **({"chat_id": chat_id} if chat_id is not None else {}),
        })


_FINALIZATION_KINDS = frozenset({
    "services_stopped",
    "services_kept",
    "service_finalization_error",
    "service_stop",
})


def service_identity_projection(service: Dict[str, Any]) -> Dict[str, Any]:
    """Bounded identity used to deduplicate idempotent teardown observations.

    The service payload's own contract, kept with the stop evidence it carries:
    ``loop_budget`` binds this function for its facade, and every field here is
    a fact the stop/keep producer already publishes. ``service_stop`` is the
    named Stop's own safe fact (``publish_stop`` above): the launch id, the
    process state, the journal's state and each declared output's retained
    versions, with no private log path or log text. Carrying it whole keeps one
    launch's facts identical whether they arrive as the Stop's own event or in
    a teardown row.
    """

    fields = (
        "service_id",
        "name",
        "task_id",
        "launch_id",
        "lifecycle",
        "backend",
        "pid",
        "port",
        "returncode",
        "artifact_outputs",
        "artifact_output_failed",
        "artifact_audit_gap",
        "log_finalization",
        "state",
        "stop_failed",
        "cleanup_dispatched",
        "stop_error",
        "service_stop",
    )
    projected = {}
    for key in fields:
        value = service.get(key)
        if key == "stop_error" and value not in (None, ""):
            value = truncate_within_limit(str(value), 2000)
        if value not in (None, "", [], {}):
            projected[key] = value
    cleanup = service.get("process_cleanup")
    if isinstance(cleanup, dict):  # what stayed unconfirmed (pid, birth, reason) and why a rest is unknown, whole
        projected["process_cleanup"] = {key: cleanup[key] for key in (
            "exit_confirmed", "census_failed", "residual", "survivors", "unverified", "custody") if key in cleanup}
    return projected


def finalization_evidence(llm_trace: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return the stable, answer-relevant part of service finalization events.

    One launch, one current outcome: the LAST event that carries a launch is
    that launch's outcome. A pending Stop a retry later confirmed is replaced
    by the confirmation instead of standing beside it, and an older teardown
    row can no longer hide a later confirmed Stop of a launch it carried. The
    superseded observation stays in the trace; only current outcomes reach the
    answer. Rows without keyed launches — kept, legacy and error evidence —
    pass through as recorded. This only reads the trace: it starts no stop and
    decides no process.
    """

    events = []
    for event in llm_trace.get("verification_events") or []:
        kind = str(event.get("kind") or "") if isinstance(event, dict) else ""
        if kind in _FINALIZATION_KINDS:
            events.append((kind, event))
    latest: Dict[str, int] = {}  # launch id -> index of its newest carrying event
    for index, (kind, event) in enumerate(events):
        if kind == "service_stop":
            facts = event.get("service_stop")
            facts = facts if isinstance(facts, dict) else {}
            launch_id = str(facts.get("launch_id") or "")
            if launch_id:
                latest[launch_id] = index
            continue
        for service in event.get("services") or []:
            facts = service.get("service_stop") if isinstance(service, dict) else None
            if isinstance(facts, dict) and facts.get("launch_id"):
                latest[str(facts["launch_id"])] = index
    rows: list[Dict[str, Any]] = []
    for index, (kind, event) in enumerate(events):
        error = truncate_within_limit(str(event.get("error") or ""), 2000)
        if kind == "service_stop":
            facts = event.get("service_stop")
            facts = facts if isinstance(facts, dict) else {}
            launch_id = str(facts.get("launch_id") or "")
            if not launch_id or latest.get(launch_id, index) != index:
                continue  # a later event already shows this launch's current outcome
            # The same bounded identity a teardown row goes through, so one launch reads
            # identically whichever event carries it.
            services = [service_identity_projection(
                {"launch_id": launch_id, "name": facts.get("name"), "service_stop": facts})]
        else:
            services = []
            superseded = False
            for service in event.get("services") or []:
                if not isinstance(service, dict):
                    continue
                projected = service_identity_projection(service)
                carried = projected.get("service_stop")
                launch_id = str(carried.get("launch_id") or "") if isinstance(carried, dict) else ""
                if launch_id and latest.get(launch_id, index) > index:
                    superseded = True  # this launch's current outcome is a later event
                    continue
                services.append(projected)
            if superseded and not services:
                continue  # every keyed launch this row carried was confirmed by a later event
        rows.append({
            "kind": kind,
            "services": services,
            "error": error,
        })
    return rows
