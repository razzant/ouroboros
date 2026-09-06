from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict

from ouroboros import skill_review_history
from ouroboros.config import get_skills_repo_path, load_settings
from ouroboros.skill_lifecycle_queue import (
    DuplicateLifecycleJobError,
    JobProgressTarget,
    LifecycleJob,
    LifecycleJobOptions,
    run_blocking_preserving_cancellation,
    run_lifecycle_job_blocking,
)
from ouroboros.contracts.schema_versions import with_schema_version
from ouroboros.skill_loader import (
    SKILL_OWNER_STATE_SCHEMA_VERSION,
    SkillPayloadUnreadable,
    compute_content_hash,
    find_skill,
    review_status_allows_execution,
    save_enabled,
    skill_identity_collision_names,
    skill_review_gate,
    skill_state_dir,
)
from ouroboros.skill_review import (
    SkillReviewOutcome,
)
from ouroboros.skill_review import (
    review_skill as _default_review_skill,
)
from ouroboros.skill_review_status import (
    STATUS_BLOCKERS,
    STATUS_PENDING,
    STATUS_WARNINGS,
    normalize_skill_review_status,
)
from ouroboros.utils import append_jsonl, atomic_write_json, read_json_dict, utc_now_iso
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    build_resolved_resource_binding,
    load_bound_skill,
)

log = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_SEC = 30.0
_STALE_REVIEW_JOB_SEC = int(os.environ.get("OUROBOROS_SKILL_REVIEW_JOB_STALE_SEC", "7200"))


ReviewImpl = Callable[..., SkillReviewOutcome]


def review_job_state_path(drive_root: pathlib.Path, skill_name: str) -> pathlib.Path:
    return skill_state_dir(pathlib.Path(drive_root), skill_name) / "review_job.json"


def _write_review_job(path: pathlib.Path, data: Dict[str, Any]) -> None:
    """Single write seam for review_job.json: every write — fresh or merge —
    lands with the ABI-2 stamp (CPL4-C10), so a job started before the upgrade
    still finishes stamped. Readers keep legacy-0 tolerance."""
    atomic_write_json(
        path,
        with_schema_version(data, SKILL_OWNER_STATE_SCHEMA_VERSION),
        trailing_newline=True,
    )


_UI_REVIEW_FIELDS = (
    "ts", "status", "review_status", "job_status", "lifecycle_status",
    "content_hash", "job_id", "group_id", "review_round", "snapshot_attempt",
    "snapshot_revised", "task_id", "root_task_id", "origin_task_id",
    "origin_root_task_id", "presentation_owner_task_id", "chat_id", "source",
    "terminal_reason", "started_at", "finished_at", "executions",
)


def _review_ui_row(value: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.review_execution_projection import normalize_review_executions

    row = {key: value[key] for key in _UI_REVIEW_FIELDS if key in value}
    if "executions" in row:
        row["executions"] = normalize_review_executions(row["executions"])
    return row


# The extensions index calls the projection for EVERY skill on EVERY request;
# reparsing an append-only history file each time is pure read amplification.
# Keyed by (drive_root, skill) and validated by both files' (mtime, size).
_UI_PROJECTION_CACHE: Dict[tuple, tuple] = {}


def _file_stamp(path: pathlib.Path) -> tuple:
    try:
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (0, 0)


def skill_review_ui_projection(
    drive_root: pathlib.Path, skill_name: str,
) -> Dict[str, Any]:
    """Sanitized current run, the last ten rows in its review group, and the
    exact count of older group rows that ten-row window leaves out."""
    cache_key = (str(drive_root), str(skill_name))
    stamp = (
        _file_stamp(review_job_state_path(drive_root, skill_name)),
        _file_stamp(skill_review_history.review_history_path(drive_root, skill_name)),
    )
    cached = _UI_PROJECTION_CACHE.get(cache_key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    current = _read_review_job(review_job_state_path(drive_root, skill_name))
    all_history = skill_review_history.load_history(drive_root, skill_name, limit=0)
    group_id = str(current.get("group_id") or "")
    if not group_id and all_history:
        group_id = str(all_history[-1].get("group_id") or "")
    group_rows = [row for row in all_history if not group_id or row.get("group_id") == group_id]
    history = group_rows[-10:]
    projection: Dict[str, Any]
    if not current and not history:
        projection = {}
    else:
        projection = {
            "current": _review_ui_row(current) if current else {},
            "history": [_review_ui_row(row) for row in history],
            # Group-scoped disclosed bound (BIBLE P1): the exact number of
            # older rows the ten-row window left out, 0 included.
            "history_omitted": max(0, len(group_rows) - len(history)),
        }
    _UI_PROJECTION_CACHE[cache_key] = (stamp, projection)
    return projection


def _events_path(drive_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(drive_root) / "logs" / "events.jsonl"


def _chat_jsonl_path(drive_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(drive_root) / "logs" / "chat.jsonl"


def _progress_jsonl_path(drive_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(drive_root) / "logs" / "progress.jsonl"


def _read_review_job(path: pathlib.Path) -> Dict[str, Any]:
    return read_json_dict(path) or {}


def _review_provenance(ctx: Any, source: str, skill_name: str) -> Dict[str, Any]:
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    task_id = str(getattr(ctx, "task_id", "") or metadata.get("task_id") or "")
    origin_task_id = str(metadata.get("origin_task_id") or task_id)
    origin_root_task_id = str(
        metadata.get("origin_root_task_id")
        or metadata.get("root_task_id")
        or getattr(ctx, "root_task_id", "")
        or origin_task_id
    )
    manual = source in {"skills", "manual", "ui", "api"} or not origin_task_id
    if manual:
        group_id = f"manual:{skill_name}"
        root_task_id = ""
        presentation_owner_task_id = ""
    else:
        # Ceiling/group root = the CURRENT task tree's root. The follow-up
        # chain marker origin_root_task_id is deliberately NOT honored here
        # (adversarial wave, machine-3): a scheduled follow-up is a fresh
        # root with a fresh review-cycle ceiling — the "fresh task" exit the
        # refusal advertises must be real. origin_* stay recorded below as
        # provenance facts.
        root_task_id = str(
            metadata.get("root_task_id")
            or getattr(ctx, "root_task_id", "")
            or task_id
        )
        group_id = f"task:{root_task_id}:{skill_name}"
        presentation_owner_task_id = root_task_id
    try:
        chat_id = int(getattr(ctx, "current_chat_id", 0) or 0)
    except (TypeError, ValueError):
        chat_id = 0
    return {
        "group_id": group_id,
        "task_id": task_id,
        "root_task_id": root_task_id,
        "origin_task_id": origin_task_id if not manual else "",
        "origin_root_task_id": origin_root_task_id if not manual else "",
        "presentation_owner_task_id": presentation_owner_task_id,
        "chat_id": chat_id,
        "source": str(source or ""),
    }


def _review_title(payload: Dict[str, Any]) -> str:
    content_hash = str(payload.get("content_hash") or "")[:12] or "unknown"
    revised = " — revised snapshot" if payload.get("snapshot_revised") else ""
    return (
        f"Skill review round {int(payload.get('review_round') or 1)} — "
        f"snapshot {content_hash} (attempt {int(payload.get('snapshot_attempt') or 1)})"
        f"{revised}"
    )


def _skill_review_executions(
    drive_root: pathlib.Path,
    skill_name: str,
    result: Any,
) -> list[Dict[str, str]]:
    """Return compact execution receipts proved by actor usage/physical rows."""
    from ouroboros.review_execution_projection import (
        normalize_review_executions,
        review_executions_from_actor_usage,
    )

    replayed_from_ts = str(getattr(result, "replayed_from_ts", "") or "")
    # A replay reuses an earlier verdict without a new physical dispatch. Even
    # if a future replay payload copies the original actors for forensics, it
    # must not present those old receipts as executions of this attempt.
    if replayed_from_ts:
        return []
    actors = list(getattr(result, "raw_actor_records", None) or []) if result is not None else []
    executions = review_executions_from_actor_usage(actors)
    wave_id = str(getattr(result, "wave_id", "") or "")
    if not bool(getattr(result, "paid", False)) or not wave_id:
        return executions
    try:
        from ouroboros.usage_accounting import skill_review_usage

        usage = skill_review_usage(
            drive_root, review_skill=skill_name, review_wave_id=wave_id,
        )
        physical_actors = []
        for attempt in usage.get("attempts") or []:
            if not isinstance(attempt, dict):
                continue
            if str(attempt.get("state") or "") not in {"dispatched", "settled", "unresolved"}:
                continue
            if str(attempt.get("source") or "") == "review_substrate.extraction":
                continue
            model = str(attempt.get("model") or "")
            if str(attempt.get("kind") or "") == "subscription_session":
                route = str(attempt.get("subscription_route") or "")
                if route:
                    physical_actors.append({"usage": {
                        "delegated_route": route, "resolved_model": model,
                    }})
            elif str(attempt.get("kind") or "") not in {"legacy_metadata", "legacy_delta"}:
                physical_actors.append({"usage": {
                    "ledger_attempt_ids": [str(attempt.get("attempt_id") or "")],
                    "resolved_model": model,
                }})
        executions.extend(review_executions_from_actor_usage(physical_actors))
    except Exception:
        log.debug("skill review execution projection unavailable", exc_info=True)
    return normalize_review_executions(executions)


def _terminal_history_payload(
    job_data: Dict[str, Any],
    *,
    status: str,
    terminal_reason: str,
    result: Any = None,
    ts: str,
) -> Dict[str, Any]:
    findings = list(getattr(result, "findings", None) or [])
    payload = {
        "ts": ts,
        "status": status,
        "job_status": str(job_data.get("lifecycle_status") or job_data.get("status") or status),
        "terminal_reason": str(terminal_reason or status),
        "content_hash": str(getattr(result, "content_hash", "") or job_data.get("content_hash") or ""),
        "failure_signature": skill_review_history.finding_signature(findings),
        "fail_findings": skill_review_history.extract_fail_findings(findings),
        "job_id": str(job_data.get("job_id") or ""),
    }
    for key in (
        "group_id", "review_round", "snapshot_attempt", "snapshot_revised",
        "task_id", "root_task_id", "origin_task_id", "origin_root_task_id",
        "presentation_owner_task_id", "chat_id", "source", "executions",
    ):
        if key in job_data:
            payload[key] = job_data[key]
    raw_actor_records = list(getattr(result, "raw_actor_records", None) or [])
    if raw_actor_records:
        payload["raw_actor_records"] = raw_actor_records
    if bool(getattr(result, "single_reviewer_no_diversity", False)):
        payload["single_reviewer_no_diversity"] = True
    # Max-Review-Cycles facts (Q17/Q23) ride the terminal row: paid panel
    # dispatch (one chunked wave = ONE cycle), its wave id, the panel contract
    # identity, the rebuttal content hash, and — for a $0 replay — the quoted
    # verdict ts. A terminal with NO result (lifecycle timeout) still gets the
    # facts: append_history_once merges the write-ahead dispatch marker by
    # wave/job id (F3), so the money spent before the timeout stays counted.
    if bool(getattr(result, "paid", False)):
        payload["paid"] = True
    for fact_key in ("wave_id", "review_contract_fingerprint", "rebuttal_sha256", "replayed_from_ts"):
        value = str(getattr(result, fact_key, "") or "")
        if value:
            payload[fact_key] = value
    return payload


def _append_terminal_history(
    drive_root: pathlib.Path,
    skill_name: str,
    job_data: Dict[str, Any],
    *,
    status: str,
    terminal_reason: str,
    result: Any = None,
    ts: str,
) -> bool:
    appended = skill_review_history.append_history_once(
        drive_root,
        skill_name,
        _terminal_history_payload(
            job_data,
            status=status,
            terminal_reason=terminal_reason,
            result=result,
            ts=ts,
        ),
    )
    if not appended:
        # LOUD failure (F3): a silently lost terminal row un-counts spent
        # review money and hides the verdict from the derived ledger. The
        # unmerged dispatch marker still preserves the paid fact.
        log.warning("skill review terminal history row did not land for %s", skill_name)
        append_jsonl(
            _events_path(drive_root),
            {
                "ts": ts,
                "type": "skill_review_history_append_failed",
                "skill": skill_name,
                "job_id": str(job_data.get("job_id") or ""),
                "status": status,
                "terminal_reason": terminal_reason,
                "reason": "terminal history or root-task projection append failed",
            },
        )
    return appended


def _append_review_chat_summary(
    drive_root: pathlib.Path,
    skill_name: str,
    payload: Dict[str, Any],
    *,
    status: str,
    ts: str,
) -> None:
    from ouroboros.review_execution_projection import normalize_review_executions

    title = _review_title(payload)
    provenance = str(payload.get("source") or "unknown")
    task_id = str(payload.get("task_id") or "")
    task_suffix = f", task={task_id}" if task_id else ""
    text = f"{title}: `{skill_name}` — status={status}, source={provenance}{task_suffix}"
    append_jsonl(
        _chat_jsonl_path(drive_root),
        {
            "ts": ts,
            "direction": "system",
            "type": "skill_review",
            "task_id": task_id,
            "root_task_id": str(payload.get("root_task_id") or ""),
            "origin_task_id": str(payload.get("origin_task_id") or ""),
            "origin_root_task_id": str(payload.get("origin_root_task_id") or ""),
            "presentation_owner_task_id": str(
                payload.get("presentation_owner_task_id") or ""
            ),
            "group_id": str(payload.get("group_id") or ""),
            "chat_id": int(payload.get("chat_id") or 0),
            "skill": skill_name,
            "status": status,
            "content_hash": str(payload.get("content_hash") or ""),
            "job_id": str(payload.get("job_id") or ""),
            "review_round": int(payload.get("review_round") or 1),
            "snapshot_attempt": int(payload.get("snapshot_attempt") or 1),
            "snapshot_revised": bool(payload.get("snapshot_revised")),
            "job_status": str(
                payload.get("lifecycle_status") or payload.get("job_status") or ""
            ),
            "terminal_reason": str(payload.get("terminal_reason") or status),
            "replayed_from_ts": str(payload.get("replayed_from_ts") or ""),
            "executions": normalize_review_executions(payload.get("executions")),
            "source": str(payload.get("source") or ""),
            "format": "markdown",
            "text": text,
        },
    )


def _pid_alive(pid: int) -> bool:
    from ouroboros.platform_layer import pid_is_alive

    return pid_is_alive(pid)


def _iso_age_sec(value: str) -> float:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return 0.0


def _review_lifecycle_chat_task_id(skill_name: str, job_id: str) -> str:
    skill_suffix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(skill_name or "skill")).strip("_")
    job_suffix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(job_id or "")).strip("_")
    return f"skill_lifecycle_review_{skill_suffix or 'skill'}_{job_suffix or 'job'}"


def _append_interrupted_review_progress(
    drive_root: pathlib.Path,
    skill_name: str,
    payload: Dict[str, Any],
    *,
    ts: str,
) -> None:
    reason = str(payload.get("interrupt_reason") or "interrupted")
    job_id = str(payload.get("job_id") or "")
    # C4: the interrupted row belongs to the SAME chat the review's other rows
    # went to — the payload records the initiator's chat (see `_review_provenance`
    # and the sibling chat.jsonl writer). Routed through the ONE notification
    # normalizer: a missing or A2A/internal (negative) chat falls back to the
    # Skill Review panel (0), never to a human stream.
    from supervisor.message_bus import notification_chat_route

    _route = notification_chat_route(payload.get("chat_id"), 0)
    chat_id = int(_route if _route is not None else 0)
    lifecycle = {
        "id": job_id,
        "kind": "review",
        "target": skill_name,
        "status": "interrupted",
        "phase": "interrupted",
        "message": "Review job was interrupted before completion.",
        "error": reason,
        "stale": False,
        "stale_reason": reason,
        "recovery_hint": "Start a fresh review for this skill before enabling or granting access.",
    }
    for key in (
        "group_id", "task_id", "root_task_id", "origin_task_id",
        "origin_root_task_id", "presentation_owner_task_id", "chat_id", "source",
    ):
        if key in payload:
            lifecycle[key] = payload[key]
    text = f"Skill review: `{skill_name}` — interrupted — {reason}"
    append_jsonl(
        _progress_jsonl_path(drive_root),
        {
            "ts": ts,
            "type": "send_message",
            "task_id": _review_lifecycle_chat_task_id(skill_name, job_id),
            "is_progress": True,
            "direction": "out",
            "chat_id": chat_id,
            "user_id": 0,
            "text": text,
            "content": text,
            "format": "",
            "lifecycle": lifecycle,
        },
    )


def mark_stale_review_job_interrupted(
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    current_content_hash: str = "",
    stale_after_sec: int = _STALE_REVIEW_JOB_SEC,
) -> None:
    path = review_job_state_path(drive_root, skill_name)
    data = _read_review_job(path)
    if str(data.get("status") or "") != "running":
        return
    pid = int(data.get("pid") or 0)
    heartbeat_age = _iso_age_sec(str(data.get("last_heartbeat_at") or data.get("started_at") or ""))
    pid_dead = bool(pid and not _pid_alive(pid))
    heartbeat_stale = bool(heartbeat_age and heartbeat_age > stale_after_sec)
    if not (pid_dead or heartbeat_stale):
        return
    now = utc_now_iso()
    payload = {
        **data,
        "status": "interrupted",
        "lifecycle_status": "interrupted",
        "finished_at": now,
        "interrupted_at": now,
        "interrupt_reason": "owner_process_exited" if pid_dead else "heartbeat_stale",
        "content_hash": data.get("content_hash") or current_content_hash,
    }
    payload["terminal_reason"] = payload["interrupt_reason"]
    _write_review_job(path, payload)
    _append_terminal_history(
        drive_root,
        skill_name,
        payload,
        status="interrupted",
        terminal_reason=str(payload["terminal_reason"]),
        ts=now,
    )
    _append_review_chat_summary(
        drive_root, skill_name, payload, status="interrupted", ts=now,
    )
    _append_interrupted_review_progress(drive_root, skill_name, payload, ts=now)
    append_jsonl(
        _events_path(drive_root),
        {
            "ts": now,
            "type": "skill_review_interrupted",
            "skill": skill_name,
            "content_hash": payload.get("content_hash", ""),
            "job_id": payload.get("job_id", ""),
            "reason": payload.get("interrupt_reason", ""),
        },
    )


def reconcile_stale_review_jobs(
    drive_root: pathlib.Path,
    *,
    repo_path: str | None = None,
    stale_after_sec: int = _STALE_REVIEW_JOB_SEC,
) -> int:
    root = pathlib.Path(drive_root) / "state" / "skills"
    if not root.exists():
        return 0
    collision_names = skill_identity_collision_names(
        pathlib.Path(drive_root),
        repo_path=get_skills_repo_path() if repo_path is None else repo_path,
    )
    count = 0
    for path in root.glob("*/review_job.json"):
        skill_name = path.parent.name
        if skill_name in collision_names:
            continue
        before = _read_review_job(path)
        if str(before.get("status") or "") != "running":
            continue
        mark_stale_review_job_interrupted(
            pathlib.Path(drive_root),
            skill_name,
            current_content_hash=str(before.get("content_hash") or ""),
            stale_after_sec=stale_after_sec,
        )
        after = _read_review_job(path)
        if str(after.get("status") or "") == "interrupted":
            count += 1
    return count


def _patch_review_job(
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    expected_job_id: str = "",
    **updates: Any,
) -> None:
    path = review_job_state_path(drive_root, skill_name)
    data = _read_review_job(path)
    current_job_id = str(data.get("job_id") or "")
    if expected_job_id and current_job_id and current_job_id != expected_job_id:
        return
    data.update(updates)
    _write_review_job(path, data)


@contextlib.contextmanager
def _review_job_heartbeat(drive_root: pathlib.Path, skill_name: str):
    stop = threading.Event()
    expected_job_id = str(
        _read_review_job(review_job_state_path(drive_root, skill_name)).get("job_id") or ""
    )

    def _beat() -> None:
        while not stop.wait(_HEARTBEAT_INTERVAL_SEC):
            try:
                _patch_review_job(
                    drive_root,
                    skill_name,
                    expected_job_id=expected_job_id,
                    last_heartbeat_at=utc_now_iso(),
                )
            except Exception:
                log.debug("skill review heartbeat update failed", exc_info=True)

    thread = threading.Thread(target=_beat, name=f"skill-review-heartbeat-{skill_name}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _load_binding_skill(binding: ResolvedResourceBinding) -> Any:
    return load_bound_skill(binding)


def _skill_content_hash(
    drive_root: pathlib.Path,
    skill_name: str,
    repo_path: str | None,
    binding: ResolvedResourceBinding | None = None,
) -> str:
    skill = _load_binding_skill(binding) if binding is not None else find_skill(
        drive_root, skill_name, repo_path=repo_path,
    )
    if skill is None or skill.load_error:
        return ""
    try:
        return compute_content_hash(
            skill.skill_dir,
            manifest_entry=skill.manifest.entry,
            manifest_scripts=skill.manifest.scripts,
        )
    except SkillPayloadUnreadable:
        return ""


def _review_dedupe_key(skill_name: str, content_hash: str) -> str:
    suffix = content_hash or "unknown"
    return f"review:{skill_name}:{suffix}"


def _call_review_with_lifecycle_guard(
    review_impl: ReviewImpl,
    ctx: Any,
    skill_name: str,
    drive_root: pathlib.Path | None = None,
    binding: ResolvedResourceBinding | None = None,
) -> SkillReviewOutcome:
    sentinel = object()
    previous = {
        "_skill_review_lifecycle_guard": getattr(ctx, "_skill_review_lifecycle_guard", sentinel),
        "_skill_review_lifecycle_job_id": getattr(ctx, "_skill_review_lifecycle_job_id", sentinel),
        "_skill_review_content_hash": getattr(ctx, "_skill_review_content_hash", sentinel),
        "_skill_review_group_id": getattr(ctx, "_skill_review_group_id", sentinel),
        "_skill_review_round": getattr(ctx, "_skill_review_round", sentinel),
        "_skill_review_snapshot_attempt": getattr(ctx, "_skill_review_snapshot_attempt", sentinel),
        "_skill_review_snapshot_revised": getattr(ctx, "_skill_review_snapshot_revised", sentinel),
        "_skill_review_resolved_binding": getattr(ctx, "_skill_review_resolved_binding", sentinel),
    }
    state_root = pathlib.Path(drive_root or ctx.drive_root)
    job_data = _read_review_job(review_job_state_path(state_root, skill_name))
    setattr(ctx, "_skill_review_lifecycle_guard", True)
    setattr(ctx, "_skill_review_lifecycle_job_id", str(job_data.get("job_id") or ""))
    setattr(ctx, "_skill_review_content_hash", str(job_data.get("content_hash") or ""))
    setattr(ctx, "_skill_review_group_id", str(job_data.get("group_id") or ""))
    setattr(ctx, "_skill_review_round", int(job_data.get("review_round") or 1))
    setattr(ctx, "_skill_review_snapshot_attempt", int(job_data.get("snapshot_attempt") or 1))
    setattr(ctx, "_skill_review_snapshot_revised", bool(job_data.get("snapshot_revised")))
    setattr(ctx, "_skill_review_resolved_binding", binding)
    try:
        if binding is not None and review_impl is _default_review_skill:
            return review_impl(ctx, skill_name, _resolved_binding=binding)
        return review_impl(ctx, skill_name)
    finally:
        for attr, value in previous.items():
            if value is sentinel:
                with contextlib.suppress(AttributeError):
                    delattr(ctx, attr)
            else:
                setattr(ctx, attr, value)


async def _to_thread_preserving_result(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return await run_blocking_preserving_cancellation(
        func,
        *args,
        log_label="blocking skill review lifecycle work",
        **kwargs,
    )


def _emit_review_persist_skipped(
    drive_root: pathlib.Path,
    skill_name: str,
    content_hash: str,
    *,
    reason: str,
    job_status: str = "",
    job_id: str = "",
) -> None:
    now = utc_now_iso()
    append_jsonl(
        _events_path(drive_root),
        {
            "ts": now,
            "type": "skill_review_persist_skipped",
            "skill": skill_name,
            "content_hash": content_hash,
            "reason": reason,
            "job_status": job_status,
            "job_id": job_id,
        },
    )


def _can_persist_review_outcome(
    drive_root: pathlib.Path,
    skill_name: str,
    content_hash: str,
    *,
    expected_job_id: str = "",
) -> bool:
    data = _read_review_job(review_job_state_path(pathlib.Path(drive_root), skill_name))
    if not data:
        return True
    status = str(data.get("status") or "")
    job_hash = str(data.get("content_hash") or "")
    job_id = str(data.get("job_id") or "")

    def _skip(reason: str) -> bool:
        _emit_review_persist_skipped(
            pathlib.Path(drive_root),
            skill_name,
            content_hash,
            reason=reason,
            job_status=status,
            job_id=job_id,
        )
        return False

    if expected_job_id and job_id and job_id != expected_job_id:
        return _skip("review job id no longer matches lifecycle owner")
    if job_hash and job_hash != content_hash:
        return _skip("content hash no longer matches current review job")
    terminal_blocking = {"interrupted", "failed", "cancelled", "timeout"}
    if status in terminal_blocking and (not job_hash or job_hash == content_hash):
        return _skip(f"review job already {status}")
    return True


def _review_job_finish_skip_reason(
    drive_root: pathlib.Path,
    skill_name: str,
    job_id: str,
) -> str:
    if not job_id:
        return ""
    data = _read_review_job(review_job_state_path(pathlib.Path(drive_root), skill_name))
    current_job_id = str(data.get("job_id") or "")
    if current_job_id and current_job_id != job_id:
        return "review job id no longer matches lifecycle owner"
    status = str(data.get("status") or "")
    if status and status != "running":
        return f"review job already {status}"
    return ""


def _mark_review_job_timeout(
    drive_root: pathlib.Path,
    skill_name: str,
    content_hash: str,
    *,
    reason: str,
) -> None:
    path = review_job_state_path(drive_root, skill_name)
    current = _read_review_job(path)
    if str(current.get("status") or "") != "running":
        return
    now = utc_now_iso()
    payload = {
        **current,
        "status": "timeout",
        "lifecycle_status": "timeout",
        "finished_at": now,
        "terminal_reason": reason or "lifecycle_timeout",
        "content_hash": current.get("content_hash") or content_hash,
    }
    _write_review_job(path, payload)
    _append_terminal_history(
        drive_root,
        skill_name,
        payload,
        status="timeout",
        terminal_reason=str(payload["terminal_reason"]),
        ts=now,
    )
    _append_review_chat_summary(
        drive_root, skill_name, payload, status="timeout", ts=now,
    )


def _reconcile_deps_after_pass_review(
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    repo_path: str | None = None,
    binding: ResolvedResourceBinding | None = None,
) -> tuple[str, str]:
    try:
        from ouroboros.marketplace.install_specs import install_specs_hash
        from ouroboros.marketplace.isolated_deps import (
            install_isolated_dependencies,
            read_deps_state,
        )

        loaded = _load_binding_skill(binding) if binding is not None else find_skill(
            drive_root, skill_name, repo_path=repo_path,
        )
        if loaded is None:
            return "failed", "skill not found during dependency reconciliation"
        from ouroboros.skill_dependencies import auto_install_specs_for_skill

        auto_specs = auto_install_specs_for_skill(drive_root, loaded)
        if not auto_specs:
            return "not_required", ""
        deps_state = read_deps_state(drive_root, skill_name, loaded.skill_dir)
        expected_hash = install_specs_hash(auto_specs)
        if (
            str(deps_state.get("status") or "") == "installed"
            and deps_state.get("specs_hash") == expected_hash
        ):
            return "installed", ""
        install_isolated_dependencies(drive_root, skill_name, loaded.skill_dir, auto_specs)
        return "installed", ""
    except Exception as exc:
        log.debug("post-review deps reconcile failed", exc_info=True)
        return "failed", f"{type(exc).__name__}: {exc}"


def _heal_mode(ctx: Any) -> bool:
    try:
        constraint = getattr(ctx, "task_constraint", None)
        return bool(constraint and getattr(constraint, "mode", "") == "skill_repair")
    except Exception:
        return False


def _outcome_payload(
    outcome: SkillReviewOutcome,
    *,
    deps_status: str,
    deps_error: str,
    extension_action: Any,
    extension_reason: Any,
    extension_process: Any = None,
    extension_server_reconcile: Any = None,
    job: LifecycleJob | None = None,
    job_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    status = normalize_skill_review_status(outcome.status)
    gate = skill_review_gate(status, findings=outcome.findings)
    payload: Dict[str, Any] = {
        "skill": outcome.skill_name,
        "status": status,
        "content_hash": outcome.content_hash,
        "reviewer_models": outcome.reviewer_models,
        "review_profile": str(getattr(outcome, "review_profile", "") or ""),
        "findings": outcome.findings,
        "raw_actor_records": list(getattr(outcome, "raw_actor_records", []) or []),
        "advisory_result": dict(getattr(outcome, "advisory_result", {}) or {}),
        "error": outcome.error,
        "review_gate": gate,
        "executable_review": gate["executable_review"],
        "auto_flow": bool(getattr(outcome, "auto_flow", False)),
        "auto_granted_keys": list(getattr(outcome, "auto_granted_keys", []) or []),
        "requested_keys": list(getattr(outcome, "requested_keys", []) or []),
        "auto_granted_permissions": list(getattr(outcome, "auto_granted_permissions", []) or []),
        "requested_permissions": list(getattr(outcome, "requested_permissions", []) or []),
        "deps_status": deps_status,
        "deps_error": deps_error,
        "extension_action": extension_action,
        "extension_reason": extension_reason,
        "extension_process": extension_process,
        "extension_server_reconcile": extension_server_reconcile,
    }
    if getattr(outcome, "convergence_hint", ""):
        payload["convergence_hint"] = outcome.convergence_hint
    if job is not None:
        payload["job_id"] = job.id
        payload["job_status"] = job.status
    for key in (
        "job_id", "group_id", "review_round", "snapshot_attempt", "snapshot_revised",
        "task_id", "root_task_id", "origin_task_id", "origin_root_task_id",
        "presentation_owner_task_id", "chat_id", "source", "terminal_reason",
        "executions",
    ):
        if job_data and key in job_data:
            payload[key] = job_data[key]
    return payload


def _reconcile_extension_payload(
    ctx: Any,
    skill_name: str,
    *,
    drive_root: pathlib.Path,
    repo_path: str | None,
    binding: ResolvedResourceBinding | None,
    heal_mode: bool,
    revert_enabled_on_error: bool = False,
) -> Dict[str, Any]:
    """Reconcile after review; the receipt also names the answering process."""
    def heal(action: str) -> Dict[str, Any]:
        return {"action": action, "reason": "heal_review_only", "process": "", "server_reconcile": ""}

    if heal_mode:
        try:
            from ouroboros import extension_loader

            if skill_name in extension_loader.snapshot()["extensions"]:
                extension_loader.unload_extension(skill_name)
                return heal("extension_unloaded")
            return heal("extension_heal_review_only")
        except Exception:
            return heal("extension_heal_review_only")
    try:
        from ouroboros import extension_loader

        live_state = extension_loader.reconcile_extension(
            skill_name,
            drive_root,
            load_settings,
            repo_path=repo_path,
            selected_skill=(
                _load_binding_skill(binding) if binding is not None else None
            ),
            retry_load_error=True,
            revert_enabled_on_error=revert_enabled_on_error,
        )
        return {
            "action": live_state.get("action"),
            "reason": live_state.get("reason"),
            "process": str(live_state.get("process") or ""),
            "server_reconcile": str(live_state.get("server_reconcile") or ""),
        }
    except Exception:
        return {"action": None, "reason": None, "process": "", "server_reconcile": ""}


def _on_started(
    drive_root: pathlib.Path,
    skill_name: str,
    content_hash: str,
    started_monotonic: Dict[str, float],
    provenance: Dict[str, Any] | None = None,
    *,
    refresh_content_hash: Callable[[], str] | None = None,
    bound_content_hash: Dict[str, str] | None = None,
) -> Callable[[LifecycleJob], None]:
    def _callback(job: LifecycleJob) -> None:
        now = utc_now_iso()
        provenance_data = dict(provenance or {})
        group_id = str(provenance_data.get("group_id") or f"manual:{skill_name}")
        previous = _read_review_job(review_job_state_path(drive_root, skill_name))
        if (
            str(previous.get("status") or "") == "running"
            and str(previous.get("job_id") or "")
            and str(previous.get("job_id") or "") != job.id
        ):
            raise RuntimeError("another process still owns this skill review lifecycle")
        current_content_hash = str(
            refresh_content_hash() if refresh_content_hash is not None else content_hash
        )
        if bound_content_hash is not None:
            bound_content_hash["value"] = current_content_hash
        started_monotonic["value"] = time.monotonic()
        review_round, snapshot_attempt, snapshot_revised = skill_review_history.allocate_ordinals(
            drive_root, skill_name, group_id, current_content_hash,
        )
        if previous and str(previous.get("job_id") or "") != job.id:
            if str(previous.get("group_id") or "") == group_id:
                review_round = max(review_round, int(previous.get("review_round") or 0) + 1)
                if str(previous.get("content_hash") or "") == current_content_hash:
                    snapshot_attempt = max(
                        snapshot_attempt, int(previous.get("snapshot_attempt") or 0) + 1,
                    )
                elif previous.get("content_hash"):
                    snapshot_revised = True
        payload = {
            "status": "running",
            "skill": skill_name,
            "content_hash": current_content_hash,
            "job_id": job.id,
            "lifecycle_status": job.status,
            "dedupe_key": job.dedupe_key,
            "started_at": job.started_at or now,
            "last_heartbeat_at": now,
            "finished_at": "",
            "duration_sec": None,
            "pid": os.getpid(),
            **provenance_data,
            "group_id": group_id,
            "review_round": review_round,
            "snapshot_attempt": snapshot_attempt,
            "snapshot_revised": snapshot_revised,
            "terminal_reason": "",
        }
        _write_review_job(review_job_state_path(drive_root, skill_name), payload)
        append_jsonl(
            _events_path(drive_root),
            {
                "ts": now,
                "type": "skill_review_started",
                "skill": skill_name,
                "content_hash": current_content_hash,
                "job_id": job.id,
            },
        )

    return _callback


def _on_finished(
    drive_root: pathlib.Path,
    skill_name: str,
    content_hash: str,
    started_monotonic: Dict[str, float],
    *,
    bound_content_hash: Dict[str, str] | None = None,
) -> Callable[[LifecycleJob, Any, BaseException | None], None]:
    def _callback(job: LifecycleJob, result: Any, exc: BaseException | None) -> None:
        lifecycle_content_hash = (
            str(bound_content_hash.get("value") or "")
            if bound_content_hash is not None and "value" in bound_content_hash
            else content_hash
        )
        now = utc_now_iso()
        duration = None
        if "value" in started_monotonic:
            duration = round(max(0.0, time.monotonic() - started_monotonic["value"]), 3)
        skip_reason = ""
        if "value" not in started_monotonic:
            skip_reason = "review job never acquired lifecycle file lock"
        else:
            skip_reason = _review_job_finish_skip_reason(drive_root, skill_name, job.id)
        if skip_reason:
            append_jsonl(
                _events_path(drive_root),
                {
                    "ts": now,
                    "type": "skill_review_finish_skipped",
                    "skill": skill_name,
                    "content_hash": lifecycle_content_hash,
                    "job_id": job.id,
                    "reason": skip_reason,
                    "duration_sec": duration,
                },
            )
            return
        review_status = normalize_skill_review_status(
            getattr(result, "status", "") if result is not None else ""
        )
        error = (
            f"{type(exc).__name__}: {exc}" if exc is not None
            else str(getattr(result, "error", "") if result is not None else "")
        )
        deps_error = str(getattr(result, "deps_error", "") if result is not None else "")
        state_status = "completed" if job.status == "succeeded" else job.status or "failed"
        current = _read_review_job(review_job_state_path(drive_root, skill_name))
        terminal_reason = error or deps_error or job.error or job.status or state_status
        payload = {
            **current,
            "status": state_status,
            "skill": skill_name,
            "content_hash": (
                getattr(result, "content_hash", "")
                or current.get("content_hash")
                or lifecycle_content_hash
            ),
            "job_id": job.id,
            "lifecycle_status": job.status,
            "dedupe_key": job.dedupe_key,
            "started_at": job.started_at,
            "last_heartbeat_at": now,
            "finished_at": job.finished_at or now,
            "duration_sec": duration,
            "pid": os.getpid(),
            "review_status": review_status,
            "error": error,
            "deps_error": deps_error,
            "terminal_reason": terminal_reason,
        }
        payload["executions"] = _skill_review_executions(
            drive_root, skill_name, result,
        )
        replayed_from_ts = str(getattr(result, "replayed_from_ts", "") or "")
        if replayed_from_ts:
            payload["replayed_from_ts"] = replayed_from_ts
        _write_review_job(review_job_state_path(drive_root, skill_name), payload)
        # Lifecycle completion is not a semantic review verdict.  A runner can
        # finish without returning a result (for example after an in-process
        # handoff), so never let the lifecycle word ``completed`` paint a
        # review green before a typed verdict exists.  Failure/cancellation
        # states remain useful terminal facts when there is no result to carry
        # the review status.
        history_status = review_status
        if result is None and state_status not in {"completed", "succeeded"}:
            history_status = state_status
        _append_terminal_history(
            drive_root,
            skill_name,
            payload,
            status=history_status,
            terminal_reason=terminal_reason,
            result=result,
            ts=now,
        )
        append_jsonl(
            _events_path(drive_root),
            {
                "ts": now,
                "type": "skill_review_completed" if state_status == "completed" else "skill_review_failed",
                "skill": skill_name,
                "content_hash": payload.get("content_hash", ""),
                "job_id": job.id,
                "duration_sec": duration,
                "status": review_status or state_status,
                "error": error or deps_error,
            },
        )
        _append_review_chat_summary(
            drive_root,
            skill_name,
            payload,
            status=history_status,
            ts=now,
        )

    return _callback


def _duplicate_payload(skill_name: str, content_hash: str, duplicate: LifecycleJob) -> Dict[str, Any]:
    return {
        "skill": skill_name,
        "status": "pending",
        "content_hash": content_hash,
        "reviewer_models": [],
        "findings": [],
        "error": f"review already {duplicate.status} for this skill/content hash",
        "review_gate": skill_review_gate("pending"),
        "executable_review": False,
        "deps_status": "not_required",
        "deps_error": "",
        "extension_action": None,
        "extension_reason": None,
        "job_id": duplicate.id,
        "job_status": duplicate.status,
    }


def _review_finding_summary(outcome: Any) -> str:
    def _is_pass(item: Dict[str, Any]) -> bool:
        signal = str(item.get("verdict") or item.get("status") or "").strip().lower()
        return signal in {"pass", "passed", "ok"}

    def _chat_headline(text: str, max_chars: int = 180) -> str:
        text = str(text or "").strip()
        if len(text) <= max_chars:
            return text
        marker = "... [omitted {count} chars; full findings in Skills page]"
        budget = max(1, max_chars - len(marker.format(count=0)))
        omitted = max(0, len(text) - budget)
        return text[:budget].rstrip() + marker.format(count=omitted)

    findings = [item for item in (getattr(outcome, "findings", None) or []) if isinstance(item, dict)]
    for item in sorted(findings, key=lambda item: 1 if _is_pass(item) else 0):
        label = str(item.get("item") or item.get("check") or item.get("title") or "finding").strip()
        verdict = str(item.get("verdict") or item.get("severity") or "").strip()
        reason = str(item.get("reason") or item.get("message") or "").strip()
        pieces = [piece for piece in (verdict, label, reason) if piece]
        if pieces:
            summary = ": ".join((" ".join(pieces[:2]), pieces[2])) if len(pieces) > 2 else " ".join(pieces)
            return _chat_headline(summary)
    return ""


def _review_result_message(outcome: Any) -> str:
    status = normalize_skill_review_status(str(getattr(outcome, "status", "") or STATUS_PENDING))
    summary = _review_finding_summary(outcome)
    gate = skill_review_gate(status)
    if gate["executable_review"]:
        prefix = "Review executable with findings" if status in {STATUS_WARNINGS, STATUS_BLOCKERS} else "Review executable"
    elif status == STATUS_WARNINGS:
        prefix = "Review warnings blocked by current enforcement"
    elif status == STATUS_BLOCKERS:
        prefix = "Review blocked: blocker findings"
    else:
        prefix = "Review pending"
    base = f"{prefix} ({status}){f': {summary}' if summary else ''}"
    auto_granted_keys = list(getattr(outcome, "auto_granted_keys", []) or [])
    auto_granted_permissions = list(getattr(outcome, "auto_granted_permissions", []) or [])
    if auto_granted_keys or auto_granted_permissions:
        auto_parts: list[str] = []
        if auto_granted_keys and not auto_granted_permissions:
            auto_parts.append(", ".join(auto_granted_keys))
        elif auto_granted_keys:
            auto_parts.append(f"keys: {', '.join(auto_granted_keys)}")
        if auto_granted_permissions:
            auto_parts.append(f"permissions: {', '.join(auto_granted_permissions)}")
        base = base + f" | auto-granted: {'; '.join(auto_parts)}"
    return base


async def run_skill_review_lifecycle(
    ctx: Any,
    skill_name: str,
    *,
    source: str = "skills",
    review_impl: ReviewImpl = _default_review_skill,
    repo_path: str | None = None,
    _resolved_binding: ResolvedResourceBinding | None = None,
) -> Dict[str, Any]:
    return await _to_thread_preserving_result(
        run_skill_review_lifecycle_blocking,
        ctx,
        skill_name,
        source=source,
        review_impl=review_impl,
        repo_path=repo_path,
        _resolved_binding=_resolved_binding,
    )


def run_skill_review_lifecycle_blocking(
    ctx: Any,
    skill_name: str,
    *,
    source: str = "tool",
    review_impl: ReviewImpl = _default_review_skill,
    repo_path: str | None = None,
    _resolved_binding: ResolvedResourceBinding | None = None,
) -> Dict[str, Any]:
    binding = _resolved_binding
    if binding is None and repo_path is None:
        try:
            binding = build_resolved_resource_binding(
                ctx, root="skill_payload", operation="review", path=".",
                skill_name=skill_name,
            )
        except Exception as exc:
            return _outcome_payload(
                SkillReviewOutcome(
                    skill_name=skill_name, status=STATUS_PENDING, error=str(exc),
                ),
                deps_status="not_run", deps_error="",
                extension_action=None, extension_reason=None,
            )
    drive_root = (
        binding.state_drive_root if binding is not None
        else pathlib.Path(ctx.drive_root)
    )
    repo_path = repo_path if repo_path is not None else get_skills_repo_path()
    selected = _load_binding_skill(binding) if binding is not None else find_skill(
        drive_root, skill_name, repo_path=repo_path,
    )
    if selected is not None and bool(getattr(selected, "identity_collision", False)):
        # A review/attestation is lifecycle state for one canonical identity.
        # Collision placeholders are topology-only, so refuse before creating a
        # review job, heartbeat, history row, grant, dependency, or enablement
        # state for an ambiguous name.
        return _outcome_payload(
            SkillReviewOutcome(
                skill_name=skill_name,
                status=STATUS_PENDING,
                error=selected.load_error or "skill identity collision",
            ),
            deps_status="not_run",
            deps_error="",
            extension_action=None,
            extension_reason=None,
        )
    if selected is None:
        return _outcome_payload(
            SkillReviewOutcome(
                skill_name=skill_name, status=STATUS_PENDING,
                error=f"Skill {skill_name!r} not found",
            ),
            deps_status="not_run", deps_error="",
            extension_action=None, extension_reason=None,
        )
    content_hash = _skill_content_hash(
        drive_root, skill_name, repo_path, binding,
    )
    mark_stale_review_job_interrupted(drive_root, skill_name, current_content_hash=content_hash)
    dedupe_key = _review_dedupe_key(skill_name, content_hash)
    started_monotonic: Dict[str, float] = {}
    bound_content_hash: Dict[str, str] = {}
    progress = JobProgressTarget()
    provenance = _review_provenance(ctx, source, skill_name)

    def _run_review() -> SkillReviewOutcome:
        with _review_job_heartbeat(drive_root, skill_name):
            progress.set("Running tri-model review…")
            outcome = _call_review_with_lifecycle_guard(
                review_impl, ctx, skill_name, drive_root, binding,
            )
            deps_status = "not_required"
            deps_error = ""
            executable_review = review_status_allows_execution(getattr(outcome, "status", ""))
            if executable_review:
                progress.set("Installing dependencies…")
                deps_status, deps_error = _reconcile_deps_after_pass_review(
                    drive_root,
                    skill_name,
                    repo_path=repo_path,
                    binding=binding,
                )
            setattr(outcome, "deps_status", deps_status)
            setattr(outcome, "deps_error", deps_error)
            if executable_review and getattr(outcome, "auto_flow", False) and deps_status == "failed":
                outcome.status = STATUS_PENDING
                outcome.error = deps_error or "self-authored dependency reconciliation failed"
                executable_review = False
            just_auto_enabled = bool(executable_review and getattr(outcome, "auto_flow", False))
            if just_auto_enabled:
                save_enabled(drive_root, skill_name, True, actor="review_auto_enable")
            progress.set("Reloading extension…")
            reconcile = _reconcile_extension_payload(
                ctx,
                skill_name,
                drive_root=drive_root,
                repo_path=repo_path,
                binding=binding,
                heal_mode=_heal_mode(ctx),
                revert_enabled_on_error=just_auto_enabled,
            )
            for key, value in reconcile.items():
                setattr(outcome, f"extension_{key}", value)
        return outcome

    try:
        outcome = run_lifecycle_job_blocking(
            kind="review",
            target=skill_name,
            source=source,
            message=f"Reviewing {skill_name}",
            dedupe_key=dedupe_key,
            # C4: a review started from a task-bound tool reports to that task's
            # chat; API/panel callers carry chat 0 and stay on the panel.
            chat_id=int(provenance.get("chat_id") or 0),
            runner=_run_review,
            options=LifecycleJobOptions(
                drive_root=drive_root,
                presentation=provenance,
                progress_target=progress,
                result_message=_review_result_message,
                result_error=lambda item: getattr(item, "error", "") or getattr(item, "deps_error", "") or "",
                on_started=_on_started(
                    drive_root, skill_name, content_hash, started_monotonic, provenance,
                    refresh_content_hash=lambda: _skill_content_hash(
                        drive_root, skill_name, repo_path, binding,
                    ),
                    bound_content_hash=bound_content_hash,
                ),
                on_finished=_on_finished(
                    drive_root,
                    skill_name,
                    content_hash,
                    started_monotonic,
                    bound_content_hash=bound_content_hash,
                ),
            ),
        )
    except DuplicateLifecycleJobError as exc:
        return _duplicate_payload(skill_name, content_hash, exc.job)
    except TimeoutError as exc:
        _mark_review_job_timeout(
            drive_root, skill_name, content_hash, reason=f"{type(exc).__name__}: {exc}",
        )
        raise

    try:
        from ouroboros.skill_loader import discover_skills
        from supervisor.queue import sync_skill_schedules

        sync_skill_schedules(discover_skills(drive_root, repo_path=repo_path), drive_root=drive_root)
    except Exception:
        log.debug("skill review schedule sync failed", exc_info=True)
    return _outcome_payload(
        outcome,
        deps_status=getattr(outcome, "deps_status", "not_required"),
        deps_error=getattr(outcome, "deps_error", ""),
        extension_action=getattr(outcome, "extension_action", None),
        extension_reason=getattr(outcome, "extension_reason", None),
        extension_process=getattr(outcome, "extension_process", None),
        extension_server_reconcile=getattr(outcome, "extension_server_reconcile", None),
        job_data=_read_review_job(review_job_state_path(drive_root, skill_name)),
    )
