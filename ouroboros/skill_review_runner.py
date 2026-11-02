"""Skill review lifecycle: the job file, the Skills-page projection, events and the post-review
reconciliation.

Writes ``review_job.json`` and the ``skill_review_*`` events. The review itself is one wave of the
shared review mechanism (``skill_review.review_skill`` → ``review_change``); its durable record is the
review ledger (``surface="skill"``), and the card's history is read from it beside the rows an earlier
version appended to ``review_history.jsonl`` (``skill_review_history``, a reader). The result payload
keeps the review verdict, ``deps_*`` and ``extension_*`` outcomes as separate fields, so a failed
dependency install or extension load never rewrites the verdict. A reused (unchanged) verdict
dispatches no reviewer panel but still resumes dependency installation when it is executable. Review
auto-enable only fires while no ``enabled.json`` exists, so it never overrides an owner disable. The
lifecycle tail runs on the wave's settlement: a wave its author left (an addressed message) returns
``pending``, enables nothing, and is collected at $0 by the next review of the same bytes.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from ouroboros import dialogue_store, skill_review_history
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
    save_enabled,
    skill_identity_collision_names,
    skill_review_gate,
    skill_state_dir,
    skill_state_dir_path,
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
from ouroboros.utils import append_jsonl, atomic_write_json, read_json_dict, update_json_locked, utc_now_iso
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    build_resolved_resource_binding,
    canonical_data_root,
    load_bound_skill,
)

log = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_SEC = 30.0
_STALE_REVIEW_JOB_SEC = int(os.environ.get("OUROBOROS_SKILL_REVIEW_JOB_STALE_SEC", "7200"))


ReviewImpl = Callable[..., SkillReviewOutcome]


_REVIEW_JOB_FILENAME = "review_job.json"


def review_job_state_path(drive_root: pathlib.Path, skill_name: str) -> pathlib.Path:
    """The job file for writers (creates the state dir); readers that must not
    materialize state resolve it through ``skill_state_dir_path``."""
    return skill_state_dir(pathlib.Path(drive_root), skill_name) / _REVIEW_JOB_FILENAME


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
    "content_hash", "job_id", "record_id", "group_id", "review_round", "snapshot_attempt",
    "snapshot_revised", "task_id", "root_task_id", "origin_task_id",
    "origin_root_task_id", "presentation_owner_task_id", "chat_id", "source",
    "terminal_reason", "started_at", "finished_at", "executions", "panel",
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
    exact count of older group rows that ten-row window leaves out.

    A ``running`` run whose process is gone or whose heartbeat is stale is
    presented as ``interrupted`` by the predicate the heal owners use; this
    read writes nothing — boot, the maintenance pass and the next review start
    persist the interruption."""
    from ouroboros.review_ledger import index_path

    cache_key = (str(drive_root), str(skill_name))
    # Non-creating: a list read must not leave state/skills/<name>/ behind.
    job_path = skill_state_dir_path(pathlib.Path(drive_root), skill_name) / _REVIEW_JOB_FILENAME
    stamp = (
        _file_stamp(job_path),
        _file_stamp(skill_review_history.review_history_path(drive_root, skill_name)),
        _file_stamp(index_path(drive_root)),
    )
    cached = _UI_PROJECTION_CACHE.get(cache_key)
    if cached is not None and cached[0] == stamp:
        projection, current = cached[1], cached[2]
    else:
        current = _read_review_job(job_path)
        # The legacy rows of this skill's group, then its ledger records (the fold's history), oldest first.
        legacy = skill_review_history.load_history(drive_root, skill_name, limit=0)
        group_id = str(current.get("group_id") or "")
        if not group_id and legacy:
            group_id = str(legacy[-1].get("group_id") or "")
        group_rows = [row for row in legacy if not group_id or row.get("group_id") == group_id]
        group_rows += _ledger_history_rows(drive_root, skill_name)
        history = group_rows[-10:]
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
        _UI_PROJECTION_CACHE[cache_key] = (stamp, projection, current)
    # Liveness is not a file fact — a pid dies without a byte changing — so the
    # memo above never answers it.
    return _present_dead_running_job(projection, current)


def _present_dead_running_job(projection: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only twin of ``mark_stale_review_job_interrupted``: the same
    ``interrupted`` status, lifecycle status and terminal reason the heal would
    persist, on a copy of the projection, with no write."""
    interrupt_reason = _running_job_interrupt_reason(current)
    if interrupt_reason is None:
        return projection
    shown = dict(projection)
    shown["current"] = {
        **projection.get("current", {}),
        "status": "interrupted",
        "lifecycle_status": "interrupted",
        "terminal_reason": interrupt_reason,
    }
    return shown


def _events_path(drive_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(drive_root) / "logs" / "events.jsonl"


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


def _ledger_history_rows(drive_root: pathlib.Path, skill_name: str) -> list[Dict[str, Any]]:
    """This skill's review ledger records as the card's history rows, oldest first: the lifecycle facts the
    record's brief kept (job, group, source), the gate's status of its verdict, its panel and executions."""
    from ouroboros import review_ledger as ledger
    from ouroboros.review_execution_projection import review_executions_from_actor_usage
    from ouroboros.tools.review_change import review_result
    from ouroboros.tools.review_skill import legacy_actor_records, outcome_status, skill_record_rows

    rows: list[Dict[str, Any]] = []
    for row in reversed(skill_record_rows(drive_root, skill_name, limit=10)):
        try:
            record = ledger.load_record(drive_root, str(row.get("record_id") or ""))
        except ValueError:
            record = None
        if not record:
            continue
        result = review_result(record, reused=False)
        status, _findings, reason = outcome_status(result)
        facts = dict((record.get("brief") or {}).get("skill") or {})
        settled = str(record.get("state") or "") == "settled"
        rows.append({
            "ts": str(record.get("ts") or ""), "status": status, "review_status": status,
            "job_status": "completed" if settled else "running", "lifecycle_status": "succeeded" if settled else "running",
            "content_hash": str((record.get("subject") or {}).get("tree_sha") or ""),
            "job_id": str(facts.get("job_id") or record.get("record_id") or ""), "record_id": str(record.get("record_id") or ""),
            "group_id": str(facts.get("group_id") or ""), "source": str(facts.get("source") or ""),
            "task_id": str(record.get("task_id") or ""), "root_task_id": str(record.get("root_task_id") or ""),
            "terminal_reason": reason, "started_at": str(record.get("ts") or ""), "finished_at": str(record.get("ts") or ""),
            "panel": dict(record.get("panel") or {}),
            "executions": review_executions_from_actor_usage(legacy_actor_records(result)),
        })
    return rows


def _allocate_ordinals(drive_root: pathlib.Path, skill_name: str, group_id: str, content_hash: str) -> tuple[int, int, bool]:
    """Presentation ordinals of the next review: its round in the group, its attempt on this snapshot, and
    whether the snapshot changed since the last one — counted over the group's legacy history rows and this
    skill's ledger records (never review authority: the card's title and the chat line)."""
    from ouroboros.tools.review_skill import skill_record_rows

    legacy = [row for row in skill_review_history.load_history(drive_root, skill_name, limit=0)
              if str(row.get("group_id") or "") == group_id]
    records = skill_record_rows(drive_root, skill_name, limit=200)
    hashes = [str(row.get("content_hash") or "") for row in legacy] + [
        str((row.get("subject") or {}).get("tree_sha") or "") for row in reversed(records)]
    last = hashes[-1] if hashes else ""
    return len(hashes) + 1, hashes.count(content_hash) + 1, bool(last and last != content_hash)


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
    """Return compact execution receipts proved by the wave's own seat rows (the record's rows, rendered
    as actor records with their usage facts). A reused record dispatched nothing: no execution of this attempt."""
    from ouroboros.review_execution_projection import (
        normalize_review_executions,
        review_executions_from_actor_usage,
    )

    if result is None or bool(getattr(result, "reused", False)) or str(getattr(result, "replayed_from_ts", "") or ""):
        return []
    actors = list(getattr(result, "raw_actor_records", None) or [])
    return normalize_review_executions(review_executions_from_actor_usage(actors))


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
    dialogue_store.append_quietly(
        drive_root, "chat",
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
            "record_id": str(payload.get("record_id") or ""),
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
    dialogue_store.append_quietly(
        drive_root, "progress",
        {
            "ts": ts,
            "type": "send_message",
            "task_id": _review_lifecycle_chat_task_id(skill_name, job_id),
            "is_progress": True,
            "direction": "out",
            "chat_id": chat_id,
            "user_id": 0,
            "role": "system", "system_type": "skill_lifecycle",
            "text": text,
            "content": text,
            "format": "",
            "lifecycle": lifecycle,
        },
    )


def _running_job_interrupt_reason(
    data: Dict[str, Any], *, stale_after_sec: int = _STALE_REVIEW_JOB_SEC,
) -> str | None:
    """Why a ``running`` review job is dead: ``owner_process_exited`` when its
    pid is gone, ``heartbeat_stale`` when the heartbeat is older than
    ``stale_after_sec``; ``None`` for a live run or any other status. Shared by
    the heal (which persists it) and the UI projection (which only shows it)."""
    if str(data.get("status") or "") != "running":
        return None
    pid = int(data.get("pid") or 0)
    heartbeat_age = _iso_age_sec(str(data.get("last_heartbeat_at") or data.get("started_at") or ""))
    if pid and not _pid_alive(pid):
        return "owner_process_exited"
    if heartbeat_age and heartbeat_age > stale_after_sec:
        return "heartbeat_stale"
    return None


def mark_stale_review_job_interrupted(
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    current_content_hash: str = "",
    stale_after_sec: int = _STALE_REVIEW_JOB_SEC,
) -> None:
    path = review_job_state_path(drive_root, skill_name)
    now = utc_now_iso()
    payload: Dict[str, Any] = {}

    def mark(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Under the job lock: a seat's pre-POST invocation checkpoint (#1582)
        # landing between a read and this write must not be erased by it.
        interrupt_reason = _running_job_interrupt_reason(data, stale_after_sec=stale_after_sec)
        if interrupt_reason is None:
            return None
        payload.update(with_schema_version({
            **data, "status": "interrupted", "lifecycle_status": "interrupted", "finished_at": now,
            "interrupted_at": now, "interrupt_reason": interrupt_reason, "terminal_reason": interrupt_reason,
            "content_hash": data.get("content_hash") or current_content_hash,
        }, SKILL_OWNER_STATE_SCHEMA_VERSION))
        return payload

    update_json_locked(path, mark)
    if not payload:
        return
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
    from ouroboros.utils import update_json_locked
    def patch(data):
        if expected_job_id and data.get("job_id") != expected_job_id:
            return None
        return with_schema_version({**data, **updates}, SKILL_OWNER_STATE_SCHEMA_VERSION)
    update_json_locked(path, patch, strict_existing_dict=True)


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
    """One review per skill at a time: the panel lends the lifecycle lane out, so a
    second review of the same skill (other bytes) gets a pointer, not a race on its
    job record."""
    return f"review:{skill_name}"


def _call_review_with_lifecycle_guard(
    review_impl: ReviewImpl,
    ctx: Any,
    skill_name: str,
    drive_root: pathlib.Path | None = None,
    binding: ResolvedResourceBinding | None = None,
    panel: Any = None,
    panel_request: Any = None,
    source: str = "",
) -> SkillReviewOutcome:
    """Run ``review_impl`` under the lifecycle job's facts: the job id the projection is bound to, the
    group the ceiling is keyed by, the panel the job composed (or the request it could not compose, which
    the wave then refuses typed or collects under an open wave's recorded panel), and the review's source."""
    sentinel = object()
    names = ("_skill_review_panel", "_skill_review_composed_panel", "_skill_review_panel_request",
             "_skill_review_lifecycle_guard", "_skill_review_lifecycle_job_id", "_skill_review_content_hash",
             "_skill_review_group_id", "_skill_review_round", "_skill_review_snapshot_revised",
             "_skill_review_resolved_binding", "_skill_review_source")
    previous = {name: getattr(ctx, name, sentinel) for name in names}
    state_root = pathlib.Path(drive_root or ctx.drive_root)
    job_data = _read_review_job(review_job_state_path(state_root, skill_name))
    setattr(ctx, "_skill_review_panel", dict(panel.facts) if panel is not None else {})
    setattr(ctx, "_skill_review_composed_panel", panel)
    setattr(ctx, "_skill_review_panel_request", panel_request)
    setattr(ctx, "_skill_review_lifecycle_guard", True)
    setattr(ctx, "_skill_review_lifecycle_job_id", str(job_data.get("job_id") or ""))
    setattr(ctx, "_skill_review_content_hash", str(job_data.get("content_hash") or ""))
    setattr(ctx, "_skill_review_group_id", str(job_data.get("group_id") or ""))
    setattr(ctx, "_skill_review_round", int(job_data.get("review_round") or 1))
    setattr(ctx, "_skill_review_snapshot_revised", bool(job_data.get("snapshot_revised")))
    setattr(ctx, "_skill_review_resolved_binding", binding)
    setattr(ctx, "_skill_review_source", str(source or ""))
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


def _review_job_finish_skip_reason(data: Dict[str, Any], job_id: str) -> str:
    """Why lifecycle owner ``job_id`` may not finish the CURRENT job record, or ''."""
    if not job_id:
        return ""
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
    job_id: str,
) -> None:
    """Time out the caller's OWN running job: ``job_id`` is the job its ``_on_started``
    recorded. A waiter that timed out before it owned one ('') leaves the running
    owner's job alone, whatever content hash the two share."""
    path = review_job_state_path(drive_root, skill_name)
    now = utc_now_iso()
    payload: Dict[str, Any] = {}

    def mark(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Locked, as for a stale interrupt, and only over this owner's own record.
        if (not job_id or str(current.get("job_id") or "") != job_id
                or str(current.get("status") or "") != "running"):
            return None
        payload.update(with_schema_version({
            **current, "status": "timeout", "lifecycle_status": "timeout", "finished_at": now,
            "terminal_reason": reason or "lifecycle_timeout",
            "content_hash": current.get("content_hash") or content_hash,
        }, SKILL_OWNER_STATE_SCHEMA_VERSION))
        return payload

    update_json_locked(path, mark)
    if not payload:
        return
    _append_review_chat_summary(
        drive_root, skill_name, payload, status="timeout", ts=now,
    )


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
        "findings": outcome.findings, "panel": dict(outcome.panel),
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
        "extension_load_error": getattr(outcome, "extension_load_error", None),
        "extension_live_loaded": getattr(outcome, "extension_live_loaded", None),
    }
    if getattr(outcome, "dispatch_refusal_code", ""):
        payload["dispatch_refusal_code"] = outcome.dispatch_refusal_code
    if getattr(outcome, "turn_on_when_ready", ""):
        payload["turn_on_when_ready"] = outcome.turn_on_when_ready
    for key in ("record_id", "reused", "pending", "paid", "seats", "governance"):
        value = getattr(outcome, key, None)
        if value:
            payload[key] = value
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
    revert_enabled_on_error: bool = False,
) -> Dict[str, Any]:
    """Reconcile after review; the receipt also names the answering process."""
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
            "load_error": live_state.get("load_error"),
            "live_loaded": live_state.get("live_loaded"),
        }
    except Exception as exc:
        return {"action": None, "reason": "reconcile_failed", "process": "", "server_reconcile": "",
                "load_error": f"{type(exc).__name__}: {exc}", "live_loaded": None}


def _on_started(
    drive_root: pathlib.Path,
    skill_name: str,
    content_hash: str,
    started_monotonic: Dict[str, float],
    provenance: Dict[str, Any] | None = None,
    *,
    refresh_content_hash: Callable[[], str] | None = None,
    bound_content_hash: Dict[str, str] | None = None,
    owned_job: Dict[str, str] | None = None,
    owner_attestation: bool = False,
) -> Callable[[LifecycleJob], None]:
    def _callback(job: LifecycleJob) -> None:
        now = utc_now_iso()
        provenance_data = dict(provenance or {})
        group_id = str(provenance_data.get("group_id") or f"manual:{skill_name}")
        previous = _read_review_job(review_job_state_path(drive_root, skill_name))
        if (
            not owner_attestation and str(previous.get("status") or "") == "running"
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
        review_round, snapshot_attempt, snapshot_revised = _allocate_ordinals(
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
            "review_predecessor_job_id": str(previous.get("job_id") or ""),
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
        if owned_job is not None:
            owned_job["job_id"] = job.id  # the one record this caller may later time out
            if owner_attestation:
                from ouroboros.skill_install_record import read_install_record

                pending = (read_install_record(drive_root, skill_name) or {}).get("pending_update") or {}
                owned_job["update_digest"] = str(pending.get("digest") or "")
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
        skip_reason = "" if "value" in started_monotonic else "review job never acquired lifecycle file lock"
        review_status = normalize_skill_review_status(
            getattr(result, "status", "") if result is not None else ""
        )
        error = (
            f"{type(exc).__name__}: {exc}" if exc is not None
            else str(getattr(result, "error", "") if result is not None else "")
        )
        deps_error = str(getattr(result, "deps_error", "") if result is not None else "")
        state_status = "completed" if job.status == "succeeded" else job.status or "failed"
        terminal_reason = error or deps_error or job.error or job.status or state_status
        payload: Dict[str, Any] = {}
        if not skip_reason:
            terminal: Dict[str, Any] = {
                "status": state_status,
                "skill": skill_name,
                "job_id": job.id,
                "lifecycle_status": job.status,
                "dedupe_key": job.dedupe_key,
                "started_at": job.started_at,
                "last_heartbeat_at": now,
                "finished_at": job.finished_at or now,
                "duration_sec": duration,
                "pid": os.getpid(),
                "panel": dict(getattr(result, "panel", {}) or {}),
                "review_status": review_status,
                "error": error,
                "deps_error": deps_error,
                "terminal_reason": terminal_reason,
                # Computed outside the job lock, which a seat's checkpoint briefly waits on.
                "executions": _skill_review_executions(drive_root, skill_name, result),
                "record_id": str(getattr(result, "record_id", "") or ""),
                "reused": bool(getattr(result, "reused", False)),
            }
            replayed_from_ts = str(getattr(result, "replayed_from_ts", "") or "")
            if replayed_from_ts:
                terminal["replayed_from_ts"] = replayed_from_ts

            def finish(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                # ONE locked update of the CURRENT job (#1582): its owner is re-checked under
                # the lock a seat's pre-POST invocation checkpoint takes, so the latest wave,
                # with every token bound meanwhile, reaches the job and its terminal history.
                nonlocal skip_reason
                skip_reason = _review_job_finish_skip_reason(current, job.id)
                if skip_reason:
                    return None
                payload.update(with_schema_version({
                    **current,
                    **terminal,
                    "content_hash": (
                        getattr(result, "content_hash", "")
                        or current.get("content_hash")
                        or lifecycle_content_hash
                    ),
                }, SKILL_OWNER_STATE_SCHEMA_VERSION))
                return payload

            update_json_locked(review_job_state_path(drive_root, skill_name), finish)
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
        # Lifecycle completion is not a semantic review verdict.  A runner can
        # finish without returning a result (for example after an in-process
        # handoff), so never let the lifecycle word ``completed`` paint a
        # review green before a typed verdict exists.  Failure/cancellation
        # states remain useful terminal facts when there is no result to carry
        # the review status.
        history_status = review_status
        if result is None and state_status not in {"completed", "succeeded"}:
            history_status = state_status
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


def _duplicate_payload(skill_name: str, content_hash: str, duplicate: LifecycleJob, gate: Dict[str, Any]) -> Dict[str, Any]:
    """The invocation stays pending; execution authority describes the selected current bytes."""
    return {
        "skill": skill_name,
        "status": "pending",
        "content_hash": content_hash,
        "reviewer_models": [],
        "findings": [],
        "error": f"review already {duplicate.status} for this skill",
        "review_gate": gate,
        "executable_review": gate["executable_review"],
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
    cyber = gate.get("blocking_reason") == "cyber_authority"
    if cyber:
        # Cyber Pro's permission to run is not the critic's outcome: name the outcome.
        prefix = {STATUS_WARNINGS: "Review passed with findings", STATUS_BLOCKERS: "Review found blockers",
                  STATUS_PENDING: "Review pending"}.get(status, "Review passed")
    elif gate["executable_review"]:
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
    return base + (" | Cyber Pro: execution is Ouroboros's decision, not a review result" if cyber else "")


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
    owner_attestation: bool = False,
    reviewers: Optional[list[str]] = None, reason: str = "", _panel: Any = None,
    progress: JobProgressTarget | None = None,
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
        else canonical_data_root(ctx)
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
    from ouroboros.reviewer_slot_config import PanelCompositionError, compose_review_panel, composed_review_pool
    content_hash = _skill_content_hash(drive_root, skill_name, repo_path, binding)
    panel_request = None
    try:
        panel = None if owner_attestation else _panel or compose_review_panel(reviewers or (), reason, surface="skill",
                                                chosen_by="ouroboros" if source == "tool" else "owner")
    except PanelCompositionError:
        # The wave decides: an open paid wave of these bytes is collected under its recorded panel (current
        # tick edits cannot revoke collection); otherwise the same refusal comes back typed from it, $0.
        panel, panel_request = None, (reviewers if reviewers is not None else [], reason)
    except ValueError:
        panel = None  # the review collects its paid wave or returns its typed configuration refusal
    mark_stale_review_job_interrupted(drive_root, skill_name, current_content_hash=content_hash)
    dedupe_key = _review_dedupe_key(skill_name, content_hash) + (":owner_attestation" if owner_attestation else "")
    started_monotonic: Dict[str, float] = {}
    bound_content_hash: Dict[str, str] = {}
    owned_job: Dict[str, str] = {}
    progress = progress or JobProgressTarget()  # an owner action hands its own handle through
    provenance = {**_review_provenance(ctx, source, skill_name), "panel": panel.facts if panel is not None else {}}

    def _run_review() -> SkillReviewOutcome:
        with _review_job_heartbeat(drive_root, skill_name):
            progress.set("Running skill review…")
            with composed_review_pool(panel.seats) if panel is not None else contextlib.nullcontext():
                outcome = _call_review_with_lifecycle_guard(review_impl, ctx, skill_name, drive_root, binding, panel,
                                                            panel_request=panel_request, source=source)
            if not outcome.panel and panel is not None:
                outcome.panel = dict(panel.facts)
            if outcome.pending:
                # The author left the wave on an addressed message, or a seat still answers: the review goes
                # on without this job, and nothing below may run on a verdict that is not there — no
                # dependency install, no turn-on, no extension reload, no update settlement (astra R1-1).
                # The job ends as a non-failure; the re-issued skill_review (a new job) collects at $0,
                # projects the record and runs this tail then. A kept update stays pending until that call.
                setattr(outcome, "deps_status", "not_run")
                setattr(outcome, "deps_error", "")
                return outcome
            deps_status = "not_required"
            deps_error = ""
            reviewed = _load_binding_skill(binding) if binding is not None else find_skill(
                drive_root, skill_name, repo_path=repo_path,
            )
            # Current persisted acceptance is authoritative, including staleness and author finish.
            executable_review = bool(reviewed is not None and reviewed.review.gate_for(reviewed.content_hash)["executable_review"])
            if executable_review:
                from ouroboros.skill_dependencies import ensure_skill_dependencies

                deps_status, deps_error = ensure_skill_dependencies(drive_root, reviewed, progress=progress.set) if (
                    reviewed is not None) else ("failed", "skill not found during dependency reconciliation")
            setattr(outcome, "deps_status", deps_status)
            setattr(outcome, "deps_error", deps_error)
            from ouroboros.contracts.task_constraint import normalize_task_constraint
            from ouroboros.skill_loader import grant_status_for_skill

            current = _load_binding_skill(binding) if binding is not None else find_skill(
                drive_root, skill_name, repo_path=repo_path,
            )
            constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
            deps_ready = deps_status in {"installed", "not_required"}
            just_auto_enabled = bool(
                executable_review and getattr(outcome, "auto_flow", False)
                and deps_ready and current is not None
                and current.content_hash == outcome.content_hash and current.review.gate_for(current.content_hash)["executable_review"]
                and grant_status_for_skill(drive_root, current).get("usable")
                and (constraint is None or (constraint.allow_enable and not constraint.has_selected_skill))
                and not (skill_state_dir(drive_root, skill_name) / "enabled.json").exists()
            )
            if just_auto_enabled:
                save_enabled(drive_root, skill_name, True, actor="review_auto_enable")
            elif deps_ready:  # a failure just recorded is not re-run by turn-on in the same job
                from ouroboros.skill_lifecycle_actions import reconcile_turn_on_when_ready

                turn_on = reconcile_turn_on_when_ready(drive_root, skill_name, repo_path=repo_path,
                                                       reason="review finished")
                setattr(outcome, "turn_on_when_ready", turn_on["action"])
            else:
                setattr(outcome, "turn_on_when_ready", "not_ready")
            progress.set("Reloading extension…")
            reconcile = _reconcile_extension_payload(
                ctx,
                skill_name,
                drive_root=drive_root,
                repo_path=repo_path,
                binding=binding,
                revert_enabled_on_error=just_auto_enabled,
            )
            for key, value in reconcile.items():
                setattr(outcome, f"extension_{key}", value)
        return outcome

    outcome = None
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
                # A review still running (the author left it) is not this job's failure: its words stay on
                # the outcome and the job file, the job ends succeeded.
                result_error=lambda item: "" if getattr(item, "pending", False) else (
                    getattr(item, "error", "") or getattr(item, "deps_error", "") or getattr(item, "extension_load_error", "") or ""),
                on_started=_on_started(
                    drive_root, skill_name, content_hash, started_monotonic, provenance,
                    refresh_content_hash=lambda: _skill_content_hash(
                        drive_root, skill_name, repo_path, binding,
                    ),
                    bound_content_hash=bound_content_hash,
                    owned_job=owned_job,
                    owner_attestation=owner_attestation,
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
        return _duplicate_payload(skill_name, content_hash, exc.job, selected.review.gate_for(content_hash))
    except TimeoutError as exc:
        _mark_review_job_timeout(
            drive_root, skill_name, content_hash, reason=f"{type(exc).__name__}: {exc}",
            job_id=owned_job.get("job_id", ""),
        )
        raise
    finally:
        from ouroboros.skill_install_record import update_backup_path

        # A left wave settles nothing here: the kept update waits for the call that collects the verdict.
        if owned_job and not getattr(outcome, "pending", False) and update_backup_path(drive_root, skill_name).exists():
            from ouroboros.skill_source_install import run_in_lane, settle_pending_update

            run_in_lane(drive_root, "update_settle", skill_name, lambda: settle_pending_update(
                drive_root, skill_name, expected_digest=owned_job.get("update_digest")
                or bound_content_hash.get("value") or content_hash,
                landing_failed=getattr(outcome, "deps_status", "") == "failed"))

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
