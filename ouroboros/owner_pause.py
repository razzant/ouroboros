"""The owner's Pause of a whole task tree: its durable fence and launch gate.

Owner Batch4 (5A/7A), amended 2026-10-08 (#1562, full variant): Pause fences
NEW effects of every member of one root's tree; operations already handed to an
executor finish; the member's OWN delegated runs get a concrete stop
(``external_runs`` policy ``stop_task_owned``) while reviewers that already
launched finish their same check (``review_episode``, armed only at the
attempt's actual handoff under this module's launch lock); each member parks
warm (owner wait) or as the exact same-ID pause the budget rail already writes
(``budget_pause``, reason ``owner``). Nothing here is a scheduler, a ledger or
a second pause store:

- The FENCE is one projection on the ROOT's task result (``owner_pause``),
  written by the supervisor's accept step through the task result's own
  locked read-modify-write (atomic replace), and released by the root's
  explicit Resume. Every member reads the same file: the queue's admission
  fence keeps new descendants from being admitted or assigned, and this
  durable projection is what a member that is ALREADY running consults at
  each launch handoff.
- Preparation and durable claims precede confirmed executor submission,
  serialized with Pause by a short per-root lock. No lock spans body/network
  completion. Claims and started operations are custody, not wire receipts;
  a crash never proves success. Nested/queued transports take their final gate
  when ready, and returned invocations cannot acquire a late background start.
- A member's SAFE BOUNDARY (the loop's round drain, a pre-dispatch model wait,
  a warm owner wait woken by the ``owner_pause`` mailbox control) reads the
  same fence and enters the exact pause without buying a model round.

The fence being closed is NOT the tree being Paused: the root's ``state``
turns ``paused`` only when no member of the tree still runs and every parked
member's own sent work (delegated runs included) has settled; reviewers the
Pause lets finish never keep it pausing (the fence lists them as
``finishing_reviews``). Until then the truthful state is ``requested``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
import os
import pathlib
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

_TOOL_OPERATION = ContextVar("owner_pause_tool_operation", default=None)
_HANDED_TOOL = ContextVar("owner_pause_handed_tool", default=None)
_HANDED_MODEL = ContextVar("owner_pause_handed_model", default="")
# One PHYSICAL review attempt's completion allowance (``review_episode``):
# armed only by that attempt's first launch the fence admitted.
_REVIEW_EPISODE = ContextVar("owner_pause_review_episode", default=None)

RAIL_OWNER_PAUSE = "owner_pause"
REASON_OWNER = "owner"

FENCE_REQUESTED = "requested"
FENCE_PAUSED = "paused"
FENCE_RELEASED = "released"
CLOSED_FENCE_STATES = frozenset({FENCE_REQUESTED, FENCE_PAUSED})

# The saved pause of a member whose sent delegated work had not settled yet: a
# checkpoint exists, but the member is not cleanly Paused (never a false Paused).
SETTLEMENT_EXTERNAL_RUNNING = "external_writers_running"
SETTLEMENT_SETTLED = "settled"

NOT_STARTED_TEXT = ("⚠️ OWNER_PAUSE_NOT_STARTED: NOT STARTED — the owner paused this task tree before this "
                    "operation was launched. Nothing ran; it may be issued again after Resume.")


class OwnerPauseRefused(Exception):
    """Pause authority refused fence installation or a new launch."""


# --- the durable fence --------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, Tuple[Tuple[int, int, int], Dict[str, Any]]] = {}


def _result_path(root_drive: Any, root_task_id: str) -> pathlib.Path:
    from ouroboros.task_results import task_result_path

    return task_result_path(pathlib.Path(root_drive), str(root_task_id), create=False)


def read_fence(root_drive: Any, root_task_id: str) -> Dict[str, Any]:
    """The root's current ``owner_pause`` projection (``{}`` when none).

    Cached on the file's identity (inode, size, mtime): writers replace the
    file atomically, so an unchanged identity is an unchanged projection and
    the launch gate costs one ``stat`` on the hot path. An unreadable root
    result raises: an unknown fence is never read as an open one.
    """
    root_task_id = str(root_task_id or "").strip()
    if not root_task_id or root_drive in (None, ""):
        return {}
    path = _result_path(root_drive, root_task_id)
    key = str(path)
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        quarantine = path.parent / "quarantine"
        with _CACHE_LOCK:
            prior = _CACHE.get(key)
        if (prior or (quarantine / path.name).exists()
                or any(quarantine.glob(f"{root_task_id}.*.json"))):
            raise ValueError("owner_pause_authority_missing")
        return {}
    stamp = (int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            return dict(cached[1])
    from ouroboros.task_results import load_task_result

    row = load_task_result(pathlib.Path(root_drive), root_task_id, strict=True) or {}
    fence = row.get("owner_pause") or {}
    if not isinstance(fence, dict) or (fence and fence.get("state") not in
                                       {FENCE_REQUESTED, FENCE_PAUSED, FENCE_RELEASED}):
        raise ValueError("owner_pause_authority_unreadable")
    with _CACHE_LOCK:
        _CACHE[key] = (stamp, dict(fence))
    return dict(fence)


def fence_closed(fence: Dict[str, Any]) -> bool:
    return str((fence or {}).get("state") or "") in CLOSED_FENCE_STATES


def _resume_outstanding(current: Dict[str, Any], fence: Dict[str, Any]) -> bool:
    """The root's unrevoked Resume grant names this closed fence.

    The fence stays closed from the grant until the resumed worker consumes it
    (``reopen_for_resume``); that grant would release whatever shares its fence.
    """
    from ouroboros.budget_pause import STATE_RESUME_GRANTED, STATE_RESUMED
    from ouroboros.post_task_checkpoint import late_phase_pause_record

    row = current.get("budget_pause") if isinstance(current.get("budget_pause"), dict) else {}
    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    late = late_phase_pause_record(current).get("grant") or {}
    return bool(fence.get("fence_id") and (
        grant.get("owner_pause_fence_id") == fence["fence_id"]
        and not grant.get("revoked_at") and row.get("state") in {STATE_RESUME_GRANTED, STATE_RESUMED}
        or late.get("fence_id") == fence["fence_id"] and not late.get("revoked_at")))


def install_fence(root_drive: Any, root_task_id: str, *, request_id: str,
                  late_work: bool = False, requested_by: str = "owner") -> Tuple[Dict[str, Any], bool]:
    """Close the root's fence durably; ``(fence, created)``.

    Idempotent by ``request_id`` across Resume and later Pause generations;
    every acknowledged action stays on this same root projection. A
    second press while closed names that fence, never an additional Pause,
    unless outstanding Resume authority names it: then this fresh Pause gets
    its own fence identity, which that grant cannot release. A
    terminal root refuses: a finished tree has nothing to pause — unless its
    answered task still owns late work (D10): an open post-task checkpoint
    read under this lock, or a live review operation the caller attests
    (``late_work``). Raises ``OwnerPauseRefused`` without any write on
    refusal; any write failure propagates, so the caller never acknowledges
    an undurable Pause.
    """
    from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
    from ouroboros.task_results import (
        _TRULY_TERMINAL_STATUSES, require_writable_task_result_schema,
        stamp_task_result_schema, task_result_path,
    )
    from ouroboros.utils import update_json_locked, utc_now_iso

    outcome: Dict[str, Any] = {}

    def update(current: dict) -> Optional[dict]:
        if not current:
            raise OwnerPauseRefused("root_result_missing")
        require_writable_task_result_schema(current)
        old = current.get("owner_pause") if isinstance(current.get("owner_pause"), dict) else {}
        requests = dict(old.get("requests") or {})
        if old.get("request_id"):
            requests.setdefault(old["request_id"], {
                "fence_id": old.get("fence_id"), "generation": old.get("generation")})
        prior = requests.get(request_id)
        if prior:
            replay = dict(old) if prior.get("fence_id") == old.get("fence_id") else {
                **prior, "request_id": request_id, "root_task_id": str(root_task_id),
                "state": FENCE_RELEASED}
            outcome.update(fence=replay, created=False)
            return None
        checkpoint = current.get("root_phase_checkpoint")
        late_open = isinstance(checkpoint, dict) and post_task_synthesis_is_open(checkpoint.get("post_task_synthesis"))
        if current.get("status") in _TRULY_TERMINAL_STATUSES and not (late_work or late_open):
            raise OwnerPauseRefused("task_terminal")
        if fence_closed(old) and not _resume_outstanding(current, old):
            requests[request_id] = {"fence_id": old["fence_id"], "generation": old.get("generation")}
            fence = {**old, "requests": requests}
            outcome.update(fence=fence, created=False)
            return stamp_task_result_schema({**current, "owner_pause": fence})
        fence = {
            "fence_id": uuid.uuid4().hex, "request_id": str(request_id or ""),
            # ``owner`` for the owner's own press; a stop door names itself
            # (``server_shutdown``, ``owner_restart``): the next start reads it.
            "state": FENCE_REQUESTED, "requested_by": str(requested_by or "owner"), "requested_at": utc_now_iso(),
            "requested_at_ts": time.time(), "root_task_id": str(root_task_id),
            "generation": int(old.get("generation") or 0) + 1,
            # Still closed here only under outstanding Resume authority: name it.
            **({"supersedes_fence_id": old["fence_id"]} if fence_closed(old) else {}),
        }
        requests[request_id] = {"fence_id": fence["fence_id"], "generation": fence["generation"]}
        fence["requests"] = requests
        outcome.update(fence=fence, created=True)
        return stamp_task_result_schema({**current, "owner_pause": fence})

    with launch_lock(root_drive, root_task_id):
        update_json_locked(task_result_path(pathlib.Path(root_drive), str(root_task_id)), update,
                           strict_existing_dict=True)
    return dict(outcome["fence"]), bool(outcome["created"])


def set_fence_state(root_drive: Any, root_task_id: str, *, fence_id: str, state: str,
                    expected_state: str = "", **fields: Any) -> Dict[str, Any]:
    """Compare-and-set the fence's state on its own ``fence_id`` (never another Pause's).

    ``expected_state`` also binds the state its caller read: a Resume that
    released this same fence after that read refuses a stale write here.
    """
    from ouroboros.task_results import (
        require_writable_task_result_schema, stamp_task_result_schema, task_result_path,
    )
    from ouroboros.utils import utc_now_iso

    written: Dict[str, Any] = {}

    def update(current: dict) -> Optional[dict]:
        require_writable_task_result_schema(current)
        old = current.get("owner_pause") if isinstance(current.get("owner_pause"), dict) else {}
        if str(old.get("fence_id") or "") != str(fence_id or ""):
            raise ValueError("owner pause fence identity changed")
        if expected_state and str(old.get("state") or "") != expected_state:
            raise ValueError("owner pause fence state changed")
        if str(old.get("state") or "") == state and not fields:
            written.update(old)
            return None
        fence = {**old, **fields, "state": state, f"{state}_at": utc_now_iso()}
        written.update(fence)
        from ouroboros.pause_notices import notice_fields
        notice = (notice_fields(current, root_drive, root_task_id, fence_id, "owner")
                  if state == FENCE_PAUSED and old.get("state") != FENCE_PAUSED else {})
        return stamp_task_result_schema({**current, **notice, "owner_pause": fence})

    from ouroboros.obligations import update_result
    update_result(task_result_path(pathlib.Path(root_drive), str(root_task_id)), update,
                  strict_existing_dict=True)
    return dict(written)


def release_fence(root_drive: Any, root_task_id: str, *, reason: str) -> Dict[str, Any]:
    """Reopen the root's fence for an explicit Resume (a no-op when already open).

    Called where the resumed ROOT actually starts: the worker consuming its
    exact grant, or the queue selecting a never-started root. A grant revoked
    before it ran (a restart, a Stop) therefore leaves the tree fenced.
    """
    fence = read_fence(root_drive, root_task_id)
    if not fence_closed(fence):
        return fence
    return set_fence_state(root_drive, root_task_id, fence_id=str(fence.get("fence_id") or ""),
                           state=FENCE_RELEASED, release_reason=str(reason or "owner_resume"))


def reopen_for_resume(root_drive: Any, root_task_id: str, task_id: str, *,
                      fence_id: str, grant_id: str) -> None:
    """Consume one explicit Resume's fence authority where its work starts.

    A root reopens its tree; an owner-selected child of a terminal root opens
    only its own member. A closed fence that ``install_fence`` minted over this
    outstanding Resume (``supersedes_fence_id``) is a newer owner Pause: the
    Resume is superseded, releases nothing, and the member parks at its next
    safe boundary. Any other identity change refuses.
    """
    with launch_lock(root_drive, root_task_id):
        current = read_fence(root_drive, root_task_id) if fence_id else {}
        if fence_id and current.get("fence_id") != fence_id:
            if not fence_closed(current) or current.get("supersedes_fence_id") != fence_id:
                raise ValueError("owner_pause_fence_changed")
        elif root_task_id == task_id:
            if fence_closed(current):
                # One root Resume restores every saved unsent owner before opening
                # their shared fence, including acceptance beside post-task work.
                from ouroboros.acceptance_late import resume_paused_acceptance_preparations
                resume_paused_acceptance_preparations(root_drive, root_task_id, current)
            release_fence(root_drive, root_task_id, reason="owner_resume_consumed")
        else:
            select_member_resume(root_drive, root_task_id, task_id, fence_id=fence_id, grant_id=grant_id)


def select_member_resume(root_drive: Any, root_task_id: str, task_id: str, *,
                         fence_id: str, grant_id: str) -> None:
    """Consume one child Resume under a terminal root; keep its siblings fenced.

    Caller holds ``launch_lock``. The consumed explicit grant, root terminality
    and exact fence are checked again here; neither lineage nor a terminal
    transition grants this exception. A new Pause has no selected members.
    """
    from ouroboros.budget_pause import budget_pause_row, STATE_RESUMED
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result
    from ouroboros.utils import utc_now_iso

    root = load_task_result(root_drive, root_task_id, strict=True) or {}
    member = load_task_result(root_drive, task_id, strict=True) or {}
    row = budget_pause_row(pathlib.Path(root_drive), task_id)
    grant = row.get("grant") or {}
    if (task_id == root_task_id or root.get("status") not in _TRULY_TERMINAL_STATUSES
            or str(member.get("root_task_id") or (member.get("metadata") or {}).get("root_task_id") or "") != root_task_id
            or row.get("state") != STATE_RESUMED or not grant_id
            or grant.get("grant_id") != grant_id or not grant.get("consumed_at")
            or grant.get("revoked_at") or grant.get("authority") != "explicit_resume"
            or grant.get("selected_by") != "owner"
            or grant.get("owner_pause_fence_id") != fence_id):
        raise OwnerPauseRefused("member_resume_authority_changed")
    fence = read_fence(root_drive, root_task_id)
    if fence.get("fence_id") != fence_id or not fence_closed(fence):
        raise OwnerPauseRefused("owner_pause_fence_changed")
    selected = dict(fence.get("selected_members") or {})
    selected[task_id] = {"grant_id": grant_id, "selected_at": utc_now_iso()}
    set_fence_state(root_drive, root_task_id, fence_id=fence_id, state=fence["state"],
                    selected_members=selected)


# --- member side: the gate every launch family shares ----------------------------------

def admit_delegated_start(drive_root: Any, payload: Dict[str, Any]) -> bool:
    """Bind billing and register START_REQUESTED before releasing launch admission.

    This is custody, not an observation of transport bytes. A crash after the
    claim must be reconciled using its existing idempotency key.
    """
    from types import SimpleNamespace
    from ouroboros.delegate_custody import emit, START_REQUESTED
    from ouroboros.usage_admission import task_billing_fields

    source = SimpleNamespace(drive_root=drive_root, task_id=str(payload.get("task_id") or ""),
                             root_task_id=str(payload.get("root_task_id") or payload.get("task_id") or ""))
    binding = task_billing_fields({"id": source.task_id}, source.root_task_id, None, drive_root)
    payload["billing_group"] = {k: v for k, v in binding.items() if k.startswith("billing_group_")}
    with launch_admission(source):
        return emit(drive_root, START_REQUESTED, payload)


def _member_coordinates(source: Any) -> Tuple[str, str, str]:
    """``(root_drive, root_task_id, task_id)`` of a tool context or usage scope."""
    if getattr(source, "non_task_operation", False) is True:
        return "", "", ""
    meta = getattr(source, "task_metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    task_id = str(getattr(source, "task_id", "") or "")
    root_task_id = str(meta.get("root_task_id") or getattr(source, "root_task_id", "") or task_id)
    root_drive = (meta.get("budget_drive_root") or getattr(source, "budget_drive_root", None)
                  or getattr(source, "drive_root", None) or "")
    return str(root_drive or ""), root_task_id, task_id


def member_fence(source: Any) -> Dict[str, Any]:
    """Read the shared fence strictly: unreadable authority refuses new effects."""
    root_drive, root_task_id, _task_id = _member_coordinates(source)
    if not root_drive or not root_task_id:
        return {}
    marker = _REVIEW_EPISODE.get()
    if (isinstance(marker, dict) and marker.get("root_task_id") == root_task_id
            and marker.get("pause_before_launch")):
        # The author detached before this slot's first handoff. Its preparation
        # may still unwind after Resume, but that does not authorize a late send.
        return {"state": FENCE_REQUESTED, "reason": "owner_pause_before_review_launch"}
    try:
        from ouroboros.model_wait import current_model_wait
        owner = current_model_wait()
        required = getattr(source, "task_lifecycle_bound", False) or (
            owner is not None and owner.task_id == _task_id)
        if required and not _result_path(root_drive, root_task_id).is_file():
            raise ValueError("owner_pause_authority_missing")
        fence = read_fence(root_drive, root_task_id)
    except Exception:
        log.warning("Owner pause authority unreadable for %s", root_task_id, exc_info=True)
        return {"state": "unknown", "reason": "owner_pause_authority_unreadable"}
    if fence_closed(fence) and _task_id in (fence.get("selected_members") or {}):
        return {}  # One consumed owner Resume, bound to this fence generation.
    if fence_closed(fence) and _episode_dispatched_before(fence, root_task_id):
        return {}  # The already-started review episode finishes its own check.
    return fence if fence_closed(fence) else {}


@contextmanager
def review_episode(root_task_id: str, operation: Any = None):
    """Bind ONE physical review attempt's Pause exception to this context (owner 2026-10-08).

    The owner's full variant: a reviewer that already started finishes the
    SAME check while its author is paused — its further model rounds,
    inspections and run starts — but nothing new starts. The marker yielded
    here is armed only by this attempt's first ACTUAL physical handoff: the
    exact model sender handed to its executor (``submit_model``), or an
    operation submitted at its final start point (``operation_start``: the
    delegated run's POST, a process start, a tool executor) — each still under
    the root's launch lock that admitted it with the fence OPEN. Registering
    an operation, a ``START_REQUESTED`` row (the POST still meets the fence), a
    noted dispatch, a preparation, a capacity probe or a ledger transition is
    no authority. A Pause accepted later has
    a newer generation, so the armed attempt alone passes it; an attempt that
    had not launched when the Pause landed is refused at $0 like any other
    launch. The author's own context never carries a marker: its tools, sends,
    commit, apply and publication stay refused.
    """
    marker = {"root_task_id": str(root_task_id or ""), "operation": operation, "armed_generation": None}
    token = _REVIEW_EPISODE.set(marker)
    try:
        yield marker
    finally:
        _REVIEW_EPISODE.reset(token)


def episode_armed(marker: Any) -> bool:
    """Whether this physical review attempt launched before any later Pause."""
    return isinstance(marker, dict) and marker.get("armed_generation") is not None


def _episode_dispatched_before(fence: Dict[str, Any], root_task_id: str) -> bool:
    marker = _REVIEW_EPISODE.get()
    if (not episode_armed(marker) or marker.get("root_task_id") != str(root_task_id or "")
            or getattr(marker.get("operation"), "closed", False)):
        return False
    try:
        return int(marker["armed_generation"]) < int(fence.get("generation") or 0)
    except (TypeError, ValueError):
        return False


def arm_review_episode(source: Any) -> None:
    """Call INSIDE ``launch_admission`` right after the actual handoff succeeded:
    the fence admitted it under the same launch lock, so this attempt's episode
    began before any Pause that lock has not yet let in. Arms once."""
    root_drive, root_task_id, _task_id = _member_coordinates(source)
    marker = _REVIEW_EPISODE.get()
    if not isinstance(marker, dict) or episode_armed(marker) or marker.get("root_task_id") != str(root_task_id or ""):
        return
    try:
        marker["armed_generation"] = int(read_fence(root_drive, root_task_id).get("generation") or 0)
    except Exception:
        # Unarmed is the conservative fact: a later Pause then refuses this attempt.
        log.warning("Review episode of %s was not armed", root_task_id, exc_info=True)


def scope_fence() -> Dict[str, Any]:
    """The closed fence over the model send bound to this execution context."""
    try:
        from ouroboros.usage_accounting import current_usage_scope

        scope = current_usage_scope()
    except Exception:
        return {}
    if scope is None or not str(getattr(scope, "task_id", "") or ""):
        return {}
    return member_fence(scope)


@contextmanager
def launch_lock(root_drive: Any, root_task_id: str, *, wait: bool = True):
    """Serialize local admission; ``wait=False`` never waits under a queue lock."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    if not root_drive or not root_task_id:
        yield
        return
    path = _result_path(root_drive, root_task_id).with_suffix(".launch.lock")
    fd = acquire_exclusive_file_lock(path, owner_aware_stale=True,
                                     **({} if wait else {"timeout_sec": 0.0}))
    if fd is None:
        raise OwnerPauseRefused("owner_launch_authority_unavailable")
    try:
        yield
    finally:
        release_exclusive_file_lock(path, fd)


@contextmanager
def launch_admission(source: Any, *, root_resume: Optional[Dict[str, Any]] = None):
    """Register under the fence, or hand a member its exact explicit Resume.

    A resumed worker must start to consume its grant. A root reopens the tree;
    an owner-selected child of a terminal root reopens only that member.
    This exception authorizes that handoff only, never its tools or sends.
    """
    root_drive, root_id, task_id = _member_coordinates(source)
    from ouroboros.model_wait import current_model_wait
    owner = current_model_wait()
    if owner is not None and owner.task_id == task_id and owner.closed:
        raise OwnerPauseRefused("operation_already_returned")
    with launch_lock(root_drive, root_id):
        fence = member_fence(source)
        selected_resume_start = False
        if fence:
            from ouroboros.budget_pause import budget_pause_row, STATE_RESUME_GRANTED

            allowed = False
            if root_resume and fence_closed(fence):
                row = budget_pause_row(pathlib.Path(root_drive), task_id)
                grant = row.get("grant") or {}
                root_allowed = task_id == root_id
                if not root_allowed and grant.get("selected_by") == "owner":
                    from ouroboros.task_results import load_task_result, _TRULY_TERMINAL_STATUSES
                    root_allowed = (load_task_result(root_drive, root_id, strict=True) or {}).get("status") in _TRULY_TERMINAL_STATUSES
                allowed = bool(row.get("state") == STATE_RESUME_GRANTED
                    and root_allowed
                    and (grant.get("owner_pause_fence_id") or row.get("owner_fence_id")) == fence.get("fence_id")
                    and row.get("pause_id") == root_resume.get("pause_id")
                    and grant.get("grant_id") == root_resume.get("grant_id")
                    and grant.get("authority") == "explicit_resume"
                    and not grant.get("revoked_at") and not grant.get("consumed_at"))
            if not allowed:
                raise OwnerPauseRefused(str(fence.get("reason") or "owner_pause"))
            selected_resume_start = task_id != root_id
        from ouroboros.budget_pause import budget_pause_row, STATE_PAUSING, STATE_PAUSED, STATE_RESUME_GRANTED

        if root_drive and task_id:
            for member_id in {root_id, task_id}:
                try:
                    row = budget_pause_row(pathlib.Path(root_drive), member_id)
                except Exception as exc:
                    raise OwnerPauseRefused("model_sleep_authority_unreadable") from exc
                if row.get("reason") != "sleep" or row.get("state") not in {
                        STATE_PAUSING, STATE_PAUSED, STATE_RESUME_GRANTED}:
                    continue
                if member_id == root_id and task_id != root_id:
                    selected = read_fence(root_drive, root_id).get("selected_members") or {}
                    if selected_resume_start or task_id in selected:
                        # Explicit member Resume outlives its terminal root's
                        # saved sleep. It does not release any sibling or root.
                        continue
                grant = row.get("grant") or {}
                resume_start = bool(root_resume and member_id == task_id
                    and row.get("state") == STATE_RESUME_GRANTED
                    and row.get("pause_id") == root_resume.get("pause_id")
                    and grant.get("grant_id") == root_resume.get("grant_id")
                    and not grant.get("revoked_at") and not grant.get("consumed_at"))
                if not resume_start:
                    raise OwnerPauseRefused("model_sleep")
        yield


def tree_member_results(root_drive: Any, root_task_id: str) -> Dict[str, Dict[str, Any]]:
    """Select complete retained membership, then strictly read its authority.

    Reuse the stat-invalidated lineage memo: unchanged foreign bodies are not
    parsed on every Pause tick. Unreadable or inadmissible facts refuse the
    census. Rows without a top-level root still need their legacy metadata read.
    The memo selects files only; it never supplies live status or custody.
    """
    from ouroboros.task_result_facts import raw_result_facts
    from ouroboros.task_results import load_task_result, task_results_dir

    facts, malformed = raw_result_facts(task_results_dir(root_drive, create=False))
    if malformed:
        raise ValueError("tree_membership_unreadable")
    members = {}
    for name, fact in facts.items():
        task_id = pathlib.Path(name).stem
        if fact.get("schema_refusal") or fact.get("task_id") != task_id:
            raise ValueError("tree_membership_unreadable")
        if fact.get("root_task_id") and fact["root_task_id"] != root_task_id:
            continue
        row = load_task_result(root_drive, task_id, strict=True)
        if row is None:
            raise ValueError("tree_member_disappeared")
        if str(row.get("root_task_id") or (row.get("metadata") or {}).get("root_task_id")
               or task_id) == root_task_id:
            members[task_id] = row
    return members


@contextmanager
def tool_handoff(source: Any, name: str):
    """Track the local invocation separately from its owned physical operations.

    Actual handler unwind closes this invocation regardless of business outcome.
    Pending host receipts and independently owned executors remain custody;
    neither a result code nor metadata can certify their completion.
    """
    from ouroboros.task_results import stamp_task_result_schema
    from ouroboros.utils import update_json_locked, utc_now_iso

    outcome: Dict[str, Any] = {"not_started": True}
    root_drive, root_id, task_id = _member_coordinates(source)
    if not root_drive or not task_id:
        yield outcome
        return
    path = _result_path(root_drive, task_id)
    op_id = uuid.uuid4().hex
    outcome["operation_id"] = op_id
    outcome["claim_path"] = path
    token = _TOOL_OPERATION.set((source, name, outcome))
    def claim(current):
        if not current.get("status") or current.get("task_id") != path.stem:
            raise OwnerPauseRefused("owner_pause_authority_unreadable")
        outstanding = dict(current.get("launch_handoffs") or {})
        outstanding[op_id] = {"tool": name, "task_id": task_id, "root_task_id": root_id,
                              "state": "claimed", "claimed_at": utc_now_iso(),
                              **({"local_owner": local_owner} if local_owner else {})}
        return stamp_task_result_schema({**current, "launch_handoffs": outstanding})
    from ouroboros.model_wait import current_model_wait
    owner = current_model_wait()
    local_owner = ({"pid": os.getpid(), "process_birth": owner.answer_owner_birth,
                    "task_attempt": owner.attempt}
                   if owner is not None and owner.task_id == task_id else {})
    claimed = False
    try:
        # Stateful tools were submitted to their sticky executor by the loop.
        # This ticket belongs only to that invocation, not its nested launches.
        handed = _HANDED_TOOL.get()
        with _CACHE_LOCK:
            outcome["handed"] = bool(handed and handed["invocation"] == (id(source), name) and not handed["claimed"])
            if outcome["handed"]:
                handed["claimed"] = True  # Shared by copied contexts: exactly one consumer.
        from ouroboros.model_wait import current_model_wait
        owner = current_model_wait()
        if owner is not None and owner.task_id == task_id and owner.closed:
            raise OwnerPauseRefused("operation_already_returned")
        with (launch_lock(root_drive, root_id) if outcome["handed"] else launch_admission(source)):
            # A standalone tool context is not a lifecycle producer. Never
            # manufacture a status-less result that the next strict read must
            # refuse. A not-yet-published member uses its existing root owner.
            if not path.exists():
                path = _result_path(root_drive, root_id)
                outcome["claim_path"] = path
            if path.exists():
                update_json_locked(path, claim, strict_existing_dict=True)
                claimed = True
            elif getattr(source, "task_lifecycle_bound", None) is not False:
                raise OwnerPauseRefused("owner_pause_authority_unreadable")
        yield outcome
    finally:
        _TOOL_OPERATION.reset(token)
        def settle(current):
            if not current.get("status") or current.get("task_id") != path.stem:
                raise OwnerPauseRefused("owner_pause_authority_unreadable")
            outstanding = dict(current.get("launch_handoffs") or {})
            claim = outstanding.get(op_id)
            if claim is not None:
                claim = {**claim, "state": "returned", "returned_at": utc_now_iso()}
                from ouroboros.tool_custody import claim_still_owns_effect
                if claim_still_owns_effect(root_drive, claim):
                    outstanding[op_id] = claim
                else:
                    outstanding.pop(op_id, None)
            return stamp_task_result_schema({**current, "launch_handoffs": outstanding})
        # Close local admission even if durable cleanup cannot acquire its lock.
        # Failure retains custody; it must not replace an executed result with
        # a false NOT STARTED answer. Settlement still serializes with starts.
        outcome["closed"] = True
        try:
            with launch_lock(root_drive, root_id):
                if claimed and (outcome.get("settled") is True or outcome.get("not_started") is True):
                    update_json_locked(path, settle, strict_existing_dict=True)
        except Exception:
            log.warning("Tool %s custody could not settle (%s)", name, op_id, exc_info=True)


def current_tool_operation(source: Any, name: str) -> str:
    """Exact synchronous invocation identity, never an exemption by tool name."""
    active = _TOOL_OPERATION.get()
    return str(active[2].get("operation_id") or "") if active and active[0] is source and active[1] == name else ""


def record_mcp_call_returned(name: str) -> None:
    """The host joined this open invocation's MCP call: its final response and
    clean session/transport exit arrived. Only that caller records it, never a
    runner still running after its timeout nor result text or provider meta;
    it settles this claim alone, not other processes, runs or money."""
    active = _TOOL_OPERATION.get()
    if active and active[1] == name and not active[2].get("closed"):
        active[2]["mcp_call_returned"] = True


@contextmanager
def operation_start(source: Any = None):
    """Final local operation-start point after preparation, serialized with Pause.

    Hold only across the local submission (for example Popen), never its wait.
    Preparation must finish first. A nested start refused here proves no effect
    for that submission, not for an earlier start in the same invocation.
    """
    active = _TOOL_OPERATION.get()
    source = source if source is not None else (active[0] if active else None)
    with launch_admission(source):
        if active and active[0] is source:
            if active[2].get("closed"):
                raise OwnerPauseRefused("operation_already_returned")
            active[2]["not_started"] = False
        yield
        arm_review_episode(source)  # the submission inside succeeded under this same lock


def submit_tool(source: Any, name: str, submit: Any, function: Any, *args: Any):
    """Hand one invocation to its existing sticky executor under Pause exclusion."""
    with launch_admission(source):
        context = copy_context()
        context.run(_HANDED_TOOL.set, {"invocation": (id(source), name), "claimed": False})
        handed = submit(context.run, function, *args)
        arm_review_episode(source)
        return handed


def run_operation(source: Any, function: Any, /, *args: Any, **kwargs: Any):
    """Submit an opaque synchronous call; join outside the short launch lock.

    The forwarders' own parameters are positional-only: every keyword, such as
    a tool argument named ``source`` or ``function``, belongs to the callee.

    The worker owns only this call. Its copied context does not authorize any
    nested process, model or delegated submission after a subsequent Pause.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="operation") as executor:
        with operation_start(source):
            context = copy_context()
            future = executor.submit(context.run, function, *args, **kwargs)
        return future.result()


def run_tool_handler(source: Any, function: Any, /, *args: Any, **kwargs: Any):
    """The host joins the actual body, including its exception unwind.

    A caller's outer timeout cannot reach this finally while the body still runs.
    This closes local launch capability, not processes, receipts or remote effects.
    """
    active = _TOOL_OPERATION.get()
    try:
        if active and active[0] is source and active[2].get("handed"):
            if active[2].get("closed"):
                raise OwnerPauseRefused("operation_already_returned")
            active[2]["not_started"] = False
            return function(*args, **kwargs)  # Keep browser/greenlet thread affinity.
        from ouroboros.tools.process_facts import process_facts_handoff
        with process_facts_handoff(function) as invoke:
            return run_operation(source, invoke, *args, **kwargs)
    finally:
        if active and active[0] is source:
            active[2]["settled"] = True


def submit_async_operation(source: Any, function: Any, /, *args: Any, **kwargs: Any):
    """Submit to the current event loop without executing the body under a lock."""
    import asyncio

    async def invoke():
        return await function(*args, **kwargs)

    with operation_start(source):
        return asyncio.create_task(invoke())


def submit_async_preparation(function: Any, *args: Any, source: Any = None, **kwargs: Any):
    """Admit a nested transport's local preparation before it opens anything.

    Entering an MCP stdio transport starts the server process: that handoff
    serializes with Pause here, so an accepted Pause refuses it. It is not the
    operation start: ``submit_async_operation`` stays the final gate for the
    call itself (and for a returned invocation). The body runs after the short
    lock, never under it. ``source`` names the task outside a tool invocation
    (a task's MCP discovery); otherwise the active invocation's task decides.
    """
    import asyncio

    active = _TOOL_OPERATION.get()

    async def invoke():
        return await function(*args, **kwargs)

    with launch_admission(source if source is not None else active[0] if active else None):
        return asyncio.create_task(invoke())


def model_handed_off(attempt_id: str = "") -> bool:
    """Only this physical attempt may finish despite a later owner Pause."""
    if not attempt_id:
        from ouroboros.usage_accounting import last_physical_attempt_capture
        attempt_id = getattr(last_physical_attempt_capture(), "attempt_id", "")
    return bool(attempt_id and _HANDED_MODEL.get() == attempt_id)


def submit_model(reservation: Any, submit: Any, function: Any):
    """Transfer the exact sender to an executor, mutually exclusive with Pause."""
    from ouroboros.llm_attempt import _PhysicalSendNotStarted, require_physical_dispatch_window

    try:
        with launch_admission(reservation.scope):
            require_physical_dispatch_window()
            context = copy_context()
            context.run(_HANDED_MODEL.set, reservation.attempt_id)
            handed = submit(context.run, function)
            arm_review_episode(reservation.scope)  # the exact sender now owns these bytes
            return handed
    except OwnerPauseRefused as exc:
        raise _PhysicalSendNotStarted(str(exc)) from exc
