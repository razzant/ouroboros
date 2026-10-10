"""A paid review operation under its author's owner Pause (owner 2026-10-08, full variant).

The owner chose: the author is truly paused while reviewers it ALREADY launched
finish the same check separately and visibly; Resume does not wait for them and
later collects their result without another purchase; nothing new starts and
the author commits, applies or publishes nothing. This module is that seam
between the review custody (``review_custody``/``review_operation``) and the
author's fence (``owner_pause``):

- the author's synchronous drain detaches on Pause, including while a slot
  prepares to launch. Under the physical handoff's same root launch lock, an
  unarmed slot latches a first-launch refusal; its still-live preparation may
  unwind later but cannot send, even if Resume came first. An armed reviewer
  (``owner_pause.review_episode``) finishes its already-started episode;
- the row the author takes with it is PENDING (``in_flight`` for a launched
  episode, ``pending_dispatch`` for a retained unsent preparation,
  ``late_result_pending``), never a verdict; the physical operation stays live
  and settles late into this process's custody, so the same unchanged request
  after Resume relays or replays it;
- the operation is listed as detached until it CLOSES (its workers settled and
  published, ``ReviewOperation.maybe_close``), which keeps the author's process
  parked warm; its state is what this process knows — running, closed here, or
  unknown — never "settled" from absence.
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Tuple

from ouroboros.owner_pause import episode_armed
from ouroboros.review_operation import _CLOSED_DETACHED, _LIVE, _LOCK, ReviewOperation

log = logging.getLogger(__name__)

# How often the author's drain re-reads its own owner fence while its review
# operation runs: a poll interval of the existing wait, not a bound.
DETACH_POLL_SEC = 1.0


@contextmanager
def slot_episode(entry: Any, scope: Any):
    """Bind one physical review attempt's Pause exception for its own worker and
    for the author's drain, which reads whether it launched (``author_detaches``)."""
    from ouroboros.owner_pause import _member_coordinates, launch_lock, review_episode

    root_drive, root_task_id, _ = _member_coordinates(scope)
    with review_episode(root_task_id, entry.operation) as marker:
        with launch_lock(root_drive, root_task_id):
            # The drain may detach before this worker even binds its marker.
            # Carry that refusal into the one marker, never replace an armed one.
            marker["pause_before_launch"] = entry.pause_before_launch
            entry.episode = marker
        yield marker


def drain_slice(operation: Any, remaining: float) -> float:
    """While its author may pause, the drain re-reads the author's fence each slice."""
    return min(remaining, DETACH_POLL_SEC) if operation is not None else remaining


def _author_coordinates(parent: Any) -> Tuple[str, str, str]:
    """``(root_drive, root_task_id, task_id)`` of the author wait an operation serves."""
    from ouroboros.owner_pause import _member_coordinates

    source = getattr(parent, "tool_context", None)
    if source is None:
        task = parent.task if isinstance(getattr(parent, "task", None), dict) else {}
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        source = SimpleNamespace(
            task_id=parent.task_id,
            root_task_id=str(task.get("root_task_id") or metadata.get("root_task_id") or parent.task_id),
            budget_drive_root=parent.canonical_root)
    return _member_coordinates(source)


def author_fence_closed(parent: Any) -> bool:
    """A NEW panel under the author's accepted (or unreadable) Pause never starts.

    An early $0 refusal only: no allowance is minted here. Each physical attempt
    earns its own Pause exception at its first admitted handoff
    (``owner_pause.review_episode``).
    """
    from ouroboros.owner_pause import member_fence

    root_drive, root_task_id, task_id = _author_coordinates(parent)
    if not root_drive or not root_task_id:
        return False
    source = getattr(parent, "tool_context", None) or SimpleNamespace(
        task_id=task_id, root_task_id=root_task_id, budget_drive_root=root_drive)
    return bool(member_fence(source))


def author_paused(operation: ReviewOperation) -> bool:
    """Whether the owner's Pause holds this operation's author now.

    Read directly (never through ``member_fence``, which answers "open" for an
    armed reviewer): the author's root fence is closed and the author is not a
    selected member. Unreadable authority reads as not paused: the author's
    drain keeps waiting, which is the pre-existing behavior.
    """
    from ouroboros.owner_pause import fence_closed, read_fence

    root_drive, root_task_id, task_id = _author_coordinates(operation.wait)
    try:
        fence = read_fence(root_drive or str(operation.result_root), root_task_id)
    except Exception:
        log.debug("author pause fence unreadable during the review drain", exc_info=True)
        return False
    return bool(fence_closed(fence) and task_id not in (fence.get("selected_members") or {}))


def author_detaches(operation: Any, pending_entries: Iterable[Any]) -> bool:
    """Detach promptly, retaining workers and refusing every still-unsent slot.

    Serialize with the first physical handoff: either that handoff armed the
    same episode first and it finishes, or Pause's refusal wins and survives
    Resume. No worker is reported ended merely because the author left.
    """
    from ouroboros.owner_pause import launch_lock

    if not author_paused(operation):
        return False
    root_drive, root_task_id, _ = _author_coordinates(operation.wait)
    with launch_lock(root_drive, root_task_id):
        if not author_paused(operation):
            return False
        for entry in pending_entries:
            if entry is None or episode_armed(entry.episode) or entry.event.is_set():
                continue
            # Fresh local work has started_at even before its first POST; its
            # write-ahead token proves no handoff. Exact recovery leaves it empty.
            if not entry.started_at and any(entry.retry_state.get(key) for key in ("pending_invocation_id", "delegated_run_id")):
                continue  # A recovered remote invocation is not an unsent local preparation.
            entry.pause_before_launch = True
            if isinstance(entry.episode, dict):
                entry.episode["pause_before_launch"] = True
        return True


def detached_actor(slot: Any, entry: Any, error_actor: Callable[..., Any], *, usage_ctx: Any,
                   request: Any, operation: Any) -> Any:
    """The PENDING row the author takes with it when the owner's Pause detaches it.

    The physical operation stays live and settles late into this process's
    custody (``_review_settled_attempts``, keyed by the attempt identity), so the
    same request after Resume replays it at $0. Never a verdict, never a stop.
    """
    from ouroboros.review_custody import _ACTIVE_LOCK
    from ouroboros.review_operation import stamp_review_controller

    with _ACTIVE_LOCK:
        if entry.event.is_set() and entry.actor is not None:
            try:
                return copy.deepcopy(entry.actor)
            except Exception:
                return entry.actor
        entry.detached = True
        if operation is not None:
            operation.author_detached = True
    preparing = bool(entry.pause_before_launch)
    message = (
        "The owner paused the author before this reviewer's first launch. Its preparation is still owned "
        "by this operation; no reviewer was sent. That preparation cannot send after Resume and must "
        "finish before its final refusal is known. No verdict yet; nothing was committed or applied."
        if preparing else
        "The owner paused the author while this reviewer was already running: it finishes the "
        "same check separately. No verdict yet; nothing was committed or applied. After Resume the "
        "same unchanged request rejoins this attempt in this process; nothing is sent again automatically.")
    actor = error_actor(slot, message, entry.operation_id, "pending_dispatch" if preparing else "in_flight")
    actor.late_result_pending = True
    actor.awaiting_since = entry.started_at
    usage = dict(getattr(actor, "usage", None) or {})
    usage["owner_pause_detached"] = True
    if preparing:
        usage["owner_pause_preparing"] = True
    usage.update({key: str(entry.retry_state.get(key) or "") for key in ("pending_invocation_id", "delegated_run_id")
                  if entry.retry_state.get(key)})
    actor.usage = usage
    stamp_review_controller(actor, entry)
    if usage_ctx is not None:
        detached = getattr(usage_ctx, "_review_detached_operations", None)
        if not isinstance(detached, list):
            detached = []
            setattr(usage_ctx, "_review_detached_operations", detached)
        detached.append({"owner_id": getattr(operation, "owner_id", ""),
                         "surface": str(getattr(request, "surface", "") or ""),
                         "retry_key": str(getattr(request, "retry_key", "") or ""),
                         "slot_id": str(getattr(slot, "slot_id", "") or ""),
                         "operation_id": entry.operation_id, "attempt_key": entry.key})
    return actor


def review_operation_state(owner_id: str) -> str:
    """``running`` while this process owns the operation live, ``closed`` once it
    closed here after its workers settled and published, else ``unknown``.

    A process-local fact for the resumed author's notice; absence is never read
    as settlement (the operation may have lived in a process that is gone).
    """
    owner_id = str(owner_id or "")
    with _LOCK:
        operation = _LIVE.get(owner_id)
        if operation is not None and not operation.closed:
            return "running"
        return "closed" if owner_id in _CLOSED_DETACHED else "unknown"


def live_detached_operations(task_id: str) -> List[Dict[str, Any]]:
    """The open review operations of ``task_id`` its paused author detached from.

    Their launched reviewers keep running in THIS process, so the author parks
    warm here and the process stays until each operation closes (after its
    settlement publication), never released on the settle event alone. Facts
    only: owner id, surface, paid identity.
    """
    task_id = str(task_id or "")
    with _LOCK:
        operations = [op for op in _LIVE.values()
                      if op.task_id == task_id and not op.closed and op.author_detached]
    return [{"owner_id": op.owner_id, "surface": op.surface, "retry_key": op.retry_key,
             "controller": dict(op.identity)} for op in operations]
