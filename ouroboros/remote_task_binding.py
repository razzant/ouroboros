"""Binding ONE task to the remote session its SEALED placement names.

The broker keys every OpenSSH session by ``(connection, project, workspace,
server generation)`` and answers a worker's operation only for a task it has
already bound to one of them (``remote_session_admission.session_for_ref``).
Nothing laid that binding on the ordinary road a task travels — ``/api/tasks``
sealed the placement, the queue carried it, the supervisor assigned it, and the
worker arrived at the broker with a ``task_id`` no session had ever heard of. The
symptom was total: every tool in a remote project answered
``REMOTE_EXECUTION_UNAVAILABLE: task_session_unbound``.

This module is the missing wire, and it is deliberately placed at ASSIGNMENT
rather than at task creation or lazily inside the broker:

* **Not at creation.** The creation surface resolves the placement in a process
  that may not be the one that runs the task. A task can wait in the queue across
  a restart, and ``_task_sessions`` is in-memory server-generation state — a
  binding laid at creation is simply gone by the time the work starts, which is
  the same hole one restart later.
* **Not inside the broker.** Opening a session needs the owner's connection row,
  and the broker must not import Home authorities (see
  ``remote_transfer.recover_pending_scopes`` — recovery is a HOOK for exactly this
  reason). A broker that reached for the connection store would become a second
  admission authority.
* **At assignment**, in the supervisor: it happens exactly once per attempt, in
  the same server generation that will run the work, AFTER any restart, and in a
  process that legitimately holds both the connection store and the broker. The
  SEALED placement is the only thing consulted about WHERE the task goes, so the
  binding cannot disagree with the routing the owner approved.

Admission is idempotent per session key, so N tasks of one project share ONE
session and each holds its own task binding; a subagent inherits its parent's
project scope on its own task record and therefore binds to the SAME session
instead of opening a second one.

The functions are pure over a task dict plus the two authorities, in the style of
``project_lease``: the supervisor calls them, they never call the supervisor.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# The per-task binding receipt, written onto the queued task dict. Underscored
# because it is TRANSIENT server-generation state: `persist_queue_snapshot` keeps
# an explicit key whitelist and does not carry it, so a task restored after a
# restart correctly reads as unbound and is bound again against the live broker.
BIND_STATE_KEY = "_remote_session_bind"
# A transient target/network failure gets a few assignment ticks before the task
# is refused terminally: a queued task must not die of a one-second hiccup, and it
# must not wait forever for a target that is gone either.
MAX_BIND_ATTEMPTS = 3


def live_server_generation() -> str:
    """The generation of the broker in THIS process, or "" when there is none.

    Read rather than remembered: a binding is only meaningful against the broker
    that holds the session, so a task carrying a receipt from another generation
    must read as unbound.
    """

    try:
        from ouroboros.remote_workspace import get_remote_workspace_service

        return str(getattr(get_remote_workspace_service(), "server_generation", "") or "")
    except Exception:
        return ""


def remote_binding_scope(task: Any) -> dict[str, Any] | None:
    """The session identity a task's SEALED placement names, or ``None``.

    ``None`` means "not a remote task" and is the answer for every local/docker
    placement, so the supervisor's remote arm is inert for ordinary work.

    The project scope is read from the task's own ``project_id`` (a subagent
    carries its parent's, which is what makes it share the parent's session), then
    from the placement fence, and finally falls back to the workspace identity —
    the same order of preference ``workspace_admission.remote_session_facts``
    uses, so the key built here is the key the placement was admitted under.
    """

    from ouroboros.project_facts import sanitize_project_id
    from ouroboros.workspace_admission import PLACEMENT_FENCE_KEY
    from ouroboros.workspace_ref import workspace_ref_for

    if not isinstance(task, Mapping):
        return None
    ref = workspace_ref_for(task)
    if ref is None or ref.kind != "ssh":
        return None
    fence = task.get(PLACEMENT_FENCE_KEY)
    fenced = str(fence.get("project_id") or "") if isinstance(fence, Mapping) else ""
    project_id = sanitize_project_id(task.get("project_id")) or fenced or ref.workspace_id
    return {
        "connection_id": ref.connection_id,
        "remote_root": ref.remote_root,
        "workspace_id": ref.workspace_id,
        "project_id": project_id,
        "workspace_ref": ref.to_payload(),
    }


def remote_binding_pending(task: Any) -> bool:
    """Whether this task still needs its remote session bound.

    Also the dispatch gate: a remote task that is not bound must NOT reach a
    worker, because the worker's first tool call would be the
    ``task_session_unbound`` refusal this module exists to remove.
    """

    if remote_binding_scope(task) is None:
        return False
    state = task.get(BIND_STATE_KEY)
    if not (isinstance(state, Mapping) and state.get("status") == "bound"):
        return True
    # The receipt names the task id AND the generation it was taken for. Both are
    # load-bearing: a timeout retry is `dict(task)` with a NEW id (so an inherited
    # receipt would let an unbound task dispatch), and a binding is meaningless
    # against a broker that is not the one holding the session.
    if str(state.get("task_id") or "") != str(task.get("id") or ""):
        return True
    return str(state.get("server_generation") or "") != live_server_generation()


def bind_remote_task_session(task: Any) -> dict[str, Any]:
    """Open-or-reuse the task's target session and BIND the task id to it.

    Returns a typed outcome the caller projects into queue state:

    * ``bound`` — the broker holds a session for this task; dispatch may proceed.
    * ``retry`` — a retryable target/transport failure with attempts remaining;
      the task stays queued and is tried again on a later assignment tick.
    * ``refused`` — a decided refusal (identity changed, connection retired, no
      broker, attempts exhausted). The caller must end the task with this
      evidence rather than dispatch it blind.
    * ``skipped`` — not a remote task, or a task with no id.

    Every failure keeps the TYPED code the target/broker produced: "the host
    identity changed" and "this build has no transport" must never arrive as one
    generic string.
    """

    from ouroboros.connection_store import get_connection
    from ouroboros.remote_workspace import get_remote_workspace_service
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError

    scope = remote_binding_scope(task)
    task_id = str(task.get("id") or "") if isinstance(task, Mapping) else ""
    if scope is None or not task_id:
        return {"status": "skipped", "code": "", "message": ""}
    state = task.get(BIND_STATE_KEY)
    attempt = (int(state.get("attempts") or 0) if isinstance(state, Mapping) else 0) + 1
    try:
        row = get_connection(scope["connection_id"], include_retired=False)
        if row is None:
            raise RemoteWorkspaceError(
                "connection_retired",
                "The task's remote connection is no longer an active trusted connection.",
                phase="authorize",
            )
        admit = getattr(get_remote_workspace_service(), "admit_workspace", None)
        if not callable(admit):
            raise RemoteWorkspaceError(
                "remote_workspace_unavailable",
                "This process holds no remote session admission; a remote task cannot start.",
                phase="connect",
            )
        admitted = admit(
            row,
            remote_root=scope["remote_root"],
            project_id=scope["project_id"],
            workspace_id=scope["workspace_id"],
            task_id=task_id,
        )
    except Exception as exc:
        code = str(getattr(exc, "code", "") or type(exc).__name__)
        retryable = bool(getattr(exc, "retryable", False)) and attempt < MAX_BIND_ATTEMPTS
        task[BIND_STATE_KEY] = {
            "status": "retry" if retryable else "refused",
            "attempts": attempt,
            "code": code,
            "task_id": task_id,
            "server_generation": live_server_generation(),
        }
        return {
            "status": "retry" if retryable else "refused",
            "code": code,
            "message": (
                f"Remote workspace session could not be opened for this task "
                f"({code}: {exc}); the task never started."
            ),
        }
    facts = admitted if isinstance(admitted, Mapping) else {}
    task[BIND_STATE_KEY] = {
        "status": "bound",
        "attempts": attempt,
        "code": "",
        "task_id": task_id,
        "server_generation": live_server_generation(),
        # The identities the broker CONFIRMED, not the ones we asked for: a receipt
        # that echoed the request would hide an admission that landed elsewhere.
        "project_id": str(facts.get("project_id") or scope["project_id"]),
        "workspace_id": str(facts.get("workspace_id") or scope["workspace_id"]),
    }
    return {"status": "bound", "code": "", "message": ""}


def release_remote_task_session(task: Any, *, cancelled: bool = False) -> bool:
    """Drop ONE task's binding when it reaches a terminal state.

    Called from the SERVER process, before the RUNNING row is cleared: the worker
    holds only a pipe proxy and has no admission door, and once the row is gone the
    sealed placement is no longer in hand to release against. The project SESSION
    stays open either way — it belongs to the next task, not this one.

    Two different endings, and conflating them is how a cancelled task keeps
    working on the target:

    * ``cancelled=True`` — the owner stopped the task, so the target must kill
      the task's process groups NOW and the binding goes with them. BOTH doors are
      used, in this order, because a cancel can land in either of two states and
      only one of them has a session yet: `cancel_admission` reaches an admission
      that is still IN FLIGHT (the task is in PENDING while assignment talks to the
      target, so the durable "queue miss" latch that also calls it never fires for
      this window), and `cancel` reaches a bound session's process groups. Using
      only the second was the hole: a cancel during admission raised
      `task_session_unbound` — the task is not in `_task_sessions` yet — and the
      target kept whatever the admission had already started.
    * otherwise — the task finished on its own, so only the lease and the Home
      import staging are released (``finish_remote_task``); the PROJECT session
      stays open for the next task.

    Never raises and never touches another task's binding: every broker door here
    is keyed by this task's id and refuses a ref that names another workspace.

    The local receipt is dropped too, so a task dict that gets requeued (a timeout
    retry is `dict(task)`) reads as unbound and is admitted again rather than
    dispatched against a binding that no longer exists.
    """

    scope = remote_binding_scope(task)
    task_id = str(task.get("id") or "") if isinstance(task, Mapping) else ""
    if scope is None or not task_id:
        return False
    if isinstance(task, dict):
        task.pop(BIND_STATE_KEY, None)
    from ouroboros.remote_workspace import (
        finish_remote_task,
        get_remote_workspace_service,
    )

    if not cancelled:
        try:
            return bool(finish_remote_task(task, task_id))
        except Exception:
            return False
    stopped = False
    try:
        abandon = getattr(get_remote_workspace_service(), "cancel_admission", None)
        stopped = bool(callable(abandon) and abandon(task_id))
    except Exception:
        stopped = False
    try:
        cancel = getattr(get_remote_workspace_service(), "cancel", None)
        if callable(cancel):
            stopped = bool(cancel(scope["workspace_ref"], task_id=task_id)) or stopped
    except Exception:
        pass
    return stopped
