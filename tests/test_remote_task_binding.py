"""The task↔session binding decisions, without a target.

`tests/test_remote_task_session_wiring.py` proves the wire reaches a real host;
this file pins the small decisions that file cannot see cheaply — which project
scope a task is admitted under, and exactly when a queued task reads as "already
bound". Both have a failure mode that looks like success: a receipt trusted one
identity too far dispatches an UNBOUND task, which is the defect the wiring test
exists to catch.
"""

from __future__ import annotations

import pytest

from ouroboros.remote_task_binding import (
    BIND_STATE_KEY,
    bind_remote_task_session,
    live_server_generation,
    release_remote_task_session,
    remote_binding_pending,
    remote_binding_scope,
)
from ouroboros.workspace_admission import PLACEMENT_FENCE_KEY
from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY, SshWorkspaceRef

_REF = SshWorkspaceRef(
    connection_id="conn-1", remote_root="/srv/app", workspace_id="ws-1"
)


def _task(**extra):
    task = {
        "id": "task-1",
        "metadata": {SEALED_WORKSPACE_REF_KEY: _REF.to_payload()},
        "workspace_root": "/srv/app",
    }
    task.update(extra)
    return task


@pytest.fixture(autouse=True)
def _no_broker(monkeypatch):
    """No broker registered: `live_server_generation()` is "" for every case here."""
    from ouroboros import remote_workspace

    monkeypatch.setattr(remote_workspace, "_REMOTE_WORKSPACE_SERVICE", None)
    assert live_server_generation() == ""


def test_a_local_task_has_no_remote_scope_and_never_blocks_dispatch():
    local = {"id": "task-1", "metadata": {}, "workspace_root": "/home/me/repo"}
    assert remote_binding_scope(local) is None
    assert remote_binding_pending(local) is False


def test_scope_prefers_the_tasks_own_project_then_the_fence_then_the_workspace():
    assert remote_binding_scope(_task(project_id="Alpha"))["project_id"] == "alpha"
    fenced = _task(**{PLACEMENT_FENCE_KEY: {"project_id": "fenced"}})
    assert remote_binding_scope(fenced)["project_id"] == "fenced"
    # No project at all: the workspace is its own scope, exactly as
    # `workspace_admission.remote_session_facts` decides it.
    assert remote_binding_scope(_task())["project_id"] == "ws-1"


def test_a_subagent_shares_the_parents_project_scope():
    # `resolve_project_id` deliberately answers "" for a subagent (memory scope),
    # but SESSION scope must be the parent's or the child opens a second session.
    child = _task(id="task-2", delegation_role="subagent", project_id="alpha",
                  parent_task_id="task-1")
    assert remote_binding_scope(child)["project_id"] == "alpha"


def test_an_unbound_or_retrying_remote_task_reads_as_pending():
    assert remote_binding_pending(_task()) is True
    retrying = _task()
    retrying[BIND_STATE_KEY] = {
        "status": "retry", "task_id": "task-1", "server_generation": "",
    }
    assert remote_binding_pending(retrying) is True


def test_a_receipt_is_only_valid_for_its_own_task_id_and_generation():
    bound = _task()
    bound[BIND_STATE_KEY] = {
        "status": "bound", "task_id": "task-1", "server_generation": "",
    }
    assert remote_binding_pending(bound) is False
    # A timeout retry is `dict(task)` under a NEW id: the inherited receipt must
    # NOT let it dispatch against a binding the broker holds for the old id.
    retried = dict(bound)
    retried["id"] = "task-1-retry"
    assert remote_binding_pending(retried) is True
    # And a receipt from another broker generation is worthless.
    stale = dict(bound)
    stale[BIND_STATE_KEY] = {
        "status": "bound", "task_id": "task-1", "server_generation": "other-generation",
    }
    assert remote_binding_pending(stale) is True


def test_binding_without_a_broker_is_a_typed_refusal_not_a_silent_wait(monkeypatch):
    from ouroboros import config

    monkeypatch.setattr(
        config, "REMOTE_CONNECTIONS_PATH", config.DATA_DIR / "no-such-store.json"
    )
    task = _task()
    outcome = bind_remote_task_session(task)
    # The connection store cannot vouch for the ref, so the task is refused with a
    # code the owner can act on — it is never dispatched to a worker.
    assert outcome["status"] == "refused"
    assert outcome["code"] == "connection_retired"
    assert "never started" in outcome["message"]
    assert task[BIND_STATE_KEY]["status"] == "refused"
    assert remote_binding_pending(task) is True


def test_a_non_remote_task_is_skipped_by_the_binder():
    assert bind_remote_task_session({"id": "task-1", "metadata": {}})["status"] == "skipped"
    assert bind_remote_task_session(_task(id=""))["status"] == "skipped"


class _RecordingService:
    """A broker stand-in that refuses `cancel` exactly as a real one does for a
    task whose admission has not finished yet."""

    server_generation = "gen-1"

    def __init__(self, *, bound: bool) -> None:
        self.bound = bound
        self.calls: list[str] = []

    def cancel_admission(self, task_id):
        self.calls.append(f"cancel_admission:{task_id}")
        return True

    def cancel(self, workspace_ref, *, task_id="", **_kwargs):
        self.calls.append(f"cancel:{task_id}")
        if not self.bound:
            from ouroboros.workspace_diagnostics import RemoteWorkspaceError

            raise RemoteWorkspaceError(
                "task_session_unbound", "not bound", phase="authorize"
            )
        return True

    def finish_task(self, workspace_ref, *, task_id=""):
        self.calls.append(f"finish_task:{task_id}")
        return True


@pytest.fixture()
def _service(monkeypatch):
    def install(*, bound: bool) -> _RecordingService:
        from ouroboros import remote_workspace

        service = _RecordingService(bound=bound)
        monkeypatch.setattr(remote_workspace, "_REMOTE_WORKSPACE_SERVICE", service)
        return service

    return install


def test_a_cancel_during_admission_still_reaches_the_target(_service):
    # The window the durable "queue miss" latch cannot see: assignment is talking
    # to the target while the task is still in PENDING, so `cancel` has no session
    # to cancel. Abandoning the in-flight ADMISSION is the only door that reaches
    # the target here, and a refusal from the other one must not suppress it.
    service = _service(bound=False)
    assert release_remote_task_session(_task(), cancelled=True) is True
    assert service.calls == ["cancel_admission:task-1", "cancel:task-1"]


def test_a_cancel_of_a_bound_task_kills_its_process_groups(_service):
    service = _service(bound=True)
    assert release_remote_task_session(_task(), cancelled=True) is True
    assert service.calls == ["cancel_admission:task-1", "cancel:task-1"]


def test_an_ordinary_terminal_releases_only_the_lease(_service):
    service = _service(bound=True)
    task = _task(drive_root="/tmp/does-not-matter")
    assert release_remote_task_session(task) is True
    assert service.calls == ["finish_task:task-1"]


def test_release_drops_the_receipt_so_a_requeue_is_admitted_again(_service):
    _service(bound=True)
    task = _task()
    task[BIND_STATE_KEY] = {
        "status": "bound", "task_id": "task-1", "server_generation": "gen-1",
    }
    assert remote_binding_pending(task) is False
    release_remote_task_session(task, cancelled=True)
    assert BIND_STATE_KEY not in task
    assert remote_binding_pending(task) is True


def test_a_retryable_failure_waits_a_few_ticks_then_ends_the_task(monkeypatch, tmp_path):
    """A queued task must not die of a one-second hiccup, nor wait forever."""
    from ouroboros import config, remote_workspace
    from ouroboros.connection_store import add_connection
    from ouroboros.remote_task_binding import MAX_BIND_ATTEMPTS
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError

    store = tmp_path / "remote_connections.json"
    monkeypatch.setattr(config, "REMOTE_CONNECTIONS_PATH", store)
    row = add_connection(name="Target", ssh_alias="target")

    class _Flaky:
        server_generation = "gen-1"

        def admit_workspace(self, *_a, **_kw):
            raise RemoteWorkspaceError(
                "remote_session_disconnected", "target asleep",
                phase="connect", retryable=True,
            )

    monkeypatch.setattr(remote_workspace, "_REMOTE_WORKSPACE_SERVICE", _Flaky())
    ref = SshWorkspaceRef(
        connection_id=str(row["id"]), remote_root="/srv/app", workspace_id="ws-1"
    )
    task = {"id": "task-1", "metadata": {SEALED_WORKSPACE_REF_KEY: ref.to_payload()}}
    seen = [bind_remote_task_session(task)["status"] for _ in range(MAX_BIND_ATTEMPTS)]
    assert seen == ["retry"] * (MAX_BIND_ATTEMPTS - 1) + ["refused"]
    # The refusal carries the TARGET's code, not a generic one, and the task is
    # still gated out of dispatch either way.
    assert task[BIND_STATE_KEY]["code"] == "remote_session_disconnected"
    assert remote_binding_pending(task) is True


def test_release_is_inert_for_a_local_task(_service):
    service = _service(bound=True)
    assert release_remote_task_session({"id": "task-1", "metadata": {}}) is False
    assert service.calls == []
