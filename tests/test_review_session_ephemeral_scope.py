"""A review reads a tree the host minted without registering it.

An isolated review checkout, a retained input view, a registry checkout or a folder
inside a task's own drive is one wave's tree: its session declares ``scope.ephemeral``
and makes no project request, and a sticky plan-review thread on it does the same on an
engine at ``CLAUDEXOR_EPHEMERAL_THREAD_MIN_VERSION`` (an older engine keeps the thread's
registration). Any other root (the system repository, an owner's project) keeps its
registration, and a sticky thread's registration outlives the run's settlement.
"""

from __future__ import annotations

import pytest

from tests._review_session_route_shared import _owned_gateway_uses_each_test_transport as __owned_gateway_uses_each_test_transport
from tests._review_session_route_shared import fake_route as __fake_route

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_owned_gateway_uses_each_test_transport = __owned_gateway_uses_each_test_transport
fake_route = __fake_route

from tests._review_session_route_shared import _run_session_directly  # noqa: E402


_ROOTS = {
    "review_checkout": ("state", "review_checkouts", "tok", "repo"),
    "review_inputs": ("artifacts", "t-plan", "source_handles", "review_inputs", "request-1"),
    # Folders inside a task's own drive: the drive custody pass retires them with it.
    "task_drive": ("task_drives", "t-plan", "work", "repo"),
    "headless_task": ("state", "headless_tasks", "t-child", "data", "repo"),
    # An owner's project that happens to live under the data root is not a minted tree.
    "owner_under_data": ("workspace",),
    "owner_project": ("projects", "p-1"),
}
_MINTED = ("review_checkout", "review_inputs", "task_drive", "headless_task")


@pytest.mark.parametrize("where", ["review_checkout", "owner_under_data", "owner_project", "elsewhere"])
def test_a_minted_review_tree_starts_ephemeral_without_a_project_request(fake_route, tmp_path, where):
    root = tmp_path.joinpath(*_ROOTS[where]) if where in _ROOTS else "/tmp/fake-repo"
    if where in _ROOTS:
        root.mkdir(parents=True)
    _run_session_directly(tmp_path, root=str(root))
    gateway = fake_route.instances[-1]
    [request] = gateway.start_requests
    if where in _MINTED:
        assert request["scope"] == {"kind": "project", "root": str(root), "ephemeral": True}
        assert gateway.project_lookups == [] and gateway.registrations == []
    else:  # an undisturbed resolution suffices for the claim
        assert request["scope"] == {"kind": "project", "root": str(root)}
        assert gateway.project_lookups == [str(root)] and gateway.registrations == []


def _thread_engine(fake_route, monkeypatch, version):
    """The fake engine with the thread routes a sticky seat drives (create, turn, read)."""
    class Engine(fake_route):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.engine_version, self.threads = version, []

        def create_thread(self, request, *, idempotency_key):
            self.threads.append(request)
            return {"id": "thread-1"}

        def start_thread_turn(self, thread_id, request, *, idempotency_key):
            self.start_requests.append(dict(request))
            return {"runId": "run-1", "turnId": "turn-1", "threadId": thread_id}

        def get_thread(self, thread_id):
            return {"thread": {"id": thread_id, "headRunId": "run-1"},
                    "turns": [{"id": "turn-1", "runId": "run-1"}], "sessions": []}

    monkeypatch.setattr("ouroboros.gateways.claudexor.ClaudexorGateway", Engine)
    return Engine


def _sticky_seat(tmp_path, root):
    from ouroboros import review_execution

    invocation = review_execution.SessionInvocation(
        task_id="t-plan", surface="plan_review", slot_id="plan_slot_1", timeout_sec=30, use_thread=True)
    return review_execution.run_delegated_review_session(
        prompt="review", root=str(root), custody_drive=tmp_path, invocation=invocation)


@pytest.mark.parametrize("threaded", [False, True], ids=["run", "thread"])
@pytest.mark.parametrize("refused", [False, True], ids=["claimed", "unsent"])
def test_review_holds_the_registration_until_start_or_refusal(fake_route, tmp_path, monkeypatch, threaded, refused):
    from ouroboros import delegate_custody, delegate_custody_usage, review_execution

    engine = _thread_engine(fake_route, monkeypatch, "6.0.0")
    prepare = review_execution.prepare_review_session_request
    observed = []

    def inspect(*args, **kwargs):
        with delegate_custody_usage.project_retirement_lock(tmp_path, "proj-1"):
            observed.append(delegate_custody_usage.project_preparing(tmp_path, "proj-1", {}))
            assert not delegate_custody.pending_invocations(tmp_path)
        if refused:
            raise review_execution.ReviewRouteUnavailable("prepare refused", code="input_too_large")
        return prepare(*args, **kwargs)

    monkeypatch.setattr(review_execution, "prepare_review_session_request", inspect)
    run = _sticky_seat if threaded else lambda drive, root: _run_session_directly(drive, root=str(root))
    if refused:
        with pytest.raises(review_execution.ReviewRouteUnavailable, match="prepare refused"):
            run(tmp_path, tmp_path)
    else:
        assert run(tmp_path, tmp_path)["custody_durable"]
    assert observed == [True] and len(engine.instances[-1].start_requests) == int(not refused)
    with delegate_custody_usage.project_retirement_lock(tmp_path, "proj-1"):
        assert not delegate_custody_usage.project_preparing(tmp_path, "proj-1", {})


def test_sticky_review_waits_before_creating_a_thread_on_an_unconfirmed_registration(
        fake_route, tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current as current
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    _thread_engine(fake_route, monkeypatch, "6.0.0")
    current.mark_removal(tmp_path, "proj-1", {}, "custody")
    with pytest.raises(ClaudexorUnavailable) as refused:
        _sticky_seat(tmp_path, tmp_path)
    assert refused.value.code == "registration_unconfirmed"
    gateway = fake_route.instances[-1]
    assert gateway.threads == gateway.start_requests == []


@pytest.mark.parametrize("below_floor", [False, True])
@pytest.mark.parametrize("where", [*_MINTED, "registered_checkout"])
def test_a_sticky_thread_on_a_minted_tree_registers_nothing_from_the_floor(
        fake_route, tmp_path, monkeypatch, where, below_floor):
    from ouroboros import delegate_custody as custody
    from ouroboros.config import CLAUDEXOR_EPHEMERAL_THREAD_MIN_VERSION
    from ouroboros.subagent_worktrees import _put_record

    root = (tmp_path / "copies" / "child") if where == "registered_checkout" else tmp_path.joinpath(*_ROOTS[where])
    root.mkdir(parents=True)
    if where == "registered_checkout":  # the host's own copy record names the root
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
        _put_record({"path": str(root)}, tmp_path)
    _thread_engine(fake_route, monkeypatch, "5.0.1" if below_floor else CLAUDEXOR_EPHEMERAL_THREAD_MIN_VERSION)
    result = _sticky_seat(tmp_path, root)
    gateway = fake_route.instances[-1]
    [thread] = gateway.threads
    assert result["thread_id"] == "thread-1" and result["run_id"] == "run-1"
    if below_floor:  # an engine that does not sweep the lanes of ephemeral threads
        assert gateway.project_lookups == [str(root)], "the thread keeps the registration it is bound to"
        assert thread["scope"] == {"kind": "project", "root": str(root)}
    else:
        assert gateway.project_lookups == [] and gateway.registrations == [], "a one-shot root never registers"
        assert thread["scope"] == {"kind": "project", "root": str(root), "ephemeral": True}
        assert custody.replay(tmp_path)["run-1"].project_id == ""


@pytest.mark.parametrize("where", ["owner_under_data", "owner_project", "elsewhere"])
def test_a_sticky_thread_on_an_owner_root_keeps_its_registration_through_settlement(
        fake_route, tmp_path, monkeypatch, where):
    from ouroboros import delegate_custody as custody
    from ouroboros.config import CLAUDEXOR_EPHEMERAL_THREAD_MIN_VERSION

    root = tmp_path.joinpath(*_ROOTS[where]) if where in _ROOTS else tmp_path.parent / f"owner-{tmp_path.name}"
    root.mkdir(parents=True, exist_ok=True)
    fake_route.project_unregistered = True  # this start creates the registration the thread binds
    _thread_engine(fake_route, monkeypatch, CLAUDEXOR_EPHEMERAL_THREAD_MIN_VERSION)
    _sticky_seat(tmp_path, root)
    gateway = fake_route.instances[-1]
    [thread] = gateway.threads
    assert gateway.registrations == [str(root)] and thread["scope"] == {"kind": "project", "root": str(root)}
    row = custody.replay(tmp_path)["run-1"]
    assert (row.project_id, row.project_owned, row.settled) == ("proj-new", False, True)
    assert not row.registration_settlement()["project_retired"]
    # The registration sweep after the owner ended: the thread's project is never this run's to retire.
    custody.retire_settled_registrations(tmp_path, gateway, live_task_ids=set())
    assert gateway.removals == [], "a sticky thread's project survives its run's settlement"
