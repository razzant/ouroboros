"""The acceptance fence travels by where the actor runs, and only owner mail revises an answer.

A Presence turn runs on a Host thread inside the server process. It used to be built without the
in-process fence transition the direct chat agent was given, so its begin travelled the worker bus,
stamped a pooled owner, and the queue refused it ``stale_attempt`` (#1183): a Presence turn is no
RUNNING worker attempt. The final seal then treated that refusal as owner mail and voided every
``presence_finish``, so a reviewed or stopped turn looped until its round limit.

Real seams, both directions: ``make_agent``, the agent's fence request, the handler the supervisor
dispatch table maps ``acceptance_fence`` to (with its ack file), ``transition_acceptance_fence`` under
the real queue lock, the loop's final seal (it runs at an author stop) and whole turns through
``run_presence_turn``. Only model and reviewer transports are scripted
(``tests.test_presence_continuation_bootstrap``).
"""
from __future__ import annotations

import json
import os
import queue as stdqueue
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_acceptance_fence import _isolated_queue

REPO = Path(__file__).resolve().parents[1]
TASK = "presence-0123456789abcdef01234567"
MARKER = "[PRESENCE_FINISH_NOT_ACCEPTED]"


class SupervisorBus:
    """The supervisor end of the process event bus, drained on its own thread.

    An ``acceptance_fence`` event goes to the handler the real dispatch table maps it to, which
    applies ``transition_acceptance_fence`` and writes the ``<token>.<req>.json`` ack the agent
    polls. Other telemetry is counted and dropped.
    """

    def __init__(self, drive_root: Path):
        from supervisor.events import EVENT_HANDLERS
        from supervisor.events_worker_reports import _handle_acceptance_fence

        assert EVENT_HANDLERS["acceptance_fence"] is _handle_acceptance_fence
        self._handler = EVENT_HANDLERS["acceptance_fence"]
        self._ctx = SimpleNamespace(DRIVE_ROOT=Path(drive_root))
        self.queue: stdqueue.Queue = stdqueue.Queue()
        self.fence_events: list = []
        self.other_events = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, name="fence-supervisor-bus", daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                evt = self.queue.get(timeout=0.02)
            except stdqueue.Empty:
                continue
            if isinstance(evt, dict) and evt.get("type") == "acceptance_fence":
                self.fence_events.append(dict(evt))
                self._handler(evt, self._ctx)
            else:
                self.other_events += 1

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        assert not self._thread.is_alive()


@pytest.fixture
def supervisor(monkeypatch, tmp_path):
    """Real supervisor queue state owned by this process, as the server owns its live maps."""
    root = tmp_path / "data"
    root.mkdir()
    queue_mod, _pending = _isolated_queue(monkeypatch, root)
    monkeypatch.setattr(queue_mod, "INITIALIZED", True)
    monkeypatch.setattr(queue_mod, "OWNER_PID", os.getpid())
    monkeypatch.delenv("OUROBOROS_IN_WORKER", raising=False)
    real = queue_mod.transition_acceptance_fence
    answers: list = []

    def recording(**kwargs):
        result = real(**kwargs)
        answers.append((dict(kwargs), dict(result)))
        return result

    # Every consumer resolves supervisor.queue.transition_acceptance_fence at call time.
    monkeypatch.setattr(queue_mod, "transition_acceptance_fence", recording)
    bus = SupervisorBus(root)
    try:
        yield SimpleNamespace(root=root, queue=queue_mod, bus=bus, answers=answers)
    finally:
        bus.close()


def _presence_built_agent(root: Path, monkeypatch, event_queue):
    """The agent as run_presence_turn builds it: make_agent(..., event_queue=bus), nothing else."""
    from ouroboros import agent as agent_module

    monkeypatch.setattr(agent_module.OuroborosAgent, "_log_worker_boot_once", lambda *_: None)
    agent = agent_module.make_agent(repo_dir=str(REPO), drive_root=str(root), event_queue=event_queue)
    # What _handle_task_scoped binds for a Presence task: its id, and ctx.task_attempt =
    # task.get("_attempt"), which a Presence task never carries.
    agent._current_task_id = TASK
    agent.tools._ctx.task_attempt = None
    return agent


def _seal_args(root: Path, agent, *, kind: str = "pooled"):
    """The ToolContext fields the final seal reads, with the agent's fence seam bound."""
    ctx = SimpleNamespace(
        task_id=TASK, drive_root=root, task_metadata={"root_task_id": TASK, "budget_drive_root": str(root),
                                                      "_execution_owner": {"kind": kind}},
        _task_acceptance_fence_token=None, _task_acceptance_sealed_fence_token=None,
        _task_acceptance_fence_generation=None, _task_acceptance_queue_descendants=[],
        _task_acceptance_fence_generation_mismatch=False, _owner_directives=[],
        owner_message_admission_lock=agent._owner_message_admission_lock, owner_message_admission_agent=agent,
        begin_acceptance_fence=agent._begin_acceptance_fence,
        inspect_acceptance_fence=agent._inspect_acceptance_fence,
        end_acceptance_fence=agent._end_acceptance_fence,
    )
    trace = {"review_decision": {}, "review_runs": []}
    return SimpleNamespace(_ctx=ctx), SimpleNamespace(task_id=TASK), trace


def _ack_unavailable(root: Path):
    path = root / "logs" / "events.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(row["op"], row["outcome"], row["reason"]) for row in rows if row.get("type") == "supervisor_ack_unavailable"]


# --- transport: the process fact, not the builder ---------------------------------------------------


def test_an_agent_in_the_queue_process_fences_in_process_without_any_injection(supervisor, monkeypatch):
    agent = _presence_built_agent(supervisor.root, monkeypatch, supervisor.bus.queue)
    assert not hasattr(agent, "fence_transition")
    token = agent._begin_acceptance_fence(root_task_id=TASK, task_id=TASK)["token"]
    assert agent._end_acceptance_fence(token=token, outcome="terminal", expected_generation=0)["status"] == "sealed"
    assert supervisor.bus.fence_events == []  # no event, no ack file, no wait
    assert [(req["action"], ans["status"]) for req, ans in supervisor.answers] == [("begin", "active"), ("end", "sealed")]
    assert all(req.get("pooled_owner") is None for req, _ans in supervisor.answers)


def test_a_worker_process_still_proves_its_attempt_over_the_bus(supervisor, monkeypatch):
    """#1183 is untouched: a bus sender without a current RUNNING attempt is refused."""
    monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")
    agent = _presence_built_agent(supervisor.root, monkeypatch, supervisor.bus.queue)
    with pytest.raises(RuntimeError, match=r"^stale_attempt$"):
        agent._begin_acceptance_fence(root_task_id=TASK, task_id=TASK)
    [begin] = supervisor.bus.fence_events
    assert begin["pooled_owner"] == {"task_attempt": 1, "pid": os.getpid()}
    assert supervisor.queue.ACCEPTANCE_FENCES == {}


def test_the_current_running_attempt_opens_over_the_bus(supervisor, monkeypatch):
    """The other direction of #1183: the same bus begin, owned by a live RUNNING attempt, opens."""
    from supervisor import workers

    monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")
    agent = _presence_built_agent(supervisor.root, monkeypatch, supervisor.bus.queue)
    worker = SimpleNamespace(proc=SimpleNamespace(pid=os.getpid(), is_alive=lambda: True),
                             busy_task_id=TASK, reaping=False)
    monkeypatch.setitem(workers.WORKERS, -901, worker)
    supervisor.queue.RUNNING[TASK] = {"task": {"id": TASK, "_attempt": 1}, "attempt": 1, "worker_id": -901}
    assert agent._begin_acceptance_fence(root_task_id=TASK, task_id=TASK)["status"] == "active"
    assert supervisor.queue.ACCEPTANCE_FENCES[TASK]["pooled_owner"] == {"task_attempt": 1, "pid": os.getpid()}


def test_a_child_forked_with_the_queue_maps_travels_the_bus(supervisor, monkeypatch):
    """INITIALIZED survives a fork; the owner's pid does not, so a copy never fences itself."""
    assert supervisor.queue.owns_live_queue(supervisor.root) is True
    monkeypatch.setattr(supervisor.queue, "OWNER_PID", os.getpid() + 1)  # what a forked child sees
    assert supervisor.queue.owns_live_queue(supervisor.root) is False
    agent = _presence_built_agent(supervisor.root, monkeypatch, supervisor.bus.queue)
    with pytest.raises(RuntimeError, match=r"^stale_attempt$"):
        agent._begin_acceptance_fence(root_task_id=TASK, task_id=TASK)
    assert supervisor.bus.fence_events and "pooled_owner" in supervisor.bus.fence_events[0]


def test_another_data_root_in_the_server_process_travels_the_bus(supervisor, monkeypatch, tmp_path):
    from ouroboros import runtime_limits

    monkeypatch.setattr(runtime_limits, "get_acceptance_fence_ack_wait_sec", lambda: 0.4)
    other = tmp_path / "other-root"
    other.mkdir()
    agent = _presence_built_agent(other, monkeypatch, supervisor.bus.queue)
    with pytest.raises((RuntimeError, TimeoutError)):
        agent._begin_acceptance_fence(root_task_id=TASK, task_id=TASK)
    assert supervisor.bus.fence_events and supervisor.bus.fence_events[0]["action"] == "begin"


# --- whole Presence turns -----------------------------------------------------------------------------


def _run_presence_turn(root: Path, monkeypatch, *, event_queue=None, supervisor=None, number=77, in_worker=False,
                       stop=False):
    """One whole turn through run_presence_turn with the production agent constructor and the real loop
    under Required/blocking result review. A message finish delivers without opening the seal (Presence
    speech is never held, and the seal runs only at an author stop); ``stop`` ends the turn with an author
    stop, which seals once right before it is delivered."""
    from ouroboros.presence_runner import presence_turn_task_id, run_presence_turn
    from ouroboros.task_results import load_task_result
    from tests.test_presence_continuation_bootstrap import ANSWER, admission, event, finish, install_bootstrap_harness

    h = install_bootstrap_harness(monkeypatch, root)
    h.addressed = ""  # nobody writes to the author during the turn
    h.release.set()
    if in_worker:  # a proactive turn started from background work runs on that worker's thread
        monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")
    turn_event = event(number)
    task_id = presence_turn_task_id(admission().binding_id, turn_event.source_event_id)

    def script(_messages):
        if stop:
            return finish(f"stop-{h.calls}", action="stop", outcome="silent", rationale="The room needs nothing from me now.")
        return finish(f"answer-{h.calls}", message=ANSWER)

    h.script = script
    outcome: dict = {}

    def run():
        try:
            outcome["result"] = run_presence_turn(
                admission=admission(), event=turn_event, repo_dir=h.repo, drive_root=h.data,
                event_queue=event_queue, agent_factory=h.factory, gate=h.gate)
        except BaseException as exc:  # recorded for the diagnosis below
            outcome["error"] = exc

    thread = threading.Thread(target=run, name="fence-presence-turn", daemon=True)
    thread.start()
    thread.join(timeout=240)
    assert not thread.is_alive(), "the Presence turn did not end within 240 s"
    row = load_task_result(h.data, task_id) or {}
    result = outcome.get("result")
    diagnosis = {
        "task_id": task_id,
        "model_calls": h.calls,
        "paid_reviews": len(h.reviews),
        "not_accepted_notes": sum(str(m.get("content") or "").count(MARKER) for m in (h.inputs[-1] if h.inputs else [])),
        "delivered": result is not None and (result.outcome == "silent" if stop else (
            result.outcome == "message" and result.text == ANSWER)),
        "error": repr(outcome.get("error")) if "error" in outcome else None,
        "row_status": row.get("status"), "row_reason_code": row.get("reason_code"),
        "supervisor_ack_unavailable": _ack_unavailable(h.data),
    }
    if supervisor is not None:
        diagnosis.update(
            bus_fence_events=[(e["action"], "pooled_owner" in e) for e in supervisor.bus.fence_events],
            queue_answers=[(req["action"], ans["status"]) for req, ans in supervisor.answers],
        )
    print("PRESENCE_TURN_DIAGNOSIS " + json.dumps(diagnosis, default=str, sort_keys=True))
    return diagnosis


def test_a_presence_host_message_turn_delivers_once_and_never_opens_the_fence(supervisor, monkeypatch):
    """The Host's configuration: run_presence_turn(event_queue=<supervisor bus>). A message finish is no
    author stop, so no seal begins; the turn delivers in one call and leaves no fence row (the stop's own
    seal is the next test)."""
    diagnosis = _run_presence_turn(supervisor.root, monkeypatch, event_queue=supervisor.bus.queue,
                                   supervisor=supervisor)
    assert diagnosis["delivered"] and diagnosis["row_status"] == "completed", diagnosis
    assert diagnosis["row_reason_code"] == "final_message", diagnosis
    assert diagnosis["supervisor_ack_unavailable"] == [] and diagnosis["model_calls"] == 1, diagnosis
    assert diagnosis["bus_fence_events"] == [] and diagnosis["queue_answers"] == [], diagnosis
    assert supervisor.queue.ACCEPTANCE_FENCES == {}


def test_a_presence_author_stop_seals_in_process_once_and_leaves_no_row(supervisor, monkeypatch):
    from ouroboros.presence_runner import presence_turn_task_id, run_presence_turn
    from ouroboros.task_results import load_task_result
    from tests.test_presence_continuation_bootstrap import admission, event, finish, install_bootstrap_harness

    h = install_bootstrap_harness(monkeypatch, supervisor.root)
    h.release.set()
    turn_event = event(91)
    task_id = presence_turn_task_id(admission().binding_id, turn_event.source_event_id)
    h.script = lambda _messages: finish(f"stop-{h.calls}", action="stop", outcome="silent",
                                        rationale="The room needs nothing from me now.")
    outcome: dict = {}

    def run():
        try:
            outcome["result"] = run_presence_turn(
                admission=admission(), event=turn_event, repo_dir=h.repo, drive_root=h.data,
                event_queue=supervisor.bus.queue, agent_factory=h.factory, gate=h.gate)
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run, name="stop-presence-turn", daemon=True)
    thread.start()
    thread.join(timeout=240)
    assert not thread.is_alive(), "the Presence turn did not end within 240 s"
    row = load_task_result(h.data, task_id) or {}
    diagnosis = {
        "calls": h.calls, "error": repr(outcome.get("error")) if "error" in outcome else None,
        "result": getattr(outcome.get("result"), "outcome", None), "status": row.get("status"),
        "acks": _ack_unavailable(h.data),
        "bus": [(e["action"], "pooled_owner" in e) for e in supervisor.bus.fence_events],
        "answers": [(req["action"], ans["status"]) for req, ans in supervisor.answers],
        "rows": dict(supervisor.queue.ACCEPTANCE_FENCES),
    }
    print("STOP_TURN_DIAGNOSIS " + json.dumps(diagnosis, default=str, sort_keys=True))
    assert diagnosis["error"] is None and diagnosis["result"] == "silent", diagnosis
    assert diagnosis["calls"] == 1 and diagnosis["acks"] == [] and diagnosis["bus"] == [], diagnosis
    assert ("end", "sealed") in diagnosis["answers"] and diagnosis["rows"] == {}, diagnosis


def test_a_raising_turn_releases_its_row(supervisor, monkeypatch):
    from ouroboros.presence_runner import presence_turn_task_id, run_presence_turn
    from tests.test_presence_continuation_bootstrap import admission, event, install_bootstrap_harness

    h = install_bootstrap_harness(monkeypatch, supervisor.root)
    turn_event = event(93)
    task_id = presence_turn_task_id(admission().binding_id, turn_event.source_event_id)

    class RaisingAgent:
        def handle_task(self, task):
            answer = supervisor.queue.transition_acceptance_fence(
                action="begin", token="raise-token", root_task_id=task["id"], task_id=task["id"])
            assert answer["status"] == "active" and task["id"] in supervisor.queue.ACCEPTANCE_FENCES
            raise RuntimeError("the executor died after opening its fence")

    with pytest.raises(BaseException):
        run_presence_turn(admission=admission(), event=turn_event, repo_dir=h.repo, drive_root=h.data,
                          event_queue=supervisor.bus.queue, agent_factory=lambda **_kw: RaisingAgent(),
                          gate=h.gate)
    assert task_id not in supervisor.queue.ACCEPTANCE_FENCES, supervisor.queue.ACCEPTANCE_FENCES


def test_a_failed_release_still_ends_the_live_turn(supervisor, monkeypatch):
    """Recovery reads a turn as live while its key is listed, so a release that fails must not keep it there."""
    from ouroboros import presence_runner
    from supervisor import queue_transitions

    def failing_release(*_args, **_kwargs):
        raise RuntimeError("release failed")

    monkeypatch.setattr(queue_transitions, "release_turn_acceptance_fence", failing_release)
    diagnosis = _run_presence_turn(supervisor.root, monkeypatch, event_queue=supervisor.bus.queue,
                                   supervisor=supervisor, number=80)
    assert "release failed" in str(diagnosis["error"]), diagnosis
    assert all(key[1] != diagnosis["task_id"] for key in presence_runner._LIVE_PRESENCE_TASKS)


def test_the_turn_end_release_leaves_other_roots_and_other_processes_alone(supervisor, monkeypatch):
    from supervisor.queue_transitions import release_turn_acceptance_fence

    supervisor.queue.ACCEPTANCE_FENCES["direct-root"] = {"token": "t", "root_task_id": "direct-root",
                                                        "task_id": "direct-root", "status": "sealed"}
    supervisor.queue.ACCEPTANCE_FENCES[TASK] = {"token": "p", "root_task_id": TASK, "task_id": TASK, "status": "sealed"}
    monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")
    assert release_turn_acceptance_fence(supervisor.root, TASK) is False  # not this process's queue
    monkeypatch.delenv("OUROBOROS_IN_WORKER")
    assert release_turn_acceptance_fence(supervisor.root, TASK) is True
    assert list(supervisor.queue.ACCEPTANCE_FENCES) == ["direct-root"]


# --- the final seal: only owner facts revise -------------------------------------------------------


def _refusing_agent(cause: str):
    return SimpleNamespace(
        _owner_message_admission_lock=threading.Lock(),
        _begin_acceptance_fence=lambda **_kw: (_ for _ in ()).throw(RuntimeError(cause)),
        _inspect_acceptance_fence=None, _end_acceptance_fence=None)


@pytest.mark.parametrize("cause", ["active", "invalid acceptance fence event"])
def test_a_refusal_that_is_not_owner_mail_delivers_disclosed(tmp_path, monkeypatch, cause):
    from ouroboros import loop
    from ouroboros.loop_delivery import _seal_admission_before_delivery

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    tools, limit_ctx, trace = _seal_args(tmp_path, _refusing_agent(cause))
    controls: list = []
    monkeypatch.setattr(loop, "_arm_delivery_control", lambda *_a, **kw: controls.append(kw["control"]))
    assert _seal_admission_before_delivery(tools, limit_ctx, trace) is True
    assert controls == []
    assert trace["review_decision"]["admission_released"] is False
    assert _ack_unavailable(tmp_path) == [("begin", "refused", cause)]  # the typed cause is the durable record


def test_a_superseded_pooled_attempt_never_publishes(tmp_path, monkeypatch):
    """``stale_attempt`` is positive evidence that this attempt lost its queue ownership (reaped or
    replaced): its answer is held, never delivered beside the current attempt's."""
    from ouroboros import loop
    from ouroboros.loop_delivery import _seal_admission_before_delivery

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    tools, limit_ctx, trace = _seal_args(tmp_path, _refusing_agent("stale_attempt"))
    monkeypatch.setattr(loop, "_arm_delivery_control", lambda *_a, **_kw: None)
    assert _seal_admission_before_delivery(tools, limit_ctx, trace) is False
    assert _ack_unavailable(tmp_path) == [("begin", "refused", "stale_attempt")]


def test_a_worker_hosted_presence_turn_delivers_with_admission_unconfirmed(supervisor, monkeypatch):
    """A proactive Presence turn started from background work runs inside that worker: its author stop's
    seal proves no RUNNING attempt, so the queue refuses ``stale_attempt``. It was never a pooled attempt,
    so nothing superseded it: the whole turn delivers once with admission unconfirmed instead of looping."""
    diagnosis = _run_presence_turn(supervisor.root, monkeypatch, event_queue=supervisor.bus.queue,
                                   supervisor=supervisor, number=78, in_worker=True, stop=True)
    assert diagnosis["error"] is None and diagnosis["delivered"], diagnosis
    assert diagnosis["model_calls"] == 1, diagnosis  # the loop was 11+ calls
    assert ("begin", "refused", "stale_attempt") in diagnosis["supervisor_ack_unavailable"], diagnosis
    assert diagnosis["bus_fence_events"] and all(pooled for _a, pooled in diagnosis["bus_fence_events"]), diagnosis
    assert supervisor.queue.ACCEPTANCE_FENCES == {}


def test_owner_mail_still_revises_after_a_refusal(tmp_path, monkeypatch):
    from ouroboros import loop
    from ouroboros.loop_delivery import _seal_admission_before_delivery
    from ouroboros.owner_mailbox import write_owner_message

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    tools, limit_ctx, trace = _seal_args(tmp_path, _refusing_agent("active"))
    assert write_owner_message(tmp_path, "Use the blue variant instead", TASK, msg_id="owner-blue")
    controls: list = []
    monkeypatch.setattr(loop, "_arm_delivery_control", lambda *_a, **kw: controls.append(kw["control"]))
    assert _seal_admission_before_delivery(tools, limit_ctx, trace) is False
    assert controls == ["owner_revision_required"]
