"""Owner-control addressability of the in-process direct-chat turn.

The class (rc.7 QA regress, item 010): a self-modification typed into the
chat runs as a DIRECT-CHAT TURN — minted by ``worker_chat_lane.py`` with an
8-hex id and executed on the long-lived chat agent inside the supervisor,
never through the queue. ``agent.py`` writes its ordinary durable
``running`` row, so ``GET /api/tasks?status=running`` listed it while every
queue-keyed owner control (cancel, hurry, decisions) answered 404 "task not
found or not active" and the spend kept growing until ``/restart``.

The contract pinned here: ``supervisor.workers.direct_chat_turn`` is the ONE
reader every ingress, the ownership predicate and the graceful-stop episode
resolve a live direct turn through — so a turn the task list shows as
running is addressable, and custody stops it COOPERATIVELY (the finalize
control on the canonical owner mailbox; there is no worker process to kill).

The two ``orphan`` cases at the top keep the honest legacy envelope for a
durable running row that NOTHING owns (no queue row, no worker, no live
chat turn): that is still 404, because there is nothing to stop.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.tasks import api_task_cancel, api_tasks_list
from ouroboros.task_results import load_task_result, write_task_result

TURN_ID = "40cc86d9"


def _isolate_queue(monkeypatch, tmp_path):
    from supervisor import queue, workers

    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(workers, "_chat_agent", None, raising=False)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    return queue


def _client(tmp_path):
    app = Starlette(routes=[
        Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"]),
        Route("/api/tasks", api_tasks_list, methods=["GET"]),
    ])
    app.state.drive_root = tmp_path
    return TestClient(app)


def _write_snapshot(tmp_path, running_ids=()):
    from ouroboros.utils import utc_now_iso

    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "queue_snapshot.json").write_text(json.dumps({
        "ts": utc_now_iso(),
        "pending": [],
        "running": [{"id": tid, "task": {"id": tid}} for tid in running_ids],
    }), encoding="utf-8")


def _live_chat_agent(monkeypatch, task_id=TURN_ID, *, accepting=True):
    """The chat agent mid-turn, shaped exactly as agent.py leaves it (the
    fields steering.py and workers.chat_turn_liveness read)."""
    from supervisor import workers

    import threading

    agent = SimpleNamespace(
        _busy=True, _accepting_owner_messages=accepting,
        _current_task_id=task_id, _current_chat_id=0,
        _current_task_metadata={"title": "Modify yourself"},
        _current_task_text="change the bubble colour and commit",
        _task_started_ts=1000.0, _last_activity_ts=1000.0,
        _owner_message_admission_lock=threading.Lock(),
    )
    monkeypatch.setattr(workers, "_chat_agent", agent, raising=False)
    return agent


def _mailbox_kinds(tmp_path, task_id=TURN_ID):
    from ouroboros.owner_mailbox import _mailbox_path

    path = _mailbox_path(tmp_path, task_id)
    if not path.exists():
        return []
    return [json.loads(line).get("kind") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- the honest residual: a running row that nothing owns ----------------------

@pytest.mark.parametrize("snapshot", ["absent", "fresh_without_row"])
def test_orphaned_running_row_keeps_the_404_envelope(tmp_path, monkeypatch, snapshot):
    """No queue row, no worker slot, no live chat turn: listed by the durable
    mirror (until the orphan projection catches it) but there is nothing to
    stop, so the cancel ingress keeps its legacy 404."""
    _isolate_queue(monkeypatch, tmp_path)
    write_task_result(tmp_path, TURN_ID, "running", chat_id=0, description="orphan")
    if snapshot == "fresh_without_row":
        _write_snapshot(tmp_path, running_ids=())

    with _client(tmp_path) as client:
        listed = client.get("/api/tasks?status=running").json()
        cancel = client.post(f"/api/tasks/{TURN_ID}/cancel")

    assert [row["task_id"] for row in listed["tasks"]] == [TURN_ID]
    assert cancel.status_code == 404


def test_running_row_present_in_queue_is_cancellable(tmp_path, monkeypatch):
    queue = _isolate_queue(monkeypatch, tmp_path)
    write_task_result(tmp_path, TURN_ID, "running", chat_id=0, description="pooled")
    queue.RUNNING[TURN_ID] = {"task": {"id": TURN_ID, "chat_id": 0}, "started_at": 1.0}
    from supervisor.queue_transitions import task_has_live_ownership

    assert task_has_live_ownership(TURN_ID) is True


# --- the direct-chat turn ----------------------------------------------------------

def test_live_direct_chat_turn_is_owned_and_admitted_by_every_owner_control(tmp_path, monkeypatch):
    queue = _isolate_queue(monkeypatch, tmp_path)
    _live_chat_agent(monkeypatch)
    write_task_result(tmp_path, TURN_ID, "running", chat_id=0, description="modify yourself")
    from supervisor.queue_transitions import task_has_live_ownership
    from supervisor.workers import direct_chat_turn
    from ouroboros.gateway.task_hurry import _admit_hurry_locked
    from ouroboros.gateway.task_decision import _live_root_task

    record = direct_chat_turn(TURN_ID)
    assert record is not None and record["_is_direct_chat"] is True
    assert record["id"] == TURN_ID and record["objective"].startswith("change the bubble")
    assert direct_chat_turn("deadbeef") is None
    assert task_has_live_ownership(TURN_ID) is True
    task, refusal, attempt = _admit_hurry_locked(TURN_ID)
    assert refusal == "" and task["id"] == TURN_ID and attempt == 1
    assert _live_root_task(TURN_ID)[0]["id"] == TURN_ID
    # A budget pause is a queue state a direct turn never enters — unchanged.
    assert queue.resume_budget_paused_task(TURN_ID)["error"] == "task_not_pending"


def test_ephemeral_decision_turn_is_not_an_owner_addressable_task(tmp_path, monkeypatch):
    _isolate_queue(monkeypatch, tmp_path)
    _live_chat_agent(monkeypatch, accepting=False)
    from supervisor.queue_transitions import task_has_live_ownership
    from supervisor.workers import direct_chat_turn

    assert direct_chat_turn(TURN_ID) is None
    assert task_has_live_ownership(TURN_ID) is False


def test_stop_now_finalizes_the_direct_turn_cooperatively_and_answers_ok(tmp_path, monkeypatch):
    """Immediate stop: custody writes the finalize control to the canonical
    mailbox, the turn reaches its next round boundary and publishes its own
    terminal result, the intent settles against it and the owner sees 200."""
    import ouroboros.config as config

    _isolate_queue(monkeypatch, tmp_path)
    agent = _live_chat_agent(monkeypatch)
    write_task_result(tmp_path, TURN_ID, "running", chat_id=0, description="modify yourself")
    _write_snapshot(tmp_path, running_ids=())
    monkeypatch.setattr(config, "get_direct_turn_stop_wait_sec", lambda: 5.0)
    from supervisor import task_reaper

    real_request = task_reaper.request_finalization_grace

    def _control_then_yield(task_drive, task_id, reason, **kwargs):
        # The loop drains the control at its round boundary and finalizes:
        # simulate exactly that — the control is really written, then the
        # turn ends and the pipeline publishes its own terminal result.
        msg_id = real_request(task_drive, task_id, reason, **kwargs)
        write_task_result(tmp_path, TURN_ID, "completed", chat_id=0, description="modify yourself")
        agent._busy = False
        return msg_id

    monkeypatch.setattr(task_reaper, "request_finalization_grace", _control_then_yield)

    with _client(tmp_path) as client:
        cancel = client.post(f"/api/tasks/{TURN_ID}/cancel", json={"stop_policy": "immediate"})

    assert cancel.status_code == 200, cancel.text
    assert cancel.json()["ok"] is True
    assert "finalize_now" in _mailbox_kinds(tmp_path)
    # The pipeline's own terminal is the truth; custody did not overwrite it.
    assert load_task_result(tmp_path, TURN_ID)["status"] == "completed"
    from ouroboros.cancel_intents import active_intent

    assert not active_intent(tmp_path, TURN_ID)


def test_stop_now_on_a_turn_inside_a_long_step_answers_still_live_and_keeps_the_stop_armed(tmp_path, monkeypatch):
    """A turn that has not reached a round boundary within the wait bound:
    503 "still live", the durable row is NOT rewritten as cancelled (the
    miss finalizer must never run for a live turn), the control stays in
    the mailbox and the intent stays open for the sweep's next custody pass."""
    import ouroboros.config as config

    _isolate_queue(monkeypatch, tmp_path)
    _live_chat_agent(monkeypatch)
    write_task_result(tmp_path, TURN_ID, "running", chat_id=0, description="modify yourself")
    _write_snapshot(tmp_path, running_ids=())
    monkeypatch.setattr(config, "get_direct_turn_stop_wait_sec", lambda: 0.5)


    with _client(tmp_path) as client:
        cancel = client.post(f"/api/tasks/{TURN_ID}/cancel", json={"stop_policy": "immediate"})

    assert cancel.status_code == 503, cancel.text
    assert "still live" in cancel.json()["error"]
    assert load_task_result(tmp_path, TURN_ID)["status"] == "running"
    assert "finalize_now" in _mailbox_kinds(tmp_path)
    from ouroboros.cancel_intents import active_intent

    assert active_intent(tmp_path, TURN_ID)


def test_wrap_up_arms_the_owner_stop_episode_for_the_direct_turn(tmp_path, monkeypatch):
    """Graceful stop (the default policy): the ingress answers 202 pending and
    the episode arms from the chat agent's record — the deterministic
    owner-stop control lands in the canonical mailbox for the loop to drain."""
    queue = _isolate_queue(monkeypatch, tmp_path)
    _live_chat_agent(monkeypatch)
    write_task_result(tmp_path, TURN_ID, "running", chat_id=0, description="modify yourself")
    _write_snapshot(tmp_path, running_ids=())
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 60, raising=False)
    from supervisor.owner_stop import begin_graceful_stop

    monkeypatch.setattr("supervisor.owner_stop.begin_graceful_stop", lambda task_id: None)
    with _client(tmp_path) as client:
        cancel = client.post(
            f"/api/tasks/{TURN_ID}/cancel", json={"stop_policy": "finalize_then_cancel"},
        )
    assert cancel.status_code == 202, cancel.text
    assert cancel.json()["cancel_state"] == "pending"

    # Run the ingress kick-off synchronously (the endpoint threads it).
    begin_graceful_stop(TURN_ID)
    from ouroboros.cancel_intents import active_intent
    from supervisor.owner_stop import owner_stop_control_id
    from ouroboros.owner_mailbox import _mailbox_path

    intent = active_intent(tmp_path, TURN_ID)
    assert intent
    rows = [json.loads(line) for line in _mailbox_path(tmp_path, TURN_ID).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["kind"] for row in rows] == ["finalize_now"]
    assert rows[0]["msg_id"] == owner_stop_control_id(intent)
    # The next sweep tick re-arms idempotently: the latch lives on the chat
    # agent's record (a pooled task keeps it on its RUNNING row), so no second
    # control and no second owner toast.
    begin_graceful_stop(TURN_ID)
    rows_again = [line for line in _mailbox_path(tmp_path, TURN_ID).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows_again) == 1
    from supervisor.workers import direct_chat_turn

    assert direct_chat_turn(TURN_ID)["finalization_control_msg_id"] == owner_stop_control_id(intent)


def test_direct_turn_progress_frames_carry_the_host_attested_cancelable_marker(tmp_path, monkeypatch):
    """The live card's Stop/Hurry control is gated on the supervisor's
    host-attested ``cancelable`` marker (web/modules/task_control_menu.js).
    A pooled root gets it from its RUNNING row; the direct turn has no row,
    so the delivery seam stamps it through the same ownership reader the
    cancel ingress uses — the card and the endpoint agree."""
    import types

    from supervisor.events import _handle_send_message

    _isolate_queue(monkeypatch, tmp_path)
    _live_chat_agent(monkeypatch)
    sent = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={},
        send_with_budget=lambda *a, **k: sent.append(k),
        append_jsonl=lambda *a, **k: None,
    )
    _handle_send_message({
        "chat_id": 0, "task_id": TURN_ID, "text": "working",
        "is_progress": True, "format": "markdown",
    }, ctx)
    _handle_send_message({
        "chat_id": 0, "task_id": "deadbeef", "text": "working",
        "is_progress": True, "format": "markdown",
    }, ctx)
    assert sent[0]["progress_meta"].get("cancelable") is True
    assert (sent[1].get("progress_meta") or {}).get("cancelable") is not True


def test_stop_now_on_a_turn_that_dies_without_a_terminal_publishes_cancelled(tmp_path, monkeypatch):
    """Adversarial finding: the lane's error path writes no task_result. If
    custody settled ``already_settled`` against a row still saying ``running``,
    the intent would be consumed and the row unreachable by any control
    forever — the rc.7 symptom re-created by a "successful" stop. Such a turn
    takes the same miss finalizer a pooled task does: durable ``cancelled``."""
    import ouroboros.config as config

    _isolate_queue(monkeypatch, tmp_path)
    agent = _live_chat_agent(monkeypatch)
    write_task_result(tmp_path, TURN_ID, "running", chat_id=0, description="modify yourself")
    _write_snapshot(tmp_path, running_ids=())
    monkeypatch.setattr(config, "get_direct_turn_stop_wait_sec", lambda: 5.0)
    from supervisor import task_reaper

    real_request = task_reaper.request_finalization_grace

    def _control_then_crash(task_drive, task_id, reason, **kwargs):
        msg_id = real_request(task_drive, task_id, reason, **kwargs)
        agent._busy = False  # the turn raised past the pipeline: no terminal written
        return msg_id

    monkeypatch.setattr(task_reaper, "request_finalization_grace", _control_then_crash)
    with _client(tmp_path) as client:
        cancel = client.post(f"/api/tasks/{TURN_ID}/cancel", json={"stop_policy": "immediate"})
    assert cancel.status_code == 200, cancel.text
    assert load_task_result(tmp_path, TURN_ID)["status"] == "cancelled"
    from ouroboros.cancel_intents import active_intent

    assert not active_intent(tmp_path, TURN_ID)


def test_stop_now_on_a_turn_that_already_ended_arms_nothing_and_answers_already_settled(tmp_path, monkeypatch):
    """Adversarial finding: a turn that finished between custody's ownership
    read and the control write must get neither a false owner toast over an
    answer that already landed nor an orphaned control; the pooled lane's
    ``already_settled`` envelope (404) applies."""
    import ouroboros.config as config

    _isolate_queue(monkeypatch, tmp_path)
    agent = _live_chat_agent(monkeypatch)
    write_task_result(tmp_path, TURN_ID, "running", chat_id=0, description="modify yourself")
    _write_snapshot(tmp_path, running_ids=())
    monkeypatch.setattr(config, "get_direct_turn_stop_wait_sec", lambda: 5.0)
    from supervisor import worker_chat_lane, workers

    real_reader = workers.direct_chat_turn
    reads = {"n": 0}

    def _turn_ends_after_custody_captured(task_id=""):
        reads["n"] += 1
        if reads["n"] == 3:  # ownership predicate, custody capture, then the lane's re-check
            write_task_result(tmp_path, TURN_ID, "completed", chat_id=0, description="modify yourself")
            agent._busy = False
        return real_reader(task_id)

    monkeypatch.setattr(workers, "direct_chat_turn", _turn_ends_after_custody_captured)
    toasts = []
    monkeypatch.setattr(workers, "get_event_q", lambda: type("Q", (), {"put": lambda self, evt: toasts.append(evt)})())
    with _client(tmp_path) as client:
        cancel = client.post(f"/api/tasks/{TURN_ID}/cancel", json={"stop_policy": "immediate"})
    assert cancel.status_code == 404, cancel.text
    assert _mailbox_kinds(tmp_path) == []
    assert toasts == []
    assert load_task_result(tmp_path, TURN_ID)["status"] == "completed"
    assert worker_chat_lane.DIRECT_TURN_STOP_GONE == "gone"


def test_custody_retry_on_an_armed_turn_does_not_wait_again(tmp_path, monkeypatch):
    """Adversarial finding: the sweep runs custody every tick; a pass that finds
    the control already stamped answers at once instead of sleeping the bound
    on every tick."""
    import time

    import ouroboros.config as config

    _isolate_queue(monkeypatch, tmp_path)
    _live_chat_agent(monkeypatch)
    from supervisor import worker_chat_lane, workers

    monkeypatch.setattr(config, "get_direct_turn_stop_wait_sec", lambda: 5.0)
    workers.stamp_direct_chat_turn(TURN_ID, stop_control_msg_id="ctl-1")
    started = time.monotonic()
    outcome = worker_chat_lane.stop_direct_chat_turn(TURN_ID, workers.direct_chat_turn(TURN_ID))
    assert outcome == worker_chat_lane.DIRECT_TURN_STOP_LIVE
    assert time.monotonic() - started < 1.0
    assert _mailbox_kinds(tmp_path) == []


def test_cascade_custody_stops_the_direct_turn_quietly(tmp_path, monkeypatch):
    """``deliver=False`` (a cascade sweep speaks for the tree once): the control
    is written, the owner toast is not."""
    import ouroboros.config as config

    _isolate_queue(monkeypatch, tmp_path)
    _live_chat_agent(monkeypatch)
    from supervisor import worker_chat_lane, workers

    monkeypatch.setattr(config, "get_direct_turn_stop_wait_sec", lambda: 0.2)
    toasts = []
    monkeypatch.setattr(workers, "get_event_q", lambda: type("Q", (), {"put": lambda self, evt: toasts.append(evt)})())
    outcome = worker_chat_lane.stop_direct_chat_turn(TURN_ID, workers.direct_chat_turn(TURN_ID), deliver=False)
    assert outcome == worker_chat_lane.DIRECT_TURN_STOP_LIVE
    assert _mailbox_kinds(tmp_path) == ["finalize_now"]
    assert toasts == []


def test_settled_row_does_not_hide_a_still_busy_direct_turn_from_custody(tmp_path, monkeypatch):
    """Scope-review finding: the pipeline persists the terminal BEFORE post-task
    cognition, so a direct turn can be settled on disk and still busy (and
    spending). Custody must not take the settled fast path past a live turn:
    it arms the stop and answers "still live" until the agent is really done,
    then settles already_settled against the stored terminal."""
    import ouroboros.config as config

    _isolate_queue(monkeypatch, tmp_path)
    agent = _live_chat_agent(monkeypatch)
    write_task_result(tmp_path, TURN_ID, "completed", chat_id=0, description="modify yourself")
    _write_snapshot(tmp_path, running_ids=())
    monkeypatch.setattr(config, "get_direct_turn_stop_wait_sec", lambda: 0.3)
    with _client(tmp_path) as client:
        first = client.post(f"/api/tasks/{TURN_ID}/cancel", json={"stop_policy": "immediate"})
        assert first.status_code == 503, first.text          # armed, still busy
        assert "finalize_now" in _mailbox_kinds(tmp_path)
        agent._busy = False                                    # the turn ends; no synthesis in flight
        second = client.post(f"/api/tasks/{TURN_ID}/cancel", json={"stop_policy": "immediate"})
    assert second.status_code == 404, second.text              # already settled, honestly
    assert load_task_result(tmp_path, TURN_ID)["status"] == "completed"
    from ouroboros.cancel_intents import active_intent

    assert not active_intent(tmp_path, TURN_ID)


def test_arming_the_stop_holds_the_turn_admission_lock_and_a_turn_that_ended_under_it_arms_nothing(tmp_path, monkeypatch):
    """Delta review: the liveness re-read and the control write must be ONE
    step under the agent's owner-message admission lock — the lock the turn's
    completion path flips ``_busy`` under — so a turn cannot end between them
    and still receive a control and a toast. Two pins: the write happens with
    the lock held; a completion that took the lock first leaves nothing."""
    import threading

    import ouroboros.config as config

    _isolate_queue(monkeypatch, tmp_path)
    agent = _live_chat_agent(monkeypatch)
    from supervisor import task_reaper, worker_chat_lane, workers

    monkeypatch.setattr(config, "get_direct_turn_stop_wait_sec", lambda: 0.2)
    real_request = task_reaper.request_finalization_grace
    held = []

    def _observe_lock(task_drive, task_id, reason, **kwargs):
        held.append(agent._owner_message_admission_lock.locked())
        return real_request(task_drive, task_id, reason, **kwargs)

    monkeypatch.setattr(task_reaper, "request_finalization_grace", _observe_lock)
    outcome = worker_chat_lane.stop_direct_chat_turn(TURN_ID, workers.direct_chat_turn(TURN_ID))
    assert outcome == worker_chat_lane.DIRECT_TURN_STOP_LIVE
    assert held == [True]
    assert _mailbox_kinds(tmp_path) == ["finalize_now"]

    # A second turn: its completion takes the lock and ends it before the arm.
    agent2 = _live_chat_agent(monkeypatch, task_id="beefcafe")
    toasts = []
    monkeypatch.setattr(workers, "get_event_q", lambda: type("Q", (), {"put": lambda self, evt: toasts.append(evt)})())
    with agent2._owner_message_admission_lock:
        agent2._busy = False
    record_before = {"id": "beefcafe", "chat_id": 0}
    outcome2 = worker_chat_lane.stop_direct_chat_turn("beefcafe", record_before)
    assert outcome2 == worker_chat_lane.DIRECT_TURN_STOP_GONE
    assert _mailbox_kinds(tmp_path, "beefcafe") == []
    assert toasts == []
    assert threading.current_thread() is threading.main_thread()


def test_wrap_up_on_a_turn_that_ended_under_its_lock_arms_nothing(tmp_path, monkeypatch):
    """Audit finding: the graceful episode must arm through the same atomic
    seam as the immediate stop. A completion that took the turn's admission
    lock first leaves no control and no toast; the episode reports "not
    armed" so custody settles the intent against the stored terminal."""
    queue = _isolate_queue(monkeypatch, tmp_path)
    agent = _live_chat_agent(monkeypatch)
    write_task_result(tmp_path, TURN_ID, "running", chat_id=0, description="modify yourself")
    _write_snapshot(tmp_path, running_ids=())
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 60, raising=False)
    from supervisor import workers
    from supervisor.owner_stop import begin_graceful_stop

    monkeypatch.setattr("supervisor.owner_stop.begin_graceful_stop", lambda task_id: None)
    with _client(tmp_path) as client:
        cancel = client.post(f"/api/tasks/{TURN_ID}/cancel", json={"stop_policy": "finalize_then_cancel"})
    assert cancel.status_code == 202, cancel.text
    toasts = []
    monkeypatch.setattr(workers, "get_event_q", lambda: type("Q", (), {"put": lambda self, evt: toasts.append(evt)})())
    real_reader = workers.direct_chat_turn
    reads = {"n": 0}

    def _ends_before_the_locked_arm(task_id=""):
        # The episode's first read still sees the turn; the completion then
        # takes the admission lock and ends it before the arm re-reads.
        reads["n"] += 1
        record = real_reader(task_id)
        if reads["n"] == 1:
            with agent._owner_message_admission_lock:
                write_task_result(tmp_path, TURN_ID, "completed", chat_id=0, description="modify yourself")
                agent._busy = False
        return record

    monkeypatch.setattr(workers, "direct_chat_turn", _ends_before_the_locked_arm)
    begin_graceful_stop(TURN_ID)
    assert _mailbox_kinds(tmp_path) == []
    assert toasts == []
    assert load_task_result(tmp_path, TURN_ID)["status"] == "completed"


# --- the turn ended, its PAID post-task synthesis is still billing (G18) -------

def _inflight_synthesis(monkeypatch, tmp_path, task_id=TURN_ID):
    """The real non-blocking post-task lane on its daemon thread, stage 1 held
    open behind an event: exactly the state ``emit_task_results`` leaves behind
    after the turn's caller returned and agent.py ended the turn's liveness."""
    import threading

    import ouroboros.agent_task_pipeline as pipeline
    import ouroboros.llm as llm_mod
    import ouroboros.post_task_evolution as pte

    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path / "repo", drive_path=lambda rel: tmp_path / rel)
    for rel in ("logs", "memory", "repo"):
        (tmp_path / rel).mkdir(exist_ok=True)
    task = {"id": task_id, "type": "task", "chat_id": 0, "_is_direct_chat": True, "text": "modify yourself"}
    arrived, release = threading.Event(), threading.Event()
    calls = []

    def _stage(name):
        def _f(*a, **k):
            calls.append(name)
            if name == "chat_consolidation":
                arrived.set()
                assert release.wait(30), "stage gate never released"
            return {"reflection": "x", "backlog_candidates": [], "memory_actions": []} if name == "reflection" else None
        return _f

    monkeypatch.setattr(llm_mod, "LLMClient", lambda *a, **k: object())
    for name, attr in (("chat_consolidation", "_run_chat_consolidation"),
                       ("scratchpad_consolidation", "_run_scratchpad_consolidation"),
                       ("summary", "_run_task_summary"), ("reflection", "_run_reflection"),
                       ("promotion", "_update_improvement_backlog")):
        monkeypatch.setattr(pipeline, attr, _stage(name))
    monkeypatch.setattr(pipeline, "_apply_reflection_memory_actions", lambda *a, **k: None)
    monkeypatch.setattr(pte, "maybe_promote", lambda *a, **k: None)
    spawned = []
    _real_thread = threading.Thread

    class CapturingThread(_real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            spawned.append(self)

    monkeypatch.setattr(pipeline.threading, "Thread", CapturingThread)
    pipeline._run_post_task_processing_async(
        env, task, {"rounds": 3, "cost": 0.01}, {"tool_calls": [], "reasoning_notes": []}, {}, tmp_path / "logs")
    assert arrived.wait(30), "the synthesis never reached its first paid stage"
    return SimpleNamespace(thread=spawned[0], release=release, calls=calls)


def test_live_ownership_holds_through_the_inflight_synthesis_and_ends_with_its_checkpoint(tmp_path, monkeypatch):
    """``task_has_live_ownership`` answers True while the pipeline holds the
    turn's in-flight key (the turn itself is gone: no busy agent, no queue
    row) and False once the terminal checkpoint is stored and the key dropped."""
    queue = _isolate_queue(monkeypatch, tmp_path)
    write_task_result(tmp_path, TURN_ID, "completed", chat_id=0, description="modify yourself")
    assert queue.task_has_live_ownership(TURN_ID) is False
    synthesis = _inflight_synthesis(monkeypatch, tmp_path)
    try:
        assert queue.task_has_live_ownership(TURN_ID) is True
    finally:
        synthesis.release.set()
        synthesis.thread.join(30)
    assert not synthesis.thread.is_alive()
    assert queue.task_has_live_ownership(TURN_ID) is False
    checkpoint = load_task_result(tmp_path, TURN_ID)["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "completed", checkpoint


def test_stop_now_during_the_inflight_synthesis_is_accepted_and_stops_the_remaining_paid_stages(tmp_path, monkeypatch):
    """The audit-point-4 class, end to end over the UI's endpoint: the turn's
    answer has landed and only its paid synthesis is running. Stop-now is
    ACCEPTED (typed 503 "still live", the durable immediate intent minted and
    left open by custody — never 404), the pipeline's per-stage gate then
    skips every remaining paid stage and discloses them; the next stop after
    the checkpoint settled answers the honest 404 with the intent drained."""
    from ouroboros.cancel_intents import INTENT_REQUESTED, active_intent

    _isolate_queue(monkeypatch, tmp_path)
    agent = _live_chat_agent(monkeypatch)
    agent._busy = False                                    # the turn's caller returned
    agent._accepting_owner_messages = False
    write_task_result(tmp_path, TURN_ID, "completed", chat_id=0, description="modify yourself")
    _write_snapshot(tmp_path, running_ids=())
    synthesis = _inflight_synthesis(monkeypatch, tmp_path)
    try:
        with _client(tmp_path) as client:
            first = client.post(f"/api/tasks/{TURN_ID}/cancel", json={"stop_policy": "immediate"})
            assert first.status_code == 503, first.text
            assert "still live" in first.json()["error"], first.text
            intent = active_intent(tmp_path, TURN_ID) or {}
            assert intent.get("state") == INTENT_REQUESTED, intent
            assert "post-task synthesis" in str(intent.get("last_error") or ""), intent
            assert _mailbox_kinds(tmp_path) == []          # no control for a loop that is gone
            synthesis.release.set()
            synthesis.thread.join(30)
            assert not synthesis.thread.is_alive()
            second = client.post(f"/api/tasks/{TURN_ID}/cancel", json={"stop_policy": "immediate"})
    finally:
        synthesis.release.set()
    assert second.status_code == 404, second.text
    assert synthesis.calls == ["chat_consolidation"], synthesis.calls
    stored = load_task_result(tmp_path, TURN_ID)
    assert stored["status"] == "completed"
    checkpoint = stored["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded", checkpoint
    assert checkpoint["post_task_stop_reason"] == (
        "owner_stopped:skipped=scratchpad_consolidation,summary,reflection,promotion"), checkpoint
    assert not active_intent(tmp_path, TURN_ID)


def test_cascade_stop_now_during_the_inflight_synthesis_keeps_the_intent_open_and_stops_the_remaining_paid_stages(tmp_path, monkeypatch):
    """The UI's Stop sends ``cascade:true`` (task_control_menu.js), so the
    audit-point-4 state must hold over the CASCADE path too: the turn's answer
    landed (stored ``completed``, no queue row, no busy agent) and only its paid
    synthesis is running. The cascade's PHYSICAL postcondition must see the
    in-flight synthesis as live — the answer is the single lane's typed 503
    "still live", the cascade-scope immediate intent stays OPEN for the
    pipeline's per-stage gate, and the remaining paid stages are skipped.
    (rc.15 review MAJOR 1: the postcondition re-judged custody's refusal
    against the stored ``completed`` alone, settled the intent and answered
    ok:true while the next stage kept billing.) The next cascade stop after
    the checkpoint settled is the already-down replay: it drains the intent."""
    from ouroboros.cancel_intents import INTENT_REQUESTED, SCOPE_CASCADE, active_intent

    _isolate_queue(monkeypatch, tmp_path)
    agent = _live_chat_agent(monkeypatch)
    agent._busy = False                                    # the turn's caller returned
    agent._accepting_owner_messages = False
    write_task_result(tmp_path, TURN_ID, "completed", chat_id=0, description="modify yourself")
    _write_snapshot(tmp_path, running_ids=())
    synthesis = _inflight_synthesis(monkeypatch, tmp_path)
    body = {"cascade": True, "stop_policy": "immediate"}
    try:
        with _client(tmp_path) as client:
            first = client.post(f"/api/tasks/{TURN_ID}/cancel", json=body)
            assert first.status_code == 503, first.text
            assert "still live" in first.json()["error"], first.text
            intent = active_intent(tmp_path, TURN_ID) or {}
            assert intent.get("state") == INTENT_REQUESTED, intent
            assert intent.get("scope") == SCOPE_CASCADE, intent
            assert "post-task synthesis" in str(intent.get("last_error") or ""), intent
            assert _mailbox_kinds(tmp_path) == []          # no control for a loop that is gone
            assert synthesis.calls == ["chat_consolidation"], synthesis.calls
            synthesis.release.set()
            synthesis.thread.join(30)
            assert not synthesis.thread.is_alive()
            assert synthesis.calls == ["chat_consolidation"], synthesis.calls
            second = client.post(f"/api/tasks/{TURN_ID}/cancel", json=body)
    finally:
        synthesis.release.set()
    assert second.status_code == 200 and second.json().get("cascade") is True, second.text
    stored = load_task_result(tmp_path, TURN_ID)
    assert stored["status"] == "completed"
    checkpoint = stored["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded", checkpoint
    assert checkpoint["post_task_stop_reason"] == (
        "owner_stopped:skipped=scratchpad_consolidation,summary,reflection,promotion"), checkpoint
    assert not active_intent(tmp_path, TURN_ID)
