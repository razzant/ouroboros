"""Native task waits (PR1696): who answers for a selected id, and what a lost source says.

A sleeping wait reads what its answer reads: the effective task result, so a retry successor answers
for the id selected, and a selected id with no record is proven only by its queue row. These cases
drive the real wait tools, ``model_sleep``'s readiness, its warm and cold pollers, the lane and cold
checks, the real loop and, for a wait ready at its call, the payload the real ``LLMClient`` builds.
Stored retries are real rows (no producer mints a new id now), one of them through the current
``tests/_silence_reaper_support.py`` fixture under real Project admission.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.test_main_retry_resume_identity import _project_retry, clock as clock, host as host, reaper as reaper  # noqa: F401
from tests.test_model_sleep import _mail
from tests.test_native_task_wait import (  # noqa: F401
    _call, _child, _ctx, _drive, _End, _header, _limit, _loop_registry, _note, _notices, _parked_cold,
    _resumed_first_request, _wire_requests, ledger as ledger)

pytestmark = pytest.mark.serial  # cold consumers bind queue module state


def _retried(root, original="kid-a", successor="kid-b", *, status="failed", successor_status="scheduled",
             parent="parent", **fields):
    """The stored state of a retried task as the effective reader follows it (``effective_task_result``): the
    successor's row, then its predecessor naming it. Real rows and no producer: none mints a new id now."""
    _child(root, successor, successor_status, parent=parent, supersedes_task_id=original, original_task_id=original,
           timeout_retry_from=original, **fields)
    _child(root, original, status, parent=parent, result="FIRST TRY", superseded_by=successor, retry_task_id=successor)


def _queue(root, *pending, running=()):
    """The queue snapshot the effective reader and the supervisor share."""
    (root / "state").mkdir(exist_ok=True)
    (root / "state" / "queue_snapshot.json").write_text(json.dumps({
        "pending": [{"id": task["id"], "task": task} for task in pending],
        "running": [{"id": task["id"], "task": task} for task in running]}))


def _woken(ctx, landing):
    """Through the real warm park: ``landing`` happens while the stack is parked. Returns the transcript."""
    from ouroboros import owner_wait

    def park(sleeper, checkpoint):
        landing()
        return owner_wait._sleep_wake(sleeper, checkpoint)

    ctx.owner_wait_callback = park
    messages = [{"role": "user", "content": "intent"}]
    owner_wait.wait_after_tools(ctx, messages, {}, {}, 1, [], set())
    assert ctx._model_sleep is None and ctx._pending_task_wait
    return messages


# --- effective identity: who answers for the id selected ---------------------------------------------


@pytest.mark.parametrize("case", ["ready", "ready_through_retry", "ready_known_hash", "ready_unknown", "armed_while_retry_runs"])
def test_a_ready_at_call_wait_reaches_the_serialized_next_request_through_its_tool_result(tmp_path, monkeypatch, case):
    """The whole path of a wait that needs no sleep, and of the one that looks ready only by its raw row."""
    from ouroboros import task_status
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256

    body = "READY-AT-CALL RESULT"
    arguments, settle = {"task_id": "kid-a", "sleep_mode": "warm"}, None
    _child(tmp_path, "parent", parent="")
    if case in {"ready", "ready_known_hash"}:
        _child(tmp_path, "kid-a", "completed", result=body)
    elif case == "ready_through_retry":
        _retried(tmp_path, successor_status="completed", result=body)
    elif case == "armed_while_retry_runs":
        _retried(tmp_path, successor_status="running")
        settle = lambda: _child(tmp_path, "kid-b", "completed", result=body)  # noqa: E731
    else:
        _child(tmp_path, "kid-a")
        real = task_status.load_effective_task_result

        def broken(root, task_id, **kwargs):
            if task_id == "kid-a":
                raise OSError("selected source torn")
            return real(root, task_id, **kwargs)

        monkeypatch.setattr(task_status, "load_effective_task_result", broken)
    if case == "ready_known_hash":
        arguments["known_result_sha256"] = _child_result_sha256(load_effective_task_result(tmp_path, "kid-a"))
    ctx, parks, sent = _wire_requests(tmp_path, monkeypatch, arguments, settle=settle)
    first, second = (json.dumps(request["messages"], ensure_ascii=False) for request in sent)
    (tool,) = [row for row in sent[1]["messages"] if row.get("tool_call_id") == "call-wait"]
    text = json.dumps(tool["content"], ensure_ascii=False)
    assert body not in first and ctx._pending_task_wait is None and getattr(ctx, "_model_sleep", None) is None
    if case == "armed_while_retry_runs":
        assert "sleep_armed" in text and body not in text and len(parks) == 1, "a failed predecessor is not ready"
        assert second.count("[BEGIN_TASK_WAIT_OUTPUT]") == 1 and second.count(body) == 1
        assert "task kid-a reaching completed for the task wait you chose" in second
        return
    assert parks == [] and "[BEGIN_TASK_WAIT_OUTPUT]" not in second, "nothing parked: the tool result answers"
    if case == "ready_unknown":
        assert "task_wait:unknown:kid-a" in text and "source_unreadable" in text
        assert "Task wait returned before terminal" in text and "not success" in text and "completed" not in text
    elif case == "ready_known_hash":
        assert second.count(body) == 0 and "result_unchanged" in text and "task:kid-a:completed" in text
    else:
        assert second.count(body) == 1 and body in text and "task:kid-a:completed" in text
        assert "FIRST TRY" not in second


@pytest.mark.parametrize("predecessor", ["failed", "interrupted"])
def test_a_retry_successor_answers_for_the_selected_id_in_exact_all_and_any(tmp_path, predecessor):
    """A raw ``failed`` row would wake early, a raw ``interrupted`` one never: the effective read decides both."""
    from ouroboros import model_sleep
    from ouroboros.task_store import task_versions
    from ouroboros.tools.control_task_results import _wait_for_task, _wait_for_tasks
    from supervisor.sleep_wake import _readiness

    _child(tmp_path, "parent", parent="")
    _child(tmp_path, "kid-c")
    _retried(tmp_path, status=predecessor)
    assert task_versions(tmp_path, ["kid-a"])["kid-a"][1] == predecessor
    exact, both, first = _ctx(tmp_path), _ctx(tmp_path), _ctx(tmp_path)
    assert json.loads(_wait_for_task(exact, "kid-a", sleep_mode="warm"))["reason"] == "sleep_armed", "pending retry"
    assert json.loads(_wait_for_tasks(both, ["kid-a", "kid-c"], sleep_mode="warm"))["reason"] == "sleep_armed"
    assert json.loads(_wait_for_tasks(first, ["kid-a", "kid-c"], mode="any_terminal", sleep_mode="warm"))[
        "reason"] == "sleep_armed"
    q = SimpleNamespace(DRIVE_ROOT=tmp_path)

    def reasons():
        return [model_sleep.wake_reason(ctx, ctx._model_sleep) for ctx in (exact, both, first)]

    assert reasons() == ["", "", ""]
    _child(tmp_path, "kid-b", "running")
    assert reasons() == ["", "", ""], "a running successor wakes nobody"
    assert _readiness(q, {"id": "parent"}, {"sleep": exact._model_sleep}) == "", "the cold poller reads the same"
    _child(tmp_path, "kid-b", "completed", result="SECOND TRY")
    settled = "task:kid-a:completed"
    assert reasons() == [settled, "", settled], "the selected identity, the successor's status; one member is not all"
    assert _readiness(q, {"id": "parent"}, {"sleep": exact._model_sleep}) == settled
    assert _readiness(q, {"id": "parent"}, {"sleep": both._model_sleep}) == ""
    _child(tmp_path, "kid-c", "failed", result="c broke")
    assert model_sleep.wake_reason(both, both._model_sleep) in {settled, "task:kid-c:failed"}


def test_the_real_loop_sleeps_through_a_running_retry_and_answers_with_what_settled_it(tmp_path, monkeypatch):
    from ouroboros import owner_wait

    _child(tmp_path, "parent", parent="")
    _retried(tmp_path, successor_status="running")
    registry = _loop_registry(tmp_path, monkeypatch)
    polls = []

    def park(sleeper, checkpoint):
        polls.append(owner_wait._sleep_wake(sleeper, checkpoint))
        _child(tmp_path, "kid-b", "completed", result="SECOND TRY")
        polls.append(owner_wait._sleep_wake(sleeper, checkpoint))
        return polls[-1]

    registry._ctx.owner_wait_callback = park

    def second(messages):
        (notice,) = _notices(messages)
        (child,) = _header(notice)["children"]
        assert child["task_id"] == "kid-a" and child["effective_task_id"] == "kid-b" and child["status"] == "completed"
        assert "SECOND TRY" in notice and "FIRST TRY" not in notice
        assert any("woken by task kid-a reaching completed for the task wait you chose" in str(row.get("content"))
                   for row in messages)
        raise _End()

    sent = _drive(tmp_path, monkeypatch, registry, [_call("call-wait", "wait_task", task_id="kid-a", sleep_mode="warm"), second])
    assert len(sent) == 2 and polls == ["", "task:kid-a:completed"], "no early wake and no fetch round"


def test_a_cold_wait_wakes_on_the_retry_that_answers_in_its_first_resumed_request(tmp_path, monkeypatch):
    from supervisor.sleep_wake import wake_ready_sleepers

    _retried(tmp_path, "foreign", "foreign-2", successor_status="running", parent="foreign-root")
    q, workers, _pause = _parked_cold(tmp_path, monkeypatch)
    assert wake_ready_sleepers(q) == [], "a failed predecessor with a running successor stays parked"
    _child(tmp_path, "foreign-2", "completed", parent="foreign-root", result="COLD SECOND TRY")
    outcome = wake_ready_sleepers(q)
    assert outcome[0]["ok"] is True and outcome[0]["ready"]["reason"] == "task:foreign:completed"

    def check(messages):
        (notice,) = _notices(messages)
        assert "COLD SECOND TRY" in notice and "FIRST TRY" not in notice
        assert _header(notice)["children"][0]["effective_task_id"] == "foreign-2"

    _resumed_first_request(tmp_path, monkeypatch, workers, check)


def test_a_queued_retry_in_the_same_project_keeps_the_lane_check_until_it_runs(tmp_path, monkeypatch, host, clock, reaper):
    """Real Project admission and a stored retry admitted as a restored row: only its queue row names the room."""
    from ouroboros import model_sleep
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.control_task_results import _wait_for_task
    from supervisor import queue as supervisor_queue
    from tests.test_retry_pause_identity import _assign

    retry = _project_retry(host, tmp_path, monkeypatch, clock, reaper)
    assert supervisor_queue.persist_queue_snapshot()
    assert not load_task_result(host.root, retry["id"]).get("project_id")
    _child(host.root, "sleeper", parent="")

    def sleeper(project):
        ctx = _ctx(host.root, "sleeper")
        ctx.project_id = project
        return ctx

    same = sleeper(retry["project_id"])
    refused = _wait_for_task(same, "held", sleep_mode="warm")
    assert "retains the project lane" in refused and not getattr(same, "_pending_task_wait", None)
    assert not getattr(same, "_model_sleep", None), "nothing sleeps on work only its own lane could admit"
    plain = sleeper(retry["project_id"])
    assert model_sleep.request_sleep(plain, model_sleep.selectors(plain, tasks=["held"]), "warm")[
        "reason"] == "sleep_armed", "an ordinary sleep keeps its own selected row's lane read"
    other = sleeper("another-room")
    assert json.loads(_wait_for_task(other, "held", sleep_mode="warm"))["reason"] == "sleep_armed"
    [running] = _assign(host, monkeypatch)
    assert running["id"] == retry["id"] and supervisor_queue.persist_queue_snapshot()
    started = sleeper(retry["project_id"])
    assert json.loads(_wait_for_task(started, "held", sleep_mode="warm"))["reason"] == "sleep_armed", "it holds its slot"
    assert model_sleep.wake_reason(started, started._model_sleep) == ""
    write_task_result(host.root, running["id"], "completed", result="RETRY ANSWER")
    assert model_sleep.wake_reason(started, started._model_sleep) == "", "its workspace capture is not final yet"
    write_task_result(host.root, running["id"], "completed", result="RETRY ANSWER", artifact_status="ready")
    assert model_sleep.wake_reason(started, started._model_sleep) == "task:held:completed"


def test_the_lane_and_cold_checks_read_who_answers_for_the_selected_id(tmp_path):
    from ouroboros.tools.control_task_results import _wait_for_task

    def refusal(task_id, mode, project=""):
        ctx = _ctx(tmp_path)
        if project:
            ctx.project_id = project
        text = _wait_for_task(ctx, task_id, sleep_mode=mode)
        return text, getattr(ctx, "_model_sleep", None)

    _child(tmp_path, "parent", parent="")
    _queue(tmp_path, {"id": "queued-root", "root_task_id": "queued-root", "project_id": "room"},
           {"id": "queued-kid", "parent_task_id": "parent", "root_task_id": "parent", "delegation_role": "subagent"},
           running=[{"id": "started-kid", "parent_task_id": "parent", "root_task_id": "parent"}])
    # A first result has not reached either queued row: the lane and cold checks still see them.
    text, armed = refusal("queued-root", "warm", "room")
    assert "retains the project lane" in text and armed is None
    assert json.loads(refusal("queued-root", "warm", "another-room")[0])["reason"] == "sleep_armed"
    assert json.loads(refusal("queued-kid", "warm", "room")[0])["reason"] == "sleep_armed", "its own tree holds no lane"
    text, armed = refusal("queued-kid", "cold")
    assert "queued_member queued-kid" in text and armed is None, "the root's fence would hold it"
    text, armed = refusal("started-kid", "cold")
    assert "running_member started-kid" in text and armed is None
    # A retry of this tree's member: the successor is what the root's fence would hold.
    _retried(tmp_path)
    text, armed = refusal("kid-a", "cold")
    assert "queued_member kid-b" in text and armed is None
    # A foreign retry queued in this Project, by its own row and with no queue evidence for a row without a room.
    _retried(tmp_path, "root-a", "root-b", parent="", project_id="room")
    assert "retains the project lane" in refusal("root-a", "warm", "room")[0]
    _child(tmp_path, "no-room", "scheduled", parent="")
    (tmp_path / "state" / "queue_snapshot.json").write_text("{torn")
    text, armed = refusal("no-room", "warm", "room")
    assert "the queue cannot be read" in text and armed is None, "an unreadable queue never authorizes the lane"


def test_a_store_state_during_the_lane_or_cold_check_never_passes_as_no_dependency(tmp_path, monkeypatch):
    from ouroboros import model_sleep, native_task_wait
    from ouroboros.store import StoreBusy
    from ouroboros.tools.control_task_results import _wait_for_task

    _child(tmp_path, "parent", parent="")
    _child(tmp_path, "kid-a")
    real, calls, busy = native_task_wait.observe, [], StoreBusy("store busy", reason="contention")

    def contended_after_the_first_read(ctx, task_ids):
        calls.append(task_ids)
        if len(calls) == 1:
            return real(ctx, task_ids)
        return {str(task_id): {} for task_id in task_ids}, {}, busy

    monkeypatch.setattr(native_task_wait, "observe", contended_after_the_first_read)
    ctx = _ctx(tmp_path)
    ctx.project_id = "room"
    with pytest.raises(StoreBusy):  # readiness read the store; the lane check that follows cannot
        _wait_for_task(ctx, "kid-a", sleep_mode="warm")
    assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_pending_task_wait", None)
    blockers = model_sleep.cold_blockers(_ctx(tmp_path), chosen={"tasks": ["kid-a"], "task_wait": {"task_ids": ["kid-a"]}})
    assert {"kind": "tree_custody_unreadable", "detail": "store busy"} in blockers


def test_an_ordinary_sleep_keeps_its_own_selected_status(tmp_path, monkeypatch):
    from ouroboros import model_sleep, native_task_wait

    monkeypatch.setattr(native_task_wait, "observe", lambda *_a, **_k: pytest.fail("an ordinary sleep read a native wait's source"))
    _child(tmp_path, "parent", parent="")
    _retried(tmp_path, successor_status="running")
    _retried(tmp_path, "kid-x", "kid-y", status="interrupted", successor_status="running")
    ctx = _ctx(tmp_path)
    assert model_sleep.wake_reason(ctx, model_sleep.selectors(ctx, tasks=["kid-a"])) == "task:kid-a:failed"
    assert model_sleep.wake_reason(ctx, model_sleep.selectors(ctx, tasks=["kid-x"])) == ""
    with pytest.raises(ValueError, match="task ghost is unknown"):
        model_sleep.selectors(ctx, tasks=["ghost"])


# --- a source lost after the arm wakes as unknown, never success and never a silent sleep --------------


def test_a_child_proven_queued_before_its_first_result_sleeps_and_losing_that_proof_wakes_unknown(tmp_path):
    from ouroboros import model_sleep, task_store
    from ouroboros.native_task_wait import deliver_owed
    from ouroboros.tools.control_task_results import _wait_for_task

    _child(tmp_path, "parent", parent="")
    _queue(tmp_path, {"id": "queued", "parent_task_id": "parent", "root_task_id": "parent"})
    ctx = _ctx(tmp_path)
    assert json.loads(_wait_for_task(ctx, "queued", sleep_mode="warm"))["reason"] == "sleep_armed"
    assert model_sleep.wake_reason(ctx, ctx._model_sleep) == "", "the queue proves it"
    _queue(tmp_path, running=[{"id": "queued", "parent_task_id": "parent", "root_task_id": "parent"}])
    assert model_sleep.wake_reason(ctx, ctx._model_sleep) == "", "started, still before its first result"
    assert model_sleep.wake_notice(ctx._model_sleep, "task_wait:unknown:queued", 1.0)["content"].count("unknown, not success") == 1
    messages = _woken(ctx, lambda: _queue(tmp_path))  # a scheduling refusal took the only proof: no terminal exists
    assert "state is unknown, not success" in str(messages[-1]["content"]), messages[-1]
    assert deliver_owed(_limit(ctx, messages)) is True
    (notice,) = _notices(messages)
    assert _header(notice)["children"] == [{"task_id": "queued", "status": "unknown", "unknown_reason": "source_missing"}]
    assert "Task wait returned before terminal" in notice and '"status": "unknown"' in notice and "completed" not in notice

    # A record removed after the arm is proven only by a queue row, then by nothing.
    _child(tmp_path, "undone", "requested")
    undone = _ctx(tmp_path)
    assert json.loads(_wait_for_task(undone, "undone", sleep_mode="warm"))["reason"] == "sleep_armed"
    _queue(tmp_path, {"id": "undone", "parent_task_id": "parent", "root_task_id": "parent"})
    assert task_store.delete_document(tmp_path, "undone")
    assert model_sleep.wake_reason(undone, undone._model_sleep) == ""
    _queue(tmp_path)
    assert model_sleep.wake_reason(undone, undone._model_sleep) == "task_wait:unknown:undone"
    # A first result ends the queue's duty: the row decides from then on.
    _queue(tmp_path, {"id": "late", "parent_task_id": "parent", "root_task_id": "parent"})
    late = _ctx(tmp_path)
    assert json.loads(_wait_for_task(late, "late", sleep_mode="warm"))["reason"] == "sleep_armed"
    _child(tmp_path, "late", "running")
    _queue(tmp_path)
    assert model_sleep.wake_reason(late, late._model_sleep) == ""
    _child(tmp_path, "late", "completed", result="late answer")
    assert model_sleep.wake_reason(late, late._model_sleep) == "task:late:completed"
    # A queue that cannot say proves nothing, so nothing arms on it.
    _queue(tmp_path, {"id": "torn", "parent_task_id": "parent", "root_task_id": "parent"})
    (tmp_path / "state" / "queue_snapshot.json").write_text("{torn")
    torn = _ctx(tmp_path)
    lead, _end = json.JSONDecoder().raw_decode(_wait_for_task(torn, "torn", sleep_mode="warm"))
    assert lead["reason"] == "ready" and lead["woke_by"] == "task_wait:unknown:torn"
    assert not getattr(torn, "_pending_task_wait", None) and not getattr(torn, "_model_sleep", None)


def test_a_wake_before_the_first_result_says_the_queue_holds_the_child(tmp_path):
    from ouroboros.native_task_wait import deliver_owed
    from ouroboros.tools.control_task_results import _wait_for_task, _wait_for_tasks

    _child(tmp_path, "parent", parent="")
    _child(tmp_path, "kid-a")
    _queue(tmp_path, {"id": "queued", "parent_task_id": "parent", "root_task_id": "parent"})
    ctx = _ctx(tmp_path)
    assert json.loads(_wait_for_tasks(ctx, ["kid-a", "queued"], sleep_mode="warm"))["reason"] == "sleep_armed"
    messages = _woken(ctx, lambda: _mail(tmp_path, "parent", "status?", sender="peer"))
    assert deliver_owed(_limit(ctx, messages)) is True
    (notice,) = _notices(messages)
    facts = {row["task_id"]: row for row in _header(notice)["children"]}
    assert facts["queued"] == {"task_id": "queued", "status": "scheduled", "pre_first_result": True}
    assert facts["kid-a"]["status"] == "running"
    payload = json.loads(notice.split("[BEGIN_TASK_WAIT_OUTPUT]\n", 1)[1].rsplit("\n[END_TASK_WAIT_OUTPUT]", 1)[0])
    assert payload["all_terminal"] is False and payload["tasks"]["queued"]["status"] == "scheduled"
    assert payload["tasks"]["queued"]["pre_first_result"] is True and payload["live_child_status"]["queued"] == "scheduled"
    # The same child through the single wait, answered at the call by the mail still unread.
    text = _wait_for_task(_ctx(tmp_path), "queued", sleep_mode="warm")
    lead, _end = json.JSONDecoder().raw_decode(text)
    assert lead["woke_by"] == "mail:peer" and "Task wait returned before terminal" in text
    assert '"pre_first_result": true' in text and "unknown or not yet registered" not in text


def test_a_named_successor_with_no_record_is_never_a_sleep_on_nothing(tmp_path):
    from ouroboros import model_sleep
    from ouroboros.outcomes import EXECUTION_INFRA_FAILED, terminal_outcome_axes
    from ouroboros.tools.control_task_results import _wait_for_task

    _child(tmp_path, "parent", parent="")
    _child(tmp_path, "kid-a")
    ctx = _ctx(tmp_path)
    assert json.loads(_wait_for_task(ctx, "kid-a", sleep_mode="warm"))["reason"] == "sleep_armed"
    _child(tmp_path, "kid-a", "interrupted", superseded_by="kid-gone", retry_task_id="kid-gone")
    assert model_sleep.wake_reason(ctx, ctx._model_sleep) == "task_wait:unknown:kid-a", "no queue to say otherwise"
    _queue(tmp_path)
    axes = terminal_outcome_axes(lifecycle="interrupted", execution=EXECUTION_INFRA_FAILED,
                                 reason_code="idle_timeout_retry", review_trigger="supervisor_terminal")
    _child(tmp_path, "kid-a", "interrupted", outcome_axes=axes)
    assert model_sleep.wake_reason(ctx, ctx._model_sleep) == "task:kid-a:failed", \
        "the stored form's own recorded failure, as the in-slot wait reads it: settled is not success"


def test_a_read_fault_wakes_unknown_and_never_hides_what_a_coincident_source_says(tmp_path, monkeypatch):
    import datetime

    from ouroboros import model_sleep, owner_wait, task_status
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, KIND_OWNER_TEXT
    from ouroboros.store import StoreBusy, StoreMigrating
    from ouroboros.tools.control_task_results import _wait_for_task, _wait_for_tasks

    _child(tmp_path, "parent", parent="")
    _child(tmp_path, "kid-a")
    _child(tmp_path, "kid-b")
    exact, first, both = _ctx(tmp_path), _ctx(tmp_path), _ctx(tmp_path)
    assert json.loads(_wait_for_task(exact, "kid-a", sleep_mode="warm"))["reason"] == "sleep_armed"
    assert json.loads(_wait_for_tasks(first, ["kid-a", "kid-b"], mode="any_terminal", sleep_mode="warm"))["reason"] == "sleep_armed"
    assert json.loads(_wait_for_tasks(both, ["kid-a", "kid-b"], sleep_mode="warm"))["reason"] == "sleep_armed"
    real, fault = task_status.load_effective_task_result, [StoreBusy("store busy", reason="contention")]

    def broken(root, task_id, **kwargs):
        if task_id == "kid-a":
            raise fault[0]
        return real(root, task_id, **kwargs)

    monkeypatch.setattr(task_status, "load_effective_task_result", broken)
    sleep, poll = exact._model_sleep, {"sleep": exact._model_sleep, "wait_id": "w"}
    for state in (fault[0], StoreMigrating("the importer runs")):
        fault[0] = state
        with pytest.raises(type(state)):
            model_sleep.wake_reason(exact, sleep)
        assert owner_wait._sleep_wake(exact, poll) == "", "a store state says nothing of the task: the next poll retries"
    for fact in (StoreBusy("store not inspectable", reason="uninspectable"), OSError("selected source torn")):
        fault[0] = fact
        assert model_sleep.wake_reason(exact, sleep) == "task_wait:unknown:kid-a"
        assert owner_wait._sleep_wake(exact, poll) == "task_wait:unknown:kid-a"
    past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)).isoformat()
    assert model_sleep.wake_reason(exact, {**sleep, "wake_at": past}) == "timeout"
    _child(tmp_path, "kid-b", "completed", result="B READY")
    assert model_sleep.wake_reason(first, first._model_sleep) == "task:kid-b:completed", "a ready sibling is not hidden"
    assert model_sleep.wake_reason(both, both._model_sleep) == "task_wait:unknown:kid-a", "all can never be proven now"
    _note("kid-a", "blocker", "need credentials")
    assert model_sleep.wake_reason(exact, sleep) == "beacon:kid-a"
    _mail(tmp_path, "parent", "status?", sender="peer")
    assert model_sleep.wake_reason(exact, sleep) == "mail:peer"
    _mail(tmp_path, "parent", "the owner speaks", kind=KIND_OWNER_TEXT)
    assert model_sleep.wake_reason(exact, sleep) == "owner_text"
    _mail(tmp_path, "parent", "wrap up", kind=KIND_FINALIZE_NOW)
    assert model_sleep.wake_reason(exact, sleep) == f"control:{KIND_FINALIZE_NOW}"


def test_the_real_loop_answers_a_read_fault_as_unknown_beside_the_sibling_that_settled(tmp_path, monkeypatch):
    from ouroboros import owner_wait, task_status

    _child(tmp_path, "parent", parent="")
    _child(tmp_path, "kid-a")
    _child(tmp_path, "kid-b")
    registry = _loop_registry(tmp_path, monkeypatch)
    real = task_status.load_effective_task_result

    def broken(root, task_id, **kwargs):
        if task_id == "kid-a":
            raise OSError("selected source torn")
        return real(root, task_id, **kwargs)

    def park(sleeper, checkpoint):
        _child(tmp_path, "kid-b", "completed", result="B READY")
        monkeypatch.setattr(task_status, "load_effective_task_result", broken)
        return owner_wait._sleep_wake(sleeper, checkpoint)

    registry._ctx.owner_wait_callback = park

    def second(messages):
        (notice,) = _notices(messages)
        facts = {row["task_id"]: row for row in _header(notice)["children"]}
        assert facts["kid-a"] == {"task_id": "kid-a", "status": "unknown", "unknown_reason": "source_unreadable"}
        assert facts["kid-b"]["status"] == "completed" and "B READY" in notice
        payload = json.loads(notice.split("[BEGIN_TASK_WAIT_OUTPUT]\n", 1)[1].rsplit("\n[END_TASK_WAIT_OUTPUT]", 1)[0])
        assert payload["all_terminal"] is False and payload["tasks"]["kid-a"]["status"] == "unknown"
        assert payload["live_child_status"]["kid-a"] == "unknown"
        assert any("task kid-a of the task wait you chose" in str(row.get("content")) for row in messages)
        raise _End()

    sent = _drive(tmp_path, monkeypatch, registry, [
        _call("call-wait", "wait_tasks", task_ids=["kid-a", "kid-b"], sleep_mode="warm"), second])
    assert len(sent) == 2 and registry._ctx._pending_task_wait is None, "no hold: a lost source is a fact to deliver"


def test_a_cold_wait_whose_source_is_lost_wakes_unknown_in_its_first_resumed_request(tmp_path, monkeypatch):
    from ouroboros import task_store
    from supervisor.sleep_wake import wake_ready_sleepers

    _child(tmp_path, "foreign", parent="foreign-root")
    q, workers, _pause = _parked_cold(tmp_path, monkeypatch)
    assert wake_ready_sleepers(q) == []
    assert task_store.delete_document(tmp_path, "foreign")
    outcome = wake_ready_sleepers(q)
    assert outcome[0]["ok"] is True and outcome[0]["ready"]["reason"] == "task_wait:unknown:foreign"

    def check(messages):
        (notice,) = _notices(messages)
        assert _header(notice)["children"] == [{"task_id": "foreign", "status": "unknown", "unknown_reason": "source_missing"}]
        assert any("state is unknown, not success" in str(row.get("content")) for row in messages)

    _resumed_first_request(tmp_path, monkeypatch, workers, check)
