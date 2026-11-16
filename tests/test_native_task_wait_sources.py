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


@pytest.mark.parametrize("case", ["ready", "ready_unknown", "armed_while_retry_runs"])
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
    reply = sleeper(retry["project_id"])  # a room's reply turn holds no lane, so the queued root can start
    reply.task_metadata["_execution_owner"] = {"kind": "direct", "lane": "conversation"}
    assert json.loads(_wait_for_task(reply, "held", sleep_mode="warm"))["reason"] == "sleep_armed"
    plain = sleeper(retry["project_id"])
    assert model_sleep.request_sleep(plain, model_sleep.selectors(plain, tasks=["held"]), "warm")[
        "reason"] == "sleep_armed", "an ordinary sleep keeps its own selected row's lane read"
    other = sleeper("another-room")
    assert json.loads(_wait_for_task(other, "held", sleep_mode="warm"))["reason"] == "sleep_armed"
    [running] = _assign(host, monkeypatch)
    assert running["id"] == retry["id"] and supervisor_queue.persist_queue_snapshot()
    # Now the caller is that restored physical retry. Its logical root no longer has a queue row.
    from ouroboros.projects_registry import create_project

    shared = running["workspace_root"]
    create_project(host.root, "neighbor-room", working_dir=shared)
    neighbor = supervisor_queue.enqueue_task(_queued_root(
        "neighbor", shared, "neighbor-room", type="task", text="Other room in the same folder", chat_id=1))
    assert not neighbor.get("_admission_blocked") and supervisor_queue.persist_queue_snapshot()
    caller = _folder_sleeper(host.root, running["id"], shared, tree="held")
    caller.project_id = running["project_id"]
    caller.workspace_root, caller.workspace_mode = shared, "external"
    refused = _wait_for_task(caller, "neighbor", sleep_mode="warm")
    assert "retains the project lane (or its folder's)" in refused, refused
    assert not getattr(caller, "_model_sleep", None) and not getattr(caller, "_pending_task_wait", None)
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
    # A cold sleep is this parent's own wait: no fence holds its queued, running or retried children, so
    # waiting for them never refuses it (the root sleep fence and its running/queued census are gone).
    _retried(tmp_path)
    for selected in ("queued-kid", "started-kid", "kid-a"):
        text, armed = refusal(selected, "cold")
        assert json.loads(text)["reason"] == "sleep_armed" and armed["mode"] == "cold", selected
    # A foreign retry queued in this Project, by its own row and with no queue evidence for a row without a room.
    _retried(tmp_path, "root-a", "root-b", parent="", project_id="room")
    assert "retains the project lane" in refusal("root-a", "warm", "room")[0]
    _child(tmp_path, "no-room", "scheduled", parent="")
    (tmp_path / "state" / "queue_snapshot.json").write_text("{torn")
    text, armed = refusal("no-room", "warm", "room")
    assert "the queue cannot be read" in text and armed is None, "an unreadable queue never authorizes the lane"


def test_a_root_never_sleeps_on_queued_work_that_waits_for_its_own_folder(tmp_path):
    """Owner 15.11.2026, one queue per recorded folder: a sleeping root that keeps its lanes also keeps its folder's,
    so waiting on a queued root of ANOTHER room in that folder would wait for itself; another folder, its own tree,
    a conversation turn and work without a folder keep the sleep."""
    from ouroboros.tools.control_task_results import _wait_for_task

    shared = str(tmp_path / "shared")

    def sleeper(*, folder=shared, lane=""):
        ctx = _ctx(tmp_path, "sleeper")
        ctx.project_id, ctx.workspace_root, ctx.workspace_mode = "room-a", folder, "external" if folder else ""
        if lane:
            ctx.task_metadata["_execution_owner"] = {"kind": "direct", "lane": lane}
        return ctx

    _child(tmp_path, "sleeper", parent="")
    for task_id in ("queued-b", "queued-c"):
        _child(tmp_path, task_id, "scheduled", parent="", project_id="room-b")
    _queue(tmp_path, {"id": "queued-b", "root_task_id": "queued-b", "project_id": "room-b",
                      "workspace_root": shared, "workspace_mode": "external"},
           {"id": "queued-c", "root_task_id": "queued-c", "project_id": "room-b",
            "workspace_root": str(tmp_path / "other"), "workspace_mode": "external"})
    ctx = sleeper()
    text = _wait_for_task(ctx, "queued-b", sleep_mode="warm")
    assert "retains the project lane (or its folder's)" in text and not getattr(ctx, "_model_sleep", None)
    assert json.loads(_wait_for_task(sleeper(), "queued-c", sleep_mode="warm"))["reason"] == "sleep_armed"
    assert json.loads(_wait_for_task(sleeper(folder=""), "queued-b", sleep_mode="warm"))["reason"] == "sleep_armed"
    assert json.loads(_wait_for_task(sleeper(lane="conversation"), "queued-b", sleep_mode="warm"))[
        "reason"] == "sleep_armed", "a room's reply turn holds no lane"


def _folder_sleeper(root, task_id, folder, *, tree=""):
    """A project-scoped actor of tree ``tree or task_id`` in room-a; a root records ``folder``, a helper its own copy."""
    ctx = _ctx(root, task_id)
    if tree:  # a helper: its tree's root, as every reader of its coordinates takes it
        ctx.root_task_id = ctx.task_metadata["root_task_id"] = ctx.model_wait_context.task["root_task_id"] = tree
    ctx.project_id = "room-a"
    ctx.workspace_root, ctx.workspace_mode = (folder, "external") if not tree else (str(root / "own-copy"), "self_worktree")
    return ctx


def _queued_root(task_id, folder, project="room-b", **fields):
    return {"id": task_id, "root_task_id": task_id, "project_id": project, "workspace_root": folder,
            "workspace_mode": "external", "metadata": {}, **fields}


@pytest.mark.parametrize("actor", ["continue", "retry", "helper", "retry-helper"])
@pytest.mark.parametrize("door", ["wait_task", "wait_tasks", "await_tasks", "await_senders"])
@pytest.mark.parametrize("mode", ["warm", "cold"])
def test_sleep_entry_uses_the_current_physical_roots_folder(tmp_path, actor, door, mode):
    """A restored retry keeps its logical tree but only its physical root occupies the queue.
    The actual wait entries refuse self-blocking warm sleep; an unencumbered cold sleep releases it.
    Continue (same id), helpers and other-folder/same-tree selections retain their existing behavior."""
    from ouroboros.project_lease import candidate_is_leasable, lane_holders, running_project_ids
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_task_results import _await_messages, _wait_for_task, _wait_for_tasks

    shared = str(tmp_path / "shared")
    physical = "root-retry" if actor.startswith("retry") else "root"
    holder = _queued_root(physical, shared, "room-a", root_task_id="root")
    helper = "helper" in actor
    caller = "helper" if helper else physical
    write_task_result(tmp_path, physical, "running", root_task_id="root", project_id="room-a")
    if physical != "root":
        write_task_result(tmp_path, "root", "interrupted", superseded_by=physical, retry_task_id=physical)
    if helper:
        _child(tmp_path, caller, parent="root")
    _child(tmp_path, "own-kid", "scheduled", parent="root", project_id="room-a")
    for selected in ("neighbor", "elsewhere"):
        _child(tmp_path, selected, "scheduled", parent="", project_id="room-b")
    neighbor = _queued_root("neighbor", shared)
    _queue(tmp_path, neighbor, _queued_root("elsewhere", str(tmp_path / "other")),
           {"id": "own-kid", "root_task_id": "root", "delegation_role": "subagent", "project_id": "room-a"},
           running=[holder])
    assert not candidate_is_leasable(neighbor, running_project_ids([holder]), holders=lane_holders([holder], []))

    def call(selected):
        ctx = _folder_sleeper(tmp_path, caller, shared, tree="root" if caller != "root" else "")
        if not helper:
            ctx.workspace_root, ctx.workspace_mode = shared, "external"
        if door == "wait_task":
            text = _wait_for_task(ctx, selected, sleep_mode=mode)
        elif door == "wait_tasks":
            text = _wait_for_tasks(ctx, [selected], sleep_mode=mode)
        else:
            text = _await_messages(ctx, mode=mode, **{door.removeprefix("await_"): [selected]})
        return ctx, text

    ctx, text = call("neighbor")
    if mode == "warm":
        assert "retains the project lane (or its folder's)" in text, text
        assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_pending_task_wait", None)
    else:
        assert json.loads(text)["reason"] == "sleep_armed", text
    for selected in ("elsewhere", "own-kid"):
        _, text = call(selected)
        assert json.loads(text)["reason"] == "sleep_armed", text


def test_cold_custody_also_refuses_a_retry_waiting_for_its_own_folder(tmp_path, monkeypatch):
    from ouroboros import sleep_custody
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_task_results import _wait_for_task

    shared = str(tmp_path / "shared")
    write_task_result(tmp_path, "retry", "running", root_task_id="root", project_id="room-a")
    _child(tmp_path, "neighbor", "scheduled", parent="", project_id="room-b")
    _queue(tmp_path, _queued_root("neighbor", shared),
           running=[_queued_root("retry", shared, "room-a", root_task_id="root")])
    # Only the presence of accepted external custody differs; refusal precedes any transfer or park.
    monkeypatch.setattr(sleep_custody, "transferable_pairs", lambda *_: [("job", "run")])
    ctx = _folder_sleeper(tmp_path, "retry", shared, tree="root")
    text = _wait_for_task(ctx, "neighbor", sleep_mode="cold")
    assert "retains the project lane (or its folder's)" in text, text
    assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_pending_task_wait", None)


@pytest.mark.parametrize("root_asleep", [False, True])
def test_a_helper_never_sleeps_on_another_rooms_work_queued_for_its_roots_folder(tmp_path, root_asleep):
    """The tree holds its root's folder whatever copy a helper works in: the helper reads it from its root's queue row,
    never from its own context. Its own tree's queued work, and a root that records no folder, keep the sleep."""
    from ouroboros import model_sleep

    shared = str(tmp_path / "shared")
    _child(tmp_path, "root", parent="")
    _child(tmp_path, "kid", parent="root")
    _child(tmp_path, "queued-b", "scheduled", parent="", project_id="room-b")
    _child(tmp_path, "queued-kid", "scheduled", parent="root")
    root_row = {"id": "root", "root_task_id": "root", "project_id": "room-a", "workspace_root": shared,
                "workspace_mode": "external"}
    kid_rows = ({"id": "queued-kid", "root_task_id": "root", "parent_task_id": "root", "delegation_role": "subagent",
                 "project_id": "room-a"},
                {"id": "kid", "root_task_id": "root", "parent_task_id": "root", "delegation_role": "subagent"})
    _queue(tmp_path, _queued_root("queued-b", shared), kid_rows[0],
           *([{**root_row, "_budget_pause": {"reason": "sleep"}}] if root_asleep else []),
           running=([kid_rows[1]] if root_asleep else [root_row, kid_rows[1]]))
    helper = _folder_sleeper(tmp_path, "kid", "", tree="root")
    with pytest.raises(ValueError, match=r"retains the project lane \(or its folder's\)"):
        model_sleep.request_sleep(helper, model_sleep.selectors(helper, tasks=["queued-b"]), model_sleep.MODE_WARM)
    for selected in ("queued-kid",):
        same = _folder_sleeper(tmp_path, "kid", "", tree="root")
        assert model_sleep.request_sleep(same, model_sleep.selectors(same, tasks=[selected]),
                                         model_sleep.MODE_WARM)["reason"] == "sleep_armed", "its own tree's work"
    _queue(tmp_path, _queued_root("queued-b", shared), kid_rows[0],
           running=[{**root_row, "workspace_root": "", "workspace_mode": ""}, kid_rows[1]])
    free = _folder_sleeper(tmp_path, "kid", "", tree="root")
    assert model_sleep.request_sleep(free, model_sleep.selectors(free, tasks=["queued-b"]),
                                     model_sleep.MODE_WARM)["reason"] == "sleep_armed", "its root holds no folder"


@pytest.mark.parametrize("pinned", [False, True])
def test_a_root_never_sleeps_on_queued_work_whose_pending_move_selects_its_folder(tmp_path, monkeypatch, pinned):
    """The folder a queued root waits for is the one its readmission will select (its durable move's), as the chooser
    reads it: a move from B into a room whose folder the sleeper holds refuses the sleep; work keeping its own
    folder B (an explicit resource) keeps it."""
    from ouroboros import model_sleep
    from ouroboros.project_placement import move_task_to_project, placement_basis
    from ouroboros.projects_registry import create_project

    held, elsewhere = (tmp_path / "folders" / name for name in ("held", "elsewhere"))
    for folder in (held, elsewhere):
        folder.mkdir(parents=True)
    create_project(tmp_path, "room-x", working_dir=str(held))
    _child(tmp_path, "sleeper", parent="")
    _child(tmp_path, "moved", "scheduled", parent="", project_id="room-b")
    row = _queued_root("moved", str(elsewhere), metadata={"resource_intent": {"kind": "explicit_resource",
                                                                             "root": str(elsewhere)}} if pinned else {})
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda **kw: True)
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda event: None))
    move_task_to_project(tmp_path, "moved", "room-x", origin={"absent": "mid_task_no_origin"},
                         basis=placement_basis(tmp_path, "moved", "room-x"), running={}, pending=[row])
    _queue(tmp_path, row)
    ctx = _folder_sleeper(tmp_path, "sleeper", str(held))
    if pinned:
        assert model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, tasks=["moved"]),
                                         model_sleep.MODE_WARM)["reason"] == "sleep_armed"
    else:
        with pytest.raises(ValueError, match=r"retains the project lane \(or its folder's\)"):
            model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, tasks=["moved"]), model_sleep.MODE_WARM)


def test_a_root_never_sleeps_on_an_original_whose_queued_retry_waits_for_its_folder(tmp_path):
    """Who answers for the id selected (its retry successor) is who waits: its queue row's folder is compared, so the
    original's id never hides it. A successor in another folder, and a cold sleep that keeps no custody, still sleep."""
    from ouroboros.tools.control_task_results import _wait_for_task

    shared = str(tmp_path / "shared")
    _child(tmp_path, "sleeper", parent="")
    _retried(tmp_path, "root-a", "root-b", parent="", project_id="room-b")
    _retried(tmp_path, "root-c", "root-d", parent="", project_id="room-b")
    _queue(tmp_path, _queued_root("root-b", shared, supersedes_task_id="root-a"),
           _queued_root("root-d", str(tmp_path / "other"), supersedes_task_id="root-c"))
    ctx = _folder_sleeper(tmp_path, "sleeper", shared)
    text = _wait_for_task(ctx, "root-a", sleep_mode="warm")
    assert "retains the project lane (or its folder's)" in text and not getattr(ctx, "_model_sleep", None)
    assert json.loads(_wait_for_task(_folder_sleeper(tmp_path, "sleeper", shared), "root-c", sleep_mode="warm"))[
        "reason"] == "sleep_armed", "its successor works in another folder"
    cold = json.loads(_wait_for_task(_folder_sleeper(tmp_path, "sleeper", shared), "root-a", sleep_mode="cold"))
    assert cold["reason"] == "sleep_armed" and cold["mode"] == "cold", "a cold sleep without custody releases the lanes"


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
    read = len(calls)  # a cold sleep fences no selected child, so its census never asks who answers for one
    assert model_sleep.cold_blockers(_ctx(tmp_path), chosen={"tasks": ["kid-a"], "task_wait": {"task_ids": ["kid-a"]}}) == []
    assert len(calls) == read


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


_LONG_REASON = ("The release waited on the owner's word about the database, which came at 14:05 in the project room. "
                * 4 + "Only the build check remains, and it must pass on both platforms before anything ships.")


@pytest.mark.parametrize(("by", "outcome", "woken"), [
    ({"actor": "owner", "door": "api_task_resume"}, "control:resume", "an explicit Resume by your human"),
    ({"actor": "ouroboros", "task_id": "c1", "text": _LONG_REASON}, "",
     f'an explicit Resume by Ouroboros (task c1), saying "{_LONG_REASON}"'),
    ({}, "control:resume", "the owner's message or control"),
    ({"actor": "owner"}, "timeout", "the wake time you chose")])
def test_a_wake_names_the_resume_that_woke_it_and_only_for_a_control_wake(by, outcome, woken):
    """A Resume that ended a sleep names who granted it, its reason whole however long (P1: no silent cut); an unnamed
    control keeps its old words, and a wake time the sleeper chose is never retold as somebody's Resume."""
    from ouroboros import model_sleep

    text = model_sleep.wake_notice({"mode": "cold"}, outcome, 5.0, by=by)["content"]
    assert f"were woken by {woken}" in text, text


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
