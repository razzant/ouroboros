"""A held queued task reads as held wherever the mind decides to wait for it.

Real owner Pause, exact park, root-only Resume, dispatch and consumption, Q9 selection through
``resume_child_task`` and the supervisor's handler, and the real readers (``get_task_result``,
``peek_task``, ``wait_task``/``wait_tasks``, ``await_messages``); only cognition and the worker
inbox are stubbed. The stored status stays ``scheduled``: queue membership is not permission to
run, so the census's reading of the row (``work_stop.task_current_stop``) rides beside it.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_budget_pause_holds import _idle_worker
from tests.test_budget_pause_lineage_restore import _consume
from tests.test_model_sleep import _ctx
from tests.test_owner_pause import _live_tree, _owner_park
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial  # the queue, its workers and the supervisor seams are module state


def _held_tree(tmp_path, monkeypatch):
    """The whole tree paused, its running child parked exactly, then the root alone resumed and consumed."""
    from ouroboros import budget_pause
    from ouroboros.task_results import STATUS_RUNNING, STATUS_SCHEDULED, write_task_result
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 5.0)
    _live_tree(tmp_path, workers)
    for task_id, status in (("child-run", STATUS_RUNNING), ("child-new", STATUS_SCHEDULED)):
        # A scheduled child's own row names its lineage, as schedule_subagent writes it.
        write_task_result(tmp_path, task_id, status, chat_id=0, parent_task_id="root-1", root_task_id="root-1",
                          delegation_role="subagent")
    assert request_owner_pause("root-1", request_id="tree-pause")["ok"]
    sup = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=queue.sort_pending,
                          persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
    for task_id in ("child-run", "root-1"):
        pause = _owner_park(tmp_path, monkeypatch, task_id, [])
        install_exact_budget_pause(sup, task_id, budget_pause.exact_pause_marker(pause)["checkpoint"])
    assert queue.resume_budget_paused_task("root-1")["ok"]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["root-1"], "root-only Resume selects no child"
    meta = workers.RUNNING["root-1"]
    ctx, limit = _loop_ctx(tmp_path, "root-1", attempt=meta["attempt"])
    _consume(monkeypatch, ctx, limit, meta["task"]["_budget_pause_resume"])
    assert queue.persist_queue_snapshot(reason="root-resumed")
    return SimpleNamespace(queue=queue, workers=workers, sent=sent, sup=sup)


def _drain(ctx, tmp_path):
    """The root's own round top read its mail (the Pause's control) into its transcript and acknowledged it,
    as its loop does; no cognition runs."""
    from ouroboros.owner_mailbox import acknowledge_transcript_entry, drain_owner_entries

    for entry in drain_owner_entries(tmp_path, ctx.task_id, set(), 1):
        assert acknowledge_transcript_entry(tmp_path, ctx.task_id, entry)
        if entry.get("msg_id"):
            ctx._loop_mailbox_seen_ids.add(entry["msg_id"])
    return ctx


def _source_sha(tmp_path, row):
    import hashlib

    from ouroboros.artifacts import read_actor_source_bytes

    return hashlib.sha256(read_actor_source_bytes(tmp_path, "child-run", row["source_ref"])).hexdigest()


def _bridge(tree, outcomes):
    return SimpleNamespace(**{**vars(tree.sup), "bridge": SimpleNamespace(push_log=outcomes.append)})


def _read(tmp_path, task_id):
    from ouroboros.task_status import load_effective_task_result

    return load_effective_task_result(tmp_path, task_id, materialize_artifacts=False)


def _outcome(text):
    return json.loads(text.split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0])


def _deliver(ctx, text):
    """The loop's delivery of one complete view (``process_tool_results``) acknowledges the holds it shows."""
    from ouroboros.model_sleep import acknowledge_hold_delivery

    acknowledge_hold_delivery(ctx, text)
    ctx._held_dependencies_pending = []


def _published(ctx):
    """The text of the result a reader published, whose meta flags the holds it shows."""
    from ouroboros.model_sleep import HOLD_NOTICE
    from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR

    result = getattr(ctx, _TOOL_RESULT_ATTR)
    assert result.meta[HOLD_NOTICE] is True, "bounded: one flag, whatever the wait set's size"
    return result.text


def _root_ctx(tmp_path):
    from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR

    ctx = _ctx(tmp_path, "root-1", root="root-1")
    setattr(ctx, _TOOL_RESULT_ATTR, None)  # the registry's sidecar slot for published results
    return _drain(ctx, tmp_path)


def test_root_only_resume_shows_each_retained_child_held_with_its_request_door(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from ouroboros.tools.control_task_results import _get_task_result, _wait_for_task, _wait_for_tasks
    from ouroboros.tools.join_ledger import _peek_task

    _held_tree(tmp_path, monkeypatch)
    before = budget_pause.budget_pause_row(tmp_path, "child-run")
    assert before["state"] == "paused" and _source_sha(tmp_path, before) == before["source_ref"]["sha256"]
    exact, never_started = _read(tmp_path, "child-run"), _read(tmp_path, "child-new")
    for row, release_code in ((exact, "awaiting_selection"), (never_started, "awaiting_selection")):
        assert row["status"] == "scheduled", "the queue row stays where it is"
        stop = row["current_stop"]
        assert (stop["state"], stop["cause"], stop["scope"], stop["code"], stop["release"]) == (
            "held", "awaiting_selection", "member", release_code, "mind")
        assert row["execution_observation"]["dispatch"] == "held"
        assert row["execution_observation"]["reason"] == "queue_ownership", "custody stays custody"
    assert exact["current_stop"]["identity"] != never_started["current_stop"]["identity"]

    root = _root_ctx(tmp_path)
    full = _outcome(_get_task_result(root, "child-run"))["current_stop"]
    assert full["door"] == "resume_child_task" and full["state"] == "held"
    sha = _get_task_result(root, "child-run").split("child_result_sha256=", 1)[1].split("\n", 1)[0]
    unchanged = _get_task_result(root, "child-run", known_result_sha256=sha)
    assert "result_unchanged" in unchanged and _outcome(unchanged)["current_stop"] == full
    for view in ("answer", "summary", "evidence"):
        assert json.loads(_get_task_result(root, "child-run", view=view))["current_stop"] == full, view
    peeked = _peek_task(root, "child-run")
    assert '"door": "resume_child_task"' in peeked and "[scheduled]" in peeked
    # Another tree's reader sees the stop, never a door that is not its own to request.
    assert "door" not in _outcome(_get_task_result(_ctx(tmp_path, "peer-root", root="peer-root"), "child-run"))[
        "current_stop"]

    waited = json.loads(_wait_for_tasks(root, ["child-run", "child-new"], timeout_sec=0))
    assert waited["early_return"]["reason"] == "dependency_held"
    assert sorted(row["task_id"] for row in waited["early_return"]["held"]) == ["child-new", "child-run"]
    assert waited["live_child_status"] == {"child-run": "held", "child-new": "held"}
    assert waited["tasks"]["child-run"]["current_stop"]["door"] == "resume_child_task"
    _deliver(root, _published(root))
    again = json.loads(_wait_for_tasks(root, ["child-run", "child-new"], timeout_sec=0))
    assert "early_return" not in again, "a hold already shown is waited on like any other state"
    assert again["live_child_status"] == {"child-run": "held", "child-new": "held"}

    single = _wait_for_task(_root_ctx(tmp_path), "child-run", timeout_sec=0)
    assert single.startswith("Task wait returned to show a held dependency") and "[HELD_DEPENDENCY]" in single
    # The sibling advice no longer counts a held child among those that will finish.
    assert "1 other child(ren) held by a current stop (child-new)" in single
    assert "still running/scheduled" not in single
    # Reading, waiting and being shown changed no saved point, source byte or task identity.
    assert budget_pause.budget_pause_row(tmp_path, "child-run") == before
    assert _source_sha(tmp_path, before) == before["source_ref"]["sha256"]


def test_the_selected_child_loses_its_stop_and_the_same_id_dispatches_with_its_saved_work(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from ouroboros.cost_projection import cost_projection
    from ouroboros.tools import control, join_ledger
    from supervisor.events_budget import _handle_budget_resume_child

    tree = _held_tree(tmp_path, monkeypatch)
    parked = budget_pause.budget_pause_row(tmp_path, "child-run")
    cost_before = cost_projection(_read(tmp_path, "child-run"))
    events, outcomes = [], []
    monkeypatch.setattr(control, "_emit_control_event", lambda _ctx, evt: events.append(evt) or "live")
    monkeypatch.setattr(join_ledger, "_record_child_decision_beacon", lambda *_a, **_k: None)
    reply = join_ledger._resume_child_task(_root_ctx(tmp_path), "child-run", reason="still needed")
    assert "A grant removes the child's current_stop" in reply and "from paused to scheduled" not in reply
    assert _read(tmp_path, "child-run")["current_stop"]["state"] == "held", "a request is not a grant"
    _handle_budget_resume_child(events[0], _bridge(tree, outcomes))
    assert outcomes[-1]["type"] == "budget_resume_child_outcome" and outcomes[-1]["ok"] is True
    assert tree.queue.persist_queue_snapshot(reason="child-selected")

    selected = _read(tmp_path, "child-run")
    assert selected["status"] == "scheduled" and selected["current_stop"] == {}
    assert selected["execution_observation"]["dispatch"] == "queued", "granted: it waits for a worker"
    assert budget_pause.budget_pause_row(tmp_path, "child-run")["state"] == "resume_granted"
    assert _read(tmp_path, "child-new")["current_stop"]["state"] == "held", "its sibling stays unselected"
    assert cost_projection(selected) == cost_before

    tree.workers.WORKERS[1] = affirm_ready(SimpleNamespace(
        wid=1, busy_task_id=None, reaping=False, in_q=SimpleNamespace(put=lambda t: tree.sent.append(dict(t)))))
    monkeypatch.setattr(tree.workers, "MAX_WORKERS", 2)
    tree.workers.assign_tasks()
    assert [row["id"] for row in tree.sent] == ["root-1", "child-run"], "the same id, and never its sibling"
    handoff = tree.workers.RUNNING["child-run"]["task"]["_budget_pause_resume"]
    assert (handoff["pause_id"], handoff["source_ref"]) == (parked["pause_id"], parked["source_ref"])
    assert _source_sha(tmp_path, parked) == parked["source_ref"]["sha256"]
    assert tree.queue.persist_queue_snapshot(reason="child-dispatched")
    running = _read(tmp_path, "child-run")
    assert running["status"] == "running" and running["current_stop"] == {}
    assert "dispatch" not in running["execution_observation"]
    assert cost_projection(running) == cost_before


def test_a_new_pause_during_selection_holds_the_child_again_under_its_tree(tmp_path, monkeypatch):
    from ouroboros.model_sleep import unshown_holds
    from ouroboros.tools import control, join_ledger
    from supervisor.events_budget import _handle_budget_resume_child
    from supervisor.owner_pause_control import request_owner_pause

    tree = _held_tree(tmp_path, monkeypatch)
    first = _read(tmp_path, "child-run")["current_stop"]
    root = _root_ctx(tmp_path)
    root._held_dependencies_seen = {"child-run": first["identity"]}
    events, outcomes = [], []
    monkeypatch.setattr(control, "_emit_control_event", lambda _ctx, evt: events.append(evt) or "live")
    monkeypatch.setattr(join_ledger, "_record_child_decision_beacon", lambda *_a, **_k: None)
    join_ledger._resume_child_task(root, "child-run")
    _handle_budget_resume_child(events[0], _bridge(tree, outcomes))
    assert outcomes[-1]["ok"] is True
    assert request_owner_pause("root-1", request_id="second-pause")["ok"]
    assert tree.queue.persist_queue_snapshot(reason="paused-again")
    again = _read(tmp_path, "child-run")
    stop = again["current_stop"]
    assert (stop["state"], stop["scope"], stop["cause"]) == ("held", "tree", "pause")
    assert "door" not in unshown_holds(root, {"child-run": again})[0], "a tree's Pause is released with its tree"
    assert stop["identity"] != first["identity"], "a newer Pause is new information"
    tree.workers.assign_tasks()
    assert "child-run" not in [row["id"] for row in tree.sent], "the newer Pause keeps it undispatched"


def test_ordinary_queue_unknown_authority_panic_and_stop_keep_their_own_words(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from ouroboros.tools.control_task_results import _await_messages

    tree = _held_tree(tmp_path, monkeypatch)
    write_task_result(tmp_path, "plain", STATUS_SCHEDULED, chat_id=0)
    tree.workers.PENDING.append({"id": "plain", "type": "task", "chat_id": 0, "_attempt": 1, "root_task_id": "plain"})
    write_task_result(tmp_path, "held-panic", STATUS_SCHEDULED, chat_id=0)
    tree.workers.PENDING.append({"id": "held-panic", "type": "task", "chat_id": 0, "_attempt": 1,
                                 "root_task_id": "held-panic", "_budget_pause_hold": {
                                     "reason": "panic_hold", "release": "owner", "held_at": "2026-10-29T15:00:00Z"}})
    assert tree.queue.persist_queue_snapshot(reason="controls")

    plain = _read(tmp_path, "plain")
    assert plain["current_stop"] == {} and plain["execution_observation"]["dispatch"] == "queued"
    root = _root_ctx(tmp_path)
    armed = json.loads(_await_messages(root, mode="warm", tasks=["plain"]))
    assert armed["reason"] == "sleep_armed", "an ordinary queued task costs no extra round"
    root._model_sleep = root._owner_wait_requested = None

    panic = _read(tmp_path, "held-panic")["current_stop"]
    assert (panic["state"], panic["cause"], panic["release"]) == ("held", "panic", "owner")
    from ouroboros.work_stop import stop_for_reader

    assert "door" not in stop_for_reader(panic, {"task_id": "held-panic", "root_task_id": "root-1"}, "root-1")

    request_cancel(tmp_path, "child-new", reason="no longer needed", requested_by="root-1")
    from ouroboros.task_status import wait_for_effective_tasks

    assert wait_for_effective_tasks(tmp_path, ["child-new"], timeout_sec=0)["live_child_status"] == {
        "child-new": "cancel_pending"}, "a Stop outranks the hold it ends"

    from ouroboros.tools.control_task_results import _get_task_result

    with monkeypatch.context() as patch:  # a supervisor that stopped publishing: every queue observation is stale
        patch.setattr("ouroboros.task_status._SNAPSHOT_OWNERSHIP_FRESH_SEC", -1.0)
        stale = _read(tmp_path, "child-run")
        assert stale["current_stop"]["queue_freshness"] == "stale" and "dispatch" not in stale["execution_observation"]
        assert "door" not in _outcome(_get_task_result(root, "child-run"))["current_stop"], "stale grants no action"

    real = owner_pause.member_fence
    monkeypatch.setattr(owner_pause, "member_fence", lambda source, **k: (
        {"state": "unknown", "reason": "owner_pause_authority_unreadable"}
        if getattr(source, "task_id", "") == "child-run" else real(source, **k)))
    unknown = _read(tmp_path, "child-run")
    stop = unknown["current_stop"]
    assert (stop["state"], stop["reason"]) == ("unknown", "control_unreadable") and "release" not in stop
    assert unknown["execution_observation"]["dispatch"] == "unknown", "unknown is neither held nor runnable"
    shown = json.loads(_await_messages(root, mode="warm", tasks=["child-run"]))
    assert shown["reason"] == "dependency_held" and shown["held"][0]["state"] == "unknown"
    assert "door" not in shown["held"][0], "unknown authority grants no action"
    _deliver(root, _published(root))
    assert json.loads(_await_messages(root, mode="warm", tasks=["child-run"]))["reason"] == "sleep_armed"


def test_a_shown_hold_lets_the_sleep_wait_for_its_external_release_and_a_newer_pause_shows_again(
        tmp_path, monkeypatch):
    from ouroboros.model_sleep import wake_reason
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control_task_results import _await_messages
    from supervisor.owner_pause_control import request_owner_pause

    tree = _held_tree(tmp_path, monkeypatch)
    root = _root_ctx(tmp_path)
    shown = json.loads(_await_messages(root, mode="warm", tasks=["child-run"], senders=["child-run"]))
    assert shown["reason"] == "dependency_held" and shown["slept"] is False
    assert not getattr(root, "_model_sleep", None), "nothing was armed"
    assert shown["held"][0]["door"] == "resume_child_task"
    _deliver(root, _published(root))
    armed = json.loads(_await_messages(root, mode="warm", tasks=["child-run"], senders=["child-run"]))
    assert armed["reason"] == "sleep_armed", "the deliberate wait for an outside release is the mind's to keep"
    chosen = {key: root._model_sleep[key] for key in ("tasks", "senders", "runs", "wake_at", "any_mail")}
    assert wake_reason(root, chosen) == "", "a held dependency is no wake; only what releases it is"

    # A newer Pause of the tree, then the root alone resumed again: the same child is new information.
    root._model_sleep = root._owner_wait_requested = None
    assert request_owner_pause("root-1", request_id="pause-b")["ok"]
    assert tree.queue.persist_queue_snapshot(reason="paused-b")
    renewed = json.loads(_await_messages(root, mode="warm", tasks=["child-run"]))
    assert renewed["reason"] == "ready" and renewed["woke_by"] == "control:owner_pause", "the Pause speaks first"
    renewed = json.loads(_await_messages(_drain(root, tmp_path), mode="warm", tasks=["child-run"]))
    assert renewed["reason"] == "dependency_held"
    assert renewed["held"][0]["identity"] != shown["held"][0]["identity"]

    write_task_result(tmp_path, "child-run", STATUS_COMPLETED, chat_id=0, result="done")
    assert wake_reason(root, chosen).startswith("task:child-run:"), "its own terminal still wakes the sleep"


def _money_cycle(tree, tmp_path, monkeypatch):
    """The resumed root spends its money (an exact root-scope pause), the owner resumes the root alone and it
    runs again: the earlier owner Resume's tree, through one more control cycle."""
    from ouroboros import budget_pause
    from supervisor.events_budget import install_exact_budget_pause
    from tests._budget_pause_exact_helpers import _fast_hold, _mock_pause_observation

    meta = tree.workers.RUNNING["root-1"]
    _root, limit = _loop_ctx(tmp_path, "root-1", attempt=meta["attempt"])
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [])
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED, scope="root",
                                   reason_text="money gone", root_task_id="root-1")
    budget_pause.end_dispatch_fence("root-1")
    install_exact_budget_pause(tree.sup, "root-1", budget_pause.exact_pause_marker(raised.value.pause)["checkpoint"])
    for worker in tree.workers.WORKERS.values():
        worker.busy_task_id = None  # the worker reported the park and is idle again
    assert tree.queue.persist_queue_snapshot(reason="root-money-paused")
    resumed = tree.queue.resume_budget_paused_task("root-1")
    assert resumed["ok"], resumed
    tree.workers.assign_tasks()
    assert tree.sent[-1]["id"] == "root-1", "the root alone runs again"
    meta = tree.workers.RUNNING["root-1"]
    ctx, limit = _loop_ctx(tmp_path, "root-1", attempt=meta["attempt"])
    _consume(monkeypatch, ctx, limit, meta["task"]["_budget_pause_resume"])
    assert tree.queue.persist_queue_snapshot(reason="root-resumed-from-money")
    return raised.value.pause


def test_a_root_money_pause_cycle_shows_each_held_child_again_and_its_selection_still_grants(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from ouroboros.cost_projection import cost_projection
    from ouroboros.tools import control, join_ledger
    from ouroboros.tools.control_task_results import _await_messages
    from supervisor.events_budget import _handle_budget_resume_child

    tree = _held_tree(tmp_path, monkeypatch)
    parked, cost = budget_pause.budget_pause_row(tmp_path, "child-run"), cost_projection(_read(tmp_path, "child-run"))
    root, selected = _root_ctx(tmp_path), ["child-run", "child-new"]
    first = json.loads(_await_messages(root, mode="warm", tasks=selected))
    assert first["reason"] == "dependency_held"
    shown = {row["task_id"]: row["identity"] for row in first["held"]}
    _deliver(root, _published(root))
    assert json.loads(_await_messages(root, mode="warm", tasks=selected))["reason"] == "sleep_armed"
    root._model_sleep = root._owner_wait_requested = None

    money = _money_cycle(tree, tmp_path, monkeypatch)
    assert money["pause_id"] != parked["pause_id"]
    again = json.loads(_await_messages(_drain(root, tmp_path), mode="warm", tasks=selected))
    assert again["reason"] == "dependency_held", again
    renewed = {row["task_id"]: row for row in again["held"]}
    assert sorted(renewed) == sorted(shown), "each held child is new information under the new root Resume"
    for task_id, identity in shown.items():
        assert renewed[task_id]["identity"] != identity and renewed[task_id]["door"] == "resume_child_task"
        assert renewed[task_id].get("from_earlier_call") is None
    _deliver(root, _published(root))
    assert json.loads(_await_messages(root, mode="warm", tasks=selected))["reason"] == "sleep_armed", \
        "shown once under this Resume: the same wait then arms"
    root._model_sleep = root._owner_wait_requested = None
    assert budget_pause.budget_pause_row(tmp_path, "child-run") == parked, "the child's saved point is untouched"

    events, outcomes = [], []
    monkeypatch.setattr(control, "_emit_control_event", lambda _ctx, evt: events.append(evt) or "live")
    monkeypatch.setattr(join_ledger, "_record_child_decision_beacon", lambda *_a, **_k: None)
    join_ledger._resume_child_task(root, "child-run", reason="still needed")
    _handle_budget_resume_child(events[0], _bridge(tree, outcomes))
    assert outcomes[-1]["ok"] is True, "the explicit selection is granted under the new root Resume"
    assert tree.queue.persist_queue_snapshot(reason="child-selected")
    chosen = _read(tmp_path, "child-run")
    assert chosen["status"] == "scheduled" and chosen["current_stop"] == {} and cost_projection(chosen) == cost
    assert _read(tmp_path, "child-new")["current_stop"]["state"] == "held", "its sibling stays unselected"
    assert _source_sha(tmp_path, parked) == parked["source_ref"]["sha256"]


def _continued(tree, tmp_path):
    """The first episode ends on a technical interruption while its helpers stay held, and the owner's Continue
    runs the next episode of the same work: a new root that names its predecessor (``owner_continue``)."""
    import time

    from ouroboros.task_results import STATUS_INTERRUPTED, STATUS_RUNNING, write_task_result

    tree.workers.RUNNING.pop("root-1")
    write_task_result(tmp_path, "root-1", STATUS_INTERRUPTED, chat_id=0, root_task_id="root-1", result="interrupted")
    metadata = {"continuation": {"predecessor_task_id": "root-1"}}
    write_task_result(tmp_path, "root-2", STATUS_RUNNING, chat_id=0, root_task_id="root-2", metadata=metadata)
    task = {"id": "root-2", "type": "task", "chat_id": 0, "root_task_id": "root-2", "_attempt": 1, "metadata": metadata}
    tree.workers.RUNNING["root-2"] = {"task": task, "attempt": 1, "started_at": time.time(), "worker_id": 0}
    tree.workers.WORKERS[0].busy_task_id = "root-2"
    ctx, limit = _loop_ctx(tmp_path, "root-2")
    ctx.task_metadata, ctx._loop_mailbox_seen_ids = metadata, set()
    return ctx, limit


def _cold_sleep(tree, tmp_path, monkeypatch, ctx, limit, **chosen):
    """The episode's own cold sleep: requested (a hold it was not shown answers first), parked at the round
    boundary under its exact pause, and queued again with the saved selection."""
    from ouroboros import budget_pause, model_sleep
    from supervisor.events_budget import install_exact_budget_pause
    from tests._budget_pause_exact_helpers import _fast_hold, _mock_pause_observation

    outcome = model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, **chosen), "cold")
    if outcome["reason"] == "dependency_held":
        assert all("door" not in row for row in outcome["held"]), "neither its root nor its parent: no door"
        model_sleep.hold_notice(ctx, outcome["held"])
        _deliver(ctx, json.dumps(outcome))
        outcome = model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, **chosen), "cold")
    assert outcome["reason"] == "sleep_armed" and outcome["mode"] == "cold", outcome
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [])
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_cold_sleep(limit)
    budget_pause.end_dispatch_fence("root-2")
    install_exact_budget_pause(tree.sup, "root-2", budget_pause.exact_pause_marker(raised.value.pause)["checkpoint"])
    assert tree.queue.persist_queue_snapshot(reason="cold-sleep")


@pytest.mark.parametrize("chosen", [{"tasks": ["child-run"]}, {"wake_after_sec": 3600}])
def test_a_cold_sleep_names_on_the_owner_line_only_the_held_helper_it_chose(tmp_path, monkeypatch, chosen):
    from tests._work_projection_shared import state_fields

    tree = _held_tree(tmp_path, monkeypatch)
    ctx, limit = _continued(tree, tmp_path)
    _cold_sleep(tree, tmp_path, monkeypatch, ctx, limit, **chosen)
    [row] = [row for row in tree.workers.PENDING if row["id"] == "root-2"]
    assert row["_budget_pause"]["sleep"]["mode"] == "cold"
    [work] = [work for work in state_fields(tmp_path)["works"] if work["episode"]["task_id"] == "root-2"]
    assert (work["activity"], work["because"][0]["code"]) == ("waiting", "sleeping"), "the sleep keeps its word"
    # The ended episode's held helpers are the owner's to select, whatever the sleep chose: the line adds no turn.
    assert sorted((entry["task_id"], entry["code"]) for entry in work["turn"]) == [
        ("child-new", "awaiting_selection"), ("child-run", "awaiting_selection")]
    held = [(tail["code"], tail["ids"], tail["state"], tail["text"]) for tail in work["tails"] if tail["kind"] == "helper"]
    if chosen.get("tasks"):  # this episode is neither the helper's root nor its parent: no releaser is named
        assert held == [("awaited_helpers_paused", ["child-run"], "held", "1 helper it waits for is paused")], held
    else:  # never inferred from every paused member of the work
        assert held == [], work["tails"]


def test_a_warm_sleep_of_the_root_names_its_held_child_and_that_ouroboros_resumes_it(tmp_path, monkeypatch):
    """The real warm park (``owner_wait.checkpoint_owner_wait``, the supervisor's ``handle_owner_wait``) of the
    resumed root on its own held child: the root holds the child's request door, so its mind is named."""
    import time

    from ouroboros import model_sleep, owner_wait
    from ouroboros.tools.control_task_results import _await_messages
    from supervisor.worker_owner_wait import handle_owner_wait
    from tests._work_projection_shared import state_fields

    tree = _held_tree(tmp_path, monkeypatch)
    root = _root_ctx(tmp_path)
    shown = json.loads(_await_messages(root, mode="warm", tasks=["child-run"]))
    assert shown["reason"] == "dependency_held" and shown["held"][0]["door"] == "resume_child_task"
    _deliver(root, _published(root))
    assert json.loads(_await_messages(root, mode="warm", tasks=["child-run"]))["reason"] == "sleep_armed"
    meta = tree.workers.RUNNING["root-1"]
    root.task_attempt = meta["attempt"]
    checkpoint = owner_wait.checkpoint_owner_wait(root, [{"role": "user", "content": "wait"}], {}, {}, 1, [], set())
    # The clock fields the loop carries into its park: the sleep's start (``wait_after_tools``) and the paused
    # total the resumed loop adopted from its handoff (no model wait clock runs in this harness).
    model_sleep.begin(root)
    checkpoint["sleep_started_at"] = root._model_sleep_started
    checkpoint["budget_paused_sec"] = float(meta.get("budget_paused_sec") or 0.0)
    tree.workers.WORKERS[0] = SimpleNamespace(wid=0, proc=SimpleNamespace(pid=77, is_alive=lambda: True),
                                              busy_task_id="root-1", reaping=False, active_capacity=True,
                                              in_q=SimpleNamespace(put=lambda _command: None))
    meta["worker_id"], meta["started_at"] = 0, time.time() - 10
    handle_owner_wait({"task_id": "root-1", "wait_id": checkpoint["wait_id"], "task_attempt": meta["attempt"],
                       "worker_id": 0, "pid": 77, "phase": "park", "checkpoint": checkpoint},
                      SimpleNamespace(RUNNING=tree.workers.RUNNING, WORKERS=tree.workers.WORKERS, DRIVE_ROOT=tmp_path))
    assert meta["owner_wait"]["state"] == "waiting" and meta["owner_wait"]["sleep"]["tasks"] == ["child-run"]
    [work] = [work for work in state_fields(tmp_path)["works"] if work["episode"]["task_id"] == "root-1"]
    assert (work["activity"], work["because"][0]["code"], work["turn"]) == ("waiting", "sleeping", [])
    assert [(tail["code"], tail["ids"], tail["text"]) for tail in work["tails"] if tail["kind"] == "helper"] == [
        ("awaited_helpers_mind", ["child-run"], "1 helper it waits for is paused until Ouroboros resumes it")]


def test_an_unreadable_root_resume_basis_is_said_waited_on_once_and_its_recovery_is_shown_again(tmp_path, monkeypatch):
    """The identity's root basis is unreadable while the child's own stop reads: the proven stop stands, the
    projection says the basis is unknown, one showing then a deliberate wait holds, and the real basis, once
    it reads again, is new information. No fail-closed showing, retry, Resume or acknowledgement gate."""
    from ouroboros import budget_pause
    from ouroboros.tools.control_task_results import _await_messages

    _held_tree(tmp_path, monkeypatch)
    readable = _read(tmp_path, "child-run")["current_stop"]
    assert "resume_basis" not in readable
    real = budget_pause.budget_pause_row

    def unreadable(root, task_id):
        if task_id == "root-1":
            raise OSError("the root's control row is unreadable")
        return real(root, task_id)

    root, kept = _root_ctx(tmp_path), ("state", "cause", "scope", "code", "release")
    with monkeypatch.context() as patch:
        patch.setattr(budget_pause, "budget_pause_row", unreadable)
        stop = _read(tmp_path, "child-run")["current_stop"]
        assert {key: stop[key] for key in kept} == {key: readable[key] for key in kept}, "the child's stop stands"
        assert stop["resume_basis"] == "unreadable", "the identity's uncertainty is said, not erased"
        assert stop["identity"] != readable["identity"]
        assert _read(tmp_path, "child-run")["current_stop"]["identity"] == stop["identity"], "one stable token"
        shown = json.loads(_await_messages(root, mode="warm", tasks=["child-run"]))
        assert shown["reason"] == "dependency_held"
        assert (shown["held"][0]["resume_basis"], shown["held"][0]["door"]) == ("unreadable", "resume_child_task")
        _deliver(root, _published(root))
        assert json.loads(_await_messages(root, mode="warm", tasks=["child-run"]))["reason"] == "sleep_armed", \
            "shown once with its uncertainty, the deliberate wait then stands"
        root._model_sleep = root._owner_wait_requested = None
    renewed = json.loads(_await_messages(root, mode="warm", tasks=["child-run"]))
    assert renewed["reason"] == "dependency_held", "the recovered real generation is new information"
    assert "resume_basis" not in renewed["held"][0] and renewed["held"][0]["identity"] == readable["identity"]
    _deliver(root, _published(root))
    assert json.loads(_await_messages(root, mode="warm", tasks=["child-run"]))["reason"] == "sleep_armed"


def _section(text, header):
    """The rows under one bracketed header of the absorption reminder."""
    if header not in text:
        return []
    return [line for line in text.split(header, 1)[1].split("\n\n[", 1)[0].splitlines() if line.startswith("- ")]


_HELD_HEADER, _RUNNING_HEADER = "[HELD BY A CURRENT STOP", "[STILL RUNNING"


def test_the_absorption_reminder_names_each_held_child_held_with_the_stop_its_parent_may_act_on(
        tmp_path, monkeypatch):
    """Astra's R2 advisory, after an actual root-only Resume: the reminder the root reads before finalizing
    (``loop_delivery``'s consumer) lists each retained child its current stop holds apart from what still runs,
    with that stop and its request door, as ``get_task_result`` shows it. When it fires and what it absorbs stay."""
    from ouroboros.loop import _compute_subagent_handoff
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.control_task_results import _get_task_result

    _held_tree(tmp_path, monkeypatch)
    write_task_result(tmp_path, "child-plain", STATUS_RUNNING, chat_id=0, parent_task_id="root-1",
                      root_task_id="root-1", delegation_role="subagent")  # an ordinary child that simply runs
    ctx = SimpleNamespace(task_metadata={"budget_drive_root": str(tmp_path), "root_task_id": "root-1"},
                          budget_drive_root=str(tmp_path), drive_root=str(tmp_path), task_id="root-1",
                          _subagent_handoff_signature="")
    text = _compute_subagent_handoff(SimpleNamespace(_ctx=ctx), tmp_path, "root-1", "done")
    held, running = _section(text, _HELD_HEADER), _section(text, _RUNNING_HEADER)
    assert [row.split(":", 1)[0] for row in held] == ["- child-run", "- child-new"]
    assert [row.split(":", 1)[0] for row in running] == ["- child-plain"], "no held child reads as running"
    assert "current_stop" not in running[0], "a child no stop holds says nothing more"
    reader = _outcome(_get_task_result(_root_ctx(tmp_path), "child-run"))["current_stop"]
    shown = json.loads(held[0].split("current_stop=", 1)[1].rsplit(", child_result_sha256=", 1)[0])
    assert shown == reader and shown["door"] == "resume_child_task", "the same stop and door the reader shows"
    assert "scheduled" in held[0], "the queue status stays what it is"
    assert _compute_subagent_handoff(SimpleNamespace(_ctx=ctx), tmp_path, "root-1", "done") == "", \
        "fires once per change, as before"


def test_the_absorption_reminder_names_a_door_only_where_the_parent_may_ask_and_absorbs_as_before():
    """Only a held member stop the mind may release offers its door; Panic's, a tree's and a stale queue's do
    not. A stop that settles a still running child rides beside it there; a finished child is absorbed as before."""
    from ouroboros.task_status import format_subagent_absorption_message

    def child(task_id, status="scheduled", **stop):
        return {"task_id": task_id, "parent_task_id": "P", "root_task_id": "P", "status": status, "result": "",
                **({"current_stop": {"state": "held", "scope": "member", "release": "mind", "identity": task_id,
                                     **stop}} if stop else {})}

    text = format_subagent_absorption_message([
        child("door", cause="awaiting_selection"), child("panic", release="owner", cause="panic"),
        child("tree", scope="tree", cause="pause"), child("stale", queue_freshness="stale"),
        child("settling", status="running", state="pausing"), child("plain", status="running"),
        {**child("done", status="completed"), "result": "the finished work"},
    ], parent_task_id="P")
    held = {row.split(":", 1)[0][2:]: row for row in _section(text, _HELD_HEADER)}
    running = {row.split(":", 1)[0][2:]: row for row in _section(text, _RUNNING_HEADER)}
    assert list(held) == ["door", "panic", "tree", "stale"] and list(running) == ["settling", "plain"]
    assert [name for name, row in held.items() if '"door": "resume_child_task"' in row] == ["door"]
    assert '"state": "pausing"' in running["settling"] and "current_stop" not in running["plain"]
    assert "## child done (" in text and "the finished work" in text
