"""The work projection of a settling owner Pause, read through its real producers.

The census (``gateway/state.py``), the members' park (``budget_pause.enter_owner_pause`` and the
supervisor's ``install_exact_budget_pause``) and the stop observation each Pause records drive
``work_projection.works_snapshot`` directly: only what still executes is said as working, and a run
whose stop nobody could confirm is said as unconfirmed, never as still working.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_owner_pause import _live_tree, _owner_park

_OWNER = {"actor": "owner", "text": "", "at": "2026-10-20T00:43:00Z"}


def _works(root):
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.gateway.work_projection import works_snapshot

    raw, owners = {}, []
    _chat_activities_snapshot_safe(root, work_facts=raw, decision_owners=owners)
    return works_snapshot(root, owners, raw)["works"]


def test_a_warm_sleep_waits_only_on_its_current_attempt_and_a_pause_still_wins(tmp_path, monkeypatch):
    from ouroboros import owner_pause, owner_wait
    from ouroboros.task_results import write_task_result

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "sleeper", "type": "task", "chat_id": 7, "root_task_id": "sleeper"}
    write_task_result(tmp_path, "sleeper", "running", chat_id=7)
    wait = owner_wait.set_owner_wait(tmp_path, "sleeper", {
        "state": "waiting", "reason": "sleep", "sleep": {"mode": "warm"}, "task_attempt": 1})
    workers.RUNNING["sleeper"] = {"task": task, "attempt": 1, "owner_wait": wait}
    [work] = _works(tmp_path)
    assert (work["activity"], work["live"], work["release"], work["turn"]) == ("waiting", False, "self", [])
    assert work["because"][0]["code"] == "sleeping"
    # Hurry and Wrap up wake a warm sleep (model_sleep), so the sleeper keeps them beside Pause.
    assert {"wrap_up", "hurry", "pause"} <= {control["action"] for control in work["controls"]}
    assert "resume" not in {control["action"] for control in work["controls"]}

    # The durable result still holds the older wait; the live worker's new observation wins.
    for changed in ({"state": "resumed"}, {"task_attempt": 2}):
        workers.RUNNING["sleeper"]["owner_wait"] = {**wait, **changed}
        [work] = _works(tmp_path)
        assert (work["activity"], work["live"]) == ("working", True)
        assert work["because"] == []

    workers.RUNNING["sleeper"]["owner_wait"] = wait
    fence, _ = owner_pause.install_fence(tmp_path, "sleeper", request_id="pause-sleeper", by=_OWNER)
    owner_pause.set_fence_state(tmp_path, "sleeper", fence_id=fence["fence_id"], state="paused")
    [work] = _works(tmp_path)
    assert (work["activity"], work["live"]) == ("paused", False)
    assert work["because"][0]["code"] == "paused_by_owner"
    assert "resume" in {control["action"] for control in work["controls"]}
    assert not {"wrap_up", "hurry"} & {control["action"] for control in work["controls"]}


def test_a_sleeping_then_queued_child_keeps_its_own_external_run(tmp_path, monkeypatch):
    from ouroboros import delegate_custody

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _live_tree(tmp_path, workers)
    run = {"run_id": "native-1", "task_id": "child-run", "root_task_id": "root-1"}
    assert delegate_custody.emit(tmp_path, delegate_custody.STARTED, run)
    worker = workers.RUNNING["child-run"]
    worker["owner_wait"] = {"state": "waiting", "reason": "sleep", "task_attempt": 1,
                            "sleep": {"mode": "warm", "runs": ["native-1"]}}

    def check(activity, reason, open_run=True):
        [work] = _works(tmp_path)
        states = work["members"]["states"]
        child = states["child-run"]
        assert (child["activity"], child["live"], child["call"], child["mark"]) == (activity, False, "", "still")
        assert [entry["code"] for entry in child["because"]] == [reason]
        if reason == "sleeping":
            assert child["because"][0]["wake"] == [{"kind": "runs", "count": 1}]
        assert [(t["code"], t["ids"]) for t in child["tails"]] == (
            [("external_runs_unconfirmed", ["native-1"])] if open_run else [])
        assert [rid for tail in work["tails"] for rid in tail["ids"]] == (["native-1"] if open_run else [])
        assert states["root-1"]["tails"] == []
        assert (states["child-new"]["activity"], states["child-new"]["live"], states["child-new"]["tails"]) == (
            "queued", False, [])
        assert work["activity"] == "working" and {c["target"] for c in work["controls"]} == {"root-1"}

    check("waiting", "sleeping")
    task = workers.RUNNING.pop("child-run")["task"]
    task["_budget_pause"] = {"rail": "model_sleep", "reason": "sleep", "exact_continuation": True,
                             "sleep": {"mode": "cold", "runs": ["native-1"]}}
    workers.PENDING.append(task)
    check("waiting", "sleeping")
    task["_budget_pause_resume"] = {"selected_by": "sleep_wake", "pause": task.pop("_budget_pause"),
                                    "external_runs": {"custody_read": "ok", "runs": [{**run, "state": "running"}]}}
    check("queued", "queued")
    # The old observation on the wake grant cannot outlive the engine's terminal observation.
    assert delegate_custody.emit(tmp_path, delegate_custody.TERMINAL_OBSERVED, {
        **run, "state": "succeeded", "observation_seq": 1, "source_ref": {"sha256": "0" * 64}})
    check("queued", "queued", open_run=False)


def test_nested_runs_need_a_current_observation_of_their_own_executor(tmp_path, monkeypatch):
    from ouroboros import budget_pause, delegate_collection, delegate_custody, delegate_feed, external_runs
    from ouroboros.gateway.work_projection import unconfirmed_stops

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _live_tree(tmp_path, workers)
    a = {"task_id": "child-run", "run_id": "job-a", "invocation_id": "inv-a", "root_task_id": "root-1"}
    b = {"task_id": "child-new", "run_id": "run-b", "invocation_id": "inv-b", "root_task_id": "root-1"}
    workers.RUNNING["root-1"]["owner_wait"] = {
        "state": "waiting", "reason": "sleep", "task_attempt": 1, "sleep": {"mode": "warm"}}
    for run in (a, b):  # a jobId-only receipt uses that id as its custody run_id
        assert delegate_custody.emit(tmp_path, delegate_custody.STARTED, run)
    workers.RUNNING["child-run"]["owner_wait"] = {
        "state": "waiting", "reason": "sleep", "task_attempt": 1, "sleep": {"mode": "warm", "runs": ["job-a"]}}
    failed = external_runs.observe_task_runs(tmp_path, "child-run", request_stop=False, read_error="unavailable")
    assert failed["custody_read"] == "failed" and failed["runs"] == []
    budget_pause.set_budget_pause(tmp_path, "child-run", {"state": budget_pause.STATE_PAUSED,
                                                         "pause_id": "a", "external_runs": failed})
    assert unconfirmed_stops(tmp_path, a["task_id"]) is None
    assert unconfirmed_stops(tmp_path, b["task_id"]) == frozenset()
    now = 100.0
    monkeypatch.setattr(delegate_feed, "time", SimpleNamespace(monotonic=lambda: now))
    seen = {"run-b": {**b, "run_state": "running", "answered": True, "at": now}}
    monkeypatch.setattr(delegate_collection, "body_runtime", lambda root: SimpleNamespace(
        observation=lambda rid: seen.get(rid, {})))

    def check(a_state, b_state):
        [work] = _works(tmp_path)
        states = work["members"]["states"]
        for run, state, reason in ((a, a_state, "sleeping"), (b, b_state, "queued")):
            member = states[run["task_id"]]
            assert not member["live"] and member["because"][0]["code"] == reason
            assert [(tail["state"], tail["ids"]) for tail in member["tails"]] == [(state, [run["run_id"]])]
        assert not states["root-1"]["tails"]
        assert {rid: tail["state"] for tail in work["tails"] for rid in tail["ids"]} == {
            a["run_id"]: a_state, b["run_id"]: b_state}
        assert work["live"] is ("live" in (a_state, b_state))
        assert [c for c in work["controls"] if c["action"] == "stop"] == [
            {"action": "stop", "target": "root-1", "scope": "cascade"}]

    check("unknown", "live")
    fresh = {**a, "answered": True, "run_state": "running", "at": now}
    for mismatch in ({"task_id": "child-new"}, {"run_id": "run-b"}, {"invocation_id": "old"},
                     {"answered": False}, {"at": now - delegate_feed._OBSERVATION_CURRENT_SEC - 1}):
        seen["job-a"] = {**fresh, **mismatch}
        check("unknown", "live")
    seen["job-a"] = {**fresh, "run_state": "queued"}
    seen["run-b"]["waiting_on_user"] = True
    check("open", "open")
    seen.pop("job-a")
    check("unknown", "open")
    seen["job-a"] = fresh
    seen["run-b"]["async_questions_only"] = True
    check("live", "live")


def test_a_settling_pause_says_what_still_executes_and_a_stop_nobody_confirmed(tmp_path, monkeypatch):
    from ouroboros import budget_pause, delegate_custody
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _live_tree(tmp_path, workers)  # a RUNNING root and child, and a child that never started
    delegate_custody.emit(tmp_path, delegate_custody.STARTED,
                          {"run_id": "run-1", "task_id": "child-run", "root_task_id": "root-1"})
    assert request_owner_pause("root-1", request_id="p", by=_OWNER)["ok"]
    [work] = _works(tmp_path)
    # The root and the running child finish their step; the child that never started is queued, not working.
    assert (work["activity"], work["transition"], work["live"]) == ("working", "pausing", True)
    assert [(tail["code"], tail["state"], tail["ids"]) for tail in work["tails"]] == [
        ("helpers_running", "live", ["child-run"]), ("external_runs_unconfirmed", "unknown", ["run-1"])]

    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=lambda: None,
                          persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
    # The child parks while the harness gateway cannot confirm its run's stop, then the root parks: the Pause
    # still settles (that run is unresolved), and nothing executes any more.
    unconfirmed = [{"run_id": "run-1", "state": "stop_unknown", "stop_outcome": "not_issued_gateway_unavailable"}]
    for task_id, runs in (("child-run", unconfirmed), ("root-1", [])):
        saved = _owner_park(tmp_path, monkeypatch, task_id, runs)
        install_exact_budget_pause(ctx, task_id, budget_pause.exact_pause_marker(saved)["checkpoint"])
    assert not workers.RUNNING
    [work] = _works(tmp_path)
    assert (work["activity"], work["transition"], work["live"]) == ("paused", "none", False)
    assert [(tail["code"], tail["state"], tail["ids"]) for tail in work["tails"]] == [
        ("external_runs_unconfirmed", "unknown", ["run-1"])]
    states = work["members"]["states"]
    assert (states["child-run"]["activity"], states["child-run"]["live"], states["child-run"]["tails"]) == (
        "paused", False, work["tails"])
    assert states["root-1"]["tails"] == states["child-new"]["tails"] == []
    assert {"stop", "resume"} <= {control["action"] for control in work["controls"]}, "its Stop still reaches that run"


def test_a_warm_park_executes_nothing_and_its_unconfirmed_stop_stays_unconfirmed(tmp_path, monkeypatch):
    """A helper whose launched review the Pause lets finish parks warm: it keeps its worker while its stack waits,
    and the stop of its own run it could not confirm is recorded on its stored result, not on a pause row."""
    from ouroboros import delegate_custody, external_runs, owner_pause, owner_wait, peer_roster
    from supervisor.owner_pause_control import request_owner_pause, warm_paused_member

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _live_tree(tmp_path, workers)
    assert request_owner_pause("root-1", request_id="p", by=_OWNER)["ok"]
    for run_id, source in (("own-run", "delegated_subagent"), ("review-run", "review_substrate")):
        delegate_custody.emit(tmp_path, delegate_custody.STARTED,
                              {"run_id": run_id, "task_id": "child-run", "root_task_id": "root-1", "source": source})
    child, limit = _loop_ctx(tmp_path, "child-run")
    child.root_task_id = "root-1"

    def retained(ctx, checkpoint):  # the worker keeps the stack; only its blocking wait is stood in for
        wait = owner_wait.set_owner_wait(tmp_path, "child-run", {**checkpoint, "state": "waiting"})
        workers.RUNNING["child-run"]["owner_wait"] = dict(wait)
        return "retained"

    child.owner_wait_callback = retained
    observed = {"custody_read": "ok", "runs": [
        {"run_id": "own-run", "state": "stop_unknown", "stop_outcome": "not_issued_gateway_unavailable"},
        {"run_id": "review-run", "state": "running", "source": "review_substrate"}]}
    monkeypatch.setattr(external_runs, "observe_task_runs", lambda *args, **kwargs: observed)
    owner_wait.park_owner_pause_warm(limit, child, fence=owner_pause.read_fence(tmp_path, "root-1"),
                                     detached=[{"operation_id": "review-op"}])
    assert warm_paused_member(workers.RUNNING["child-run"])
    [work] = _works(tmp_path)
    # The root still finishes its step. The warm helper is not said as working: its review finishes beside the
    # work, and its own run is the one nobody can confirm.
    assert (work["activity"], work["transition"], work["live"]) == ("working", "pausing", True)
    assert [(tail["code"], tail["state"], tail["ids"]) for tail in work["tails"]] == [
        ("review_work_finishing", "live", ["review-run"]), ("external_runs_unconfirmed", "unknown", ["own-run"])]
    states = work["members"]["states"]
    assert states["child-run"]["tails"] == work["tails"]
    assert states["root-1"]["tails"] == states["child-new"]["tails"] == []
    # Ouroboros's own roster reads the same tree the same way.
    queue.persist_queue_snapshot(reason="warm-park")
    tree = peer_roster._live_trees(tmp_path, peer_roster._load_queue_snapshot(tmp_path))["root-1"]
    assert tree["running"] == [] and tree["pending"] == 2
    tails = peer_roster._tree_tails(tmp_path, tree, owner_pause.read_fence(tmp_path, "root-1"))
    assert [(tail["code"], tail["state"]) for tail in tails] == [
        ("review_work_finishing", "live"), ("external_runs_unconfirmed", "unknown")]

    # The root parks cold and the helper's custody review settles, while a native review it already sent still runs:
    # its work says that review and stays live; once it ends, only the unconfirmed run remains.
    from ouroboros import budget_pause, review_operation
    from supervisor.events_budget import install_exact_budget_pause

    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=lambda: None,
                          persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
    saved = _owner_park(tmp_path, monkeypatch, "root-1", [])
    install_exact_budget_pause(ctx, "root-1", budget_pause.exact_pause_marker(saved)["checkpoint"])
    assert delegate_custody.emit(tmp_path, delegate_custody.SETTLED,
                                 {"run_id": "review-run", "task_id": "child-run", "state": "succeeded"})
    native = {"child-run"}
    monkeypatch.setattr(review_operation, "task_has_live_review_operation",
                        lambda root, task_id, sent_only=False, **_: task_id in native)
    [work] = _works(tmp_path)
    assert work["live"] is True
    assert [(tail["code"], tail["state"]) for tail in work["tails"]] == [
        ("review_work_finishing", "live"), ("external_runs_unconfirmed", "unknown")]
    native.clear()
    [work] = _works(tmp_path)
    assert work["live"] is False
    assert [(tail["code"], tail["state"]) for tail in work["tails"]] == [("external_runs_unconfirmed", "unknown")]


def test_an_answered_roots_late_work_under_a_settling_pause_executes_and_keeps_its_identity(tmp_path, monkeypatch):
    """An answered Continue episode whose post-answer synthesis is still in flight when the owner pauses: the census
    lists it as settling and running, and its work keeps the id and name of the work it continues."""
    from ouroboros import post_task_checkpoint
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "ans", "completed", chat_id=0, root_phase_checkpoint={"post_task_synthesis": "pending_once"},
                      metadata={"continuation": {"predecessor_task_id": "prev", "work_id": "prev",
                                                 "work_name": "Whole work"}})
    monkeypatch.setattr(post_task_checkpoint, "post_task_synthesis_in_flight", lambda root, task_id: task_id == "ans")
    assert request_owner_pause("ans", request_id="p", by=_OWNER)["ok"]
    [work] = _works(tmp_path)
    assert (work["work_id"], work["name"]) == ("prev", "Whole work")
    assert (work["activity"], work["transition"], work["live"]) == ("working", "pausing", True)
