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
        ("helpers_running", "live", ["child-run"]), ("delegated_running", "live", ["run-1"])]

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
