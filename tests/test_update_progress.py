"""Progress belongs to the running executor; its last attempt outlives the process (#1818)."""

import asyncio
import inspect
import json
import threading
from types import SimpleNamespace

import pytest
from starlette.responses import JSONResponse

from ouroboros.gateway import control, update_progress as progress


@pytest.fixture(autouse=True)
def isolated_progress(monkeypatch):
    """A disposable retained record behind the same load/update seam as state.json."""
    from supervisor import state as sup_state, update_merge

    real_state = SimpleNamespace(read_state=sup_state.read_state, update_state=sup_state.update_state)
    saved: dict = {}
    lock = threading.Lock()

    def update_state(mutator, **_kwargs):
        with lock:
            mutator(saved)
            return dict(saved)

    state = SimpleNamespace(quality="current")
    monkeypatch.setattr(sup_state, "read_state",
                        lambda *_a, **_k: SimpleNamespace(quality=state.quality, values=dict(saved)))
    monkeypatch.setattr(sup_state, "update_state", update_state)
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {})
    monkeypatch.setattr(progress, "_current", {})
    monkeypatch.setattr(progress, "_boot_settled", threading.Event())
    server = SimpleNamespace(generation="server-a")
    monkeypatch.setattr(progress, "_generation", lambda: server.generation)
    notices = []
    monkeypatch.setattr(progress, "_notify", lambda: notices.append(progress.snapshot({})))
    return SimpleNamespace(saved=saved, notices=notices, server=server, state=state, real_state=real_state)


def _next_server(isolated, monkeypatch, generation="server-b"):
    """The next process over the same data root: nothing in memory, another generation."""
    monkeypatch.setattr(progress, "_current", {})
    isolated.server.generation = generation


def _plan(**extra):
    return {"available": True, "kind": "clean", "base_sha": "a" * 40, "target_sha": "b" * 40,
            "target_ref": "official", "update_channel": "stable", "local_dirty_count": 0, **extra}


def _restart_requested_attempt(attempt="tx-1"):
    @progress.observe_result
    def execute():
        progress.begin(base_sha="a" * 40, target_sha="b" * 40)
        progress.link({"attempt_id": attempt}, keep_branch="local-keep-20261021_032000-a1b2c3")
        progress.advance("restart_requested")
        return JSONResponse({"status": "ok", "restarting": True})

    execute()


def _seen(outcome, attempt="tx-1", commit="c" * 40):
    return {"observed": {"outcome": outcome, "attempt_id": attempt, "commit": commit}}


def _boot(monkeypatch, result, *, baseline="c" * 40, bootstrap_ok=True):
    from ouroboros import server_process

    monkeypatch.setattr(server_process, "server_source_baseline", lambda: baseline)
    seen = []

    def finalize(*, supervisor_ready):
        seen.append(supervisor_ready)
        return result

    assert progress.observe_boot(finalize, ready=True, bootstrap_ok=bootstrap_ok) is result
    assert seen == [True] and progress._boot_settled.is_set()
    return progress.snapshot({})


def test_progress_is_passive_and_stage_notifications_are_not_boot_ready(isolated_progress):
    assert progress.snapshot({}) == {}
    progress.begin()
    progress.advance("stopping_workers")
    progress.advance("stopping_workers")
    view = progress.snapshot({})
    assert view["active"] and view["stage"] == "stopping_workers"
    assert view["operation_id"] and view["generation"] and view["stage_started_at"]
    assert "owner_thread" not in view and "owner_thread" not in isolated_progress.saved[progress.STATE_KEY]
    view["stage"] = "changed by reader"
    assert progress.snapshot({})["stage"] == "stopping_workers"
    assert len(isolated_progress.notices) == 2


@pytest.mark.parametrize("body,code,result,outcome", [
    ({"error": "writers remain", "restart_required": True}, 409, "failed", "not_applied"),
    ({"error": "rollback failed", "rolled_back": False, "restart_required": True}, 500, "failed", "not_applied"),
    ({"status": "restart_required", "error": "callback unavailable"}, 200, "restart_required", ""),
    ({"status": "assisted_started", "task_id": "resolver"}, 200, "assisted_started", ""),
    ({"status": "ok", "restarting": True}, 200, "restart_requested", ""),
])
def test_executor_result_survives_panel_reopen(isolated_progress, body, code, result, outcome):
    @progress.observe_result
    def execute():
        progress.begin()
        progress.link({"attempt_id": "tx-1"})
        progress.advance("checking")
        return JSONResponse(body, status_code=code)

    response = execute()
    view = progress.snapshot({"attempt_id": "tx-1"})
    assert response.status_code == code
    assert view["result"] == result and not view["active"] and view["outcome"] == outcome
    assert view["error"] == body.get("error", "")
    assert view["restart_required"] == (result == "restart_required" or body.get("restart_required", False))
    assert isolated_progress.saved[progress.STATE_KEY]["result"] == result


@pytest.mark.serial
def test_cancelled_http_await_does_not_finish_the_running_executor(monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    @progress.observe_result
    def execute(*_args, **_kwargs):
        progress.begin()
        progress.advance("stopping_workers")
        entered.set()
        try:
            assert release.wait(5)
            return JSONResponse({"status": "assisted_started"})
        finally:
            finished.set()

    monkeypatch.setattr(control, "_apply_smart_update_fenced", execute)

    async def exercise():
        task = asyncio.create_task(control._apply_smart_update(None, expected_base_sha="a", expected_target_sha="b"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert progress.snapshot()["active"] is True
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 5)

    asyncio.run(exercise())
    assert progress.snapshot()["result"] == "assisted_started"
    assert progress.snapshot()["active"] is False


@pytest.mark.serial
def test_rejected_competing_executor_cannot_clear_or_advance_owner():
    progress.begin()
    progress.advance("stopping_workers")
    owner = progress.snapshot()

    @progress.observe_result
    def refused():
        progress.advance("applying")
        return JSONResponse({"error": "update lock is held"}, status_code=409)

    thread = threading.Thread(target=refused)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    assert progress.snapshot() == owner


def test_telemetry_failure_does_not_change_update_outcome(monkeypatch):
    def fail():
        raise RuntimeError("socket unavailable")

    monkeypatch.setattr(progress, "_notify", fail)

    @progress.observe_result
    def execute():
        progress.begin()
        progress.advance("applying")
        return JSONResponse({"status": "ok", "restarting": True})

    assert execute().status_code == 200
    assert progress.snapshot()["result"] == "restart_requested"


def test_an_unwritable_record_changes_no_update_outcome(monkeypatch):
    from supervisor import state as sup_state

    def refuse(*_args, **_kwargs):
        raise sup_state.StateUnavailable("lock_timeout", "fixture")

    monkeypatch.setattr(sup_state, "update_state", refuse)

    @progress.observe_result
    def execute():
        progress.begin()
        return JSONResponse({"status": "ok", "restarting": True})

    assert execute().status_code == 200
    assert progress.snapshot({})["result"] == "restart_requested"
    assert progress.snapshot({})["retention_unconfirmed"] is True


def test_a_restarted_server_reads_the_attempt_and_names_a_missing_conclusion(isolated_progress, monkeypatch):
    _restart_requested_attempt()
    _next_server(isolated_progress, monkeypatch)
    waiting = progress.snapshot({})
    assert (waiting["active"], waiting["outcome"], waiting["current_generation"]) == (False, "", False)
    assert waiting["base_sha"] == "a" * 40 and waiting["tx_attempt_id"] == "tx-1"
    progress._boot_settled.set()
    assert progress.snapshot({"attempt_id": "tx-1"})["outcome"] == ""  # its transaction still decides
    unknown = progress.snapshot({})
    assert (unknown["outcome"], unknown["reason"]) == ("unknown", "no_completion_record")


def test_an_interrupted_executor_is_never_shown_running_by_a_later_server(isolated_progress, monkeypatch):
    progress.begin()
    progress.advance("stopping_workers")  # the process died here: _finish never ran
    _next_server(isolated_progress, monkeypatch)
    progress._boot_settled.set()
    view = progress.snapshot({})
    assert view["active"] is False and view["outcome"] == "unknown"





@pytest.mark.parametrize("restart_required", [False, True])
def test_explicit_check_refreshes_availability_and_retains_the_last_attempt(isolated_progress, monkeypatch,
                                                                           restart_required):
    @progress.observe_result
    def failed():
        progress.begin()
        return JSONResponse({"error": "stopped", "restart_required": restart_required}, status_code=409)

    failed()
    before = progress.snapshot({})
    monkeypatch.setattr(control, "_managed_update_payload", lambda **_: {"check_ok": True})
    asyncio.run(control.api_update_check(None))
    assert progress.snapshot({}) == before and before["outcome"] == "not_applied"
    assert isolated_progress.notices[-1] == before  # other clients re-read; nothing was cleared


def test_explicit_check_does_not_clear_a_running_executor(monkeypatch):
    progress.begin()
    owner = progress.snapshot({})
    monkeypatch.setattr(control, "_managed_update_payload", lambda **_: {"check_ok": True})
    asyncio.run(control.api_update_check(None))
    assert progress.snapshot({}) == owner





def test_a_drain_refusal_is_not_applied_stopped_nobody_and_retries_nothing(isolated_progress, monkeypatch):
    from supervisor import update_merge, workers, workspace_preparation

    locks, kills = [], []
    monkeypatch.setattr(update_merge, "plan_managed_update_merge", lambda **_kw: _plan())
    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda: locks.append(1) or object())
    monkeypatch.setattr(update_merge, "release_update_lock", lambda _fh: None)
    monkeypatch.setattr(workspace_preparation, "pause_admission", lambda: None)
    monkeypatch.setattr(workspace_preparation, "resume_admission", lambda: None)
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda *_a: None)
    monkeypatch.setattr(workers, "open_repo_writer_admission", lambda: None)
    monkeypatch.setattr(workers, "drain_repo_writers", lambda: ["Main conversation"])
    monkeypatch.setattr(workers, "kill_workers_for_update", lambda **_kw: kills.append(1) or [])

    monkeypatch.setattr(control, "_index_lock_precondition", lambda: None)  # synthetic repo
    response = control._apply_smart_update_fenced(None, expected_base_sha="a" * 40, expected_target_sha="b" * 40)

    view = progress.snapshot({})
    assert response.status_code == 409 and locks == [1] and kills == []
    assert (view["outcome"], view["work_stop"], view["restart_required"]) == ("not_applied", "none", False)
    assert (view["reason"], view["blockers"]) == ("update_writer_fence_blocked", ["active:Main conversation"])
    assert (view["base_sha"], view["target_sha"]) == ("a" * 40, "b" * 40)
    assert isolated_progress.saved[progress.STATE_KEY]["outcome"] == "not_applied"





def test_an_attempt_past_teardown_records_that_it_reached_teardown(isolated_progress):
    @progress.observe_result
    def execute():
        progress.begin()
        progress.advance("stopping_workers")
        return JSONResponse({"error": "writers remain", "reason": "update_writer_fence_blocked",
                             "blockers": ["worker:42"], "restart_required": True}, status_code=409)

    execute()
    view = progress.snapshot({})
    assert (view["outcome"], view["work_stop"], view["blockers"]) == ("not_applied", "invoked", ["worker:42"])








def test_a_late_executor_write_cannot_downgrade_a_boot_conclusion(isolated_progress, monkeypatch):
    _restart_requested_attempt()
    late = {key: value for key, value in progress._current.items() if key != "owner_thread"}
    _next_server(isolated_progress, monkeypatch)
    _boot(monkeypatch, {"finalized": True, **_seen("applied")})
    progress._persist({**late, "result": "failed", "outcome": "failed"})
    assert progress.snapshot({})["outcome"] == "applied"
    assert isolated_progress.saved[progress.STATE_KEY]["concluded_by"] == "boot"


@pytest.mark.parametrize("keep", ["local-keep-20261021_032000-a1b2c3", ""])
def test_replace_restart_names_only_the_keep_branch_it_made(keep):
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(request_restart=lambda **_kw: None)))
    body = json.loads(control._restart_response(request, strategy="replace", plan={}, keep_branch=keep).body)
    assert body["restarting"] is True and ("keep_branch" in body) is bool(keep)
    assert body.get("keep_branch", "") == keep

@pytest.mark.parametrize("quality", ["unavailable", "recovered_transient"])
def test_an_unreadable_state_is_a_named_unknown_not_an_absence(isolated_progress, quality):
    isolated_progress.state.quality = quality
    assert progress.snapshot({}) == {"operation_id": "state_unreadable", "active": False, "outcome": "unknown",
                                     "reason": "state_unreadable", "current_generation": False}
    isolated_progress.state.quality = "uninitialized"
    assert progress.snapshot({}) == {}


def test_a_failing_state_read_or_a_torn_value_is_a_named_unknown(isolated_progress, monkeypatch):
    from supervisor import state as sup_state

    isolated_progress.saved[progress.STATE_KEY] = "torn"
    assert progress.snapshot({})["reason"] == "record_unreadable"

    def broken(*_args, **_kwargs):
        raise OSError("EIO")

    monkeypatch.setattr(sup_state, "read_state", broken)
    assert progress.snapshot({})["reason"] == "state_unreadable"


def test_a_crashed_attempt_never_carries_a_stopped_nothing_claim(isolated_progress, monkeypatch):
    progress.begin()  # stored with work_stop "none"; the process dies before its conclusion
    _next_server(isolated_progress, monkeypatch)
    progress._boot_settled.set()
    view = progress.snapshot({})
    assert view["outcome"] == "unknown" and "work_stop" not in view


@pytest.mark.parametrize("stages,body,code,expected", [
    ((), {"error": "stopped"}, 500, "not_applied"),
    (("rolling_back",), {"error": "smart_update_exception", "rolled_back": True}, 409, "not_applied"),
    (("applying",), {"error": "boom"}, 500, "failed"),
    (("applying",), {"error": "unexpected"}, 409, "failed"),
    (("applying", "rolling_back"), {"error": "smoke", "rolled_back": True}, 409, "rolled_back"),
    (("applying",), {"error": "git refused", "reason": "replace_not_applied", "abandoned": True}, 409,
     "not_applied"),
])
def test_the_executor_concludes_only_what_its_own_steps_prove(isolated_progress, stages, body, code, expected):
    @progress.observe_result
    def execute():
        progress.begin()
        for stage in stages:
            progress.advance(stage)
        return JSONResponse(body, status_code=code)

    execute()
    assert progress.snapshot({})["outcome"] == expected


def test_each_lane_announces_its_repository_step_and_only_the_drain_names_active_writers():
    for function, step in ((control._apply_clean_merge_fenced, "apply_managed_merge_update(branch"),
                           (control._start_assisted_merge_fenced, "materialize_assisted_merge_live(branch"),
                           (control._apply_replace_recovery_fenced, "switch_or_reset(")):
        source = inspect.getsource(function)
        assert source.index('update_progress.advance("applying")') < source.index(step), function.__name__
    stop = inspect.getsource(control._stop_repo_writers)
    drained = stop.index('return [f"active:{label}" for label in blocked]')
    assert drained < stop.index("prepare_owner_wait_handoffs(") < stop.index('update_progress.advance("stopping_workers")')
    assert stop.index('update_progress.advance("stopping_workers")') < stop.index("kill_workers_for_update(")
    assert stop.index('update_progress.advance("stopping_services")') < stop.index("kill_all_services(")
    assert stop.count("active:") == 1, "only the drain refusal names an active writer"


def test_a_request_refused_before_admission_leaves_the_retained_attempt(isolated_progress, monkeypatch):
    from supervisor import update_merge

    _restart_requested_attempt()
    before = dict(isolated_progress.saved[progress.STATE_KEY])
    monkeypatch.setattr(update_merge, "plan_managed_update_merge", lambda **_kw: _plan(target_sha="c" * 40))
    monkeypatch.setattr(control, "_index_lock_precondition", lambda: None)  # synthetic repo
    refused = control._apply_smart_update_fenced(None, expected_base_sha="a" * 40, expected_target_sha="b" * 40)
    assert refused.status_code == 409 and json.loads(refused.body)["reason"] == "release_moved"
    assert isolated_progress.saved[progress.STATE_KEY] == before


def test_returned_refs_notes_and_blockers_stay_whole(isolated_progress):
    note = "local changes are present, so the stash was NOT auto-applied; " + "x" * 2000
    keep = "local-keep-20261021_032000-" + "a" * 80
    blockers = [f"active:conversation {index}" for index in range(12)]

    @progress.observe_result
    def execute():
        progress.begin()
        return JSONResponse({"error": "e" * 3000, "reason": "update_writer_fence_blocked", "blockers": blockers,
                             "stash_note": note, "keep_branch": keep}, status_code=409)

    execute()
    stored = isolated_progress.saved[progress.STATE_KEY]
    assert (stored["stash_note"], stored["keep_branch"], stored["blockers"]) == (note, keep, blockers)
    assert len(stored["error"]) == 3000


@pytest.mark.parametrize("result,bootstrap_ok,baseline,expected", [
    ({"finalized": True, **_seen("applied")}, True, "c" * 40, ("applied", "")),
    ({"finalized": True, **_seen("applied")}, False, "c" * 40, ("unknown", "startup_unconfirmed")),
    ({"finalized": True, **_seen("applied")}, True, "d" * 40, ("unknown", "running_source_mismatch")),
    ({"finalized": True, **_seen("rolled_back", commit="a" * 40)}, True, "a" * 40, ("rolled_back", "")),
    ({"finalized": True, **_seen("rolled_back", commit="a" * 40)}, True, "c" * 40, ("unknown", "running_source_mismatch")),
    ({"abandoned": True, **_seen("not_applied", commit="a" * 40)}, True, "a" * 40, ("not_applied", "")),
    ({"finalized": True}, True, "c" * 40, ("unknown", "no_completion_record")),
    ({"finalized": False, "rolled_back": True}, True, "a" * 40, ("unknown", "no_completion_record")),
    ({"finalized": True, **_seen("applied", attempt="tx-2")}, True, "c" * 40, ("unknown", "no_completion_record")),
])
def test_a_later_server_concludes_only_the_bound_attempt_on_its_served_commit(isolated_progress, monkeypatch,
                                                                              result, bootstrap_ok, baseline, expected):
    _restart_requested_attempt()
    _next_server(isolated_progress, monkeypatch)
    view = _boot(monkeypatch, result, baseline=baseline, bootstrap_ok=bootstrap_ok)
    assert (view["outcome"], view.get("reason", "")) == expected
    assert view["keep_branch"] == "local-keep-20261021_032000-a1b2c3" and "work_stop" not in view


def test_the_same_server_never_borrows_a_boot_conclusion(isolated_progress, monkeypatch):
    _restart_requested_attempt()
    same = _boot(monkeypatch, {"finalized": True, **_seen("applied")})
    assert (same["outcome"], same["current_generation"]) == ("", True)


@pytest.mark.parametrize('fields,outcome', [
    ({'stash_restore': {'context': 'boot_finalize'}}, 'applied'),
    ({'stash_restore': {'context': 'boot_stash_recovery'}}, 'not_applied'),
    ({'stash_restore': {'context': 'boot_diverged_abandon'}}, ''),
    ({'stash_restore': {'context': 'boot_restore_resume', 'evidence': {'interrupted': {'context': 'boot_stash_recovery'}}}}, 'not_applied'),
    ({'stash_restore': {'context': 'boot_restore_resume', 'evidence': {'interrupted': {'context': 'boot_diverged_abandon'}}}}, ''),
    ({'stash_restore': {'context': 'boot_restore_resume'}}, ''),
    ({'gate_blocked_reason': 'rollback_restart_pending'}, 'rolled_back'),
    ({'gate_blocked_reason': 'replace_abandon_tx_cleanup_failed'}, 'not_applied'),
    ({'gate_blocked_reason': 'stash_marker_cleanup_failed', 'gate_blocked_from_phase': 'pending_boot_smoke'}, 'applied'),
    ({'gate_blocked_reason': 'stash_marker_cleanup_failed', 'gate_blocked_from_phase': 'assisted_resolution'}, ''),
    ({'gate_blocked_from_phase': 'pending_boot_smoke'}, ''),
])
def test_cleanup_keeps_only_an_already_established_meaning(fields, outcome):
    from supervisor.update_merge import _cleanup_observed

    tx = {'attempt_id': 'bound', 'merge_commit': 'b' * 40, 'pre_update_sha': 'a' * 40, **fields}
    result = _cleanup_observed(tx)
    if not outcome:
        assert result == {}
    else:
        assert result['observed'] == {'attempt_id': 'bound', 'outcome': outcome,
                                      'commit': ('b' if outcome == 'applied' else 'a') * 40}


def test_a_later_successful_save_clears_only_the_live_retention_warning(isolated_progress, monkeypatch):
    from supervisor import state as sup_state

    with monkeypatch.context() as failure:
        failure.setattr(sup_state, 'update_state', lambda *_a, **_k: (_ for _ in ()).throw(OSError('unwritable')))
        progress.begin()
    assert progress.snapshot({})['retention_unconfirmed'] is True
    progress.link({'attempt_id': 'retained'})
    assert 'retention_unconfirmed' not in progress.snapshot({})
    assert 'retention_unconfirmed' not in isolated_progress.saved[progress.STATE_KEY]


@pytest.fixture
def retained_state(isolated_progress, tmp_path, monkeypatch):
    """Real state locking, primary/backup writes and reads, all under tmp_path."""
    from supervisor import state

    for name, path in {
        "DRIVE_ROOT": tmp_path,
        "STATE_PATH": tmp_path / "state" / "state.json",
        "STATE_LAST_GOOD_PATH": tmp_path / "state" / "state.last_good.json",
        "STATE_LOCK_PATH": tmp_path / "locks" / "state.lock",
    }.items():
        monkeypatch.setattr(state, name, path)
    monkeypatch.setattr(state, "read_state", isolated_progress.real_state.read_state)
    monkeypatch.setattr(state, "update_state", isolated_progress.real_state.update_state)
    state.save_state({progress.STATE_KEY: {
        "operation_id": "old", "generation": "previous-server", "active": False,
        "outcome": "applied", "concluded_by": "boot", "outcome_at": "previous-boot",
        "tx_attempt_id": "old-tx",
    }})
    return state


def _fail_attempt_save(monkeypatch, state, point):
    if point == "before":
        def refuse(*_args, **_kwargs):
            raise state.StateUnavailable("lock_timeout", "fixture")

        monkeypatch.setattr(state, "update_state", refuse)
        return
    write = state.atomic_write_text
    failed_path = state.STATE_PATH if point == "primary" else state.STATE_LAST_GOOD_PATH

    def interrupted(path, body):
        if path == failed_path:
            raise OSError("fixture write failure")
        return write(path, body)

    monkeypatch.setattr(state, "atomic_write_text", interrupted)


@pytest.mark.parametrize("point", ["before", "primary", "backup"])
@pytest.mark.parametrize("recovery", ["link", "finish"])
def test_failed_begin_replaces_the_unchanged_previous_terminal_attempt(
        retained_state, isolated_progress, monkeypatch, point, recovery):
    state = retained_state
    with monkeypatch.context() as failure:
        _fail_attempt_save(failure, state, point)
        progress.begin()
    live = progress.snapshot({})
    operation = live["operation_id"]
    assert live["retention_unconfirmed"] is True
    first = state.read_state().values[progress.STATE_KEY]
    assert first["operation_id"] == (operation if point == "backup" else "old")

    if recovery == "link":
        progress.link({"attempt_id": "new-tx"})
    else:
        progress._finish(JSONResponse({"error": "refused"}, status_code=409), None)

    saved = state.read_state().values[progress.STATE_KEY]
    assert saved["operation_id"] == operation
    assert "retention_unconfirmed" not in progress.snapshot({})
    assert "retention_unconfirmed" not in saved and "_retention_base" not in saved
    assert "_retention_base" not in progress.snapshot({})
    if recovery == "link":
        assert saved["tx_attempt_id"] == "new-tx"
    else:
        assert saved["outcome"] == "not_applied" and saved["active"] is False
    _next_server(isolated_progress, monkeypatch)
    rebooted = progress.snapshot({})
    assert rebooted["operation_id"] == operation and rebooted["outcome"] != "applied"


@pytest.mark.parametrize("point", ["before", "primary", "backup"])
def test_an_unknown_baseline_needs_a_mutator_observation_before_recovery(
        retained_state, monkeypatch, point):
    state = retained_state

    def unreadable():
        raise OSError("fixture read failure")

    with monkeypatch.context() as failure:
        failure.setattr(state, "read_state", unreadable)
        _fail_attempt_save(failure, state, point)
        progress.begin()
    operation = progress.snapshot({})["operation_id"]
    assert progress.snapshot({})["retention_unconfirmed"] is True
    progress.link({"attempt_id": "new-tx"})
    saved = state.read_state().values[progress.STATE_KEY]
    if point == "before":
        assert saved["operation_id"] == "old"
        assert progress.snapshot({})["retention_unconfirmed"] is True
    else:
        # The first mutator supplied the baseline even though its save raised.
        assert saved["operation_id"] == operation and saved["tx_attempt_id"] == "new-tx"
        assert "retention_unconfirmed" not in progress.snapshot({})


@pytest.mark.parametrize("winner,point", [
    ("newer", "before"),
    ("predecessor_conclusion", "before"),
    ("boot", "backup"),
    ("boot", "none"),
])
def test_a_refused_finish_preserves_the_winner_and_reports_unconfirmed_retention(
        retained_state, isolated_progress, monkeypatch, winner, point):
    state = retained_state
    with monkeypatch.context() as failure:
        if point != "none":
            _fail_attempt_save(failure, state, point)
        progress.begin()
        if winner == "boot":
            progress.link({"attempt_id": "new-tx"})

    # Another process wins after admission's update lock has been released.
    def competing_write(st):
        record = st[progress.STATE_KEY]
        if winner == "newer":
            record.update(operation_id="newer", generation="other-server", active=False,
                          outcome="not_applied", concluded_by="executor")
        elif winner == "predecessor_conclusion":
            record.update(outcome_at="later-boot")  # Same ID, changed conclusion.
        else:
            record.update(active=False, outcome="applied", concluded_by="boot", outcome_at="later-boot")

    state.update_state(competing_write)
    before = state.read_state().values[progress.STATE_KEY]
    progress._finish(JSONResponse({"error": "refused"}, status_code=409), None)
    assert state.read_state().values[progress.STATE_KEY] == before
    assert progress.snapshot({})["retention_unconfirmed"] is True
    _next_server(isolated_progress, monkeypatch)
    assert progress.snapshot({})["operation_id"] == before["operation_id"]
    assert progress.snapshot({})["outcome"] == before["outcome"]


def test_a_late_save_cannot_overwrite_or_mark_a_newer_live_attempt(retained_state, monkeypatch):
    state = retained_state
    with monkeypatch.context() as failure:
        _fail_attempt_save(failure, state, "before")
        progress.begin()
    late = dict(progress._current)
    progress.begin()
    live = progress.snapshot({})
    saved = state.read_state().values[progress.STATE_KEY]
    progress._persist({**late, "active": False, "outcome": "failed", "concluded_by": "executor"})
    assert progress.snapshot({}) == live
    assert state.read_state().values[progress.STATE_KEY] == saved
