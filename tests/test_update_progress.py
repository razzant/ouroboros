"""Progress belongs to the running executor; its last attempt outlives the process (#1818)."""

import asyncio
import inspect
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.responses import JSONResponse

from ouroboros.gateway import control, update_progress as progress
from supervisor import update_merge as _update_merge

_ACTIVE_UPDATE_TX = _update_merge.active_update_tx  # the real reader, before a fixture stands in for it


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








def _unfinished(attempt="tx-1", msg="rollback reset failed", note=None, **extra):
    return {"finalized": False, "rolled_back": False, "msg": msg, "stash_note": msg if note is None else note,
            "observed": {"outcome": "failed", "attempt_id": attempt, "reason": "rollback_incomplete"}, **extra}


@pytest.mark.parametrize("conclusion", [{"finalized": True, **_seen("applied")}, _unfinished()])
def test_a_late_executor_write_cannot_downgrade_a_boot_conclusion(isolated_progress, monkeypatch, conclusion):
    _restart_requested_attempt()
    late = {key: value for key, value in progress._current.items() if key != "owner_thread"}
    _next_server(isolated_progress, monkeypatch)
    expected = _boot(monkeypatch, conclusion)["outcome"]
    progress._persist({**late, "result": "ok", "outcome": "not_applied", "error": "late"})
    assert progress.snapshot({})["outcome"] == expected
    assert isolated_progress.saved[progress.STATE_KEY]["concluded_by"] == "boot"


@pytest.mark.parametrize("bootstrap_ok,baseline", [(True, "c" * 40), (False, "d" * 40)])
def test_a_known_unfinished_recovery_concludes_without_claiming_a_checkout(isolated_progress, monkeypatch,
                                                                           bootstrap_ok, baseline):
    _restart_requested_attempt()
    _next_server(isolated_progress, monkeypatch)
    view = _boot(monkeypatch, _unfinished(msg="conflicts could not be staged", note="rollback clean failed",
                                          blockers=["worker:7"]), baseline=baseline, bootstrap_ok=bootstrap_ok)
    assert (view["outcome"], view["reason"], view["concluded_by"]) == ("failed", "rollback_incomplete", "boot")
    assert (view["error"], view["stash_note"], view["blockers"]) == (
        "conflicts could not be staged", "rollback clean failed", ["worker:7"])
    assert view["generation_facts"]["after"] is None
    # The next boot repeats the same report once, then a restored generation replaces it whole.
    _next_server(isolated_progress, monkeypatch, "server-c")
    view = _boot(monkeypatch, _unfinished())
    assert (view["error"], view["stash_note"], view["blockers"]) == ("rollback reset failed", "", [])
    _next_server(isolated_progress, monkeypatch, "server-d")
    view = _boot(monkeypatch, {"finalized": True, **_seen("rolled_back", commit="a" * 40)}, baseline="a" * 40)
    assert (view["outcome"], view["reason"], view["error"], view["blockers"]) == ("rolled_back", "", "", [])
    assert view["keep_branch"] == "local-keep-20261021_032000-a1b2c3"


def test_boot_recovery_keeps_the_executor_cause_and_replaces_its_own_note(isolated_progress, monkeypatch):
    cause = "merge apply failed: index.lock exists"

    @progress.observe_result
    def execute():
        progress.begin()
        progress.link({"attempt_id": "tx-1"})
        progress.advance("applying")
        return JSONResponse({"error": cause, "rolled_back": False, "stash_note": "first rollback failed"},
                            status_code=500)

    execute()
    for generation, message, note in [("server-b", "rollback intent failed", None),
                                       ("server-c", "restaging failed", "second rollback failed")]:
        _next_server(isolated_progress, monkeypatch, generation)
        view = _boot(monkeypatch, _unfinished(msg=message, note=note))
        assert view["error"] == cause
        assert view["stash_note"] == "\n".join([message, *([note] if note else [])])
        assert "first rollback failed" not in view["stash_note"]
    _next_server(isolated_progress, monkeypatch, "server-d")
    view = _boot(monkeypatch, {"finalized": True, **_seen("rolled_back")})
    assert (view["outcome"], view["error"], view["reason"], view["stash_note"], view["blockers"]) == (
        "rolled_back", cause, "", "", [])


@pytest.mark.parametrize("result", [
    {"finalized": False, "rolled_back": False, "msg": "no observation named"},
    _unfinished(attempt="tx-2"),
    {**_unfinished(), "observed": {"outcome": "failed", "attempt_id": "tx-1"}},
])
def test_an_unnamed_or_foreign_failure_concludes_nothing(isolated_progress, monkeypatch, result):
    _restart_requested_attempt()
    _next_server(isolated_progress, monkeypatch)
    view = _boot(monkeypatch, result)
    assert (view["outcome"], view["reason"]) == ("unknown", "no_completion_record")
    assert "concluded_by" not in isolated_progress.saved[progress.STATE_KEY]


def test_the_same_server_never_borrows_a_known_failure(isolated_progress, monkeypatch):
    _restart_requested_attempt()
    same = _boot(monkeypatch, _unfinished())
    assert (same["outcome"], same["current_generation"]) == ("", True)


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
    ({"finalized": True, **_seen("rolled_back", commit="a" * 40)}, True, "a" * 40, ("rolled_back", "")),
    ({"abandoned": True, **_seen("not_applied", commit="a" * 40)}, True, "a" * 40, ("not_applied", "")),
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
    ({'stash_restore': {'context': 'boot_diverged_abandon'}}, ''),
    ({'stash_restore': {'context': 'boot_restore_resume', 'evidence': {'interrupted': {'context': 'boot_diverged_abandon'}}}}, ''),
    ({'stash_restore': {'context': 'boot_restore_resume'}}, ''),
    ({'gate_blocked_reason': 'replace_abandon_tx_cleanup_failed'}, 'not_applied'),
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


@pytest.mark.parametrize(("recovery", "point"), [("link", "before"), ("link", "backup"), ("finish", "primary")])
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


def test_a_restored_boot_replaces_the_executor_own_conclusion_detail(isolated_progress, monkeypatch):
    """Executor-concluded failure, then a boot whose rollback finished: the stale executor note is not shown
    under 'previous version restored'; the executor's cause stays, as it does after a failed boot in between."""
    cause = "merge apply failed: index.lock exists"

    @progress.observe_result
    def execute():
        progress.begin()
        progress.link({"attempt_id": "tx-1"}, keep_branch="local-keep-x")
        progress.advance("applying")
        return JSONResponse({"error": cause, "rolled_back": False, "reason": "rollback_gate_blocked",
                             "blockers": ["worker:9"],
                             "stash_note": "rollback could not clear update intent before restoring the repository"},
                            status_code=500)

    execute()
    assert isolated_progress.saved[progress.STATE_KEY]["concluded_by"] == "executor"
    _next_server(isolated_progress, monkeypatch)
    view = _boot(monkeypatch, {"finalized": True, **_seen("rolled_back")})
    assert (view["outcome"], view["concluded_by"], view["error"]) == ("rolled_back", "boot", cause)
    assert (view["reason"], view["stash_note"], view["blockers"], view["keep_branch"]) == ("", "", [], "local-keep-x")


def _failing_store(isolated_progress, monkeypatch, *, read_first):
    def update_state(mutator, **_kwargs):
        if read_first:
            mutator(dict(isolated_progress.saved))  # the replacement is computed, then the write fails
        raise OSError(28, "No space left on device")

    from supervisor import state as sup_state

    monkeypatch.setattr(sup_state, "update_state", update_state)


@pytest.mark.parametrize("read_first", [True, False])
def test_an_unsaved_boot_conclusion_stays_live_until_this_process_ends(isolated_progress, monkeypatch, read_first):
    _restart_requested_attempt()
    unconcluded = dict(isolated_progress.saved[progress.STATE_KEY])
    _next_server(isolated_progress, monkeypatch)
    _failing_store(isolated_progress, monkeypatch, read_first=read_first)
    view = _boot(monkeypatch, _unfinished(blockers=["worker:7"]))
    assert (view["outcome"], view["reason"], view["error"], view["blockers"]) == (
        "failed", "rollback_incomplete", "rollback reset failed", ["worker:7"])
    assert view["retention_unconfirmed"] is True and view["current_generation"] is False
    assert isolated_progress.saved[progress.STATE_KEY] == unconcluded
    _next_server(isolated_progress, monkeypatch, "server-c")  # the live fact ends with its process, as it says
    assert progress.snapshot({})["outcome"] == "unknown"


@pytest.mark.parametrize("case", ["foreign", "same_server", "newer_attempt"])
def test_an_unsaved_boot_conclusion_keeps_the_conclusion_guards(isolated_progress, monkeypatch, case):
    _restart_requested_attempt()
    if case != "same_server":
        _next_server(isolated_progress, monkeypatch)
    _failing_store(isolated_progress, monkeypatch, read_first=True)
    result = _unfinished(attempt="tx-2" if case == "foreign" else "tx-1")
    newer = {"operation_id": "newer", "active": True, "generation": "server-b"}
    finalize = (lambda **_kw: progress._current.update(newer) or result) if case == "newer_attempt" else (
        lambda **_kw: result)
    progress.observe_boot(finalize, ready=True, bootstrap_ok=True)
    if case == "newer_attempt":
        assert progress._current == newer
    else:
        assert progress.snapshot({}).get("outcome") != "failed"
        assert not progress._current.get("retention_unconfirmed")


def test_the_client_phase_readings_mirror_the_supervisor_phases():
    """update_status.js keeps one assisted-phase set and one completion phase; both name the supervisor's own."""
    import pathlib
    import re

    from ouroboros.server_restart import _RESTARTABLE_UPDATE_PHASES
    from supervisor import update_merge

    modules = pathlib.Path(__file__).resolve().parents[1] / "web" / "modules"
    source = (modules / "update_status.js").read_text(encoding="utf-8")
    assisted = re.search(r"const ASSISTED_PHASES = new Set\(\[([^\]]*)\]\);", source).group(1)
    assert tuple(re.findall(r"'([a-z_]+)'", assisted)) == update_merge._ASSISTED_PHASES
    assert re.search(r"const COMPLETION_RECORDED_PHASE = '([a-z_]+)';", source).group(1) == (
        update_merge.MARKER_CLEANUP_RETRY_PHASE)
    updates = (modules / "updates.js").read_text(encoding="utf-8")
    assert "includes('assisted')" not in updates and "assistedUpdatePhase(txPhase)" in updates
    # The restart readiness: the phase the restart admits, read with the two check values every lane writes and the
    # status projects, by one reader both surfaces call.
    assert re.search(r"const RESTART_READY_PHASE = '([a-z_]+)';", source).group(1) in _RESTARTABLE_UPDATE_PHASES
    reader = re.search(r"export function updateRestartReadiness\(tx\) \{(.*?)\n\}", source, re.S).group(1)
    checks = (update_merge._PRE_RESTART_SMOKE_PENDING, update_merge._PRE_RESTART_SMOKE_PASSED)
    assert set(re.findall(r"smoke === '([a-z]+)'", reader)) == set(checks)
    assert f'in ("{checks[0]}", "{checks[1]}")' in inspect.getsource(control._managed_update_payload)
    assert "updateRestartReadiness(data.update_tx)" in updates and "updateRestartReadiness(bound)" in source


# R4 (Astra, rl-20261125T100917-7711b6c1fd40): every lane records pending_boot_smoke BEFORE its dependency and import
# check and a failed check rolls the update back, so the phase alone made the rail say Restart needed and Dashboard →
# Updates offer Restart now while the check still ran. Only the attempt's own passed check, or its apply's own
# restart_required answer, makes the restart the next step.

_BASE, _TARGET, _COMMIT = "a" * 40, "b" * 40, "c" * 40
_PLAN = {"available": True, "kind": "clean", "base_sha": _BASE, "target_sha": _TARGET, "merge_commit": _COMMIT,
         "target_ref": "refs/ouroboros-managed/tags/v7.13.0", "update_channel": "stable", "local_dirty_count": 0}
_CONSUMERS = """
import { readFileSync } from 'node:fs';
import { updateAttemptView } from './modules/update_status.js';
import { updateVerdict } from './modules/updates.js';
const seen = {};
for (const [label, status] of Object.entries(JSON.parse(readFileSync(process.argv[1], 'utf8')))) {
    const rail = updateAttemptView(status.update_progress, status.update_tx);
    const panel = updateVerdict(status, '');
    seen[label] = { rail: rail && { pill: rail.pill, says: rail.says, headline: rail.headline },
        panel: { state: panel.state, headline: panel.headline, action: panel.action?.id || '' } };
}
process.stdout.write(JSON.stringify(seen));
"""


@pytest.fixture
def real_marker(isolated_progress, tmp_path, monkeypatch):
    """The transaction carrier as the lanes write it and the status reads it: the real marker writer, strict reader
    and active reader over a disposable checkout's .git and data root. Each test stands in only for the repository's
    Git steps, the dependency and import check, and the restart request (it answers that none could be requested)."""
    import ouroboros.update_letter as update_letter
    from ouroboros.delegate_recovery import PLANNED_RESTART_TRANSACTION_ENV
    from supervisor import git_ops, update_merge

    monkeypatch.delenv(PLANNED_RESTART_TRANSACTION_ENV, raising=False)  # an unrequested restart disarms it
    repo, data = tmp_path / "repo", tmp_path / "data"
    (repo / ".git").mkdir(parents=True)
    (data / "logs").mkdir(parents=True)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", data)
    monkeypatch.setattr(update_merge, "active_update_tx", _ACTIVE_UPDATE_TX)
    monkeypatch.setattr(git_ops, "compute_managed_update_status", lambda fetch: {
        "managed": True, "available": False, "current_sha": _BASE, "latest_sha": "", "update_channel": "stable"})
    monkeypatch.setattr(update_letter, "runtime_status", lambda status: status)
    monkeypatch.setattr(update_letter, "project_letter_for_panel", lambda status: None)
    monkeypatch.setattr(control, "_request_restart", lambda *_args, **_kwargs: False)
    statuses = {}

    def status(label):
        """The status read the server serves now, checked against its published shape."""
        from ouroboros.gateway.schema import contract_typed_dict, validate_ingress

        payload = control._managed_update_payload(fetch=False, include_tags=False)
        assert validate_ingress(payload, contract_typed_dict("UpdateStatusResponse")) == [], label
        statuses[label] = json.loads(json.dumps(payload))
        return statuses[label]

    def checked(ok):
        """The dependency and import check, read while it runs: the server's status at that moment."""
        def run():
            status(f"checking:{len(statuses)}")
            return {"ok": ok}
        return run

    def consumers():
        """Both readers of each status, as production JS: the rail's capsule and Dashboard → Updates."""
        from ouroboros.preflight_node import resolve_node
        from tests.group_run import run_in_group

        node = resolve_node()
        assert node, "Node is required for the real rail and Dashboard readings"
        path = tmp_path / "statuses.json"
        path.write_text(json.dumps(statuses), encoding="utf-8")
        result = run_in_group([node, "--input-type=module", "-e", _CONSUMERS, str(path)],
                              cwd=Path(__file__).resolve().parents[1] / "web", timeout=60)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    return SimpleNamespace(update_merge=update_merge, status=status, checked=checked, consumers=consumers,
                           statuses=statuses)


def _executor(attempt, body):
    """One admitted executor bound to ``attempt``; ``body`` is its own remaining work and returns its answer."""
    @progress.observe_result
    def execute():
        progress.begin(base_sha=_BASE, target_sha=_TARGET)
        progress.link({"attempt_id": attempt})
        return body()

    return execute()


def test_ordinary_and_replace_lanes_offer_the_restart_only_after_their_own_check(real_marker, monkeypatch):
    from ouroboros import store_switch
    from supervisor import git_ops, update_merge

    monkeypatch.setattr(git_ops, "_create_rescue_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(git_ops, "_collect_repo_sync_state", lambda: {})
    monkeypatch.setattr(update_merge, "destructive_apply_guard", lambda *_args: "")
    monkeypatch.setattr(update_merge, "apply_managed_merge_update", lambda *_args: (True, ""))
    monkeypatch.setattr(update_merge, "update_restart_smoke", real_marker.checked(True))
    # The ordinary clean merge, from the clean-tree prologue's transaction (control._stash_local_work_fenced).
    prologue = {"phase": "stashing_local_work", "pre_update_sha": _BASE, "pre_update_branch": "ouroboros",
                "base_sha": _BASE, "target_sha": _TARGET, "target_ref": _PLAN["target_ref"], "update_channel": "stable",
                "attempt_id": "tx-merge", "stash_sha": "", "local_work_carrier": "none", "requested_at": "2026-11-25"}
    answer = _executor("tx-merge", lambda: control._apply_clean_merge_fenced(None, dict(_PLAN), dict(prologue)))
    assert json.loads(answer.body)["status"] == "restart_required"
    real_marker.status("merge:after")
    # The Replace lane, its check before the restart and its store-less form whose check waits for the next start.
    monkeypatch.setattr(control, "_index_lock_precondition", lambda: None)
    monkeypatch.setattr(update_merge, "plan_managed_update_merge", lambda **_kwargs: dict(_PLAN))
    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda **_kwargs: object())
    monkeypatch.setattr(update_merge, "release_update_lock", lambda _fh: None)
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda _reason: [])
    monkeypatch.setattr(git_ops, "prepare_managed_update", lambda *_args, **_kwargs: (
        True, {"update_intent": {"strategy": "replace"}, "keep_branch": "local-keep-x"}))
    monkeypatch.setattr(git_ops, "_write_update_intent", lambda _intent: None)
    for deferred in (False, True):
        update_merge.clear_update_tx()
        monkeypatch.setattr(store_switch, "switch_or_reset", lambda *_args, deferred=deferred, **_kwargs: (
            True, "checked out", deferred))
        answer = control._apply_replace_recovery_fenced(None, expected_base_sha=_BASE, expected_target_sha=_TARGET)
        assert json.loads(answer.body)["status"] == "restart_required"
        real_marker.status(f"replace:{'deferred' if deferred else 'checked'}")
    statuses, seen = real_marker.statuses, real_marker.consumers()
    checking = [label for label in statuses if label.startswith("checking:")]
    assert len(checking) == 2, "the merge and the checked Replace each ran their check once; the deferred one none"
    for label in checking:
        # While the check runs the transaction is pending_boot_smoke, its check pending, and the executor's stage
        # says so in both places: nobody is asked to restart.
        assert statuses[label]["update_tx"]["phase"] == "pending_boot_smoke", label
        assert statuses[label]["update_tx"]["pre_restart_smoke"] == "pending", label
        assert seen[label]["rail"]["says"] == "stage" and seen[label]["rail"]["pill"] == "Checking update…", label
        assert seen[label]["panel"]["headline"] == "Checking the update and dependencies…", label
        assert seen[label]["panel"]["action"] != "restart", label
    for label in ("merge:after", "replace:checked", "replace:deferred"):
        expected = "pending" if label == "replace:deferred" else "passed"
        assert statuses[label]["update_tx"]["pre_restart_smoke"] == expected, label
        assert statuses[label]["update_progress"]["result"] == "restart_required", label
        # The apply's own answer that its restart could not be requested: the restart, in both places.
        assert (seen[label]["rail"]["says"], seen[label]["rail"]["pill"]) == ("restart", "Restart needed"), label
        assert (seen[label]["panel"]["state"], seen[label]["panel"]["action"]) == ("restart_required", "restart"), label


@pytest.mark.parametrize("passes", [True, False], ids=["check-passes", "check-fails"])
def test_a_reviewed_merge_is_the_check_at_work_until_its_own_check_passed(real_marker, monkeypatch, passes):
    update_merge = real_marker.update_merge
    monkeypatch.setattr(update_merge, "update_restart_smoke", real_marker.checked(passes))
    monkeypatch.setattr(update_merge, "_log_supervisor", lambda _row: True)

    def rollback(reason, **_kwargs):  # the restored checkout's end state: the transaction is gone
        assert update_merge.clear_update_tx() is True
        return True, f"restored the previous version ({reason})"

    monkeypatch.setattr(update_merge, "rollback_managed_update", rollback)
    _executor("tx-assisted", lambda: JSONResponse({"status": "assisted_started", "task_id": "update_assisted_merge_1"}))
    update_merge.write_update_tx({"phase": "committing_assisted", "attempt_id": "tx-assisted", "pre_update_sha": _BASE,
                                  "target_sha": _TARGET, "task_id": "update_assisted_merge_1"})
    ok, message = update_merge.managed_assisted_postcommit(update_merge.read_update_tx(), _COMMIT)
    assert ok is passes, message
    real_marker.status("after")
    statuses, seen = real_marker.statuses, real_marker.consumers()
    during = statuses["checking:0"]
    assert (during["update_tx"]["phase"], during["update_tx"]["pre_restart_smoke"]) == ("pending_boot_smoke", "pending")
    assert during["update_progress"]["result"] == "assisted_started"
    assert seen["checking:0"]["rail"] == {"pill": "Checking update…", "says": "stage",
                                          "headline": "Checking the update and dependencies…"}
    assert seen["checking:0"]["panel"] == {"state": "resolving", "headline": "Checking the update and dependencies…",
                                           "action": "check"}
    if passes:
        assert statuses["after"]["update_tx"]["pre_restart_smoke"] == "passed"
        assert seen["after"]["rail"]["says"] == "restart" and seen["after"]["rail"]["pill"] == "Restart needed"
        assert seen["after"]["panel"]["action"] == "restart"
        assert seen["after"]["rail"]["headline"] == seen["after"]["panel"]["headline"], "the panel's own sentence"
    else:
        # Rolled back in this generation: no transaction, the start's outcome unconfirmed (CLIENT_CONTRACT §15), and
        # nothing anywhere says restart.
        assert statuses["after"]["update_tx"] == {"active": False}
        rail = seen["after"]["rail"]
        assert (rail["says"], rail["pill"]) == ("outcome", "Update outcome unconfirmed")
        assert seen["after"]["panel"]["action"] != "restart"


def test_an_older_or_newer_record_is_ready_only_on_its_own_answer(real_marker):
    """A marker an older updater wrote (unstamped, no check recorded) and one a newer release wrote (its stamp above
    this version's) are not made ready by their phase; the apply's own restart_required answer still is."""
    update_merge = real_marker.update_merge
    path = update_merge._update_tx_marker_path()
    older = {"phase": "pending_boot_smoke", "attempt_id": "tx-older", "pre_update_sha": _BASE, "merge_commit": _COMMIT}
    path.write_text(json.dumps(older), encoding="utf-8")
    assert update_merge.read_update_tx_strict() == ("valid", older)
    _executor("tx-older", lambda: JSONResponse({"status": "assisted_started"}))
    real_marker.status("older:started")
    _executor("tx-older", lambda: control._restart_response(None, strategy="auto_merge", plan=dict(_PLAN)))
    real_marker.status("older:answered")
    newer = {"_schema_version": update_merge.UPDATE_TX_SCHEMA_VERSION + 1, "phase": "pending_boot_smoke",
             "pre_restart_smoke": "passed", "attempt_id": "tx-newer"}
    path.write_text(json.dumps(newer), encoding="utf-8")
    assert update_merge.read_update_tx_strict()[0] == "future"
    _executor("tx-newer", lambda: JSONResponse({"status": "assisted_started"}))
    real_marker.status("newer:started")
    statuses, seen = real_marker.statuses, real_marker.consumers()
    for label in statuses:
        assert "pre_restart_smoke" not in statuses[label]["update_tx"], label
        assert statuses[label]["update_tx"]["phase"] == "pending_boot_smoke", label
    for label in ("older:started", "newer:started"):
        assert seen[label]["rail"]["says"] == "unresolved" and seen[label]["panel"]["action"] != "restart", label
    assert (seen["older:answered"]["rail"]["says"], seen["older:answered"]["panel"]["action"]) == ("restart", "restart")
