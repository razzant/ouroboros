"""Retained update outcomes through real Git executors and modeled boot generations.

Readiness, smoke checks and restart exit are controlled boundaries; no native
restart is claimed. All storage and Git work belong to replace_scene.
"""
import json
from types import SimpleNamespace

import pytest
from supervisor import git_ops
from tests import test_git_lock_owners as git_fixtures
from tests.test_git_lock_owners import _git, _rows

# Export only the requested fixtures into this module, not the source's autouse sleep patch.
replacement_history = git_fixtures.replacement_history
replace_scene = git_fixtures.replace_scene
update_lanes = git_fixtures.update_lanes

@pytest.fixture
def retained_attempt(replace_scene, update_lanes, monkeypatch):
    import asyncio
    import threading

    import server
    from ouroboros import server_process, update_letter
    from ouroboros.gateway import control, update_progress as progress
    from supervisor import state, update_merge, workers

    scene = replace_scene
    for name, value in {
        "DRIVE_ROOT": scene.data,
        "STATE_PATH": scene.data / "state" / "state.json",
        "STATE_LAST_GOOD_PATH": scene.data / "state" / "state.last_good.json",
        "STATE_LOCK_PATH": scene.data / "locks" / "state.lock",
    }.items():
        monkeypatch.setattr(state, name, value)
    state.save_state({"owner_chat_id": 0})

    generation = SimpleNamespace(value="server-a")
    monkeypatch.setattr(progress, "_generation", lambda: generation.value)
    monkeypatch.setattr(progress, "_current", {})
    monkeypatch.setattr(progress, "_boot_settled", threading.Event())
    monkeypatch.setattr(progress, "_notify", lambda: None)
    monkeypatch.setattr(
        git_ops, "compute_managed_update_status", lambda **_kw: {}
    )
    monkeypatch.setattr(update_letter, "runtime_status", lambda value: value)
    monkeypatch.setattr(
        update_letter, "project_letter_for_panel", lambda _value: None
    )
    monkeypatch.setattr(
        update_letter, "refresh_after_check", lambda _value: None
    )
    monkeypatch.setattr(
        update_merge, "external_host_restart_smoke", lambda: {"ok": True}
    )
    monkeypatch.setattr(server, "DATA_DIR", scene.data)
    monkeypatch.setattr(server, "REPO_DIR", scene.repo)
    monkeypatch.setattr(server, "broadcast_ws_sync", lambda _value: None)
    monkeypatch.setattr(
        server.body_adoption, "settle_on_boot", lambda *_a, **_kw: None
    )
    boot_restarts = []
    monkeypatch.setattr(
        server, "_request_restart_exit", lambda: boot_restarts.append(True)
    )
    real_finalize = update_merge.finalize_managed_update_on_boot

    def saved():
        return json.loads(
            state.STATE_PATH.read_text(encoding="utf-8")
        )[progress.STATE_KEY]

    def status():
        response = asyncio.run(control.api_update_status(None))
        assert response.status_code == 200, response.body
        return json.loads(response.body)

    def view():
        return status()["update_progress"]

    def boot(name, *, bootstrap_ok=True, served=None):
        generation.value = name
        monkeypatch.setattr(progress, "_current", {})
        monkeypatch.setattr(progress, "_boot_settled", threading.Event())
        monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")
        monkeypatch.setattr(server_process, "_source_baseline", None)
        captured = server_process.capture_server_source_baseline(scene.repo)
        assert captured == _git(scene.repo, "rev-parse", "HEAD")
        if served is not None:
            # Negative case: this generation serves older source.
            monkeypatch.setattr(server_process, "_source_baseline", served)
        monkeypatch.setattr(server, "_bootstrap_ok", bootstrap_ok)
        monkeypatch.setattr(
            server, "_wait_for_supervisor_update_finalize", lambda: True
        )
        results = []

        def finalize(*, supervisor_ready):
            result = real_finalize(supervisor_ready=supervisor_ready)
            results.append(result)
            return result

        with monkeypatch.context() as capture:
            capture.setattr(
                update_merge, "finalize_managed_update_on_boot", finalize
            )
            server._boot_managed_update_tasks()
        assert len(results) == 1 and progress._boot_settled.is_set()
        return results[0]

    return SimpleNamespace(
        saved=saved, view=view, status=status, boot=boot, boot_restarts=boot_restarts
    )


@pytest.mark.parametrize("lane", ["replace", "auto_merge"])
def test_real_update_retains_saved_work_and_concludes_on_boot(
    replace_scene, update_lanes, retained_attempt, lane
):
    from supervisor import update_merge

    scene = replace_scene
    if lane == "auto_merge":
        (scene.repo / "seed.txt").write_text("owner edit\n", encoding="utf-8")
        (scene.repo / "draft.txt").write_text("owner draft\n", encoding="utf-8")
    code, body = (
        scene.apply(scene.target) if lane == "replace"
        else update_lanes(lane)
    )
    assert code == 200 and body["restarting"] is True, body

    tx = update_merge.read_update_tx_strict()[1]
    saved = retained_attempt.saved()
    assert tx["phase"] == "pending_boot_smoke"
    assert saved["tx_attempt_id"] == tx["attempt_id"]
    assert (saved["base_sha"], saved["target_sha"]) == (
        scene.base, scene.target
    )
    assert (saved["result"], saved["outcome"]) == ("restart_requested", "")
    assert _git(scene.repo, "rev-parse", "HEAD") == scene.target
    assert retained_attempt.view()["outcome"] == ""

    if lane == "replace":
        keep = body["keep_branch"]
        assert saved["keep_branch"] == keep
        assert _git(scene.repo, "rev-parse", keep) == scene.base
        assert _git(scene.repo, "show", f"{keep}:local.txt") == "owner commit"
        assert not (scene.repo / "local.txt").exists()
    else:
        stash = tx["stash_sha"]
        assert _git(scene.repo, "show", f"{stash}:seed.txt") == "owner edit"
        assert _git(scene.repo, "show", f"{stash}^3:draft.txt") == "owner draft"
        assert _git(
            scene.repo, "rev-parse", f"rescue-local-{stash[:12]}"
        ) == stash
        assert (scene.repo / "seed.txt").read_text(encoding="utf-8") == "seed\n"
        assert not (scene.repo / "draft.txt").exists()

    result = retained_attempt.boot("server-b")
    assert result["finalized"] is True
    assert result["observed"] == {
        "outcome": "applied",
        "attempt_id": tx["attempt_id"],
        "commit": scene.target,
    }
    assert update_merge.read_update_tx_strict()[0] == "absent"

    view = retained_attempt.view()
    assert (view["operation_id"], view["tx_attempt_id"]) == (
        saved["operation_id"], tx["attempt_id"]
    )
    assert (view["outcome"], view["concluded_by"], view["active"]) == (
        "applied", "boot", False
    )
    assert view["current_generation"] is False and "work_stop" not in view
    assert retained_attempt.saved()["outcome"] == "applied"

    if lane == "replace":
        assert view["keep_branch"] == keep
        assert _git(scene.repo, "show", f"{keep}:local.txt") == "owner commit"
    else:
        assert "local changes restored" in view["stash_note"]
        assert (scene.repo / "seed.txt").read_text(
            encoding="utf-8"
        ) == "owner edit\n"
        assert (scene.repo / "draft.txt").read_text(
            encoding="utf-8"
        ) == "owner draft\n"
        assert _git(scene.repo, "rev-parse", stash) == stash
        assert _git(
            scene.repo, "rev-parse", f"rescue-local-{stash[:12]}"
        ) == stash
    assert retained_attempt.view() == view


@pytest.mark.parametrize("missing_proof,reason", [
    ("bootstrap", "startup_unconfirmed"),
    ("source", "running_source_mismatch"),
])
def test_real_finalizer_success_still_requires_the_served_source_and_bootstrap(
    replace_scene, retained_attempt, missing_proof, reason
):
    code, body = replace_scene.apply(replace_scene.target)
    assert code == 200 and body["restarting"] is True
    result = retained_attempt.boot(
        "server-b",
        bootstrap_ok=missing_proof != "bootstrap",
        served=replace_scene.base if missing_proof == "source" else None,
    )
    assert result["observed"]["outcome"] == "applied"
    view = retained_attempt.view()
    assert (view["outcome"], view["reason"]) == ("unknown", reason)
    assert retained_attempt.saved()["outcome"] == "unknown"


def test_real_boot_rollback_concludes_only_in_the_restored_generation(
    replace_scene, update_lanes, retained_attempt, monkeypatch
):
    from supervisor import update_merge

    scene = replace_scene
    code, body = update_lanes("auto_merge")
    assert code == 200 and body["restarting"] is True
    tx = update_merge.read_update_tx_strict()[1]
    saved = retained_attempt.saved()
    monkeypatch.setattr(
        update_merge, "external_host_restart_smoke",
        lambda: {"ok": False, "error": "fixture boot failure"},
    )

    result = retained_attempt.boot("server-b")
    assert result["rolled_back"] is True and "observed" not in result
    assert _git(scene.repo, "rev-parse", "HEAD") == scene.base
    assert (scene.repo / "local.txt").read_text(
        encoding="utf-8"
    ) == "owner commit\n"
    assert not (scene.repo / "official.txt").exists()

    pending = update_merge.read_update_tx_strict()[1]
    assert pending["phase"] == update_merge.MARKER_CLEANUP_RETRY_PHASE
    assert pending["gate_blocked_reason"] == "rollback_restart_pending"
    assert pending["attempt_id"] == tx["attempt_id"]
    assert _git(
        scene.repo, "show", f"{pending['failed_update_ref']}:official.txt"
    ) == "release"
    [row] = _rows(scene.data, "managed_update_rolled_back")
    assert "checkout_already_restored" not in row
    assert retained_attempt.view()["outcome"] == ""
    assert retained_attempt.boot_restarts == [True]

    result = retained_attempt.boot("server-c")
    assert result["observed"] == {
        "outcome": "rolled_back",
        "attempt_id": tx["attempt_id"],
        "commit": scene.base,
    }
    assert update_merge.read_update_tx_strict()[0] == "absent"
    view = retained_attempt.view()
    assert (view["operation_id"], view["outcome"], view["concluded_by"]) == (
        saved["operation_id"], "rolled_back", "boot"
    )
    assert retained_attempt.saved()["outcome"] == "rolled_back"
    assert retained_attempt.boot_restarts == [True]


def test_explicit_retry_creates_distinct_real_attempt_identities(
    replace_scene, retained_attempt, monkeypatch
):
    from supervisor import update_merge

    scene = replace_scene
    monkeypatch.setattr(
        update_merge, "update_restart_smoke",
        lambda: {"ok": False, "error": "fixture pre-restart failure"},
    )
    code, body = scene.apply(scene.target)
    assert code == 409 and body["rolled_back"] is True
    first = retained_attempt.saved()
    assert (first["outcome"], first["concluded_by"]) == (
        "rolled_back", "executor"
    )
    assert _git(scene.repo, "rev-parse", "HEAD") == scene.base
    assert update_merge.read_update_tx_strict()[0] == "absent"
    assert scene.events["restarts"] == []
    assert _git(
        scene.repo, "show", f"{first['keep_branch']}:local.txt"
    ) == "owner commit"

    monkeypatch.setattr(update_merge, "update_restart_smoke", lambda: {"ok": True})
    code, body = scene.apply(scene.target)  # Another explicit request.
    assert code == 200 and body["restarting"] is True
    second = retained_attempt.saved()
    assert second["operation_id"] != first["operation_id"]
    assert second["tx_attempt_id"] != first["tx_attempt_id"]
    assert second["tx_attempt_id"] == (
        update_merge.read_update_tx_strict()[1]["attempt_id"]
    )
    assert second["outcome"] == "" and "concluded_by" not in second
    assert retained_attempt.boot("server-b")["observed"]["attempt_id"] == (
        second["tx_attempt_id"]
    )
    view = retained_attempt.view()
    assert (view["operation_id"], view["outcome"]) == (
        second["operation_id"], "applied"
    )


def test_real_executor_drain_refusal_stops_nobody_and_schedules_nothing(
    replace_scene, retained_attempt, monkeypatch
):
    from ouroboros.gateway import control
    from ouroboros.tools import services
    from supervisor import update_merge, workers

    scene, drains, stops = replace_scene, [], []
    # replace_scene bypasses quiescence; use the real drain/teardown body.
    monkeypatch.setattr(
        control, "_quiesce_repo_writers", control._stop_repo_writers
    )
    monkeypatch.setattr(
        workers, "drain_repo_writers",
        lambda: drains.append(True) or ["Main conversation"],
    )
    monkeypatch.setattr(
        workers, "kill_workers_for_update",
        lambda **_kw: stops.append("workers") or [],
    )
    monkeypatch.setattr(
        services, "kill_all_services",
        lambda *_a, **_kw: stops.append("services") or [],
    )

    code, body = scene.apply(scene.target)
    assert code == 409 and body["reason"] == "update_writer_fence_blocked"
    saved = retained_attempt.saved()
    assert (saved["outcome"], saved["work_stop"], saved["restart_required"]) == (
        "not_applied", "none", False
    )
    assert saved["blockers"] == ["active:Main conversation"]
    assert not saved.get("tx_attempt_id")
    assert update_merge.read_update_tx_strict()[0] == "absent"
    assert workers.repo_writer_admission_closed() == ""
    assert _git(scene.repo, "rev-parse", "HEAD") == scene.base
    assert retained_attempt.view()["operation_id"] == saved["operation_id"]

    assert retained_attempt.boot("server-b")["reason"] == "no pending update"
    assert retained_attempt.view()["outcome"] == "not_applied"
    assert drains == [True] and stops == [] and scene.events["restarts"] == []




@pytest.mark.parametrize("with_stash", [False, True])
def test_completed_update_keeps_its_outcome_after_marker_cleanup_retry(
    replace_scene, update_lanes, retained_attempt, monkeypatch, with_stash
):
    from supervisor import update_merge

    scene = replace_scene
    if with_stash:
        (scene.repo / "seed.txt").write_text("owner edit\n", encoding="utf-8")
        (scene.repo / "draft.txt").write_text("owner draft\n", encoding="utf-8")

    code, body = update_lanes("auto_merge")
    assert code == 200 and body["restarting"] is True
    tx = update_merge.read_update_tx_strict()[1]
    operation = retained_attempt.saved()["operation_id"]
    with monkeypatch.context() as failure:
        failure.setattr(update_merge, "clear_update_tx", lambda: False)
        first = retained_attempt.boot("server-b")

    assert first["reason"] == "could not clear update marker"
    pending = update_merge.read_update_tx_strict()[1]
    assert pending["phase"] == update_merge.MARKER_CLEANUP_RETRY_PHASE
    assert pending["gate_blocked_from_phase"] == "pending_boot_smoke"
    if with_stash:
        assert pending["stash_restore"]["status"] == "restored"
        assert pending["stash_restore"]["context"] == "boot_finalize"
    assert retained_attempt.view()["outcome"] == ""

    second = retained_attempt.boot("server-c")
    assert second["finalized"] is True
    assert update_merge.read_update_tx_strict()[0] == "absent"
    assert _git(scene.repo, "rev-parse", "HEAD") == scene.target
    if with_stash:
        stash = tx["stash_sha"]
        assert _git(
            scene.repo, "rev-parse", f"rescue-local-{stash[:12]}"
        ) == stash
        assert (scene.repo / "seed.txt").read_text(
            encoding="utf-8"
        ) == "owner edit\n"
        assert (scene.repo / "draft.txt").read_text(
            encoding="utf-8"
        ) == "owner draft\n"
    assert retained_attempt.saved()["operation_id"] == operation

    # Cleanup retry retains the proven outcome instead of losing it with the marker.
    assert (second.get("observed"), retained_attempt.view()["outcome"]) == (
        {
            "outcome": "applied",
            "attempt_id": tx["attempt_id"],
            "commit": scene.target,
        },
        "applied",
    )
    assert retained_attempt.saved()["concluded_by"] == "boot"


@pytest.mark.parametrize('original_context,expected', [
    ('boot_stash_recovery', 'not_applied'), ('boot_finalize', 'applied'),
])
def test_interrupted_restore_keeps_meaning_after_preservation_and_cleanup_retry(
    replace_scene, retained_attempt, monkeypatch, original_context, expected
):
    """A seeded interrupted restore uses the real marker/restoration/cleanup owners."""
    from ouroboros.gateway import update_progress as progress
    from supervisor import update_merge

    scene = replace_scene
    (scene.repo / 'seed.txt').write_text('owner edit\n', encoding='utf-8')
    _git(scene.repo, 'stash', 'push', '-u', '-m', 'interrupted update fixture')
    stash = _git(scene.repo, 'rev-parse', 'refs/stash')
    progress.begin(base_sha=scene.base, target_sha=scene.target)
    phase = 'pending_boot_smoke' if original_context == 'boot_finalize' else 'stashing_local_work'
    tx = {'schema_version': 2, 'phase': phase, 'attempt_id': 'interrupted',
          'pre_update_sha': scene.base, 'target_sha': scene.target, 'merge_commit': scene.base,
          'branch': _git(scene.repo, 'branch', '--show-current'), 'stash_sha': stash,
          'stash_restore': {'status': 'applying', 'context': original_context, 'stash_sha': stash}}
    progress.link(tx)
    update_merge.write_update_tx(tx)
    with monkeypatch.context() as failure:
        failure.setattr(update_merge, 'clear_update_tx', lambda: False)
        retained_attempt.boot('server-b', served=scene.base)
    pending = update_merge.read_update_tx_strict()[1]
    assert pending['stash_restore']['context'] == 'boot_restore_resume'
    assert pending['stash_restore']['evidence']['interrupted']['context'] == original_context
    result = retained_attempt.boot('server-c', served=scene.base)
    assert result['observed']['outcome'] == expected
    assert retained_attempt.view()['outcome'] == expected
    assert update_merge.read_update_tx_strict()[0] == 'absent'
    assert _git(scene.repo, 'rev-parse', stash) == stash
    assert not _git(scene.repo, 'status', '--porcelain')  # no unknown stash application replay


def _resolver_started(scene, attempt, **fields):
    """The real executor's assisted handoff: the attempt record bound to the transaction it wrote."""
    from starlette.responses import JSONResponse

    from ouroboros.gateway import update_progress as progress
    from supervisor import update_merge

    tx = {"schema_version": 2, "phase": "assisted_resolution", "attempt_id": attempt,
          "pre_update_sha": scene.base, "pre_update_branch": "ouroboros", "target_sha": scene.target,
          "local_snapshot": scene.base, "owner_chat_id": 0, "resolver_submitted_id": "", **fields}

    @progress.observe_result
    def started():
        progress.begin(base_sha=scene.base, target_sha=scene.target)
        progress.link(tx)
        update_merge.write_update_tx(tx)
        return JSONResponse({"status": "assisted_started"})

    started()
    return tx


def test_failed_boot_unwind_is_a_known_failure_until_the_restored_generation(
    replace_scene, retained_attempt, monkeypatch
):
    """#2671: an unready resolver's boot rollback that does not finish is no live resolution; its
    replay stays failed; a completed rollback is success only in the generation it restores."""
    from supervisor import update_candidate, update_merge, workers

    scene = replace_scene
    tx = _resolver_started(scene, "unwind")
    operation = retained_attempt.saved()["operation_id"]
    stops = []
    monkeypatch.setattr(update_candidate, "_boot_resolver_ready", lambda _wait: False)
    # An unproven stop is disclosed with the outcome, never a veto: the rollback still runs.
    monkeypatch.setattr(workers, "kill_workers_for_update", lambda **_kw: stops.append(True) or ["worker:4242"])
    monkeypatch.setattr(update_merge, "_assisted_head_state", lambda _tx: "in_progress")
    monkeypatch.setattr(update_merge, "_merge_head_sha", lambda: "")
    monkeypatch.setattr(git_ops, "_clear_update_intent", lambda: False)

    # Served source differs from every commit the transaction names: a known failure needs none.
    first = retained_attempt.boot("server-b", served=scene.target)
    assert first["rolled_back"] is False and first["blockers"] == ["worker:4242"]
    assert first["observed"] == {"outcome": "failed", "attempt_id": tx["attempt_id"],
                                 "reason": "rollback_incomplete"}
    assert "rollback did not finish" in first["stash_note"]
    assert stops == [True] and retained_attempt.boot_restarts == []
    assert update_merge.read_update_tx_strict()[1]["phase"] == "rolling_back"
    assert _git(scene.repo, "rev-parse", "HEAD") == scene.base
    status = retained_attempt.status()
    view = status["update_progress"]
    assert status["update_tx"]["attempt_id"] == view["tx_attempt_id"] == tx["attempt_id"]
    assert status["update_tx"]["phase"] == "rolling_back"
    assert (view["operation_id"], view["result"], view["concluded_by"], view["active"]) == (
        operation, "assisted_started", "boot", False)
    assert (view["outcome"], view["reason"], view["error"]) == (
        "failed", "rollback_incomplete", first["msg"])
    assert view["stash_note"] == first["stash_note"] and view["blockers"] == ["worker:4242"]

    # Replay: the durable rolling_back phase retries and fails again; the record restates that boot's own report.
    second = retained_attempt.boot("server-c")
    assert second["rolled_back"] is False and second["observed"] == first["observed"]
    view = retained_attempt.view()
    assert (view["operation_id"], view["outcome"], view["error"]) == (operation, "failed", second["msg"])
    assert view["stash_note"] == "" and view["blockers"] == []

    # The rollback completes: success is not claimed before the restored generation serves it.
    monkeypatch.setattr(git_ops, "_clear_update_intent", lambda: True)
    third = retained_attempt.boot("server-d")
    assert third["rolled_back"] is True and "observed" not in third
    assert retained_attempt.boot_restarts == [True]
    assert retained_attempt.view()["outcome"] == "failed"
    pending = update_merge.read_update_tx_strict()[1]
    assert pending["gate_blocked_reason"] == "rollback_restart_pending"

    last = retained_attempt.boot("server-e")
    assert last["observed"] == {"outcome": "rolled_back", "attempt_id": tx["attempt_id"], "commit": scene.base}
    assert update_merge.read_update_tx_strict()[0] == "absent"
    view = retained_attempt.view()
    assert (view["operation_id"], view["outcome"], view["concluded_by"]) == (operation, "rolled_back", "boot")
    assert (view["error"], view["reason"], view["blockers"]) == ("", "", [])
    assert view["stash_note"] == last.get("stash_note", "")
    assert retained_attempt.boot_restarts == [True]


@pytest.mark.parametrize(("rolled_back", "phase", "fields"), [
    pytest.param(False, "pending_boot_smoke", {"merge_commit": "target", "boot_attempts": 0}, id='False-pending_boot_smoke-fields0'),
    pytest.param(False, "assisted_resolution", {}, id='False-assisted_resolution-fields3'),
    pytest.param(True, "applying_replace", {"pre_update_sha": "target"}, id='True-applying_replace-fields1'),
    pytest.param(True, "gate_blocked", {"gate_blocked_reason": "fixture"}, id='True-gate_blocked-fields4'),
    pytest.param(None, "rolling_back", {"rollback_reason": "fixture"}, id='None-rolling_back-fields2'),
])
def test_every_boot_rollback_branch_reports_its_own_unfinished_rollback(
    replace_scene, retained_attempt, monkeypatch, phase, fields, rolled_back
):
    from supervisor import update_merge

    scene = replace_scene
    fields = {key: scene.target if value == "target" else value for key, value in fields.items()}
    tx = _resolver_started(scene, f"branch-{phase}", **{**fields, "phase": phase})
    attempts = []

    def rollback(reason, **_kw):
        attempts.append(reason)
        if rolled_back is None:
            raise OSError("fixture rollback write failed")
        return rolled_back, f"fixture rollback {'finished' if rolled_back else 'refused'}"

    monkeypatch.setattr(update_merge, "rollback_managed_update", rollback)
    monkeypatch.setattr(update_merge, "_boot_smoke_rollback",
                        lambda _tx, reason: update_merge.rollback_managed_update_on_boot(reason))
    monkeypatch.setattr(update_merge, "_assisted_head_state", lambda _tx: "committed")

    result = retained_attempt.boot("server-b")
    assert len(attempts) == 1 and result["rolled_back"] is (rolled_back is True)
    view = retained_attempt.view()
    if rolled_back:  # Success keeps its two-generation proof: nothing concludes here.
        assert "observed" not in result and view["outcome"] == "" and retained_attempt.boot_restarts == [True]
        return
    assert result["observed"] == {"outcome": "failed", "attempt_id": tx["attempt_id"],
                                  "reason": "rollback_incomplete"}
    cause = "rollback raised OSError: fixture rollback write failed" if rolled_back is None else "fixture rollback refused"
    assert (view["outcome"], view["reason"], view["error"], view["stash_note"]) == (
        "failed", "rollback_incomplete", cause, "")
    assert retained_attempt.boot_restarts == []


@pytest.mark.parametrize("rollback_reached", [True, False])
def test_a_raised_rollback_is_reported_without_reclassifying_other_boot_failures(
    replace_scene, retained_attempt, monkeypatch, rollback_reached
):
    from supervisor import update_candidate, update_merge, workers

    tx = _resolver_started(replace_scene, "intent-write-failure")
    monkeypatch.setattr(update_candidate, "_boot_resolver_ready", lambda _wait: False)
    monkeypatch.setattr(workers, "kill_workers_for_update", lambda **_kw: [])
    monkeypatch.setattr(update_merge, "_assisted_head_state", lambda _tx: "in_progress")
    monkeypatch.setattr(update_merge, "_merge_head_sha", lambda: "")

    def cannot_write(*_args):
        raise OSError(28, "No space left on device", "update-tx.json")

    monkeypatch.setattr(update_merge, "write_update_tx" if rollback_reached else "_assisted_head_state", cannot_write)
    result = retained_attempt.boot("server-b")
    status = retained_attempt.status()
    assert status["update_tx"]["phase"] == "assisted_resolution"
    assert status["update_tx"]["attempt_id"] == tx["attempt_id"]
    assert _git(replace_scene.repo, "rev-parse", "HEAD") == replace_scene.base
    assert retained_attempt.boot_restarts == []
    if rollback_reached:
        assert result["rolled_back"] is False and result["observed"]["attempt_id"] == tx["attempt_id"]
        assert status["update_progress"]["outcome"] == "failed"
        assert status["update_progress"]["reason"] == "rollback_incomplete"
        assert "No space left on device" in status["update_progress"]["error"]
    else:
        assert result == {"finalized": False, "error": "exception"}
        assert status["update_progress"]["outcome"] == ""  # no rollback was attempted


def test_a_failed_rematerialization_keeps_its_cause_apart_from_the_rollback_failure(
    replace_scene, retained_attempt, monkeypatch
):
    from supervisor import update_merge, worker_chat_lane

    scene = replace_scene
    _resolver_started(scene, "rematerialize")
    monkeypatch.setattr(update_merge, "_assisted_head_state", lambda _tx: "in_progress")
    monkeypatch.setattr(update_merge, "_merge_head_sha", lambda: "")
    monkeypatch.setattr(update_merge, "refuse_unready_boot_resolver", lambda: {})
    monkeypatch.setattr(worker_chat_lane, "preload_owner_control_path", lambda: None)
    monkeypatch.setattr(git_ops, "rescue_into_tx", lambda *_a, **_kw: {})
    monkeypatch.setattr(update_merge, "materialize_assisted_merge_live",
                        lambda *_a: (False, "fixture conflicts could not be staged", ""))
    monkeypatch.setattr(update_merge, "rollback_managed_update",
                        lambda *_a, **_kw: (False, "fixture rollback refused"))

    result = retained_attempt.boot("server-b")
    assert result["observed"]["outcome"] == "failed"
    view = retained_attempt.view()
    assert (view["error"], view["stash_note"]) == ("fixture conflicts could not be staged", "fixture rollback refused")


def test_a_boot_smoke_rollback_door_that_raises_is_a_known_failure(replace_scene, retained_attempt, monkeypatch):
    """#2671 (A): the authority switch's own Git probe raising before any reset is the door's unfinished rollback,
    not a finalizer exception that leaves the attempt unconcluded; the phase keeps its own Restart."""
    from ouroboros import store_switch
    from supervisor import update_merge

    scene = replace_scene
    tx = _resolver_started(scene, "smoke-door", phase="pending_boot_smoke", merge_commit=scene.target,
                           boot_attempts=0)
    resets = []
    monkeypatch.setattr(update_merge, "rollback_managed_update", lambda *_a, **_kw: resets.append(True))

    def exhausted(*_args, **_kwargs):
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(store_switch, "subprocess", SimpleNamespace(run=exhausted))
    result = retained_attempt.boot("server-b")
    assert result.get("error") != "exception" and result["rolled_back"] is False and resets == []
    assert result["observed"] == {"outcome": "failed", "attempt_id": tx["attempt_id"], "reason": "rollback_incomplete"}
    assert result["msg"] == "rollback raised OSError: [Errno 24] Too many open files"
    status = retained_attempt.status()
    assert (status["update_tx"]["phase"], status["update_tx"]["attempt_id"]) == ("pending_boot_smoke", tx["attempt_id"])
    view = status["update_progress"]
    assert (view["outcome"], view["reason"], view["error"]) == ("failed", "rollback_incomplete", result["msg"])
    assert "retention_unconfirmed" not in view and retained_attempt.boot_restarts == []
    assert _git(scene.repo, "rev-parse", "HEAD") == scene.base


@pytest.mark.parametrize("lost", ["write", "lock"])
def test_an_unsaved_boot_failure_stays_known_live_and_says_it_is_unsaved(
    replace_scene, retained_attempt, monkeypatch, lost
):
    """#2671 (B): disk full on the same volume refuses the rollback's intent write AND the state write. The known
    failure still reaches the API this generation, marked unsaved; the next generation retries and saves it."""
    from supervisor import state, update_candidate, update_merge, workers

    tx = _resolver_started(replace_scene, "disk-full")
    before = retained_attempt.saved()
    monkeypatch.setattr(update_candidate, "_boot_resolver_ready", lambda _wait: False)
    monkeypatch.setattr(workers, "kill_workers_for_update", lambda **_kw: [])
    monkeypatch.setattr(update_merge, "_assisted_head_state", lambda _tx: "in_progress")
    monkeypatch.setattr(update_merge, "_merge_head_sha", lambda: "")

    def full(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as disk:
        disk.setattr(update_merge, "write_update_tx", full)
        if lost == "write":  # the record was read under the lock; only its replacement failed
            disk.setattr(state, "_save_state_unlocked", full)
        else:  # nothing was read under the lock
            disk.setattr(state, "update_state", lambda *_a, **_kw: (_ for _ in ()).throw(
                state.StateUnavailable("lock_timeout")))
        result = retained_attempt.boot("server-b")
        status = retained_attempt.status()
    assert result["rolled_back"] is False and "No space left on device" in result["msg"]
    assert retained_attempt.saved() == before  # nothing saved: the observation is live only
    assert (status["update_tx"]["phase"], status["update_tx"]["attempt_id"]) == ("assisted_resolution", tx["attempt_id"])
    view = status["update_progress"]
    assert (view["operation_id"], view["result"], view["outcome"], view["reason"], view["concluded_by"]) == (
        before["operation_id"], "assisted_started", "failed", "rollback_incomplete", "boot")
    assert view["retention_unconfirmed"] is True and view["current_generation"] is False
    assert "No space left on device" in view["error"]

    retry = retained_attempt.boot("server-c")  # space came back: the same door retries, nothing new decides it
    assert retry["rolled_back"] is True and retained_attempt.boot_restarts == [True]
    status = retained_attempt.status()
    assert status["update_tx"]["phase"] == "marker_cleanup_retry"  # a finished rollback awaiting its restart
    assert status["update_progress"]["outcome"] == "" and "retention_unconfirmed" not in status["update_progress"]
    last = retained_attempt.boot("server-d")
    assert last["observed"]["outcome"] == "rolled_back"
    assert (retained_attempt.view()["outcome"], retained_attempt.saved()["outcome"]) == ("rolled_back", "rolled_back")


def test_an_unsaved_boot_conclusion_keeps_the_attempt_guards(replace_scene, retained_attempt, monkeypatch):
    """The live fallback reuses the conclusion's own guards: another attempt's record is never concluded."""
    from ouroboros.gateway import update_progress as progress
    from supervisor import state, update_merge

    _resolver_started(replace_scene, "guarded", phase="rolling_back", rollback_reason="fixture")
    stored = retained_attempt.saved()
    state.update_state(lambda st: st[progress.STATE_KEY].update(tx_attempt_id="another-transaction"))
    monkeypatch.setattr(update_merge, "rollback_managed_update", lambda *_a, **_kw: (False, "fixture refused"))

    def full(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as disk:
        disk.setattr(state, "_save_state_unlocked", full)
        result = retained_attempt.boot("server-b")
        view = retained_attempt.view()
    assert result["observed"]["outcome"] == "failed"
    assert progress._current == {} and view["operation_id"] == stored["operation_id"]
    assert view["outcome"] != "failed" and "retention_unconfirmed" not in view
