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

    def view():
        response = asyncio.run(control.api_update_status(None))
        assert response.status_code == 200, response.body
        return json.loads(response.body)["update_progress"]

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
        saved=saved, view=view, boot=boot, boot_restarts=boot_restarts
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
