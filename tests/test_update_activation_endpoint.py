"""``POST /api/update/apply`` for the unified Update: prepare, activate and discard over a miniature body.

The writer fence is replaced by a recorder (its own matrix lives in test_update_apply_routing);
everything else is real: the copy registry, Git, the handoff and the restart request.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import body_adoption
from ouroboros.gateway import control
from supervisor import update_prepare
from tests.test_update_activation import Scene

pytestmark = pytest.mark.serial


class _Request:
    def __init__(self, body, restart=True):
        self._body = body
        self.restarts = []
        callback = (lambda owner=False: self.restarts.append(owner)) if restart else None
        self.app = SimpleNamespace(state=SimpleNamespace(request_restart=callback))

    async def json(self):
        return self._body


@pytest.fixture
def scene(tmp_path, monkeypatch):
    import supervisor.update_checks as checks

    scene = Scene(tmp_path, monkeypatch)
    scene.fence = []
    monkeypatch.setattr(control, "_index_lock_precondition", lambda: None)
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda reason: scene.fence.append(("stop", reason)) or [])
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update",
                        lambda: scene.fence.append(("reopen", "")) or True)
    monkeypatch.setattr(checks, "workspace_checks", lambda _row: {"ok": True, "stub": True})
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    yield scene
    _join_retirement()  # while this scene's bindings still hold, never against the next test's


def _join_retirement():
    """A discard retires its copy on a thread (``update_activation._discard``): wait for it to end."""
    for thread in [thread for thread in threading.enumerate() if thread.name == "update-candidate-retire"]:
        thread.join(timeout=60)
        assert not thread.is_alive(), "the discarded candidate's retirement did not end"


def _apply(body, **kwargs):
    request = _Request(body, **kwargs)
    response = asyncio.run(control.api_update_apply(request))
    return response.status_code, json.loads(response.body), request


def _prepared():
    deadline = time.monotonic() + 60
    while update_prepare._running.locked() and time.monotonic() < deadline:
        time.sleep(0.05)
    return update_prepare.current()


def test_prepare_answers_at_once_and_the_candidate_becomes_ready_beside_the_checkout(scene):
    before = scene.composition()
    status, body, _request = _apply({"strategy": "prepare"})
    assert status == 202 and body == {"status": "preparing", "strategy": "prepare"}
    row = _prepared()
    assert row["state"] == "ready" and scene.composition() == before and scene.fence == []
    assert control._managed_update_payload  # the status projection carries it
    assert update_prepare.projection()["actions"] == ["activate", "discard_candidate"]


def test_activate_fences_writers_authorizes_the_switch_and_requests_the_restart(scene):
    _apply({"strategy": "prepare"})
    row = _prepared()
    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
    assert status == 200 and body["status"] == "ok" and body["restarting"] is True
    assert scene.fence == [("stop", "activation")] and request.restarts == [True]
    handoff = body_adoption.read(scene.data)
    assert handoff["kind"] == "update" and handoff["phase"] == "authorized" and handoff["restart_bound"] is True
    assert update_prepare.find(row["candidate_id"])["state"] == "activating"


def test_a_late_local_change_reopens_the_writers_and_makes_the_candidate_stale(scene):
    _apply({"strategy": "prepare"})
    row = _prepared()
    (scene.serving / "notes-written-meanwhile.txt").write_text("late\n")
    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
    assert status == 409 and body["reason"] == "local_changed" and body["outcome"] == "not_applied"
    assert scene.fence == [("stop", "activation"), ("reopen", "")] and request.restarts == []
    assert update_prepare.find(row["candidate_id"])["state"] == "stale" and body_adoption.read(scene.data) == {}


def test_without_a_restart_callback_nothing_is_armed_and_the_candidate_stays_ready(scene):
    _apply({"strategy": "prepare"})
    row = _prepared()
    status, body, _request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]}, restart=False)
    assert status == 500 and body["outcome"] == "not_applied"
    assert body_adoption.read(scene.data) == {} and update_prepare.find(row["candidate_id"])["state"] == "ready"


def test_activate_names_its_candidate_and_discard_settles_it(scene):
    _apply({"strategy": "prepare"})
    row = _prepared()
    status, body, _request = _apply({"strategy": "activate", "candidate_id": "update_other"})
    assert status == 409 and body["reason"] == "candidate_changed" and scene.fence == []
    status, body, _request = _apply({"strategy": "discard_candidate", "candidate_id": row["candidate_id"]})
    assert status == 200 and body["status"] == "discarded"
    _join_retirement()
    record = update_prepare.find(row["candidate_id"])  # retired with its checkout, or still its settled record
    assert record is None or record["state"] == "discarded"
    assert update_prepare.current() is None


def _answered():
    """The preparation thread's end: its refusal noted and every client told (``_start_preparation.run``)."""
    for thread in [thread for thread in threading.enumerate() if thread.name == "update-prepare"]:
        thread.join(timeout=60)
        assert not thread.is_alive(), "the preparation thread did not end"
    return update_prepare.current()


def test_a_copy_git_cannot_make_settles_and_every_client_is_offered_preparing_again(scene, monkeypatch):
    """Whole-Astra review: the copy is recorded before Git makes its checkout. When Git cannot (here the copies
    folder is a file), the preparation thread ended with that record ``preparing`` while this process lived, and
    every client kept "Preparing…" with no action. Now the copy settles at once: the last state every client hears
    is no candidate, the refusal in Git's words and Prepare; the serving checkout is untouched, and preparing again
    makes a ready candidate."""
    from ouroboros.gateway import update_activation, update_progress

    monkeypatch.setattr(update_prepare, "_last_refusal", {})  # this test's own panel memory
    heard = []
    monkeypatch.setattr(update_progress, "notify_changed", lambda: heard.append(update_activation.candidate_view()))
    copies = scene.tmp / "worktrees" / "copies.noindex"
    copies.parent.mkdir(parents=True, exist_ok=True)
    copies.write_text("a file where the copies are born\n")
    before = scene.composition()

    status, _body, _request = _apply({"strategy": "prepare"})

    assert status == 202 and _answered() is None, update_prepare.current()  # never left preparing
    last = heard[-1]  # the candidate /api/update/status carries, as every client re-reads it at the thread's end
    assert (last["state"], last["actions"], last["refusal"]["code"]) == ("none", ["prepare"], "UPDATE_WORKSPACE_FAILED")
    assert "git worktree add exited" in last["refusal"]["text"] and scene.composition() == before
    assert [row["state"] for row in update_prepare._rows()] == ["discarded"]
    copies.unlink()
    _apply({"strategy": "prepare"})
    assert _answered()["state"] == "ready" and heard[-1]["actions"] == ["activate", "discard_candidate"]


def test_a_newer_copy_whose_checkout_timed_out_leaves_the_ready_candidate_current_and_activatable(scene,
                                                                                                  monkeypatch):
    """Whole-Astra review: a newer preparation is admitted only once its copy exists. Its checkout ending at the
    bounded Git owner's deadline left that copy ``preparing`` and current, hiding the ready candidate it never
    replaced. Now the copy settles at once: the ready candidate stays current with its actions, and Restart to
    update activates exactly it."""
    from ouroboros.gateway import update_activation
    from tests.test_update_prepare import deadline_at_checkout

    monkeypatch.setattr(update_prepare, "_last_refusal", {})  # this test's own panel memory
    _apply({"strategy": "prepare"})
    ready = _answered()
    assert ready["state"] == "ready"
    undo = deadline_at_checkout(monkeypatch)

    status, _body, _request = _apply({"strategy": "prepare"})

    assert status == 202 and _answered()["candidate_id"] == ready["candidate_id"], update_prepare.current()
    view = update_activation.candidate_view()
    assert (view["state"], view["actions"]) == ("ready", ["activate", "discard_candidate"])
    made = [row for row in update_prepare._rows() if row["candidate_id"] != ready["candidate_id"]]
    assert [row["state"] for row in made] == ["discarded"] and "timed out" in made[0]["reason"]
    assert made[0]["reason"].startswith("UPDATE_WORKSPACE_FAILED: ")
    undo()
    status, body, request = _apply({"strategy": "activate", "candidate_id": ready["candidate_id"]})
    assert (status, body["restarting"], request.restarts) == (200, True, [True])
    assert update_prepare.find(ready["candidate_id"])["state"] == "activating"


def test_a_ready_candidate_whose_record_is_held_through_the_admission_stays_current_and_the_copy_settles(scene,
                                                                                                         monkeypatch):
    """Whole-Astra follow-up: the admission replaces the live candidate under that candidate's record lock. Held by
    another owner past the registry's wait, the lock refuses untyped (``WorktreeOpsLockBusy``), and the newer copy
    stayed ``preparing`` while this process lived, current over the ready candidate it never replaced, also once the
    lock was free: no action, nothing that would settle it. Now that copy is discarded at once like any copy the
    admission does not admit: the ready candidate stays current with its actions, and Restart to update activates it."""
    import os

    from ouroboros import subagent_worktrees
    from ouroboros.gateway import update_activation

    monkeypatch.setattr(update_prepare, "_last_refusal", {})  # this test's own panel memory
    _apply({"strategy": "prepare"})
    ready = _answered()
    assert ready["state"] == "ready"
    held, free = threading.Event(), threading.Event()

    def holder():  # another owner keeps the ready candidate's record lock, through the registry's own primitive
        with subagent_worktrees._record_lock(ready["path"], None, op="held_by_test", task_id=ready["candidate_id"]):
            held.set()
            free.wait(60)

    monkeypatch.setattr(subagent_worktrees, "_LOCK_TIMEOUT_SEC", 0.5)  # the registry's 120 s wait, shortened
    other = threading.Thread(target=holder, name="record-lock-holder", daemon=True)
    other.start()
    assert held.wait(10)
    try:
        status, _body, _request = _apply({"strategy": "prepare"})
        _answered()
    finally:
        free.set()
        other.join(10)

    rows = [(row["candidate_id"], row["state"], (row.get("worker") or {}).get("pid")) for row in update_prepare._rows()]
    # Read once the producer ended and the lock is free again: never the newer copy left preparing.
    assert status == 202 and update_prepare.current()["candidate_id"] == ready["candidate_id"], (rows, os.getpid())
    made = [row for row in update_prepare._rows() if row["candidate_id"] != ready["candidate_id"]]
    assert [row["state"] for row in made] == ["discarded"], rows
    assert made[0]["reason"].startswith("UPDATE_ADMISSION_FAILED: WorktreeOpsLockBusy: private copy record lock")
    view = update_activation.candidate_view()
    assert (view["state"], view["actions"]) == ("ready", ["activate", "discard_candidate"])
    assert update_prepare.reconcile_interrupted(scene.data) == []
    status, body, request = _apply({"strategy": "activate", "candidate_id": ready["candidate_id"]})
    assert (status, body["restarting"], request.restarts) == (200, True, [True])
    assert update_prepare.find(ready["candidate_id"])["state"] == "activating"


def test_a_discard_while_the_writers_stop_is_never_activated(scene, monkeypatch):
    """The fence waits; the owner discards meanwhile: the request re-reads and authorizes nothing."""
    _apply({"strategy": "prepare"})
    row = _prepared()

    def fence_while_discarded(reason):
        scene.fence.append(("stop", reason))
        update_prepare.discard(update_prepare.find(row["candidate_id"]))
        return []

    monkeypatch.setattr(control, "_quiesce_repo_writers", fence_while_discarded)
    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})

    assert status == 409 and body["reason"] == "candidate_changed" and request.restarts == []
    assert body_adoption.read(scene.data) == {} and update_prepare.find(row["candidate_id"])["state"] == "discarded"
    assert scene.fence == [("stop", "activation"), ("reopen", "")]


def test_a_new_preparation_while_the_writers_stop_is_not_activated_in_its_place(scene, monkeypatch):
    _apply({"strategy": "prepare"})
    first = _prepared()

    def fence_while_superseded(reason):
        scene.fence.append(("stop", reason))
        scene.prepare()  # another preparation supersedes the confirmed candidate
        return []

    monkeypatch.setattr(control, "_quiesce_repo_writers", fence_while_superseded)
    status, body, request = _apply({"strategy": "activate", "candidate_id": first["candidate_id"]})

    assert status == 409 and body["reason"] == "candidate_changed" and request.restarts == []
    assert body_adoption.read(scene.data) == {} and update_prepare.current()["state"] == "ready"
    assert update_prepare.current()["candidate_id"] != first["candidate_id"]


def test_a_stale_screen_cannot_leave_out_the_local_work_of_a_newer_candidate(scene, monkeypatch):
    """Two devices: A failed and offered the exclusion; B was prepared meanwhile. A's request answers 409 and
    changes nothing; the same request naming B is admitted and prepares without the local work."""
    import supervisor.update_checks as checks

    monkeypatch.setattr(checks, "workspace_checks", lambda _row: {"ok": False, "stub": True})
    _apply({"strategy": "prepare"})
    first = _prepared()
    assert first["state"] == "failed" and "apply_without_local_changes" in update_prepare.projection()["actions"]
    _apply({"strategy": "prepare"})  # the other device
    second = _prepared()
    assert second["state"] == "failed" and second["candidate_id"] != first["candidate_id"]
    rows, before = update_prepare._rows(), scene.composition()

    status, body, _request = _apply({"strategy": "apply_without_local_changes", "candidate_id": first["candidate_id"]})

    assert (status, body["reason"], body["outcome"]) == (409, "candidate_changed", "not_applied")
    assert body["error"] == "the prepared update changed; nothing was prepared"  # words; the ids are details
    assert (body["details"]["expected"], body["details"]["live"]) == (first["candidate_id"], second["candidate_id"])
    assert not update_prepare._running.locked()  # nothing was admitted, no thread started
    assert update_prepare._rows() == rows and scene.composition() == before and scene.fence == []
    monkeypatch.setattr(checks, "workspace_checks", lambda _row: {"ok": True, "stub": True})
    status, body, _request = _apply({"strategy": "apply_without_local_changes", "candidate_id": second["candidate_id"]})
    assert (status, body["strategy"]) == (202, "apply_without_local_changes")
    plain = _prepared()
    assert plain["state"] == "ready" and plain["without_local_changes"] is True
    assert update_prepare.find(second["candidate_id"])["state"] == "superseded" and scene.composition() == before


def test_every_client_hears_the_preparation_start_and_the_activation_begin(scene, monkeypatch):
    """The indicator and other screens re-read on update_progress_changed: preparing at admission, the outcome at the
    end, activating before the restart (a stale screen would otherwise keep "Update available" or "Update ready")."""
    from ouroboros.gateway import update_progress

    heard = []
    monkeypatch.setattr(update_progress, "notify_changed", lambda: heard.append(update_prepare.projection()["state"]))
    _apply({"strategy": "prepare"})
    row = _prepared()
    assert heard[0] == "preparing" and heard[-1] == "ready", heard
    heard.clear()
    status, _body, _request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
    assert status == 200 and heard == ["activating"]


def test_leaving_the_work_out_without_a_live_candidate_is_refused(scene):
    status, body, _request = _apply({"strategy": "apply_without_local_changes"})
    assert (status, body["reason"]) == (409, "candidate_changed") and update_prepare.current() is None
    assert not update_prepare._running.locked()


def _staged_installer(scene, monkeypatch, *, starts=True):
    """A candidate whose official app is staged: the installer's two calls and the restart request, in order."""
    import sys

    from supervisor import update_native

    events = []
    apps = scene.tmp / "Apps"
    user_base = scene.tmp / "userbase"
    (user_base / "lib").mkdir(parents=True)
    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("user_site", user_base))
    monkeypatch.setattr(update_prepare, "_install_lock", lambda *_args, **_kw: {"returncode": 0})
    from ouroboros import body_switch

    installed, new = apps / "Ouroboros.app", apps / ".Ouroboros-update-9.9.9" / "Ouroboros.app"
    for bundle in (installed, new):
        bundle.mkdir(parents=True)
    staged = {"status": "staged", "version": "9.9.9", "installed": str(installed), "staged_app": str(new),
              "backup_app": str(apps / ".Ouroboros-backup.app"), "staging_dir": str(new.parent),
              "python": sys.executable, "runtime": {"version": "3.10.22", "sqlite": "3.53.1"},
              "previous_identity": body_switch.bundle_identity(str(installed)),
              "new_identity": body_switch.bundle_identity(str(new))}
    monkeypatch.setattr(update_native, "prepare", lambda _row: dict(staged))
    monkeypatch.setattr(update_native, "launcher_pid", lambda: 4242)
    monkeypatch.setattr(update_native, "installer_command", lambda _native, **kw: events.append(
        ("command", kw["adoption_id"])) or {"ok": True, "reason": "", "argv": ["installer"], "log": "", "entry": ""})
    monkeypatch.setattr(update_native, "hand_over", lambda command, **kw: events.append(
        ("start", command["argv"][0])) or ({"ok": True, "reason": "", "pid": 1} if starts
                                           else {"ok": False, "reason": "installer_not_started: refused"}))
    request_restart = control._request_restart
    monkeypatch.setattr(control, "_request_restart",
                        lambda request: events.append(("restart", "")) or request_restart(request))
    return events


def test_the_app_installer_starts_before_the_restart_is_requested(scene, monkeypatch):
    """It acts only once that restart armed the switch; started first, it can be named in the record and held."""
    events = _staged_installer(scene, monkeypatch)
    _apply({"strategy": "prepare"})
    row = _prepared()
    assert row["native"]["status"] == "staged" and row["environment"]["generation"]["python"]
    status, _body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
    handoff = body_adoption.read(scene.data)
    assert status == 200 and request.restarts == [True] and handoff["native"]["version"] == "9.9.9"
    assert handoff["native"]["previous"]["launcher"]["pid"] == 4242 and handoff["environment"]["kind"] == "user_site"
    assert events == [("command", handoff["id"]), ("start", "installer"), ("restart", "")]


def test_without_a_restart_the_started_installer_finds_its_activation_closed(scene, monkeypatch):
    events = _staged_installer(scene, monkeypatch)
    _apply({"strategy": "prepare"})
    row = _prepared()
    status, body, _request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]}, restart=False)
    assert status == 500 and body["outcome"] == "not_applied" and "nothing was switched" in body["error"]
    assert [kind for kind, _ in events] == ["command", "start", "restart"]
    assert body_adoption.read(scene.data) == {} and update_prepare.find(row["candidate_id"])["state"] == "ready"


def test_an_installer_that_cannot_start_requests_no_restart_and_arms_nothing(scene, monkeypatch):
    events = _staged_installer(scene, monkeypatch, starts=False)
    _apply({"strategy": "prepare"})
    row = _prepared()
    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
    assert (status, body["reason"], body["outcome"]) == (409, "native_hand_over_failed", "not_applied")
    assert [kind for kind, _ in events] == ["command", "start"] and request.restarts == []
    assert body_adoption.read(scene.data) == {} and update_prepare.find(row["candidate_id"])["state"] == "ready"
    assert scene.fence == [("stop", "activation"), ("reopen", "")]  # the writers reopen: nothing restarts


@pytest.fixture
def lock_scene(tmp_path, monkeypatch):
    """A candidate that brings its own packages (a changed lock), over the endpoint's recorded fence."""
    import supervisor.update_checks as checks

    scene = Scene(tmp_path, monkeypatch)
    scene.fence = []
    monkeypatch.setattr(control, "_index_lock_precondition", lambda: None)
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda reason: scene.fence.append(("stop", reason)) or [])
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update",
                        lambda: scene.fence.append(("reopen", "")) or True)
    monkeypatch.setattr(checks, "workspace_checks", lambda _row: {"ok": True, "stub": True})
    venv = tmp_path / "venv"
    (venv / "lib").mkdir(parents=True)
    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("venv", venv))
    monkeypatch.setattr(update_prepare, "_install_lock", lambda *_a, **_kw: {"returncode": 0})
    scene.official({"requirements-runtime.lock": "toydep==2.0\n", "ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    yield scene
    _join_retirement()


def test_activation_holds_the_local_model_runtime_install_while_it_switches_packages(lock_scene, monkeypatch):
    """The writer fence stops task writers; the server's own package writer is the local model runtime's install.
    While it writes the serving packages an activation that replaces them is refused (nothing armed, writers
    reopened, the candidate still ready); otherwise the install is held from the package check until the restart
    switches them, and a refused activation releases it."""
    from ouroboros import local_model

    manager = local_model.LocalModelManager()
    monkeypatch.setattr(local_model, "_manager", manager)
    monkeypatch.setattr(manager, "_run_install", lambda: None)  # an admitted install starts nothing here
    _apply({"strategy": "prepare"})
    row = _prepared()
    assert row["state"] == "ready" and row["environment"]["generation"], row.get("reason")
    manager._runtime_status = "installing"

    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})

    assert status == 409 and body["reason"] == "packages_busy" and request.restarts == []
    assert lock_scene.fence == [("stop", "activation"), ("reopen", "")] and body_adoption.read(lock_scene.data) == {}
    assert update_prepare.find(row["candidate_id"])["state"] == "ready"
    manager._runtime_status = "install_ok"  # it finished
    (lock_scene.serving / "notes-written-meanwhile.txt").write_text("late\n")
    status, body, _request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
    assert status == 409 and body["reason"] == "local_changed"
    manager.install_runtime()  # the refused activation released its hold
    assert manager._runtime_status == "installing"
    manager._runtime_status = "install_ok"

    again = update_prepare.prepare(fetch=False, checks=lambda _row: {"ok": True, "stub": True})
    status, body, request = _apply({"strategy": "activate", "candidate_id": again["candidate_id"]})

    assert status == 200 and body["restarting"] is True and request.restarts == [True]
    with pytest.raises(RuntimeError, match="an update is being activated"):
        manager.install_runtime()  # held until the restart switches the packages
