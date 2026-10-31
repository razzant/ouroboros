"""A Restart to update that ends after its writer fence and before its restart: what it may have left is settled
before the writers return, and what cannot be settled stays actionable.

Most of these keep the endpoint's recorded fence and everything else real (the copy registry, Git, the handoff, the
restart request). ``real`` keeps the fence, the saved-work owner, the queue and the pool real too, only OS processes
simulated; ``native_scene`` keeps the detached installer and its custody real, and is no native qualification.
"""
from __future__ import annotations

import errno
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading

import pytest

from ouroboros import body_adoption, body_switch
from ouroboros.gateway import control, update_activation, update_progress
from supervisor import update_merge, update_native, update_prepare
from tests import test_update_activation, test_update_activation_endpoint, test_update_native
from tests.body_candidate_support import REPO, git
from tests.test_native_package import INSTALLED
from tests.test_update_abort_retention import _assert_returned, _system
from tests.test_update_activation import EXITED, STOPPED, Scene
from tests.test_update_activation_endpoint import _apply, _prepared, _staged_installer
from tests.test_update_native import _ended, _stop, _target_speaks, _tool, _until, _version_at

pytestmark = pytest.mark.serial  # real Git and child processes

owned_state = test_update_activation.owned_state
scene, lock_scene = test_update_activation_endpoint.scene, test_update_activation_endpoint.lock_scene
native_scene = test_update_native.scene


def _pins(scene):
    return git(scene.serving, "for-each-ref", "--format=%(refname)", body_adoption.PIN_PREFIX).split()


def deadline_at_snapshot(monkeypatch):
    """The activation snapshot's ``add -A`` (300 s live) gets a deadline it cannot meet: the real Git starts and the
    bounded Git owner ends it, answering as at any deadline. Returns the undo."""
    from supervisor import git_ops

    bounded = git_ops._run_git_process_bounded

    def owner(cmd, *, timeout, **kwargs):
        return bounded(cmd, timeout=1e-6 if "add" in cmd and "-A" in cmd else timeout, **kwargs)

    monkeypatch.setattr(git_ops, "_run_git_process_bounded", owner)
    return lambda: monkeypatch.setattr(git_ops, "_run_git_process_bounded", bounded)


def full_at_copy(monkeypatch, name):
    """The disk refuses authorization's copy of the helper file ``name``. Returns the undo."""
    copy = shutil.copyfile

    def refused(source, target, **kwargs):
        if str(target).endswith(name):
            raise OSError(errno.ENOSPC, "no space left on device", str(target))
        return copy(source, target, **kwargs)

    monkeypatch.setattr(body_adoption.shutil, "copyfile", refused)
    return lambda: monkeypatch.setattr(body_adoption.shutil, "copyfile", copy)


def _assert_recovered(scene, row, status, body, request, code=500):
    """Nothing armed, nothing pinned, the writers back, the candidate ready, the update lock free."""
    assert (status, body["outcome"]) == (code, "not_applied") and "nothing was switched" in body["error"], body
    assert "restart_required" not in body and request.restarts == []
    assert scene.fence == [("stop", "activation"), ("reopen", "")]
    assert body_adoption.read(scene.data) == {} and _pins(scene) == []
    assert update_prepare.find(row["candidate_id"])["state"] == "ready"
    update_merge.release_update_lock(update_merge.acquire_update_lock())


def _activates(row):
    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
    assert (status, body["restarting"], request.restarts) == (200, True, [True]), body
    assert update_prepare.find(row["candidate_id"])["state"] == "activating"


_CLAUDEXOR_NODE = str(pathlib.Path.home() / ".claudexor" / "node" / "bin" / "node")
NODE = _CLAUDEXOR_NODE if os.path.exists(_CLAUDEXOR_NODE) else shutil.which("node") or "node"
_PANEL = """
import { updateVerdict } from './modules/updates.js';
import { activationAttemptView } from './modules/update_candidate.js';
import { updateNotice } from './modules/update_status.js';
let raw = '';
for await (const chunk of process.stdin) raw += chunk;
const status = JSON.parse(raw);
const verdict = updateVerdict(status);
const last = activationAttemptView(status);
process.stdout.write(JSON.stringify({ state: verdict.state, headline: verdict.headline, action: verdict.action,
    last: last && last.headline, pill: updateNotice(status)?.pill || '' }));
"""


def _panel():
    """Every client's Update panel now: what ``/api/update/status`` carries for it (the candidate and the retained
    attempt), read by the real browser modules: its verdict, action, Last attempt and pill."""
    status = {"managed": True, "candidate": update_activation.candidate_view(),
              "update_progress": update_progress.snapshot(update_merge.active_update_tx())}
    shown = subprocess.run([NODE, "--input-type=module", "-e", _PANEL], input=json.dumps(status), text=True,
                           encoding="utf-8", capture_output=True, cwd=REPO / "web", timeout=60)
    assert shown.returncode == 0, shown.stderr
    return json.loads(shown.stdout)


def test_an_activation_snapshot_at_its_deadline_brings_the_writers_back_and_the_candidate_activates(scene, monkeypatch):
    """Whole review, F2: ``_run`` raises ``TimeoutExpired`` at Git's deadline. It escaped past the fence: the server
    lived on with writer admission closed, its pool disabled and no restart requested. Nothing was authorized, so the
    writers return, the candidate stays ready (a passing moment does not make it stale), and the next press works."""
    row = scene.prepare()
    undo = deadline_at_snapshot(monkeypatch)

    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})

    _assert_recovered(scene, row, status, body, request)
    assert body["reason"] == "activation_failed" and "TimeoutExpired" in body["error"]
    undo()
    _activates(row)


@pytest.mark.parametrize("name", ["switch.py", "installation_stop.py", "handoff.json"])
def test_a_partial_authorization_leaves_nothing_that_can_arm_and_no_pin_without_a_record(scene, monkeypatch, name):
    """Whole review, F2: authorization's helper copies and its record raise a bare ``OSError`` after the pins are set.
    The helper's folder may stay half written (the next authorization retires it); no record names those pins, so
    authorization takes them back itself."""
    row = scene.prepare()
    if name == "handoff.json":
        write = body_switch._write_json

        def refused(path, value):
            raise OSError(errno.ENOSPC, "no space left on device", str(path))

        monkeypatch.setattr(body_switch, "_write_json", refused)

        def undo():
            monkeypatch.setattr(body_switch, "_write_json", write)
    else:
        undo = full_at_copy(monkeypatch, name)

    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})

    _assert_recovered(scene, row, status, body, request)
    assert "OSError" in body["error"] and body_adoption.helper_dir(scene.data).exists()
    undo()
    _activates(row)


def test_a_binding_that_fails_closes_the_handoff_it_was_binding(scene, monkeypatch):
    row = scene.prepare()
    record = body_switch.record

    def refused_at_binding(directory, handoff, phase, detail=""):
        if detail == "bound to the activation restart":
            raise OSError(errno.EIO, "input/output error")
        return record(directory, handoff, phase, detail)

    monkeypatch.setattr(body_switch, "record", refused_at_binding)
    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
    _assert_recovered(scene, row, status, body, request)
    monkeypatch.setattr(body_switch, "record", record)
    scene.fence.clear()
    monkeypatch.setattr(body_adoption, "bind_update_restart", lambda _data: False)  # read back as nothing to bind

    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})

    _assert_recovered(scene, row, status, body, request, code=409)
    assert body["reason"] == "restart_binding_failed"


def test_a_failed_mark_or_a_restart_with_no_callback_closes_the_handoff_and_stays_ready(scene, monkeypatch):
    """Each ends after authorization with no restart accepted: no exit tail can arm what is closed."""
    row = scene.prepare()
    mark = update_prepare.mark

    def unwritable(live, **fields):
        if fields.get("state") == "activating":
            raise OSError(errno.ENOSPC, "no space left on device")
        return mark(live, **fields)

    monkeypatch.setattr(update_prepare, "mark", unwritable)
    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
    _assert_recovered(scene, row, status, body, request)
    monkeypatch.setattr(update_prepare, "mark", mark)
    scene.fence.clear()

    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]}, restart=False)

    _assert_recovered(scene, row, status, body, request)  # marked, then back to ready
    assert (body["reason"], body["status"]) == ("restart_unavailable", "restart_required")
    _activates(row)


def test_an_installer_that_raises_or_a_mark_after_it_started_ends_with_its_activation_closed(scene, monkeypatch):
    """Either way the installer's record is gone before the writers return: it waits for its arming on that record
    (``app_install.run``), so a started one ends with nothing changed."""
    _staged_installer(scene, monkeypatch)
    _apply({"strategy": "prepare"})
    row = _prepared()
    hand_over = update_native.hand_over

    def crashed(_command, **_kw):
        raise RuntimeError("the installer's kill failed too")

    monkeypatch.setattr(update_native, "hand_over", crashed)
    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
    _assert_recovered(scene, row, status, body, request)
    monkeypatch.setattr(update_native, "hand_over", hand_over)
    scene.fence.clear()
    mark = update_prepare.mark

    def unwritable(live, **fields):  # the installer is already started and waiting on the record
        if fields.get("state") == "activating":
            raise OSError(errno.ENOSPC, "no space left on device")
        return mark(live, **fields)

    monkeypatch.setattr(update_prepare, "mark", unwritable)
    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
    _assert_recovered(scene, row, status, body, request)


def test_after_the_restart_was_requested_a_failure_touches_nothing_it_owns(scene, monkeypatch):
    """The restart owns the transition from there: no abandonment, no writers, the failure stays an error."""
    from ouroboros.gateway import update_activation

    row = scene.prepare()
    monkeypatch.setattr(update_activation, "candidate_view", lambda *_a, **_k: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})

    assert scene.fence == [("stop", "activation")]
    assert body_adoption.read(scene.data)["restart_bound"] is True
    assert update_prepare.find(row["candidate_id"])["state"] == "activating"


def test_before_the_fence_a_failure_stays_an_error_and_recovers_nothing(scene, monkeypatch):
    row = scene.prepare()
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})

    assert scene.fence == [] and body_adoption.read(scene.data) == {}


def _unresolved(scene, status, body, request):
    """Its writers stopped, its transition open and every client told to restart; never "not applied"."""
    assert (status, body["reason"], body["restart_required"]) == (500, "activation_unresolved", True), body
    assert "outcome" not in body and "next restart either switches" in body["error"] and request.restarts == []
    assert scene.fence == [("stop", "activation")]
    attempt = update_progress.snapshot({})
    assert (attempt["result"], attempt["restart_required"], attempt["outcome"]) == ("failed", True, "unknown"), attempt
    assert attempt["error"] == body["error"] and attempt["current_generation"] is True


def _settles_at_the_next_boot(scene, row, armed):
    """The restart it asked for: armed, the switch runs and the record follows; not armed, nothing was applied."""
    if armed:
        assert body_adoption.arm(scene.data, worker_exits=EXITED, live_children=[], owned_stop=STOPPED,
                                 owner_restart=True) == "armed"
        assert scene.boot().returncode == 0
        assert git(scene.serving, "rev-parse", "HEAD") == row["layers"]["new_commit"]
        assert scene.reports()[-1]["a"] == "GEN_CAND"
        return
    result = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)
    update_prepare.record_activation(scene.data, result)
    assert result["outcome"] == "not_applied" and body_adoption.read(scene.data) == {}
    assert update_prepare.find(row["candidate_id"])["state"] == "ready"


@pytest.mark.parametrize("armed", [False, True], ids=["not_armed", "armed"])
def test_a_handoff_that_cannot_close_keeps_the_writers_stopped_and_says_what_a_restart_does(scene, monkeypatch,
                                                                                            owned_state, armed):
    """Authorized and bound, this handoff arms at any later restart. Reopening the writers over it would let them edit
    what that switch then reads, so they stay stopped and the attempt stays actionable for every client."""
    monkeypatch.setattr(update_progress, "_current", {})
    row = scene.prepare()
    archive = body_adoption._archive

    def unremovable(*_args):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(body_adoption, "_archive", unremovable)

    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]}, restart=False)

    _unresolved(scene, status, body, request)
    assert "could not be closed (authorized)" in body["error"]
    handoff = body_adoption.read(scene.data)
    assert handoff["phase"] == "authorized" and handoff["restart_bound"] is True
    assert update_prepare.find(row["candidate_id"])["state"] == "ready"
    monkeypatch.setattr(body_adoption, "_archive", archive)
    _settles_at_the_next_boot(scene, row, armed)


def test_an_unreadable_record_is_never_read_as_closed(scene, monkeypatch, owned_state):
    """``body_adoption.read`` answers ``{}`` for an unreadable record; the proof that lets writers return does not."""
    monkeypatch.setattr(update_progress, "_current", {})
    row = scene.prepare()
    real, close_update = body_switch.read_handoff, body_adoption.close_update
    closing = []

    def corrupt_once_closing(directory):
        if closing:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return real(directory)

    def close(*args):
        closing.append(True)
        return close_update(*args)

    monkeypatch.setattr(body_switch, "read_handoff", corrupt_once_closing)
    monkeypatch.setattr(body_adoption, "close_update", close)

    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]}, restart=False)

    _unresolved(scene, status, body, request)
    assert "unreadable: ValueError" in body["error"]


@pytest.mark.parametrize("lock_file", ["kept", "missing"])
def test_a_close_on_a_full_disk_uses_the_lock_file_authorization_made_and_no_other(scene, monkeypatch, lock_file):
    """A full disk creates nothing, and this abort's own failure may be exactly that. Authorization makes the lock file
    before any record, so the close needs no new one. Without it there is no exclusion to act on: nothing is removed and
    the record says what stays open."""
    row = scene.prepare()
    body_adoption.authorize_update(scene.data, row)
    lock = body_adoption.helper_dir(scene.data) / body_switch.LOCK_NAME
    assert lock.is_file()
    if lock_file == "missing":
        lock.unlink()

    def full(path, mode="r", **kwargs):
        if "a" in mode and not os.path.exists(path):
            raise OSError(errno.ENOSPC, "no space left on device", path)
        return open(path, mode, **kwargs)

    monkeypatch.setattr(body_switch, "open", full, raising=False)

    closed = body_adoption.close_update(scene.data, "activation_failed", row["candidate_id"])

    assert closed == ("" if lock_file == "kept" else "authorized")
    assert body_adoption.read(scene.data).get("phase") == (None if lock_file == "kept" else "authorized")
    assert len(_pins(scene)) == (0 if lock_file == "kept" else 5)


def test_a_close_waits_for_an_installers_record_step_so_nothing_writes_the_handoff_back(scene):
    """The installer journals under the helper's lock, reading the record and writing it whole (``native_journal``). A
    close between the two would pass its read-back and then find the record written again, still armable."""
    row = scene.prepare()
    body_adoption.authorize_update(scene.data, row)
    assert body_adoption.bind_update_restart(scene.data)
    helper = str(body_adoption.helper_dir(scene.data))
    has_read, may_write, closed = threading.Event(), threading.Event(), []

    def journal():
        with body_switch._locked(helper):
            handoff = body_switch.read_handoff(helper)
            has_read.set()
            may_write.wait(30)
            body_switch.write_handoff(helper, handoff)

    installer = threading.Thread(target=journal, name="installer-journal-step", daemon=True)
    closer = threading.Thread(target=lambda: closed.append(
        body_adoption.close_update(scene.data, "activation_failed", row["candidate_id"])),
        name="activation-abort-close", daemon=True)
    installer.start()
    assert has_read.wait(30)
    closer.start()
    closer.join(0.5)
    assert closer.is_alive() and closed == []  # it waits for that step
    may_write.set()
    installer.join(30)
    closer.join(30)
    assert closed == [""] and body_adoption.read(scene.data) == {}


def test_a_restart_callback_that_signalled_then_raised_keeps_its_custody(scene, monkeypatch, owned_state):
    """It may have accepted the restart before it raised, so this is no abort: the handoff stays bound and armable, the
    record says activating, and neither the hold nor the fence moves. The restart it accepted switches as asked."""
    monkeypatch.setattr(update_progress, "_current", {})
    row = scene.prepare()
    signalled = []

    def signalled_then_raised(request):
        signalled.append(request.app.state.request_restart(owner=True))
        raise RuntimeError("the restart callback failed after signalling")

    monkeypatch.setattr(control, "_request_restart", signalled_then_raised)

    status, body, _request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})

    assert signalled == [None] and (status, body["reason"]) == (500, "activation_unresolved")
    assert "its restart may already have been accepted" in body["error"] and "outcome" not in body
    assert scene.fence == [("stop", "activation")]
    assert body_adoption.read(scene.data)["restart_bound"] is True
    assert update_prepare.find(row["candidate_id"])["state"] == "activating"
    _settles_at_the_next_boot(scene, row, armed=True)


def test_a_record_that_cannot_be_put_back_stays_with_its_handoff_for_the_next_boot(scene, monkeypatch, owned_state):
    """A closed handoff under a record still saying ``activating`` would be a dead end no boot settles, so the handoff
    stays: that restart settles both."""
    monkeypatch.setattr(update_progress, "_current", {})
    row = scene.prepare()
    mark = update_prepare.mark

    def unwritable(live, **fields):
        if fields.get("state") == "ready":
            raise OSError(errno.ENOSPC, "no space left on device")
        return mark(live, **fields)

    monkeypatch.setattr(update_prepare, "mark", unwritable)

    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]}, restart=False)

    _unresolved(scene, status, body, request)
    assert "OSError" in body["error"] and update_prepare.find(row["candidate_id"])["state"] == "activating"
    assert body_adoption.read(scene.data)["phase"] == "authorized"
    monkeypatch.setattr(update_prepare, "mark", mark)
    _settles_at_the_next_boot(scene, row, armed=False)


def test_a_restart_already_accepted_elsewhere_keeps_its_custody_too(scene, monkeypatch, owned_state):
    """A late owner Restart passed its serialized step just before this activation took the lock. Its exit tail may be
    arming this handoff now, so it is not closed under that: the failure leaves it to that restart."""
    from ouroboros import server_process

    monkeypatch.setattr(update_progress, "_current", {})
    accepted = threading.Event()
    accepted.set()
    monkeypatch.setattr(server_process, "_restart_requested", accepted)
    row = scene.prepare()
    mark = update_prepare.mark

    def unwritable(live, **fields):
        if fields.get("state") == "activating":
            raise OSError(errno.ENOSPC, "no space left on device")
        return mark(live, **fields)

    monkeypatch.setattr(update_prepare, "mark", unwritable)

    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})

    _unresolved(scene, status, body, request)
    assert "its restart may already have been accepted" in body["error"]
    assert body_adoption.read(scene.data)["restart_bound"] is True
    assert update_prepare.find(row["candidate_id"])["state"] == "ready"
    _settles_at_the_next_boot(scene, row, armed=True)


def test_a_handoff_the_exit_tail_already_armed_is_never_abandoned_or_reopened_over(scene, monkeypatch, owned_state):
    """A late owner Restart armed it while this request failed: the record says what the next boot does."""
    monkeypatch.setattr(update_progress, "_current", {})
    row = scene.prepare()
    mark = update_prepare.mark

    def armed_then_failed(live, **fields):
        if fields.get("state") == "activating":
            assert body_adoption.arm(scene.data, worker_exits=EXITED, live_children=[], owned_stop=STOPPED,
                                     owner_restart=True) == "armed"
            raise OSError(errno.ENOSPC, "no space left on device")
        return mark(live, **fields)

    monkeypatch.setattr(update_prepare, "mark", armed_then_failed)

    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})

    _unresolved(scene, status, body, request)
    assert "(armed)" in body["error"] and body_adoption.read(scene.data)["phase"] == "armed"
    assert scene.boot().returncode == 0
    assert git(scene.serving, "rev-parse", "HEAD") == row["layers"]["new_commit"]


def test_an_open_transition_keeps_the_runtime_install_hold_a_closed_one_releases_it(lock_scene, monkeypatch):
    """The hold protects the packages the restart switches: it stays exactly while that switch can still happen."""
    from ouroboros import local_model

    monkeypatch.setattr(update_progress, "_current", {})
    manager = local_model.LocalModelManager()
    monkeypatch.setattr(local_model, "_manager", manager)
    monkeypatch.setattr(manager, "_run_install", lambda: None)
    _apply({"strategy": "prepare"})
    row = _prepared()
    assert row["environment"]["generation"], row.get("reason")
    undo = full_at_copy(monkeypatch, "switch.py")

    status, body, _request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})

    assert status == 500 and body["outcome"] == "not_applied"
    manager.install_runtime()  # closed: released
    assert manager._runtime_status == "installing"
    manager._runtime_status = "install_ok"
    undo()

    archive = body_adoption._archive
    monkeypatch.setattr(body_adoption, "_archive", lambda data, handoff, outcome: (
        1 / 0 if outcome.startswith("abandoned") else archive(data, handoff, outcome)))
    status, body, _request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]}, restart=False)

    assert (status, body["reason"]) == (500, "activation_unresolved")
    with pytest.raises(RuntimeError, match="an update is being activated"):
        manager.install_runtime()  # still open: held


@pytest.fixture
def real(tmp_path, monkeypatch):
    """A ready candidate over the real writer fence, saved-work owner, queue and pool, with one task's saved work
    running: only OS processes are simulated (``test_update_abort_retention._system``)."""
    from supervisor import git_ops

    scene = Scene(tmp_path, monkeypatch)
    system = _system(scene.data, monkeypatch)
    monkeypatch.setattr(git_ops, "REPO_DIR", scene.serving)
    monkeypatch.setattr(git_ops, "_git_dir", lambda: scene.serving / ".git")
    monkeypatch.setattr(update_progress, "_current", {})
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    system.scene, system.row = scene, scene.prepare()
    system.saved()
    return system


def _after_the_fence(monkeypatch, then):
    """Run ``then`` once the real fence has stopped everything."""
    quiesce = control._quiesce_repo_writers

    def fence(reason):
        blockers = quiesce(reason)
        assert blockers == []
        then()
        return blockers

    monkeypatch.setattr(control, "_quiesce_repo_writers", fence)


def _activate_real(system, **kwargs):
    return _apply({"strategy": "activate", "candidate_id": system.row["candidate_id"]}, **kwargs)


def _saved_work_resumes(system):
    assert not system.workers.repo_writer_admission_closed()
    _assert_returned(system, system.scene.data)
    assert len(system.launches) == 1
    system.workers.assign_tasks()
    sent = system.launches[0].get_nowait()
    assert (sent["id"], sent["_attempt"]) == ("saved", 2)


def _nothing_started(system):
    assert system.workers.repo_writer_admission_closed() and system.launches == []
    assert system.workers.RUNNING["saved"]["retained_for_boot"] is True


def test_the_real_fence_ordinary_success_keeps_what_the_restart_owns(real):
    status, body, request = _activate_real(real)
    assert (status, body["restarting"], request.restarts) == (200, True, [True])
    _nothing_started(real)
    assert body_adoption.read(real.scene.data)["restart_bound"] is True
    assert update_progress.snapshot({}) == {}  # an ordinary activation keeps no attempt of its own


def test_the_real_fence_known_refusal_returns_the_saved_work(real):
    (real.scene.serving / "notes-written-meanwhile.txt").write_text("late\n")
    status, body, _request = _activate_real(real)
    assert (status, body["reason"], body["outcome"]) == (409, "local_changed", "not_applied")
    assert "restart_required" not in body
    _saved_work_resumes(real)


def test_a_fence_that_raises_before_it_stopped_anything_readmits_the_writers(real, monkeypatch):
    """The drain raised right after the gate closed: the abort owner finds no stopped cohort and opens it again."""
    def broken(*_args, **_kwargs):
        raise RuntimeError("the activity registry could not be read")

    monkeypatch.setattr(real.workers, "drain_repo_writers", broken)

    status, body, request = _activate_real(real)

    assert (status, body["reason"], body["outcome"]) == (500, "activation_failed", "not_applied"), body
    assert "restart_required" not in body and request.restarts == []
    assert not real.workers.repo_writer_admission_closed()
    assert "retained_for_boot" not in real.workers.RUNNING["saved"] and real.launches == []  # never stopped


@pytest.mark.parametrize("fault", ["snapshot_deadline", "helper_copy"])
def test_a_transient_failure_after_the_real_fence_resumes_saved_work_and_activates_after_it(real, monkeypatch, fault):
    """The review's two producers, through the real owners: ordinary work runs again in this server, no restart."""
    undo = deadline_at_snapshot(monkeypatch) if fault == "snapshot_deadline" else full_at_copy(monkeypatch, "switch.py")

    status, body, request = _activate_real(real)

    assert (status, body["reason"], body["outcome"]) == (500, "activation_failed", "not_applied"), body
    assert "restart_required" not in body and request.restarts == []
    assert body_adoption.read(real.scene.data) == {} and _pins(real.scene) == []
    _saved_work_resumes(real)
    undo()
    status, body, request = _activate_real(real)
    assert (status, body["restarting"], request.restarts) == (200, True, [True]), body


def test_saved_work_that_cannot_be_handed_back_keeps_the_gate_closed_and_tells_every_client(real, monkeypatch):
    """No failed queue snapshot opens writers (the abort owner's own rule). The answer is true about both facts, nothing
    was switched and nothing resumed, and the retained attempt is the panel's Restart now on every client."""
    original = real.queue.atomic_write_text

    def full(path, *args, **kwargs):
        if path == real.queue.QUEUE_SNAPSHOT_PATH:
            raise OSError(errno.ENOSPC, "no space left on device")
        return original(path, *args, **kwargs)

    def fail_after_the_fence():
        monkeypatch.setattr(real.queue, "atomic_write_text", full)

    _after_the_fence(monkeypatch, fail_after_the_fence)
    full_at_copy(monkeypatch, "switch.py")

    status, body, _request = _activate_real(real)

    assert (status, body["outcome"], body["restart_required"]) == (500, "not_applied", True), body
    assert "running work stays stopped until Ouroboros restarts" in body["error"]
    assert body_adoption.read(real.scene.data) == {}
    assert real.workers.repo_writer_admission_closed() and real.launches == []  # its owner keeps the successor to retry
    attempt = update_progress.snapshot({})
    assert (attempt["result"], attempt["restart_required"], attempt["outcome"], attempt["work_stop"]) == (
        "failed", True, "not_applied", "invoked"), attempt
    monkeypatch.setattr(real.queue, "atomic_write_text", original)
    assert control._respawn_workers_after_failed_update() is True  # the owner's own retry
    _saved_work_resumes(real)


@pytest.mark.parametrize("blocker", ["a_service_survives", "the_service_stop_raises", "custody_keeps_one"])
def test_a_stop_the_fence_cannot_prove_keeps_its_writers_stopped_and_its_restart_for_every_client(real, monkeypatch,
                                                                                                  blocker):
    """Whole review R3: the pool stopped, then a service outlived its stop (or the stop raised, or custody kept a
    process). The fence answers ``restart_required`` and nothing may start into that unproven stop, but formerly no
    client kept it: the status said Update ready with no Last attempt and a refresh lost even the toast. Now that
    answer is the retained attempt every client reads (Restart now). The restart is the recovery: this generation
    still holds the stopped work for it, so a retry once the blocker is gone is refused the same way, and after it the
    panel offers Restart to update again."""
    from ouroboros import process_custody
    from ouroboros.tools import services

    _initialized(real, monkeypatch)
    stop, quiesce = services.kill_all_services, process_custody.quiesce_custodied_services

    def raises(*_args, **_kwargs):
        raise OSError(errno.EIO, "the service registry could not be read")

    if blocker == "a_service_survives":
        survivor = [{"service_id": "kept", "state": "running"}]
        monkeypatch.setattr(services, "kill_all_services", lambda *_a, **_k: survivor)
    elif blocker == "the_service_stop_raises":
        monkeypatch.setattr(services, "kill_all_services", raises)
    else:
        monkeypatch.setattr(process_custody, "quiesce_custodied_services", lambda _root: (False, ["custody:kept"]))

    status, body, request = _activate_real(real)

    assert (status, body["reason"], body["restart_required"], request.restarts) == (
        409, "update_writer_fence_blocked", True, []), body
    _nothing_started(real)
    attempt, (saved, kept) = update_progress.snapshot({}), update_progress._read_saved()
    assert (attempt.get("result"), attempt.get("restart_required"), attempt.get("outcome"), attempt.get("work_stop"),
            attempt.get("blockers")) == ("failed", True, "not_applied", "invoked", body["blockers"]), attempt
    assert (saved, kept.get("blockers"), kept.get("restart_required")) == ("valid", body["blockers"], True), kept
    assert (_panel()["state"], _panel()["action"]) == ("restart_needed", {"id": "restart", "label": "Restart now"})
    monkeypatch.setattr(services, "kill_all_services", stop)
    monkeypatch.setattr(process_custody, "quiesce_custodied_services", quiesce)

    status, body, request = _activate_real(real)

    assert (status, body["restart_required"], request.restarts, body_adoption.read(real.scene.data)) == (
        409, True, [], {}), body
    _nothing_started(real)
    attempt = update_progress.snapshot({})  # the latest answer, kept the same way: the pool is still down for it
    assert (attempt["blockers"], attempt["work_stop"]) == (body["blockers"], "invoked"), attempt
    monkeypatch.setattr(update_progress, "_generation", lambda: "after-the-restart")
    monkeypatch.setattr(update_progress, "_current", {})
    assert (_panel()["state"], _panel()["action"]["id"]) == ("candidate_ready", "activate")


def test_a_drain_refusal_stays_its_own_answer_with_its_writers_back(real, monkeypatch):
    """The control: active work refused the drain before any teardown, so the gate reopened at once; nothing is
    retained and no restart is asked."""
    monkeypatch.setattr(real.workers, "drain_repo_writers", lambda: ["review:task-x"])

    status, body, _request = _activate_real(real)

    assert (status, body["blockers"], body["restart_required"]) == (409, ["active:review:task-x"], False), body
    assert not real.workers.repo_writer_admission_closed() and real.launches == []
    assert "retained_for_boot" not in real.workers.RUNNING["saved"]
    assert update_progress.snapshot({}) == {} and _panel()["state"] == "candidate_ready"


def _initialized(system, monkeypatch):
    """The serving generation's state, where a retained attempt is saved (this root's history came first)."""
    from supervisor import state_initialization

    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    state_initialization.mark_pending(system.scene.data, origin="owner_reset")
    assert system.state.init_state().quality == "current"


def test_a_restart_handoff_that_cannot_be_prepared_stops_nothing_and_stays_its_own_answer(real, monkeypatch):
    """Whole review R3, the refusal before any teardown: a waiting task's restart handoff could not be prepared, so
    the fence reopened the writers and the pool never stopped. Nothing needs a restart: the answer asks none, nothing
    is retained, and every client (a refresh, another one) is offered Restart to update again."""
    from ouroboros import owner_wait

    def unreadable(*_args, **_kwargs):
        raise OSError("owner wait source bytes are unreadable")

    _initialized(real, monkeypatch)
    monkeypatch.setattr(owner_wait, "prepare_owner_wait_handoffs", unreadable)

    status, body, request = _activate_real(real)

    assert (status, body["blockers"], body["restart_required"], request.restarts) == (
        409, ["owner_wait_handoff:OSError: owner wait source bytes are unreadable"], False, []), body
    assert "No work was stopped" in body["error"]
    assert not real.workers.repo_writer_admission_closed() and real.launches == []
    assert "retained_for_boot" not in real.workers.RUNNING["saved"]  # its pool never stopped
    assert (update_progress.snapshot({}), update_progress._read_saved()[0]) == ({}, "absent")
    assert (_panel()["state"], _panel()["action"]["id"]) == ("candidate_ready", "activate")


def test_saved_work_that_cannot_be_captured_keeps_its_writers_refused_and_claims_no_stopped_work(real, monkeypatch):
    """The saved-work owner could not capture the running work, before any teardown: the fence keeps the writers
    refused and nothing reopens them before a restart, so every client keeps Restart now; but its pool runs on, and
    the attempt says no running work was interrupted."""
    from supervisor import restart_retention

    def unreadable(*_args, **_kwargs):
        raise RuntimeError("the running work could not be read")

    _initialized(real, monkeypatch)
    monkeypatch.setattr(restart_retention, "capture_update_returns", unreadable)

    status, body, request = _activate_real(real)

    assert (status, body["blockers"], body["restart_required"], request.restarts) == (
        409, ["saved_work_handoff:RuntimeError: the running work could not be read"], True, []), body
    assert real.workers.repo_writer_admission_closed() and real.launches == []
    assert "retained_for_boot" not in real.workers.RUNNING["saved"]  # its pool never stopped
    attempt, (saved, kept) = update_progress.snapshot({}), update_progress._read_saved()
    assert (attempt.get("result"), attempt.get("restart_required"), attempt.get("work_stop"), saved,
            kept.get("work_stop")) == ("failed", True, "none", "valid", "none"), attempt
    assert (_panel()["state"], _panel()["action"]["id"]) == ("restart_needed", "restart")


def _stop_begins(system):
    from ouroboros import owned_shutdown

    owned_shutdown.begin_owned_stop(system.scene.data, cause="owner_restart")


def test_a_late_stop_answers_what_it_refused_and_starts_no_pool_into_the_stop(real, monkeypatch):
    """This server's stop began under the activation: its next Git step refuses typed (``update_prepare.stopped``),
    which also escaped the fence. Its exit keeps the stopped work for the next boot, so nothing starts here."""
    from ouroboros import owned_shutdown

    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    _after_the_fence(monkeypatch, lambda: _stop_begins(real))

    status, body, request = _activate_real(real)

    assert (status, body["reason"], body["outcome"]) == (409, "UPDATE_PREPARE_STOPPED", "not_applied"), body
    assert body["restart_required"] is True and request.restarts == []
    _nothing_started(real)


def test_a_late_stop_after_authorization_closes_the_handoff_so_its_exit_tail_arms_nothing(real, monkeypatch):
    """The owner's Restart enters the stop door after the binding: the handoff closes (its pins stay: they keep objects
    only, and no child may start), so that restart's exit tail finds nothing, and no pool starts."""
    from ouroboros import owned_shutdown

    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    bind = body_adoption.bind_update_restart

    def bind_then_stop(data):
        bound = bind(data)
        _stop_begins(real)
        return bound

    monkeypatch.setattr(body_adoption, "bind_update_restart", bind_then_stop)

    status, body, _request = _activate_real(real, restart=False)

    assert (status, body["outcome"], body["restart_required"]) == (500, "not_applied", True), body
    assert body_adoption.read(real.scene.data) == {} and len(_pins(real.scene)) == 5
    assert update_prepare.find(real.row["candidate_id"])["state"] == "ready"
    assert body_adoption.arm(real.scene.data, worker_exits=EXITED, live_children=[], owned_stop=STOPPED,
                             owner_restart=True) == ""
    _nothing_started(real)


@pytest.mark.parametrize("panic", ["latch", "command_latch", "input"])
def test_a_late_panic_starts_no_pool_and_closes_what_it_can(real, monkeypatch, panic):
    """Panic forbids every automatic step: the failure still closes this activation's handoff, and nothing is revived
    into the emergency stop, whichever way the Panic is read."""
    from ouroboros import config, server_control
    from ouroboros.tools import shell_process

    def strike():
        if panic == "latch":
            monkeypatch.setattr(config, "DATA_DIR", real.scene.data)
            monkeypatch.setattr(server_control, "_restart_stop_requested", True)
        elif panic == "command_latch":
            monkeypatch.setattr(shell_process, "_panic_requested", True)
        else:
            (real.scene.data / "state" / "panic_stop.flag").write_text("panic")

    _after_the_fence(monkeypatch, strike)
    full_at_copy(monkeypatch, "switch.py")

    status, body, _request = _activate_real(real)

    assert status in (409, 500) and body["outcome"] == "not_applied" and body["restart_required"] is True, body
    assert body_adoption.read(real.scene.data) == {}
    _nothing_started(real)


@pytest.mark.skipif(sys.platform != "darwin", reason="the macOS app step: renamex_np and launchd")
def test_an_activation_that_fails_after_its_installer_started_ends_that_real_installer(native_scene, monkeypatch):
    """Whole review, F2: the installer is already running when the activation fails at its record. The writers may
    return only once it cannot act: its handoff closes first, and it reads that and ends with nothing quit, swapped or
    opened. The activation's own request answers, over the real installer and its custody."""
    scene = native_scene
    _target_speaks(scene, 2)
    row = scene.prepare()
    entry = pathlib.Path(row["native"]["staged_app"]) / "Contents" / "MacOS" / "ouroboros-shell"
    entry.write_text(f"#!/bin/sh\nexec {json.dumps(sys.executable)} \"$@\"\n")
    entry.chmod(0o755)
    fence = []
    monkeypatch.setattr(update_native, "launcher_pid", lambda: os.getppid())
    monkeypatch.setattr(control, "_index_lock_precondition", lambda: None)
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda reason: fence.append("stop") or [])
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: fence.append("reopen") or True)
    tools = {"osascript": _tool(scene.tmp / "osascript", "exit 0\n"), "open": _tool(scene.tmp / "open", "exit 0\n")}
    command, mark = update_native.installer_command, update_prepare.mark
    monkeypatch.setattr(update_native, "installer_command", lambda native, **kw: command(native, tools=tools, **kw))
    started = []

    log = pathlib.Path(row["native"]["staged_app"]).parent / "install.jsonl"
    in_custody = []

    def unwritable(live, **fields):  # what it sees it keeps for the test: an assertion here would read as this failure
        if fields.get("state") == "activating":
            started.append(body_adoption.read(scene.data)["native"]["installer"]["pid"])
            _until(lambda: log.exists() and '"custody"' in log.read_text(), timeout=30)
            native = body_adoption.read(scene.data)["native"]
            in_custody.append((body_switch.process_gone(native["installer"]), native["steps"]))
            raise OSError(errno.ENOSPC, "no space left on device")
        return mark(live, **fields)

    monkeypatch.setattr(update_prepare, "mark", unwritable)
    try:
        status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})

        assert (status, body["reason"], body["outcome"]) == (500, "activation_failed", "not_applied"), body
        assert fence == ["stop", "reopen"] and request.restarts == [] and body_adoption.read(scene.data) == {}
        assert in_custody == [(False, [])]  # alive, its record unwritten: ``arm`` never waits for that lock
        result, _lines = _ended(log)
        assert (result["status"], result["reason"]) == ("refused", "not_armed")
        assert _version_at(scene.app) == INSTALLED and pathlib.Path(row["native"]["staged_app"]).is_dir()
        assert update_prepare.find(row["candidate_id"])["state"] == "ready"
    finally:
        for pid in started:
            _stop(pid)
