"""The exit tail's ``arm`` against a retirement of the same handoff, on real Git, real owners and real cold entries.

The update lock excludes neither: the owner's Restart and the agent's hold it only around their checkout step, the
exit tail takes none, and an abort can wait after its one read of the restart event. Formerly ``arm`` read the record,
the close removed the helper, ``arm`` published a pointer to nothing, and every later start refused its boot (exit 3).
Both now hold the helper's lock and ``arm`` never waits for it, so the first is the one outcome: armed and kept, or gone
with no pointer.
"""
from __future__ import annotations

import contextlib
import errno
import json
import pathlib
import shutil
import subprocess
import sys
import threading

import pytest

from ouroboros import body_adoption, body_switch
from ouroboros.gateway import update_progress
from supervisor import update_prepare
from tests import test_update_activation, test_update_activation_endpoint
from tests.body_candidate_support import git
from tests.test_update_activation import EXITED, STOPPED
from tests.test_update_activation_abort import _activates, _panel, _unresolved
from tests.test_update_activation_endpoint import _apply

pytestmark = pytest.mark.serial  # real Git and child processes

owned_state = test_update_activation.owned_state
scene = test_update_activation_endpoint.scene

EXIT_TAIL = {"worker_exits": EXITED, "live_children": [], "owned_stop": STOPPED, "owner_restart": True}


def _bound(scene):
    """A prepared Update authorized and bound to a restart: what an exit tail arms and an abort closes."""
    row = scene.prepare()
    body_adoption.authorize_update(scene.data, row)
    assert body_adoption.bind_update_restart(scene.data)
    return row


def _pointer(scene):
    return pathlib.Path(body_switch.pointer_path(str(scene.serving)))


def _next_entry(scene):
    """The real next cold entry: its exit code, the generation it imported, and its refusal, if any."""
    before = len(scene.reports())
    entered = scene.boot()
    imported = scene.reports()[before:]
    return (entered.returncode, imported[-1]["a"] if imported else None,
            entered.stderr[-200:] if entered.returncode else "")


def _not_armed_rows(scene):
    rows = [json.loads(line) for line in (scene.data / "logs" / "supervisor.jsonl").read_text().splitlines()]
    return [row["blockers"] for row in rows if row.get("type") == "body_adoption_not_armed"]


@pytest.fixture
def exit_tail(scene, monkeypatch):
    """The real ``_emergency_process_cleanup`` of an accepted restart over this root, its stop door included: only the
    pool, the owned stop and the process table answer as an empty server's do."""
    import server
    from ouroboros import owned_shutdown
    from supervisor import workers

    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())  # this test's own generation
    monkeypatch.setattr(server, "DATA_DIR", scene.data)
    monkeypatch.setattr(workers, "kill_workers", lambda **_kw: True)
    monkeypatch.setattr(workers, "last_worker_exit_census", lambda: EXITED)
    monkeypatch.setattr(server, "stop_owned_work", lambda _root: STOPPED)
    monkeypatch.setattr(server, "_stop_owned_daemon_for_new_pin", lambda: None)
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda _port: None)
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])

    def run():
        server._restart_requested.set()
        server._emergency_process_cleanup()

    yield run
    server._restart_requested.clear()


def _unwritable_activating(monkeypatch):
    """The record's ``activating`` write fails after the handoff is authorized and bound: the abort runs."""
    mark = update_prepare.mark

    def unwritable(live, **fields):
        if fields.get("state") == "activating":
            raise OSError(errno.ENOSPC, "no space left on device")
        return mark(live, **fields)

    monkeypatch.setattr(update_prepare, "mark", unwritable)


def test_success_the_real_exit_tail_arms_an_ordinary_activation_and_the_next_entry_switches(scene, exit_tail):
    row = scene.prepare()
    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
    assert (status, body["restarting"], request.restarts) == (200, True, [True]), body
    exit_tail()
    assert body_adoption.read(scene.data)["phase"] == "armed" and _pointer(scene).exists()
    assert _next_entry(scene) == (0, "GEN_CAND", "")
    assert git(scene.serving, "rev-parse", "HEAD") == row["layers"]["new_commit"]


def test_abort_a_close_before_the_exit_tail_leaves_it_nothing_to_arm(scene, exit_tail):
    row = _bound(scene)
    assert body_adoption.close_update(scene.data, "activation_failed", row["candidate_id"]) == ""
    exit_tail()
    assert not _pointer(scene).exists() and _next_entry(scene) == (0, "GEN_OLD", "")


def test_arm_wins_a_restart_accepted_after_the_aborts_check_keeps_its_custody_through_the_close(scene, monkeypatch,
                                                                                               owned_state, exit_tail):
    """Formerly admitted. This abort read no restart accepted, then waited (``lifecycle()``, its record's lock, up to
    the registry's 120 s). The owner's Restart reached its exit tail meanwhile: ``arm`` read the record, the close
    removed the helper, ``arm`` published a pointer to nothing, and every later start refused its boot. Now ``arm``
    decides under the helper's lock, so the close waits and finds it armed: the abort says so, and the restart it
    accepted switches."""
    monkeypatch.setattr(update_progress, "_current", {})
    row = scene.prepare()
    _unwritable_activating(monkeypatch)
    find, close, locked, panic = (update_prepare.find, body_adoption.close_update, body_switch._locked,
                                  body_switch.installation_panic)
    arm_read, close_reached, activation = threading.Event(), threading.Event(), []
    tail = threading.Thread(target=exit_tail, name="exit-tail", daemon=True)

    def past_the_check(candidate_id, *args):  # the abort's first step after it read the restart event
        if not activation:
            activation.append(threading.current_thread())
            tail.start()
            assert arm_read.wait(120)
        return find(candidate_id, *args)

    def mid_decision(root):  # ``arm`` reads Panic after it read the record, before it publishes
        if threading.current_thread() is tail and not arm_read.is_set():
            arm_read.set()
            assert close_reached.wait(120)
        return panic(root)

    def noting(directory, *args, **kwargs):  # the close, about to wait for that lock
        if threading.current_thread() in activation:
            close_reached.set()
        return locked(directory, *args, **kwargs)

    def closing(*args):
        try:
            return close(*args)
        finally:
            close_reached.set()

    monkeypatch.setattr(update_prepare, "find", past_the_check)
    monkeypatch.setattr(body_switch, "installation_panic", mid_decision)
    monkeypatch.setattr(body_switch, "_locked", noting)
    monkeypatch.setattr(body_adoption, "close_update", closing)

    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
    tail.join(120)
    assert not tail.is_alive()
    published, entry = _pointer(scene).exists(), _next_entry(scene)

    assert (body.get("reason"), published, entry) == ("activation_unresolved", True, (0, "GEN_CAND", ""))
    _unresolved(scene, status, body, request)
    assert "(armed)" in body["error"] and git(scene.serving, "rev-parse", "HEAD") == row["layers"]["new_commit"]


def test_close_wins_a_restart_accepted_during_the_aborts_removal_arms_nothing_and_never_waits(scene, monkeypatch,
                                                                                             exit_tail):
    """The real exit tail reaches ``arm`` while this abort is removing the record (its pins can take Git's 300 s). It
    refuses at once: the removal is the one outcome, the answer is true and no pointer outlives the record."""
    row = scene.prepare()
    _unwritable_activating(monkeypatch)
    release, ran = body_adoption._release_pins, []

    def tail_mid_removal(repo, handoff):
        if not ran:
            ran.append(True)
            tail = threading.Thread(target=exit_tail, name="exit-tail", daemon=True)
            tail.start()
            tail.join(120)
            assert not tail.is_alive()  # it did not wait for this holder
        return release(repo, handoff)

    monkeypatch.setattr(body_adoption, "_release_pins", tail_mid_removal)

    status, body, request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
    published, entry = _pointer(scene).exists(), _next_entry(scene)

    assert (status, body["reason"], body["outcome"], published, entry) == (
        500, "activation_failed", "not_applied", False, (0, "GEN_OLD", ""))
    assert scene.fence == [("stop", "activation"), ("reopen", "")] and request.restarts == []
    assert body_adoption.read(scene.data) == {} and _not_armed_rows(scene)[-1] == ["handoff_busy"]
    assert update_prepare.find(row["candidate_id"])["state"] == "ready"


@pytest.mark.parametrize("remover", ["close", "discard", "boot", "replacement"])
def test_every_retirement_of_a_bound_record_refuses_an_arm_at_once(scene, monkeypatch, remover):
    """Whoever retires the record holds the helper's lock: this same thread's ``arm`` would hang if it ever waited."""
    row = _bound(scene)
    release, armed = body_adoption._release_pins, []

    def arm_mid_removal(repo, handoff):
        if not armed:
            armed.append(body_adoption.arm(scene.data, **EXIT_TAIL))
        return release(repo, handoff)

    monkeypatch.setattr(body_adoption, "_release_pins", arm_mid_removal)
    {"close": lambda: body_adoption.close_update(scene.data, "activation_failed", row["candidate_id"]),
     "discard": lambda: update_prepare.discard(row),
     "boot": lambda: body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True),
     "replacement": lambda: body_adoption.authorize_update(scene.data, row)}[remover]()

    assert (armed, _pointer(scene).exists(), _next_entry(scene)) == (["authorized"], False, (0, "GEN_OLD", ""))


def test_the_folder_and_its_lock_file_outlive_the_record_so_every_later_holder_takes_the_same_lock(scene):
    """Formerly the folder went after the lock was released, its lock file with it, and a later ``open`` made a new one
    nobody held. Now the record and its helpers go under the lock, and the folder keeps that one file for the next."""
    row = _bound(scene)
    helper = body_adoption.helper_dir(scene.data)
    lock = (helper / "lock").stat().st_ino

    closed = body_adoption.close_update(scene.data, "activation_failed", row["candidate_id"])

    assert (body_adoption.arm(scene.data, **EXIT_TAIL), closed, _pointer(scene).exists(), _next_entry(scene)) == (
        "", "", False, (0, "GEN_OLD", ""))
    assert [path.name for path in helper.iterdir()] == ["lock"] and (helper / "lock").stat().st_ino == lock
    body_adoption.authorize_update(scene.data, row)
    assert body_adoption.read(scene.data)["phase"] == "authorized" and (helper / "lock").stat().st_ino == lock


def test_arm_wins_a_close_that_read_the_record_before_it_was_armed_leaves_it_armed(scene, monkeypatch):
    """The close's own read is stale by the time it holds the lock: what was armed meanwhile stays, and it says so."""
    row = _bound(scene)
    archive, armed = body_adoption._archive, []

    def arm_first(*args):
        armed.append(body_adoption.arm(scene.data, **EXIT_TAIL))
        return archive(*args)

    monkeypatch.setattr(body_adoption, "_archive", arm_first)
    closed = body_adoption.close_update(scene.data, "activation_failed", row["candidate_id"])

    assert (armed, closed, _pointer(scene).exists(), _next_entry(scene)) == (
        ["armed"], "armed", True, (0, "GEN_CAND", ""))


@pytest.mark.parametrize("taker", ["discard", "replacement"])
def test_arm_wins_whoever_read_the_record_before_it_was_armed_is_refused(scene, monkeypatch, taker):
    """A candidate armed meanwhile is being activated: a discard would drop its switch's outcome, and a new
    authorization would overwrite the record its restart armed. Neither happens; a replacement takes back its pins."""
    row = _bound(scene)
    handoff, locked, armed = body_adoption.read(scene.data), body_switch._locked, []

    def arm_first(directory, wait=True):  # the taker read the record; the exit tail arms before it holds the lock
        if wait and not armed:
            armed.append(body_adoption.arm(scene.data, **EXIT_TAIL))
        return locked(directory, wait)

    monkeypatch.setattr(body_switch, "_locked", arm_first)
    with pytest.raises((update_prepare.PrepareRefused, body_adoption.AdoptionRefused)) as refused:
        update_prepare.discard(row) if taker == "discard" else body_adoption.authorize_update(scene.data, row)

    assert refused.value.code == {"discard": "UPDATE_ACTIVATION_PENDING", "replacement": "ADOPTION_PENDING"}[taker]
    assert armed == ["armed"] and body_adoption.read(scene.data)["id"] == handoff["id"]
    assert len(git(scene.serving, "for-each-ref", body_adoption.PIN_PREFIX).splitlines()) == 5
    assert update_prepare.find(row["candidate_id"])["state"] == "ready"
    assert _next_entry(scene) == (0, "GEN_CAND", "")


def test_a_retirement_acts_only_on_the_record_its_caller_read(scene):
    """One authorized again since then has the same phase and another identity: it stays, with its pointer and pins."""
    row = _bound(scene)
    stale = {**body_adoption.read(scene.data), "id": "replaced0000"}
    before = body_adoption.read(scene.data)

    body_adoption._archive(scene.data, stale, "abandoned:stale_read")

    assert body_adoption.read(scene.data) == before and row["candidate_id"] == before["candidate_id"]
    assert len(git(scene.serving, "for-each-ref", body_adoption.PIN_PREFIX).splitlines()) == 5


@pytest.mark.parametrize("left", ["its_lock_file", "a_new_lock_file"])
def test_close_wins_an_arm_overtaken_between_its_first_read_and_the_lock_finds_nothing(scene, monkeypatch, left):
    """What it read outside the lock proves nothing: the whole close ran in between, and only its read under the lock
    counts. The folder keeps its lock file; one made anew (as an earlier body's retirement left it) holds no record
    either."""
    row = _bound(scene)
    helper, locked, closed = body_adoption.helper_dir(scene.data), body_switch._locked, []

    def overtaken(directory, *args, **kwargs):
        if not closed and kwargs.get("wait") is False:
            monkeypatch.setattr(body_switch, "_locked", locked)
            closed.append(body_adoption.close_update(scene.data, "activation_failed", row["candidate_id"]))
            if left != "its_lock_file":
                (helper / "lock").unlink()
                (helper / "lock").touch()
        return locked(directory, *args, **kwargs)

    monkeypatch.setattr(body_switch, "_locked", overtaken)
    armed = body_adoption.arm(scene.data, **EXIT_TAIL)

    assert (armed, closed, _pointer(scene).exists(), _next_entry(scene)) == ("", [""], False, (0, "GEN_OLD", ""))


HOLD_WITH_THE_CAPTURED_HELPER = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("captured_switch", sys.argv[1] + "/switch.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
with helper._locked(sys.argv[1]):
    print("held", flush=True)
    sys.stdin.read()
"""


def test_refusal_a_lock_held_by_another_process_arms_nothing_and_a_later_arm_goes_on(scene):
    """Against the captured helper's own lock, the one the installer and every entry take. Its holder's end frees it: a
    later restart of this generation arms the same record, unchanged by the refusal."""
    _bound(scene)
    helper = body_adoption.helper_dir(scene.data)
    record = (helper / "handoff.json").read_bytes()
    holder = subprocess.Popen([sys.executable, "-c", HOLD_WITH_THE_CAPTURED_HELPER, str(helper)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline() == "held\n"
        assert body_adoption.arm(scene.data, **EXIT_TAIL) == "authorized"
        assert (helper / "handoff.json").read_bytes() == record and not _pointer(scene).exists()
        assert _not_armed_rows(scene) == [["handoff_busy"]]
    finally:
        holder.stdin.close()
        holder.wait(30)
        holder.stdout.close()
    assert body_adoption.arm(scene.data, **EXIT_TAIL) == "armed"
    assert _next_entry(scene) == (0, "GEN_CAND", "")


def test_refusal_the_next_boot_says_not_applied_and_the_candidate_may_go_again(scene):
    row = _bound(scene)
    helper = body_adoption.helper_dir(scene.data)
    with body_switch._locked(str(helper)):  # this same thread's ``arm`` would hang here if it ever waited
        assert body_adoption.arm(scene.data, **EXIT_TAIL) == "authorized"
    assert not _pointer(scene).exists() and _next_entry(scene) == (0, "GEN_OLD", "")

    result = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)
    update_prepare.record_activation(scene.data, result)

    assert result["outcome"] == "not_applied" and body_adoption.read(scene.data) == {}
    record = update_prepare.find(row["candidate_id"])
    assert record["state"] == "ready" and update_prepare.activation_check(record) == ""


def test_panic_stays_immediate_with_the_lock_held_and_free(scene):
    """Panic waits for nobody: a held lock arms nothing and writes nothing; a free one records why, as ever."""
    _bound(scene)
    helper = body_adoption.helper_dir(scene.data)
    (scene.data / "state" / "panic_stop.flag").write_text("panic")
    with body_switch._locked(str(helper)):
        assert body_adoption.arm(scene.data, **EXIT_TAIL) == "authorized"
    assert "panic_stop" not in json.dumps(body_adoption.read(scene.data)["events"]) and not _pointer(scene).exists()
    assert body_adoption.arm(scene.data, **EXIT_TAIL) == "authorized"
    handoff = body_adoption.read(scene.data)
    assert handoff["restart_bound"] is False and "panic_stop" in handoff["events"][-1]["detail"]
    assert not _pointer(scene).exists()


def test_refusal_neither_side_acts_when_the_lock_cannot_be_taken(scene):
    """No lock, no exclusion: ``arm`` publishes nothing and the close removes nothing; what the record holds says what
    stays open, and it settles at the boot once the lock can be taken."""
    row = _bound(scene)
    lock = body_adoption.helper_dir(scene.data) / "lock"
    assert lock.is_file()  # made with the helper, before any record: a full disk cannot then stop a close
    lock.unlink()
    lock.mkdir()  # ``open`` refuses it

    assert body_adoption.arm(scene.data, **EXIT_TAIL) == "" and not _pointer(scene).exists()
    assert body_adoption.close_update(scene.data, "activation_failed", row["candidate_id"]) == "authorized"
    assert body_adoption.read(scene.data)["phase"] == "authorized"
    lock.rmdir()

    assert _next_entry(scene) == (0, "GEN_OLD", "")
    assert body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)["outcome"] == "not_applied"


def test_the_windows_branch_of_a_lock_that_does_not_wait_tries_once_and_never_sleeps(tmp_path, monkeypatch):
    """Synthetic: no Windows runs here. The branch's own decision, over a stand-in ``msvcrt``."""
    attempts = []

    def locking(_fd, _mode, _size):
        attempts.append(1)
        raise PermissionError(errno.EACCES, "held")

    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "msvcrt", type("msvcrt", (), {"locking": staticmethod(locking), "LK_NBLCK": 2}))
        patch.setattr(sys, "platform", "win32")
        patch.setattr(body_switch.time, "sleep", lambda _sec: pytest.fail("a lock that does not wait never sleeps"))
        with pytest.raises(BlockingIOError):
            with body_switch._locked(str(tmp_path), wait=False):
                pytest.fail("not held")
    assert attempts == [1]


def _held_after_release(monkeypatch, name):
    """Holds the thread called ``name`` right after it releases the helper's lock, where a retirement formerly went on
    to remove the folder by path. Returns the events (released, resume)."""
    locked, released, resume = body_switch._locked, threading.Event(), threading.Event()

    @contextlib.contextmanager
    def holding(directory, wait=True):
        with locked(directory, wait):
            yield
        if threading.current_thread().name == name and not released.is_set():
            released.set()
            assert resume.wait(120)

    monkeypatch.setattr(body_switch, "_locked", holding)
    return released, resume


def _boot_thread(settle):
    settled = []
    thread = threading.Thread(target=lambda: settled.append(settle()), name="boot-settlement", daemon=True)
    thread.start()
    return thread, settled


def test_an_old_boots_retirement_never_reaches_a_new_activation_in_its_folder_and_the_next_entry_switches(
        scene, monkeypatch):
    """Whole review R1: an earlier generation's activation was interrupted before its record said activating, and this
    boot retires the handoff it left while the owner's Restart to update runs again. 05a removed the folder by path once
    its lock was released: the new activation made, bound and armed its handoff there meanwhile, that removal then took
    its helper, and every later start refused at the pointer (exit 3). Now the retirement is whole under the lock that
    creation holds too, the folder outlives both, and the earlier handoff's result leaves the new attempt's record."""
    row = scene.prepare()
    leftover = body_adoption.authorize_update(scene.data, row)
    assert body_adoption.bind_update_restart(scene.data)
    released, resume = _held_after_release(monkeypatch, "boot-settlement")
    boot, settled = _boot_thread(lambda: body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True))
    try:
        assert released.wait(120)
        status, body, _request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})
        armed, new = body_adoption.arm(scene.data, **EXIT_TAIL), body_adoption.read(scene.data)
    finally:
        resume.set()
        boot.join(120)

    assert (status, body.get("restarting"), armed, _next_entry(scene)) == (
        200, True, "armed", (0, "GEN_CAND", "")), body
    assert not boot.is_alive() and (settled[0]["outcome"], settled[0]["adoption_id"]) == ("not_applied", leftover["id"])
    record = update_prepare.find(row["candidate_id"])
    assert (record["state"], record["activation"]["adoption_id"]) == ("activating", new["id"])


def test_an_old_retirement_never_reaches_a_body_adoption_authorized_again_in_its_folder(tmp_path, monkeypatch):
    """R1 for schema 1, the same folder and lock: a task's earlier authorization retires at boot while the task
    authorizes again and its restart arms; the next entry switches to the commit it named."""
    from tests.test_body_adoption import Scene as BodyScene

    body = BodyScene(tmp_path, monkeypatch)
    body.commit()
    leftover = body.authorize()
    released, resume = _held_after_release(monkeypatch, "boot-settlement")
    boot, settled = _boot_thread(lambda: body_adoption.finalize_on_boot(body.data, body.serving, supervisor_ready=True))
    try:
        assert released.wait(120)
        again, armed = body.authorize(), body.arm()
    finally:
        resume.set()
        boot.join(120)
    entered = body.boot()

    assert (armed, entered.returncode, entered.stderr[-200:] if entered.returncode else "") == ("armed", 0, "")
    assert body.reports()[-1]["a"] == "GEN_CAND" and git(body.serving, "rev-parse", "HEAD") == body.cand
    assert not boot.is_alive() and settled[0]["adoption_id"] == leftover["id"] != again["id"]


def test_a_creation_holds_the_lock_from_its_first_helper_to_its_record_so_a_stale_retirement_waits(scene, monkeypatch):
    """A boot that read the earlier record retires it while a new authorization writes its helpers: it waits for that
    record, then leaves it. 05a wrote them outside the lock, so nothing ordered the two."""
    row = scene.prepare()
    stale = body_adoption.authorize_update(scene.data, row)
    copy, copied, proceed, created, retired = shutil.copyfile, threading.Event(), threading.Event(), [], []

    def mid_population(source, target, **kwargs):
        done = copy(source, target, **kwargs)
        if threading.current_thread().name == "creation" and pathlib.Path(target).name == "switch.py":
            copied.set()
            assert proceed.wait(120)
        return done

    monkeypatch.setattr(shutil, "copyfile", mid_population)
    creation = threading.Thread(target=lambda: created.append(body_adoption.authorize_update(scene.data, row)),
                                name="creation", daemon=True)
    retirement = threading.Thread(target=lambda: retired.append(body_adoption._archive(
        scene.data, stale, "abandoned:not_armed")), name="boot-settlement", daemon=True)
    creation.start()
    assert copied.wait(120)
    retirement.start()
    retirement.join(1)
    waited = retirement.is_alive()
    proceed.set()
    creation.join(120)
    retirement.join(120)

    assert (waited, retired, body_adoption.read(scene.data)["id"]) == (True, [True], created[0]["id"])
    assert body_adoption.bind_update_restart(scene.data) and body_adoption.arm(scene.data, **EXIT_TAIL) == "armed"
    assert _next_entry(scene) == (0, "GEN_CAND", "")


def test_arm_wins_the_boots_retirement_so_that_boot_says_nothing_and_every_client_sees_the_switch(scene, monkeypatch):
    """Whole review R2: the boot read a handoff an earlier activation left authorized and bound, and a restart's exit
    tail armed it before the boot's retirement held the lock. 05a still answered not_applied: the record went back to
    ready, its Last attempt said nothing was applied and the panel offered Restart to update while that restart was
    switching. Now a boot settles only what it closed: the record stays activating, and the switch's own boot settles
    it."""
    monkeypatch.setattr(update_progress, "_current", {})
    row = scene.prepare()
    _activates(row)
    handoff, archive, armed = body_adoption.read(scene.data), body_adoption._archive, []

    def arm_first(*args):  # the exit tail arms after the boot read the record, before its retirement holds the lock
        armed.append(body_adoption.arm(scene.data, **EXIT_TAIL))
        return archive(*args)

    monkeypatch.setattr(body_adoption, "_archive", arm_first)
    settled = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    monkeypatch.setattr(body_adoption, "_archive", archive)
    record, panel = update_prepare.find(row["candidate_id"]), _panel()

    assert (settled, armed, body_adoption.read(scene.data)["phase"]) == ({}, ["armed"], "armed")
    assert (record["state"], record["activation"], "last_activation" in update_prepare.projection()) == (
        "activating", {"adoption_id": handoff["id"]}, False)
    assert (panel["state"], panel["action"], panel["last"], panel["pill"]) == (
        "candidate_activating", None, None, "Restart requested…"), panel
    assert _next_entry(scene) == (0, "GEN_CAND", "")
    monkeypatch.setattr(sys, "_ouroboros_body_generation", None, raising=False)
    body_switch._attest_loaded(str(scene.serving), body_adoption.read(scene.data))
    adopted = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    assert (adopted["outcome"], update_prepare.find(row["candidate_id"])["state"]) == ("adopted", "activated")


def test_the_boots_retirement_wins_the_arm_so_its_not_applied_is_true_and_restart_to_update_is_offered(scene,
                                                                                                    monkeypatch):
    """The other order, the control: the boot retired the handoff, so the exit tail found it held, then gone; its word
    is true, and the record is offered again as the panel says."""
    monkeypatch.setattr(update_progress, "_current", {})
    row = scene.prepare()
    _activates(row)
    release, armed = body_adoption._release_pins, []

    def arm_mid_removal(repo, handoff):
        armed.append(body_adoption.arm(scene.data, **EXIT_TAIL))
        return release(repo, handoff)

    monkeypatch.setattr(body_adoption, "_release_pins", arm_mid_removal)
    settled = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    armed.append(body_adoption.arm(scene.data, **EXIT_TAIL))
    panel = _panel()

    assert (settled["outcome"], armed, _pointer(scene).exists()) == ("not_applied", ["authorized", ""], False)
    assert update_prepare.find(row["candidate_id"])["state"] == "ready"
    assert (panel["state"], panel["action"], panel["last"], panel["pill"]) == (
        "candidate_ready", {"id": "activate", "label": "Restart to update"}, "No update was applied.", "Update ready")
    assert _next_entry(scene) == (0, "GEN_OLD", "")


def test_an_earlier_handoffs_boot_result_never_overwrites_a_later_restart_to_update_of_the_same_candidate(
        scene, monkeypatch):
    """Whole review R2, its other half: the boot really retired the earlier handoff, but the owner's next Restart to
    update of the same candidate was admitted before the boot's result reached the record (both under the same
    lifecycle lock). That result is the earlier handoff's: the record stays the new attempt's, activating."""
    monkeypatch.setattr(update_progress, "_current", {})
    row = scene.prepare()
    leftover = body_adoption.authorize_update(scene.data, row)  # interrupted before its record said activating
    record_activation, settled = update_prepare.record_activation, []

    def after_the_new_attempt(data_dir, result):
        settled.append(result)
        _activates(row)
        return record_activation(data_dir, result)

    monkeypatch.setattr(update_prepare, "record_activation", after_the_new_attempt)
    body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    new, record = body_adoption.read(scene.data), update_prepare.find(row["candidate_id"])

    assert (record["state"], record["activation"], "last_activation" in update_prepare.projection()) == (
        "activating", {"adoption_id": new["id"]}, False)
    assert _panel()["state"] == "candidate_activating"
    assert [(result["outcome"], result.get("adoption_id")) for result in settled] == [("not_applied", leftover["id"])]
    assert body_adoption.arm(scene.data, **EXIT_TAIL) == "armed" and _next_entry(scene) == (0, "GEN_CAND", "")


@pytest.mark.parametrize("ending", ["its_activating_mark_failed", "no_restart_could_be_requested"])
def test_an_earlier_handoffs_late_result_never_overwrites_a_later_attempt_that_ended_before_any_restart(
        scene, monkeypatch, ending):
    """Whole review R2, the later attempt that ended: the owner's next Restart to update of the same candidate ended
    before any restart (its activating mark failed, or no restart could be requested), closed its own handoff and put
    the record back to ready. The boot's late result is still the earlier handoff's, retired by that boot: it changes
    neither the record that later attempt left nor the Last attempt, however alike both outcomes read."""
    monkeypatch.setattr(update_progress, "_current", {})
    row = scene.prepare()
    leftover = body_adoption.authorize_update(scene.data, row)  # interrupted before its record said activating
    record_activation, mark, named, ended, settled = update_prepare.record_activation, update_prepare.mark, [], {}, []

    def marking(live, **fields):  # the later attempt's own writes of its record, as they are made
        named.append((fields.get("activation") or {}).get("adoption_id"))
        if ending == "its_activating_mark_failed" and fields.get("state") == "activating":
            raise OSError(errno.ENOSPC, "no space left on device")
        return mark(live, **fields)

    def after_the_later_attempt(data_dir, result):
        settled.append(result)
        with pytest.MonkeyPatch.context() as later:
            later.setattr(update_prepare, "mark", marking)
            answer = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]},
                            restart=ending != "no_restart_could_be_requested")
        ended.update(answer=answer[:2], handoff=body_adoption.read(scene.data),
                     record=update_prepare.find(row["candidate_id"]), last=update_prepare.projection().get(
                         "last_activation"))
        return record_activation(data_dir, result)

    monkeypatch.setattr(update_prepare, "record_activation", after_the_later_attempt)
    body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    record, later = update_prepare.find(row["candidate_id"]), next(attempt for attempt in named if attempt)

    status, body = ended["answer"]
    assert (status, body["reason"], body["outcome"], ended["handoff"]) == (500, {
        "its_activating_mark_failed": "activation_failed", "no_restart_could_be_requested": "restart_unavailable"}[
        ending], "not_applied", {}), body
    assert (record["state"], record.get("activation"), record.get("reason")) == (
        ended["record"]["state"], ended["record"].get("activation"), ended["record"].get("reason"))
    assert (update_prepare.projection().get("last_activation"), _panel()["last"]) == (ended["last"], None)
    assert (record.get("activation") or {}).get("adoption_id") == later != leftover["id"]
    assert [(result["outcome"], result.get("adoption_id")) for result in settled] == [("not_applied", leftover["id"])]


class _ProcessEnded(BaseException):
    """The process ends here: no ``except Exception`` of the activation runs, as none would."""


def _activation_ending_at(row, boundary):
    """The real Restart to update whose process ends at ``boundary`` of its creation."""
    mark, record = update_prepare.mark, body_switch.record

    def marking(live, **fields):
        naming = "state" not in fields and (fields.get("activation") or {}).get("adoption_id")
        if naming and boundary == "as_the_record_would_name_it":
            raise _ProcessEnded()
        done = mark(live, **fields)
        if naming and boundary == "right_after_the_record_named_it":
            raise _ProcessEnded()
        return done

    def recording(directory, handoff, phase, detail=""):
        done = record(directory, handoff, phase, detail)
        if boundary == "right_after_its_handoff_is_durable" and str(detail).startswith("restart reason"):
            raise _ProcessEnded()
        return done

    with pytest.MonkeyPatch.context() as ending:
        ending.setattr(update_prepare, "mark", marking)
        ending.setattr(body_switch, "record", recording)
        with pytest.raises(_ProcessEnded):
            _apply({"strategy": "activate", "candidate_id": row["candidate_id"]})


@pytest.mark.parametrize("boundary", ["right_after_its_handoff_is_durable", "as_the_record_would_name_it",
                                      "right_after_the_record_named_it"])
@pytest.mark.parametrize("earlier", ["settled_by_its_boot", "ended_before_its_restart", "still_authorized"])
def test_a_process_ending_during_a_creation_never_leaves_a_handoff_its_record_does_not_name(
        scene, monkeypatch, earlier, boundary):
    """Whole review R2, its own attempt: a ready candidate keeps its earlier attempt's name, and the process of the next
    Restart to update ends during its creation. Formerly the handoff became durable first: the next boot really retired
    it, its result read as an earlier one's and was dropped, and the Last attempt lost its newest outcome. Now the
    record names the attempt before its handoff exists: what the next boot inherits is what its record names, and with
    nothing inherited there is nothing to settle. Either way the next Restart to update goes on."""
    monkeypatch.setattr(update_progress, "_current", {})
    row = scene.prepare()
    if earlier == "settled_by_its_boot":
        _activates(row)
        body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    elif earlier == "ended_before_its_restart":
        status, body, _request = _apply({"strategy": "activate", "candidate_id": row["candidate_id"]}, restart=False)
        assert (status, body["reason"]) == (500, "restart_unavailable"), body
    else:
        _activation_ending_at(row, "right_after_its_handoff_is_durable")
    before, older = update_prepare.find(row["candidate_id"]), body_adoption.read(scene.data).get("id")
    last = update_prepare.projection().get("last_activation")
    assert before["state"] == "ready"

    _activation_ending_at(row, boundary)
    inherited = body_adoption.read(scene.data)  # what the next boot finds; what the creation replaced is never there
    settled = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    record, newest, panel = update_prepare.find(row["candidate_id"]), update_prepare.projection().get(
        "last_activation"), _panel()

    assert inherited.get("id") != older or older is None
    assert record["state"] == "ready" and (panel["state"], panel["action"]["id"]) == ("candidate_ready", "activate")
    if inherited:
        own = inherited["id"]
        assert (settled["outcome"], settled["adoption_id"], record["activation"].get("adoption_id"),
                record["activation"].get("outcome")) == ("not_applied", own, own, "not_applied"), record["activation"]
        assert newest != last and newest["outcome"] == "not_applied" and panel["last"] == "No update was applied."
    else:
        assert (settled, newest) == ({}, last)

    _activates(row)
    assert body_adoption.arm(scene.data, **EXIT_TAIL) == "armed" and _next_entry(scene) == (0, "GEN_CAND", "")


def test_a_creation_names_its_attempt_after_what_it_replaces_is_gone_and_before_its_handoff_exists(scene):
    """The order that makes the above hold: no name beside a handoff still live, no handoff its record does not name."""
    row = scene.prepare()
    body_adoption.authorize_update(scene.data, row)  # an earlier one, still authorized when this creation starts
    mark, record, seen = update_prepare.mark, body_switch.record, []

    def marking(live, **fields):
        named = (fields.get("activation") or {}).get("adoption_id")
        if named and "state" not in fields:
            seen.append(("named", named, body_adoption.read(scene.data).get("id")))
        return mark(live, **fields)

    def recording(directory, handoff, phase, detail=""):
        done = record(directory, handoff, phase, detail)
        if str(detail).startswith("restart reason"):
            seen.append(("durable", handoff["id"], handoff["id"]))
        return done

    with pytest.MonkeyPatch.context() as watching:
        watching.setattr(update_prepare, "mark", marking)
        watching.setattr(body_switch, "record", recording)
        _activates(row)

    new = body_adoption.read(scene.data)["id"]
    assert seen == [("named", new, None), ("durable", new, new)]


@pytest.mark.parametrize("form", ["cleared_by_an_abort", "settled_without_a_name", "a_result_naming_none"])
def test_a_record_or_a_result_of_an_earlier_body_that_names_no_attempt_still_settles(scene, form):
    """The forms an earlier body left: its abort cleared the name and its settlement kept none, so nothing there names
    another attempt, and a handoff's own result lands as it always did. Only two names that differ decline."""
    row = scene.prepare()
    handoff = body_adoption.authorize_update(scene.data, row)
    if form != "a_result_naming_none":
        update_prepare.mark(row, activation={} if form == "cleared_by_an_abort" else {"outcome": "returned"})
    result = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)
    if form == "a_result_naming_none":
        result.pop("adoption_id")

    update_prepare.record_activation(scene.data, result)

    record = update_prepare.find(row["candidate_id"])
    assert (record["state"], record["activation"]["outcome"]) == ("ready", "not_applied")
    assert record["activation"].get("adoption_id") == (None if form == "a_result_naming_none" else handoff["id"])
    assert update_prepare.projection()["last_activation"]["outcome"] == "not_applied"
