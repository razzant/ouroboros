"""What an Update's activation is bound to: real Git, the miniature body, the real switch helper.

The import probe the switch runs carries the installation's scope marker and nothing of a task or an engine
run; an activation the restart never applied is offered again while what it was prepared on holds; a
candidate prepared on another serving interpreter is stale at activation, never switched under it; once this
generation's stop began, authorization and abandonment start no Git and pin nothing; an entry that cannot
record itself imports nothing, so no later return reads its data as untouched.
"""
from __future__ import annotations

import contextlib
import errno
import importlib.util
import os
import subprocess
import sys

import pytest

from ouroboros import body_adoption, body_switch
from supervisor import update_prepare
from tests.body_candidate_support import git
from tests.test_update_activation import STYLE_NEW, Scene, _passive_base

pytestmark = pytest.mark.serial  # real Git and child processes


@pytest.fixture
def scene(tmp_path, monkeypatch):
    return Scene(tmp_path, monkeypatch)


def test_the_import_probe_carries_the_installation_marker_and_no_task_or_run_marker(tmp_path, monkeypatch):
    """Under the detached app installer (its own session) no process group of the entry holds the import probe:
    Panic's census finds it by the installation's scope marker, which the probe's allowlisted environment keeps,
    and only that one (a task's or an engine run's marker would end it with that work)."""
    from ouroboros import process_census

    for name in [name for name in os.environ if process_census.is_scope_name(name)]:
        monkeypatch.delenv(name)  # a run inside an installation or an engine run carries its own markers
    install, task, run = (process_census.mint_marker(prefix) for prefix in (
        process_census.INSTALL_SCOPE_PREFIX, process_census.TASK_SCOPE_PREFIX, process_census.RUN_SCOPE_PREFIX))
    for name in (install, task, run):
        monkeypatch.setenv(name, "1")
    seen = ("import os, sys; scopes = sorted(name for name in os.environ if '_SCOPE_' in name); "
            f"sys.exit(0 if scopes == [{install!r}] else 3)")

    assert body_switch._probe(str(tmp_path), {"id": "probe1", "probe_command": [sys.executable, "-c", seen]}) == ""
    env = body_switch._probe_env({"id": "probe1"}, str(tmp_path / "scratch"))
    assert process_census.installation_scope_entries(env) == {install: "1"}
    assert process_census.scope_entries(env) == {install: "1"}  # the census grammar: exactly one marker crosses
    assert body_switch._INSTALL_SCOPE_PREFIX == process_census.INSTALL_SCOPE_PREFIX  # the stdlib restatement


@pytest.mark.parametrize("moved", [False, True], ids=["intact", "local_changed"])
def test_an_activation_that_was_never_applied_is_offered_again_while_its_base_holds(scene, moved):
    """``not_applied`` (the restart's exit tail never armed, or the transition was abandoned before any switch):
    nothing changed, so an intact candidate whose base and environment still hold is ``ready`` again, its Last
    attempt kept, and Restart to update may retry without preparing again. A base that moved makes it stale."""
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    row = scene.prepare()
    handoff = body_adoption.authorize_update(scene.data, row)
    update_prepare.mark(row, state="activating", activation={"adoption_id": handoff["id"]})
    if moved:
        (scene.serving / "ouroboros" / "mod_b.py").write_text("GEN = 'GEN_OLD'\nLATER = 1\n")

    result = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)
    update_prepare.record_activation(scene.data, result)

    record = update_prepare.find(row["candidate_id"])
    assert result["outcome"] == "not_applied" and record["activation"]["outcome"] == "not_applied"
    assert (record["state"], record["reason"]) == (("stale", "local_changed") if moved
                                                   else ("ready", "activation:not_applied"))
    assert update_prepare.projection()["last_activation"]["outcome"] == "not_applied"
    if not moved:
        assert update_prepare.activation_check(record) == ""  # Restart to update may go again as prepared


def test_a_candidate_prepared_on_another_python_is_stale_at_activation(scene, monkeypatch):
    """The serving runtime is part of what a candidate was prepared and checked on: one the owner now serves on
    another interpreter (or SQLite) is ``stale:environment_changed`` at activation, never switched under it."""
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    row = scene.prepare()
    real = body_switch.runtime_identity()
    monkeypatch.setattr(body_switch, "runtime_identity", lambda: {**real, "executable": "/elsewhere/bin/python3"})

    why = update_prepare.activation_check(row)

    assert why.startswith("environment_changed: the Python that serves changed (executable)"), why
    assert update_prepare.find(row["candidate_id"])["state"] == "stale"


def _pins(scene):
    """``{ref name below the adoption prefix: object}`` of every adoption pin in the serving repository."""
    listing = git(scene.serving, "for-each-ref", "--format=%(refname) %(objectname)", body_adoption.PIN_PREFIX)
    return {ref[len(body_adoption.PIN_PREFIX):]: sha for ref, sha in (line.split() for line in listing.splitlines())}


@contextlib.contextmanager
def _recorded_starts():
    """Every child this process starts inside the block, with whether this generation's stop door had opened."""
    from ouroboros import owned_shutdown

    started, real = [], subprocess.Popen

    class Recorded(real):
        def __init__(self, args, *rest, **kwargs):
            started.append((list(args) if isinstance(args, (list, tuple)) else [args], owned_shutdown.stop_begun()))
            super().__init__(args, *rest, **kwargs)

    subprocess.Popen = Recorded
    try:
        yield started
    finally:
        subprocess.Popen = real  # the test's own reads after the block start their children as usual


@pytest.mark.parametrize("door", ["shut", "opens_before_the_pins"])
def test_authorizing_an_update_pins_through_the_tracked_admission_and_changes_nothing_after_the_stop(
        scene, monkeypatch, door):
    """Producer -> Stop -> next spawn in the activation's authorization, which replaces an earlier authorized
    handoff of the same candidate: the five new pins are set, then the earlier ones removed, each in ONE ref
    transaction through the shell's tracked admission. When an owner Restart enters this generation's stop door
    after authorization's last read, no child starts: ``ADOPTION_STOPPED``, no new pin, and the earlier handoff, its
    helper and its pins stay exactly as they were (the activation then answers ``not_applied``)."""
    from ouroboros import owned_shutdown

    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())  # this test's own generation
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    row = scene.prepare()
    earlier = body_adoption.authorize_update(scene.data, row)
    helper = body_adoption.helper_dir(scene.data)
    before, kept = _pins(scene), {path.name: path.read_bytes() for path in helper.iterdir() if path.is_file()}
    assert set(before) == {earlier["id"], *(f"{earlier['id']}-{name}" for name in body_adoption._LAYER_PINS)}
    real_lock, reads = body_adoption.dependency_lock, []

    def lock_then_restart(*args, **kwargs):  # authorization's last read; the owner's Restart enters the door after it
        found = real_lock(*args, **kwargs)
        reads.append(found)
        if door != "shut" and len(reads) == 2:
            owned_shutdown.begin_owned_stop(scene.data, cause="owner_restart")
        return found

    monkeypatch.setattr(body_adoption, "dependency_lock", lock_then_restart)
    with _recorded_starts() as started:
        try:
            handoff, refused = body_adoption.authorize_update(scene.data, row), None
        except body_adoption.AdoptionRefused as exc:
            handoff, refused = None, exc

    if door == "shut":
        assert refused is None and handoff["id"] != earlier["id"]
        assert body_adoption.read(scene.data)["id"] == handoff["id"]
        assert [args for args, _stopping in started if "update-ref" in args] == [["git", "update-ref", "--stdin"]] * 2
        assert _pins(scene) == {handoff["id"]: row["layers"]["new_commit"], **{
            f"{handoff['id']}-{name}": row["layers"][layer] for name, layer in body_adoption._LAYER_PINS.items()}}
        return
    assert refused.code == "ADOPTION_STOPPED" and refused.text.startswith(
        "this server generation's stop has begun"), refused.text
    assert len(reads) == 2 and [args for args, stopping in started if stopping] == []
    assert _pins(scene) == before
    assert body_adoption.read(scene.data) == earlier
    assert {path.name: path.read_bytes() for path in helper.iterdir() if path.is_file()} == kept


def test_an_abandon_after_the_stop_starts_no_git_closes_the_handoff_and_keeps_its_pins(scene, monkeypatch):
    """Producer -> Stop -> next spawn when an authorized handoff is abandoned once this generation's stop door
    opened (an installer that was then refused to start, or a refused restart): the removal of its pins is one ref
    transaction the tracked admission refuses, so no child starts and the pins stay (they keep objects only), while
    the handoff still closes, so nothing can arm it."""
    from ouroboros import owned_shutdown

    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    row = scene.prepare()
    handoff = body_adoption.authorize_update(scene.data, row)
    before = _pins(scene)
    owned_shutdown.begin_owned_stop(scene.data, cause="owner_restart")

    with _recorded_starts() as started:
        body_adoption.abandon(scene.data, "native_hand_over_failed", candidate_id=row["candidate_id"])

    # The one child is a read: the stdlib switch helper's own Git-dir lookup for the pointer (``body_switch.git_dir``).
    assert [args for args, _stopping in started] == [["git", "rev-parse", "--absolute-git-dir"]]
    assert body_adoption.read(scene.data) == {}
    assert [path.name for path in body_adoption.helper_dir(scene.data).iterdir()] == ["lock"]
    assert _pins(scene) == before and len(before) == 5 and handoff["id"] in before


# One start under a launcher (this test process's own children never inherit its marker): it exits 42 after the switch.
_MANAGED_START = ("import os, subprocess, sys; sys.exit(subprocess.call([sys.executable, 'server.py'], "
                  "env=dict(os.environ, OUROBOROS_MANAGED_BY_LAUNCHER='1')))")


def _switched_without_an_entry(scene, files=None):
    """The real switch of an armed Update under a launcher: its exit 42 leaves the tree switched, no entry recorded."""
    scene.official(files or {"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    if not files:
        scene.local_work(commit=False)
    scene.activate(scene.prepare())
    first = subprocess.run([sys.executable, "-c", _MANAGED_START], cwd=str(scene.serving), capture_output=True,
                           text=True, timeout=120)
    assert first.returncode == 42, first.stderr
    handoff = scene.handoff()
    assert handoff["phase"] == "switched" and handoff["entries"] == [] and "census" not in handoff
    return handoff


@contextlib.contextmanager
def _record_refused(scene):
    """The next record really fails in a cold entry: nothing can be created beside it (the lock file already exists)."""
    helper = body_adoption.helper_dir(scene.data)
    helper.chmod(0o555)
    try:
        if os.access(helper, os.W_OK):
            pytest.skip("this user writes through a read-only folder, so the real write cannot fail here")
        yield
    finally:
        helper.chmod(0o755)


def _captured_helper(scene, monkeypatch, argv0):
    """The helper the hook runs (the captured copy, beside its record) for an entry named ``argv0``, its writes full."""
    spec = importlib.util.spec_from_file_location("captured_switch", body_adoption.helper_dir(scene.data) / "switch.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setattr(sys, "argv", [str(scene.serving / argv0)])
    monkeypatch.setattr(sys, "_ouroboros_body_generation", None, raising=False)

    def full(_directory, _handoff):
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(helper, "write_handoff", full)
    return helper


def test_an_entry_that_cannot_record_itself_imports_nothing_and_the_next_start_retries(scene):
    """Whole review, F1: the first entry's record failed, the entry only logged it and imported the new tree, and its
    data writes later read as "no successor entered": ``compatible``. Now that entry stops before its imports with the
    record as it was, so that reading stays true. The next start records it and the census guards what follows."""
    _switched_without_an_entry(scene)
    settings = scene.data / "settings.json"
    settings.write_text('{"schema": 1}')
    before = body_switch.data_census(str(scene.data))

    with _record_refused(scene):
        refused = scene.boot()

    assert refused.returncode == body_switch.STUCK_EXIT_CODE, refused.stderr
    assert "the update's entry could not be recorded (PermissionError" in refused.stderr
    assert "the next start retries it" in refused.stderr and scene.reports() == []  # no body module was imported
    stored = scene.handoff()
    assert stored["phase"] == "switched" and stored["entries"] == []
    assert "census" not in stored and "return" not in stored
    assert body_switch.data_census(str(scene.data)) == before

    retried = scene.boot()

    assert retried.returncode == 0, retried.stderr
    assert [report["a"] for report in scene.reports()] == ["GEN_CAND"]
    stored = scene.handoff()
    assert [entry["kind"] for entry in stored["entries"]] == ["server"] and stored["census"]["settings.json"]
    settings.write_text('{"schema": 2}')  # the successor migrates its data: the probe's own normal control
    assert body_switch.qualify_return(str(scene.serving), stored) == (
        "unknown", ["successor_entered:server", "successor_changed_data:settings.json"])


@pytest.mark.parametrize("argv0", ["server.py", "launcher.py", "cli.py"])
def test_every_kind_of_entry_stops_at_a_first_record_it_cannot_make(scene, monkeypatch, argv0):
    """A full disk, through the captured helper: a module CLI writes data like a server, and nothing is attested."""
    handoff = _switched_without_an_entry(scene)
    helper = _captured_helper(scene, monkeypatch, argv0)

    with pytest.raises(SystemExit) as stopped:
        helper.main(str(scene.serving))

    assert stopped.value.code == body_switch.STUCK_EXIT_CODE and sys._ouroboros_body_generation is None
    assert scene.handoff() == handoff


def test_an_entry_after_a_recorded_one_still_starts_when_its_own_record_fails(scene, monkeypatch, capsys):
    """The boundary already holds, so a later record's failure refuses nothing: this entry starts as it did, its
    generation attested, and a return still reads the recorded entry."""
    _switched_without_an_entry(scene)
    assert scene.boot().returncode == 0
    stored = scene.handoff()
    helper = _captured_helper(scene, monkeypatch, "server.py")

    helper.main(str(scene.serving))

    assert sys._ouroboros_body_generation["sha"] == stored["cand"]
    assert "update entry not recorded: OSError" in capsys.readouterr().err
    assert scene.handoff() == stored
    assert body_switch.qualify_return(str(scene.serving), stored) == ("unknown", ["successor_entered:server"])


@pytest.mark.skipif(os.name == "nt", reason="the app transition is macOS-only: process_gone reads POSIX process facts")
def test_an_app_intent_on_record_is_no_entry_for_the_return_that_reads_only_entries_of_the_tree(scene, monkeypatch):
    """``_app_not_exchanged`` looks past the installer's own intent. An entry under a live installer that imported
    unrecorded would make the later return of the previous app read "nothing ran": it stops there instead, and once the
    installer is gone that return is true and happens."""
    _switched_without_an_entry(scene)
    app = scene.tmp / "Apps" / "Ouroboros.app"
    (app / "Contents").mkdir(parents=True)
    stored = scene.handoff()
    stored["entries"] = [{"kind": "app", "argv0": "Ouroboros.app", "pid": 1, "ready": False}]
    stored["native"] = {"installed": str(app), "previous_identity": body_switch.bundle_identity(str(app)),
                        "installer": {"pid": os.getpid()}}  # alive: unknown is never gone
    body_switch.write_handoff(str(body_adoption.helper_dir(scene.data)), stored)
    helper = _captured_helper(scene, monkeypatch, "launcher.py")

    with pytest.raises(SystemExit) as stopped:
        helper.main(str(scene.serving))

    assert stopped.value.code == body_switch.STUCK_EXIT_CODE and scene.handoff() == stored
    stored["native"]["installer"] = {"pid": 0}
    body_switch.write_handoff(str(body_adoption.helper_dir(scene.data)), stored)
    returned = scene.boot()
    assert returned.returncode == 0, returned.stderr
    assert scene.handoff()["return"]["trigger"] == "app_not_exchanged" and scene.reports()[-1]["a"] == "GEN_OLD"


def test_a_refused_first_entry_leaves_the_proven_stylesheet_return_as_it_was(scene):
    """The compatible return is untouched: refused once, the stylesheet-only successor then enters, writes data and
    fails, and the next entry returns the previous composition on that evidence."""
    _passive_base(scene)
    _switched_without_an_entry(scene, {"web/style.css": STYLE_NEW})
    with _record_refused(scene):
        assert scene.boot().returncode == body_switch.STUCK_EXIT_CODE
    assert scene.boot(TOY_WRITE_THEN_FAIL="1").returncode != 0
    assert body_switch.qualify_return(str(scene.serving), scene.handoff()) == ("compatible", [])

    returned = scene.boot()

    assert returned.returncode == 0, returned.stderr
    decision = scene.handoff()["return"]
    assert (decision["outcome"], decision["trigger"]) == ("returned", "successor_start_failed")
    assert [row["form"] for row in scene.reports()[-1]["read"]] == ["v1"]  # the previous generation read what it wrote


def test_panic_still_forbids_every_automatic_step_at_a_refused_entry(scene):
    """Under Panic an entry that cannot record itself returns nothing and imports nothing; a later entry whose record
    fails returns nothing either."""
    handoff = _switched_without_an_entry(scene)
    panic = scene.data / "state" / "panic_stop.flag"
    panic.write_text("panic")
    with _record_refused(scene):
        refused = scene.boot()
    assert refused.returncode == body_switch.STUCK_EXIT_CODE and scene.reports() == []
    assert git(scene.serving, "rev-parse", "HEAD") == handoff["cand"] and "return" not in scene.handoff()
    panic.unlink()
    assert scene.boot().returncode == 0  # entered
    panic.write_text("panic")
    with _record_refused(scene):
        later = scene.boot()
    assert later.returncode == 0, later.stderr
    assert git(scene.serving, "rev-parse", "HEAD") == handoff["cand"] and "return" not in scene.handoff()
    assert body_switch.qualify_return(str(scene.serving), scene.handoff()) == ("refused", ["panic_stop"])
