"""What an Update's activation is bound to: real Git, the miniature body, the real switch helper.

The import probe the switch runs carries the installation's scope marker and nothing of a task or an engine
run; an activation the restart never applied is offered again while what it was prepared on holds; a
candidate prepared on another serving interpreter is stale at activation, never switched under it; once this
generation's stop began, authorization and abandonment start no Git and pin nothing.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys

import pytest

from ouroboros import body_adoption, body_switch
from supervisor import update_prepare
from tests.body_candidate_support import git
from tests.test_update_activation import Scene

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
    assert body_adoption.read(scene.data) == {} and not body_adoption.helper_dir(scene.data).exists()
    assert _pins(scene) == before and len(before) == 5 and handoff["id"] in before
