"""An environment selection interrupted between its two renames, resumed by real cold entries.

The switch helper moves the serving packages aside, then the prepared generation into the one
path every launcher starts the body with (``body_switch._select_environment``). A process that
dies between the two renames (a Panic during activation, a power loss) leaves that path empty,
and the next start makes it again before any hook resumes: the packaged launcher's user base
(``launcher_bootstrap.embedded_python_env``) or a fresh ``--runtime-venv``
(``runtime_venv.prepare_runtime_venv``). The resume tells the recorded directories apart from
that one by the identities authorization recorded: the made directory is kept aside, never
selected, and a return then restores exactly the previous generation's packages. A selection
that cannot complete returns the whole transition before any entry. The fault point is the
helper's own (``OUROBOROS_BODY_SWITCH_FAIL_AT=mid_environment``); entries are real processes.
"""
from __future__ import annotations

import contextlib
import errno
import os
import pathlib
import shutil
import stat
import subprocess
import types

import pytest

from ouroboros import body_adoption, body_switch, launcher_bootstrap, runtime_venv
from supervisor import update_prepare
from tests.body_candidate_support import git
from tests.test_update_activation import Scene

pytestmark = pytest.mark.serial  # real cold-entry processes


@pytest.fixture
def scene(tmp_path, monkeypatch):
    return Scene(tmp_path, monkeypatch)


def _ready_lock_change(scene, kind, current, *, breaks):
    """A prepared, armed Update whose new lock gets its own packages (``lib/toydep-2.0`` in the generation)."""
    (current / "lib").mkdir(parents=True)
    (current / "lib" / "toydep-1.0").write_text("old\n")
    scene.monkeypatch.setattr(update_prepare, "serving_environment", lambda: (kind, current))

    def install(target, lock, installed_kind, python=None):  # pip into the generation directory only
        (pathlib.Path(target) / "lib" / "toydep-1.0").unlink()
        (pathlib.Path(target) / "lib" / "toydep-2.0").write_text("new\n")
        return {"returncode": 0}

    scene.monkeypatch.setattr(update_prepare, "_install_lock", install)
    files = {"requirements-runtime.lock": "toydep==2.0\n", "ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"}
    if breaks:  # the new tree does not import on its own packages: the scratch probe fails before any entry
        files["server.py"] = (scene.serving / "server.py").read_text().replace(
            "import ouroboros  #", f"if os.path.exists({str(current / 'lib' / 'toydep-2.0')!r}):\n"
            "    raise SystemExit('the new packages break this tree')\nimport ouroboros  #")
    scene.official(files)
    scene.local_work(commit=False)
    row = scene.prepare()
    assert row["state"] == "ready", row.get("reason")
    handoff = scene.activate(row)
    assert handoff["environment"]["identity"]["current"] and handoff["environment"]["identity"]["generation"]
    return row, handoff, pathlib.Path(row["environment"]["generation"]["user_base"])


def _recreate_user_base(scene, current):
    launcher_bootstrap.embedded_python_env(scene.data)  # the packaged launcher's fixed user base, made again
    assert current == scene.data / "state" / "python-userbase"


def _recreate_runtime_venv(scene, current):
    runtime_venv.prepare_runtime_venv(current)  # a launcher's --runtime-venv at an absent path: a fresh venv


@pytest.mark.parametrize("breaks", [False, True], ids=["selected", "returned"])
@pytest.mark.parametrize("kind", ["user_site", "venv"])
def test_a_selection_interrupted_between_its_renames_resumes_by_identity_and_keeps_what_was_made(
        scene, kind, breaks):
    current = scene.data / "state" / "python-userbase" if kind == "user_site" else scene.tmp / "runtime-venv"
    row, handoff, generation = _ready_lock_change(scene, kind, current, breaks=breaks)
    previous = pathlib.Path(handoff["environment"]["previous"])

    crashed = scene.boot(OUROBOROS_BODY_SWITCH_FAIL_AT="mid_environment")

    assert crashed.returncode == 137, crashed.stderr
    assert (scene.handoff()["phase"], scene.handoff()["environment"]["state"]) == ("switching", "selecting")
    assert not current.exists() and (previous / "lib" / "toydep-1.0").exists()
    assert (generation / "lib" / "toydep-2.0").exists()
    (_recreate_user_base if kind == "user_site" else _recreate_runtime_venv)(scene, current)
    made = sorted(path.name for path in current.iterdir())
    assert "lib" not in made or not (current / "lib" / "toydep-2.0").exists()  # neither recorded generation

    resumed = scene.boot()

    assert resumed.returncode == 0, resumed.stderr
    handoff = scene.handoff()
    kept = pathlib.Path(f"{current}.foreign-{handoff['id']}")
    assert handoff["environment"]["foreign"] == [str(kept)]
    assert sorted(path.name for path in kept.iterdir()) == made  # kept as it was made, never deleted or selected
    if not breaks:
        assert (handoff["phase"], handoff["environment"]["state"]) == ("switched", "selected")
        assert sorted(path.name for path in (current / "lib").iterdir()) == ["toydep-2.0"]
        assert (previous / "lib" / "toydep-1.0").exists() and not generation.exists()
        assert scene.reports()[-1]["a"] == "GEN_CAND"
        return
    # The probe failed on the resumed selection: files, index, branch and the exact previous packages return.
    assert (handoff["return"]["outcome"], handoff["return"]["trigger"]) == ("returned", "import_probe_failed")
    assert (handoff["phase"], handoff["environment"]["state"]) == ("returned", "restored")
    assert sorted(path.name for path in (current / "lib").iterdir()) == ["toydep-1.0"]
    assert (generation / "lib" / "toydep-2.0").exists() and not previous.exists()
    assert git(scene.serving, "rev-parse", "HEAD") == row["layers"]["old_commit"]
    assert [report["a"] for report in scene.reports()] == ["GEN_OLD"]


def test_a_selection_that_cannot_complete_returns_the_whole_transition_before_any_entry(scene):
    current = scene.data / "state" / "python-userbase"
    row, handoff, generation = _ready_lock_change(scene, "user_site", current, breaks=False)
    shutil.rmtree(generation)
    generation.mkdir()  # something else holds the generation's name now: not the prepared packages

    first = scene.boot()

    assert first.returncode == 0, first.stderr  # no refused start (exit 3) for the launcher's crash fuse to count
    handoff = scene.handoff()
    assert (handoff["return"]["outcome"], handoff["return"]["trigger"]) == ("returned", "environment_selection_failed")
    assert (handoff["phase"], handoff["environment"]["state"], handoff["entries"]) == ("returned", "restored", [])
    assert sorted(path.name for path in (current / "lib").iterdir()) == ["toydep-1.0"]
    assert generation.is_dir() and not any(generation.iterdir())  # the other directory is left as found
    assert git(scene.serving, "rev-parse", "HEAD") == row["layers"]["old_commit"]
    assert [report["a"] for report in scene.reports()] == ["GEN_OLD"]


def test_serving_packages_that_cannot_move_aside_are_never_set_aside_and_return_in_place(tmp_path):
    """The previous generation's name is taken by something else: the selection stops (the switch then returns
    the whole transition), the serving packages never leave the fixed path and the other directory stays."""
    from ouroboros import body_switch

    current, generation = tmp_path / "python-userbase", tmp_path / "python-userbase.update-c1"
    (current / "lib" / "toydep-1.0").mkdir(parents=True)
    (generation / "lib" / "toydep-2.0").mkdir(parents=True)
    taken = tmp_path / "python-userbase.previous-a1"
    taken.mkdir()
    helper = tmp_path / "helper"
    helper.mkdir()
    handoff = {"id": "a1", "phase": "switching", "environment": {
        "state": "prepared", "kind": "user_site", "current": str(current), "generation": str(generation),
        "previous": str(taken), "identity": {"current": body_switch._path_identity(current),
                                             "generation": body_switch._path_identity(generation)}}}

    with pytest.raises(body_switch._Stuck):
        body_switch._select_environment(str(helper), handoff)

    assert (current / "lib" / "toydep-1.0").is_dir() and not handoff["environment"].get("foreign")
    body_switch._restore_environment(str(helper), handoff)
    assert handoff["environment"]["state"] == "restored" and (current / "lib" / "toydep-1.0").is_dir()
    assert (generation / "lib" / "toydep-2.0").is_dir() and taken.is_dir() and not any(taken.iterdir())


# A real refusal of a directory rename by the kernel: the owner-settable immutable flag (macOS, BSD) refuses
# rename(2) with EPERM, as Windows refuses one with WinError 5 while a process works inside the directory.
needs_refusal = pytest.mark.skipif(not (hasattr(os, "chflags") and hasattr(stat, "UF_IMMUTABLE")),
                                   reason="a real refused rename needs the owner-settable immutable flag")


@pytest.fixture
def held():
    """Directories the system refuses to move during one test; always released, so their scratch can go."""
    paths = []

    def hold(path):
        os.chflags(path, stat.UF_IMMUTABLE)
        paths.append(path)

    def release(path):
        os.chflags(path, 0)
        paths.remove(path)

    yield types.SimpleNamespace(hold=hold, release=release)
    for path in paths:
        with contextlib.suppress(OSError):
            os.chflags(path, 0)


def _settle(scene, row):
    """The boot's own settlement of the returned handoff (``settle_on_boot``'s two steps), then the live record."""
    update_prepare.record_activation(scene.data, body_adoption.finalize_on_boot(scene.data, scene.serving,
                                                                                supervisor_ready=True))
    return update_prepare.find(row["candidate_id"])


@needs_refusal
def test_a_refused_move_of_the_serving_packages_returns_before_any_entry_and_restart_to_update_retries(scene, held):
    current = scene.data / "state" / "python-userbase"
    row, handoff, generation = _ready_lock_change(scene, "user_site", current, breaks=False)
    held.hold(current)  # the system refuses to move the packages the old generation serves from

    first = scene.boot()

    assert first.returncode == 0, first.stderr  # no refused start (exit 3) for the launcher's crash fuse to count
    handoff = scene.handoff()
    decision = handoff["return"]
    assert (decision["outcome"], decision["trigger"], decision["retry"]) == ("returned", "environment_selection_failed",
                                                                             True)
    assert (decision["error"]["step"], decision["error"]["errno"]) == ("serving_aside", errno.EPERM)
    assert decision["error"]["filename"] == str(current)  # the system's own words, never a guessed holder
    assert (handoff["phase"], handoff["environment"]["state"], handoff["entries"]) == ("returned", "restored", [])
    assert sorted(path.name for path in (current / "lib").iterdir()) == ["toydep-1.0"]
    assert (generation / "lib" / "toydep-2.0").exists() and not pathlib.Path(handoff["environment"]["previous"]).exists()
    assert git(scene.serving, "rev-parse", "HEAD") == row["layers"]["old_commit"]
    assert [report["a"] for report in scene.reports()] == ["GEN_OLD"]  # this entry went on with the old tree
    again = _settle(scene, row)
    assert again["state"] == "ready" and update_prepare.projection()["actions"] == ["activate", "discard_candidate"]
    last = update_prepare.projection()["last_activation"]
    assert (last["outcome"], last["trigger"]) == ("returned", "environment_selection_failed") and str(current) in last[
        "detail"]
    held.release(current)  # whatever held it is gone

    scene.activate(again)
    second = scene.boot()

    assert second.returncode == 0, second.stderr
    assert (scene.handoff()["phase"], scene.handoff()["environment"]["state"]) == ("switched", "selected")
    assert sorted(path.name for path in (current / "lib").iterdir()) == ["toydep-2.0"]
    assert scene.reports()[-1]["a"] == "GEN_CAND"


@needs_refusal
def test_a_refused_move_of_the_prepared_packages_puts_the_serving_ones_back_in_the_same_process(scene, held):
    current = scene.data / "state" / "python-userbase"
    row, handoff, generation = _ready_lock_change(scene, "user_site", current, breaks=False)
    held.hold(generation)  # the serving packages move aside; the prepared ones are refused

    first = scene.boot()

    assert first.returncode == 0, first.stderr
    handoff = scene.handoff()
    assert (handoff["return"]["error"]["step"], handoff["return"]["error"]["errno"]) == ("prepared_in", errno.EPERM)
    assert (handoff["phase"], handoff["environment"]["state"]) == ("returned", "restored")
    assert sorted(path.name for path in (current / "lib").iterdir()) == ["toydep-1.0"]  # back before any start
    assert not pathlib.Path(handoff["environment"]["previous"]).exists() and (generation / "lib" / "toydep-2.0").exists()
    assert [report["a"] for report in scene.reports()] == ["GEN_OLD"]
    assert _settle(scene, row)["state"] == "ready"


@needs_refusal
def test_packages_that_cannot_be_put_back_stop_the_tree_as_stuck_with_the_record(tmp_path, held):
    """The prepared packages are refused, then so is putting the serving ones back: the previous tree without its own
    packages is no generation, so the return stops (``_Stuck``: exit 3 with the record), the paths named."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "ouroboros", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--allow-empty", "-m", "base"], check=True)
    head = git(repo, "rev-parse", "HEAD")
    current, generation = tmp_path / "python-userbase", tmp_path / "python-userbase.update-c1"
    (current / "lib" / "toydep-1.0").mkdir(parents=True)
    (generation / "lib" / "toydep-2.0").mkdir(parents=True)
    helper = tmp_path / "helper"
    helper.mkdir()
    handoff = {"id": "a1", "phase": "switching", "switch": [], "old": head, "cand": head, "branch": "ouroboros",
               "environment": {"state": "prepared", "kind": "user_site", "current": str(current),
                               "generation": str(generation), "previous": str(tmp_path / "python-userbase.previous-a1"),
                               "identity": {"current": body_switch._path_identity(current),
                                            "generation": body_switch._path_identity(generation)}}}
    held.hold(generation)
    with pytest.raises(OSError) as refused:
        body_switch._select_environment(str(helper), handoff)
    assert refused.value.step == "prepared_in" and not current.exists()
    held.hold(pathlib.Path(handoff["environment"]["previous"]))  # and now the put-back is refused too

    with pytest.raises(body_switch._Stuck) as stuck:
        body_switch._return_to_old(str(repo), str(helper), handoff, "environment_selection_failed", {})

    assert handoff["environment"]["state"] == "unrestored" and "PermissionError" in handoff["environment"][
        "restore_error"]
    assert str(current) in str(stuck.value) and handoff["environment"]["previous"] in str(stuck.value)
    assert body_switch.read_handoff(str(helper))["environment"]["state"] == "unrestored"  # recorded before the stop


@needs_refusal
def test_the_reclaim_reports_only_what_it_actually_removed(tmp_path, monkeypatch, held):
    """Windows keeps a file a surviving process still maps; such a stray is not reported reclaimed and the next pass,
    once it can be deleted, removes it."""
    current = tmp_path / "data" / "state" / "python-userbase"
    stray = current.with_name("python-userbase.previous-gone")
    (stray / "lib").mkdir(parents=True)
    current.mkdir()
    (stray / "lib" / "mapped.pyd").write_bytes(b"in use")
    monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("user_site", current))
    held.hold(stray / "lib" / "mapped.pyd")

    assert update_prepare.reclaim_environments(tmp_path / "data", rows=[]) == []
    assert (stray / "lib" / "mapped.pyd").exists()
    held.release(stray / "lib" / "mapped.pyd")
    assert update_prepare.reclaim_environments(tmp_path / "data", rows=[]) == [str(stray)] and not stray.exists()


def test_after_a_start_that_was_not_returned_the_previous_packages_stay_until_the_next_admitted_preparation(scene):
    """PERSISTENCE: the previous generation's packages are kept after a start that was not returned. The switch
    renamed them aside; the successor started but never became ready and nothing returned it (``start_failed``).
    The Last attempt names them for a Restore, so every reclaim pass keeps them; the next admitted preparation
    replaces that Last attempt (``_forget_activation``) and the next pass reclaims them."""
    current = scene.tmp / "runtime-venv"
    _row, handoff, _generation = _ready_lock_change(scene, "venv", current, breaks=False)
    previous = pathlib.Path(handoff["environment"]["previous"])
    assert scene.boot().returncode == 0 and scene.handoff()["environment"]["state"] == "selected"
    body_switch._attest_loaded(str(scene.serving), scene.handoff())  # this start verified the switched tree

    result = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=False)
    update_prepare.record_activation(scene.data, result)

    assert result["outcome"] == "start_failed" and body_adoption.read(scene.data) == {}  # the handoff closed
    assert update_prepare.projection()["last_activation"]["previous_environment"] == str(previous)
    for _ in range(2):
        assert update_prepare.reclaim_environments(scene.data) == [] and (previous / "lib" / "toydep-1.0").exists()
    update_prepare._forget_activation(scene.data)

    assert update_prepare.reclaim_environments(scene.data) == [str(previous)] and not previous.exists()
