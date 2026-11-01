"""A recorded Panic forbids every automatic step of an Update's switch: real Git, the REAL hook and captured helper.

Over ``test_update_activation``'s cold-entry scene (the owner's commit, staged, unstaged and untracked work): an armed
transition is abandoned untouched (already covered there); a source switch interrupted midway, a macOS-app transition
whose installer died mid-switch, a Panic that arrives while the import probe of the new tree succeeds, and a switch
another entry finished while this one waited are each held as found (``stuck``, exit 3): nothing resumes, returns or
hands the tree to a fresh process. Without the Panic the same starts go on as before (the controls).
"""
from __future__ import annotations

import pytest

from ouroboros import body_adoption, body_switch
from tests.body_candidate_support import git
from tests.test_update_activation import UNSTAGED, Scene

pytestmark = pytest.mark.serial  # real cold-entry processes


@pytest.fixture
def scene(tmp_path, monkeypatch):
    return Scene(tmp_path, monkeypatch)


def _panic(scene):
    (scene.data / "state" / "panic_stop.flag").write_text("panic")


def _held(scene, booted, trigger="panic_stop"):
    """The transition is held as found: the entry stopped before any import, the record names the Panic."""
    handoff = scene.handoff()
    assert booted.returncode == body_switch.STUCK_EXIT_CODE, booted.stderr
    assert "Panic is recorded" in booted.stderr and scene.reports() == []  # no generation started, no fresh process
    assert handoff["phase"] == "stuck" and handoff["return"]["outcome"] == "refused"
    assert (handoff["return"]["trigger"], handoff["return"]["reasons"]) == (trigger, ["panic_stop"])
    return handoff


def _interrupted(scene):
    """An activated Update whose first entry died after writing the files, before the index and the branch."""
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    before = scene.composition()
    row = scene.prepare()
    scene.activate(row)
    crashed = scene.boot(OUROBOROS_BODY_SWITCH_FAIL_AT="after_files")
    assert crashed.returncode == 137 and scene.handoff()["phase"] == "switching"
    assert (scene.serving / "ouroboros" / "mod_a.py").read_text() == "GEN = 'GEN_CAND'\n"  # a mixed tree
    return before, row


@pytest.mark.parametrize("panic", [True, False], ids=["panic", "control"])
def test_an_interrupted_source_switch_is_never_resumed_after_a_panic(scene, panic):
    before, row = _interrupted(scene)
    if panic:
        _panic(scene)

    booted = scene.boot()

    if panic:
        _held(scene, booted)
        assert git(scene.serving, "rev-parse", "HEAD") == before["head"]  # the branch never moved
        assert (scene.serving / "ouroboros" / "mod_a.py").read_text() == "GEN = 'GEN_CAND'\n"  # as found, unreturned
        assert scene.boot().returncode == body_switch.STUCK_EXIT_CODE  # held for the owner at every later start
    else:  # the existing resume by content
        assert booted.returncode == 0, booted.stderr
        assert git(scene.serving, "rev-parse", "HEAD") == row["layers"]["new_commit"]
        assert (scene.serving / "ouroboros" / "mod_b.py").read_text() == UNSTAGED
        assert scene.reports()[-1]["a"] == "GEN_CAND"


@pytest.mark.parametrize("panic", [True, False], ids=["panic", "control"])
def test_an_app_transition_whose_installer_died_mid_switch_is_never_returned_after_a_panic(scene, panic):
    """``native`` in the record: the hook would return such a switch by content and hand the returned tree to a fresh
    process. After a Panic it is held where the installer left it."""
    before, _row = _interrupted(scene)
    helper = str(body_adoption.helper_dir(scene.data))
    handoff = body_switch.read_handoff(helper)
    handoff["native"] = {"installer": {"pid": 0}}  # its installer is gone: nothing that runs is named
    body_switch.write_handoff(helper, handoff)
    if panic:
        _panic(scene)

    booted = scene.boot()

    if panic:
        _held(scene, booted)
        assert (scene.serving / "ouroboros" / "mod_a.py").read_text() == "GEN = 'GEN_CAND'\n"
    else:  # returned by content, then a fresh process started the previous composition
        assert booted.returncode == 0, booted.stderr
        assert scene.handoff()["phase"] == "returned"
        assert scene.handoff()["return"]["trigger"] == "native_switch_interrupted"
        assert scene.composition() == before and [report["a"] for report in scene.reports()] == ["GEN_OLD"]


@pytest.mark.parametrize("panic", [True, False], ids=["panic", "control"])
def test_a_panic_that_arrives_while_the_import_probe_succeeds_is_never_handed_over(scene, monkeypatch, panic):
    """The probe of the switched tree itself records the Panic (the owner pressing it meanwhile) and then succeeds."""
    flag = scene.data / "state" / ("panic_stop.flag" if panic else "probe-ran.txt")
    monkeypatch.setattr(body_adoption, "UPDATE_PROBE_IMPORTS", f"import server; open({str(flag)!r}, 'w').write('panic')")
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    row = scene.prepare()
    scene.activate(row)

    booted = scene.boot()

    assert flag.read_text() == "panic"  # the probe ran and succeeded
    assert git(scene.serving, "rev-parse", "HEAD") == row["layers"]["new_commit"]  # the coherent switched tree
    if panic:
        _held(scene, booted)
    else:
        assert booted.returncode == 0, booted.stderr
        assert scene.handoff()["phase"] == "switched" and scene.reports()[-1]["a"] == "GEN_CAND"


@pytest.mark.parametrize("panic", [True, False], ids=["panic", "control"])
def test_a_switch_another_entry_finished_is_not_handed_over_by_a_waiting_entry_after_a_panic(scene, monkeypatch,
                                                                                            panic):
    """An entry that waited for the helper's lock while another finished the switch finds ``switched``: it would
    hand the tree to a fresh process. The switch itself is the real one; only the waiting entry is this process."""
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.activate(scene.prepare())
    assert scene.boot().returncode == 0 and scene.handoff()["phase"] == "switched"  # the other entry's switch
    helper = str(body_adoption.helper_dir(scene.data))
    handoff = body_switch.read_handoff(helper)
    if panic:
        _panic(scene)

    if panic:
        with pytest.raises(SystemExit) as stopped:
            body_switch._hook_switch(str(scene.serving), helper, handoff, "switched")
        assert stopped.value.code == body_switch.STUCK_EXIT_CODE
        record = scene.handoff()
        assert record["phase"] == "stuck" and record["return"]["reasons"] == ["panic_stop"]
    else:
        assert body_switch._hook_switch(str(scene.serving), helper, handoff, "switched") is True
        assert scene.handoff()["phase"] == "switched"
