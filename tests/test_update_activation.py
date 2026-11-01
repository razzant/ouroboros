"""The unified Update's activation and return: real Git, the real hook and helper, real cold entries.

The serving body is the miniature tree of ``body_candidate_support`` (the REAL package-init
hook and the REAL switch helper). Its owner keeps three layers of local work: a local
commit, a staged edit and a further unstaged edit of the same file, and an untracked
file. Preparation builds the candidate in its own workspace; activation switches the
checkout before the next entry imports anything; a successor that does not start is
returned only on positive evidence.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import threading

import pytest

from ouroboros import body_adoption, body_switch
from supervisor import update_adaptation, update_prepare
from tests.body_candidate_support import REPO, git, isolate, make_serving, run_entry, stage_then_edit_in_one_second

pytestmark = pytest.mark.serial  # real cold-entry processes

STOPPED = {"state": "completed", "unconfirmed": []}
EXITED = {"doomed": [1], "dead": [1], "unconfirmed": [], "cleanup_ok": True, "snapshot_ok": True}
STAGED = "GEN = 'GEN_OLD'\nSTAGED = 1\n"
UNSTAGED = "GEN = 'GEN_OLD'\nSTAGED = 1\nUNSTAGED = 1\n"


class Scene:
    def __init__(self, tmp_path, monkeypatch):
        from supervisor import git_ops

        self.tmp = tmp_path
        self.serving = make_serving(tmp_path)
        self.data = isolate(monkeypatch, tmp_path)
        self.report = tmp_path / "report.jsonl"
        monkeypatch.setattr(git_ops, "REPO_DIR", self.serving)
        monkeypatch.setattr(git_ops, "DRIVE_ROOT", self.data)
        monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
        monkeypatch.setattr(body_adoption, "UPDATE_PROBE_IMPORTS", "import server")
        self.channel = "development"  # the owner's selected channel, the one the target below is pinned from
        monkeypatch.setattr("ouroboros.update_channels.get_update_channel", lambda: self.channel)
        self.monkeypatch = monkeypatch
        self.base = git(self.serving, "rev-parse", "HEAD")

    def official(self, files, message="official update"):
        """A target commit on branch ``official`` from the base, made without touching the checkout."""
        git(self.serving, "checkout", "-q", "-b", "official")
        for rel, text in files.items():
            (self.serving / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.serving / rel).write_text(text)
        git(self.serving, "add", "-A")
        git(self.serving, "commit", "-q", "-m", message)
        self.target = git(self.serving, "rev-parse", "HEAD")
        git(self.serving, "checkout", "-q", "ouroboros")
        self.monkeypatch.setattr(update_prepare, "resolve_target", lambda **_kw: {
            "channel": self.channel, "target_ref": "official", "target_sha": self.target})
        return self.target

    def local_work(self, *, commit=True):
        if commit:
            (self.serving / "ouroboros" / "local.py").write_text("LOCAL = 1\n")
            git(self.serving, "add", "-A")
            git(self.serving, "commit", "-q", "-m", "local feature")
        (self.serving / "ouroboros" / "mod_b.py").write_text(STAGED)
        git(self.serving, "add", "ouroboros/mod_b.py")
        (self.serving / "ouroboros" / "mod_b.py").write_text(UNSTAGED)
        (self.serving / "local-scratch.txt").write_text("mine, untracked\n")

    def composition(self):
        """Exactly what the owner has: HEAD, the index bytes, every file's bytes."""
        snap = update_prepare.snapshot(self.serving, "ouroboros")
        return {"head": snap["head"], "index": git(self.serving, "ls-files", "-s"),
                "files": snap["files_tree"], "status": git(self.serving, "status", "--porcelain")}

    def prepare(self, **kwargs):
        return update_prepare.prepare(fetch=False, checks=lambda _row: {"ok": True, "stub": True}, **kwargs)

    def activate(self, row):
        handoff = body_adoption.authorize_update(self.data, row)
        assert body_adoption.bind_update_restart(self.data)
        # The owner's activation restart: the exit tail arms on the stop owners' evidence only.
        assert body_adoption.arm(self.data, worker_exits=EXITED, live_children=[], owned_stop=STOPPED,
                                 owner_restart=True) == "armed"
        return handoff

    def boot(self, **env):
        return run_entry(self.serving, env={"TOY_REPORT": str(self.report), **env})

    def reports(self):
        return [json.loads(line) for line in self.report.read_text().splitlines()] if self.report.exists() else []

    def handoff(self):
        return body_adoption.read(self.data)


@pytest.fixture
def scene(tmp_path, monkeypatch):
    return Scene(tmp_path, monkeypatch)


@pytest.fixture
def owned_state(scene, monkeypatch):
    """The retained activation outcome is one ``state.json`` value: this scene's own state, initialized."""
    from supervisor import state

    for name, path in {"DRIVE_ROOT": scene.data, "STATE_PATH": scene.data / "state/state.json",
                       "STATE_LAST_GOOD_PATH": scene.data / "state/state.last_good.json",
                       "STATE_LOCK_PATH": scene.data / "locks/state.lock"}.items():
        monkeypatch.setattr(state, name, path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    assert state.init_state().quality == "current"
    return state


def test_activation_carries_commit_index_and_files_as_separate_layers(scene):
    target = scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n", "web/sentinel.txt": "official ui\n"})
    scene.local_work()
    before = scene.composition()
    row = scene.prepare()
    assert row["state"] == "ready", row.get("reason")
    assert scene.composition() == before  # preparation changed nothing that serves
    scene.activate(row)

    booted = scene.boot()

    assert booted.returncode == 0, booted.stderr
    assert scene.reports()[-1]["a"] == "GEN_CAND"  # the fresh process imported the new tree
    merge = row["layers"]["new_commit"]
    assert git(scene.serving, "rev-parse", "HEAD") == merge and merge != target
    assert git(scene.serving, "rev-list", "--parents", "-n", "1", "HEAD").split()[1:] == [before["head"], target]
    # Staged stays staged, unstaged stays unstaged, untracked stays untracked; none of it is history.
    assert git(scene.serving, "show", ":ouroboros/mod_b.py") + "\n" == STAGED
    assert (scene.serving / "ouroboros" / "mod_b.py").read_text() == UNSTAGED
    assert git(scene.serving, "status", "--porcelain").splitlines() == [
        "MM ouroboros/mod_b.py", "?? local-scratch.txt"]
    assert "layer" not in git(scene.serving, "log", "--format=%s", "HEAD")
    assert (scene.serving / "ouroboros" / "local.py").read_text() == "LOCAL = 1\n"
    body_switch._attest_loaded(str(scene.serving), scene.handoff())
    result = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)
    assert result["outcome"] == "adopted" and result["kind"] == "update"


def test_a_confirmed_success_outlives_its_retired_record_until_the_next_admitted_preparation(scene, owned_state):
    """The panel's Last attempt: the settled record goes with its checkout; the outcome stays until a new admission."""
    from ouroboros import subagent_worktrees

    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n", "VERSION": "1.0.1\n"})
    scene.local_work()
    before = scene.composition()
    row = scene.prepare()
    scene.activate(row)
    assert scene.boot().returncode == 0
    body_switch._attest_loaded(str(scene.serving), scene.handoff())
    update_prepare.record_activation(scene.data, body_adoption.finalize_on_boot(scene.data, scene.serving,
                                                                                supervisor_ready=True))

    last = update_prepare.projection()["last_activation"]
    assert (last["outcome"], last["base_version"], last["target_version"]) == ("adopted", "1.0.0", "1.0.1")
    assert last["previous"] == before["head"] and last["sha"] == row["layers"]["new_commit"]
    assert (last["local_commits"], last["local_changes"], last["carried_files"]) == (1, True, 2)
    assert update_prepare.projection()["state"] == "none"  # settled: the panel's verdict is the ordinary one
    subagent_worktrees.prune_orphans(retention_days=0, live=lambda _task: False)
    assert update_prepare.find(row["candidate_id"]) is None  # the GC retired the record with its checkout
    assert update_prepare.projection()["last_activation"] == last  # and the Last attempt stays
    with pytest.raises(update_prepare.PrepareRefused) as current:
        scene.prepare()  # a refusal before any admission is no attempt
    assert current.value.code == "UPDATE_CURRENT" and update_prepare.projection()["last_activation"] == last

    newer = git(scene.serving, "commit-tree", f"{scene.target}^{{tree}}", "-p", scene.target, "-m", "newer official")
    scene.monkeypatch.setattr(update_prepare, "resolve_target", lambda **_kw: {
        "channel": scene.channel, "target_ref": "official", "target_sha": newer})
    scene.prepare()
    assert "last_activation" not in update_prepare.projection()  # the next admitted attempt replaced it


def test_a_start_that_does_not_attest_its_switch_is_shown_unconfirmed_until_one_that_does(scene, owned_state,
                                                                                             monkeypatch):
    """The boot's real observation: HEAD is the switched tree, but this process verified nothing before its imports.
    The candidate keeps activating with that observation beside its adoption id: no outcome is retained, the handoff
    stays switched and the owner's chat gets nothing. The next start that attests the tree settles it as adopted.
    The healthy composition follows the same evidence: an ordinary ready start records the one it ran, the
    unattested start leaves it (so a later import failure of the switched tree is an unverified edit, rescued, and
    never blamed on the environment), and the start that attests the switched tree records that one."""
    from supervisor import git_ops_reset, message_bus

    def healthy():  # the keep policy's own retained fact, beside the Last attempt (never supervisor state)
        return (update_prepare.retained_fact("healthy_composition", scene.data) or {}).get("composition")

    def on_disk():
        return update_prepare.fingerprint(update_prepare.snapshot(scene.serving, "ouroboros"))

    sent = []
    monkeypatch.setattr(message_bus, "send_with_budget", lambda *args, **kwargs: sent.append(args))
    # The owner's chat exists: an Update names no task, and its outcomes still never become an owner-wide notice.
    owned_state.update_state(lambda live: live.update({"owner_chat_id": 7}))
    monkeypatch.setattr(sys, "_ouroboros_body_generation", None, raising=False)  # nothing attested in this process
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    assert body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True) == {}  # an ordinary start
    before = healthy()
    assert before == on_disk()
    row = scene.prepare()
    handoff = scene.activate(row)
    update_prepare.mark(row, state="activating", activation={"adoption_id": handoff["id"]})  # as the activation does
    assert scene.boot().returncode == 0 and scene.handoff()["phase"] == "switched"

    observed = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)

    assert (observed["outcome"], observed["kind"], observed["why"]) == ("unconfirmed", "update", "loaded_tree_unattested")
    record = update_prepare.find(row["candidate_id"])
    assert (record["state"], record["reason"]) == ("activating", "activation:unconfirmed")
    assert record["activation"]["adoption_id"] == handoff["id"] and record["activation"]["observed_at"]
    assert scene.handoff()["phase"] == "switched"  # the adoption's own mechanics are untouched
    assert "last_activation" not in update_prepare.projection() and sent == []
    assert healthy() == before != on_disk()  # ready, but nothing proves it ran the switched tree
    kept, why = git_ops_reset._kept_import_failure("bootstrap", {"returncode": 1, "stderr": "SyntaxError: boom"})
    failure = update_prepare.projection()["start_failure"]
    assert kept is False and "cause is unknown" in why and failure["composition"] == "unverified"
    assert pathlib.Path(failure["rescue"]).is_dir()
    body_switch._attest_loaded(str(scene.serving), scene.handoff())

    settled = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)

    assert settled["outcome"] == "adopted" and update_prepare.find(row["candidate_id"])["state"] == "activated"
    assert update_prepare.projection()["last_activation"]["outcome"] == "adopted" and sent == []
    assert healthy() == on_disk() != before  # this start attested the switched tree and reached ready
    assert "environment, data or something outside" in git_ops_reset._kept_import_failure(
        "bootstrap", {"returncode": 1, "stderr": "SyntaxError: boom"})[1]


def test_a_new_tree_that_does_not_import_returns_the_exact_previous_composition(scene):
    scene.official({"ouroboros/mod_a.py": "GEN = (\n"})  # the official commit does not even parse
    scene.local_work()
    before = scene.composition()
    row = scene.prepare()
    assert row["state"] == "ready"
    scene.activate(row)

    booted = scene.boot()

    assert booted.returncode == 0, booted.stderr
    assert [report["a"] for report in scene.reports()] == ["GEN_OLD"]  # this entry went on with the old tree
    assert scene.composition() == before  # HEAD, staged, unstaged and untracked bytes exactly as they were
    handoff = scene.handoff()
    assert handoff["phase"] == "returned" and handoff["return"]["trigger"] == "import_probe_failed"
    assert not pathlib.Path(body_switch.pointer_path(str(scene.serving))).exists()
    result = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)
    assert result["outcome"] == "returned" and "data was not rolled back" in result["note"]
    assert result["why"] == "import_probe_failed"  # the panel's Why, as a code, beside the prose note
    # A cold reopen of the previous generation keeps the same composition and starts nothing new.
    assert scene.boot().returncode == 0 and scene.composition() == before
    assert body_adoption.read(scene.data) == {}


def test_a_successor_that_entered_is_never_returned_automatically(scene):
    """The census alone never decides: this successor wrote nothing it watches, but it entered the new tree."""
    server = (scene.serving / "server.py").read_text().replace(
        'if __name__ == "__main__":', 'if __name__ == "__main__":\n    raise SystemExit("successor crashed")')
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n", "server.py": server})
    scene.local_work(commit=False)
    scene.activate(scene.prepare())
    first = scene.boot()  # the scratch probe imports it fine; the real entry then dies
    assert first.returncode != 0 and "successor crashed" in first.stderr
    assert [(entry["kind"], entry["argv0"]) for entry in scene.handoff()["entries"]] == [("server", "server.py")]

    again = scene.boot()

    assert again.returncode != 0  # nothing returned: the new tree starts (and fails) again
    decision = scene.handoff()["return"]
    assert decision["outcome"] == "unknown" and decision["reasons"] == ["successor_entered:server"]
    assert git(scene.serving, "rev-parse", "HEAD") == scene.handoff()["cand"]


def test_successor_that_wrote_a_first_read_store_stays_for_recovery(scene):
    """The negative consumer: a real successor process changes settings.json, then dies."""
    server = (scene.serving / "server.py").read_text().replace(
        'if __name__ == "__main__":',
        'if __name__ == "__main__":\n'
        '    with open(os.path.join(os.environ["OUROBOROS_DATA_DIR"], "settings.json"), "w") as fh:\n'
        '        fh.write("{\\"migrated_by\\": \\"successor\\"}")\n'
        '    raise SystemExit("successor crashed after writing")')
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n", "server.py": server})
    scene.local_work(commit=False)
    (scene.data / "settings.json").write_text("{}")  # what the previous generation left
    scene.activate(scene.prepare())
    first = scene.boot()
    assert first.returncode != 0 and "after writing" in first.stderr

    again = scene.boot()

    assert again.returncode != 0  # nothing was returned: the new tree starts (and fails) again
    decision = scene.handoff()["return"]
    assert decision["outcome"] == "unknown" and decision["reasons"] == [
        "successor_entered:server", "successor_changed_data:settings.json"]
    assert git(scene.serving, "rev-parse", "HEAD") == scene.handoff()["cand"]  # data and new tree kept
    assert (scene.data / "settings.json").read_text() == '{"migrated_by": "successor"}'


def test_late_edit_at_a_switch_path_abandons_the_switch_and_keeps_the_edit(scene):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    scene.activate(scene.prepare())
    (scene.serving / "ouroboros" / "mod_a.py").write_text("GEN = 'GEN_LATE'\n")  # after the fence, at a switch path

    assert scene.boot().returncode == 0
    assert scene.reports()[-1]["a"] == "GEN_LATE"  # this entry went on with the old tree and the late edit
    assert scene.handoff()["phase"] == "abandoned"
    assert (scene.serving / "ouroboros" / "mod_a.py").read_text() == "GEN = 'GEN_LATE'\n"


def test_late_edit_outside_the_switch_set_is_carried_through_unchecked(scene):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    scene.activate(scene.prepare())
    (scene.serving / "ouroboros" / "mod_b.py").write_text("GEN = 'GEN_OLD'\nLATE = 1\n")  # no layer moves mod_b

    assert scene.boot().returncode == 0
    assert scene.reports()[-1]["a"] == "GEN_CAND"
    assert (scene.serving / "ouroboros" / "mod_b.py").read_text() == "GEN = 'GEN_OLD'\nLATE = 1\n"


def test_authorize_refuses_a_switch_path_that_moved_after_preparation(scene):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    row = scene.prepare()
    (scene.serving / "ouroboros" / "mod_a.py").write_text("GEN = 'GEN_NEWER'\n")  # a path the switch moves
    with pytest.raises(body_adoption.AdoptionRefused) as refused:
        body_adoption.authorize_update(scene.data, row)
    assert refused.value.code == "ADOPTION_SWITCH_SET_DIRTY"


def test_any_late_local_change_makes_the_ready_candidate_stale_before_authorization(scene):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    row = scene.prepare()
    (scene.serving / "ouroboros" / "mod_b.py").write_text("GEN = 'GEN_OLD'\nNEWER = 1\n")  # outside the switch set
    assert update_prepare.activation_check(row).startswith("local_changed")
    assert update_prepare.find(row["candidate_id"])["state"] == "stale"
    assert "activate" not in update_prepare.projection()["actions"]


def _switched_handoff(scene):
    """A switched handoff whose successor has one recorded server entry (its census taken then)."""
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    scene.activate(scene.prepare())
    handoff = scene.handoff()
    handoff["phase"] = "switched"
    handoff["entries"] = [{"kind": "server", "pid": 1, "ready": False}]
    handoff["census"] = body_switch.data_census(str(scene.data))
    return handoff


def test_the_census_is_a_further_check_on_the_first_read_stores(scene):
    (scene.data / "state" / "state.json").write_text("{}")
    (scene.data / "state" / "dialog.sqlite").write_bytes(b"db")
    handoff = _switched_handoff(scene)
    assert body_switch._census_changes(handoff) == []
    (scene.data / "state" / "dialog.sqlite-wal").write_bytes(b"")  # an empty WAL is no write
    assert body_switch._census_changes(handoff) == []
    (scene.data / "state" / "dialog.sqlite-wal").write_bytes(b"frame")
    assert body_switch._census_changes(handoff) == ["state/dialog.sqlite-wal"]
    (scene.data / "state" / "dialog.sqlite-wal").unlink()
    (scene.data / "state" / "state.json").write_text('{"session_id": "successor"}')
    assert body_switch._census_changes(handoff) == ["state/state.json"]
    assert body_switch._census_changes({**handoff, "census": None}) == ["census_missing"]
    outcome, reasons = body_switch.qualify_return(str(scene.serving), {**handoff, "census": None})
    assert outcome == "unknown" and "data_unobserved" in reasons  # nothing to compare is never compatible
    # The boundary is any recorded entry, a module CLI's included; before one, nothing of the new tree ran.
    other = [{"kind": "other", "argv0": "cli.py", "ready": False}]
    assert "successor_entered:other" in body_switch.qualify_return(str(scene.serving), {**handoff, "entries": other})[1]
    assert not any(reason.startswith("successor_entered") for reason in body_switch.qualify_return(
        str(scene.serving), {**handoff, "entries": []})[1])


def test_panic_refuses_every_automatic_return(scene):
    handoff = _switched_handoff(scene)
    scene.monkeypatch.setattr(body_switch, "installation_panic", lambda _root: True)
    assert body_switch.qualify_return(str(scene.serving), handoff) == ("refused", ["panic_stop"])


def test_a_lock_change_without_a_switchable_environment_is_refused_not_prepared_on_serving_packages(scene, tmp_path):
    scene.monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("", tmp_path / "system-python"))
    scene.official({"requirements-runtime.lock": "toydep==2.0\n"})
    scene.local_work(commit=False)
    row = scene.prepare()
    assert row["state"] == "failed" and row["reason"].startswith("UPDATE_ENVIRONMENT_UNAVAILABLE")
    assert row["environment"]["same"] is False


@pytest.mark.parametrize("kind", ["user_site", "venv"])
def test_a_lock_change_runs_and_returns_on_its_own_generation_environment(scene, tmp_path, kind):
    user_base = tmp_path / ("userbase" if kind == "user_site" else "venv")  # the packages' one serving path
    (user_base / "lib").mkdir(parents=True)
    (user_base / "lib" / "toydep-1.0").write_text("old\n")
    scene.monkeypatch.setattr(update_prepare, "serving_environment", lambda: (kind, user_base))
    installs = []

    def install(target, lock, installed_kind, python=None):  # pip into the generation directory only
        assert installed_kind == kind
        installs.append((pathlib.Path(target), lock.read_text()))
        (pathlib.Path(target) / "lib" / "toydep-1.0").unlink()
        (pathlib.Path(target) / "lib" / "toydep-2.0").write_text("new\n")
        return {"returncode": 0}

    scene.monkeypatch.setattr(update_prepare, "_install_lock", install)
    # The new tree imports only with its own packages missing: the scratch probe sees the generation's.
    server = (scene.serving / "server.py").read_text().replace(
        "import ouroboros  #", f"if os.path.exists({str(user_base / 'lib' / 'toydep-2.0')!r}):\n"
        "    raise SystemExit('the new packages break this tree')\nimport ouroboros  #")
    scene.official({"requirements-runtime.lock": "toydep==2.0\n", "server.py": server})
    scene.local_work(commit=False)
    row = scene.prepare()
    assert row["state"] == "ready", row.get("reason")
    generation = pathlib.Path(row["environment"]["generation"]["user_base"])
    assert installs == [(generation, "toydep==2.0\n")] and sorted(p.name for p in (user_base / "lib").iterdir()) == [
        "toydep-1.0"]  # the serving packages are untouched by the preparation
    handoff = scene.activate(row)
    assert handoff["environment"]["state"] == "prepared" and handoff["deps_command"] is None

    # Files, index, branch, then the environment switch; the scratch probe of the new tree, run on the
    # generation's packages, fails before any entry of it: the real pre-data return, environment included.
    first = scene.boot()

    assert first.returncode == 0, first.stderr
    handoff = scene.handoff()
    assert (handoff["return"]["outcome"], handoff["return"]["trigger"]) == ("returned", "import_probe_failed")
    assert handoff["environment"]["state"] == "restored" and handoff["entries"] == []
    assert sorted(p.name for p in (user_base / "lib").iterdir()) == ["toydep-1.0"]
    assert (generation / "lib" / "toydep-2.0").exists() and not pathlib.Path(handoff["environment"]["previous"]).exists()
    assert [report["a"] for report in scene.reports()] == ["GEN_OLD"]


def test_an_engine_pin_or_runtime_change_never_qualifies_a_return(scene):
    handoff = _switched_handoff(scene)
    # Each condition alone, before any entry: here the checkout is still at the base, not the candidate.
    assert body_switch.qualify_return(str(scene.serving), {**handoff, "entries": []})[1] == ["checkout_moved"]
    handoff["qualification"]["engine_pin_same"] = False
    handoff["qualification"]["runtime"] = {**handoff["qualification"]["runtime"], "sqlite": "0.0.0"}
    outcome, reasons = body_switch.qualify_return(str(scene.serving), handoff)
    assert outcome == "unknown" and {"engine_pin_changed", "runtime_changed"} <= set(reasons)


def test_a_switch_interrupted_between_files_and_branch_resumes_by_content(scene):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n", "web/sentinel.txt": "official ui\n"})
    scene.local_work(commit=False)
    before = scene.composition()
    row = scene.prepare()
    scene.activate(row)

    crashed = scene.boot(OUROBOROS_BODY_SWITCH_FAIL_AT="after_files")

    assert crashed.returncode == 137 and scene.handoff()["phase"] == "switching"
    assert git(scene.serving, "rev-parse", "HEAD") == before["head"]  # the branch moves last
    resumed = scene.boot()
    assert resumed.returncode == 0, resumed.stderr
    assert git(scene.serving, "rev-parse", "HEAD") == row["layers"]["new_commit"]
    assert git(scene.serving, "show", ":ouroboros/mod_b.py") + "\n" == STAGED
    assert (scene.serving / "ouroboros" / "mod_b.py").read_text() == UNSTAGED
    assert scene.reports()[-1]["a"] == "GEN_CAND"


def test_a_newer_target_keeps_the_pinned_candidate_activatable(scene):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    row = scene.prepare()
    # The channel moved forward: the same channel names a newer commit than the pinned target.
    git(scene.serving, "checkout", "-q", "official")
    (scene.serving / "ouroboros" / "mod_a.py").write_text("GEN = 'GEN_NEWER'\n")
    git(scene.serving, "commit", "-q", "-am", "a newer official commit")
    newer = git(scene.serving, "rev-parse", "HEAD")
    git(scene.serving, "checkout", "-q", "ouroboros")
    projection = update_prepare.projection(latest_sha=newer)
    assert projection["newer_available"] is True and projection["target_sha"] == row["target"]["target_sha"]
    assert "activate" in projection["actions"]
    with update_prepare.lifecycle():
        assert update_prepare.activation_check(update_prepare.find(row["candidate_id"])) == ""


def test_the_owners_channel_change_makes_the_prepared_candidate_stale(scene):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    row = scene.prepare()
    scene.channel = "stable"  # the owner chose another channel after preparing
    with update_prepare.lifecycle():
        why = update_prepare.activation_check(update_prepare.find(row["candidate_id"]))
    assert why.startswith("channel_changed") and update_prepare.find(row["candidate_id"])["reason"] == "channel_changed"
    assert update_prepare.find(row["candidate_id"])["state"] == "stale"
    assert "activate" not in update_prepare.projection()["actions"]


def test_a_requirements_txt_edit_beside_an_unchanged_runtime_lock_activates_without_a_generation(scene):
    """The launcher installs the runtime lock; preparation and authorization compare exactly that input."""
    assert (scene.serving / "requirements-runtime.lock").read_text() == "toydep==1.0\n"  # the body's own lock
    scene.official({"requirements.txt": "toydep==9.9\n", "ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    row = scene.prepare()
    assert row["state"] == "ready" and row["environment"]["same"] is True and "generation" not in row["environment"]
    assert row["environment"]["candidate"]["lock"] == "requirements-runtime.lock"

    handoff = scene.activate(row)

    assert handoff["environment"] is None and handoff["qualification"]["dependencies_same"] is True
    assert scene.boot().returncode == 0 and scene.reports()[-1]["a"] == "GEN_CAND"
    assert (scene.serving / "requirements.txt").read_text() == "toydep==9.9\n"


def test_a_discarded_venv_generation_goes_with_its_retired_candidate_and_strays_are_reclaimed(scene, tmp_path):
    from ouroboros import subagent_worktrees

    venv = tmp_path / "venv"
    (venv / "lib").mkdir(parents=True)
    scene.monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("venv", venv))

    def install(target, _lock, _kind, _python=None):
        (pathlib.Path(target) / "lib" / "toydep-2.0").write_text("new\n")
        return {"returncode": 0}

    scene.monkeypatch.setattr(update_prepare, "_install_lock", install)
    scene.official({"requirements-runtime.lock": "toydep==2.0\n"})
    row = scene.prepare()
    generation = pathlib.Path(row["environment"]["generation"]["user_base"])
    assert generation == venv.with_name(f"venv.update-{row['candidate_id']}") and generation.is_dir()
    crashed, aside = venv.with_name("venv.update-crashed"), venv.with_name("venv.previous-0123456789ab")
    crashed.mkdir()
    aside.mkdir()  # a previous generation whose handoff closed: nothing names it

    update_prepare.discard(update_prepare.find(row["candidate_id"]))
    assert generation.is_dir()  # kept with its checkout until the GC retires the candidate
    subagent_worktrees.prune_orphans(retention_days=0, live=lambda _task_id: False)

    assert not generation.exists() and not crashed.exists() and not aside.exists()
    assert venv.is_dir() and (venv / "lib").is_dir() and update_prepare.find(row["candidate_id"]) is None


def _venv_with_a_lock_change(scene, tmp_path):
    """The serving packages are a venv, and the official update changes the lock: each candidate gets a generation."""
    venv = tmp_path / "venv"
    (venv / "lib").mkdir(parents=True)
    scene.monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("venv", venv))

    def install(target, _lock, _kind, _python=None):
        (pathlib.Path(target) / "lib" / "toydep-2.0").write_text("new\n")
        return {"returncode": 0}

    scene.monkeypatch.setattr(update_prepare, "_install_lock", install)
    scene.official({"requirements-runtime.lock": "toydep==2.0\n"})
    return venv


def test_a_generation_whose_candidate_became_ready_after_the_gc_listed_the_records_is_kept(scene, tmp_path):
    """A worktree GC pass lists the copy records; a preparation then creates and finishes its generation; the
    pass reclaims afterwards. Its earlier listing names no owner, so only the records as they are at the
    deletion may decide: the ready candidate keeps its packages."""
    from ouroboros import subagent_worktrees

    venv = _venv_with_a_lock_change(scene, tmp_path)
    listed, prepared = subagent_worktrees._records, []

    def records(data_dir=None, *, op="", strict=False):
        rows = listed(data_dir, op=op, strict=strict)
        if op == "prune_orphans" and not prepared:
            prepared.append(scene.prepare())  # after this pass listed, before it reclaims
        return rows

    scene.monkeypatch.setattr(subagent_worktrees, "_records", records)

    subagent_worktrees.prune_orphans(retention_days=0, live=lambda _task_id: False)

    [row] = prepared
    assert row["state"] == "ready" and update_prepare.current()["candidate_id"] == row["candidate_id"]
    generation = pathlib.Path(row["environment"]["generation"]["user_base"])
    assert generation.parent == venv.parent and (generation / "lib" / "toydep-2.0").exists()


def test_generations_this_process_works_in_survive_a_pass_that_looked_before_they_began(scene, tmp_path):
    """Both start after the pass took its listing and its view of this process's steps: a preparation still
    cloning (unsettled, its generation not recorded yet) and one superseded meanwhile whose step still runs.
    Once that step ended, its clone is a proven orphan and the next pass takes it."""
    from ouroboros import subagent_worktrees

    venv = tmp_path / "venv"
    (venv / "lib").mkdir(parents=True)
    listing = subagent_worktrees._records(scene.data, op="prune_orphans")
    started = []

    def serving_then_meanwhile():
        for cid, state in (("update_preparing_0001", "preparing"), ("update_superseded_02", "superseded")):
            subagent_worktrees._put_record({"kind": update_prepare.KIND, "path": str(tmp_path / "worktrees" / cid),
                                            "candidate_id": cid, "task_id": cid, "state": state})
            update_prepare._working.add(cid)
            venv.with_name(f"venv.update-{cid}").mkdir()
            started.append(cid)
        scene.monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("venv", venv))
        return "venv", venv

    scene.monkeypatch.setattr(update_prepare, "serving_environment", serving_then_meanwhile)
    try:
        assert update_prepare.reclaim_environments(scene.data, rows=listing) == []
        assert all(venv.with_name(f"venv.update-{cid}").is_dir() for cid in started)
    finally:
        update_prepare._working.difference_update(started)

    removed = update_prepare.reclaim_environments(scene.data, rows=subagent_worktrees._records(scene.data))

    assert removed == [str(venv.with_name("venv.update-update_superseded_02"))]
    assert venv.with_name("venv.update-update_preparing_0001").is_dir() and (venv / "lib").is_dir()


def test_a_generation_judged_unowned_regains_no_owner_before_its_deletion(scene, tmp_path, monkeypatch):
    """Between the fresh ownership read (under the lifecycle lock) and the deletion outside it, each supported
    producer refuses the doomed generation's settled candidate: its record takes no write, its resolver continues
    nothing, and activation does not find it current. A guard of the invariant the deletion relies on."""
    from ouroboros import subagent_worktrees

    venv = tmp_path / "venv"
    (venv / "lib").mkdir(parents=True)
    scene.monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("venv", venv))
    cid, resolver = "update_superseded_03", {"update_candidate_resolver": "update_superseded_03",
                                             "root_task_id": "update_adapt_03"}
    subagent_worktrees._put_record({"kind": update_prepare.KIND, "path": str(tmp_path / "worktrees" / cid),
                                    "candidate_id": cid, "task_id": cid, "state": "superseded",
                                    "resolver": {"task_id": "update_adapt_03"}})
    doomed, real, seen = venv.with_name(f"venv.update-{cid}"), shutil.rmtree, []
    doomed.mkdir()

    def at_the_deletion(path, *args, **kwargs):
        if pathlib.Path(path) == doomed:
            with pytest.raises(update_prepare.PrepareRefused) as refused:
                update_prepare._update(update_prepare.find(cid), environment={"generation": {"user_base": str(doomed)}})
            seen.append((refused.value.code, update_adaptation.resolver_row(resolver), update_prepare.current()))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", at_the_deletion)

    assert update_prepare.reclaim_environments(scene.data, rows=subagent_worktrees._records(scene.data)) == [
        str(doomed)]
    assert seen == [("UPDATE_CANDIDATE_SETTLED", None, None)] and not doomed.exists()


def test_environments_an_open_handoff_names_outlive_any_pass_and_go_once_it_closes(scene, tmp_path):
    """Active and returned: the previous packages a switch kept for a return, and a generation a return put
    back beside them, belong to the open handoff at the deletion; once it closed nothing names them."""
    from ouroboros import subagent_worktrees

    venv = tmp_path / "venv"
    (venv / "lib").mkdir(parents=True)
    scene.monkeypatch.setattr(update_prepare, "serving_environment", lambda: ("venv", venv))
    listing = subagent_worktrees._records(scene.data, op="prune_orphans")
    previous, returned = venv.with_name("venv.previous-0123456789ab"), venv.with_name("venv.update-update_back_01")
    previous.mkdir()
    returned.mkdir()
    helper = body_adoption.helper_dir(scene.data)
    helper.mkdir(parents=True)
    body_switch.write_handoff(str(helper), {"id": "0123456789ab", "kind": "update", "phase": "switched",
                                            "environment": {"state": "selected", "current": str(venv),
                                                            "generation": str(returned), "previous": str(previous)}})

    assert update_prepare.reclaim_environments(scene.data, rows=listing) == []
    assert previous.is_dir() and returned.is_dir()
    shutil.rmtree(helper)  # the adoption closed

    removed = update_prepare.reclaim_environments(scene.data, rows=listing)
    assert sorted(removed) == sorted([str(previous), str(returned)])


def test_a_source_only_activation_authorizes_where_the_macos_app_installer_cannot_import(scene):
    """Windows has no ``fcntl``, which the macOS app installer imports at its top. Authorizing a source-only
    Update runs after the writer fence; it must never need that installer, so here it runs in an interpreter
    where importing it fails as it does on Windows, against the real checkout and record. The helper's lock it
    holds meanwhile is the platform's own (``body_switch._locked``: ``msvcrt`` there, ``fcntl`` here)."""
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    row = scene.prepare()
    assert row["state"] == "ready" and row["native"]["status"] == "none"
    script = textwrap.dedent("""
        import json, sys
        sys.modules["ouroboros.app_install"] = None  # as on Windows, where its own ``import fcntl`` fails
        from ouroboros import body_adoption
        handoff = body_adoption.authorize_update(sys.argv[1], json.loads(sys.argv[2]))
        installed = sys.modules.get("ouroboros.app_install") is not None
        print(json.dumps({"phase": handoff["phase"], "installer_imported": installed}))
    """)

    done = subprocess.run([sys.executable, "-c", script, str(scene.data), json.dumps(row)], cwd=str(REPO),
                          capture_output=True, text=True, timeout=180, env={**os.environ, "PYTHONPATH": str(REPO)})

    assert done.returncode == 0, done.stderr[-3000:]
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"phase": "authorized", "installer_imported": False}
    assert scene.handoff()["phase"] == "authorized"
    assert not (body_adoption.helper_dir(scene.data) / "app_install.py").exists()


@pytest.mark.parametrize("recovered, settled", [(True, ("activated", "adopted")), (False, ("failed", "start_failed"))])
def test_a_providerless_generation_adopts_its_update_once_its_recovery_made_it_ready(scene, monkeypatch, recovered,
                                                                                      settled):
    """No provider: no supervisor starts and no bootstrap runs, so ``_bootstrap_ok`` stays None while the
    generation serves. It is ready once its providerless recovery is (``_supervisor_ready``), and this Update's
    own switch probe imported the tree first; the real boot task then settles it as adopted. One that stops
    before its recovery made it ready is not."""
    import server

    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    row = scene.prepare()
    scene.activate(row)
    assert scene.boot().returncode == 0 and scene.handoff()["phase"] == "switched"
    body_switch._attest_loaded(str(scene.serving), scene.handoff())  # this process verified the tree first
    for name, value in (("DATA_DIR", scene.data), ("REPO_DIR", scene.serving), ("_supervisor_error", None),
                        ("_bootstrap_ok", None), ("broadcast_ws_sync", lambda *_a: None)):
        monkeypatch.setattr(server, name, value)
    monkeypatch.setattr("supervisor.update_merge.finalize_managed_update_on_boot", lambda **_kw: {})
    monkeypatch.setattr("supervisor.update_merge.active_update_tx", lambda: None)
    monkeypatch.setattr("supervisor.git_ops.compute_managed_update_status", lambda *_a, **_kw: {})
    monkeypatch.delenv("OUROBOROS_EXTERNAL_HOST_UPDATE", raising=False)  # no external host to read back
    init_done, ready, stop = threading.Event(), threading.Event(), threading.Event()
    init_done.set()  # a providerless boot publishes its init outcome at once
    (ready if recovered else stop).set()  # its recovery finished, or the generation stopped before it did
    for name, event in (("_supervisor_init_done", init_done), ("_supervisor_ready", ready), ("_supervisor_stop", stop)):
        monkeypatch.setattr(server, name, event)

    server._boot_managed_update_tasks()

    record = update_prepare.find(row["candidate_id"])
    assert (record["state"], record["activation"]["outcome"]) == settled, record.get("activation")
    assert record["activation"].get("why") == (None if recovered else "supervisor_not_ready")


def test_providerless_readiness_never_stands_in_for_a_body_adoption(scene, monkeypatch):
    """A body adoption has no Update switch probe: without a bootstrap it stays unconfirmed, readiness unasked."""
    head = git(scene.serving, "rev-parse", "HEAD")
    helper = body_adoption.helper_dir(scene.data)
    helper.mkdir(parents=True)
    body_switch.write_handoff(str(helper), {"schema": 1, "id": "body00000001", "phase": "switched", "cand": head,
                                            "old": head, "repo_dir": str(scene.serving), "branch": "ouroboros"})
    monkeypatch.setattr(sys, "_ouroboros_body_generation", {"adoption_id": "body00000001", "sha": head}, raising=False)
    consulted = []

    result = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=False,
                                            providerless_ready=lambda: consulted.append(1) or True)

    # A body adoption's settled facts (``_settled``: its identity and card-row order), never an Update's outcome.
    assert (result["outcome"], result["note"], consulted) == ("unconfirmed", "", [])
    assert (result["adoption_id"], result["candidate"], result["card_row_revision"]) == ("body00000001", head, 0)
    assert "kind" not in result and "candidate_id" not in result


# A base whose server writes one record through its own writer and reads every record back; with
# TOY_WRITE_THEN_FAIL it writes and then dies before ready (an outside failure: the code is the base's).
PASSIVE_SERVER = '''"""Toy entry with a data writer and reader (both generations run exactly this)."""
import json
import multiprocessing
import os
import sys

import ouroboros  # the package-init hook runs here, before any other body module
from ouroboros import mod_a, mod_b


def _child():
    pass


if __name__ == "__main__":
    if os.environ.get("TOY_SPAWN_CHILD"):  # as the real supervisor starts its event-queue manager
        child = multiprocessing.get_context("spawn").Process(target=_child)
        child.start()
        child.join(60)
    records = os.path.join(os.environ["OUROBOROS_DATA_DIR"], "records.jsonl")
    if os.environ.get("TOY_WRITE_THEN_FAIL"):
        with open(records, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"form": mod_b.FORM, "pid": os.getpid()}) + "\\n")
        raise SystemExit("successor not ready")
    read = [json.loads(line) for line in open(records, encoding="utf-8")] if os.path.exists(records) else []
    report = {"a": mod_a.GEN, "read": [row for row in read if row.get("form") == mod_b.FORM], "all": len(read)}
    with open(os.environ["TOY_REPORT"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(report) + "\\n")
'''
STYLE_OLD, STYLE_NEW = "body { color: black; }\n", "body { color: navy; }\n"


def _passive_base(scene):
    """The installation's own commit: a served stylesheet, a record form and the writing/reading server."""
    (scene.serving / "web").mkdir(exist_ok=True)
    (scene.serving / "web" / "style.css").write_text(STYLE_OLD)
    (scene.serving / "ouroboros" / "mod_b.py").write_text("GEN = 'GEN_OLD'\nFORM = 'v1'\n")
    (scene.serving / "server.py").write_text(PASSIVE_SERVER)
    git(scene.serving, "add", "-A")
    git(scene.serving, "commit", "-q", "-m", "local: stylesheet, record form, writer and reader")


def test_a_stylesheet_only_successor_that_wrote_data_and_failed_is_returned_and_read(scene):
    _passive_base(scene)
    scene.official({"web/style.css": STYLE_NEW})
    row = scene.prepare()
    handoff = scene.activate(row)
    assert handoff["qualification"]["web_root"] == "web" and handoff["qualification"]["owned_engine"] == "absent"

    failed = scene.boot(TOY_WRITE_THEN_FAIL="1")  # switched, entered, wrote with the base's writer, died

    assert failed.returncode != 0 and "successor not ready" in failed.stderr
    assert (scene.serving / "web" / "style.css").read_text() == STYLE_NEW
    returned = scene.boot()  # the next entry: the stylesheet-only successor is returned
    assert returned.returncode == 0, returned.stderr
    decision = scene.handoff()["return"]
    assert (decision["outcome"], decision["trigger"], decision["reasons"]) == ("returned", "successor_start_failed", [])
    assert (scene.serving / "web" / "style.css").read_text() == STYLE_OLD
    assert git(scene.serving, "rev-parse", "HEAD") == handoff["old"]
    assert [entry["read"][0]["form"] for entry in scene.reports()] == ["v1"]  # the previous generation read it
    assert body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)["outcome"] == "returned"


def test_a_multiprocessing_child_of_the_successor_is_another_entry_never_its_server(scene):
    """A spawn child re-imports ``server.py`` after taking its parent's ``sys.argv``: before the
    successor is ready, that child must neither ask for nor perform the return."""
    _passive_base(scene)
    scene.official({"web/style.css": STYLE_NEW})
    scene.activate(scene.prepare())

    failed = scene.boot(TOY_SPAWN_CHILD="1", TOY_WRITE_THEN_FAIL="1")

    assert failed.returncode != 0 and "successor not ready" in failed.stderr
    handoff = scene.handoff()
    assert [entry["kind"] for entry in handoff["entries"]] == ["server", "other"], handoff["entries"]
    assert "return" not in handoff and handoff["phase"] == "switched"
    assert (scene.serving / "web" / "style.css").read_text() == STYLE_NEW
    assert scene.boot().returncode == 0 and scene.handoff()["return"]["outcome"] == "returned"


def test_a_changed_writer_under_the_same_record_schema_is_never_returned(scene):
    """Negative control: the successor writes the same keys in a form the previous code does not read."""
    _passive_base(scene)
    scene.official({"web/style.css": STYLE_NEW, "ouroboros/mod_b.py": "GEN = 'GEN_OLD'\nFORM = 'v2'\n"})
    scene.activate(scene.prepare())
    assert scene.boot(TOY_WRITE_THEN_FAIL="1").returncode != 0

    again = scene.boot()

    decision = scene.handoff()["return"]
    assert decision["outcome"] == "unknown" and decision["reasons"][0] == "successor_entered:server"
    assert (scene.serving / "web" / "style.css").read_text() == STYLE_NEW  # nothing returned
    assert again.returncode == 0  # the new tree starts again; its record stays for recovery
    assert json.loads((scene.data / "records.jsonl").read_text().splitlines()[0])["form"] == "v2"  # previous reads v1


def _passive_switched(scene):
    _passive_base(scene)
    scene.official({"web/style.css": STYLE_NEW})
    scene.activate(scene.prepare())
    assert scene.boot(TOY_WRITE_THEN_FAIL="1").returncode != 0  # switched and entered
    return scene.handoff()


def test_a_stylesheet_only_return_still_needs_every_actual_fact(scene):
    handoff = _passive_switched(scene)
    assert body_switch.qualify_return(str(scene.serving), handoff) == ("compatible", [])
    descriptor = scene.data / "claudexor" / "daemon" / "control-api.json"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text("{}")  # an owned engine provisioned on this root after the switch
    assert body_switch.qualify_return(str(scene.serving), handoff) == ("unknown", ["engine_floor_unproven"])
    shutil.rmtree(scene.data / "claudexor")
    switched_env = {**handoff, "environment": {"state": "selected", "kind": "venv"}}
    assert body_switch.qualify_return(str(scene.serving), switched_env)[1] == ["environment_switched"]
    other_runtime = {**handoff, "qualification": {**handoff["qualification"],
                                                  "runtime": {**handoff["qualification"]["runtime"], "sqlite": "0"}}}
    assert body_switch.qualify_return(str(scene.serving), other_runtime)[1] == ["runtime_changed"]
    (scene.serving / "web" / "style.css").write_text("body { color: red; }\n")  # a later edit at the switched path
    assert body_switch.qualify_return(str(scene.serving), handoff) == ("unknown", ["later_edits:web/style.css"])
    (scene.serving / "web" / "style.css").write_text(STYLE_NEW)
    (scene.data / "state" / "panic_stop.flag").write_text("panic")
    assert body_switch.qualify_return(str(scene.serving), handoff) == ("refused", ["panic_stop"])


def test_an_owned_engine_home_present_before_the_update_never_proves_its_floor(scene):
    """Only the observed absence of any owned engine home proves the floor: a home present before the
    update, its descriptor unchanged throughout, stays unknown, and so does an unreadable one."""
    home = scene.data / "claudexor"
    descriptor = home / "daemon" / "control-api.json"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text("{}")
    before = descriptor.stat()
    handoff = _passive_switched(scene)
    after = descriptor.stat()
    assert handoff["qualification"]["owned_engine"] == "present"
    assert (before.st_ino, before.st_mtime_ns, before.st_size) == (after.st_ino, after.st_mtime_ns, after.st_size)
    assert body_switch.qualify_return(str(scene.serving), handoff) == ("unknown", ["engine_floor_unproven"])
    assert scene.boot().returncode == 0  # the real next entry
    assert (scene.handoff()["return"]["outcome"], scene.handoff()["return"]["reasons"]) == (
        "unknown", ["engine_floor_unproven"])
    assert (scene.serving / "web" / "style.css").read_text() == STYLE_NEW  # nothing returned
    recorded_absent = {**handoff, "qualification": {**handoff["qualification"], "owned_engine": "absent"}}
    home.chmod(0)  # present and unreadable: its state cannot be read at all
    try:
        assert body_switch.qualify_return(str(scene.serving), recorded_absent) == ("unknown", ["engine_floor_unproven"])
    finally:
        home.chmod(0o755)
    shutil.rmtree(home)
    unread = {**handoff, "qualification": {**handoff["qualification"], "owned_engine": "unreadable"}}
    assert body_switch.qualify_return(str(scene.serving), unread) == ("unknown", ["engine_floor_unproven"])
    locked = scene.tmp / "locked"
    (locked / "data").mkdir(parents=True)
    locked.chmod(0)
    try:
        assert body_switch.owned_engine(str(locked / "data")) == "unreadable"
    finally:
        locked.chmod(0o755)
    assert body_switch.qualify_return(str(scene.serving), recorded_absent) == ("compatible", [])  # absent both times


def test_an_edit_anywhere_after_the_switch_is_later_work_never_a_return(scene):
    """The stylesheet-only premise is the whole composition: an edit outside the switched path, a new
    script, or a staged one, after the switch keeps the decision unknown; the real entry returns nothing."""
    handoff = _passive_switched(scene)
    mod_a = scene.serving / "ouroboros" / "mod_a.py"
    original = mod_a.read_text()
    mod_a.write_text(original + "AFTER_SWITCH = 1\n")
    assert body_switch.qualify_return(str(scene.serving), handoff) == ("unknown", ["later_edits:ouroboros/mod_a.py"])
    mod_a.write_text(original)
    (scene.serving / "web" / "late.js").write_text("console.log('late');\n")
    assert body_switch.qualify_return(str(scene.serving), handoff) == ("unknown", ["later_edits:web/late.js"])
    git(scene.serving, "add", "web/late.js")
    assert body_switch.qualify_return(str(scene.serving), handoff) == ("unknown", ["later_edits:web/late.js"])
    git(scene.serving, "rm", "-q", "--cached", "web/late.js")
    (scene.serving / "web" / "late.js").unlink()
    assert body_switch.qualify_return(str(scene.serving), handoff) == ("compatible", [])  # the composition again
    mod_a.write_text(original + "AFTER_SWITCH = 1\n")

    entered = scene.boot()

    assert entered.returncode == 0, entered.stderr
    decision = scene.handoff()["return"]
    assert (decision["outcome"], decision["reasons"]) == ("unknown", ["later_edits:ouroboros/mod_a.py"])
    assert (scene.serving / "web" / "style.css").read_text() == STYLE_NEW and "AFTER_SWITCH" in mod_a.read_text()
    if not stage_then_edit_in_one_second(scene.serving, "ouroboros/mod_a.py", original, "GEN = 'GEN_NEW'\n"):
        pytest.skip("LABELLED CONTROLLED TIMING: the stage and a same-size edit never fit one second here")
    assert body_switch.qualify_return(str(scene.serving), handoff) == ("unknown", ["later_edits:ouroboros/mod_a.py"])


def test_a_stylesheet_return_keeps_every_layer_the_owner_had(scene):
    """Positive over the owner's committed, staged, unstaged and untracked work, carried through the switch:
    the whole-composition check counts none of it, and the return restores it exactly."""
    _passive_base(scene)
    scene.official({"web/style.css": STYLE_NEW})
    (scene.serving / "ouroboros" / "local.py").write_text("LOCAL = 1\n")
    git(scene.serving, "add", "-A")
    git(scene.serving, "commit", "-q", "-m", "local feature")
    mod_a = scene.serving / "ouroboros" / "mod_a.py"
    mod_a.write_text("GEN = 'GEN_OLD'\nSTAGED = 1\n")
    git(scene.serving, "add", "ouroboros/mod_a.py")
    mod_a.write_text("GEN = 'GEN_OLD'\nSTAGED = 1\nUNSTAGED = 1\n")
    (scene.serving / "local-scratch.txt").write_text("mine, untracked\n")
    before = scene.composition()
    handoff = scene.activate(scene.prepare())
    assert [row["path"] for row in handoff["switch"]] == ["web/style.css"]
    assert scene.boot(TOY_WRITE_THEN_FAIL="1").returncode != 0

    assert body_switch.qualify_return(str(scene.serving), scene.handoff()) == ("compatible", [])
    returned = scene.boot()

    assert returned.returncode == 0, returned.stderr
    assert scene.handoff()["return"]["outcome"] == "returned"
    assert scene.composition() == before  # HEAD, index and every file as the owner had them


@pytest.mark.parametrize("path, mode, why", [
    ("web/style.css", "100644", ""),
    ("web/modules/app.js", "100644", "not_passive:web/modules/app.js"),
    ("web/index.html", "100644", "not_passive:web/index.html"),
    ("web/icon.svg", "100644", "not_passive:web/icon.svg"),  # an image that can carry script
    ("docs/print.css", "100644", "not_passive:docs/print.css"),  # a stylesheet outside the served root
    ("ouroboros/report.css", "100644", "not_passive:ouroboros/report.css"),
    ("web/style.css", "100755", "not_regular:web/style.css"),
    ("web/style.css", "120000", "not_regular:web/style.css"),  # a link can point anywhere
])
def test_passive_means_a_regular_stylesheet_the_gateway_serves_never_an_extension_alone(path, mode, why):
    row = {"path": path, **{f"{side}{key}_{field}": value for side in ("old", "new") for key in ("", "_index", "_file")
                            for field, value in (("sha", "a" * 40), ("mode", mode))}}
    assert body_switch._passive_presentation([row], "web") == why
    assert body_switch._passive_presentation([row], "") == "web_root_unknown"


def test_a_return_keeps_its_why_for_the_panel_and_retains_the_outcome(scene, owned_state):
    scene.official({"ouroboros/mod_a.py": "GEN = (\n"})  # the official commit does not even parse
    scene.local_work(commit=False)
    before = scene.composition()
    row = scene.prepare()
    scene.activate(row)
    assert scene.boot().returncode == 0 and scene.composition() == before

    update_prepare.record_activation(scene.data, body_adoption.finalize_on_boot(scene.data, scene.serving,
                                                                                supervisor_ready=True))

    projection = update_prepare.projection()
    assert (projection["state"], projection["reason"]) == ("failed", "activation:returned")
    assert projection["activation"]["why"] == "import_probe_failed"
    last = projection["last_activation"]
    assert (last["outcome"], last["why"], last["trigger"]) == ("returned", "import_probe_failed", "import_probe_failed")
    assert last["sha"] == before["head"] and last["local_changes"] is True and last["candidate_id"] == row["candidate_id"]
