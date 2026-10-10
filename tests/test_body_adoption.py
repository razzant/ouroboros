"""Restart-bound adoption of a body candidate: real Git, real cold entries (#1539).

The serving body is a miniature tree carrying the REAL package-init hook and
the REAL switch helper. Entries are real processes: ``python server.py``, the
module CLI, a source launcher that imports the body itself, and a launcher
loop honouring exit 42 (a test double for the packaged launcher, whose own
frozen code is not exercised here).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from ouroboros import body_adoption, body_candidate, body_switch
from tests.body_candidate_support import (
    RICH_CHANGE, candidate_commit, git, isolate, make_ctx, make_serving, restart_receipt, rich_commit,
    run_entry,
)

pytestmark = pytest.mark.serial  # real cold-entry processes throughout

STOPPED = {"state": "completed", "unconfirmed": []}
EXITED = {"doomed": [4001, 4002], "dead": [4001, 4002], "unconfirmed": [], "cleanup_ok": True, "snapshot_ok": True}


class Scene:
    def __init__(self, tmp_path, monkeypatch, task_id="root-adopt"):
        self.serving = make_serving(tmp_path)
        self.data = isolate(monkeypatch, tmp_path)
        monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
        self.report = tmp_path / "report.jsonl"
        self.ctx = make_ctx(self.serving, self.data, task_id)
        body_candidate.prepare(self.ctx)
        self.candidate = pathlib.Path(self.ctx.repo_dir)
        self.old = git(self.serving, "rev-parse", "HEAD")
        self.cand = ""

    def commit(self, **kwargs):
        self.cand = candidate_commit(self.candidate, **kwargs) if kwargs else rich_commit(self.candidate)
        body_candidate.record_reviewed_commit(self.ctx, self.cand)
        return self.cand

    def authorize(self, reason="adopt it"):
        handoff = body_adoption.authorize(self.ctx, self.cand, reason=reason)
        restart_receipt(self.data, self.cand, reason)  # what request_restart writes beside it
        return handoff

    def arm(self, reason="adopt it", **overrides):
        ok, msg = body_adoption.bind_restart(lambda **_kw: (True, "ok"), self.data, reason)(
            reason="agent_restart_request", unsynced_policy="rescue_and_block")
        assert ok, msg
        return body_adoption.arm(self.data, **{"worker_exits": EXITED, "live_children": [], "owned_stop": STOPPED,
                                                "owner_restart": False, **overrides})

    def boot(self, *argv, entry="server.py", **env):
        return run_entry(self.serving, *argv, entry=entry, env={"TOY_REPORT": str(self.report), **env})

    def reports(self):
        return [json.loads(line) for line in self.report.read_text().splitlines()] if self.report.exists() else []

    def phase(self):
        return body_adoption.read(self.data).get("phase", "")

    def attest(self):
        """What the booting process records before its imports (the hook's ``switched`` branch)."""
        body_switch._attest_loaded(str(self.serving), body_adoption.read(self.data))
        return getattr(sys, "_ouroboros_body_generation", {})

    def pointer(self):
        return pathlib.Path(body_switch.pointer_path(str(self.serving)))

    def switch_paths(self):
        return [*RICH_CHANGE, "server.py", "ouroboros/gone.py"]

    def side_of(self, path):
        """'old' / 'new' / 'foreign' for the bytes on disk at one switch path."""
        full = self.serving / path
        sides = {}
        for name, rev in (("old", self.old), ("new", self.cand)):
            shown = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=self.serving, capture_output=True)
            sides[name] = shown.stdout if shown.returncode == 0 else None
        have = full.read_bytes() if full.exists() else None
        return "new" if have == sides["new"] else "old" if have == sides["old"] else "foreign"


@pytest.fixture
def scene(tmp_path, monkeypatch):
    return Scene(tmp_path, monkeypatch)


def _adopted_tree_checks(scene):
    assert git(scene.serving, "rev-parse", "HEAD") == scene.cand
    assert git(scene.serving, "rev-parse", "--abbrev-ref", "HEAD") == "ouroboros"
    assert git(scene.serving, "status", "--porcelain") == ""
    assert not (scene.serving / "ouroboros/gone.py").exists()
    assert (scene.serving / "web/sentinel.txt").read_text() == "candidate ui\n"


# --------------------------------------------------------------------------- #
# Nothing adopts by itself
# --------------------------------------------------------------------------- #
def test_plain_boots_and_an_unarmed_authorization_adopt_nothing(scene):
    scene.commit()
    assert scene.boot().returncode == 0  # candidate exists, nothing authorized
    scene.authorize()
    assert scene.phase() == "authorized" and not scene.pointer().exists()
    first = scene.boot()
    assert first.returncode == 0, first.stderr
    assert [r["a"] for r in scene.reports()] == ["GEN_OLD", "GEN_OLD"]
    assert git(scene.serving, "rev-parse", "HEAD") == scene.old and git(scene.serving, "status", "--porcelain") == ""
    # The next generation reports the authorization that no restart armed, and closes it.
    result = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)
    assert result["outcome"] == "not_applied" and "unchanged" in result["note"]
    assert scene.phase() == "" and git(scene.serving, "for-each-ref", body_adoption.PIN_PREFIX) == ""
    assert scene.candidate.is_dir()  # the candidate itself is retained


# --------------------------------------------------------------------------- #
# The real entries
# --------------------------------------------------------------------------- #
def test_direct_server_entry_switches_before_body_imports_and_keeps_argv(scene):
    scene.commit()
    scene.authorize()
    (scene.serving / "owner-notes.txt").write_text("unrelated owner file\n")
    assert scene.arm() == "armed" and scene.pointer().exists()

    booted = scene.boot("--port", "8765")

    assert booted.returncode == 0, booted.stderr
    # One report only: the generation that switched never reached a body import.
    assert scene.reports() == [{"server": "GEN_CAND", "a": "GEN_CAND", "b": "GEN_CAND",
                                "pid": scene.reports()[0]["pid"], "argv": ["--port", "8765"]}]
    _adopted_tree_checks_with_dirt = git(scene.serving, "status", "--porcelain")
    assert _adopted_tree_checks_with_dirt == "?? owner-notes.txt"
    assert (scene.serving / "owner-notes.txt").read_text() == "unrelated owner file\n"
    assert scene.phase() == "switched"
    assert body_adoption.holds_checkout(scene.data, scene.serving) is True
    assert git(scene.serving, "reflog", "-1", "--format=%gs", "refs/heads/ouroboros").startswith("body adoption")

    # Disk HEAD alone proves nothing about what THIS process imported: unattested stays unconfirmed.
    unattested = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)
    assert unattested["outcome"] == "unconfirmed" and "did not verify" in unattested["note"]
    assert scene.attest()["sha"] == scene.cand and scene.attest()["later_edits"] == []
    not_ready = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=False)
    assert not_ready["outcome"] == "unconfirmed" and scene.phase() == "switched"
    unproven = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True,
                                              native_smoke=lambda: {"ok": False, "error": "host artifact"})
    assert unproven["outcome"] == "unconfirmed" and scene.phase() == "switched"
    adopted = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True,
                                             native_smoke=lambda: {"ok": True})
    assert adopted["outcome"] == "adopted" and adopted["sha"] == scene.cand
    assert scene.phase() == "" and not scene.pointer().exists()
    assert not body_adoption.helper_dir(scene.data).exists()
    assert git(scene.serving, "for-each-ref", body_adoption.PIN_PREFIX) == ""
    closed = [json.loads(line) for line in (scene.data / "logs/supervisor.jsonl").read_text().splitlines()]
    assert closed[-1]["type"] == "body_adoption_closed" and closed[-1]["outcome"] == "adopted"
    assert scene.boot().returncode == 0 and scene.reports()[-1]["a"] == "GEN_CAND"  # ordinary boots stay ordinary


def test_module_cli_entry_reaches_the_hook_through_the_package_import(scene):
    scene.commit()
    scene.authorize()
    scene.arm()

    booted = scene.boot("status", "--json", entry="ouroboros.cli")

    assert booted.returncode == 0, booted.stderr
    assert scene.reports() == [{"entry": "cli", "a": "GEN_CAND", "b": "GEN_CAND", "argv": ["status", "--json"]}]
    _adopted_tree_checks(scene)


def test_source_launcher_cold_start_switches_inside_the_launcher_process(scene):
    scene.commit()
    scene.authorize()
    scene.arm()

    booted = scene.boot(entry="launcher.py")

    assert booted.returncode == 0, booted.stderr
    # The launcher process itself re-executed on the new tree before starting the server.
    assert "LAUNCHER_GEN=GEN_CAND codes=[0]" in booted.stdout
    assert [r["server"] for r in scene.reports()] == ["GEN_CAND"]
    _adopted_tree_checks(scene)


def test_launcher_managed_server_hands_over_with_exit_42_and_leaves_dependencies_to_the_launcher(scene, tmp_path):
    scene.commit()
    handoff = scene.authorize()
    marker = tmp_path / "deps-ran.txt"
    handoff["deps_command"] = [sys.executable, "-c", f"open({str(marker)!r}, 'a').write('ran')"]
    body_switch.write_handoff(str(body_adoption.helper_dir(scene.data)), handoff)
    scene.arm()

    # A launcher that is already running and whose code is NOT the tree's (the packaged launcher's
    # position): it starts repo/server.py as launcher-managed and only relaunches on exit 42. It is a
    # separate process because the suite strips the launcher marker from the test's own children.
    loop = tmp_path / "packaged_launcher_double.py"
    loop.write_text(
        "import os, subprocess, sys\n"
        "codes = []\n"
        "for _ in range(3):\n"
        "    codes.append(subprocess.run([sys.executable, 'server.py'], cwd=sys.argv[1],\n"
        "                 env=dict(os.environ, OUROBOROS_MANAGED_BY_LAUNCHER='1')).returncode)\n"
        "    if codes[-1] != 42:\n"
        "        break\n"
        "print('codes=%s' % codes)\n", encoding="utf-8")
    ran = subprocess.run([sys.executable, str(loop), str(scene.serving)], capture_output=True, text=True,
                         timeout=120, env={**os.environ, "TOY_REPORT": str(scene.report)})

    assert "codes=[42, 0]" in ran.stdout, ran.stdout + ran.stderr
    assert [r["server"] for r in scene.reports()] == ["GEN_CAND"]
    assert not marker.exists()  # the launcher's own install step owns dependencies after the 42
    _adopted_tree_checks(scene)


def test_direct_mode_syncs_dependencies_after_the_switch_and_a_failure_returns_the_tree(scene, tmp_path):
    scene.commit()
    handoff = scene.authorize()
    order = tmp_path / "order.txt"
    probe = ("import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("
             "pathlib.Path('ouroboros/mod_a.py').read_text()); sys.exit(int(sys.argv[2]))")
    handoff["deps_command"] = [sys.executable, "-c", probe, str(order), "7"]
    body_switch.write_handoff(str(body_adoption.helper_dir(scene.data)), handoff)
    before = {p: (scene.serving / p).read_bytes() for p in scene.switch_paths() if (scene.serving / p).exists()}
    scene.arm()

    booted = scene.boot()

    assert booted.returncode == 0, booted.stderr
    assert "GEN_CAND" in order.read_text()  # the install saw the NEW tree, with the old readers gone
    assert scene.reports() == [{"server": "GEN_OLD", "a": "GEN_OLD", "b": "GEN_OLD",
                                "pid": scene.reports()[0]["pid"], "argv": []}]
    assert git(scene.serving, "rev-parse", "HEAD") == scene.old
    assert git(scene.serving, "status", "--porcelain") == ""
    assert {p: (scene.serving / p).read_bytes() for p in before} == before
    assert not (scene.serving / "ouroboros/added.py").exists()
    result = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)
    assert result["outcome"] == "not_applied" and "dependency_sync_failed:rc=7" in result["note"]
    assert "may already have changed" in result["note"]  # Git returned; the interpreter is not claimed restored
    closed = json.loads((scene.data / "logs/supervisor.jsonl").read_text().splitlines()[-1])
    assert closed["interpreter_state"] == "may_have_changed"


# --------------------------------------------------------------------------- #
# Interruption at every boundary: old-or-new bytes, then convergence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("boundary,head_after", [
    ("mid_files", "old"), ("after_files", "old"), ("before_ref", "old"), ("after_ref", "cand")])
def test_interrupted_switch_never_leaves_a_torn_file_and_the_next_entry_converges(scene, boundary, head_after):
    scene.commit()
    scene.authorize()
    (scene.serving / "owner-notes.txt").write_text("unrelated owner file\n")
    (scene.serving / "VERSION").write_text("1.0.0-owner-edit\n")  # tracked edit OUTSIDE the switch set
    scene.arm()

    died = scene.boot(OUROBOROS_BODY_SWITCH_FAIL_AT=boundary)

    assert died.returncode == 137 and scene.reports() == []  # no body module was imported
    assert scene.phase() == "switching"
    assert git(scene.serving, "rev-parse", "HEAD") == (scene.old if head_after == "old" else scene.cand)
    sides = {path: scene.side_of(path) for path in scene.switch_paths()}
    assert "foreign" not in sides.values(), sides
    if boundary == "mid_files":
        assert set(sides.values()) == {"old", "new"}  # genuinely mid-way, each file whole
    assert not [p for p in scene.serving.rglob(".body-switch-*")]

    resumed = scene.boot()

    assert resumed.returncode == 0, resumed.stderr
    assert [r["server"] for r in scene.reports()] == ["GEN_CAND"]
    assert git(scene.serving, "rev-parse", "HEAD") == scene.cand
    assert git(scene.serving, "status", "--porcelain").splitlines() == [" M VERSION", "?? owner-notes.txt"]
    assert (scene.serving / "VERSION").read_text() == "1.0.0-owner-edit\n"
    assert all(scene.side_of(path) == "new" for path in scene.switch_paths())
    assert body_adoption.read(scene.data)["attempts"] == 2


# --------------------------------------------------------------------------- #
# Later owner work is never overwritten
# --------------------------------------------------------------------------- #
def test_edit_made_after_arming_abandons_with_the_tree_untouched(scene):
    scene.commit()
    scene.authorize()
    scene.arm()
    (scene.serving / "ouroboros/mod_b.py").write_text("GEN = 'OWNER_LATER'\n")
    (scene.serving / "ouroboros/added.py").write_text("owner created this path first\n")

    booted = scene.boot()

    assert booted.returncode == 0, booted.stderr
    assert scene.reports()[0] | {"pid": 0} == {"server": "GEN_OLD", "a": "GEN_OLD", "b": "OWNER_LATER",
                                               "pid": 0, "argv": []}
    assert git(scene.serving, "rev-parse", "HEAD") == scene.old
    assert (scene.serving / "ouroboros/added.py").read_text() == "owner created this path first\n"
    handoff = body_adoption.read(scene.data)
    assert handoff["phase"] == "abandoned" and handoff["events"][-1]["detail"].startswith("switch_set_dirty")
    assert not scene.pointer().exists()


def test_foreign_bytes_met_mid_switch_are_kept_and_our_writes_are_returned(scene):
    scene.commit()
    scene.authorize()
    scene.arm()
    assert scene.boot(OUROBOROS_BODY_SWITCH_FAIL_AT="mid_files").returncode == 137
    switched = [p for p in scene.switch_paths() if scene.side_of(p) == "new"]
    victim = "web/sentinel.txt"  # a path the candidate ADDS, not yet written when the switch died
    assert "ouroboros/mod_a.py" in switched and scene.side_of(victim) == "old"
    (scene.serving / "web").mkdir(exist_ok=True)
    (scene.serving / victim).write_text("owner created this while the switch was interrupted\n")

    booted = scene.boot()

    # Never rescue-then-overwrite: the owner's bytes stay, our half-switch is undone, HEAD never moved.
    assert (scene.serving / victim).read_text() == "owner created this while the switch was interrupted\n"
    assert git(scene.serving, "rev-parse", "HEAD") == scene.old
    others = [p for p in scene.switch_paths() if p != victim]
    assert all(scene.side_of(p) == "old" for p in others), {p: scene.side_of(p) for p in others}
    assert git(scene.serving, "status", "--porcelain") == "?? web/"
    handoff = body_adoption.read(scene.data)
    assert handoff["phase"] == "abandoned" and handoff["events"][-1]["detail"] == f"foreign_content:{victim}"
    assert not scene.pointer().exists()
    # The same process then imports the coherent OLD tree: no mixed generation is ever reported.
    assert booted.returncode == 0, booted.stderr
    assert [(r["server"], r["a"], r["b"]) for r in scene.reports()] == [("GEN_OLD", "GEN_OLD", "GEN_OLD")]


def test_commit_made_before_the_switch_abandons_and_one_made_mid_switch_stops_every_boot(scene):
    scene.commit()
    scene.authorize()
    scene.arm()
    (scene.serving / "owner.txt").write_text("owner commit\n")
    git(scene.serving, "add", "owner.txt")
    git(scene.serving, "commit", "-q", "-m", "owner moved the base")
    moved = git(scene.serving, "rev-parse", "HEAD")

    assert scene.boot().returncode == 0
    assert body_adoption.read(scene.data)["events"][-1]["detail"] == f"base_moved:{moved[:12]}"
    assert git(scene.serving, "rev-parse", "HEAD") == moved and git(scene.serving, "status", "--porcelain") == ""


def test_head_moved_during_an_interrupted_switch_is_stuck_and_never_falls_through(scene):
    scene.commit()
    scene.authorize()
    scene.arm()
    assert scene.boot(OUROBOROS_BODY_SWITCH_FAIL_AT="mid_files").returncode == 137
    (scene.serving / "owner.txt").write_text("owner commit\n")
    git(scene.serving, "add", "owner.txt")
    git(scene.serving, "commit", "-q", "-m", "owner committed during the interrupted switch")
    snapshot = {p: scene.side_of(p) for p in scene.switch_paths()}

    first = scene.boot()
    second = scene.boot(entry="ouroboros.cli")

    assert first.returncode == body_switch.STUCK_EXIT_CODE and second.returncode == body_switch.STUCK_EXIT_CODE
    assert "neither the recorded base nor the candidate" in first.stderr and "needs a decision" in second.stderr
    assert str(scene.pointer()) in second.stderr  # the message names the one file that ends the hold
    assert scene.reports() == []  # a mixed tree was never imported, on either entry
    assert {p: scene.side_of(p) for p in scene.switch_paths()} == snapshot  # nothing further overwritten
    assert scene.phase() == "stuck" and scene.pointer().exists()
    assert body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True) == {}


def test_transient_failure_mid_switch_stops_this_boot_and_the_next_one_retries(scene):
    scene.commit()
    scene.authorize()
    scene.arm()
    lock = pathlib.Path(body_switch.git_dir(str(scene.serving))) / "index.lock"
    lock.write_text("held by another git process\n")  # update-index cannot take the index now

    blocked = scene.boot()

    assert blocked.returncode == body_switch.STUCK_EXIT_CODE and "the next start retries it" in blocked.stderr
    assert scene.reports() == [] and scene.phase() == "switching"  # not "stuck": nothing needs a decision
    assert git(scene.serving, "rev-parse", "HEAD") == scene.old
    lock.unlink()

    assert scene.boot().returncode == 0
    assert [r["server"] for r in scene.reports()] == ["GEN_CAND"] and git(scene.serving, "status", "--porcelain") == ""


def test_pending_transition_with_an_unreadable_helper_refuses_the_boot(scene):
    scene.commit()
    scene.authorize()
    scene.arm()
    (body_adoption.helper_dir(scene.data) / "switch.py").unlink()

    booted = scene.boot()

    assert booted.returncode == 3 and "refusing to import a possibly mixed tree" in booted.stderr
    assert scene.reports() == []


# --------------------------------------------------------------------------- #
# Arming needs the existing stop owners' confirmation; newer controls govern
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("overrides,blocker", [
    ({"owned_stop": {"state": "unconfirmed", "unconfirmed": [{"pid": 4242}]}}, "owned_work_unconfirmed"),
    ({"worker_exits": None}, "worker_exits_unconfirmed"),  # kill_workers never reached its census
    ({"worker_exits": {**EXITED, "unconfirmed": [4002]}}, "worker_exits_unconfirmed"),  # a worker outlived the join
    ({"live_children": [object()]}, "child_processes_alive"),
    ({"owner_restart": True}, "owner_restart"),
    ({"owned_stop": None}, "owned_work_unconfirmed"),
])
def test_surviving_readers_or_a_newer_owner_control_never_arm(scene, overrides, blocker):
    scene.commit()
    scene.authorize()

    assert scene.arm(**overrides) == "authorized"

    assert not scene.pointer().exists()
    assert scene.boot().returncode == 0 and scene.reports()[0]["a"] == "GEN_OLD"
    assert git(scene.serving, "rev-parse", "HEAD") == scene.old
    result = body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)
    assert result["outcome"] == "not_applied" and blocker in result["note"]


def test_only_the_restart_that_carried_the_adoption_can_arm_and_panic_abandons(scene):
    scene.commit()
    scene.authorize(reason="adopt it")
    # A different restart (or the exit tail of a plain one) arms nothing.
    assert body_adoption.arm(scene.data, worker_exits=EXITED, live_children=[], owned_stop=STOPPED,
                             owner_restart=False) == "authorized"
    ok, _msg = body_adoption.bind_restart(lambda **_kw: (True, "ok"), scene.data, "some other restart")()
    assert ok and not body_adoption.read(scene.data).get("restart_bound")
    # Another task's plain restart with the SAME reason carries its own receipt (the serving commit).
    restart_receipt(scene.data, scene.old, "adopt it")
    ok, _msg = body_adoption.bind_restart(lambda **_kw: (True, "ok"), scene.data, "adopt it")()
    assert ok and not body_adoption.read(scene.data).get("restart_bound")
    assert not scene.pointer().exists()
    restart_receipt(scene.data, scene.cand, "adopt it")

    (scene.data / "state" / "panic_stop.flag").write_text("panic")
    assert scene.arm() == "authorized"  # Panic before the exit tail: never armed, nothing waited for
    (scene.data / "state" / "panic_stop.flag").unlink()
    assert scene.arm() == "armed"
    (scene.data / "state" / "panic_stop.flag").write_text("panic")  # Panic after arming
    assert scene.boot().returncode == 0 and scene.reports()[0]["a"] == "GEN_OLD"
    assert body_adoption.read(scene.data)["events"][-1]["detail"] == "panic_stop"
    assert git(scene.serving, "rev-parse", "HEAD") == scene.old


def test_refused_or_moved_restart_abandons_the_authorization(scene):
    scene.commit()
    scene.authorize()
    ok, msg = body_adoption.bind_restart(
        lambda **_kw: (False, "Unsynced state rescued; restart blocked."), scene.data, "adopt it")()
    assert not ok and "was not armed" in msg and scene.phase() == ""

    scene.authorize()
    git(scene.serving, "commit", "-q", "--allow-empty", "-m", "moved")
    ok, msg = body_adoption.bind_restart(lambda **_kw: (True, "ok"), scene.data, "adopt it")()
    assert not ok and "moved to" in msg and scene.phase() == ""
    assert git(scene.serving, "for-each-ref", body_adoption.PIN_PREFIX) == ""


# --------------------------------------------------------------------------- #
# Authorization: exact, reviewed, built on the serving commit
# --------------------------------------------------------------------------- #
def _refusal(scene, **kwargs):
    with pytest.raises(body_adoption.AdoptionRefused) as refused:
        body_adoption.authorize(scene.ctx, kwargs.get("commit", scene.cand), reason="adopt it")
    return refused.value.code


def test_authorization_refusals_leave_the_serving_tree_and_no_handoff(scene, monkeypatch):
    before = git(scene.serving, "rev-parse", "HEAD")
    raw = candidate_commit(scene.candidate)  # committed by a shell, not by commit_reviewed
    scene.cand = raw
    assert _refusal(scene) == "ADOPTION_UNREVIEWED"
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")  # Cyber records it instead of refusing
    assert body_adoption.authorize(scene.ctx, raw, reason="adopt it")["unreviewed_commits"] == [raw]
    body_adoption.abandon(scene.data, "test")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    body_candidate.record_reviewed_commit(scene.ctx, raw)

    assert _refusal(scene, commit="deadbeef") == "ADOPTION_COMMIT_UNKNOWN"
    assert _refusal(scene, commit=before) == "ADOPTION_NOTHING_TO_ADOPT"

    (scene.serving / "ouroboros/mod_a.py").write_text("GEN = 'OWNER'\n")
    assert _refusal(scene) == "ADOPTION_SWITCH_SET_DIRTY"
    git(scene.serving, "checkout", "--", "ouroboros/mod_a.py")

    bible = scene.commit(files={"BIBLE.md": "# Constitution\n\nAmended.\n"})
    assert _refusal(scene) == "ADOPTION_CONSTITUTION_NEEDS_RELEASE"
    release = scene.commit(files={"VERSION": "1.0.1\n"})  # the same change as a numbered release
    assert body_adoption.authorize(scene.ctx, release, reason="adopt it")["cand"] == release
    body_adoption.abandon(scene.data, "test")
    assert bible != release

    hookless = scene.commit(files={"ouroboros/__init__.py": "from .version import get_version\n"})
    assert _refusal(scene) == "ADOPTION_HOOK_MISSING"
    assert hookless and scene.phase() == ""
    assert git(scene.serving, "rev-parse", "HEAD") == before and git(scene.serving, "status", "--porcelain") == ""
    assert git(scene.serving, "for-each-ref", body_adoption.PIN_PREFIX) == ""

    unbound = make_ctx(scene.serving, scene.data, "no-candidate")
    with pytest.raises(body_adoption.AdoptionRefused) as refused:
        body_adoption.authorize(unbound, release, reason="x")
    assert refused.value.code == "ADOPTION_NO_CANDIDATE"


def test_moved_or_rolled_back_base_needs_a_reviewed_merge_and_a_second_adoption_works(tmp_path, monkeypatch):
    scene = Scene(tmp_path, monkeypatch, task_id="first")
    scene.commit()
    scene.authorize()
    scene.arm()
    assert scene.boot().returncode == 0
    scene.attest()
    assert body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)["outcome"] == "adopted"
    first_adopted = git(scene.serving, "rev-parse", "HEAD")

    # A candidate prepared BEFORE that adoption is stale: it does not build on the serving commit.
    stale_ctx = make_ctx(scene.serving, scene.data, "stale")
    stale_row_base = scene.old
    stale = body_candidate._provision(scene.serving, "stale", None)
    git(stale["path"], "reset", "-q", "--hard", stale_row_base)
    body_candidate._update_row(stale, lambda entry: entry.__setitem__("base_sha", stale_row_base))
    body_candidate.restore(stale_ctx)
    stale_sha = candidate_commit(stale_ctx.repo_dir, files={"ouroboros/extra.py": "GEN = 'STALE'\n"})
    body_candidate.record_reviewed_commit(stale_ctx, stale_sha)
    with pytest.raises(body_adoption.AdoptionRefused) as refused:
        body_adoption.authorize(stale_ctx, stale_sha, reason="second")
    assert refused.value.code == "ADOPTION_BASE_MOVED"

    # Merging the serving commit into the candidate, through review, makes it adoptable.
    git(stale_ctx.repo_dir, "merge", "-q", "--no-edit", first_adopted)
    merged = git(stale_ctx.repo_dir, "rev-parse", "HEAD")
    body_candidate.record_reviewed_commit(stale_ctx, merged)
    handoff = body_adoption.authorize(stale_ctx, merged, reason="second")
    restart_receipt(scene.data, merged, "second")
    assert handoff["old"] == first_adopted and [r["path"] for r in handoff["switch"]] == ["ouroboros/extra.py"]
    ok, _ = body_adoption.bind_restart(lambda **_kw: (True, "ok"), scene.data, "second")()
    assert ok and body_adoption.arm(scene.data, worker_exits=EXITED, live_children=[], owned_stop=STOPPED,
                                    owner_restart=False) == "armed"
    assert scene.boot().returncode == 0
    assert git(scene.serving, "rev-parse", "HEAD") == merged and git(scene.serving, "status", "--porcelain") == ""
    scene.attest()
    assert body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)["outcome"] == "adopted"

    # A serving checkout rolled BACK behind a candidate's base is not silently rolled forward again.
    fresh_ctx = make_ctx(scene.serving, scene.data, "fresh")
    body_candidate.prepare(fresh_ctx)
    fresh_sha = candidate_commit(fresh_ctx.repo_dir, files={"ouroboros/more.py": "GEN = 'FRESH'\n"})
    body_candidate.record_reviewed_commit(fresh_ctx, fresh_sha)
    git(scene.serving, "reset", "-q", "--hard", first_adopted)
    with pytest.raises(body_adoption.AdoptionRefused) as rolled_back:
        body_adoption.authorize(fresh_ctx, fresh_sha, reason="third")
    assert rolled_back.value.code == "ADOPTION_BASE_MOVED"


def test_unnumbered_adoption_does_not_conflict_with_the_next_official_release_on_version_carriers(scene, tmp_path):
    official = tmp_path / "official.git"
    git(tmp_path, "clone", "-q", "--bare", str(scene.serving), str(official))
    scene.commit(files={"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})  # version-neutral: VERSION untouched
    scene.authorize()
    scene.arm()
    assert scene.boot().returncode == 0
    assert (scene.serving / "VERSION").read_text() == "1.0.0\n"  # identity is the commit, not a number

    upstream = tmp_path / "upstream"
    git(tmp_path, "clone", "-q", str(official), str(upstream))
    (upstream / "VERSION").write_text("1.1.0\n")
    (upstream / "ouroboros/mod_b.py").write_text("GEN = 'OFFICIAL_1_1_0'\n")
    git(upstream, "commit", "-q", "-am", "official release 1.1.0")
    git(upstream, "push", "-q", "origin", "HEAD:ouroboros")

    git(scene.serving, "fetch", "-q", str(official), "ouroboros")
    merged = subprocess.run(["git", "merge", "--no-edit", "FETCH_HEAD"], cwd=scene.serving, capture_output=True,
                            text=True, env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e.invalid",
                                            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e.invalid"})
    assert merged.returncode == 0, merged.stdout + merged.stderr
    assert (scene.serving / "VERSION").read_text() == "1.1.0\n"
    assert (scene.serving / "ouroboros/mod_a.py").read_text() == "GEN = 'GEN_CAND'\n"  # the adoption survives
    assert git(scene.serving, "merge-base", "--is-ancestor", scene.cand, "HEAD", check=False) == ""


# --------------------------------------------------------------------------- #
# The restart tool and the supervisor's evolution path
# --------------------------------------------------------------------------- #
def test_request_restart_adopts_only_when_the_exact_commit_is_named(scene):
    from ouroboros.tools.registry import ToolRegistry

    scene.commit()
    registry = ToolRegistry(repo_dir=scene.serving, drive_root=scene.data)
    registry.set_context(scene.ctx)
    marker = scene.data / "state" / "pending_restart_verify.json"

    plain = registry.execute("request_restart", {"reason": "just restart"})
    assert "NOT adopted" in plain and scene.phase() == ""
    assert json.loads(marker.read_text())["expected_sha"] == scene.old  # the SERVING commit, not the candidate's

    refused = registry.execute("request_restart", {"reason": "adopt", "adopt_commit": "0" * 40})
    assert "RESTART_BLOCKED: ADOPTION_COMMIT_UNKNOWN" in refused

    adopting = registry.execute("request_restart", {"reason": "adopt reviewed fix", "adopt_commit": scene.cand})
    assert f"adopts candidate commit {scene.cand[:12]} onto ouroboros" in adopting
    written = json.loads(marker.read_text())
    assert (written["expected_sha"], written["expected_branch"], written["reason"]) == (
        scene.cand, "ouroboros", "adopt reviewed fix")
    handoff = body_adoption.read(scene.data)
    assert handoff["phase"] == "authorized" and handoff["reason"] == "adopt reviewed fix"
    assert scene.ctx.pending_restart_reason == "adopt reviewed fix"
    assert body_adoption.authorized_base(scene.data, scene.cand, "adopt reviewed fix") == scene.old
    assert body_adoption.authorized_base(scene.data, scene.cand, "another reason") == ""
    assert git(scene.serving, "rev-parse", "HEAD") == scene.old  # authorizing touched nothing


def test_supervisor_evolution_restart_authorizes_the_candidate_commit_or_stops(scene):
    scene.commit()
    # No candidate for that task: a commit made in the serving checkout needs no adoption.
    assert body_adoption.authorize_for_task(scene.data, "other-task", scene.old, "evo") is True
    assert scene.phase() == ""
    assert body_adoption.authorize_for_task(scene.data, "root-adopt", scene.cand, "evo") is True
    assert body_adoption.read(scene.data)["reason"] == "evo"
    first_id = body_adoption.read(scene.data)["id"]
    assert body_adoption.authorize_for_task(scene.data, "root-adopt", scene.cand, "evo") is True
    assert body_adoption.read(scene.data)["id"] == first_id  # the task's own authorization is reused
    body_adoption.abandon(scene.data, "test")

    (scene.serving / "ouroboros/mod_a.py").write_text("GEN = 'OWNER'\n")
    assert body_adoption.authorize_for_task(scene.data, "root-adopt", scene.cand, "evo") is False
    assert (scene.serving / "ouroboros/mod_a.py").read_text() == "GEN = 'OWNER'\n"


# --------------------------------------------------------------------------- #
# The REAL entry files keep the hook reachable before any mutable body import
# --------------------------------------------------------------------------- #
def _imports_before_first_body_import(path: pathlib.Path) -> tuple[list[str], str]:
    import ast

    before: list[str] = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                 else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        for name in names:
            if name.split(".")[0] in ("ouroboros", "supervisor"):
                return before, name
            before.append(name.split(".")[0])
    return before, ""


@pytest.mark.parametrize("entry", ["server.py", "launcher.py"])
def test_real_entries_reach_the_package_hook_through_stdlib_only(entry):
    repo = pathlib.Path(__file__).resolve().parents[1]

    before, first_body = _imports_before_first_body_import(repo / entry)

    assert first_body.startswith("ouroboros"), first_body  # the package init (the hook) is the first body code
    assert set(before) <= set(sys.stdlib_module_names), sorted(set(before) - set(sys.stdlib_module_names))


def test_real_hook_runs_before_the_package_imports_anything_and_the_helper_is_stdlib_only():
    import ast

    repo = pathlib.Path(__file__).resolve().parents[1]
    init = ast.parse((repo / "ouroboros" / "__init__.py").read_text(encoding="utf-8")).body
    call = next(i for i, node in enumerate(init) if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == body_adoption.HOOK_MARKER)
    first_import = next(i for i, node in enumerate(init) if isinstance(node, (ast.Import, ast.ImportFrom)))
    assert call < first_import
    hook = next(node for node in init if isinstance(node, ast.FunctionDef) and node.name == body_adoption.HOOK_MARKER)
    hook_imports = {alias.name for node in ast.walk(hook) if isinstance(node, ast.Import) for alias in node.names}
    assert hook_imports <= {"os", "sys"} and not [n for n in ast.walk(hook) if isinstance(n, ast.ImportFrom)]

    helper = ast.parse((repo / "ouroboros" / "body_switch.py").read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(helper) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {(node.module or "").split(".")[0] for node in ast.walk(helper) if isinstance(node, ast.ImportFrom)}
    assert imported <= set(sys.stdlib_module_names), imported
    # The module CLI has no body import of its own ahead of the package: `-m` imports the package first.
    cli_before, _ = _imports_before_first_body_import(repo / "ouroboros" / "cli.py")
    assert set(cli_before) <= set(sys.stdlib_module_names)
