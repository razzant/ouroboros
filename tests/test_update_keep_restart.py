"""Restart and cold boot keep the owner's composition (owner decision 3): real Git, real rescue.

``keep`` never rescues, resets or cleans the dev branch. A composition that no longer
imports is never replaced by its last commit: an unverified edit is copied to a rescue
snapshot and stays in place, a composition that was healthy names its environment, and
no restart starts an older branch on the current data (the stable fallback is retired).
"""
from __future__ import annotations

import pathlib

import pytest

from supervisor import git_ops, git_ops_reset, update_prepare
from tests.body_candidate_support import git, make_serving
from tests.test_update_activation import Scene

pytestmark = pytest.mark.serial


@pytest.fixture(params=["provider", "providerless"])
def repo(tmp_path, monkeypatch, request):
    """A dirty dev checkout on a root whose supervisor state was initialized, or on one no provider ever started
    (only ``init_state`` creates that state): the keep facts are the Update's retained ones on both."""
    from supervisor import state

    serving = make_serving(tmp_path)
    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    (data / "logs").mkdir(parents=True)
    for name, path in {"DRIVE_ROOT": data, "STATE_PATH": data / "state/state.json",
                       "STATE_LAST_GOOD_PATH": data / "state/state.last_good.json",
                       "STATE_LOCK_PATH": data / "locks/state.lock"}.items():
        monkeypatch.setattr(state, name, path)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    if request.param == "provider":
        assert state.init_state().quality == "current"
    else:
        assert not (data / "state" / "state.json").exists()
        assert state.load_state()["_state_read"]["quality"] == "uninitialized"
    monkeypatch.setattr(git_ops, "REPO_DIR", serving)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", data)
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    monkeypatch.setattr(git_ops, "BRANCH_STABLE", "ouroboros-stable")
    monkeypatch.setattr(git_ops, "sync_runtime_dependencies", lambda **_kw: (True, "fixture"))
    git(serving, "branch", "ouroboros-stable")
    (serving / "ouroboros" / "mod_a.py").write_text("GEN = 'GEN_LOCAL'\n")  # unstaged
    (serving / "ouroboros" / "mod_b.py").write_text("GEN = 'GEN_STAGED'\n")
    git(serving, "add", "ouroboros/mod_b.py")
    (serving / "notes.txt").write_text("untracked owner notes\n")
    return serving


def _composition(repo):
    files = {path.relative_to(repo).as_posix(): path.read_bytes() for path in sorted(repo.rglob("*"))
             if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts}
    return git(repo, "rev-parse", "HEAD"), git(repo, "ls-files", "-s"), files


def _failing_import(monkeypatch):
    monkeypatch.setattr(git_ops, "import_test", lambda: {"ok": False, "returncode": 1, "stderr": "SyntaxError: boom"})


def test_owner_restart_and_cold_boot_keep_the_dirty_composition(repo, monkeypatch):
    monkeypatch.setattr(git_ops, "import_test", lambda: {"ok": True})
    before = _composition(repo)
    for reason in ("owner_restart", "bootstrap"):
        assert git_ops_reset.safe_restart(reason, unsynced_policy="keep") == (True, "OK: ouroboros")
        assert _composition(repo) == before
    assert not any((git_ops.DRIVE_ROOT / "archive" / "rescue").glob("*"))  # keep makes no rescue copy either


def test_unverified_breaking_edit_is_kept_and_stops_in_recovery(repo, monkeypatch):
    _failing_import(monkeypatch)
    before = _composition(repo)

    ok, message = git_ops_reset.safe_restart("bootstrap", unsynced_policy="keep")

    assert ok is False and "last commit was not started" in message
    assert "the cause is unknown; its 3 local change(s) stay in place and were copied to" in message
    assert _composition(repo) == before  # nothing reset, nothing cleaned, no fallback checkout
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "ouroboros"
    failure = update_prepare.projection()["start_failure"]  # what the Updates panel reads
    assert failure["composition"] == "unverified" and failure["detail"].endswith("SyntaxError: boom")
    rescue = pathlib.Path(failure["rescue"])
    assert rescue.is_dir() and (rescue / "changes.diff").exists()


def test_a_healthy_composition_that_stops_importing_names_its_environment(repo, monkeypatch):
    git_ops_reset.record_healthy_composition()
    _failing_import(monkeypatch)
    before = _composition(repo)

    ok, message = git_ops_reset.safe_restart("owner_restart", unsynced_policy="keep")

    assert ok is False and "environment, data or something outside the checkout" in message
    assert _composition(repo) == before
    assert update_prepare.projection()["start_failure"]["composition"] == "healthy"
    monkeypatch.setattr(git_ops, "import_test", lambda: {"ok": True})
    assert git_ops_reset.safe_restart("owner_restart", unsynced_policy="keep")[0] is True
    assert "start_failure" not in update_prepare.projection()  # the next good start clears it
    assert update_prepare.retained_fact("healthy_composition", git_ops.DRIVE_ROOT)["composition"]


def test_a_clean_checkout_with_no_ready_record_is_unverified_without_blaming_local_changes(repo, monkeypatch):
    """No healthy receipt yet (a first start) and nothing edited: the failure is reported as unexplained, never as
    local changes that break it; the import error stays readable and the checkout stays for Restore."""
    git(repo, "reset", "-q", "--hard")
    (repo / "notes.txt").unlink()
    head = git(repo, "rev-parse", "HEAD")
    error = "ModuleNotFoundError: No module named 'absent_dependency'"
    monkeypatch.setattr(git_ops, "import_test", lambda: {"ok": False, "returncode": 1, "stderr": error})

    ok, message = git_ops_reset.safe_restart("bootstrap", unsynced_policy="keep")

    assert ok is False and "no ready start of this composition is recorded, so the cause is unknown." in message
    assert "local change" not in message and "last commit was not started" in message
    failure = update_prepare.projection()["start_failure"]
    assert (failure["composition"], failure["paths"], failure["detail"]) == ("unverified", [], error)
    assert git(repo, "rev-parse", "HEAD") == head and git(repo, "status", "--porcelain") == ""


UPDATE = {"kind": "update", "candidate_id": "update_c_1"}


@pytest.mark.parametrize(("settled", "records"), [
    ({}, True),  # nothing inherited: an ordinary ready start
    ({**UPDATE, "outcome": "adopted"}, True),
    ({**UPDATE, "outcome": "returned", "why": "import_probe_failed"}, True),  # the previous tree, back and ready
    ({**UPDATE, "outcome": "not_applied", "note": "n"}, True),  # nothing was switched
    ({"outcome": "superseded", "note": "n"}, True),  # the checkout moved off the candidate: judged as any ready start
    ({"outcome": "unconfirmed", "note": "n"}, True),  # an adoption ready on its attested tree, its native host unproven
    ({**UPDATE, "outcome": "unconfirmed", "why": "loaded_tree_unattested"}, False),
    ({"outcome": "unconfirmed", "why": "loaded_tree_unattested", "note": "n"}, False),  # the same for an adoption
    ({**UPDATE, "outcome": "returning"}, False),
    ({**UPDATE, "outcome": "start_failed"}, False),
])
def test_a_ready_start_records_its_composition_healthy_unless_its_switched_tree_went_unattested(
        monkeypatch, settled, records):
    """Which boot settlement writes the composition ``keep`` later trusts (``body_adoption.settle_on_boot``); the
    real path, from an unattested start to the one that attests, is in test_update_activation."""
    from ouroboros import body_adoption
    from supervisor import message_bus, update_prepare

    calls = []
    monkeypatch.setattr(body_adoption, "finalize_on_boot", lambda *_a, **_kw: dict(settled))
    monkeypatch.setattr(git_ops_reset, "record_healthy_composition", lambda: calls.append("healthy"))
    monkeypatch.setattr(update_prepare, "record_activation", lambda *_a: None)
    monkeypatch.setattr(message_bus, "send_with_budget", lambda *_a, **_kw: None)
    monkeypatch.setattr("supervisor.state.load_state", lambda: {})
    body_adoption.settle_on_boot("data", "repo", supervisor_ready=False)
    # A start that is not ready records nothing, except an Update adopted on a providerless generation's readiness:
    # ``finalize_on_boot`` adopts only a ready start, and its switch probe imported the tree on this interpreter.
    adopted_update = settled.get("kind") == "update" and settled.get("outcome") == "adopted"
    assert calls == (["healthy"] if adopted_update else [])
    calls.clear()
    body_adoption.settle_on_boot("data", "repo", supervisor_ready=True)
    assert calls == (["healthy"] if records else [])


def test_a_failed_import_never_starts_an_older_branch_on_the_current_data(repo, monkeypatch):
    """The automatic fallback to ``ouroboros-stable`` is retired: nothing proves an older branch runs on data this
    generation may have migrated, so a failed import keeps the dev checkout exactly and records the start failure."""
    git(repo, "reset", "-q", "--hard")  # a clean tree on the dev branch (this fixture's own repository)
    (repo / "notes.txt").unlink()
    head = git(repo, "rev-parse", "HEAD")
    _failing_import(monkeypatch)

    ok, message = git_ops_reset.safe_restart("agent_restart_request", unsynced_policy="rescue_and_block")

    assert ok is False and "the checkout was kept" in message and "no older branch" in message
    assert git(repo, "rev-parse", "HEAD") == head and git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "ouroboros"


def test_an_update_adopted_on_providerless_readiness_records_its_composition_healthy(tmp_path, monkeypatch):
    """A generation without a provider runs no supervisor and no supervisor state exists (``providerless_ready``
    observes its recovery's readiness): the adopted Update's composition is still the healthy one ``keep`` trusts,
    its switch probe having imported the tree here, and it lives with the Update's retained facts."""
    import sys

    from ouroboros import body_adoption, body_switch

    scene = Scene(tmp_path, monkeypatch)
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work(commit=False)
    row = scene.prepare()
    scene.activate(row)
    assert scene.boot().returncode == 0 and scene.handoff()["phase"] == "switched"
    monkeypatch.setattr(sys, "_ouroboros_body_generation", None, raising=False)
    body_switch._attest_loaded(str(scene.serving), scene.handoff())  # as this start's own entry did

    settled = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=False,
                                           providerless_ready=lambda: True)

    assert settled["outcome"] == "adopted" and not (scene.data / "state" / "state.json").exists()
    on_disk = update_prepare.fingerprint(update_prepare.snapshot(scene.serving, "ouroboros"))
    assert update_prepare.retained_fact("healthy_composition", scene.data)["composition"] == on_disk
    assert update_prepare.projection()["last_activation"]["outcome"] == "adopted"
    assert update_prepare.find(row["candidate_id"])["state"] == "activated"
