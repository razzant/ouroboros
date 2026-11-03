"""The local recovery point follows the commits this installation proved, and a fallback to it goes
through the store-switch owner (``supervisor/git_ops_reset.py``). Real Git in tmp.

A ready boot moves ``ouroboros-stable`` (``update-ref``, local, never pushed) to the commit whose import
smoke passed in that process on a clean, launcher-managed dev checkout, an update being finalized
included; never live HEAD. A start whose dev branch fails its smoke never boots a recovery point that
cannot read this installation's stores: it stops with a typed failure; one that can gets the stores
handed off and is checked out as before, the dev branch untouched.
"""
from __future__ import annotations

import subprocess

import pytest

from supervisor import git_ops, git_ops_reset
from tests._git_ops_recovery_shared import _git
from tests.test_git_bootstrap_receipt import bootstrap  # noqa: F401 - the shared real-Git fixture

STORES = ("ouroboros/store.py", "ouroboros/usage_store.py", "ouroboros/evidence_packs.py", "ouroboros/dialogue_store.py")


@pytest.fixture
def managed(bootstrap, monkeypatch):  # noqa: F811
    repo, drive, calls = bootstrap
    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.setattr(git_ops_reset, "_SMOKED", {})  # one process: one generation
    first = _git(repo, "rev-list", "--max-parents=0", "HEAD")
    _git(repo, "branch", "ouroboros-stable", first)
    return repo, drive, calls, first


def _stable(repo) -> str:
    return _git(repo, "rev-parse", "ouroboros-stable")


def test_a_ready_boot_moves_the_recovery_point_to_the_commit_its_smoke_proved(managed):
    repo, _drive, calls, first = managed
    head = _git(repo, "rev-parse", "HEAD")
    assert git_ops.safe_restart(reason="bootstrap")[0] and "smoke" in calls
    assert git_ops.advance_recovery_point(False) == "" and _stable(repo) == first, "not ready: nothing moves"
    assert git_ops.advance_recovery_point(True) == head and _stable(repo) == head, "the old boot left July's point"
    assert git_ops.advance_recovery_point(True) == "", "already there"
    (repo / "value.txt").write_text("a later commit nobody smoked\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "later")
    assert git_ops.advance_recovery_point(True) == "" and _stable(repo) == head, "never live HEAD"


def test_an_update_being_finalized_counts_though_its_receipt_stays_off(managed):
    repo, drive, _calls, _first = managed
    (repo / ".git" / "ouroboros-update-tx.json").write_text("{}", encoding="utf-8")
    assert git_ops.import_test()["ok"] and not (drive / "state" / "git_bootstrap.json").exists()
    assert git_ops.advance_recovery_point(True) == _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("case", ["dirty", "other_branch", "unmanaged", "no_smoke_this_process"])
def test_only_a_clean_managed_dev_smoke_moves_it(managed, monkeypatch, case):
    repo, _drive, _calls, first = managed
    if case == "dirty":
        (repo / "untracked.txt").write_text("owner work\n", encoding="utf-8")
    elif case == "other_branch":
        _git(repo, "checkout", "-q", "-b", "experiment")
    elif case == "unmanaged":
        monkeypatch.delenv("OUROBOROS_MANAGED_BY_LAUNCHER")
        monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {})
    if case != "no_smoke_this_process":
        assert git_ops.import_test()["ok"]
    assert git_ops.advance_recovery_point(True) == "" and _stable(repo) == first


def test_a_commit_landing_during_the_smoke_is_never_promoted(managed, monkeypatch):
    repo, _drive, _calls, first = managed
    run = subprocess.run

    def smoke_then_commit(cmd, **kwargs):
        if len(cmd) == 3 and cmd[1] == "-c" and cmd[2].startswith("import ouroboros,"):
            (repo / "value.txt").write_text("a commit nobody smoked\n", encoding="utf-8")
            _git(repo, "commit", "-qam", "landed while the smoke ran")
        return run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", smoke_then_commit)
    assert git_ops.import_test()["ok"]
    assert git_ops.advance_recovery_point(True) == "" and _stable(repo) == first, "the later, unsmoked SHA moved it"


@pytest.mark.parametrize("case", ["dev", "other_branch", "unmanaged"])
def test_a_ready_boot_whose_receipt_skips_the_smoke_still_advances_a_lagging_point(managed, monkeypatch, case):
    """An earlier boot passed this exact snapshot's smoke and died before it was ready: the receipt now skips the
    smoke, and the recovery point still follows on a clean managed dev checkout (owner 1A: after every start)."""
    repo, _drive, calls, first = managed
    if case == "other_branch":
        _git(repo, "checkout", "-q", "-b", "experiment")
    assert git_ops.import_test()["ok"] and calls == ["smoke"]
    monkeypatch.setattr(git_ops_reset, "_SMOKED", {})  # the next process
    if case == "unmanaged":
        monkeypatch.delenv("OUROBOROS_MANAGED_BY_LAUNCHER")
        monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {})
    assert git_ops.import_test()["skipped"] == "unchanged_bootstrap" and calls == ["smoke"]
    head = _git(repo, "rev-parse", "HEAD")
    expected = (head, head) if case == "dev" else ("", first)
    assert (git_ops.advance_recovery_point(True), _stable(repo)) == expected


def test_a_commit_landing_after_the_receipt_matched_is_never_promoted(managed, monkeypatch):
    repo, _drive, calls, _first = managed
    assert git_ops.import_test()["ok"] and calls == ["smoke"]  # an earlier boot: smoke passed, receipt kept
    proven = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(git_ops_reset, "_SMOKED", {})  # the next process
    receipt = git_ops_reset._bootstrap_receipt

    def receipt_then_commit():
        saved = receipt()
        (repo / "value.txt").write_text("a commit nobody smoked\n", encoding="utf-8")
        _git(repo, "commit", "-qam", "landed after the receipt matched")
        return saved

    monkeypatch.setattr(git_ops_reset, "_bootstrap_receipt", receipt_then_commit)
    assert git_ops.import_test()["skipped"] == "unchanged_bootstrap" and calls == ["smoke"]
    assert git_ops.advance_recovery_point(True) == proven and _stable(repo) == proven, "the unsmoked commit moved it"


def _failing_dev_smoke(monkeypatch, repo, dev_head):
    run = subprocess.run

    def smoke(cmd, **kwargs):
        if len(cmd) == 3 and cmd[1] == "-c" and cmd[2].startswith("import ouroboros,"):
            broken = run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip() == dev_head
            return subprocess.CompletedProcess(cmd, int(broken), "" if broken else "import_ok", "broken dev" if broken else "")
        return run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", smoke)


def test_a_recovery_point_that_cannot_read_the_stores_is_never_started(managed, monkeypatch):
    repo, _drive, _calls, first = managed
    dev_head = _git(repo, "rev-parse", "HEAD")
    _failing_dev_smoke(monkeypatch, repo, dev_head)
    ok, message = git_ops.safe_restart(reason="bootstrap")
    assert not ok and message.startswith("recovery_point_predates_stores:"), message
    assert dev_head[:12] in message and first[:12] in message and "use Restore on the Updates page" in message
    assert _git(repo, "rev-parse", "HEAD") == dev_head, "the old fallback checked July's code out over today's stores"
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "ouroboros"


def _store_aware_point(repo):
    """Move the recovery point to a commit that carries the store modules; dev HEAD and that commit."""
    dev_head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "ouroboros-stable")
    for rel in STORES:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("# a store-aware release\n", encoding="utf-8")
    _git(repo, "add", "ouroboros")
    _git(repo, "commit", "-qm", "store-aware recovery point")
    stable = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "ouroboros")
    return dev_head, stable


def test_a_recovery_point_that_reads_the_stores_is_handed_off_and_checked_out_dev_untouched(managed, monkeypatch):
    repo, _drive, _calls, _first = managed
    dev_head, stable = _store_aware_point(repo)
    handed = []
    monkeypatch.setattr("ouroboros.store_switch._hand_off_stores", lambda root: handed.append(root) or "")
    _failing_dev_smoke(monkeypatch, repo, dev_head)
    assert git_ops.safe_restart(reason="bootstrap") == (True, "OK: fell back to ouroboros-stable")
    assert handed, "the stores were handed off before the checkout"
    assert _git(repo, "rev-parse", "HEAD") == stable and _git(repo, "rev-parse", "ouroboros") == dev_head



def test_the_fallback_hands_off_this_roots_real_stores_with_the_dialogue_open(managed, monkeypatch):
    """The dialogue opens before repairs (ARCHITECTURE ch.2): its idle handle is this process's own, closed by the
    handoff, and every WAL store leaves WAL before the store-aware recovery point is checked out."""
    import sqlite3

    from ouroboros import dialogue_store, store, usage_store
    from tests.test_store import publish

    repo, drive, _calls, _first = managed
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 53, 1))
    monkeypatch.setattr(store, "_filesystem_type", lambda _path: "apfs")  # the stores open in WAL
    publish(drive)
    usage_store.migrate_from_journal(drive)
    with usage_store.hold(drive) as txn:
        txn.set_meta("retained", 1)
    dialogue_store.ensure(drive)
    dialogue_store.append(drive, "chat", {"direction": "out", "chat_id": 1, "text": "before the repairs"})
    paths = [store.store_path(drive), drive / usage_store.STORE_REL, dialogue_store.path(drive)]
    assert all(store.wal_file(path) for path in paths)
    dev_head, stable = _store_aware_point(repo)
    _failing_dev_smoke(monkeypatch, repo, dev_head)
    try:
        assert git_ops.safe_restart(reason="bootstrap") == (True, "OK: fell back to ouroboros-stable")
        assert not any(store.wal_file(path) for path in paths), "an older body opens no WAL file"
        assert _git(repo, "rev-parse", "HEAD") == stable and _git(repo, "rev-parse", "ouroboros") == dev_head
    finally:
        store.forget_handoffs()
        store.close_connections()
