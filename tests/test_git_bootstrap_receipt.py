"""A durable successful smoke skips only an unchanged, clean bootstrap."""

import json
import subprocess

import pytest

from supervisor import git_ops, git_ops_reset
from tests._git_ops_recovery_shared import _git, _history_repo


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    repo, _, _ = _history_repo(tmp_path)
    drive = tmp_path / "data"
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", drive)
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    monkeypatch.setattr(git_ops, "_has_remote", lambda *_: False)
    monkeypatch.setattr(git_ops, "update_state", lambda fn: fn({}))
    monkeypatch.setattr(git_ops_reset.sys, "frozen", False, raising=False)
    monkeypatch.delenv("OUROBOROS_DISABLE_MANAGED_UPDATES", raising=False)
    calls = []
    run = subprocess.run

    def measured_run(cmd, **kwargs):
        if len(cmd) == 3 and cmd[1] == "-c" and cmd[2].startswith("import ouroboros,"):
            calls.append("smoke")
            return subprocess.CompletedProcess(cmd, 0, "import_ok", "")
        if cmd[:3] == ["git", "reset", "--hard"]:
            calls.append("reset")
        return run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", measured_run)
    return repo, drive, calls


def test_success_receipt_skips_both_work_units_and_head_or_dependencies_invalidate(bootstrap, monkeypatch):
    repo, drive, calls = bootstrap
    assert git_ops.safe_restart(reason="bootstrap")[0]
    assert calls == ["reset", "reset", "smoke"]
    receipt = drive / "state" / "git_bootstrap.json"
    saved = receipt.read_bytes()
    assert git_ops.safe_restart(reason="bootstrap")[0]
    assert receipt.read_bytes() == saved
    assert calls == ["reset", "reset", "smoke"]

    (repo / "value.txt").write_text("new HEAD\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "changed head")
    assert git_ops.safe_restart(reason="bootstrap")[0]
    assert calls == ["reset", "reset", "smoke"] * 2
    assert json.loads(receipt.read_text(encoding="utf-8"))["head"] == _git(repo, "rev-parse", "HEAD")

    monkeypatch.setenv("PYTHONUSERBASE", str(drive / "new-dependencies"))
    assert git_ops.safe_restart(reason="bootstrap")[0]
    assert calls == ["reset", "reset", "smoke"] * 3
    assert git_ops.safe_restart(reason="bootstrap")[0]
    assert calls == ["reset", "reset", "smoke"] * 3


@pytest.mark.parametrize("change", ["tracked", "untracked", "missing", "corrupt", "rollback", "explicit_update"])
def test_dirty_tree_or_invalidated_receipt_cannot_skip(bootstrap, change):
    repo, drive, calls = bootstrap
    assert git_ops.safe_restart(reason="bootstrap")[0]
    receipt = drive / "state" / "git_bootstrap.json"
    if change in {"tracked", "untracked"}:
        path = repo / ("value.txt" if change == "tracked" else "untracked.txt")
        path.write_text("owner edit\n", encoding="utf-8")
    elif change == "missing":
        receipt.unlink()
    elif change == "corrupt":
        receipt.write_text("{", encoding="utf-8")
    elif change == "rollback":
        # Restore's common fact writer invalidates even a rollback to the same SHA.
        git_ops_reset._record_checkout_facts({"current_sha": _git(repo, "rev-parse", "HEAD")})
    reason = "ui_update_apply" if change == "explicit_update" else "bootstrap"
    assert git_ops.safe_restart(reason=reason)[0]
    assert calls == ["reset", "reset", "smoke"] * 2
    assert git_ops.safe_restart(reason="bootstrap")[0]
    assert calls == ["reset", "reset", "smoke"] * 2


@pytest.mark.parametrize("marker", [
    "ouroboros-update-tx.json", "ouroboros-update-intent.json", "ouroboros-bootstrap-pending", "MERGE_HEAD",
])
def test_transaction_markers_force_smoke_and_prevent_a_receipt(bootstrap, marker):
    repo, drive, calls = bootstrap
    assert git_ops.safe_restart(reason="bootstrap")[0]
    path = repo / ".git" / marker
    path.write_text("{", encoding="utf-8")  # unreadable is also unfinished
    assert git_ops_reset._bootstrap_snapshot() is None
    assert git_ops.import_test()["ok"]
    assert calls == ["reset", "reset", "smoke", "smoke"]
    assert not (drive / "state" / "git_bootstrap.json").exists()
    path.unlink()
    assert git_ops.safe_restart(reason="bootstrap")[0]
    assert calls == ["reset", "reset", "smoke", "smoke", "reset", "reset", "smoke"]
    assert git_ops.safe_restart(reason="bootstrap")[0]
    assert calls == ["reset", "reset", "smoke", "smoke", "reset", "reset", "smoke"]


def test_smoke_failure_or_changed_tree_never_certifies_bootstrap(bootstrap, monkeypatch):
    repo, drive, calls = bootstrap
    receipt = drive / "state" / "git_bootstrap.json"
    run = subprocess.run
    failing = [True]

    def failed_or_mutating_smoke(cmd, **kwargs):
        if len(cmd) == 3 and cmd[1] == "-c":
            if not failing[0]:
                (repo / "value.txt").write_text("edited during import\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, int(failing[0]), "", "")
        return run(cmd, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", failed_or_mutating_smoke)
        assert not git_ops.import_test()["ok"]
        assert not receipt.exists()
        failing[0] = False
        assert git_ops.import_test()["ok"]
        assert not receipt.exists()
    assert git_ops.safe_restart(reason="bootstrap")[0]
    assert receipt.exists()
    assert calls == ["reset", "reset", "smoke"]
    assert git_ops.safe_restart(reason="bootstrap")[0]
    assert calls == ["reset", "reset", "smoke"]
