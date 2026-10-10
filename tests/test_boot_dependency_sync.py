"""Requirements changes, targets and failed installs determine dependency sync."""
from unittest.mock import Mock

import pytest

from supervisor import git_ops, git_ops_reset
from ouroboros import startup_migrations as m


def test_success_fingerprint_skips_only_unchanged_install(tmp_path, monkeypatch):
    req = tmp_path / "requirements-runtime.lock"
    req.write_text("example==1\n", encoding="utf-8")
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.delenv("OUROBOROS_PYTEST_ACTIVE")
    monkeypatch.setattr("ouroboros.platform_layer.pip_install_target_args", lambda _exe: ["--target", str(tmp_path / "env")])
    calls, code = [], [0]
    monkeypatch.setattr(git_ops_reset.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or Mock(wait=lambda **kw: code[0]))
    assert git_ops_reset.sync_runtime_dependencies("test")[0]
    assert git_ops_reset.sync_runtime_dependencies("test")[1].startswith("unchanged:")
    assert len(calls) == 1
    req.write_text("example==2\n", encoding="utf-8")
    assert git_ops_reset.sync_runtime_dependencies("test")[0]
    assert len(calls) == 2
    marker = m.watermarks(tmp_path / "data")["dependencies"]
    monkeypatch.setenv("PYTHONUSERBASE", str(tmp_path / "other-target"))
    code[0] = 1
    assert not git_ops_reset.sync_runtime_dependencies("test")[0]
    assert m.watermarks(tmp_path / "data")["dependencies"] == marker
    code[0] = 0
    assert git_ops_reset.sync_runtime_dependencies("test")[0]
    assert len(calls) == 4


def test_send_path_never_invokes_readback(tmp_path, monkeypatch):
    from ouroboros import model_send_seal
    from tests.test_model_send_seal import _dispatch
    monkeypatch.setattr(model_send_seal, "verify_sealed_candidate", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("send readback")))
    assert _dispatch(tmp_path, "task")["state"] == "settled"


def test_system_scope_seals_are_readable_and_not_false_unlogged(tmp_path, monkeypatch):
    from ouroboros import model_send_seal, observability
    from ouroboros.usage_accounting import UsageScope
    from tests import test_model_send_seal as fixture
    # System calls carry the production non-task scope, outside task Pause fences.
    monkeypatch.setattr(fixture, "_scope", lambda root, task_id: UsageScope(
        drive_root=root, task_id=task_id, root_task_id=task_id, non_task_operation=True,
        category="system", source="test.model_send"))
    for task in ("system:ui_translation", "system:update_letter"):
        row = fixture._dispatch(tmp_path, task)
        manifest = observability.read_call_manifest_ref(tmp_path, row["candidate_manifest_ref"], task_id=task)
        assert manifest["model_send_seal"]["attempt_id"] == row["attempt_id"]
    report = model_send_seal.reconcile_model_send_seals(tmp_path)
    assert report["unlogged_attempts"] == report["orphan_seals"] == report["facts_written"] == 0


@pytest.mark.parametrize("changed", [False, True])
def test_checkout_wipes_bytecode_only_when_head_moved(tmp_path, monkeypatch, changed):
    import subprocess
    cache = tmp_path / "pkg/__pycache__"
    cache.mkdir(parents=True)
    (cache / "code.pyc").write_bytes(b"old bytecode")
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "git_capture", lambda *a, **kw: (0, "before", ""))
    monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {})
    monkeypatch.setattr(git_ops, "_read_update_intent", lambda: {})
    monkeypatch.setattr(git_ops, "_has_remote", lambda *a: False)
    monkeypatch.setattr(git_ops, "_run_git_resilient", lambda *a, **kw: None)
    monkeypatch.setattr(git_ops, "update_state", lambda fn: fn({}))
    monkeypatch.setattr(git_ops_reset.subprocess, "run", lambda cmd, **kw:
        subprocess.CompletedProcess(cmd, 0, stdout="after" if changed else "before", stderr=""))
    assert git_ops_reset.checkout_and_reset("ouroboros")[0]
    assert cache.exists() is not changed


def test_a_failed_import_test_forgets_the_fingerprint_so_the_next_sync_runs_pip(tmp_path, monkeypatch):
    """A drifted environment (packages removed after a sync) must be re-synced: the import test that
    finds it broken clears the fingerprint; a passing import test keeps it."""
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops_reset.sys, "frozen", False, raising=False)
    m.stamp(tmp_path / "data", dependencies="fingerprint")
    result = Mock(returncode=0, stdout="import_ok", stderr="")
    monkeypatch.setattr(git_ops_reset.subprocess, "run", lambda *a, **kw: result)
    assert git_ops_reset.import_test()["ok"]
    assert m.watermarks(tmp_path / "data")["dependencies"] == "fingerprint"
    result.returncode = 1
    assert not git_ops_reset.import_test()["ok"]
    assert m.watermarks(tmp_path / "data")["dependencies"] is None


def test_torn_watermarks_never_fail_a_checkout_or_a_sync(tmp_path, monkeypatch):
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    (tmp_path / "data" / "state").mkdir(parents=True)
    (tmp_path / "data" / "state" / "migrations.json").write_bytes(b"")  # torn
    monkeypatch.setattr(git_ops, "update_state", lambda mutator: None)
    git_ops_reset._record_checkout_facts({"current_sha": "abc"})  # logs; never raises into the checkout
    req = tmp_path / "requirements-runtime.lock"
    req.write_text("example==1\n", encoding="utf-8")
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.delenv("OUROBOROS_PYTEST_ACTIVE")
    monkeypatch.setattr("ouroboros.platform_layer.pip_install_target_args", lambda _exe: ["--target", str(tmp_path / "env")])
    calls = []
    monkeypatch.setattr(git_ops_reset.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or Mock(wait=lambda **kw: 0))
    assert git_ops_reset.sync_runtime_dependencies("test")[0] and len(calls) == 1  # not known synced: pip runs
