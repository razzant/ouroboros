"""Requirements changes, targets and failed installs determine dependency sync."""
import json
import os
import shutil
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


def test_a_private_environment_that_does_not_serve_this_checkout_is_never_replaced_by_the_interpreter(
        tmp_path, monkeypatch):
    """A bare interpreter whose private environment does not serve this checkout (its marker gone, another
    checkout's, or the whole directory gone after an Update selected it) refuses the boot sync and the switch's
    install, never installing into its own packages; only one that never selected one installs where it always did."""
    from ouroboros import runtime_venv

    (tmp_path / "requirements-runtime.lock").write_text("example==1\n", encoding="utf-8")
    data, bare = tmp_path / "data", str(tmp_path / "system" / "bin" / "python3")
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", data)
    monkeypatch.delenv("OUROBOROS_PYTEST_ACTIVE")
    monkeypatch.setattr(git_ops_reset.sys, "executable", bare)
    calls, done = [], Mock(wait=lambda **_: 0)
    monkeypatch.setattr(git_ops_reset.subprocess, "Popen", lambda cmd, **_: calls.append(cmd) or done)
    private = data / "state" / runtime_venv.PRIVATE_NAME
    (private / "bin").mkdir(parents=True)  # an Update selected it once; its marker is gone
    ok, message = git_ops_reset.sync_runtime_dependencies("test")
    assert not ok and "does not serve" in message and calls == []
    (private / runtime_venv.PRIVATE_MARKER).write_text(json.dumps({"repo_dir": str(tmp_path / "elsewhere")}))
    assert not git_ops_reset.sync_runtime_dependencies("test")[0] and calls == []
    with pytest.raises(runtime_venv.RuntimeVenvRefused):
        git_ops_reset.runtime_dependency_command(tmp_path)  # a switch with no launcher is refused the same way
    from supervisor import update_prepare

    update_prepare.retain_fact("private_environment", {"root": str(private)}, data)  # an adopted Update selected it
    shutil.rmtree(private)  # ...and at this cold entry, or after a return, it is gone: still never the interpreter's
    ok, message = git_ops_reset.sync_runtime_dependencies("test")
    assert not ok and "is missing" in message and calls == []
    update_prepare.retain_fact("private_environment", None, data)  # never selected: its own packages, as always
    assert git_ops_reset.sync_runtime_dependencies("test")[0] and calls[0][0] == bare


@pytest.mark.parametrize("record", ["cut", "empty", "not_utf8", "not_an_object", "sharing_violation"])
def test_an_activation_record_that_cannot_be_read_is_not_one_that_never_selected_a_private_environment(
        tmp_path, monkeypatch, record):
    """With ``python-private`` gone, only a record that says no Update selected one (none, or one read without
    ``private_environment``) lets a bare interpreter install into its own packages. One that exists but cannot be
    read may hold that fact (an atomic write is not flushed, so a crash can leave it cut or empty; an owner edits it
    by hand, as the refusal says; Windows refuses an open during a replace or a scan): the boot sync, the switch's
    install and the launcher refuse it as they refuse a selected environment that is gone."""
    from ouroboros import launcher_bootstrap, runtime_venv
    from ouroboros.utils import atomic_write_json
    from supervisor import update_prepare

    (tmp_path / "requirements-runtime.lock").write_text("example==1\n", encoding="utf-8")
    data, bare = tmp_path / "data", str(tmp_path / "system" / "bin" / "python3")
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", data)
    monkeypatch.delenv("OUROBOROS_PYTEST_ACTIVE")
    monkeypatch.setattr(git_ops_reset.sys, "executable", bare)
    calls, done, warned = [], Mock(wait=lambda **_: 0), []
    monkeypatch.setattr(git_ops_reset.subprocess, "Popen", lambda cmd, **_: calls.append(cmd) or done)
    context = Mock(repo_dir=tmp_path, data_dir=data, embedded_python=bare, deps_timeout_sec=1,
                   log=Mock(warning=lambda *a, **_: warned.append(a[-1])),
                   hidden_run=lambda cmd, **_: calls.append(cmd) or Mock(returncode=0))
    path = data / update_prepare.ACTIVATION_REL
    path.parent.mkdir(parents=True)
    atomic_write_json(path, {"private_environment": {"root": str(data / "state" / runtime_venv.PRIVATE_NAME)}})
    whole = path.read_bytes()
    path.write_bytes({"cut": whole[: len(whole) // 2], "empty": b"", "not_utf8": b"\xff" + whole,
                      "not_an_object": b"[]"}.get(record, whole))
    if record == "sharing_violation":
        def opened(file, *args, **kwargs):
            if os.path.abspath(file) == os.path.abspath(path):
                raise PermissionError(13, "The process cannot access the file", str(file))
            return open(file, *args, **kwargs)
        monkeypatch.setattr(runtime_venv, "open", opened, raising=False)
    ok, message = git_ops_reset.sync_runtime_dependencies("test")
    assert not ok and "cannot be read" in message and calls == []
    with pytest.raises(runtime_venv.RuntimeVenvRefused, match="cannot be read"):
        git_ops_reset.runtime_dependency_command(tmp_path)
    assert launcher_bootstrap.install_deps(context) is False and calls == [] and "cannot be read" in str(warned[-1])
    # Read, it tells: an adopted Update selected one, which is gone (refused), or none (its own packages, as always).
    monkeypatch.delattr(runtime_venv, "open", raising=False)
    atomic_write_json(path, json.loads(whole))
    assert launcher_bootstrap.install_deps(context) is False and calls == [] and "is missing" in str(warned[-1])
    atomic_write_json(path, {"outcome": "adopted"})
    assert git_ops_reset.runtime_dependency_command(tmp_path)[0] == bare
    assert launcher_bootstrap.install_deps(context) is True and calls[-1][0] == bare
    path.unlink()
    assert git_ops_reset.sync_runtime_dependencies("test")[0] and calls[-1][0] == bare


@pytest.mark.parametrize("writer", ["start_failure", "healthy_composition", "private_environment", "last_attempt",
                                    "forget"])
def test_an_activation_record_that_cannot_be_read_is_never_rewritten_or_deleted_as_if_empty(
        tmp_path, monkeypatch, writer):
    """Its own writers (a start failure, each ready boot's composition, an adopted private selection, the Last attempt
    and its replacement) keep a record that exists but cannot be read as it is: it may hold the app backup or the
    private selection. After any of them the boot sync, the switch's install and the launcher still refuse it; an
    absent record is still a first use, and a readable one keeps its other facts beside the one written."""
    from contextlib import suppress

    from ouroboros import launcher_bootstrap, runtime_venv
    from ouroboros.utils import atomic_write_json
    from supervisor import update_prepare

    (tmp_path / "requirements-runtime.lock").write_text("example==1\n", encoding="utf-8")
    data, bare = tmp_path / "data", str(tmp_path / "system" / "bin" / "python3")
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", data)
    monkeypatch.delenv("OUROBOROS_PYTEST_ACTIVE")
    monkeypatch.setattr(git_ops_reset.sys, "executable", bare)
    calls, done, warned = [], Mock(wait=lambda **_: 0), []
    monkeypatch.setattr(git_ops_reset.subprocess, "Popen", lambda cmd, **_: calls.append(cmd) or done)
    context = Mock(repo_dir=tmp_path, data_dir=data, embedded_python=bare, deps_timeout_sec=1,
                   log=Mock(warning=lambda *a, **_: warned.append(a[-1])),
                   hidden_run=lambda cmd, **_: calls.append(cmd) or Mock(returncode=0))
    private = {"root": str(data / "state" / runtime_venv.PRIVATE_NAME)}
    write = {"start_failure": lambda: git_ops_reset._note_start_failure({"reason": "boot", "cause": "import_failed"}),
             "healthy_composition": lambda: update_prepare.retain_fact("healthy_composition", {"head": "a"}, data),
             "private_environment": lambda: update_prepare.retain_fact("private_environment", private, data),
             "last_attempt": lambda: update_prepare._retain_activation({"candidate_id": "c"}, {}, data),
             "forget": lambda: update_prepare._forget_activation(data)}[writer]
    path = data / update_prepare.ACTIVATION_REL
    path.parent.mkdir(parents=True)
    atomic_write_json(path, {"private_environment": private, "backup_app": str(tmp_path / "Ouroboros.app.backup")})
    path.write_bytes(path.read_bytes()[:40])  # cut, as a crash after an unflushed replace leaves it
    cut = path.read_bytes()
    with suppress(OSError):  # its callers log it (git_ops_reset's keep policy, body_adoption's boot follow-ups)
        write()
    ok, message = git_ops_reset.sync_runtime_dependencies("test")
    assert not ok and "cannot be read" in message and calls == []
    with pytest.raises(runtime_venv.RuntimeVenvRefused, match="cannot be read"):
        git_ops_reset.runtime_dependency_command(tmp_path)
    assert launcher_bootstrap.install_deps(context) is False and calls == [] and "cannot be read" in str(warned[-1])
    assert path.read_bytes() == cut  # kept as it is: nothing it may hold was erased
    path.unlink()
    write()
    assert path.exists() is (writer != "forget")  # absent: a first use writes it (nothing to keep: none)
    atomic_write_json(path, {"backup_app": "kept", "start_failure": {"reason": "earlier"}})
    write()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["backup_app"] == "kept" and "start_failure" in saved


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
