"""A completed Panic cannot acquire authority over work started after manual return.

Mocks exercise the actual membership/identity/signal and journal consumers without
signalling host processes; one POSIX case uses only this test's own children.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ouroboros import platform_layer as pl
from ouroboros import process_scope as ps

pytestmark = [pytest.mark.skipif(os.name == "nt", reason="POSIX census contract")]
REPO = Path(__file__).resolve().parents[1]
BIRTH = "Sat Oct 10 20:10:23 2026"
OLD, NEW, TASK, ENGINE = 120001, 120002, 120003, 120004


@pytest.fixture
def world(tmp_path, monkeypatch):
    marker = ps.installation_marker(tmp_path)
    rows = {pid: ps._Row(pid, 1, pid, BIRTH, ps.birth_epoch(BIRTH), markers={marker})
            for pid in (OLD, NEW, TASK, ENGINE)}
    signalled = []
    monkeypatch.setattr(ps, "census", lambda markers=(): dict(rows))
    monkeypatch.setattr(ps, "read_identities", lambda pids: {
        int(pid): ps._Identity(ps.IDENTITY_ALIVE, rows[pid].birth, rows[pid].pgid)
        for pid in pids if pid in rows})
    monkeypatch.setattr(ps, "_await_exit", lambda targets, deadline: ({}, {}))

    def signal(pid, *, graceful):
        signalled.append(pid)
        rows.pop(pid, None)
        return pl.SIGNAL_SENT

    monkeypatch.setattr(pl, "deliver_signal", signal)
    pending = ps.panic_sweep_pending_path(tmp_path)
    record = {"requested_at": "before manual start", "excluded": [],
              "known_members": [{"pid": OLD, "birth": BIRTH}]}
    pending.write_text(json.dumps(record), encoding="utf-8")
    return marker, rows, signalled, pending, record


def test_both_startup_entries_retry_saved_identity_without_enrolling_new_sources(world, tmp_path, monkeypatch):
    marker, rows, signalled, pending, record = world
    task_marker = ps.mint_marker(ps.TASK_SCOPE_PREFIX)
    engine_marker = ps.mint_marker(ps.RUN_SCOPE_PREFIX)
    monkeypatch.setattr(ps, "task_scope_records", lambda root: [{
        "task_id": "new", "marker": task_marker, "groups": [
            {"pgid": TASK, "leader_pid": TASK, "leader_birth": BIRTH}]}])
    monkeypatch.setattr(ps, "engine_scope_index", lambda root: ([{
        "run_id": "new", "marker": engine_marker, "scope_start": BIRTH,
        "captured": [{"pid": ENGINE, "pgid": ENGINE, "birth": BIRTH}]}], {"status": "read"}))
    (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    monkeypatch.setenv(marker, "0")  # restore the caller environment at teardown
    monkeypatch.setattr(ps, "_PLANTED", {})
    real_update = ps.update_json_locked

    def failed_keep(path, *args, **kwargs):
        if path == pending:
            raise OSError("pending directory cannot accept writes")
        return real_update(path, *args, **kwargs)

    monkeypatch.setattr(ps, "update_json_locked", failed_keep)
    monkeypatch.setattr(ps, "append_jsonl", lambda *args, **kwargs: False)
    # Both a main() call and its later lifespan call, even when no previous update landed.
    for _ in range(2):
        rows[OLD] = ps._Row(OLD, 1, OLD, BIRTH, ps.birth_epoch(BIRTH), markers=set())
        assert ps.initialize_installation_scope(tmp_path) == marker
        assert json.loads(pending.read_text()) == record
    assert signalled == [OLD, OLD]
    assert set(rows) == {NEW, TASK, ENGINE}, "marker, task group and engine membership confer no old-Panic authority"


def test_known_only_does_not_even_read_current_membership_sources(world, tmp_path, monkeypatch):
    _, _, signalled, _, _ = world

    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary completion read a fresh membership source")

    for name in ("installation_marker", "task_scope_records", "engine_scope_index"):
        monkeypatch.setattr(ps, name, forbidden)
    result = ps.complete_pending_panic_sweep(tmp_path)
    assert signalled == [OLD] and result["scope"] == "known_members"
    assert result["engine_index"]["status"] == "not_read"
    assert result["settled"] is True and result["pending_retired"] is True
    assert "not a full cleanup" in result["residual"]
    row = json.loads((tmp_path / "logs" / "supervisor.jsonl").read_text().splitlines()[-1])
    assert row["scope"] == "known_members" and row["engine_index"]["status"] == "not_read"


def test_pid_reuse_never_becomes_membership_even_when_installation_marker_matches(world, tmp_path):
    _, rows, signalled, _, _ = world
    rows[OLD].birth = "Sun Oct 11 20:10:23 2026"
    result = ps.complete_pending_panic_sweep(tmp_path)
    assert signalled == [] and result["captured_gone"] == [OLD]
    assert result["pending_retired"] is True


@pytest.mark.parametrize("body", ["{broken", "[]", '{"known_members": {}}',
                                  '{"excluded": "bad"}', '{"known_members": [null]}',
                                  '{"known_members": [{"pid": 120001, "birth": {}}]}'])
def test_unreadable_or_malformed_pending_stays_byte_exact(world, tmp_path, body):
    _, _, signalled, pending, _ = world
    pending.write_text(body, encoding="utf-8")
    before = pending.read_bytes()
    result = ps.complete_pending_panic_sweep(tmp_path)
    assert signalled == [] and pending.read_bytes() == before
    assert result["pending_record"] == "unreadable"
    assert result["settled"] is False and result["pending_retired"] is False
    row = json.loads((tmp_path / "logs" / "supervisor.jsonl").read_text().splitlines()[-1])
    assert row["pending_record"] == "unreadable" and row["settled"] is False


def test_mode_zero_pending_is_not_empty(world, tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root reads mode-zero files")
    _, _, signalled, pending, _ = world
    before = pending.read_bytes()
    pending.chmod(0)
    try:
        result = ps.complete_pending_panic_sweep(tmp_path)
        assert signalled == [] and result["pending_retired"] is False
        assert result["pending_record"] == "unreadable"
    finally:
        pending.chmod(0o600)
    assert pending.read_bytes() == before


def test_explicit_launcher_pass_can_still_reach_members_but_does_not_erase_a_corrupt_record(world, tmp_path):
    _, _, signalled, pending, _ = world
    pending.write_text("{broken", encoding="utf-8")
    result = ps.complete_pending_panic_sweep(tmp_path, installation=True)
    assert set(signalled) == {OLD, NEW, TASK, ENGINE}
    assert result["scope"] == "installation" and result["pending_retired"] is False
    assert result["pending_record"] == "unreadable" and pending.read_text() == "{broken"


def test_failed_retirement_remains_pending_until_a_real_unlink(world, tmp_path, monkeypatch):
    _, rows, _, pending, _ = world
    rows.clear()  # no process remains, independent of the unlink fault
    monkeypatch.setattr(ps, "read_identities", lambda pids: {pid: ps._Identity("absent") for pid in pids})
    real_unlink = Path.unlink

    def refuse(path, *args, **kwargs):
        if path == pending:
            raise PermissionError("pending record is not deletable")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse)
    result = ps.complete_pending_panic_sweep(tmp_path)
    assert result["settled"] is True and result["recorded"] is True
    assert result["pending_retired"] is False and pending.exists()
    assert json.loads(pending.read_text())["last_outcome"]["scope"] == "known_members"
    monkeypatch.setattr(Path, "unlink", real_unlink)
    assert ps.complete_pending_panic_sweep(tmp_path)["pending_retired"] is True
    assert not pending.exists()


def test_new_panic_is_still_broad_and_replaces_corrupt_pending(world, tmp_path):
    _, _, signalled, pending, _ = world
    pending.write_text("{broken", encoding="utf-8")
    result = ps.panic_scope_census(tmp_path)
    assert set(signalled) == {OLD, NEW, TASK, ENGINE}
    assert result["scope"] == "installation" and result["pending_retired"] is True
    assert not pending.exists()


def test_only_the_launchers_observed_panic_exit_requests_installation_completion():
    launcher = ast.parse((REPO / "launcher.py").read_text())
    calls = [node for node in ast.walk(launcher) if isinstance(node, ast.Call)
             and any(kw.arg == "target" and isinstance(kw.value, ast.Name)
                     and kw.value.id == "complete_pending_panic_sweep" for kw in node.keywords)]
    assert len(calls) == 1
    kwargs = next(kw.value for kw in calls[0].keywords if kw.arg == "kwargs")
    assert any(isinstance(k, ast.Constant) and k.value == "installation"
               and isinstance(v, ast.Constant) and v.value is True for k, v in zip(kwargs.keys, kwargs.values))
    panic_branch = next(node for node in ast.walk(launcher) if isinstance(node, ast.If)
                        and ast.unparse(node.test) == "exit_code == PANIC_EXIT_CODE")
    assert calls[0] in list(ast.walk(panic_branch))
    for path in (REPO / "server.py", REPO / "ouroboros" / "process_scope.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "complete_pending_panic_sweep":
                assert not any(kw.arg == "installation" for kw in node.keywords), str(path)


def test_real_known_old_process_ends_but_new_marker_process_survives_manual_start(tmp_path, monkeypatch):
    marker = ps.installation_marker(tmp_path)
    old = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           start_new_session=True)
    new = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                           env={**os.environ, marker: "1"}, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        birth = ps.birth_text(old.pid)
        assert birth, "the test must actually capture the old process identity"
        ps.panic_sweep_pending_path(tmp_path).write_text(json.dumps({
            "excluded": [], "known_members": [{"pid": old.pid, "birth": birth}]}), encoding="utf-8")
        monkeypatch.setenv(marker, "0")
        monkeypatch.setattr(ps, "_PLANTED", {})
        assert ps.initialize_installation_scope(tmp_path) == marker
        assert old.wait(timeout=5) != 0
        assert new.poll() is None
        assert ps.initialize_installation_scope(tmp_path) == marker and new.poll() is None
        assert not ps.panic_sweep_pending_path(tmp_path).exists()
    finally:
        for child in (old, new):
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
