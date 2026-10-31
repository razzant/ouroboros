"""Hermetic data-root CLASS pins (issue #455).

Three structural guarantees, pinned independently of any single writer:

1. The shared jsonl writer helper fails CLOSED under pytest when a path
   resolves into the live data tree — the guard half that was missing while
   ``state.atomic_write_text`` already had it, which is why the supervisor.jsonl
   leak landed silently.
2. The same for the shared ATOMIC-overwrite helper, which is the other whole
   half of the durable-write surface: every full-file writer in the tree
   (``write_bytes_atomic``, ``write_text_atomic``, ``atomic_write_json``,
   ``supervisor.state.atomic_write_text``) replaces through it, so guarding it
   guards them all rather than one writer at a time (RES-14b).
3. Batch recipe: a real child pytest running supervisor writers under a throwaway
   HOME plus full OUROBOROS_* isolation leaves the live-shaped data root untouched
   (pre/post inventory delta must be empty) — the AGENTS hermeticity etalon
   (``find <live-root> -newermt <start>`` empty) as a regression test.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

from ouroboros.utils import append_jsonl, atomic_write_json, write_bytes_atomic, write_text_atomic

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _run_hermetic_child_pytest(args, env):
    from ouroboros.platform_layer import kill_process_tree
    from ouroboros.process_containment import ProcessContainer

    container = ProcessContainer()
    proc = None
    try:
        proc = container.spawn(
            [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *args],
            cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        # Bound a stuck child; this is a custody budget, never a speed assertion.
        stdout, stderr = proc.communicate(timeout=900)
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    finally:
        try:
            if proc is not None and proc.poll() is None:
                kill_process_tree(proc)
        finally:
            try:
                error = container.reap()  # Includes descendants after an already-exited root.
                if proc is not None:
                    # Bound kernel reaping after the kill, not application behavior.
                    proc.wait(timeout=10)
                assert not error, error
            finally:
                container.close()
                if proc is not None:
                    for stream in (proc.stdout, proc.stderr):
                        if stream is not None:
                            stream.close()


_HERMETIC_WRITER_SUITE = '''
import json
import os
import pathlib

import pytest


def test_supervisor_roots_and_real_writers_are_isolated(monkeypatch):
    from ouroboros import config
    from ouroboros.utils import append_jsonl, atomic_write_json, write_bytes_atomic, write_text_atomic
    from supervisor import git_ops, queue, state, update_merge, workers

    root = pathlib.Path(os.environ["OUROBOROS_DATA_DIR"]).resolve()
    live = pathlib.Path(LIVE_SHAPED).resolve()
    assert root != live and live not in root.parents
    assert config.DATA_DIR.resolve() == root
    assert config.SETTINGS_PATH.resolve() == root / "settings.json"
    for owner in (git_ops, queue, state, workers):
        assert owner.DRIVE_ROOT.resolve() == root, owner.__name__
    assert state.STATE_PATH == root / "state" / "state.json"
    assert queue.QUEUE_SNAPSHOT_PATH == root / "state" / "queue_snapshot.json"
    assert git_ops.REPO_DIR == workers.REPO_DIR == root / "repo"

    # The update-merge logging path that originally leaked, with no fake writer.
    assert update_merge._log_supervisor({"type": "hermetic_writer_probe", "probe": True})
    rows = [json.loads(line) for line in
            (root / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["probe"] for row in rows if row.get("type") == "hermetic_writer_probe"] == [True]
    direct = root / "logs" / "direct.jsonl"
    assert append_jsonl(direct, {"probe": True})
    assert json.loads(direct.read_text(encoding="utf-8")) == {"probe": True}

    writers = (
        ("bytes", lambda p: write_bytes_atomic(p, b'{"probe": true}')),
        ("text", lambda p: write_text_atomic(p, '{"probe": true}')),
        ("json", lambda p: atomic_write_json(p, {"probe": True})),
        ("state", lambda p: state.atomic_write_text(p, '{"probe": true}')),
    )
    for label, write in writers:
        target = root / "writer-probe" / (label + ".json")
        write(target)
        assert json.loads(target.read_text(encoding="utf-8")) == {"probe": True}, label

    # Both shared writer seams must also refuse the parent's synthetic live root.
    monkeypatch.setenv("OUROBOROS_TEST_LIVE_DATA_ROOT", str(live))
    monkeypatch.delenv("OUROBOROS_ALLOW_LIVE_DATA_TESTS", raising=False)
    with pytest.raises(RuntimeError, match="PYTEST_LIVE_DATA_WRITE_BLOCKED"):
        append_jsonl(live / "logs" / "probe.jsonl", {"probe": True})
    with pytest.raises(RuntimeError, match="PYTEST_LIVE_DATA_WRITE_BLOCKED"):
        atomic_write_json(live / "state" / "probe.json", {"probe": True})
'''


def test_append_jsonl_fails_closed_on_live_root_write(monkeypatch, tmp_path):
    fake_live = tmp_path / "fake-live"
    monkeypatch.setenv("OUROBOROS_PYTEST_ACTIVE", "1")
    monkeypatch.delenv("OUROBOROS_ALLOW_LIVE_DATA_TESTS", raising=False)
    monkeypatch.setenv("OUROBOROS_TEST_LIVE_DATA_ROOT", str(fake_live))
    with pytest.raises(RuntimeError, match="PYTEST_LIVE_DATA_WRITE_BLOCKED"):
        append_jsonl(fake_live / "logs" / "supervisor.jsonl", {"type": "probe"})
    assert not (fake_live / "logs" / "supervisor.jsonl").exists()
    # A non-live path keeps working.
    assert append_jsonl(tmp_path / "ok" / "log.jsonl", {"type": "probe"}) is True


def test_every_atomic_writer_fails_closed_on_live_root_write(monkeypatch, tmp_path):
    """The class fix (RES-14b): the guard belongs on ``_atomic_overwrite``, the
    one seam every full-file writer replaces through — not on each writer, where
    the next one added would silently arrive unguarded."""
    fake_live = tmp_path / "fake-live"
    monkeypatch.setenv("OUROBOROS_PYTEST_ACTIVE", "1")
    monkeypatch.delenv("OUROBOROS_ALLOW_LIVE_DATA_TESTS", raising=False)
    monkeypatch.setenv("OUROBOROS_TEST_LIVE_DATA_ROOT", str(fake_live))
    from supervisor import state as supervisor_state

    writers = (
        ("write_bytes_atomic", lambda p: write_bytes_atomic(p, b"probe")),
        ("write_text_atomic", lambda p: write_text_atomic(p, "probe")),
        ("atomic_write_json", lambda p: atomic_write_json(p, {"probe": True})),
        ("state.atomic_write_text", lambda p: supervisor_state.atomic_write_text(p, "probe")),
    )
    for label, write in writers:
        target = fake_live / "state" / f"{label}.bin"
        with pytest.raises(RuntimeError, match="PYTEST_LIVE_DATA_WRITE_BLOCKED"):
            write(target)
        assert not target.exists(), label
    # A non-live path keeps working, and the temp file is cleaned up either way.
    ok = tmp_path / "ok" / "state.json"
    atomic_write_json(ok, {"probe": True})
    assert json.loads(ok.read_text(encoding="utf-8")) == {"probe": True}
    assert sorted(p.name for p in ok.parent.iterdir()) == ["state.json"]


@pytest.mark.serial
def test_update_merge_suites_leave_live_shaped_root_untouched(tmp_path):
    """Keep the historical node's child-startup/writer isolation witness.

    Update/stash scenarios run in their own modules; replaying them here only
    repeats their git fixtures. This child reaches their actual logging writer
    and both durable-write seams after the real conftest isolation boundary.
    """
    fake_home = tmp_path / "home"
    iso = tmp_path / "iso"
    fake_home.mkdir()
    env = {**os.environ,
           "HOME": str(fake_home),
           "USERPROFILE": str(fake_home),
           "OUROBOROS_APP_ROOT": str(iso),
           "OUROBOROS_REPO_DIR": str(iso / "repo"),
           "OUROBOROS_DATA_DIR": str(iso / "data"),
           "OUROBOROS_SETTINGS_PATH": str(iso / "data" / "settings.json")}
    for stale in ("OUROBOROS_PYTEST_ACTIVE", "OUROBOROS_TEST_LIVE_DATA_ROOT",
                  "OUROBOROS_BENCH_RUNS_ROOT", "PYTEST_CURRENT_TEST", "PYTEST_XDIST_WORKER"):
        env.pop(stale, None)

    live_shaped = fake_home / "Ouroboros" / "data"
    pre_inventory = sorted(str(p) for p in live_shaped.rglob("*")) if live_shaped.exists() else []
    assert pre_inventory == []

    suite = tmp_path / "child-suite"
    suite.mkdir()
    child = suite / "test_root_writers.py"
    child.write_text(f"LIVE_SHAPED = {str(live_shaped)!r}\n" + _HERMETIC_WRITER_SUITE,
                     encoding="utf-8")
    proc = _run_hermetic_child_pytest(
        ["-p", "tests.conftest", "-o", "addopts=", "-n", "0",
         "--rootdir", str(suite), "--confcutdir", str(suite), str(child)], env,
    )
    assert proc.returncode == 0, (proc.stdout or "")[-4000:] + (proc.stderr or "")[-2000:]

    post_inventory = sorted(str(p) for p in live_shaped.rglob("*")) if live_shaped.exists() else []
    assert post_inventory == [], (
        f"supervisor writers leaked into the live-shaped data root: {post_inventory}"
    )
