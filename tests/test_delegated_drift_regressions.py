"""Focused regressions kept outside the size-ratcheted isolation module: what a capture says while
the target moves (no target-drift diagnostics any more), SQLite WAL indexes and nested repositories."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from ouroboros import delegate_custody as custody
from ouroboros.subagent_worktrees import provision_execution_snapshot
from test_delegated_run_isolation import _isolated_entry, _nanny_ctx, _seed_target


_DEV_SERVER = ("import sqlite3, sys\n"
               "conn = sqlite3.connect(sys.argv[1], isolation_level=None)\n"
               "conn.execute('PRAGMA journal_mode=WAL')\n"
               "conn.execute('PRAGMA wal_autocheckpoint=0')\n"
               "conn.execute('CREATE TABLE IF NOT EXISTS t(v)')\n"
               "conn.execute(\"INSERT INTO t VALUES ('dev server row')\")\n"
               "print('ready', flush=True)\n"
               "for line in sys.stdin:\n"
               "    conn.execute(\"INSERT INTO t VALUES ('another row')\")\n"
               "    print('committed', flush=True)\n")


def _hold(database, *, killed):
    """Another process holding DATABASE open in WAL (a dev server), or killed after its first commit
    (its index and log left behind); ``commit()`` makes a live one commit again."""
    import subprocess

    holder = subprocess.Popen([sys.executable, "-I", "-c", _DEV_SERVER, str(database)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "ready"
    if killed:
        holder.kill()
        holder.wait()

    def commit():
        holder.stdin.write("commit\n")
        holder.stdin.flush()
        assert holder.stdout.readline().strip() == "committed"

    def release():
        if holder.poll() is None:
            holder.stdin.close()
            try:
                holder.wait(timeout=60)
            except subprocess.TimeoutExpired:
                holder.kill()
                holder.wait()

    return commit, release


def test_an_index_a_child_stages_is_neither_captured_nor_applied_over_the_target_s(tmp_path, monkeypatch):
    """A child that runs the app in its copy and stages everything (``git add -A``) stages that copy's
    own index: the capture leaves it out of the patch and of the paths the integration stages, so the
    apply lands the child's change and never writes over the target's live index (a new-file patch for
    it was drift the apply refused, losing the child's change with it)."""
    from ouroboros.tools.delegate import _capture_terminal_patch
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    target = _seed_target(tmp_path)
    (target / ".gitignore").write_text("*.db\n*.db-wal\n", encoding="utf-8")
    ctx = _nanny_ctx(tmp_path, target, monkeypatch)
    index = target / "app.db-shm"
    _commit, release = _hold(target / "app.db", killed=False)
    try:
        handle = provision_execution_snapshot(target_root=target, task_id="t-nanny", snapshot_id="snapStaged")
        copy = Path(handle.path)
        _hold(copy / "app.db", killed=True)  # the app the child ran there, its index left behind
        (copy / "tracked.txt").write_text("one\ntwo\nchild\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=copy, check=True, capture_output=True)
        before = index.stat()
        entry = _isolated_entry(ctx, target, handle, run_id="run-staged-index")
        capture = _capture_terminal_patch(ctx, entry)
        manifest = json.loads((custody.delegated_capture_dir(custody.custody_root(ctx), "t-nanny", "snapStaged")
                               / "workspace_patch.json").read_text(encoding="utf-8"))
        out = _integrate_delegated_patch(ctx, "run-staged-index", "apply", "the child's change")
        after = index.stat()
    finally:
        release()
    assert capture["status"] == "ready_with_changes", capture
    assert "✅ Integrated" in out, out
    assert (target / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\nchild\n"
    assert manifest["tracked_changed"] == ["tracked.txt"] and manifest["untracked_included"] == []
    # The copy's own staged index leaves the result commit by its code, so it never reaches the patch.
    assert {"path": "app.db-shm", "reason": "sqlite_wal_index"} in manifest["untracked_excluded"]
    assert manifest["tracked_excluded"] == [] and "app.db-shm" not in Path(capture["patch_artifact"]).read_text()
    assert (after.st_ino, after.st_mtime_ns, after.st_size) == (before.st_ino, before.st_mtime_ns, before.st_size)
    assert subprocess.run(["git", "ls-files", "app.db-shm"], cwd=target, check=True,
                          capture_output=True, text=True).stdout == ""
    custody._CUSTODY.clear()
