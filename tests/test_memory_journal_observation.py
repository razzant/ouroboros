"""Boot measures the full-text memory journals once per generation and never touches them.

``server_maintenance._startup_prune_sweeps`` emits one ``memory_journal_observation`` event with
each observed journal's byte size, read by ``lstat`` alone. Neither repeated boots nor the
maintenance child's retirement and temp sweeps digest, rewrite, delete or create journal bytes,
however old the rows and the files are; an odd entry or an unwritable event never stops boot.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import pytest

from ouroboros import server_maintenance as sm
from ouroboros.maintenance import run_job
from ouroboros.startup_migrations import prepare_startup_state

_OBSERVED = ("memory/identity_journal.jsonl", "memory/world_journal.jsonl",
             "memory/knowledge_history.jsonl", "memory/knowledge/patterns_history.jsonl")
# Journals the event does not measure; boot and maintenance leave them whole all the same.
_UNOBSERVED = ("memory/scratchpad_journal.jsonl", "projects/example/knowledge_history.jsonl")
_OLD = time.time() - 400 * 86400  # past any GC retention (at most 365 days)


def _history(relative: str) -> bytes:
    """Rows as earlier releases left them: Unicode old/new snapshots under a stale legacy hash,
    a digest-only row, a torn line and a final record without its newline."""
    rows = [{"ts": "2020-01-01T00:00:00+00:00", "old_content": f"old {relative}\nUnicode Я ✓",
             "new_content": f"new {relative}\nUnicode Ё 🙂", "old_sha256": "legacy-mismatch"},
            {"ts": "2020-01-02T00:00:00+00:00", "content_digested": True, "old_sha256": "0" * 64}]
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return (text + "{legacy broken row\n" + '{"ts": "2020-01-03T00:00:00+00:00"').encode("utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    data = tmp_path / "data"
    prepare_startup_state(data)
    monkeypatch.setattr(sm, "DATA_DIR", data)
    monkeypatch.setattr(sm, "_STARTUP_RECOVERY_GAPS", [False])
    return data


def _seed(root: pathlib.Path) -> None:
    for relative in (*_OBSERVED, *_UNOBSERVED):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_history(relative))
        os.utime(path, (_OLD, _OLD))


def _snapshot(root: pathlib.Path) -> dict:
    """Every file under the journal planes with its exact bytes and modification time."""
    return {path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for plane in ("memory", "projects") for path in sorted((root / plane).rglob("*"))
            if path.is_file()}


def _observations(root: pathlib.Path) -> list:
    events = root / "logs" / "events.jsonl"
    lines = events.read_text(encoding="utf-8").splitlines() if events.exists() else []
    return [row for row in map(json.loads, lines) if row.get("type") == "memory_journal_observation"]


def _assert_observation(row: dict, journal_bytes: dict, errors: list) -> None:
    """The event's whole envelope: ``ts``, ``type`` and a report of exactly these two fields."""
    assert set(row) == {"ts", "type", "report"}
    assert row["report"] == {"errors": errors, "journal_bytes": journal_bytes}
    assert list(row["report"]["journal_bytes"]) == list(_OBSERVED)


def test_old_journal_bytes_survive_repeated_boots_and_the_maintenance_child(root):
    _seed(root)
    before = _snapshot(root)
    orphan = root / "memory" / ".identity_journal.jsonl.tmp.1.2.abc"
    orphan.write_bytes(b"torn write")
    os.utime(orphan, (_OLD, _OLD))

    sm._startup_prune_sweeps()
    assert run_job(root, "runtime_retirement")["state"] == "done"
    assert run_job(root, "temp_sweep")["items"] == 1, "the child's temp sweep walked memory/"
    sm._startup_prune_sweeps(preserve_task_sources=True, recovery_report={"errors": ["unknown"]})

    assert not orphan.exists()
    assert _snapshot(root) == before, "every journal keeps its bytes and mtime; nothing new appears"
    rows = _observations(root)
    assert len(rows) == 2, "one observation per boot, none from the maintenance child"
    for row in rows:
        _assert_observation(row, {relative: len(_history(relative)) for relative in _OBSERVED}, [])


def test_boot_without_journals_reports_each_missing_and_creates_none(root):
    sm._startup_prune_sweeps()
    sm._startup_prune_sweeps()

    rows = _observations(root)
    assert len(rows) == 2, "missing journals are still observed, once per boot"
    for row in rows:
        _assert_observation(row, dict.fromkeys(_OBSERVED), [])
    assert not (root / "memory").exists() and not (root / "projects").exists()


def test_measuring_an_absent_root_creates_nothing(tmp_path):
    absent = tmp_path / "absent"
    assert sm._memory_journal_sizes(absent) == {"errors": [], "journal_bytes": dict.fromkeys(_OBSERVED)}
    assert not absent.exists()


@pytest.mark.parametrize("entry", ["directory", "symlink", "unreadable"])
def test_an_odd_journal_entry_is_reported_and_boot_goes_on(root, tmp_path, monkeypatch, entry):
    _seed(root)
    journal = root / "memory" / "world_journal.jsonl"
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"not a journal\n")
    error = "not_regular"
    if entry == "directory":
        journal.unlink()
        journal.mkdir()
    elif entry == "symlink":
        journal.unlink()
        try:
            journal.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation unavailable")
    before = _snapshot(root)

    with monkeypatch.context() as patch:
        if entry == "unreadable":
            error, real_lstat = "PermissionError", pathlib.Path.lstat

            def lstat(path):
                if path == journal:
                    raise PermissionError(13, "Permission denied", str(path))
                return real_lstat(path)
            patch.setattr(pathlib.Path, "lstat", lstat)
        sm._startup_prune_sweeps()

    [row] = _observations(root)
    sizes = {relative: len(_history(relative)) for relative in _OBSERVED}
    _assert_observation(row, {**sizes, "memory/world_journal.jsonl": None},
                        [f"memory/world_journal.jsonl: {error}"])
    assert _snapshot(root) == before and outside.read_bytes() == b"not a journal\n"


def test_an_unwritable_event_leaves_boot_and_journals_intact(root, monkeypatch):
    import supervisor.state as state

    _seed(root)
    before = _snapshot(root)
    real_append = state.append_jsonl

    def append(path, row, **kwargs):
        if row.get("type") == "memory_journal_observation":
            raise OSError(28, "No space left on device")
        return real_append(path, row, **kwargs)
    monkeypatch.setattr(state, "append_jsonl", append)

    sm._startup_prune_sweeps()  # returns: the server's lifespan goes on

    assert _observations(root) == []
    assert _snapshot(root) == before
