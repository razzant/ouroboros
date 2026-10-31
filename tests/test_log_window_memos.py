"""A task's log window reuses what cannot have changed: the archive listing and immutable archives.

Every context build reads the task's own newest tools and events rows (``Memory.read_task_recent``).
A task with no rows yet meets no quota: the whole live file plus the newest archives, every turn.
Output and coverage stay those of a fresh read.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from ouroboros import jsonl_tail, utils
from ouroboros.memory import Memory


def _write(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join((row if isinstance(row, str) else json.dumps(row)) + "\n" for row in rows),
                    encoding="utf-8")


@pytest.fixture
def logs(tmp_path):
    """events.jsonl beside three rotated archives of another task's rows."""
    for day in (1, 2, 3):
        _write(tmp_path / "archive" / f"events_2026100{day}T000000.jsonl",
               [{"task_id": "other", "type": "llm_round", "day": day}] * 3)
    _write(tmp_path / "logs" / "events.jsonl", [{"task_id": "other", "type": "llm_round"}])
    return tmp_path


def _listings(monkeypatch, root) -> list:
    seen, real = [], os.scandir
    monkeypatch.setattr(jsonl_tail.os, "scandir", lambda path: (
        seen.append(path) if pathlib.Path(path) == root / "archive" else None) or real(path))
    return seen


def _rotate(root, stamp) -> None:
    """The rotator's move: the live file becomes the newest archive, a new live file follows."""
    os.replace(root / "logs" / "events.jsonl", root / "archive" / f"events_{stamp}.jsonl")
    _write(root / "logs" / "events.jsonl", [{"task_id": "other", "type": "llm_round"}])


def test_the_archive_listing_is_reused_until_the_log_rotates(logs, monkeypatch):
    monkeypatch.setattr(utils, "_RACY_NS", 0)  # trust the directory as written (the tick rule has its own case)
    listed, memory = _listings(monkeypatch, logs), Memory(drive_root=logs)

    first = memory.read_task_recent("events.jsonl", "fresh", 200)
    assert memory.read_task_recent("events.jsonl", "fresh", 200) == first
    assert len(listed) == 1 and first[1]["archives_available"] == 3

    archive = logs / "archive"
    before = archive.stat()
    _rotate(logs, "20261004T000000")
    os.utime(archive, ns=(before.st_atime_ns, before.st_mtime_ns))  # a coarse clock: only the live inode moved
    after = memory.read_task_recent("events.jsonl", "fresh", 200)[1]
    assert len(listed) == 2 and after["archives_available"] == 4
    assert after["archive_sources"][0].endswith("events_20261004T000000.jsonl")


def test_a_directory_changed_within_a_clock_tick_is_listed_again(logs, monkeypatch):
    listed, memory = _listings(monkeypatch, logs), Memory(drive_root=logs)

    memory.read_task_recent("events.jsonl", "fresh", 200)
    memory.read_task_recent("events.jsonl", "fresh", 200)
    assert len(listed) == 2


def test_a_callers_own_seams_keep_their_fresh_listing(logs, monkeypatch):
    monkeypatch.setattr(utils, "_RACY_NS", 0)
    listed, memory = _listings(monkeypatch, logs), Memory(drive_root=logs)

    for _ in range(2):
        memory.read_task_recent("events.jsonl", "fresh", 200, stat=pathlib.Path.stat)
    assert len(listed) == 2


def _archive_parses(monkeypatch, root) -> list:
    seen, real = [], jsonl_tail.iter_jsonl_objects
    monkeypatch.setattr(jsonl_tail, "iter_jsonl_objects", lambda path, *args, **kwargs: (
        seen.append(pathlib.Path(path).name) if pathlib.Path(path).parent == root / "archive" else None)
        or real(path, *args, **kwargs))
    return seen


def test_an_archive_without_the_tasks_rows_is_not_parsed_again_for_it(logs, monkeypatch):
    _write(logs / "archive" / "events_20261002T000000.jsonl", [{"task_id": "other"}, "{torn"])  # a recorded gap
    parsed, memory = _archive_parses(monkeypatch, logs), Memory(drive_root=logs)

    first = memory.read_task_recent("events.jsonl", "fresh", 200)
    assert len(parsed) == 3 and first[0] == [] and first[1]["gaps"] == ["malformed_jsonl"]
    assert memory.read_task_recent("events.jsonl", "fresh", 200) == first  # the same coverage, gaps replayed
    assert len(parsed) == 3
    memory.read_task_recent("events.jsonl", "other", 200)  # a task every archive holds: parsed as ever
    assert len(parsed) == 6


def test_an_archive_that_holds_the_task_or_was_replaced_is_parsed(logs, monkeypatch):
    memory = Memory(drive_root=logs)
    assert memory.read_task_recent("events.jsonl", "late", 200)[0] == []
    newest = logs / "archive" / "events_20261003T000000.jsonl"
    utils.write_text_atomic(newest, newest.read_text(encoding="utf-8") + json.dumps({"task_id": "late"}) + "\n")
    parsed = _archive_parses(monkeypatch, logs)

    rows, coverage = memory.read_task_recent("events.jsonl", "late", 200)
    assert rows == [{"task_id": "late"}] and parsed == [newest.name]  # the older two still hold none of it
    assert memory.read_task_recent("events.jsonl", "late", 200) == (rows, coverage)
    assert parsed.count(newest.name) == 2


def test_only_the_archives_a_read_can_consult_are_kept(logs, monkeypatch):
    monkeypatch.setattr(utils, "_RACY_NS", 0)
    memory = Memory(drive_root=logs)
    for stamp in ("20261004T000000", "20261005T000000"):
        memory.read_task_recent("events.jsonl", "fresh", 200)
        _rotate(logs, stamp)
    memory.read_task_recent("events.jsonl", "fresh", 200)

    kept = jsonl_tail._ARCHIVE_TASKS[str(memory.logs_path("events.jsonl"))]
    assert sorted(pathlib.Path(path).name for path in kept) == [
        "events_20261003T000000.jsonl", "events_20261004T000000.jsonl", "events_20261005T000000.jsonl"]


@pytest.mark.skipif(os.name == "nt", reason="a symlinked data root spelling; Windows needs a privilege for it")
def test_the_boots_listing_spares_the_first_owner_turn_its_archive_listing(logs, monkeypatch):
    """The boot lists archive/ once for the logs a turn's recent sections read (server_boot.remember_log_listings);
    the process's first owner turn then lists nothing, whichever spelling of the data root each of them uses."""
    from ouroboros import server_boot

    monkeypatch.setattr(utils, "_RACY_NS", 0)
    _write(logs / "logs" / "tools.jsonl", [{"task_id": "other", "tool": "read_file"}])
    _write(logs / "archive" / "tools_20261001T000000.jsonl", [{"task_id": "other", "tool": "read_file"}])
    from ouroboros import dialogue_store

    dialogue_store.ensure(logs)  # an installed root: its dialogue imported, as the turns find it
    alias = logs.parent / "data-alias"
    alias.symlink_to(logs, target_is_directory=True)
    server_boot.remember_log_listings(alias).join()  # the boot's listing, off its steps
    listed = _listings(monkeypatch, logs)

    sections = Memory(drive_root=logs).recent_activity_sections("fresh-task")
    assert listed == [] and sections == []
