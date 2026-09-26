"""Old memory-journal snapshots remain readable after every maintenance pass.

Previously this startup sweep digested old knowledge, identity and Pattern
Register old/new contents. A digest cannot restore the complete source after
retention; the compatibility entry point is intentionally non-destructive.
"""

from __future__ import annotations

import inspect
import json

import pytest

from ouroboros.memory_journal_compaction import compact_memory_journal_snapshots

_JOURNALS = (
    "memory/identity_journal.jsonl",
    "memory/knowledge_history.jsonl",
    "memory/knowledge/patterns_history.jsonl",
    "projects/example/knowledge_history.jsonl",
    "memory/scratchpad_journal.jsonl",
)


@pytest.mark.parametrize("journal", _JOURNALS)
def test_old_journal_bytes_remain_complete_through_repeated_maintenance(tmp_path, journal):
    path = tmp_path / journal
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": "2020-01-01T00:00:00+00:00", "old_content": "old\nwith Unicode Я",
           "new_content": "new\nwith Unicode Ё", "old_sha256": "legacy-mismatch"}
    original = (json.dumps(row, ensure_ascii=False) + "\n{legacy broken row\n").encode("utf-8")
    path.write_bytes(original)

    for _ in range(2):
        report = compact_memory_journal_snapshots(tmp_path, retention_days=0,
                                                  now=2_000_000_000.0)
        assert report["digested"] == report["digest_mismatch"] == {}
        assert report["errors"] == []
        assert report["journal_bytes"].get(journal) == (len(original) if journal in
            ("memory/identity_journal.jsonl", "memory/knowledge_history.jsonl",
             "memory/knowledge/patterns_history.jsonl") else None)
        assert path.read_bytes() == original
    # Content digested in an older release cannot be restored, but is not deleted either.
    assert b"old_content" in path.read_bytes() and b"new_content" in path.read_bytes()


def test_maintenance_does_not_create_missing_journals_or_directories(tmp_path):
    root = tmp_path / "absent"
    compact_memory_journal_snapshots(root)
    assert not root.exists()


def test_startup_prune_still_reaches_compatibility_entry_point():
    import ouroboros.server_maintenance as maintenance

    assert "compact_memory_journal_snapshots" in inspect.getsource(maintenance._startup_prune_sweeps)


def test_size_observation_does_not_follow_a_journal_symlink(tmp_path):
    target = tmp_path / "elsewhere"
    target.write_bytes(b"secret data")
    link = tmp_path / "memory" / "knowledge_history.jsonl"
    link.parent.mkdir()
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    report = compact_memory_journal_snapshots(tmp_path)
    assert report["journal_bytes"]["memory/knowledge_history.jsonl"] is None
    assert "memory/knowledge_history.jsonl: not_regular" in report["errors"]
    assert target.read_bytes() == b"secret data"


def test_startup_event_publishes_normal_journal_sizes(tmp_path, monkeypatch):
    import ouroboros.server_maintenance as maintenance
    import supervisor.state as state

    journal = tmp_path / "memory" / "knowledge_history.jsonl"
    journal.parent.mkdir()
    journal.write_bytes(b"full historical text\n")
    rows = []
    monkeypatch.setattr(maintenance, "DATA_DIR", tmp_path)
    monkeypatch.setattr(state, "append_jsonl", lambda _path, row: rows.append(row))
    report = compact_memory_journal_snapshots(tmp_path)
    maintenance._prune_event("memory_journal_observation", ("journal_bytes", "errors"), report=report)
    assert len(rows) == 1
    assert rows[0]["report"]["journal_bytes"]["memory/knowledge_history.jsonl"] == len(b"full historical text\n")
    assert journal.read_bytes() == b"full historical text\n"
    # Pin the real startup caller, not only this unit invocation.
    source = inspect.getsource(maintenance._startup_prune_sweeps)
    assert '_prune_event("memory_journal_observation", ("journal_bytes", "errors")' in source
