"""CPL4-C16 pins (owner batch №8, 4A): old journal snapshots go digest-only.

Fresh entries keep their full old/new text; entries older than GC retention
keep only sha256 + length and gain ``content_digested``. Unparseable lines and
rows without a readable ``ts`` survive byte-identical; the consciousness
observation inbox is out of scope.

Audit #15-11 corrective lane: this compactor is the one sweep that destroys
CONTENT, so it also pins that a stored digest is verified before the text it
describes is deleted, that the rewrite publishes only a source nothing else
touched, and that the journal is never loaded whole.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

import pytest

from ouroboros import memory_journal_compaction as mjc
from ouroboros.memory_journal_compaction import compact_memory_journal_snapshots
from ouroboros.utils import utc_now_iso

_OLD_TS = "2020-01-01T00:00:00+00:00"


def _journal(tmp_path, rel):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_old_rows_digested_fresh_rows_kept_full(tmp_path):
    path = _journal(tmp_path, "memory/identity_journal.jsonl")
    old_row = {
        "ts": _OLD_TS, "old_content": "I was v1", "new_content": "I am v2",
        "old_sha256": hashlib.sha256(b"I was v1").hexdigest(), "old_len": 8,
    }
    fresh_row = {"ts": utc_now_iso(), "old_content": "I am v2", "new_content": "I am v3"}
    broken_line = "{not json at all\n"
    path.write_text(
        json.dumps(old_row) + "\n" + broken_line + json.dumps(fresh_row) + "\n",
        encoding="utf-8",
    )

    report = compact_memory_journal_snapshots(tmp_path)

    lines = path.read_text(encoding="utf-8").splitlines()
    digested = json.loads(lines[0])
    assert "old_content" not in digested and "new_content" not in digested
    assert digested["content_digested"] is True
    assert digested["old_sha256"] == hashlib.sha256(b"I was v1").hexdigest()
    assert digested["new_sha256"] == hashlib.sha256(b"I am v2").hexdigest()
    assert digested["new_len"] == len("I am v2")
    assert lines[1] == broken_line.rstrip("\n")  # unreadable: byte-identical
    kept = json.loads(lines[2])
    assert kept["old_content"] == "I am v2" and kept["new_content"] == "I am v3"
    assert report["digested"] == {"memory/identity_journal.jsonl": 1}
    assert not report["digest_mismatch"] and not report["errors"]


@pytest.mark.parametrize("false_fact", [
    {"old_sha256": "pinned-old-hash"},
    {"old_len": 999},
])
def test_a_false_stored_digest_never_costs_the_text(tmp_path, false_fact):
    """Audit #15-11: the compactor used ``setdefault``, so a stored digest that
    contradicted its own text was KEPT while the only correct copy of the
    content was deleted — the lie became the whole record. The pre-fix pin in
    this file asserted exactly that behavior (``old_sha256 == "pinned-old-hash"``
    survives the deletion of ``old_content``); it was cementing the defect and
    is reshaped above to a truthful stored digest.

    A row whose stored fact does not match its text now keeps its FULL content
    and is reported as a typed fact."""
    path = _journal(tmp_path, "memory/identity_journal.jsonl")
    row = {"ts": _OLD_TS, "old_content": "I was v1", "new_content": "I am v2", **false_fact}
    original = json.dumps(row) + "\n"
    path.write_text(original, encoding="utf-8")

    report = compact_memory_journal_snapshots(tmp_path)

    assert path.read_text(encoding="utf-8") == original  # byte-identical
    assert not report["digested"]
    assert report["digest_mismatch"] == {"memory/identity_journal.jsonl": 1}


def test_a_concurrent_append_is_never_dropped_by_the_rewrite(tmp_path):
    """``append_jsonl`` appends WITHOUT the sidecar lock once its own
    acquisition times out, so a row can land mid-rewrite. Whether this pass
    carries it over or abandons the rewrite, the row must survive."""
    path = _journal(tmp_path, "memory/knowledge_history.jsonl")
    old_row = {"ts": _OLD_TS, "old_content": "a", "new_content": "b"}
    path.write_text(json.dumps(old_row) + "\n", encoding="utf-8")
    racing = {"ts": utc_now_iso(), "old_content": "b", "new_content": "c"}
    real_digest_line = mjc._digest_line
    fired = {"done": False}

    def racing_digest_line(raw, cutoff):
        result = real_digest_line(raw, cutoff)
        if not fired["done"]:
            fired["done"] = True
            with path.open("ab") as unlocked_appender:
                unlocked_appender.write(json.dumps(racing).encode("utf-8") + b"\n")
        return result

    mjc._digest_line = racing_digest_line
    try:
        compact_memory_journal_snapshots(tmp_path)
    finally:
        mjc._digest_line = real_digest_line

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[1] == racing  # the concurrent row is intact whichever branch ran


@pytest.mark.skipif(os.name == "nt", reason="the reader holds the journal open; Windows refuses to unlink it (no FILE_SHARE_DELETE)")
def test_a_replaced_source_aborts_the_publish(tmp_path):
    """Identity, not just size: if the journal is swapped for a different file
    under the rewrite, the finished temp must be dropped, not published over
    whatever now lives there."""
    path = _journal(tmp_path, "memory/knowledge/patterns_history.jsonl")
    path.write_text(json.dumps({"ts": _OLD_TS, "old_content": "a", "new_content": "b"}) + "\n",
                    encoding="utf-8")
    replacement = json.dumps({"ts": _OLD_TS, "topic": "someone else's file"}) + "\n"
    real_digest_line = mjc._digest_line
    fired = {"done": False}

    def swapping_digest_line(raw, cutoff):
        result = real_digest_line(raw, cutoff)
        if not fired["done"]:
            fired["done"] = True
            path.unlink()
            path.write_text(replacement, encoding="utf-8")
        return result

    mjc._digest_line = swapping_digest_line
    try:
        report = compact_memory_journal_snapshots(tmp_path)
    finally:
        mjc._digest_line = real_digest_line

    assert path.read_text(encoding="utf-8") == replacement
    assert not report["digested"]
    assert report["errors"] == [
        {"journal": "memory/knowledge/patterns_history.jsonl", "error": "source_changed"},
    ]
    assert not list(path.parent.glob("*.compact.tmp"))


def test_the_journal_is_never_loaded_whole(tmp_path, monkeypatch):
    """Bounded/streaming (audit #15-11c): these journals are the worst byte
    offenders in the memory plane; a whole-file read is the thing being fixed.
    Poison the whole-file readers and the compaction must still work."""
    path = _journal(tmp_path, "memory/identity_journal.jsonl")
    rows = [{"ts": _OLD_TS, "old_content": f"o{i}", "new_content": f"n{i}"} for i in range(50)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _boom(self, *args, **kwargs):
        raise AssertionError(f"whole-file read of {self}")

    monkeypatch.setattr(pathlib.Path, "read_bytes", _boom)
    report = compact_memory_journal_snapshots(tmp_path)

    assert report["digested"] == {"memory/identity_journal.jsonl": 50}


def test_the_rewrite_takes_an_unstealable_lock():
    """Owner-aware stale: elapsed time alone must never hand a second writer
    the journal this destructive rewrite is holding."""
    import inspect

    src = inspect.getsource(mjc._compact_one)
    assert "owner_aware_stale=True" in src


def test_patterns_history_gains_derived_digests(tmp_path):
    path = _journal(tmp_path, "memory/knowledge/patterns_history.jsonl")
    path.write_text(json.dumps({
        "ts": _OLD_TS, "task_id": "t", "markers": ["m"],
        "old_content": "old body", "new_content": "new body\n",
    }) + "\n", encoding="utf-8")

    compact_memory_journal_snapshots(tmp_path)

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["old_sha256"] == hashlib.sha256(b"old body").hexdigest()
    assert row["new_len"] == len("new body\n")
    assert "old_content" not in row and row["content_digested"] is True


def test_row_without_readable_ts_keeps_full_text(tmp_path):
    path = _journal(tmp_path, "memory/knowledge_history.jsonl")
    original = json.dumps({"topic": "x", "old_content": "a", "new_content": "b"}) + "\n"
    path.write_text(original, encoding="utf-8")

    report = compact_memory_journal_snapshots(tmp_path)

    assert path.read_text(encoding="utf-8") == original
    assert not report["digested"] and not report["errors"]


def test_observation_inbox_is_out_of_scope(tmp_path):
    inbox = tmp_path / "state" / "consciousness_observations.jsonl"
    inbox.parent.mkdir(parents=True)
    original = json.dumps({"ts": _OLD_TS, "op": "enqueue", "payload": "keep me"}) + "\n"
    inbox.write_text(original, encoding="utf-8")

    compact_memory_journal_snapshots(tmp_path)

    assert inbox.read_text(encoding="utf-8") == original


def test_startup_prune_sweeps_run_the_compaction():
    import inspect

    import ouroboros.server_maintenance as sm

    assert "compact_memory_journal_snapshots" in inspect.getsource(sm._startup_prune_sweeps)
