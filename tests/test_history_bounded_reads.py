"""Unit tests for the bounded rotation-aware JSONL reader (v6.90.x P2).

`gateway/_helpers.read_rotated_jsonl_entries` is the window-doubling tail read
behind /api/chat/history (chat + progress) and /api/logs/{name}: a 512KB live
byte tail that doubles until the FILTERED quota is satisfied (degenerate case =
full read), plus a newest-first archive backfill bounded to 3 files.
"""

from __future__ import annotations

import json

import pytest

from ouroboros.gateway._helpers import read_rotated_jsonl_entries
from ouroboros.jsonl_tail import TAIL_WINDOW_START_BYTES as _TAIL_WINDOW_START_BYTES


def _write_rows(path, count, *, prefix="row", pad=0, start=0):
    lines = []
    for i in range(start, start + count):
        row = {"ts": "2026-08-08T00:00:00Z", "i": i, "kind": prefix}
        if pad:
            row["pad"] = "x" * pad
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _always(entry):
    return isinstance(entry, dict)


def test_small_file_is_read_fully(tmp_path):
    live = tmp_path / "log.jsonl"
    _write_rows(live, 5)
    entries = read_rotated_jsonl_entries(live, tmp_path / "archive", "log", 100, _always)
    assert [e["i"] for e in entries] == [0, 1, 2, 3, 4]


def test_missing_file_yields_empty(tmp_path):
    entries = read_rotated_jsonl_entries(
        tmp_path / "absent.jsonl", tmp_path / "archive", "log", 10, _always
    )
    assert entries == []


def test_first_window_stops_when_quota_satisfied(tmp_path):
    """A large live file with enough matching rows in the last 512KB is read from
    the first tail window only — a strict suffix, not the whole file."""
    live = tmp_path / "log.jsonl"
    # ~2KB per row, ~1500 rows ≈ 3MB; the last 512KB holds ~250 rows.
    _write_rows(live, 1500, pad=2000)
    assert live.stat().st_size > 4 * _TAIL_WINDOW_START_BYTES

    entries = read_rotated_jsonl_entries(live, tmp_path / "archive", "log", 50, _always)

    assert len(entries) >= 50
    assert len(entries) < 1500  # bounded: not a full read
    assert entries[-1]["i"] == 1499  # it is the newest suffix
    indices = [e["i"] for e in entries]
    assert indices == list(range(indices[0], 1500))  # contiguous suffix


def test_window_doubles_until_quota_satisfied(tmp_path):
    """When the first 512KB window holds fewer FILTERED rows than the quota, the
    window doubles until it does."""
    live = tmp_path / "log.jsonl"
    # 1500 padded rows; only every 10th row counts toward the quota.
    _write_rows(live, 1500, pad=2000)

    def _every_tenth(entry):
        return isinstance(entry, dict) and entry.get("i", 0) % 10 == 0

    # First window ≈ 250 rows ≈ 25 matching; ask for 40 → needs a doubled window.
    entries = read_rotated_jsonl_entries(live, tmp_path / "archive", "log", 40, _every_tenth)
    matching = [e for e in entries if _every_tenth(e)]
    assert len(matching) >= 40
    assert entries[-1]["i"] == 1499


def test_degenerate_full_read_when_quota_unsatisfiable(tmp_path):
    """A quota the file cannot satisfy degrades into exactly one full read."""
    live = tmp_path / "log.jsonl"
    _write_rows(live, 1500, pad=2000)
    entries = read_rotated_jsonl_entries(live, tmp_path / "archive", "log", 10**6, _always)
    assert [e["i"] for e in entries][:3] == [0, 1, 2]
    assert len(entries) == 1500  # the whole file, once


def test_archive_backfill_newest_first_until_quota(tmp_path):
    live = tmp_path / "log.jsonl"
    adir = tmp_path / "archive"
    adir.mkdir()
    _write_rows(live, 2, prefix="live", start=200)
    _write_rows(adir / "log_20260808T010000.jsonl", 3, prefix="old", start=0)
    _write_rows(adir / "log_20260808T020000.jsonl", 3, prefix="new", start=100)

    entries = read_rotated_jsonl_entries(live, adir, "log", 4, _always)

    kinds = [e["kind"] for e in entries]
    # Newest archive satisfied the quota; the older one was skipped, and the
    # chosen archive precedes the live rows chronologically.
    assert "old" not in kinds
    assert kinds == ["new", "new", "new", "live", "live"]


def test_archive_backfill_bounded_to_three_files(tmp_path):
    live = tmp_path / "log.jsonl"
    adir = tmp_path / "archive"
    adir.mkdir()
    live.write_text("", encoding="utf-8")
    for gen in range(5):
        _write_rows(adir / f"log_2026080{gen}T000000.jsonl", 1, start=gen)

    entries = read_rotated_jsonl_entries(live, adir, "log", 100, _always)

    # Only the 3 newest archives are consulted even though the quota is unmet.
    assert [e["i"] for e in entries] == [2, 3, 4]


@pytest.mark.parametrize("want", [0, 1])
def test_tiny_quotas_still_return_live_window(tmp_path, want):
    live = tmp_path / "log.jsonl"
    _write_rows(live, 3)
    entries = read_rotated_jsonl_entries(live, tmp_path / "archive", "log", want, _always)
    assert [e["i"] for e in entries] == [0, 1, 2]
