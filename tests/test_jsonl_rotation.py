"""Tests for the generalized JSONL log rotation (v6.90.x P2).

`supervisor/state.rotate_jsonl_log_if_needed` rotates logs/<name> into
archive/<prefix>_<ts>.jsonl under the append sidecar lock, suppresses itself in
sentinel-marked isolated benchmark roots, and never collides archive names.
`rotate_chat_log_if_needed` stays as a thin compatibility wrapper.
"""

from __future__ import annotations

import json

import pytest

from supervisor.state import (
    ISOLATED_BENCHMARK_SENTINEL,
    rotate_chat_log_if_needed,
    rotate_jsonl_log_if_needed,
)


# (live filename, archive prefix) — v6.109.29 added events + tools to the
# supervisor-tick rotation alongside chat/progress. Each pair must produce
# the same byte-preserving rename + empty-live side effect.
ROTATED_LOG_NAMES = [
    ("progress.jsonl", "progress"),
    ("events.jsonl", "events"),
    ("tools.jsonl", "tools"),
]


def _seed_log(root, name, rows=50):
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / name
    path.write_text(
        "\n".join(json.dumps({"i": i, "pad": "x" * 100}) for i in range(rows)) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(("name", "prefix"), ROTATED_LOG_NAMES)
def test_rotation_moves_live_to_archive(tmp_path, name, prefix):
    """A >max_bytes file for chat/progress/events/tools rotates to
    archive/<prefix>_<ts>.jsonl, the live file is recreated empty, and no
    line is lost (v6.109.29)."""
    live = _seed_log(tmp_path, name)
    original = live.read_text(encoding="utf-8")

    rotate_jsonl_log_if_needed(tmp_path, name, prefix, max_bytes=100)

    archives = sorted((tmp_path / "archive").glob(f"{prefix}_*.jsonl"))
    assert len(archives) == 1
    assert archives[0].read_text(encoding="utf-8") == original
    assert live.exists()
    assert live.stat().st_size == 0


def test_rotation_noop_below_threshold(tmp_path):
    live = _seed_log(tmp_path, "progress.jsonl", rows=2)
    rotate_jsonl_log_if_needed(tmp_path, "progress.jsonl", "progress", max_bytes=10**9)
    assert not (tmp_path / "archive").exists()
    assert live.stat().st_size > 0


def test_rotation_suppressed_by_isolated_benchmark_sentinel(tmp_path):
    live = _seed_log(tmp_path, "progress.jsonl")
    (tmp_path / ISOLATED_BENCHMARK_SENTINEL).write_text("", encoding="utf-8")

    rotate_jsonl_log_if_needed(tmp_path, "progress.jsonl", "progress", max_bytes=100)

    assert not (tmp_path / "archive").exists()
    assert live.stat().st_size > 0  # untouched: bench harness readers see one file


def test_chat_wrapper_still_rotates_chat(tmp_path):
    live = _seed_log(tmp_path, "chat.jsonl")
    rotate_chat_log_if_needed(tmp_path, max_bytes=100)
    archives = sorted((tmp_path / "archive").glob("chat_*.jsonl"))
    assert len(archives) == 1
    assert live.stat().st_size == 0


def test_same_second_rotations_do_not_clobber_archives(tmp_path, monkeypatch):
    """Second-resolution archive names collide under a fast writer; the second
    rotation must pick a suffixed name that sorts AFTER the first."""
    import supervisor.state as state_mod

    monkeypatch.setattr(state_mod, "utc_now_iso", lambda: "2026-08-08T06:00:00.000000Z")

    _seed_log(tmp_path, "progress.jsonl")
    rotate_jsonl_log_if_needed(tmp_path, "progress.jsonl", "progress", max_bytes=100)
    _seed_log(tmp_path, "progress.jsonl")  # refill within the "same second"
    rotate_jsonl_log_if_needed(tmp_path, "progress.jsonl", "progress", max_bytes=100)

    archives = sorted(p.name for p in (tmp_path / "archive").glob("progress_*.jsonl"))
    assert len(archives) == 2
    # Name-ordered readers must keep chronological order: base name first.
    assert archives[0] == "progress_20260808T060000.jsonl"
    assert archives[1] == "progress_20260808T060000_1.jsonl"
