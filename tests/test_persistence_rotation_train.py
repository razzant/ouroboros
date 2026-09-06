"""CPL4-C1..C5 rotation-train pins: readers stay whole across a rotated chain.

The generalized rotator (``supervisor/state.rotate_jsonl_log_if_needed``) now
also rotates events/tools/supervisor/task_reflections on the supervisor tick.
events.jsonl is custody authority, so every custody reader must replay the
archive chain, the fault tail-scan must window across it, the settled-terminal
cursor must stay monotonic over it, and the one-time legacy usage import must
snapshot it whole. tools/supervisor/task_reflections tail readers backfill
from the newest archive segments.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.utils import append_jsonl, iter_jsonl_chain_objects
from supervisor.state import rotate_jsonl_log_if_needed


@pytest.fixture(autouse=True)
def _clean_custody_memo():
    custody._CUSTODY.clear()
    yield
    custody._CUSTODY.clear()


def _emit(root, kind, **payload):
    assert custody.emit(root, kind, payload)


def _rotate_events(root):
    rotate_jsonl_log_if_needed(root, "events.jsonl", "events", max_bytes=1)
    assert sorted((root / "archive").glob("events_*.jsonl"))
    assert (root / "logs" / "events.jsonl").stat().st_size == 0


def test_replay_reads_rotated_custody_rows(tmp_path):
    _emit(tmp_path, custody.STARTED, run_id="r1", task_id="t1")
    _emit(tmp_path, custody.SETTLED, run_id="r1", task_id="t1", state="succeeded")
    _rotate_events(tmp_path)
    _emit(tmp_path, custody.STARTED, run_id="r2", task_id="t2")

    state = custody.replay(tmp_path)
    assert state["r1"].settled and state["r1"].terminal_state == "succeeded"
    assert state["r2"].task_id == "t2" and not state["r2"].settled
    assert custody.lookup(tmp_path, "t1", "r1")[0] == custody.OWNED


def test_open_runs_and_pending_invocations_span_chain(tmp_path):
    _emit(tmp_path, custody.START_REQUESTED, invocation_id="inv1", task_id="t1",
          request={"prompt": "x"})
    _emit(tmp_path, custody.STARTED, run_id="r1", task_id="t1", invocation_id="inv1")
    _rotate_events(tmp_path)
    _emit(tmp_path, custody.START_REQUESTED, invocation_id="inv2", task_id="t2",
          request={"prompt": "y"})

    assert [row.run_id for row in custody.open_runs(tmp_path)] == ["r1"]
    pending = {row["invocation_id"] for row in custody.pending_invocations(tmp_path)}
    assert pending == {"inv2"}
    record = custody.invocation_record(tmp_path, "inv1")
    assert record is not None and record["state"] == "started" and record["run_id"] == "r1"


def test_fault_tail_scan_spans_rotated_chain(tmp_path):
    # The fault row lands ONLY in the event log (compact-projection write lost),
    # then rotation buries it in the archive: the 4MB tail must still find it.
    _emit(tmp_path, custody.CONTAINMENT_FAULT, run_id="r1", task_id="t1",
          reason="cancel_unverified")
    _rotate_events(tmp_path)
    _emit(tmp_path, custody.STARTED, run_id="r2", task_id="t2")

    faults = custody.open_containment_faults(tmp_path)
    assert [row.get("run_id") for row in faults] == ["r1"]

    _emit(tmp_path, custody.CONTAINMENT_RESOLVED, run_id="r1", task_id="t1",
          reason="verified_terminal")
    assert custody.open_containment_faults(tmp_path) == []


def test_iter_rows_tail_bytes_bounds_across_chain(tmp_path):
    for index in range(4):
        _emit(tmp_path, custody.STARTED, run_id=f"r{index}", task_id=f"t{index}")
        rotate_jsonl_log_if_needed(tmp_path, "events.jsonl", "events", max_bytes=1)
    segments = sorted((tmp_path / "archive").glob("events_*.jsonl"))
    assert len(segments) == 4

    events_path = custody.event_log_path(tmp_path)
    all_rows = [row["run_id"] for row in custody._iter_rows(events_path)]
    assert all_rows == ["r0", "r1", "r2", "r3"]

    newest_size = segments[-1].stat().st_size
    bounded = [
        row["run_id"]
        for row in custody._iter_rows(events_path, tail_bytes=newest_size)
    ]
    assert bounded == ["r3"]  # the window covers only the newest chain bytes


def test_run_timing_reads_rotated_started_row(tmp_path):
    _emit(tmp_path, custody.STARTED, run_id="r1", task_id="t1", max_seconds=120)
    _rotate_events(tmp_path)
    started_ts, max_seconds = custody.run_timing(tmp_path, "r1")
    assert started_ts and max_seconds == 120


def test_complete_custody_rows_spans_chain(tmp_path):
    from ouroboros.delegate_custody_usage import complete_custody_rows

    _emit(tmp_path, custody.STARTED, run_id="r1", task_id="t1")
    _rotate_events(tmp_path)
    _emit(tmp_path, custody.SETTLED, run_id="r1", task_id="t1", state="succeeded")

    rows = complete_custody_rows(
        custody.event_log_path(tmp_path), "delegate_run", started_type=custody.STARTED,
    )
    assert rows is not None
    assert [row["type"] for row in rows] == [custody.STARTED, custody.SETTLED]


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod-based unreadability needs a non-root POSIX user",
)
def test_unreadable_archive_segment_fails_closed(tmp_path):
    from ouroboros.delegate_custody_usage import complete_custody_rows

    _emit(tmp_path, custody.STARTED, run_id="r1", task_id="t1")
    _rotate_events(tmp_path)
    segment = sorted((tmp_path / "archive").glob("events_*.jsonl"))[0]
    segment.chmod(0)
    try:
        # The strict authority reader refuses an incomplete view...
        assert complete_custody_rows(
            custody.event_log_path(tmp_path), "delegate_run",
        ) is None
        # ...and the unreadable-log probe covers the chain (GR6-4).
        assert custody.custody_log_unreadable(tmp_path) is True
    finally:
        segment.chmod(0o644)
    assert custody.custody_log_unreadable(tmp_path) is False


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod-based unreadability needs a non-root POSIX user",
)
def test_unreadable_archive_directory_is_typed_not_empty(tmp_path):
    """Audit #14-6a: ``Path.glob`` SWALLOWS a PermissionError on archive/ and
    yields nothing, so every chain reader read "this store never rotated" from
    a directory it could not open. Authority readers must see a TYPED
    incomplete view instead; fail-soft readers keep the lenient answer."""
    from ouroboros.delegate_custody_usage import complete_custody_rows
    from ouroboros.utils import JsonlChainUnreadable, jsonl_archive_segments

    _emit(tmp_path, custody.STARTED, run_id="r1", task_id="t1")
    _rotate_events(tmp_path)
    path = custody.event_log_path(tmp_path)
    archive_dir = tmp_path / "archive"
    archive_dir.chmod(0)
    try:
        assert jsonl_archive_segments(path) == []  # fail-soft: degraded window
        with pytest.raises(JsonlChainUnreadable):
            jsonl_archive_segments(path, strict=True)
        assert complete_custody_rows(path, "delegate_run") is None
        assert custody.custody_log_unreadable(tmp_path) is True
    finally:
        archive_dir.chmod(0o755)
    assert custody.custody_log_unreadable(tmp_path) is False
    assert complete_custody_rows(path, "delegate_run") is not None


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod-based unreadability needs a non-root POSIX user",
)
@pytest.mark.parametrize("hide", ["segment", "directory"])
def test_legacy_money_import_fails_closed_on_an_unreadable_chain(tmp_path, hide):
    """Audit #14-6a: the legacy usage import EXPECTS an OSError and fails
    closed — but a permission error on a rotated segment (or on archive/)
    never reached that except, and the hidden ``llm_usage`` rows would have
    been silently excluded from the imported monetary baseline."""
    from ouroboros.usage_ledger import UsageAccountingError
    from ouroboros.usage_legacy_import import _legacy_snapshot

    events_path = tmp_path / "logs" / "events.jsonl"
    append_jsonl(events_path, {"type": "llm_usage", "model": "m1", "cost": 0.5})
    _rotate_events(tmp_path)
    append_jsonl(events_path, {"type": "llm_usage", "model": "m2", "cost": 0.25})
    hidden = (
        tmp_path / "archive" if hide == "directory"
        else sorted((tmp_path / "archive").glob("events_*.jsonl"))[0]
    )
    hidden.chmod(0)
    try:
        with pytest.raises(UsageAccountingError):
            _legacy_snapshot(tmp_path)
    finally:
        hidden.chmod(0o755 if hide == "directory" else 0o644)
    rows, _state, _hashes = _legacy_snapshot(tmp_path)
    assert [row["model"] for row in rows] == ["m1", "m2"]


def test_settled_cursor_advances_across_rotation(tmp_path, monkeypatch):
    import ouroboros.delegate_terminal as dt

    refreshed: list[str] = []
    monkeypatch.setattr(dt, "_task_is_terminal", lambda root, tid: True)
    monkeypatch.setattr(
        dt, "refresh_terminal_reconciliation",
        lambda root, tid: refreshed.append(tid) or True,
    )

    _emit(tmp_path, custody.SETTLED, run_id="r1", task_id="t1", state="succeeded")
    assert dt.refresh_recently_settled_terminals(tmp_path) == 1
    assert refreshed == ["t1"]

    _rotate_events(tmp_path)
    _emit(tmp_path, custody.SETTLED, run_id="r2", task_id="t2", state="succeeded")
    # The chain offset is monotonic across the rotation: t1's row is already
    # consumed even though it now lives in the archive; only t2 is new.
    assert dt.refresh_recently_settled_terminals(tmp_path) == 1
    assert refreshed == ["t1", "t2"]
    assert dt.refresh_recently_settled_terminals(tmp_path) == 0


def test_worker_boot_event_found_after_rotation(tmp_path, monkeypatch):
    from supervisor import workers
    from supervisor.worker_pool_lifecycle import _first_worker_event_since, events_log_cursor

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path, raising=False)
    events_path = tmp_path / "logs" / "events.jsonl"
    append_jsonl(events_path, {"type": "noise"})
    cursor = events_log_cursor()
    append_jsonl(events_path, {"type": "worker_boot", "git_sha": "abc123", "pid": 7})
    _rotate_events(tmp_path)

    evt = _first_worker_event_since(cursor)
    assert evt is not None and evt.get("git_sha") == "abc123"


def test_worker_boot_event_found_when_the_new_live_log_outgrew_the_cursor(tmp_path, monkeypatch):
    """Audit #14-6c: rotation was detected as "the live file is smaller than my
    offset". Under a busy supervisor the FRESH live file is already bigger than
    the old offset by the time the verify reads, so the rotation went
    unnoticed and the read seeked into an unrelated file at a meaningless
    offset — the boot event vanished. Detect by file IDENTITY."""
    from supervisor import workers
    from supervisor.worker_pool_lifecycle import _first_worker_event_since, events_log_cursor

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path, raising=False)
    events_path = tmp_path / "logs" / "events.jsonl"
    append_jsonl(events_path, {"type": "noise", "pad": "x" * 200})
    cursor = events_log_cursor()
    append_jsonl(events_path, {"type": "worker_boot", "git_sha": "abc123", "pid": 7})
    _rotate_events(tmp_path)
    # The new live file overtakes the old offset before the verify runs.
    for i in range(20):
        append_jsonl(events_path, {"type": "llm_round", "i": i, "pad": "y" * 200})
    assert events_path.stat().st_size > cursor[0]

    evt = _first_worker_event_since(cursor)
    assert evt is not None and evt.get("git_sha") == "abc123"
    # The pre-cursor noise row stays excluded: the offset is honored inside the
    # segment the cursor actually named.
    assert _first_worker_event_since(cursor, "noise") is None


def test_legacy_snapshot_includes_rotated_llm_usage(tmp_path):
    from ouroboros.usage_legacy_import import _legacy_snapshot

    events_path = tmp_path / "logs" / "events.jsonl"
    append_jsonl(events_path, {"type": "llm_usage", "model": "m1", "cost": 0.5,
                               "prompt_tokens": 10, "completion_tokens": 5})
    _rotate_events(tmp_path)
    append_jsonl(events_path, {"type": "llm_usage", "model": "m2", "cost": 0.25,
                               "prompt_tokens": 4, "completion_tokens": 2})

    rows, _state, hashes = _legacy_snapshot(tmp_path)
    assert [row["model"] for row in rows] == ["m1", "m2"]
    assert hashes["events.jsonl"]  # the chain snapshot was hashed and archived


def test_memory_tail_backfills_from_rotated_archive(tmp_path):
    from ouroboros.memory import Memory

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "task_reflections_20260101T000000.jsonl").write_text(
        "\n".join(json.dumps({"i": i}) for i in range(15)) + "\n", encoding="utf-8",
    )
    (logs / "task_reflections.jsonl").write_text(
        "\n".join(json.dumps({"i": i}) for i in range(15, 20)) + "\n", encoding="utf-8",
    )

    entries = Memory(tmp_path).read_jsonl_tail("task_reflections.jsonl", 20)
    assert [entry["i"] for entry in entries] == list(range(20))
    # An unrotated store behaves exactly as before.
    short = Memory(tmp_path).read_jsonl_tail("task_reflections.jsonl", 3)
    assert [entry["i"] for entry in short] == [17, 18, 19]


def test_iter_jsonl_chain_objects_chronological(tmp_path):
    path = tmp_path / "logs" / "events.jsonl"
    append_jsonl(path, {"i": 0})
    rotate_jsonl_log_if_needed(tmp_path, "events.jsonl", "events", max_bytes=1)
    append_jsonl(path, {"i": 1})
    rotate_jsonl_log_if_needed(tmp_path, "events.jsonl", "events", max_bytes=1)
    append_jsonl(path, {"i": 2})

    assert [row["i"] for row in iter_jsonl_chain_objects(path)] == [0, 1, 2]


def test_launcher_stdout_copy_is_size_capped():
    """Source pin (CPL4-C5): the pipe-copy thread rotates agent_stdout.log."""
    import inspect

    import launcher

    src = inspect.getsource(launcher.start_agent)
    assert "max_bytes = 2 * 1024 * 1024" in src
    assert 'os.replace(log_path, log_path.with_name(f"{log_path.name}.1"))' in src


def test_supervisor_tick_rotates_the_train():
    """Source pin: the four train logs rotate beside chat/progress."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent / "server.py").read_text(
        encoding="utf-8",
    )
    for name, prefix in (
        ("events.jsonl", "events"),
        ("tools.jsonl", "tools"),
        ("supervisor.jsonl", "supervisor"),
        ("task_reflections.jsonl", "task_reflections"),
    ):
        assert f'rotate_jsonl_log_if_needed(DATA_DIR, "{name}", "{prefix}")' in src
