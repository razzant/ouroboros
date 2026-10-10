"""A seal audit reads its retained history once per pass, including the unknown path."""

import json
import os
import pathlib
import sys

import pytest

from ouroboros import model_send_seal as seals
from tests import fixtures_usage_store as usage_fixtures
from tests import test_model_send_seal as seal_fixtures


data_root = seal_fixtures.data_root
_UNREADABLE = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod-based unreadability needs a non-root POSIX user")


def _manifest(root, attempt_id):
    """Only the manifest projection consumed by the reverse direction."""
    path = root / "observability" / "calls" / "batch" / (attempt_id + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "task_id": "batch", "call_id": attempt_id,
        "model_send_seal": {"attempt_id": attempt_id},
    }), encoding="utf-8")


def _archive(root, count):
    """``count`` sealed attempts folded before the store existed: their rows
    live only in a retained archive segment."""
    attempts = [usage_fixtures._settle(root, cost=0.0, cost_final=True) for _ in range(count)]
    usage_fixtures.fold_into_archive(root, [attempt.attempt_id for attempt in attempts])
    for attempt in attempts:
        _manifest(root, attempt.attempt_id)


def _recording(monkeypatch, result=None):
    calls = []
    scan = seals._retained_attempt_ids

    def recorded(root, wanted):
        calls.append(set(wanted))
        return scan(root, wanted) if result is None else result(root, wanted)

    monkeypatch.setattr(seals, "_retained_attempt_ids", recorded)
    return calls


def test_live_only_sweep_never_loads_archive(data_root, monkeypatch):
    seal_fixtures._dispatch(data_root, "live")
    calls = _recording(monkeypatch)

    report = seals.reconcile_model_send_seals(data_root)

    assert report["seals"] == report["sealed_attempts"] == 1
    assert report["facts_written"] == 0
    assert calls == [set()]  # every seal joined the store: nothing retained is wanted


def test_large_seal_batch_reads_retained_history_once(data_root, monkeypatch):
    # The incident had 415 generations of folded history.
    count = 415
    _archive(data_root, count)
    calls = _recording(monkeypatch)
    opened = []
    real_open = open

    def counting_open(path, *args, **kwargs):
        if "usage_attempts" in str(path) or "usage_ledger" in str(path):
            opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)
    report = seals.reconcile_model_send_seals(data_root)

    assert report["seals"] == count and report["facts_written"] == 0
    assert len(calls) == 1 and len(calls[0]) == count
    assert len(opened) == len(set(opened))  # each retained file at most once per pass


@_UNREADABLE
def test_each_sweep_rechecks_retained_history_and_keeps_no_stale_answer(data_root, monkeypatch):
    from tests.fixtures_usage_store import ARCHIVE_SEGMENT_REL

    _archive(data_root, 2)
    calls = _recording(monkeypatch)
    assert seals.reconcile_model_send_seals(data_root)["facts_written"] == 0
    segment = data_root / ARCHIVE_SEGMENT_REL
    segment.chmod(0)
    _manifest(data_root, "actually-absent")
    try:
        report = seals.reconcile_model_send_seals(data_root)
    finally:
        segment.chmod(0o644)

    assert len(calls) == 2
    assert report["status"] == "unknown" and report["seals"] == 3
    assert report["orphan_seals"] == report["facts_written"] == 0


def test_unknown_history_is_attempted_once_and_forward_checks_continue(data_root, monkeypatch):
    _archive(data_root, 3)
    live = seal_fixtures._dispatch(data_root, "live-missing-seal")
    pathlib.Path(live["candidate_manifest_ref"]["path"]).unlink()
    calls = _recording(monkeypatch, result=lambda _root, _wanted: None)

    report = seals.reconcile_model_send_seals(data_root)

    assert len(calls) == 1
    assert report["seals"] == 3 and report["orphan_seals"] == 0
    assert report["unlogged_attempts"] == report["facts_written"] == 1
    assert seal_fixtures._violation_events(data_root)[0]["kind"] == "unlogged_attempt"


def test_manifest_selection_precedes_live_snapshot(data_root, monkeypatch):
    seal_fixtures._dispatch(data_root, "selected")
    select = seals._seal_manifest_paths

    def new_attempt_after_selection(root, limit):
        selected = select(root, limit)
        seal_fixtures._dispatch(root, "arrived-after-selection")
        return selected

    monkeypatch.setattr(seals, "_seal_manifest_paths", new_attempt_after_selection)
    report = seals.reconcile_model_send_seals(data_root)

    # Reverse audit is pinned to the selected manifests, while the later live
    # snapshot also sees the new reservation and its forward sealing evidence.
    assert report["seals"] == 1 and report["sealed_attempts"] == 2
    assert report["facts_written"] == 0
