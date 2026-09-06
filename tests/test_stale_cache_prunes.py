"""CPL4-C14/C15 pins: pure-cache and dead-marker age prunes at startup.

``state/code_intel/<root-sha>/`` roots age out by their ``inventory.json``
mtime; ``state/extension_reconcile/failed/`` markers age out by file mtime
(the failure fact is already durable in events.jsonl). Fresh entries and
anything that refuses removal stay.
"""

from __future__ import annotations

import json
import os
import time
from unittest import mock

from ouroboros.code_intelligence import prune_stale_code_intel_roots
from ouroboros.extension_reconcile_queue import prune_failed_reconcile_markers

_OLD = time.time() - 400 * 86400


def test_code_intel_roots_age_out_by_inventory_mtime(tmp_path):
    cache = tmp_path / "state" / "code_intel"
    stale = cache / ("a" * 16)
    stale.mkdir(parents=True)
    (stale / "inventory.json").write_text("{}", encoding="utf-8")
    os.utime(stale / "inventory.json", (_OLD, _OLD))
    fresh = cache / ("b" * 16)
    fresh.mkdir(parents=True)
    (fresh / "inventory.json").write_text("{}", encoding="utf-8")
    orphan_dir = cache / ("c" * 16)  # no inventory: ages by the dir itself
    orphan_dir.mkdir(parents=True)
    os.utime(orphan_dir, (_OLD, _OLD))

    report = prune_stale_code_intel_roots(tmp_path)

    assert not stale.exists() and not orphan_dir.exists()
    assert fresh.exists()
    assert sorted(report["removed"]) == ["a" * 16, "c" * 16]
    assert report["kept"] == 1 and not report["errors"]


def test_code_intel_prune_missing_cache_dir_is_a_noop(tmp_path):
    report = prune_stale_code_intel_roots(tmp_path)
    assert report == {"removed": [], "kept": 0, "errors": []}


def test_failed_reconcile_markers_age_out(tmp_path):
    failed = tmp_path / "state" / "extension_reconcile" / "failed"
    failed.mkdir(parents=True)
    old = failed / "skill-abc-1.json"
    old.write_text(json.dumps({"status": "failed", "attempts": 5}), encoding="utf-8")
    os.utime(old, (_OLD, _OLD))
    fresh = failed / "skill-def-2.json"
    fresh.write_text(json.dumps({"status": "failed", "attempts": 5}), encoding="utf-8")
    pending = tmp_path / "state" / "extension_reconcile" / "pending.json"
    pending.write_text("{}", encoding="utf-8")  # the active queue is untouched
    os.utime(pending, (_OLD, _OLD))

    report = prune_failed_reconcile_markers(tmp_path)

    assert not old.exists() and fresh.exists() and pending.exists()
    assert report["removed"] == ["skill-abc-1.json"] and report["kept"] == 1


def test_the_terminal_failure_survives_the_marker_it_is_cached_in(tmp_path):
    """Audit #15-13: the C15 GC was chosen on the premise that the failure fact
    'is already durable in events.jsonl' — but ``_mark_failed`` wrote only the
    marker, so pruning it destroyed the last-error detail forever. The terminal
    failure is now appended to the event log BEFORE the marker exists, and the
    marker really is only a cache of it."""
    from ouroboros.extension_reconcile_queue import (
        list_extension_reconcile_requests,
        process_extension_reconcile_requests,
        request_extension_reconcile,
    )

    marker = request_extension_reconcile(tmp_path, "telegram", reason="enable")
    for _ in range(5):  # MAX_ATTEMPTS
        with mock.patch(
            "ouroboros.extension_loader.reconcile_extension",
            side_effect=RuntimeError("companion port already bound"),
        ):
            process_extension_reconcile_requests(tmp_path, lambda: {})

    assert not marker.exists()
    failed_dir = tmp_path / "state" / "extension_reconcile" / "failed"
    (failed_marker,) = list(failed_dir.glob("*.json"))
    os.utime(failed_marker, (_OLD, _OLD))
    prune_failed_reconcile_markers(tmp_path)
    assert not failed_marker.exists()
    assert not list_extension_reconcile_requests(tmp_path)

    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
    ]
    terminal = [row for row in rows if row["type"] == "extension_reconcile_failed"]
    assert len(terminal) == 1  # exactly one terminal fact, at the last attempt
    assert terminal[0]["skill"] == "telegram" and terminal[0]["attempts"] == 5
    assert "companion port already bound" in terminal[0]["last_error"]


def test_startup_prune_sweeps_run_both():
    import inspect

    import ouroboros.server_maintenance as sm

    src = inspect.getsource(sm._startup_prune_sweeps)
    assert "prune_stale_code_intel_roots" in src
    assert "prune_failed_reconcile_markers" in src
