"""Observability census pins (CPL4-C22, owner 7A: the retention knob is GONE).

``prune_observability_blobs`` counts manifests and blobs for startup
telemetry and deletes NOTHING — the preserve-indefinitely contract. The
former ``OUROBOROS_OBSERVABILITY_RETENTION_DAYS`` knob (parsed, clamped,
reported, deleting nothing) is retired entirely: absent from the module and
listed in ``RETIRED_SETTING_KEYS`` so stored ghosts drop on settings load.
"""

from __future__ import annotations

import os
import time

from ouroboros.observability import prune_observability_blobs


def _seed_store(tmp_path):
    calls = tmp_path / "observability" / "calls" / "t1"
    blobs = tmp_path / "observability" / "blobs"
    calls.mkdir(parents=True)
    blobs.mkdir(parents=True)
    aged = time.time() - 4000 * 86400
    manifests = [calls / "a.json", calls / "b.json"]
    blob_files = [blobs / "x.gz", blobs / "y.gz", blobs / "z.gz"]
    for path in (*manifests, *blob_files):
        path.write_bytes(b"data")
        os.utime(path, (aged, aged))
    return manifests, blob_files


def test_census_counts_and_preserves_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_OBSERVABILITY_RETENTION_DAYS", "1")  # inert: retired
    manifests, blob_files = _seed_store(tmp_path)

    report = prune_observability_blobs(tmp_path)

    assert report["preserved_indefinitely"] is True
    assert report["manifest_count"] == 2 and report["blob_count"] == 3
    assert not report["errors"]
    assert all(path.exists() for path in (*manifests, *blob_files))


def test_absent_store_reports_empty_census(tmp_path):
    report = prune_observability_blobs(tmp_path)
    assert report == {
        "preserved_indefinitely": True, "manifest_count": 0, "blob_count": 0, "errors": [],
    }


def test_retention_knob_is_retired_everywhere():
    import inspect

    import ouroboros.observability as observability
    from ouroboros.settings_defaults import RETIRED_SETTING_KEYS

    source = inspect.getsource(observability)
    # The docstring may still NAME the retired knob; nothing may READ it.
    assert 'environ.get("OUROBOROS_OBSERVABILITY_RETENTION_DAYS"' not in source
    assert "OUROBOROS_OBSERVABILITY_RETENTION_DAYS" in RETIRED_SETTING_KEYS
