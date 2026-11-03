"""Durable job facts survive the process that wrote them (``ouroboros/maintenance_jobs.py``)."""

from __future__ import annotations

import pytest

from ouroboros import maintenance_jobs, store, store_hot_facts, store_import

# A fresh store carries the import's own row and the hot-fact job its schema publication records
# (``store_hot_facts.upgrade``: done, nothing to classify); no other row of ``meta`` is a job.
FRESH = {"current_store_import", store_hot_facts.JOB}


def test_job_facts_round_trip_and_survive_a_fresh_read(tmp_path):
    store_import.ensure_current_store(tmp_path)
    assert maintenance_jobs.job(tmp_path, "legacy_reconcile") is None
    imported = maintenance_jobs.job(tmp_path, "current_store_import")
    assert imported["state"] == "done" and imported["done"] == imported["total"] == 1
    assert maintenance_jobs.job(tmp_path, store_hot_facts.JOB)["state"] == "done", "a fresh store owes no classification"
    assert set(maintenance_jobs.jobs(tmp_path)) == FRESH

    maintenance_jobs.record(tmp_path, "legacy_reconcile", state="running", cursor="task-0042", done=42, total=None)
    maintenance_jobs.record(tmp_path, "activity_backfill", state="done", negative=["proj_x"])
    row = maintenance_jobs.record(tmp_path, "legacy_reconcile", state="done", done=100)

    assert row["state"] == "done" and row["cursor"] == "task-0042" and row["done"] == 100 and row["updated_at"]
    assert maintenance_jobs.is_done(tmp_path, "legacy_reconcile")
    assert set(maintenance_jobs.jobs(tmp_path)) == FRESH | {"legacy_reconcile", "activity_backfill"}
    assert maintenance_jobs.job(tmp_path, "activity_backfill")["negative"] == ["proj_x"]


def test_job_range_read_ignores_other_meta_keys(tmp_path):
    store_import.ensure_current_store(tmp_path)
    maintenance_jobs.record(tmp_path, "a", state="pending")
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") and txn.meta("lock_tier")  # meta rows that are not jobs
    assert set(maintenance_jobs.jobs(tmp_path)) == {"a", *FRESH}  # other meta keys are not jobs


@pytest.mark.parametrize("name", ["", "has:colon", None])
def test_invalid_job_names_are_refused(tmp_path, name):
    store_import.ensure_current_store(tmp_path)
    with pytest.raises(ValueError):
        maintenance_jobs.record(tmp_path, name, state="pending")


def test_invalid_state_is_refused(tmp_path):
    store_import.ensure_current_store(tmp_path)
    with pytest.raises(ValueError):
        maintenance_jobs.record(tmp_path, "x", state="almost")


def test_named_rows_are_read_without_decoding_the_others(tmp_path):
    """The dispatcher's look names its jobs: a row another owner keeps proportional to its own inventory
    (the drive settlement's verdicts once lived on ``job:drive_settlement``) is neither returned nor
    decoded for it; the unnamed read still returns every job row."""
    from tests._cost_meter import CostMeter, tagged

    store_import.ensure_current_store(tmp_path)
    maintenance_jobs.record(tmp_path, "small", state="done")
    maintenance_jobs.record(tmp_path, "wide", state="done", verdicts={f"drive-{index}": "x" * 100 for index in range(300)})
    with CostMeter(tmp_path) as meter:
        named = tagged("named", lambda: maintenance_jobs.jobs(tmp_path, ["small", "absent"]))
        every = tagged("every", lambda: maintenance_jobs.jobs(tmp_path))
    assert set(named) == {"small"} and {"small", "wide"} <= set(every)
    assert meter.snapshot("named")["counts"]["json_chars"] < 1000 < meter.snapshot("every")["counts"]["json_chars"]
    assert maintenance_jobs.jobs(tmp_path, []) == {}
