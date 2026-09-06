"""ABI-2 (v7.0, owner decision Q8=B): ``_schema_version`` stamp-on-write for
durable task results and the QUARANTINE reader — the F12 semantics family:
future-refusal, malformed, idempotency, N−1, rollback.

Q8=B verbatim: no legacy converter and no compat machinery — an inadmissible
stored row (unstamped pre-7.0 history, future/invalid stamp, malformed JSON,
or the retired ``improvement_policy: "until_deadline"`` contract form; ledger
f30 entry 4) moves byte-unchanged into ``task_results/quarantine/`` and the
read reports no result. Owner decision 6.3=B pins log-only visibility: ONE
durable ``task_results_quarantined`` events row per read/scan batch — no UI
counter, no chat notice. ``state.json`` / ``queue_snapshot.json`` ride the
same decision as stamp-on-write ONLY (form unchanged, readers require no
stamp). The N−1/rollback fixtures here are shared property with the ABI-7
updater shim suite (tests/test_update_tx_nminus1_shim.py).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
from ouroboros.task_results import (
    TASK_RESULT_QUARANTINE_DIR,
    TASK_RESULT_SCHEMA_VERSION,
    list_task_results,
    load_task_result,
    task_result_schema_refusal,
    write_task_result,
)


def _results_dir(root: pathlib.Path) -> pathlib.Path:
    return root / "task_results"


def _write_raw(root: pathlib.Path, task_id: str, payload) -> pathlib.Path:
    path = _results_dir(root) / f"{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, (dict, list)):
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text(str(payload), encoding="utf-8")
    return path


def _quarantine_files(root: pathlib.Path):
    qdir = _results_dir(root) / TASK_RESULT_QUARANTINE_DIR
    return sorted(qdir.glob("*.json")) if qdir.is_dir() else []


def _quarantine_events(root: pathlib.Path):
    path = root / "logs" / "events.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("type") == "task_results_quarantined"]


# --------------------------------------------------------------------- writers


def test_write_task_result_stamps_the_schema_version(tmp_path):
    write_task_result(tmp_path, "t1", "completed", result="done")
    raw = json.loads((_results_dir(tmp_path) / "t1.json").read_text(encoding="utf-8"))
    assert raw[SCHEMA_VERSION_KEY] == TASK_RESULT_SCHEMA_VERSION
    assert load_task_result(tmp_path, "t1")["result"] == "done"


def test_lifecycle_write_stamps_a_live_unstamped_row(tmp_path):
    """The N−1 transition path is stamp-on-write, NOT a converter: a live
    pre-upgrade task's next lifecycle write stamps the merged row in place."""
    _write_raw(tmp_path, "t1", {"task_id": "t1", "status": "running", "note": "old"})
    write_task_result(tmp_path, "t1", "completed", result="done")
    raw = json.loads((_results_dir(tmp_path) / "t1.json").read_text(encoding="utf-8"))
    assert raw[SCHEMA_VERSION_KEY] == TASK_RESULT_SCHEMA_VERSION
    assert raw["note"] == "old" and raw["status"] == "completed"
    assert not _quarantine_files(tmp_path)


def test_plan_review_writer_stamps_the_row(tmp_path):
    from ouroboros.task_results import PLAN_REVIEW_STATE_KEY, _update_plan_review_state

    _update_plan_review_state(tmp_path, "t2", lambda state: state)
    raw = json.loads((_results_dir(tmp_path) / "t2.json").read_text(encoding="utf-8"))
    assert raw[SCHEMA_VERSION_KEY] == TASK_RESULT_SCHEMA_VERSION
    assert PLAN_REVIEW_STATE_KEY in raw


def test_owner_hurry_writer_stamps_the_row(tmp_path):
    from ouroboros.owner_hurry import record_requested

    record_requested(tmp_path, "t3", request_id="r1", attempt=1)
    raw = json.loads((_results_dir(tmp_path) / "t3.json").read_text(encoding="utf-8"))
    assert raw[SCHEMA_VERSION_KEY] == TASK_RESULT_SCHEMA_VERSION


def test_writer_refuses_a_row_owned_by_a_newer_schema(tmp_path):
    """Future-refusal on the WRITE side: after a binary rollback, fields of
    schema 1 must never be merged over (and silently downgrade) a row a newer
    release stamped."""
    path = _write_raw(tmp_path, "t4", {
        SCHEMA_VERSION_KEY: TASK_RESULT_SCHEMA_VERSION + 1,
        "task_id": "t4", "status": "running",
    })
    before = path.read_bytes()
    with pytest.raises(ValueError, match="TASK_RESULT_SCHEMA_REFUSED"):
        write_task_result(tmp_path, "t4", "completed", result="late")
    assert path.read_bytes() == before


# -------------------------------------------------------------------- readers


def test_unstamped_history_is_quarantined_not_converted(tmp_path):
    legacy = {"task_id": "t5", "status": "completed", "result": "pre-7.0"}
    path = _write_raw(tmp_path, "t5", legacy)
    original = path.read_bytes()

    assert load_task_result(tmp_path, "t5") is None

    assert not path.exists()
    moved = _quarantine_files(tmp_path)
    assert [p.name for p in moved] == ["t5.json"]
    assert moved[0].read_bytes() == original  # bytes preserved, never rewritten
    events = _quarantine_events(tmp_path)
    assert len(events) == 1
    assert events[0]["count"] == 1 and events[0]["first_task_id"] == "t5"
    assert events[0]["reasons"] == {"unstamped_pre_7_0": 1}
    # 6.3=B: log-only visibility — no chat notice is minted anywhere.
    assert not (tmp_path / "logs" / "chat.jsonl").exists()


def test_quarantine_is_idempotent_for_repeat_reads(tmp_path):
    _write_raw(tmp_path, "t6", {"task_id": "t6", "status": "completed"})
    assert load_task_result(tmp_path, "t6") is None
    assert load_task_result(tmp_path, "t6") is None  # second read: plain absence
    assert len(_quarantine_files(tmp_path)) == 1
    assert len(_quarantine_events(tmp_path)) == 1  # no second event


@pytest.mark.parametrize("payload, reason", [
    ("{ not json", "malformed"),
    (["not", "an", "object"], "malformed"),
    ({SCHEMA_VERSION_KEY: TASK_RESULT_SCHEMA_VERSION + 1, "task_id": "t7", "status": "completed"},
     "future_schema"),
    ({SCHEMA_VERSION_KEY: "1", "task_id": "t7", "status": "completed"}, "invalid_stamp"),
    ({SCHEMA_VERSION_KEY: True, "task_id": "t7", "status": "completed"}, "invalid_stamp"),
    ({SCHEMA_VERSION_KEY: 0, "task_id": "t7", "status": "completed"}, "invalid_stamp"),
])
def test_inadmissible_rows_are_quarantined_with_the_typed_reason(tmp_path, payload, reason):
    _write_raw(tmp_path, "t7", payload)
    assert load_task_result(tmp_path, "t7") is None
    assert len(_quarantine_files(tmp_path)) == 1
    events = _quarantine_events(tmp_path)
    assert len(events) == 1 and events[0]["reasons"] == {reason: 1}


def test_retired_until_deadline_contract_is_quarantined(tmp_path):
    """Ledger f30 entry 4 residual: a stored contract still carrying the
    retired pre-7.0 ``until_deadline`` pacing alias is malformed to the 7.0
    acceptance-wallet authority — same quarantine, even when stamped."""
    row = {
        SCHEMA_VERSION_KEY: TASK_RESULT_SCHEMA_VERSION,
        "task_id": "t8", "status": "completed",
        "task_contract": {
            "schema_version": 1, "deadline_at": "",
            "budget_profile": {"improvement_policy": "until_deadline"},
        },
    }
    assert task_result_schema_refusal(row) == "retired_contract_until_deadline"
    _write_raw(tmp_path, "t8", row)
    assert load_task_result(tmp_path, "t8") is None
    events = _quarantine_events(tmp_path)
    assert len(events) == 1
    assert events[0]["reasons"] == {"retired_contract_until_deadline": 1}


def test_list_scan_quarantines_the_whole_batch_as_one_event(tmp_path):
    for tid in ("bad1", "bad2", "bad3"):
        _write_raw(tmp_path, tid, {"task_id": tid, "status": "completed"})
    write_task_result(tmp_path, "good1", "completed")
    write_task_result(tmp_path, "good2", "failed")

    rows = list_task_results(tmp_path)

    assert sorted(row["task_id"] for row in rows) == ["good1", "good2"]
    assert len(_quarantine_files(tmp_path)) == 3
    events = _quarantine_events(tmp_path)
    assert len(events) == 1  # one batched event, never one per file
    assert events[0]["count"] == 3 and events[0]["first_task_id"] == "bad1"
    # A later scan sees a clean directory: quarantined rows are invisible.
    assert sorted(r["task_id"] for r in list_task_results(tmp_path)) == ["good1", "good2"]
    assert len(_quarantine_events(tmp_path)) == 1


def test_strict_read_raises_and_never_mutates_storage(tmp_path):
    path = _write_raw(tmp_path, "t9", {"task_id": "t9", "status": "completed"})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="quarantined_schema: unstamped_pre_7_0"):
        load_task_result(tmp_path, "t9", strict=True)
    with pytest.raises(ValueError, match="quarantined_schema"):
        list_task_results(tmp_path, strict=True)
    assert path.read_bytes() == before
    assert not _quarantine_files(tmp_path)
    assert not _quarantine_events(tmp_path)


def test_plain_absence_has_no_side_effects(tmp_path):
    assert load_task_result(tmp_path, "never-written") is None
    assert not (_results_dir(tmp_path) / TASK_RESULT_QUARANTINE_DIR).exists()
    assert not _quarantine_events(tmp_path)


def test_rollback_a_stamped_row_reads_unchanged_under_an_n_minus_1_reader(tmp_path):
    """Rollback direction of F12: the stamp is ONE additive key, so the N−1
    reader (plain JSON, no admission gate) sees the full row unchanged."""
    write_task_result(tmp_path, "t10", "completed", result="done")
    raw = json.loads((_results_dir(tmp_path) / "t10.json").read_text(encoding="utf-8"))
    unstamped = {key: value for key, value in raw.items() if key != SCHEMA_VERSION_KEY}
    assert set(raw) - set(unstamped) == {SCHEMA_VERSION_KEY}
    assert unstamped["task_id"] == "t10" and unstamped["status"] == "completed"
    assert unstamped["result"] == "done"


# ------------------------------------------- state.json / queue_snapshot.json


def test_state_save_stamps_and_an_unstamped_state_loads_unchanged(tmp_path, monkeypatch):
    from supervisor import state

    monkeypatch.setattr(state, "STATE_PATH", tmp_path / "state" / "state.json")
    monkeypatch.setattr(state, "STATE_LAST_GOOD_PATH", tmp_path / "state" / "state.last_good.json")
    monkeypatch.setattr(state, "STATE_LOCK_PATH", tmp_path / "locks" / "state.lock")

    state.save_state({"spent_usd": 1.25})
    for path in (state.STATE_PATH, state.STATE_LAST_GOOD_PATH):
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw[SCHEMA_VERSION_KEY] == state.STATE_SCHEMA_VERSION

    # N−1 direction: readers require no stamp — a pre-7.0 file loads unchanged.
    unstamped = {key: value for key, value in raw.items() if key != SCHEMA_VERSION_KEY}
    state.STATE_PATH.write_text(json.dumps(unstamped), encoding="utf-8")
    loaded = state.load_state()
    assert loaded["spent_usd"] == 1.25


def test_queue_snapshot_stamps_and_an_unstamped_snapshot_still_parses(tmp_path, monkeypatch):
    from supervisor import queue as queue_mod
    from supervisor.queue_snapshot import (
        QUEUE_SNAPSHOT_SCHEMA_VERSION,
        persist_queue_snapshot,
        restore_pending_from_snapshot,
    )
    from ouroboros.utils import utc_now_iso

    snapshot_path = tmp_path / "state" / "queue_snapshot.json"
    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_mod, "QUEUE_SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(queue_mod, "PENDING", [])
    monkeypatch.setattr(queue_mod, "RUNNING", {})
    monkeypatch.setattr(queue_mod, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr(queue_mod, "BUDGET_ROOT_FENCES", {})

    assert persist_queue_snapshot("test") is True
    raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert raw[SCHEMA_VERSION_KEY] == QUEUE_SNAPSHOT_SCHEMA_VERSION
    assert raw["pending"] == [] and raw["running"] == []

    # N−1 direction: the restore path requires no stamp (form unchanged).
    unstamped = {key: value for key, value in raw.items() if key != SCHEMA_VERSION_KEY}
    unstamped["ts"] = utc_now_iso()
    snapshot_path.write_text(json.dumps(unstamped), encoding="utf-8")
    assert restore_pending_from_snapshot() == 0  # parsed past the gate, nothing to restore
