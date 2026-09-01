"""CyberGym abandoned-residue cost grace and cancel-404 custody recovery tests.

A transient provider failure can leave a usage-ledger row ``unresolved``
forever, so a completed gateway frame's ``cost_final`` never turns true.
These tests pin the bounded grace that delivers such frames with the residue
disclosed, the refusal to deliver any other cost-pending frame early, and
the cancel-404 recovery that adopts an already-terminal task instead of
failing it.
"""

from __future__ import annotations

import datetime
import hashlib
import json

import pytest

from devtools.benchmarks.cybergym.cybergym_adapter import BudgetLedger, run_campaign
from devtools.benchmarks.cybergym.cybergym_executor import (
    CyberGymExecutor,
    ExecutorFailure,
)
from devtools.benchmarks.cybergym.cybergym_wire import _CostGraceTracker
from tests.test_cybergym_executor import _config, dataclasses_replace


def _completed_abandoned_residue_frame(task_id, *, residue=0.035398, age_sec=3600):
    """A completed gateway frame blocked only by an abandoned unresolved row."""
    moment = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=age_sec)
    ).isoformat()
    return {
        "task_id": task_id,
        "status": "completed",
        "ts": moment,
        "updated_at": moment,
        "cost_usd": 0.386527,
        "accounted_upper_bound_usd": 0.386527,
        "unresolved_upper_bound_usd": residue,
        "reserved_usd": 0.0,
        "unknown_unmetered": 0,
        "non_final_rows": 1,
        "cost_final": False,
        "cost_estimated": False,
        "cost_accounting_status": "available",
        "ledger_integrity_degraded": False,
    }


def test_gateway_completed_abandoned_residue_accepted_after_grace(tmp_path):
    config = _config(tmp_path, provider_probe=False, task_timeout_sec=60)
    task_id = "cybergym-cost-grace"
    frame = _completed_abandoned_residue_frame(task_id)
    calls = []

    def http(method, _url, **_kwargs):
        calls.append(method)
        if method == "POST":
            return {"task_id": task_id, "status": "scheduled"}
        return frame

    executor = CyberGymExecutor(
        dataclasses_replace(config, http_runner=http, sleep=lambda _seconds: None)
    )
    result = executor._gateway_wait(  # noqa: SLF001 - accounting contract
        {"task_id": task_id, "description": "test"},
        config.run_root / "checkpoint.json",
    )

    assert calls == ["POST", "GET"]
    grace = result["cost_grace_acceptance"]
    assert grace["reason"] == "abandoned_unresolved_residue"
    assert grace["unresolved_upper_bound_usd"] == pytest.approx(0.035398)
    # The residue is disclosed, never silently called fully final.
    assert result["cost_final"] is False


def test_gateway_live_reservation_is_never_grace_accepted(tmp_path):
    config = _config(tmp_path, provider_probe=False, task_timeout_sec=60)
    task_id = "cybergym-cost-live-reservation"
    pending = _completed_abandoned_residue_frame(task_id)
    pending["reserved_usd"] = 0.5  # a genuinely in-flight reservation
    final = dict(pending)
    final["reserved_usd"] = 0.0
    final["unresolved_upper_bound_usd"] = 0.0
    final["non_final_rows"] = 0
    final["cost_final"] = True
    calls = []
    status_rows = iter((pending, pending, final))

    def http(method, _url, **_kwargs):
        calls.append(method)
        if method == "POST":
            return {"task_id": task_id, "status": "scheduled"}
        return next(status_rows)

    executor = CyberGymExecutor(
        dataclasses_replace(config, http_runner=http, sleep=lambda _seconds: None)
    )
    result = executor._gateway_wait(  # noqa: SLF001 - accounting contract
        {"task_id": task_id, "description": "test"},
        config.run_root / "checkpoint.json",
    )

    assert calls == ["POST", "GET", "GET", "GET"]
    assert result["cost_final"] is True
    assert "cost_grace_acceptance" not in result


@pytest.mark.parametrize(
    "missing",
    [
        "ledger_integrity_degraded",
        "cost_accounting_status",
        "cost_estimated",
        "unknown_unmetered",
        "reserved_usd",
    ],
)
def test_cost_grace_requires_every_accounting_axis(missing):
    from devtools.benchmarks.cybergym.cybergym_wire import (
        _abandoned_cost_residue_usd,
    )

    frame = _completed_abandoned_residue_frame("cybergym-grace-sparse")
    frame.pop(missing)
    assert _abandoned_cost_residue_usd(frame) is None


def test_cost_grace_marker_must_prove_full_wait():
    from devtools.benchmarks.cybergym.cybergym_wire import _valid_cost_grace

    frame = _completed_abandoned_residue_frame("cybergym-grace-marker")
    marked = _CostGraceTracker().accept(
        frame,
        now=0.0,
        wall_now=datetime.datetime.now(datetime.timezone.utc).timestamp(),
    )
    assert marked is not None
    marked["cost_grace_acceptance"]["waited_sec"] = 1
    assert _valid_cost_grace(marked) is None


def test_cancel_404_recovers_terminal_gateway_payload(tmp_path):
    config = _config(tmp_path, poll_interval_sec=0)
    task_id = "cybergym-cancel-404"
    terminal = _completed_abandoned_residue_frame(task_id)
    calls = []
    responses = iter(
        (
            {"status_code": 404, "body": {"detail": "task not found"}},
            {"status_code": 200, "body": terminal},
        )
    )

    def http(method, _url, **_kwargs):
        calls.append(method)
        return next(responses)

    executor = CyberGymExecutor(dataclasses_replace(config, http_runner=http))
    executor._gateway_attempts[task_id] = {  # noqa: SLF001 - custody assertion
        "gateway_task_id": task_id,
        "status": "submitted",
    }
    checkpoint = config.run_root / "checkpoint.json"
    result = executor._cancel_gateway_task(task_id, checkpoint)  # noqa: SLF001

    assert result["status"] == "completed"
    assert result["cost_grace_acceptance"]["unresolved_upper_bound_usd"] == pytest.approx(0.035398)
    assert calls == ["POST", "GET"]
    assert task_id not in executor._gateway_attempts
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["status"] == "completed"
    assert saved["cancel_status_code"] == 404


def test_cancel_404_without_terminal_record_keeps_typed_failure(tmp_path):
    config = _config(tmp_path, poll_interval_sec=0)
    task_id = "cybergym-cancel-404-gone"
    calls = []

    def http(method, _url, **_kwargs):
        calls.append(method)
        return {"status_code": 404, "body": {"detail": "task not found"}}

    executor = CyberGymExecutor(dataclasses_replace(config, http_runner=http))
    executor._gateway_attempts[task_id] = {  # noqa: SLF001 - custody assertion
        "gateway_task_id": task_id,
        "status": "submitted",
    }
    with pytest.raises(ExecutorFailure):
        executor._cancel_gateway_task(task_id, config.run_root / "checkpoint.json")  # noqa: SLF001
    assert calls == ["POST", "GET"]
    assert task_id in executor._gateway_attempts


def test_grace_residue_bound_admits_multi_row_residue_and_refuses_pathological():
    from devtools.benchmarks.cybergym.cybergym_wire import _abandoned_cost_residue_usd

    frame = _completed_abandoned_residue_frame("cybergym-grace-bound")
    # A provider error loop leaves many abandoned rows; their summed known
    # residue stays deliverable while it is small against the task envelope.
    frame["unresolved_upper_bound_usd"] = 1.60
    frame["non_final_rows"] = 46
    assert _abandoned_cost_residue_usd(frame) == pytest.approx(1.60)
    # A pathological residue (a large fraction of the task's own cost) is not.
    frame["unresolved_upper_bound_usd"] = 6.00
    assert _abandoned_cost_residue_usd(frame) is None


def test_isolate_disk_terminal_record_grace_accepts_abandoned_residue(tmp_path):
    isolate_root = tmp_path / "isolate-data"
    config = _config(tmp_path, provider_probe=False, isolate_data_root=isolate_root)
    executor = CyberGymExecutor(config)
    task_id = "cybergym-disk-grace"
    frame = _completed_abandoned_residue_frame(task_id)
    target = isolate_root / "task_results" / f"{task_id}.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(frame), encoding="utf-8")

    accepted = executor._terminal_result_from_isolate_disk(task_id)  # noqa: SLF001
    assert accepted is not None
    assert accepted["cost_grace_acceptance"]["reason"] == "abandoned_unresolved_residue"

    pending = dict(frame)
    pending["reserved_usd"] = 0.5
    target.write_text(json.dumps(pending), encoding="utf-8")
    assert executor._terminal_result_from_isolate_disk(task_id) is None  # noqa: SLF001


def test_run_campaign_grace_accepted_row_discloses_residue(tmp_path):
    """A grace-accepted completed outcome stays completed and carries the residue."""
    task_id = "cybergym-campaign-grace"
    frame = _completed_abandoned_residue_frame(task_id)
    marked = _CostGraceTracker().accept(
        frame,
        now=0.0,
        wall_now=datetime.datetime.now(datetime.timezone.utc).timestamp(),
    )
    assert marked is not None

    def callback(_task, task_dir):
        (task_dir / "final.poc").write_bytes(b"poc")
        digest = hashlib.sha256(b"poc").hexdigest()
        return {
            "status": "completed",
            "observed_effort": "high",
            "trials": [
                {
                    "trial_id": "final",
                    "is_final": True,
                    "poc_hash": digest,
                    "vul_exit_code": 1,
                    "fix_exit_code": 0,
                }
            ],
            "runtime_result": marked,
        }

    rows = run_campaign(
        ["arvo:1"],
        run_root=tmp_path / "grace-row",
        executor=callback,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )
    assert rows[0]["status"] == "completed"
    assert rows[0]["official_success"] is True
    # Honest recording: the residue rides the row, cost is not called final.
    assert rows[0]["cost_final"] is False
    assert rows[0]["unresolved_upper_bound_usd"] == pytest.approx(0.035398)
    assert rows[0]["cost_grace_acceptance"]["reason"] == "abandoned_unresolved_residue"
    assert rows[0]["cost_usd"] == pytest.approx(0.386527)
    projection = BudgetLedger(tmp_path / "grace-row" / "claims.jsonl", cap_usd=2).projection()
    assert projection.settled_usd == pytest.approx(0.386527)
    assert projection.unresolved_upper_bound_usd == 0
    assert projection.projected_usd == pytest.approx(0.386527)


def test_grace_marker_on_failure_outcome_does_not_break_row_build(tmp_path):
    """A grace marker riding a failed delivery outcome must not crash the row."""
    from devtools.benchmarks.cybergym.cybergym_adapter import TaskSpec, finalize_outcome_row

    frame = _completed_abandoned_residue_frame("cybergym-grace-failure")
    marked = _CostGraceTracker().accept(
        frame,
        now=0.0,
        wall_now=datetime.datetime.now(datetime.timezone.utc).timestamp(),
    )
    assert marked is not None
    outcome = {
        "status": "infra_failed",
        "lifecycle": "post_gateway_evaluation_failed",
        "runtime_result": marked,
        "cost_grace_acceptance": marked["cost_grace_acceptance"],
    }
    row = finalize_outcome_row(
        tmp_path,
        TaskSpec("arvo:9", "arvo"),
        tmp_path,
        outcome,
        attempt_id="a1",
        contract=None,
    )
    assert row["status"] == "infra_failed"
    assert row["cost_final"] is False
    assert row["unresolved_upper_bound_usd"] == pytest.approx(0.035398)
