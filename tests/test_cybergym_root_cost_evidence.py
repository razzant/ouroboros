"""Canonical terminal accounting reaches live and saved CyberGym consumers."""

from __future__ import annotations

import copy
import datetime
import hashlib
import time
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from devtools.benchmarks.cybergym import cybergym_custody, cybergym_lifecycle, cybergym_wire
from devtools.benchmarks.cybergym.cybergym_adapter import (
    BudgetLedger, _terminal_gateway_accounting, run_campaign,
)
from devtools.benchmarks.cybergym.cybergym_executor import CyberGymExecutor
from devtools.benchmarks.cybergym.cybergym_wire import (
    _CostGraceTracker, _abandoned_cost_residue_usd, _valid_cost_grace,
)
from ouroboros import usage_accounting as ua
from ouroboros.gateway.tasks import api_task_get
from ouroboros.post_task_checkpoint import set_root_post_task_checkpoint
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_cybergym_executor import _config, dataclasses_replace
from tests.test_cybergym_cost_grace import _completed_abandoned_residue_frame
from tests._store_shared import as_install, read_result


def _frame():
    return {
        "task_id": "cybergym-root", "root_task_id": "cybergym-root",
        "status": "completed", "cost_final": False,
        "accounted_upper_bound_usd": 0.35,
        "accounted_upper_bound_usd_with_children": 1.05,
        "root_phase_checkpoint": {
            "post_task_synthesis": "completed",
            "accounting": {
                "schema": "ouroboros.root_cost_snapshot.v1", "scope": "root_tree",
                "root_task_id": "cybergym-root", "cost_accounting_status": "available",
                "accounted_upper_bound_usd": 1.05, "unresolved_upper_bound_usd": 0.05,
                "reserved_usd": 0.0, "non_final_rows": 1,
                "attempt_counts": {"unresolved": 1}, "unknown_unmetered": 0,
                "ledger_integrity_degraded": False,
            },
        },
    }


def _accept(frame):
    tracker = _CostGraceTracker()
    assert tracker.accept(frame, now=0) is None
    return tracker.accept(frame, now=120)


@pytest.fixture
def canonical_terminal(tmp_path, monkeypatch):
    data = tmp_path / "isolated" / "data"
    # The server's roots belong only to the producer. The host adapter must
    # retain its own environment so its live-root overlap guard stays real.
    with monkeypatch.context() as producer_env:
        for key, value in {
            "OUROBOROS_APP_ROOT": data.parent,
            "OUROBOROS_REPO_DIR": tmp_path / "repo",
            "OUROBOROS_DATA_DIR": data,
            "OUROBOROS_SETTINGS_PATH": data / "settings.json",
        }.items():
            producer_env.setenv(key, str(value))
        root_id = "cybergym-root"
        task = {"id": root_id, "root_task_id": root_id, "budget_drive_root": str(data)}
        write_task_result(
            data, root_id, "completed", root_task_id=root_id,
            root_phase_checkpoint={"post_task_synthesis": "running"},
            artifact_status="ready", result="done",
        )
        for task_id, amount, unresolved in ((root_id, 0.3, False), ("child", 0.7, False), (root_id, 0.05, True)):
            reservation = ua.reserve_attempt(ua.AttemptRequest(
                model="openai/gpt-5.2", provider="openai", reservation_usd=amount,
                drive_root=data, task_id=task_id, root_task_id=root_id,
                global_limit_usd=10, root_limit_usd=10,
            ))
            ua.mark_dispatched(reservation)
            if unresolved:
                ua.mark_unresolved(reservation, "provider_outcome_unknown")
            else:
                ua.settle_attempt(reservation, {"prompt_tokens": 10, "completion_tokens": 5}, cost_usd=amount, cost_final=True)
        stored = set_root_post_task_checkpoint(SimpleNamespace(drive_root=data), task, "completed")
    assert stored is not None
    return data, stored


def test_real_ledger_checkpoint_gateway_and_disk_reach_grace(canonical_terminal, tmp_path, monkeypatch):
    data, stored = canonical_terminal
    assert "cost_estimated" not in stored
    # Feed the actual producer result into the actual gate before any mock.
    assert _abandoned_cost_residue_usd(stored) == pytest.approx(0.05)
    assert stored["root_phase_checkpoint"]["accounting"]["attempt_counts"]["unresolved"] == 1
    app = Starlette(routes=[Route("/api/tasks/{task_id}", api_task_get)])
    app.state.drive_root = data
    with as_install(data), TestClient(app) as client:  # the producer's own server answers for its root
        response = client.get("/api/tasks/cybergym-root")
    assert response.status_code == 200
    live = response.json()
    assert live["root_phase_checkpoint"]["accounting"] == stored["root_phase_checkpoint"]["accounting"]
    assert _abandoned_cost_residue_usd(live) == pytest.approx(0.05)
    with as_install(data):
        raw = read_result(data, "cybergym-root")
    assert _abandoned_cost_residue_usd(raw) == pytest.approx(0.05)
    root = tmp_path / "executor"
    root.mkdir()
    config = _config(root, isolate_data_root=data)
    epoch = datetime.datetime.fromisoformat(stored["updated_at"]).timestamp() + 121
    clock = SimpleNamespace(time=lambda: epoch, monotonic=time.monotonic,
                            strftime=time.strftime, gmtime=time.gmtime)
    for module in (cybergym_wire, cybergym_lifecycle, cybergym_custody):
        monkeypatch.setattr(module, "time", clock)
    calls = []

    def http(method, _url, **_kwargs):
        calls.append(method)
        return {"task_id": "cybergym-root", "status": "scheduled"} if method == "POST" else live

    executor = CyberGymExecutor(dataclasses_replace(config, http_runner=http))
    delivered = executor._gateway_wait({"task_id": "cybergym-root"}, config.run_root / "checkpoint.json")
    assert calls == ["POST", "GET"]
    assert delivered["cost_final"] is False
    assert _valid_cost_grace(delivered)["accounting_schema"] == "ouroboros.root_cost_snapshot.v1"
    disk = executor._terminal_result_from_isolate_disk("cybergym-root")
    assert _valid_cost_grace(disk) is not None
    with as_install(data):
        assert load_task_result(data, "cybergym-root")["root_phase_checkpoint"] == stored["root_phase_checkpoint"]
    responses = iter(({"status_code": 404, "body": {}}, live))
    custody = CyberGymExecutor(dataclasses_replace(config, http_runner=lambda *_args, **_kwargs: next(responses)))
    custody._gateway_attempts["cybergym-root"] = {"status": "submitted"}
    recovered = custody._cancel_gateway_task("cybergym-root", config.run_root / "cancel.json")
    assert _valid_cost_grace(recovered) is not None
    assert "cybergym-root" not in custody._gateway_attempts


@pytest.mark.parametrize("patch", [
    {"schema": "other"}, {"scope": "own"}, {"root_task_id": "other"},
    {"cost_accounting_status": "unavailable"}, {"ledger_integrity_degraded": True},
    {"unknown_unmetered": 1}, {"reserved_usd": 0.01},
    {"non_final_rows": 2}, {"non_final_rows": True}, {"non_final_rows": -1},
    {"attempt_counts": {"unresolved": 0}}, {"attempt_counts": {"unresolved": 1.0}},
    {"accounted_upper_bound_usd": None}, {"accounted_upper_bound_usd": float("nan")},
    {"accounted_upper_bound_usd": True}, {"unresolved_upper_bound_usd": float("inf")},
    {"unresolved_upper_bound_usd": 2}, {"unresolved_upper_bound_usd": -1},
])
def test_explicit_invalid_or_open_new_proof_never_falls_back(patch):
    frame = _frame()
    # A complete legacy shape would otherwise admit this frame.
    frame.update(cost_estimated=False, cost_accounting_status="available",
                 ledger_integrity_degraded=False, unknown_unmetered=0,
                 reserved_usd=0, unresolved_upper_bound_usd=0.05)
    frame["root_phase_checkpoint"]["accounting"].update(patch)
    assert _accept(frame) is None


@pytest.mark.parametrize("value", [None, True, "false", 0])
def test_explicit_legacy_estimate_is_not_repaired_by_new_proof(value):
    frame = _frame()
    frame["cost_estimated"] = value
    assert _accept(frame) is None
    projected = _terminal_gateway_accounting(frame)
    assert projected["cost_estimated"] is True
    assert projected["cost_final"] is False


@pytest.mark.parametrize("phase", ["pending_once", "running", "", None])
def test_open_phase_cannot_release_the_new_accounting_path(phase):
    frame = _frame()
    frame["root_phase_checkpoint"]["post_task_synthesis"] = phase
    assert _accept(frame) is None


@pytest.mark.parametrize("field,value", [
    ("_is_direct_chat", True), ("artifact_status", "pending"),
    ("artifact_status", "finalizing"), ("reserved_usd", 0.1),
    ("unknown_unmetered", 1), ("ledger_integrity_degraded", True),
])
def test_conflicting_live_evidence_cannot_release_new_path(field, value):
    frame = _frame()
    frame[field] = value
    assert _accept(frame) is None


def test_scope_and_replica_agreement_without_summing_own_cost():
    frame = _frame()
    assert _terminal_gateway_accounting(frame)["cost_usd"] == 1.05
    assert _accept(frame) is not None
    frame["task_result"] = copy.deepcopy(frame)
    assert _accept(frame) is not None
    frame["task_result"]["root_phase_checkpoint"]["accounting"]["non_final_rows"] = 2
    assert _accept(frame) is None


@pytest.mark.parametrize("total", [0.9, None, True, float("nan")])
def test_conflicting_root_display_amount_cannot_authorize_grace(total):
    frame = _frame()
    frame["cost_breakdown"] = {
        "authority": "physical_attempt_ledger", "non_final_rows": 1,
        "accounted_upper_bound_usd": total,
    }
    assert _accept(frame) is None


def test_cost_grace_revalidates_identity_and_resets_after_new_open_work():
    frame = _frame()
    tracker = _CostGraceTracker()
    assert tracker.accept(frame, now=0) is None
    opened = copy.deepcopy(frame)
    opened["root_phase_checkpoint"]["accounting"]["non_final_rows"] = 2
    assert tracker.accept(opened, now=119) is None
    assert tracker.accept(frame, now=120) is None
    accepted = tracker.accept(frame, now=240)
    assert accepted is not None
    accepted["cost_grace_acceptance"]["root_task_id"] = "foreign"
    assert _valid_cost_grace(accepted) is None


def test_canonical_grace_preserves_whole_liability_and_absent_estimate(tmp_path):
    accepted = _accept(_frame())
    assert accepted is not None

    def callback(_task, task_dir):
        (task_dir / "final.poc").write_bytes(b"poc")
        return {
            "status": "completed", "observed_effort": "high", "runtime_result": accepted,
            "trials": [{"trial_id": "final", "is_final": True,
                        "poc_hash": hashlib.sha256(b"poc").hexdigest(),
                        "vul_exit_code": 1, "fix_exit_code": 0}],
        }

    root = tmp_path / "campaign"
    rows = run_campaign(["arvo:1"], run_root=root, executor=callback,
                        estimated_cost_usd=5, budget_cap_usd=10)
    assert rows[0]["status"] == "completed"
    assert rows[0]["official_success"] is True
    assert rows[0]["cost_final"] is False
    assert "cost_estimated" not in rows[0]
    projection = BudgetLedger(root / "claims.jsonl", cap_usd=10).projection()
    assert projection.settled_usd == 0
    assert projection.unresolved_upper_bound_usd == pytest.approx(1.05)
    assert projection.projected_usd == pytest.approx(1.05)


def test_old_sparse_record_is_not_upgraded():
    frame = _frame()
    del frame["root_phase_checkpoint"]["accounting"]
    assert _accept(frame) is None


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("field", ["waited_sec", "unresolved_upper_bound_usd"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 10**400], ids=["nan", "infinity", "huge-integer"])
def test_cost_grace_rejects_invalid_marker_numbers(legacy, field, value):
    frame = _completed_abandoned_residue_frame("legacy") if legacy else _frame()
    marked = _accept(frame)
    assert _valid_cost_grace(marked) is not None
    marked["cost_grace_acceptance"][field] = value
    assert _valid_cost_grace(marked) is None


def test_canonical_marker_cannot_fall_back_after_its_snapshot_is_removed():
    frame = _accept(_frame())
    del frame["root_phase_checkpoint"]["accounting"]
    frame.update(cost_estimated=False, cost_accounting_status="available",
                 ledger_integrity_degraded=False, unknown_unmetered=0,
                 reserved_usd=0, unresolved_upper_bound_usd=0.05)
    assert _abandoned_cost_residue_usd(frame) == pytest.approx(0.05)
    assert _valid_cost_grace(frame) is None


@pytest.mark.parametrize("marker", [False, None, "absent", True])
def test_closed_snapshot_preserves_explicit_finality(marker):
    frame = _frame()
    snapshot = frame["root_phase_checkpoint"]["accounting"]
    snapshot.update(non_final_rows=0, attempt_counts={"unresolved": 0},
                    unresolved_upper_bound_usd=0)
    if marker == "absent":
        frame.pop("cost_final")
    else:
        frame["cost_final"] = marker
    projected = _terminal_gateway_accounting(frame)
    assert projected["cost_usd"] == pytest.approx(1.05)
    assert projected.get("cost_final") is (True if marker is True else False if marker != "absent" else None)
    if marker is True:
        assert projected["cost_estimated"] is False
    else:
        assert "cost_estimated" not in projected


def test_closed_snapshot_cannot_override_explicit_partiality():
    frame = _frame()
    frame["cost_final"] = True
    frame["cost_with_children_partial"] = True
    frame["root_phase_checkpoint"]["accounting"].update(
        non_final_rows=0, attempt_counts={"unresolved": 0}, unresolved_upper_bound_usd=0,
    )
    assert _terminal_gateway_accounting(frame)["cost_final"] is False


def test_delivery_does_not_invent_an_estimate_flag(tmp_path, monkeypatch):
    frame = _accept(_frame())
    frame.update(prompt_tokens=10, completion_tokens=1,
                 outcome_axes={"execution": {"status": "ok"}})
    config = _config(tmp_path)
    monkeypatch.setattr(
        "devtools.benchmarks.cybergym.cybergym_lifecycle._served_telemetry",
        lambda *_args, **_kwargs: {
            "observed_model": config.model, "observed_provider": "test",
            "observed_effort": "high",
        },
    )
    from devtools.benchmarks.cybergym.cybergym_adapter import TaskSpec

    result = CyberGymExecutor(config)._deliver_gateway_result(
        TaskSpec("arvo:1", "arvo"), config.run_root, config.run_root,
        "container", "agent", frame, checkpoint=config.run_root / "checkpoint.json",
        cleanup_ref=config.run_root / "cleanup.json", alias_ref=config.run_root / "alias.json",
        attestation_ref="", sidecar_attestation={}, terminal_evidence={},
    )
    assert result["lifecycle"] == "final_poc_missing_after_fair_completion"
    assert result["status"] == "failed"
    assert result["cost_final"] is False
    assert "cost_estimated" not in result
    assert result["cost_usd"] == pytest.approx(1.05)
