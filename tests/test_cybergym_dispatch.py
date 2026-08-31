"""Contract tests for the dead-gateway dispatch circuit breaker.

Run 3 (2026-08-31) lost its isolate server to an external SIGTERM and the
launcher kept dispatching into the dead gateway, burning 234 tasks into
``ExecutorFailure: HTTP GET transport failed`` rows.  These tests pin the
breaker: N consecutive transport-class failures stop admission, in-flight
tasks settle, and the campaign fails fast with a typed outcome.
"""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request

import pytest

from devtools.benchmarks.cybergym.cybergym_adapter import (
    BudgetLedger,
    GatewayCircuitOpen,
    run_campaign,
    safe_task_path,
)
from devtools.benchmarks.cybergym.cybergym_dispatch import (
    GATEWAY_CIRCUIT_BREAKER_THRESHOLD,
    GATEWAY_TRANSPORT_INFRA_REASON,
    is_gateway_transport_row,
)
from devtools.benchmarks.cybergym.cybergym_protocol import CyberGymError
from devtools.benchmarks.cybergym.cybergym_wire import (
    ExecutorFailure,
    GatewayTransportError,
    HttpStatusError,
    urllib_json,
)


def _completed(_task, task_dir):
    marker = task_dir / "final.poc"
    marker.write_bytes(b"poc")
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
        "cost_usd": 0.1,
        "cost_final": True,
    }


def _transport_failure(_task, _task_dir):
    raise GatewayTransportError("HTTP POST transport failed")


def _result_index(root):
    path = root / "result_index.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_transport_failure_is_a_typed_executor_failure():
    assert issubclass(GatewayTransportError, ExecutorFailure)
    assert not issubclass(HttpStatusError, GatewayTransportError)
    # The dispatch engine cannot import the wire class (import cycle), so the
    # row classification key is the class name; pin the coupling.
    assert GATEWAY_TRANSPORT_INFRA_REASON == GatewayTransportError.__name__
    assert is_gateway_transport_row(
        {"status": "infra_failed", "infra_reason": "GatewayTransportError"}
    )
    assert not is_gateway_transport_row(
        {"status": "infra_failed", "infra_reason": "ExecutorFailure"}
    )
    assert not is_gateway_transport_row(
        {"status": "infra_failed", "infra_reason": "HttpStatusError"}
    )
    assert not is_gateway_transport_row(
        {"status": "completed", "infra_reason": "GatewayTransportError"}
    )
    assert not is_gateway_transport_row({"status": "failed"})


def test_urllib_transport_failure_raises_typed_gateway_error(monkeypatch):
    def refused(_request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refused)
    with pytest.raises(GatewayTransportError, match="transport failed"):
        urllib_json("GET", "http://127.0.0.1:9/api/tasks")

    def http_error(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 503, "unavailable", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", http_error)
    # A status answer proves the gateway is alive: not circuit-class.
    with pytest.raises(HttpStatusError):
        urllib_json("GET", "http://127.0.0.1:9/api/tasks")


def test_consecutive_transport_failures_open_circuit_and_skip_remaining(tmp_path):
    root = tmp_path / "circuit"
    tasks = [f"arvo:{index}" for index in range(1, 7)]
    with pytest.raises(GatewayCircuitOpen) as excinfo:
        run_campaign(
            tasks,
            run_root=root,
            executor=_transport_failure,
            estimated_cost_usd=1,
            budget_cap_usd=10,
        )

    exc = excinfo.value
    assert isinstance(exc, CyberGymError)
    assert exc.threshold == GATEWAY_CIRCUIT_BREAKER_THRESHOLD
    assert [row["task_id"] for row in exc.rows] == tasks[:3]
    assert all(row["status"] == "infra_failed" for row in exc.rows)
    assert all(row["infra_reason"] == "GatewayTransportError" for row in exc.rows)
    assert exc.remaining_task_ids == tasks[3:]
    assert exc.as_dict() == {
        "outcome": "gateway_unreachable",
        "consecutive_transport_failures": GATEWAY_CIRCUIT_BREAKER_THRESHOLD,
        "dispatched_rows": 3,
        "remaining_task_ids": tasks[3:],
    }
    # Undispatched tasks are not burned into infra rows and leave no trace.
    assert [row["task_id"] for row in _result_index(root)] == tasks[:3]
    for task_id in tasks[3:]:
        assert not safe_task_path(root, task_id).exists()
    projection = BudgetLedger(root / "claims.jsonl", cap_usd=10).projection()
    assert projection.reserved_usd == 0
    assert projection.unresolved_upper_bound_usd == 0
    assert projection.can_dispatch is True


def test_success_resets_the_consecutive_transport_streak(tmp_path):
    outcomes = iter(["transport", "transport", "ok", "transport", "transport"])

    def mixed(task, task_dir):
        if next(outcomes) == "ok":
            return _completed(task, task_dir)
        return _transport_failure(task, task_dir)

    rows = run_campaign(
        ["arvo:1", "arvo:2", "arvo:3", "arvo:4", "arvo:5"],
        run_root=tmp_path / "reset",
        executor=mixed,
        estimated_cost_usd=1,
        budget_cap_usd=10,
        gateway_circuit_threshold=3,
    )

    assert len(rows) == 5
    assert [row["status"] for row in rows] == [
        "infra_failed",
        "infra_failed",
        "completed",
        "infra_failed",
        "infra_failed",
    ]


def test_streak_trips_again_after_a_reset(tmp_path):
    outcomes = iter(["transport", "transport", "ok", "transport", "transport", "transport"])

    def mixed(task, task_dir):
        if next(outcomes) == "ok":
            return _completed(task, task_dir)
        return _transport_failure(task, task_dir)

    with pytest.raises(GatewayCircuitOpen) as excinfo:
        run_campaign(
            ["arvo:1", "arvo:2", "arvo:3", "arvo:4", "arvo:5", "arvo:6", "arvo:7"],
            run_root=tmp_path / "retrip",
            executor=mixed,
            estimated_cost_usd=1,
            budget_cap_usd=10,
            gateway_circuit_threshold=3,
        )

    exc = excinfo.value
    assert [row["task_id"] for row in exc.rows] == [f"arvo:{index}" for index in range(1, 7)]
    assert exc.remaining_task_ids == ["arvo:7"]


def test_non_transport_infra_failures_never_trip_the_breaker(tmp_path):
    def per_task_infra(_task, _task_dir):
        raise ExecutorFailure("generation failed")

    rows = run_campaign(
        ["arvo:1", "arvo:2", "arvo:3", "arvo:4", "arvo:5"],
        run_root=tmp_path / "infra",
        executor=per_task_infra,
        estimated_cost_usd=1,
        budget_cap_usd=10,
    )

    assert [row["status"] for row in rows] == ["infra_failed"] * 5
    assert all(row["infra_reason"] == "ExecutorFailure" for row in rows)


def test_in_flight_rows_land_and_breaker_latches_in_parallel(tmp_path):
    release = threading.Event()
    called: list[str] = []
    called_lock = threading.Lock()

    def callback(task, task_dir):
        with called_lock:
            called.append(task.task_id)
        if task.task_id == "arvo:1":
            raise GatewayTransportError("HTTP GET transport failed")
        if task.task_id == "arvo:2":
            # In flight when the breaker opens; released by the failing lane
            # and must still land its own completed row.
            release.wait(timeout=30)
            return _completed(task, task_dir)
        raise AssertionError(f"{task.task_id} must never be dispatched")

    def releasing(task, task_dir):
        try:
            return callback(task, task_dir)
        finally:
            if task.task_id == "arvo:1":
                release.set()

    with pytest.raises(GatewayCircuitOpen) as excinfo:
        run_campaign(
            ["arvo:1", "arvo:2", "arvo:3", "arvo:4"],
            run_root=tmp_path / "parallel",
            executor=releasing,
            estimated_cost_usd=1,
            budget_cap_usd=10,
            max_workers=2,
            gateway_circuit_threshold=1,
        )

    exc = excinfo.value
    assert [row["task_id"] for row in exc.rows] == ["arvo:1", "arvo:2"]
    assert exc.rows[0]["infra_reason"] == "GatewayTransportError"
    assert exc.rows[1]["status"] == "completed"
    assert exc.remaining_task_ids == ["arvo:3", "arvo:4"]
    assert sorted(called) == ["arvo:1", "arvo:2"]
    assert [row["task_id"] for row in _result_index(tmp_path / "parallel")] == ["arvo:1", "arvo:2"]


def test_healthy_gateway_parallel_campaign_is_unchanged(tmp_path):
    rows = run_campaign(
        ["arvo:1", "arvo:2", "arvo:3", "arvo:4", "arvo:5"],
        run_root=tmp_path / "healthy",
        executor=_completed,
        estimated_cost_usd=1,
        budget_cap_usd=10,
        max_workers=3,
    )

    assert [row["task_id"] for row in rows] == [f"arvo:{index}" for index in range(1, 6)]
    assert all(row["status"] == "completed" for row in rows)


def test_gateway_circuit_threshold_is_validated(tmp_path):
    for invalid in (0, -1, True, 2.5, "3"):
        with pytest.raises(ValueError, match="gateway_circuit_threshold"):
            run_campaign(
                ["arvo:1"],
                run_root=tmp_path / "invalid",
                executor=_completed,
                estimated_cost_usd=1,
                budget_cap_usd=2,
                gateway_circuit_threshold=invalid,
            )
