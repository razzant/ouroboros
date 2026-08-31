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
    DEFAULT_LEVEL,
    OFFICIAL_MODEL,
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
    second_started = threading.Event()
    called: list[str] = []
    called_lock = threading.Lock()

    def callback(task, task_dir):
        with called_lock:
            called.append(task.task_id)
        if task.task_id == "arvo:1":
            assert second_started.wait(timeout=30)
            raise GatewayTransportError("HTTP GET transport failed")
        if task.task_id == "arvo:2":
            # In flight when the breaker opens; released by the failing lane
            # and must still land its own completed row.
            second_started.set()
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


def test_launcher_finalizes_gateway_unreachable_when_circuit_opens(monkeypatch, tmp_path):
    """Pin the launcher branch that finalizes a circuit-open campaign.

    ``run_campaign`` raising ``GatewayCircuitOpen`` must still produce a
    finalized manifest: the rows that landed stay accounted, the undispatched
    tasks are named under ``extra.gateway_circuit.remaining_task_ids``, and
    the run records outcome ``gateway_unreachable`` with exit code 2 instead
    of a generic failure.
    """
    from types import SimpleNamespace

    import devtools.benchmarks.cybergym.run_cybergym as launcher

    repo = tmp_path / "seed"
    source = tmp_path / "cybergym-source"
    data = tmp_path / "cybergym-data"
    tasks = tmp_path / "tasks.json"
    mask_map = tmp_path / "mask-map.json"
    settings_template = tmp_path / "settings.json"
    server_root = tmp_path / "server-root"
    binary_dir = server_root / "bin"
    for directory in (repo, source, data, server_root, binary_dir):
        directory.mkdir(parents=True)
    tasks.write_text("{}", encoding="utf-8")
    mask_map.write_text("{}", encoding="utf-8")
    settings_template.write_text("{}", encoding="utf-8")
    applied = tmp_path / "run" / "settings_applied.json"
    expected_commit = "a" * 40
    task_ids = ["arvo:1", "arvo:2", "arvo:3", "arvo:4"]
    events: list[str] = []

    class FakeServer:
        base_url = "http://127.0.0.1:18181"
        attestation = {"repo_head": expected_commit}

        def close(self):
            events.append("server.close")

    class FakeExecutor:
        def prepare(self):
            events.append("executor.prepare")
            return {"prepared": True}

        def close(self):
            events.append("executor.close")
            return {"ok": True, "status": "closed"}

    def fake_prepare(_template, _out_root, _args):
        applied.parent.mkdir(parents=True, exist_ok=True)
        applied.write_text("{}", encoding="utf-8")
        return applied, {
            "model": OFFICIAL_MODEL,
            "model_slots": {"OUROBOROS_MODEL": OFFICIAL_MODEL},
            "provider_credentials": {},
        }

    args = SimpleNamespace(
        repo_dir=repo,
        source_root=source,
        data_root=data,
        tasks_file=tasks,
        task_id=list(task_ids),
        server="http://cybergym-internal:8666",
        ouroboros_url="",
        docker_host="unix:///run/user/1006/docker.sock",
        server_image="cybergym-server",
        server_image_digest="sha256:" + "b" * 64,
        workspace_image="ouroboros-workspace",
        workspace_image_digest="sha256:" + "c" * 64,
        server_root=server_root,
        binary_dir=binary_dir,
        cybergym_api_key_env="CYBERGYM_API_KEY",
        mask_map=mask_map,
        difficulty=DEFAULT_LEVEL,
        model=OFFICIAL_MODEL,
        settings_path=settings_template,
        out_dir=tmp_path / "run",
        run_id="",
        budget_usd=2.0,
        per_task_cost_usd=1.0,
        per_task_estimate_usd=1.0,
        timeout_sec=1,
        workers=1,
        executor="",
        dry_run=False,
        allow_dirty_seed=False,
        expected_source_sha256="",
        expected_data_sha256="a" * 64,
        expected_binary_sha256="b" * 64,
        expected_tasks_sha256="",
        expected_mask_sha256="mask-digest",
        cybergym_python="python3",
        provider_only=["provider-a"],
        provider_order=["provider-a"],
    )
    monkeypatch.setattr(launcher, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(launcher, "pre_admission_report", lambda **_kwargs: {"ok": True, "reasons": []})
    monkeypatch.setattr(
        launcher,
        "admit_benchmark_run",
        lambda _path, **_kwargs: {
            "source": {"head": expected_commit},
            "extra": dict(_kwargs.get("extra") or {}),
            "harness": {},
            "output_paths": {},
        },
    )
    monkeypatch.setattr(launcher, "verify_source_checkout", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(launcher, "source_tree_digest", lambda *_args, **_kwargs: "source-digest")
    monkeypatch.setattr(
        launcher,
        "verify_mask_map",
        lambda *_args, **_kwargs: {"sha256": "mask-digest"},
    )
    monkeypatch.setattr(
        launcher,
        "load_task_catalog",
        lambda *_args, **_kwargs: {"task_ids": list(task_ids)},
    )
    monkeypatch.setattr(launcher, "_prepare_applied_settings", fake_prepare)
    monkeypatch.setattr(
        launcher,
        "_start_isolated_ouroboros_server",
        lambda *_args, **_kwargs: FakeServer(),
    )
    monkeypatch.setattr(launcher, "_build_default_executor", lambda *_args, **_kwargs: FakeExecutor())
    monkeypatch.setattr(
        launcher,
        "_validate_paid_observations",
        lambda *_args, **_kwargs: (
            {"status": "passed", "model": OFFICIAL_MODEL},
            {"sha256": "a" * 64},
            {"sha256": "b" * 64},
            0.0,
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_record_provider_probe_cost",
        lambda *_args, **_kwargs: {"attempt_id": "campaign-overhead-provider_probe"},
    )

    landed_rows = [
        {
            "task_id": "arvo:1",
            "status": "infra_failed",
            "infra_reason": "GatewayTransportError",
            "final_submission_success": False,
        },
        {
            "task_id": "arvo:2",
            "status": "completed",
            "final_submission_success": True,
        },
    ]
    dispatched: list[str] = []

    def circuit_open(specs, **_kwargs):
        dispatched.extend(spec.task_id for spec in specs)
        raise GatewayCircuitOpen(rows=landed_rows, threshold=3, remaining=task_ids[2:])

    monkeypatch.setattr(launcher, "run_campaign", circuit_open)
    rc = launcher.main()

    assert rc == 2
    assert dispatched == task_ids
    assert events == ["executor.prepare", "executor.close", "server.close"]
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["requested_task_ids"] == task_ids
    extra = manifest["extra"]
    assert extra["outcome"] == "gateway_unreachable"
    assert extra["exit_code"] == 2
    assert extra["gateway_circuit"] == {
        "outcome": "gateway_unreachable",
        "consecutive_transport_failures": 3,
        "dispatched_rows": 2,
        "remaining_task_ids": ["arvo:3", "arvo:4"],
    }
    # Rows that landed before the breaker opened stay accounted, not dropped.
    assert extra["rows_written"] == 2
    assert extra["completed_count"] == 1
    assert extra["infra_count"] == 1
    assert extra["close_skipped"] is False
    assert extra["server_cleanup"]["status"] == "closed"
