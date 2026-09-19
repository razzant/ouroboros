"""Contract tests for the dead-gateway dispatch circuit breaker.

Run 3 (2026-08-31) lost its isolate server to an external SIGTERM and the
launcher kept dispatching into the dead gateway, burning 234 tasks into
``ExecutorFailure: HTTP GET transport failed`` rows.  These tests pin the
breaker: N consecutive transport-class failures stop admission, in-flight
tasks settle, and the campaign fails fast with a typed outcome.

The second half pins the budget gate (run 20260907T233516Z flushed 1145
never-dispatched tasks into ``infra_failed`` rows on claim refusals): a
claim-time ``BudgetRefused`` now pauses admission, in-flight settlements are
probed for freed headroom, and only a drained pool with the cap still
refusing ends the campaign with ``BudgetCapReached`` — undispatched tasks
stay row-free for a later resume campaign.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
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
        "cost_estimated": False,
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
    # Transport-dead attempts have no terminal frame, so each claim settles
    # terminally at its reservation instead of leaking an eternal unresolved
    # liability (run 20260907T233516Z tripped the cap on such phantom spend).
    assert projection.settled_usd == 3
    assert projection.unresolved_upper_bound_usd == 0
    assert projection.projected_usd == 3
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
        if task.task_id == "arvo:3":
            # A just-freed lane may accept this task before the failure
            # future is observed by the dispatcher.  It is genuine
            # in-flight work and must settle; the breaker must prevent
            # any later admission after it latches.
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
    assert [row["task_id"] for row in exc.rows][:2] == ["arvo:1", "arvo:2"]
    assert exc.rows[0]["infra_reason"] == "GatewayTransportError"
    assert exc.rows[1]["status"] == "completed"
    assert exc.remaining_task_ids == ["arvo:4"] if "arvo:3" in called else ["arvo:3", "arvo:4"]
    assert set(called).issubset({"arvo:1", "arvo:2", "arvo:3"})
    assert {row["task_id"] for row in _result_index(tmp_path / "parallel")} == set(called)


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


def test_buffered_completions_do_not_hold_admission_lanes(tmp_path):
    """r8 (2026-09-04): 9 finished lanes sat idle behind one long position 0.

    A slow early task must not idle the lanes whose rows are merely waiting
    to be recorded behind it; recording (and result_index) still lands in
    source order, while durable completion handling must not wait for it.
    """

    from devtools.benchmarks.cybergym.cybergym_dispatch import run_dispatched

    first_may_finish = threading.Event()
    first_finished = threading.Event()
    head_still_running_at: list[str] = []
    lock = threading.Lock()

    def run_one(task):
        if task.task_id == "arvo:1":
            assert first_may_finish.wait(timeout=30)
            first_finished.set()
        else:
            with lock:
                if not first_finished.is_set():
                    head_still_running_at.append(task.task_id)
            if task.task_id == "arvo:4":
                # Two refills of the second lane happened behind a running
                # position 0; now let the head of the line finish.
                first_may_finish.set()
        return {"task_id": task.task_id, "status": "completed"}

    recorded: list[str] = []
    rows = run_dispatched(
        [_Task(f"arvo:{index}") for index in range(1, 6)],
        run_one,
        max_workers=2,
        on_row=lambda row: recorded.append(row["task_id"]),
    )

    # arvo:3 and arvo:4 were admitted into the lane arvo:2 freed while arvo:1
    # (position 0) was still running.
    assert head_still_running_at[:3] == ["arvo:2", "arvo:3", "arvo:4"]
    # Completed rows are handed to the durable callback before the blocked
    # first task settles.  This releases their budget claims before either
    # lane admits another task.
    assert recorded.index("arvo:2") < recorded.index("arvo:1")
    assert recorded.index("arvo:3") < recorded.index("arvo:1")
    # Reporting remains deterministic even though delivery was completion
    # ordered.
    assert [row["task_id"] for row in rows] == [f"arvo:{index}" for index in range(1, 6)]


def test_completion_settlement_releases_budget_before_lane_refill(tmp_path):
    """A slow first task cannot turn a finished sibling into a held reserve."""

    release_first = threading.Event()
    third_started = threading.Event()

    def callback(task, task_dir):
        if task.task_id == "arvo:1":
            assert release_first.wait(timeout=10)
        if task.task_id == "arvo:3":
            third_started.set()
            release_first.set()
        return _completed(task, task_dir)

    root = tmp_path / "settlement-before-refill"
    rows = run_campaign(
        ["arvo:1", "arvo:2", "arvo:3"],
        run_root=root,
        executor=callback,
        estimated_cost_usd=1,
        budget_cap_usd=2.5,
        max_workers=2,
    )

    assert third_started.is_set()
    assert [row["status"] for row in rows] == ["completed", "completed", "completed"]
    assert _result_index(root)[0]["task_id"] == "arvo:2"
    assert BudgetLedger(root / "claims.jsonl", cap_usd=2.5).projection().reserved_usd == 0


def test_breaker_sees_transport_failures_in_completion_order(tmp_path):
    """A transport failure at a later position pauses admission at once, even
    while an earlier position is still running."""

    from devtools.benchmarks.cybergym.cybergym_dispatch import run_dispatched

    first_may_finish = threading.Event()
    started: list[str] = []
    lock = threading.Lock()

    def run_one(task):
        with lock:
            started.append(task.task_id)
        if task.task_id == "arvo:1":
            assert first_may_finish.wait(timeout=30)
            return {"task_id": task.task_id, "status": "completed"}
        if task.task_id == "arvo:2":
            return _transport_row(task.task_id)
        raise AssertionError(f"{task.task_id} must never be dispatched")

    events: list[dict] = []

    def probe() -> bool:
        first_may_finish.set()
        return False

    # Real clock: the in-flight wait is bounded by the probe schedule.
    with pytest.raises(GatewayCircuitOpen) as excinfo:
        run_dispatched(
            [_Task(f"arvo:{index}") for index in range(1, 5)],
            run_one,
            max_workers=2,
            threshold=1,
            gateway_probe=probe,
            probe_backoff_sec=(0.2,),
            pause_budget_sec=1.0,
            on_event=events.append,
        )

    assert sorted(started) == ["arvo:1", "arvo:2"]
    assert [row["task_id"] for row in excinfo.value.rows] == ["arvo:1", "arvo:2"]
    assert excinfo.value.rows[0]["status"] == "completed"
    assert excinfo.value.remaining_task_ids == ["arvo:3", "arvo:4"]
    assert events[0]["event"] == "gateway_pause"


class _Task:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id


class _PausingClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(float(seconds))
        self.now += float(seconds)


def _transport_row(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "status": "infra_failed",
        "infra_reason": GATEWAY_TRANSPORT_INFRA_REASON,
    }


def test_breaker_pauses_probes_and_resumes_instead_of_abandoning(tmp_path):
    """full1507: three transport rows from a ~100 s stall, not a dead isolate."""

    from devtools.benchmarks.cybergym.cybergym_dispatch import run_dispatched

    clock = _PausingClock()
    probes: list[float] = []
    # Gateway is "stalled" for the first two probes and answers on the third.
    probe_answers = iter([False, False, True])

    def probe() -> bool:
        probes.append(clock.now)
        return next(probe_answers)

    outcomes = iter(["transport", "transport", "transport", "ok", "ok", "ok"])

    def run_one(task):
        if next(outcomes) == "ok":
            return {"task_id": task.task_id, "status": "completed"}
        return _transport_row(task.task_id)

    events: list[dict] = []
    tasks = [_Task(f"arvo:{index}") for index in range(1, 7)]
    for workers in (1, 3):
        outcomes = iter(["transport", "transport", "transport", "ok", "ok", "ok"])
        probe_answers = iter([False, False, True])
        probes.clear()
        events.clear()
        clock.now = 0.0
        clock.slept.clear()
        rows = run_dispatched(
            tasks,
            run_one,
            max_workers=workers,
            threshold=3,
            gateway_probe=probe,
            probe_backoff_sec=(30.0, 60.0, 120.0),
            pause_budget_sec=3600.0,
            on_event=events.append,
            sleep=clock.sleep,
            clock=clock.monotonic,
        )

        assert [row["status"] for row in rows] == ["infra_failed"] * 3 + ["completed"] * 3
        # Backoff schedule: first probe after 30 s, then +60 s, then +120 s.
        assert probes == [30.0, 90.0, 210.0]
        names = [event["event"] for event in events]
        assert names == [
            "gateway_pause",
            "gateway_probe_failed",
            "gateway_probe_failed",
            "gateway_resume",
        ]
        assert events[-1]["paused_sec"] == 210.0
        assert events[-1]["failed_probes"] == 2


def test_breaker_opens_only_after_pause_budget_is_exhausted(tmp_path):
    from devtools.benchmarks.cybergym.cybergym_dispatch import run_dispatched

    clock = _PausingClock()
    probes: list[float] = []

    def dead_probe() -> bool:
        probes.append(clock.now)
        return False

    def run_one(task):
        return _transport_row(task.task_id)

    tasks = [_Task(f"arvo:{index}") for index in range(1, 9)]
    with pytest.raises(GatewayCircuitOpen) as excinfo:
        run_dispatched(
            tasks,
            run_one,
            max_workers=1,
            threshold=3,
            gateway_probe=dead_probe,
            probe_backoff_sec=(30.0, 60.0),
            pause_budget_sec=200.0,
            sleep=clock.sleep,
            clock=clock.monotonic,
        )

    exc = excinfo.value
    assert [row["task_id"] for row in exc.rows] == ["arvo:1", "arvo:2", "arvo:3"]
    assert exc.remaining_task_ids == [f"arvo:{index}" for index in range(4, 9)]
    # 30, 90, 150, 210 >= 200 budget -> open on the fourth failed probe.
    assert probes == [30.0, 90.0, 150.0, 210.0]
    assert exc.as_dict()["pause"]["pauses"][0]["failed_probes"] == 4
    assert exc.as_dict()["pause"]["pauses"][0]["paused_sec"] == 210.0


def test_breaker_without_probe_keeps_the_fail_fast_contract(tmp_path):
    from devtools.benchmarks.cybergym.cybergym_dispatch import run_dispatched

    def run_one(task):
        return _transport_row(task.task_id)

    tasks = [_Task(f"arvo:{index}") for index in range(1, 6)]
    with pytest.raises(GatewayCircuitOpen) as excinfo:
        run_dispatched(
            tasks,
            run_one,
            max_workers=1,
            threshold=3,
            sleep=lambda _s: pytest.fail("no probe: must not sleep"),
        )
    assert "pause" not in excinfo.value.as_dict()
    assert excinfo.value.remaining_task_ids == ["arvo:4", "arvo:5"]


def test_run_campaign_wires_executor_probe_and_records_dispatch_events(tmp_path, monkeypatch):
    from devtools.benchmarks.cybergym import cybergym_dispatch

    captured = {}
    real = cybergym_dispatch.run_dispatched

    def spy(tasks, run_one, **kwargs):
        captured.update(kwargs)
        return real(tasks, run_one, **kwargs)

    monkeypatch.setattr(
        "devtools.benchmarks.cybergym.cybergym_adapter.run_dispatched", spy
    )

    class Owner:
        def probe_gateway_alive(self) -> bool:
            return True

        def run(self, task, task_dir):
            return _completed(task, task_dir)

    owner = Owner()
    rows = run_campaign(
        ["arvo:1", "arvo:2"],
        run_root=tmp_path / "wired",
        executor=owner.run,
        estimated_cost_usd=1,
        budget_cap_usd=10,
    )

    assert [row["status"] for row in rows] == ["completed", "completed"]
    assert captured["gateway_probe"] == owner.probe_gateway_alive
    captured["on_event"]({"event": "gateway_pause", "x": 1})
    logged = (tmp_path / "wired" / "dispatch_events.jsonl").read_text(encoding="utf-8")
    entry = json.loads(logged.strip())
    assert entry["event"] == "gateway_pause"
    assert entry["ts"].endswith("Z")


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


def _launch_to_stop(monkeypatch, tmp_path, stop, *, close_report=None):
    """Drive ``run_cybergym.main`` through a paid dispatch that ends in ``stop``.

    ``stop(task_ids)`` builds the typed stop that the fake ``run_campaign``
    raises after asserting the durable ready-to-dispatch checkpoint.  Returns
    the exit code, the finalized manifest and the resource-close order.
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
            return dict(close_report or {"ok": True, "status": "closed"})

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
    dispatched: list[str] = []

    def stopped(specs, **_kwargs):
        durable = json.loads(
            (tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert durable["requested_task_ids"] == task_ids
        assert durable["requested_count"] == len(task_ids)
        assert durable["extra"]["provider_probe"]["model"] == OFFICIAL_MODEL
        assert "data_root" in durable["extra"]["state_layout"]
        assert durable["extra"]["outcome"] == "running"
        assert durable["extra"]["exit_code"] is None
        assert durable["extra"]["recovery_checkpoint"]["phase"] == "ready_to_dispatch"
        dispatched.extend(spec.task_id for spec in specs)
        raise stop(task_ids)

    monkeypatch.setattr(launcher, "run_campaign", stopped)
    rc = launcher.main()
    assert dispatched == task_ids
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["requested_task_ids"] == task_ids
    return rc, manifest, events


def test_launcher_finalizes_gateway_unreachable_when_circuit_opens(monkeypatch, tmp_path):
    """Pin the launcher branch that finalizes a circuit-open campaign.

    ``run_campaign`` raising ``GatewayCircuitOpen`` must still produce a
    finalized manifest: the rows that landed stay accounted, the undispatched
    tasks are named under ``extra.gateway_circuit.remaining_task_ids``, and
    the run records outcome ``gateway_unreachable`` with exit code 2 instead
    of a generic failure.
    """
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
    rc, manifest, events = _launch_to_stop(
        monkeypatch,
        tmp_path,
        lambda task_ids: GatewayCircuitOpen(rows=landed_rows, threshold=3, remaining=task_ids[2:]),
    )

    assert rc == 2
    assert events == ["executor.prepare", "executor.close", "server.close"]
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


def test_budget_refusal_name_is_pinned_and_overspend_never_trips_the_gate():
    from devtools.benchmarks.cybergym.cybergym_adapter import (
        BudgetOverspend,
        BudgetRefused,
    )
    from devtools.benchmarks.cybergym.cybergym_dispatch import (
        BUDGET_REFUSED_ERROR_NAME,
        _is_budget_refusal,
    )

    assert BUDGET_REFUSED_ERROR_NAME == BudgetRefused.__name__
    assert _is_budget_refusal(BudgetRefused("reservation would exceed cap"))
    # The settlement-time subclass already has a row path through
    # ``_run_one``; it must never be re-queued as a claim-time refusal.
    assert issubclass(BudgetOverspend, BudgetRefused)
    assert not _is_budget_refusal(BudgetOverspend("settlement overspend"))
    assert not _is_budget_refusal(ExecutorFailure("workspace failed"))


def test_budget_refusal_pauses_and_settlement_frees_headroom(tmp_path):
    """A refused claim pauses admission; a cheap settlement resumes it.

    Run 20260907T233516Z burned 1145 never-dispatched tasks into infra rows
    here.  Now the refused task is re-queued without a row, and once the
    in-flight attempt settles below its reservation the probe sees headroom
    and admission resumes.
    """

    root = tmp_path / "budget-resume"
    events_path = root / "dispatch_events.jsonl"

    def callback(task, task_dir):
        if task.task_id == "arvo:1":
            # Hold the reservation until the second claim has been refused
            # and the gate paused; the dispatcher writes the pause event
            # synchronously before waiting on the in-flight lane.
            deadline = time.monotonic() + 30
            while True:
                if (
                    events_path.exists()
                    and '"budget_pause"' in events_path.read_text(encoding="utf-8")
                ):
                    break
                if time.monotonic() > deadline:
                    raise AssertionError("budget pause was never recorded")
                time.sleep(0.01)
            outcome = _completed(task, task_dir)
            outcome["cost_usd"] = 5.0
            return outcome
        return _completed(task, task_dir)

    rows = run_campaign(
        ["arvo:1", "arvo:2"],
        run_root=root,
        executor=callback,
        estimated_cost_usd=20,
        budget_cap_usd=25,
        max_workers=2,
    )

    assert [row["task_id"] for row in rows] == ["arvo:1", "arvo:2"]
    assert all(row["status"] == "completed" for row in rows)
    assert {row["task_id"] for row in _result_index(root)} == {"arvo:1", "arvo:2"}
    logged = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    names = [entry["event"] for entry in logged]
    assert "budget_pause" in names
    assert names[-1] == "budget_resume"
    assert names.index("budget_pause") < names.index("budget_resume")
    projection = BudgetLedger(root / "claims.jsonl", cap_usd=25).projection()
    assert projection.settled_usd == pytest.approx(5.1)
    assert projection.reserved_usd == 0
    assert projection.can_dispatch is True


def test_budget_cap_reached_leaves_undispatched_tasks_row_free(tmp_path):
    """When no settlement frees headroom the campaign stops, row-free."""

    def callback(task, task_dir):
        outcome = _completed(task, task_dir)
        outcome["cost_usd"] = 20.0
        return outcome

    root = tmp_path / "budget-cap"
    from devtools.benchmarks.cybergym.cybergym_dispatch import BudgetCapReached

    with pytest.raises(BudgetCapReached) as excinfo:
        run_campaign(
            ["arvo:1", "arvo:2", "arvo:3"],
            run_root=root,
            executor=callback,
            estimated_cost_usd=20,
            budget_cap_usd=20,
            max_workers=2,
        )

    exc = excinfo.value
    assert isinstance(exc, CyberGymError)
    # With two lanes the first claim wins the whole cap; which of the first
    # pair wins is a thread-scheduling race, so assert the invariants:
    # exactly one row landed, the other two tasks stay row-free for resume.
    assert len(exc.rows) == 1
    winner = exc.rows[0]["task_id"]
    assert winner in {"arvo:1", "arvo:2"}
    assert exc.remaining_task_ids == sorted(
        {"arvo:1", "arvo:2", "arvo:3"} - {winner}
    )
    payload = exc.as_dict()
    assert payload["outcome"] == "budget_cap_reached"
    assert payload["dispatched_rows"] == 1
    assert payload["pause"]["refusals"] >= 1
    # Never-dispatched tasks leave no row and no task directory, so a later
    # resume campaign re-runs them without any retry flag.
    assert [row["task_id"] for row in _result_index(root)] == [winner]
    for task_id in exc.remaining_task_ids:
        assert not safe_task_path(root, task_id).exists()
    projection = BudgetLedger(root / "claims.jsonl", cap_usd=20).projection()
    assert projection.settled_usd == pytest.approx(20)
    assert projection.reserved_usd == 0
    logged = (root / "dispatch_events.jsonl").read_text(encoding="utf-8")
    assert '"budget_pause"' in logged
    assert '"budget_gate_closed"' in logged
    assert '"cap_exhausted"' in logged


def test_budget_cap_returns_completed_rows_behind_a_refused_position():
    """A refusal at an EARLIER position must not strand later completed rows.

    The source-order drain keys on ``next_record``; a refused position never
    produces a row, so without the union drain every later completion stays
    stuck in ``completed`` and the campaign would report zero landed rows
    (seen under parallel load in the campaign-level cap test).
    """

    from devtools.benchmarks.cybergym.cybergym_adapter import BudgetRefused
    from devtools.benchmarks.cybergym.cybergym_dispatch import (
        BudgetCapReached,
        run_dispatched,
    )

    def run_one(task):
        if task.task_id == "arvo:1":
            raise BudgetRefused("reservation would exceed campaign budget cap")
        return {"task_id": task.task_id, "status": "completed"}

    with pytest.raises(BudgetCapReached) as excinfo:
        run_dispatched(
            [_Task(f"arvo:{index}") for index in range(1, 4)],
            run_one,
            max_workers=2,
            budget_probe=lambda: False,
        )

    exc = excinfo.value
    assert [row["task_id"] for row in exc.rows] == ["arvo:2"]
    assert exc.remaining_task_ids == ["arvo:1", "arvo:3"]


def test_budget_refusal_in_a_serial_lane_ends_the_campaign_cleanly(tmp_path):
    """A serial lane holds no in-flight work that could free headroom."""

    def callback(task, task_dir):
        outcome = _completed(task, task_dir)
        outcome["cost_usd"] = 20.0
        return outcome

    from devtools.benchmarks.cybergym.cybergym_dispatch import BudgetCapReached

    root = tmp_path / "budget-serial"
    with pytest.raises(BudgetCapReached) as excinfo:
        run_campaign(
            ["arvo:1", "arvo:2", "arvo:3"],
            run_root=root,
            executor=callback,
            estimated_cost_usd=20,
            budget_cap_usd=20,
            max_workers=1,
        )

    assert [row["task_id"] for row in excinfo.value.rows] == ["arvo:1"]
    assert excinfo.value.remaining_task_ids == ["arvo:2", "arvo:3"]
    assert [row["task_id"] for row in _result_index(root)] == ["arvo:1"]


def test_budget_gate_without_probe_closes_on_first_refusal(tmp_path):
    """Without a probe the first refusal is terminal; in-flight rows land."""

    from devtools.benchmarks.cybergym.cybergym_adapter import BudgetRefused
    from devtools.benchmarks.cybergym.cybergym_dispatch import (
        BudgetCapReached,
        run_dispatched,
    )

    second_refused = threading.Event()

    def run_one(task):
        if task.task_id == "arvo:1":
            assert second_refused.wait(timeout=30)
            return {"task_id": task.task_id, "status": "completed"}
        if task.task_id == "arvo:2":
            second_refused.set()
            raise BudgetRefused("reservation would exceed campaign budget cap")
        raise AssertionError(f"{task.task_id} must never be dispatched")

    events: list[dict] = []
    with pytest.raises(BudgetCapReached) as excinfo:
        run_dispatched(
            [_Task(f"arvo:{index}") for index in range(1, 4)],
            run_one,
            max_workers=2,
            budget_probe=None,
            on_event=events.append,
        )

    exc = excinfo.value
    assert [row["task_id"] for row in exc.rows] == ["arvo:1"]
    assert exc.remaining_task_ids == ["arvo:2", "arvo:3"]
    assert exc.as_dict()["pause"]["refusals"] == 1
    assert [event["event"] for event in events] == ["budget_gate_closed"]
    assert events[0]["reason"] == "no_probe"


def test_budget_overspend_is_not_a_claim_refusal(tmp_path):
    """The settlement-time subclass propagates; it never re-queues a task."""

    from devtools.benchmarks.cybergym.cybergym_adapter import BudgetOverspend
    from devtools.benchmarks.cybergym.cybergym_dispatch import run_dispatched

    def run_one(task):
        raise BudgetOverspend("measured settlement exceeds campaign budget cap")

    with pytest.raises(BudgetOverspend):
        run_dispatched(
            [_Task("arvo:1"), _Task("arvo:2")],
            run_one,
            max_workers=2,
        )


def test_launcher_finalizes_budget_cap_reached(monkeypatch, tmp_path):
    """Pin the launcher branch that finalizes a budget-stopped campaign.

    ``run_campaign`` raising ``BudgetCapReached`` must still produce a
    finalized manifest: the rows that landed stay accounted, the
    undispatched tasks are named under ``extra.budget_cap`` and the run
    records outcome ``budget_cap_reached`` with exit code 2.
    """
    from devtools.benchmarks.cybergym.cybergym_dispatch import BudgetCapReached

    landed_rows = [{"task_id": "arvo:1", "status": "completed", "final_submission_success": True}]
    rc, manifest, events = _launch_to_stop(
        monkeypatch,
        tmp_path,
        lambda task_ids: BudgetCapReached(
            rows=landed_rows,
            remaining=task_ids[1:],
            pause={"refusals": 1, "probes": 2, "pauses": []},
        ),
    )

    assert rc == 2
    assert events == ["executor.prepare", "executor.close", "server.close"]
    extra = manifest["extra"]
    assert extra["outcome"] == "budget_cap_reached"
    assert extra["exit_code"] == 2
    assert extra["budget_cap"] == {
        "outcome": "budget_cap_reached",
        "dispatched_rows": 1,
        "remaining_task_ids": ["arvo:2", "arvo:3", "arvo:4"],
        "pause": {"refusals": 1, "probes": 2, "pauses": []},
    }
    # Rows that landed before the cap stopped admission stay accounted.
    assert extra["rows_written"] == 1
    assert extra["completed_count"] == 1
    assert extra["close_skipped"] is False
    assert extra["server_cleanup"]["status"] == "closed"


def test_launcher_finalizes_workspace_custody_timeout_ahead_of_custody_pending(monkeypatch, tmp_path):
    """A custody stop keeps its unresolved residue alive (``custody_pending``
    close, server kept for rescue) yet finalizes as the distinct, truthful
    ``workspace_custody_timeout`` with its row-free remainder."""
    from devtools.benchmarks.cybergym.cybergym_dispatch import WorkspaceCustodyTimeout

    landed_rows = [{"task_id": "arvo:1", "status": "infra_failed", "infra_reason": "ExecutorFailure"}]
    pause = {"pauses": [{"paused_sec": 300.0, "failed_probes": 10}], "pause_budget_sec": 300.0}
    residue = {"cybergym-workspace-agent-" + "a" * 24: "run timed out; name inspect failed"}
    rc, manifest, events = _launch_to_stop(
        monkeypatch,
        tmp_path,
        lambda task_ids: WorkspaceCustodyTimeout(rows=landed_rows, remaining=task_ids[1:], pause=pause),
        close_report={
            "ok": False,
            "status": "custody_pending",
            "workspace_custody_unresolved": residue,
        },
    )

    assert rc == 2
    assert events == ["executor.prepare", "executor.close"]
    extra = manifest["extra"]
    assert (extra["outcome"], extra["exit_code"]) == ("workspace_custody_timeout", 2)
    assert extra["workspace_custody"] == {
        "outcome": "workspace_custody_timeout",
        "dispatched_rows": 1,
        "remaining_task_ids": ["arvo:2", "arvo:3", "arvo:4"],
        "pause": pause,
    }
    assert extra["close_skipped"] is True
    assert extra["executor_cleanup"]["workspace_custody_unresolved"] == residue
    assert extra["rows_written"] == 1


def _custody_signal():
    from devtools.benchmarks.cybergym.cybergym_dispatch import WorkspaceCustodyPending

    return WorkspaceCustodyPending({"cybergym-workspace-agent-" + "f" * 24: "run timed out"})


def test_custody_pause_requeues_row_free_and_resumes_after_a_clean_heal():
    from devtools.benchmarks.cybergym.cybergym_dispatch import run_dispatched

    clock = _PausingClock()
    probes = iter([False, True])
    calls: list[str] = []
    lock = threading.Lock()

    def run_one(task):
        with lock:
            calls.append(task.task_id)
            first_try = calls.count(task.task_id) == 1
        if task.task_id == "arvo:2" and first_try:
            raise _custody_signal()
        return {"task_id": task.task_id, "status": "completed"}

    events: list[dict] = []
    rows = run_dispatched(
        [_Task(f"arvo:{index}") for index in range(1, 5)],
        run_one,
        max_workers=2,
        custody_probe=lambda: next(probes),
        on_event=events.append,
        sleep=clock.sleep,
        clock=clock.monotonic,
    )

    assert [row["task_id"] for row in rows] == ["arvo:1", "arvo:2", "arvo:3", "arvo:4"]
    assert calls.count("arvo:2") == 2
    assert [event["event"] for event in events] == [
        "workspace_custody_pause", "workspace_custody_probe_failed", "workspace_custody_resume",
    ]
    assert list(events[0]["unresolved"]) == ["cybergym-workspace-agent-" + "f" * 24]
    assert clock.slept == [30.0, 30.0]


def test_custody_pause_is_never_reset_and_drains_in_flight_work_at_expiry():
    """Later signals cannot restart the five-minute budget; a row already in
    flight when it expires still lands before the typed stop."""
    from devtools.benchmarks.cybergym.cybergym_dispatch import (
        WorkspaceCustodyTimeout,
        run_dispatched,
    )

    clock = _PausingClock()
    paused, second_signal = threading.Event(), threading.Event()

    def run_one(task):
        if task.task_id == "arvo:1":
            raise _custody_signal()
        assert paused.wait(10), "custody gate never paused"
        if task.task_id == "arvo:2":
            clock.now = 200.0
            second_signal.set()
            raise _custody_signal()
        assert second_signal.wait(10)
        clock.now = 301.0
        return {"task_id": task.task_id, "status": "completed"}

    events: list[dict] = []

    def on_event(event):
        events.append(event)
        if event["event"] == "workspace_custody_pause":
            paused.set()

    landed: list[str] = []
    with pytest.raises(WorkspaceCustodyTimeout) as excinfo:
        run_dispatched(
            [_Task(f"arvo:{index}") for index in range(1, 6)],
            run_one,
            max_workers=3,
            on_row=lambda row: landed.append(row["task_id"]),
            custody_probe=lambda: False,
            on_event=on_event,
            sleep=clock.sleep,
            clock=clock.monotonic,
        )

    stop = excinfo.value.as_dict()
    assert landed == ["arvo:3"] and stop["dispatched_rows"] == 1
    assert stop["remaining_task_ids"] == ["arvo:1", "arvo:2", "arvo:4", "arvo:5"]
    assert stop["pause"]["pauses"][0]["paused_sec"] == 301.0
    assert [event["event"] for event in events].count("workspace_custody_pause") == 1
    assert events[-1]["event"] == "workspace_custody_timeout"


@pytest.mark.parametrize("workers", [1, 3], ids=["serial", "concurrent"])
@pytest.mark.parametrize(
    ("probe_at", "healed_at", "stops"),
    [(270.0, 299.5, False), (300.0, 300.0, True), (270.0, 300.5, True)],
    ids=["heal-inside-budget", "heal-at-expiry", "slow-heal-past-expiry"],
)
def test_custody_budget_is_hard_so_a_heal_at_or_after_expiry_still_stops(
    workers, probe_at, healed_at, stops
):
    """The contiguous five-minute custody budget is a hard deadline: a heal
    landing at or after 300 s cannot resume admission.  Serially the gate
    sleeps between probes; in parallel an in-flight attempt carries the clock
    to the due probe and its row still lands."""
    from devtools.benchmarks.cybergym.cybergym_dispatch import (
        WorkspaceCustodyTimeout,
        run_dispatched,
    )

    clock = _PausingClock()
    paused = threading.Event()
    attempts: list[str] = []
    lock = threading.Lock()

    def run_one(task):
        with lock:
            attempts.append(task.task_id)
            first_try = attempts.count(task.task_id) == 1
        if task.task_id == "arvo:1" and first_try:
            raise _custody_signal()
        if task.task_id == "arvo:2" and workers > 1:
            assert paused.wait(10), "custody gate never paused"
            clock.now = probe_at
        return {"task_id": task.task_id, "status": "completed"}

    def probe():
        if clock.now < probe_at:
            return False
        clock.now = healed_at  # the healer finishes its clean pass here
        return True

    events: list[dict] = []

    def on_event(event):
        events.append(event)
        if event["event"] == "workspace_custody_pause":
            paused.set()

    landed: list[str] = []

    def dispatch():
        return run_dispatched(
            [_Task(f"arvo:{index}") for index in range(1, 4)],
            run_one,
            max_workers=workers,
            on_row=lambda row: landed.append(row["task_id"]),
            custody_probe=probe,
            on_event=on_event,
            sleep=clock.sleep,
            clock=clock.monotonic,
        )

    if not stops:
        assert [row["task_id"] for row in dispatch()] == ["arvo:1", "arvo:2", "arvo:3"]
        assert (events[-1]["event"], events[-1]["paused_sec"]) == ("workspace_custody_resume", healed_at)
        return
    with pytest.raises(WorkspaceCustodyTimeout) as excinfo:
        dispatch()
    stop = excinfo.value.as_dict()
    assert stop["remaining_task_ids"][0] == "arvo:1" and attempts.count("arvo:1") == 1
    assert sorted(landed + stop["remaining_task_ids"]) == ["arvo:1", "arvo:2", "arvo:3"]
    assert stop["pause"]["pauses"][-1]["paused_sec"] == healed_at
    assert events[-1]["event"] == "workspace_custody_timeout"


def test_gateway_breaker_still_resumes_on_a_healthy_probe_at_its_budget():
    """Only the custody budget is hard: the transport breaker keeps its rule
    that any healthy probe resumes admission, even one at the budget."""
    from devtools.benchmarks.cybergym.cybergym_dispatch import run_dispatched

    clock = _PausingClock()

    def run_one(task):
        if task.task_id == "arvo:1":
            return _transport_row(task.task_id)
        return {"task_id": task.task_id, "status": "completed"}

    rows = run_dispatched(
        [_Task("arvo:1"), _Task("arvo:2")],
        run_one,
        max_workers=1,
        threshold=1,
        gateway_probe=lambda: clock.now >= 200.0,
        probe_backoff_sec=(100.0,),
        pause_budget_sec=200.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
    )
    assert [row["status"] for row in rows] == ["infra_failed", "completed"]
    assert clock.slept == [100.0, 100.0]


def test_custody_without_probe_stops_at_once_and_serial_keeps_the_position():
    from devtools.benchmarks.cybergym.cybergym_dispatch import (
        WorkspaceCustodyTimeout,
        run_dispatched,
    )

    def run_one(task):
        if task.task_id == "arvo:2":
            raise _custody_signal()
        return {"task_id": task.task_id, "status": "completed"}

    for workers in (1, 2):
        with pytest.raises(WorkspaceCustodyTimeout) as excinfo:
            run_dispatched(
                [_Task(f"arvo:{index}") for index in range(1, 4)],
                run_one,
                max_workers=workers,
                sleep=lambda _seconds: pytest.fail("no probe: must not sleep"),
            )
        stop = excinfo.value.as_dict()
        landed = [row["task_id"] for row in excinfo.value.rows]
        assert "pause" not in stop and stop["remaining_task_ids"][0] == "arvo:2"
        assert sorted(landed + stop["remaining_task_ids"]) == ["arvo:1", "arvo:2", "arvo:3"]
        if workers == 1:
            assert (landed, stop["remaining_task_ids"]) == (["arvo:1"], ["arvo:2", "arvo:3"])


@pytest.mark.parametrize(
    ("signals", "outcome"),
    [
        ({"custody", "budget", "gateway"}, "workspace_custody_timeout"),
        ({"budget", "gateway"}, "budget_cap_reached"),
        ({"gateway"}, "gateway_unreachable"),
    ],
)
def test_coinciding_stops_resolve_custody_then_budget_then_gateway(signals, outcome):
    from devtools.benchmarks.cybergym.cybergym_adapter import BudgetRefused
    from devtools.benchmarks.cybergym.cybergym_dispatch import run_dispatched

    kinds = sorted(signals)
    barrier = threading.Barrier(len(kinds))
    by_task = {f"arvo:{index}": kind for index, kind in enumerate(kinds, 1)}

    def run_one(task):
        barrier.wait(10)
        kind = by_task[task.task_id]
        if kind == "custody":
            raise _custody_signal()
        if kind == "budget":
            raise BudgetRefused("reservation would exceed campaign budget cap")
        return _transport_row(task.task_id)

    with pytest.raises(CyberGymError) as excinfo:
        run_dispatched(
            [_Task(task_id) for task_id in by_task], run_one, max_workers=len(kinds), threshold=1,
        )
    assert excinfo.value.as_dict()["outcome"] == outcome


def test_last_requeued_position_waits_out_a_gateway_pause_instead_of_vanishing():
    """A budget-requeued final position met a paused gateway with nothing in
    flight; it must be re-admitted after the pause, never silently dropped."""
    from devtools.benchmarks.cybergym.cybergym_adapter import BudgetRefused
    from devtools.benchmarks.cybergym.cybergym_dispatch import run_dispatched

    clock = _PausingClock()
    gateway_paused = threading.Event()
    attempts: list[str] = []

    def run_one(task):
        attempts.append(task.task_id)
        if task.task_id == "arvo:1":
            return _transport_row(task.task_id)
        if attempts.count("arvo:2") == 1:
            assert gateway_paused.wait(10), "the gateway pause was never recorded"
            raise BudgetRefused("reservation would exceed campaign budget cap")
        return {"task_id": task.task_id, "status": "completed"}

    def on_event(event):
        if event["event"] == "gateway_pause":
            gateway_paused.set()

    rows = run_dispatched(
        [_Task("arvo:1"), _Task("arvo:2")],
        run_one,
        max_workers=2,
        threshold=1,
        gateway_probe=lambda: True,
        budget_probe=lambda: True,
        on_event=on_event,
        sleep=clock.sleep,
        clock=clock.monotonic,
    )
    assert [(row["task_id"], row["status"]) for row in rows] == [
        ("arvo:1", "infra_failed"), ("arvo:2", "completed"),
    ]
    assert clock.slept == [30.0]
