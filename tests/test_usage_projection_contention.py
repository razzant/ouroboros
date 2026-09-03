"""Battle test: display projections never stall behind a real 64-way ledger convoy.

CyberGym remaining1150 postmortem class: 64 lanes of monetary writes held the
usage ledger lock almost continuously, and every display/compatibility read on
a concurrency-critical thread waited out the 45s monetary timeout. py-spy
caught the supervisor loop parked in ``_handle_task_heartbeat`` →
``live_root_cost_projection`` → ``_memoized_final_rows`` → ``_locked`` (three
dumps, stalls of 90-105s past the liveness deadline); the llm_usage refresh
(``update_budget_from_usage``) and the ``assign_tasks`` budget gate share the
shape; on the gateway side ``api_task_get`` ran the same locked read inline on
the asyncio loop, so the executor's 60s status polls died at transport level —
67 GatewayTransportError write-offs, then the crash storm.

This test runs the REAL monetary write path
(``reserve_attempt``/``mark_dispatched``/``settle_attempt``) from 64 writer
threads — the production lane count — against a real on-disk ledger, proves
the convoy is real (a non-stale short-timeout read really fails), then drives
the REAL loop and gateway display readers with their production arguments and
proves none of them ever waits out the monetary timeout: contended reads serve
the last validated snapshot, a cold memo still fails closed exactly once, and
the memo converges to the fresh ledger the moment the convoy eases.
"""
from __future__ import annotations

import threading
import time

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.usage_ledger import DISPLAY_LOCK_TIMEOUT_SEC, UsageLockUnavailable

_WORKERS = 64  # the production CyberGym lane count
_CONVOY_SEC = 4.0
# Hard bound for one display read under the convoy: 20x the display timeout,
# 18x under the supervisor liveness deadline (90s) that the old shape blew past.
_MAX_READ_SEC = 5.0


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    (root / "state").mkdir(parents=True)
    return root


def _request(data_root, worker_index):
    return ua.AttemptRequest(
        model="openai/gpt-5.2",
        provider="openai",
        reservation_usd=0.0001,
        global_limit_usd=1_000_000.0,
        drive_root=data_root,
        task_id=f"task-{worker_index}",
        root_task_id="root",
        source="test",
    )


def _writer(data_root, worker_index, stop, errors):
    while not stop.is_set():
        try:
            reservation = ua.reserve_attempt(_request(data_root, worker_index))
            ua.mark_dispatched(reservation)
            ua.settle_attempt(
                reservation,
                {"prompt_tokens": 5, "completion_tokens": 2},
                cost_usd=0.0001,
                cost_final=True,
            )
        except Exception as exc:  # the monetary write path must never fail here
            errors.append(f"writer {worker_index}: {type(exc).__name__}: {exc}")


def _prove_convoy(data_root, stop):
    """A non-stale short-timeout read must really lose the lock race — otherwise
    the convoy is vacuous and the battery below proves nothing."""
    deadline = time.monotonic() + _CONVOY_SEC
    while time.monotonic() < deadline and not stop.is_set():
        try:
            ua.usage_projection(data_root, lock_timeout_sec=DISPLAY_LOCK_TIMEOUT_SEC)
        except UsageLockUnavailable:
            return True
    return False


def test_display_reads_never_stall_under_64way_write_convoy(data_root, monkeypatch):
    from ouroboros.cost_projection import live_root_cost_projection
    from ouroboros.gateway.tasks import _task_cost_breakdown_view
    from supervisor import state as supervisor_state

    # The llm_usage refresh writes the legacy state.json projection; point the
    # supervisor state module at the test root and keep the network drift probe
    # out of the test.
    monkeypatch.setattr(supervisor_state, "DRIVE_ROOT", data_root)
    monkeypatch.setattr(supervisor_state, "STATE_PATH", data_root / "state" / "state.json")
    monkeypatch.setattr(
        supervisor_state, "STATE_LAST_GOOD_PATH", data_root / "state" / "state.last_good.json"
    )
    monkeypatch.setattr(supervisor_state, "STATE_LOCK_PATH", data_root / "locks" / "state.lock")
    monkeypatch.setattr(supervisor_state, "TOTAL_BUDGET_LIMIT", 1_000_000.0)
    monkeypatch.setattr(supervisor_state, "check_openrouter_ground_truth", lambda: None)

    # Seed + warm the memo exactly as the first loop tick would.
    reservation = ua.reserve_attempt(_request(data_root, 0))
    ua.mark_dispatched(reservation)
    ua.settle_attempt(reservation, {"prompt_tokens": 5, "completion_tokens": 2}, cost_usd=0.0001, cost_final=True)
    warm = ua.usage_projection(data_root, root_task_id="root")
    assert warm["attempt_counts"]

    stop = threading.Event()
    errors: list[str] = []
    writers = [
        threading.Thread(target=_writer, args=(data_root, index, stop, errors), daemon=True)
        for index in range(_WORKERS)
    ]
    for thread in writers:
        thread.start()

    try:
        assert _prove_convoy(data_root, stop), "64 writers never contended the ledger lock"

        task = {"id": "root", "root_task_id": "root", "budget_drive_root": str(data_root)}
        result_row = {"task_id": "root", "root_task_id": "root"}
        readers = [
            ("heartbeat_cost_projection", lambda: live_root_cost_projection("root", task, {}, data_root)),
            (
                "assign_tasks_budget_gate",
                lambda: supervisor_state.budget_remaining(
                    {}, strict=True,
                    lock_timeout_sec=DISPLAY_LOCK_TIMEOUT_SEC, allow_stale=True,
                ),
            ),
            ("gateway_task_cost_view", lambda: _task_cost_breakdown_view(data_root, result_row)),
            ("llm_usage_budget_refresh", lambda: supervisor_state.update_budget_from_usage({})),
        ]

        durations: dict[str, float] = {name: 0.0 for name, _ in readers}
        deadline = time.monotonic() + _CONVOY_SEC
        rounds = 0
        while time.monotonic() < deadline:
            for name, read in readers:
                started = time.monotonic()
                read()
                durations[name] = max(durations[name], time.monotonic() - started)
            rounds += 1

        assert rounds >= 2, f"reader battery starved: {rounds} round(s) in {_CONVOY_SEC}s"
        for name, worst in durations.items():
            assert worst < _MAX_READ_SEC, f"{name} stalled {worst:.2f}s behind the convoy"

        # The readers kept serving real data from the last validated snapshot.
        projection = live_root_cost_projection("root", task, {}, data_root)
        assert projection.get("cost_accounting_status") == "available"
        assert projection.get("cost_usd_with_children") is not None
    finally:
        stop.set()
        for thread in writers:
            thread.join(timeout=30)
        assert not any(thread.is_alive() for thread in writers), "writer never stopped"

    assert not errors, errors

    # Convergence: once the convoy eases, the very next display read revalidates
    # against the ledger — stale serving is a convoy-only posture, not a wedge.
    final = ua.usage_breakdown(data_root)
    final_calls = int(final.get("physical_calls") or 0)
    assert final_calls >= _WORKERS
    converged = ua.usage_breakdown(
        data_root, lock_timeout_sec=DISPLAY_LOCK_TIMEOUT_SEC, allow_stale=True
    )
    assert int(converged.get("physical_calls") or 0) == final_calls


def test_cold_memo_fails_closed_once_under_contention(data_root):
    """No validated snapshot yet + a contended lock → the caller's unavailable
    branch, exactly as before — display reads never invent a $0 authority."""
    from ouroboros.usage_ledger import _locked

    hold = threading.Event()
    release = threading.Event()

    def gatekeeper():
        with _locked(data_root):
            hold.set()
            release.wait(timeout=30)

    thread = threading.Thread(target=gatekeeper, daemon=True)
    thread.start()
    try:
        assert hold.wait(timeout=10), "gatekeeper never took the lock"
        with pytest.raises(UsageLockUnavailable):
            ua.usage_projection(
                data_root,
                lock_timeout_sec=DISPLAY_LOCK_TIMEOUT_SEC,
                allow_stale=True,
            )
    finally:
        release.set()
        thread.join(timeout=30)
    assert not thread.is_alive()

    # After the gatekeeper releases, the same read succeeds and seeds the memo.
    projection = ua.usage_projection(
        data_root, lock_timeout_sec=DISPLAY_LOCK_TIMEOUT_SEC, allow_stale=True
    )
    assert projection is not None
