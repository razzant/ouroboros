"""Battle test: display projections never stall behind a real 64-way ledger convoy.

CyberGym remaining1150 postmortem class: 64 lanes of monetary writes held the
usage ledger lock almost continuously, and every display/compatibility read on
a concurrency-critical thread waited out the 45s monetary timeout. py-spy
caught the supervisor loop parked in ``_handle_task_heartbeat`` →
``live_root_cost_projection`` → ``_memoized_final_rows`` → ``_locked`` (three
dumps, stalls of 90-105s past the liveness deadline); the llm_usage refresh
(``update_budget_from_usage``) and the ``assign_tasks`` budget pre-check share
the shape, and on the gateway side the cost views ran the same locked read.

The first test runs the REAL monetary write path
(``reserve_attempt``/``mark_dispatched``/``settle_attempt``) from 64 writer
threads — the production lane count — against a real on-disk ledger, proves
the convoy is real (a short-timeout lock attempt really fails), then drives
the REAL loop and gateway display readers with their production arguments and
proves none of them ever waits out the monetary timeout.

The rest pin the money contract of that stale-while-revalidate path from both
sides: a display reader is served the last validated snapshot quickly while
``reserve_attempt`` keeps waiting for the lock and refuses an exhausted
budget exactly; a snapshot may say "there is money" but never refuses by
itself; the owner's live limit is applied to the snapshot, never remembered
with it; a lagging snapshot cannot regress ``state.json``; a cold memo fails
closed and the loop handlers survive that without publishing a zero.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time

import pytest

from ouroboros import _usage_rows_memo as rows_memo
from ouroboros import usage_accounting as ua
from ouroboros.runtime_limits import USAGE_DISPLAY_LOCK_TIMEOUT_SEC
from ouroboros.usage_ledger import UsageLockUnavailable

pytestmark = pytest.mark.serial  # 64 real threads + supervisor.state module globals

_WORKERS = 64  # the production CyberGym lane count
_CONVOY_SEC = 3.0
# Hard bound for one display read under contention: 20x the display timeout,
# 9x under the 45s monetary timeout the old shape waited out on every read.
_MAX_READ_SEC = 5.0


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    (root / "state").mkdir(parents=True)
    (root / "logs").mkdir()
    ua._reset_task_cache_splits()
    return root


@pytest.fixture
def supervisor_state(data_root, monkeypatch):
    """``supervisor.state`` pointed at the test root, network probe removed."""
    from supervisor import state

    monkeypatch.setattr(state, "DRIVE_ROOT", data_root)
    monkeypatch.setattr(state, "STATE_PATH", data_root / "state" / "state.json")
    monkeypatch.setattr(state, "STATE_LAST_GOOD_PATH", data_root / "state" / "state.last_good.json")
    monkeypatch.setattr(state, "STATE_LOCK_PATH", data_root / "locks" / "state.lock")
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 1_000_000.0)
    monkeypatch.setattr(state, "check_openrouter_ground_truth", lambda: None)
    return state


def _request(data_root, task_id="task", *, reservation_usd=0.0001, limit=1_000_000.0):
    return ua.AttemptRequest(
        model="openai/gpt-5.2",
        provider="openai",
        reservation_usd=reservation_usd,
        global_limit_usd=limit,
        drive_root=data_root,
        task_id=task_id,
        root_task_id="root",
        source="test",
    )


def _spend(data_root, cost, *, task_id="task", limit=1_000_000.0):
    reservation = ua.reserve_attempt(_request(data_root, task_id, reservation_usd=cost, limit=limit))
    ua.mark_dispatched(reservation)
    ua.settle_attempt(
        reservation, {"prompt_tokens": 5, "completion_tokens": 2}, cost_usd=cost, cost_final=True,
    )


@contextlib.contextmanager
def _held_ledger_lock(data_root):
    """One writer sitting on the monetary lock: the convoy, made deterministic."""
    from ouroboros.usage_ledger import _locked

    held, release = threading.Event(), threading.Event()

    def gatekeeper():
        with _locked(data_root):
            held.set()
            release.wait(timeout=60)

    thread = threading.Thread(target=gatekeeper, daemon=True)
    thread.start()
    assert held.wait(timeout=10), "gatekeeper never took the lock"
    try:
        yield release
    finally:
        release.set()
        thread.join(timeout=30)
        assert not thread.is_alive(), "gatekeeper never released the lock"


def _llm_usage_rows(data_root):
    lines = (data_root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [row for row in map(json.loads, lines) if row.get("type") == "llm_usage"]


def _timed(read):
    started = time.monotonic()
    value = read()
    return value, time.monotonic() - started


def _writer(data_root, worker_index, stop, errors):
    while not stop.is_set():
        try:
            _spend(data_root, 0.0001, task_id=f"task-{worker_index}")
        except Exception as exc:  # the monetary write path must never fail here
            errors.append(f"writer {worker_index}: {type(exc).__name__}: {exc}")


def _prove_convoy(data_root, stop):
    """A short-timeout attempt on the monetary lock must really lose the race —
    otherwise the convoy is vacuous and the battery below proves nothing."""
    from ouroboros.usage_ledger import _locked

    deadline = time.monotonic() + _CONVOY_SEC
    while time.monotonic() < deadline and not stop.is_set():
        try:
            with _locked(data_root, timeout_sec=USAGE_DISPLAY_LOCK_TIMEOUT_SEC):
                pass
        except UsageLockUnavailable:
            return True
    return False


def test_display_reads_never_stall_under_64way_write_convoy(data_root, supervisor_state):
    from ouroboros.cost_projection import live_root_cost_projection
    from ouroboros.gateway.cost_breakdown import _task_cost_breakdown_view

    # Seed + warm the memo exactly as the first loop tick would.
    _spend(data_root, 0.0001, task_id="task-0")
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
                "assign_tasks_budget_pre_check",
                lambda: supervisor_state.budget_remaining({}, strict=True, allow_stale=True),
            ),
            ("gateway_task_cost_view", lambda: _task_cost_breakdown_view(data_root, result_row)),
            ("llm_usage_budget_refresh", lambda: supervisor_state.update_budget_from_usage({})),
        ]

        durations: dict[str, float] = {name: 0.0 for name, _ in readers}
        deadline = time.monotonic() + _CONVOY_SEC
        rounds = 0
        while time.monotonic() < deadline:
            for name, read in readers:
                _, elapsed = _timed(read)
                durations[name] = max(durations[name], elapsed)
            rounds += 1

        assert rounds >= 2, f"reader battery starved: {rounds} round(s) in {_CONVOY_SEC}s"
        for name, worst in durations.items():
            assert worst < _MAX_READ_SEC, f"{name} stalled {worst:.2f}s behind the convoy"

        # The readers kept serving real data from the last validated snapshot.
        projection = live_root_cost_projection("root", task, {}, data_root)
        assert projection.get("cost_accounting_status") == "available"
        assert projection.get("accounted_upper_bound_usd_with_children") is not None
    finally:
        stop.set()
        for thread in writers:
            thread.join(timeout=60)
        assert not any(thread.is_alive() for thread in writers), "writer never stopped"

    assert not errors, errors[:5]

    # Convergence: stale serving is a convoy-only posture, not a wedge. Once the
    # convoy eases, display reads alone catch up with the ledger.
    final_calls = int(ua.usage_breakdown(data_root).get("physical_calls") or 0)
    assert final_calls >= _WORKERS
    converged = ua.usage_breakdown(data_root, allow_stale=True)
    assert int(converged.get("physical_calls") or 0) == final_calls


def test_display_reader_serves_the_snapshot_while_the_monetary_gate_stays_exact(data_root):
    """Rule of the port: money stays exact, only display readers go stale."""
    _spend(data_root, 0.40, limit=1.0)
    assert ua.usage_projection(data_root, global_limit_usd=1.0)["remaining_known_usd"] == pytest.approx(0.60)
    _spend(data_root, 0.60, task_id="second", limit=1.0)  # the budget is now genuinely exhausted

    verdict: list = []

    def next_paid_attempt():
        try:
            verdict.append(ua.reserve_attempt(_request(data_root, "third", reservation_usd=0.01, limit=1.0)))
        except Exception as exc:
            verdict.append(exc)

    with _held_ledger_lock(data_root):
        shown, elapsed = _timed(
            lambda: ua.usage_projection(data_root, global_limit_usd=1.0, allow_stale=True))
        assert elapsed < _MAX_READ_SEC
        assert shown["accounted_usd"] == pytest.approx(0.40)  # the snapshot lags, by design
        # Without the display contract the very same read fails closed instead.
        with pytest.raises(UsageLockUnavailable):
            with ua._locked(data_root, timeout_sec=USAGE_DISPLAY_LOCK_TIMEOUT_SEC):
                pass

        gate = threading.Thread(target=next_paid_attempt, daemon=True)
        gate.start()
        gate.join(timeout=1.5)
        assert gate.is_alive() and not verdict, "reserve_attempt must wait for the lock, never a snapshot"

    gate.join(timeout=60)
    assert not gate.is_alive()
    assert len(verdict) == 1 and isinstance(verdict[0], ua.BudgetExceeded), verdict
    assert ua.usage_projection(data_root, global_limit_usd=1.0)["accounted_usd"] == pytest.approx(1.0)


def test_a_snapshot_may_admit_but_never_refuses(data_root, supervisor_state, monkeypatch):
    """A lagging snapshot that still shows an open reservation must not pause or
    fail queued work: the pre-check decides a refusal on the exact read only."""
    monkeypatch.setattr(supervisor_state, "TOTAL_BUDGET_LIMIT", 1.0)
    reservation = ua.reserve_attempt(_request(data_root, reservation_usd=1.0, limit=1.0))
    ua.mark_dispatched(reservation)
    assert supervisor_state.budget_remaining({}, strict=True, allow_stale=True) == pytest.approx(0.0)
    ua.settle_attempt(reservation, {"prompt_tokens": 5, "completion_tokens": 2}, cost_usd=0.10, cost_final=True)

    answer: list = []
    with _held_ledger_lock(data_root):
        shown = ua.usage_projection(data_root, global_limit_usd=1.0, allow_stale=True)
        assert shown["remaining_known_usd"] == pytest.approx(0.0)  # what the snapshot alone would say
        pre_check = threading.Thread(
            target=lambda: answer.append(supervisor_state.budget_remaining({}, strict=True, allow_stale=True)),
            daemon=True,
        )
        pre_check.start()
        pre_check.join(timeout=1.5)
        assert pre_check.is_alive() and not answer, "a refusal was decided on a snapshot"
    pre_check.join(timeout=60)
    assert answer == [pytest.approx(0.90)]

    # The quiet side: a snapshot that shows money answers at once, lock held or not.
    with _held_ledger_lock(data_root):
        remaining, elapsed = _timed(
            lambda: supervisor_state.budget_remaining({}, strict=True, allow_stale=True))
    assert remaining == pytest.approx(0.90) and elapsed < _MAX_READ_SEC
    # A caller that refuses below its own reserve names it, and gets the exact read there too.
    with _held_ledger_lock(data_root):
        waiting = threading.Thread(
            target=lambda: answer.append(
                supervisor_state.budget_remaining({}, strict=True, allow_stale=True, refuse_below=5.0)),
            daemon=True,
        )
        waiting.start()
        waiting.join(timeout=1.5)
        assert waiting.is_alive(), "a reserve-floor refusal was decided on a snapshot"
    waiting.join(timeout=60)
    assert not waiting.is_alive()


def _assignment_with_an_evolution_row(data_root, monkeypatch, *, settle_at):
    """The real ``assign_tasks`` over a real ledger: a $5 limit, one lane that reserved $4
    (the snapshot says $1 left, under the $2 evolution reserve) and then settled."""
    from types import SimpleNamespace

    from supervisor import queue, state, workers

    state.init(data_root, total_budget_limit=5.0)
    queue.init(data_root)
    for module in (workers, queue):
        monkeypatch.setattr(module, "DRIVE_ROOT", data_root)
    monkeypatch.setattr(state, "check_openrouter_ground_truth", lambda: None)
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 0})
    monkeypatch.setattr(workers, "_evolution_assignment_error", lambda _task: "")
    pending, running, pool = [], {}, {}
    for name, value in (("PENDING", pending), ("RUNNING", running), ("WORKERS", pool)):
        monkeypatch.setattr(workers, name, value)
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    queue.init_queue_refs(pending, running, workers.QUEUE_SEQ_COUNTER_REF)
    reservation = ua.reserve_attempt(_request(data_root, reservation_usd=4.0, limit=5.0))
    ua.mark_dispatched(reservation)
    assert state.budget_remaining({}, strict=True, allow_stale=True) == pytest.approx(1.0)  # warms the memo
    ua.settle_attempt(reservation, {"prompt_tokens": 5, "completion_tokens": 2}, cost_usd=settle_at, cost_final=True)
    sent: list = []
    pool[0] = SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                              in_q=SimpleNamespace(put=lambda task: sent.append(dict(task))))
    pending.append({"id": "evo-1", "type": "evolution", "chat_id": 0, "priority": 1})
    reasons: list = []
    real_persist = queue.persist_queue_snapshot
    monkeypatch.setattr(queue, "persist_queue_snapshot",
                        lambda reason="", **kw: (reasons.append(reason), real_persist(reason=reason, **kw))[1])
    return workers, pending, reasons


def test_an_evolution_row_is_never_dropped_on_a_snapshot(data_root, monkeypatch):
    """``assign_tasks`` refuses at TWO floors: zero, and the evolution reserve. A lagging
    snapshot inside the reserve must not drop an evolution row the exact budget affords."""
    workers, pending, reasons = _assignment_with_an_evolution_row(data_root, monkeypatch, settle_at=0.10)
    with _held_ledger_lock(data_root):
        tick = threading.Thread(target=workers.assign_tasks, daemon=True)
        tick.start()
        tick.join(timeout=1.5)
        assert tick.is_alive(), "the evolution reserve refusal was decided on a snapshot"
        assert [row["id"] for row in pending] == ["evo-1"] and not reasons
    tick.join(timeout=60)
    assert not tick.is_alive() and "evolution_dropped_budget" not in reasons  # exact budget: $4.90


def test_an_evolution_row_the_exact_budget_cannot_afford_is_still_dropped(data_root, monkeypatch):
    """The quiet direction: the refusal itself is untouched when the money is really gone."""
    workers, pending, reasons = _assignment_with_an_evolution_row(data_root, monkeypatch, settle_at=3.50)
    workers.assign_tasks()
    assert pending == [] and "evolution_dropped_budget" in reasons  # exact budget: $1.50


def test_a_cold_memo_reads_exactly_instead_of_refusing(data_root, supervisor_state, monkeypatch):
    """"May admit, never refuses" includes the refusal by absence: with no validated
    snapshot a pre-check waits for the exact read, as it did before, instead of telling
    the owner that cost accounting is unavailable after a quarter of a second."""
    monkeypatch.setattr(supervisor_state, "TOTAL_BUDGET_LIMIT", 5.0)
    _spend(data_root, 0.40, limit=5.0)
    rows_memo._ROWS_MEMO.clear()  # a fresh supervisor generation: nothing validated yet
    answer: list = []
    with _held_ledger_lock(data_root):
        pre_check = threading.Thread(
            target=lambda: answer.append(supervisor_state.budget_remaining({}, strict=True, allow_stale=True)),
            daemon=True,
        )
        pre_check.start()
        pre_check.join(timeout=1.5)
        assert pre_check.is_alive() and not answer, "a cold memo refused instead of reading exactly"
    pre_check.join(timeout=60)
    assert answer == [pytest.approx(4.60)]
    # A DISPLAY reader keeps failing closed on a cold memo (the test below): it has no refusal to decide.


def test_assignment_rides_the_snapshot_when_it_shows_money(data_root, monkeypatch):
    """The other direction of the two tests above: with money plainly there the tick
    never touches the lock, which is why this package exists."""
    from supervisor import state

    workers, pending, reasons = _assignment_with_an_evolution_row(data_root, monkeypatch, settle_at=0.10)
    assert state.budget_remaining({}, strict=True, allow_stale=True) == pytest.approx(4.90)  # revalidates
    with _held_ledger_lock(data_root):
        _, elapsed = _timed(workers.assign_tasks)
    assert elapsed < _MAX_READ_SEC and pending == [] and "evolution_dropped_budget" not in reasons


def _display_readers(data_root, state):
    """Every DISPLAY reader this package moved onto the snapshot, as the real callable."""
    from ouroboros.consciousness_allowance import allowance_window
    from ouroboros.gateway.cost_breakdown import _task_cost_breakdown_view
    from supervisor import message_bus, queue

    return {
        "status_text": lambda: state.status_text({}, [], {}),
        "budget_breakdown": lambda: state.budget_breakdown({}),
        "model_breakdown": lambda: state.model_breakdown({}),
        "budget_line": lambda: message_bus._format_budget_line({}),
        "evolution_status": lambda: queue.get_evolution_status_snapshot(),
        "task_cost_view": lambda: _task_cost_breakdown_view(data_root, {"task_id": "task", "root_task_id": "task"}),
        "consciousness_status": lambda: allowance_window(data_root, allow_stale=True),
    }


@pytest.mark.parametrize("reader", [
    "status_text", "budget_breakdown", "model_breakdown", "budget_line",
    "evolution_status", "task_cost_view", "consciousness_status",
])
def test_every_display_reader_answers_under_a_held_monetary_lock(data_root, supervisor_state, monkeypatch, reader):
    """The wiring, caller by caller: a silent return to the exact read would park the
    supervisor loop (or a gateway worker) for the 45-second monetary timeout again."""
    from supervisor import message_bus, queue

    for module in (message_bus, queue):
        monkeypatch.setattr(module, "DRIVE_ROOT", data_root, raising=False)
    monkeypatch.setattr(message_bus, "TOTAL_BUDGET_LIMIT", 1_000_000.0, raising=False)
    _spend(data_root, 0.40)
    read = _display_readers(data_root, supervisor_state)[reader]
    read()  # warm: the first read validates a snapshot
    with _held_ledger_lock(data_root):
        _, elapsed = _timed(read)
    assert elapsed < _MAX_READ_SEC, f"{reader} waited {elapsed:.1f}s on the monetary lock"


def test_a_wake_admission_reads_the_allowance_exactly(data_root):
    """The dangerous direction of the status view's snapshot read: the SAME reader
    admits a consciousness wake, and an admission waits for the exact read however
    warm the memo is (``allow_stale`` is opt-in; the status view alone opts in)."""
    from ouroboros.consciousness_allowance import allowance_window

    _spend(data_root, 0.40)
    assert allowance_window(data_root)["status"]  # warms the memo
    verdict: list = []
    with _held_ledger_lock(data_root):
        admission = threading.Thread(target=lambda: verdict.append(allowance_window(data_root)), daemon=True)
        admission.start()
        admission.join(timeout=1.5)
        assert admission.is_alive() and not verdict, "a wake admission was decided on a snapshot"
    admission.join(timeout=60)
    assert verdict and verdict[0]["status"] != "allowance_unknown"


def test_the_live_limit_is_applied_to_the_snapshot_never_remembered_with_it(
    data_root, supervisor_state, monkeypatch,
):
    monkeypatch.setattr(supervisor_state, "TOTAL_BUDGET_LIMIT", 1.0)
    _spend(data_root, 1.0, limit=1.0)
    assert supervisor_state.budget_remaining({}, strict=True) == pytest.approx(0.0)

    monkeypatch.setattr(supervisor_state, "TOTAL_BUDGET_LIMIT", 5.0)  # the owner raises the budget
    with _held_ledger_lock(data_root):
        remaining, elapsed = _timed(
            lambda: supervisor_state.budget_remaining({}, strict=True, allow_stale=True))
        shown = ua.usage_projection(data_root, global_limit_usd=5.0, allow_stale=True)
    assert remaining == pytest.approx(4.0) and elapsed < _MAX_READ_SEC
    assert shown["limit_usd"] == pytest.approx(5.0)


def test_contended_display_reads_back_off_instead_of_paying_the_timeout_each(data_root, monkeypatch):
    monkeypatch.setattr(rows_memo, "USAGE_DISPLAY_REVALIDATE_AFTER_SEC", 600.0)
    _spend(data_root, 0.25)
    assert ua.usage_breakdown(data_root)["physical_calls"] == 1
    attempts: list[dict] = []
    real_locked = ua._locked

    def counting_locked(root, **kwargs):
        attempts.append(kwargs)
        return real_locked(root, **kwargs)

    monkeypatch.setattr(ua, "_locked", counting_locked)
    with _held_ledger_lock(data_root):
        for _ in range(25):
            assert ua.usage_breakdown(data_root, allow_stale=True)["physical_calls"] == 1
        assert attempts == [{"timeout_sec": USAGE_DISPLAY_LOCK_TIMEOUT_SEC}]
    # The exact path is untouched by the backoff: it takes the lock every time.
    ua.usage_breakdown(data_root)
    assert attempts[-1] == {} and len(attempts) == 2


def test_a_lagging_snapshot_cannot_regress_state_json(data_root, supervisor_state):
    _spend(data_root, 0.40)
    assert supervisor_state.update_budget_from_usage({}) is True
    saved = supervisor_state.load_state()
    assert saved["spent_usd"] == pytest.approx(0.40)
    # Another process, reading a later ledger position, already saved newer money.
    newer = dict(saved, spent_usd=9.75, usage_ledger_high_water_seq=[0, 10_000])
    supervisor_state.save_state(newer)

    with _held_ledger_lock(data_root):
        refreshed, elapsed = _timed(lambda: supervisor_state.update_budget_from_usage({}))
    assert refreshed is False and elapsed < _MAX_READ_SEC
    after = supervisor_state.load_state()
    assert after["spent_usd"] == pytest.approx(9.75)
    assert after["usage_ledger_high_water_seq"] == [0, 10_000]


def test_cold_memo_fails_closed_once_under_contention(data_root):
    """No validated snapshot yet + a contended lock → the caller's unavailable
    branch, exactly as before — display reads never invent a $0 authority."""
    with _held_ledger_lock(data_root):
        with pytest.raises(UsageLockUnavailable):
            ua.usage_projection(data_root, allow_stale=True)
    # After the writer releases, the same read succeeds and seeds the memo.
    assert ua.usage_projection(data_root, allow_stale=True)["accounted_usd"] == 0.0


def test_loop_handlers_survive_a_cold_memo_without_publishing_a_zero(data_root, supervisor_state, monkeypatch):
    from ouroboros import server_liveness
    from ouroboros.cost_projection import live_root_cost_projection
    from supervisor import events_budget

    monkeypatch.setattr(server_liveness, "BUDGET_PROJECTION_RETRY_SEC", 0.0)  # retry on the very next turn
    _spend(data_root, 0.40)
    rows_memo._ROWS_MEMO.clear()  # a fresh supervisor generation: nothing validated yet
    supervisor_state.save_state(dict(supervisor_state.load_state(), spent_usd=7.5))

    class Ctx:
        DRIVE_ROOT = data_root
        update_budget_from_usage = staticmethod(supervisor_state.update_budget_from_usage)

    ctx = Ctx()
    task = {"id": "root", "root_task_id": "root", "budget_drive_root": str(data_root)}
    event = {"type": "llm_usage", "task_id": "task", "usage": {"prompt_tokens": 5, "cost": 0.4}}
    with _held_ledger_lock(data_root):
        heartbeat, heartbeat_sec = _timed(lambda: live_root_cost_projection("root", task, {}, data_root))
        events_budget._handle_llm_usage(dict(event), ctx)
        _, usage_sec = _timed(lambda: server_liveness.flush_budget_projection(ctx))  # the turn's one write
    assert heartbeat_sec < _MAX_READ_SEC and usage_sec < _MAX_READ_SEC
    assert heartbeat["cost_accounting_status"] == "unavailable"
    assert heartbeat["accounted_upper_bound_usd_with_children"] is None  # unknown, never zero
    assert [row["projection_update_status"] for row in _llm_usage_rows(data_root)] == ["deferred"]
    assert ctx.budget_projection_dirty is True  # the refused write is owed to the next turn
    assert supervisor_state.load_state()["spent_usd"] == pytest.approx(7.5)  # the last value stays

    # The working case stays quiet: with the lock free the same handlers publish real money.
    events_budget._handle_llm_usage(dict(event), ctx)
    server_liveness.flush_budget_projection(ctx)
    assert _llm_usage_rows(data_root)[-1]["projection_update_status"] == "deferred"
    assert ctx.budget_projection_dirty is False
    assert supervisor_state.load_state()["spent_usd"] == pytest.approx(0.40)
    heartbeat = live_root_cost_projection("root", task, {}, data_root)
    assert heartbeat["cost_accounting_status"] == "available"
    assert heartbeat["accounted_upper_bound_usd_with_children"] == pytest.approx(0.40)


def test_cost_breakdown_endpoint_reads_the_ledger_off_the_event_loop(data_root, monkeypatch):
    """The gateway half: an inline ledger read parks every HTTP client behind it."""
    from ouroboros.gateway.cost_breakdown import make_cost_breakdown_endpoint

    _spend(data_root, 0.25)
    real_breakdown = ua.usage_breakdown
    seen: dict = {}

    def recording_breakdown(root, **kwargs):
        seen.update(thread=threading.get_ident(), kwargs=kwargs)
        return real_breakdown(root, **kwargs)

    monkeypatch.setattr(ua, "usage_breakdown", recording_breakdown)

    async def call():
        return threading.get_ident(), await make_cost_breakdown_endpoint(data_root)(None)

    loop_thread, response = asyncio.run(call())
    assert seen["thread"] != loop_thread, "the ledger read ran on the asyncio loop thread"
    assert seen["kwargs"] == {"allow_stale": True}
    assert response.status_code == 200
    assert json.loads(response.body)["total_cost"] == pytest.approx(0.25)
