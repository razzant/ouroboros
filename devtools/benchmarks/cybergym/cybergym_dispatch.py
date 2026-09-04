"""CyberGym campaign dispatch engine: bounded fan-out plus a dead-gateway breaker.

Extracted from ``cybergym_adapter.run_campaign`` so the stateful adapter stays
inside its module-size band.  This module owns only dispatch policy: it never
touches the budget ledger, the result index, workspaces, or containers.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from devtools.benchmarks.cybergym.cybergym_protocol import CyberGymError

# Consecutive transport-class gateway failures that prove the isolate is dead
# and open the dispatch circuit breaker.  Small on purpose: a healthy gateway
# never produces even one transport failure, so three in a row is already a
# deterministic dead-transport signal (run 3 burned 234 tasks without this).
GATEWAY_CIRCUIT_BREAKER_THRESHOLD = 3

# When the caller supplies a liveness probe, the breaker first PAUSES
# admission and probes the gateway on this backoff schedule instead of
# abandoning the campaign outright: full1507 lost 1360 never-dispatched tasks
# to three transport rows produced by a ~100 s supervisor stall, not by a dead
# isolate.  Only a pause that exhausts ``GATEWAY_PAUSE_BUDGET_SEC`` without a
# single healthy probe opens the circuit for good.
GATEWAY_PROBE_BACKOFF_SEC: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0)
GATEWAY_PAUSE_BUDGET_SEC = 3600.0

# Rows record ``infra_reason=type(exc).__name__``; the wire layer's typed
# transport failure is the only circuit-class reason.  A test pins this string
# to ``cybergym_wire.GatewayTransportError.__name__`` — importing the class
# here would close an import cycle (wire <- adapter <- this module).
GATEWAY_TRANSPORT_INFRA_REASON = "GatewayTransportError"


class GatewayCircuitOpen(CyberGymError):
    """Dispatch halted: the isolate gateway is unreachable at transport level.

    Carries every row that landed before the breaker opened so the launcher
    can still account for each dispatched task; never-dispatched tasks are
    named in ``remaining_task_ids`` and deliberately have no result row.
    """

    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        threshold: int,
        remaining: Sequence[str],
        pause: Mapping[str, Any] | None = None,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.threshold = int(threshold)
        self.remaining_task_ids = [str(task_id) for task_id in remaining]
        self.pause = dict(pause or {})
        super().__init__(
            f"gateway unreachable: {self.threshold} consecutive transport "
            f"failures, {len(self.remaining_task_ids)} task(s) not dispatched"
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "outcome": "gateway_unreachable",
            "consecutive_transport_failures": self.threshold,
            "dispatched_rows": len(self.rows),
            "remaining_task_ids": list(self.remaining_task_ids),
        }
        if self.pause:
            payload["pause"] = dict(self.pause)
        return payload


def is_gateway_transport_row(row: Mapping[str, Any]) -> bool:
    """True only when the row proves the gateway itself could not be reached.

    Per-task infrastructure failures (container, workspace, generation) and
    gateway responses carrying a status or a malformed body are not
    circuit-class: the gateway demonstrably answered in those cases.
    """
    return (
        str(row.get("status") or "") == "infra_failed"
        and str(row.get("infra_reason") or "") == GATEWAY_TRANSPORT_INFRA_REASON
    )


class _Breaker:
    """Transport-failure streak -> pause-and-probe -> open, under one lock."""

    def __init__(
        self,
        *,
        threshold: int,
        probe: Callable[[], bool] | None,
        pause_budget_sec: float,
        backoff_sec: Sequence[float],
        clock: Callable[[], float],
        on_event: Callable[[Mapping[str, Any]], None] | None,
    ) -> None:
        self.threshold = int(threshold)
        self.probe = probe
        self.pause_budget_sec = float(pause_budget_sec)
        self.backoff_sec = tuple(float(value) for value in backoff_sec) or (30.0,)
        self.clock = clock
        self.on_event = on_event
        self.lock = threading.Lock()
        self.streak = 0
        self.open = False
        self.paused_since: float | None = None
        self.next_probe_at: float | None = None
        self.probe_failures = 0
        self.pauses: list[dict[str, Any]] = []

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(dict(event))
        except Exception:  # noqa: BLE001 - observers never steer dispatch
            pass

    @property
    def paused(self) -> bool:
        return self.paused_since is not None

    def admission_allowed(self) -> bool:
        with self.lock:
            return not self.open and self.paused_since is None

    def record(self, row: Mapping[str, Any]) -> None:
        with self.lock:
            if self.open:
                return
            self.streak = self.streak + 1 if is_gateway_transport_row(row) else 0
            if self.streak < self.threshold or self.paused_since is not None:
                return
            if self.probe is None:
                self.open = True
                return
            now = self.clock()
            self.paused_since = now
            self.probe_failures = 0
            self.next_probe_at = now + self.backoff_sec[0]
            event = {
                "event": "gateway_pause",
                "consecutive_transport_failures": self.streak,
                "first_probe_in_sec": self.backoff_sec[0],
                "pause_budget_sec": self.pause_budget_sec,
            }
        self._emit(event)

    def seconds_until_probe(self) -> float | None:
        with self.lock:
            if self.next_probe_at is None or self.open:
                return None
            return max(0.0, self.next_probe_at - self.clock())

    def tick(self) -> None:
        """Run one due liveness probe; resume admission or open the circuit."""

        with self.lock:
            if self.open or self.paused_since is None or self.next_probe_at is None:
                return
            now = self.clock()
            if now < self.next_probe_at:
                return
            probe = self.probe
        healthy = False
        try:
            healthy = bool(probe()) if probe is not None else False
        except Exception:  # noqa: BLE001 - a raising probe is an unhealthy gateway
            healthy = False
        with self.lock:
            if self.open or self.paused_since is None:
                return
            now = self.clock()
            paused_for = now - self.paused_since
            if healthy:
                summary = {
                    "event": "gateway_resume",
                    "paused_sec": round(paused_for, 3),
                    "failed_probes": self.probe_failures,
                }
                self.pauses.append({k: v for k, v in summary.items() if k != "event"})
                self.streak = 0
                self.paused_since = None
                self.next_probe_at = None
                self.probe_failures = 0
                event = summary
            else:
                self.probe_failures += 1
                if paused_for >= self.pause_budget_sec:
                    self.open = True
                    self.next_probe_at = None
                    event = {
                        "event": "gateway_circuit_open",
                        "paused_sec": round(paused_for, 3),
                        "failed_probes": self.probe_failures,
                    }
                    self.pauses.append({k: v for k, v in event.items() if k != "event"})
                else:
                    step = min(self.probe_failures, len(self.backoff_sec) - 1)
                    self.next_probe_at = now + self.backoff_sec[step]
                    event = {
                        "event": "gateway_probe_failed",
                        "paused_sec": round(paused_for, 3),
                        "failed_probes": self.probe_failures,
                        "next_probe_in_sec": self.backoff_sec[step],
                    }
        self._emit(event)

    def pause_summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                "pauses": [dict(item) for item in self.pauses],
                "pause_budget_sec": self.pause_budget_sec,
                "probe_backoff_sec": list(self.backoff_sec),
            }


def run_dispatched(
    tasks: Sequence[Any],
    run_one: Callable[[Any], dict[str, Any]],
    *,
    max_workers: int,
    threshold: int = GATEWAY_CIRCUIT_BREAKER_THRESHOLD,
    on_row: Callable[[Mapping[str, Any]], None] | None = None,
    gateway_probe: Callable[[], bool] | None = None,
    pause_budget_sec: float = GATEWAY_PAUSE_BUDGET_SEC,
    probe_backoff_sec: Sequence[float] = GATEWAY_PROBE_BACKOFF_SEC,
    on_event: Callable[[Mapping[str, Any]], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    """Run ``run_one`` over ``tasks``, stopping admission on a dead gateway.

    ``tasks`` are duck-typed ``TaskSpec`` values (importing the class would
    close the same import cycle).  The breaker counts consecutive circuit-class
    rows; at ``threshold`` it pauses admission and, when ``gateway_probe`` is
    given, probes the gateway on ``probe_backoff_sec`` until a healthy answer
    resumes admission or ``pause_budget_sec`` elapses and the circuit opens.
    Without a probe the circuit opens immediately.  Already-dispatched
    in-flight tasks always settle and their rows land, never-dispatched tasks
    get no row, and an open circuit fails the campaign with
    ``GatewayCircuitOpen``.  A healthy gateway changes nothing: every task is
    dispatched and rows keep task order.
    """

    breaker = _Breaker(
        threshold=threshold,
        probe=gateway_probe,
        pause_budget_sec=pause_budget_sec,
        backoff_sec=probe_backoff_sec,
        clock=clock,
        on_event=on_event,
    )

    def settle(rows: list[dict[str, Any]], submitted: int) -> list[dict[str, Any]]:
        if breaker.open:
            summary = breaker.pause_summary()
            raise GatewayCircuitOpen(
                rows=rows,
                threshold=threshold,
                remaining=[str(task.task_id) for task in tasks[submitted:]],
                pause=summary if summary["pauses"] else None,
            )
        return rows

    def wait_out_pause() -> None:
        """Block the admission loop while paused; returns when resumed or open."""

        while breaker.paused and not breaker.open:
            due_in = breaker.seconds_until_probe()
            if due_in is None:
                break
            if due_in > 0:
                sleep(due_in)
            breaker.tick()

    if max_workers == 1 or len(tasks) <= 1:
        rows: list[dict[str, Any]] = []
        submitted = 0
        for task in tasks:
            if breaker.paused:
                wait_out_pause()
            if breaker.open:
                break
            row = run_one(task)
            breaker.record(row)
            if on_row is not None:
                on_row(row)
            rows.append(row)
            submitted += 1
        return settle(rows, submitted)

    dispatched: dict[int, dict[str, Any]] = {}
    completed: dict[int, dict[str, Any]] = {}
    submitted = 0
    next_record = 0
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cybergym") as pool:
        in_flight: dict[Future[dict[str, Any]], int] = {}
        while True:
            # Admission is bounded by lanes actually running.  Rows waiting in
            # ``completed`` for an earlier position to settle must not hold a
            # lane: counting them made every window of ``max_workers`` tasks
            # wait for its slowest member (a 2 h deadline task idled 63 lanes
            # for up to 2 h — r7/r8 ran near single-digit effective
            # concurrency for long stretches).
            while (
                breaker.admission_allowed()
                and len(in_flight) < max_workers
                and submitted < len(tasks)
            ):
                in_flight[pool.submit(run_one, tasks[submitted])] = submitted
                submitted += 1
            if not in_flight:
                if breaker.paused and not breaker.open and submitted < len(tasks):
                    wait_out_pause()
                    continue
                break
            timeout = breaker.seconds_until_probe() if breaker.paused else None
            done, _pending = wait(
                tuple(in_flight), timeout=timeout, return_when=FIRST_COMPLETED
            )
            for future in done:
                position = in_flight.pop(future)
                row = future.result()
                # The breaker sees rows as they settle, so a transport failure
                # pauses admission immediately instead of waiting behind an
                # earlier long-running position.
                breaker.record(row)
                completed[position] = row
            # Source order is the campaign treatment: rows are recorded (and
            # result_index appended) strictly in task order, buffering
            # out-of-order completions until their predecessors settle.
            while next_record in completed:
                row = completed.pop(next_record)
                if on_row is not None:
                    on_row(row)
                dispatched[next_record] = row
                next_record += 1
            if breaker.paused:
                breaker.tick()
    return settle([dispatched[position] for position in sorted(dispatched)], submitted)
