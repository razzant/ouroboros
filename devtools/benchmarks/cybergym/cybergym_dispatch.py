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


# ``run_one`` signals a claim-time budget refusal by raising; the adapter's
# ledger error class is pinned by name for the same import-cycle reason as
# GATEWAY_TRANSPORT_INFRA_REASON (wire <- adapter <- this module).  A test
# pins this string to ``cybergym_adapter.BudgetRefused.__name__``.  The exact
# class name is matched, so the ``BudgetOverspend`` subclass — a settlement
# condition that already has a row path — never trips this gate.
BUDGET_REFUSED_ERROR_NAME = "BudgetRefused"


class BudgetCapReached(CyberGymError):
    """Dispatch halted: the budget projection refuses every further claim.

    Raised only after admission paused on a claim refusal and no in-flight
    settlement freed enough headroom for the next reservation (or the caller
    supplied no probe at all).  Carries every row that landed before the stop
    so the launcher can still account for each dispatched task;
    never-dispatched tasks are named in ``remaining_task_ids`` and
    deliberately have no result row, so a later resume campaign re-runs them
    without any retry flag.
    """

    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        remaining: Sequence[str],
        pause: Mapping[str, Any] | None = None,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.remaining_task_ids = [str(task_id) for task_id in remaining]
        self.pause = dict(pause or {})
        super().__init__(
            "campaign budget cap reached: "
            f"{len(self.remaining_task_ids)} task(s) not dispatched"
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "outcome": "budget_cap_reached",
            "dispatched_rows": len(self.rows),
            "remaining_task_ids": list(self.remaining_task_ids),
        }
        if self.pause:
            payload["pause"] = dict(self.pause)
        return payload


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


class _BudgetGate:
    """Claim refusal -> pause-and-probe -> closed, under one lock.

    Unlike the transport breaker there is no backoff clock: the only event
    that can free headroom is an in-flight attempt settling below its
    reservation, so probes run exactly after settlements (and once more when
    the pool has drained) instead of on a timer.  Without a probe the first
    refusal closes the gate terminally — admission stops, in-flight work
    still drains, and the campaign ends with ``BudgetCapReached``.
    """

    def __init__(
        self,
        *,
        probe: Callable[[], bool] | None,
        clock: Callable[[], float],
        on_event: Callable[[Mapping[str, Any]], None] | None,
    ) -> None:
        self.probe = probe
        self.clock = clock
        self.on_event = on_event
        self.lock = threading.Lock()
        self.paused_since: float | None = None
        self.closed = False
        self.refusals = 0
        self.probes = 0
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
        return self.paused_since is not None and not self.closed

    def admission_allowed(self) -> bool:
        with self.lock:
            return not self.closed and self.paused_since is None

    def record_refusal(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.refusals += 1
            if self.probe is None:
                self.closed = True
                event = {
                    "event": "budget_gate_closed",
                    "reason": "no_probe",
                    "refusals": self.refusals,
                }
            elif self.paused_since is None:
                self.paused_since = self.clock()
                event = {"event": "budget_pause", "refusals": self.refusals}
            else:
                # Already paused; further refusals just re-queue the task.
                return
        self._emit(event)

    def probe_now(self) -> None:
        """Probe after a settlement; resume admission when headroom reappeared."""

        with self.lock:
            if self.closed or self.paused_since is None:
                return
            probe = self.probe
        freed = False
        try:
            freed = bool(probe()) if probe is not None else False
        except Exception:  # noqa: BLE001 - a raising probe is not freed budget
            freed = False
        with self.lock:
            if self.closed or self.paused_since is None:
                return
            self.probes += 1
            if freed:
                event = {
                    "event": "budget_resume",
                    "paused_sec": round(self.clock() - self.paused_since, 3),
                    "probes": self.probes,
                    "refusals": self.refusals,
                }
                self.pauses.append({k: v for k, v in event.items() if k != "event"})
                self.paused_since = None
            else:
                event = {
                    "event": "budget_probe_waiting",
                    "probes": self.probes,
                    "refusals": self.refusals,
                }
        self._emit(event)

    def close(self) -> None:
        """Terminally close once the pool drained with the cap still refusing."""

        with self.lock:
            if self.closed:
                return
            self.closed = True
            event: dict[str, Any] = {
                "event": "budget_gate_closed",
                "reason": "cap_exhausted",
                "refusals": self.refusals,
                "probes": self.probes,
            }
            if self.paused_since is not None:
                event["paused_sec"] = round(self.clock() - self.paused_since, 3)
                self.pauses.append({k: v for k, v in event.items() if k != "event"})
                self.paused_since = None
        self._emit(event)

    def pause_summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                "pauses": [dict(item) for item in self.pauses],
                "refusals": self.refusals,
                "probes": self.probes,
            }


def _is_budget_refusal(exc: BaseException) -> bool:
    """True for the adapter's claim-time budget refusal, matched by exact name."""

    return type(exc).__name__ == BUDGET_REFUSED_ERROR_NAME


def run_dispatched(
    tasks: Sequence[Any],
    run_one: Callable[[Any], dict[str, Any]],
    *,
    max_workers: int,
    threshold: int = GATEWAY_CIRCUIT_BREAKER_THRESHOLD,
    on_row: Callable[[Mapping[str, Any]], None] | None = None,
    gateway_probe: Callable[[], bool] | None = None,
    budget_probe: Callable[[], bool] | None = None,
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
    ``GatewayCircuitOpen``.  ``on_row`` runs in completion order so its caller
    can durably settle a finished attempt before another task is admitted.
    The returned rows retain source order for reproducible reporting.

    A claim-time budget refusal raised by ``run_one`` is the second stop
    class: the task was never dispatched, so it gets no row and is re-queued.
    Admission then pauses while in-flight attempts settle below their
    reservations; ``budget_probe`` runs after each settlement and resumes
    admission once a further claim fits the cap.  When the pool has drained
    and the probe still refuses, the campaign ends with ``BudgetCapReached``
    and the undispatched ids stay row-free for a later resume campaign.
    """

    breaker = _Breaker(
        threshold=threshold,
        probe=gateway_probe,
        pause_budget_sec=pause_budget_sec,
        backoff_sec=probe_backoff_sec,
        clock=clock,
        on_event=on_event,
    )
    gate = _BudgetGate(probe=budget_probe, clock=clock, on_event=on_event)
    refused: list[int] = []

    def remaining_ids(submitted: int) -> list[str]:
        return [str(tasks[position].task_id) for position in sorted(refused)] + [
            str(task.task_id) for task in tasks[submitted:]
        ]

    def settle(rows: list[dict[str, Any]], submitted: int) -> list[dict[str, Any]]:
        if gate.closed:
            summary = gate.pause_summary()
            raise BudgetCapReached(
                rows=rows,
                remaining=remaining_ids(submitted),
                pause=summary if summary["refusals"] else None,
            )
        if breaker.open:
            summary = breaker.pause_summary()
            raise GatewayCircuitOpen(
                rows=rows,
                threshold=threshold,
                remaining=remaining_ids(submitted),
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
            if breaker.open or gate.closed:
                break
            try:
                row = run_one(task)
            except Exception as exc:
                if not _is_budget_refusal(exc):
                    raise
                # A serial lane holds no in-flight work whose settlement
                # could free headroom later, so a refusal is terminal here.
                gate.record_refusal()
                gate.close()
                break
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
            # concurrency for long stretches).  Budget-refused positions are
            # re-admitted before fresh ones so a pause never reorders work.
            while (
                breaker.admission_allowed()
                and gate.admission_allowed()
                and len(in_flight) < max_workers
                and (refused or submitted < len(tasks))
            ):
                if refused:
                    position = refused.pop(0)
                else:
                    position = submitted
                    submitted += 1
                in_flight[pool.submit(run_one, tasks[position])] = position
            if not in_flight:
                if gate.paused:
                    # Nothing in flight can free headroom any more: one final
                    # probe decides between resume and a terminal stop.
                    gate.probe_now()
                    if gate.paused:
                        gate.close()
                        break
                    refused.sort()
                    continue
                if gate.closed:
                    break
                if breaker.paused and not breaker.open and submitted < len(tasks):
                    wait_out_pause()
                    continue
                break
            timeout = breaker.seconds_until_probe() if breaker.paused else None
            done, _pending = wait(
                tuple(in_flight), timeout=timeout, return_when=FIRST_COMPLETED
            )
            newly_completed: list[dict[str, Any]] = []
            for future in done:
                position = in_flight.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    if not _is_budget_refusal(exc):
                        raise
                    # Claim refused before dispatch: the task was never
                    # attempted, gets no row, and is re-admitted once a
                    # settlement frees headroom under the cap.
                    refused.append(position)
                    gate.record_refusal()
                    continue
                # The breaker sees rows as they settle, so a transport failure
                # pauses admission immediately instead of waiting behind an
                # earlier long-running position.
                breaker.record(row)
                completed[position] = row
                newly_completed.append(row)
            # Durably settle each finished attempt before refilling a lane.
            # Keeping settlement behind source-order reporting held completed
            # attempts' budget reservations while the freed lanes admitted
            # replacements, eventually refusing undispatched tasks despite
            # ample actual budget headroom.
            if on_row is not None:
                for row in newly_completed:
                    on_row(row)
            # Settlements are the only event that can free budget headroom,
            # so the budget probe runs exactly here instead of on a timer.
            if gate.paused:
                gate.probe_now()
                if not gate.paused:
                    refused.sort()
            # Source order remains a reporting/provenance property.  The
            # caller receives a stable ordered sequence after all completed
            # rows have already been settled.
            while next_record in completed:
                row = completed.pop(next_record)
                dispatched[next_record] = row
                next_record += 1
            if breaker.paused:
                breaker.tick()
    return settle([dispatched[position] for position in sorted(dispatched)], submitted)
