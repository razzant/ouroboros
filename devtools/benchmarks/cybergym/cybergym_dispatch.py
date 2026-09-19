"""CyberGym campaign dispatch engine: bounded fan-out plus its admission gates.

Extracted from ``cybergym_adapter.run_campaign`` so the stateful adapter stays
inside its module-size band.  This module owns only dispatch policy: it never
touches the budget ledger, the result index, workspaces, or containers.  Three
gates can pause admission — a dead gateway, a refused budget claim, and a
sibling's unresolved workspace startup — and each ends a campaign with a typed
stop whose never-dispatched task ids stay row-free.
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

# A sibling's unresolved pre-gateway workspace pauses admission instead of
# turning innocent tasks into infra rows (the 2026-09-13 campaign turned 11
# failed starts into 12 collateral rows).  The gate re-runs the executor's
# healer every 30 s; a contiguous five-minute pause without a clean heal stops
# the campaign.  This logical budget is its own rail, distinct from the Docker
# command timeout, the gateway transport budget, the task deadline, the
# finalization grace, and the campaign budget.
WORKSPACE_CUSTODY_BUDGET_SEC = 300.0
WORKSPACE_CUSTODY_PROBE_INTERVAL_SEC = 30.0


class _DispatchStop(CyberGymError):
    """Admission halted: every row that landed plus the row-free remainder.

    The launcher still accounts for each dispatched task; never-dispatched
    tasks are named in ``remaining_task_ids`` and deliberately have no result
    row, so a later append-only campaign runs them without any retry flag.
    """

    outcome = ""
    reason = ""

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
        super().__init__(f"{self.reason}: {len(self.remaining_task_ids)} task(s) not dispatched")

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "outcome": self.outcome,
            "dispatched_rows": len(self.rows),
            "remaining_task_ids": list(self.remaining_task_ids),
        }
        if self.pause:
            payload["pause"] = dict(self.pause)
        return payload


class GatewayCircuitOpen(_DispatchStop):
    """Dispatch halted: the isolate gateway is unreachable at transport level."""

    outcome = "gateway_unreachable"

    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        threshold: int,
        remaining: Sequence[str],
        pause: Mapping[str, Any] | None = None,
    ) -> None:
        self.threshold = int(threshold)
        self.reason = f"gateway unreachable after {self.threshold} consecutive transport failures"
        super().__init__(rows=rows, remaining=remaining, pause=pause)

    def as_dict(self) -> dict[str, Any]:
        return {**super().as_dict(), "consecutive_transport_failures": self.threshold}


class WorkspaceCustodyPending(CyberGymError):
    """Zero-send pause: a sibling's pre-gateway workspace custody is unresolved.

    Raised before gateway admission by an attempt that met another start's
    latch, never by the failed start itself (which keeps its honest infra
    row).  ``run_campaign`` durably releases the attempt's claim and re-raises;
    the dispatcher requeues the task row-free and pauses the custody gate.
    """

    def __init__(self, unresolved: Mapping[str, str]) -> None:
        self.unresolved = {str(name): str(reason) for name, reason in unresolved.items()}
        super().__init__(
            "workspace startup custody is unresolved: " + ", ".join(sorted(self.unresolved))
        )


class WorkspaceCustodyTimeout(_DispatchStop):
    """Dispatch halted: startup custody stayed unresolved for the whole budget."""

    outcome = "workspace_custody_timeout"
    reason = "workspace startup custody stayed unresolved"


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


class BudgetCapReached(_DispatchStop):
    """Dispatch halted: the budget projection refuses every further claim.

    Raised only after admission paused on a claim refusal and no in-flight
    settlement freed enough headroom for the next reservation (or the caller
    supplied no probe at all).
    """

    outcome = "budget_cap_reached"
    reason = "campaign budget cap reached"


class _Breaker:
    """Failure signal -> pause-and-probe -> open, under one lock.

    One timed gate serves two rails: the gateway breaker trips on a streak of
    transport rows (``record``), the workspace-custody gate on a zero-send
    custody signal (``trip``).  Further signals never reset a live pause; only
    a healthy probe resumes admission, and a contiguous pause that exhausts
    ``pause_budget_sec`` opens the gate for good.  A ``hard_budget`` (the
    custody gate) is a deadline: a probe that turns healthy only at or after
    it cannot resume, while the gateway still resumes on any healthy probe.
    """

    def __init__(
        self,
        *,
        kind: str,
        open_event: str,
        threshold: int,
        probe: Callable[[], bool] | None,
        pause_budget_sec: float,
        hard_budget: bool = False,
        backoff_sec: Sequence[float],
        clock: Callable[[], float],
        on_event: Callable[[Mapping[str, Any]], None] | None,
    ) -> None:
        self.kind = kind
        self.open_event = open_event
        self.threshold = int(threshold)
        self.probe = probe
        self.pause_budget_sec = float(pause_budget_sec)
        self.hard_budget = bool(hard_budget)
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

    def _emit(self, event: dict[str, Any] | None) -> None:
        if self.on_event is None or event is None:
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
            if self.streak < self.threshold:
                return
            event = self._trip_locked({"consecutive_transport_failures": self.streak})
        self._emit(event)

    def trip(self, detail: Mapping[str, Any]) -> None:
        with self.lock:
            event = self._trip_locked(detail)
        self._emit(event)

    def _trip_locked(self, detail: Mapping[str, Any]) -> dict[str, Any] | None:
        if self.open or self.paused_since is not None:
            return None
        if self.probe is None:
            self.open = True
            return None
        now = self.clock()
        self.paused_since = now
        self.probe_failures = 0
        self.next_probe_at = now + self.backoff_sec[0]
        return {
            "event": f"{self.kind}_pause",
            **dict(detail),
            "first_probe_in_sec": self.backoff_sec[0],
            "pause_budget_sec": self.pause_budget_sec,
        }

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
            expired = paused_for >= self.pause_budget_sec
            if healthy and not (self.hard_budget and expired):
                summary = {
                    "event": f"{self.kind}_resume",
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
                # A heal landing at a hard deadline is not a failed probe.
                if not healthy:
                    self.probe_failures += 1
                if expired:
                    self.open = True
                    self.next_probe_at = None
                    event = {
                        "event": self.open_event,
                        "paused_sec": round(paused_for, 3),
                        "failed_probes": self.probe_failures,
                    }
                    self.pauses.append({k: v for k, v in event.items() if k != "event"})
                else:
                    step = min(self.probe_failures, len(self.backoff_sec) - 1)
                    self.next_probe_at = now + self.backoff_sec[step]
                    event = {
                        "event": f"{self.kind}_probe_failed",
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
    custody_probe: Callable[[], bool] | None = None,
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

    ``WorkspaceCustodyPending`` is the third: a sibling's pre-gateway start
    left unresolved custody, so this zero-send attempt (its claim already
    released) is re-queued row-free and admission pauses.  ``custody_probe``
    re-runs the executor's healer every ``WORKSPACE_CUSTODY_PROBE_INTERVAL_SEC``
    and a clean heal before the deadline resumes admission; a contiguous pause
    that reaches ``WORKSPACE_CUSTODY_BUDGET_SEC`` (at once without a probe)
    drains in-flight work and ends the campaign with
    ``WorkspaceCustodyTimeout``, even if its last probe healed.  When stops
    coincide, custody precedes the budget, which precedes the gateway.
    """

    breaker = _Breaker(
        kind="gateway",
        open_event="gateway_circuit_open",
        threshold=threshold,
        probe=gateway_probe,
        pause_budget_sec=pause_budget_sec,
        backoff_sec=probe_backoff_sec,
        clock=clock,
        on_event=on_event,
    )
    custody = _Breaker(
        kind="workspace_custody",
        open_event="workspace_custody_timeout",
        threshold=1,
        probe=custody_probe,
        pause_budget_sec=WORKSPACE_CUSTODY_BUDGET_SEC,
        hard_budget=True,
        backoff_sec=(WORKSPACE_CUSTODY_PROBE_INTERVAL_SEC,),
        clock=clock,
        on_event=on_event,
    )
    gate = _BudgetGate(probe=budget_probe, clock=clock, on_event=on_event)
    # Never-dispatched positions (budget refusals, custody pauses) are
    # re-admitted in source order before fresh ones, so a pause never
    # reorders work; while unadmitted they are part of the row-free remainder.
    requeued: list[int] = []

    def remaining_ids(submitted: int) -> list[str]:
        return [str(tasks[position].task_id) for position in sorted(requeued)] + [
            str(task.task_id) for task in tasks[submitted:]
        ]

    def settle(rows: list[dict[str, Any]], submitted: int) -> list[dict[str, Any]]:
        if custody.open:
            summary = custody.pause_summary()
            raise WorkspaceCustodyTimeout(
                rows=rows,
                remaining=remaining_ids(submitted),
                pause=summary if summary["pauses"] else None,
            )
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

    def wait_out(paused: _Breaker) -> None:
        """Block the admission loop while paused; returns when resumed or open."""

        while paused.paused and not paused.open:
            due_in = paused.seconds_until_probe()
            if due_in is None:
                break
            if due_in > 0:
                sleep(due_in)
            paused.tick()

    if max_workers == 1 or len(tasks) <= 1:
        rows: list[dict[str, Any]] = []
        position = 0
        while position < len(tasks):
            for paused in (custody, breaker):
                if paused.paused:
                    wait_out(paused)
            if breaker.open or gate.closed or custody.open:
                break
            try:
                row = run_one(tasks[position])
            except WorkspaceCustodyPending as exc:
                # Zero-send and already released: retry this same position
                # once the custody gate resumes.
                custody.trip({"unresolved": dict(exc.unresolved)})
                continue
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
            position += 1
        return settle(rows, position)

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
            requeued.sort()
            while (
                breaker.admission_allowed()
                and custody.admission_allowed()
                and gate.admission_allowed()
                and len(in_flight) < max_workers
                and (requeued or submitted < len(tasks))
            ):
                if requeued:
                    position = requeued.pop(0)
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
                    continue
                if gate.closed or custody.open or breaker.open:
                    break
                # A requeued last position is still undispatched work: wait
                # out the pause for it instead of ending the campaign early.
                paused = next((item for item in (custody, breaker) if item.paused), None)
                if paused is not None and (requeued or submitted < len(tasks)):
                    wait_out(paused)
                    continue
                break
            due = [item.seconds_until_probe() for item in (custody, breaker) if item.paused]
            due = [value for value in due if value is not None]
            done, _pending = wait(
                tuple(in_flight), timeout=min(due) if due else None, return_when=FIRST_COMPLETED
            )
            newly_completed: list[dict[str, Any]] = []
            for future in done:
                position = in_flight.pop(future)
                try:
                    row = future.result()
                except WorkspaceCustodyPending as exc:
                    # Zero-send and already released: requeue row-free while
                    # the custody gate pauses (a live pause is never reset).
                    requeued.append(position)
                    custody.trip({"unresolved": dict(exc.unresolved)})
                    continue
                except Exception as exc:
                    if not _is_budget_refusal(exc):
                        raise
                    # Claim refused before dispatch: the task was never
                    # attempted, gets no row, and is re-admitted once a
                    # settlement frees headroom under the cap.
                    requeued.append(position)
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
            # Source order remains a reporting/provenance property.  The
            # caller receives a stable ordered sequence after all completed
            # rows have already been settled.
            while next_record in completed:
                row = completed.pop(next_record)
                dispatched[next_record] = row
                next_record += 1
            for paused in (custody, breaker):
                if paused.paused:
                    paused.tick()
    # A requeued position never produces a row, so the source-order drain
    # above can strand later completed rows behind that row-less position in
    # ``completed``.  The campaign's landed rows are the union of both maps
    # (they are disjoint: ``dispatched`` is drained FROM ``completed``), still
    # reported in source order.
    landed = {**completed, **dispatched}
    return settle([landed[position] for position in sorted(landed)], submitted)
