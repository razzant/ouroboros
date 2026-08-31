"""CyberGym campaign dispatch engine: bounded fan-out plus a dead-gateway breaker.

Extracted from ``cybergym_adapter.run_campaign`` so the stateful adapter stays
inside its module-size band.  This module owns only dispatch policy: it never
touches the budget ledger, the result index, workspaces, or containers.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from devtools.benchmarks.cybergym.cybergym_protocol import CyberGymError

# Consecutive transport-class gateway failures that prove the isolate is dead
# and open the dispatch circuit breaker.  Small on purpose: a healthy gateway
# never produces even one transport failure, so three in a row is already a
# deterministic dead-transport signal (run 3 burned 234 tasks without this).
GATEWAY_CIRCUIT_BREAKER_THRESHOLD = 3

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
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.threshold = int(threshold)
        self.remaining_task_ids = [str(task_id) for task_id in remaining]
        super().__init__(
            f"gateway unreachable: {self.threshold} consecutive transport "
            f"failures, {len(self.remaining_task_ids)} task(s) not dispatched"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": "gateway_unreachable",
            "consecutive_transport_failures": self.threshold,
            "dispatched_rows": len(self.rows),
            "remaining_task_ids": list(self.remaining_task_ids),
        }


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


def run_dispatched(
    tasks: Sequence[Any],
    run_one: Callable[[Any], dict[str, Any]],
    *,
    max_workers: int,
    threshold: int = GATEWAY_CIRCUIT_BREAKER_THRESHOLD,
) -> list[dict[str, Any]]:
    """Run ``run_one`` over ``tasks``, stopping admission on a dead gateway.

    ``tasks`` are duck-typed ``TaskSpec`` values (importing the class would
    close the same import cycle).  The breaker counts consecutive circuit-class
    rows and opens at ``threshold``; already-dispatched in-flight tasks always
    settle and their rows land, while never-dispatched tasks get no row and the
    campaign fails fast with ``GatewayCircuitOpen``.  A healthy gateway changes
    nothing: every task is dispatched and rows keep task order.
    """

    circuit_open = threading.Event()
    streak: list[int] = [0]
    streak_lock = threading.Lock()

    def record(row: Mapping[str, Any]) -> None:
        with streak_lock:
            streak[0] = streak[0] + 1 if is_gateway_transport_row(row) else 0
            if streak[0] >= threshold:
                circuit_open.set()

    def settle(rows: list[dict[str, Any]], submitted: int) -> list[dict[str, Any]]:
        if circuit_open.is_set():
            raise GatewayCircuitOpen(
                rows=rows,
                threshold=threshold,
                remaining=[str(task.task_id) for task in tasks[submitted:]],
            )
        return rows

    if max_workers == 1 or len(tasks) <= 1:
        rows: list[dict[str, Any]] = []
        submitted = 0
        for task in tasks:
            if circuit_open.is_set():
                break
            row = run_one(task)
            record(row)
            rows.append(row)
            submitted += 1
        return settle(rows, submitted)

    dispatched: dict[int, dict[str, Any]] = {}
    submitted = 0
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cybergym") as pool:
        in_flight: dict[Future[dict[str, Any]], int] = {}
        while True:
            while (
                not circuit_open.is_set()
                and len(in_flight) < max_workers
                and submitted < len(tasks)
            ):
                in_flight[pool.submit(run_one, tasks[submitted])] = submitted
                submitted += 1
            if not in_flight:
                break
            done, _pending = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            # Record a simultaneous batch in task order so the streak is
            # deterministic and matches the sequential lane's semantics.
            finished = sorted((in_flight.pop(future), future) for future in done)
            for position, future in finished:
                row = future.result()
                record(row)
                dispatched[position] = row
    return settle([dispatched[position] for position in sorted(dispatched)], submitted)
