"""The run-wide money ledger behind ``--total-budget``: reservation, admission, settlement and the dispatch order.

``run_live_lanes`` owns the lane pool; this module owns what the lanes may spend together. A lane's spend is read
from its own durable usage (``lane_spend``). Its known part is what the core counts toward a limit: confirmed prices
plus disclosed estimates (``_usage_money.exceeds_limit``, #1487), so an estimate binds and is never called final. It
is UNKNOWN when that read fails, when a lane whose server process was spawned has no usage record at all, when
attempts were sent without a final price, or when the core calls the lane's ledger degraded: never zero. A live lane
in that state keeps working under its reservation, and its read failure is never raised into another attempt's
admission. An ended lane always leaves the flight and
wakes its waiters, and stays charged ``max(reservation, known spend)``: the known part plus an unknown liability. So
does a lane that ended without proof its processes are gone, whatever its prices say (``note_stop_unproved``). An
attempt only that liability keeps out is refused as ``accounting_unavailable``, not as ``budget_cap``.
"""
from __future__ import annotations

import pathlib
import sqlite3
import threading
import time
from decimal import Decimal
from typing import Callable

from devtools.e2e_live.scenarios import SCENARIOS, now_iso

# A lane's TOTAL_BUDGET must stay POSITIVE: the runtime reads a non-positive value as "no finite
# global budget" (``settings_setup_contract.resolve_total_budget_usd``), the opposite of a cap.
LANE_BUDGET_FLOOR_USD = 0.01
RESERVATION_RULE = (f"max({LANE_BUDGET_FLOOR_USD:g}, per_task_usd x (root_tasks + 1 if --self-mod and the scenario absorbs else root_tasks)) "
                    "— the runtime fences each root task tree at OUROBOROS_PER_TASK_COST_USD; --self-mod adds one root for the post-task "
                    "cycle of a lane that promotes (SM1; SW1/SK1 pin it off). The lane's TOTAL_BUDGET is that reservation — the true fence")


class LaneSpendAbsent(LookupError):
    """Neither ``state/usage.sqlite`` nor the journal exists under a lane's data root: nothing on disk says what it
    spent. That is zero only for a lane that never spawned a server process; otherwise it is unknown."""


def lane_spend(data_root: pathlib.Path) -> tuple[float, int]:
    """``(known USD, attempts sent without a final price)`` from ``state/usage.sqlite``, the money authority (``llm_usage`` misses review/advisory/synthesis: 114.81 vs 141.63), or a journal-era ``--seed`` lane's journal; ``LaneSpendAbsent`` when neither exists.

    Known is every settled amount, a confirmed price or a disclosed estimate, as the core sums its ``settled`` bucket (``_usage_money.cash_contribution``), each decoded as the core decodes money (``_usage_money.amount``: a numeric string is money, as the core's own export writes a literal a float would change). Without a final price means an estimate or partial receipt (its amount is in the known sum, as a floor), a settlement at nothing, or an attempt sent and never settled (``dispatched``, ``unresolved``). A ``reserved`` or ``released`` attempt was never sent and costs nothing. A ledger the core calls degraded (``usage_store.integrity_degraded``: a journal row it once quarantined, whose money nobody can read) is never final, from the store or the journal: one more unknown, never a known zero.

    The store keeps one current row per attempt, decoded by its own codec (``usage_store._decode``: a value its typed column cannot hold, a numeric string, stays in ``extra``). The journal appends one per transition, so it is first reduced as the store's import reduces it: each attempt's latest row (``usage_ledger._final_rows``), the compaction header (no money) left out. Otherwise an estimate and the price that replaced it both counted. The journal is only read, never truncated or quarantined as the core's own reader does. A row it cannot read as a JSON object (torn at the tail or before it, undecodable, not an object: ``iter_jsonl_objects``' ``gap_reasons``), or one the core's row rule refuses (``usage_ledger.validate_row_fields``: a cost that is negative or not a number, ``UsageLedgerCorrupt``; a non-finite one, ``UsageNonFiniteMoney``), makes the whole reading unknown, never a skipped row: a torn one-shot row is its attempt's only record. So does a journal gone before it opens, never an empty read."""
    from ouroboros._usage_money import amount
    from ouroboros.usage_store import integrity_degraded

    store, journal = (pathlib.Path(data_root) / "state" / name for name in ("usage.sqlite", "usage_attempts.jsonl"))
    if store.is_file():
        from ouroboros.usage_store import _ATTEMPT_CAPTURE, _decode
        conn = sqlite3.connect(f"{store.resolve().as_uri()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            latest = [_decode(record) for record in conn.execute(
                f"SELECT {', '.join(_ATTEMPT_CAPTURE)} FROM attempts WHERE state IN ('settled', 'dispatched', 'unresolved')")]
        finally:
            conn.close()
    elif journal.is_file():
        from ouroboros.usage_ledger import UsageLedgerCorrupt, _final_rows, validate_row_fields
        from ouroboros.utils import iter_jsonl_objects
        gaps: set[str] = set()
        with journal.open("rb") as handle:                         # opened here: a vanished journal raises, not reads as empty
            records = [row for row in iter_jsonl_objects(journal, gap_reasons=gaps, _handle=handle) if row.get("kind") != "usage_baseline"]
        if gaps:
            raise UsageLedgerCorrupt(f"usage journal rows not readable as JSON objects ({', '.join(sorted(gaps))}): {journal}")
        for row in records:
            validate_row_fields(row, row.get("seq"))
        latest = [row for row in _final_rows(records).values() if row.get("state") in ("settled", "dispatched", "unresolved")]
    else:
        raise LaneSpendAbsent(f"no state/usage.sqlite or state/usage_attempts.jsonl under {data_root}")
    costs = [amount(row.get("cost_usd")) if row.get("state") == "settled" else None for row in latest]
    known = float(sum((cost for cost in costs if cost is not None), Decimal(0)))
    unpriced = sum(1 for cost, row in zip(costs, latest) if cost is None or row.get("cost_final") is not True)
    return known, unpriced + int(integrity_degraded(pathlib.Path(data_root)))


def lane_spend_fact(data_root: pathlib.Path, *, spawned: bool) -> dict:
    """The lane row's spend: the reading, a zero only for a lane that never spawned a server, else unknown (``None``, with why)."""
    try:
        spent, unknown = lane_spend(data_root)
        return {"spent_usd": spent, "unknown_cost_rows": unknown}
    except LaneSpendAbsent as exc:
        if not spawned:
            return {"spent_usd": 0.0, "unknown_cost_rows": 0}
        error: Exception = exc
    except Exception as exc:  # noqa: BLE001 - an unreadable store is an unknown spend, recorded, never zero
        error = exc
    return {"spent_usd": None, "unknown_cost_rows": None, "spend_unknown": f"{type(error).__name__}: {error}"[:300]}


class RunBudget:
    """The RUN-WIDE ledger behind ``--total-budget`` (the first paid run gave every lane the whole cap).

    Reservation rule, per attempt: ``per_task_usd x (root_tasks + int(self_mod and absorbs))`` — the runtime fences
    each ROOT task tree at ``OUROBOROS_PER_TASK_COST_USD`` and children spend under their root's ceiling, so SW1 (one
    root, two scouts) reserves for one root, SK1 (author + dispatch) for two, and ``--self-mod`` adds one root for
    the post-task cycle of a lane that promotes (SM1 only), all under the lane fence. Admission asks, PER attempt:
    ``spent + unknown liability + reservation > cap`` — it can NEVER fit, refused and recorded ``not_run`` (a later,
    smaller attempt is asked on its own; nothing halts the run); otherwise it reserves only when no earlier-dispatched
    attempt is still asking (FIFO by ``dispatch_index``, see ``admit``) and ``spent + liability + reserved(in flight)
    + reservation <= cap``, else it waits on the ledger and re-asks after every settle (the first paid run wrote
    SW1/SK1 off at t=+21 min while the blocker was two SM1 reservations in flight, not the cap). ``spent`` is re-read
    from the lanes' durable usage on every question (it grows while they spend, so a waiter can end refused — its
    ``waited_sec`` is recorded; a late final price may lower an estimate); a live lane whose read fails counts its last good reading, because its whole reservation is already
    counted in flight, and ``spend_unknown`` names it. Each lane's TOTAL_BUDGET is its OWN reservation (``ceiling``): the ceilings in flight are
    disjoint and their sum plus the settled spend never exceeds the cap. No ``halted`` flag: ``refusals`` lists every
    refused attempt's facts, ``first_refused`` the first.
    """

    def __init__(self, cap_usd: float, per_task_usd: float, reader: Callable[[pathlib.Path], tuple[float, int]] | None = None,
                 *, self_mod: bool = False) -> None:
        self.cap, self.per_task, self.self_mod = float(cap_usd), float(per_task_usd), bool(self_mod)
        self._read = reader or lane_spend
        self._lock = threading.Condition()                         # reserve/refuse/settle wake every waiting admission
        self._pending: dict[int, str] = {}                         # dispatch index -> attempt name, while asking
        self._live: dict[tuple, tuple[pathlib.Path, float]] = {}   # job -> (data root, reservation)
        self._final: dict[tuple, tuple[float, int, float]] = {}    # ended job -> (known spent, unknown rows, liability)
        self._last: dict[tuple, tuple[float, int]] = {}            # live job -> its last good reading
        self._spawned: set[tuple] = set()                          # jobs whose lane spawned a server process
        self._unstopped: set[tuple] = set()                        # jobs whose processes were not proven gone at the end
        self.spend_unknown: dict[str, dict] = {}                   # attempt -> why its spend is unknown right now
        self.refusals: list[dict] = []
        self.not_run: list[str] = []

    def reservation(self, root_tasks: int, absorbs: bool = False) -> float:
        """The ONE effective ceiling of an attempt (admission, the lane's TOTAL_BUDGET and the reports carry this
        same number): ``per_task_usd x (root_tasks + int(self_mod and absorbs))``, floored, never rounded up."""
        return max(LANE_BUDGET_FLOOR_USD, self.per_task * (max(1, int(root_tasks or 1)) + int(self.self_mod and absorbs)))

    def note_spawned(self, job: tuple) -> None:
        """The lane spawned a server process: from now on its missing usage record is unknown spend, not zero."""
        with self._lock:
            self._spawned.add(job)

    def note_stop_unproved(self, job: tuple) -> None:
        """The lane ended without proof that its processes are gone: one may still spend under its own TOTAL_BUDGET, so
        settlement keeps its reservation charged even when every price read so far is final."""
        with self._lock:
            self._unstopped.add(job)

    def _mark_unknown(self, job: tuple, why: str) -> None:
        self.spend_unknown[f"{job[0]}_a{job[1]}"] = {"reason": why[:300], "at": now_iso()}

    def _reading(self, job: tuple, root: pathlib.Path) -> tuple[float, int]:
        """A live lane's spend for admission: a failed read is recorded and answers its last good reading."""
        try:
            self._last[job] = self._read(root)
            self.spend_unknown.pop(f"{job[0]}_a{job[1]}", None)
        except LaneSpendAbsent:
            pass                                                   # a booting lane has not written its store yet
        except Exception as exc:  # noqa: BLE001 - never raised into another attempt's question
            self._mark_unknown(job, f"read failed in flight: {type(exc).__name__}: {exc}")
        return self._last.get(job, (0.0, 0))

    def _spent_locked(self) -> tuple[float, int, float]:
        ended = list(self._final.values())
        rows = [(usd, n) for usd, n, _charge in ended] + [self._reading(job, root) for job, (root, _reserved) in self._live.items()]
        return sum((usd for usd, _n in rows), 0.0), sum(n for _usd, n in rows), sum((charge for *_known, charge in ended), 0.0)

    def _reserved_locked(self) -> float:
        return sum(reserved for _root, reserved in self._live.values())

    def admit(self, job: tuple, root_tasks: int, data_root: pathlib.Path, *, dispatch_index: int,
              on_wait: Callable[[str], None] | None = None, absorbs: bool = False) -> tuple[bool, dict]:
        """FIFO by ``dispatch_index`` (``dispatch_order``'s position): an attempt reserves only when no earlier-
        dispatched attempt is still asking and the cap has room beside the reservations in flight; one that can
        never fit is refused at once and leaves the line. Without the line the freed lane's NEXT job took the lock
        before the woken waiter every time (300/300: settle in ``run_attempt``'s finally, return, the executor's
        next job, ``admit`` on the same thread), so a round overflowing the cap let later attempts leapfrog the
        waiter. No deadlock: a waiter waits only while something is in flight or an earlier attempt is asking; the
        earliest never waits on the line and, with nothing in flight, either fits or is refused; reserve, refusal
        and settle wake every waiter, the line predicate re-parks the later ones. Cost: a large head can idle lanes
        a smaller attempt would use. ``on_wait`` runs UNDER the budget lock (it must not touch the budget), told
        once, with the reason the wait begins with; ``facts["waited_sec"]`` is how long. A refusal's facts carry its
        ``reason``: ``budget_cap``, or ``accounting_unavailable`` when the known spend alone would have fitted."""
        need, name, waited_from = self.reservation(root_tasks, absorbs), f"{job[0]}_a{job[1]}", None
        with self._lock:
            self._pending[dispatch_index] = name
            try:
                while True:
                    spent, unknown, liability = self._spent_locked()
                    reserved = self._reserved_locked()
                    waited = round(time.monotonic() - waited_from, 3) if waited_from is not None else 0.0
                    facts = {"cap_usd": self.cap, "spent_usd": round(spent, 4), "unknown_liability_usd": round(liability, 4),
                             "reserved_usd": round(reserved, 4), "reservation_usd": need, "unknown_cost_rows": unknown,
                             "waited_sec": waited}
                    head = min(self._pending)                      # the earliest attempt still asking
                    if spent + liability + need > self.cap:        # can never fit: refused, this attempt only
                        reason = "budget_cap" if spent + need > self.cap else "accounting_unavailable"
                        self.refusals.append({"attempt": name, "reason": reason, "at": now_iso(), **facts})
                        self.not_run.append(name)
                        return False, {**facts, "reason": reason}
                    if head < dispatch_index or (reserved > 0 and spent + liability + reserved + need > self.cap):
                        if waited_from is None:
                            waited_from = time.monotonic()
                            if on_wait is not None:
                                on_wait((f"waiting — behind {self._pending[head]} in dispatch order" if head < dispatch_index
                                         else f"waiting — in flight reserved ${reserved:.2f}")
                                        + f", needs ${need:.2f}, spent ${spent:.2f}, cap ${self.cap:.2f}")
                        self._lock.wait()
                        continue
                    self._live[job] = (pathlib.Path(data_root), need)
                    return True, facts
            finally:                                               # out of the line, even on a reader error: the next may reserve
                del self._pending[dispatch_index]
                self._lock.notify_all()

    def ceiling(self, job: tuple) -> float:
        """The lane's TOTAL_BUDGET: its own reservation — immutable, disjoint from every other
        lane's, never non-positive (the runtime reads a non-positive budget as NO cap)."""
        with self._lock:
            entry = self._live.get(job)
            return entry[1] if entry else LANE_BUDGET_FLOOR_USD   # exactly what admission reserved

    def settle(self, job: tuple) -> None:
        """The lane ended: out of flight on EVERY path (an entry left live would hold its reservation forever and
        park a waiter no settle wakes), its final reading kept, and every waiter woken even when that read fails."""
        with self._lock:
            try:
                entry = self._live.pop(job, None)
                if entry is not None:
                    self._final[job] = self._final_reading(job, *entry)
            finally:
                self._lock.notify_all()                            # every waiter re-asks against the new ledger

    def _final_reading(self, job: tuple, root: pathlib.Path, reserved: float) -> tuple[float, int, float]:
        """``(known spent, unknown rows, liability)``: an unknown spend keeps ``max(reservation, known)`` charged, the
        known part being the reading, or the last good one when this read fails. A process that may still run makes
        the spend unknown whatever the read says: it can go on spending after it."""
        known = self._last.pop(job, (0.0, 0))
        try:
            known = self._read(root)
            why = f"{known[1]} attempts without a final price" if known[1] else ""
        except LaneSpendAbsent as exc:
            why = f"{type(exc).__name__}: {exc}" if job in self._spawned else ""   # no server process: nothing could spend
        except Exception as exc:  # noqa: BLE001 - an unknown spend, charged conservatively, never zero
            why = f"read failed at settlement: {type(exc).__name__}: {exc}"
        if job in self._unstopped:
            why = "; ".join(filter(None, ("server stop unproved: a process may still spend", why)))
        self.spend_unknown.pop(f"{job[0]}_a{job[1]}", None)
        if not why:
            return known[0], known[1], 0.0
        self._mark_unknown(job, why)
        return known[0], known[1], max(float(reserved) - known[0], 0.0)

    def snapshot(self) -> dict:
        with self._lock:
            spent, unknown, liability = self._spent_locked()
            return {"cap_usd": self.cap, "per_task_usd": self.per_task, "reservation_rule": RESERVATION_RULE,
                    "spent_usd": round(spent, 4), "unknown_liability_usd": round(liability, 4),
                    "reserved_usd": round(self._reserved_locked(), 4), "unknown_cost_rows": unknown,
                    "lanes_in_flight": len(self._live), "lanes_settled": len(self._final),
                    "spend_unknown": {name: dict(fact) for name, fact in self.spend_unknown.items()},
                    "refusals": [dict(r) for r in self.refusals],
                    "first_refused": self.refusals[0]["attempt"] if self.refusals else None,
                    "attempts_not_run": list(self.not_run)}


def budget_preflight(budget: RunBudget, scenario_ids: list[str], attempts: int, lanes: int) -> dict:
    """The reservation arithmetic BEFORE any lane spends, in ``credit_preflight``'s typed shape: a row per
    scenario, the worst case ``sum(reservation x attempts)`` against the cap, the per-ROUND worst case (the
    ``lanes`` largest reservations, ONE attempt per scenario — above the cap the round's last lane WAITS on
    a settle, no refusal; with ``lanes`` above the scenario count it understates what admission can put in
    flight) and ``unreachable``: a reservation above the cap, or equal to it with attempts >= 2 (the second
    can never be admitted after any spend). No override flag: the operator changes the flags."""
    rows = []
    for sid in scenario_ids:
        need = budget.reservation(SCENARIOS[sid].root_tasks, SCENARIOS[sid].expects_absorb)
        rows.append({"scenario": sid, "root_tasks": SCENARIOS[sid].root_tasks, "reservation_usd": need,
                     "attempts": int(attempts), "worst_case_usd": round(need * attempts, 4),
                     "unreachable": need > budget.cap or (attempts >= 2 and need >= budget.cap)})
    return {"cap_usd": budget.cap, "per_task_usd": budget.per_task, "self_mod": budget.self_mod,
            "reservation_rule": RESERVATION_RULE, "scenarios": rows,
            "worst_case_usd": round(sum(r["worst_case_usd"] for r in rows), 4), "lanes": int(lanes),
            "round_worst_case_usd": round(sum(sorted((r["reservation_usd"] for r in rows), reverse=True)[:int(lanes)]), 4),
            "unreachable": [r["scenario"] for r in rows if r["unreachable"]]}


def dispatch_order(budget: RunBudget, requested: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Round-robin by attempt (a1 of every scenario, then a2, ...), largest reservation first within a round
    (stable among equals); admission keeps this order (``RunBudget.admit``). The verdict is pass-of PER
    scenario, so the MINIMUM admitted attempts per scenario is what the order protects, not the sum."""
    return sorted(requested, key=lambda job: (job[1], -budget.reservation(SCENARIOS[job[0]].root_tasks, SCENARIOS[job[0]].expects_absorb)))
