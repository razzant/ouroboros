"""Usage and terminal-state projections for delegated runs and review seats."""

from __future__ import annotations

import hashlib
import logging
import pathlib
import time
from contextlib import contextmanager
# Standard library only: scripts/safe_test.py imports this chain without site-packages.
from typing import Any, Dict, Literal, Optional, Tuple, TypedDict

from ouroboros._usage_response import provider_cost_value

log = logging.getLogger(__name__)


class _CostCustodyRequired(TypedDict):
    status: Literal["available", "unavailable"]
    open_run_ids: list[str]
    unrecorded_run_ids: list[str]
    pending_invocation_ids: list[str]


class CostCustody(_CostCustodyRequired, total=False):
    """Execution coverage for the amount's scope; never another monetary store.

    ``unrecorded_run_ids`` names every run without a matching receipt. Its subset
    ``closed_unrecorded_run_ids`` names those whose custody already closed (settled or
    closed absent): no collector writes for them, so their cost is unknown, not pending.
    Records written before that field lack it; absence keeps their earlier reading.
    """

    closed_unrecorded_run_ids: list[str]


def unavailable_cost_custody() -> CostCustody:
    """A failed coverage read is unknown, even when no run identity is readable."""
    return {"status": "unavailable", "open_run_ids": [], "unrecorded_run_ids": [], "closed_unrecorded_run_ids": [],
            "pending_invocation_ids": []}


@contextmanager
def project_retirement_lock(drive_root: Any, project_id: str):
    """Cross-process lock for one project's settlement/retirement decision."""
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock, release_exclusive_file_lock,
    )

    digest = hashlib.sha256(str(project_id or "").encode("utf-8")).hexdigest()[:24]
    lock_path = pathlib.Path(drive_root) / "state" / "delegate_project_retirements" / f"{digest}.lock"
    fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=20.0, stale_sec=120.0, owner_aware_stale=True,
    )
    if fd is None:
        raise TimeoutError("project retirement lock is unavailable")
    try:
        yield
    finally:
        release_exclusive_file_lock(lock_path, fd)


def summary_of(detail: Dict[str, Any]) -> Dict[str, Any]:
    return detail.get("summary") if isinstance(detail.get("summary"), dict) else {}


def run_names(detail: Dict[str, Any]) -> frozenset[str]:
    """Both names the engine gives this detail's run: its run id and the job id backing it.

    A start answered before its run bound (HTTP 202) hands the host only the job id, and the
    engine addresses a run by either name (control-api ``findRun``), so a held run id is one
    of these exactly. Empty when the detail names no run.
    """
    summary = summary_of(detail)
    run = summary.get("runId") or summary.get("id") or detail.get("runId")
    return frozenset(str(name) for name in (run, summary.get("jobId")) if name)


def names_another_run(detail: Dict[str, Any], run_id: Any) -> bool:
    """The detail names a run, and ``run_id`` is neither of its names."""
    names = run_names(detail)
    return bool(names) and str(run_id or "") not in names


def held_run_names(detail: Dict[str, Any], run_id: Any) -> frozenset[str]:
    """The names the held run may carry in evidence beside this detail (telemetry, turns,
    continuity rows): ``run_id`` and, only when the detail is of that run, its two names."""
    held = str(run_id or "")
    return frozenset({held}) if names_another_run(detail, held) else frozenset({held, *run_names(detail)})


def disclosed_cost(summary: Dict[str, Any], *, attempt_execution: Any = None) -> Dict[str, Any]:
    """Known cash, estimation and missing cash are separate facts.

    An exact attempt plus an unpriced attempt is a known subtotal, never an
    estimated total or an upper bound. Only reported attempts are considered.
    """
    partial = False
    if isinstance(attempt_execution, list) and attempt_execution:
        cash, priced, estimated = 0.0, False, False
        for attempt in attempt_execution:
            cost = attempt.get("usageCost") if isinstance(attempt, dict) else None
            cost = cost if isinstance(cost, dict) else {}
            amount = provider_cost_value(cost.get("cashUsd"))
            knowledge = cost.get("cashKnowledge")
            if amount is None or knowledge not in {"exact", "estimated"}:
                partial = True
            else:
                cash += amount
                priced = True
                estimated |= knowledge == "estimated"
        cash = provider_cost_value(cash) if priced else None
    else:
        cash = provider_cost_value(summary.get("spendUsd"))
        estimated = cash is not None and summary.get("spendEstimated") is True
    return {"cost_usd": cash, "cost_estimated": estimated, "cost_partial": partial,
            "cost_final": cash is not None and not estimated and not partial}


def disclosed_spend(summary: Dict[str, Any], *, attempt_execution: Any = None) -> Tuple[Optional[float], bool]:
    """Compatibility pair: the second value withholds finality for partial cash too."""
    facts = disclosed_cost(summary, attempt_execution=attempt_execution)
    return facts["cost_usd"], facts["cost_usd"] is not None and not facts["cost_final"]


def model_identity_delta(requested: str, applied: str) -> Dict[str, Any]:
    """The one record of a delegated session that names its model differently from the request (a review
    seat's and a delegate run's): the engine writes its own alias, so the note says what was not compared."""
    return {"kind": "capability_delta", "requested": f"model {requested}", "effective": f"model {applied}",
            "reason": "session_route_resolves_its_own_model",
            "note": "The engine names the model it ran by its own alias; the names differ, capabilities were "
                    "not compared. A disclosure, not a failure."}


def session_attempt_id(session_id: str) -> str:
    """The existing one-shot ledger identity, also used by coverage readers."""
    return "session-" + hashlib.sha256(str(session_id or "").strip().encode("utf-8")).hexdigest()[:24]


def cost_snapshot(root, *, task_id="", root_task_id="", custody_root=None, allow_stale=False):
    """Observe custody BEFORE money, over addressed indexes; never replay history.

    Canonical custody and an explicitly bound execution store may differ. A
    missing/unreadable side is a coverage gap, not an empty set. This is not an
    atomic fence against a new invocation starting after this observation.
    """
    from ouroboros import store
    from ouroboros.delegate_custody_current import task_snapshot
    from ouroboros.runtime_limits import USAGE_DISPLAY_LOCK_TIMEOUT_SEC

    snapshot = {"runs": {}, "pending": [], "status": "available"}
    if not (task_id or root_task_id):
        return snapshot
    roots = {pathlib.Path(root).resolve(), pathlib.Path(custody_root or root).resolve()}
    deadline = time.monotonic() + USAGE_DISPLAY_LOCK_TIMEOUT_SEC if allow_stale else None
    for address in sorted(roots):
        try:
            view = task_snapshot(address, task_id, root_task_id=root_task_id,
                                 wait_sec=max(0.0, deadline - time.monotonic()) if deadline is not None else store.STORE_WAIT_SEC)
            for rid, entry in view["runs"].items():
                if rid in snapshot["runs"] and vars(entry) != vars(snapshot["runs"][rid]):
                    snapshot["status"] = "unavailable"  # conflicting custody cannot close money
                snapshot["runs"][rid] = entry
            snapshot["pending"].extend(view["pending"])
        except store.StoreBusy:
            log.debug("Cost custody busy for %s/%s", root_task_id, task_id)
            snapshot["status"] = "unavailable"
        except Exception:
            log.warning("Cost custody unavailable for %s/%s", root_task_id, task_id, exc_info=True)
            snapshot["status"] = "unavailable"
    return snapshot


def cost_coverage(snapshot, txn, root) -> dict:
    """Join fresh ledger receipts and amounts in the SAME ledger transaction.

    Counts/dollars stay ledger-owned. Even SETTLED/LEDGER_RECORDED is only a
    custody claim until its actual one-shot receipt belongs to this scope.
    Cleanup holds (patches, output, project retirement) are not monetary holds.
    A settled run without that receipt is ``closed_unrecorded``: its custody is over
    (SETTLED or CLOSED_ABSENT, whatever ``ledger_recorded`` claims) and no collection
    follows, so no money will arrive. An unsettled run, terminal or not, may still
    be collected and stays unrecorded only.
    """
    from ouroboros.delegate_custody import TERMINAL_STATES

    root = pathlib.Path(root).resolve()
    ledger_roots = {"": root}
    missing = set()
    for rid, entry in sorted(snapshot["runs"].items()):
        address = str(entry.ledger_root or "")
        if address not in ledger_roots:
            ledger_roots[address] = pathlib.Path(address).resolve()
        receipt = txn.attempt(session_attempt_id(rid))
        matching = (receipt is not None and receipt.get("kind") == "subscription_session" and receipt.get("state") == "settled"
                    and receipt.get("task_id") == entry.task_id
                    and receipt.get("root_task_id") == (entry.root_task_id or entry.task_id)
                    and ledger_roots[address] == root)
        if not matching:
            missing.add(rid)
    return {**snapshot, "unrecorded": missing,
            "closed_unrecorded": {rid for rid in missing if snapshot["runs"][rid].settled},
            "open": {rid for rid, entry in snapshot["runs"].items()
                     if not entry.settled and entry.terminal_state not in TERMINAL_STATES}}


def with_cost_coverage(summary, joined, *, task_id="", root_task_id=""):
    """Project the already joined facts onto one bucket without rereading receipts."""
    def selected(task, tree):
        return (not task_id or task == task_id) and (not root_task_id or (tree or task) == root_task_id)

    runs = {rid for rid, entry in joined["runs"].items() if selected(entry.task_id, entry.root_task_id)}
    coverage: CostCustody = {
        "status": joined["status"], "open_run_ids": sorted(runs & joined["open"]),
        "unrecorded_run_ids": sorted(runs & joined["unrecorded"]),
        "closed_unrecorded_run_ids": sorted(runs & joined["closed_unrecorded"]),
        "pending_invocation_ids": sorted({str(p["invocation_id"]) for p in joined["pending"]
                                          if selected(p.get("task_id"), p.get("root_task_id"))}),
    }
    result = {**summary, "cost_custody": coverage}
    if cost_coverage_open(result):
        result["cost_final"] = False
    return result


def cost_coverage_open(summary) -> bool:
    """Coverage leaves the amount incomplete: unreadable, live, uncollected or closed without a receipt."""
    coverage = summary.get("cost_custody") or {}
    return bool(coverage.get("status", "available") != "available"
                or coverage.get("open_run_ids") or coverage.get("unrecorded_run_ids")
                or coverage.get("pending_invocation_ids"))


def cost_coverage_pending(summary) -> bool:
    """Coverage may still bring money: as ``cost_coverage_open`` minus runs closed without a receipt."""
    coverage = summary.get("cost_custody") or {}
    closed = set(coverage.get("closed_unrecorded_run_ids") or ())
    return bool(coverage.get("status", "available") != "available"
                or coverage.get("open_run_ids") or set(coverage.get("unrecorded_run_ids") or ()) - closed
                or coverage.get("pending_invocation_ids"))


def disclosed_tokens(raw: Any) -> Optional[int]:
    """A reported token count, or ``None`` when the harness reported nothing.

    The control schema keeps token counts null until a harness reports them;
    converting absence to zero would erase that distinction.
    """
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return None


def is_terminal(detail: Dict[str, Any], terminal_states: frozenset[str]) -> bool:
    return str(summary_of(detail).get("state") or "") in terminal_states


# ---- reviewer usage observation (sessions and physical extraction sends stay separate) ----

_EMITTED_SESSION_USAGE: Dict[str, float] = {}  # delegated run id -> first llm_usage emission


def session_usage_once(run_id: str) -> bool:
    """One llm_usage row per delegated run, across executors (process-local)."""
    key = str(run_id or "")
    if key in _EMITTED_SESSION_USAGE:
        return False
    if key:
        _EMITTED_SESSION_USAGE.clear() if len(_EMITTED_SESSION_USAGE) >= 256 else None
        _EMITTED_SESSION_USAGE[key] = time.monotonic()
    return True


def observe_session_usage(observer: Any, usage: Dict[str, Any]) -> None:
    """A continued reviewer emits each run, never its chain total as the last run."""
    for part in usage.get("session_usage_events") or [usage]:
        if session_usage_once(part.get("delegated_run_id", "")):
            observe_review_usage(observer, part)


def observe_review_usage(
    observer: Any, usage: Optional[Dict[str, Any]], *, known: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Hand one row per physical send to the reviewer usage observer.

    An earlier attempt a later reply names is its own ledger fact, so its event states only
    what its own receipt holds, route and price alike, never the reply's: the seat's
    already-recorded fact when ``known`` carries one (``ReviewAttemptCosts._observed``),
    otherwise the ledger row read through the same indexed ``_review_attempt_rows`` the seat
    projects. The read is returned, so that projection reuses it instead of repeating it and
    keeps every id for custody and pricing. A receipt read as a hold that never left the
    host (``_proven_unsent``) presents no send. A single ordinary exact receipt keeps the
    zero-read path — an empty earlier set never opens the store — and an unread receipt
    stays unpriced rather than stating an unknown this module did not observe; its route
    is stated absent (``_receipt_route``), never borrowed.
    """
    if observer is None:
        return {}
    row = dict(usage or {})
    attempt_ids = [str(value) for value in (row.get("ledger_attempt_ids") or []) if value]
    earlier_ids = attempt_ids[:-1]
    recorded = known or {}
    receipts = {**{key: recorded[key] for key in earlier_ids if recorded.get(key)},
                **_review_attempt_rows([key for key in earlier_ids if not recorded.get(key)])}
    for attempt_id in earlier_ids:
        receipt = receipts.get(attempt_id)
        if receipt is not None and _proven_unsent(receipt):
            continue
        observer({
            **_receipt_route(receipt), "ledger_attempt_ids": [attempt_id],
            **({} if receipt is None else _settled_receipt_price(receipt)),
        })
    if attempt_ids:
        row["ledger_attempt_ids"] = attempt_ids[-1:]
    observer(row)
    return receipts


def _review_attempt_rows(attempt_ids: list[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """Read only these physical ids in the current review's billing scope."""
    from ouroboros import usage_store
    from ouroboros.usage_accounting import _drive_root, current_usage_scope

    if attempt_ids:
        try:
            scope = current_usage_scope()
            with usage_store.read(_drive_root(getattr(scope, "drive_root", None)), allow_stale=True) as txn:
                return {attempt_id: txn.attempt(attempt_id) for attempt_id in attempt_ids}
        except Exception:
            log.debug("failed to resolve review attempt states", exc_info=True)
    return dict.fromkeys(attempt_ids)


def _receipt_cash(row: Dict[str, Any]) -> Any:
    """A settled receipt row's price, in either shape this module holds one.

    A row read from the ledger carries the ledger's own private column; the fact
    ``ReviewAttemptCosts.observe`` already accounted from an ordinary exact receipt
    carries the same amount under the usage projection's own name, so no store-shaped
    row is fabricated here just to borrow the ledger's spelling.
    """
    return row.get("cost_usd") if "cost_usd" in row else row.get("cost")


def _receipt_route(receipt: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The route one attempt's own receipt names, in the usage projection's vocabulary.

    A ledger row names its model ``model``; the fact ``ReviewAttemptCosts.observe`` accounted
    keeps its reply's own ``resolved_model``. Both keys are always stated, and ``None`` is
    this attempt's route explicitly absent (an unread receipt names none): the usage emitter
    keeps that unknown instead of the seat's configured model, the reply's route or another
    attempt's.
    """
    row = receipt or {}
    model = row.get("model") if "model" in row else row.get("resolved_model")
    return {"resolved_model": str(model) if model else None,
            "provider": str(row["provider"]) if row.get("provider") else None}


def _unsent_states() -> frozenset:
    """The ledger's own states of a hold that never left the host (reserved, released)."""
    from ouroboros.usage_accounting import PHYSICAL_ATTEMPT_STATES, POSITIVE_PHYSICAL_ATTEMPT_STATES

    return PHYSICAL_ATTEMPT_STATES - POSITIVE_PHYSICAL_ATTEMPT_STATES


def _proven_unsent(receipt: Dict[str, Any]) -> bool:
    """A read receipt in one of those states: its id stays in custody, no send is presented."""
    return receipt.get("state") in _unsent_states()


def _settled_receipt_price(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """One read receipt row's price, in the usage projection's own vocabulary.

    Shared by a prior physical attempt's event and a failed send's event, so the two
    lanes cannot fork: a settled row states its exact cash (a reported zero included),
    any other read state states an explicit unknown, and an estimate keeps its mark.
    """
    cash = provider_cost_value(_receipt_cash(receipt)) if receipt.get("state") == "settled" else None
    return {"cost": cash,
            "cost_final": cash is not None and receipt.get("cost_final") is True
            and receipt.get("cost_partial") is not True,
            "cost_estimated": cash is not None and _review_receipt_estimated(receipt)}


def _review_receipt_estimated(row: Dict[str, Any]) -> bool:
    """For a priced row, keep an exact subtotal distinct from an estimate."""
    return row.get("cost_final") is not True and not (
        row.get("cost_partial") is True and row.get("cost_estimated") is False)


def observe_failed_review_send(observer: Any, exc: BaseException) -> Dict[str, Optional[Dict[str, Any]]]:
    """Emit the failed send's physical events; share the same read with its seat's money projection."""
    from ouroboros._usage_response import UnusableReplyError
    from ouroboros.usage_accounting import PHYSICAL_ATTEMPT_STATES, POSITIVE_PHYSICAL_ATTEMPT_STATES

    capture = getattr(exc, "physical_attempt_capture", None)
    attempt_ids = dict.fromkeys(str(value) for ids in (
        getattr(exc, "ledger_attempt_ids", None),
        exc.usage.get("ledger_attempt_ids") if isinstance(exc, UnusableReplyError) else None,
    ) for value in ids or () if value)
    capture_id = str(getattr(capture, "attempt_id", "") or "")
    if capture_id:
        attempt_ids[capture_id] = None
    rows = _review_attempt_rows(list(attempt_ids))
    capture_state = str(getattr(capture, "state", "") or "")
    any_read = any(row is not None for row in rows.values())
    for attempt_id, receipt in rows.items():
        row = receipt or {}
        state = str(row.get("state") or (capture_state if attempt_id == capture_id else ""))
        if state not in POSITIVE_PHYSICAL_ATTEMPT_STATES and not (
            not any_read and attempt_id != capture_id
        ):
            continue
        if attempt_id == capture_id:  # share this exact route with its event and seat, never with another id
            row = rows[attempt_id] = {**row, "state": state, **{
                key: row.get(key) or getattr(capture, key, None) for key in ("model", "provider")}}
        usage = {**_receipt_route(row), "ledger_attempt_ids": [attempt_id]}
        if receipt is not None:
            usage.update(_settled_receipt_price(receipt))
        observe_review_usage(observer, usage)
    # Preserve the physical fact, but never infer money from an unread row.
    if capture_state in PHYSICAL_ATTEMPT_STATES and capture_id and rows[capture_id] is None:
        rows[capture_id] = {"state": capture_state}
    return rows


class ReviewAttemptCosts:
    """Project an API review seat's settled spend once per physical id, not once per answer."""

    def __init__(self) -> None:
        self._observed: Dict[str, Optional[Dict[str, Any]]] = {}
        self._unread: dict[str, None] = {}
        self._returned_reply = False
        self._last_failed_route: Optional[Dict[str, Any]] = None

    def observe(self, usage: Optional[Dict[str, Any]], observer: Any) -> None:
        from ouroboros.usage_accounting import last_physical_attempt_capture

        self._returned_reply = True
        self._last_failed_route = None  # a subsequent returned reply supersedes an earlier transport failure
        reply = usage or {}
        ids = [str(value) for value in reply.get("ledger_attempt_ids") or () if value]
        for attempt_id in ids:
            if attempt_id not in self._observed:
                self._observed[attempt_id] = None
                self._unread[attempt_id] = None
        capture = last_physical_attempt_capture()
        # Exact ordinary receipts already pass through settle_attempt. Read back only prices not
        # accounted for by that receipt (internal resends, nonfinal and unreadable settlement);
        # the accounted fact stays in this module's own row vocabulary, not the store's, and
        # keeps the route its own reply stated for an event a later reply may ask of it.
        if (ids and capture is not None and capture.attempt_id == ids[-1] and capture.state == "settled"
                and reply.get("cost_final") is True and not any(reply.get(key) for key in (
                    "cost_partial", "cost_estimated", "cost_invalid"))
                and (cash := provider_cost_value(reply.get("cost"))) is not None):
            self._observed[ids[-1]] = {"state": "settled", "cost": cash, "cost_final": True, **{
                key: reply[key] for key in ("resolved_model", "provider") if reply.get(key)}}
            self._unread.pop(ids[-1], None)
        # The events below price the earlier attempts from what this seat already recorded
        # or read once; the projection shares that same result instead of repeating it.
        for attempt_id, receipt in observe_review_usage(observer, usage, known=self._observed).items():
            self._observed[attempt_id] = receipt
            self._unread.pop(attempt_id, None)

    def failed(self, rows: Optional[Dict[str, Optional[Dict[str, Any]]]]) -> None:
        from ouroboros.usage_accounting import POSITIVE_PHYSICAL_ATTEMPT_STATES

        for attempt_id, row in (rows or {}).items():
            self._unread.pop(attempt_id, None)
            if row is not None or not self._observed.get(attempt_id):
                self._observed[attempt_id] = row
            if row and row.get("state") in POSITIVE_PHYSICAL_ATTEMPT_STATES:
                self._last_failed_route = _receipt_route(row)

    def project(self, usage: Dict[str, Any], *, failed: bool = False) -> Dict[str, Any]:
        from ouroboros._usage_money import amount, exact_money
        from ouroboros.usage_accounting import POSITIVE_PHYSICAL_ATTEMPT_STATES

        if not self._observed:
            return usage  # uninstrumented legacy replies keep their existing usage grammar
        self._observed.update(_review_attempt_rows(list(self._unread)))
        self._unread.clear()
        amounts = []
        states = set()
        partial = estimated = False
        for row in (receipt or {} for receipt in self._observed.values()):
            state = row.get("state")
            states.add(state)
            if state == "released":
                continue
            cash = amount(_receipt_cash(row)) if state == "settled" else None
            partial |= cash is None or row.get("cost_partial") is True
            if cash is not None:
                amounts.append(cash)
                estimated |= _review_receipt_estimated(row)
        with exact_money():
            subtotal = float(sum(amounts)) if amounts else None
        projected = dict(usage)
        projected.pop("cost_invalid", None)  # the ledger, not the last reply's price, is the projection's source
        projected.update(ledger_attempt_ids=list(self._observed), cost=subtotal, cost_partial=partial,
                         cost_estimated=estimated, cost_final=subtotal is not None and not (partial or estimated))
        # Billing IDs can include holds that never left the host. Disambiguate only
        # those receipts; an ordinary returned reply keeps its existing projection.
        if states & _unsent_states():
            projected["physical_dispatch_observed"] = self._returned_reply or bool(states & POSITIVE_PHYSICAL_ATTEMPT_STATES)
        if failed and self._last_failed_route is not None:
            projected.update(self._last_failed_route, physical_dispatch_observed=True)
        return projected
