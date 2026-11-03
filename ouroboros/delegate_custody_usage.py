"""Pure usage and terminal-state projections for delegated-run custody."""

from __future__ import annotations

import hashlib
import logging
import pathlib
import time
from contextlib import contextmanager
from typing import Any, Dict, Literal, Optional, Tuple, TypedDict

from ouroboros._usage_response import provider_cost_value

log = logging.getLogger(__name__)


class CostCustody(TypedDict):
    """Execution coverage for the amount's scope; never another monetary store."""

    status: Literal["available", "unavailable"]
    open_run_ids: list[str]
    unrecorded_run_ids: list[str]
    pending_invocation_ids: list[str]


def unavailable_cost_custody() -> CostCustody:
    """A failed coverage read is unknown, even when no run identity is readable."""
    return {"status": "unavailable", "open_run_ids": [], "unrecorded_run_ids": [], "pending_invocation_ids": []}


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
        "pending_invocation_ids": sorted({str(p["invocation_id"]) for p in joined["pending"]
                                          if selected(p.get("task_id"), p.get("root_task_id"))}),
    }
    result = {**summary, "cost_custody": coverage}
    if cost_coverage_open(result):
        result["cost_final"] = False
    return result


def cost_coverage_open(summary) -> bool:
    coverage = summary.get("cost_custody") or {}
    return bool(coverage.get("status", "available") != "available"
                or coverage.get("open_run_ids") or coverage.get("unrecorded_run_ids")
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


def observe_review_usage(observer: Any, usage: Optional[Dict[str, Any]]) -> None:
    """Hand one row per ledger attempt to the reviewer usage observer."""
    if observer is None:
        return
    row = dict(usage or {})
    attempt_ids = [str(value) for value in (row.get("ledger_attempt_ids") or []) if value]
    for attempt_id in attempt_ids[:-1]:
        observer({
            "resolved_model": row.get("resolved_model"), "provider": row.get("provider"),
            "ledger_attempt_ids": [attempt_id],
        })
    if attempt_ids:
        row["ledger_attempt_ids"] = attempt_ids[-1:]
    observer(row)


def observe_failed_review_send(observer: Any, exc: BaseException) -> None:
    """Rows for every physically dispatched attempt behind a failed reviewer send."""
    from ouroboros import usage_store
    from ouroboros.usage_accounting import POSITIVE_PHYSICAL_ATTEMPT_STATES, _drive_root, current_usage_scope

    capture = getattr(exc, "physical_attempt_capture", None)
    attempt_ids = [str(value) for value in (getattr(exc, "ledger_attempt_ids", None) or []) if value]
    capture_id = str(getattr(capture, "attempt_id", "") or "")
    if capture_id and capture_id not in attempt_ids:
        attempt_ids.append(capture_id)
    rows: Dict[str, Dict[str, Any]] = {}
    try:
        scope = current_usage_scope()
        root = _drive_root(getattr(scope, "drive_root", None))
        with usage_store.read(root) as txn:  # one PK lookup per attempt id
            found = {attempt_id: txn.attempt(attempt_id) for attempt_id in attempt_ids}
        rows = {attempt_id: row for attempt_id, row in found.items() if row is not None}
    except Exception:
        log.debug("failed to resolve review attempt states", exc_info=True)
    capture_state = str(getattr(capture, "state", "") or "")
    for attempt_id in attempt_ids:
        row = rows.get(attempt_id, {})
        state = str(row.get("state") or (capture_state if attempt_id == capture_id else ""))
        if state not in POSITIVE_PHYSICAL_ATTEMPT_STATES and not (
            not rows and attempt_id != capture_id
        ):
            continue
        observe_review_usage(observer, {
            "resolved_model": str(row.get("model") or getattr(capture, "model", "") or ""),
            "provider": str(row.get("provider") or getattr(capture, "provider", "") or ""),
            "ledger_attempt_ids": [attempt_id],
        })
