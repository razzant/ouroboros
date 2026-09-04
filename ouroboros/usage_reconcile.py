"""Reconciliation of abandoned ``unresolved`` usage-attempt rows.

A provider send whose outcome is unknown (transient 429/5xx, a transport reset
after dispatch) is marked ``unresolved`` — a terminal ledger state that keeps the
attempt's reservation upper bound inside ``accounted_usd``. The row is dead on
arrival: no writer can add information to a terminal row, and no provider
re-probe handle is stored (a rate-limited send has no generation id to query),
so without a reconciliation path the row blocked ``cost_final`` for its task
FOREVER — the incident class where completed benchmark tasks were dropped while
waiting on cost finality.

This is the ONE production reconciliation path, reached from two existing seams
(no new scheduler):

* the terminal cost authority — ``supervisor.state.reconstruct_task_cost`` calls
  it with ``task_id`` before projecting, so a terminal task's stored frame is
  born final (its dead rows are written off immediately); and
* the supervisor maintenance tick (server.py) — a periodic sweep without
  ``task_id`` that writes off rows older than
  ``OUROBOROS_USAGE_UNRESOLVED_WRITEOFF_SEC`` (config.py SSOT), healing orphans
  whose task never reached a terminal projection (crashes, pre-feature ledgers,
  unattributed rows).

A write-off settles the row AT its carried reservation bound through the
existing ``terminalize_abandoned_attempt`` (no duplicated transition logic):
``accounted_usd`` does not move, only finality does, and the origin reason is
preserved on the settled row. A row whose bound is genuinely unknown is never
given a fabricated number — it stays honestly unresolved.
"""

from __future__ import annotations

import datetime as _dt
import logging
import pathlib
import time
from typing import Any, Dict, Optional

from ouroboros.usage_accounting import (
    AttemptReservation,
    _drive_root,
    _final_rows,
    _locked,
    _number,
    _read_records_locked_cached,
    terminalize_abandoned_attempt,
)
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)


def _row_age_sec(row: Dict[str, Any], now: float) -> float:
    try:
        ts = _dt.datetime.fromisoformat(str(row.get("ts") or ""))
    except ValueError:
        return float("inf")  # an undatable row is the oldest one
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_dt.timezone.utc)
    return max(0.0, now - ts.timestamp())


def reconcile_abandoned_unresolved_attempts(
    drive_root: pathlib.Path | str | None = None,
    *,
    task_id: str = "",
    now: Optional[float] = None,
    max_age_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Terminalize abandoned unresolved attempt rows; bounded and idempotent.

    With ``task_id`` (the terminal cost-authority seam) only that task's rows —
    its own and its subtree's (``root_task_id``) — are eligible, at ANY age: the
    task is terminalizing, so those rows are known dead. Without it (the
    periodic sweep) a row must be older than the configured write-off TTL.

    At the terminal seam the task's OWN still-open ``dispatched`` / ``reserved``
    rows are dead too: the process that would have settled them was killed
    (CyberGym r8, 2026-09-04 — the reaper's deadline kill landed mid
    post-task synthesis, the completed frame was born ``cost_final: false`` and
    the launcher's cancellation custody timed out on a solved task).  A
    dispatched row settles at its carried reservation bound (money does not
    move — the bound already stands in ``accounted_usd``); a reserved row never
    reached a provider and is released.  A worker that is in fact still alive
    and settles later simply appends the measured row, which then supersedes
    the bound.  Subtree rows keep the ``unresolved``-only rule: a child may be
    legitimately mid-call while its root terminalizes.
    """
    from ouroboros.config import get_usage_unresolved_writeoff_sec

    root = _drive_root(drive_root)
    task_id = str(task_id or "").strip()
    ttl = float(max_age_sec) if max_age_sec is not None else get_usage_unresolved_writeoff_sec()
    now = time.time() if now is None else float(now)
    with _locked(root):
        finals = list(_final_rows(_read_records_locked_cached(root)).values())
    candidates = []
    for row in finals:
        if str(row.get("kind") or "attempt") != "attempt":
            continue  # legacy_metadata rows carry no bound and never block finality
        state = str(row.get("state") or "")
        if task_id:
            own = str(row.get("task_id") or "") == task_id
            subtree = str(row.get("root_task_id") or "") == task_id
            if state == "unresolved":
                if not (own or subtree):
                    continue
            elif state in {"dispatched", "reserved"}:
                if not own:
                    continue
            else:
                continue
        else:
            if state != "unresolved":
                continue
            if _row_age_sec(row, now) < ttl:
                continue
        candidates.append(row)
    terminalized: list[str] = []
    released: list[str] = []
    kept_unknown_bound: list[str] = []
    for row in candidates:
        reservation = AttemptReservation(
            str(row["attempt_id"]),
            root,
            str(row.get("model") or ""),
            str(row.get("provider") or ""),
            _number(row.get("reservation_upper_bound_usd")),
        )
        was_open = str(row.get("state") or "") in {"dispatched", "reserved"}
        reason = "reconcile_terminal_task_open_attempt" if was_open else "reconcile_abandoned_unresolved"
        try:
            state = terminalize_abandoned_attempt(reservation, reason=reason)
            if state == "unresolved" and was_open:
                # dispatched -> unresolved (no measured usage); the second pass is
                # the ordinary bound write-off of a now-unresolved dead row.
                state = terminalize_abandoned_attempt(reservation, reason=reason)
        except Exception:
            log.warning("Abandoned-attempt write-off failed for %s", row.get("attempt_id"), exc_info=True)
            continue
        if state == "settled":
            terminalized.append(reservation.attempt_id)
        elif state == "released":
            released.append(reservation.attempt_id)
        else:
            kept_unknown_bound.append(reservation.attempt_id)
    if terminalized or released:
        try:
            append_jsonl(
                root / "logs" / "events.jsonl",
                {
                    "type": "usage_unresolved_writeoff",
                    "ts": utc_now_iso(),
                    "task_id": task_id,
                    "attempt_ids": terminalized,
                    "released_attempt_ids": released,
                },
            )
        except Exception:
            log.debug("Failed to append usage write-off event", exc_info=True)
    return {
        "terminalized": terminalized,
        "released": released,
        "kept_unknown_bound": kept_unknown_bound,
    }
