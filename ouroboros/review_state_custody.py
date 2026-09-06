"""Durable custody of in-flight review invocations and attempt-history hygiene.

The checkpoint that records a pending review invocation before dispatch, the
roster-row predicates that decide whether an attempt still holds active review
custody, the eviction rule that keeps settled attempts from starving the
bounded history, and the heavy-payload stripper applied on eviction. This is
post-cutoff upstream growth (the adaptive-timeout/custody train) cut out of
ouroboros/review_state.py by the v7next D06 lane as a NEW owner leaf — no
MIGRATION_v7 row names these symbols; review_state.py re-exports every name.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.review_state_model import AdvisoryReviewState
    from ouroboros.review_state_records import CommitAttemptRecord


def _rs():
    """The parent review-state module, read at call time.

    The review-state members stay monkeypatch-addressable at their historical
    ``ouroboros.review_state`` bindings (tests rebind them there), so this leaf
    resolves every such cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import review_state

    return review_state


_ACTIVE_REVIEW_OPERATION_STATES = frozenset({"in_flight", "custody_lost"})


def _attempt_review_roster_rows(item: "CommitAttemptRecord") -> List[Dict[str, Any]]:
    """Return mutable row objects from both existing commit-review surfaces."""
    rows = [row for row in (item.triad_raw_results or []) if isinstance(row, dict)]
    scope = item.scope_raw_result
    if not isinstance(scope, dict) or not scope:
        return rows
    raw_results = scope.get("raw_results")
    if isinstance(raw_results, list):
        rows.extend(row for row in raw_results if isinstance(row, dict))
    elif "raw_results" not in scope:
        # Historical single-scope rows used the wrapper itself as the actor.
        rows.append(scope)
    return rows


def _review_roster_row_is_pending(row: Dict[str, Any]) -> bool:
    return bool(
        str(row.get("pending_invocation_id") or "").strip()
        or row.get("late_result_pending")
        or str(row.get("operation_state") or "") in _ACTIVE_REVIEW_OPERATION_STATES
    )


def _attempt_has_active_review_custody(item: "CommitAttemptRecord") -> bool:
    """One SSOT for overlap, startup, eviction, and payload compaction."""
    if str(getattr(item, "status", "") or "") == "reviewing" or bool(
        getattr(item, "late_result_pending", False)
    ):
        return True
    # A terminal lifecycle projection can still race a live delegated/API row;
    # preserve that exact roster. A synthetic malformed-container
    # ``custody_lost`` row on a historical terminal attempt is forensic damage,
    # not reopened physical work, so it does not make the old attempt active.
    return any(
        str(row.get("pending_invocation_id") or "").strip()
        or str(row.get("operation_state") or "") == "in_flight"
        for row in _attempt_review_roster_rows(item)
    )


def checkpoint_pending_review_invocation(
    drive_root: pathlib.Path,
    *,
    repo_key: str,
    tool_name: str,
    task_id: str,
    attempt: int,
    review_retry_key: str,
    surface: str,
    slot_id: str,
    operation_id: str,
    invocation_id: str,
) -> None:
    """Bind one delegated start token to its pre-reserved commit-review row.

    The existing advisory-review state remains the only ledger.  This narrow
    locked patch is deliberately stricter than a whole-attempt merge: triad and
    scope start concurrently, so each may update only its exact reserved row
    and may never overwrite the other surface's token.
    """
    expected = {
        "review_retry_key": str(review_retry_key or ""),
        "surface": str(surface or ""),
        "slot_id": str(slot_id or ""),
        "operation_id": str(operation_id or ""),
        "invocation_id": str(invocation_id or ""),
    }
    if not all(expected.values()):
        raise ValueError("pending review invocation checkpoint is incomplete")
    if expected["surface"] not in {"multi_model_review", "scope_review"}:
        raise ValueError(f"unsupported commit-review surface {surface!r}")

    def _mutate(state: AdvisoryReviewState) -> None:
        current = state.latest_attempt_for(
            repo_key=repo_key,
            tool_name=tool_name,
            task_id=task_id,
            attempt=int(attempt or 0),
        )
        if (
            current is None
            or current.status != "reviewing"
            or not current.paid
            or current.review_retry_key != expected["review_retry_key"]
        ):
            raise ValueError("reserved commit-review attempt is unavailable")
        if expected["surface"] == "multi_model_review":
            rows = current.triad_raw_results
        else:
            wrapper = current.scope_raw_result
            rows = wrapper.get("raw_results") if isinstance(wrapper, dict) else None
        if not isinstance(rows, list):
            raise ValueError("reserved commit-review roster is unavailable")
        matches = [
            row for row in rows
            if isinstance(row, dict)
            and str(row.get("slot_id") or row.get("slot") or "") == expected["slot_id"]
        ]
        if len(matches) != 1:
            raise ValueError("reserved commit-review slot identity is ambiguous")
        row = matches[0]
        if str(row.get("operation_id") or "") != expected["operation_id"]:
            raise ValueError("reserved commit-review operation identity changed")
        if str(row.get("operation_state") or "") != "in_flight":
            raise ValueError("reserved commit-review operation is not in flight")
        prior = str(row.get("pending_invocation_id") or "")
        if prior and prior != expected["invocation_id"]:
            raise ValueError("reserved commit-review invocation identity changed")
        row["pending_invocation_id"] = expected["invocation_id"]
        row["late_result_pending"] = True
        current.late_result_pending = True
        current.updated_ts = _rs()._utc_now()

    _rs().update_state(pathlib.Path(drive_root), _mutate)


def _attempt_history_evictable(item: "CommitAttemptRecord") -> bool:
    """True only for rows the Max-Review-Cycles machinery derives NO authority
    from. Never evictable: paid rows (the per-root-task money count), rows of
    an attempt still in flight (their upcoming terminal/paid merge must find
    them), and review-VERDICT blocks (the anchors an identical-diff refusal
    quotes; legacy rows classify by recorded reason via the commit gate's
    ``attempt_block_class``). Unclassifiable rows are kept — fail toward
    remembering."""
    if _attempt_has_active_review_custody(item):
        return False
    if bool(getattr(item, "paid", False)):
        return False
    try:
        from ouroboros.tools.commit_gate import BLOCK_CLASS_VERDICT, attempt_block_class

        return attempt_block_class(item) != BLOCK_CLASS_VERDICT
    except Exception:
        return False


_STRIPPED_DETAILS_LIMIT = 600


_STRIPPED_MESSAGE_LIMIT = 300


def _strip_attempt_heavy_payload(item: CommitAttemptRecord, *, force: bool = False) -> None:
    """Compact one preserved accounting row that fell outside the newest-50
    ledger window (F1 follow-up: paid-row immortality must not make the hot
    load-modify-save state file grow by full reviewer raw output forever).

    Dropped: the heavy forensic payloads NO gate reads from over-window rows —
    ``triad_raw_results`` and ``scope_raw_result`` (full reviewer raw_text) —
    plus the free-text fields bounded to what consumers render. Kept, exactly
    the facts the Max-Review-Cycles gates consume: ``paid``/``root_task_id``
    (the ceiling count), status/phase/``block_reason``/``block_class``/
    ``pre_review_fingerprint``/``review_contract_fingerprint``/
    ``rebuttal_sha256`` (the streak walker), and ``critical_findings`` plus a
    600-char ``block_details`` excerpt (everything ``_quote_verdict_attempt``
    renders in an identical-diff refusal). A legacy scope verdict classifies
    THROUGH ``scope_raw_result``, so ``block_class`` is materialized before
    that evidence is dropped — the row keeps its refusal-anchor authority.
    ``raw_stripped=True`` marks the compaction for audits; full payloads live
    only in the newest-50 window. ``force=True`` re-compacts a row ALREADY
    marked stripped — the terminal-merge path uses it when a merge onto a
    stripped row would otherwise resurrect the heavy payloads it carries in."""
    if _attempt_has_active_review_custody(item):
        return
    if bool(getattr(item, "raw_stripped", False)) and not force:
        return
    if not item.block_class:
        try:
            from ouroboros.tools.commit_gate import attempt_block_class

            klass = attempt_block_class(item)
            if klass:
                item.block_class = klass
        except Exception:
            pass
    item.triad_raw_results = []
    item.scope_raw_result = {}
    item.block_details = _rs()._truncate_review_artifact(
        str(item.block_details or ""), limit=_STRIPPED_DETAILS_LIMIT
    )
    item.commit_message = _rs()._truncate_review_artifact(
        str(item.commit_message or ""), limit=_STRIPPED_MESSAGE_LIMIT
    )
    item.raw_stripped = True
