"""Reviewer dispatch primitives: the row-identity mint (moved whole from
``review_substrate.py`` at the module-size gate, same split shape as
``tools/review_change_custody.py``) and the write-ahead PAID stamp seam (owner
Q16/Q17; Max-Review-Cycles fix round).

The ``paid`` fact of Max-Review-Cycles accounting is recorded at PHYSICAL
dispatch: a gate that must durably record "this wave spent reviewer money"
installs a :class:`ReviewPaidStamp` on ``ctx._review_paid_stamp`` for the
duration of its wave, and the shared reviewer transport entry
(``review_custody.run_custodied_review_slots``) invokes it after slot resolution and
immediately before worker fan-out. The coordinator also captures that exact
once-only object: session routes invoke it before their replayable
``START_REQUESTED`` row, while API routes bind it for the canonical physical-
attempt boundary. Assembly-only refusals (triad fit ladder, scope pack signals,
skill prompt building) exit before the seam, so a $0 attempt stays outside
every ceiling; a worker that outlives its logical caller cannot race the
write-ahead fact, and a crash after dispatch keeps the durable paid fact.
Commit review verifies this write fail-closed; other callers retain historical
fail-open accounting. This seam also hosts the L-review lane's two-phase admission.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
from typing import Any, Callable, Iterator

log = logging.getLogger(__name__)
_BOUND_API_PAID_STAMP: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "ouroboros_review_api_paid_stamp", default=None,
)

# Identity prefixes for the configured reviewer surfaces. A surface that fans
# rows out registers its prefix here rather than spelling one inline, so
# ``slot_id_for_row`` stays the only place a row id is built.
SLOT_ID_PREFIX = "slot"
SCOPE_SLOT_ID_PREFIX = "scope_slot"
PLAN_SLOT_ID_PREFIX = "plan_slot"


def slot_id_for_row(index: int, *, prefix: str = SLOT_ID_PREFIX) -> str:
    """Identity of the ``index``-th (1-based) configured reviewer row.

    The single mint for reviewer-slot identity, and the reason the substrate
    contract says slot identity is separate from model identity. Naming a row
    after its own model instead collides two rows that share a model (a supported
    configuration — the factory pool repeats Main on purpose),
    collides two model spellings that sanitize alike (``openai::gpt-5`` and
    ``openai/gpt/5``), and moves a row's identity the moment the owner edits its
    model, so the row's receipts stop lining up with its own history. The model,
    the route and the effort are PROPERTIES of a row, never its name.
    """
    return f"{prefix}_{int(index)}"


class ReviewPaidStamp:
    """Idempotent, thread-safe stamp shared by the seats of one review wave.

    The first caller attempts the durable write-ahead; siblings wait for its
    result. A failed write is not retried here and still marks the stamp fired.
    Ordinary cost accounting fails open, with the terminal record authoritative.
    Commit-gate, review_change and resumed-skill stamps use
    ``fail_closed=True``: every caller observes the failure and no reviewer
    transport proceeds.
    """

    def __init__(
        self, write: Callable[[], None], *, fail_closed: bool = False,
    ) -> None:
        self._write = write
        self._lock = threading.Lock()
        self.fail_closed = bool(fail_closed)
        self._failure: Exception | None = None
        self.fired = False

    def __call__(self) -> None:
        with self._lock:
            if self.fired:
                if self.fail_closed and self._failure is not None:
                    raise RuntimeError(str(self._failure)) from self._failure
                return
            try:
                self._write()
            except Exception as exc:
                self._failure = exc
                raise
            finally:
                self.fired = True


def invoke_review_paid_stamp(stamp: Any) -> None:
    """Invoke one captured write-ahead stamp; a strict (fail-closed) stamp propagates."""
    if not callable(stamp):
        return
    try:
        stamp()
    except Exception:
        if bool(getattr(stamp, "fail_closed", False)):
            raise
        log.debug("review paid dispatch stamp failed (fail-open)", exc_info=True)


def invoke_late_review_paid_stamp(stamp: Any) -> bool:
    """Stamp a send that already failed after dispatch; True when a strict stamp failed.

    Only a transport that crossed the boundary without usage accounting's canonical
    marker (a compatibility or test adapter) reaches this; the canonical lane stamps
    before its claim. That send happened, so its own exception, with its capture and
    attempt ids, stays the failure the caller raises for the retry rail, the native
    cache and the failed-send observer (#1652). A strict stamp's failure is logged
    here, never raised. The stamp keeps the failure for the canonical pre-send
    marker; this fallback neither grants dispatch nor changes the retry policy.
    """
    try:
        invoke_review_paid_stamp(stamp)
    except Exception:
        log.warning("review paid stamp failed after a dispatched send; the send's own failure stays raised",
                    exc_info=True)
        return True
    return False


@contextlib.contextmanager
def bind_api_review_paid_stamp(stamp: Any) -> Iterator[None]:
    """Bind one API review stamp until a canonical physical dispatch occurs."""
    token = _BOUND_API_PAID_STAMP.set(stamp)
    try:
        yield
    finally:
        _BOUND_API_PAID_STAMP.reset(token)


def invoke_bound_api_review_paid_stamp(
    *, fail_closed: bool | None = None,
) -> None:
    """Invoke the bound API stamp at its matching dispatch phase.

    A strict (fail-closed) stamp runs immediately before the usage ledger
    crosses into ``dispatched`` so a veto remains an honest released attempt;
    a fail-open stamp keeps its existing post-transition write-ahead point.
    ``None`` retains the historical unconditional helper behavior for direct
    callers.
    """
    stamp = _BOUND_API_PAID_STAMP.get()
    if fail_closed is not None and bool(getattr(stamp, "fail_closed", False)) != fail_closed:
        return
    invoke_review_paid_stamp(stamp)


def stamp_review_paid_on_dispatch(ctx: Any) -> None:
    """Invoke the caller-installed stamp at the shared dispatch boundary."""
    invoke_review_paid_stamp(
        getattr(ctx, "_review_paid_stamp", None) if ctx is not None else None
    )


def review_operation_binding(request: Any, slot: Any, operation_id: str) -> dict:
    """Bind one physical result to its existing task, material and panel owners."""
    from ouroboros.review_custody import _attempt_key

    return {
        **dict(getattr(request, "reconciliation_identity", {}) or {}),
        "task_id": str(getattr(request, "task_id", "") or ""),
        "surface": str(getattr(request, "surface", "") or ""),
        "slot_id": str(getattr(slot, "slot_id", "") or ""),
        "operation_id": str(operation_id or ""),
        "request_key": _attempt_key(request, slot),
    }


def recorded_binding_conflicts(binding: Any, expected: dict) -> bool:
    """Does a recorded binding name another operation than ``expected``? An absent or
    empty one is no conflict; its pending invocation token is a separate fact."""
    if binding is None:
        return False
    return not isinstance(binding, dict) or bool(binding) and (
        {key: value for key, value in binding.items() if key != "pending_invocation_id"} != expected)


def recorded_review_slot(request: Any, slot: Any, operation_id: str, binding: Any) -> Any:
    """Interpret a historical blank only when its full saved binding proves it.

    Old writers omitted blank processing from identity; new captured blanks are
    distinct from unspecified processing. Never change new requests or saved bytes.
    Both prompt reconstruction and pending collection use this interpretation.
    """
    from dataclasses import replace

    if slot.processing_preference == "" and isinstance(binding, dict):
        saved = {key: value for key, value in binding.items() if key != "pending_invocation_id"}
        if (all(saved.get(key) for key in ("subject_hash", "review_contract", "roster_hash", "root_task_id"))
                and saved != review_operation_binding(request, slot, operation_id)):
            legacy = replace(slot, processing_preference=None)
            if saved == review_operation_binding(request, legacy, operation_id):
                return legacy
    return slot


def recorded_invocation_tokens(row: dict) -> set:
    """Every pending invocation token a frozen seat row recorded (row, usage, binding)."""
    return {str(value) for value in (
        row.get("pending_invocation_id"), (row.get("usage") or {}).get("pending_invocation_id"),
        (row.get("recovery_binding") or {}).get("pending_invocation_id")) if value}


def review_reconciliation_identity(request: Any, slots: list, *, root_task_id: str, contract: Any = "") -> dict:
    """Reuse the caller's material/cycle identity and the exact configured roster."""
    import hashlib
    import json
    from dataclasses import asdict, is_dataclass
    from ouroboros.review_execution import review_output_contract

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                         default=str).encode("utf-8")).hexdigest()

    supplied = dict(getattr(request, "reconciliation_identity", None) or {})
    roster = []
    for slot in slots:
        values = asdict(slot) if is_dataclass(slot) else dict(getattr(slot, "__dict__", {}) or {})
        # A read-only seat keeps the roster identity recorded before seats carried
        # rights (#1609); writable rights are identity, as in ``_attempt_key``.
        roster.append({k: v for k, v in values.items()
                       if k not in {"timeout_sec", "transport_timeout_sec"}
                       and (k != "processing_preference" or v is not None)
                       and (k != "session_access" or str(v or "readonly") != "readonly")})
    retry_key = getattr(request, "retry_key", None)
    return {
        "subject_hash": digest(retry_key or {
            "subject": getattr(request, "subject", ""), "goal": getattr(request, "goal", ""),
            "scope": getattr(request, "scope", ""), "evidence": getattr(request, "evidence", ""),
            "evidence_refs": getattr(request, "evidence_refs", []), "messages": getattr(request, "messages", []),
            "slot_messages": getattr(request, "slot_messages", []),
        }),
        "review_contract": str(contract or digest({
            "rendered": review_output_contract(request) if hasattr(request, "policy") else "",
            "policy": getattr(request, "policy", ""),
        })),
        "roster_hash": digest(roster), "epoch": str(retry_key or ""),
        **supplied,
        "root_task_id": str(root_task_id), "task_attempt": getattr(request, "task_attempt", None),
    }
