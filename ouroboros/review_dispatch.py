"""Reviewer dispatch primitives: the row-identity mint (moved whole from
``review_substrate.py`` at the module-size gate, same split shape as
``skill_review_cycles.py``) and the write-ahead PAID stamp seam (owner
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
import math
import pathlib
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


def task_acceptance_zero_physical_refusal(evidence: Any, *, retrieving: bool = False) -> dict[str, str]:
    """Describe an acceptance refusal that needs no reviewer transport.

    A retrieving row (native episode, agent session) reads the exact source
    itself, so a partial tool-result PROJECTION does not refuse it; the
    immutable-core overflow refuses every delivery — no owner requirement is
    truncated for any reviewer."""
    packet = evidence if isinstance(evidence, dict) else {}
    # Only a genuinely UNAVAILABLE source withholds the panel. A row the budget
    # ladder shed still has a durable, actor-resolvable source ref, so it is a
    # disclosed omission — refusing on it burned real acceptance panels for $0
    # while the reviewer could have read the exact bytes.
    partials = packet.get("__unresolved_partial_artifacts__")
    unavailable = (
        [row for row in partials
         if isinstance(row, dict) and str(row.get("status") or "") == "source_unavailable"]
        if isinstance(partials, list) else ([partials] if partials else [])
    )
    if unavailable and not retrieving:
        return {
            "status": "degraded_partial_source",
            "summary": (
                "A decision-bearing tool result remains partial and its exact source "
                "is unavailable; acceptance cannot treat that projection as complete."
            ),
        }
    overflow = packet.get("__immutable_core_overflow__")
    if overflow:
        reason = str((overflow if isinstance(overflow, dict) else {}).get("reason") or "").strip()
        return {
            "status": "degraded_core_overflow",
            "summary": (
                "Immutable owner requirements do not fit the acceptance evidence "
                "budget; no requirement was silently truncated."
                + (f" {reason}" if reason else "")
            ),
        }
    return {}


def acceptance_slot_fit(
    slot: Any, executor: Any, *, slot_input_caps: Any = None,
) -> tuple[int, int]:
    """This slot's calibrated input cap and the rendered prompt's token estimate.

    The packet ceiling is resolved once against the review QUORUM's windows, so
    a narrower slot in the same panel needs its own fit check before any send.
    An unmeasurable prompt or an absent cached cap reads ``(0, 0)`` and
    dispatches — the fit check is a backstop, never a new way to withhold a
    panel.
    """
    from ouroboros.review_evidence import _ACCEPT_DENSE_CHARS_PER_TOKEN

    try:
        chars = int(executor.prompt_chars())
        cap = int((slot_input_caps or {}).get(slot.model, 0) or 0)
        return cap, math.ceil(
            chars / _ACCEPT_DENSE_CHARS_PER_TOKEN
        )
    except Exception:
        log.debug("acceptance per-slot fit check failed; dispatching", exc_info=True)
        return 0, 0


def run_zero_physical_task_acceptance(
    request: Any, slots: Any, *, drive_root: Any, usage_ctx: Any,
) -> Any:
    """Return the substrate's synthetic refusal when EVERY row would be refused
    free, or ``None`` for physical work — a mixed panel refuses its packet rows
    inside `_run_slot` ($0) and runs its retrieving rows."""
    if not all(
        task_acceptance_zero_physical_refusal(
            request.evidence, retrieving=bool(getattr(slot, "retrieves", False)))
        for slot in slots
    ):
        return None
    from ouroboros.review_substrate import run_review_request

    return run_review_request(
        request, slots=slots, drive_root=pathlib.Path(drive_root), usage_ctx=usage_ctx,
    )


def claim_task_acceptance_dispatch(
    drive_root: Any,
    root_task_id: str,
    task_id: str,
    binding: dict[str, Any],
) -> dict[str, Any]:
    """Atomically claim the canonical wallet immediately before dispatch."""
    from ouroboros.task_results import claim_task_acceptance_review_cycle

    return claim_task_acceptance_review_cycle(
        drive_root, root_task_id, binding, claimed_by_task_id=task_id,
    )


def task_acceptance_preclaim_refusal(ctx: Any) -> Any:
    """Project every free refusal before assembly and again at dispatch."""
    from ouroboros.review_substrate import ReviewRunResult
    from ouroboros.task_results import project_task_acceptance_review_capacity

    projection = project_task_acceptance_review_capacity(
        ctx.tools._ctx,
        binding_hash=str((ctx.review_binding or {}).get("binding_hash") or ""),
        task_id=str(ctx.task_id or ""),
        # A-material: refuse a PAID dispatch whose material the tree already
        # bought, even when the binding hash moved (a cosmetic tool call moves it).
        paid_identity=str((ctx.review_binding or {}).get("paid_identity") or ""),
    )
    if projection.get("state") == "available" and not projection.get("binding_seen"):
        return None
    reason = (
        "binding_dispatch_already_claimed"
        if projection.get("binding_seen")
        else str(projection.get("reason") or "review_capacity_unknown")
    )
    return ReviewRunResult(
        request={"surface": "task_acceptance", "task_id": str(ctx.task_id)},
        actors=[], parsed_findings=[], aggregate_signal="DEGRADED", degraded=True,
        degraded_reasons=[f"{reason} (no reviewer was called)"],
    )


def slot_id_for_row(index: int, *, prefix: str = SLOT_ID_PREFIX) -> str:
    """Identity of the ``index``-th (1-based) configured reviewer row.

    The single mint for reviewer-slot identity, and the reason the substrate
    contract says slot identity is separate from model identity. Naming a row
    after its own model instead collides two rows that share a model (a supported
    configuration — ``get_scope_review_models`` preserves duplicates on purpose),
    collides two model spellings that sanitize alike (``openai::gpt-5`` and
    ``openai/gpt/5``), and moves a row's identity the moment the owner edits its
    model, so the row's receipts stop lining up with its own history. The model,
    the route and the effort are PROPERTIES of a row, never its name.
    """
    return f"{prefix}_{int(index)}"


class TaskAcceptanceDispatchUnavailable(RuntimeError):
    """A task-acceptance panel was refused before reviewer transport."""


class ReviewPaidStamp:
    """Idempotent, thread-safe once-only wrapper around one durable write.

    Parallel dispatch means two sides can race to be "the first transport
    call" (the commit gate dispatches triad and scope concurrently): the first
    caller performs the durable write-ahead, later callers block on the lock
    until it lands and then no-op — so EVERY side is guaranteed the paid fact
    is durable before its own transport begins. A failing default write is not
    retried and still marks the stamp fired: the terminal record is the primary
    ledger, and ordinary cost accounting remains fail-open. Task acceptance
    uses ``fail_closed=True`` for its already-hard shared wallet authority;
    every parallel caller then observes the same failure and no reviewer
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
                    raise TaskAcceptanceDispatchUnavailable(
                        str(self._failure)
                    ) from self._failure
                return
            try:
                self._write()
            except Exception as exc:
                self._failure = exc
                raise
            finally:
                self.fired = True


def task_acceptance_paid_dispatch_stamp(
    ctx: Any,
    drive_root: Any,
    root_task_id: str,
    task_id: str,
    binding: dict[str, Any],
) -> ReviewPaidStamp:
    """Build the strict once-only wallet claim for a physical panel dispatch.

    The claim checks cancellation and the paid-cycle wallet only (owner R55):
    the launch floor is evaluated once per panel, at loop admission, and a
    running panel is bounded by the R23 deadline clamps and the per-send
    wallet fence."""

    def _claim() -> None:
        refusal = task_acceptance_preclaim_refusal(ctx)
        if refusal is not None:
            reasons = list(getattr(refusal, "degraded_reasons", None) or [])
            raise TaskAcceptanceDispatchUnavailable(
                reasons[0] if reasons else "review_dispatch_refused"
            )
        claim = claim_task_acceptance_dispatch(
            drive_root, root_task_id, task_id, binding,
        )
        if claim.get("status") != "claimed":
            raise TaskAcceptanceDispatchUnavailable(
                str(claim.get("reason") or "review_capacity_unknown")
            )

    return ReviewPaidStamp(_claim, fail_closed=True)


@contextlib.contextmanager
def bind_task_acceptance_paid_dispatch(ctx: Any) -> Iterator[Any]:
    """Bind the canonical tree-wallet claim for this panel's physical seam."""
    from ouroboros.task_results import resolve_task_lineage

    tools_ctx = ctx.tools._ctx
    metadata = getattr(tools_ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    lineage = resolve_task_lineage(ctx.task_id, metadata=metadata)
    root_task_id = str(lineage.get("root_task_id") or ctx.task_id)
    accounting_root = pathlib.Path(str(
        metadata.get("budget_drive_root")
        or getattr(tools_ctx, "budget_drive_root", "")
        or ctx.drive_root
        or getattr(tools_ctx, "drive_root", ".")
    ))
    prior = getattr(tools_ctx, "_review_paid_stamp", None)
    if prior is not None:
        raise TaskAcceptanceDispatchUnavailable("review_dispatch_stamp_already_bound")
    tools_ctx._review_paid_stamp = task_acceptance_paid_dispatch_stamp(
        ctx, accounting_root, root_task_id, ctx.task_id, ctx.review_binding,
    )
    try:
        yield tools_ctx
    finally:
        tools_ctx._review_paid_stamp = prior


def invoke_review_paid_stamp(stamp: Any) -> None:
    """Invoke one captured write-ahead stamp; strict wallet claims propagate."""
    if not callable(stamp):
        return
    try:
        stamp()
    except Exception:
        if bool(getattr(stamp, "fail_closed", False)):
            raise
        log.debug("review paid dispatch stamp failed (fail-open)", exc_info=True)


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

    Strict task-acceptance authority runs immediately before the usage ledger
    crosses into ``dispatched`` so a veto remains an honest released attempt.
    Ordinary commit/skill accounting keeps its existing post-transition
    write-ahead point.  ``None`` retains the historical unconditional helper
    behavior for direct callers.
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
