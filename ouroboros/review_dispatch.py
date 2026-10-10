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

Task acceptance binds one strict, exact-hash claim on the locked
``task_acceptance_review_accounting`` tree wallet per panel to this stamp
(``task_acceptance_paid_dispatch_stamp``); a binding or paid identity already
claimed is ``unknown``, never resend authority. Before a new panel is prepared,
``reconcile_pending_acceptance_runs`` collects already-paid panels at $0 from
their original requests and rosters, and concurrent progress forces recollection
so publication keeps the settled facts.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import math
import pathlib
import threading
from typing import Any, Callable, Dict, Iterator

log = logging.getLogger(__name__)
_BOUND_API_PAID_STAMP: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "ouroboros_review_api_paid_stamp", default=None,
)
_ACCEPTANCE_COLLECTION_LOCK = threading.Lock()

# Identity prefixes for the configured reviewer surfaces. A surface that fans
# rows out registers its prefix here rather than spelling one inline, so
# ``slot_id_for_row`` stays the only place a row id is built.
SLOT_ID_PREFIX = "slot"
SCOPE_SLOT_ID_PREFIX = "scope_slot"
PLAN_SLOT_ID_PREFIX = "plan_slot"


def task_acceptance_zero_physical_refusal(
    evidence: Any, *, retrieving: bool = False, delivery: Any = None,
) -> dict[str, str]:
    """Describe an acceptance refusal that needs no reviewer transport.

    A retrieving row (native episode, agent session) reads the exact source
    itself, so a partial tool-result PROJECTION does not refuse it. The
    immutable-core overflow refuses every PACKET row — no owner requirement is
    truncated for any reviewer — but a retrieving row only when its work order
    found no exact packet source it can open (``delivery``, one row of
    ``ReviewRequest.slot_source_delivery``): the packet ceiling sizes a
    delivery, it never grants or removes a reader's right to review (#1329)."""
    delivery = delivery if isinstance(delivery, dict) else {}
    if retrieving and delivery.get("status") == "unavailable":
        preparation = delivery.get("preparation_error")
        if isinstance(preparation, dict):
            return {"status": str(preparation.get("code") or "review_source_preparation_failed"),
                    "summary": "Retrieving reviewer preparation failed; nothing was sent. " + str(delivery.get("reason") or "")}
        return {
            "status": "degraded_source_unreachable",
            "summary": ("This retrieving reviewer has no exact source of the complete acceptance "
                        "packet it can open; nothing was sent. " + str(delivery.get("reason") or "")).strip(),
        }
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
    if overflow and not (retrieving and delivery.get("status") == "paged"):
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


def task_acceptance_row_refusal(request: Any, slot: Any) -> dict[str, str]:
    """The zero-physical refusal of ONE panel row, with its own source delivery."""
    return task_acceptance_zero_physical_refusal(
        request.evidence, retrieving=bool(getattr(slot, "retrieves", False)),
        delivery=(getattr(request, "slot_source_delivery", None) or {}).get(str(getattr(slot, "slot_id", ""))))


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

    caps = slot_input_caps or {}
    slot_id = str(getattr(slot, "slot_id", "") or "")
    if caps and slot_id not in caps and slot.model not in caps:
        raise ValueError("Acceptance capacity has no entry for the frozen reviewer slot")
    try:
        chars = int(executor.prompt_chars())
        cap = int(caps.get(slot_id, caps.get(slot.model, 0)) or 0)
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
    if not all(task_acceptance_row_refusal(request, slot) for slot in slots):
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


def collect_task_acceptance_run(run: dict, *, drive_root: Any, usage_ctx: Any, controller: Any = None) -> Any:
    """Collect the recorded operation at zero new dispatch, using its exact inputs.

    The existing host review record owns the request and roster; custody owns live
    workers and complete producer artifacts. Collection branches BEFORE the
    ordinary runner (``review_operation.collect_recorded_acceptance_run``): exact
    producer CAS, a live local worker, or an attach-only read of a proven
    delegated run, parsed locally. Missing custody cannot turn it into a send.
    """
    from ouroboros.review_operation import collect_recorded_acceptance_run

    return collect_recorded_acceptance_run(run, drive_root=pathlib.Path(drive_root), usage_ctx=usage_ctx,
                                           controller=controller)


def reconcile_pending_acceptance_runs(
    llm_trace: dict, *, drive_root: Any, usage_ctx: Any, controller: Any = None,
) -> int:
    """Collect every already-paid acceptance panel still recorded as running, $0.

    The dispatch barrier (``ReviewRequest.drain_deadline``) returns the host right
    after dispatch, so a panel whose subject was re-authored before it settled is
    left with ``pending_dispatch`` rows that nothing reads: the free-replay lookup
    matches only the CURRENT binding or paid identity, so verdicts the tree already
    bought were discarded. This is the acceptance twin of plan review's
    reconcile-before-supersede (I3): it sends nothing, pays nothing, samples no new
    evidence, and advances only producer facts on runs the tree already owns.
    Idempotent by ``acceptance_run_pending`` alone -- a settled, custody-lost or
    already-collected run is never collected again. Returns how many runs advanced.
    """
    from ouroboros.loop_acceptance_review import acceptance_run_pending

    advanced = 0
    for run in (llm_trace.get("review_runs") or []):
        # Agent-tool acceptance runs carry no barrier and drain synchronously.
        if not isinstance(run, dict) or run.get("authority") != "host_root":
            continue
        if not isinstance(run.get("request"), dict) or not run.get("slot_roster"):
            continue
        while True:
            with _ACCEPTANCE_COLLECTION_LOCK:
                if not acceptance_run_pending(run):
                    break
                actors = run.get("actors")
                snapshot = dict(run)
            try:
                # Collection may read a delegated run; never hold the lock over I/O.
                result = collect_task_acceptance_run(
                    snapshot, drive_root=drive_root, usage_ctx=usage_ctx,
                    **({"controller": controller} if controller is not None else {}),
                )
            except (OSError, TimeoutError, ValueError, KeyError) as exc:
                log.warning("acceptance run %s could not be reconciled: %s",
                            str(run.get("panel_id") or "")[:16], exc)
                break
            with _ACCEPTANCE_COLLECTION_LOCK:
                if not acceptance_run_pending(run):
                    # This caller observed pending too and may own publication
                    # of the transition, even when another collector applied it.
                    advanced += 1
                    break
                # Collectors replace this list, never mutate its rows. A concurrent
                # collection won: recollect its facts instead of erasing progress.
                if run.get("actors") is not actors:
                    continue
                # Keep the host panel identity and paid request.
                run.update({key: value for key, value in vars(result).items()
                            if key not in {"request", "panel_id"}})
                advanced += not acceptance_run_pending(run)
                break
    return advanced


def task_acceptance_preclaim_refusal(ctx: Any) -> Any:
    """Project every free refusal before assembly and again at dispatch."""
    from ouroboros.review_substrate import ReviewRunResult
    from ouroboros.task_results import project_task_acceptance_review_capacity

    if getattr(ctx, "historical_purpose", None):
        from ouroboros.acceptance_late import historical_preclaim_refusal

        reason = historical_preclaim_refusal(ctx)
        if reason:
            return ReviewRunResult(request={"surface": "task_acceptance", "task_id": str(ctx.task_id)},
                actors=[], parsed_findings=[], aggregate_signal="DEGRADED", degraded=True,
                degraded_reasons=[f"{reason} (no reviewer was called)"])
    projection = project_task_acceptance_review_capacity(
        ctx.tools._ctx,
        binding_hash=str((ctx.review_binding or {}).get("binding_hash") or ""),
        task_id=str(ctx.task_id or ""),
        # A-material: refuse a PAID dispatch whose material the tree already
        # bought, even when the binding hash moved (a cosmetic tool call moves it).
        paid_identity=str((ctx.review_binding or {}).get("paid_identity") or ""),
        purpose=str(getattr(ctx, "purpose", "") or ""),
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
    configuration — the factory pool repeats Main on purpose),
    collides two model spellings that sanitize alike (``openai::gpt-5`` and
    ``openai/gpt/5``), and moves a row's identity the moment the owner edits its
    model, so the row's receipts stop lining up with its own history. The model,
    the route and the effort are PROPERTIES of a row, never its name.
    """
    return f"{prefix}_{int(index)}"


class TaskAcceptanceDispatchUnavailable(RuntimeError):
    """A task-acceptance panel was refused before reviewer transport."""


class ReviewPaidStamp:
    """Idempotent, thread-safe stamp shared by the seats of one review wave.

    The first caller attempts the durable write-ahead; siblings wait for its
    result. A failed write is not retried here and still marks the stamp fired.
    Ordinary cost accounting fails open, with the terminal record authoritative.
    Commit-gate, review_change, acceptance and resumed-skill stamps use
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
    prior_authority = getattr(tools_ctx, "_review_paid_authority", None)
    if prior is not None:
        raise TaskAcceptanceDispatchUnavailable("review_dispatch_stamp_already_bound")
    tools_ctx._review_paid_stamp = task_acceptance_paid_dispatch_stamp(
        ctx, accounting_root, root_task_id, ctx.task_id, ctx.review_binding,
    )
    # The immutable operation checkpoint retains the wallet binding before the
    # physical seam claims it. A missing claim still never authorizes recovery.
    tools_ctx._review_paid_authority = ({
        "schema_version": 1, "authority": "host_root", "lineage": dict(lineage),
        "binding": {key: ctx.review_binding.get(key) for key in (
            "binding_hash", "candidate_hash", "evidence_revision", "fence_hash", "paid_identity")},
    } if lineage.get("is_root_task") else None)
    try:
        yield tools_ctx
    finally:
        tools_ctx._review_paid_stamp = prior
        tools_ctx._review_paid_authority = prior_authority


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
        roster.append({k: v for k, v in values.items()
                       if k not in {"timeout_sec", "transport_timeout_sec"}
                       and (k != "processing_preference" or v)})
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


def retrieving_acceptance_packet(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """The packet a NATIVE row receives (R4/R15): the same host-attested exhibits
    WITHOUT the freely degradable tail the api ladder spends first — the
    tool-trajectory rows and artifact previews — because that row reads those
    sources itself at the pointers. Every section key survives, so an
    `evidence_ref` naming it still resolves against the FULL dict (the ref
    authority never changes), and the omission is manifested like every other."""
    packet = dict(evidence)
    manifest_present = "omissions_manifest" in packet
    manifest = packet.get("omissions_manifest")
    # A sequence is a manifest; anything else present (None, a dict, a string) is
    # malformed and is normalized to an empty list — never carried as-is, never its keys.
    omissions = list(manifest) if isinstance(manifest, (list, tuple)) else []
    trajectory = packet.get("tool_trajectory")
    if isinstance(trajectory, list) and trajectory:
        packet["tool_trajectory"] = [{
            "retrieve": "tool-trajectory rows withheld from this delivery; read the trajectory log at the pointer",
            "calls": len(trajectory),
        }]
        omissions.append({"section": "tool_trajectory", "omitted": len(trajectory), "reason": "retrieving_delivery"})
    artifacts = packet.get("artifacts")
    if isinstance(artifacts, list):
        rows = [
            {k: v for k, v in row.items() if k != "preview"} if isinstance(row, dict) and row.get("preview") else row
            for row in artifacts
        ]
        stripped = sum(1 for before, after in zip(artifacts, rows) if before is not after)
        if stripped:
            packet["artifacts"] = rows
            omissions.append({"section": "artifact_previews", "omitted": stripped, "reason": "retrieving_delivery"})
    if omissions or (manifest_present and not isinstance(manifest, list)):
        packet["omissions_manifest"] = omissions  # normalized whenever present and not a list; an absent key stays absent
    return packet
