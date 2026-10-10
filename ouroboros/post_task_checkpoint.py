"""Durable root post-task phase and final-cost checkpoint helpers.

Owner D10 (late-phase Pause, soft same-task Resume): after the answer is
delivered, the root's post-task synthesis is the same task's remaining work.
An owner Pause reaching it at a paid boundary leaves the open, non-terminal
``post_task_synthesis="paused"`` with one ``post_task_pause`` record: the
interrupted stage, the completed stages and an actor-source payload with the
task's frozen synthesis inputs, its original money scope and what the phase
already holds (``LatePhaseRun``). Resume mints one single-use ``grant``; the
existing late-phase executor consumes it, reopens the fence and re-enters the
unfinished stage on CURRENT memory inputs, reassessing with those saved
results — never replaying requests. Completed stages and applied effects do
not repeat. Restart keeps the pause and revokes an unconsumed grant; Stop
degrades the remainder ``owner_stopped``; ``running`` across a restart stays
indeterminate exactly as before.
"""

from __future__ import annotations

import contextvars
import logging
import pathlib
import threading
from typing import Any, Callable, Dict

from ouroboros.cost_projection import (
    COST_ALIAS_PAIRS,
    COST_SCOPE_ROOT_TREE,
    build_cost_presentation,
    carry_cost_meta,
    honest_accounted_amount,
    with_cost_aliases,
)
from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.task_results import (
    TASK_COST_META_FIELDS,
    STATUS_COMPLETED,
    load_task_result,
    merge_review_projection,
    resolve_task_lineage,
    write_task_result,
)
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

POST_TASK_SYNTHESIS_LOCK = threading.Lock()
# None reserves dispatch before the thread binds its live model-wait owner.
POST_TASK_SYNTHESIS_INFLIGHT: dict[tuple[str, str], Any] = {}
POST_TASK_SYNTHESIS_OPEN_STATUSES = frozenset({"pending_once", "running", "paused"})
POST_TASK_SYNTHESIS_TERMINAL_STATUSES = frozenset({"completed", "degraded"})
_TERMINAL_ACCOUNTING_FIELDS = (
    *TASK_COST_META_FIELDS,
    "total_rounds",
    "prompt_tokens",
    "completion_tokens",
)
# Scrub set for the stale-accounting pops below: ABI-3 keeps the retired
# alias spellings HERE (read/scrub tolerance, never an emission) so a legacy
# replica/patch overlay cannot smuggle a stale `cost_usd` past a scrub that
# only knows the honest names — deprecated-wins at the write seam would then
# resurrect the stale amount.
_TERMINAL_ACCOUNTING_SCRUB_FIELDS = (
    *_TERMINAL_ACCOUNTING_FIELDS,
    *(old for _new, old in COST_ALIAS_PAIRS),
)


def post_task_synthesis_is_open(value: Any) -> bool:
    """Return whether a root still owes post-task synthesis."""
    return str(value or "") in POST_TASK_SYNTHESIS_OPEN_STATUSES


def post_task_synthesis_is_terminal(value: Any) -> bool:
    """Return whether canonical post-task synthesis has settled."""
    return str(value or "") in POST_TASK_SYNTHESIS_TERMINAL_STATUSES


def post_task_synthesis_is_paused(value: Any) -> bool:
    """The owner's saved late phase: open work that nothing runs until Resume."""
    return str(value or "") == "paused"


def late_phase_pause_record(row: Any) -> Dict[str, Any]:
    """The saved late-phase record of a paused root result, else ``{}``."""
    checkpoint = row.get("root_phase_checkpoint") if isinstance(row, dict) else None
    if not isinstance(checkpoint, dict) or not post_task_synthesis_is_paused(checkpoint.get("post_task_synthesis")):
        return {}
    record = checkpoint.get("post_task_pause")
    return dict(record) if isinstance(record, dict) else {}


def late_phase_state(drive_root: Any, task_id: str) -> str:
    """``running`` / ``paused`` / ``""`` / ``unknown`` for one root's late phase.

    ``running`` is this process's in-flight key or a durable open phase another
    process may own (a pooled worker); ``unknown`` is unreadable authority.
    """
    if post_task_synthesis_in_flight(drive_root, task_id):
        return "running"
    try:
        row = load_task_result(pathlib.Path(drive_root), str(task_id), strict=True) or {}
    except Exception:
        return "unknown"
    checkpoint = row.get("root_phase_checkpoint") if isinstance(row.get("root_phase_checkpoint"), dict) else {}
    phase = str(checkpoint.get("post_task_synthesis") or "")
    if post_task_synthesis_is_paused(phase):
        return "paused"
    return "running" if post_task_synthesis_is_open(phase) else ""


def update_late_phase_pause(drive_root: Any, task_id: str,
                            mutate: Callable[[Dict[str, Any]], Dict[str, Any] | None]) -> Dict[str, Any] | None:
    """Compare-and-set the paused record under the result lock; None when not applied.

    ``mutate`` sees the current record of a still-``paused`` phase and returns
    its replacement, or None to refuse. Grant, consumption, revocation and Stop
    serialize here, so two Resumes or a Resume racing Stop apply at most once.
    Like the owner fence, this is a field write on the result's own locked
    read-modify-write: it never writes a lifecycle status.
    """
    from ouroboros.task_results import (
        require_writable_task_result_schema,
        stamp_task_result_schema,
        task_result_path,
    )
    from ouroboros.obligations import update_result

    applied: Dict[str, Any] = {}

    def update(latest: Dict[str, Any]) -> Dict[str, Any] | None:
        checkpoint = latest.get("root_phase_checkpoint") if latest else None
        if not isinstance(checkpoint, dict) or not post_task_synthesis_is_paused(
                checkpoint.get("post_task_synthesis")):
            return None
        require_writable_task_result_schema(latest)
        record = mutate(dict(checkpoint.get("post_task_pause") or {}))
        if record is None:
            return None
        applied["record"] = record
        return stamp_task_result_schema({**latest, "root_phase_checkpoint": {**checkpoint, "post_task_pause": record}})

    path = task_result_path(pathlib.Path(drive_root), str(task_id), create=False)
    if not path.is_file():
        return None
    update_result(path, update, strict_existing_dict=True)
    return applied.get("record")


class LatePhaseRun:
    """What one root late phase already holds, carried across an owner Pause.

    Stage-owned facts, never request replay: ``marks`` name completed steps
    whose effects must not repeat (a backlog append, a promotion step).
    ``drafts`` held the retired dialogue writer's confirmed room drafts; no stage
    writes it any more, and it stays only so a phase parked by an earlier
    version still loads.
    """

    def __init__(self, marks: Any = (), drafts: Any = None) -> None:
        self.marks = {str(mark) for mark in marks or ()}
        self.drafts = {str(key): str(value) for key, value in (drafts or {}).items()} if isinstance(drafts, dict) else {}

    def mark(self, name: str) -> None:
        self.marks.add(str(name))

    def marked(self, name: str) -> bool:
        return str(name) in self.marks


_LATE_PHASE_RUN: contextvars.ContextVar = contextvars.ContextVar("late_phase_run", default=None)


def current_late_phase_run() -> LatePhaseRun | None:
    return _LATE_PHASE_RUN.get()


def bind_late_phase_run(run: LatePhaseRun | None):
    """Bind the run to this execution context; returns the reset token."""
    return _LATE_PHASE_RUN.set(run)


def reset_late_phase_run(token: Any) -> None:
    _LATE_PHASE_RUN.reset(token)


def post_task_synthesis_in_flight(drive_root: Any, task_id: str) -> bool:
    """Whether THIS process is still running the paid post-task synthesis of
    ``task_id`` on ``drive_root`` — the in-flight key the pipeline holds from
    dispatch until its terminal checkpoint is stored (GR6-1, widened to the
    non-blocking lane): a direct-chat turn's loop returns and its liveness
    ends while the synthesis thread still bills, so the key is the live
    physical ownership the stop ingress and custody must see. Process-local
    on purpose: a durable ``running`` phase alone cannot tell a live worker
    from one that died before the boot reconciler degraded it."""
    tid = str(task_id or "").strip()
    if not tid or not drive_root:
        return False
    try:
        root_key = str(pathlib.Path(drive_root).resolve(strict=False))
    except (TypeError, OSError, ValueError):
        return False
    with POST_TASK_SYNTHESIS_LOCK:
        return (root_key, tid) in POST_TASK_SYNTHESIS_INFLIGHT


def post_task_model_wait(drive_root: Any, task_id: str):
    """The existing process-local synthesis owner, never a durable liveness guess."""
    key = (str(pathlib.Path(drive_root).resolve(strict=False)), str(task_id))
    with POST_TASK_SYNTHESIS_LOCK:
        owner = POST_TASK_SYNTHESIS_INFLIGHT.get(key)
    return owner if owner is not None and not owner.closed else None


def post_task_model_waits(drive_root: Any) -> list:
    root = str(pathlib.Path(drive_root).resolve(strict=False))
    with POST_TASK_SYNTHESIS_LOCK:
        owners = [owner for (path, _task), owner in POST_TASK_SYNTHESIS_INFLIGHT.items() if path == root]
    return [owner for owner in owners if owner is not None and not owner.closed]


def _delegated_receipt_counts(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, dict) or value.get("evidence_read_failed"):
        return None
    counts = (value.get("delegated_runs_started"), value.get("delegated_runs_settled"))
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts):
        return None
    return counts


def project_replica_task_result_fields(
    canonical_fields: Dict[str, Any],
    replica_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """Return the replica overlay permitted over a canonical task result.

    A terminal canonical post-task checkpoint owns its synthesis fields and
    accounting snapshot. Canonical custody also retains non-regressing delegation
    receipts and canonical-if-present reconciliation disclosures under the narrow
    rules below. The replica continues to own acceptance, result, and trace fields;
    review snapshots retain the newest host publication of each panel.
    ``updated_at`` is monotonic metadata only; it never selects field authority.
    """
    from ouroboros.terminal_time import preserve_terminal_attempt, replica_terminal_time
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES
    from ouroboros.acceptance_history import preserve_acceptance_history

    overlay = preserve_terminal_attempt(canonical_fields, replica_fields)
    overlay = preserve_acceptance_history(canonical_fields, overlay)
    if (canonical_fields.get("status") in _TRULY_TERMINAL_STATUSES
            or replica_fields.get("status") in _TRULY_TERMINAL_STATUSES):
        overlay["terminal_time"] = replica_terminal_time(canonical_fields, replica_fields)
    # The receiving drive's first accepted terminal transition owns provenance,
    # including its absence on historical rows; replicas cannot originate it.
    overlay.pop("canonical_terminal_projection_origin", None)
    overlay.pop("pause_notices", None)  # confirmed host transitions alone own pending chat disclosure
    # Unread-mail custody is a union: a stale replica never drops a canonical row.
    from ouroboros.task_custody import merge_unread_mail

    custody = merge_unread_mail(canonical_fields.get("unread_mailbox"), overlay.get("unread_mailbox"))
    if custody is not None:
        overlay["unread_mailbox"] = custody
    canonical_cost = canonical_fields.get("cost_presentation")
    replica_cost = overlay.get("cost_presentation")
    if (isinstance(canonical_cost, dict) and canonical_cost.get("scope") == COST_SCOPE_ROOT_TREE
            and (not isinstance(replica_cost, dict) or replica_cost.get("scope") != COST_SCOPE_ROOT_TREE)):
        # A worker's own bucket cannot replace an already published tree bucket.
        # Its own monetary fields remain own; the scoped amount/facts stay paired.
        overlay.pop("cost_presentation", None)
    if "review_projection" in overlay:
        overlay["review_projection"] = merge_review_projection(
            canonical_fields.get("review_projection"), overlay["review_projection"],
        )
    canonical_checkpoint = canonical_fields.get("root_phase_checkpoint")
    canonical_post_task = (
        str(canonical_checkpoint.get("post_task_synthesis") or "")
        if isinstance(canonical_checkpoint, dict)
        else ""
    )
    if post_task_synthesis_is_terminal(canonical_post_task):
        replica_checkpoint = overlay.get("root_phase_checkpoint")
        merged_checkpoint = dict(canonical_checkpoint)
        if isinstance(replica_checkpoint, dict):
            merged_checkpoint.update(replica_checkpoint)
        merged_checkpoint["post_task_synthesis"] = canonical_post_task
        # The canonical phase owns this tree observation, including its absence
        # on older records. A replica cannot invent or replace that evidence.
        merged_checkpoint.pop("accounting", None)
        if "accounting" in canonical_checkpoint:
            merged_checkpoint["accounting"] = canonical_checkpoint["accounting"]
        if "post_task_stop_reason" in canonical_checkpoint:
            merged_checkpoint["post_task_stop_reason"] = canonical_checkpoint[
                "post_task_stop_reason"
            ]
        overlay["root_phase_checkpoint"] = merged_checkpoint
        for field in _TERMINAL_ACCOUNTING_SCRUB_FIELDS:
            overlay.pop(field, None)
    elif post_task_synthesis_is_paused(canonical_post_task) and isinstance(
            overlay.get("root_phase_checkpoint"), dict):
        # A saved late phase is canonical custody: no replica reopens or drops it.
        overlay["root_phase_checkpoint"] = {**overlay["root_phase_checkpoint"], **{
            key: canonical_checkpoint[key] for key in ("post_task_synthesis", "post_task_pause")
            if key in canonical_checkpoint}}

    # Non-Project split synthesis writes this field in the canonical parent
    # root.  A later child replica must not replace it with stale child text.
    if isinstance(canonical_fields.get("continuation_narrative"), dict):
        if str(canonical_fields["continuation_narrative"].get("text") or "").strip():
            overlay.pop("continuation_narrative", None)

    # Write-side custody heals must survive both reducer consumers. A canonical
    # absence still accepts the first replica value.
    for field in (
        "delegated_runs_unreconciled",
        "delegate_terminal_reconciliation",
        # update_focus writes the canonical result only; a split root's worker
        # replica carries the stale (often null) execution-local copy.
        "focus",
        # The terminal-projection obligation and its receipt are canonical
        # bookkeeping (#1154): a replica that still carried the readiness row
        # would resurrect an obligation this drive had already settled, and a
        # replica marker would claim a Project row nobody appended here.
        "canonical_terminal_projection",
        "canonical_terminal_projection_ready",
    ):
        if field in canonical_fields:
            overlay.pop(field, None)

    canonical_envelope = canonical_fields.get("subagent_envelope")
    canonical_evidence = (
        canonical_envelope.get("execution_evidence")
        if isinstance(canonical_envelope, dict)
        else None
    )
    if isinstance(canonical_evidence, dict) and canonical_evidence:
        replica_envelope = overlay.get("subagent_envelope")
        replica_evidence = (
            replica_envelope.get("execution_evidence")
            if isinstance(replica_envelope, dict)
            else None
        )

        canonical_counts = _delegated_receipt_counts(canonical_evidence)
        replica_counts = _delegated_receipt_counts(replica_evidence)
        canonical_wins = not isinstance(replica_evidence, dict) or not replica_evidence
        if isinstance(replica_evidence, dict) and replica_evidence:
            canonical_wins = bool(
                canonical_counts is not None
                and (
                    replica_counts is None
                    or all(a >= b for a, b in zip(canonical_counts, replica_counts))
                )
            )
        if canonical_wins:
            merged_envelope = (
                dict(replica_envelope)
                if isinstance(replica_envelope, dict)
                else dict(canonical_envelope)
            )
            merged_envelope["execution_evidence"] = dict(canonical_evidence)
            canonical_substrate = str(
                canonical_envelope.get("actual_substrate")
                or canonical_fields.get("actual_substrate")
                or ""
            ).strip()
            if canonical_substrate:
                merged_envelope["actual_substrate"] = canonical_substrate
                overlay["actual_substrate"] = canonical_substrate
            if "native_contribution" in canonical_envelope:
                merged_envelope["native_contribution"] = canonical_envelope[
                    "native_contribution"
                ]
            overlay["subagent_envelope"] = merged_envelope

    canonical_updated_at = parse_deadline_ts(canonical_fields.get("updated_at"))
    replica_updated_at = parse_deadline_ts(overlay.get("updated_at"))
    if canonical_updated_at is not None and (
        replica_updated_at is None or canonical_updated_at > replica_updated_at
    ):
        overlay["updated_at"] = canonical_fields["updated_at"]
    return overlay


def project_root_post_task_checkpoint_fields(
    canonical_fields: Dict[str, Any],
    patch_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge a root post-task patch against the CURRENT canonical checkpoint.

    The root writer owns only post-task synthesis and its accounting snapshot;
    acceptance remains whatever the current record says. Once post-task state
    is terminal, an open or different-terminal stale patch cannot replace that
    state or its accounting. A same-terminal patch remains valid so an explicit
    ``refresh`` can update the final cost snapshot.
    """
    overlay = dict(patch_fields)
    if canonical_fields.get("status"):
        # This writer enriches the current lifecycle record; it never owns a
        # possibly stale pre-lock lifecycle transition.
        overlay["status"] = canonical_fields["status"]
    canonical_checkpoint = canonical_fields.get("root_phase_checkpoint")
    patch_checkpoint = overlay.get("root_phase_checkpoint")
    current = (
        dict(canonical_checkpoint)
        if isinstance(canonical_checkpoint, dict)
        else {"phase": "task_acceptance", "status": "not_required", "pass_index": 0}
    )
    patch = dict(patch_checkpoint) if isinstance(patch_checkpoint, dict) else {}
    canonical_post_task = str(current.get("post_task_synthesis") or "")
    patch_post_task = str(patch.get("post_task_synthesis") or "")

    if post_task_synthesis_is_terminal(canonical_post_task):
        current["post_task_synthesis"] = canonical_post_task
        if isinstance(canonical_checkpoint, dict) and "post_task_stop_reason" in canonical_checkpoint:
            current["post_task_stop_reason"] = canonical_checkpoint[
                "post_task_stop_reason"
            ]
        if patch_post_task != canonical_post_task:
            for field in _TERMINAL_ACCOUNTING_SCRUB_FIELDS:
                overlay.pop(field, None)
    else:
        if "post_task_stop_reason" in patch:
            current["post_task_stop_reason"] = patch["post_task_stop_reason"]
        if "post_task_pause" in patch:
            current["post_task_pause"] = patch["post_task_pause"]
        if patch_post_task:
            current["post_task_synthesis"] = patch_post_task
    if patch_post_task and (
        not post_task_synthesis_is_terminal(canonical_post_task)
        or patch_post_task == canonical_post_task
    ):
        current.pop("accounting", None)
        if post_task_synthesis_is_terminal(patch_post_task) and "accounting" in patch:
            current["accounting"] = patch["accounting"]
    overlay["root_phase_checkpoint"] = current
    return overlay


def is_root_post_task(task: Dict[str, Any]) -> bool:
    """Structural root test for the single global post-task synthesis authority."""
    if bool(task.get("_skip_post_task_synthesis")):
        return False
    meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    task_id = str(task.get("id") or task.get("task_id") or "")
    return bool(resolve_task_lineage(
        task_id,
        metadata=meta,
        root_task_id=task.get("root_task_id"),
        parent_task_id=task.get("parent_task_id"),
        delegation_role=task.get("delegation_role"),
        original_task_id=task.get("original_task_id"),
        timeout_retry_from=task.get("timeout_retry_from"),
    )["is_root_task"])


def root_checkpoint_roots(env: Any, task: Dict[str, Any]) -> list[pathlib.Path]:
    """Return the one durable phase authority (compatibility list shape)."""
    raw = task.get("budget_drive_root") or getattr(env, "drive_root", None)
    if not raw:
        return []
    try:
        return [pathlib.Path(raw).resolve(strict=False)]
    except (TypeError, OSError, ValueError):
        return []


def _root_accounting_snapshot(root_task_id: str, subtree: Dict[str, Any] | None) -> Dict[str, Any]:
    """Keep one root-tree ledger observation distinct from own-task money.

    This records row states, not an invoice or local-work closure. The phase
    owner supplies a fresh breakdown; unavailable refreshes retain no old proof.
    """
    source = subtree if isinstance(subtree, dict) else {}
    counts = source.get("attempt_counts")
    return {
        "schema": "ouroboros.root_cost_snapshot.v1",
        "scope": "root_tree",
        "root_task_id": root_task_id,
        "cost_accounting_status": "available" if subtree is not None else "unavailable",
        "accounted_upper_bound_usd": honest_accounted_amount(source),
        **{key: source.get(key) for key in (
            "unresolved_upper_bound_usd", "reserved_usd", "non_final_rows", "unknown_unmetered",
        )},
        "attempt_counts": (
            {"unresolved": counts.get("unresolved", 0)} if isinstance(counts, dict) else None
        ),
        "ledger_integrity_degraded": (
            source.get("integrity_degraded") if subtree is not None else True
        ),
    }


def set_root_post_task_checkpoint(
    env: Any,
    task: Dict[str, Any],
    status: str,
    *,
    stop_reason: str = "",
    pause: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """Merge the phase marker and return the record actually stored, if any."""
    if not is_root_post_task(task):
        return
    task_id = str(task.get("id") or task.get("task_id") or "")
    if not task_id:
        return
    requested_status = str(status)
    roots = root_checkpoint_roots(env, task)
    if not roots:
        return
    authority_root = roots[0]
    finalized_event: Dict[str, Any] | None = None
    # A late cost refresh can settle concurrently with post-task synthesis. A
    # shared critical section makes that refresh and the final snapshot linear.
    with POST_TASK_SYNTHESIS_LOCK:
        existing = load_task_result(authority_root, task_id) or {}
        checkpoint = existing.get("root_phase_checkpoint")
        saved = str(checkpoint.get("post_task_synthesis") or "") if isinstance(checkpoint, dict) else ""
        effective_status = saved if requested_status == "refresh" and saved else requested_status
        cost_fields: Dict[str, Any] = {"cost_final": False, "cost_with_children_partial": True}
        accounting = None
        if post_task_synthesis_is_terminal(effective_status):
            metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
            logical_root_id = str(task.get("root_task_id") or metadata.get("root_task_id") or task_id)
            accounting = _root_accounting_snapshot(logical_root_id, None)
            try:
                from ouroboros.usage_accounting import usage_breakdown
                from supervisor.state import reconstruct_task_cost

                cost_fields.update(reconstruct_task_cost(task_id, fields=True, drive_root=authority_root))
                subtree = usage_breakdown(
                    authority_root, root_task_id=logical_root_id
                )
                accounting = _root_accounting_snapshot(logical_root_id, subtree)
                subtree_final = bool(subtree.get("cost_final"))
                subtree_amount = honest_accounted_amount(subtree)
                cost_fields.update({
                    "accounted_upper_bound_usd_with_children": (
                        round(subtree_amount, 6) if subtree_amount is not None else None
                    ),
                    "cost_with_children_partial": not subtree_final,
                    "cost_final": bool(cost_fields.get("cost_final") and subtree_final),
                    # #498: a ROOT's terminal record speaks for its whole tree, so
                    # the carrier is rebuilt from the SUBTREE bucket, replacing the
                    # own-scope one `reconstruct_task_cost` just attached. Presence
                    # is what selects it — a null subtree amount stays null and never
                    # falls back to the root's own (often zero) number.
                    "cost_presentation": build_cost_presentation(
                        subtree, scope=COST_SCOPE_ROOT_TREE),
                })
            except Exception:
                log.error("Failed to refresh final root cost projection for %s", task_id, exc_info=True)
                accounting = _root_accounting_snapshot(logical_root_id, None)
                cost_fields.update({
                    "cost_accounting_status": "unavailable",
                    "cost_accounting_error": "ledger_unavailable",
                    "accounted_upper_bound_usd": None,
                    "accounted_upper_bound_usd_with_children": None,
                    "cost_presentation": None,
                })
        # SSOT cost naming (C2/F12/ABI-3): every branch above writes the honest
        # names directly onto the honest-named `reconstruct_task_cost` fields
        # (Ф3.1 fix-round — producers no longer touch the retired spellings);
        # the seam stays as the LAST step as the idempotent invariant guard —
        # it re-normalizes amounts and would strip any retired key a future
        # mutation leaked, so this producer can never persist a diverged pair.
        cost_fields = with_cost_aliases(cost_fields)
        checkpoint_patch = {"post_task_synthesis": effective_status}
        if accounting is not None:
            checkpoint_patch["accounting"] = accounting
        if stop_reason:
            checkpoint_patch["post_task_stop_reason"] = str(stop_reason)
        if pause is not None:
            checkpoint_patch["post_task_pause"] = dict(pause)
        stored: Dict[str, Any] | None = None
        try:
            stored = write_task_result(
                authority_root,
                task_id,
                str(existing.get("status") or task.get("status") or STATUS_COMPLETED),
                _field_projector=project_root_post_task_checkpoint_fields,
                root_task_id=str(task.get("root_task_id") or task_id),
                parent_task_id=task.get("parent_task_id"),
                budget_drive_root=str(authority_root),
                child_drive_root=task.get("child_drive_root") or task.get("drive_root"),
                project_id=str(task.get("project_id") or ""),
                root_phase_checkpoint=checkpoint_patch,
                **cost_fields,
            )
        except Exception:
            log.debug("Failed to update root post-task checkpoint", exc_info=True)
        stored_checkpoint = (
            stored.get("root_phase_checkpoint") if isinstance(stored, dict) else None
        )
        stored_post_task = (
            str(stored_checkpoint.get("post_task_synthesis") or "")
            if isinstance(stored_checkpoint, dict)
            else ""
        )
        if post_task_synthesis_is_terminal(stored_post_task):
            finalized_event = {
                "type": "task_cost_finalized",
                "ts": utc_now_iso(),
                "task_id": task_id,
                "root_task_id": str(stored.get("root_task_id") or task.get("root_task_id") or task_id),
                "post_task_status": stored_post_task,
                # The typed stop disclosure rides the same event (owner Stop-now
                # during synthesis, restart recovery): absent when nothing stopped.
                **({"post_task_stop_reason": str(stored_checkpoint.get("post_task_stop_reason"))}
                   if stored_checkpoint.get("post_task_stop_reason") else {}),
                # ABI-3: cost pair CONVERTED from a possibly-legacy stored row
                # (deprecated-wins) — the event carries honest names only.
                **carry_cost_meta(stored),
                **{
                    field: stored[field]
                    for field in ("total_rounds", "prompt_tokens", "completion_tokens")
                    if field in stored
                },
            }
    if finalized_event is not None:
        try:
            append_jsonl(authority_root / "logs" / "events.jsonl", finalized_event)
        except Exception:
            log.warning("Failed to persist finalized task cost for %s", task_id, exc_info=True)
        else:
            # v6.74.0 (D3): the durable append above is the record of truth; the
            # live UI push is best-effort. `get_bridge()` ASSERTS `init()` was
            # called and raised in post-task contexts without a live bus
            # (benchmark workers, headless finalization) — pure log noise.
            # `try_get_bridge` pushes only when a bridge actually exists.
            try:
                from supervisor.message_bus import try_get_bridge

                bridge = try_get_bridge()
                if bridge is not None:
                    # A bridge exists only in the server process, where the
                    # live RUNNING table is available for addressing.
                    from supervisor.log_addressing import address_handler_push

                    bridge.push_log(address_handler_push(authority_root, dict(finalized_event)))
            except Exception:
                log.debug("Live push of finalized task cost skipped for %s", task_id, exc_info=True)
    settle_terminal_projection(authority_root, task_id, task=task)
    if stored is None:
        # The write failed: nothing was stored, and the contract is "the record
        # actually stored, if any" — a pre-existing row must not impersonate a
        # persisted checkpoint (callers treat None as "not persisted").
        return None
    # Settlement writes receipts/retirement and can race another enrichment.
    # Never hand a caller the pre-settlement obligation as current authority.
    try:
        return load_task_result(authority_root, task_id, strict=True)
    except Exception:
        log.warning("Failed to read settled root post-task checkpoint for %s", task_id, exc_info=True)
        return None


# Compatibility exports: the continuation owns no synthesis or result lock.
from ouroboros.terminal_projection import (  # noqa: E402, F401
    SETTLEMENT_NONE, SETTLEMENT_DEFERRED, SETTLEMENT_SETTLED,
    clear_terminal_projection_obligation as _clear_terminal_projection_obligation,
    settle_terminal_projection,
)


def _root_post_task_phase(env: Any, task: Dict[str, Any]) -> str:
    if not is_root_post_task(task):
        return ""
    task_id = str(task.get("id") or task.get("task_id") or "")
    roots = root_checkpoint_roots(env, task)
    existing = load_task_result(roots[0], task_id) if roots and task_id else None
    checkpoint = existing.get("root_phase_checkpoint") if isinstance(existing, dict) else None
    return str(checkpoint.get("post_task_synthesis") or "") if isinstance(checkpoint, dict) else ""


def root_post_task_already_completed(env: Any, task: Dict[str, Any]) -> bool:
    return post_task_synthesis_is_terminal(_root_post_task_phase(env, task))


def root_post_task_paused(env: Any, task: Dict[str, Any]) -> bool:
    """A saved late phase: only the owner's Resume grant continues it."""
    return post_task_synthesis_is_paused(_root_post_task_phase(env, task))
