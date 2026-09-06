"""Durable root post-task phase and final-cost checkpoint helpers."""

from __future__ import annotations

import logging
import pathlib
import threading
from datetime import datetime, timezone
from typing import Any, Dict

from ouroboros.cost_projection import (
    COST_ALIAS_PAIRS,
    carry_cost_meta,
    honest_accounted_amount,
    with_cost_aliases,
)
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
POST_TASK_SYNTHESIS_INFLIGHT: set[tuple[str, str]] = set()
POST_TASK_SYNTHESIS_OPEN_STATUSES = frozenset({"pending_once", "running"})
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


def _parse_updated_at(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
    overlay = dict(replica_fields)
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
        if "post_task_stop_reason" in canonical_checkpoint:
            merged_checkpoint["post_task_stop_reason"] = canonical_checkpoint[
                "post_task_stop_reason"
            ]
        overlay["root_phase_checkpoint"] = merged_checkpoint
        for field in _TERMINAL_ACCOUNTING_SCRUB_FIELDS:
            overlay.pop(field, None)

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

    canonical_updated_at = _parse_updated_at(canonical_fields.get("updated_at"))
    replica_updated_at = _parse_updated_at(overlay.get("updated_at"))
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
    state or its accounting. A same-terminal patch remains valid so the
    proactive namer's explicit ``refresh`` can update the final cost snapshot.
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
        if patch_post_task:
            current["post_task_synthesis"] = patch_post_task
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


def set_root_post_task_checkpoint(
    env: Any,
    task: Dict[str, Any],
    status: str,
    *,
    stop_reason: str = "",
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
    # The proactive namer can settle concurrently with post-task synthesis. A
    # shared critical section makes its refresh and the final snapshot linear.
    with POST_TASK_SYNTHESIS_LOCK:
        existing = load_task_result(authority_root, task_id) or {}
        checkpoint = existing.get("root_phase_checkpoint")
        saved = str(checkpoint.get("post_task_synthesis") or "") if isinstance(checkpoint, dict) else ""
        effective_status = saved if requested_status == "refresh" and saved else requested_status
        cost_fields: Dict[str, Any] = {"cost_final": False, "cost_with_children_partial": True}
        if post_task_synthesis_is_terminal(effective_status):
            try:
                from ouroboros.usage_accounting import usage_breakdown
                from supervisor.state import reconstruct_task_cost

                cost_fields.update(reconstruct_task_cost(task_id, fields=True, drive_root=authority_root))
                metadata = (
                    task.get("metadata")
                    if isinstance(task.get("metadata"), dict)
                    else {}
                )
                logical_root_id = str(
                    task.get("root_task_id")
                    or metadata.get("root_task_id")
                    or task_id
                )
                subtree = usage_breakdown(
                    authority_root, root_task_id=logical_root_id
                )
                subtree_final = bool(subtree.get("cost_final"))
                subtree_amount = honest_accounted_amount(subtree)
                cost_fields.update({
                    "accounted_upper_bound_usd_with_children": (
                        round(subtree_amount, 6) if subtree_amount is not None else None
                    ),
                    "cost_with_children_partial": not subtree_final,
                    "cost_final": bool(cost_fields.get("cost_final") and subtree_final),
                })
            except Exception:
                log.error("Failed to refresh final root cost projection for %s", task_id, exc_info=True)
                cost_fields.update({
                    "cost_accounting_status": "unavailable",
                    "cost_accounting_error": "ledger_unavailable",
                    "accounted_upper_bound_usd": None,
                    "accounted_upper_bound_usd_with_children": None,
                })
        # SSOT cost naming (C2/F12/ABI-3): every branch above writes the honest
        # names directly onto the honest-named `reconstruct_task_cost` fields
        # (Ф3.1 fix-round — producers no longer touch the retired spellings);
        # the seam stays as the LAST step as the idempotent invariant guard —
        # it re-normalizes amounts and would strip any retired key a future
        # mutation leaked, so this producer can never persist a diverged pair.
        cost_fields = with_cost_aliases(cost_fields)
        checkpoint_patch = {"post_task_synthesis": effective_status}
        if stop_reason:
            checkpoint_patch["post_task_stop_reason"] = str(stop_reason)
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
    pending_projection = (
        stored.get("canonical_terminal_projection_ready")
        if isinstance(stored, dict) else None
    )
    if (
        isinstance(pending_projection, dict)
        and post_task_synthesis_is_terminal(stored_post_task)
        and not isinstance((stored or {}).get("canonical_terminal_projection"), dict)
    ):
        try:
            from ouroboros.project_dialogue import append_terminal_task_projection

            projection_task = {**task, **(stored or {}), "id": task_id}
            append_terminal_task_projection(
                authority_root,
                task_id,
                projection_task,
                stored or {},
                {
                    "ts": str(pending_projection.get("task_done_ts") or utc_now_iso()),
                    "chat_id": int(pending_projection.get("chat_id") or 0),
                    "status": str((stored or {}).get("status") or STATUS_COMPLETED),
                },
            )
        except Exception:
            log.warning(
                "Failed to settle canonical terminal projection for %s",
                task_id,
                exc_info=True,
            )
    return stored


def root_post_task_already_completed(env: Any, task: Dict[str, Any]) -> bool:
    if not is_root_post_task(task):
        return False
    task_id = str(task.get("id") or task.get("task_id") or "")
    roots = root_checkpoint_roots(env, task)
    existing = load_task_result(roots[0], task_id) if roots and task_id else None
    checkpoint = existing.get("root_phase_checkpoint") if isinstance(existing, dict) else None
    return bool(
        isinstance(checkpoint, dict)
        and post_task_synthesis_is_terminal(checkpoint.get("post_task_synthesis"))
    )
