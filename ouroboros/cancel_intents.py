"""Durable cancel-intent projection — the ONE ingress for task cancellation.

The Poltergeist incident class (2026-08-11): cancel intent used to travel as a
``cancel_requested`` value of the canonical task ``status``, which made one field
carry both INTENT and OUTCOME. A lost control event left four children latched
forever, the post-kill re-check read the latch back as a terminal result, and a
nonterminal ``task_done`` wedged the parent. This module separates the two:

- ``state/cancel_intents.json`` (per canonical data root) is a COMPACT, locked
  projection of ACTIVE intents only — ``requested`` → ``claimed`` → settled rows
  LEAVE the projection — so consulting it on queue restore, worker assignment,
  and effective-status reads is one small locked read, never a scan of a growing
  ledger.
- Every state change also appends a typed ``cancel_intent`` row to
  ``logs/supervisor.jsonl`` as a forensic trail (reusing the existing supervisor
  ledger; the trail is never read back for state).

Ownership: every cancel ingress (agent ``cancel_task`` tool, HTTP single, HTTP
cascade, boot migration of legacy latch files) writes an intent through
``request_cancel``. The supervisor's ``cancel_task_custody`` is the ONE settle
owner: it CLAIMS the intent before teardown and SETTLES it with the terminal
outcome; the supervisor-tick watchdog only re-feeds unclaimed/stale intents back
into custody. The canonical ``status`` never carries intent again.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import pathlib
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ouroboros.utils import append_jsonl, update_json_locked, utc_now_iso

log = logging.getLogger(__name__)


class CancelIntentProjectionCorrupt(RuntimeError):
    """The intent projection file exists but is malformed/non-object JSON.

    Raised (GR3-9) instead of silently collapsing the projection to ``{}`` and
    overwriting it — which would lose EVERY active intent in one write. The
    append-only forensic trail in ``logs/supervisor.jsonl`` keeps the evidence;
    the caller fails closed (the cancel ingress reports the intent write as
    failed rather than pretending the intent is durable).
    """


class CancelIntentLineageIndeterminate(RuntimeError):
    """A historical id points through an invalid timeout-retry chain."""


_SCHEMA_VERSION = 1
# A claim older than this is presumed abandoned (custody crashed mid-teardown)
# and the watchdog may re-feed the intent into custody.
CLAIM_STALE_SEC = 180.0

INTENT_REQUESTED = "requested"
INTENT_CLAIMED = "claimed"

# Cancel SCOPE: a cascade intent must be re-fed as a cascade, not as a single
# cancel — a watchdog replay of the root alone would leave descendants live.
SCOPE_SINGLE = "single"
SCOPE_CASCADE = "cascade"

# Terminalization POLICY (S3, Q1 2026-08-15) — an INDEPENDENT axis from scope
# (§13.1): absence/empty means today's immediate hard cancellation, byte-
# identical for every existing caller; the explicit graceful value buys the
# owner-stop finalization episode (one bounded tool-less model turn inside the
# shared finalization grace) before custody kills. MONOTONIC: an immediate
# request over a graceful intent HARDENS it in place (same durable stop
# request, single kill-owner); graceful can never soften an accepted immediate.
STOP_POLICY_IMMEDIATE = "immediate"
STOP_POLICY_FINALIZE = "finalize_then_cancel"

# Integrity bound only: the queue currently permits far fewer physical retry
# hops, but a corrupted on-disk chain must never make one cancel ingress scan
# without limit.
_RETRY_LINEAGE_MAX_HOPS = 32


def stop_policy(intent: Any) -> str:
    """The intent's terminalization policy; absent/unknown reads IMMEDIATE."""
    value = str((intent or {}).get("stop_policy") or "") if isinstance(intent, dict) else ""
    return STOP_POLICY_FINALIZE if value == STOP_POLICY_FINALIZE else STOP_POLICY_IMMEDIATE

# Settle outcomes (forensic vocabulary; the projection row is removed on settle).
SETTLED_CANCELLED = "cancelled"
SETTLED_ALREADY_SETTLED = "already_settled"
SETTLED_NOT_FOUND = "not_found"


def _intents_path(drive_root: Any) -> pathlib.Path:
    root = pathlib.Path(drive_root) / "state"
    root.mkdir(parents=True, exist_ok=True)
    return root / "cancel_intents.json"


@contextmanager
def cancellation_projection_lock(drive_root: Any) -> Iterator[None]:
    """Hold the same short lock used by cancel-intent mutators.

    Cross-store decisions such as a paid acceptance claim use this only while
    committing their own atomic row.  If cancellation linearizes first they see
    it; if this lock wins first, the claim linearizes before cancellation.
    """
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )

    path = _intents_path(drive_root)
    lock_path = path.with_name(path.name + ".lock")
    lock_fd = acquire_exclusive_file_lock(lock_path)
    if lock_fd is None:
        raise TimeoutError("cancel-intent projection lock unavailable")
    try:
        yield
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)


def _forensic(drive_root: Any, row: Dict[str, Any]) -> None:
    """Append one typed forensic row; the trail is evidence, never state."""
    try:
        append_jsonl(
            pathlib.Path(drive_root) / "logs" / "supervisor.jsonl",
            {"ts": utc_now_iso(), "type": "cancel_intent", **row},
        )
    except Exception:
        log.debug("cancel-intent forensic append failed", exc_info=True)


def _valid_task_id(task_id: Any) -> str:
    from ouroboros.task_results import validate_task_id

    return validate_task_id(task_id)


def _refuse_corrupt(
    drive_root: Any, task_id: str, op: str, exc: Exception,
) -> CancelIntentProjectionCorrupt:
    """Disclose a corrupt projection and build the typed refusal for ``op``.

    Every mutation of the projection AUTHORS a durable record, so all of them
    fail closed on a file they could not read — the mint refuses to record an
    intent, and the four lifecycle mutators refuse to claim, release, settle or
    re-scope one. Reading the file for BEHAVIOUR is the separate, deliberately
    fail-soft path (``active_intents``): the split is what keeps one unreadable
    file from wedging the supervisor tick while still never letting corruption
    masquerade as "no cancel was requested".
    """
    _forensic(drive_root, {
        "event": "projection_corrupt_refused", "task_id": task_id,
        "op": op, "error": str(exc)[:200],
    })
    log.error(
        "cancel-intent projection is corrupt; refusing %s for %s", op, task_id,
    )
    return CancelIntentProjectionCorrupt(str(exc))


def _load_intents(data: Dict[str, Any], *, strict: bool = False) -> Dict[str, Any]:
    """The active-intent rows; ``strict`` refuses a malformed nested value.

    GR5-6 (v7 delta D08 re-derived on the hardened custody bytes): every
    MUTATOR passes ``strict=True`` — a present-but-non-dict ``intents`` under
    a valid top-level dict used to be coerced to ``{}``, so the next mint
    rewrote the file and silently dropped every other active intent (the
    exact loss the top-level ``strict_existing_dict`` check refuses), and the
    non-minting mutators read that same ``{}`` as "nobody requested a cancel"
    — an answer indistinguishable from a lost claim-first fence. The raise is
    the typed ``ValueError`` each mutator turns into
    ``CancelIntentProjectionCorrupt``. Read paths disclose separately and
    stay fail-soft (``active_intents``)."""
    intents = data.get("intents")
    if isinstance(intents, dict):
        return intents
    if strict and intents is not None:
        raise ValueError(
            "cancel-intent projection 'intents' is malformed (not an object)"
        )
    return {}


def settled_status(drive_root: Any, task_id: str) -> str:
    """The task's own already-settled durable status, or "" — fail-soft."""
    try:
        from ouroboros.task_results import load_task_result
        from ouroboros.task_status import SETTLED_STATUSES

        status = str((load_task_result(drive_root, task_id) or {}).get("status") or "")
        return status if status in SETTLED_STATUSES else ""
    except Exception:
        log.debug("cancel-intent settled-status read failed for %s", task_id, exc_info=True)
        return ""


def _validated_single_cancel_target(drive_root: Any, task_id: str) -> str:
    """Resolve a historical root id to its durable physical retry leaf.

    Only reciprocal host-written result lineage is authority.  Same-id retries
    (subagent/evolution) stay exact.  This helper is called while the cancel
    projection lock is held, which makes retry publication and intent minting
    one ordered decision with the reaper's admission boundary.
    """
    from ouroboros.task_results import load_task_result, resolve_task_lineage

    current_id = _valid_task_id(task_id)
    seen = {current_id}
    logical_root = current_id
    for _hop in range(_RETRY_LINEAGE_MAX_HOPS + 1):
        try:
            current = load_task_result(drive_root, current_id, strict=True) or {}
        except Exception as exc:
            raise CancelIntentLineageIndeterminate(
                f"task-result authority is unreadable for {current_id}"
            ) from exc
        if _hop == 0:
            logical_root = str(current.get("root_task_id") or current_id)
        superseded_by = str(current.get("superseded_by") or "").strip()
        retry_task_id = str(current.get("retry_task_id") or "").strip()
        if not superseded_by and retry_task_id == current_id:
            return current_id
        if not superseded_by and not retry_task_id:
            return current_id
        if not superseded_by or superseded_by != retry_task_id:
            raise CancelIntentLineageIndeterminate(
                f"timeout retry lineage from {current_id} is not reciprocal"
            )
        successor_id = _valid_task_id(superseded_by)
        if successor_id in seen:
            raise CancelIntentLineageIndeterminate(
                f"timeout retry lineage from {task_id} contains a cycle"
            )
        if _hop >= _RETRY_LINEAGE_MAX_HOPS:
            raise CancelIntentLineageIndeterminate(
                f"timeout retry lineage from {task_id} exceeds the integrity bound"
            )
        try:
            successor = load_task_result(drive_root, successor_id, strict=True)
        except Exception as exc:
            raise CancelIntentLineageIndeterminate(
                f"task-result authority is unreadable for {successor_id}"
            ) from exc
        if not isinstance(successor, dict):
            raise CancelIntentLineageIndeterminate(
                f"timeout retry successor {successor_id} has no durable result"
            )
        lineage = resolve_task_lineage(
            successor_id,
            metadata=successor.get("metadata"),
            root_task_id=successor.get("root_task_id"),
            parent_task_id=successor.get("parent_task_id"),
            delegation_role=successor.get("delegation_role"),
            original_task_id=successor.get("original_task_id"),
            timeout_retry_from=successor.get("timeout_retry_from"),
        )
        if (
            str(successor.get("supersedes_task_id") or "") != current_id
            or str(successor.get("original_task_id") or "") != current_id
            or str(successor.get("timeout_retry_from") or "") != current_id
            or not lineage.get("is_retry_root_attempt")
            or str(lineage.get("root_task_id") or "") != logical_root
        ):
            raise CancelIntentLineageIndeterminate(
                f"timeout retry lineage {current_id} -> {successor_id} is incomplete"
            )
        seen.add(successor_id)
        current_id = successor_id
    raise CancelIntentLineageIndeterminate(
        f"timeout retry lineage from {task_id} exceeds the integrity bound"
    )


def _validated_retry_root_cancel_key(
    drive_root: Any, task_id: str, *, task_hint: Any = None,
) -> str:
    """Return the logical root key for a proven physical root retry leaf.

    Ordinary tasks and same-id retries return ``""``.  A new-id retry is an
    alias of its logical root only when its durable row has the typed root
    attempt shape *and* the reciprocal forward chain from that root ends at
    this exact physical id.  Retry admission uses this while checking cancel
    authority so a cascade/finalize intent that deliberately remains keyed by
    the logical root cannot be escaped by a second timeout retry.
    """
    from ouroboros.task_results import load_task_result, resolve_task_lineage

    tid = _valid_task_id(task_id)
    try:
        current = load_task_result(drive_root, tid, strict=True)
    except Exception as exc:
        raise CancelIntentLineageIndeterminate(
            f"task-result authority is unreadable for {tid}"
        ) from exc
    if not isinstance(current, dict):
        # A first physical attempt (or a same-id descendant retry) can reach
        # timeout during the narrow spawn->first-result window.  It has no
        # logical alias to escape, so the live host-owned task row is enough to
        # classify it as exact.  A new-id root retry is different: its missing
        # durable reciprocal row destroys the only proof that a logical-root
        # intent covers this physical id, so admission must fail closed.
        hint = task_hint if isinstance(task_hint, dict) else {}
        hinted = resolve_task_lineage(
            tid,
            metadata=hint.get("metadata"),
            root_task_id=hint.get("root_task_id"),
            parent_task_id=hint.get("parent_task_id"),
            delegation_role=hint.get("delegation_role"),
            original_task_id=hint.get("original_task_id"),
            timeout_retry_from=hint.get("timeout_retry_from"),
        )
        if hinted.get("is_retry_root_attempt"):
            raise CancelIntentLineageIndeterminate(
                f"task-result authority has no durable row for retry leaf {tid}"
            )
        return ""
    lineage = resolve_task_lineage(
        tid,
        metadata=current.get("metadata"),
        root_task_id=current.get("root_task_id"),
        parent_task_id=current.get("parent_task_id"),
        delegation_role=current.get("delegation_role"),
        original_task_id=current.get("original_task_id"),
        timeout_retry_from=current.get("timeout_retry_from"),
    )
    if not lineage.get("is_retry_root_attempt"):
        return ""
    logical_root = _valid_task_id(str(lineage.get("root_task_id") or ""))
    if _validated_single_cancel_target(drive_root, logical_root) != tid:
        raise CancelIntentLineageIndeterminate(
            f"logical retry root {logical_root} does not resolve to {tid}"
        )
    return logical_root


def request_cancel(
    drive_root: Any,
    task_id: str,
    *,
    reason: str = "",
    source: str = "",
    requested_by: str = "",
    scope: str = "",
    allow_settled_target: bool = False,
    requested_stop_policy: str = "",
    observation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record durable cancel intent for ``task_id`` — idempotent per task.

    Returns the ACTIVE intent row (existing or newly minted) plus
    ``already_requested``. Never touches the canonical task status: teardown and
    the terminal write belong to the supervisor's cancellation custody.

    An ALREADY-SETTLED task with NO live ownership mints nothing: an intent for
    a task that finished on its own would show a false "Cancelling…" badge on a
    settled card until the watchdog cleaned it up, and nothing is left to tear
    down. The caller gets ``already_settled`` plus the real ``status`` instead
    (completion wins).

    ``allow_settled_target`` is the LIVE-OWNERSHIP exception (GR6-1, widening
    the GR2-1b cascade case): the pipeline persists the durable terminal result
    BEFORE post-task cognition ends, so a settled STATUS alone does not prove a
    dead WORKER — ``already_settled`` is a terminal answer only when no live
    physical ownership remains (no RUNNING row / busy worker). This module
    stays pure (it never reads the queue): each INGRESS checks its own live
    ownership fact (`supervisor.queue.task_has_live_ownership` in-process, the
    queue-snapshot read worker-side) and passes ``allow_settled_target=True``
    when ownership is live, so custody can kill the still-spending worker while
    completion-wins preserves the stored result. The cascade-coordination
    ingress (GR2-1b) passes it for a settled root with live descendants — the
    intent is the watchdog's replay trigger and settles only at the cascade
    postcondition. The settled-card badge hazard does not apply: the
    effective-status read only projects ``cancel_state`` onto NON-settled
    results.

    ``scope`` is stored on the row so a watchdog replay re-runs the SAME shape:
    a ``cascade`` intent re-fed as a single cancel would settle the root while
    its descendants kept running.
    """
    tid = _valid_task_id(task_id)
    reason_text = " ".join(str(reason or "").split())[:500]
    policy_text = str(requested_stop_policy or "").strip()
    minted: Dict[str, Any] = {}
    scope_text = str(scope or "").strip()
    resolved = {"task_id": tid, "status": ""}
    # Whether THIS mutation recorded a new hardening — a duplicate immediate
    # request over an already-hardened intent must not re-emit the forensic row.
    newly_hardened = {"value": False}
    observed = copy.deepcopy(observation) if isinstance(observation, dict) else None

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        newly_hardened["value"] = False
        intents = _load_intents(current, strict=True)
        # A pre-existing logical-root intent remains its own authority.  New
        # SINGLE requests canonicalize to the physical retry leaf; cascades
        # deliberately stay on the logical root because their subtree owner
        # uses that id for fences, descendant capture, and the one summary.
        requested_existing = intents.get(tid)
        if requested_existing is not None and not isinstance(requested_existing, dict):
            raise ValueError(
                f"cancel-intent row for {tid} is malformed (not an object)"
            )
        target_id = tid
        rekeyed = False
        # A physical timeout-retry leaf may already be governed by a cascade
        # intent that deliberately remains keyed by the stable logical root.
        # Reuse that proven authority instead of minting a second SINGLE intent
        # on the leaf (Stop-now must harden the same owner request).  Do not
        # infer a new logical cascade: only an already-present, validated
        # cascade row is eligible for this reverse alias.
        if not (
            isinstance(requested_existing, dict)
            and requested_existing.get("request_id")
        ):
            retry_root_id = _validated_retry_root_cancel_key(
                drive_root, tid,
            )
            retry_root_existing = (
                intents.get(retry_root_id) if retry_root_id else None
            )
            if retry_root_existing is not None and not isinstance(
                retry_root_existing, dict,
            ):
                raise ValueError(
                    f"cancel-intent row for {retry_root_id} is malformed "
                    "(not an object)"
                )
            if (
                isinstance(retry_root_existing, dict)
                and retry_root_existing.get("request_id")
                and str(retry_root_existing.get("scope") or "")
                == SCOPE_CASCADE
            ):
                target_id = retry_root_id
                requested_existing = retry_root_existing
        if (
            not (
                isinstance(requested_existing, dict)
                and requested_existing.get("request_id")
            )
            and scope_text == SCOPE_CASCADE
        ):
            leaf_id = _validated_single_cancel_target(drive_root, tid)
            leaf_existing = intents.get(leaf_id) if leaf_id != tid else None
            if leaf_existing is not None and not isinstance(leaf_existing, dict):
                raise ValueError(
                    f"cancel-intent row for {leaf_id} is malformed (not an object)"
                )
            if isinstance(leaf_existing, dict) and leaf_existing.get("request_id"):
                if leaf_existing.get("state") == INTENT_CLAIMED:
                    raise CancelIntentLineageIndeterminate(
                        f"cannot re-key a claimed retry intent from {leaf_id} to {tid}"
                    )
                requested_existing = {
                    **leaf_existing,
                    "task_id": tid,
                    "scope": SCOPE_CASCADE,
                }
                intents.pop(leaf_id, None)
                intents[tid] = requested_existing
                rekeyed = True
        elif not (
            isinstance(requested_existing, dict)
            and requested_existing.get("request_id")
        ):
            target_id = _validated_single_cancel_target(drive_root, tid)
        resolved["task_id"] = target_id
        if observed is not None:
            observed["matches_cancel_target"] = observed.get("observed_task_id") == target_id
        existing = intents.get(target_id)
        if existing is not None and not isinstance(existing, dict):
            # GR6-3 row strictness: a present-but-malformed ROW is corruption,
            # not an absent intent — silently overwriting it would destroy the
            # forensic bytes exactly like the {}-collapse the container check
            # refuses. Same typed ValueError → CancelIntentProjectionCorrupt.
            raise ValueError(
                f"cancel-intent row for {target_id} is malformed (not an object)"
            )
        from ouroboros.task_results import load_task_result
        from ouroboros.task_status import SETTLED_STATUSES

        try:
            target_result = load_task_result(drive_root, target_id, strict=True) or {}
        except Exception as exc:
            raise CancelIntentLineageIndeterminate(
                f"task-result authority is unreadable for {target_id}"
            ) from exc
        target_status = str(target_result.get("status") or "")
        resolved["status"] = target_status if target_status in SETTLED_STATUSES else ""
        if (
            resolved["status"]
            and not allow_settled_target
            and not (
                isinstance(existing, dict)
                and existing.get("request_id")
            )
        ):
            # An existing intent can still own unfinished coordination (most
            # importantly a settled logical root with live descendants).  The
            # already-settled fast path applies only when there is no durable
            # owner request left to harden or settle.
            minted.update({
                "task_id": target_id,
                "already_requested": False,
                "already_settled": True,
                "status": resolved["status"],
                **({"observation": observed} if observed is not None else {}),
            })
            return None
        if isinstance(existing, dict) and existing.get("request_id"):
            minted.update(existing)
            minted["already_requested"] = True
            updated_row = dict(existing)
            changed = rekeyed
            if rekeyed and isinstance(updated_row.get("observation"), dict):
                updated_row["observation"] = {**updated_row["observation"],
                    "matches_cancel_target": updated_row["observation"].get("observed_task_id") == target_id}
            if scope_text == SCOPE_CASCADE and str(existing.get("scope") or "") != SCOPE_CASCADE:
                # A single-cancel intent later re-entered through the cascade
                # ingress must be replayed as a cascade. WIDEN-ONLY (GR2-1d):
                # cascade → single is never written back — narrowing a recorded
                # cascade would let a watchdog replay settle the root and leave
                # its descendants running.
                updated_row["scope"] = scope_text
                changed = True
            if (
                policy_text == STOP_POLICY_IMMEDIATE
                and stop_policy(existing) == STOP_POLICY_FINALIZE
            ):
                # MONOTONIC hardening (§12.2 item 10): Stop-now during the
                # graceful wait tightens the SAME durable stop request — single
                # kill-owner, no second intent. Graceful over immediate is the
                # forbidden softening direction and falls through unchanged.
                updated_row["stop_policy"] = STOP_POLICY_IMMEDIATE
                updated_row["hardened_at"] = utc_now_iso()
                newly_hardened["value"] = True
                changed = True
            if changed:
                intents[target_id] = updated_row
                minted.update(updated_row)
                return {"schema_version": _SCHEMA_VERSION, "intents": intents}
            return None
        row = {
            "request_id": f"ci_{uuid.uuid4().hex[:12]}",
            "task_id": target_id,
            "state": INTENT_REQUESTED,
            "reason": reason_text,
            "source": str(source or ""),
            "requested_by": str(requested_by or ""),
            "requested_at": utc_now_iso(),
            "generation": 0,
            "scope": scope_text or SCOPE_SINGLE,
            **({"observation": observed} if observed is not None else {}),
        }
        if policy_text == STOP_POLICY_FINALIZE:
            row["stop_policy"] = STOP_POLICY_FINALIZE
        intents[target_id] = row
        minted.update(row)
        minted["already_requested"] = False
        return {"schema_version": _SCHEMA_VERSION, "intents": intents}

    try:
        # GR3-9 strict read: a malformed projection file must REFUSE the mutation
        # loudly instead of collapsing to {} and being overwritten — that write
        # would silently drop every other active intent. Absent file (first
        # write) is unaffected.
        update_json_locked(_intents_path(drive_root), _mutate, strict_existing_dict=True)
    except CancelIntentLineageIndeterminate as exc:
        _forensic(drive_root, {
            "event": "retry_lineage_refused", "task_id": tid,
            "op": "request_cancel", "error": str(exc)[:200],
        })
        log.error(
            "cancel-intent retry lineage is indeterminate for %s; refusing intent",
            tid,
        )
        raise
    except ValueError as exc:
        _forensic(drive_root, {
            "event": "projection_corrupt_refused", "task_id": tid,
            "op": "request_cancel", "error": str(exc)[:200],
        })
        log.error(
            "cancel-intent projection is corrupt; refusing to record intent for %s", tid,
        )
        raise CancelIntentProjectionCorrupt(str(exc)) from exc
    if minted.get("already_settled"):
        _forensic(drive_root, {
            "event": "already_settled",
            "task_id": str(resolved.get("task_id") or tid),
            "requested_task_id": tid,
            "status": str(minted.get("status") or ""),
            "source": str(source or ""),
            "reason": reason_text,
            **({"observation": observed} if observed is not None else {}),
        })
    elif not minted.get("already_requested"):
        _forensic(drive_root, {
            "event": "requested", "task_id": str(resolved.get("task_id") or tid),
            **(
                {"requested_task_id": tid}
                if str(resolved.get("task_id") or tid) != tid else {}
            ),
            "request_id": minted.get("request_id"),
            "source": minted.get("source"), "requested_by": minted.get("requested_by"),
            "scope": minted.get("scope"), "reason": reason_text,
            **({"observation": minted["observation"]} if "observation" in minted else {}),
            **({"stop_policy": minted.get("stop_policy")} if minted.get("stop_policy") else {}),
            **(
                {"settled_target_status": resolved.get("status")}
                if resolved.get("status") else {}
            ),
        })
    elif newly_hardened["value"]:
        _forensic(drive_root, {
            "event": "stop_policy_hardened",
            "task_id": str(resolved.get("task_id") or tid),
            **(
                {"requested_task_id": tid}
                if str(resolved.get("task_id") or tid) != tid else {}
            ),
            "request_id": minted.get("request_id"), "stop_policy": STOP_POLICY_IMMEDIATE,
        })
    minted.setdefault("already_settled", False)
    result = dict(minted)
    if str(resolved.get("task_id") or tid) != tid:
        result["requested_task_id"] = tid
    return result


def mark_finalize_control_drained(
    drive_root: Any, task_id: str, *, drained_at: str = "",
) -> bool:
    """Record when the loop actually DELIVERED the finalize control to the model.

    S3 owner decision (2026-08-15, 1=A): the finalization-episode budget starts
    at DELIVERY (the round-boundary mailbox drain), not at the stop request —
    a task inside a long blocking tool call still gets its bounded final turn.
    The worker calls this from the production drain; the custody sweep reads
    ``control_drained_at`` back to compute the effective episode deadline
    (``supervisor/owner_stop.py``). FIRST DRAIN WINS: a restart re-drain (the
    control is replayable until terminal cleanup) never moves the stamp, so a
    worker crash cannot resurrect an unlimited episode. No-op for absent
    intents and for non-finalize policies. Fail-soft for ordinary write
    failures (a projection error never breaks the round loop) but fail-CLOSED
    for a corrupt projection: ``CancelIntentProjectionCorrupt`` is raised
    rather than answering "no intent" over a file nobody could read. Returns
    whether THIS call recorded the stamp.
    """
    try:
        requested_tid = _valid_task_id(task_id)
    except ValueError:
        return False
    tid, _resolved_intent = resolve_owner_stop_intent(drive_root, requested_tid)
    stamp = str(drained_at or "") or utc_now_iso()
    recorded: Dict[str, Any] = {}

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        recorded.clear()
        intents = _load_intents(current, strict=True)
        row = intents.get(tid)
        if not isinstance(row, dict) or stop_policy(row) != STOP_POLICY_FINALIZE:
            return None
        if str(row.get("control_drained_at") or ""):
            return None  # first drain wins: the stamp is immutable
        intents[tid] = {**row, "control_drained_at": stamp}
        recorded.update(intents[tid])
        return {"schema_version": _SCHEMA_VERSION, "intents": intents}

    try:
        update_json_locked(_intents_path(drive_root), _mutate, strict_existing_dict=True)
    except ValueError as exc:
        raise _refuse_corrupt(
            drive_root, tid, "mark_finalize_control_drained", exc,
        ) from exc
    except Exception:
        log.debug(
            "finalize-control drain stamp failed for %s", task_id, exc_info=True,
        )
        return False
    if recorded:
        _forensic(drive_root, {
            "event": "finalize_control_drained", "task_id": tid,
            "request_id": recorded.get("request_id"),
            "control_drained_at": stamp,
        })
    return bool(recorded)


def mark_intent_scope(drive_root: Any, task_id: str, scope: str) -> bool:
    """Widen an EXISTING intent's scope; never mints one. Returns whether it changed.

    The ingress owns minting (owner batch-4 1=A); the cascade only records the
    SHAPE it is running, so a watchdog replay re-runs a cascade as a cascade
    instead of a single cancel that would settle the root and leave descendants
    running.

    WIDEN-ONLY (GR2-1d): ``single`` → ``cascade`` is the one legal transition.
    ``cascade`` → ``single`` is refused as a no-op plus a forensic row — a
    narrowed record would make the watchdog replay the root alone while its
    descendants kept running, exactly the shape the scope exists to prevent.

    A CORRUPT projection raises ``CancelIntentProjectionCorrupt``: the caller
    (the cascade's scope stamp) already treats a failure here as loud, and
    "no intent to widen" is not an answer this file can honestly give.
    """
    try:
        tid = _valid_task_id(task_id)
    except ValueError:
        return False
    scope_text = str(scope or "").strip()
    if not scope_text:
        return False
    changed = {"value": False}
    narrowed: Dict[str, Any] = {}

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        narrowed.clear()
        intents = _load_intents(current, strict=True)
        row = intents.get(tid)
        if not isinstance(row, dict) or str(row.get("scope") or "") == scope_text:
            return None
        if str(row.get("scope") or "") == SCOPE_CASCADE:
            narrowed.update(row)
            return None
        intents[tid] = {**row, "scope": scope_text}
        changed["value"] = True
        return {"schema_version": _SCHEMA_VERSION, "intents": intents}

    try:
        update_json_locked(_intents_path(drive_root), _mutate, strict_existing_dict=True)
    except ValueError as exc:
        raise _refuse_corrupt(drive_root, tid, "mark_intent_scope", exc) from exc
    except Exception:
        log.debug("cancel-intent scope update failed for %s", task_id, exc_info=True)
        return False
    if narrowed:
        _forensic(drive_root, {
            "event": "scope_narrow_refused", "task_id": tid,
            "request_id": narrowed.get("request_id"),
            "scope": SCOPE_CASCADE, "requested_scope": scope_text,
        })
        return False
    if changed["value"]:
        _forensic(drive_root, {"event": "scope_recorded", "task_id": tid, "scope": scope_text})
    return changed["value"]


def active_intents(
    drive_root: Any, *, disclose_corruption: bool = False, strict: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """All active intents keyed by task id (a private copy; read-only callers).

    GR5-6 read semantics: an ABSENT projection is an ordinary empty read; an
    UNREADABLE/MALFORMED one (file or nested ``intents``) is a real gap — the
    enforcement readers would otherwise collapse corruption to "no intent".
    It is disclosed with a typed ``log.error`` (and, for the supervisor
    watchdog's enforcement read via ``disclose_corruption=True``, the existing
    typed ``projection_corrupt_refused`` forensic row so the owner can see
    enforcement is degraded) before the read still returns ``{}`` — fail-soft
    read, fail-closed write.
    """
    path = _intents_path_read(drive_root)
    if not path.is_file():
        return {}
    try:
        from ouroboros.utils import read_json_dict

        data = read_json_dict(path)
        if data is None:
            raise ValueError("projection file is malformed or is not an object")
        intents = _load_intents(data, strict=True)
    except Exception as exc:
        log.error(
            "cancel-intent projection is unreadable/malformed (%s); enforcement "
            "reads see NO active intents until the file is repaired", exc,
        )
        if disclose_corruption:
            _forensic(drive_root, {
                "event": "projection_corrupt_refused",
                "op": "active_intents", "error": str(exc)[:200],
            })
        if strict:
            raise CancelIntentProjectionCorrupt(str(exc)) from exc
        return {}
    malformed = [str(tid) for tid, row in intents.items() if not isinstance(row, dict)]
    if malformed:
        # GR6-3 row strictness on the enforcement read: a malformed per-task
        # row used to be silently filtered — the watchdog then saw "no intent"
        # over bytes that still claim one. Disclose loudly once per read and
        # QUARANTINE the row (skipped here, never dropped from the durable
        # file — only the strict mutators rewrite it, and they refuse).
        log.error(
            "cancel-intent projection holds %d malformed row(s) (%s); "
            "quarantined — skipped by reads, bytes kept on disk",
            len(malformed), ", ".join(malformed[:5]),
        )
        # GR7-5: the log.error above stays per sweep, but the typed EVENT is
        # emitted once per distinct row content (in-process memo keyed by
        # row hash) — the watchdog re-reads every ~20s, so a lingering
        # quarantined row used to append the same forensic row forever. A
        # restart re-announcing once is honest.
        fresh = [
            tid for tid in malformed
            if _malformed_row_memo_key(drive_root, tid, intents.get(tid))
            not in _DISCLOSED_MALFORMED_ROWS
        ]
        if disclose_corruption and fresh:
            for tid in fresh:
                if len(_DISCLOSED_MALFORMED_ROWS) < _DISCLOSED_MALFORMED_ROWS_CAP:
                    _DISCLOSED_MALFORMED_ROWS.add(
                        _malformed_row_memo_key(drive_root, tid, intents.get(tid)),
                    )
            _forensic(drive_root, {
                "event": "projection_corrupt_refused",
                "op": "active_intents_row",
                "error": f"malformed intent row(s): {', '.join(fresh[:5])}"[:200],
            })
        if strict:
            raise CancelIntentProjectionCorrupt(
                f"malformed intent row(s): {', '.join(malformed[:5])}"
            )
    return {
        str(tid): dict(row)
        for tid, row in intents.items()
        if isinstance(row, dict)
    }


def _intents_path_read(drive_root: Any) -> pathlib.Path:
    # Read path without the mkdir side effect: a scan of a never-provisioned
    # root must not materialise its state directory.
    return pathlib.Path(drive_root) / "state" / "cancel_intents.json"


# GR7-5: in-process memo of already-disclosed quarantined rows (typed EVENT
# once per row content; the per-sweep log.error stays). Bounded; a restart
# re-announces each row once, which is honest.
_DISCLOSED_MALFORMED_ROWS: set[str] = set()
_DISCLOSED_MALFORMED_ROWS_CAP = 1024


def _malformed_row_memo_key(drive_root: Any, task_id: str, row: Any) -> str:
    # Keyed per data root too: one process can serve several roots (tests,
    # child drives) and a memo hit on one must not mute another's disclosure.
    return hashlib.sha256(
        f"{pathlib.Path(drive_root)}:{task_id}:{row!r}".encode("utf-8", "replace")
    ).hexdigest()[:16]


def active_intent(
    drive_root: Any, task_id: str, *, strict: bool = False,
) -> Optional[Dict[str, Any]]:
    try:
        tid = _valid_task_id(task_id)
    except ValueError:
        if strict:
            raise
        return None
    row = active_intents(drive_root, strict=strict).get(tid)
    return dict(row) if isinstance(row, dict) else None


def resolve_owner_stop_intent(
    drive_root: Any, task_id: str,
) -> Tuple[str, Dict[str, Any]]:
    """Return the durable owner-stop key/row for a physical retry task.

    SINGLE intents are already canonicalized to the physical leaf.  Graceful
    cascades deliberately remain keyed by their logical root, so a worker on a
    new-id retry needs this validated reverse lookup for its drain stamp and
    deadline check.  No alias state is persisted; reciprocal result lineage is
    the authority.
    """
    try:
        tid = _valid_task_id(task_id)
    except ValueError:
        return "", {}
    exact = active_intent(drive_root, tid)
    if isinstance(exact, dict) and stop_policy(exact) == STOP_POLICY_FINALIZE:
        return tid, exact
    try:
        from ouroboros.task_results import load_task_result

        result = load_task_result(drive_root, tid, strict=True) or {}
        logical_root = str(result.get("root_task_id") or "").strip()
        if not logical_root or logical_root == tid:
            return tid, {}
        logical = active_intent(drive_root, logical_root)
        if not (
            isinstance(logical, dict)
            and str(logical.get("scope") or "") == SCOPE_CASCADE
            and stop_policy(logical) == STOP_POLICY_FINALIZE
            and _validated_single_cancel_target(drive_root, logical_root) == tid
        ):
            return tid, {}
        return logical_root, logical
    except Exception:
        log.debug(
            "owner-stop intent resolution failed for physical task %s",
            tid,
            exc_info=True,
        )
        return tid, {}


def has_active_intent(drive_root: Any, task_id: str, *, strict: bool = False) -> bool:
    return active_intent(drive_root, task_id, strict=strict) is not None


def claim_intent(drive_root: Any, task_id: str, *, owner: str) -> Optional[Dict[str, Any]]:
    """Mark the intent claimed by one custody attempt; bumps the generation.

    EXCLUSIVE while the holder is alive: a LIVE claim is never stolen, because a
    second custody attempt that took the claim from a live one would let both
    write the terminal result and both emit ``task_done`` for the same task (the
    concurrent-ingress double-settle a reviewer probe reproduced on a pending
    task). A refused claim comes back as the existing row plus
    ``claim_refused: True`` so the caller can restore custody and let the real
    owner — or the watchdog, once the claim is ABANDONED — finish.

    An ABANDONED claim (its process is gone, or it aged past ``CLAIM_STALE_SEC``)
    is taken over and the generation is bumped, which is exactly what makes the
    old holder's late ``settle``/``release`` a no-op (see ``expected_generation``).
    Returns None when no active intent exists.
    """
    try:
        tid = _valid_task_id(task_id)
    except ValueError:
        return None
    claimed: Dict[str, Any] = {}
    refused: Dict[str, Any] = {}

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        intents = _load_intents(current, strict=True)
        malformed = [
            str(intent_id)
            for intent_id, intent_row in intents.items()
            if not isinstance(intent_row, dict)
        ]
        if malformed:
            raise ValueError(
                f"malformed intent row(s): {', '.join(malformed[:5])}"
            )
        row = intents.get(tid)
        if not isinstance(row, dict):
            return None
        if row.get("state") == INTENT_CLAIMED and not claim_is_abandoned(row):
            refused.update(row)
            return None
        row = dict(row)
        row["state"] = INTENT_CLAIMED
        row["claim_owner"] = str(owner or "")
        row["claim_pid"] = int(os.getpid())
        row["claimed_at"] = utc_now_iso()
        row["generation"] = int(row.get("generation") or 0) + 1
        intents[tid] = row
        claimed.update(row)
        return {"schema_version": _SCHEMA_VERSION, "intents": intents}

    # Claiming is an enforcement write: an existing malformed projection must
    # raise instead of being treated as an empty store and returning ``None``.
    # The latter would look like a harmless claim race to the pending-drop
    # caller and could let a cancelled row cross the dispatch boundary.  An
    # absent file still initializes normally, so first-use behavior is intact.
    try:
        update_json_locked(
            _intents_path(drive_root), _mutate, strict_existing_dict=True,
        )
    except ValueError as exc:
        _forensic(drive_root, {
            "event": "projection_corrupt_refused", "task_id": tid,
            "op": "claim_intent", "error": str(exc)[:200],
        })
        log.error(
            "cancel-intent projection is corrupt; refusing claim for %s", tid,
        )
        raise CancelIntentProjectionCorrupt(str(exc)) from exc
    if claimed:
        _forensic(drive_root, {
            "event": "claimed", "task_id": tid,
            "request_id": claimed.get("request_id"),
            "claim_owner": claimed.get("claim_owner"),
            "generation": claimed.get("generation"),
        })
        return dict(claimed)
    if refused:
        _forensic(drive_root, {
            "event": "claim_refused", "task_id": tid,
            "request_id": refused.get("request_id"),
            "claim_owner": refused.get("claim_owner"),
            "generation": refused.get("generation"), "owner": str(owner or ""),
        })
        return {**refused, "claim_refused": True}
    return None


def _generation_mismatch(
    row: Dict[str, Any], expected_generation: Optional[int], request_id: str,
) -> str:
    """"" when this row is the caller's own claim, else why it is not.

    ``generation`` used to be forensic decoration. It is a FENCE now: a custody
    attempt whose claim was taken over (crash, stale takeover) must not mutate
    the projection afterwards — its late ``release`` would revert a newer claim
    and its late ``settle`` would delete an intent the new owner is still
    working. A mismatch records a forensic row and changes nothing.
    """
    if request_id and str(row.get("request_id") or "") != str(request_id):
        return f"request_id {row.get('request_id')!r} != {request_id!r}"
    if expected_generation is None:
        return ""
    try:
        current = int(row.get("generation") or 0)
    except (TypeError, ValueError):
        current = 0
    return "" if current == int(expected_generation) else (
        f"generation {current} != {int(expected_generation)}"
    )


def release_claim(
    drive_root: Any, task_id: str, *, error: str = "",
    expected_generation: Optional[int] = None, request_id: str = "",
) -> bool:
    """Return a claimed intent to ``requested`` after a failed custody attempt.

    Fenced by ``expected_generation``/``request_id``: a stale claimant's release
    must never revert the claim of the custody attempt that took over from it.
    Returns ``True`` only when this exact claim was durably reopened; ``False``
    means the row was absent, already changed, or did not match the fence.

    A corrupt projection raises ``CancelIntentProjectionCorrupt``; the caller
    logs it and the intent stays CLAIMED for the watchdog, which is the same
    conservative outcome an unwritable projection already produces.
    """
    try:
        tid = _valid_task_id(task_id)
    except ValueError:
        return False
    released: Dict[str, Any] = {}
    mismatch: Dict[str, Any] = {}

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        intents = _load_intents(current, strict=True)
        row = intents.get(tid)
        if not isinstance(row, dict) or row.get("state") != INTENT_CLAIMED:
            return None
        reason = _generation_mismatch(row, expected_generation, request_id)
        if reason:
            mismatch.update({**row, "_reason": reason})
            return None
        row = dict(row)
        row["state"] = INTENT_REQUESTED
        row["last_error"] = str(error or "")[:500]
        row.pop("claim_owner", None)
        row.pop("claim_pid", None)
        row.pop("claimed_at", None)
        intents[tid] = row
        released.update(row)
        return {"schema_version": _SCHEMA_VERSION, "intents": intents}

    try:
        update_json_locked(_intents_path(drive_root), _mutate, strict_existing_dict=True)
    except ValueError as exc:
        raise _refuse_corrupt(drive_root, tid, "release_claim", exc) from exc
    if released:
        _forensic(drive_root, {
            "event": "claim_released", "task_id": tid,
            "request_id": released.get("request_id"), "error": str(error or "")[:500],
        })
    elif mismatch:
        _forensic(drive_root, {
            "event": "claim_release_refused", "task_id": tid,
            "request_id": mismatch.get("request_id"),
            "reason": mismatch.get("_reason"),
            "expected_generation": expected_generation,
        })
    return bool(released)


def settle_intent(
    drive_root: Any, task_id: str, *, outcome: str, detail: str = "",
    expected_generation: Optional[int] = None, request_id: str = "",
    allow_cascade_scope: bool = False,
) -> Optional[Dict[str, Any]]:
    """Remove the active intent with its terminal ``outcome`` (forensic row kept).

    Called only by the supervisor settle paths (custody, pending drop, budget
    drain) — the ONE settle ownership the redesign establishes. Fenced by
    ``expected_generation``/``request_id``: a settle from a claim that was taken
    over is a NO-OP plus a forensic row, never a mutation.

    CASCADE OWNERSHIP (GR3-1): a ``scope=cascade`` intent is settled EXCLUSIVELY
    by the cascade postcondition (``allow_cascade_scope=True``). Every other
    settle site is refused ATOMICALLY inside the locked mutate — the scope is
    re-read from the CURRENT durable row, so a claim snapshot that went stale
    when the intent was widened mid-flight cannot settle the tree's replay
    trigger. When the refused caller holds the matching fenced claim, that
    claim is RELEASED in the same write (state back to ``requested``) so the
    watchdog can re-feed the cascade instead of waiting out a dead claim.

    A corrupt projection raises ``CancelIntentProjectionCorrupt`` instead of
    reporting the no-op every settle caller reads as "nothing left to settle":
    the intent must stay OPEN for the watchdog when nobody could read the file.
    """
    try:
        tid = _valid_task_id(task_id)
    except ValueError:
        return None
    settled: Dict[str, Any] = {}
    mismatch: Dict[str, Any] = {}
    cascade_deferred: Dict[str, Any] = {}

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        settled.clear()
        mismatch.clear()
        cascade_deferred.clear()
        intents = _load_intents(current, strict=True)
        row = intents.get(tid)
        if not isinstance(row, dict):
            return None
        if not allow_cascade_scope and str(row.get("scope") or "") == SCOPE_CASCADE:
            cascade_deferred.update(row)
            fence_ok = not _generation_mismatch(row, expected_generation, request_id)
            # GR4-7: the auto-release requires a GENERATION proof. ``request_id``
            # alone is durable on the intent row (identical for every claimant),
            # so a fence-less ``settle_intent(..., request_id=X)`` from a caller
            # that holds NO claim would otherwise release a DIFFERENT owner's
            # live claim. Only a caller that captured the claimed generation may
            # have its refused claim returned to ``requested``.
            if row.get("state") == INTENT_CLAIMED and fence_ok and (
                expected_generation is not None
            ):
                released = dict(row)
                released["state"] = INTENT_REQUESTED
                released["last_error"] = "cascade settle deferred to postcondition"
                released.pop("claim_owner", None)
                released.pop("claim_pid", None)
                released.pop("claimed_at", None)
                intents[tid] = released
                cascade_deferred["_claim_released"] = True
                return {"schema_version": _SCHEMA_VERSION, "intents": intents}
            return None
        reason = _generation_mismatch(row, expected_generation, request_id)
        if reason:
            mismatch.update({**row, "_reason": reason})
            return None
        intents.pop(tid, None)
        settled.update(row)
        return {"schema_version": _SCHEMA_VERSION, "intents": intents}

    try:
        update_json_locked(_intents_path(drive_root), _mutate, strict_existing_dict=True)
    except ValueError as exc:
        raise _refuse_corrupt(drive_root, tid, "settle_intent", exc) from exc
    if settled:
        _forensic(drive_root, {
            "event": "settled", "task_id": tid,
            "request_id": settled.get("request_id"),
            "outcome": str(outcome or ""), "detail": str(detail or "")[:500],
            "generation": settled.get("generation"),
        })
        return dict(settled)
    if cascade_deferred:
        _forensic(drive_root, {
            "event": "cascade_settle_deferred", "task_id": tid,
            "request_id": cascade_deferred.get("request_id"),
            "outcome": str(outcome or ""), "detail": str(detail or "")[:500],
            "claim_released": bool(cascade_deferred.get("_claim_released")),
        })
        return None
    if mismatch:
        _forensic(drive_root, {
            "event": "settle_refused", "task_id": tid,
            "request_id": mismatch.get("request_id"),
            "outcome": str(outcome or ""), "reason": mismatch.get("_reason"),
            "expected_generation": expected_generation,
        })
    return None


def claim_is_stale(intent: Dict[str, Any], *, now: Optional[float] = None) -> bool:
    """Whether a claimed intent's custody attempt is presumed dead by AGE."""
    if not isinstance(intent, dict) or intent.get("state") != INTENT_CLAIMED:
        return False
    raw = str(intent.get("claimed_at") or "").replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
        claimed_ts = (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
    except (TypeError, ValueError):
        return True  # unreadable claim provenance: let the watchdog retry
    current = now if now is not None else datetime.now(timezone.utc).timestamp()
    return (current - claimed_ts) > CLAIM_STALE_SEC


def claim_is_abandoned(intent: Dict[str, Any], *, now: Optional[float] = None) -> bool:
    """Whether a claimed intent has no live owner left.

    ABANDONED (GR3-2) means the claimant pid is provably DEAD, or the claim is
    age-stale while liveness is UNKNOWN (pid missing/unparseable, or the probe
    raised). A claimant the probe just proved ALIVE is NEVER abandoned — age
    alone must not let a second custody steal a live claim and double-settle;
    the live owner settles or releases, and a genuinely wedged live claimant is
    the fenced write's problem, not the watchdog's. A demonstrably dead
    claimant is taken over immediately (waiting out ``CLAIM_STALE_SEC`` would
    keep a worker slot marked ``reaping`` — skipped by assignment and by the
    crash detector — for three minutes).
    """
    if not isinstance(intent, dict) or intent.get("state") != INTENT_CLAIMED:
        return False
    try:
        pid = int(intent.get("claim_pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    if pid > 0:
        try:
            from ouroboros.platform_layer import pid_is_alive

            alive = bool(pid_is_alive(pid))
        except Exception:
            log.debug("claim owner liveness probe failed", exc_info=True)
        else:
            if not alive:
                return True
            return False  # probed alive: never abandoned, regardless of age
    return claim_is_stale(intent, now=now)


def claim_still_owned(drive_root: Any, task_id: str, claim: Dict[str, Any]) -> bool:
    """Whether OUR fenced claim (pid + request_id + generation) still owns the intent.

    The minimal write-fence (GR3-2): cancellation custody re-checks this
    immediately BEFORE the durable terminal write — after the kill/join window,
    where a stale-takeover could have re-claimed. A lost claim aborts the
    publication (the caller restores custody and returns ``failed``); this is
    deliberately NOT a renewable-lease subsystem. Fail-CLOSED: an unreadable
    projection cannot prove ownership. A claim-less custody (legacy/no-intent
    path) passes trivially — capture under the queue lock is its exclusion.
    """
    if not isinstance(claim, dict) or not claim.get("request_id"):
        return True
    try:
        row = active_intent(drive_root, task_id)
        if not isinstance(row, dict):
            return False
        return (
            row.get("state") == INTENT_CLAIMED
            and str(row.get("request_id") or "") == str(claim.get("request_id") or "")
            and int(row.get("generation") or 0) == int(claim.get("generation") or 0)
            and int(row.get("claim_pid") or 0) == int(os.getpid())
        )
    except Exception:
        log.debug("claim ownership re-check failed for %s", task_id, exc_info=True)
        return False


def migrate_legacy_cancel_latches(drive_root: Any) -> List[str]:
    """Boot migration: legacy ``cancel_requested`` status files → synthetic intents.

    Pre-redesign task results may still sit in the ``cancel_requested`` latch (the
    incident's wedged shape). Each becomes an ordinary active intent so the
    supervisor watchdog drives it through custody to a settled outcome. The file
    itself is left untouched here (legacy read-path; custody writes the terminal).

    ABI-2 CARVE-OUT (owner 4A) — the ONE exception to the Q8=B wholesale
    quarantine, and the reason this function opens with a scan of its own. A
    pre-redesign latch file is by definition UNSTAMPED, so under ABI-2 the first
    ordinary read moves it byte-unchanged into ``task_results/quarantine/``:
    the scan below stopped finding wedged tasks, and custody's own fail-soft
    read then saw no durable result and settled ``not_found`` — the task
    disappeared without ever reaching a terminal. Boot therefore performs, for
    the LATCHED rows and nothing else, the same stamp-on-write that a live
    pre-upgrade task performs on its next lifecycle write (the transition
    ``require_writable_task_result_schema`` already admits as lawful, and a
    wedged task has no worker left to make it): same status, same fields, no
    conversion — not a converter, and deliberately not a general one. The row
    is then an ordinary latch, this scan adopts it, and custody drives it to
    the ``cancelled`` terminal. Every other unstamped row still quarantines on
    the very next read, including the one below. Applying the carve-out is ONE
    typed durable events row per boot (never one per file), the same log-only
    visibility owner decision 6.3=B gives the quarantine itself.
    """
    from ouroboros.task_result_schema import task_result_schema_refusal
    from ouroboros.task_results import (
        STATUS_CANCEL_REQUESTED, list_task_results, task_results_dir,
        write_task_result,
    )
    from ouroboros.utils import read_json_dict

    admitted: List[str] = []
    for path in sorted(task_results_dir(drive_root, create=False).glob("*.json")):
        raw = read_json_dict(path)
        if task_result_schema_refusal(raw) != "unstamped_pre_7_0":
            continue
        if (
            str(raw.get("status") or "") != STATUS_CANCEL_REQUESTED
            or str(raw.get("task_id") or "") != path.stem
        ):
            continue  # not a latch, or an identity this write would rewrite
        try:
            write_task_result(drive_root, path.stem, STATUS_CANCEL_REQUESTED)
        except Exception:
            log.warning("cancel-latch schema admission failed for %s", path.name,
                        exc_info=True)
            continue
        admitted.append(path.stem)
    if admitted:
        try:
            append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "task_result_cancel_latch_admitted",
                "count": len(admitted),
                "task_ids": admitted,
                "reason": "unstamped_pre_7_0",
            })
        except Exception:
            log.warning("failed to record the cancel-latch admission", exc_info=True)

    migrated: List[str] = []
    try:
        latched = list_task_results(
            pathlib.Path(drive_root), statuses=[STATUS_CANCEL_REQUESTED],
        )
    except Exception:
        log.debug("legacy cancel-latch scan failed", exc_info=True)
        return migrated
    for row in latched:
        tid = str(row.get("task_id") or row.get("id") or "")
        if not tid:
            continue
        try:
            intent = request_cancel(
                pathlib.Path(drive_root), tid,
                reason="legacy cancel_requested latch migrated at boot",
                source="boot_migration",
            )
        except Exception:
            log.debug("legacy cancel-latch migration failed for %s", tid, exc_info=True)
            continue
        if not intent.get("already_requested") and not intent.get("already_settled"):
            migrated.append(tid)
    return migrated


def cancel_state_fields(drive_root: Any, task_id: str) -> Dict[str, Any]:
    """Typed public projection: ``{"cancel_state": "pending", ...}`` or ``{}``."""
    intent = active_intent(drive_root, task_id)
    if intent is None:
        return {}
    fields: Dict[str, Any] = {"cancel_state": "pending"}
    if intent.get("reason"):
        fields["cancel_reason"] = str(intent.get("reason") or "")
    if stop_policy(intent) == STOP_POLICY_FINALIZE:
        # The minimal reload-visible stop-policy projection (§12.2 item 2): the
        # card can honestly show "finalizing before stop" instead of the
        # immediate "Cancelling…" while the graceful episode runs.
        fields["stop_policy"] = STOP_POLICY_FINALIZE
    return fields


def cancel_pending(drive_root: Any, task_id: str, *, strict: bool = False) -> bool:
    """Both cancel-pending carriers in ONE predicate — fail-soft.

    The durable intent projection is the live authority; the legacy
    ``cancel_requested`` STATUS latch covers pre-redesign result files that boot
    migration has not converted yet. Steering-refusal call sites must consult
    both, or a task wedged in the old shape still accepts new owner messages
    while the supervisor is tearing it down.
    """
    try:
        if has_active_intent(drive_root, task_id, strict=strict):
            return True
    except Exception:
        if strict:
            raise
        log.debug("cancel-pending intent read failed for %s", task_id, exc_info=True)
    try:
        from ouroboros.task_results import STATUS_CANCEL_REQUESTED, load_task_result

        status = str(
            (load_task_result(drive_root, task_id, strict=strict) or {}).get("status") or ""
        )
        return status == STATUS_CANCEL_REQUESTED
    except Exception:
        if strict:
            raise
        log.debug("cancel-pending legacy latch read failed for %s", task_id, exc_info=True)
        return False
