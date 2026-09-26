"""Helpers for durable task result/status files."""

from __future__ import annotations

import copy
import json
import logging
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional

from ouroboros.cost_projection import (
    COST_ALIAS_PAIRS, COST_OPENNESS_FIELDS,
    normalize_task_result_cost_planes,
)
from ouroboros.utils import read_json_dict, update_json_locked, utc_now_iso
# Read-side custody of a published review projection belongs with the projection
# owner; the historical name stays resolvable through this module.
from ouroboros.review_projection import merge_review_projection as merge_review_projection
from ouroboros.review_records import validate_author_disposition

log = logging.getLogger(__name__)

STATUS_REQUESTED = "requested"
STATUS_SCHEDULED = "scheduled"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_REJECTED_DUPLICATE = "rejected_duplicate"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"
STATUS_CANCELLED = "cancelled"

# ABI 7.0 (Q8=B): re-export schema admission, stamping and quarantine for callers.
from ouroboros.task_result_schema import (  # noqa: F401
    QUARANTINED_SCHEMA_REASON, TASK_RESULT_QUARANTINE_DIR, TASK_RESULT_SCHEMA_VERSION,
    emit_quarantine_event as _emit_quarantine_event,
    quarantine_task_result as _quarantine_task_result,
    require_writable_task_result_schema, stamp_task_result_schema,
    task_result_schema_refusal,
)


def review_binding_hash(
    *, candidate_hash: str, evidence_revision: str, fence_hash: str,
) -> str:
    """Digest the immutable task-acceptance binding components."""

    import hashlib

    payload = {
        "candidate_hash": str(candidate_hash or ""),
        "evidence_revision": str(evidence_revision or ""),
        "fence_hash": str(fence_hash or ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def effective_task_acceptance_review_cycles(
    profile: Dict[str, Any], *,
    required_blocking: bool = False,
) -> Optional[int]:
    """Paid capacity is independent of the author's last response opportunity.

    Explicit task-local pass profiles retain their established p+1 paid ceiling.
    """
    from ouroboros.review_cycles import review_max_cycles

    passes = profile.get("max_improvement_passes")
    return review_max_cycles() if passes is None else max(1, int(passes) + 1)


def _root_task_acceptance_review_cap(
    root_result: Dict[str, Any],
) -> Optional[int]:
    """Resolve the cap only from the canonical root result: no claimant fallback.
    Otherwise a descendant's deadline could widen the shared wallet, or
    process-local state could silently recreate a deleted root.
    """

    if not root_result or "task_contract" not in root_result:
        raise ValueError(
            "TASK_ACCEPTANCE_REVIEW_STATE_UNKNOWN: root cap authority is absent"
        )

    from ouroboros.contracts.task_contract import (
        VALID_IMPROVEMENT_POLICIES,
        normalize_budget_profile,
    )
    from ouroboros.deadline_utils import parse_deadline_ts

    root_contract = root_result.get("task_contract")
    if (
        not isinstance(root_contract, dict)
        or root_contract.get("schema_version") != 1
        or "deadline_at" not in root_contract
        or not isinstance(root_contract.get("budget_profile"), dict)
    ):
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_UNKNOWN: root contract is malformed")
    raw_profile = root_contract["budget_profile"]
    policy = raw_profile.get("improvement_policy")
    max_passes = raw_profile.get("max_improvement_passes")
    if policy not in VALID_IMPROVEMENT_POLICIES or (
        max_passes is not None
        and (not isinstance(max_passes, int) or max_passes < 0)
    ):
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_UNKNOWN: root budget profile is malformed")
    deadline_at = root_contract["deadline_at"]
    if not isinstance(deadline_at, str):
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_UNKNOWN: root deadline is malformed")
    deadline = parse_deadline_ts(deadline_at)
    if deadline_at.strip() and deadline is None:
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_UNKNOWN: root deadline is malformed")
    profile = normalize_budget_profile(raw_profile)
    return effective_task_acceptance_review_cycles(
        profile,
        required_blocking=task_acceptance_required_blocking(),
    )


def task_acceptance_required_blocking() -> bool:
    """Shared Required+Blocking predicate for the cap and capacity readers.
    It matches ``loop_acceptance``: ``ctx.mode == 'required'`` (captured from
    ``config.get_task_review_mode()`` at ``_run_task_acceptance_review_once``)
    and ``loop.get_review_enforcement() == 'blocking'``. That getter is the
    same config getter used here, so these projections cannot drift apart."""
    from ouroboros import config

    return bool(
        config.get_task_review_mode() == "required"
        and config.get_review_enforcement() == "blocking"
    )


def _claim_for_paid_identity(claims: Any, paid_identity: str) -> Optional[Dict[str, Any]]:
    """The claim row this tree already bought for one A-material paid identity.

    ONE answer shared by the free-refusal projection and the atomic claim, so the
    dispatch seam can never refuse what the wallet would have allowed (or vice
    versa). An empty identity matches nothing: a pre-A-material row keys on its
    binding hash alone."""
    identity = str(paid_identity or "").strip().lower()
    if not identity:
        return None
    return next(
        (
            row for row in (claims or {}).values()
            if isinstance(row, dict) and str(row.get("paid_identity") or "") == identity
        ),
        None,
    )


def project_task_acceptance_review_capacity(
    ctx: Any, *, binding_hash: str = "", task_id: str = "", paid_identity: str = "",
) -> Dict[str, Any]:
    """Read the canonical root's paid acceptance-wallet projection.

    Descendants may observe but never initialize root authority. A missing or
    malformed canonical result is UNKNOWN for them; a live root may begin with
    the known empty state. The atomic claim remains dispatch authority.

    WALLET AND CANCELLATION ONLY (owner R52). The TIME axis is not projected:
    the launch rule (``task_pacing.review_launch_allowed``) is evaluated once,
    at loop admission (owner R55; the paid claim inside the dispatch stamp
    checks cancellation and the wallet only), and a descendant reading this
    reads its own deadline window from the adjacent coordination ``time``
    fact. So no budget profile, no snapshot and no duration prediction is read
    here, and the answer cannot drift from the rule it used to imitate.
    """

    from ouroboros import config

    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    task_id = str(task_id or getattr(ctx, "task_id", "") or "")
    lineage = resolve_task_lineage(
        task_id,
        metadata=metadata,
        root_task_id=getattr(ctx, "root_task_id", None),
        parent_task_id=getattr(ctx, "parent_task_id", None),
        delegation_role=getattr(ctx, "delegation_role", None),
        original_task_id=getattr(ctx, "original_task_id", None),
        timeout_retry_from=getattr(ctx, "timeout_retry_from", None),
    )
    root_task_id = str(lineage.get("root_task_id") or task_id)
    root = pathlib.Path(str(
        metadata.get("budget_drive_root")
        or getattr(ctx, "budget_drive_root", "")
        or getattr(ctx, "drive_root", ".")
    ))
    base = {
        "root_task_id": root_task_id,
        "cap_cycles": None,
        "claimed_cycles": None,
        "remaining_cycles": None,
        "binding_seen": False,
        "dedupe": "task_acceptance_binding_sha256",
    }
    if config.get_task_review_mode() == "off":
        return {
            **base,
            "state": "unavailable",
            "reason": "task_review_mode_off",
        }
    try:
        state = load_task_acceptance_review_state(
            root,
            root_task_id,
            require_root_result=not bool(lineage.get("is_root_task")),
        )
        path = task_result_path(root, root_task_id, create=False)
        root_result = read_json_dict(path) if path.is_file() else {}
        if path.is_file() and root_result is None:
            raise ValueError("root result is malformed")
        root_result = root_result if isinstance(root_result, dict) else {}
        cap = _root_task_acceptance_review_cap(root_result)
        claims = state.get("claims_by_binding") or {}
        claimed = len(claims)
        remaining = None if cap is None else max(0, cap - claimed)
        requested_binding = str(binding_hash or "").strip().lower()
        # Seen = this dispatch was already PAID for, under either identity: the
        # exact binding (as before) or the A-material paid identity, so a resubmit
        # that only moved the evidence revision cannot buy a second panel.
        binding_seen = bool(requested_binding and requested_binding in claims) or (
            _claim_for_paid_identity(claims, paid_identity) is not None
        )
        projection = {
            **base,
            "state": "available",
            "reason": "",
            "cap_cycles": cap,
            "claimed_cycles": claimed,
            "remaining_cycles": remaining,
            "binding_seen": binding_seen,
        }
        try:
            from ouroboros.cancel_intents import cancel_pending

            if cancel_pending(root, root_task_id, strict=True) or (
                task_id != root_task_id
                and cancel_pending(root, task_id, strict=True)
            ):
                projection.update({
                    "state": "unavailable", "reason": "cancellation_pending",
                })
                return projection
        except Exception as exc:
            return {
                **projection,
                "state": "unknown",
                "reason": f"cancellation_state_unknown:{type(exc).__name__}",
            }
        if remaining == 0:
            projection.update({
                "state": "unavailable", "reason": "review_cycles_exhausted",
            })
        return projection
    except Exception as exc:
        return {
            **base,
            "state": "unknown",
            "reason": f"review_capacity_unknown:{type(exc).__name__}",
        }

# Cancel intent outranks running mirrors but not terminal states: stale progress
# cannot resurrect the task, and the supervisor's final CANCELLED write still lands.
STATUS_CANCEL_REQUESTED = "cancel_requested"

# Events, progress replay, chat summaries and results share the cost-projection
# SSOT's current names and ALL openness/integrity markers. Legacy cost_usd aliases
# are read-only compatibility: use carry_cost_meta, not a key loop, to copy them.
# In particular, non_final_rows explains cost_final; omitting it or
# ledger_integrity_degraded would silently lose the authority's uncertainty.
TASK_COST_META_FIELDS = tuple(dict.fromkeys(
    [new for new, _old in COST_ALIAS_PAIRS] + list(COST_OPENNESS_FIELDS)
))

# Monotonic lifecycle: stale scheduled/running mirrors cannot overwrite cancellation
# or terminal outcomes (the ghost-subagent class). Unknown statuses never block.
_TRULY_TERMINAL_STATUSES = frozenset({
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_CANCELLED,
    STATUS_REJECTED_DUPLICATE,
})
_STATUS_RANK = {
    STATUS_REQUESTED: 0,
    STATUS_SCHEDULED: 1,
    STATUS_RUNNING: 2,
    STATUS_INTERRUPTED: 2,
    STATUS_CANCEL_REQUESTED: 3,
    STATUS_COMPLETED: 4,
    STATUS_FAILED: 4,
    STATUS_CANCELLED: 4,
    STATUS_REJECTED_DUPLICATE: 4,
}
# Regressions are only blocked once a task reaches the cancel-intent latch or a
# terminal state; normal forward progress (requested->scheduled->running) and
# unknown statuses are always allowed.
_REGRESSION_GUARD_FLOOR = _STATUS_RANK[STATUS_CANCEL_REQUESTED]

_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

PLAN_REVIEW_STATE_KEY = "plan_review_state"
_PLAN_REVIEW_STATE_VERSION = 2
_PLAN_REVIEW_STATE_MAX_BYTES = 1_000_000
_PLAN_REVIEW_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_PLAN_REVIEW_REASON_MAX_CHARS = 2_000

TASK_ACCEPTANCE_REVIEW_STATE_KEY = "task_acceptance_review_accounting"
_TASK_ACCEPTANCE_REVIEW_STATE_VERSION = 1
_TASK_ACCEPTANCE_REVIEW_STATE_MAX_BYTES = 1_000_000
_TASK_ACCEPTANCE_REVIEW_CLAIM_FIELDS = frozenset({
    "binding_hash", "candidate_hash", "evidence_revision", "fence_hash",
    "claimed_at", "claimed_by_task_id",
})
# A-material (2026-08-30), additive: the identity the panel was actually PAID
# for — candidate answer plus new obligation dispositions, evidence revision
# deliberately excluded. Rows written before it carry no such key and stay valid;
# new rows carry both, so the binding-keyed reads above never change meaning.
_TASK_ACCEPTANCE_REVIEW_CLAIM_OPTIONAL_FIELDS = frozenset({"paid_identity"})


def _empty_task_acceptance_review_state(root_task_id: str) -> Dict[str, Any]:
    return {
        "schema_version": _TASK_ACCEPTANCE_REVIEW_STATE_VERSION,
        "root_task_id": str(root_task_id),
        "claims_by_binding": {},
    }


def _validated_task_acceptance_review_state(
    value: Any, root_task_id: str,
) -> Dict[str, Any]:
    """Strict private copy of the root tree's paid acceptance claims."""

    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: unsupported schema")
    if set(value) != {"schema_version", "root_task_id", "claims_by_binding"}:
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: state shape is invalid")
    if str(value.get("root_task_id") or "") != str(root_task_id):
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: root identity mismatch")
    claims = value.get("claims_by_binding")
    if not isinstance(claims, dict):
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: claims must be an object")
    for binding_hash, claim in claims.items():
        if not _PLAN_REVIEW_HASH_RE.fullmatch(str(binding_hash or "")):
            raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: claim key is invalid")
        if not isinstance(claim, dict) or not (
            _TASK_ACCEPTANCE_REVIEW_CLAIM_FIELDS
            <= set(claim)
            <= (_TASK_ACCEPTANCE_REVIEW_CLAIM_FIELDS | _TASK_ACCEPTANCE_REVIEW_CLAIM_OPTIONAL_FIELDS)
        ):
            raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: claim shape is invalid")
        if str(claim.get("binding_hash") or "") != str(binding_hash):
            raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: claim identity mismatch")
        for key in ("binding_hash", "candidate_hash", "evidence_revision", "fence_hash"):
            if not _PLAN_REVIEW_HASH_RE.fullmatch(str(claim.get(key) or "")):
                raise ValueError(
                    f"TASK_ACCEPTANCE_REVIEW_STATE_INVALID: {key} is invalid"
                )
        if "paid_identity" in claim and not _PLAN_REVIEW_HASH_RE.fullmatch(
            str(claim.get("paid_identity") or "")
        ):
            raise ValueError(
                "TASK_ACCEPTANCE_REVIEW_STATE_INVALID: paid_identity is invalid"
            )
        expected_binding = review_binding_hash(
            candidate_hash=str(claim["candidate_hash"]),
            evidence_revision=str(claim["evidence_revision"]),
            fence_hash=str(claim["fence_hash"]),
        )
        if expected_binding != str(binding_hash):
            raise ValueError(
                "TASK_ACCEPTANCE_REVIEW_STATE_INVALID: binding digest mismatch"
            )
        for key in ("claimed_at", "claimed_by_task_id"):
            if not isinstance(claim.get(key), str) or not str(claim.get(key) or ""):
                raise ValueError(
                    f"TASK_ACCEPTANCE_REVIEW_STATE_INVALID: {key} must be non-empty text"
                )
    copied = copy.deepcopy(value)
    if len(json.dumps(copied, ensure_ascii=False).encode("utf-8")) > _TASK_ACCEPTANCE_REVIEW_STATE_MAX_BYTES:
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: state is too large")
    return copied


def load_task_acceptance_review_state(
    results_drive_root: Any,
    root_task_id: str,
    *,
    require_root_result: bool = False,
) -> Dict[str, Any]:
    """Read the canonical root's shared paid-review claims without mutation."""

    path = task_result_path(results_drive_root, root_task_id, create=False)
    if not path.is_file():
        if require_root_result:
            raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_UNKNOWN: root result is absent")
        return _empty_task_acceptance_review_state(root_task_id)
    result = read_json_dict(path)
    if result is None:
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: root result is malformed")
    if str(result.get("task_id") or "") != str(root_task_id):
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: root identity mismatch")
    stored_root_id = str(result.get("root_task_id") or "")
    if stored_root_id and stored_root_id != str(root_task_id):
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: root identity mismatch")
    if TASK_ACCEPTANCE_REVIEW_STATE_KEY not in result:
        return _empty_task_acceptance_review_state(root_task_id)
    return _validated_task_acceptance_review_state(
        result[TASK_ACCEPTANCE_REVIEW_STATE_KEY], root_task_id,
    )


def _update_task_acceptance_review_state(
    results_drive_root: Any,
    root_task_id: str,
    mutator: Callable[
        [Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]
    ],
    *,
    allow_create: bool,
) -> Dict[str, Any]:
    """Strict root-result update; the file lock is the tree-wide claim fence."""

    path = task_result_path(results_drive_root, root_task_id, create=allow_create)
    if not allow_create and not path.is_file():
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_UNKNOWN: root result is absent")
    observed_state: Dict[str, Any] = {}

    def _merge(existing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not allow_create and not existing:
            raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_UNKNOWN: root result is absent")
        require_writable_task_result_schema(existing, path)
        if existing and str(existing.get("task_id") or "") != str(root_task_id):
            raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: root identity mismatch")
        stored_root_id = str(existing.get("root_task_id") or "")
        if stored_root_id and stored_root_id != str(root_task_id):
            raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: root identity mismatch")
        state = (
            _validated_task_acceptance_review_state(
                existing[TASK_ACCEPTANCE_REVIEW_STATE_KEY], root_task_id,
            )
            if TASK_ACCEPTANCE_REVIEW_STATE_KEY in existing
            else _empty_task_acceptance_review_state(root_task_id)
        )
        observed_state.clear()
        observed_state.update(copy.deepcopy(state))
        candidate = mutator(state, existing)
        if candidate is None:
            return None
        updated = _validated_task_acceptance_review_state(
            candidate, root_task_id,
        )
        now = utc_now_iso()
        return stamp_task_result_schema({
            **existing,
            TASK_ACCEPTANCE_REVIEW_STATE_KEY: updated,
            "task_id": str(root_task_id),
            "status": str(existing.get("status") or STATUS_RUNNING),
            "ts": str(existing.get("ts") or now),
            "updated_at": now,
        })

    try:
        updated = update_json_locked(
            path,
            _merge,
            strict_existing_dict=True,
            reject_existing_empty_dict=True,
        )
    except ValueError as exc:
        if str(exc).startswith("update_json_locked:"):
            raise ValueError(
                "TASK_ACCEPTANCE_REVIEW_STATE_INVALID: root result is malformed"
            ) from exc
        raise
    if TASK_ACCEPTANCE_REVIEW_STATE_KEY not in updated:
        return _validated_task_acceptance_review_state(observed_state, root_task_id)
    return _validated_task_acceptance_review_state(
        updated[TASK_ACCEPTANCE_REVIEW_STATE_KEY], root_task_id,
    )


def claim_task_acceptance_review_cycle(
    results_drive_root: Any,
    root_task_id: str,
    review_binding: Dict[str, Any],
    *,
    claimed_by_task_id: str,
) -> Dict[str, Any]:
    """Atomically dedupe and claim one paid root-acceptance panel dispatch."""

    binding_fields = {
        key: str((review_binding or {}).get(key) or "").strip().lower()
        for key in ("binding_hash", "candidate_hash", "evidence_revision", "fence_hash")
    }
    if any(
        not _PLAN_REVIEW_HASH_RE.fullmatch(value)
        for value in binding_fields.values()
    ):
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: review binding is invalid")
    expected_binding = review_binding_hash(
        candidate_hash=binding_fields["candidate_hash"],
        evidence_revision=binding_fields["evidence_revision"],
        fence_hash=binding_fields["fence_hash"],
    )
    if binding_fields["binding_hash"] != expected_binding:
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: binding digest mismatch")
    binding = binding_fields["binding_hash"]
    # A-material: the identity the panel is PAID for. Absent (a pre-A-material
    # caller) => the binding hash keeps being the paid identity, i.e. exactly the
    # old behaviour; present => it also refuses a second claim for the same
    # material under a different binding.
    paid_identity = str((review_binding or {}).get("paid_identity") or "").strip().lower()
    if paid_identity and not _PLAN_REVIEW_HASH_RE.fullmatch(paid_identity):
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: paid identity is invalid")
    claimant = str(claimed_by_task_id or "").strip()
    if not claimant:
        raise ValueError("TASK_ACCEPTANCE_REVIEW_STATE_INVALID: claimant is absent")
    decision: Dict[str, Any] = {}
    resolved: Dict[str, Optional[int]] = {}

    def _claim(
        state: Dict[str, Any], root_result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        from ouroboros.cancel_intents import cancel_pending

        if cancel_pending(results_drive_root, root_task_id, strict=True) or (
            claimant != str(root_task_id)
            and cancel_pending(results_drive_root, claimant, strict=True)
        ):
            decision.update({"status": "unavailable", "reason": "cancellation_pending"})
            return None
        cap = _root_task_acceptance_review_cap(root_result)
        resolved["cap"] = cap
        claims = dict(state.get("claims_by_binding") or {})
        prior = claims.get(binding) or _claim_for_paid_identity(claims, paid_identity)
        if prior is not None:
            decision.update({
                "status": "unknown",
                "reason": "binding_dispatch_already_claimed",
            })
            return None
        if cap is not None and len(claims) >= cap:
            decision.update({"status": "unavailable", "reason": "review_cycles_exhausted"})
            return None
        claims[binding] = {
            **binding_fields,
            **({"paid_identity": paid_identity} if paid_identity else {}),
            "claimed_at": utc_now_iso(),
            "claimed_by_task_id": claimant,
        }
        state["claims_by_binding"] = claims
        decision.update({"status": "claimed", "reason": ""})
        return state

    from ouroboros.cancel_intents import cancellation_projection_lock

    with cancellation_projection_lock(results_drive_root):
        state = _update_task_acceptance_review_state(
            results_drive_root,
            root_task_id,
            _claim,
            allow_create=False,
        )
    paid = len(state.get("claims_by_binding") or {})
    cap = resolved.get("cap")
    return {
        **decision,
        "binding_hash": binding,
        "paid_identity": paid_identity,
        "cycles_paid": paid,
        "max_cycles": cap,
        "remaining_cycles": None if cap is None else max(0, cap - paid),
    }


def cancellation_blocks_child_result(result: Any) -> bool:
    """Return whether canonical cancellation forbids child-drive promotion.

    Only a supervisor-SETTLED ``cancelled`` blocks: cancel INTENT no longer rides
    the canonical status (it lives in the durable ``cancel_intents`` projection),
    and natural completion WINS a late cancel (owner decision 4=A, 2026-08-11) —
    a child that finished before the teardown keeps its completed result and
    artifacts, so copy-back paths must promote it rather than refuse it.
    """

    if not isinstance(result, dict):
        return False
    return str(result.get("status") or "").strip().lower() == STATUS_CANCELLED


def resolve_task_lineage(
    task_id: Any,
    *,
    metadata: Any = None,
    root_task_id: Any = None,
    parent_task_id: Any = None,
    delegation_role: Any = None,
    original_task_id: Any = None,
    timeout_retry_from: Any = None,
) -> Dict[str, Any]:
    """Return one typed lineage projection for root-owned lifecycle gates.

    ``root_task_id`` is the logical subtree/budget authority and intentionally
    survives a top-level hard-timeout retry that receives a fresh physical
    ``task_id``.  Such a retry is a root *attempt* only when the two independent
    host-written retry markers agree.  This keeps malformed lineage fail-closed
    without splitting budget, fence, task-tree, or cost authorities.
    """

    meta = metadata if isinstance(metadata, dict) else {}

    def _field(explicit: Any, key: str) -> str:
        # ``None`` means the canonical carrier is absent.  An explicit empty
        # parent is meaningful and must override stale copied metadata.
        value = explicit if explicit is not None else meta.get(key)
        return str(value or "").strip()

    resolved_task_id = str(task_id or "").strip()
    resolved_root_id = _field(root_task_id, "root_task_id") or resolved_task_id
    resolved_parent_id = _field(parent_task_id, "parent_task_id")
    resolved_role = _field(delegation_role, "delegation_role").lower()
    resolved_original_id = _field(original_task_id, "original_task_id")
    resolved_retry_from = _field(timeout_retry_from, "timeout_retry_from")
    is_regular_root = bool(resolved_task_id and resolved_root_id == resolved_task_id
                           and not resolved_parent_id and resolved_role != "subagent")
    is_retry_root = bool(resolved_task_id and resolved_root_id and resolved_root_id != resolved_task_id
                         and not resolved_parent_id and resolved_role == "root" and resolved_original_id
                         and resolved_original_id == resolved_retry_from and resolved_original_id != resolved_task_id)
    return {
        "task_id": resolved_task_id,
        "root_task_id": resolved_root_id,
        "parent_task_id": resolved_parent_id,
        "delegation_role": resolved_role,
        "original_task_id": resolved_original_id,
        "timeout_retry_from": resolved_retry_from,
        "is_retry_root_attempt": is_retry_root,
        "is_root_task": bool(is_regular_root or is_retry_root),
    }


def _is_status_regression(existing_status: str, new_status: str) -> bool:
    """Return True when writing *new_status* over *existing_status* would
    regress or corrupt a task that has already reached cancel-intent or a
    terminal state.

    Rules:
      - Unknown statuses never block (forward-compatible).
      - Truly-terminal is sticky: once completed/failed/cancelled/rejected, only
        a same-status rewrite is allowed (result/trace enrichment). Switching to
        a *different* terminal status (e.g. cancelled -> completed) is blocked.
      - cancel-intent (cancel_requested) blocks regress to running/scheduled but
        still allows the supervisor's eventual terminal write (rank 3 -> 4).
    """
    existing = str(existing_status or "")
    new = str(new_status or "")
    # Sticky terminal FIRST — independent of whether the new status is ranked, so
    # a typo/unknown/future status can never overwrite a terminal one. Only an
    # identical-status rewrite (result/trace enrichment) is allowed.
    if existing in _TRULY_TERMINAL_STATUSES:
        return new != existing
    if existing == STATUS_CANCEL_REQUESTED:
        # LEGACY read-path only (pre-intent files): the latch status is no longer
        # written — cancel intent lives in the durable ``cancel_intents``
        # projection. Natural completion WINS (owner decision 4=A): any terminal
        # write, including a racing ``completed``, may land over an old latch;
        # only a regression to scheduled/running is refused.
        new_rank = _STATUS_RANK.get(new)
        return new_rank is not None and new_rank < _STATUS_RANK[STATUS_CANCEL_REQUESTED]
    existing_rank = _STATUS_RANK.get(existing)
    new_rank = _STATUS_RANK.get(new)
    if existing_rank is None or new_rank is None:
        return False
    if existing_rank >= _REGRESSION_GUARD_FLOOR:
        return new_rank < existing_rank
    return False


def is_reconciled_presence_placeholder(row: Dict[str, Any]) -> bool:
    """The orphan reconciler's failed mark on a lost presence turn: no terminal of its own ever ran."""
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return (row.get("status") == STATUS_FAILED and not row.get("terminal_origin")
            and row.get("status_reconciled_from") in {STATUS_RUNNING, STATUS_INTERRUPTED}
            and row.get("reason_code") in {"orphaned_running_after_worker_restart", "interrupted_retry_lost"}
            and (meta.get("source") == "presence" or isinstance(meta.get("presence"), dict)))


def reopen_reconciled_presence_placeholder(drive_root: Any, task_id: str) -> bool:
    """Return a placeholder to ``running`` for its re-run; the host mark moves to ``superseded_placeholder``."""
    path, reopened = task_result_path(drive_root, task_id), []
    # The terminal-projection bookkeeping of the placeholder's failed transition goes with it: the
    # re-run's own terminal transition must originate its own room row, never inherit a failed one.
    projection = ("canonical_terminal_projection", "canonical_terminal_projection_ready",
                  "canonical_terminal_projection_origin")
    cleared = {"reason_code", "outcome_axes", "artifact_status", "artifact_bundle", "result",
               "status_reconciled_from", *projection}

    def _reopen(existing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not is_reconciled_presence_placeholder(existing):
            return None
        require_writable_task_result_schema(existing, path)
        mark = {key: existing.get(key) for key in ("status", "reason_code", "status_reconciled_from", "ts", *projection)
                if key in existing}
        reopened.append(utc_now_iso())
        return stamp_task_result_schema({
            **{key: value for key, value in existing.items() if key not in cleared},
            "status": STATUS_RUNNING, "ts": reopened[0], "updated_at": reopened[0],
            "superseded_placeholder": {**mark, "result": str(existing.get("result") or "")[-500:]},
        })

    update_json_locked(path, _reopen, strict_existing_dict=True)
    return bool(reopened)


def validate_task_id(task_id: Any) -> str:
    text = str(task_id or "").strip()
    if not _TASK_ID_RE.fullmatch(text):
        raise ValueError("task_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
    return text


def task_results_dir(drive_root: Any, *, create: bool = True) -> pathlib.Path:
    """Resolve ``<drive_root>/task_results``; writers create it, readers pass ``create=False``.

    A read-side mkdir once let a scan with a MagicMock-derived root create a stray
    ``MagicMock/.../task_results`` tree in cwd, so a never-provisioned root reads as empty."""
    path = pathlib.Path(drive_root) / "task_results"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def task_result_path(drive_root: Any, task_id: str, *, create: bool = True) -> pathlib.Path:
    return task_results_dir(drive_root, create=create) / f"{validate_task_id(task_id)}.json"


def _normalize_swarm_efficiency(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read the two retired fan-out names without rewriting historical records."""
    value = data.get("swarm_efficiency")
    if not isinstance(value, dict):
        return data
    value = dict(value)
    for old, new in (("wave_count", "fanout_count"),
                     ("inter_wave_latency_sec_total", "fanout_interval_sec_total")):
        if old in value:
            value.setdefault(new, value.pop(old))
    return {**data, "swarm_efficiency": value}


def _admit_task_result(
    path: pathlib.Path, data: Any, task_id: str, *, strict: bool, noun: str,
) -> tuple[Optional[Dict[str, Any]], str]:
    """Schema admission shared by exact and scan reads: ``(row or None, refusal when quarantined)``.

    Fail-soft refusal quarantines; strict refusal raises without moving, so
    unreadable identities cannot be readmitted."""
    refusal = task_result_schema_refusal(data)
    if refusal and strict:  # "malformed" keeps the pre-ABI-2 strict message for unreadable rows
        raise ValueError(f"{noun} is unreadable or invalid: {path}" if refusal == "malformed" else
                         f"task result schema is inadmissible ({QUARANTINED_SCHEMA_REASON}: {refusal}): {path}")
    if refusal:
        outcome = _quarantine_task_result(path, refusal)
        data = read_json_dict(path) if outcome == "kept_admissible" else None
        if data is None or task_result_schema_refusal(data):
            return None, refusal if outcome == "moved" else ""
    if strict and (str(data.get("task_id") or "") != task_id or not isinstance(data.get("status"), str)
                   or not str(data.get("status") or "").strip()):
        raise ValueError(f"{noun} is unreadable or invalid: {path}")
    return data, ""


def load_task_result(
    drive_root: Any, task_id: str, *, strict: bool = False,
) -> Optional[Dict[str, Any]]:
    """Read one exact result through schema admission, tolerating old fan-out names.

    Fail-soft refusal quarantines with a batched event and returns None; strict
    refusal raises without moving, so unreadable identities cannot be readmitted.
    Neither mode materializes deliverables or independent state axes."""
    try:
        tid = validate_task_id(task_id)
        path = task_result_path(drive_root, tid, create=False)
    except ValueError:
        if strict:
            raise
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None  # This read saw absence even if a writer publishes immediately after it.
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        data = None
    data, moved = _admit_task_result(path, data, tid, strict=strict, noun="task result authority")
    if moved:
        _emit_quarantine_event(drive_root, [{"task_id": tid, "reason": moved}])
    return _normalize_swarm_efficiency(data) if data is not None else None


def list_task_results(
    drive_root: Any,
    *,
    statuses: Optional[List[str]] = None,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """List task results, optionally refusing an incomplete authority scan.

    Most observational callers remain tolerant of a malformed historical row.
    Authority reducers such as direct-child admission pass ``strict=True`` so
    an unreadable row cannot be silently reinterpreted as an absent child.

    Schema admission (ABI 7.0, Q8=B): the fail-soft scan QUARANTINES every
    inadmissible row it meets and reports the whole sweep as ONE durable
    event; the strict scan raises WITHOUT moving anything. Rows already under
    ``task_results/quarantine/`` are outside this scan (non-recursive glob).
    """
    wanted = {str(item) for item in list(statuses or []) if str(item).strip()}
    results: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, str]] = []
    for path in sorted(task_results_dir(drive_root, create=False).glob("*.json")):
        data = read_json_dict(path)
        if data is None and not path.is_file():
            continue  # vanished mid-scan — nothing to admit or quarantine
        data, moved = _admit_task_result(path, data, path.stem, strict=strict, noun="task result")
        if moved:
            quarantined.append({"task_id": path.stem, "reason": moved})
        if data is None or (wanted and str(data.get("status") or "") not in wanted):
            continue
        results.append(data)
    _emit_quarantine_event(drive_root, quarantined)
    return results


def write_task_result(
    results_drive_root: Any,
    task_id: str,
    status: str,
    *,
    _field_projector: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    strict_existing_dict: bool = False,
    create_only: bool = False,
    **fields: Any,
) -> Dict[str, Any]:
    """Merge-write a task result under a per-file lock.

    Workers, supervisor and gateways share CURRENT under this lock: the monotonic
    reducer decides terminal races, so natural completion wins a late cancel.
    Discarding a result remains the separate explicit ``discard_child_result``.
    ``_field_projector`` runs after review-publication selection under the same
    lock, deriving fields/status from CURRENT or publishing its verified refs;
    repeating the incoming-review merge afterward would undo that handoff.
    ``strict_existing_dict`` refuses malformed/non-object, empty, wrong-identity
    or wrong-schema authority under the write lock, never replacing it with {}.
    ``create_only`` aborts on an existing nonempty row after those checks, before
    projection, normalization or timestamps. Pair it with strict validation to
    initialize only absence while preserving unknown bytes.
    """
    path = task_result_path(results_drive_root, task_id)
    explicit_ts = str(fields.pop("ts", "") or "")
    from ouroboros.task_custody import capture_unread_mail, merge_unread_mail

    # TZ-1 V10: the mailbox bytes are read BEFORE the row lock (a bounded union happens
    # under it); a terminal write that the projector turns terminal captures under it.
    captured = capture_unread_mail(results_drive_root, task_id) if status in _TRULY_TERMINAL_STATUSES else None

    def _merge(existing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if strict_existing_dict and existing and (
            str(existing.get("task_id") or "") != str(task_id)
            or not isinstance(existing.get("status"), str)
            or not str(existing.get("status") or "").strip()
        ):
            raise ValueError(
                f"task result authority is unreadable or invalid: {path}"
            )
        # ABI 7.0: every write stamps the row; another schema's row is never downgraded.
        require_writable_task_result_schema(existing, path)
        if create_only and existing:
            return None
        prepared_fields = dict(fields)
        if "review_projection" in prepared_fields:
            prepared_fields["review_projection"] = merge_review_projection(
                existing.get("review_projection"), prepared_fields["review_projection"],
            )
        projected_fields = _field_projector(existing, {**prepared_fields, "status": status}) if _field_projector else prepared_fields
        if projected_fields is None:  # projector saw a terminal/stale row: no mutation
            return None
        projected_status = str(projected_fields.pop("status", status))
        # Monotonic lifecycle: no stale mirror may overwrite a terminal outcome.
        existing_status = str(existing.get("status") or "")
        if existing and _is_status_regression(existing_status, projected_status):
            # Debugging a "stuck" task: the only signal that a stale/late write was dropped.
            log.debug("Blocked status regression %s -> %s for task %s",
                      existing.get("status"), projected_status, task_id)
            return None
        # Only this accepted transition can originate terminal delivery debt:
        # enrichment/replica fields cannot adopt historical terminal rows or
        # erase provenance persisted before the separate readiness write.
        projected_fields.pop("canonical_terminal_projection_origin", None)
        if projected_status in _TRULY_TERMINAL_STATUSES and existing_status not in _TRULY_TERMINAL_STATUSES:
            merged = {**existing, **projected_fields}
            lineage_keys = ("root_task_id", "parent_task_id", "delegation_role", "original_task_id", "timeout_retry_from")
            if resolve_task_lineage(task_id, metadata=merged.get("metadata"),
                                    **{key: merged.get(key) for key in lineage_keys})["is_root_task"]:
                projected_fields["canonical_terminal_projection_origin"] = "terminal_transition"
            # TZ-1 V10: this accepted transition keeps the mail no attempt read (no ACK written);
            # later late mail joins through settlement, never through a rejected write.
            projected_fields["unread_mailbox"] = merge_unread_mail(
                projected_fields.get("unread_mailbox"),
                captured if status in _TRULY_TERMINAL_STATUSES else capture_unread_mail(results_drive_root, task_id))
        # Unread-mail custody only grows: no replica or partial write can shrink it.
        projected_fields["unread_mailbox"] = merge_unread_mail(
            existing.get("unread_mailbox"), projected_fields.get("unread_mailbox"))
        if projected_fields["unread_mailbox"] is None:
            projected_fields.pop("unread_mailbox")
        now = utc_now_iso()
        # ABI-3 write seam: the merge BASE is normalized onto honest cost names first, so a stored alias
        # neither survives nor outranks this write's fresh value; a legacy spelling IN this write still
        # wins the final pass and leaves under the honest name. Both passes use the shared deep
        # normalizer, so nested cost planes (subagent envelope, loop-outcome usage) are rewritten too.
        return stamp_task_result_schema(normalize_task_result_cost_planes({
            **normalize_task_result_cost_planes(existing),
            **projected_fields,
            "task_id": task_id,
            "status": projected_status,
            "ts": explicit_ts or str(existing.get("ts") or now),
            "updated_at": now,
        }))

    # Never fall back to an unlocked read/merge/write: stale state would let timing, not the monotonic
    # reducer, pick a completed-vs-cancelled winner. Callers retry or fail their transition explicitly.
    return update_json_locked(
        path,
        _merge,
        strict_existing_dict=bool(strict_existing_dict),
        reject_existing_empty_dict=bool(strict_existing_dict),
    )


# plan_review_state v2 records each task's plan_task cycles: a fresh series_id,
# cycles_paid bounded by review_max_cycles(), need_evidence_seen (one request per
# locator), and current_attempt (fingerprint + open/unavailable/rail_degraded).
# Keep _PLAN_REVIEW_FULL_WAVES whole, compact older waves and count waves_omitted
# beyond _PLAN_REVIEW_MAX_WAVES. v1 remains read-only under legacy_v1; an open wave
# requires resubmission until a new plan_task starts v2, never automatic closure.

_PLAN_REVIEW_FULL_WAVES = 8
_PLAN_REVIEW_MAX_WAVES = 64
_PLAN_REVIEW_ATTEMPT_STATUSES = frozenset({"open", "unavailable", "rail_degraded", "cycles_exhausted"})
_PLAN_REVIEW_AGGREGATES = frozenset({"GREEN", "REVIEW_REQUIRED", "REVISE_PLAN", "DEGRADED"})
LEGACY_OPEN_STATUS = "legacy_open_requires_resubmission"


def _empty_plan_review_state() -> Dict[str, Any]:
    return {
        "schema_version": _PLAN_REVIEW_STATE_VERSION,
        "series_id": "",
        "cycles_paid": 0,
        "need_evidence_seen": [],
        "current_attempt": {},
        "waves": [],
        "waves_omitted": 0,
    }


def legacy_plan_review_projection(value: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only projection of a v1 record: ``{fingerprint, status, outcome, closed,
    acceptance_claims}`` where status ∈ closed | rail_degraded | open (every non-closed
    v1 ATTEMPT: open review, unavailable) | pending (a wave without any attempt) | absent."""
    attempt = value.get("current_attempt") if isinstance(value.get("current_attempt"), dict) else {}
    fingerprint = str(attempt.get("fingerprint") or "") or str(value.get("latest_review_fingerprint") or "")
    waves = value.get("waves") if isinstance(value.get("waves"), list) else []
    wave = next((w for w in waves if isinstance(w, dict)
                 and str(w.get("request_fingerprint") or "") == fingerprint), None) if fingerprint else None
    review = wave.get("review") if isinstance((wave or {}).get("review"), dict) else {}
    outcome = str(review.get("aggregate_signal") or "")
    integrated = bool(review and str((wave or {}).get("phase") or "") == "reviewed"
                      and str((wave or {}).get("review_evidence_status") or "integrated") != "pending")
    closed = integrated and bool(review.get("closed")) and outcome in {"GREEN", "REVIEW_REQUIRED"}
    if closed:
        status = "closed"
    elif str(attempt.get("status") or "") == "rail_degraded":
        status = "rail_degraded"
    elif fingerprint:
        status = "open"
    elif waves:
        status = "pending"  # a v1 wave that never reached a panel: hold under every policy
    else:
        status = "absent"
    return {
        "fingerprint": fingerprint, "status": status, "outcome": outcome, "closed": closed,
        "reason": str(attempt.get("reason") or ""),
        "acceptance_claims": list((wave or {}).get("acceptance_claims") or []),
    }


def _validated_plan_review_state(value: Any) -> Dict[str, Any]:
    """Validate bounded v2 state; wrap v1 read-only without migrating authority."""
    if value in (None, {}):
        return _empty_plan_review_state()
    if not isinstance(value, dict):
        raise ValueError("PLAN_REVIEW_STATE_INVALID: unsupported or malformed schema")
    if value.get("schema_version") == 1:
        state = _empty_plan_review_state()
        state["legacy_v1"] = copy.deepcopy(value)
        state["legacy_v1_projection"] = legacy_plan_review_projection(value)
        return state
    if value.get("schema_version") != _PLAN_REVIEW_STATE_VERSION:
        raise ValueError("PLAN_REVIEW_STATE_INVALID: unsupported or malformed schema")
    waves = value.get("waves")
    if not isinstance(waves, list) or len(waves) > _PLAN_REVIEW_MAX_WAVES:
        raise ValueError("PLAN_REVIEW_STATE_INVALID: waves must be a bounded list")
    seen: set[str] = set()
    for wave in waves:
        if not isinstance(wave, dict):
            raise ValueError("PLAN_REVIEW_STATE_INVALID: wave must be an object")
        fingerprint = str(wave.get("request_fingerprint") or "")
        if not _PLAN_REVIEW_HASH_RE.fullmatch(fingerprint) or fingerprint in seen:
            raise ValueError("PLAN_REVIEW_STATE_INVALID: wave fingerprints must be unique")
        if str(wave.get("aggregate") or "") not in _PLAN_REVIEW_AGGREGATES:
            raise ValueError("PLAN_REVIEW_STATE_INVALID: invalid wave aggregate")
        if not isinstance(wave.get("closed"), bool):
            raise ValueError("PLAN_REVIEW_STATE_INVALID: wave closed must be boolean")
        if wave["aggregate"] == "GREEN" and not wave["closed"]:
            raise ValueError("PLAN_REVIEW_STATE_INVALID: GREEN wave must be closed")
        if wave["aggregate"] in {"REVISE_PLAN", "DEGRADED"} and wave["closed"]:
            raise ValueError("PLAN_REVIEW_STATE_INVALID: REVISE_PLAN/DEGRADED wave cannot be closed")
        if not wave.get("compact"):
            if not isinstance(wave.get("spec"), dict) or not isinstance(wave.get("findings"), list):
                raise ValueError("PLAN_REVIEW_STATE_INVALID: full wave needs spec and findings")
            if not isinstance(wave.get("dispositions", []), list):
                raise ValueError("PLAN_REVIEW_STATE_INVALID: dispositions must be a list")
        if "author_disposition" in wave and validate_author_disposition(
            wave["author_disposition"], subject_hash=fingerprint,
        ) is None:
            raise ValueError("PLAN_REVIEW_STATE_INVALID: author_disposition is malformed or stale")
        seen.add(fingerprint)
    cycles_paid = value.get("cycles_paid", 0)
    if not isinstance(cycles_paid, int) or isinstance(cycles_paid, bool) or cycles_paid < 0:
        raise ValueError("PLAN_REVIEW_STATE_INVALID: cycles_paid must be a non-negative count")
    if not isinstance(value.get("need_evidence_seen", []), list):
        raise ValueError("PLAN_REVIEW_STATE_INVALID: need_evidence_seen must be a list")
    copied = {**_empty_plan_review_state(), **copy.deepcopy(value)}
    attempt = copied.get("current_attempt")
    if not isinstance(attempt, dict):
        raise ValueError("PLAN_REVIEW_STATE_INVALID: current_attempt must be an object")
    if attempt:
        if set(attempt) - {"fingerprint", "status", "reason", "author_subject"} or not {"fingerprint", "status", "reason"} <= set(attempt):
            raise ValueError("PLAN_REVIEW_STATE_INVALID: current_attempt shape is invalid")
        if not _PLAN_REVIEW_HASH_RE.fullmatch(str(attempt.get("fingerprint") or "")):
            raise ValueError("PLAN_REVIEW_STATE_INVALID: current attempt fingerprint is invalid")
        if str(attempt.get("status") or "") not in _PLAN_REVIEW_ATTEMPT_STATUSES:
            raise ValueError("PLAN_REVIEW_STATE_INVALID: current attempt status is invalid")
        if "author_subject" in attempt:
            subject = attempt["author_subject"]
            if (not isinstance(subject, dict) or not isinstance(subject.get("source_ref"), dict)
                    or not _PLAN_REVIEW_HASH_RE.fullmatch(str(subject.get("review_fingerprint") or ""))
                    or validate_author_disposition(subject.get("author_disposition"), subject_hash=attempt["fingerprint"]) is None):
                raise ValueError("PLAN_REVIEW_STATE_INVALID: current author subject is invalid")
        if len(str(attempt.get("reason") or "")) > _PLAN_REVIEW_REASON_MAX_CHARS:
            raise ValueError("PLAN_REVIEW_STATE_INVALID: current attempt reason is too large")
    if len(json.dumps(copied, ensure_ascii=False, default=str).encode("utf-8")) > _PLAN_REVIEW_STATE_MAX_BYTES:
        raise ValueError("PLAN_REVIEW_STATE_INVALID: state exceeds the bounded size limit")
    return copied


def load_plan_review_state(results_drive_root: Any, task_id: str) -> Dict[str, Any]:
    """Read bounded state, then resolve the full specs used by decision consumers."""
    path = task_result_path(results_drive_root, task_id, create=False)
    if not path.is_file():
        return _empty_plan_review_state()
    # Share the writer's lock: Windows may deny reads during atomic replacement.
    try:
        result = update_json_locked(path, lambda _: None, strict_existing_dict=True)
    except ValueError as exc:
        raise ValueError("PLAN_REVIEW_STATE_INVALID: parent task result JSON is malformed") from exc
    from ouroboros.tools.plan_review_artifacts import authority_state

    return authority_state(
        results_drive_root, task_id, _validated_plan_review_state(result.get(PLAN_REVIEW_STATE_KEY)),
    )


def plan_review_wave(state: Dict[str, Any], fingerprint: str) -> Optional[Dict[str, Any]]:
    """The wave recorded under ``fingerprint`` (full or compact), as a private copy."""
    for wave in (state or {}).get("waves") or []:
        if str(wave.get("request_fingerprint") or "") == fingerprint:
            return copy.deepcopy(wave)
    return None


def current_plan_review_wave(state: Any) -> Optional[Dict[str, Any]]:
    """Copy the current fingerprint's wave; only an absent fingerprint selects the latest."""
    if not isinstance(state, dict):
        return None
    attempt = state.get("current_attempt") if isinstance(state.get("current_attempt"), dict) else {}
    fingerprint = str(attempt.get("fingerprint") or "")
    if fingerprint:
        return plan_review_wave(state, fingerprint)
    waves = state.get("waves") if isinstance(state.get("waves"), list) else []
    return copy.deepcopy(waves[-1]) if waves else None


def _legacy_projection_of(state: Any) -> Dict[str, Any]:
    state = state if isinstance(state, dict) else {}
    projection = legacy_plan_review_projection(state) if state.get("schema_version") == 1 else state.get("legacy_v1_projection")
    return projection if isinstance(projection, dict) else {}


def plan_review_gate_projection(
    state: Any,
    enforcement: str,
    *,
    hard_rail: str = "",
) -> Dict[str, Any]:
    """Project finalization permission without changing the durable review facts.

    Current-attempt identity prevents stale closure. Blocking holds open reviews;
    spent cycles, unreachable quorum and hard rails permit honest blocked exits.
    Advisory/Cyber allow is not closure or PASS. Accepts v2 and raw/wrapped v1.
    """
    policy = "blocking" if str(enforcement or "").lower() == "blocking" else "advisory"
    control: Dict[str, Any] = {}
    attempted = False
    attempt = state.get("current_attempt") if isinstance(state, dict) and isinstance(state.get("current_attempt"), dict) else {}
    if isinstance(state, dict):
        legacy = _legacy_projection_of(state)
        wave = current_plan_review_wave(state) if state.get("schema_version") != 1 else None
        if wave is not None or (attempt and state.get("schema_version") != 1):
            attempted = True
            outcome = str((wave or {}).get("aggregate") or "")
            if wave is not None and bool(wave.get("closed")):
                control = {"status": "closed", "outcome": outcome, "closed": True,
                           "fingerprint": str(wave.get("request_fingerprint") or "")}
            elif str(attempt.get("status") or "") == "rail_degraded":
                control = {"status": "rail_degraded", "reason": str(attempt.get("reason") or ""),
                           "outcome": outcome}
            elif wave is not None:
                # B2b typed fact: a wave whose own rows prove no re-dispatch can meet
                # quorum (structurally dead lanes) carries its earliest recorded reset.
                control = {
                    "status": "cycles_exhausted" if wave.get("cycles_exhausted") or (attempt.get("status") == "cycles_exhausted" and not wave.get("custody_pending")) else "open",
                    "outcome": outcome, "closed": False,
                    "fingerprint": str(wave.get("request_fingerprint") or ""),
                    "reviewer_slots_degraded": outcome == "DEGRADED", "custody_pending": bool(wave.get("custody_pending")),
                    **({"quorum_unreachable": True, "earliest_reset": str(wave.get("earliest_reset") or "")}
                       if wave.get("quorum_unreachable") else {}),
                }
            else:
                control = {"status": str(attempt.get("status") or "open"),
                           "reason": str(attempt.get("reason") or "")}
        elif legacy.get("status") not in (None, "", "absent"):
            attempted = True
            if legacy["status"] == "closed":
                control = {"status": "closed", "outcome": legacy["outcome"], "closed": True,
                           "fingerprint": legacy["fingerprint"], "legacy_v1": True}
            elif legacy["status"] == "rail_degraded":
                control = {"status": "rail_degraded", "reason": legacy["reason"],
                           "outcome": legacy["outcome"], "legacy_v1": True}
            elif legacy["status"] == "pending":
                control = {"status": "pending", "legacy_v1": True}
            else:
                control = {"status": LEGACY_OPEN_STATUS, "outcome": legacy["outcome"],
                           "closed": False, "fingerprint": legacy["fingerprint"], "legacy_v1": True,
                           "reason": "an open v1 plan wave cannot be honored: re-call plan_task"}
        elif state.get("waves"):
            control = {"status": "pending"}
        else:
            control = {"status": "absent"}
    else:
        control = {"status": "invalid"}

    subject = attempt.get("author_subject") or {}
    author = validate_author_disposition(subject.get("author_disposition"), subject_hash=str(attempt.get("fingerprint") or ""))
    if author and attempt.get("reason") in {"author_current_plan", "author_stop"}:
        # Follow historical criticism only here, never in current_plan_review_wave: its
        # closed_plan_review_wave consumer binds CURRENT acceptance authority. EVIDENCE
        # into a GAP only: it never moves this attempt's lifecycle or outcome (§6 why).
        critic = plan_review_wave(state, str(subject.get("review_fingerprint") or ""))
        if critic is not None and not control.get("outcome"):
            outcome = str(critic.get("aggregate") or "")
            control.update(outcome=outcome, historical_critic=True,
                           custody_pending=bool(critic.get("custody_pending")),
                           reviewer_slots_degraded=outcome == "DEGRADED")
    if author and author.get("action") == "stop":
        control.update(status="author_stopped", reason="author_stop", closed=False)
    status = str(control.get("status") or "unavailable")
    closed = bool(control.get("closed"))
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    cyber = not review_enforcement_blocks("blocking")
    if status == "closed" and closed:
        gate_status, allow = "closed", True
    elif status == "author_stopped":
        gate_status, allow = "author_stopped", True
    elif cyber:
        gate_status, allow = "advisory_open", True
    elif hard_rail or status == "rail_degraded":
        gate_status, allow = "rail_degraded", True
    elif policy == "advisory" and status in {
        "open", "unavailable", "cycles_exhausted", LEGACY_OPEN_STATUS,
    }:
        gate_status, allow = "advisory_open", True
    elif status == "cycles_exhausted":
        gate_status, allow = "cycles_exhausted", True
    elif status == "open" and control.get("quorum_unreachable"):
        # B2b: quorum structurally unreachable under blocking — finalization is
        # RELEASED so the agent MAY choose an honest blocked terminal (outcomes maps
        # it to blocked_with_evidence). The review stays open, implementation stays
        # held, and nothing here auto-finalizes: the gate merely stops refusing.
        # Staleness residual (disclosed): the stamped facts are as old as the wave
        # record — earliest_reset may already have passed by finalization time; the
        # agent SEES the reset instant below and can re-call plan_task to re-probe.
        gate_status, allow = "open", True
    else:
        gate_status, allow = status, False
    return {
        "enforcement": policy,
        **({"decision_authority": "cyber_pro", "review_status": status} if cyber else {}),
        "status": gate_status,
        "allow": allow,
        "attempted": attempted,
        "outcome": str(control.get("outcome") or ""),
        "historical_critic": bool(control.get("historical_critic")),  # whose verdict: label it, never this plan's own
        "closed": closed,
        "review_capacity_reason": "review_cycles_exhausted" if attempt.get("status") == "cycles_exhausted" else "",
        "reviewer_slots_degraded": bool(control.get("reviewer_slots_degraded")),
        "custody_pending": bool(control.get("custody_pending")),  # reviewers still working: read before aggregate
        "quorum_unreachable": bool(control.get("quorum_unreachable")),
        "earliest_reset": str(control.get("earliest_reset") or ""),
        "reason": str(hard_rail or control.get("reason") or ""),
        "cycles_paid": int((state or {}).get("cycles_paid") or 0) if isinstance(state, dict) else 0,
        "legacy_v1": bool(control.get("legacy_v1")),
        "source": "durable_state",
        "author_action": str(author.get("action") or "") if author else "",
    }


def closed_plan_review_wave(state: Any) -> Optional[Dict[str, Any]]:
    """The wave holding the CURRENT CLOSED plan authority, or None (pure read).

    v2: the current wave when ``closed``. A v1 record with a closed current review
    projects a minimal legacy wave (``{"acceptance_claims": [...], "legacy_v1": True}``)
    so ``effective_acceptance_claims``' v1 fallback still binds those claims."""
    if not isinstance(state, dict):
        return None
    wave = current_plan_review_wave(state) if state.get("schema_version") != 1 else None
    if wave is not None:
        return wave if bool(wave.get("closed")) else None
    legacy = _legacy_projection_of(state)
    if legacy.get("status") == "closed":
        return {"legacy_v1": True, "request_fingerprint": legacy["fingerprint"],
                "aggregate": legacy["outcome"], "closed": True,
                "acceptance_claims": list(legacy.get("acceptance_claims") or [])}
    return None


def record_plan_review_attempt(
    results_drive_root: Any,
    task_id: str,
    *,
    fingerprint: str,
    status: str = "open",
    reason: str = "",
    author_subject: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select one canonical plan fingerprint as current (open | unavailable | rail_degraded)."""
    if not _PLAN_REVIEW_HASH_RE.fullmatch(str(fingerprint or "")):
        raise ValueError("PLAN_REVIEW_STATE_INVALID: current attempt fingerprint is invalid")
    if status not in _PLAN_REVIEW_ATTEMPT_STATUSES:
        raise ValueError("PLAN_REVIEW_STATE_INVALID: current attempt status is invalid")

    def _record(state: Dict[str, Any]) -> Dict[str, Any]:
        state["current_attempt"] = {
            "fingerprint": fingerprint,
            "status": status,
            "reason": str(reason or "")[:_PLAN_REVIEW_REASON_MAX_CHARS],
        }
        if author_subject is not None:
            state["current_attempt"]["author_subject"] = copy.deepcopy(author_subject)
        return state

    return _update_plan_review_state(results_drive_root, task_id, _record)


def mark_current_plan_review_unavailable(
    results_drive_root: Any,
    task_id: str,
    *,
    reason: str,
) -> Dict[str, Any]:
    """Mark the current fingerprint retryable-unavailable (a failed/timed-out panel)."""

    def _mark(state: Dict[str, Any]) -> Dict[str, Any]:
        current = state.get("current_attempt")
        if isinstance(current, dict) and current.get("fingerprint"):
            current["status"] = "unavailable"
            current["reason"] = str(reason or "review_unavailable")[:_PLAN_REVIEW_REASON_MAX_CHARS]
        return state

    return _update_plan_review_state(results_drive_root, task_id, _mark)


def _fit_plan_review_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fit every hot-index write so paid cycles and closure/cap stamps can persist.

    A refused paid-state write could rebill the panel. Full specs stay source-backed;
    compact older waves, then tier-cut newest-wave text and request memory with markers.
    Preserve ids/classes/breaks/hashes/fingerprints, verdicts and paid state, plus
    legacy inline goal/acceptance claims. Mark cut legacy specs: original hashes
    cannot imply complete delta input; cut locators become named omissions.
    """
    def _size() -> int:
        return len(json.dumps(state, ensure_ascii=False, default=str).encode("utf-8"))

    # The fit target keeps headroom for the post-loop stamps (`spec_body_truncated`,
    # `request_memory_truncated`) and a later one-key writer (`cycles_exhausted`), so a state
    # that fits here still fits after them (delta review R10-2).
    fit_limit = _PLAN_REVIEW_STATE_MAX_BYTES - 512

    waves = list(state.get("waves") or [])
    for index in range(len(waves) - 1):
        if _size() <= fit_limit:
            break
        if not waves[index].get("compact"):
            waves[index] = _compact_plan_review_wave(waves[index])
            state["waves"] = waves
    if waves and _size() > fit_limit:
        newest = waves[-1]
        wave_before = json.dumps(newest, sort_keys=True, ensure_ascii=False, default=str)
        spec_before = json.dumps(newest.get("spec"), sort_keys=True, ensure_ascii=False, default=str)
        for cut in _PLAN_REVIEW_TRUNCATION_TIERS:
            _truncate_wave_texts(newest, cut)
            memory = [str(x) for x in state.get("need_evidence_seen") or []]
            if any(len(x) > cut for x in memory):
                state["need_evidence_seen"] = sorted(
                    {x[:cut] + _PLAN_REVIEW_TRUNCATION_MARKER if len(x) > cut else x for x in memory})
                state["request_memory_truncated"] = True
            if _size() <= fit_limit:
                break
        # The stamps are honest: set only for what the cut actually touched (delta review F10-2).
        if json.dumps(newest, sort_keys=True, ensure_ascii=False, default=str) != wave_before:
            newest["findings_texts_truncated"] = True
        if json.dumps(newest.get("spec"), sort_keys=True, ensure_ascii=False, default=str) != spec_before:
            newest["spec_body_truncated"] = True
    return state


def _update_plan_review_state(
    results_drive_root: Any,
    task_id: str,
    mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Strict locked update; unlike lifecycle writes, planning authority has no unlocked fallback."""
    path = task_result_path(results_drive_root, task_id)

    def _merge(existing: Dict[str, Any]) -> Dict[str, Any]:
        require_writable_task_result_schema(existing, path)
        state = _validated_plan_review_state(existing.get(PLAN_REVIEW_STATE_KEY))
        state.pop("legacy_v1_projection", None)  # derived on load, never persisted
        updated_state = _validated_plan_review_state(_fit_plan_review_state(mutator(state)))
        updated_state.pop("legacy_v1_projection", None)
        now = utc_now_iso()
        return stamp_task_result_schema({
            **existing,
            PLAN_REVIEW_STATE_KEY: updated_state,
            "task_id": task_id,
            "status": str(existing.get("status") or STATUS_RUNNING),
            "ts": str(existing.get("ts") or now),
            "updated_at": now,
        })

    try:
        updated = update_json_locked(path, _merge, strict_existing_dict=True)
    except ValueError as exc:
        if str(exc).startswith("update_json_locked:"):
            raise ValueError(
                "PLAN_REVIEW_STATE_INVALID: parent task result JSON is malformed"
            ) from exc
        raise
    return _validated_plan_review_state(updated.get(PLAN_REVIEW_STATE_KEY))


def _compact_plan_review_wave(wave: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.tools.plan_review_artifacts import compact_wave

    return compact_wave(wave)


def plan_review_authority_core(state: Dict[str, Any], *, source_ref: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Project plan authority through the exact-wave companion."""
    from ouroboros.tools.plan_review_artifacts import plan_review_authority_core as project

    return project(state, source_ref=source_ref)


_PLAN_REVIEW_TRUNCATION_MARKER = "…[truncated to fit the durable state]"
_PLAN_REVIEW_TRUNCATION_TIERS = (600, 200, 80, 40)
# Identity-bearing keys the last-resort cut never touches (authority, not prose).
_PLAN_REVIEW_IDENTITY_KEYS = frozenset({
    "id", "finding_id", "slot", "slot_id", "class", "breaks", "aggregate", "request_fingerprint",
    "previous_fingerprint", "spec_hash", "evidence_manifest_hash", "plan_prose_hash", "sha256",
    "model", "request_model", "route", "host_file_read_attestation", "reason", "decision", "kind",
    "goal", "acceptance_claims", "cycle_index", "series_id", "schema_version", "retry_key",
    "wave_artifact", "previous_wave_artifact", "spec_source_ref", "dialogue_source_ref", "dialogue_chat_id", "author_request_fingerprint",
    "historical_supplements",
})


def _truncate_wave_texts(node: Any, cut: int, *, key: str = "") -> Any:
    """Cut every free-text string leaf under ``node`` to ``cut`` chars (marker appended), in
    place for containers; identity keys are left whole. Returns the (possibly new) leaf."""
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if k in _PLAN_REVIEW_IDENTITY_KEYS:
                continue
            node[k] = _truncate_wave_texts(v, cut, key=k)
        return node
    if isinstance(node, list):
        for i, v in enumerate(node):
            node[i] = _truncate_wave_texts(v, cut, key=key)
        return node
    if isinstance(node, str) and len(node) > cut and not node.endswith(_PLAN_REVIEW_TRUNCATION_MARKER):
        return node[:cut] + _PLAN_REVIEW_TRUNCATION_MARKER
    if isinstance(node, str) and node.endswith(_PLAN_REVIEW_TRUNCATION_MARKER):
        body = node[: -len(_PLAN_REVIEW_TRUNCATION_MARKER)]
        return (body[:cut] + _PLAN_REVIEW_TRUNCATION_MARKER) if len(body) > cut else node
    return node


def record_plan_review_wave(
    results_drive_root: Any,
    task_id: str,
    wave: Dict[str, Any],
    *,
    need_evidence_seen: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Retain a reviewed wave and charge only physical dispatch, including failed panels.

    Free collection preserves a separately selected author plan. Existing paid
    cycles, full-wave history and exact source handles remain the authority.
    """
    fingerprint = str(wave.get("request_fingerprint") or "")
    if not _PLAN_REVIEW_HASH_RE.fullmatch(fingerprint):
        raise ValueError("PLAN_REVIEW_STATE_INVALID: wave fingerprint is invalid")

    def _record(state: Dict[str, Any]) -> Dict[str, Any]:
        selected = state.get("current_attempt") or {}
        retained_author = (selected.get("author_subject") or {}).get("review_fingerprint") == fingerprint
        previous = [w for w in state.get("waves") or [] if str(w.get("request_fingerprint") or "") == fingerprint]
        # D2, deliberately NARROWED by B2 (explicit wave-record authority change): only an
        # UNPAID wave — one in which NOTHING was physically dispatched (typed $0 skip rows
        # only) — preserves a PAID predecessor with the same fingerprint as the durable
        # authority (that predecessor holds the findings and rejections the promised delta
        # cycle rides on; a free attempt must not erase them). A PAID DEGRADED wave — the
        # panel WAS dispatched but returned no parseable quorum — is recordable as current
        # like any other paid wave: it replaces the predecessor and charges its cycle.
        # The degraded_retries counter therefore now counts only nothing-dispatched
        # attempts; the caller still renders the attempt it was handed.
        if not wave.get("paid") and not wave.get("custody_pending") and any(w.get("paid") for w in previous):
            for w in state.get("waves") or []:
                if str(w.get("request_fingerprint") or "") == fingerprint and w.get("paid"):
                    w["degraded_retries"] = int(w.get("degraded_retries") or 0) + 1
                    # An unpaid all-skip attempt can DISCOVER structural quorum
                    # unreachability the paid predecessor never saw. The typed fact
                    # must land on the DURABLE record the gate projects, or the tool
                    # answer (finalization released) and the gate (hold) contradict.
                    if wave.get("quorum_unreachable") and not w.get("closed"):
                        w["quorum_unreachable"] = True
                        w["structurally_dead_slots"] = list(wave.get("structurally_dead_slots") or [])
                        w["earliest_reset"] = str(wave.get("earliest_reset") or "")
            state["current_attempt"] = selected if retained_author else {"fingerprint": fingerprint, "status": "open", "reason": ""}
            if need_evidence_seen is not None:
                state["need_evidence_seen"] = sorted({str(s) for s in need_evidence_seen if str(s)})
            return state
        waves = [w for w in state.get("waves") or [] if str(w.get("request_fingerprint") or "") != fingerprint]
        recorded = copy.deepcopy(wave)
        history = [item for prior in previous for item in prior.get("historical_supplements") or []]
        if history:
            recorded["historical_supplements"] = copy.deepcopy(history)
            if any(prior.get("paid") and prior.get("cycle_index") == wave.get("cycle_index") for prior in previous):
                recorded["paid"] = True
        waves.append(recorded)
        if not state.get("series_id"):
            state["series_id"] = fingerprint[:16]
        # C-07: replacement writes don't charge again; a fully-rejected wave's
        # earned delta advances cycle_index and charges its new physical panel.
        already_paid = any(
            w.get("paid") and int(w.get("cycle_index") or 0) >= int(wave.get("cycle_index") or 0)
            for w in previous
        )
        if wave.get("paid") and not already_paid:
            state["cycles_paid"] = int(state.get("cycles_paid") or 0) + 1
        if need_evidence_seen is not None:
            state["need_evidence_seen"] = sorted({str(s) for s in need_evidence_seen if str(s)})
        full_from = max(0, len(waves) - _PLAN_REVIEW_FULL_WAVES)
        waves = [
            (_compact_plan_review_wave(w) if idx < full_from and not w.get("compact") and not w.get("custody_pending") else w)
            for idx, w in enumerate(waves)
        ]
        overflow = max(0, len(waves) - _PLAN_REVIEW_MAX_WAVES)
        if overflow:  # the newest PAID wave stays reachable: the next dispatch judges against it
            keep = next((i for i in range(len(waves) - 1, -1, -1) if waves[i].get("paid")), None)
            dropped = set([i for i in range(len(waves)) if i != keep][:overflow])
            state["waves_omitted"] = int(state.get("waves_omitted") or 0) + len(dropped)
            waves = [w for i, w in enumerate(waves) if i not in dropped]
        # I-02: size-fitting (older-wave compaction, then the last-resort text cut) runs for
        # EVERY writer in `_update_plan_review_state` → `_fit_plan_review_state`.
        state["waves"] = waves
        state["current_attempt"] = selected if retained_author else {"fingerprint": fingerprint, "status": "open", "reason": ""}
        return state

    state = _update_plan_review_state(results_drive_root, task_id, _record)
    return plan_review_wave(state, fingerprint) or {}


def plan_review_notes_are_annotatable(wave: Dict[str, Any]) -> bool:
    """Optional notes remain discussable after automatic closure, not new authority.

    A note-only wave is recorded GREEN (notes never change the verdict); older
    records carry it as a closed REVIEW_REQUIRED, and both stay annotatable."""
    findings = wave.get("findings") or []
    return bool(findings) and wave.get("aggregate") in {"GREEN", "REVIEW_REQUIRED"} and all(
        finding.get("class") == "note" for finding in findings
    )


def record_plan_review_dispositions(
    results_drive_root: Any,
    task_id: str,
    *,
    fingerprint: str,
    dispositions: List[Dict[str, Any]],
    closed: bool,
    aggregate: str = "",
    closure_notes: Optional[List[str]] = None,
    wave_artifact: Optional[Dict[str, Any]] = None,
    recorded_at: str = "",
    author_disposition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Store the agent's dispositions on one FULL wave and its resulting closure.
    Only note-only closed waves accept annotations. Closure authority remains
    ``plan_spec.closure_after_disposition`` (``aggregate`` is the verdict that
    table says to record — GREEN when a REVIEW_REQUIRED open set emptied); this
    writer is rule-free. Other closed waves are immutable."""

    def _record(state: Dict[str, Any]) -> Dict[str, Any]:
        wave = next((w for w in state["waves"] if str(w.get("request_fingerprint") or "") == fingerprint), None)
        if wave is None or wave.get("compact"):
            raise ValueError("PLAN_REVIEW_DISPOSITION_UNBINDABLE: no full wave holds this fingerprint")
        current = state.get("current_attempt") or {}
        if current.get("fingerprint") and current["fingerprint"] != fingerprint:
            raise ValueError("PLAN_REVIEW_DISPOSITION_STALE: a newer attempt supersedes this wave")
        if wave.get("closed") and not plan_review_notes_are_annotatable(wave):
            raise ValueError("PLAN_REVIEW_DISPOSITION_IMMUTABLE: a closed wave cannot be changed")
        wave["dispositions"] = copy.deepcopy(list(dispositions))
        wave["disposition_recorded_at"] = recorded_at or utc_now_iso()
        if closure_notes is not None:
            wave["closure_notes"] = list(closure_notes)
        if wave_artifact is not None:
            wave["wave_artifact"] = copy.deepcopy(wave_artifact)
        if author_disposition is not None:
            author = validate_author_disposition(
                author_disposition,
                subject_hash=fingerprint,
            )
            if author is None:
                raise ValueError("PLAN_REVIEW_AUTHOR_DISPOSITION_INVALID: stale or malformed record")
            wave["author_disposition"] = author
        if closed and str(wave.get("aggregate") or "") == "REVIEW_REQUIRED":
            wave.update(closed=True, aggregate=aggregate or wave["aggregate"])
        state["current_attempt"] = {"fingerprint": fingerprint, "status": "open", "reason": ""}
        return state

    state = _update_plan_review_state(results_drive_root, task_id, _record)
    return plan_review_wave(state, fingerprint) or {}

def mark_plan_review_cycles_exhausted(
    results_drive_root: Any, task_id: str, *, fingerprint: str,
) -> Dict[str, Any]:
    """Stamp the current open wave ``cycles_exhausted`` (typed hold state; no panel ran)."""

    def _mark(state: Dict[str, Any]) -> Dict[str, Any]:
        wave = next((w for w in state["waves"] if str(w.get("request_fingerprint") or "") == fingerprint), None)
        if wave is not None and not wave.get("closed"):
            wave["cycles_exhausted"] = True
        return state

    state = _update_plan_review_state(results_drive_root, task_id, _mark)
    return plan_review_wave(state, fingerprint) or {}
