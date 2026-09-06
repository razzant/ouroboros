"""The one SSOT projection of task cost for every producer surface (C2, owner 10=B).

``accounted_upper_bound_usd`` is the honest NAME for what ``cost_usd`` always was:
the settled + reserved + unresolved upper bound from the physical attempt ledger,
not a settled receipt. The semantics do not change (owner 10=B); the name does.
ABI 7.0 (ABI-3, the approved ABI break): the deprecated ``cost_usd`` /
``cost_usd_with_children`` aliases are no longer EMITTED anywhere — the write
seams strip them — while the READ side keeps resolving both spellings so every
stored legacy record (task results, chat.jsonl rows, event snapshots) stays
readable, with the historical deprecated-wins precedence for diverged pairs.

Null is null: a missing or unknown cost projects ``None`` on BOTH names and must
never render as ``$0.00``, and finality is never fabricated — ``cost_final`` is
True only when the SOURCE says so about a known amount.

Producers do not hand-assemble these fields. They pass their source mapping
through :func:`cost_projection` (read-side projections: peek/recent/handoff/wait)
or :func:`with_cost_aliases` (write-side field dicts that already carry the full
meta set) so every surface tells the same story with the same names.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, Mapping, Optional

log = logging.getLogger(__name__)

# (honest name, retired legacy spelling). Since ABI 7.0 the legacy spelling is
# READ-ONLY tolerance for stored records — emitters never write it again.
COST_ALIAS_PAIRS = (
    ("accounted_upper_bound_usd", "cost_usd"),
    ("accounted_upper_bound_usd_with_children", "cost_usd_with_children"),
)

# The accounting OPENNESS / INTEGRITY markers that must ride BESIDE an amount
# whenever the source carries them: an upper bound without them reads like a
# receipt. ONE list, here — producers import it instead of hand-building a
# closed whitelist each (three of those had silently dropped `reserved_usd`,
# `unresolved_upper_bound_usd` and the ledger-integrity marker, so a frame said
# "not final" with nothing on it explaining what was still open). Adding a new
# marker means adding it here, once.
COST_OPENNESS_FIELDS = (
    "cost_accounting_status",
    "cost_accounting_error",
    "cost_final",
    "cost_with_children_partial",
    "unknown_unmetered",
    "non_final_rows",
    "reserved_usd",
    "unresolved_upper_bound_usd",
    "ledger_integrity_degraded",
)


def _as_amount(value: Any) -> Optional[float]:
    """A dollar amount, or None. Booleans and unparseable values are None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def honest_accounted_amount(source: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Return a known accounted subtotal without fabricating unknown zero.

    The physical ledger uses ``accounted_usd`` as the settled plus reserved /
    unresolved upper bound. An external or unknown-price row can therefore
    leave that subtotal at ``0.0`` while ``unknown_unmetered`` is positive.
    That zero is not a measured free result unless a bound exists, so terminal
    and live producers must carry ``None`` instead. A non-zero known subtotal,
    including a mixed known+unknown tree, remains useful and is disclosed by
    the openness fields beside it.
    """
    src = source if isinstance(source, Mapping) else {}
    amount = _as_amount(src.get("accounted_usd"))
    if amount is None:
        return None
    try:
        unknown = int(src.get("unknown_unmetered") or 0)
        reserved = float(src.get("reserved_usd") or 0)
        unresolved = float(src.get("unresolved_upper_bound_usd") or 0)
    except (TypeError, ValueError):
        return amount
    if amount == 0 and unknown > 0 and reserved == 0 and unresolved == 0:
        return None
    return amount


def resolve_cost_pair(source: Optional[Mapping[str, Any]], new: str, old: str) -> tuple[bool, Any]:
    """``(present, raw_value)`` for one alias pair under the ONE precedence rule.

    THE DEPRECATED NAME WINS when both are present. Every seam — read and write,
    Python and JS — asks this question here, because two seams answering it
    differently is how a diverged pair starts telling two stories about the same
    record: the write side re-converged on ``cost_usd`` while the read side
    displayed ``accounted_upper_bound_usd``. Deprecated-wins is the direction the
    write side must take anyway (legacy mutators between two seam crossings edit
    ``cost_usd``), so reading the same way keeps a record self-consistent no
    matter which producer last touched it.
    """
    src = source if isinstance(source, Mapping) else {}
    if old in src:
        return True, src[old]
    if new in src:
        return True, src[new]
    return False, None


def honest_cost_pair_amount(
    source: Optional[Mapping[str, Any]], new: str, old: str,
) -> tuple[bool, Optional[float]]:
    """Resolve one cost pair and apply the shared unknown-zero rule."""
    src = source if isinstance(source, Mapping) else {}
    present, raw = resolve_cost_pair(src, new, old)
    if not present:
        return False, None
    probe = dict(src)
    probe["accounted_usd"] = raw
    return True, honest_accounted_amount(probe)


def with_cost_aliases(fields: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """A COPY of ``fields`` normalized onto the honest cost names (ABI 7.0).

    Write-side seam: a producer that already assembled the full cost-meta dict
    (``reconstruct_task_cost``, the terminal frame builder) passes through here
    — and it must be the LAST step after any cost mutation. A legacy spelling
    present on the input still WINS resolution (:func:`resolve_cost_pair`,
    deprecated wins — a legacy mutator's edit is honored) but is STRIPPED from
    the output: since ABI-3 nothing emits or persists the retired alias keys.
    A pair absent on both sides STAYS absent — normalization never invents a
    field, and an explicit None stays None. Idempotent.
    """
    out = dict(fields or {})
    for new, old in COST_ALIAS_PAIRS:
        present, value = honest_cost_pair_amount(out, new, old)
        out.pop(old, None)
        if present:
            out[new] = value
    return out


def normalize_task_result_cost_planes(fields: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``with_cost_aliases`` over a task-result row AND its nested public cost planes.

    ABI-3 (fix-round-3): the projection seam (``public_task_result``) and the
    rewrite seam (``write_task_result``) share this ONE normalizer, so the
    stored row and the outbound payload cannot diverge on WHICH planes are
    normalized. The nested planes are the known public ones — the subagent
    envelope (its own amount AND its ``usage`` snapshot, the actually
    supported producer path) and the loop-outcome ``usage`` dict. Internal
    evidence planes that merely share the spelling (review receipts, ledger
    rows) are their own schemas and pass through untouched — the emission
    sweep allowlist names them per-site, and the projection-boundary tests
    deep-scan the public planes instead.
    """
    out = with_cost_aliases(fields)
    envelope = out.get("subagent_envelope")
    if isinstance(envelope, dict):
        envelope = with_cost_aliases(envelope)
        if isinstance(envelope.get("usage"), dict):
            envelope["usage"] = with_cost_aliases(envelope["usage"])
        out["subagent_envelope"] = envelope
    loop_outcome = out.get("loop_outcome")
    if isinstance(loop_outcome, dict) and isinstance(loop_outcome.get("usage"), dict):
        out["loop_outcome"] = {
            **loop_outcome, "usage": with_cost_aliases(loop_outcome["usage"]),
        }
    return out


def carry_cost_meta(source: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The honest cost names plus every openness/integrity marker the source carries.

    For a producer assembling a NEW frame out of an existing payload: it copies
    what accounting actually said instead of re-listing field names by hand.
    A legacy alias spelling on the SOURCE still resolves (deprecated wins) but
    only the honest name is emitted (ABI 7.0). Absent stays absent; unknown
    extra keys are the caller's business (this returns only accounting fields,
    so a caller merging it never loses its own).
    """
    src = source if isinstance(source, Mapping) else {}
    out: Dict[str, Any] = {}
    for new, old in COST_ALIAS_PAIRS:
        present, value = honest_cost_pair_amount(src, new, old)
        if present:
            out[new] = value
    for key in COST_OPENNESS_FIELDS:
        if key in src:
            out[key] = src[key]
    return out


def live_root_cost_projection(
    task_id: str, task: Mapping[str, Any], event: Mapping[str, Any], drive_root: pathlib.Path,
) -> Dict[str, Any]:
    """Return a non-final root-subtree projection for the existing heartbeat."""
    from ouroboros.task_results import resolve_task_lineage

    lineage = resolve_task_lineage(
        task_id, metadata=task.get("metadata"),
        **{key: task.get(key, event.get(key)) for key in (
            "root_task_id", "parent_task_id", "delegation_role",
            "original_task_id", "timeout_retry_from",
        )},
    )
    if not lineage["is_root_task"]:
        return {}
    try:
        from ouroboros.usage_accounting import usage_projection

        usage = usage_projection(
            pathlib.Path(task.get("budget_drive_root") or drive_root),
            root_task_id=str(lineage["root_task_id"] or task_id),
        )
        # A root-filtered summary with no attributable rows is an empty view,
        # not measured zero.  The task-detail reader applies the same rule:
        # legacy metadata alone and an empty subtree must not publish a money
        # amount whose source never observed this root.
        counts = usage.get("attempt_counts")
        if isinstance(counts, Mapping):
            attributable_rows = sum(
                max(0, int(value or 0))
                for key, value in counts.items()
                if key != "metadata_only"
            )
            sessions = max(0, int(usage.get("subscription_sessions") or 0))
            if attributable_rows == 0 and sessions == 0:
                return {}
        unknown = int(usage.get("unknown_unmetered") or 0)
        reserved = float(usage.get("reserved_usd") or 0)
        unresolved = float(usage.get("unresolved_upper_bound_usd") or 0)
        accounted = honest_accounted_amount(usage)
        projection = {
            "cost_accounting_status": "available",
            "accounted_upper_bound_usd_with_children": (
                round(accounted, 6) if accounted is not None else None
            ),
            "reserved_usd": reserved,
            "unresolved_upper_bound_usd": unresolved,
            "unknown_unmetered": unknown,
            "non_final_rows": int(usage.get("non_final_rows") or 0),
            "ledger_integrity_degraded": bool(usage.get("integrity_degraded")),
        }
    except Exception:
        log.debug("Root heartbeat cost unavailable for %s", task_id, exc_info=True)
        projection = {
            "cost_accounting_status": "unavailable",
            "cost_accounting_error": "ledger_unavailable",
            "accounted_upper_bound_usd_with_children": None,
        }
    projection.update(cost_final=False, cost_with_children_partial=True)
    return with_cost_aliases(projection)


def cost_projection(source: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Read-side projection of a task-result-like mapping into the SSOT shape.

    Emits ``accounted_upper_bound_usd`` (None when unknown; a stored legacy
    ``cost_usd`` spelling still resolves, deprecated wins — ABI 7.0 read
    tolerance), ``cost_known`` (whether ANY amount is accounted),
    ``cost_final`` (source-claimed finality about a KNOWN amount, never
    fabricated), the with-children name when the source carries the pair, and
    every openness flag the source has.
    """
    src = source if isinstance(source, Mapping) else {}
    _, amount = honest_cost_pair_amount(src, *COST_ALIAS_PAIRS[0])
    out: Dict[str, Any] = {
        "accounted_upper_bound_usd": amount,
        "cost_known": amount is not None,
        "cost_final": bool(src.get("cost_final")) and amount is not None,
    }
    present, with_children = honest_cost_pair_amount(src, *COST_ALIAS_PAIRS[1])
    if present:
        out["accounted_upper_bound_usd_with_children"] = with_children
    for key in COST_OPENNESS_FIELDS:
        if key in src and key not in out:
            out[key] = src[key]
    return out


def cost_display(source: Optional[Mapping[str, Any]], *, decimals: int = 2) -> str:
    """Human rendering of one task's cost. Null NEVER renders as ``$0.00``.

    ``$1.23`` for a final amount, ``$1.23 (upper bound, not final)`` for an open
    one, ``unknown (accounting unavailable or not settled)`` when nothing is
    accounted.
    """
    projection = cost_projection(source)
    amount = projection["accounted_upper_bound_usd"]
    if amount is None:
        return "unknown (accounting unavailable or not settled)"
    text = f"${amount:.{decimals}f}"
    if projection["cost_final"]:
        return text
    return f"{text} (upper bound, not final)"


__all__ = [
    "COST_ALIAS_PAIRS",
    "COST_OPENNESS_FIELDS",
    "carry_cost_meta",
    "cost_display",
    "cost_projection",
    "honest_cost_pair_amount",
    "honest_accounted_amount",
    "live_root_cost_projection",
    "normalize_task_result_cost_planes",
    "resolve_cost_pair",
    "with_cost_aliases",
]
