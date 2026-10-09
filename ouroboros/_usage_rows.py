"""Pure row math over physical-attempt rows: the ONE money reducer.

``summary_delta`` is one row's contribution to every summary member; a
``Bucket`` holds the running sum (reversible: a transition subtracts the old
row and adds the new one). The usage store maintains its ``summaries`` with
these deltas and ``_summary``/``_breakdown_bucket`` fold them over addressed
rows, so both readings are one implementation. Also: the limit/integrity
decorations, the per-row physical-call count, the earliest-binding index the
journal import folds, and the per-wave Skill Review projection. A LEAF over
plain row dicts — no file I/O, no locks; ``usage_accounting`` re-exports the
historical names.
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple
from decimal import Decimal, InvalidOperation

from ouroboros.usage_ledger import _number
from ouroboros._usage_money import billing_group_key, monetary_scope_key, ZERO_CASH, cash_contribution, render_cash, exact_money, decimal_of

REVIEW_ATTRIBUTION_KEYS = ("review_skill", "review_wave_id", "review_slot_id")
# A row under a review's own custody: every review-substrate send carries its reviewer slot and a
# skill review its skill. A wave alone marks a review that runs its executor directly (advisory,
# deep self-review); its rows stay with the generic reconciliation and cost refresh (#1544).
REVIEW_CUSTODY_KEYS = ("review_skill", "review_slot_id")

# Earliest cap/attribution binding per root and billing group (usage_admission).
BINDING_KEYS = ("root_task_id", "billing_group_id", "billing_group_limit_usd", "billing_group_limit_source",
                "billing_group_limit_revision", "root_limit_usd", "root_limit_source", "root_limit_revision")
_BINDING_CAPS = ("billing_group_limit_usd", "root_limit_usd")
BINDING_AUTHORITY_FIELD, BINDING_CARRIED = "binding_authority", "carried"  # baseline header stamp
CARRIED_ROOT_BINDING, CARRIED_GROUP_BINDING = "original_root_binding", "original_group_binding"
# Carried value for an identity the SOURCE had not bound yet: it stays unbound,
# so a later original row still binds it exactly as it would have uncompacted.
NO_ORIGINAL_BINDING = "unbound"
# Source label of a binding taken from a block row's own cap literal: an older
# or unstamped block that carried nothing usable for that member.
LEGACY_LIVE_SOURCE = "legacy_live"


def _carried(value: Any, key: str, owner) -> Optional[dict]:
    """A carried binding that is well formed and belongs to ``key``; else ``None``."""
    if (not isinstance(value, dict) or not set(value) <= set(BINDING_KEYS) or owner(value) != key
            or not any(cap in value for cap in _BINDING_CAPS)):
        return None
    for cap in _BINDING_CAPS:
        item = value.get(cap)
        if item is not None and (_number(item) is None or not math.isfinite(_number(item))):
            return None
    return dict(value)


_GROUP_AXIS_KEYS = ("billing_group_id", "billing_group_limit_usd", "billing_group_limit_source",
                    "billing_group_limit_revision")


def _block_literal(row: Dict[str, Any], keys: tuple) -> Optional[tuple]:
    """The binding fields a block row carries on one axis, caps normalized so rows can be
    compared; ``None`` when it carries no cap on that axis."""
    present = tuple((k, None if row[k] is None else _number(row[k]) if k in _BINDING_CAPS else row[k])
                    for k in keys if k in row)
    return present if any(k in _BINDING_CAPS for k, _ in present) else None


def _axis_keys(row: Dict[str, Any], group_axis: bool) -> tuple:
    """Which fields must agree across a member's block rows: a group's own cap fields (the member
    roots' caps differ by right), else everything the row carries."""
    if not group_axis:
        return BINDING_KEYS
    return _GROUP_AXIS_KEYS if "billing_group_limit_usd" in row else ("root_task_id", "root_limit_usd")


@dataclass
class BindingIndex:
    """Earliest binding per root and billing group: the first ORIGINAL row wins.

    A baseline aggregate's cap is a minimum and its position a sort order, so a
    carried block (``binding_authority=carried``) restores the source's bindings
    verbatim from the first group row of each root/group; an explicit
    ``NO_ORIGINAL_BINDING`` there leaves that member open for a later original
    row. Any other block row — an unstamped or ``unknown`` header, a missing,
    ``unknown`` or malformed carriage — binds its member from the row's OWN cap
    literal (``legacy_live``) while every block row of that member that carries
    a cap carries the same one; a block row without a cap carries no literal and
    does not vote (so neither row order nor the moment of compaction changes
    the answer). A member whose block rows carry different literals stays
    unbound, and admission applies the configured cap, disclosed on the task
    (``legacy_default``, usage_admission). No archive is ever read. Equality
    compares the indexes.
    """

    roots: Dict[str, Any] = field(default_factory=dict)
    groups: Dict[str, Any] = field(default_factory=dict)
    carried: bool = field(default=False, compare=False)
    unbound: set = field(default_factory=set, compare=False)  # (carrier, key) the block left open
    live: dict = field(default_factory=dict, compare=False)  # (carrier, key) -> block literal; None once disputed

    def fold(self, row: Dict[str, Any]) -> None:
        kind = str(row.get("kind") or "")
        if kind == "usage_baseline":
            self.carried = row.get(BINDING_AUTHORITY_FIELD) == BINDING_CARRIED
            return
        root, group = monetary_scope_key(row), billing_group_key(row)
        if kind == "usage_baseline_group":
            for index, key, name, owner, keys in (
                    (self.roots, root, CARRIED_ROOT_BINDING, monetary_scope_key, _axis_keys(row, False)),
                    (self.groups, group, CARRIED_GROUP_BINDING, billing_group_key, _axis_keys(row, True))):
                if not key or (name, key) in self.unbound:
                    continue
                if (name, key) in self.live:  # bound from a block literal: every capped block row must agree
                    literal = _block_literal(row, keys)
                    if literal is not None and self.live[name, key] is not None and literal != self.live[name, key]:
                        index.pop(key, None)
                        self.live[name, key] = None
                    continue
                if key in index:
                    continue  # the first block row of each member decides it
                carriage = row.get(name) if self.carried else None
                if carriage == NO_ORIGINAL_BINDING:
                    self.unbound.add((name, key))
                    continue
                bound = _carried(carriage, key, owner)
                if bound is None:  # nothing usable was carried: the row's own literal (legacy_live)
                    literal = _block_literal(row, keys)
                    if literal is None:
                        continue  # no cap on this row: no literal, no vote
                    self.live[name, key] = literal
                    bound = {k: row[k] for k in BINDING_KEYS if k in row}
                    bound["billing_group_limit_source"] = LEGACY_LIVE_SOURCE  # disclosed as a block literal
                index[key] = bound
            return
        if ((root and root not in self.roots) or (group and group not in self.groups)) and any(
                cap in row for cap in _BINDING_CAPS):
            binding = {key: row[key] for key in BINDING_KEYS if key in row}  # verbatim, key presence kept
            for index, key in ((self.roots, root), (self.groups, group)):
                if key:
                    index.setdefault(key, binding)


def row_ts_epoch(row: Any) -> Optional[float]:
    """A ledger row's ``ts`` (UTC ISO, the appender's stamp) as an epoch second;
    ``None`` when absent or unparseable — a reader must not guess a time."""
    text = str(row.get("ts") or "").strip() if isinstance(row, dict) else ""
    if not text:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.timestamp()


def _merge_processing_summary(total: dict, addition: dict) -> None:
    """Sum existing evidence components; preserve absent amounts and exact literals."""
    for key in ("valuation_usd", "unclassified_usd"):
        value = addition.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            amount = decimal_of(value)
        except (InvalidOperation, ValueError):
            continue
        if amount.is_finite() and amount >= 0:
            with exact_money():
                total[key] = total.get(key, Decimal(0)) + amount
    for key in ("unknown_cash_rows", "unknown_valuation_rows"):
        if key in addition:
            total[key] = total.get(key, 0) + int(addition[key])
    for key in ("observed_modes", "billing_kinds"):
        for label, count in (addition.get(key) or {}).items():
            bucket = total.setdefault(key, {})
            bucket[label] = bucket.get(label, 0) + int(count)


def _processing_summary(rows: Sequence[Dict[str, Any]], *, decimal_values: bool = False) -> dict:
    """Valuation and unclassified amounts are disclosures, never added to cash."""
    total: dict = {}
    for row in rows:
        if isinstance(row.get("processing_summary"), dict):
            _merge_processing_summary(total, row["processing_summary"])
            continue
        attempts = row.get("attempt_execution")
        evidence = row.get("cost_evidence") or {}
        entries = attempts if isinstance(attempts, list) else [{
            "processing": row.get("processing"),
            "processingCostBasis": evidence.get("processing") or row.get("processing_basis"),
            "usageCost": {"valuationUsd": evidence.get("valuationUsd"),
                          "valuationKnowledge": evidence.get("valuationKnowledge", "unknown"),
                          "cashUsd": evidence.get("cashUsd") if evidence.get("cashUsd") is not None else evidence.get("estimatedUsd"),
                          "cashKnowledge": evidence.get("knowledge", "unknown")} if evidence else {},
        }]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            receipt = entry.get("processing") or {}
            basis = entry.get("processingCostBasis") or {}
            cost = entry.get("usageCost") or {}
            facts = {}
            if receipt.get("observed"):
                facts["observed_modes"] = {receipt["observed"]: 1}
            if basis.get("kind"):
                facts["billing_kinds"] = {basis["kind"]: 1}
            if cost:
                facts["unknown_cash_rows"] = int(cost.get("cashKnowledge") not in {"exact", "estimated"} or cost.get("cashUsd") is None)
                known_valuation = cost.get("valuationKnowledge") in {"exact", "estimated"}
                facts["unknown_valuation_rows"] = int(not known_valuation or cost.get("valuationUsd") is None)
                facts["valuation_usd"] = cost.get("valuationUsd") if known_valuation else None
                facts["unclassified_usd"] = cost.get("unknownUsd")
            _merge_processing_summary(total, facts)
    if total:
        for key in ("valuation_usd", "unclassified_usd"):
            total.setdefault(key, None)
    return {key: (value if decimal_values or not isinstance(value, Decimal) else float(value))
            for key, value in total.items()}


_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens")
_PROCESSING_AMOUNTS = ("valuation_usd", "unclassified_usd")
_PROCESSING_COUNTERS = ("unknown_cash_rows", "unknown_valuation_rows")
_PROCESSING_LABELS = ("observed_modes", "billing_kinds")


@dataclass
class Bucket:
    """The running reducer state of one set of final rows (the usage store's
    ``summaries`` row). Every member is the exact sum of per-row contributions
    (``summary_delta``), so a transition subtracts the old row and adds the new
    one. Presence travels with each sum (``[sum, rows that reported it]``): a
    token axis nobody reported stays ``None``, never a measured zero. The cap
    multiset (``caps``: decimal literal -> rows) and the subscription windows
    (max reset per route; one-shot rows never leave a bucket) are the two
    non-additive members."""

    rows: int = 0
    cash: tuple = ZERO_CASH
    counts: Dict[str, int] = field(default_factory=dict)
    unknown: int = 0
    priced: int = 0
    tracked_nonfinal: int = 0
    accounting_open: int = 0
    non_final: int = 0
    sessions: int = 0
    windows: Dict[str, str] = field(default_factory=dict)
    processing: Dict[str, Any] = field(default_factory=dict)
    tokens: Dict[str, list] = field(default_factory=dict)
    physical: int = 0
    ttls: Dict[str, list] = field(default_factory=dict)
    caps: Dict[str, int] = field(default_factory=dict)

    def add(self, other: "Bucket", sign: int = 1, *, cap: Optional[str] = None) -> "Bucket":
        """Add (``sign=1``) or subtract (``sign=-1``) ``other``; ``cap`` is the cap
        literal the row carries on THIS bucket's axis (root or group scope)."""
        self.rows += sign * other.rows
        with exact_money():
            self.cash = tuple(a + sign * b for a, b in zip(self.cash, other.cash))
        for name in ("unknown", "priced", "tracked_nonfinal", "accounting_open", "non_final", "sessions", "physical"):
            setattr(self, name, getattr(self, name) + sign * getattr(other, name))
        _add_counts(self.counts, other.counts, sign)
        for route, reset_at in other.windows.items():
            if sign > 0:
                self.windows[route] = max(self.windows.get(route, ""), reset_at)
        _add_pairs(self.tokens, other.tokens, sign)
        _add_pairs(self.ttls, other.ttls, sign)
        for key in (*_PROCESSING_AMOUNTS, *_PROCESSING_COUNTERS):
            if key in other.processing:
                _add_pairs(self.processing, {key: other.processing[key]}, sign)
        for key in _PROCESSING_LABELS:
            if key in other.processing:
                labels = self.processing.setdefault(key, {})
                _add_pairs(labels, other.processing[key], sign)
                if not labels:
                    self.processing.pop(key)
        if cap is not None:
            _add_counts(self.caps, {cap: 1}, sign)
        return self

    def min_cap(self) -> Optional[float]:
        known = [decimal_of(literal) for literal, count in self.caps.items() if count > 0]
        return float(min(known)) if known else None

    def render_summary(self) -> Dict[str, Any]:
        """Exactly ``_summary``'s shape for the rows this bucket holds."""
        result = {
            **render_cash(self.cash),
            "unknown_unmetered": self.unknown,
            "priced_rows": self.priced,
            "tracked_nonfinal_rows": self.tracked_nonfinal,
            "accounting_open_rows": self.accounting_open,
            # Every row that increments `unknown` is open, so it is subsumed here.
            "non_final_rows": self.non_final,
            "cost_final": not self.non_final,
            "attempt_counts": {state: count for state, count in self.counts.items() if count},
            "subscription_sessions": self.sessions,
            "subscription_windows": dict(self.windows),
        }
        processing = self._render_processing()
        if processing:
            result["processing_summary"] = processing
        return result

    def render_breakdown(self) -> Dict[str, Any]:
        """Exactly ``_breakdown_bucket``'s shape for the rows this bucket holds."""
        def summed(name: str) -> Optional[int]:
            total, present = self.tokens.get(name) or (0, 0)
            return int(total) if present > 0 else None

        prompt, completion = summed("prompt_tokens"), summed("completion_tokens")
        return {
            **self.render_summary(),
            "physical_calls": self.physical,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            # Unknown on BOTH halves is an unknown total; one known half is a real total.
            "total_tokens": None if prompt is None and completion is None else (prompt or 0) + (completion or 0),
            "cached_tokens": summed("cached_tokens"),
            "cache_write_tokens": summed("cache_write_tokens"),
            "prompt_cache_ttls": {ttl: int(total) for ttl, (total, present) in self.ttls.items() if present > 0},
        }

    def _render_processing(self) -> dict:
        total: dict = {}
        for key in (*_PROCESSING_AMOUNTS, *_PROCESSING_COUNTERS):
            value, present = self.processing.get(key) or (0, 0)
            if present > 0:
                total[key] = float(value) if key in _PROCESSING_AMOUNTS else int(value)
        for key in _PROCESSING_LABELS:
            labels = {label: int(count) for label, (count, present) in (self.processing.get(key) or {}).items()
                      if present > 0}
            if labels:
                total[key] = labels
        if total:
            for key in _PROCESSING_AMOUNTS:
                total.setdefault(key, None)
        return total


def _add_counts(target: Dict[str, int], addition: Dict[str, int], sign: int) -> None:
    for key, value in addition.items():
        updated = target.get(key, 0) + sign * value
        if updated:
            target[key] = updated
        else:
            target.pop(key, None)


def _add_pairs(target: Dict[str, list], addition: Dict[str, list], sign: int) -> None:
    """``[sum, rows that reported it]`` pairs; a pair no row reports any more leaves."""
    for key, (value, present) in addition.items():
        old_value, old_present = target.get(key) or (0, 0)
        with exact_money():
            pair = [old_value + sign * value, old_present + sign * present]
        if pair[1]:
            target[key] = pair
        else:
            target.pop(key, None)


def _processing_delta(row: Dict[str, Any]) -> Dict[str, Any]:
    """One row's processing disclosure as ``[sum, 1]`` pairs (the merge rules
    of ``_processing_summary``, so presence matches the list-based fold)."""
    total = _processing_summary([row], decimal_values=True)
    delta: Dict[str, Any] = {}
    for key in _PROCESSING_AMOUNTS:
        if total.get(key) is not None:
            delta[key] = [decimal_of(total[key]), 1]
    for key in _PROCESSING_COUNTERS:
        if key in total:
            delta[key] = [int(total[key]), 1]
    for key in _PROCESSING_LABELS:
        labels = total.get(key) or {}
        if labels:
            delta[key] = {label: [int(count), 1] for label, count in labels.items()}
    return delta


def summary_delta(row: Dict[str, Any]) -> Bucket:
    """The contribution of ONE final row to every running summary member.

    The ``_summary`` branch rules, ``cash_contribution``, ``_physical_call_count``
    and ``_processing_summary`` merge rules, applied to one row. A baseline
    group row (an imported aggregate) carries ``folded_attempt_count`` rows'
    PRE-SUMMED cash/tokens, so sums add once and count axes add the weight;
    every other row has weight 1. ``_summary``/``_breakdown_bucket`` fold these
    deltas, and the usage store maintains its summaries with the same deltas.
    """
    delta = Bucket(rows=1, cash=cash_contribution(row))
    kind = str(row.get("kind") or "")
    state = str(row.get("state") or "")
    delta.processing = _processing_delta(row)
    delta.physical = _physical_call_count(row)
    ttl = str(row.get("prompt_cache_ttl") or "").strip()
    if ttl:
        delta.ttls[ttl] = [delta.physical, 1]
    for name in _TOKEN_FIELDS:
        if row.get(name) is not None:
            delta.tokens[name] = [max(0, int(row[name])), 1]
    if kind == "usage_baseline":
        # Compaction stamp header: pure provenance, no money, no counts.
        return delta
    weight = max(1, int(row.get("folded_attempt_count") or 1)) if kind == "usage_baseline_group" else 1
    if kind == "subscription_session":
        delta.sessions = 1
        route = str(row.get("subscription_route") or "")
        reset_at = str(row.get("subscription_reset_at") or "")
        if route and reset_at:
            delta.windows[route] = reset_at
    if kind == "legacy_metadata":
        delta.counts["metadata_only"] = max(1, int(row.get("ambiguous_call_count") or 1))
        return delta
    delta.counts[state] = weight
    pricing_unknown = row.get("pricing_known") is False
    bound = _number(row.get("reservation_upper_bound_usd"))
    # Finality is a COUNT of OPEN ROWS, never a truthiness test on dollar sums:
    # an estimated $0.00 and a dispatched row whose reservation is exactly 0.0
    # are open however little they cost. ``priced`` is the evidence behind a
    # zero — "nothing was spent" vs "nothing is known" (#498).
    if state == "settled":
        if _number(row.get("cost_usd")) is None:
            delta.unknown = delta.non_final = weight
            if bound is not None:
                delta.priced = delta.tracked_nonfinal = delta.accounting_open = weight
        else:
            delta.priced = weight
            if not bool(row.get("cost_final")):
                delta.non_final = delta.tracked_nonfinal = delta.accounting_open = weight
    elif state in {"reserved", "dispatched", "unresolved"}:
        delta.non_final = delta.accounting_open = weight
        if bound is None or pricing_unknown:
            delta.unknown = weight
        if bound is not None:
            delta.priced = delta.tracked_nonfinal = weight
    return delta


def _fold(rows: Sequence[Dict[str, Any]]) -> Bucket:
    bucket = Bucket()
    for row in rows:
        bucket.add(summary_delta(row))
    return bucket


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The money summary of these final rows (``_fold`` + ``render_summary``).

    A row is final when it is SETTLED at a known price its writer called final;
    anything else is open. ``non_final_rows`` is the DISCLOSED CAUSE of
    ``cost_final: false``; ``priced_rows`` separates a measured zero from an
    unknown one; ``unknown_unmetered`` counts rows without a known price."""
    return _fold(rows).render_summary()


def _with_limit(summary: Dict[str, Any], limit: Optional[float]) -> Dict[str, Any]:
    """Decorate a summary with its configured limit and the room left above KNOWN
    (settled) spend — the admission rule's own number; open holds stay beside it."""
    if limit is None:
        return summary
    summary["limit_usd"] = round(max(0.0, float(limit)), 6)
    summary["remaining_known_usd"] = round(max(0.0, summary["limit_usd"] - float(summary["settled_usd"])), 6)
    return summary


def _with_integrity(summary: Dict[str, Any], degraded: bool) -> Dict[str, Any]:
    """Attach ledger integrity and prevent a torn tail from claiming final cost."""
    summary["integrity_degraded"] = bool(degraded)
    if degraded:
        summary["cost_final"] = False
    return summary


def _physical_call_count(row: Dict[str, Any]) -> int:
    kind = str(row.get("kind") or "attempt")
    if kind == "legacy_metadata":
        return 0
    if kind == "legacy_delta":
        return 0
    # A subscription session is not a core-mediated physical provider send; it is
    # counted on the sessions axis instead (see record_subscription_session).
    if kind == "subscription_session":
        return 0
    # CPL4-C6 baseline rows: the stamp header carries no calls; a group row
    # stands for `folded_attempt_count` final attempt rows of one state.
    if kind == "usage_baseline":
        return 0
    if kind == "usage_baseline_group":
        if str(row.get("state") or "") in {"settled", "unresolved"}:
            return max(1, int(row.get("folded_attempt_count") or 1))
        return 0
    return 1 if str(row.get("state") or "") in {"dispatched", "settled", "unresolved"} else 0


def _breakdown_bucket(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """``_summary`` plus physical calls, token sums (``None`` when no row
    reported that count: an absent count is never a confident zero) and the
    per-TTL physical-call counts."""
    return _fold(rows).render_breakdown()


_SKILL_ATTEMPT_FIELDS = (
    "attempt_id", "review_slot_id", "kind", "state", "model", "provider", "source",
    "cost_usd", "cost_final", "reservation_upper_bound_usd", "pricing_known",
    "prompt_tokens", "completion_tokens", "cached_tokens", "subscription_route",
    "subscription_reset_at", "credential_profile_id", "access_profile",
    "effort", "effort_resolution", "processing", "processing_basis", "cost_evidence", "attempt_execution",
)


def _skill_review_usage_bucket(
    rows: Sequence[Dict[str, Any]], *, review_skill: str, review_wave_id: str,
    integrity_degraded: bool,
) -> Dict[str, Any]:
    """Project one exact Skill Review wave from canonical final attempt rows."""
    selected = sorted(
        (
            row for row in rows
            if str(row.get("review_skill") or "") == review_skill
            and str(row.get("review_wave_id") or "") == review_wave_id
        ),
        key=lambda row: str(row.get("attempt_id") or ""),
    )
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(str(row.get("review_slot_id") or "(unattributed)"), []).append(row)
    by_slot = {
        slot: _with_integrity(_breakdown_bucket(grouped[slot]), integrity_degraded)
        for slot in sorted(grouped)
    }
    result = _with_integrity(_breakdown_bucket(selected), integrity_degraded)
    result.update({
        "review_skill": review_skill,
        "review_wave_id": review_wave_id,
        "attempt_ids": [str(row.get("attempt_id") or "") for row in selected],
        "attempts": [
            {key: row.get(key) for key in _SKILL_ATTEMPT_FIELDS if key in row}
            for row in selected
        ],
        "by_slot": by_slot,
        "attribution_complete": bool(selected) and all(
            str(row.get("review_slot_id") or "") for row in selected
        ),
    })
    return result


_VIEW_AXES = ("model", "provider", "category")
_VIEW_LEGACY_UNATTRIBUTED = frozenset({"legacy_metadata", "legacy_delta"})


# ---- read views: the historical projection/breakdown shapes from summary rows
# (``txn`` is a usage-store transaction: ``bucket``/``buckets``/``attempts``) ----

def _rendered(bucket: Bucket, degraded: bool, *, decorate: bool) -> Dict[str, Any]:
    rendered = bucket.render_breakdown()
    return _with_integrity(rendered, degraded) if decorate or degraded else rendered


def breakdown_view(txn: Any, *, root_task_id: str = "", task_id: str = "", include_owners: bool = False,
                   degraded: bool = False) -> Dict[str, Any]:
    """``usage_breakdown``'s shape for the global scope, one root or one task,
    from summary rows only. The global per-task/per-root maps enumerate every
    owner, so they are rendered only on request (``include_owners``)."""
    if root_task_id and task_id:
        return breakdown_of_rows(txn.attempts("root_task_id=? AND task_id=?", (root_task_id, task_id)), degraded)
    empty = Bucket()
    if root_task_id:
        top = txn.bucket("root", root_task_id)
        axes = {axis: txn.buckets(f"root_{axis}", root_task_id) for axis in _VIEW_AXES}
        axes["task"] = txn.buckets("root_task", root_task_id)
        axes["root"] = {root_task_id: top} if top.rows else {}
        delegated = txn.bucket("root_delegated", root_task_id)
    elif task_id:
        top = txn.bucket("task", task_id)
        axes = {axis: txn.buckets(f"task_{axis}", task_id) for axis in _VIEW_AXES}
        axes["root"] = txn.buckets("task_root", task_id)
        axes["task"] = {task_id: top} if top.rows else {}
        delegated = txn.bucket("task_delegated", task_id)
    else:
        top = txn.bucket("global")
        axes = {axis: {**txn.buckets(axis), "": txn.bucket("unattributed", axis)} for axis in _VIEW_AXES}
        for axis in ("task", "root"):
            owners = txn.buckets(axis) if include_owners else {}
            axes[axis] = {**{key: bucket for key, bucket in owners.items() if key},
                          "": txn.bucket("unattributed", axis)}
        delegated = txn.bucket("kind", "subscription_session")
    result: Dict[str, Any] = _rendered(top, degraded, decorate=True)
    for axis in ("model", "provider", "category", "task", "root"):
        found = axes[axis]
        result[f"by_{axis}"] = {key: _rendered(found[key], degraded, decorate=False) for key in sorted(found) if key}
    result["delegated"] = _rendered(delegated, degraded, decorate=True)
    result["unattributed"] = {axis: _rendered(axes[axis].get("") or empty, degraded, decorate=False)
                              for axis in ("model", "provider", "category", "task", "root")}
    return result


def breakdown_of_rows(rows: Sequence[Dict[str, Any]], degraded: bool) -> Dict[str, Any]:
    """The same shape aggregated from ADDRESSED rows (one root and task)."""
    def grouped(name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        groups: Dict[str, list] = {}
        unattributed: list = []
        for row in rows:
            key = str(row.get(name) or "")
            if str(row.get("kind") or "") in _VIEW_LEGACY_UNATTRIBUTED or not key:
                unattributed.append(row)
            else:
                groups.setdefault(key, []).append(row)
        return {key: _breakdown_bucket(groups[key]) for key in sorted(groups)}, _breakdown_bucket(unattributed)

    result = _with_integrity(_breakdown_bucket(rows), degraded)
    unattributed = {}
    for axis, name in (("model", "model"), ("provider", "provider"), ("category", "category"),
                       ("task", "task_id"), ("root", "root_task_id")):
        result[f"by_{axis}"], unattributed[axis] = grouped(name)
        if degraded:
            for bucket in (*result[f"by_{axis}"].values(), unattributed[axis]):
                _with_integrity(bucket, True)
    result["delegated"] = _with_integrity(_breakdown_bucket(
        [row for row in rows if str(row.get("kind") or "") == "subscription_session"]), degraded)
    result["unattributed"] = unattributed
    return result


def projection_view(txn: Any, *, root_task_id: str = "", billing_group_id: str = "",
                    limit: Optional[float] = None, include_roots: bool = False,
                    degraded: bool = False) -> Dict[str, Any]:
    """``usage_projection``'s shape: a group or root (with its minimum
    recorded cap) or the global total under ``limit``."""
    if billing_group_id or root_task_id:
        bucket = txn.bucket("group", billing_group_id) if billing_group_id else txn.bucket("root", root_task_id)
        return _with_integrity(_with_limit(bucket.render_summary(), bucket.min_cap()), degraded)
    result = _with_limit(txn.bucket("global").render_summary(), limit)
    if include_roots:
        result["by_root"] = {
            key: _with_integrity(_with_limit(bucket.render_summary(), bucket.min_cap()), degraded)
            for key, bucket in txn.buckets("root").items() if key}
    return _with_integrity(result, degraded)
