"""Pure row-math projections over physical-attempt ledger rows.

Extracted from ``usage_accounting.py`` at the v6.91 gate-3 fix round so the
substrate stays under the hard module gate: the accounted/summary arithmetic,
its limit/integrity decorations, and the per-row physical-call/bucket
projections. A LEAF over plain row dicts — no file I/O, no locks, and it never
imports ``ouroboros.usage_accounting``; that module re-exports every name here
so historical import and monkeypatch sites keep working unchanged.
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence
from decimal import Decimal, InvalidOperation

from ouroboros.usage_ledger import _number
from ouroboros._usage_money import billing_group_key, monetary_scope_key, ZERO_CASH, cash_contribution, change_cash, render_cash, exact_money, decimal_of

REVIEW_ATTRIBUTION_KEYS = ("review_skill", "review_wave_id", "review_slot_id")

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


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cash = ZERO_CASH
    unknown = 0
    # Finality is a COUNT of OPEN ROWS, not a truthiness test on dollar sums. Three of the
    # four old terms asked a STATE question of a float, so any row that is genuinely open
    # while holding $0.00 disappeared from the predicate entirely:
    #   * an ESTIMATED $0.00 — what the engine reports for a delegated subscription run
    #     whose cash it has not settled (all 8 estimated rows on a live 60-run page); and
    #   * a DISPATCHED row whose reservation is exactly 0.0, which `_reservation_cost`
    #     returns for `provider="local"`, so the projection claimed `cost_final: True`
    #     with a physical send still in flight.
    # A row is final when it is SETTLED at a known price its writer called final; anything
    # else is open, however little it costs. `_final_rows` keys by attempt_id, so a settled
    # row REPLACES its own reserved/dispatched predecessor — a row still open here is
    # really still open, and a released reservation is in neither branch.
    #
    # The count is also the DISCLOSED CAUSE, returned as `non_final_rows`: a projection
    # reporting `cost_final: false` with every dollar bucket at zero and `unknown` at zero
    # is a flag no reader can reconstruct.
    non_final_rows = 0
    # Rows whose money is KNOWN: a settled price, or a reservation upper bound a
    # still-open row is carrying. It is the evidence behind a ZERO — an empty
    # ledger and a ledger of exclusively unpriced rows both sum to 0.0, and only
    # this count separates "nothing was spent" from "nothing is known" (#498).
    # Weighted like every other axis, so a compacted baseline group answers the
    # same as the final attempt rows it folded.
    priced_rows = 0
    # Presentation facts are independent of cost_final: a settled unpriced
    # attempt leaves the total unknown without making a known subtotal inexact.
    tracked_nonfinal_rows = accounting_open_rows = 0
    counts: Dict[str, int] = {}
    # Session count/quota and incremental cash remain separate observed axes.
    sessions = 0
    session_windows: Dict[str, str] = {}
    for row in rows:
        cash = change_cash(cash, new=cash_contribution(row))
        state = str(row.get("state") or "")
        kind = str(row.get("kind") or "")
        if kind == "usage_baseline":
            # Compaction stamp header (CPL4-C6): pure provenance, no money,
            # no counts — the folded contributions live on its group rows.
            continue
        # A baseline group row carries the PRE-SUMMED monetary/token totals of
        # ``folded_attempt_count`` final attempt rows that shared every branch
        # predicate below (the compactor's group key), so sums add ONCE and
        # count axes add the weight. Ordinary rows keep weight 1 exactly.
        weight = (
            max(1, int(row.get("folded_attempt_count") or 1))
            if kind == "usage_baseline_group"
            else 1
        )
        if kind == "subscription_session":
            sessions += 1
            route = str(row.get("subscription_route") or "")
            reset_at = str(row.get("subscription_reset_at") or "")
            if route and reset_at:
                session_windows[route] = max(session_windows.get(route, ""), reset_at)
        if kind == "legacy_metadata":
            ambiguous = max(1, int(row.get("ambiguous_call_count") or 1))
            counts["metadata_only"] = counts.get("metadata_only", 0) + ambiguous
            continue
        counts[state] = counts.get(state, 0) + weight
        pricing_unknown = row.get("pricing_known") is False
        if state == "settled":
            cost = _number(row.get("cost_usd"))
            if cost is None:
                unknown += weight
                non_final_rows += weight
                bound = _number(row.get("reservation_upper_bound_usd"))
                if bound is not None:
                    priced_rows += weight
                    tracked_nonfinal_rows += weight
                    accounting_open_rows += weight
            else:
                priced_rows += weight
                if not bool(row.get("cost_final")):
                    non_final_rows += weight
                    tracked_nonfinal_rows += weight
                    accounting_open_rows += weight
        elif state == "reserved":
            non_final_rows += weight
            accounting_open_rows += weight
            bound = _number(row.get("reservation_upper_bound_usd"))
            if bound is None or pricing_unknown:
                unknown += weight
            if bound is not None:
                priced_rows += weight
                tracked_nonfinal_rows += weight
        elif state in {"dispatched", "unresolved"}:
            non_final_rows += weight
            accounting_open_rows += weight
            bound = _number(row.get("reservation_upper_bound_usd"))
            if bound is None or pricing_unknown:
                unknown += weight
            if bound is not None:
                priced_rows += weight
                tracked_nonfinal_rows += weight
    return {
        **render_cash(cash),
        "unknown_unmetered": unknown,
        "priced_rows": priced_rows,
        "tracked_nonfinal_rows": tracked_nonfinal_rows,
        "accounting_open_rows": accounting_open_rows,
        # Every row that increments `unknown` is open, so the old `not unknown` term is
        # subsumed here rather than dropped.
        "non_final_rows": non_final_rows,
        "cost_final": not non_final_rows,
        "attempt_counts": counts,
        "subscription_sessions": sessions,
        "subscription_windows": session_windows,
        **({"processing_summary": processing} if (processing := _processing_summary(rows)) else {}),
    }


def _with_limit(summary: Dict[str, Any], limit: Optional[float]) -> Dict[str, Any]:
    """Decorate a summary with its configured limit and remaining headroom."""
    if limit is None:
        return summary
    summary["limit_usd"] = round(max(0.0, float(limit)), 6)
    summary["remaining_known_usd"] = round(max(0.0, summary["limit_usd"] - float(summary["accounted_usd"])), 6)
    return summary


def _with_integrity(summary: Dict[str, Any], degraded: bool) -> Dict[str, Any]:
    """Attach ledger integrity and prevent a torn tail from claiming final cost."""
    summary["integrity_degraded"] = bool(degraded)
    if degraded:
        summary["cost_final"] = False
    return summary


def _projection_from_final(
    final: list, integrity_degraded: bool, configured_limit: Optional[float] = None,
    *, root_task_id: str = "", include_roots: bool = True, billing_group_id: str = "",
) -> Dict[str, Any]:
    """Render the money projection from ALREADY-VALIDATED final rows: one
    snapshot, one projection, so a caller deriving the ordering marker from
    the SAME rows writes both under one authority instead of pairing a marker
    with a second, later ledger read."""
    def limit_of(rows: list) -> Optional[float]:
        known = [v for v in (_number(row.get("root_limit_usd")) for row in rows) if v is not None]
        return min(known) if known else None
    if billing_group_id:  # a whole-work group: its own rows plus legacy rows of its original root
        from ouroboros.usage_admission import group_rows

        rows = group_rows(final, billing_group_id)
        # Explicit None is original unlimited provenance, not a missing cap.
        caps = [row.get("billing_group_limit_usd", row.get("root_limit_usd")) for row in rows]
        known = [value for value in map(_number, caps) if value is not None]
        return _with_integrity(_with_limit(_summary(rows), min(known) if known else None), integrity_degraded)
    if root_task_id:
        rows = [row for row in final if monetary_scope_key(row) == root_task_id]
        return _with_integrity(_with_limit(_summary(rows), limit_of(rows)), integrity_degraded)
    result = _with_limit(_summary(final), configured_limit)
    if include_roots:
        grouped: Dict[str, list] = {}
        for row in final:
            rid = monetary_scope_key(row)
            if rid:
                grouped.setdefault(rid, []).append(row)
        result["by_root"] = {
            rid: _with_integrity(_with_limit(_summary(grouped[rid]), limit_of(grouped[rid])),
                                 integrity_degraded)
            for rid in sorted(grouped)
        }
    return _with_integrity(result, integrity_degraded)


def _marker_from_final(final: Sequence[Dict[str, Any]]) -> Optional[list]:
    """The ordered ``[compaction_epoch, seq]`` fact of these validated rows.

    Compaction advances the epoch in its leading ``usage_baseline`` header
    while renumbering live rows, so the PAIR stays ordered even when the file
    gets shorter. ``None`` means unknown ordering — never zero — so a
    compatibility writer can fail safe instead of writing money it cannot
    place in time.
    """
    try:
        baselines = [row for row in final
                     if isinstance(row, dict) and str(row.get("kind") or "") == "usage_baseline"]
        if len(baselines) > 1:
            raise ValueError("multiple usage baseline headers")
        epoch = baselines[0].get("compaction_epoch", 0) if baselines else 0
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("invalid usage baseline compaction epoch")
        seqs = []
        for row in final:
            value = row.get("seq") if isinstance(row, dict) else None
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("invalid usage ledger sequence marker")
            seqs.append(value)
        return [epoch, max(seqs, default=0)]
    except (OSError, TypeError, ValueError):
        return None




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
    bucket = _summary(rows)

    def summed(field: str) -> Optional[int]:
        """Sum the rows that HAVE this count; None when not one of them does.

        `int(row.get(field) or 0)` collapsed an ABSENT count into a reported zero, so a
        bucket whose provider returned no token counts at all published a confident
        "0 tokens" — the render-unknown-as-zero shape this module refuses everywhere
        else (`cost = None  # legacy zero may mean unknown pricing, never "free"`). A
        zero here is now only ever a MEASURED zero, and a partially-reporting bucket
        still sums the rows that did report rather than being erased by the ones that
        did not."""
        present = [row.get(field) for row in rows if row.get(field) is not None]
        return sum(max(0, int(value)) for value in present) if present else None

    prompt = summed("prompt_tokens")
    completion = summed("completion_tokens")
    prompt_cache_ttls: Dict[str, int] = {}
    for row in rows:
        ttl = str(row.get("prompt_cache_ttl") or "").strip()
        if ttl:
            prompt_cache_ttls[ttl] = prompt_cache_ttls.get(ttl, 0) + _physical_call_count(row)
    bucket.update({
        "physical_calls": sum(_physical_call_count(row) for row in rows),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        # Unknown on BOTH halves is an unknown total; one known half is a real total
        # of what was measured, reported as the number it is.
        "total_tokens": (None if prompt is None and completion is None
                         else (prompt or 0) + (completion or 0)),
        "cached_tokens": summed("cached_tokens"),
        "cache_write_tokens": summed("cache_write_tokens"),
        "prompt_cache_ttls": prompt_cache_ttls,
    })
    return bucket


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
