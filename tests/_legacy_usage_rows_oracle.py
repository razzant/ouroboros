"""Frozen copy of the pre-store usage row math (base 84febbdd3): the oracle.

Test fixture only. The usage store maintains every summary incrementally
(``_usage_rows.summary_delta`` / ``Bucket``); the property tests compare those
running buckets with this verbatim copy of the list-based ``_summary`` /
``_breakdown_bucket`` / ``_processing_summary`` and the cash helpers they used,
so the oracle cannot drift with the implementation it judges.
"""
from __future__ import annotations

import contextlib
import decimal
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterator, Optional, Sequence


def _number(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 and parsed == parsed else None


@contextlib.contextmanager
def exact_money() -> Iterator[None]:
    with decimal.localcontext(decimal.Context(prec=60)) as context:
        context.traps[decimal.Inexact] = True
        yield


class LiteralFloat(float):
    """Ordinary JSON row shape with the original numeric literal retained.

Only arithmetic consumes ``literal``. No private field is added to durable
rows or public projections. A copied monetary field is serialized as its exact
decimal string if binary float serialization would change the literal.
    """

    def __new__(cls, literal: str):
        value = super().__new__(cls, literal)
        value.literal = literal
        return value


def decimal_of(value: Any) -> Decimal:
    if isinstance(value, bool):
        # The ledger's historical numeric grammar accepts JSON booleans as 0/1.
        return Decimal(int(value))
    return value if isinstance(value, Decimal) else Decimal(
        value.literal if isinstance(value, LiteralFloat) else str(value))


def amount(value: Any) -> Decimal | None:
    """Decode cash; nonfinite evidence is an integrity failure, never unknown."""
    if value is None:
        return None
    try:
        parsed = decimal_of(value)
    except (decimal.InvalidOperation, ValueError):
        return None
    if not parsed.is_finite():
        # The substrate imports this arithmetic leaf. Resolve its error lazily
        # so every reader fails typed without making it quarantinable corruption.
        raise ValueError("non-finite monetary value; source data requires repair")
    return parsed if parsed >= 0 else None


CASH_KEYS = ("settled_usd", "confirmed_usd", "estimated_usd", "reserved_usd",
             "unresolved_upper_bound_usd")
ZERO_CASH = (Decimal(0),) * len(CASH_KEYS)


def cash_contribution(row: dict) -> tuple[Decimal, ...]:
    if row.get("kind") in {"usage_baseline", "legacy_metadata"}:
        return ZERO_CASH
    cost = amount(row.get("cost_usd"))
    bound = amount(row.get("reservation_upper_bound_usd")) or Decimal(0)
    state = row.get("state")
    if state == "settled" and cost is not None:
        return (cost, cost if row.get("cost_final") else Decimal(0),
                Decimal(0) if row.get("cost_final") else cost, Decimal(0), Decimal(0))
    if state == "reserved":
        return (Decimal(0), Decimal(0), Decimal(0), bound, Decimal(0))
    if state in {"dispatched", "unresolved", "settled"}:
        return (Decimal(0), Decimal(0), Decimal(0), Decimal(0), bound)
    return ZERO_CASH


def change_cash(total: tuple, old: tuple = ZERO_CASH, new: tuple = ZERO_CASH) -> tuple:
    with exact_money():
        return tuple(a - b + c for a, b, c in zip(total, old, new))


def rounded_cash(total: tuple) -> tuple:
    # Quantization is the ONE deliberate rounding operation. Summation still
    # traps loss of precision, including the sum of the rounded buckets.
    with exact_money():
        with decimal.localcontext() as context:
            context.traps[decimal.Inexact] = False
            rounded = [value.quantize(Decimal("0.000001"), rounding=decimal.ROUND_HALF_EVEN)
                       for value in total]
        accounted = rounded[0] + rounded[3] + rounded[4]
    return (*rounded, accounted)


def render_cash(total: tuple) -> dict:
    return dict(zip((*CASH_KEYS, "accounted_usd"), map(float, rounded_cash(total))))


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

