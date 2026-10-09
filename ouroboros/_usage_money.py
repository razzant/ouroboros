"""Exact reversible cash arithmetic shared by replay, writers and compaction.

The public contract rounds each bucket, then its accounted sum, to six decimal
places (half even). Accumulators never round. The precision-60, Inexact-trapping
context is the compactor's storage contract, independent of ambient Decimal
settings. A representational overflow is an error, never a float fallback.
"""
from __future__ import annotations

import contextlib
import decimal
from decimal import Decimal
from typing import Any, Iterator


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
        from ouroboros.usage_ledger import UsageNonFiniteMoney

        raise UsageNonFiniteMoney("non-finite monetary value; source data requires repair")
    return parsed if parsed >= 0 else None


def durable_literals(value: Any, key: str = "") -> Any:
    """Preserve monetary literals; unrelated numeric metadata keeps its shape."""
    if isinstance(value, LiteralFloat):
        monetary = key.endswith("_usd") or key in {"cashUsd", "valuationUsd"}
        return value.literal if monetary and decimal_of(value) != Decimal(str(float(value))) else float(value)
    if isinstance(value, dict):
        return {name: durable_literals(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [durable_literals(item, key) for item in value]
    return value


CASH_KEYS = ("settled_usd", "confirmed_usd", "estimated_usd", "reserved_usd",
             "unresolved_upper_bound_usd")
ZERO_CASH = (Decimal(0),) * len(CASH_KEYS)


def monetary_scope_key(row: dict) -> str:
    """Canonical root monetary scope, shared by replay and incremental indexes.

    Absent/None/empty roots keep the existing unattributed key. Billing groups
    are a separate derived axis; they never replace the real root identity.
    """
    return str(row.get("root_task_id") or "")


def billing_group_key(row: dict) -> str:
    """Whole-work scope, including pre-group rows of the original root."""
    return str(row.get("billing_group_id") or monetary_scope_key(row))


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


def exceeds_limit(total: tuple, limit) -> bool:
    """Whether KNOWN spend has reached ``limit`` (owner Q4-A, #1487).

    Known is the settled bucket: confirmed prices plus disclosed estimates.
    Reservations and unresolved bounds stay in the accounted exposure for
    display; they are not counted as spending and refuse nothing here. The
    same predicate guards reservation and dispatch, so a crossing in between
    refuses without sending. Concurrent and late charges can still overshoot.
    """
    with exact_money():
        cap, known, allowance = decimal_of(limit), rounded_cash(total)[0], Decimal("1e-9")
        return cap <= 0 or known >= cap - allowance
