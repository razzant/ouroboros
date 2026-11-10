"""One external call priced by a published per-request tariff, and the ledger row that says so.

A vendor that bills per SUCCESSFUL REQUEST — not per token — never passes through the physical-attempt
ledger, so without a row of its own the call reads as free: ``cost_usd`` stays NULL, "unknown" collapses
into "nothing spent", and an otherwise empty ledger reads as a search that cost nothing.

The row is PRICED and ESTIMATED, never final. A tariff is a document about the provider's list price, not
that provider's receipt, and only a settled charge may be called final. It stays deliberately distinct
from ``usage_accounting.record_unmetered_external_dispatch``: that function's contract is "nobody could
meter this call", and the two facts must stay distinguishable in the ledger. It lives here rather than in
``usage_accounting`` because that module sits at its size ceiling, and because a published per-request
price is a policy about a document, not about attempt custody.
"""

from __future__ import annotations

import hashlib
import math
import pathlib
from typing import Any, Dict  # noqa: F401 — the row shape is part of this module's contract


def record_external_charge(
    charge_id: str,
    *,
    cost_usd: float,
    drive_root: pathlib.Path | str | None = None,
    model: str = "",
    provider: str = "external",
    task_id: str = "",
    root_task_id: str = "",
    parent_task_id: str = "",
    category: str = "external",
    source: str = "external_charge",
    non_task_operation: bool = False,
) -> str:
    """One delivered external request priced by a published per-request tariff.

    ``charge_id`` names ONE request: a distinct request needs a distinct id, and a replayed id returns
    the stored row (``usage_store.ONE_SHOT_IDENTITY``) instead of billing twice. A non-finite or
    negative price is refused rather than booked — the store's structural validator would otherwise
    report a corrupt ledger instead of the caller error it is.

    The row keeps the store's ``external_unmetered`` kind: that kind names where the row came from — an
    external dispatch outside core metering — not whether it carries a price.
    """
    from ouroboros import usage_accounting as ua

    stable_id = str(charge_id or "").strip()
    if not stable_id:
        raise ua.UsageAccountingError("external charge requires a stable charge_id")
    # The store's row encoder is JSON: money is a float here and becomes Decimal in the money layer.
    price = float(cost_usd)
    if not math.isfinite(price) or price < 0:
        raise ua.UsageAccountingError(f"external charge refuses a non-finite or negative price: {cost_usd!r}")
    bound = ua._CURRENT_SCOPE.get() or ua.UsageScope()
    root = ua._drive_root(drive_root or bound.drive_root)
    identity = hashlib.sha256(stable_id.encode("utf-8")).hexdigest()
    from ouroboros.usage_admission import settlement_billing_fields

    real_task = str(task_id or bound.task_id or "")
    real_root = str(root_task_id or (bound.root_task_id if not task_id else "") or real_task)
    billing = ({"billing_group_id": real_root} if non_task_operation
               else settlement_billing_fields(root, real_task, real_root))
    row = {
        "kind": "external_unmetered",
        "attempt_id": f"external-{identity[:24]}",
        "state": "settled",
        "model": str(model or ""),
        "provider": str(provider or "external"),
        "cost_usd": price,
        "cost_final": False,
        "cost_estimated": True,
        "reservation_upper_bound_usd": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "task_id": real_task,
        "root_task_id": real_root,
        **({"consciousness_cause": bound.consciousness_cause}
           if bound.consciousness_cause and real_root == (bound.root_task_id or bound.task_id) else {}),
        "parent_task_id": str(parent_task_id or bound.parent_task_id or ""),
        **{key: value for key, value in billing.items() if key.startswith("billing_group_")},
        "category": str(category or bound.category or "external"),
        "source": str(source or bound.source or "external_charge"),
        "external_dispatch_id_sha256": identity,
    }
    return ua._record_one_shot(root, row)
