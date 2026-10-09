"""Catalog rosters for tests that need a review pool (PR-3).

A reviewer is a marked row of ``OUROBOROS_SUBAGENTS``; there is no shipped
default panel at runtime (the one-time migration mints the factory rows into the
install's catalog). Tests that used to rely on the lane-era default panel or set
``OUROBOROS_REVIEWER_SLOTS`` build their pool here instead. Seats default to the
``packet`` delivery because that is what the lane triad rows those tests pinned
received; pass ``delivery="native"`` (or omit the key via ``delivery=None``) for a
natively retrieving api seat.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Sequence

from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS

FACTORY_MODELS: tuple = tuple(OPENROUTER_REVIEW_DEFAULTS["triad"])


def pool_seat(subagent_id: str, target_id: str, *, kind: str = "api_model", delivery: Optional[str] = "packet",
              effort: str = "", profile_id: str = "", marked: bool = True, **extra: Any) -> Dict[str, Any]:
    """One catalog row; ``marked`` rows are in the review pool."""
    route: Dict[str, Any] = {"kind": kind, "target_id": target_id}
    if profile_id:
        route["credential_profile_id"] = profile_id
    row: Dict[str, Any] = {"subagent_id": subagent_id, "recommended_use": f"{subagent_id} reviewer.", "route": route}
    if marked:
        row["review_eligible"] = True
    if kind == "api_model" and delivery:
        row["delivery"] = delivery
    if effort:
        row["effort"] = effort
    row.update(extra)
    return row


def pool_roster(*rows: Dict[str, Any], enabled: bool = True) -> str:
    """The ``OUROBOROS_SUBAGENTS`` text for these rows."""
    return json.dumps({"enabled": enabled, "items": list(rows)})


def packet_pool(models: Sequence[str] = FACTORY_MODELS, *, prefix: str = "review", delivery: Optional[str] = "packet",
                effort: str = "", enabled: bool = True) -> str:
    """A pool of one api seat per model (``review-1``, ``review-2``, …), the shape the
    factory rows take after the migration — packet by default."""
    return pool_roster(*(
        pool_seat(f"{prefix}-{index}", model, delivery=delivery, effort=effort)
        for index, model in enumerate(models, start=1)
    ), enabled=enabled)


def mixed_pool_rows(models: Sequence[str] = FACTORY_MODELS, *, prefix: str = "review") -> list:
    """Three pool rows the way a mixed install reads them: two natively retrieving api
    seats (asked both parts of the brief) and one packet seat (asked ``change``)."""
    return [pool_seat(f"{prefix}-1", models[0], delivery="native"),
            pool_seat(f"{prefix}-2", models[1], delivery="native"),
            pool_seat(f"{prefix}-3", models[2])]


def set_review_pool(monkeypatch, roster_or_models: Any = FACTORY_MODELS, **kwargs: Any) -> str:
    """Put a pool in the environment: a roster text, or the models of a packet pool."""
    raw = roster_or_models if isinstance(roster_or_models, str) else packet_pool(list(roster_or_models), **kwargs)
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", raw)
    for key in ("OUROBOROS_REVIEWER_SLOTS", "OUROBOROS_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODELS",
                "OUROBOROS_SCOPE_REVIEW_MODEL"):
        monkeypatch.delenv(key, raising=False)
    return raw


def pool_targets(raw: str) -> list:
    """The marked rows' targets of a roster text, in order (a quick assertion helper)."""
    return [row["route"]["target_id"] for row in json.loads(raw)["items"] if row.get("review_eligible")]


def iter_marked(raw: str) -> Iterable[Dict[str, Any]]:
    return (row for row in json.loads(raw)["items"] if row.get("review_eligible"))
