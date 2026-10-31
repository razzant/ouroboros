"""Retained identity writer from 88a70130d481abecd9c051ad1191a2f1c718e0f7.

Exact source excerpts, including the old ReviewSlot default. Tests use these only
while the real producer writes its prompt/result, then restore current collectors.
The provenance hashes name the complete original modules, not regenerated values.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

from ouroboros.review_execution import ReviewRouteKind, delivery_retrieves


# ouroboros/review_records.py:203-259; full source sha256=99d8e3233d6a20f6ee9a4d4870e76f5fcc9f8784f0b7d4feb50443db3252bde3
@dataclass(frozen=True)
class ReviewSlot:
    slot_id: str
    model: str
    effort: str = "medium"
    timeout_sec: Optional[float] = None
    max_tokens: int = 16_384
    temperature: float | None = None
    role_hint: str = ""
    use_local: bool = False
    # Delivery route for this slot. ``use_local`` above is the existing
    # precedent for a per-slot transport hint; ``route`` is the general axis.
    route: ReviewRouteKind = ReviewRouteKind.API_CHAT
    # agent_session rows only: THIS row's opaque ``harness[=model]`` target
    # (6.1 — every slot is independently harness-or-API). Empty falls back to
    # the shared session-route key, which is the whole legacy behavior.
    session_target: str = ""
    # Optional manual credential pin (Q2-в); '' = the daemon's rotation (D28).
    session_profile: str = ""
    transport_timeout_sec: Optional[float] = None
    # The catalog row this seat is (``slot_id == subagent_id`` for a pool row;
    # '' for a row built from a bare model). An IDENTITY fact only: it says
    # nothing about delivery (F8) — see ``native_retrieval_override``.
    subagent_id: str = ""
    # Host sampling hint, resolved at dispatch; an explicit temperature wins.
    default_temperature: float | None = None
    # The effort this row runs at because the CALLER declared it for one order
    # (plan review's ``reviewer_effort``): '' when the row's own effort, a
    # compound route slug or the surface setting applied. Disclosure for the
    # last-execution projection; identity already rides ``effort``.
    declared_effort: str = ""
    # Captured preference; empty explicitly preserves the legacy request shape.
    processing_preference: str = ""
    # An api row's delivery, stated by whoever built the row: ``True`` — the
    # bounded native inspection episode on its own route (the catalog's
    # ``native``), ``False`` — the assembled packet (``packet``). The pool
    # builder sets it explicitly in both directions; ``None`` (a row built from a
    # bare model, or a session row) means the packet. Never derived from the
    # presence of a subagent id (F8).
    native_retrieval_override: Optional[bool] = None
    # Configured technical rights; absent historical seats preserve readonly.
    session_access: str = ""

    @property
    def native_retrieval(self) -> bool:
        return (
            str(getattr(self.route, "value", self.route) or "") == ReviewRouteKind.API_CHAT.value
            and self.native_retrieval_override is True
        )

    @property
    def retrieves(self) -> bool:
        # DELIVERY class for admission/fit/authority; transport tests the route.
        # A session row always retrieves; an api row retrieves exactly when it
        # reads natively. ``delivery_retrieves`` is the same predicate for
        # callers that hold only a route and the native fact.
        return delivery_retrieves(self.route, self.native_retrieval)


# ouroboros/review_custody.py:689-742; full source sha256=371ea7e113924676477661e3b840f5f8e38d98e18b7b60b6fcb1d1bd333fe56f
def _attempt_key(request: Any, slot: Any) -> str:
    retry_key = str(getattr(request, "retry_key", "") or "").strip()
    slot_messages = (getattr(request, "slot_messages", {}) or {}).get(
        str(getattr(slot, "slot_id", "") or "")
    )
    slot_data = {
        key: getattr(slot, key, None) for key in (
            "slot_id", "model", "effort", "max_tokens", "temperature", "role_hint",
            "use_local", "route", "session_target", "session_profile",
            # Actor binding is operation identity: a changed configured-subagent
            # reference mints a new attempt key, exactly like a changed route.
            "subagent_id",
        )
    }
    if getattr(slot, "default_temperature", None) is not None:
        slot_data["default_temperature"] = slot.default_temperature
    if getattr(slot, "processing_preference", ""):
        slot_data["processing_preference"] = slot.processing_preference
    if str(getattr(slot, "session_access", "") or "readonly") != "readonly":
        slot_data["session_access"] = slot.session_access  # writable rights are identity
    identity = {
        "retry_key": retry_key,
        "surface": getattr(request, "surface", ""),
        "task_id": getattr(request, "task_id", ""),
        "call_type": getattr(request, "call_type", ""),
        "session_root": getattr(request, "session_root", ""),
        "slot": slot_data,
    }
    # An explicit retry key is the logical-operation identity. Mutable prompt
    # history must not turn a retry of that operation into a second paid send.
    # Callers change the key for genuinely new material/cycles; old callers
    # without one retain the full content-addressed behavior byte-for-byte.
    if retry_key:
        payload = identity
    else:
        payload = {
            **identity,
            "task_attempt": getattr(request, "task_attempt", None),
            "max_tokens": getattr(request, "max_tokens", None),
            "temperature": getattr(request, "temperature", None),
            "no_proxy": getattr(request, "no_proxy", None),
            "messages": slot_messages if slot_messages is not None else getattr(request, "messages", []),
            "goal": getattr(request, "goal", ""),
            "scope": getattr(request, "scope", ""),
            "subject": getattr(request, "subject", ""),
            "evidence_refs": getattr(request, "evidence_refs", []),
            "evidence": getattr(request, "evidence", {}),
            "policy": getattr(request, "policy", {}),
            "session_task": getattr(request, "session_task", ""),
        }
        if getattr(request, "default_temperature", None) is not None:
            payload["default_temperature"] = request.default_temperature
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


# ouroboros/review_dispatch.py:484-520; full source sha256=7251de0c854719c22f85972747bce6af4ab0422d16483df945f6fe618ea7bcd1
def review_reconciliation_identity(request: Any, slots: list, *, root_task_id: str, contract: Any = "") -> dict:
    """Reuse the caller's material/cycle identity and the exact configured roster."""
    import hashlib
    import json
    from dataclasses import asdict, is_dataclass
    from ouroboros.review_execution import review_output_contract

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                         default=str).encode("utf-8")).hexdigest()

    supplied = dict(getattr(request, "reconciliation_identity", None) or {})
    roster = []
    for slot in slots:
        values = asdict(slot) if is_dataclass(slot) else dict(getattr(slot, "__dict__", {}) or {})
        # A read-only seat keeps the roster identity recorded before seats carried
        # rights (#1609); writable rights are identity, as in ``_attempt_key``.
        roster.append({k: v for k, v in values.items()
                       if k not in {"timeout_sec", "transport_timeout_sec"}
                       and (k != "processing_preference" or v)
                       and (k != "session_access" or str(v or "readonly") != "readonly")})
    retry_key = getattr(request, "retry_key", None)
    return {
        "subject_hash": digest(retry_key or {
            "subject": getattr(request, "subject", ""), "goal": getattr(request, "goal", ""),
            "scope": getattr(request, "scope", ""), "evidence": getattr(request, "evidence", ""),
            "evidence_refs": getattr(request, "evidence_refs", []), "messages": getattr(request, "messages", []),
            "slot_messages": getattr(request, "slot_messages", []),
        }),
        "review_contract": str(contract or digest({
            "rendered": review_output_contract(request) if hasattr(request, "policy") else "",
            "policy": getattr(request, "policy", ""),
        })),
        "roster_hash": digest(roster), "epoch": str(retry_key or ""),
        **supplied,
        "root_task_id": str(root_task_id), "task_attempt": getattr(request, "task_attempt", None),
    }


# Historical collector provenance (unused executable excerpt removed):
# ouroboros/review_operation.py:1223-1258; full source sha256=1d6b69500fbc46ceb870db2cb2d8eadfe204e39df354aec53f4de1fe7941c326
