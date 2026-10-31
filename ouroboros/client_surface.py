"""Owner Surface Fact: per-message client-surface provenance helpers.

The SPA measures raw sending-surface observables at send time and attaches
them to the chat frame; this module owns the fact's normalization boundary,
its surface-identity projection, and the mailbox surface-change note. It is
a dedicated module (rather than message_bus/loop code) so those size-gated
modules stay inside their ratchet limits; the ``owner_client`` context
rendering lives in ``ouroboros/context.py`` and host channel stamps are
assembled at their producers.

``normalize_client_surface`` is the ingress boundary for the untrusted-ish
browser payload: a CLOSED key set with bounded values — unknown keys are
dropped, every string bound is DISCLOSED (strict wire bound, marker inside the
limit), and an empty/non-dict result is None so ``{}`` can never slip through
metadata filters as a phantom fact. Host-built channel facts
(``{"channel": <source>}``) do not pass through here; they are trusted host
stamps assembled at their producers.
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, Optional

from ouroboros.utils import truncate_within_limit

log = logging.getLogger(__name__)

CLIENT_SURFACE_UA_LIMIT = 512
_CLIENT_SURFACE_SHORT_LIMIT = 64
_CLIENT_SURFACE_VIEWPORT_MAX = 100_000
_INTERFACE_SIZE_MIN, _INTERFACE_SIZE_MAX = 25, 500  # percent
_PIXEL_RATIO_MIN, _PIXEL_RATIO_MAX = 0.25, 16.0
# An IANA key (``Area/City``, ``UTC``, ``Etc/GMT+3``): no path traversal, no spaces.
_IANA_ZONE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*(?:/[A-Za-z0-9_+\-]+){0,2}")


def normalize_timezone_name(value: Any) -> str:
    """The sender's IANA time-zone name when the host can load it, else ``""``.

    The browser reports ``Intl...resolvedOptions().timeZone``; a missing,
    malformed or unloadable name is dropped rather than guessed, so a reader
    falls back to an explicitly labelled UTC and never to the server's zone.
    """
    name = value.strip() if isinstance(value, str) else ""
    if not name or len(name) > _CLIENT_SURFACE_SHORT_LIMIT or not _IANA_ZONE_RE.fullmatch(name):
        return ""
    try:
        from zoneinfo import ZoneInfo

        ZoneInfo(name)
    except Exception:
        return ""
    return name


def normalize_client_surface(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize a browser-supplied client_surface payload; None when empty."""
    if not isinstance(value, dict) or not value:
        return None
    fact: Dict[str, Any] = {}
    for key in ("pywebview", "narrow_layout", "coarse_pointer"):
        # Real booleans only — a string "false" would coerce to True and hand
        # the model an INVERTED fact; drop-don't-guess, like every other key.
        if isinstance(value.get(key), bool):
            fact[key] = value[key]
    ua = value.get("ua")
    if isinstance(ua, str) and ua.strip():
        fact["ua"] = truncate_within_limit(ua.strip(), CLIENT_SURFACE_UA_LIMIT)
    viewport = value.get("viewport")
    if isinstance(viewport, dict):
        try:
            # OverflowError: stock json.loads accepts the Infinity literal, and
            # int(inf) raises it — an uncaught escape here drops the owner's
            # whole message at the ws chat branch (adversarial finding, proven).
            w = max(0, min(int(viewport.get("w", 0)), _CLIENT_SURFACE_VIEWPORT_MAX))
            h = max(0, min(int(viewport.get("h", 0)), _CLIENT_SURFACE_VIEWPORT_MAX))
            fact["viewport"] = {"w": w, "h": h}
        except (TypeError, ValueError, OverflowError):
            pass
    # The page's Interface size in percent and the window's pixels per CSS px (web/modules/interface_size.js);
    # numbers only, bounded, a bool never counting as one.
    size = value.get("interface_size")
    if isinstance(size, (int, float)) and not isinstance(size, bool) and math.isfinite(size):
        fact["interface_size"] = int(max(_INTERFACE_SIZE_MIN, min(round(size), _INTERFACE_SIZE_MAX)))
    ratio = value.get("device_pixel_ratio")
    if isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and math.isfinite(ratio) and ratio > 0:
        fact["device_pixel_ratio"] = round(max(_PIXEL_RATIO_MIN, min(float(ratio), _PIXEL_RATIO_MAX)), 2)
    captured_at = value.get("captured_at")
    if isinstance(captured_at, str) and captured_at.strip():
        fact["captured_at"] = truncate_within_limit(captured_at.strip(), _CLIENT_SURFACE_SHORT_LIMIT)
    timezone = normalize_timezone_name(value.get("timezone"))
    if timezone:
        fact["timezone"] = timezone
    return fact or None


def owner_client_fact(meta: Any) -> Optional[Dict[str, Any]]:
    """The ONE projection of a task's sending-surface baseline from metadata.

    Reads ONLY ``metadata.client_surface`` — the fact is assembled at its
    PRODUCER (web ingress normalizes browser observables; bridge routing and
    /api/command host-stamp their channel; external /api/tasks admission
    stamps its caller-declared channel). The renderer never infers a surface
    from ``metadata.source``: that field is overloaded by internal producers
    (scheduler "scheduled_task"/"skill_scheduled_task"), and guessing here
    once dressed machine traffic as an owner message. Shared by the
    runtime-context ``owner_client`` render and the mailbox surface-note
    baseline so the two projections can never disagree.
    """
    if not isinstance(meta, dict):
        return None
    fact = meta.get("client_surface")
    if isinstance(fact, dict) and fact:
        return fact
    return None


def client_surface_identity(fact: Any) -> Optional[tuple]:
    """The SURFACE IDENTITY a mid-task change-note keys on.

    Deliberately excludes viewport, narrow_layout, timestamps and the time
    zone: a window resize across the 980px breakpoint, a phone rotation or a
    trip across zones is NOT a surface change and must never inject a false
    "different client surface" note.
    """
    if not isinstance(fact, dict) or not fact:
        return None
    if not any(key in fact for key in ("pywebview", "coarse_pointer", "ua", "channel")):
        # A fact carrying NONE of the identity keys has no identity: comparing
        # it would collapse to the all-empty tuple and mint a false "DIFFERENT
        # client surface" note against any real baseline (absence of evidence
        # is not observed change).
        return None
    return (
        bool(fact.get("pywebview")),
        bool(fact.get("coarse_pointer")),
        str(fact.get("ua") or ""),
        str(fact.get("channel") or ""),
    )


def owner_surface_note(owner_ctx: Any, fact: Any) -> str:
    """Surface-change note for a mailbox follow-up (Owner Surface Fact, Q4=A).

    Appended ONLY when the sending-surface IDENTITY differs from the last one
    seen this attempt; the first observed fact with no baseline gets a neutral
    wording — never a false "different". The note is a HOST annotation: the
    caller keeps the owner's exact words in the directive record and appends
    this to the injected message only.
    """
    if owner_ctx is None:
        return ""
    try:
        identity = client_surface_identity(fact)
        if identity is None:
            return ""
        last = getattr(owner_ctx, "_last_owner_surface_identity", None)
        if last is None:
            # Baseline through the SAME projection the context render uses —
            # so a CLI/API-admitted task (channel fact stamped at admission)
            # reads its first web follow-up as a surface CHANGE.
            last = client_surface_identity(owner_client_fact(getattr(owner_ctx, "task_metadata", None)))
        # Render BEFORE advancing the baseline: a dumps failure must not eat
        # the change-note forever by leaving the baseline already moved.
        rendered = json.dumps(fact, ensure_ascii=False, sort_keys=True)
        owner_ctx._last_owner_surface_identity = identity
        if last is None:
            return f"\n[note: sent from client surface: {rendered}]"
        if identity != last:
            return f"\n[note: this follow-up was sent from a DIFFERENT client surface: {rendered}]"
        return ""
    except Exception:
        log.debug("owner surface note failed", exc_info=True)
        return ""


def noted_owner_text(owner_ctx: Any, entry: Dict[str, Any], text: str) -> str:
    """The injected mailbox text plus its surface note when one is due."""
    note = owner_surface_note(owner_ctx, entry.get("client_surface"))
    return text + note if note else text
