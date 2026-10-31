"""One passive quota read contract for Accounts, actor inspection, admission and the primary observation.

The engine owns applicability; a selector sent is not a selector honored. Old
engines' observations remain readable without authorizing exact availability.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

QUOTA_VIEW = "resources_freshness"
# Never project arbitrary engine error codes or exception prose: either can
# contain paths, credential material or an unbounded upstream response body.
_SAFE_ERROR_CODES = frozenset({
    "daemon_not_discovered", "daemon_descriptor_unreadable",
    "daemon_descriptor_incomplete", "daemon_token_unreadable",
    "daemon_endpoint_not_loopback", "daemon_unreachable",
    "malformed_response", "protocol_incompatible", "daemon_recovery_only",
})


def passive_read_error(exc: Exception) -> dict[str, Any]:
    """A failed passive read as a safe code and optional HTTP 400-599 status, never prose."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    if not isinstance(exc, ClaudexorUnavailable):
        return {"code": "read_failed"}
    code = exc.code if exc.code in _SAFE_ERROR_CODES else "read_refused"
    if exc.observation_reason == "observation_read_timeout":
        code = "observation_read_timeout"
    result: dict[str, Any] = {"code": code}
    if 400 <= exc.status_code <= 599:
        result["status_code"] = exc.status_code
    return result


def quota_envelope_valid(value: Any) -> bool:
    """Reject partial lists/freshness, preserving legacy envelopes and new fields."""
    if not isinstance(value, dict) or not isinstance(value.get("snapshots"), list):
        return False
    for key in ("snapshots", "absences", "resources"):
        if key in value and (not isinstance(value[key], list)
                             or not all(isinstance(row, dict) for row in value[key])):
            return False
    groups = [snapshot.get("constraints", []) for snapshot in value["snapshots"]]
    constraints = [row for group in groups if isinstance(group, list) for row in group]
    if any(isinstance(row, dict) and "freshness" in row for row in constraints):
        if not all(isinstance(group, list) for group in groups) or not all(
            isinstance(row, dict) and row.get("freshness") in ("fresh", "stale", "unknown")
            for row in constraints
        ):
            return False
    return True


def read_quota(gateway: Any, *, model: str | None = None, request_kind: str = "text",
               require_resources: bool = False, timeout_sec: float | None = None) -> dict[str, Any]:
    """No discovery/refresh. Accounts may require its already-advertised resources.

    A definite 400 permits a legacy view: Accounts requires resources, passive
    readers try view-only constraint freshness, then plain after a second 400.
    Legacy windows remain readable without implying honored model/kind filters.
    Accounts' existing catalog also permits resources after an ignored selector.
    Return that whole epoch, never combine reads or retry an unknown failure.
    ``timeout_sec`` is one budget for every view asked; a spent one raises TimeoutError.
    """
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    if request_kind not in ("text", "image"):
        raise ValueError("request_kind must be text or image")
    started = time.monotonic()

    def quota_state(**selectors: Any) -> Any:
        if timeout_sec is None:
            return gateway.quota_state(**selectors)
        remaining = timeout_sec - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Quota read deadline elapsed")
        return gateway.quota_state(**selectors, timeout_sec=remaining)

    try:
        value = quota_state(view=QUOTA_VIEW, model=model, request_kind=request_kind)
    except ClaudexorUnavailable as exc:
        if exc.status_code != 400:
            raise
        if require_resources:
            value = quota_state(view="resources")
        else:
            try:
                value = quota_state(view="constraint_freshness")
            except ClaudexorUnavailable as legacy_exc:
                if legacy_exc.status_code != 400:
                    raise
                value = quota_state()
    else:
        if (require_resources and quota_envelope_valid(value)
                and "resources" not in value and "availability_scope" not in value):
            value = quota_state(view="resources")
    if not quota_envelope_valid(value) or (require_resources and "resources" not in value):
        raise ClaudexorUnavailable("malformed_response", "Quota envelope is malformed")
    return value


def scope_honored(value: dict, model: str | None, request_kind: str = "text") -> bool:
    scope = value.get("availability_scope")
    return (isinstance(scope, dict) and "model" in scope and scope["model"] == model
            and scope.get("requestKind") == request_kind and scope.get("basis") == "quota_block")


def subject_rows(rows: Any, harness: str, profile: str = "") -> list[dict]:
    """A pin matches only its literal engine subject, never a healthy sibling.

    A null subject is an unprofiled route's own window (an explicit API key, a
    bootstrap or rollback state), never a named row's, whatever that row's id;
    Auto (no pin) still sees every subject of the harness.
    """
    relevant = [row for row in rows if isinstance(row, dict)
            and isinstance(row.get("subject"), dict)
            and row["subject"].get("harness") == harness] if isinstance(rows, list) else []
    return [row for row in relevant if not profile or row["subject"].get("subject_id") == profile]


def resource_rows(rows: Any, harness: str, profile: str = "") -> list[dict]:
    """AccountResourceSnapshot uses AccountTarget, not a nullable quota subject.

    Resource targets always name a profile: a null-target row cannot widen
    this independently typed identity or borrow another account's money.
    """
    return [row for row in rows if isinstance(row, dict)
            and isinstance(row.get("target"), dict)
            and row["target"].get("harness") == harness
            and isinstance(row["target"].get("profile_id"), str) and row["target"]["profile_id"]
            and (not profile or row["target"]["profile_id"] == profile)] if isinstance(rows, list) else []


# Field ranges are explicit (subagent_status_primitives.js has the same pattern):
# Python 3.10 accepts an offset like +05:60, JS a 24:00 hour. The calendar day is
# datetime's to reject (Feb 29 of a non-leap year, Apr 31).
_INSTANT = re.compile(r"\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d"
                      r"(?:\.\d+)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)")


def _future_instant(value: Any, now: float) -> float | None:
    """An engine ISO instant with an offset that is still ahead, else None (invalid or past)."""
    if not isinstance(value, str) or not _INSTANT.fullmatch(value):
        return None
    head, _, tail = value.partition(".")
    if tail:  # Python 3.10 parses only 3 or 6 fractional digits; JS Date.parse takes any.
        digits = re.match(r"\d+", tail).group()
        head = f"{head}.{(digits + '000000')[:6]}{tail[len(digits):]}"
    try:
        at = datetime.fromisoformat(head.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
    return at if at > now else None


def _applicable_measurement(constraint: Any, request_kind: str) -> bool:
    """One window that measured this request: a finite ratio, every model, the requested kind.

    The ratio is evidence that the window was observed, never a headroom policy.
    Model-scoped windows stay unknown: the host does not match model spellings.
    """
    if not isinstance(constraint, dict):
        return False
    ratio = constraint.get("used_ratio")
    models = constraint.get("applies_to_models")
    prefixes = constraint.get("applies_to_model_prefixes", [])
    kinds = constraint.get("applies_to_request_kinds", ["text", "image"])
    return (isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and 0 <= ratio <= 1
            and (models is None or models == []) and prefixes == []
            and isinstance(kinds, list) and request_kind in kinds)


def _subject_verdict(snapshots: list[dict], absent: bool, request_kind: str, now: float) -> dict:
    """Simultaneous rows of ONE subject are constraints on it, never alternatives.

    A dated, still-live engine block wins beside stale, free or absent rows, and
    on a stale row too: the engine keeps such a cap in its own admission ledger.
    Its reset is the next time to look again. Available needs every row fresh and
    engine-available, no absence, and one applicable measured window.
    """
    blocks = [(at, row["availability"]) for row in snapshots
              if isinstance(row.get("availability"), dict)
              and row["availability"].get("state") in ("exhausted", "cooldown")
              and (at := _future_instant(row["availability"].get("resets_at"), now)) is not None]
    if blocks:
        at, earliest = min(blocks, key=lambda block: block[0])
        return {"state": "exhausted" if any(v["state"] == "exhausted" for _, v in blocks) else "cooldown",
                "resets_at": earliest["resets_at"], "at": at}
    available = snapshots and not absent and all(
        row.get("freshness") == "fresh" and isinstance(row.get("availability"), dict)
        and row["availability"].get("state") == "available" for row in snapshots)
    if available and any(_applicable_measurement(constraint, request_kind) for row in snapshots
                         for constraint in (row.get("constraints") if isinstance(row.get("constraints"), list) else [])):
        return {"state": "available"}
    return {"state": "unknown"}


def _subject_key(row: dict) -> tuple:
    subject = row["subject"]
    return subject.get("harness"), subject.get("credential_route"), subject.get("subject_id")


def quota_availability(value: dict, harness: str, *, model: str | None = None,
                       profile: str = "", request_kind: str = "text", now: float | None = None) -> dict:
    """Project engine verdicts, never infer applicability from model spellings.

    Unknowns fail open at admission, not as a claim of execution readiness.
    Each full subject (harness, credential route, subject id) is judged alone;
    observed subjects are alternatives: one available suffices, one unknown
    prevents an all-blocked claim, otherwise the earliest reset is a recheck.
    This is the read roster, not the engine's enabled and ready pool.
    Monetary resources are deliberately not consulted.
    """
    unknown = {"state": "unknown", "resets_at": None}
    if not scope_honored(value, model, request_kind):
        return {**unknown, "reason": "availability_scope_unhonored"}
    now = time.time() if now is None else now
    snapshots = subject_rows(value.get("snapshots"), harness, profile)
    absences = subject_rows(value.get("absences"), harness, profile)
    keys = list(dict.fromkeys(_subject_key(row) for row in snapshots + absences))
    if not keys:
        return {**unknown, "reason": "quota_evidence_missing"}
    absent = {_subject_key(row) for row in absences}
    verdicts = [_subject_verdict([row for row in snapshots if _subject_key(row) == key], key in absent,
                                 request_kind, now) for key in keys]
    if any(v["state"] == "available" for v in verdicts):
        return {"state": "available", "resets_at": None, "execution_guaranteed": False}
    if any(v["state"] == "unknown" for v in verdicts):
        return {**unknown, "reason": "quota_evidence_stale_or_incomplete" if snapshots else "quota_evidence_missing"}
    earliest = min(verdicts, key=lambda v: v["at"])
    return {"state": "exhausted" if any(v["state"] == "exhausted" for v in verdicts) else "cooldown",
            "resets_at": earliest["resets_at"], "execution_guaranteed": False}


def _dated_row(row: dict, *, snapshot: bool) -> dict:
    """One engine row's subject, source and dates; no labels, ratios, money or detail prose."""
    fact = {"subject_id": row["subject"].get("subject_id"), "observed_at": row.get("observed_at")}
    if not snapshot:
        return {**fact, "reason": row.get("reason")}
    availability = row.get("availability") if isinstance(row.get("availability"), dict) else {}
    constraints = row.get("constraints")
    windows = [c for c in constraints if isinstance(c, dict)] if isinstance(constraints, list) else []
    return {**fact, "source": row.get("source"), "freshness": row.get("freshness"),
            "fresh_windows": sum(c.get("freshness") == "fresh" for c in windows), "windows": len(windows),
            "state": availability.get("state"), "resets_at": availability.get("resets_at")}


def quota_observation(harness: str, model: str | None, profile: str = "", *, timeout_sec: float,
                      request_kind: str = "text") -> dict[str, Any]:
    """Quota-only facts for one exact binding: the shared verdict and every scoped row, compact and dated.

    One deadline covers the metadata connection and every legacy view. A failed read,
    an unread envelope and an empty one stay distinct. Nothing refreshes, starts,
    selects, reads the roster or reads money. Auto sees every subject of the harness.
    """
    from ouroboros.claudexor_daemon import read_owned_gateway

    started = time.monotonic()

    def left() -> float:
        remaining = timeout_sec - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Quota observation deadline elapsed")
        return remaining

    fact: dict[str, Any] = {"requested_scope": {"model": model, "requestKind": request_kind,
                                                "account": profile or "Auto"}}
    try:
        gateway = read_owned_gateway(timeout_sec=left())
        try:
            value = read_quota(gateway, model=model, request_kind=request_kind, timeout_sec=left())
        finally:
            gateway.close()
    except TimeoutError:
        return {**fact, "read": "not_read", "reason": "observation_deadline_elapsed"}
    except Exception as exc:
        return {**fact, "read": "failed", "error": passive_read_error(exc)}
    snapshots = subject_rows(value.get("snapshots"), harness, profile)
    absences = subject_rows(value.get("absences"), harness, profile)
    verdict = quota_availability(value, harness, model=model, profile=profile, request_kind=request_kind)
    fact.update(read="ok", scope_honored=scope_honored(value, model, request_kind),
                subjects=("every engine subject of the harness, enabled or not" if not profile
                          else "the exact pin"),
                availability={**verdict, "scope": "quota_only"},
                snapshots=[_dated_row(row, snapshot=True) for row in snapshots],
                absences=[_dated_row(row, snapshot=False) for row in absences])
    return fact
