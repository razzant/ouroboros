"""The subscription census: the engine's verdicts on this install's subscription accounts, counted per family,
kept as one dated file and shown in the round's facts row when what it states changes (owner 21.11.2026 «1A»).

The engine owns every verdict (``claudexor_observations.quota_availability``, the reader Accounts, admission and the
primary observation share); this module only counts them. ``build_census`` is the pure count over one passive read
(``gateway/claudexor_passive``: the roster and one quota epoch, no handshake, no refresh) and ``census_text`` its
facts-only words. The census duty (``server_maintenance``) runs ``refresh`` every ``DUTY_SEC`` while a live actor
exists, or ``REQUEST_MIN_SEC`` after a refresh request (a delegated run settled on a limit code, a spent window the
engine named, a confirmed account change); it writes ``CENSUS_FILE`` only when what the text states changed.
``quota_part`` is the round's reader: the file alone, never a connection, for the foci that may choose an executor,
change-only against the latest part still in view. Nothing here selects, starts, refreshes, switches or advises.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import re
import threading
import time
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ouroboros.claudexor_observations import (
    _applicable_measurement, _future_instant, quota_availability, scope_honored,
)

log = logging.getLogger(__name__)

CENSUS_TYPE = "subscription_census"
CENSUS_FILE = "state/subscription_census.json"  # one record replaced in place: about 2 KB for 50 accounts
REQUEST_FILE = "state/subscription_census.request"  # a refresh request: its mtime against the duty's last pass
DUTY_SEC = 600.0  # the duty's cadence while a live actor exists; an idle install reads nothing
REQUEST_MIN_SEC = 15.0  # a refresh request is answered no sooner than this after the previous pass
_MAX_BYTES = 262_144  # the read bound of the file
# ``subject.credential_route`` of a vendor login; ``managed_api_key`` is a paid key and ``local`` a local model.
SUBSCRIPTION_ROUTES = frozenset({"vendor_native"})
# A settled delegated run that failed on one of these engine codes asks for a fresh census.
LIMIT_FAILURE_CODES = frozenset({"vendor_limit_rejected", "subscription_window_exhausted",
                                 "credential_pool_exhausted", "rate_limited"})
LABEL = "subscription accounts"
RETRACTION = f"({LABEL}, engine verdicts): no enabled subscription accounts now"
UNCONFIRMED = f"({LABEL}): the previous summary is no longer confirmed"
_SLOT = "quota "  # the facts row writes each part as ``quota <text>`` (``loop_messages.context_facts_line``)
_RETRACTED, _UNCONFIRMED = "retracted", "unconfirmed"
# A Presence task receives the part only when its capability ceiling grants one of these.
_EXECUTOR_TOOLS = ("schedule_subagent", "delegate_start", "start_task", "review_change", "commit_reviewed", "plan_task")
# Counted classes in their text order; with room and at limit carry their own details.
_CLASSES = (("with_room", "with room"), ("at_limit", "at limit"), ("verification_failed", "verification failed"),
            ("no_measured_window", "available (no measured general window)"),
            ("no_quota_source", "no quota source"), ("not_freshly_read", "not freshly read"))
_SAFE_CODE = re.compile(r"[a-z0-9_]{1,64}")

_LOCK = threading.Lock()  # one publication, failure record or invalidation at a time in this process
_PASS: Dict[str, Any] = {"confirmed_at": None, "failures": 0}  # memory only: the last successful read and the streak
_WRITES = frozenset({"written", "recovered", "confirmed", "none", "failure_recorded"})


# ---- the count -----------------------------------------------------------------------------------------------------

def verification_failed(status: Any) -> bool:
    """The Accounts rule (``web/modules/harness_accounts.js`` ``verificationBadge``'s error tone, which ``needsSignIn``
    reads): a reported verification that is neither ``passed`` nor ``not_run``."""
    value = status.get("verification") if isinstance(status, Mapping) else None
    return isinstance(value, str) and bool(value) and value not in ("passed", "not_run")


def build_census(payload: Mapping[str, Any], now: float) -> Optional[Dict[str, Dict[str, Any]]]:
    """Each family's counted accounts from one passive read (``claudexor_passive._quota_payload``'s shape), or None
    when the engine did not certify the model-agnostic text scope (an older engine): there is then no census.

    Counted: the roster's enabled named profiles of a subscription credential kind (the core's include-list,
    ``gateway.onboarding._SUBSCRIPTION_CREDENTIAL_KINDS``), each judged on its subscription-route rows and absences
    alone, so a paid key never gives a subscription account room. In this order: verification failed (the Accounts
    rule), the engine's verdict at the limit or with room, then the engine's unknowns by cause. Profiles with room add
    each window length's highest fresh measured share; model-scoped blocks still ahead count once per profile and
    engine model list. The result is JSON-safe: dicts keyed by family, lists, strings and numbers.
    """
    from ouroboros.gateway.onboarding import _SUBSCRIPTION_CREDENTIAL_KINDS

    scope = payload.get("availability_scope")
    if not scope_honored({"availability_scope": scope}, None, "text"):
        return None
    roster = _mapping(payload.get("profiles"))
    snapshots, absences = _dicts(payload.get("quota")), _dicts(payload.get("quota_absences"))
    tallies: Dict[str, Dict[str, Any]] = {}
    for wrapper in _dicts(roster.get("profiles")):
        profile = _mapping(wrapper.get("profile"))
        harness, profile_id = profile.get("harness_id"), profile.get("profile_id")
        if (not _named(harness) or not _named(profile_id) or profile.get("enabled") is False
                or profile.get("credential_kind") not in _SUBSCRIPTION_CREDENTIAL_KINDS):
            continue
        tally = tallies.setdefault(harness, {"counts": dict.fromkeys((key for key, _ in _CLASSES), 0), "resets": [],
                                             "windows": {}, "blocks": {}})
        rows, gone = _own_rows(snapshots, harness, profile_id), _own_rows(absences, harness, profile_id)
        for key, block in _profile_blocks(rows, now).items():
            tally["blocks"].setdefault(key, []).append(block)
        category, reset = _judge(rows, gone, _mapping(wrapper.get("status")), harness, profile_id, scope, now)
        tally["counts"][category] += 1
        if reset is not None:
            tally["resets"].append(reset)
        if category == "with_room":
            for length, share in _highest_shares(rows).items():
                tally["windows"].setdefault(length, []).append(share)
    return {harness: _family(tally) for harness, tally in sorted(tallies.items())}


def _judge(rows: List[dict], gone: List[dict], status: Mapping[str, Any], harness: str, profile_id: str,
           scope: Any, now: float) -> Tuple[str, Optional[Tuple[float, str]]]:
    """One profile's class and, at the limit, its engine reset ``(instant, text)``."""
    if verification_failed(status):
        return "verification_failed", None
    verdict = quota_availability({"snapshots": rows, "absences": gone, "availability_scope": scope}, harness,
                                 profile=profile_id, request_kind="text", now=now)
    if verdict["state"] in ("exhausted", "cooldown"):
        at = _future_instant(verdict.get("resets_at"), now)
        return "at_limit", (at, verdict["resets_at"]) if at is not None else None
    if verdict["state"] == "available":
        return "with_room", None
    fresh = [row for row in rows if row.get("freshness") == "fresh"]
    if rows and not gone and len(fresh) == len(rows) and all(_state(row) == "available" for row in rows):
        return "no_measured_window", None  # the engine says available; no finite general text window measured it
    if not fresh and any(row.get("reason") == "no_source" for row in gone):
        return "no_quota_source", None
    return "not_freshly_read", None


def _highest_shares(rows: List[dict]) -> Dict[int, float]:
    """Each window length's highest fresh measured share among a profile's fresh engine-available rows: one value per
    profile and length, so a family's range is over accounts, never over one account's windows."""
    shares: Dict[int, float] = {}
    for row in rows:
        if row.get("freshness") != "fresh" or _state(row) != "available":
            continue
        for constraint in _dicts(row.get("constraints")):
            if _applicable_measurement(constraint, "text") and constraint.get("freshness", "fresh") == "fresh":
                length, share = _seconds(constraint.get("window_seconds")), float(constraint["used_ratio"])
                shares[length] = max(shares[length], share) if length in shares else share
    return shares


def _profile_blocks(rows: List[dict], now: float) -> Dict[tuple, Tuple[float, str, Optional[str]]]:
    """A profile's model-scoped blocks still ahead, once per engine model list, as ``(instant, reset, seen)``: a fresh
    row's reading wins over a stale one of the same list; a stale one carries its observation time; passed or
    malformed resets drop (``_future_instant``, the core's parser)."""
    found: Dict[tuple, Tuple[bool, float, str, Optional[str]]] = {}
    for row in rows:
        fresh = row.get("freshness") == "fresh"
        for block in _dicts(_mapping(row.get("availability")).get("model_scoped_exhaustions")):
            at = _future_instant(block.get("resets_at"), now)
            if at is None:
                continue
            key = (_strings(block.get("applies_to_models")), _strings(block.get("applies_to_model_prefixes")))
            held = found.get(key)
            if held is None or (fresh, -at) > (held[0], -held[1]):
                seen = None if fresh else (row.get("observed_at") if isinstance(row.get("observed_at"), str) else "")
                found[key] = (fresh, at, block["resets_at"], seen)
    return {key: (at, reset, seen) for key, (_fresh, at, reset, seen) in found.items()}


def _family(tally: Dict[str, Any]) -> Dict[str, Any]:
    family: Dict[str, Any] = dict(tally["counts"])
    family["earliest_reset"] = min(tally["resets"])[1] if tally["resets"] else None
    family["windows"] = [{"window_seconds": length, "least": min(shares), "greatest": max(shares),
                          "profiles": len(shares)} for length, shares in sorted(tally["windows"].items())]
    family["model_scoped"] = []
    for (models, prefixes), blocks in sorted(tally["blocks"].items()):
        stale = [(_instant(seen) or float("inf"), seen) for _at, _reset, seen in blocks if seen is not None]
        family["model_scoped"].append({"models": list(models), "prefixes": list(prefixes), "profiles": len(blocks),
                                       "earliest_reset": min(blocks, key=lambda block: block[0])[1],
                                       "seen": min(stale)[1] if stale else None})
    return family


# ---- the words -----------------------------------------------------------------------------------------------------

def census_text(census: Mapping[str, Any]) -> str:
    """The facts row's part without its slot word (the row writes ``quota <text>``): the engine's verdicts as counted,
    dated by the first read that gave them; after a failed read, also when the engine was last read and why not."""
    since, code = _clock(census.get("since")), census.get("code")
    if census.get("read") == "failed":
        code = code if isinstance(code, str) and _SAFE_CODE.fullmatch(code) else "read_failed"
        head = f"({LABEL}, engine not read since {_clock(census.get('confirmed_at'))} ({code}); values from {since})"
    else:
        head = f"({LABEL}, engine verdicts, unchanged since {since})"
    return f"{head}: {body_text(_mapping(census.get('families')))}"


def body_text(families: Mapping[str, Any]) -> str:
    """The families as the text states them, in name order: what the signature and the in-view comparison cover."""
    return "; ".join(_family_text(name, family) for name, family in sorted(families.items())
                     if isinstance(family, Mapping)) or "none"


def _family_text(name: str, family: Mapping[str, Any]) -> str:
    bits = []
    room = _count(family.get("with_room"))
    for key, words in _CLASSES:
        number = _count(family.get(key))
        if not number:
            continue
        detail = ""
        if key == "with_room":
            spans = [f"{_length(window.get('window_seconds'))} {_shares(window)}% used, "
                     f"{_count(window.get('profiles'))} of {room} measured" for window in _dicts(family.get("windows"))]
            detail = f" ({'; '.join(spans)})" if spans else ""
        elif key == "at_limit" and family.get("earliest_reset"):
            detail = f" (earliest reset {_clock(family['earliest_reset'])})"
        bits.append(f"{number} {words}{detail}")
    for block in _dicts(family.get("model_scoped")):
        seen = f", seen {_clock(block['seen'])}" if block.get("seen") is not None else ""
        bits.append(f"{_models(block)} spent on {_count(block.get('profiles'))} "
                    f"(earliest reset {_clock(block.get('earliest_reset'))}{seen})")
    return f"{_clean(name)} " + ", ".join(bits)


def _shares(window: Mapping[str, Any]) -> str:
    least, greatest = _percent(window.get("least")), _percent(window.get("greatest"))
    return least if least == greatest else f"{least}–{greatest}"


def _percent(ratio: Any) -> str:
    """The engine's share as percent to at most two decimals, cut (never rounded up), trailing zeros dropped."""
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        return "?"
    value = (Decimal(repr(float(ratio))) * 100).quantize(Decimal("0.01"), rounding=ROUND_DOWN).normalize()
    return f"{value:f}"


def _length(seconds: Any) -> str:
    seconds = _seconds(seconds)
    for unit, size in (("d", 86_400), ("h", 3_600), ("m", 60), ("s", 1)):
        if seconds and seconds % size == 0:
            return f"{seconds // size}{unit}"
    return "window of unknown length"


def _models(block: Mapping[str, Any]) -> str:
    names = [*_strings(block.get("models")), *(f"{prefix}*" for prefix in _strings(block.get("prefixes")))]
    return _clean(",".join(names)) or "models the engine did not name"


def _clock(value: Any) -> str:
    at = _instant(value)
    return datetime.fromtimestamp(at, timezone.utc).strftime("%d.%m %H:%MZ") if at is not None else "an unreadable time"


def _instant(value: Any) -> Optional[float]:
    return _future_instant(value, float("-inf"))  # the core's parser, any instant


def census_signature(census: Mapping[str, Any]) -> str:
    """What the text states without its times or code: the read status (ok or failed) and the families."""
    return _signature("failed" if census.get("read") == "failed" else "ok", body_text(_mapping(census.get("families"))))


def _signature(read: str, body: str) -> str:
    return hashlib.sha256(f"{read}\n{body}".encode("utf-8")).hexdigest()[:20]


# ---- the file ------------------------------------------------------------------------------------------------------

def _data_dir(root: Any = None) -> pathlib.Path:
    if root is not None:
        return pathlib.Path(root)
    from ouroboros import config

    return pathlib.Path(config.DATA_DIR)


def _file_state(root: Any = None) -> Tuple[str, Optional[Dict[str, Any]]]:
    """``(state, record)``: ``absent``, ``unreadable`` (also oversized or foreign), ``pending`` (an account change
    awaits its read) or ``ok``. A bounded read that never raises."""
    path = _data_dir(root) / CENSUS_FILE
    try:
        with path.open("rb") as stream:
            data = stream.read(_MAX_BYTES + 1)
        record = json.loads(data.decode("utf-8")) if len(data) <= _MAX_BYTES else None
    except FileNotFoundError:
        return "absent", None
    except (OSError, UnicodeDecodeError, ValueError):
        log.debug("Subscription census unreadable", exc_info=True)
        return "unreadable", None
    if (not isinstance(record, dict) or record.get("type") != CENSUS_TYPE
            or not isinstance(record.get("families"), dict) or record.get("read") not in ("ok", "failed")):
        return "unreadable", None
    return ("pending" if record.get("pending") else "ok"), record


def read_file(root: Any = None) -> Optional[Dict[str, Any]]:
    """The stored census (pending included), or None when absent or unreadable."""
    return _file_state(root)[1]


def _write(record: Dict[str, Any], root: Any) -> None:
    from ouroboros.utils import atomic_write_json

    atomic_write_json(_data_dir(root) / CENSUS_FILE, record)  # the pytest live-data guard sits in its seam


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


# ---- the writer ----------------------------------------------------------------------------------------------------

def refresh(*, stop_requested: Optional[Callable[[], bool]] = None, root: Any = None) -> str:
    """One census pass of the census duty (``server_maintenance``), its outcome logged here (nothing in it raises into
    the duty's step helpers). Never raises."""
    try:
        outcome = _refresh(stop_requested or (lambda: False), root)
    except Exception:
        log.warning("Subscription census pass failed", exc_info=True)
        return "error"
    log.log(logging.INFO if outcome in _WRITES else logging.DEBUG, "subscription census pass: %s", outcome)
    return outcome


def _refresh(stop: Callable[[], bool], root: Any) -> str:
    from ouroboros.claudexor_daemon import owned_daemon_provisioned
    from ouroboros.gateway import claudexor_accounts, claudexor_passive

    if not owned_daemon_provisioned() and not (_data_dir(root) / CENSUS_FILE).exists():
        return "no_engine"  # never provisioned: nothing is read and no file appears
    revision = claudexor_accounts.account_read_revision()
    payload = claudexor_passive._quota_payload()  # discovery, roster and one quota epoch; no handshake or refresh
    if stop():
        return "stopped"
    now, reads = time.time(), _mapping(payload.get("reads"))
    if reads.get("accounts") != "ok" or reads.get("quota") != "ok":
        return _record_failure(_failure_code(payload), root)
    families = build_census(payload, now)
    with _LOCK:
        _PASS.update(confirmed_at=now, failures=0)
    if families is None:
        return "scope_unhonored"  # an older engine certifies no text scope: nothing to state, a known file stays
    return _publish(families, now, revision, claudexor_accounts.account_read_revision, stop, root)


def _failure_code(payload: Mapping[str, Any]) -> str:
    """The read's safe code (``passive_read_error``'s or the roster's typed refusal), discovery first."""
    errors = _mapping(payload.get("read_errors"))
    for facet in ("discovery", "accounts", "quota"):
        code = _mapping(errors.get(facet)).get("code")
        if isinstance(code, str) and _SAFE_CODE.fullmatch(code):
            return code
    return "read_failed"


def _record_failure(code: str, root: Any) -> str:
    """A failed read keeps the values, ``since`` and ``confirmed_at``; it is recorded on the second failed pass in a
    row (a first pass after a start can meet an engine still warming up), once until a read succeeds."""
    with _LOCK:
        _PASS["failures"] += 1
        if _PASS["failures"] < 2:
            return "failure_debounced"
        existing = read_file(root)
        if existing is None:
            return "failure_without_values"
        if existing["read"] == "failed":
            return "failure_unchanged"  # the code may alternate; it is printed, never compared
        confirmed = _PASS["confirmed_at"]
        record = {**existing, "read": "failed", "code": code,
                  "confirmed_at": _iso(confirmed) if confirmed is not None else existing.get("confirmed_at")}
        record["digest"] = census_signature(record)
        _write(record, root)
    return "failure_recorded"


def _publish(families: Dict[str, Any], now: float, revision: int, current_revision: Callable[[], int],
             stop: Callable[[], bool], root: Any) -> str:
    """Write the census when what it states changed (or a pending file awaits it); never over a confirmed account
    change that crossed this read (the accounts module's revision), which requested its own pass."""
    record: Dict[str, Any] = {"type": CENSUS_TYPE, "read": "ok", "code": None, "families": families}
    digest = census_signature(record)
    with _LOCK:
        if current_revision() != revision:
            return "roster_changed"
        existing = read_file(root)
        if existing is not None and not existing.get("pending") and census_signature(existing) == digest:
            return "unchanged"
        if stop():
            return "stopped"
        same = existing is not None and body_text(existing["families"]) == body_text(families)
        record.update(digest=digest, since=existing.get("since") if same and existing.get("since") else _iso(now),
                      confirmed_at=_iso(now))
        _write(record, root)
    if not same:
        return "written" if families else "none"
    return "recovered" if existing["read"] == "failed" else "confirmed"


# ---- requests and invalidation -------------------------------------------------------------------------------------

def request_refresh(reason: str, root: Any = None) -> None:
    """Ask the census duty for a pass, answered at most every ``REQUEST_MIN_SEC``: the request is a file whose mtime
    the duty compares with its last pass, so any process may ask. Never raises."""
    try:
        from ouroboros.utils import atomic_write_json, utc_now_iso

        atomic_write_json(_data_dir(root) / REQUEST_FILE, {"type": "subscription_census_request",
                                                           "reason": str(reason)[:64], "requested_at": utc_now_iso()})
    except Exception:
        log.debug("Subscription census refresh request not written", exc_info=True)


def note_limit_failure(code: Any) -> None:
    """A settled delegated run failed: a limit code asks for a fresh census (``LIMIT_FAILURE_CODES``)."""
    if code in LIMIT_FAILURE_CODES:
        request_refresh(str(code))


def invalidate(reason: str, root: Any = None) -> None:
    """A confirmed account disable or removal: the file turns pending at once, so the round keeps what is in view
    instead of restating a roster that changed, and a pass is requested. Never raises."""
    try:
        with _LOCK:
            existing = read_file(root)
            if existing is not None and not existing.get("pending"):
                _write({**existing, "pending": True}, root)
    except Exception:
        log.debug("Subscription census not marked pending", exc_info=True)
    request_refresh(reason, root)


def account_changed(enabled: Optional[bool]) -> None:
    """A confirmed Settings › Accounts change (``gateway.claudexor_accounts._account_changed``): a disable or removal
    invalidates, an enable only asks for a pass."""
    if enabled is True:
        request_refresh("account_enabled")
    else:
        invalidate("account_disabled" if enabled is False else "account_removed")


def request_mtime(root: Any = None) -> Optional[float]:
    try:
        return os.stat(_data_dir(root) / REQUEST_FILE).st_mtime
    except OSError:
        return None


def duty_due(now: float, last_pass: float, has_actors: bool, root: Any = None) -> bool:
    """The census duty's due condition: ``DUTY_SEC`` after the last pass while a live actor exists, or a refresh
    request newer than the last pass once ``REQUEST_MIN_SEC`` have passed since it."""
    if now - last_pass < REQUEST_MIN_SEC:
        return False
    if has_actors and now - last_pass >= DUTY_SEC:
        return True
    requested = request_mtime(root)
    return requested is not None and requested > last_pass


# ---- the round's part ----------------------------------------------------------------------------------------------

def quota_part(ctx: Any) -> Tuple[Optional[str], str]:
    """``(part, term)``: this round's part for the facts row's quota slot, or None, and the term the row's reuse key
    carries (the file's signature for a recipient, ``""`` for any other focus).

    Reads the census file only, never a connection. The part is due when what the file states differs from the
    latest part of this module in an intact facts row still in view, every facts row searched newest first, or when
    none is in view; times are not compared, so A→B→A shows B, then A. A stored «none» retracts a part in view once;
    a file gone absent or unreadable says once that the summary in view is no longer confirmed; a pending file (an
    account change awaiting its read) keeps what is in view.
    """
    try:
        if not chooses_executors(ctx):
            return None, ""
        state, census = _file_state()
        if state == "pending":
            return None, "pending:" + census_signature(census)
        latest = _latest_in_view(getattr(ctx, "messages", None) or [])
        if census is None:
            return (UNCONFIRMED if latest not in (None, _UNCONFIRMED) else None), state
        signature = census_signature(census)
        if not census["families"]:
            return (RETRACTION if latest not in (None, _RETRACTED) else None), signature
        return (census_text(census) if latest != signature else None), signature
    except Exception:
        log.debug("Subscription census part unavailable for the facts row", exc_info=True)
        return None, ""


def _latest_in_view(messages: List[Any]) -> Optional[str]:
    """The newest part of this module in an intact facts row in view: its signature, ``_RETRACTED`` or
    ``_UNCONFIRMED``; None when no facts row in view carries one."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, host_row_intact
    from ouroboros.context_view_facts import CONTEXT_FACTS_NAME

    marker = f"{_SLOT}({LABEL}"
    for row in reversed(messages):
        content = row.get("content") if isinstance(row, dict) else None
        if (isinstance(content, str) and marker in content and row.get(HOST_CONTEXT_KIND_KEY) == CONTEXT_FACTS_NAME
                and host_row_intact(row)):
            for segment in content.split("\n", 1)[0].split(" | "):
                if segment.startswith(marker):
                    return _part_signature(segment[len(_SLOT):])
    return None


def _part_signature(text: str) -> str:
    if text == RETRACTION:
        return _RETRACTED
    if text == UNCONFIRMED:
        return _UNCONFIRMED
    head, _sep, body = text.partition("): ")
    return _signature("failed" if head.startswith(f"({LABEL}, engine not read") else "ok", body)


def chooses_executors(ctx: Any) -> bool:
    """Whether this focus may choose an executor, so the census concerns it: every integrating focus (Main, project
    roots, consciousness); a Presence task only when its ceiling grants an executor-starting or reviewer tool; a
    delegated child or nanny that may call ``review_change`` (an acting profile; a read-only child is refused there)
    or holds the delegation right."""
    from ouroboros.chronicle_store import DELEGATED_FOCUS_ROLES
    from ouroboros.memory_view import view_role

    tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    metadata = getattr(tool_ctx, "task_metadata", None)
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    contract = getattr(tool_ctx, "task_contract", None)
    contract = contract if isinstance(contract, Mapping) else _mapping(metadata.get("task_contract"))
    role = view_role({"metadata": metadata, "task_contract": dict(contract)})
    if role == "presence":
        from ouroboros.presence_authority import presence_ceiling_allows_tool, presence_ceiling_from_context

        ceiling = presence_ceiling_from_context(tool_ctx)
        return ceiling is None or any(presence_ceiling_allows_tool(ceiling, name) for name in _EXECUTOR_TOOLS)
    if role not in DELEGATED_FOCUS_ROLES:
        return True
    from ouroboros.tool_access import active_tool_profile

    acting = active_tool_profile(tool_ctx) == "acting_subagent"  # review_change is callable for this profile
    return acting or holds_delegation_right(contract.get("delegation_budget"))


def holds_delegation_right(budget: Any) -> bool:
    """The delegation right itself, without counting children (``tools/control_delegation.check_delegation_admission``
    reads the same facts at a real start, where the count binds too): ``may_delegate`` (omitted: legacy permissive),
    depth left (omitted or malformed: unknown, not spent) and the room's subagents mode, Off withholding it."""
    from ouroboros.contracts.task_contract import normalize_bool

    budget = budget if isinstance(budget, Mapping) else {}
    if not normalize_bool(budget.get("may_delegate", True)):
        return False
    try:
        if budget.get("depth_remaining") is not None and int(budget["depth_remaining"]) <= 0:
            return False
    except (TypeError, ValueError):
        pass
    from ouroboros.settings_integrity import effective_subagents_mode

    return effective_subagents_mode() != "off"


# ---- small readers -------------------------------------------------------------------------------------------------

def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _dicts(value: Any) -> List[dict]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _named(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _seconds(value: Any) -> int:
    """A window length in whole seconds; 0 for an unknown one."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return _count(value)


def _strings(value: Any) -> tuple:
    return tuple(item for item in value if isinstance(item, str)) if isinstance(value, list) else ()


def _state(row: Mapping[str, Any]) -> Any:
    return _mapping(row.get("availability")).get("state")


def _own_rows(rows: List[dict], harness: str, profile_id: str) -> List[dict]:
    """A profile's rows of a subscription credential route; a paid key's rows of the same id stay out."""
    return [row for row in rows if isinstance(row.get("subject"), dict)
            and row["subject"].get("harness") == harness and row["subject"].get("subject_id") == profile_id
            and row["subject"].get("credential_route") in SUBSCRIPTION_ROUTES]


def _clean(text: str) -> str:
    """An engine name inside the row's text: one line, no row separator."""
    return " ".join(str(text).replace("|", "/").split())
