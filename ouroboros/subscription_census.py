"""The subscription census: the engine's verdicts on this install's subscription accounts, counted per family,
kept as one dated file and shown in the round's facts row when its counts change (owner 21.11.2026 «1A»).

The engine owns every verdict (``claudexor_observations.quota_availability``, the reader Accounts, admission and the
primary observation share); this module only counts them. ``build_census`` is the pure count over one passive read
(``gateway/claudexor_passive``: the roster and one quota epoch, no handshake, no refresh) and ``census_text`` its
facts-only words. The census duty (``dispatch``, ticked by ``server_maintenance``) runs ``refresh`` every ``DUTY_SEC``
while someone works, or at the next tick after a refresh request, never sooner than ``REQUEST_MIN_SEC`` after its
previous pass (a run, thread turn, seat or model call refused or settled on a spent window or quota pool, a confirmed
account change; a plain rate limit asks nothing); each dispatch removes the request it answers, so a failed read is
retried on the cadence, never sooner, and a restart never answers it again. ``refresh`` rewrites ``CENSUS_FILE`` when
what the text states changed; its ``since`` moves only with the trigger (the counts, resets and model-scoped blocks;
share ranges and a stale block's ``seen`` are not in it). ``quota_part`` is the round's reader: the file alone, never
a connection, for the foci that may choose an executor, change-only against the latest part still in view. Nothing
here selects, starts, refreshes, switches or advises.
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
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from ouroboros.claudexor_observations import (
    _applicable_measurement, _future_instant, quota_availability, scope_honored,
)

log = logging.getLogger(__name__)

CENSUS_TYPE = "subscription_census"
CENSUS_FILE = "state/subscription_census.json"  # one record replaced in place: about 2 KB for 50 accounts
REQUEST_FILE = "state/subscription_census.request"  # a refresh request: its mtime against the duty's passes
DUTY_SEC = 600.0  # the duty's cadence while someone works; an idle install reads nothing
REQUEST_MIN_SEC = 15.0  # a refresh request is answered no sooner than this after the previous pass
_MAX_BYTES = 262_144  # the read bound of the file
# ``subject.credential_route`` of a vendor login; ``managed_api_key`` is a paid key and ``local`` a local model.
SUBSCRIPTION_ROUTES = frozenset({"vendor_native"})
# A run, seat or model call refused or settled on one of these engine codes (a spent window or quota pool) asks for a
# fresh census; a plain rate limit (``rate_limited``) does not: a burst of 429s on a loaded engine adds no reads.
LIMIT_FAILURE_CODES = frozenset({"vendor_limit_rejected", "subscription_window_exhausted",
                                 "credential_pool_exhausted"})
LABEL = "subscription accounts"
RETRACTION = f"({LABEL}, engine verdicts): no enabled subscription accounts now"
UNCONFIRMED = f"({LABEL}): the previous summary is no longer confirmed"
_SLOT = "quota "  # the facts row writes each part as ``quota <text>`` (``loop_messages.context_facts_line``)
_FAILED_HEAD = f"({LABEL}, engine not read since "
_SINCE = re.compile(r"counts unchanged since ([^;]*)")  # the part's ``since``, to the second, in either head
_ROOM_DETAIL = re.compile(r" with room \([^()]*\)")  # shares and coverage: shown, never a trigger
_SEEN = re.compile(r", seen [^)]*\)")  # a stale block's observation time: shown, never a trigger
_RETRACTED, _UNCONFIRMED = "retracted", "unconfirmed"
# A Presence task receives the part only when its ceiling grants one of these; an explicit denial withholds it.
_EXECUTOR_TOOLS = ("schedule_subagent", "delegate_start", "start_task", "review_change", "commit_reviewed",
                   "vcs_commit_reviewed", "preflight_review", "advisory_review", "plan_task")
_DELEGATION_TOOLS = ("schedule_subagent", "delegate_start")  # the entries the delegation right opens
# Counted classes in their text order; with room and at limit carry their own details.
_CLASSES = (("with_room", "with room"), ("at_limit", "at limit"), ("verification_failed", "verification failed"),
            ("no_measured_window", "available (no measured general window)"),
            ("no_quota_source", "no quota source"), ("not_freshly_read", "not freshly read"))
_SAFE_CODE = re.compile(r"[a-z0-9_]{1,64}")

_LOCK = threading.Lock()  # one publication, failure record or invalidation at a time in this process
# Memory only: the last accepted read, the start of the current failed streak and the duty's last dispatch (an
# attempt, which answers every request older than it, whatever the read's outcome).
_PASS: Dict[str, Any] = {"confirmed_at": None, "failing_since": None, "dispatched": 0.0}
_WRITES = frozenset({"written", "refreshed", "recovered", "confirmed", "none", "failure_recorded"})


# ---- the count -----------------------------------------------------------------------------------------------------

def verification_failed(status: Any) -> bool:
    """The Accounts rule (``web/modules/harness_accounts.js`` ``verificationBadge``'s error tone, which ``needsSignIn``
    reads): a reported verification that is neither ``passed`` nor ``not_run``."""
    value = status.get("verification") if isinstance(status, Mapping) else None
    return isinstance(value, str) and bool(value) and value not in ("passed", "not_run")


def _subscription_profiles(payload: Mapping[str, Any]) -> Iterator[Tuple[str, str, Mapping[str, Any]]]:
    """``(harness, profile_id, status)`` of the roster's enabled named profiles of a subscription credential kind
    (the core's include-list, ``gateway.onboarding._SUBSCRIPTION_CREDENTIAL_KINDS``)."""
    from ouroboros.gateway.onboarding import _SUBSCRIPTION_CREDENTIAL_KINDS

    for wrapper in _dicts(_mapping(payload.get("profiles")).get("profiles")):
        profile = _mapping(wrapper.get("profile"))
        harness, profile_id = profile.get("harness_id"), profile.get("profile_id")
        if (_named(harness) and _named(profile_id) and profile.get("enabled") is not False
                and profile.get("credential_kind") in _SUBSCRIPTION_CREDENTIAL_KINDS):
            yield harness, profile_id, _mapping(wrapper.get("status"))


def build_census(payload: Mapping[str, Any], now: float) -> Optional[Dict[str, Dict[str, Any]]]:
    """Each family's counted accounts from one passive read (``claudexor_passive._quota_payload``'s shape), or None
    when the engine did not certify the model-agnostic text scope (an older engine): there is then no census.

    Counted: the roster's subscription profiles (``_subscription_profiles``), each judged on its subscription-route
    rows and absences alone, so a paid key never gives a subscription account room. In this order: verification failed
    (the Accounts rule), the engine's verdict at the limit or with room, then the engine's unknowns by cause. Profiles
    with room add each window length's highest fresh measured share; model-scoped blocks still ahead count once per
    profile and model list. The result is JSON-safe: dicts keyed by family, lists, strings and numbers.
    """
    scope = payload.get("availability_scope")
    if not scope_honored({"availability_scope": scope}, None, "text"):
        return None
    snapshots, absences = _dicts(payload.get("quota")), _dicts(payload.get("quota_absences"))
    tallies: Dict[str, Dict[str, Any]] = {}
    for harness, profile_id, status in _subscription_profiles(payload):
        tally = tallies.setdefault(harness, {"counts": dict.fromkeys((key for key, _ in _CLASSES), 0), "resets": [],
                                             "windows": {}, "blocks": {}})
        rows, gone = _own_rows(snapshots, harness, profile_id), _own_rows(absences, harness, profile_id)
        for key, block in _profile_blocks(rows, now).items():
            tally["blocks"].setdefault(key, []).append(block)
        category, reset = _judge(rows, gone, status, harness, profile_id, scope, now)
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
    """A profile's model-scoped blocks still ahead, once per model list, as ``(instant, reset, seen)``: the earliest
    reset across its rows (a fresh reading breaks a tie), ``seen`` the observation time of that block's row when it
    is stale. The list is the engine's prefixes when it names any (shorter), else its sorted model names; passed or
    malformed resets drop (``_future_instant``, the core's parser)."""
    found: Dict[tuple, Tuple[float, bool, str, Optional[str]]] = {}
    for row in rows:
        fresh = row.get("freshness") == "fresh"
        for block in _dicts(_mapping(row.get("availability")).get("model_scoped_exhaustions")):
            at = _future_instant(block.get("resets_at"), now)
            if at is None:
                continue
            prefixes = tuple(sorted(_strings(block.get("applies_to_model_prefixes"))))
            key = (() if prefixes else tuple(sorted(_strings(block.get("applies_to_models")))), prefixes)
            if key not in found or (at, not fresh) < found[key][:2]:
                seen = None if fresh else (row.get("observed_at") if isinstance(row.get("observed_at"), str) else "")
                found[key] = (at, not fresh, block["resets_at"], seen)
    return {key: (at, reset, seen) for key, (at, _stale, reset, seen) in found.items()}


def _family(tally: Dict[str, Any]) -> Dict[str, Any]:
    family: Dict[str, Any] = dict(tally["counts"])
    family["earliest_reset"] = min(tally["resets"])[1] if tally["resets"] else None
    family["windows"] = [{"window_seconds": length, "least": min(shares), "greatest": max(shares),
                          "profiles": len(shares)} for length, shares in sorted(tally["windows"].items())]
    family["model_scoped"] = []
    for (models, prefixes), blocks in sorted(tally["blocks"].items()):
        _at, reset, seen = min(blocks, key=lambda block: (block[0], block[2] is not None))  # its own row's seen
        family["model_scoped"].append({"models": list(models), "prefixes": list(prefixes), "profiles": len(blocks),
                                       "earliest_reset": reset, "seen": seen})
    return family


# ---- the words -----------------------------------------------------------------------------------------------------

def census_text(census: Mapping[str, Any]) -> str:
    """The facts row's part without its slot word (the row writes ``quota <text>``): the engine's verdicts as counted,
    with the read that first gave these counts (to the second) and the read the shares come from; after a failed
    read, also when the engine was last read and why not."""
    code = census.get("code")
    times = f"counts unchanged since {_clock(census.get('since'), seconds=True)}; shares read {_clock(census.get('read_at'))}"
    if census.get("read") == "failed":
        code = code if isinstance(code, str) and _SAFE_CODE.fullmatch(code) else "read_failed"
        head = f"{_FAILED_HEAD}{_clock(census.get('confirmed_at'))} ({code}); {times})"
    else:
        head = f"({LABEL}, engine verdicts; {times})"
    return f"{head}: {body_text(_mapping(census.get('families')))}"


def body_text(families: Mapping[str, Any]) -> str:
    """The families as the text states them, in name order."""
    return "; ".join(_family_text(name, family) for name, family in sorted(families.items())
                     if isinstance(family, Mapping)) or "none"


def _trigger(body: str) -> str:
    """What makes the part due: the body without the with-room shares and coverage and without a stale block's
    ``seen`` (counts, resets and model-scoped blocks stay), read the same way from the file and from a part in view."""
    return _SEEN.sub(")", _ROOM_DETAIL.sub(" with room", body))


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
            detail = f" ({'; '.join(spans)})" if spans else " (no fresh measured window)"
        elif key == "at_limit" and family.get("earliest_reset"):
            detail = f" (earliest reset {_clock(family['earliest_reset'], nearest=True)})"
        bits.append(f"{number} {words}{detail}")
    for block in _dicts(family.get("model_scoped")):
        seen = f", seen {_clock(block['seen'])}" if block.get("seen") is not None else ""
        bits.append(f"{_models(block)} spent on {_count(block.get('profiles'))} "
                    f"(earliest reset {_clock(block.get('earliest_reset'), nearest=True)}{seen})")
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


def _clock(value: Any, *, nearest: bool = False, seconds: bool = False) -> str:
    """An instant to the minute (``seconds``: to the second, the counts epoch the comparison reads); an engine reset to
    the NEAREST minute (vendor resets jitter around :00)."""
    at = _instant(value)
    if at is None:
        return "an unreadable time"
    shown = datetime.fromtimestamp(round(at / 60) * 60 if nearest else at, timezone.utc)
    return shown.strftime("%d.%m %H:%M:%SZ" if seconds else "%d.%m %H:%MZ")


def _instant(value: Any) -> Optional[float]:
    return _future_instant(value, float("-inf"))  # the core's parser, any instant


def census_signature(census: Mapping[str, Any]) -> str:
    """What makes the part due: the read status (ok or failed), the trigger and its ``since`` as the text states it;
    the shares, ``read_at``, ``confirmed_at`` and the code are shown, never compared."""
    return _signature("failed" if census.get("read") == "failed" else "ok", _clock(census.get("since"), seconds=True),
                      _trigger(body_text(_mapping(census.get("families")))))


def _signature(read: str, since: str, trigger: str) -> str:
    return hashlib.sha256(f"{read}\n{since}\n{trigger}".encode("utf-8")).hexdigest()[:20]


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
    """One census pass of the census duty, its outcome logged here (nothing in it raises into the duty's step
    helpers). Never raises."""
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
    started = time.time()
    revision = claudexor_accounts.account_read_revision()
    payload = claudexor_passive._quota_payload()  # discovery, roster and one quota epoch; no handshake or refresh
    if stop():
        return "stopped"
    now, reads = time.time(), _mapping(payload.get("reads"))
    if reads.get("accounts") == "ok" and next(_subscription_profiles(payload), None) is None:
        families: Optional[Dict[str, Any]] = {}  # the roster answered: no subscription account, whatever the quota
    elif reads.get("accounts") != "ok" or reads.get("quota") != "ok":
        return _record_failure(_failure_code(payload), started, root)
    else:
        families = build_census(payload, now)
    if families is None:  # an older engine certifies no text scope: its verdicts are no census
        return _record_failure("availability_scope_unhonored", started, root)
    return _publish(families, now, revision, claudexor_accounts.account_read_revision, stop, root)


def _failure_code(payload: Mapping[str, Any]) -> str:
    """The read's safe code (``passive_read_error``'s or the roster's typed refusal), discovery first."""
    errors = _mapping(payload.get("read_errors"))
    for facet in ("discovery", "accounts", "quota"):
        code = _mapping(errors.get(facet)).get("code")
        if isinstance(code, str) and _SAFE_CODE.fullmatch(code):
            return code
    return "read_failed"


def _record_failure(code: str, started: float, root: Any) -> str:
    """A failed read keeps the values, ``since`` and ``confirmed_at``; it is recorded once a failed streak has lasted
    ``DUTY_SEC`` (an engine warming up after a start fails a pass or two), once until a read is accepted."""
    with _LOCK:
        if _PASS["failing_since"] is None:
            _PASS["failing_since"] = started
        if started - _PASS["failing_since"] < DUTY_SEC:
            return "failure_debounced"
        existing = read_file(root)
        if existing is None:
            return "failure_without_values"
        if existing["read"] == "failed":
            return "failure_unchanged"  # the code may alternate; it is printed, never compared
        confirmed = _PASS["confirmed_at"]
        _write({**existing, "read": "failed", "code": code, "confirmed_at": _iso(confirmed) if confirmed is not None
                else existing.get("confirmed_at") or existing.get("read_at")}, root)
    return "failure_recorded"


def _publish(families: Dict[str, Any], now: float, revision: int, current_revision: Callable[[], int],
             stop: Callable[[], bool], root: Any) -> str:
    """Accept the read and write the census when what it states changed (or a pending or failed file awaits it);
    never over a confirmed account change that crossed this read (the accounts module's revision), which requested
    its own pass. Only an accepted read that is stored (or already was) confirms: ``confirmed_at`` and the end of a
    failed streak."""
    body = body_text(families)
    with _LOCK:
        if current_revision() != revision:
            return "roster_changed"
        if stop():
            return "stopped"
        existing = read_file(root)
        if (existing is not None and existing["read"] == "ok" and not existing.get("pending")
                and body_text(existing["families"]) == body):
            _PASS.update(confirmed_at=now, failing_since=None)
            return "unchanged"
        same = existing is not None and _trigger(body_text(existing["families"])) == _trigger(body)
        _write({"type": CENSUS_TYPE, "read": "ok", "code": None, "families": families,
                "since": existing.get("since") if same and existing.get("since") else _iso(now), "read_at": _iso(now)},
               root)
        _PASS.update(confirmed_at=now, failing_since=None)  # after the write: a failed replace confirms nothing
    if not same:
        return "written" if families else "none"
    return "recovered" if existing["read"] == "failed" else "confirmed" if existing.get("pending") else "refreshed"


# ---- requests, invalidation and the duty ---------------------------------------------------------------------------

def request_refresh(reason: str, root: Any = None) -> None:
    """Ask the census duty for a pass: the request is a file whose mtime the duty compares with its passes, so any
    process may ask. A no-op when no engine was ever provisioned (nothing to read). Never raises."""
    try:
        from ouroboros.claudexor_daemon import owned_daemon_provisioned
        from ouroboros.utils import atomic_write_json, utc_now_iso

        if owned_daemon_provisioned():
            atomic_write_json(_data_dir(root) / REQUEST_FILE, {"type": "subscription_census_request",
                                                               "reason": str(reason)[:64], "requested_at": utc_now_iso()})
    except Exception:
        log.debug("Subscription census refresh request not written", exc_info=True)


def note_limit_failure(code: Any, problem: Any = None, root: Any = None) -> None:
    """A run, seat or model call refused or settled on a limit code (``LIMIT_FAILURE_CODES``) asks for a fresh census,
    except a pool the engine calls structural (its problem's ``context.poolCause``). Never raises."""
    if (isinstance(code, str) and code in LIMIT_FAILURE_CODES
            and _mapping(_mapping(problem).get("context")).get("poolCause") != "structural"):
        request_refresh(code, root)


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
    """A confirmed change in Settings › Accounts and keys (``gateway.claudexor_accounts._account_changed``): a disable
    or removal invalidates, an enable or a new account only asks for a pass."""
    if enabled is True:
        request_refresh("account_enabled")
    else:
        invalidate("account_disabled" if enabled is False else "account_removed")


def request_mtime(root: Any = None) -> Optional[float]:
    try:
        return os.stat(_data_dir(root) / REQUEST_FILE).st_mtime
    except OSError:
        return None


def duty_due(now: float, working: bool, root: Any = None) -> bool:
    """The census duty's due condition, never sooner than ``REQUEST_MIN_SEC`` after its last dispatch: ``DUTY_SEC``
    after it while someone works (a running task or a direct actor; a pending-only queue reads nothing), or a refresh
    request on disk. Each dispatch removes the request it answers (``dispatch``), failed read or not, so an engine
    that does not answer is asked again on the cadence or a new request, never in a 15 s loop, and a restart never
    answers an answered request again. Only an engine once provisioned, or a census file to keep, has anything to
    read."""
    last = _PASS["dispatched"]
    if now - last < REQUEST_MIN_SEC:
        return False
    due = (working and now - last >= DUTY_SEC) or request_mtime(root) is not None
    if not due:
        return False
    from ouroboros.claudexor_daemon import owned_daemon_provisioned

    return owned_daemon_provisioned() or (_data_dir(root) / CENSUS_FILE).exists()


def dispatch(now: float, working: bool, latch: Any, root: Any, start: Callable[[], Any],
             busy: Callable[[Any, float], Any]) -> None:
    """The census duty's tick (``server_maintenance``): when due and its latch is free, stamp, remove the request the
    pass answers (one written during the read stays for the next) and ``start`` one pass (``duty_pass``); a due pass
    whose latch is still held reports the stall (``busy``)."""
    if not duty_due(now, working, root):
        return
    if latch.acquire(blocking=False):
        _PASS["dispatched"] = now
        try:
            os.unlink(_data_dir(root) / REQUEST_FILE)
        except OSError:
            pass  # none on disk (a cadence pass), or not removable: the next dispatch tries again
        start()
    else:
        busy(latch, DUTY_SEC)


def duty_pass(stop_requested: Callable[[], bool], latch: Any, root: Any = None) -> str:
    """One pass of the census duty, then its latch goes back; a closed generation reads nothing. Never raises."""
    try:
        return "stopped" if stop_requested() else refresh(stop_requested=stop_requested, root=root)
    finally:
        latch.release()


# ---- the round's part ----------------------------------------------------------------------------------------------

def quota_part(ctx: Any) -> Tuple[Optional[str], str]:
    """``(part, term)``: this round's part for the facts row's quota slot, or None, and the term the row's reuse key
    carries (the file's signature for a recipient, ``""`` for any other focus).

    Reads the census file only, never a connection. The part is due when what the file states (its read status, its
    trigger and that trigger's ``since``) differs from the latest part of this module in an intact facts row still in
    view, every facts row searched newest first, or when none is in view; so A→B→A shows B, then A, also once B's
    row has left the view. A stored «none» retracts a part in view once; a failed read of an empty roster says so
    once where a part is in view; a file gone absent or unreadable, or a pending file whose read failed, says once
    that the summary in view is no longer confirmed; a pending file (an account change awaiting its read) otherwise
    keeps what is in view.
    """
    try:
        if not chooses_executors(ctx):
            return None, ""
        state, census = _file_state()
        if census is None:
            latest = _latest_in_view(getattr(ctx, "messages", None) or [])
            return (UNCONFIRMED if latest not in (None, _UNCONFIRMED) else None), state
        signature = census_signature(census)
        if state == "pending" and census["read"] != "failed":
            return None, signature
        latest = _latest_in_view(getattr(ctx, "messages", None) or [])
        if state == "pending":
            return (UNCONFIRMED if latest not in (None, _UNCONFIRMED) else None), signature
        if not census["families"] and census["read"] != "failed":
            return (RETRACTION if latest not in (None, _RETRACTED) else None), signature
        if not census["families"] and latest is None:
            return None, signature  # a failed read of an empty roster tells a fresh view nothing
        return (census_text(census) if latest != signature else None), signature
    except Exception:
        log.debug("Subscription census part unavailable for the facts row", exc_info=True)
        return None, ""


def _latest_in_view(messages: List[Any]) -> Optional[str]:
    """The newest part of this module in an intact facts row in view: its signature, ``_RETRACTED`` or
    ``_UNCONFIRMED``; None when no facts row in view carries one. The row's kind is checked first; its whole text
    is searched (an earlier part, an owner's quote say, may span lines)."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, host_row_intact
    from ouroboros.context_view_facts import CONTEXT_FACTS_NAME

    marker = f"{_SLOT}({LABEL}"
    for row in reversed(messages):
        if not isinstance(row, dict) or row.get(HOST_CONTEXT_KIND_KEY) != CONTEXT_FACTS_NAME:
            continue
        content = row.get("content")
        if isinstance(content, str) and marker in content and host_row_intact(row):
            for segment in content.split(" | "):
                if segment.startswith(marker):
                    return _part_signature(segment.split("\n", 1)[0][len(_SLOT):])
    return None


def _part_signature(text: str) -> str:
    if text == RETRACTION:
        return _RETRACTED
    if text == UNCONFIRMED:
        return _UNCONFIRMED
    head, _sep, body = text.partition("): ")
    since = _SINCE.search(head)
    return _signature("failed" if head.startswith(_FAILED_HEAD) else "ok", since.group(1) if since else "",
                      _trigger(body))


def chooses_executors(ctx: Any) -> bool:
    """Whether this focus may choose an executor, so the census concerns it: every integrating focus (Main, project
    roots, consciousness); a Presence task when its ceiling grants an executor-starting or reviewer tool; a delegated
    child or nanny that may call ``review_change`` (an acting profile) or holds the delegation right with an entry
    it opens left (the scheduler or a direct start). A tool the task's contract withholds (``disabled_tools``) grants
    nothing."""
    from ouroboros.chronicle_store import DELEGATED_FOCUS_ROLES
    from ouroboros.memory_view import view_role

    tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    metadata = getattr(tool_ctx, "task_metadata", None)
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    contract = getattr(tool_ctx, "task_contract", None)
    contract = contract if isinstance(contract, Mapping) else _mapping(metadata.get("task_contract"))
    role = view_role({"metadata": metadata, "task_contract": dict(contract)})
    if role != "presence" and role not in DELEGATED_FOCUS_ROLES:
        return True
    from ouroboros.tools.registry_guards import _disabled_tools

    denied = _disabled_tools(tool_ctx)
    if role == "presence":
        from ouroboros.presence_authority import presence_ceiling_allows_tool, presence_ceiling_from_context

        ceiling = presence_ceiling_from_context(tool_ctx)
        return any(name not in denied and (ceiling is None or presence_ceiling_allows_tool(ceiling, name))
                   for name in _EXECUTOR_TOOLS)
    from ouroboros.tool_access import active_tool_profile

    acting = active_tool_profile(tool_ctx) == "acting_subagent" and "review_change" not in denied
    return acting or (any(name not in denied for name in _DELEGATION_TOOLS)
                      and holds_delegation_right(contract.get("delegation_budget")))


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
