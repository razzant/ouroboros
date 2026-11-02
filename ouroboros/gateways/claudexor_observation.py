"""Ordinary observation of delegated runs: ONE adapter over two engine surfaces.

Transport only, beside ``gateways/claudexor.py`` (docs/DEVELOPMENT.md "Gateway Boundary
Pattern"). A host watching a run needs three facts between the moments it must act: the
lifecycle state, whether a question is pending, and an opaque token that says "nothing of
that changed since you last looked". The engine's compact wait, ``POST /v2/runs/wait-many``,
answers exactly those for up to a thousand runs in one body-free read and holds the call
until something needs attention or its window ends; the full run detail (events, timeline,
artifacts) is an addressed read a consumer takes on a reason — attention, terminal, a person
opening the run — never as the heartbeat.

An engine may list that route and still serve it unrepaired (claudexor #513–#515: shutdown,
deadline passes, problem rows; the repair carries no marker in the handshake or the catalog,
and release numbers do not say which build ships it), so support is a contract: the catalog
lists the route AND the route accepts the problem acknowledgement (``seen: null``), which only
the repaired wait-many does. That is one side-effect-free probe per exact engine identity,
remembered like the event stream's catalog answer. Without the contract, or when the route
is missing (a 404 on the route is the engine's answer about the ROUTE, never about a run),
``wait_attention`` reads each run's detail once, as the old pollers did, and hands those
details back so a consumer keeps today's behaviour without a second code path. No probe reads
a timeout as "unsupported": a hole leaves support unknown and this beat reads as before.

Tokens are the engine's; a consumer echoes a row's token as ``seen`` only after it fetched
what the attention was about (pending rows, terminal detail), so an unfetched question can
never be acknowledged away. A problem row (a run the engine could not place) is answered at
once, not held on; after the addressed read that decides what the run is, the consumer
acknowledges the problem (``ACKNOWLEDGED_PROBLEM``, wire ``seen: null``) and the engine holds
again. Fallback rows carry a local token (``local:`` prefix) that is never sent to an engine.
Kinds this adapter does not know (a future wake reason or problem kind) are attention, never
a refusal of the whole answer: unknown is not no.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

WAIT_MANY_OPERATION = ("POST", "/v2/runs/wait-many")
# The contract probe's target: a run id no engine can own, acknowledged (``seen: null``) with a
# zero wait. A repaired engine answers 200 with that row's ``not_found`` problem; an engine
# without the repair answers 400 (``seen: null`` fails its schema). No run is touched.
PROBE_RUN_ID = "capability-probe"
# The engine's own bounds (packages/schema/src/run-observation.ts): targets per call,
# wait window and the shape of a token it minted.
WAIT_MANY_MAX_TARGETS = 1000
WAIT_MANY_MAX_WAIT_SEC = 50.0
_ENGINE_TOKEN = re.compile(r"^[A-Za-z0-9_-]{43}$")
_LOCAL_TOKEN = "local:"
# A consumer's ``seen`` for a problem row it has since read on its own: sent as ``seen: null``,
# which the repaired engine takes as "this problem is known" (run-wait-many.ts: a problem row
# wakes the call only while its ``seen`` is not null), so the call holds again.
ACKNOWLEDGED_PROBLEM = _LOCAL_TOKEN + "problem"
PROBLEMS = frozenset({"not_found", "snapshot_unavailable", "not_observed"})
WAKE_REASONS = frozenset({"attention", "problems", "timeout", "aborted"})
# The contract's answer per exact engine identity (version, build sha): an engine cannot
# change what its route accepts without changing, so one probe per identity per process (a
# same-build restart keeps it). The catalog itself is the event stream module's shared one.
_SUPPORT: Dict[Tuple[str, str], bool] = {}
_SUPPORT_LOCK = threading.Lock()


class ClaudexorObservationGateway:
    """The compact attention wait, as the owned HTTP client speaks it (mixed into the gateway)."""

    def wait_run_attention(self, targets: List[Dict[str, Any]], *, wait_ms: Optional[int],
                           timeout_sec: Optional[float]) -> Dict[str, Any]:
        """``POST /v2/runs/wait-many``: the engine's own result object, unprojected.

        ``wait_ms=None`` leaves the hold to the engine's default. The read bound covers the
        hold plus the caller's transport allowance: a completed pass may overrun the nominal
        window by itself.
        """
        from ouroboros.gateways.claudexor import DAEMON_BUSY_CODES, ClaudexorUnavailable

        hold = WAIT_MANY_MAX_WAIT_SEC if wait_ms is None else wait_ms / 1000.0
        bound = None if timeout_sec is None else float(timeout_sec) + hold
        request = {"targets": targets, **({} if wait_ms is None else {"waitMs": int(wait_ms)})}
        try:
            body = self._request("POST", "/v2/runs/wait-many", json_body=request, timeout_sec=bound)
        except ClaudexorUnavailable as exc:
            # A read in POST clothing: the engine's typed "cannot serve now" (its busy and
            # stopping codes, or any 503 it marks retryable) is the same quiet observation hole
            # it is for a GET. ``_problem`` stamps GETs only, since a lost answer to a mutating
            # POST never proves the mutation was not accepted.
            if exc.status_code == 503 and (exc.code in DAEMON_BUSY_CODES or exc.code == "daemon_stopping"
                                           or _retryable_refusal(exc)):
                exc.observation_timeout, exc.observation_reason, exc.observation_answered = True, exc.code, True
            raise
        if not isinstance(body, dict):
            raise ClaudexorUnavailable("malformed_response", "Claudexor wait-many returned no object")
        return body


@dataclass
class RunAttention:
    """One observed run: the engine's row, or the fallback's reading of its detail."""

    run_id: str
    status: Optional[str] = None
    pending: Optional[bool] = None
    token: Optional[str] = None
    problem: Optional[str] = None
    detail: Optional[Dict[str, Any]] = None  # the fallback's detail read, for the consumer's own projection
    error: Optional[BaseException] = None  # the fallback's received refusal for this run (a 404, a 5xx)

    @property
    def observed(self) -> bool:
        return self.problem is None and self.error is None

    @property
    def terminal(self) -> bool:
        from ouroboros.delegate_custody import TERMINAL_STATES

        return self.status in TERMINAL_STATES


@dataclass
class AttentionWake:
    """What one observation call learned. ``attention`` indexes ``rows``."""

    reason: str  # attention | problems | timeout | aborted | polled | unobserved
    rows: List[RunAttention]
    attention: List[int] = field(default_factory=list)
    source: str = "wait_many"  # wait_many | detail
    waited: bool = False  # the engine held the call (a window, or until attention); a consumer sleeps nothing
    hole: Optional[BaseException] = None  # unobserved: the typed hole nothing was learned through

    def needs_attention(self, index: int) -> bool:
        return index in self.attention


def wait_many_supported(gateway: Any, *, timeout_sec: Optional[float]) -> bool:
    """Does the handshaken engine behind ``gateway`` serve the repaired compact wait?

    Support is the contract, not a release number: the catalog lists the route AND the
    route accepts the problem acknowledgement. The probe (``PROBE_RUN_ID``, ``seen: null``,
    zero wait) is asked once per exact engine identity. Only a RECEIVED answer is
    remembered: 200 with the ``not_found`` row (supported), a 400 or a 200 of another shape
    (the route rejects or predates the acknowledgement), a 404 (the route is gone). Anything
    else the probe meets — a hole (our bound expiring, a transport failure, the engine's busy,
    stopping or retryable 503) or any other refusal of it — leaves support unknown: nothing is
    remembered, this beat reads details as before (the old read says what the refusal is) and
    the next beat asks again. An unhandshaken transport, or a double without the seam, is ``False``
    without a request; the catalog read is the one shared with the event stream's question
    (``claudexor_run_events.route_listed``), and a hole on it propagates typed.
    """
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.gateways.claudexor_run_events import engine_identity, route_listed

    if not getattr(gateway, "engine_version", "") or not callable(getattr(gateway, "wait_run_attention", None)):
        return False
    key = engine_identity(gateway)
    with _SUPPORT_LOCK:
        if key in _SUPPORT:
            return _SUPPORT[key]
    supported = route_listed(gateway, WAIT_MANY_OPERATION, timeout_sec=timeout_sec)
    if supported:
        try:
            body = gateway.wait_run_attention([{"runId": PROBE_RUN_ID, "seen": None}], wait_ms=0,
                                              timeout_sec=timeout_sec)
        except ClaudexorUnavailable as exc:
            if int(getattr(exc, "status_code", 0) or 0) not in (400, 404):
                return False  # unknown: nothing remembered, today's reads this beat, asked again next beat
            supported = False  # received: the route rejects the acknowledgement, or is gone
        else:
            supported = _accepts_acknowledgement(body)
    with _SUPPORT_LOCK:
        _SUPPORT[key] = supported
    return supported


def _accepts_acknowledgement(body: Any) -> bool:
    """The repaired answer to the probe: one row, the ``not_found`` problem, nothing else claimed."""
    rows = body.get("summaries") if isinstance(body, dict) else None
    return isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict) and rows[0].get("problem") == "not_found"


def forget_support(gateway: Any) -> None:
    """The route is unusable on this exact engine (absent, or answering outside its contract):
    every consumer reads details as before until the engine identity (version, build sha)
    changes; a same-build restart of the engine keeps this process memo."""
    from ouroboros.gateways.claudexor_run_events import engine_identity

    with _SUPPORT_LOCK:
        _SUPPORT[engine_identity(gateway)] = False


def is_observation_hole(exc: BaseException) -> bool:
    """A read the daemon did not answer, or answered that it cannot serve now (its own typed
    busy/stopping/recovery 503s, or any 503 it marks retryable): nothing is known about the
    runs, nothing was claimed, and nothing is remembered of it."""
    from ouroboros.gateways.claudexor import DAEMON_BUSY_CODES

    code = str(getattr(exc, "code", "") or "")
    return (bool(getattr(exc, "observation_timeout", False)) or code in DAEMON_BUSY_CODES
            or code == "daemon_stopping" or _retryable_refusal(exc))


def _retryable_refusal(exc: BaseException) -> bool:
    """A received 503 whose problem says ``retryable: true`` (``store_closed``,
    ``store_maintenance_failed``): the engine's own word that the read is to be asked again."""
    problem = getattr(exc, "problem", None)
    return (int(getattr(exc, "status_code", 0) or 0) == 503 and isinstance(problem, dict)
            and problem.get("retryable") is True)


def detail_attention(run_id: str, detail: Dict[str, Any]) -> RunAttention:
    """The compact facts of one full detail, with a local token over exactly those facts."""
    from ouroboros.delegate_custody_usage import summary_of
    from ouroboros.gateways.claudexor import pending_interactions

    summary = summary_of(detail)
    status = str(summary.get("state") or "") or None
    ids = sorted(str(row.get("interaction_id") or "") for row in pending_interactions(detail))
    pending = bool(ids) or bool(summary.get("waitingOnUser"))
    digest = hashlib.sha256(json.dumps(["ouroboros.detail-observation", 1, run_id, status, pending, ids],
                                       sort_keys=True).encode("utf-8")).hexdigest()
    return RunAttention(run_id=run_id, status=status, pending=pending, token=_LOCAL_TOKEN + digest, detail=detail)


def wait_attention(gateway: Any, targets: Sequence[Tuple[str, Optional[str]]], *,
                   wait_sec: Optional[float], timeout_sec: Optional[float]) -> AttentionWake:
    """Observe ``targets`` (``(run_id, seen_token)`` pairs) once.

    On a supporting engine the call is held up to ``wait_sec`` (the engine caps it; ``None``
    leaves the hold to the engine's default) and returns the engine's rows and attention;
    ``timeout_sec`` bounds the transport beyond that hold. ``waited`` says the engine held
    the call (its window ran out, or attention ended it); an answer about an unacknowledged
    problem row, an abort or a zero wait is no hold, and the consumer sleeps its own beat.
    More targets than the engine takes in one call go in chunks; the hold is shared by the
    chunks and nothing holds past a chunk's notice. Elsewhere every target's detail is read
    once under ``timeout_sec`` and the rows are derived from those details (``waited`` is
    False: the consumer keeps its own beat). A typed hole (transport, busy, stopping) is
    returned as an ``unobserved`` wake, never raised, so a consumer treats it as the quiet
    hole it already knows; a received refusal about the READ itself (a schema or auth
    refusal) or an answer outside the contract propagates typed.
    """
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    pairs = [(str(run_id), seen) for run_id, seen in targets]
    if not pairs:
        return AttentionWake("timeout", [], waited=False)
    if not wait_many_supported(gateway, timeout_sec=timeout_sec):
        return _detail_wake(gateway, pairs, timeout_sec)
    wakes: List[AttentionWake] = []
    budget = wait_sec
    for start in range(0, len(pairs), WAIT_MANY_MAX_TARGETS):
        chunk = pairs[start:start + WAIT_MANY_MAX_TARGETS]
        wait_ms = None if budget is None else int(max(0.0, min(float(budget), WAIT_MANY_MAX_WAIT_SEC)) * 1000)
        began = time.monotonic()
        try:
            body = gateway.wait_run_attention([_target(run_id, seen) for run_id, seen in chunk],
                                              wait_ms=wait_ms, timeout_sec=timeout_sec)
        except ClaudexorUnavailable as exc:
            if int(getattr(exc, "status_code", 0) or 0) == 404:
                forget_support(gateway)  # the ROUTE is absent on this engine; no run was named absent
                return _detail_wake(gateway, pairs, timeout_sec)
            if is_observation_hole(exc):
                return _unobserved(pairs, exc, "wait_many")
            raise
        wake = _engine_wake(chunk, body, held=wait_ms is None or wait_ms > 0)
        wakes.append(wake)
        if wake.attention or wake.reason == "problems":
            budget = 0.0  # a notice is due now: the rest of the pass reads the present state
        elif budget is not None:
            budget = max(0.0, float(budget) - (time.monotonic() - began))
    return wakes[0] if len(wakes) == 1 else _joined(wakes)


def _target(run_id: str, seen: Optional[str]) -> Dict[str, Any]:
    if seen == ACKNOWLEDGED_PROBLEM:
        return {"runId": run_id, "seen": None}
    if isinstance(seen, str) and _ENGINE_TOKEN.match(seen):
        return {"runId": run_id, "seen": seen}
    return {"runId": run_id}


def _joined(wakes: List[AttentionWake]) -> AttentionWake:
    """One pass out of its chunks: rows in target order, attention re-indexed."""
    rows: List[RunAttention] = []
    attention: List[int] = []
    for wake in wakes:
        attention.extend(index + len(rows) for index in wake.attention)
        rows.extend(wake.rows)
    reasons = {wake.reason for wake in wakes}
    reason = ("attention" if attention else "problems" if "problems" in reasons
              else "aborted" if "aborted" in reasons else "timeout")
    return AttentionWake(reason, rows, attention=attention, waited=any(wake.waited for wake in wakes))


def _unobserved(pairs: List[Tuple[str, Optional[str]]], hole: BaseException, source: str) -> AttentionWake:
    """Nothing reached any row: every row is ``not_observed``, the hole rides the wake."""
    return AttentionWake("unobserved", [RunAttention(run_id, problem="not_observed") for run_id, _ in pairs],
                         source=source, hole=hole)


def _engine_wake(pairs: List[Tuple[str, Optional[str]]], body: Dict[str, Any], *, held: bool) -> AttentionWake:
    """The engine's answer projected. Structure outside the contract (a batch of the wrong
    size, indexes outside it, a token the engine could not have minted) is a typed refusal;
    a kind outside this adapter's vocabulary is not: an unknown problem kind stays a problem
    row (the addressed read decides what the run is) and an unknown wake reason names every
    row for one addressed read and promises no hold."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    def malformed(what: str) -> ClaudexorUnavailable:
        return ClaudexorUnavailable("malformed_response", f"Claudexor wait-many answered with {what}")

    reason = body.get("wakeReason")
    summaries = body.get("summaries")
    attention = body.get("attention")
    if not isinstance(reason, str) or not isinstance(summaries, list) or len(summaries) != len(pairs):
        raise malformed("no wake reason or a summary batch of the wrong size")
    if not isinstance(attention, list) or any(not isinstance(i, int) or isinstance(i, bool)
                                              or not 0 <= i < len(pairs) for i in attention):
        raise malformed("attention indexes outside the batch")
    rows: List[RunAttention] = []
    for (run_id, _seen), row in zip(pairs, summaries):
        if not isinstance(row, dict):
            raise malformed("a summary that is not an object")
        problem = row.get("problem")
        if problem is None:
            status, pending, token = row.get("status"), row.get("pending"), row.get("token")
            if not isinstance(status, str) or not isinstance(pending, bool) \
                    or not isinstance(token, str) or not _ENGINE_TOKEN.match(token):
                raise malformed("a summary without lifecycle, pending flag or token")
            rows.append(RunAttention(run_id=run_id, status=status, pending=pending, token=token))
        elif isinstance(problem, str) and problem:
            rows.append(RunAttention(run_id=run_id, problem=problem))
        else:
            raise malformed("a problem that is not a kind")
    if reason not in WAKE_REASONS:
        return AttentionWake(reason, rows, attention=list(range(len(rows))), waited=False)
    # The engine answers at once about an unacknowledged problem row, and when it is aborting.
    return AttentionWake(reason, rows, attention=sorted(set(attention)), waited=held and reason in ("attention", "timeout"))


def _detail_wake(gateway: Any, pairs: List[Tuple[str, Optional[str]]],
                 timeout_sec: Optional[float]) -> AttentionWake:
    """The engines without the compact wait: one detail read per target, rows derived from it."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    rows: List[RunAttention] = []
    attention: List[int] = []
    for index, (run_id, seen) in enumerate(pairs):
        try:
            detail = gateway.get_run(run_id, **({"timeout_sec": timeout_sec} if timeout_sec is not None else {}))
        except ClaudexorUnavailable as exc:
            if is_observation_hole(exc):
                return _unobserved(pairs, exc, "detail")
            if int(getattr(exc, "status_code", 0) or 0) >= 400:
                rows.append(RunAttention(run_id=run_id, problem="not_found" if exc.status_code == 404 else None,
                                         error=exc))
                continue
            raise
        row = detail_attention(run_id, detail if isinstance(detail, dict) else {})
        rows.append(row)
        if (row.pending or row.terminal) and row.token != seen:
            attention.append(index)
    reason = "attention" if attention else ("problems" if any(not row.observed for row in rows) else "polled")
    return AttentionWake(reason, rows, attention=attention, source="detail", waited=False)
