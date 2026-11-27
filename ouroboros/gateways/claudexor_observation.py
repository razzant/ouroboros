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

Attention is not a hold: ``question_attention`` judges a run's whole question set once, a
``delivery: async`` question leaves the agent working, and only a question that holds the
agent ends a reviewer slot (``review_execution._poll_session_terminal``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

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
# ``identity + ("blocking",)`` records, beside a supporting identity, whether the same probe
# was answered with the ``includeBlocking`` opt-in (claudexor #628), so later calls ask for it.
_SUPPORT: Dict[Tuple[str, ...], bool] = {}
_SUPPORT_LOCK = threading.Lock()
_RUN_READ = re.compile(r"/v2/runs/[^/]+")
_ABSENT_RUN_SITES: set = set()  # the code sites seen asking for a run the engine did not know


def tag_absent_run_read(response: Any, error: Any) -> None:
    """Name the code that read a run the engine does not know — a diagnostic, never a verdict.

    ``GET /v2/runs/:id`` answered 404 95,697 times on 01.11 (~2.9/s) from a poller nobody could
    name. The 404 proves nothing about the run here: completion, custody and ownership stay with
    their owners (``delegate_custody.daemon_says_absent`` decides absence on its own terms). The
    refusal carries ``caller_site``, the innermost frames outside the gateway package and the
    standard library, and each distinct site is logged once per process, so the caller is found
    without a line per poll."""
    try:
        request = response.request
        if response.status_code != 404 or request.method != "GET" or not _RUN_READ.fullmatch(request.url.path):
            return
        frames, frame = [], sys._getframe(1)
        while frame is not None and len(frames) < 3:
            module = str(frame.f_globals.get("__name__", ""))
            if not (module.startswith("ouroboros.gateways") or module == "ouroboros.observability"
                    or module.split(".")[0] in sys.stdlib_module_names):
                frames.append(f"{module}:{frame.f_code.co_name}:{frame.f_lineno}")
            frame = frame.f_back
        error.caller_site = " < ".join(frames)
        if error.caller_site not in _ABSENT_RUN_SITES:
            _ABSENT_RUN_SITES.add(error.caller_site)
            log.info("Claudexor answered 404 to GET /v2/runs/:id (a run it does not know) for %s", error.caller_site)
    except Exception:
        log.debug("absent-run read was not tagged", exc_info=True)  # the refusal stands as raised


def note_problem_provenance(response: Any, body: Any, error: Any) -> None:
    """Record parsed-JSON and HTTP route provenance on a gateway refusal without failing
    on an unattached ``httpx.Response`` or replacing an ordinary refusal."""
    error._problem_parsed = isinstance(body, dict)
    try:
        req = response.request
        url = getattr(req, "url", None) if req is not None else None
        error._response_route = (
            (str(req.method), str(url.path), bool(url.query))
            if req is not None and url is not None else None
        )
    except Exception:
        error._response_route = None


# Exact (engine version, full 40-char build sha) pairs proved to emit ``POST /v2/runs``
# 409 ``continuation_superseded`` strictly before run/job admission. Authority:
# ``ouroboros/claudexor_runtime_pin.json`` (6.4.2 / 3b8b87982a9362a27d4731f3e411437c0e63041a;
# 6.4.1 / d5a5c399bdc8ad40b95dde3add16b792e27d48d0, 6.4.0 / 87bd5757578754ee68285134adf1c026cfc81768 and
# 6.3.0 / 0ff3df717c2e807fdbd3a233487e35d81993fdea remain proven historical producers).
# No control-plane capability bit certifies arbitrary older, newer, or unhashed builds, so
# unproven producers stay outside this rule and fall back to unknown/pending custody.
# Maintenance: when advancing ``claudexor_runtime_pin.json``, verify that ``POST /v2/runs``
# still rejects ``continuation_superseded`` strictly pre-admission and add the new
# ``(version, build_sha)`` pair here (no new control-plane probe is required).
PRE_ADMISSION_CONTINUATION_PRODUCERS: frozenset[Tuple[str, str]] = frozenset({
    ("6.3.0", "0ff3df717c2e807fdbd3a233487e35d81993fdea"),
    ("6.4.0", "87bd5757578754ee68285134adf1c026cfc81768"),
    ("6.4.1", "d5a5c399bdc8ad40b95dde3add16b792e27d48d0"),
    ("6.4.2", "3b8b87982a9362a27d4731f3e411437c0e63041a"),
})


def verified_pre_admission_producer(
    gateway: Any, *, expected_generation: Any = None,
) -> Optional[Tuple[str, str]]:
    """Return ``(engine_version, engine_build_sha)`` only when THIS gateway's live
    descriptor generation completed a normal (non-``recovery_only``) handshake proving
    an allowlisted pre-admission ``continuation_superseded`` producer."""
    from ouroboros.gateways import claudexor_metadata as metadata

    generation = metadata.endpoint_generation(getattr(gateway, "_endpoint", None))
    if generation is None or (expected_generation is not None and generation != expected_generation):
        return None
    identity = (
        str(getattr(gateway, "engine_version", "") or ""),
        str(getattr(gateway, "engine_build_sha", "") or ""),
    )
    if identity not in PRE_ADMISSION_CONTINUATION_PRODUCERS:
        return None
    return identity if metadata.remembered_engine(gateway) == identity else None


@dataclass(frozen=True)
class PreAdmissionContinuationRejection:
    """Typed transport observation that a verified bound producer's direct ``POST /v2/runs``
    answered 409 ``continuation_superseded`` before admission for this continuation request.

    ``requested_predecessor_id`` is the ``continueFrom`` id sent on the wire; ``predecessor_run_id``
    is the ``context.runId`` the engine reported. They match on direct run-id continuations; when
    custody bound a queued predecessor by its ``jobId``, pinned Claudexor canonicalizes it to the
    predecessor's ``runId`` before pre-admission checks, and core must prove that alias in the same
    store before settling."""

    route: Tuple[str, str]
    status_code: int
    code: str
    predecessor_run_id: str
    engine_head: str
    store_id: str
    idempotency_key: str
    request_digest: str
    endpoint_generation: Tuple[Any, ...] = ()
    engine_ref: Tuple[Tuple[str, str], ...] = ()
    engine_version: str = ""
    engine_build_sha: str = ""
    requested_predecessor_id: str = ""

    @property
    def predecessor_matched(self) -> bool:
        return bool(self.predecessor_run_id) and self.requested_predecessor_id == self.predecessor_run_id

    def verified_on(self, gateway: Any) -> bool:
        """Whether this transport observation still holds for ``gateway``'s live descriptor
        generation, verified allowlisted producer, and serving store identity."""
        return (
            self.route == ("POST", "/v2/runs")
            and self.status_code == 409
            and self.code == "continuation_superseded"
            and bool(self.requested_predecessor_id)
            and self.requested_predecessor_id == self.requested_predecessor_id.strip()
            and bool(self.predecessor_run_id)
            and self.predecessor_run_id == self.predecessor_run_id.strip()
            and bool(self.engine_head)
            and self.engine_head == self.engine_head.strip()
            and bool(self.store_id)
            and self.store_id == self.store_id.strip()
            and bool(self.idempotency_key)
            and self.idempotency_key not in (self.requested_predecessor_id, self.predecessor_run_id, self.engine_head)
            and self.engine_head not in (self.requested_predecessor_id, self.predecessor_run_id)
            and bool(self.endpoint_generation)
            and verified_pre_admission_producer(gateway, expected_generation=self.endpoint_generation)
            == (self.engine_version, self.engine_build_sha)
            and str(getattr(gateway, "store_identity", "") or "") == self.store_id
        )


def canonical_request_digest(request: Any) -> str:
    """Deterministic SHA-256 digest of a canonical JSON run-start request body."""
    if not isinstance(request, dict) or not request:
        return ""
    try:
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return ""
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def pre_admission_continuation_snapshot(gateway: Any, request: dict) -> Tuple[Optional[tuple], str]:
    """Before a direct run POST, sample only a candidate for continuation proof.

    One generation read guards the handshake's store field, as ``gateway.store_identity`` does.
    The response still needs a fresh verified producer/store, and only custody knows this was the first send.
    """
    predecessor = request.get("continueFrom")
    if (not isinstance(predecessor, str) or not predecessor or predecessor != predecessor.strip()
            or request.get("_thread_id") or request.get("_use_thread")):
        return None, ""
    # A gateway without the optional proof metadata still uses the ordinary start path.
    store = getattr(gateway, "_store_identity", "")
    if not isinstance(store, str) or not store or (
            getattr(gateway, "engine_version", ""),
            getattr(gateway, "engine_build_sha", "")) not in PRE_ADMISSION_CONTINUATION_PRODUCERS:
        return None, ""
    from ouroboros.gateways import claudexor_metadata as metadata

    generation = metadata.endpoint_generation(getattr(gateway, "_endpoint", None))
    return generation, store if generation is not None else ""


def tag_pre_admission_continuation_rejection(
    gateway: Any,
    error: Any,
    *,
    request: Any,
    idempotency_key: str,
    before_generation: Any,
    before_store: str,
) -> None:
    """Attach ``pre_admission_continuation_rejection`` ONLY when a verified allowlisted
    producer's direct ``POST /v2/runs`` returned a parsed JSON 409 ``continuation_superseded``
    pre-admission problem for this exact continuation request on the same live descriptor
    generation and store.

    Preserves both the wire ``continueFrom`` id (``requested_predecessor_id``) and the
    engine's reported ``context.runId`` (``predecessor_run_id``); when they differ (e.g.
    ``jobId`` canonicalized to ``runId``), core must prove the same-store alias before settling.
    Any malformed, non-JSON, ambiguous, thread-routed, admitted-job, unproven-producer,
    or store-drifted response leaves ``error`` untagged so the start stays unknown.
    """
    if getattr(error, "_problem_parsed", False) is not True:
        return
    if getattr(error, "_response_route", None) != ("POST", "/v2/runs", False):
        return
    if int(getattr(error, "status_code", 0) or 0) != 409:
        return
    problem = getattr(error, "problem", None)
    if not isinstance(problem, dict):
        return
    if getattr(error, "code", "") != "continuation_superseded" or problem.get("code") != "continuation_superseded":
        return
    if problem.get("retryable") is not False:
        return
    context = problem.get("context")
    if not isinstance(context, dict):
        return
    if any(k in problem for k in ("jobId", "state", "handle", "runId")):
        return
    if any(k in context for k in ("jobId", "state", "handle")):
        return
    if not isinstance(request, dict) or request.get("_thread_id") or request.get("_use_thread"):
        return
    sent_pred = request.get("continueFrom")
    if not isinstance(sent_pred, str) or not sent_pred or sent_pred != sent_pred.strip():
        return
    ctx_run, ctx_head, wire_key = context.get("runId"), context.get("head"), str(idempotency_key or "")
    if (not isinstance(ctx_run, str) or not ctx_run or ctx_run != ctx_run.strip()
            or not isinstance(ctx_head, str) or not ctx_head or ctx_head != ctx_head.strip()
            or not wire_key or wire_key in (sent_pred, ctx_run, ctx_head)
            or ctx_head in (ctx_run, sent_pred)):
        return
    if before_generation is None:
        return
    producer = verified_pre_admission_producer(gateway, expected_generation=before_generation)
    after_store = str(getattr(gateway, "store_identity", "") or "")
    if (producer is None
            or not isinstance(before_store, str) or not before_store or before_store != before_store.strip()
            or after_store != before_store):
        return
    for container in (problem, context):
        if "storeId" in container and (not isinstance(container["storeId"], str) or container["storeId"] != before_store):
            return
    ref = getattr(gateway, "engine_ref", None)
    frozen_ref: Tuple[Tuple[str, str], ...] = ()
    if isinstance(ref, dict) and ref:
        if ref.get("kind") == "work" and ref.get("store") != before_store:
            return
        frozen_ref = tuple(sorted((str(k), str(v)) for k, v in ref.items()))
    digest = canonical_request_digest(request)
    if not digest:
        return
    error.pre_admission_continuation_rejection = PreAdmissionContinuationRejection(
        route=("POST", "/v2/runs"),
        status_code=409,
        code="continuation_superseded",
        predecessor_run_id=ctx_run,
        requested_predecessor_id=sent_pred,
        engine_head=ctx_head,
        store_id=before_store,
        idempotency_key=wire_key,
        request_digest=digest,
        endpoint_generation=before_generation,
        engine_ref=frozen_ref,
        engine_version=producer[0],
        engine_build_sha=producer[1],
    )


class ClaudexorObservationGateway:
    """The compact attention wait and the noncreating start lookup, as the owned HTTP client speaks them
    (mixed into the gateway)."""

    def wait_run_attention(self, targets: List[Dict[str, Any]], *, wait_ms: Optional[int],
                           timeout_sec: Optional[float], include_blocking: bool = False) -> Dict[str, Any]:
        """``POST /v2/runs/wait-many``: the engine's own result object, unprojected.

        ``wait_ms=None`` leaves the hold to the engine's default. The read bound covers the
        hold plus the caller's transport allowance: a completed pass may overrun the nominal
        window by itself. ``include_blocking`` asks a supporting engine to add ``blocking``
        to each row with pending questions; without it the request is byte-identical to the
        old one.
        """
        from ouroboros.gateways.claudexor import DAEMON_BUSY_CODES, ClaudexorUnavailable

        hold = WAIT_MANY_MAX_WAIT_SEC if wait_ms is None else wait_ms / 1000.0
        bound = None if timeout_sec is None else float(timeout_sec) + hold
        request = {"targets": targets, **({} if wait_ms is None else {"waitMs": int(wait_ms)}),
                   **({"includeBlocking": True} if include_blocking else {})}
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

    def lookup_run(self, request: Dict[str, Any], *, idempotency_key: str) -> Dict[str, Any]:
        """``POST /v2/runs/lookup``: the command a recorded ``POST /v2/runs`` holds, read with that start's own
        body and key; it never creates one. Every answer names its store unasked; it is returned unprojected."""
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        body = self._request("POST", "/v2/runs/lookup", json_body=dict(request),
                             headers={"Idempotency-Key": str(idempotency_key)})
        if not isinstance(body, dict):
            raise ClaudexorUnavailable("malformed_response", "Claudexor run lookup returned no object")
        return body


@dataclass
class RunAttention:
    """One observed run: the engine's row, or the fallback's reading of its detail."""

    run_id: str
    status: Optional[str] = None
    pending: Optional[bool] = None
    # Whether a pending question may hold the agent: False only when every pending question
    # is known async (the agent keeps working); None when unknown (an engine that does not
    # say, a problem row), which every consumer treats as the old blocking meaning.
    blocking: Optional[bool] = None
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
    supported, blocking = route_listed(gateway, WAIT_MANY_OPERATION, timeout_sec=timeout_sec), False
    if supported:
        probe = [{"runId": PROBE_RUN_ID, "seen": None}]
        try:
            # Asked with the opt-in first. An engine whose strict input predates it refuses the
            # whole request (400); the acknowledgement alone is then asked as before, so that
            # refusal is about the opt-in, never about support.
            body = gateway.wait_run_attention(probe, wait_ms=0, timeout_sec=timeout_sec, include_blocking=True)
        except ClaudexorUnavailable as exc:
            status = int(getattr(exc, "status_code", 0) or 0)
            if status not in (400, 404):
                return False  # unknown: nothing remembered, today's reads this beat, asked again next beat
            body = None
            if status == 400:
                try:
                    body = gateway.wait_run_attention(probe, wait_ms=0, timeout_sec=timeout_sec)
                except ClaudexorUnavailable as again:
                    if int(getattr(again, "status_code", 0) or 0) not in (400, 404):
                        return False  # unknown, as above
            # received: the route rejects the acknowledgement, or is gone, unless the old ask passed
            supported = body is not None and _accepts_acknowledgement(body)
        else:
            supported = blocking = _accepts_acknowledgement(body)
    with _SUPPORT_LOCK:
        _SUPPORT[key] = supported
        if supported:
            _SUPPORT[key + ("blocking",)] = blocking
    return supported


def reports_blocking(gateway: Any) -> bool:
    """Whether the supporting engine behind ``gateway`` answered the contract probe with the
    ``includeBlocking`` opt-in: only then do compact calls ask for it. A read of the memo the
    probe wrote, never a request; unknown (not yet probed, a hole) is ``False``."""
    from ouroboros.gateways.claudexor_run_events import engine_identity

    if not getattr(gateway, "engine_version", ""):
        return False
    key = engine_identity(gateway)
    with _SUPPORT_LOCK:
        return _SUPPORT.get(key) is True and _SUPPORT.get(key + ("blocking",)) is True


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


def question_attention(detail: Any, pending: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, bool]:
    """``(waiting, async_only)`` of one run detail, judged once over its COMPLETE question set.

    ``waiting`` is attention: a question awaits an answer (a row, or the summary's
    ``waitingOnUser``), never by itself the fact that the agent stopped. ``async_only`` says
    no pending question holds the agent: every row of a complete set is ``delivery: async``
    and the summary does not contradict it, or, with no complete rows to read, the summary's
    own ``asyncQuestionsOnly`` is true. A legacy row (no delivery: blocking), a kind this host
    does not know, a malformed or dropped row all keep the old blocking meaning. ``pending``
    is the caller's normalization of this same detail when it holds one: the whole set,
    never a bounded preview, which can shed the only blocking row."""
    from ouroboros.delegate_custody_usage import summary_of
    from ouroboros.gateways.claudexor import pending_interactions

    if not isinstance(detail, dict):
        return bool(pending), False
    rows = pending_interactions(detail) if pending is None else pending
    raw = detail.get("pendingInteractions")
    summary = summary_of(detail)
    if not (rows or summary.get("waitingOnUser") or (isinstance(raw, list) and raw)):
        return False, False
    said = summary.get("asyncQuestionsOnly")
    if rows and any(row.get("delivery") != "async" for row in rows):
        return True, False
    if rows and isinstance(raw, list) and len(raw) == len(rows):
        return True, said is not False
    return True, said is True


def detail_attention(run_id: str, detail: Dict[str, Any]) -> RunAttention:
    """The compact facts of one full detail, with a local token over exactly those facts."""
    from ouroboros.delegate_custody_usage import summary_of
    from ouroboros.gateways.claudexor import pending_interactions

    summary = summary_of(detail)
    status = str(summary.get("state") or "") or None
    rows = pending_interactions(detail)
    ids = sorted(str(row.get("interaction_id") or "") for row in rows)
    pending = bool(ids) or bool(summary.get("waitingOnUser"))
    digest = hashlib.sha256(json.dumps(["ouroboros.detail-observation", 1, run_id, status, pending, ids],
                                       sort_keys=True).encode("utf-8")).hexdigest()
    blocking = not question_attention(detail, rows)[1] if pending else None
    return RunAttention(run_id=run_id, status=status, pending=pending, blocking=blocking,
                        token=_LOCAL_TOKEN + digest, detail=detail)


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
    opt_in = {"include_blocking": True} if reports_blocking(gateway) else {}
    wakes: List[AttentionWake] = []
    budget = wait_sec
    for start in range(0, len(pairs), WAIT_MANY_MAX_TARGETS):
        chunk = pairs[start:start + WAIT_MANY_MAX_TARGETS]
        wait_ms = None if budget is None else int(max(0.0, min(float(budget), WAIT_MANY_MAX_WAIT_SEC)) * 1000)
        began = time.monotonic()
        try:
            body = gateway.wait_run_attention([_target(run_id, seen) for run_id, seen in chunk],
                                              wait_ms=wait_ms, timeout_sec=timeout_sec, **opt_in)
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
            # The opt-in's fact about this row's pending questions; absent is unknown (blocking).
            blocking = row.get("blocking")
            if blocking is not None and not isinstance(blocking, bool):
                raise malformed("a blocking fact that is not a boolean")
            rows.append(RunAttention(run_id=run_id, status=status, pending=pending,
                                     blocking=blocking if pending else None, token=token))
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
