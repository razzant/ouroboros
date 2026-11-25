"""The accepted work behind one recorded delegated start, read without creating any (#3138).

A start whose POST answer was lost may have been accepted. Claudexor answers that question through
two declared, noncreating reads that take the start's own recorded body and wire key (the invocation
id): ``POST /v2/runs/lookup`` for a run start and ``POST /v2/threads/:id/turns/lookup`` for a thread
turn, whose body is the projection the append sent (``review_thread_continuity.thread_turn_request``).
``locate`` negotiates that route from the serving engine's catalog, sends exactly that request to the
engine the invocation recorded and validates the engine's union:

- ``found``: the accepted command (job id; run id or null; command state, terminal ones included; the
  failure facts the job recorded as ``handle.problem``, kept whole);
- ``turn_only``: an accepted thread turn with no retained command (run id or null; its enqueue error);
- ``not_found``: this store's current partition holds no matching binding.

Everything else is a typed gap: an engine that does not declare the route (``unsupported``, an old
route's bare 404 and a build without durable turn reads, 501, too), a transport fault, a 503 or an
answer missing its store identity (``unavailable``), a
typed 4xx such as a request/key conflict, a missing project or thread, or a continuation predecessor no
longer retained, 410 (``refused``), and a binding in a store other than the recorded admitting one
(``store_mismatch``). No outcome proves that the original POST created nothing, and none authorizes a
POST: a miss leaves the start pending and its outcome unknown; a refusal or a foreign store forbids any
replay, other misses leave it to ``delegate_pending.replay_store_unproven``. A positive binding names
the store that accepted the work, its own receipt (``storeId``); a handshake never does.

``bind_started`` is where the STARTED writers of one invocation converge (the start's own receipt, a
review seat's, every read here: explicit retry, sweep, a seat's rejoin, a start or replay whose POST the
engine answered with an error, ``accepted_after_refusal``): under the actor decision lock a
later writer adopts the run its invocation already holds, whichever engine name (job or run id) the
first writer knew it by, so one accepted command keeps one custody and cost row. ``recover`` reads and
binds; observation, settlement and any stop stay with their existing owners. Nothing here lifts a
Pause, sends a model request, applies a patch or repeats a review.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, NamedTuple, Optional, Tuple

log = logging.getLogger(__name__)

RUN_LOOKUP = ("POST", "/v2/runs/lookup")
TURN_LOOKUP = ("POST", "/v2/threads/:id/turns/lookup")
_HANDLE_FACTS = ("jobId", "runId", "state", "runDir", "taskId", "threadId", "turnId", "problem")
# The engine's ``ControlRunState``: a found job's ``state`` is one of these or the answer is malformed.
RUN_STATES = frozenset({"queued", "running", "succeeded", "failed", "cancelled", "interrupted"})
# ``recover`` reasons that keep a start pending with no replay at all: work the engine accepted that
# custody cannot bind now (a runless turn, an unavailable decision), an answer from a store this start is
# not for, or a typed refusal to read it (the same body's create would be refused alike, a definite failure
# retiring a start that may have been accepted). Any other miss keeps the send rule (``replay_store_unproven``).
NO_REPLAY = frozenset({"accepted_turn_without_run", "accepted_start_unbound", "accepted_start_store_mismatch",
                       "accepted_start_refused"})


class Lookup(NamedTuple):
    """One validated lookup answer. ``run_id`` is the engine name custody binds: a found command's run id,
    else its job id (the engine addresses a run by either); a retained turn's run id; else ""."""

    outcome: str
    store_id: str = ""
    run_id: str = ""
    state: str = ""
    facts: Optional[Dict[str, Any]] = None
    detail: str = ""

    @property
    def bindable(self) -> bool:
        return self.outcome in ("found", "turn_only") and bool(self.run_id)


class Bound(NamedTuple):
    custody: Any  # the run custody now holding the invocation; None when nothing could be bound
    durable: bool  # its STARTED row is durable
    adopted: bool  # an earlier writer had bound the invocation: this call bound no other run


def _field(container: Dict[str, Any], key: str, *, nullable: bool = False) -> Optional[str]:
    """A required field of the strict union: its nonempty string, "" for an explicit null where the schema
    allows one, ``None`` (malformed) when it is missing, null where none is allowed, or not a string."""
    if key not in container:
        return None
    value = container[key]
    if value is None:
        return "" if nullable else None
    return value if isinstance(value, str) and value else None


def locate(gateway: Any, record: Dict[str, Any]) -> Lookup:
    """Read the accepted work behind ``record`` (an invocation record with its resolved ``request``)."""
    from ouroboros.execution_domain import is_peer_ref, recorded_ref
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    body, key = record.get("request"), str(record.get("invocation_id") or "")
    if not isinstance(body, dict) or not body or not key:
        return Lookup("unavailable", detail="the recorded request or its invocation key is unavailable")
    if is_peer_ref(recorded_ref(record)) or is_peer_ref(getattr(gateway, "engine_ref", None)):
        return _locate_peer(gateway, key)
    thread_id = str(body.get("_thread_id") or "")
    method, path = TURN_LOOKUP if thread_id else RUN_LOOKUP
    try:
        catalog = gateway.operations() if callable(getattr(gateway, "operations", None)) else []
    except ClaudexorUnavailable as exc:
        return Lookup("unavailable", detail=f"the engine catalog was not read: {exc.code}")
    if not any(isinstance(row, dict) and row.get("method") == method and row.get("path") == path
               for row in catalog or ()):
        return Lookup("unsupported", detail=f"the serving engine does not declare {method} {path}")
    try:
        if thread_id:
            from ouroboros.review_thread_continuity import thread_turn_request

            answer = gateway.lookup_thread_turn(thread_id, thread_turn_request(body), idempotency_key=key)
        else:
            answer = gateway.lookup_run(dict(body), idempotency_key=key)
    except ClaudexorUnavailable as exc:
        status = int(getattr(exc, "status_code", 0) or 0)
        outcome = ("unsupported" if status == 501 or (status == 404 and exc.code == "http_404") else
                   "refused" if 400 <= status < 500 else "unavailable")
        return Lookup(outcome, detail=f"{exc.code}: {exc}"[:500])
    return _validated(answer, record, thread_id, getattr(gateway, "engine_ref", None))


def _locate_peer(gateway: Any, letter_id: str) -> Lookup:
    """Non-creating read of a peer letter (``GET /api/letters/{id}``): the receiver answers it while its side is
    paused, while a Pause of my own place for the link leaves it ``unavailable`` here (only a Stop's read crosses it)."""
    from ouroboros.delegate_custody_usage import summary_of
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        detail = gateway.get_run(letter_id)
    except ClaudexorUnavailable as exc:
        status = int(getattr(exc, "status_code", 0) or 0)
        named_refusal = str(getattr(exc, "refusal_code", "") or "")
        if status == 404 and (named_refusal or exc.code) == "letter_not_found":
            return Lookup("not_found")
        if named_refusal and 400 <= status < 500:
            return Lookup("refused", detail=f"{exc.code}: {exc}"[:500], facts={"refusal_code": named_refusal})
        return Lookup("unavailable", detail=f"{exc.code}: {exc}"[:500])
    state = str(summary_of(detail).get("state") or "")
    if not isinstance(detail, dict) or str(detail.get("runId") or "") != letter_id or state not in RUN_STATES:
        return Lookup("unavailable", detail="malformed peer letter lookup answer")
    return Lookup("found", run_id=letter_id, state=state, facts={"letterId": letter_id, "state": state})


def _validated(answer: Any, record: Dict[str, Any], thread_id: str, engine: Any) -> Lookup:
    def malformed(what: str) -> Lookup:
        return Lookup("unavailable", detail=f"malformed lookup answer: {what}")

    if not isinstance(answer, dict) or answer.get("status") not in ("found", "turn_only", "not_found"):
        return malformed("no known status")
    store = answer.get("storeId")
    if not isinstance(store, str) or not store:
        return Lookup("unavailable", detail="the lookup answer names no store, so it speaks for none")
    admitted = str(record.get("daemon_store_id") or "")
    proven = str(engine.get("store") or "") if isinstance(engine, dict) else ""
    if (admitted and admitted != store) or (proven and proven != store):
        return Lookup("store_mismatch", store_id=store,
                      detail=f"store {store} answered, but this start belongs to store {admitted or proven}")
    if answer["status"] == "not_found":
        return Lookup("not_found", store_id=store)
    if answer["status"] == "turn_only":
        turn, thread, run = _field(answer, "turnId"), _field(answer, "threadId"), _field(answer, "runId", nullable=True)
        if not turn or not thread or run is None or thread != thread_id:  # a run start has no turn to answer
            return malformed("turn_only without the asked thread, its turn and a run id or explicit null")
        facts = {"threadId": thread, "turnId": turn, "runId": run or None,
                 "enqueueError": copy.deepcopy(answer.get("enqueueError"))}
        return Lookup("turn_only", store_id=store, run_id=run, facts=facts)
    handle = answer.get("handle")
    if not isinstance(handle, dict):
        return malformed("found without a handle")
    job, run, state = _field(handle, "jobId"), _field(handle, "runId", nullable=True), handle.get("state")
    if not job or run is None or not isinstance(state, str) or state not in RUN_STATES or (
            thread_id and handle.get("threadId") not in (None, thread_id)):
        return malformed("found without a job id, a run id or explicit null, and a known command state")
    facts = {key: copy.deepcopy(handle[key]) for key in _HANDLE_FACTS if key in handle}  # ``problem``: the job's own
    return Lookup("found", store_id=store, run_id=run or job, state=state, facts=facts)


def bind_started(drive_root: Any, entry: Any, *, shape: Optional[Dict[str, Any]] = None,
                 exclusive: bool = False, observed_state: str = "", accepted_only: bool = False) -> Bound:
    """Write ``entry``'s STARTED row unless its invocation already holds a run.

    Under the actor decision lock (no network inside it) the invocation's record decides: a run it
    already holds is adopted. A recovery (``exclusive``) then writes nothing, the earlier row being the
    start's own or an equal recovery; the start's own receipt instead records its facts on that run's
    row (an idempotent STARTED of the same run merges, first-wins), so it keeps one row whichever name
    it was answered with. No STARTED is published outside that lock, where it could rival a recovery
    holding it: when the lock or the record cannot be read, a recovery binds nothing, and the start's
    own receipt keeps its live run in this process only, as a STARTED whose publication failed
    (``started_uncustodied`` with its retry token); the invocation stays pending for a read to bind once,
    and that run's money waits for the bound row (``deferred_money``).
    ``accepted_only`` lets Stop republish its known physical handle only under the recorded decision.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import run as current_run
    from ouroboros.delegate_registration_policy import bind_registration_owner

    invocation_id, task_id = str(getattr(entry, "invocation_id", "") or ""), str(getattr(entry, "task_id", "") or "")
    if invocation_id and not task_id and entry.project_id:
        try:
            prior = custody.invocation_record(drive_root, invocation_id) or {}
            bind_registration_owner(drive_root, entry, prior)
        except Exception:
            log.warning("Cannot bind the registration of %s", invocation_id, exc_info=True)
            custody.record_started(drive_root, entry, shape=shape, publish=False)
            return Bound(entry, False, False)
    if not invocation_id or not task_id:
        return Bound(entry, bool(custody.record_started(drive_root, entry, shape=shape)), False)
    decided = False
    try:
        with custody.actor_decision_lock(drive_root, task_id):
            prior = custody.invocation_record(drive_root, invocation_id) or {}
            held = str(prior.get("run_id") or "") if prior.get("state") == "started" else ""
            existing = current_run(drive_root, held) if held else None
            decided = True
            if accepted_only and (not prior.get("resume_decisions") or (not held and prior.get("state") != "pending")):
                return Bound(None, False, False)
            if prior and (prior.get("task_id") != task_id
                          or (prior.get("root_task_id") or task_id) != (entry.root_task_id or task_id)
                          or existing and (existing.task_id != task_id or existing.invocation_id != invocation_id)):
                return Bound(None if exclusive else entry, False, False)
            if held and existing is None:
                entry = copy.copy(entry)
                entry.run_id = held
                return Bound(None if exclusive else entry, False, True)
            if prior.get("resume_decisions"):
                from ouroboros.delegate_resume import observe_bound_state

                if existing is not None:
                    if observe_bound_state(drive_root, existing, observed_state) is False:
                        return Bound(existing, False, True)
                else:
                    shape = {**(shape or {}), "resume_start_state": observed_state if observed_state in RUN_STATES else "unknown",
                             "root_task_id": prior.get("root_task_id") or task_id}
            if existing is not None and exclusive:
                return Bound(existing, True, True)
            if existing is not None:  # the caller's object stays as it was: a copy takes the held run's name
                entry = copy.copy(entry)
                entry.run_id = existing.run_id
                entry.daemon_store_id = existing.daemon_store_id or entry.daemon_store_id
                entry.project_owned = existing.project_owned and entry.project_owned
                entry.project_resolution = existing.project_resolution
            if not entry.project_resolution:
                decided = False  # an owner that cannot be bound publishes nothing; the invocation stays pending
                bind_registration_owner(drive_root, entry, prior)
                decided = True
            return Bound(entry, bool(custody.record_started(drive_root, entry, shape=shape)), existing is not None)
    except Exception:
        if decided:
            raise  # the decision was taken and its write failed loudly: never write it twice
        log.warning("STARTED of %s not published: its invocation record or decision lock is unavailable",
                    invocation_id, exc_info=True)
    if exclusive:
        return Bound(None, False, False)
    custody.record_started(drive_root, entry, shape=shape, publish=False)  # this process's handle, no rival row
    return Bound(entry, False, False)


def deferred_money(drive_root: Any, entry: Any) -> Dict[str, str]:
    """Why ``entry``'s money waits for its invocation's STARTED row; ``{}`` when it is recorded now.

    An invocation's money is recorded once, on its one durable STARTED row. A run known only in this
    process (its STARTED unpublished: a failed emit, a lock ``bind_started`` could not take) records
    none under its own name while that binding is missing, pending or unreadable, since a read may still
    bind the accepted work under its other engine name, nor once it is bound under that other name: the
    bound row's settlement (body observer, sweep) records it. The handle and its terminal answer stay
    as they were. A durable row, a legacy run without invocation, or a binding of this very name records.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import run as current_run

    invocation_id, run_id = str(getattr(entry, "invocation_id", "") or ""), str(getattr(entry, "run_id", "") or "")
    if not invocation_id or not run_id:
        return {}
    try:
        held_entry = current_run(drive_root, run_id)
        if held_entry is not None:
            if held_entry.task_id != entry.task_id or held_entry.invocation_id != invocation_id:
                return {"deferred": "custody_owner_mismatch"}
            return {}
        record = custody.invocation_record(drive_root, invocation_id) or {}
    except Exception:
        log.debug("Binding of %s unreadable: its money waits for its STARTED row", invocation_id, exc_info=True)
        record = {}
    held = str(record.get("run_id") or "") if record.get("state") == "started" else ""
    if held == run_id:
        return {} if record.get("task_id") == entry.task_id else {"deferred": "custody_owner_mismatch"}
    return ({"deferred": "invocation_held_elsewhere", "held_by": held} if held
            else {"deferred": "custody_publication_unconfirmed"})


def recover(drive_root: Any, gateway: Any, record: Dict[str, Any], *,
            facts: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
    """Read the accepted work behind one pending invocation and bind it once; nothing is POSTed.

    ``record`` carries the resolved ``request`` and the ``invocation_id``. Returns ``(custody, receipt)``:
    the custody now holding the invocation's run (bound here, or adopted from an earlier writer) or
    ``None`` with the typed reason the start stays pending and unknown. A runless retained turn is named
    with its thread, turn and recorded enqueue error and stays unknown: custody has no state that would
    record it as a definite failure or as a run. ``facts`` ride a STARTED row this read writes.
    """
    from ouroboros.delegate_custody_reconcile import _mark_root_cost, recovered_custody

    located = locate(gateway, record)
    receipt: Dict[str, Any] = {"invocation_id": str(record.get("invocation_id") or ""),
                               "task_id": str(record.get("task_id") or ""), "lookup": located.outcome,
                               **({"store_id": located.store_id} if located.store_id else {}),
                               **({"lookup_detail": located.detail} if located.detail else {})}
    if not located.bindable:
        reason = "accepted_turn_without_run" if located.outcome == "turn_only" else f"accepted_start_{located.outcome}"
        turn = ({"thread_id": located.facts["threadId"], "turn_id": located.facts["turnId"],
                 "enqueue_error": located.facts["enqueueError"]} if located.outcome == "turn_only" else {})
        refusal = ({"refusal_code": located.facts["refusal_code"]}
                   if located.outcome == "refused" and isinstance(located.facts, dict)
                   and located.facts.get("refusal_code") else {})
        return None, {**receipt, "action": "invocation_retained", "reason": reason, **turn, **refusal}
    entry, shape = recovered_custody(drive_root, gateway, record, located.run_id, located.store_id)
    shape.update(facts or {}, accepted_start_lookup=located.outcome)
    bound = bind_started(drive_root, entry, shape=shape, exclusive=True, observed_state=located.state)
    if bound.custody is not None and bound.adopted and not bound.durable:
        return bound.custody, {**receipt, "action": "accepted_start_adopted", "run_id": bound.custody.run_id,
                               "reason": "start_state_not_recorded"}
    if bound.custody is None or not bound.durable:
        return None, {**receipt, "action": "invocation_retained", "reason": "accepted_start_unbound"}
    if not bound.adopted:
        _mark_root_cost(drive_root, bound.custody)
    problem = (located.facts or {}).get("problem")  # a failure the accepted job recorded: its outcome, kept whole
    return bound.custody, {**receipt, "action": "accepted_start_adopted" if bound.adopted else "accepted_start_bound",
                           "run_id": bound.custody.run_id, **({"state": located.state} if located.state else {}),
                           **({"problem": problem} if isinstance(problem, dict) else {})}


def accepted_after_refusal(drive_root: Any, gateway: Any, invocation_id: str,
                           error: BaseException) -> Tuple[Any, Dict[str, Any]]:
    """Read the work behind a start whose POST the engine answered with an error and no run; bind it once.

    An HTTP answer to an entered POST generally proves no non-admission (the gateway types a non-JSON body
    ``http_<status>``; an accepted command can end before execution and answer its start 400). ``START_UNSENT``
    belongs strictly to a fresh start that never entered its POST. Here the still-pending invocation is read
    first (``recover``): a found command is bound for its existing observation, capture and settlement. Every
    other answer (not found, unsupported, refused, another store, a record no longer pending) binds nothing,
    sends nothing and removes nothing; only a verified allowlisted producer's direct ``POST /v2/runs`` 409
    ``continuation_superseded`` on a fresh ordinary continuation may then settle durably as a sent
    pre-admission refusal (``delegate_continuation.settle_pre_admission_continuation_rejection``, ``start_sent=True``,
    never ``START_UNSENT`` and never retiring snapshots or owned registrations) once this read returns same-store
    ``not_found``. A transport fault, no answer at all, is left to the start's existing readers.
    Returns ``(custody or None, receipt)``; the receipt, and a STARTED row it writes, name the refusal it read
    behind (``start_refusal``). The run's own terminal stays its outcome: its typed failure, never a success."""
    refusal = {"code": str(getattr(error, "code", "") or type(error).__name__),
               "status": int(getattr(error, "status_code", 0) or 0)}
    receipt: Dict[str, Any] = {"invocation_id": str(invocation_id or ""), "lookup": "not_read",
                               "start_refusal": refusal}
    if not invocation_id or not refusal["status"]:
        return None, receipt
    from ouroboros import delegate_custody as custody

    try:
        record = custody.invocation_record(drive_root, invocation_id) or {}
        if record.get("state") != "pending":  # a STARTED another reader bound, or no record: its own readers decide
            return None, {**receipt, "reason": f"invocation_{record.get('state') or 'unrecorded'}"}
        held, read = recover(drive_root, gateway, {**record, "invocation_id": invocation_id},
                             facts={"start_refusal": refusal})
    except Exception:  # a read that failed outright binds nothing; the start stays as unknown as before it
        log.warning("Accepted-start read behind the refused start of %s failed", invocation_id, exc_info=True)
        return None, {**receipt, "lookup": "failed"}
    return held, {**read, "start_refusal": refusal}


def prove_job_run_alias(
    drive_root: Any,
    gateway: Any,
    job_id: str,
    canonical_run_id: str,
    *,
    store_id: str,
    engine_ref: Optional[Dict[str, Any]] = None,
    exclude_invocation_id: str = "",
) -> bool:
    """Confirm via noncreating lookup that ``job_id`` is the durable predecessor bound to
    ``canonical_run_id`` in ``store_id``.

    Pinned Claudexor canonicalizes ``continueFrom=<jobId>`` to the predecessor's ``runId`` before
    pre-admission checks, while custody keeps the ``jobId`` when a queued start was bound before
    its run existed. Only an exact same-store ``locate`` lookup of the predecessor's landed durable
    invocation proves that link — never ID shape, a missing record, or ``GET /v2/runs/:id`` 200
    (which does not promise ``storeId``)."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import run as current_run
    from ouroboros.execution_domain import is_work_ref, recorded_ref, same_engine

    job, canonical, expected_store = str(job_id or ""), str(canonical_run_id or ""), str(store_id or "")
    if (not job or job != job.strip() or not canonical or canonical != canonical.strip()
            or job == canonical or not expected_store or expected_store != expected_store.strip()):
        return False
    expected_ref = recorded_ref(engine_ref or {})
    if (str(getattr(gateway, "store_identity", "") or "") != expected_store
            or not same_engine(gateway, {"engine_ref": expected_ref})):
        return False
    try:
        if custody.custody_log_unreadable(drive_root):
            return False
        prior = current_run(drive_root, job)
    except Exception:
        log.debug("Predecessor custody unreadable while checking alias %s -> %s", job, canonical, exc_info=True)
        return False
    if (prior is None or prior.run_id != job or prior.review_owned
            or prior.terminal_state not in custody.TERMINAL_STATES
            or prior.daemon_store_id != expected_store or not same_engine(gateway, prior)):
        return False
    prior_ref = recorded_ref(prior)
    if is_work_ref(prior_ref) != is_work_ref(expected_ref) or (is_work_ref(expected_ref) and prior_ref != expected_ref):
        return False
    prior_iid = str(prior.invocation_id or "").strip()
    if not prior_iid or prior_iid == str(exclude_invocation_id or ""):
        return False
    try:
        record = custody.invocation_record(drive_root, prior_iid)
    except Exception:
        log.debug("Predecessor invocation %s unreadable while checking alias %s -> %s", prior_iid, job, canonical, exc_info=True)
        return False
    if (not isinstance(record, dict) or record.get("state") != "started"
            or str(record.get("run_id") or "") != job
            or str(record.get("task_id") or "") != prior.task_id
            or custody.review_owned_source(record.get("source"))):
        return False
    rec_store = str(record.get("daemon_store_id") or "")
    if rec_store and rec_store != expected_store:
        return False
    rec_ref = recorded_ref(record)
    if is_work_ref(rec_ref) != is_work_ref(expected_ref) or (is_work_ref(expected_ref) and rec_ref != expected_ref):
        return False
    body = record.get("request")
    if not isinstance(body, dict) or not body:
        return False
    thread_id = str(body.get("_thread_id") or "")
    if (body.get("_use_thread") and not thread_id) or (thread_id and thread_id != str(prior.thread_id or "")):
        return False
    located = locate(gateway, {**record, "invocation_id": prior_iid, "daemon_store_id": expected_store})
    if located.outcome != "found" or located.store_id != expected_store:
        return False
    facts = located.facts if isinstance(located.facts, dict) else {}
    return facts.get("jobId") == job and facts.get("runId") == canonical


def predecessor_alias_custody_valid(
    txn: Any,
    job_id: str,
    canonical_run_id: str,
    *,
    store_id: str,
    engine_ref: Optional[Dict[str, Any]] = None,
    exclude_invocation_id: str = "",
) -> bool:
    """Atomic write-txn check for an aliased pre-admission rejection (zero CAS I/O): ``job_id``
    remains a durable non-review terminal run bound to ``store_id`` on the same engine with its
    own invocation, and ``canonical_run_id`` is not bound to the rejected continuation."""
    from ouroboros import delegate_custody as custody
    from ouroboros.execution_domain import is_work_ref, recorded_ref

    pred_row = txn.custody_row(job_id)
    if not isinstance(pred_row, dict) or custody.review_owned_source(pred_row.get("source")):
        return False
    pred_iid = str(pred_row.get("invocation_id") or "")
    pred_ref, expected_ref = recorded_ref(pred_row), recorded_ref(engine_ref or {})
    if (not pred_iid or pred_iid == str(exclude_invocation_id or "")
            or str(pred_row.get("daemon_store_id") or "") != str(store_id or "")
            or str(pred_row.get("terminal_state") or "") not in custody.TERMINAL_STATES
            or is_work_ref(pred_ref) != is_work_ref(expected_ref)
            or (is_work_ref(expected_ref) and pred_ref != expected_ref)):
        return False
    canon_row = txn.custody_row(canonical_run_id)
    return not (isinstance(canon_row, dict) and str(canon_row.get("invocation_id") or "") == str(exclude_invocation_id or ""))


__all__ = ["Bound", "Lookup", "NO_REPLAY", "RUN_LOOKUP", "RUN_STATES", "TURN_LOOKUP", "accepted_after_refusal",
           "bind_started", "deferred_money", "locate", "predecessor_alias_custody_valid", "prove_job_run_alias", "recover"]
