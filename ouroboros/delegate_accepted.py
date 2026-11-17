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
review seat's, every read here: explicit retry, sweep, a seat's rejoin): under the actor decision lock a
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
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    body, key = record.get("request"), str(record.get("invocation_id") or "")
    if not isinstance(body, dict) or not body or not key:
        return Lookup("unavailable", detail="the recorded request or its invocation key is unavailable")
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
                 exclusive: bool = False) -> Bound:
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
    """
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import run as current_run

    invocation_id, task_id = str(getattr(entry, "invocation_id", "") or ""), str(getattr(entry, "task_id", "") or "")
    if not invocation_id or not task_id:
        return Bound(entry, bool(custody.record_started(drive_root, entry, shape=shape)), False)
    decided = False
    try:
        with custody.actor_decision_lock(drive_root, task_id):
            prior = custody.invocation_record(drive_root, invocation_id) or {}
            held = str(prior.get("run_id") or "") if prior.get("state") == "started" else ""
            existing = current_run(drive_root, held) if held else None
            decided = True
            if held and existing is None and exclusive:
                return Bound(None, False, False)  # a bound run whose row is unreadable: nothing is added
            if existing is not None and exclusive:
                return Bound(existing, True, True)
            if existing is not None:  # the caller's object stays as it was: a copy takes the held run's name
                entry = copy.copy(entry)
                entry.run_id = existing.run_id
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
        if current_run(drive_root, run_id) is not None:
            return {}
        record = custody.invocation_record(drive_root, invocation_id) or {}
    except Exception:
        log.debug("Binding of %s unreadable: its money waits for its STARTED row", invocation_id, exc_info=True)
        record = {}
    held = str(record.get("run_id") or "") if record.get("state") == "started" else ""
    if held == run_id:
        return {}
    return ({"deferred": "invocation_held_elsewhere", "held_by": held} if held
            else {"deferred": "custody_publication_unconfirmed"})


def recover(drive_root: Any, gateway: Any, record: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Read the accepted work behind one pending invocation and bind it once; nothing is POSTed.

    ``record`` carries the resolved ``request`` and the ``invocation_id``. Returns ``(custody, receipt)``:
    the custody now holding the invocation's run (bound here, or adopted from an earlier writer) or
    ``None`` with the typed reason the start stays pending and unknown. A runless retained turn is named
    with its thread, turn and recorded enqueue error and stays unknown: custody has no state that would
    record it as a definite failure or as a run.
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
        return None, {**receipt, "action": "invocation_retained", "reason": reason, **turn}
    entry, shape = recovered_custody(drive_root, gateway, record, located.run_id, located.store_id)
    shape["accepted_start_lookup"] = located.outcome
    bound = bind_started(drive_root, entry, shape=shape, exclusive=True)
    if bound.custody is None or not bound.durable:
        return None, {**receipt, "action": "invocation_retained", "reason": "accepted_start_unbound"}
    if not bound.adopted:
        _mark_root_cost(drive_root, bound.custody)
    problem = (located.facts or {}).get("problem")  # a failure the accepted job recorded: its outcome, kept whole
    return bound.custody, {**receipt, "action": "accepted_start_adopted" if bound.adopted else "accepted_start_bound",
                           "run_id": bound.custody.run_id, **({"state": located.state} if located.state else {}),
                           **({"problem": problem} if isinstance(problem, dict) else {})}


__all__ = ["Bound", "Lookup", "NO_REPLAY", "RUN_LOOKUP", "RUN_STATES", "TURN_LOOKUP", "bind_started", "deferred_money",
           "locate", "recover"]
