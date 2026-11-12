"""#3138: a start the engine already accepted is read back and bound once, never sent again.

The engine double serves the declared noncreating reads (``POST /v2/runs/lookup``,
``POST /v2/threads/:id/turns/lookup``: same body, same ``Idempotency-Key``) in the engine's own strict
``ControlStartLookupResponse`` shapes (HTTP 200 ``found`` with ``handle{jobId, runId|null, state, runDir,
taskId, problem?}`` | ``turn_only`` with ``enqueueError`` | ``not_found``, each naming its ``storeId``; typed
ControlProblems otherwise) beside its creating ``POST /v2/runs``, and addresses a command by its job id or
run id as ``GET /v2/runs/:id`` does. Every consumer runs for
real through the production gateway: an explicit ``delegate_start(retry_of=...)``, the admitted
startup/periodic sweep for orphaned and still-owned starts, a review seat's own rejoin and an owner
Pause's settle and Resume pass. Only the engine is scripted; every creating POST is counted, so a
test that must send nothing proves it sent nothing.
"""
from __future__ import annotations

import copy
import json
import threading
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from ouroboros import delegate_collection as body, delegate_custody as c, delegate_custody_current as current
from ouroboros import delegate_custody_reconcile as reconcile
from ouroboros.config import CLAUDEXOR_MIN_VERSION, CLAUDEXOR_PROTOCOL_MAJOR
from ouroboros.gateways import claudexor as cx, claudexor_metadata as metadata
from ouroboros.delegate_custody_usage import run_names
from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _owned_gateway_uses_each_test_transport,
    _select_transport_access,
)
from tests._review_session_route_shared import fake_route as _fake_route_fixture
from tests._delegated_transport_shared import _delegate_registry

fake_route = _fake_route_fixture  # the review seat's scripted engine (FakeGateway), for the panel's own rejoin

_REAL_GATEWAY = cx.ClaudexorGateway
ENQUEUE_ERROR = {"message": "fixture turn preflight refused", "code": "fixture_preflight_refusal", "retryable": False,
                 "requiredActions": [], "context": {}, "failedAt": "2026-11-11T12:37:12.000Z"}
_LOOKUP_ROW = {"mutability": "read_only", "idempotency": "key_required", "completion": "immediate",
               "responseSchema": "ControlStartLookupResponse", "parameters": []}
LOOKUPS = [{"method": "POST", "path": "/v2/runs/lookup", "requestSchema": "ControlRunStartRequest",
            "applicability": "run", **_LOOKUP_ROW},
           {"method": "POST", "path": "/v2/threads/:id/turns/lookup", "requestSchema": "ControlThreadTurnRequest",
            "applicability": "thread", **_LOOKUP_ROW}]  # the engine's catalog rows (sibling INTERFACE.md)


class Engine:
    """One serving store, its accepted commands and thread turns, in Claudexor's wire shapes."""

    def __init__(self, tmp_path, store="store-a", lookup=True):
        descriptor = tmp_path / "control-api.json"
        descriptor.write_text("{}", encoding="utf-8")
        self.endpoint = cx.DaemonEndpoint("127.0.0.1", 1, "secret-token", str(descriptor),
                                          metadata.descriptor_stamp(descriptor))
        self.root, self.store, self.lookup = str(tmp_path), store, lookup
        self.commands, self.turns = {}, {}
        self.creates, self.lookups, self.paths, self.cancels = [], [], [], []
        self.lose, self.queue_runs, self.health, self.answer = 0, False, "ok", None
        self.replace_at_post, self.on_accept = "", None  # the store a POST lands in; a reader inside the POST
        self.create_refusal = None  # a ControlProblem every creating POST answers with, as the engine's own validation

    def accept(self, key, request, *, run=True, state="running", spend=0.25, failure=None, problem=None):
        number = len(self.commands) + 1
        job = f"job-{number}"
        self.commands[job] = {"store": self.store, "key": key, "body": copy.deepcopy(request), "seq": 2,
                              "run": f"run-{number}" if run else None, "state": state if run or state != "running"
                              else "queued", "spend": spend, "failure": failure, "problem": problem,
                              "text": "the complete answer"}
        return job

    def _handle(self, job, **turn):
        """``ControlStartLookupHandle``: runDir/taskId present (null before a run), ``problem`` only when recorded."""
        row = self.commands[job]
        return {"jobId": job, "runId": row["run"], "state": row["state"],
                "runDir": f"/runs/{row['run']}" if row["run"] else None, "taskId": f"task-{job}" if row["run"] else None,
                **turn, **({"problem": copy.deepcopy(row["problem"])} if row["problem"] else {})}

    def bind_run(self, job, state="running"):
        command = self.commands[job]
        command["run"] = command["run"] or "run-" + job.split("-", 1)[1]
        command["state"], command["seq"] = state, command["seq"] + 1
        return command["run"]

    def _held(self, key):
        return next((job for job, row in self.commands.items() if (row["store"], row["key"]) == (self.store, key)), None)

    def _named(self, request):
        return {"storeId": self.store} if request.headers.get("x-claudexor-store-identity") == "1" and self.store else {}

    def _detail(self, job):
        row = self.commands[job]
        summary = {"runId": row["run"], "jobId": job, "state": row["state"], "effectiveAccess": "readonly",
                   **({"spendUsd": row["spend"], "spendEstimated": False} if row["spend"] is not None else {}),
                   **({"failure": row["failure"]} if row["failure"] else {})}
        detail = {"summary": summary, "lastSeq": row["seq"], "pendingInteractions": []}
        if row["state"] == "succeeded":
            detail["primaryOutput"] = {"text": row["text"], "bytes": len(row["text"].encode()), "truncated": False}
        return detail

    def _lookup(self, request, key, path):
        sent = json.loads(request.read())
        thread = path.split("/")[3] if path.startswith("/v2/threads/") else ""
        self.lookups.append((thread or "run", key, sent))
        if self.answer is not None:
            return httpx.Response(self.answer[0], json=self.answer[1])
        if thread:
            turn = self.turns.get((self.store, thread, key))
            if turn is None:
                return httpx.Response(200, json={"status": "not_found", "storeId": self.store})
            if turn["job"]:
                return httpx.Response(200, json={"status": "found", "storeId": self.store,
                                                 "handle": self._handle(turn["job"], threadId=thread, turnId=turn["turn"])})
            return httpx.Response(200, json={"status": "turn_only", "storeId": self.store, "threadId": thread,
                                             "turnId": turn["turn"], "runId": turn["run"], "enqueueError": turn["error"]})
        job = self._held(key)
        if job is None:
            return httpx.Response(200, json={"status": "not_found", "storeId": self.store})
        if self.commands[job]["body"] != sent:
            return httpx.Response(409, json={"code": "idempotency_conflict", "message": "the key holds another request",
                                             "retryable": False})
        return httpx.Response(200, json={"status": "found", "storeId": self.store, "handle": self._handle(job)})

    def __call__(self, request):
        path, method = request.url.path, request.method
        self.paths.append((method, path))
        if path == "/v2/handshake":
            return httpx.Response(200, json={"protocolMajor": CLAUDEXOR_PROTOCOL_MAJOR, "compatible": True,
                                             "engine": {"version": CLAUDEXOR_MIN_VERSION}, **self._named(request)})
        if path == "/v2/agent-capabilities":
            return httpx.Response(200, json={"harnesses": [{"id": "some-route", "enabled": True, "status": self.health,
                                                            "accessProfilesSupported": ["readonly"]}]})
        if path == "/v2/quota":
            return httpx.Response(200, json={"snapshots": []})
        if path == "/v2/operations":
            return httpx.Response(200, json={"operations": LOOKUPS if self.lookup else []})
        if path == "/v2/projects":
            return httpx.Response(200, json={"projects": [{"id": "prj-existing", "root": self.root}]})
        key = request.headers.get("Idempotency-Key")
        if path == "/v2/runs/lookup" or path.endswith("/turns/lookup"):
            return self._lookup(request, key, path)
        if path == "/v2/runs" and method == "POST":
            self.store, self.replace_at_post = self.replace_at_post or self.store, ""
            sent = json.loads(request.read())
            self.creates.append((key, self.store, sent))
            if self.create_refusal is not None:
                return httpx.Response(self.create_refusal[0], json=self.create_refusal[1])
            job = self._held(key) or self.accept(key, sent, run=not self.queue_runs)
            if self.on_accept is not None:
                hook, self.on_accept = self.on_accept, None
                hook(job)
            if self.lose:
                self.lose -= 1
                raise httpx.ConnectError("the reply was lost after the POST was accepted")
            return httpx.Response(200, json={"jobId": job, "runId": self.commands[job]["run"], "runDir": "/tmp/run",
                                             **self._named(request)})
        name = path.split("/")[3]
        job = next((job for job, row in self.commands.items() if row["store"] == self.store and name in (job, row["run"])),
                   None)
        if path.endswith("/control"):
            self.cancels.append(name)
            return httpx.Response(200 if job else 404, json={"accepted": True} if job else {"error": "no such run"})
        if "/artifacts/" in path or job is None:
            return httpx.Response(404, json={"error": "no such run", "context": {"storeId": self.store}})
        assert method == "GET" and path == f"/v2/runs/{name}", path
        return httpx.Response(200, json=self._detail(job))

    def gateway(self):
        gateway = _REAL_GATEWAY(self.endpoint)
        gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(self),
                                       headers=dict(gateway._client.headers))
        gateway.handshake()  # the serving store, as every owned gateway proves it
        return gateway


def _delegating(tmp_path, monkeypatch, **engine):
    engine = Engine(tmp_path, **engine)
    _select_transport_access(monkeypatch, "readonly")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(cx, "ClaudexorGateway", lambda *_a, **_k: engine.gateway())
    return _delegate_registry(tmp_path, monkeypatch), engine


def _start(registry, retry_of=""):
    from ouroboros.tools import delegate

    if retry_of:
        return json.loads(registry.execute_result("delegate_start", {"retry_of": retry_of}).text)
    return json.loads(delegate._delegate_start(registry._ctx, "Summarize the layout.", max_seconds=120).text)


def _lost_start(tmp_path, monkeypatch, **engine):
    """A real start whose reply was lost after the engine accepted it: only START_REQUESTED names it."""
    registry, engine = _delegating(tmp_path, monkeypatch, **engine)
    engine.lose = 1
    token = _start(registry)["pending_invocation_id"]
    c._CUSTODY.clear()  # the worker that held the reply is gone
    return registry, engine, token


def _rows(root, kind, invocation_id="", **match):
    """The chain's rows of ``kind`` for one invocation (STARTED, START_FAILED) or one run (``run_id=``: SETTLED)."""
    return [row for row in c.custody_rows(root) if row["type"] == kind
            and (not invocation_id or row.get("invocation_id") == invocation_id)
            and all(row.get(key) == value for key, value in match.items())]


def _held(root, invocation_id):
    return [run for run in current.runs(root).values() if run.invocation_id == invocation_id]


def _sweep(root, engine, live=(), *, observe=lambda: True, replay=lambda: True, factory=None):
    """The admitted startup/periodic sweep's shape: one not-stopping predicate admits the read and the replay."""
    return c.reconcile_orphaned_runs(root, set(live), gateway_factory=factory or engine.gateway,
                                     replay_pending=replay, observe_pending=observe)


def test_an_explicit_retry_reads_its_accepted_start_before_route_health_and_sends_nothing(tmp_path, monkeypatch):
    registry, engine, token = _lost_start(tmp_path, monkeypatch)
    (job, accepted), = engine.commands.items()
    engine.health, engine.paths[:] = "unavailable", []  # the model route is unhealthy now: accepted work is still read
    payload = _start(registry, retry_of=token)
    assert payload["status"] == "started" and payload["recovered_without_start"] is True, payload
    assert payload["run_id"] == accepted["run"] and payload["idempotent_recovery"] is True
    assert ("GET", "/v2/agent-capabilities") not in engine.paths, "no route health is asked of work already accepted"
    assert [(kind, key) for kind, key, _ in engine.lookups] == [("run", token)]
    assert engine.lookups[0][2] == engine.creates[0][2], "the lookup carries the recorded start's exact body"
    assert len(engine.creates) == 1, "the accepted start is read, never sent again"
    held, = _held(tmp_path, token)
    assert (held.run_id, held.daemon_store_id, held.task_id) == (accepted["run"], "store-a", "t-family")
    started, = _rows(tmp_path, c.STARTED, token)
    assert started["accepted_start_lookup"] == "found" and started["recovered_from_pending_invocation"] is True
    assert c.invocation_record(tmp_path, token)["state"] == "started" and c.pending_invocations(tmp_path) == []
    again = _start(registry, retry_of=token)
    assert (again["reason"], again["run_id"]) == ("invocation_already_started", accepted["run"])
    assert len(engine.creates) == 1 and len(engine.lookups) == 1


@pytest.mark.parametrize("consumer", ["retry", "sweep"])
def test_a_queued_job_with_no_run_yet_is_bound_by_its_job_id_and_settles_once_through_it(tmp_path, monkeypatch, consumer):
    registry, engine, token = _lost_start(tmp_path, monkeypatch)
    engine.queue_runs = True
    engine.commands.clear()
    job = engine.accept(token, engine.creates[0][2], run=False)  # accepted, still queued: no run bound
    if consumer == "retry":
        payload = _start(registry, retry_of=token)
        assert (payload["status"], payload["run_id"], payload["lookup"]["state"]) == ("started", job, "queued"), payload
    else:  # the owner is live: the sweep reads and binds, never replays, cancels or settles
        outcome, = _sweep(tmp_path, engine, live={"t-family"})
        assert (outcome["action"], outcome["run_id"], outcome["state"]) == ("accepted_start_bound", job, "queued")
    held, = _held(tmp_path, token)
    assert held.run_id == job and not held.settled and engine.cancels == []
    run = engine.bind_run(job, state="succeeded")  # the job's run binds and ends; custody keeps the job's name
    detail = engine.gateway().get_run(job)
    assert run_names(detail) == {job, run}
    for _ in range(2):  # the body's ordinary collection, repeated: one settlement, one cost row
        body.collect_terminal(tmp_path, job, gateway=engine.gateway(), detail=detail)
    settled = current.run(tmp_path, job)
    assert settled.settled and settled.terminal_state == "succeeded" and settled.cost_usd == 0.25
    assert len(_rows(tmp_path, c.SETTLED, run_id=job)) == 1 and len(_rows(tmp_path, c.LEDGER_RECORDED)) == 1
    assert current.run(tmp_path, run) is None and len(engine.creates) == 1


@pytest.mark.parametrize("consumer", ["retry", "sweep"])
def test_a_job_that_failed_before_any_run_settles_as_an_observed_failure_with_unknown_cost(tmp_path, monkeypatch, consumer):
    registry, engine, token = _lost_start(tmp_path, monkeypatch)
    engine.commands.clear()
    failure = {"code": "route_preflight_refused", "safeMessage": "the route refused before any run started"}
    problem = {"message": "the route refused before any run started", "code": "route_preflight_refused", "status": 403,
               "retryable": False, "requiredActions": ["Change the route."], "context": {"detail": "kept"}}
    job = engine.accept(token, engine.creates[0][2], run=False, state="failed", spend=None, failure=failure,
                        problem=problem)
    if consumer == "retry":
        payload = _start(registry, retry_of=token)
        assert (payload["run_id"], payload["lookup"]["state"]) == (job, "failed"), payload
        assert payload["lookup"]["problem"] == problem, "the job's recorded failure is its outcome, kept whole"
        body.collect_terminal(tmp_path, job, gateway=engine.gateway(), detail=engine.gateway().get_run(job))
    else:  # an orphaned start: bound, then reconciled like any open run of an ended owner
        outcome, = _sweep(tmp_path, engine)
        assert (outcome["run_id"], outcome["action"], outcome["settled"]) == (job, "settle_attempted", True), outcome
        bound, = [row for row in _rows(tmp_path, c.RECONCILED) if row.get("action") == "accepted_start_bound"]
        assert (bound["run_id"], bound["state"], bound["problem"]) == (job, "failed", problem)
    settled = current.run(tmp_path, job)
    assert settled.settled and settled.terminal_state == "failed"
    assert settled.cost_usd is None and not settled.spend_disclosed, "no cost was reported: it stays unknown, never zero"
    row, = _rows(tmp_path, c.SETTLED, run_id=job)
    assert (row["state"], row["failure_code"], row["cost_usd"]) == ("failed", "route_preflight_refused", None)
    # Only the lost POST's own unknown-outcome row: the recovery adds no refusal, definite or not.
    assert [row.get("definite") for row in _rows(tmp_path, c.START_FAILED, token)] == [False]
    assert c.invocation_record(tmp_path, token)["state"] == "started" and len(engine.creates) == 1


def test_a_sweep_that_binds_the_job_while_its_start_waits_converges_with_the_starts_own_receipt(tmp_path, monkeypatch):
    """The ordinary race, recovery first: the original POST waits for its run, the periodic sweep reads the
    accepted job and binds its job id, then the original reply names the run. One custody and cost row."""
    registry, engine = _delegating(tmp_path, monkeypatch)
    engine.queue_runs = True
    seen = {}

    def sweep_while_the_post_waits(job):
        seen["outcomes"] = _sweep(tmp_path, engine, live={"t-family"})
        seen["run"] = engine.bind_run(job)
    engine.on_accept = sweep_while_the_post_waits
    payload = _start(registry)
    (job, _command), = engine.commands.items()
    token = payload["invocation_id"]
    assert [(row["action"], row["run_id"]) for row in seen["outcomes"]] == [("accepted_start_bound", job)]
    assert payload["status"] == "started" and payload["run_id"] == job, "the start answers with the bound run's name"
    held, = _held(tmp_path, token)
    assert held.run_id == job and held.max_seconds == 120, "the start's own facts land on the one row"
    assert {row["run_id"] for row in _rows(tmp_path, c.STARTED, token)} == {job}
    assert current.run(tmp_path, seen["run"]) is None and c.invocation_record(tmp_path, token)["run_id"] == job
    engine.commands[job]["state"] = "succeeded"
    for _ in range(2):  # the owner ended: the orphan sweep settles the one row once
        _sweep(tmp_path, engine)
    assert len(_rows(tmp_path, c.SETTLED, run_id=job)) == 1 and len(_rows(tmp_path, c.LEDGER_RECORDED)) == 1
    assert _rows(tmp_path, c.SETTLED, run_id=seen["run"]) == [], "the run's other name holds no second row"
    assert len(engine.creates) == 1


def _short_lock_wait(monkeypatch, bound=0.3):
    """The actor decision lock's real O_EXCL file lock, its contention bound cut from 20 s for a fast test."""
    from ouroboros import platform_layer

    real = platform_layer.acquire_exclusive_file_lock

    def acquire(path, *args, timeout_sec=4.0, **kwargs):
        return real(path, *args, timeout_sec=min(float(timeout_sec), bound), **kwargs)
    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", acquire)


class _DecisionHeld:
    """Another writer of the task still inside its decision, holding the real actor lock until released."""

    def __init__(self, root, task_id):
        self.root, self.task_id = root, task_id
        self.held, self.release = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self):
        with c.actor_decision_lock(self.root, self.task_id):
            self.held.set()
            self.release.wait(10)

    def __enter__(self):
        self.thread.start()
        assert self.held.wait(5)
        return self

    def __exit__(self, *_exc):
        self.release.set()
        self.thread.join(10)


@pytest.mark.parametrize("first", ["the_bound_row_settles", "the_in_session_cancel"])
def test_a_start_whose_decision_lock_a_recovery_holds_publishes_no_rival_row(tmp_path, monkeypatch, first):
    """Recovery first: the periodic read binds the queued job's id and is still inside its decision when the
    start's own receipt, naming the run id, cannot take the lock. That receipt publishes nothing: the job's row
    is the one row, the run stays this process's ``started_uncustodied`` handle with its retry token, and money
    is recorded once whichever settles first (``delegate_accepted.deferred_money``)."""
    registry, engine = _delegating(tmp_path, monkeypatch)
    engine.queue_runs = True
    _short_lock_wait(monkeypatch)
    seen = {}

    def read_then_hold(job):
        seen["outcomes"] = _sweep(tmp_path, engine, live={"t-family"})
        seen["run"] = engine.bind_run(job)
        seen["lock"] = _DecisionHeld(tmp_path, "t-family").__enter__()
    engine.on_accept = read_then_hold
    try:
        payload = _start(registry)
    finally:
        seen["lock"].__exit__()
    (job, _command), = engine.commands.items()
    token, run = payload["invocation_id"], seen["run"]
    assert [(row["action"], row["run_id"]) for row in seen["outcomes"]] == [("accepted_start_bound", job)]
    assert {row["run_id"] for row in _rows(tmp_path, c.STARTED, token)} == {job}, "no rival row under the run's name"
    assert (payload["status"], payload["run_id"], payload.get("pending_invocation_id")) == ("started_uncustodied", run, token)
    assert current.run(tmp_path, run) is None and c.invocation_record(tmp_path, token)["run_id"] == job
    engine.commands[job]["state"] = "succeeded"
    assert body.record_terminal_money(tmp_path, c._CUSTODY[run], engine._detail(job)) == {
        "settled": False, "ledger_recorded": False, "retried": False, "deferred": "invocation_held_elsewhere",
        "held_by": job}, "the bound row's settlement records it, never this name"

    def cancel():
        return json.loads(registry.execute_result("delegate_cancel", {"run_id": run}).text)
    if first == "the_in_session_cancel":
        assert cancel()["status"] == c.CANCEL_CONFIRMED, "the process-local handle still reaches its run"
        _sweep(tmp_path, engine)  # its owner ended: the one row settles
    else:
        _sweep(tmp_path, engine)
        assert cancel()["status"] == c.CANCEL_CONFIRMED
    assert len(_rows(tmp_path, c.LEDGER_RECORDED)) == 1 and len(_rows(tmp_path, c.SETTLED, run_id=job)) == 1
    assert _rows(tmp_path, c.SETTLED, run_id=run) == [] and current.run(tmp_path, run) is None
    assert len(engine.creates) == 1


def _start_under_a_held_decision(tmp_path, monkeypatch):
    """A recovery has read the accepted job and is still inside its decision, nothing published yet, when the
    start's own receipt (naming the run id) cannot take the lock: the run stays this process's handle with its
    invocation pending, and the engine then finishes it. The held decision is returned for the test to end."""
    registry, engine = _delegating(tmp_path, monkeypatch)
    _short_lock_wait(monkeypatch)
    seen = {}

    def hold(job):
        seen["job"], seen["decision"] = job, _DecisionHeld(tmp_path, "t-family").__enter__()
    engine.on_accept = hold
    try:
        payload = _start(registry)
        run, token, job = payload["run_id"], payload["invocation_id"], seen["job"]
        assert (payload["status"], c.invocation_record(tmp_path, token)["state"]) == ("started_uncustodied", "pending")
    except BaseException:
        if "decision" in seen:
            seen["decision"].__exit__()
        raise
    engine.commands[job]["state"] = "succeeded"
    return registry, engine, run, token, job, seen["decision"]


def _stale_read(engine, job):
    """What the held read saw: the command queued, no run yet."""
    engine.answer = (200, {"status": "found", "storeId": "store-a",
                           "handle": {"jobId": job, "runId": None, "state": "queued"}})


@pytest.mark.parametrize("consumer", ["money_writer", "in_session_cancel", "in_session_wait"])
def test_a_process_local_run_records_no_money_before_the_reads_binding_lands(tmp_path, monkeypatch, consumer):
    """The nanny's repro (2026-11-11, money before the read's publication), through every consumer of the
    handle: the money writer itself, the in-session cancel and the in-session wait meet the terminal while the
    invocation is still pending, so a read may yet bind its other engine name. None records money under the
    run's name; the read's job binding then records it once, and a later wait on the handle delivers the
    answer with no second cost."""
    registry, engine, run, token, job, decision = _start_under_a_held_decision(tmp_path, monkeypatch)

    def wait():
        return json.loads(registry.execute_result("delegate_wait", {"run_id": run}).text)
    try:
        if consumer == "money_writer":
            assert body.record_terminal_money(tmp_path, c._CUSTODY[run], engine._detail(job)) == {
                "settled": False, "ledger_recorded": False, "retried": False,
                "deferred": "custody_publication_unconfirmed"}
        elif consumer == "in_session_cancel":
            cancelled = json.loads(registry.execute_result("delegate_cancel", {"run_id": run}).text)
            assert (cancelled["status"], cancelled["state"]) == (c.CANCEL_CONFIRMED, "succeeded"), cancelled
        else:  # its own supervision save needs the held decision too: typed, nothing recorded yet
            assert wait()["reason"] == "supervision_state_unsaved"
        assert _rows(tmp_path, c.LEDGER_RECORDED) == [] and not c._CUSTODY[run].settled
    finally:
        decision.__exit__()
    _stale_read(engine, job)
    _sweep(tmp_path, engine)  # its owner ended: the read binds the job id and its one row settles
    waited = wait()
    assert (waited.get("collection") or {}).get("reason") == "custody_publication_unconfirmed", waited
    assert "the complete answer" in json.dumps(waited), "the handle still delivers its terminal answer"
    assert [row["run_id"] for row in _rows(tmp_path, c.LEDGER_RECORDED)] == [job]
    assert len(_rows(tmp_path, c.SETTLED, run_id=job)) == 1 and _rows(tmp_path, c.SETTLED, run_id=run) == []
    assert {row["run_id"] for row in _rows(tmp_path, c.STARTED, token)} == {job} and len(engine.creates) == 1
    _sweep(tmp_path, engine)
    assert len(_rows(tmp_path, c.LEDGER_RECORDED)) == 1 and current.run(tmp_path, job).settled


def test_a_deferred_run_its_wait_publishes_first_records_once_and_the_stale_read_adopts_it(tmp_path, monkeypatch):
    """The adjacent order: the same handle's cancel defers its money, then the decision ends before any read
    binds. The in-session wait re-publishes the run's own STARTED (its invocation still pending) and records
    the money there; a read listed before, answering the stale queued job, adopts that row. One row, one cost."""
    registry, engine, run, token, job, decision = _start_under_a_held_decision(tmp_path, monkeypatch)
    try:
        assert json.loads(registry.execute_result("delegate_cancel", {"run_id": run}).text)["status"] == c.CANCEL_CONFIRMED
        listed, = [row for row in c.pending_invocations(tmp_path) if row["invocation_id"] == token]
    finally:
        decision.__exit__()
    assert _rows(tmp_path, c.LEDGER_RECORDED) == []
    waited = json.loads(registry.execute_result("delegate_wait", {"run_id": run}).text)
    assert (waited["settlement"]["settled"], waited["settlement"]["ledger_recorded"]) == (True, True), waited
    _stale_read(engine, job)
    outcome = reconcile._observe_pending_invocation(tmp_path, engine.gateway(), listed)
    assert (outcome["action"], outcome["run_id"]) == ("accepted_start_adopted", run), outcome
    _sweep(tmp_path, engine)
    assert [row["run_id"] for row in _rows(tmp_path, c.LEDGER_RECORDED)] == [run]
    assert {row["run_id"] for row in _rows(tmp_path, c.STARTED, token)} == {run} and len(engine.creates) == 1


@pytest.mark.parametrize("handle_first", [False, True])
def test_a_wait_beside_a_running_process_local_handle_still_delivers_a_ready_durable_terminal(tmp_path, monkeypatch,
                                                                                              handle_first):
    """A wait selecting this process's handle (its invocation pending, its run still running) beside a durable run
    whose terminal the body recorded answers with that terminal in either order, as a wait on it alone would: its
    whole answer, settled once, the handle not even read. Beside a quiet durable run, the handle is the run read.
    The handle is neither published nor charged, and nothing is sent again."""
    registry, engine, run, token, job, decision = _start_under_a_held_decision(tmp_path, monkeypatch)
    decision.__exit__()
    engine.commands[job]["state"] = "running"
    done, quiet = _start(registry)["run_id"], _start(registry)["run_id"]
    done_job = next(name for name, row in engine.commands.items() if row["run"] == done)
    engine.commands[done_job]["state"] = "succeeded"
    assert body.observe_run(tmp_path, current.run(tmp_path, done), engine._detail(done_job))  # as the body records it

    def wait(other):
        engine.paths[:] = []
        selection = [run, other] if handle_first else [other, run]
        return json.loads(registry.execute_result("delegate_wait", {"runs": selection}).text)
    waited = wait(done)
    assert (waited.get("run_id"), (waited.get("sleep") or {}).get("woke_by")) == (done, f"run:{done}"), waited
    assert "the complete answer" in json.dumps(waited) and current.run(tmp_path, done).settled
    assert ("GET", f"/v2/runs/{run}") not in engine.paths
    read = wait(quiet)
    woke = "woke_by" in (read.get("sleep") or {})
    assert (read.get("run_id"), read.get("state"), woke) == (run, "running", False), read
    assert ("GET", f"/v2/runs/{run}") in engine.paths and ("GET", f"/v2/runs/{quiet}") not in engine.paths
    assert [row["run_id"] for row in _rows(tmp_path, c.LEDGER_RECORDED)] == [done] and len(engine.creates) == 3
    assert _rows(tmp_path, c.STARTED, token) == [] and c.invocation_record(tmp_path, token)["state"] == "pending"


@pytest.mark.parametrize("binding", ["unreadable", "missing", "no_invocation"])
def test_process_local_money_waits_wherever_its_invocation_could_still_be_bound(tmp_path, monkeypatch, binding):
    """A run known only to this process: when its invocation's binding cannot be read or has no record, the
    money waits (unknown, never guessed and never a second row); a legacy run with no invocation at all,
    which no read can bind, records under its own name as before."""
    from ouroboros.tools import delegate

    entry = delegate._RunCustody(run_id="run-x", task_id="t-a", route_id="r", model="m", project_id="p",
                                 project_owned=False, root_task_id="t-a", ledger_root=str(tmp_path),
                                 invocation_id="" if binding == "no_invocation" else "inv-x")
    c.record_started(tmp_path, entry, publish=False)
    if binding == "unreadable":
        def unreadable(*_args, **_kwargs):
            raise OSError("the custody store did not answer")
        monkeypatch.setattr(c, "invocation_record", unreadable)
    detail = {"summary": {"runId": "run-x", "state": "succeeded", "spendUsd": 0.25, "spendEstimated": False,
                          "effectiveAccess": "readonly"}, "lastSeq": 2, "pendingInteractions": []}
    money = body.record_terminal_money(tmp_path, entry, detail)
    if binding == "no_invocation":
        assert money["ledger_recorded"] is True, money
        assert [row["run_id"] for row in _rows(tmp_path, c.LEDGER_RECORDED)] == ["run-x"]
    else:
        assert money == {"settled": False, "ledger_recorded": False, "retried": False,
                         "deferred": "custody_publication_unconfirmed"}
        assert _rows(tmp_path, c.LEDGER_RECORDED) == [] and c._CUSTODY["run-x"] is entry and not entry.settled


@pytest.mark.parametrize("consumer", ["retry", "sweep"])
@pytest.mark.parametrize("fault", ["decision_lock", "started_publication"])
def test_a_recovery_that_cannot_publish_binds_nothing_and_sends_nothing(tmp_path, monkeypatch, consumer, fault):
    """The read finds accepted work, but either the decision lock or STARTED publication fails. Neither consumer
    claims durable custody or sends again; the pending token survives, and a later read binds it once."""
    from contextlib import nullcontext

    registry, engine, token = _lost_start(tmp_path, monkeypatch)
    _short_lock_wait(monkeypatch)
    with monkeypatch.context() as patch, (_DecisionHeld(tmp_path, "t-family") if fault == "decision_lock" else nullcontext()):
        if fault == "started_publication":
            emit = c.emit
            patch.setattr(c, "emit", lambda root, kind, row: False if kind == c.STARTED else emit(root, kind, row))
        if consumer == "retry":
            result = _start(registry, retry_of=token)
            assert result["reason"] == "accepted_start_unbound" and result["pending_invocation_id"] == token, result
            assert result.get("custody_durable") is not True
            outcome = result["lookup"]
        else:
            outcome, = _sweep(tmp_path, engine)  # its owner ended: an orphan read
    assert (outcome["action"], outcome["reason"], outcome["lookup"]) == (
        "invocation_retained", "accepted_start_unbound", "found"), outcome
    assert _rows(tmp_path, c.STARTED, token) == [] and len(engine.creates) == 1, "no row and no replay"
    assert c.invocation_record(tmp_path, token)["state"] == "pending" and _held(tmp_path, token) == []
    assert _rows(tmp_path, c.LEDGER_RECORDED) == []
    outcome, = _sweep(tmp_path, engine)
    assert outcome["run_id"] == engine.commands["job-1"]["run"] and len(_rows(tmp_path, c.STARTED, token)) == 1


def test_an_orphan_replay_adopts_the_job_bound_while_its_post_waited(tmp_path, monkeypatch):
    """The sweep's liveness snapshot predates an owner's return. A supported legacy replay waits for its
    handle while that owner reads and binds the job; the replay's run id cannot create a second money row."""
    registry, engine = _delegating(tmp_path, monkeypatch)
    token = "legacy-orphan-replay"
    request = {"prompt": "Summarize the layout.", "access": "readonly", "mode": "ask",
               "primaryHarness": "some-route", "maxSeconds": 120}
    assert c.record_start_requested(tmp_path, task_id="t-family", invocation_id=token, idempotency_key=token,
                                    request=request, project_id="", project_owned=False, route="some-route",
                                    store_identity_unproven=False)
    job = engine.accept(token, request, run=False)

    def returned_owner_reads(job_id):
        payload = _start(registry, retry_of=token)
        assert payload["status"] == "started" and payload["run_id"] == job_id, payload
        engine.bind_run(job_id, state="succeeded")
    engine.on_accept = returned_owner_reads
    outcome, = _sweep(tmp_path, engine, observe=None)  # legacy creating recovery, no preliminary read
    assert outcome["run_id"] == job and outcome["settled"] is True, outcome
    _sweep(tmp_path, engine)
    assert [held.run_id for held in _held(tmp_path, token)] == [job]
    assert [row["run_id"] for row in _rows(tmp_path, c.STARTED, token)] == [job]
    assert [row["run_id"] for row in _rows(tmp_path, c.LEDGER_RECORDED)] == [job]
    assert len(_rows(tmp_path, c.SETTLED, run_id=job)) == 1 and current.run(tmp_path, "run-1") is None
    assert len(engine.creates) == 1 and len(engine.lookups) == 1


def test_an_attach_only_caller_reads_and_settles_accepted_work_with_no_create_ever(tmp_path, monkeypatch):
    """``observe_pending`` without ``replay_pending``: an explicit attach-only recovery caller reads, binds and
    settles the exact found work; a miss is retained, and no create is attempted on any row."""
    registry, engine, token = _lost_start(tmp_path, monkeypatch)
    engine.commands["job-1"]["state"] = "succeeded"
    engine.lose = 1
    missed = _start(registry)["pending_invocation_id"]  # a second lost start the engine forgets entirely
    engine.commands = {job: row for job, row in engine.commands.items() if row["key"] != missed}

    def attach_only():
        gateway = engine.gateway()
        gateway.start_run = lambda *_a, **_k: pytest.fail("an attach-only reader attempted a create")
        return gateway
    outcomes = c.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=attach_only, observe_pending=lambda: True)
    by_invocation = {row.get("invocation_id") or row.get("run_id"): row for row in outcomes}
    settled = next(row for row in outcomes if row.get("action") == "settle_attempted")
    assert settled["settled"] is True and current.run(tmp_path, settled["run_id"]).invocation_id == token
    assert (by_invocation[missed]["reason"], by_invocation[missed]["lookup"]) == ("replay_not_admitted", "not_found")
    assert len(engine.creates) == 2 and len(_rows(tmp_path, c.LEDGER_RECORDED)) == 1


def test_the_creating_replay_alone_reads_nothing(tmp_path, monkeypatch):
    """``replay_pending`` without ``observe_pending`` keeps the legacy creating replay exactly as before:
    no lookup is sent, and the unproven store still refuses the replay."""
    registry, engine, token = _lost_start(tmp_path, monkeypatch)
    outcome, = c.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=engine.gateway, replay_pending=lambda: True)
    assert (outcome["action"], outcome["reason"]) == ("invocation_retained", "invocation_store_unproven"), outcome
    assert engine.lookups == [] and len(engine.creates) == 1 and _rows(tmp_path, c.STARTED, token) == []


@pytest.mark.parametrize("stop_during_lookup", [False, True])
def test_a_creating_replay_rechecks_stop_after_lookup(tmp_path, monkeypatch, stop_during_lookup):
    engine = Engine(tmp_path)
    token = "legacy-missed-start"
    assert c.record_start_requested(tmp_path, run_id="", task_id="gone-owner", invocation_id=token,
                                    idempotency_key=token, request={"prompt": "work", "primaryHarness": "some-route"},
                                    route="some-route", project_id="", project_owned=False)
    stopped = False
    lookup = engine._lookup

    def read_then_stop(request, key, path):
        nonlocal stopped
        answer = lookup(request, key, path)
        stopped = stop_during_lookup
        return answer

    monkeypatch.setattr(engine, "_lookup", read_then_stop)
    outcome, = _sweep(tmp_path, engine, observe=lambda: not stopped, replay=lambda: not stopped)
    assert len(engine.lookups) == 1
    if stop_during_lookup:
        assert outcome["reason"] == "replay_not_admitted" and engine.creates == []
        assert c.invocation_record(tmp_path, token)["state"] == "pending" and not _held(tmp_path, token)
    else:
        assert outcome["action"] == "left_live" and len(engine.creates) == 1
        assert len(_held(tmp_path, token)) == 1


def test_a_stop_arriving_mid_pass_admits_no_new_lookup(tmp_path, monkeypatch):
    """One not-stopping predicate, re-asked before each row: the first row is read, the second meets the stop
    and is neither read nor replayed. Every call is pinned: one more ask would exhaust the script."""
    registry, engine, first = _lost_start(tmp_path, monkeypatch)
    engine.lose = 1
    second = _start(registry)["pending_invocation_id"]
    answers = iter([True, True, False])  # the gate, the first row, the stop before the second row
    outcomes = _sweep(tmp_path, engine, observe=lambda: next(answers), replay=lambda: False)
    assert len(engine.lookups) == 1 and len(engine.creates) == 2
    retained, = [row for row in outcomes if row.get("reason") == "replay_not_admitted"]
    read = {first, second} - {retained["invocation_id"]}
    assert len(read) == 1 and len(_rows(tmp_path, c.STARTED, read.pop())) == 1
    assert "lookup" not in retained or retained["lookup"] == "not_read"


@pytest.mark.parametrize("stale", ["job_only", "job_and_run"])
def test_a_read_listed_before_the_starts_receipt_adopts_its_run(tmp_path, monkeypatch, stale):
    """The same race, start first: a sweep listed the start as pending before its receipt landed, then read
    the engine (the command still runless, or already named). It adopts the bound run and writes nothing."""
    registry, engine = _delegating(tmp_path, monkeypatch)
    engine.queue_runs = True
    listed = {}

    def list_then_bind(job):
        listed["rows"] = c.pending_invocations(tmp_path)
        engine.bind_run(job)
    engine.on_accept = list_then_bind
    payload = _start(registry)
    (job, command), = engine.commands.items()
    token, row = payload["invocation_id"], listed["rows"][0]
    assert payload["run_id"] == command["run"] and row["invocation_id"] == token
    if stale == "job_only":
        engine.answer = (200, {"status": "found", "storeId": "store-a",
                               "handle": {"jobId": job, "runId": None, "state": "queued"}})
    outcome = reconcile._observe_pending_invocation(tmp_path, engine.gateway(), row)
    assert (outcome["action"], outcome["run_id"]) == ("accepted_start_adopted", command["run"]), outcome
    assert [held.run_id for held in _held(tmp_path, token)] == [command["run"]]
    assert len(_rows(tmp_path, c.STARTED, token)) == 1, "the adopting reader writes no row of its own"
    assert len(engine.creates) == 1


@pytest.mark.parametrize("drift", ["root_moved", "snapshot_gone", "workspace_inactive"])
def test_accepted_mutating_work_is_bound_whatever_the_present_target_or_snapshot(tmp_path, monkeypatch, drift):
    """A target, snapshot or write-authority check governs only a new POST: the run the engine already accepted
    is live in its recorded snapshot either way, so its retry binds it (custody, never a second writer) and the
    recorded C1 binding rides the row verbatim; the same drift with nothing accepted still refuses typed."""
    from ouroboros.tools import delegate
    from tests._delegated_transport_shared import _delegating_ctx

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    engine = Engine(tmp_path)
    monkeypatch.setattr(cx, "ClaudexorGateway", lambda *_a, **_k: engine.gateway())
    ctx = _delegating_ctx(tmp_path, acting=True)
    target = tmp_path / "moved" if drift == "root_moved" else ctx.workspace_root
    request = {"prompt": "edit the docs", "access": "workspace_write", "mode": "agent",
               "scope": {"kind": "project", "root": str(target)},
               "execution": {"isolation": "live", "delegated": True}, "primaryHarness": "some-route"}
    for token in ("inv-accepted", "inv-unsent"):
        assert c.record_start_requested(
            tmp_path, run_id="", task_id=ctx.task_id, idempotency_key=token, invocation_id=token, request=request,
            project_id="p", project_owned=False, route="some-route", snapshot_id="snap-gone",
            execution_root=str(tmp_path / "exec"), baseline_sha="b" * 40, target_root=str(target),
            authority_source="acting_constraint", handshake_store_id="store-a", store_identity_unproven=True)
    job = engine.accept("inv-accepted", request)
    if drift == "workspace_inactive":
        ctx.workspace_mode = ""
    bound = json.loads(delegate._delegate_start(ctx, "", retry_of="inv-accepted").text)
    assert (bound["status"], bound["run_id"]) == ("started", engine.commands[job]["run"]), bound
    held, = _held(tmp_path, "inv-accepted")
    assert (held.snapshot_id, held.baseline_sha, held.execution_root, held.target_root, held.access) == (
        "snap-gone", "b" * 40, str(tmp_path / "exec"), str(target), "workspace_write")
    refused = json.loads(delegate._delegate_start(ctx, "", retry_of="inv-unsent").text)
    assert refused["reason"] == {"root_moved": "retry_root_divergence", "snapshot_gone": "execution_snapshot_missing",
                                 "workspace_inactive": "workspace_not_active"}[drift], refused
    assert engine.creates == [] and c.invocation_record(tmp_path, "inv-unsent")["state"] == "pending"


def test_a_binding_found_in_another_store_than_the_handshake_names_its_accepting_store(tmp_path, monkeypatch):
    """The store changed after the proving handshake, before the POST landed: the handshake is no admission
    proof, the exact positive binding is, so the run is bound to the store whose answer named it."""
    registry, engine = _delegating(tmp_path, monkeypatch)
    engine.lose, engine.replace_at_post = 1, "store-b"
    token = _start(registry)["pending_invocation_id"]
    assert c.invocation_record(tmp_path, token)["handshake_store_id"] == "store-a"
    payload = _start(registry, retry_of=token)
    assert payload["status"] == "started", payload
    held, = _held(tmp_path, token)
    assert held.daemon_store_id == "store-b" and len(engine.creates) == 1


def test_a_known_admitting_store_refuses_an_answer_from_another_store(tmp_path, monkeypatch):
    registry, engine = _delegating(tmp_path, monkeypatch, store="store-b")
    request = {"prompt": "continue the stopped run", "mode": "ask", "access": "readonly"}
    assert c.record_start_requested(tmp_path, run_id="", task_id="t-gone", invocation_id="inv-known", idempotency_key="k",
                                    request=request, route="some-route", project_id="", project_owned=False,
                                    daemon_store_id="store-a", store_identity_unproven=False)
    engine.accept("inv-known", request, state="succeeded")  # the same key and body, held by store-b
    outcome, = _sweep(tmp_path, engine)
    assert (outcome["action"], outcome["reason"], outcome["lookup"]) == (
        "invocation_retained", "accepted_start_store_mismatch", "store_mismatch"), outcome
    assert _held(tmp_path, "inv-known") == [] and engine.creates == [], "neither adopted nor re-sent"
    assert c.invocation_record(tmp_path, "inv-known")["state"] == "pending"


PREDECESSOR_NOT_RETAINED = (410, {"code": "continuation_predecessor_not_retained", "retryable": False,
                                  "message": "the run this start continues is no longer retained",
                                  "context": {"continueFrom": "run-predecessor"}})  # the engine's exact 410
REFUSED = {"request_conflict", "project_not_registered", "continuation_predecessor_not_retained"}


@pytest.mark.parametrize("case", ["other_store_partition", "request_conflict", "transport_fault", "no_store_identity",
                                  "malformed", "old_route_bare_404", "catalog_without_lookup", "project_not_registered",
                                  "store_identity_unavailable", "recovery_unavailable", "run_id_missing",
                                  "run_id_not_a_name", "state_unknown", "state_missing", "turn_only_for_a_run",
                                  "continuation_predecessor_not_retained"])
def test_every_answer_short_of_a_binding_keeps_the_start_pending_unknown_and_unsent(tmp_path, monkeypatch, case):
    registry, engine, token = _lost_start(tmp_path, monkeypatch, lookup=case != "catalog_without_lookup")
    if case == "other_store_partition":
        engine.store = "store-b"  # the same key lives only in store-a's partition: a scoped miss, never non-dispatch
    elif case == "request_conflict":
        engine.commands["job-1"]["body"] = {**engine.creates[0][2], "prompt": "another request under this key"}
    else:
        engine.answer = {"transport_fault": (503, {"code": "daemon_busy", "message": "busy"}),
                         "no_store_identity": (200, {"status": "found", "handle": {
                             "jobId": "job-1", "runId": "run-1", "state": "running"}}),
                         "malformed": (200, {"status": "found", "storeId": "store-a", "handle": {"runId": "run-1"}}),
                         "old_route_bare_404": (404, None),
                         "project_not_registered": (404, {"code": "project_not_registered", "retryable": False,
                                                          "message": "the run's project root is not registered"}),
                         "store_identity_unavailable": (503, {"code": "store_identity_unavailable", "retryable": True,
                                                              "message": "no store identity"}),
                         "recovery_unavailable": (501, {"code": "thread_recovery_unavailable",
                                                        "message": "no durable thread-turn reads"}),
                         # The strict union's required fields: a missing runId is no explicit null, a state outside
                         # ControlRunState is no command state; neither may bind the job id as an accepted run.
                         "run_id_missing": (200, {"status": "found", "storeId": "store-a", "handle": {
                             "jobId": "job-1", "state": "queued", "runDir": None, "taskId": None}}),
                         "run_id_not_a_name": (200, {"status": "found", "storeId": "store-a", "handle": {
                             "jobId": "job-1", "runId": 7, "state": "running", "runDir": None, "taskId": None}}),
                         "state_unknown": (200, {"status": "found", "storeId": "store-a", "handle": {
                             "jobId": "job-1", "runId": None, "state": "accepted", "runDir": None, "taskId": None}}),
                         "state_missing": (200, {"status": "found", "storeId": "store-a", "handle": {
                             "jobId": "job-1", "runId": None, "runDir": None, "taskId": None}}),
                         "turn_only_for_a_run": (200, {"status": "turn_only", "storeId": "store-a", "threadId": "th-1",
                                                       "turnId": "tn-1", "runId": "run-1", "enqueueError": None}),
                         "continuation_predecessor_not_retained": PREDECESSOR_NOT_RETAINED,
                         }.get(case)
    expected = {"other_store_partition": "not_found", "request_conflict": "refused", "transport_fault": "unavailable",
                "no_store_identity": "unavailable", "malformed": "unavailable", "old_route_bare_404": "unsupported",
                "catalog_without_lookup": "unsupported", "project_not_registered": "refused",
                "store_identity_unavailable": "unavailable", "recovery_unavailable": "unsupported",
                "continuation_predecessor_not_retained": "refused"}.get(case, "unavailable")
    kept = "accepted_start_refused" if case in REFUSED else "invocation_store_unproven"  # a refusal: no replay at all
    outcome, = _sweep(tmp_path, engine)
    assert (outcome["action"], outcome["reason"], outcome["lookup"]) == ("invocation_retained", kept, expected), outcome
    refused = _start(registry, retry_of=token)
    assert refused["reason"] == kept, refused
    assert len(engine.creates) == 1, "a miss never licenses a POST"
    assert len(engine.lookups) == (0 if case == "catalog_without_lookup" else 2)
    assert _held(tmp_path, token) == [] and not any(row.get("definite") for row in _rows(tmp_path, c.START_FAILED, token))
    assert c.invocation_record(tmp_path, token)["state"] == "pending"


@pytest.mark.parametrize("answer", ["not_found", "continuation_predecessor_not_retained", "request_conflict"])
def test_a_typed_read_refusal_never_falls_through_to_a_replay_its_proven_store_would_allow(tmp_path, monkeypatch, answer):
    """An orphaned continuation start whose admitting store is proven (it names the store its predecessor ran in,
    the store serving now), its reply lost. A plain miss there is replayed, the branch's unchanged rule (control).
    A typed refusal to read it, the engine's 410 for a predecessor no longer retained or a key conflict, sends
    nothing: the same body's create would be refused alike and recorded a definite failure, retiring a start
    the engine may have accepted. It stays pending and unknown."""
    registry, engine = _delegating(tmp_path, monkeypatch)
    request = {"prompt": "continue the stopped run", "mode": "ask", "access": "readonly", "continueFrom": "run-predecessor"}
    assert c.record_start_requested(tmp_path, run_id="", task_id="t-gone", invocation_id="inv-cont", idempotency_key="k",
                                    request=request, route="some-route", project_id="", project_owned=False,
                                    daemon_store_id="store-a", store_identity_unproven=False)
    if answer != "not_found":
        engine.answer = PREDECESSOR_NOT_RETAINED if answer != "request_conflict" else (
            409, {"code": "idempotency_conflict", "message": "the key holds another request", "retryable": False})
        engine.create_refusal = engine.answer  # what the engine's create would answer the same body
    outcome, = _sweep(tmp_path, engine)
    if answer == "not_found":
        assert len(engine.creates) == 1 and len(engine.lookups) == 1, ("the proven store replays a plain miss", outcome)
        return
    assert engine.creates == [] and _rows(tmp_path, c.START_FAILED, "inv-cont") == [], ("no POST, no definite failure",
                                                                                          outcome)
    assert (outcome["action"], outcome["reason"], outcome["lookup"]) == (
        "invocation_retained", "accepted_start_refused", "refused"), outcome
    assert outcome["lookup_detail"].startswith(engine.answer[1]["code"]), outcome
    assert c.invocation_record(tmp_path, "inv-cont")["state"] == "pending"


def test_an_explicit_retry_whose_read_is_refused_sends_nothing_where_its_store_permits_replay(tmp_path, monkeypatch):
    """The explicit retry, in a branch whose store rule permits replay (an engine naming no store identity:
    legacy single-store), reads its start, meets the engine's 410 and refuses typed: no POST, no definite
    failure, the invocation pending and unknown."""
    registry, engine, token = _lost_start(tmp_path, monkeypatch, store="")
    engine.answer = engine.create_refusal = PREDECESSOR_NOT_RETAINED
    refused = _start(registry, retry_of=token)
    assert len(engine.creates) == 1 and len(engine.lookups) == 1, ("only the lost original POST was ever sent", refused)
    assert (refused["reason"], refused["pending_invocation_id"], refused["lookup"]["lookup"]) == (
        "accepted_start_refused", token, "refused"), refused
    assert not any(row.get("definite") for row in _rows(tmp_path, c.START_FAILED, token))
    assert c.invocation_record(tmp_path, token)["state"] == "pending"


def _thread_start(tmp_path, engine, *, invocation_id="inv-turn", run=None, job=False):
    """A recorded thread-turn start whose command the engine no longer retains (or still does)."""
    request = {"prompt": "next turn", "mode": "ask", "access": "readonly", "_use_thread": True, "_thread_id": "th-1",
               "scope": {"kind": "project", "root": str(tmp_path)}, "execution": {"isolation": "envelope"}}
    assert c.record_start_requested(tmp_path, run_id="", task_id="t-gone", invocation_id=invocation_id,
                                    idempotency_key="k-turn", request=request, route="some-route", project_id="",
                                    project_owned=False, handshake_store_id="store-a", store_identity_unproven=True)
    command = engine.accept("unused-key", {}, state="succeeded") if job or run else None
    if run:  # the turn's run outlived its pruned command
        engine.commands[command]["key"] = None
    engine.turns[("store-a", "th-1", invocation_id)] = {
        "turn": "tn-1", "job": command if job else None, "run": engine.commands[command]["run"] if run else None,
        "error": None if run or job else ENQUEUE_ERROR}
    return request


def test_a_runless_retained_turn_stays_named_unknown_and_is_never_enqueued(tmp_path, monkeypatch):
    registry, engine = _delegating(tmp_path, monkeypatch)
    _thread_start(tmp_path, engine)
    outcome, = _sweep(tmp_path, engine)
    assert (outcome["action"], outcome["reason"], outcome["thread_id"], outcome["turn_id"]) == (
        "invocation_retained", "accepted_turn_without_run", "th-1", "tn-1"), outcome
    assert outcome["enqueue_error"] == ENQUEUE_ERROR, "the turn's recorded refusal, kept whole"
    (kind, key, sent), = engine.lookups
    assert (kind, key) == ("th-1", "inv-turn")
    assert "scope" not in sent and "_thread_id" not in sent and sent["execution"] == {"isolation": "envelope"}, (
        "the turn lookup carries the append's own projected body")
    assert engine.creates == [] and _held(tmp_path, "inv-turn") == []
    assert _rows(tmp_path, c.START_FAILED, "inv-turn") == [], "no fabricated failure, job or zero run"
    assert c.invocation_record(tmp_path, "inv-turn")["state"] == "pending"


@pytest.mark.parametrize("retained", ["run", "command"])
def test_a_retained_turn_binds_its_exact_run(tmp_path, monkeypatch, retained):
    registry, engine = _delegating(tmp_path, monkeypatch)
    _thread_start(tmp_path, engine, run=retained == "run", job=retained == "command")
    run = next(iter(engine.commands.values()))["run"]
    outcome, = _sweep(tmp_path, engine)
    assert (outcome["run_id"], outcome["settled"]) == (run, True), outcome
    started, = _rows(tmp_path, c.STARTED, "inv-turn")
    assert started["accepted_start_lookup"] == ("turn_only" if retained == "run" else "found")
    assert engine.creates == []


def test_lost_response_then_pause_then_lookup_settles_under_pause_and_resume_returns_the_full_result(
        tmp_path, monkeypatch):
    """The NX shape: an ended member's start was accepted, its reply lost; the owner's Pause holds the tree
    UNKNOWN. The admitted sweep reads and settles that run while the Pause stands; the authorized Resume then
    proceeds with the complete retained output, and nothing was sent again."""
    from ouroboros.gateways import claudexor as gw
    from supervisor.owner_pause_control import request_owner_pause
    from tests._owner_pause_ended_helpers import _fence, _kept, _park_root, _tail_tree, _tick

    request = {"prompt": "continue the review", "mode": "ask", "access": "readonly", "continueFrom": "run-earlier",
               "scope": {"kind": "project", "root": str(tmp_path)}, "primaryHarness": "codex", "harnesses": ["codex"]}

    def lost_start(root):
        assert c.record_start_requested(root, run_id="", task_id="child-dead", root_task_id="root-1",
                                        invocation_id="inv-1", idempotency_key="k-1", request=request, route="codex",
                                        project_id="", project_owned=False, continuation_of="run-earlier",
                                        handshake_store_id="store-a", store_identity_unproven=True)
    queue, workers, _stand_in = _tail_tree(tmp_path, monkeypatch, tail=lost_start)
    engine = Engine(tmp_path)
    job = engine.accept("inv-1", request, state="succeeded")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *_a, **_k: engine.gateway())

    assert request_owner_pause("root-1", request_id="press")["ok"]
    assert _tick(queue) == []
    assert [row["stop_outcome"] for row in _kept(tmp_path)["external_runs"]["runs"]] == ["pending_invocation_unbound"]
    _park_root(tmp_path, monkeypatch, queue, workers)
    refused = queue.resume_budget_paused_task("root-1")
    assert refused["ok"] is False and refused["error"] == "owner_pause_effects_unsettled", refused

    outcome, = c.reconcile_orphaned_runs(tmp_path, set(workers.RUNNING), gateway_factory=engine.gateway,
                                         replay_pending=lambda: True, observe_pending=lambda: True)
    run = engine.commands[job]["run"]
    assert (outcome["run_id"], outcome["action"], outcome["settled"]) == (run, "settle_attempted", True), outcome
    assert _fence(tmp_path)["state"] == "requested", "reading and settling accepted work lifts no Pause"
    _tick(queue)
    assert _fence(tmp_path)["state"] == "paused" and engine.cancels == []
    resumed = queue.resume_budget_paused_task("root-1")
    assert resumed["ok"] is True, resumed
    held = current.run(tmp_path, run)
    assert held.settled and held.terminal_state == "succeeded" and held.cost_usd == 0.25
    assert held.continuation_of == "run-earlier" and held.task_id == "child-dead"
    assert body._read_source(tmp_path, held, held.terminal_source)["primaryOutput"]["text"] == "the complete answer"
    assert engine.creates == [], "no POST at any step: the review is not repeated"


# -- a review seat's own rejoin ---------------------------------------------------------------------------------

def _review_seat(tmp_path, *, operation_id="", generation_end_at="", restore=None, replayable=False):
    """A delegated review seat's pending START: recorded, its POST outcome unknown. ``replayable``: a row
    naming no store, which the store rule lets the seat replay (the earlier single-store behavior)."""
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "t-root", "running", root_task_id="t-root")
    request = {"prompt": "review this", "instructions": "review", "authPreference": "subscription", "mode": "ask",
               "access": "readonly", "scope": {"kind": "project", "root": "/tmp/fake-repo"},
               "harnesses": ["fake-review"], "primaryHarness": "fake-review", "maxSeconds": 30, "model": "fake-small"}
    assert c.record_start_requested(
        tmp_path, run_id="", task_id="t-b", idempotency_key="k-seat", invocation_id="inv-seat", max_seconds=30,
        request=request, project_id="", project_owned=False, route="fake-review", surface="scope_review",
        slot_id="scope_slot_1", operation_id=operation_id, root_task_id="t-root", source="review_substrate",
        **({"store_identity_unproven": False} if replayable else {"handshake_store_id": "store-a",
                                                                  "store_identity_unproven": True}),
        **({"generation_end_at": generation_end_at} if generation_end_at else {}),
        **({"review_subject_restore": restore} if restore else {}))


def _review_lookup(monkeypatch, gateway_class, *, supported=True):
    reads = []

    def lookup_run(self, request, *, idempotency_key):
        reads.append((idempotency_key, request))
        return {"status": "found", "storeId": "store-a",
                "handle": {"jobId": "job-seat", "runId": "run-1", "state": "succeeded"}}
    monkeypatch.setattr(gateway_class, "operations", lambda self: LOOKUPS if supported else [], raising=False)
    monkeypatch.setattr(gateway_class, "lookup_run", lookup_run, raising=False)
    return reads


def test_a_review_seat_rejoins_its_accepted_start_with_no_second_post(tmp_path, monkeypatch, fake_route):
    from tests._review_session_route_shared import _run_session_directly, _terminal_detail

    _review_seat(tmp_path)
    reads = _review_lookup(monkeypatch, fake_route)
    fake_route.detail = _terminal_detail('{"findings": []}', conformance="passed")
    state = {"pending_invocation_id": "inv-seat"}
    facts = _run_session_directly(tmp_path, retry_state=state)
    assert facts["run_id"] == "run-1" and facts["idempotent_recovery"] is True and '"findings"' in facts["text"]
    assert [key for key, _ in reads] == ["inv-seat"]
    assert [request for gateway in fake_route.instances for request in gateway.start_requests] == []
    started, = _rows(tmp_path, c.STARTED, "inv-seat")
    assert started["accepted_start_lookup"] == "found" and started["source"] == "review_substrate"
    assert c.invocation_record(tmp_path, "inv-seat")["state"] == "started"
    assert "pending_invocation_id" not in state, "the seat's collected result releases its token"


def test_a_review_seat_whose_read_is_refused_sends_nothing_where_its_store_permits_replay(
        tmp_path, monkeypatch, fake_route):
    """The panel's own rejoin, its seat replayable under the store rule: the engine's 410 for the read refuses
    the seat typed, with no second POST and no definite failure, the start kept pending and unknown."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.review_execution import ReviewRouteUnavailable
    from tests._review_session_route_shared import _run_session_directly

    _review_seat(tmp_path, replayable=True)
    status, problem = PREDECESSOR_NOT_RETAINED

    def refused_read(self, request, *, idempotency_key):
        raise ClaudexorUnavailable(problem["code"], problem["message"], status_code=status)
    monkeypatch.setattr(fake_route, "operations", lambda self: LOOKUPS, raising=False)
    monkeypatch.setattr(fake_route, "lookup_run", refused_read, raising=False)
    try:
        _run_session_directly(tmp_path, retry_state={"pending_invocation_id": "inv-seat"})
        code = ""
    except ReviewRouteUnavailable as exc:
        code = exc.code
    posted = [request for gateway in fake_route.instances for request in gateway.start_requests]
    assert (posted, code) == ([], "accepted_start_refused"), "the seat's read refusal re-sent nothing"
    assert not any(row.get("definite") for row in _rows(tmp_path, c.START_FAILED, "inv-seat"))
    assert c.invocation_record(tmp_path, "inv-seat")["state"] == "pending"


@pytest.mark.parametrize("supported", [True, False])
def test_an_accepted_continuation_is_collected_after_its_window_closed_and_its_checkout_went(
        tmp_path, monkeypatch, fake_route, supported):
    """An explicit continuation's recorded start, its generation window over and its released checkout
    gone: both bound only a new POST. Read first, its accepted run is collected; without the read the
    start is refused before any POST, exactly as before."""
    from ouroboros.review_continuation import PREFIX
    from ouroboros.review_execution import ReviewRouteUnavailable
    from tests._review_session_route_shared import _run_session_directly, _terminal_detail

    closed = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    gone = tmp_path / "released" / "checkout"
    restore = {"root": str(gone), "owner_task": "t-b", "sources": [],
               "subject": {"root": str(tmp_path / "missing-repo"), "base": "0" * 40, "tree_sha": "1" * 40}}
    operation = PREFIX + "seat"
    _review_seat(tmp_path, operation_id=operation, generation_end_at=closed, restore=restore)
    reads = _review_lookup(monkeypatch, fake_route, supported=supported)
    fake_route.detail = _terminal_detail('{"findings": []}', conformance="passed")
    state = {"pending_invocation_id": "inv-seat"}
    if supported:
        facts = _run_session_directly(tmp_path, retry_state=state, operation_id=operation)
        assert facts["run_id"] == "run-1" and facts["late_success_accepted"] is True
        assert not gone.exists(), "a known run needs no checkout put back"
    else:
        with pytest.raises(ReviewRouteUnavailable) as refused:
            _run_session_directly(tmp_path, retry_state=state, operation_id=operation)
        assert refused.value.code == "review_subject_unavailable"
        assert reads == [] and c.invocation_record(tmp_path, "inv-seat")["state"] == "pending"
    assert [request for gateway in fake_route.instances for request in gateway.start_requests] == []
