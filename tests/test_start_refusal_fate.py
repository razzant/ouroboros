"""An engine answer to an ENTERED start POST is no proof the start created nothing (#3138, audit 980c602f).

Claudexor admits a command, can end it before any run and answer its own start with that command's 4xx
(``context.jobId``/``state``) while the POST still waits for the run's binding; later the same fact arrives
as 202 with the job id. The gateway also synthesizes a ControlProblem for a non-JSON body. So the delegated
start (``delegate_start``) and the admitted sweep's replay (``_recover_pending_invocation``) keep such a start's
token, registration and execution snapshot, read its accepted work through the engine's declared noncreating
lookup (``delegate_accepted.accepted_after_refusal``) and bind it once for the ordinary terminal owners. A miss
(not found, no lookup route, another store, a refused read) binds, sends and removes nothing. Only a start that
never entered its POST is a definite refusal.

Engine doubles only: the production gateway over a scripted transport (``Engine``) or a stub. No live engine,
provider or money.
"""
from __future__ import annotations

import json

import httpx
import pytest

from ouroboros import delegate_collection as body, delegate_custody as c, delegate_custody_current as current
from ouroboros.gateways import claudexor as cx
from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _delegate_registry,
    _owned_gateway_uses_each_test_transport,
    _select_transport_access,
)
from tests._delegated_transport_shared import Engine, _rows, _start, _sweep
from tests.test_delegate_continuation import _snapshot_count, _start as _stub_start

_REAL_GATEWAY = cx.ClaudexorGateway  # bound at import, before any test patches the name
WINDOW = {"code": "subscription_window_exhausted", "resetsAt": "2026-11-16T12:00:00Z",
          "safeMessage": "Subscription window spent before the run started"}


class _AcceptThenRefuse(Engine):
    """A creating POST admits its command, which ends before any run; the POST answers with that command's own
    ControlProblem (its status, ``context.jobId``, ``context.state``), as the engine does inside its bind wait."""

    def __init__(self, tmp_path, *, refuse=None, **kwargs):
        super().__init__(tmp_path, **kwargs)
        self.refuse = refuse  # (status, failure) the admitted command ends with, answered to its own POST

    def __call__(self, request):
        if not (request.url.path == "/v2/runs" and request.method == "POST" and self.refuse):
            return super().__call__(request)
        key, sent = request.headers.get("Idempotency-Key"), json.loads(request.read())
        self.creates.append((key, self.store, sent))
        status, failure = self.refuse
        job = self._held(key) or self.accept(key, sent, run=False, state="failed", spend=None, failure=failure,
                                             problem={"code": failure["code"], "message": failure["safeMessage"],
                                                      "status": status})
        return httpx.Response(status, json={"code": failure["code"], "message": failure["safeMessage"],
                                            "retryable": False, "context": {"jobId": job, "state": "failed"}})


def _delegating(tmp_path, monkeypatch, engine):
    _select_transport_access(monkeypatch, "readonly")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(cx, "ClaudexorGateway", lambda *_a, **_k: engine.gateway())
    return _delegate_registry(tmp_path, monkeypatch)


def _synthesized(status):
    """The gateway's own ControlProblem for an engine answer whose body is not JSON (``http_<status>``)."""
    gateway = _REAL_GATEWAY(cx.DaemonEndpoint(host="127.0.0.1", port=1, token="fixture-token"))
    try:
        return gateway._problem(httpx.Response(status, text="<html>proxy refused the request</html>"), "POST")
    finally:
        gateway.close()


def _requested(root):
    return [row for row in c.custody_rows(root) if row["type"] == c.START_REQUESTED]


# -- the delegated start --------------------------------------------------------------------------------------

@pytest.mark.parametrize("timing", ["early_4xx", "late_202"])
def test_an_accepted_job_bound_from_its_refused_start_settles_once_like_its_late_receipt(tmp_path, monkeypatch,
                                                                                         timing):
    """The same admitted command, ended before any run with a spent subscription window. Answered early (its own
    429 inside the bind wait) the start reads and binds the job; answered late (202 with the job id) the start's
    own receipt binds it. Either way: started on the job, one STARTED row, one settlement through the job with the
    engine's typed failure and no cost (unknown, never zero), nothing sent again, nothing retired."""
    engine = _AcceptThenRefuse(tmp_path, refuse=(429, WINDOW) if timing == "early_4xx" else None)
    registry = _delegating(tmp_path, monkeypatch, engine)
    if timing == "late_202":
        engine.queue_runs = True  # admitted, its 202 names only the job
        payload = _start(registry)
    else:  # the early answer still names the one file this start prepared before its first send
        from ouroboros.tools import delegate

        drive = tmp_path / "task_drives" / registry._ctx.task_id
        drive.mkdir(parents=True, exist_ok=True)
        (drive / "one.md").write_bytes(b"known")
        payload = json.loads(delegate._delegate_start(registry._ctx, "Summarize the layout.", max_seconds=120,
                                                      input_files=[{"root": "task_drive", "path": "one.md"}]).text)
    (job, command), = engine.commands.items()
    if timing == "late_202":
        command.update(state="failed", failure=dict(WINDOW), spend=None)  # it ends before any run, after the 202
    token = _requested(tmp_path)[0]["invocation_id"]
    assert (payload["status"], payload["run_id"], payload["invocation_id"]) == ("started", job, token), payload
    started, = _rows(tmp_path, c.STARTED, token)
    if timing == "early_4xx":
        assert payload["lookup"]["start_refusal"] == {"code": "subscription_window_exhausted", "status": 429}
        assert (started["accepted_start_lookup"], started["start_refusal"]["status"]) == ("found", 429)
        assert "the engine answered this start with an error" in payload["note"].lower()
        assert "inputs: 1 file, 5 bytes, prepared before the first send" in payload["note"], payload["note"]
    assert c.invocation_record(tmp_path, token)["state"] == "started" and c.pending_invocations(tmp_path) == []
    for _ in range(2):  # the body's ordinary collection, repeated: one settlement, one money row
        body.collect_terminal(tmp_path, job, gateway=engine.gateway(), detail=engine.gateway().get_run(job))
    settled = current.run(tmp_path, job)
    assert settled.settled and settled.terminal_state == "failed" and settled.cost_usd is None
    row, = _rows(tmp_path, c.SETTLED, run_id=job)
    assert (row["failure_code"], row["cost_usd"]) == ("subscription_window_exhausted", None), "typed, never success"
    assert len(_rows(tmp_path, c.LEDGER_RECORDED)) == 1 and _rows(tmp_path, c.START_FAILED, token) == []
    assert len(engine.creates) == 1 and not any(method == "DELETE" for method, _path in engine.paths)


@pytest.mark.parametrize("miss", ["not_found", "unsupported", "refused"])
def test_a_refused_start_whose_read_binds_nothing_keeps_its_token_and_never_creates(tmp_path, monkeypatch, miss):
    """The engine refused the POST without admitting it (here truly: nothing is held), but no answer says so:
    the read finds no binding in its store, the engine declares no read, or the read is refused. None is proof
    of non-admission and none licenses a POST: the start keeps its token, pending and unknown, and an explicit
    retry reads again and still sends nothing (its admitting store stays unproven, or the read refuses it). A
    found binding in another store is covered where an admitting store is known (the review seat's suite)."""
    engine = Engine(tmp_path, lookup=miss != "unsupported")
    registry = _delegating(tmp_path, monkeypatch, engine)
    engine.create_refusal = (400, {"code": "execution_workspace_invalid", "message": "workspace refused",
                                   "retryable": False})
    engine.answer = (409, {"code": "idempotency_conflict", "message": "the key holds another request",
                           "retryable": False}) if miss == "refused" else None
    refused = _start(registry)
    token = _requested(tmp_path)[0]["invocation_id"]
    assert (refused["status"], refused["reason"], refused["pending_invocation_id"]) == (
        "refused", "execution_workspace_invalid", token), refused
    assert "definitely_unrun" not in refused
    assert len(engine.lookups) == (0 if miss == "unsupported" else 1)
    assert c.invocation_record(tmp_path, token)["state"] == "pending" and _rows(tmp_path, c.STARTED, token) == []
    assert [row["definite"] for row in _rows(tmp_path, c.START_FAILED, token)] == [False]
    retried = _start(registry, retry_of=token)
    assert (retried["status"], retried["reason"]) == ("refused", "accepted_start_refused" if miss == "refused"
                                                      else "invocation_store_unproven"), retried
    assert len(engine.creates) == 1, "a miss never licenses a POST"
    assert c.invocation_record(tmp_path, token)["state"] == "pending"
    assert not any(method == "DELETE" for method, _path in engine.paths)


ANSWERED = {
    "engine_400": lambda: cx.ClaudexorUnavailable("request_invalid", "the engine refused the start", status_code=400),
    "retired_profile_409": lambda: cx.ClaudexorUnavailable("retired_access_profile", "retired", status_code=409),
    "handshake_426": lambda: cx.ClaudexorUnavailable("handshake_required", "handshake first", status_code=426),
    "non_json_400": lambda: _synthesized(400),
}


@pytest.mark.parametrize("answer", [*ANSWERED, "launch_fence"])
def test_a_writable_start_keeps_its_execution_snapshot_unless_its_post_was_never_entered(tmp_path, monkeypatch,
                                                                                       answer):
    """A mutating start provisions its private execution snapshot before the POST. Any answer to the entered POST
    keeps the snapshot in custody's keep-set beside the pending token (the engine stub declares no read, so
    nothing is bound). Only the launch fence, refusing before the POST, proves no run can use it: then, and only
    then, the snapshot is removed and the token retired."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.owner_pause import OwnerPauseRefused

    def fenced(*_args, **_kwargs):
        raise OwnerPauseRefused("owner_pause_fenced")

    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snap_root"))  # as the start helper sets
    before = _snapshot_count()
    if answer == "launch_fence":
        monkeypatch.setattr(delegate, "run_operation", fenced)
    _request, payload, _ctx, calls = _stub_start(tmp_path, monkeypatch, acting=True, run_id="run-never",
                                                 start_error=ANSWERED[answer]() if answer in ANSWERED else None)
    requested, = _requested(tmp_path)
    token, snapshot = requested["invocation_id"], requested["snapshot_id"]
    assert snapshot, requested
    fate = c.invocation_record(tmp_path, token)["state"]
    failed, = [row["definite"] for row in _rows(tmp_path, c.START_FAILED, token)]
    if answer == "launch_fence":
        assert (len(calls), payload["definitely_unrun"], "pending_invocation_id" in payload) == (0, True, False)
        assert (fate, failed, _snapshot_count()) == ("failed_definite", True, before)
        assert snapshot not in c.open_snapshot_ids(tmp_path)
    else:
        assert (len(calls), payload["pending_invocation_id"], "definitely_unrun" in payload) == (1, token, False)
        assert (fate, failed, _snapshot_count()) == ("pending", False, before + 1)
        assert snapshot in c.open_snapshot_ids(tmp_path), "the pending invocation keeps its snapshot open"


# -- the admitted sweep's replay ------------------------------------------------------------------------------

def _orphan(tmp_path, *, proven=True):
    """A pending start of an ended task, with an admitting-store receipt unless testing a legacy gap."""
    request = {"prompt": "summarize", "mode": "ask", "access": "readonly", "primaryHarness": "some-route",
               "harnesses": ["some-route"], "scope": {"kind": "project", "root": str(tmp_path)}}
    assert c.record_start_requested(tmp_path, run_id="", task_id="t-gone", invocation_id="inv-orphan",
                                    idempotency_key="k-orphan", request=request, route="some-route",
                                    project_id="prj-owned", project_owned=True, store_identity_unproven=False,
                                    daemon_store_id="store-a" if proven else "")
    return request


@pytest.mark.parametrize("read", ["admitted", "replay_only"])
def test_a_sweep_replay_answered_with_its_jobs_refusal_is_read_and_bound_or_kept(tmp_path, monkeypatch, read):
    """The sweep replays an orphan's recorded start; the engine admits it, the command ends before any run and
    answers the replay with its own 400. With a read admitted the pass reads the job at once and settles it
    through the ordinary reconciliation; with only the replay admitted it retires nothing (the owned
    registration, the pending invocation) and a later pass's read binds the same job without another POST.
    (Once settled, the run's owned registration takes its ordinary settled-run retirement lane.)"""
    failure = {"code": "execution_workspace_invalid", "safeMessage": "the workspace went before the run"}
    engine = _AcceptThenRefuse(tmp_path, refuse=(400, failure))
    _orphan(tmp_path)
    outcome, = _sweep(tmp_path, engine, observe=(lambda: True) if read == "admitted" else None)
    (job, _command), = engine.commands.items()
    if read == "replay_only":
        assert (outcome["action"], outcome["reason"], outcome["lookup"]) == (
            "invocation_retained", "start_refusal_unproven", "not_read"), outcome
        assert c.invocation_record(tmp_path, "inv-orphan")["state"] == "pending"
        kept, = _rows(tmp_path, c.START_FAILED, "inv-orphan")
        assert (kept["definite"], kept["project_retired"], kept["reason"]) == (
            False, False, "recovery_refused_execution_workspace_invalid")
        assert not any(method == "DELETE" for method, _path in engine.paths), "the owned registration is kept"
        outcome, = _sweep(tmp_path, engine)  # the next admitted pass reads first: found, bound, never re-sent
    assert (outcome["run_id"], outcome["action"], outcome["settled"]) == (job, "settle_attempted", True), outcome
    bound, = [row for row in _rows(tmp_path, c.RECONCILED) if row.get("action") == "accepted_start_bound"]
    assert (bound["run_id"], bound["state"]) == (job, "failed")
    if read == "admitted":
        assert bound["start_refusal"] == {"code": "execution_workspace_invalid", "status": 400}
    row, = _rows(tmp_path, c.SETTLED, run_id=job)
    assert (row["state"], row["failure_code"], row["cost_usd"]) == ("failed", "execution_workspace_invalid", None)
    assert c.invocation_record(tmp_path, "inv-orphan")["state"] == "started"
    assert not any(r.get("definite") for r in _rows(tmp_path, c.START_FAILED, "inv-orphan"))
    assert len(engine.creates) == 1


@pytest.mark.parametrize("origin", ["fresh", "legacy"])
@pytest.mark.parametrize("lookup", ["not_found", "unsupported", "refused", "unavailable", "store_mismatch"])
def test_an_automatic_sweep_never_reposts_identityless_pending_after_refusal(
        tmp_path, monkeypatch, origin, lookup):
    """Unknown entered POST → ended owner → replay-only passes → read pass. Without the admitting
    store none of those later passes may create. A handshake or a negative read isn't that receipt."""
    engine = Engine(tmp_path, store="", lookup=lookup != "unsupported")
    if origin == "fresh":
        registry = _delegating(tmp_path, monkeypatch, engine)
        engine.create_refusal = (400, {"code": "execution_workspace_invalid", "message": "refused", "retryable": False})
        refused = _start(registry)
        token = refused["pending_invocation_id"]
        assert refused["reason"] == "execution_workspace_invalid"
    else:
        _orphan(tmp_path, proven=False)  # the old row has no admitting store and no proven receipt
        token = "inv-orphan"
    original_creates = len(engine.creates)
    engine.create_refusal = None  # replay would now create fresh work if the sweep attempted it
    for _ in range(2):
        outcome, = _sweep(tmp_path, engine, observe=None)
        assert outcome["reason"] == "invocation_store_unproven", outcome
        assert len(engine.creates) == original_creates
        c._CUSTODY.clear()  # another pass after the owner's process ended, only durable rows remain
    engine.store = "store-a"  # a newly identifying handshake still doesn't prove admission
    if lookup == "refused":
        engine.answer = (409, {"code": "idempotency_conflict", "message": "other body", "retryable": False})
    elif lookup == "unavailable":
        engine.answer = (503, {"code": "unavailable", "message": "read unavailable", "retryable": True})
    elif lookup == "store_mismatch":
        engine.answer = (200, {"status": "not_found", "storeId": "store-b"})
    outcome, = _sweep(tmp_path, engine)
    assert outcome["action"] == "invocation_retained"
    assert len(engine.creates) == original_creates
    assert c.invocation_record(tmp_path, token)["state"] == "pending"
    assert not any(method == "DELETE" for method, _path in engine.paths)
    # A later positive read of the exact key/body/store still joins accepted work without POST.
    engine.lookup, engine.answer = True, None
    if lookup == "unsupported":  # a later engine build publishes a new descriptor generation
        from dataclasses import replace
        from pathlib import Path
        from ouroboros.gateways.claudexor_metadata import descriptor_stamp
        descriptor = Path(engine.endpoint.descriptor_path)
        descriptor.write_text('{"generation": 2}')
        engine.endpoint = replace(engine.endpoint, descriptor_identity=descriptor_stamp(descriptor))
    request = c.invocation_record(tmp_path, token)["request"]
    job = engine.accept(token, request, run=False, state="failed", failure=WINDOW, spend=None)
    outcome, = _sweep(tmp_path, engine)
    assert (outcome["run_id"], outcome["settled"]) == (job, True)
    assert len(engine.creates) == original_creates
    assert len(_rows(tmp_path, c.LEDGER_RECORDED)) == 1
    settled, = _rows(tmp_path, c.SETTLED, run_id=job)
    assert settled["failure_code"] == "subscription_window_exhausted" and settled["cost_usd"] is None
