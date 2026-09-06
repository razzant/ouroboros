"""Custody of a delegated run is durable, not process-local.

Split verbatim out of ``tests/test_delegated_subagent_transport.py`` by theme. This
module owns the custody rows that outlive the worker that wrote them: invocation
identity across retries, the registration a failed start must not leave behind,
recovery of a pending invocation whose worker died, and the shared project
registration only its canonical sharer retires.
"""

from __future__ import annotations

import json
import pathlib
import httpx
import pytest
from ouroboros import subagents
from ouroboros.config import (
    CLAUDEXOR_MIN_VERSION,
    CLAUDEXOR_PROTOCOL_MAJOR,
)
from ouroboros.gateways import claudexor as cx

from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _LiveRunStub,
    _event_types,
    _nanny_ctx,
    _owned_gateway_uses_each_test_transport,
    _transport_snapshot,
)


def test_custody_survives_the_worker_that_started_the_run(tmp_path, monkeypatch):
    """A worker crash, a restart or a lost response used to leave a LIVE mutating run
    that nothing could wait on, cancel or settle — and the process-local dict then
    refused the OWNING task itself, because the only record of ownership died with the
    process. Ownership now replays from the durable `delegate_run_started` row, and an
    id with no durable record at all is UNKNOWN, which is a different answer from
    "belongs to someone else"."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    stub = _LiveRunStub()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: stub)
    delegate._CUSTODY.clear()
    ctx = _nanny_ctx(tmp_path)
    assert json.loads(delegate._delegate_start(ctx, "review the diff"))["status"] == "started"

    delegate._CUSTODY.clear()          # the worker died; only the durable rows remain
    resumed = json.loads(delegate._delegate_wait(ctx, "run-live", wait_sec=1))
    assert resumed["status"] == "no_progress", resumed
    cancelled = json.loads(delegate._delegate_cancel(ctx, "run-live", reason="restart"))
    assert cancelled["status"] in {"requested", "confirmed"}, cancelled
    assert stub.cancels, "the restarted owner must be able to actually stop its own run"

    delegate._CUSTODY.clear()
    sibling = json.loads(delegate._delegate_wait(_nanny_ctx(tmp_path, "t-b"), "run-live", wait_sec=1))
    assert sibling["reason"] == "run_not_owned", sibling
    unknown = json.loads(delegate._delegate_wait(ctx, "run-never-seen", wait_sec=1))
    assert unknown["reason"] == "run_ownership_unknown", unknown
    delegate._CUSTODY.clear()


def test_the_invocation_id_is_reused_on_retry_and_fresh_per_intended_start(
        tmp_path, monkeypatch):
    """Only retry_of reuses an unsettled invocation; replacement waits for settlement."""
    import httpx

    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    script = ["transport_error", "ok", "ok", "definite_refusal", "transport_error"]
    keys, bodies = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v2/handshake":
            return httpx.Response(200, json={"protocolMajor": CLAUDEXOR_PROTOCOL_MAJOR,
                                             "compatible": True,
                                             "engine": {"version": CLAUDEXOR_MIN_VERSION}})
        if path == "/v2/agent-capabilities":
            return httpx.Response(200, json={"harnesses": [
                {"id": "some-route", "enabled": True, "status": "ok",
                 "accessProfilesSupported": ["readonly"]}]})
        if path == "/v2/quota":
            return httpx.Response(200, json={"snapshots": []})
        if path == "/v2/projects":
            return httpx.Response(200, json={"projects": [{"id": "prj-existing", "root": str(tmp_path)}]})
        keys.append(request.headers.get("Idempotency-Key"))
        bodies.append(json.loads(request.read()))
        action = script.pop(0)
        if action == "transport_error":
            raise httpx.ConnectError("daemon fell over mid-POST")
        if action == "definite_refusal":
            return httpx.Response(400, json={"code": "bad_request", "message": "no"})
        return httpx.Response(200, json={"runId": f"run-{len(keys)}"})

    real_gateway = cx.ClaudexorGateway   # captured before the name is patched below

    def _fresh(*_a, **_k):
        gateway = real_gateway(cx.DaemonEndpoint("127.0.0.1", 1, "secret-token"))
        gateway._client = httpx.Client(base_url="http://127.0.0.1:1",
                                       transport=httpx.MockTransport(handler),
                                       headers=dict(gateway._client.headers))
        return gateway

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", _fresh)
    delegate._CUSTODY.clear()
    ctx = _nanny_ctx(tmp_path)
    prompt = "the same intended work"

    # 1. Outcome unknown: the refusal HANDS BACK the retry token. Nothing else may
    #    ever resurrect this invocation.
    lost = json.loads(delegate._delegate_start(ctx, prompt, max_seconds=120))
    assert lost["status"] == "refused" and lost["reason"] == "daemon_unreachable"
    token = lost["pending_invocation_id"]
    assert token == keys[0] and "retry_of" in lost["retry_hint"]

    # 2. A plain identical call is an intended replacement, but the unknown
    # first POST has not settled. The replacement fence refuses before the wire.
    blocked = json.loads(delegate._delegate_start(ctx, prompt))
    assert blocked["reason"] == "replacement_requires_settlement"
    assert len(keys) == 1

    # 3. Only the EXPLICIT token replays the invocation -- the STORED body verbatim,
    #    even though the route config drifted between the attempts.
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:high")
    retried = json.loads(delegate._delegate_start(ctx, prompt, retry_of=token))
    assert retried["status"] == "started" and retried["idempotent_recovery"] is True
    assert keys[1] == token, "the retry must present the original invocation id"
    assert bodies[1] == bodies[0], "the retry must replay the RECORDED body, not re-derive it"
    assert bodies[1]["maxSeconds"] == 120 and bodies[1]["effort"] == "low"

    # The id lives in the run's durable record and survives the worker.
    delegate._CUSTODY.clear()
    assert dc.replay(tmp_path)[retried["run_id"]].invocation_id == token
    assert dc.emit(tmp_path, dc.SETTLED, {
        "run_id": retried["run_id"], "task_id": "t-a",
    })

    # Once the old invocation is durably settled, a plain identical call is a
    # genuinely NEW intention: fresh id, never content-matched reuse.
    fresh = json.loads(delegate._delegate_start(ctx, prompt))
    assert fresh["status"] == "started"
    assert keys[2] != token, "content-matched reuse is forbidden: new intention, new id"
    assert fresh["idempotent_recovery"] is False
    assert dc.emit(tmp_path, dc.SETTLED, {
        "run_id": fresh["run_id"], "task_id": "t-a",
    })

    # 4-5. A bound invocation is never re-posted; an unknown token is refused.
    again = json.loads(delegate._delegate_start(ctx, prompt, retry_of=token))
    assert again["reason"] == "invocation_already_started"
    assert again["run_id"] == retried["run_id"]
    ghost = json.loads(delegate._delegate_start(ctx, prompt, retry_of="no-such-invocation"))
    assert ghost["reason"] == "unknown_invocation"

    # 6. A DEFINITE refusal offers no token: the id is dead, the next start is new.
    refused = json.loads(delegate._delegate_start(ctx, prompt))
    assert refused["status"] == "refused" and "pending_invocation_id" not in refused

    # 7-8. The token replays the recorded invocation, so a divergent prompt is a
    #    confusion, not a merge.
    lost2 = json.loads(delegate._delegate_start(ctx, prompt))
    assert lost2["reason"] == "daemon_unreachable"
    mismatch = json.loads(delegate._delegate_start(
        ctx, "an entirely different ask", retry_of=lost2["pending_invocation_id"]))
    assert mismatch["reason"] == "retry_prompt_mismatch"

    assert len(keys) == 5, "refused retry_of shapes must never reach the wire"
    assert len({keys[0], keys[2], keys[3], keys[4]}) == 4, "one id per intended invocation"
    delegate._CUSTODY.clear()


def test_a_retry_testifies_about_the_stored_invocation_not_the_current_config(
        tmp_path, monkeypatch):
    """A retry POSTs the STORED canonical body — so every fact written or said about
    it must come from the stored invocation too. The old branch re-derived the
    pre-flight health check, the root, the project and the custody/attempt rows from
    the CURRENT route/model/workspace context, so the durable record and the parent's
    result described a configuration the run never had (Codex audit
    run-b62c202d72db). Drift EVERYTHING before the retry — route id, model, effort,
    active root, and make the current route unknown to the daemon — and the retry
    must still replay, health-check and testify the recorded invocation."""
    import httpx

    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    root_a = tmp_path / "root-a"; root_a.mkdir()
    root_b = tmp_path / "root-b"; root_b.mkdir()
    drive = tmp_path / "drive"; drive.mkdir()

    script = ["transport_error", "ok"]
    keys, bodies, registrations, removals = [], [], [], []
    projects: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v2/handshake":
            return httpx.Response(200, json={"protocolMajor": CLAUDEXOR_PROTOCOL_MAJOR,
                                             "compatible": True,
                                             "engine": {"version": CLAUDEXOR_MIN_VERSION}})
        if path == "/v2/agent-capabilities":
            # Only the ORIGINAL route exists. The drifted current config below names
            # route-b, which the daemon has never heard of: a health check asked about
            # the current route refuses the retry outright.
            return httpx.Response(200, json={"harnesses": [
                {"id": "route-a", "enabled": True, "status": "ok",
                 "accessProfilesSupported": ["readonly"]}]})
        if path == "/v2/quota":
            return httpx.Response(200, json={"snapshots": []})
        if path == "/v2/projects" and request.method == "GET":
            return httpx.Response(200, json={"projects": [
                {"id": pid, "root": known} for known, pid in projects.items()]})
        if path == "/v2/projects" and request.method == "POST":
            body = json.loads(request.read())
            pid = f"prj-{len(projects) + 1}"
            projects[str(body["root"])] = pid
            registrations.append(str(body["root"]))
            return httpx.Response(200, json={"id": pid})
        if request.method == "DELETE" and path.startswith("/v2/projects/"):
            removals.append(path.rsplit("/", 1)[-1])
            return httpx.Response(200, json={})
        assert path == "/v2/runs", path
        keys.append(request.headers.get("Idempotency-Key"))
        bodies.append(json.loads(request.read()))
        action = script.pop(0)
        if action == "transport_error":
            raise httpx.ConnectError("daemon fell over mid-POST")
        if action == "definite_refusal":
            return httpx.Response(400, json={"code": "bad_request", "message": "no"})
        return httpx.Response(200, json={"runId": f"run-{len(keys)}"})

    real_gateway = cx.ClaudexorGateway

    def _fresh(*_a, **_k):
        gateway = real_gateway(cx.DaemonEndpoint("127.0.0.1", 1, "secret-token"))
        gateway._client = httpx.Client(base_url="http://127.0.0.1:1",
                                       transport=httpx.MockTransport(handler),
                                       headers=dict(gateway._client.headers))
        return gateway

    def _ctx(repo_dir):
        from ouroboros.tools.registry import ToolContext

        ctx = ToolContext(repo_dir=repo_dir, drive_root=drive)
        ctx.task_id = "t-a"
        ctx.task_metadata = {"root_task_id": "t-a", "parent_task_id": "t-a"}
        return ctx

    monkeypatch.setattr(gw, "ClaudexorGateway", _fresh)
    delegate._CUSTODY.clear()

    # 1. The intended start: route-a=model-old:low at root-a. It registers and OWNS
    #    the project for root-a, then the POST's outcome is lost.
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "route-a=model-old:low")
    lost = json.loads(delegate._delegate_start(_ctx(root_a), "the intended work",
                                               max_seconds=120))
    assert lost["reason"] == "daemon_unreachable"
    token = lost["pending_invocation_id"]
    prj_a = projects[str(root_a)]

    # 2. EVERYTHING drifts before the retry: route id, model, effort and active root.
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "route-b=model-new:high")

    # 2a. A refused token performs no daemon work (old branch registered a
    #     project for the CURRENT root before reading the record).
    ghost = json.loads(delegate._delegate_start(_ctx(root_b), "the intended work",
                                                retry_of="no-such-invocation"))
    assert ghost["reason"] == "unknown_invocation"
    assert str(root_b) not in projects, "refused retry registers no projects"

    # 2b. An unwritable retry row keeps the ORIGINAL attempt's facts alive:
    #     project kept, invocation pending (a run may exist behind the lost POST).
    monkeypatch.setattr(dc, "record_start_requested", lambda *a, **k: False)
    unwritable = json.loads(delegate._delegate_start(_ctx(root_b), "the intended work",
                                                     retry_of=token))
    assert unwritable["reason"] == "start_request_row_unwritable"
    assert "definitely_unrun" not in unwritable
    assert removals == [], "unknown original outcome keeps its project"
    monkeypatch.undo()
    monkeypatch.setattr(gw, "ClaudexorGateway", _fresh)
    from ouroboros import claudexor_daemon
    monkeypatch.setattr(
        claudexor_daemon,
        "ensure_owned_gateway",
        lambda: gw.ClaudexorGateway(),
    )
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "route-b=model-new:high")

    # 3. The real retry: health is asked about the STORED route (the current route-b
    #    is not in the daemon's catalog at all), the wire carries the STORED body,
    #    and no project is registered for the drifted root.
    retried = json.loads(delegate._delegate_start(_ctx(root_b), "the intended work",
                                                  retry_of=token))
    assert retried["status"] == "started", retried
    assert bodies[-1] == bodies[0], "the retry replays the RECORDED body"
    assert keys[-1] == token
    assert str(root_b) not in projects, "a retry binds no NEW resources"

    # THE CLAIM: the tool result testifies the invocation it REPLAYED.
    assert retried["route"] == "route-a"
    assert retried["model"] == "model-old"
    assert retried["effort"] == "low"
    assert retried["root"] == str(root_a)

    # ... and so do the durable rows, attempt and custody alike.
    rows = [json.loads(line) for line
            in (drive / "logs" / "events.jsonl").read_text().splitlines() if line.strip()]
    attempts = [r for r in rows if r.get("type") == dc.START_REQUESTED
                and r.get("invocation_id") == token]
    started = [r for r in rows if r.get("type") == dc.STARTED
               and r.get("run_id") == retried["run_id"]][-1]
    original = attempts[0]
    for row in attempts[1:]:
        for fact in ("route", "project_id", "project_owned", "idempotency_key",
                     "max_seconds", "request"):
            assert row[fact] == original[fact], f"retry attempt re-derived {fact}"
    for fact, expected in (("route", "route-a"), ("model", "model-old"),
                           ("effort", "low"), ("root", str(root_a)),
                           ("project_id", prj_a), ("project_owned", True),
                           ("idempotency_key", original["idempotency_key"])):
        assert started[fact] == expected, f"custody row lies about {fact}: {started[fact]!r}"
    assert dc.replay(drive)[retried["run_id"]].model == "model-old"
    assert dc.emit(drive, dc.SETTLED, {"run_id": retried["run_id"], "task_id": "t-a"})

    # 4. A DEFINITE refusal of a retry settles the STORED attempt's resources: the
    #    project the original start registered and owned is the one retired.
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "route-a=model-old:low")
    root_c = tmp_path / "root-c"; root_c.mkdir()
    script[:] = ["transport_error", "definite_refusal"]
    lost2 = json.loads(delegate.exact_start(
        _ctx(root_c), "other work",
        {"snapshot": _transport_snapshot(subagents.parse_subagent_harness("route-a=model-old:low"))},
    ))
    assert lost2["reason"] == "daemon_unreachable"
    prj_c = projects[str(root_c)]
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "route-b=model-new:high")
    refused = json.loads(delegate._delegate_start(
        _ctx(root_b), "other work", retry_of=lost2["pending_invocation_id"]))
    assert refused["status"] == "refused" and refused["project_retired"] is True
    assert removals == [prj_c], "the retired project is the stored attempt's own"
    delegate._CUSTODY.clear()


def test_custody_rows_outlive_the_child_drive_they_were_written_from(tmp_path, monkeypatch):
    """A live subagent runs on an isolated child drive that headless pruning DELETES, so a
    custody row written there cannot outlive the run it governs. The rows go to the
    canonical (budget) root instead — the existing SSOT for "survives the child" — and
    every fixture that passes only `drive_root` makes the two the same directory, so
    nothing here is proved unless the roots actually differ."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.registry import ToolContext

    canonical, child = tmp_path / "canonical", tmp_path / "child"
    child.mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _LiveRunStub())
    delegate._CUSTODY.clear()
    ctx = ToolContext(repo_dir=tmp_path, drive_root=child)
    ctx.task_id = "t-a"
    ctx.task_metadata = {"root_task_id": "t-a", "budget_drive_root": str(canonical)}

    assert json.loads(delegate._delegate_start(ctx, "review the diff"))["status"] == "started"
    assert (canonical / "logs" / "events.jsonl").exists(), "custody must live on the canonical root"
    assert not (child / "logs" / "events.jsonl").exists(), "not on the drive that gets pruned"

    import shutil

    shutil.rmtree(child)                # headless pruning reaps the child drive
    delegate._CUSTODY.clear()           # and the worker that held the memo is gone
    root = dc.custody_root(ctx)
    assert dc.lookup(root, "t-a", "run-live")[0] == dc.OWNED
    assert [c.run_id for c in dc.open_runs(root)] == ["run-live"]
    delegate._CUSTODY.clear()


def test_delegated_spend_settles_into_the_canonical_budget_ledger(tmp_path, monkeypatch):
    """P34R.1: `ledger_root` was stored from ctx.drive_root — the DISPOSABLE child
    drive on a split-root task — while the custody rows themselves already went to the
    canonical root. `settle_run` then wrote the subscription-session ledger row to
    `custody.ledger_root`, so the delegated spend never reached the canonical budget
    ledger and was erased with the child drive's pruning. The ledger row and the
    custody row must share the same durable root, and the durable STARTED row must
    NAME that root, because a restarted worker settles from the row, not from a ctx."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.registry import ToolContext

    canonical, child = tmp_path / "canonical", tmp_path / "child"
    child.mkdir(parents=True)
    canonical.mkdir(parents=True)

    class _Terminal(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 3, "summary": {"state": "succeeded", "spendUsd": 1.25,
                                              "effectiveAccess": "readonly",
                                              "inputTokens": 10, "outputTokens": 5}}

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Terminal())
    delegate._CUSTODY.clear()
    ctx = ToolContext(repo_dir=tmp_path, drive_root=child)
    ctx.task_id = "t-a"
    ctx.task_metadata = {"root_task_id": "t-a", "budget_drive_root": str(canonical)}

    assert json.loads(delegate._delegate_start(ctx, "review the diff"))["status"] == "started"
    done = json.loads(delegate._delegate_wait(ctx, "run-live", wait_sec=1))
    assert done["settlement"]["settled"] is True
    assert done["settlement"]["ledger_recorded"] is True

    ledger = pathlib.Path("state") / "usage_attempts.jsonl"
    assert (canonical / ledger).exists(), \
        "delegated spend must land in the canonical budget ledger"
    assert not (child / ledger).exists(), \
        "never on the child drive that headless pruning deletes"
    rows = [json.loads(line) for line in (canonical / ledger).read_text().splitlines()
            if '"subscription_session"' in line]
    assert rows and rows[-1]["cost_usd"] == 1.25 and rows[-1]["cost_final"] is True
    started = [json.loads(line) for line
               in (canonical / "logs" / "events.jsonl").read_text().splitlines()
               if '"delegate_run_started"' in line][-1]
    assert started["ledger_root"] == str(dc.custody_root(ctx)), \
        "the durable row must name the canonical root, not the disposable child drive"
    delegate._CUSTODY.clear()


def test_durable_truncation_is_disclosed_never_a_bare_slice(tmp_path):
    """P34R.5: durable/cognitive surfaces in the delegation core hand-rolled `[:N]`
    slices — the containment-incident row cut its EVIDENCE at 500 chars with no
    marker at all, and the primary-output disclosure reason at 300. Every bound now
    goes through the shared `truncate_review_artifact` contract: the cut is marked,
    the original length is named, and the anti-waste floor never spends a marker
    longer than the text it saves."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate

    entry = dc.RunCustody(run_id="run-x", task_id="t-a", route_id="r")
    dc.record_containment_fault(tmp_path, entry, "cancel_unverified", "E" * 5000)
    fault = dc.open_containment_faults(tmp_path)[0]
    assert fault["detail"].startswith("E" * 2000)
    assert "OMISSION NOTE" in fault["detail"] and "original length 5000" in fault["detail"]

    # The anti-waste floor: a cut that saves fewer chars than its own marker
    # passes the text through whole instead of destroying it.
    entry2 = dc.RunCustody(run_id="run-y", task_id="t-a", route_id="r")
    dc.record_containment_fault(tmp_path, entry2, "cancel_unverified", "F" * 2010)
    fault2 = [f for f in dc.open_containment_faults(tmp_path) if f["run_id"] == "run-y"][0]
    assert fault2["detail"] == "F" * 2010

    class _Boom:
        def get_run_artifact(self, rid, path):
            raise RuntimeError("Z" * 900)

    primary = {"truncated": True, "path": "out.md", "bytes": 10, "text": "abc"}
    _resolved_primary, ok, disclosure = delegate._resolve_full_primary_output(
        _Boom(), "run-x", primary)
    assert ok is False
    assert "OMISSION NOTE" in disclosure["reason"] and "original length" in disclosure["reason"]


def test_an_unresolved_containment_fault_cannot_age_out_of_the_health_view(tmp_path):
    """P34R.3: `open_containment_faults` scanned only the last 4 MB of the canonical
    event log, so an UNRESOLVED containment fault — an overpowered run that may still
    be live — silently vanished from the health invariants once later unrelated
    traffic buried its row, despite the stated contract that it stays CRITICAL until
    a terminal receipt resolves it. Incidents now live in their own compact durable
    projection that is read WHOLE; the event-log tail remains as the fallback surface
    for a fault whose compact write failed."""
    import ouroboros.delegate_custody as dc

    entry = dc.RunCustody(run_id="run-fault", task_id="t-a", route_id="r")
    dc.record_containment_fault(tmp_path, entry, "cancel_unverified", "engine went dark")

    # Bury the fault under MORE than the tail window of later unrelated custody rows.
    noise = json.dumps({"type": "delegate_run_reconciled", "run_id": "run-noise",
                        "task_id": "t-b", "pad": "x" * 1500})
    events = dc.event_log_path(tmp_path)
    with events.open("a", encoding="utf-8") as fh:
        for _ in range(3000):
            fh.write(noise + "\n")
    assert events.stat().st_size > dc._FAULT_SCAN_TAIL_BYTES, "the fault is outside the tail"

    open_faults = dc.open_containment_faults(tmp_path)
    assert [f["run_id"] for f in open_faults] == ["run-fault"], \
        "an unresolved incident must never age out of the health view"
    assert open_faults[0]["reason"] == "cancel_unverified"

    # A resolution clears it durably, and later noise cannot reopen it.
    dc.resolve_containment_fault(tmp_path, entry, "verified_terminal")
    assert dc.open_containment_faults(tmp_path) == []
    with events.open("a", encoding="utf-8") as fh:
        for _ in range(200):
            fh.write(noise + "\n")
    assert dc.open_containment_faults(tmp_path) == []

    # Fallback surface: a fault whose COMPACT write failed is still visible through
    # the event-log tail — either landing alone keeps the incident visible.
    other = tmp_path / "other-drive"
    (other / "logs").mkdir(parents=True)
    dc._faults_path(other).mkdir()          # the compact append will fail loudly
    dc.record_containment_fault(other, entry, "cancel_unreachable", "")
    assert [f["run_id"] for f in dc.open_containment_faults(other)] == ["run-fault"]


def test_every_pre_custody_exit_names_the_registration_it_created(tmp_path, monkeypatch):
    """P34P1.7: a registration created before start_run is retired on every TYPED
    pre-custody exit, but an UNTYPED one — a bug here, a timeout, a signal — left the
    durable trail with a bare `start_requested` row and no disposition. The row already
    named the project (so the reviewer's "permanently orphaned" was not literally true,
    proven by execution), but nothing said the attempt had ended, so a reader could not
    tell a live start from a dead one.

    The registration is still NOT retired on an untyped exit: that outcome says nothing
    about whether the POST reached the daemon, and destroying state on missing
    information is the one thing this module forbids. It is NAMED, with a typed reason,
    and the exception continues on its way — disclosure, not a swallow."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    class _Untyped(_LiveRunStub):
        removed: list = []

        def find_project_id(self, root): return ""
        def register_project(self, root): return "prj-owned"
        def remove_project(self, pid): _Untyped.removed.append(pid)
        def start_run(self, request, *, idempotency_key=""):
            raise MemoryError("an untyped failure between register_project and custody")

    stub = _Untyped()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: stub)
    delegate._CUSTODY.clear()

    with pytest.raises(MemoryError):
        delegate._delegate_start(_nanny_ctx(tmp_path), "work")

    rows = [json.loads(l) for l in
            (tmp_path / "logs" / "events.jsonl").read_text().splitlines() if l.strip()]
    failed = [r for r in rows if r.get("type") == dc.START_FAILED]
    assert [r["project_id"] for r in failed] == ["prj-owned"]
    assert failed[0]["reason"] == "pre_custody_exit_MemoryError"
    assert failed[0]["definite"] is False, "an untyped exit is not a definite refusal"
    assert failed[0]["project_retired"] is False
    assert failed[0]["invocation_id"], "the invocation is named, so it can be recovered"
    assert stub.removed == [], "an unknown outcome never destroys the registration"

    # The invocation stays recoverable by the durable sweep (P34R.2), which is what
    # makes retaining the registration the right answer rather than a leak.
    pending = dc.pending_invocations(tmp_path)
    assert [p["project_id"] for p in pending] == ["prj-owned"]
    delegate._CUSTODY.clear()


def test_reconciliation_recovers_a_pending_invocation_whose_worker_died(tmp_path, monkeypatch):
    """P34R.2: /v2/runs accepts the POST, the response is lost, and the worker dies
    before record_started — only the START_REQUESTED row remains. The run-keyed sweep
    could not see it: a live mutating run stayed uncollected FOREVER, and the retry
    token never reached any model. The durable sweep now recovers pending invocations
    on the SAME owner-is-gone predicate: the stored canonical body is re-POSTed under
    the invocation's own wire key (the engine replay returns the ORIGINAL handle), the
    recovered run gets its custody row from the stored invocation facts, and the
    ordinary settle-or-cancel path collects it. Negative shapes: a live owner's pending
    invocation is untouched; a definite refusal retires the invocation AND the
    registration the original attempt owned; an unreachable daemon leaves it pending."""
    import httpx

    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    script = ["transport_error"]
    posted = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v2/handshake":
            return httpx.Response(200, json={"protocolMajor": CLAUDEXOR_PROTOCOL_MAJOR,
                                             "compatible": True,
                                             "engine": {"version": CLAUDEXOR_MIN_VERSION}})
        if path == "/v2/agent-capabilities":
            return httpx.Response(200, json={"harnesses": [
                {"id": "some-route", "enabled": True, "status": "ok",
                 "accessProfilesSupported": ["readonly"]}]})
        if path == "/v2/quota":
            return httpx.Response(200, json={"snapshots": []})
        if path == "/v2/projects":
            return httpx.Response(200, json={"projects": []}) if request.method == "GET" \
                else httpx.Response(200, json={"id": "prj-owned"})
        assert path == "/v2/runs", path
        posted.append((request.headers.get("Idempotency-Key"), json.loads(request.read())))
        if script.pop(0) == "transport_error":
            raise httpx.ConnectError("daemon fell over mid-POST")
        return httpx.Response(200, json={"runId": "run-recovered"})

    real_gateway = cx.ClaudexorGateway

    def _fresh(*_a, **_k):
        gateway = real_gateway(cx.DaemonEndpoint("127.0.0.1", 1, "secret-token"))
        gateway._client = httpx.Client(base_url="http://127.0.0.1:1",
                                       transport=httpx.MockTransport(handler),
                                       headers=dict(gateway._client.headers))
        return gateway

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", _fresh)
    delegate._CUSTODY.clear()
    ctx = _nanny_ctx(tmp_path)

    # The durable residue of the crash, produced through the REAL path: an accepted
    # POST whose response was lost. Only START_REQUESTED names the invocation.
    lost = json.loads(delegate._delegate_start(ctx, "the intended work", max_seconds=60))
    token = lost["pending_invocation_id"]
    delegate._CUSTODY.clear()            # the worker that knew the token is gone
    assert [r["invocation_id"] for r in dc.pending_invocations(tmp_path)] == [token]
    assert dc.open_runs(tmp_path) == [], "no run row exists: the run-keyed sweep is blind here"

    # 1. The owner is ALIVE: its pending invocation is untouched (the owner holds
    #    the retry token and decides).
    assert dc.reconcile_orphaned_runs(tmp_path, {"t-a"}, gateway_factory=_fresh) == []
    assert len(posted) == 1

    # 2. The owner is GONE: the sweep replays the stored body under the stored key,
    #    the daemon returns the run it (now) holds, and the ordinary path collects it.
    class _TerminalRecovery:
        removed: list = []

        def handshake(self, **_kw): return {}
        def start_run(self, request, *, idempotency_key=""):
            posted.append((idempotency_key, dict(request)))
            return {"runId": "run-recovered"}
        def get_run(self, rid, **_kw):
            return {"lastSeq": 2, "summary": {"state": "succeeded", "spendUsd": 0.5,
                                              "effectiveAccess": "readonly"}}
        def remove_project(self, pid): _TerminalRecovery.removed.append(pid)
        def close(self): pass

    outcomes = dc.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=lambda: _TerminalRecovery())
    assert [o["action"] for o in outcomes] == ["settle_attempted"] and outcomes[0]["settled"] is True
    key, body = posted[-1]
    assert key == token, "recovery must present the invocation's own wire key"
    assert body == posted[0][1], "recovery must replay the RECORDED canonical body"
    started = [json.loads(line) for line
               in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()
               if '"delegate_run_started"' in line][-1]
    assert started["run_id"] == "run-recovered"
    assert started["recovered_from_pending_invocation"] is True
    assert started["route"] == "some-route" and started["model"] == "weak-model"
    assert started["idempotency_key"], "the stored lookup key rides the recovered row"
    assert dc.pending_invocations(tmp_path) == [], "a recovered invocation is bound, not pending"
    again = dc.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=lambda: _TerminalRecovery())
    assert again == [], "a settled recovery does not repeat"

    # 3. A DEFINITE refusal at recovery retires the invocation and the registration
    #    the original attempt owned; an unreachable daemon leaves it pending.
    script[:] = ["transport_error"]
    lost2 = json.loads(delegate._delegate_start(ctx, "other intended work"))
    token2 = lost2["pending_invocation_id"]
    delegate._CUSTODY.clear()

    class _Refusing:
        def __init__(self): self.removed = []
        def handshake(self, **_kw): return {}
        def start_run(self, request, *, idempotency_key=""):
            raise ClaudexorUnavailable("bad_request", "no", status_code=400)
        def remove_project(self, pid): self.removed.append(pid)
        def close(self): pass

    class _Unreachable:
        def handshake(self, **_kw): return {}
        def start_run(self, request, *, idempotency_key=""):
            raise ClaudexorUnavailable("daemon_unreachable", "down", status_code=0)
        def close(self): pass

    down = dc.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=lambda: _Unreachable())
    assert [o["action"] for o in down] == ["recovery_unreachable"]
    assert [r["invocation_id"] for r in dc.pending_invocations(tmp_path)] == [token2], \
        "an unknown outcome never destroys the invocation"
    refusing = _Refusing()
    gone = dc.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=lambda: refusing)
    assert [o["action"] for o in gone] == ["invocation_retired"]
    assert refusing.removed == ["prj-owned"], "the ORIGINAL attempt's owned registration is discharged"
    assert dc.pending_invocations(tmp_path) == []
    assert dc.invocation_record(tmp_path, token2)["state"] == "failed_definite"
    delegate._CUSTODY.clear()


def test_a_start_whose_custody_row_did_not_land_does_not_claim_to_be_custodied(tmp_path, monkeypatch):
    """A successful POST without a durable STARTED row is a named custody fault.

    START_REQUESTED lands, so the live run keeps its original invocation identity;
    only STARTED/SETTLED persistence is faulted here.
    """
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _LiveRunStub())
    real_append = dc.append_jsonl

    def _started_row_lost(path, obj):
        if obj.get("type") in ("delegate_run_started", "delegate_run_settled"):
            return False
        return real_append(path, obj)

    monkeypatch.setattr(dc, "append_jsonl", _started_row_lost)
    delegate._CUSTODY.clear()
    out = json.loads(delegate._delegate_start(_nanny_ctx(tmp_path), "review the diff"))
    delegate._CUSTODY.clear()

    assert out["run_id"] == "run-live", "the run really did start; that is not in doubt"
    assert out["custody_durable"] is False
    assert out["invocation_id"] and out["pending_invocation_id"] == out["invocation_id"]
    assert out["status"] == "started_uncustodied", (
        "a start nothing outside this worker can name must not wear the plain name")
    assert "CUSTODY IS NOT DURABLE" in out["note"]
    assert dc.lookup(tmp_path, "t-a", "run-live")[0] == dc.UNKNOWN, "the premise of the claim"

    entry = dc.RunCustody(run_id="run-2", task_id="t-a", route_id="r", model="m",
                          project_id="p", project_owned=False, ledger_root=str(tmp_path))
    entry.ledger_recorded = True
    settlement = dc.settle_run(tmp_path, _LiveRunStub(), entry,
                               {"summary": {"state": "succeeded", "spendUsd": 0.0}})
    assert settlement["settled"] is False and entry.settled is False


@pytest.mark.parametrize("status_code,retired,remove_absent", [
    (422, True, False),     # the daemon ANSWERED and refused: no run was bound
    (0, False, False),      # transport error: the POST's fate is unknown, a run may be live
    (503, False, False),    # 5xx: same — an unverified outcome is not grounds to destroy state
    # The daemon has no such registration: absence IS discharge, the same answer
    # `retire_project` settles on, not a failure to report.
    (422, True, True),
])
def test_a_failed_start_does_not_leave_the_registration_it_created(
    tmp_path, monkeypatch, status_code, retired, remove_absent,
):
    """The project is registered BEFORE `start_run`. A start failure used to leave that
    registration behind with nothing anywhere naming its id — and the id must be durably
    named whether or not the registration can be safely retired."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    live = {"prj-new"}

    class _Stub(_LiveRunStub):
        def find_project_id(self, root): return ""
        def register_project(self, root): return "prj-new"
        def remove_project(self, pid):
            if remove_absent:
                live.discard(pid)   # it was never there to begin with
                raise gw.ClaudexorUnavailable("project_not_found", "gone", status_code=404)
            live.discard(pid)
        def start_run(self, request, *, idempotency_key=""):
            raise gw.ClaudexorUnavailable("run_start_failed", "no run", status_code=status_code)

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    out = json.loads(delegate._delegate_start(_nanny_ctx(tmp_path), "x"))
    delegate._CUSTODY.clear()
    assert out["status"] == "refused" and out["reason"] == "run_start_failed"
    assert out["project_retired"] is retired, out
    assert (live == set()) is retired, "only a definite refusal may retire the registration"
    if not retired:
        assert out["project_retention_reason"] == "start_outcome_unknown_run_may_exist"
    rows = [json.loads(l) for l in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()]
    named = [r for r in rows if r.get("type") == "delegate_run_start_failed"]
    assert named and named[0]["project_id"] == "prj-new", "the id must be durably named"


def test_a_queued_handle_with_no_run_id_names_its_registration_like_its_twin(tmp_path, monkeypatch):
    """The untreated twin of the branch above. Here the POST SUCCEEDED (2xx) and only the
    handle was unusable, so a run is MORE likely live against the registration — yet this
    branch retired nothing and durably named nothing, and with no run id the orphan
    reconciler can never see it either. Both branches now leave the same durable trace."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    live = {"prj-new"}

    class _Stub(_LiveRunStub):
        def find_project_id(self, root): return ""
        def register_project(self, root): return "prj-new"
        def remove_project(self, pid): live.discard(pid)
        def start_run(self, request, *, idempotency_key=""): return {"status": "queued"}

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    out = json.loads(delegate._delegate_start(_nanny_ctx(tmp_path), "x"))
    delegate._CUSTODY.clear()
    assert out["reason"] == "queued_without_run_id"
    assert out["project_id"] == "prj-new", "the retained registration must be named"
    assert out["project_retired"] is False and live == {"prj-new"}, (
        "an accepted POST is never grounds to destroy the registration a run may use")
    assert out["project_retention_reason"] == "start_outcome_unknown_run_may_exist"
    rows = [json.loads(l) for l in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()]
    named = [r for r in rows if r.get("type") == "delegate_run_start_failed"]
    assert named and named[0]["project_id"] == "prj-new", "the id must be durably named"
    assert named[0]["reason"] == "queued_without_run_id"


def test_shared_project_retirement_defers_quietly_for_non_canonical_sharers(tmp_path):
    """Any unsettled sibling defers every sharer QUIETLY; once all settle
    the LOWEST-run_id sharer carries the retry lane."""
    import json as _json

    import ouroboros.delegate_custody as dc
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    class _RefusingGateway:
        def __init__(self):
            self.removals = []
            self.refuse = True

        def remove_project(self, pid):
            self.removals.append(pid)
            if self.refuse:
                raise ClaudexorUnavailable("project_busy", "project has live runs", status_code=409)

    gateway = _RefusingGateway()
    for rid, tid in (("run-aa", "t-1"), ("run-bb", "t-2")):
        dc.record_started(tmp_path, dc.RunCustody(
            run_id=rid, task_id=tid, route_id="r", model="m",
            project_id="prj-shared", project_owned=True, ledger_root=str(tmp_path)))
    dc._CUSTODY.clear()

    # Non-canonical sharer: quiet deferral.
    custody_b = dc.replay(tmp_path)["run-bb"]
    dc.retire_project(tmp_path, gateway, custody_b)
    assert gateway.removals == []
    assert "delegate_run_project_retire_failed" not in _event_types(tmp_path)
    assert custody_b.project_owned is True

    # Canonical too defers while a sibling is unsettled.
    custody_a = dc.replay(tmp_path)["run-aa"]
    dc.retire_project(tmp_path, gateway, custody_a)
    assert gateway.removals == []
    # Sibling settles: canonical attempts; refusal is typed.
    dc.emit(tmp_path, dc.SETTLED, {"run_id": "run-bb", "task_id": "t-2", "route": "r"})
    dc._CUSTODY.clear()
    custody_a = dc.replay(tmp_path)["run-aa"]
    dc.retire_project(tmp_path, gateway, custody_a)
    assert gateway.removals == ["prj-shared"]
    rows = [
        _json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed = [r for r in rows if r.get("type") == "delegate_run_project_retire_failed"]
    assert len(failed) == 1
    assert "live runs" in str(failed[0].get("reason"))

    # Once the daemon accepts, the canonical sharer discharges the registration.
    gateway.refuse = False
    dc.retire_project(tmp_path, gateway, custody_a)
    assert custody_a.project_owned is False
    assert "delegate_run_project_retired" in _event_types(tmp_path)
    dc._CUSTODY.clear()
