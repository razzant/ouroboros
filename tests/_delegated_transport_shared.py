"""Fixtures, stubs and context builders shared by the delegated-transport suites.

Split out of ``tests/test_delegated_subagent_transport.py`` when that module was
divided by theme (the v7 S7a split, re-cut on the v7next tip bytes); every
definition is the tip's exact bytes, so each sibling suite keeps the exact
semantics it was written against. ``_owned_gateway_uses_each_test_transport`` is
autouse, so importing it into a test module re-applies it there.
"""

from __future__ import annotations

import copy
import json
import sys
import httpx
import pytest
from ouroboros import delegate_custody as c, delegate_custody_current as current, subagents
from ouroboros.config import CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION, CLAUDEXOR_MIN_VERSION, CLAUDEXOR_PROTOCOL_MAJOR
from ouroboros.gateways import claudexor as cx, claudexor_metadata as metadata

# The technical access the autouse fixture's selected session row captured (#1609).
# ``None`` keeps the legacy snapshot with no "access" key (workspace_write), which an
# acting child needs. Technical access alone now decides the run shape, so a test
# about a READ-ONLY child must say so: ``_select_transport_access(monkeypatch,
# "readonly")`` (or ``_started_request(..., access="readonly")``).
_TRANSPORT_ACCESS = None


def _transport_snapshot(route, access=None):
    target = route.route_id + (f"={route.model}" if route.model else "")
    return {
        "schema": 1, "selected_subagent_id": "transport-fixture",
        "config_fingerprint": "transport-fixture-v1",
        "route": {"kind": "agent_session", "target_id": target,
                  "credential_profile_id": route.profile_id},
        "effort": route.effort,
        **({"access": access} if access else {}),
    }


def _select_transport_access(monkeypatch, access):
    """Make the fixture's selected row capture ``access`` for the rest of this test."""
    monkeypatch.setattr(sys.modules[__name__], "_TRANSPORT_ACCESS", access)


@pytest.fixture(autouse=True)
def _owned_gateway_uses_each_test_transport(monkeypatch):
    """Bind route ID."""
    from ouroboros import claudexor_daemon, delegate_custody, subagent_runtime
    from ouroboros.gateways import claudexor as gateway_module
    from ouroboros.tools import delegate

    monkeypatch.setattr(
        claudexor_daemon,
        "ensure_owned_gateway",
        lambda: gateway_module.ClaudexorGateway(),
    )
    # The fixture models a SERVING daemon, so the attach-only release surfaces
    # (loop exit, kill boundaries) reach the same test transport the admitted
    # sweep does; neither starts anything here.
    monkeypatch.setattr(
        claudexor_daemon,
        "read_owned_gateway",
        lambda **_kw: gateway_module.ClaudexorGateway(),
    )
    original_actor = delegate.prepare_delegate_start_actor

    def explicit_transport_actor(ctx, drive_root, **kwargs):
        if kwargs.get("recovering"):
            return original_actor(ctx, drive_root, **kwargs)
        route = subagents.get_subagent_harness()
        if route is None:
            return original_actor(ctx, drive_root, **kwargs)
        # A suite may swap `_transport_snapshot` for its own one-argument builder;
        # only an explicit lowering passes the access through.
        snapshot = (_transport_snapshot(route) if _TRANSPORT_ACCESS is None
                    else _transport_snapshot(route, _TRANSPORT_ACCESS))
        token = subagent_runtime._EXACT_START_SELECTION.set({"snapshot": snapshot})
        try:
            return original_actor(ctx, drive_root, **kwargs)
        finally:
            subagent_runtime._EXACT_START_SELECTION.reset(token)

    delegate_custody._CUSTODY.clear()
    monkeypatch.setattr(delegate, "prepare_delegate_start_actor", explicit_transport_actor)
    yield
    delegate_custody._CUSTODY.clear()


def _gateway(handler) -> cx.ClaudexorGateway:
    gateway = cx.ClaudexorGateway(cx.DaemonEndpoint("127.0.0.1", 1, "secret-token"))
    gateway._client = httpx.Client(
        base_url="http://127.0.0.1:1",
        transport=httpx.MockTransport(handler),
        headers=dict(gateway._client.headers),
    )
    return gateway


class _HealthStub:
    """A daemon that answers the manifest questions the rule table needs.

    `engine_version` is part of that answer, not decoration: the real gateway sets it
    at handshake and the mutating lane's floor reads it, so a stub without one models a
    daemon that never negotiated.
    """

    def __init__(self, *, status="ok", profiles=("readonly", "workspace_write"), reset_at="",
                 engine_version=CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION):
        self.status, self.profiles, self.reset_at = status, profiles, reset_at
        self.engine_version = engine_version

    def handshake(self, **_kw): return {}
    def agent_capabilities(self):
        return {"harnesses": [{
            "id": "some-route", "enabled": self.status == "ok", "status": self.status,
            "accessProfilesSupported": list(self.profiles),
        }]}

    def quota_snapshots(self):
        if not self.reset_at:
            return []
        return [{
            "subject": {"harness": "some-route"}, "freshness": "fresh",
            "constraints": [{"used_ratio": 1.0, "resets_at": self.reset_at}],
        }]

    def quota_state(self, *, model=None, request_kind="text", **_kw):
        return {"snapshots": [{**row, "availability": {
            "state": "exhausted" if self.reset_at else "available", "resets_at": self.reset_at or None,
        }} for row in self.quota_snapshots()], "absences": [],
            "availability_scope": {"model": model, "requestKind": request_kind, "basis": "quota_block"}}

    def close(self): pass


def _session_dispatch(*, route="some-route=weak:low", stub=None, monkeypatch=None,
                      raises=None, acting=False):
    """The executor resolution of a configured agent-session row at the REAL dispatch
    entry point, against a daemon double: the one dispatch that health-checks a
    session route before a token is spent. A read-only child's row says so."""
    from ouroboros.gateways import claudexor as gw

    def _make(*a, **k):
        if raises is not None:
            raise raises
        return stub if stub is not None else _HealthStub()

    monkeypatch.setattr(gw, "ClaudexorGateway", _make)
    parsed = subagents.parse_subagent_harness(route)
    task = {"delegation_role": "subagent", "parent_cognitive_route": {"model": "nanny-model"},
            "configured_subagent": _transport_snapshot(parsed, None if acting else "readonly")}
    if acting:
        task["task_constraint"] = {"mode": "acting_subagent", "surface": "self_worktree"}
    return subagents.resolve_subagent_dispatch(task, task_type="task").executor_resolution


def _delegating_ctx(tmp_path, *, acting: bool, task_id: str = "t-nanny"):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    # An acting child's run root must equal the granted write_root (its own
    # worktree, OUTSIDE the data drive - the overlap is what
    # workspace_mode_block_reason refuses). Pre-v6.87.30 this pinned
    # "hand the shell the live tree" as correct.
    worktree = tmp_path.parent / f"wt-{tmp_path.name}"
    worktree.mkdir(exist_ok=True)
    if acting and not (worktree / ".git").exists():
        # C1: a mutating run's authority target must be a git tree — the private
        # execution snapshot is a worktree of it, at a baseline built from it.
        import subprocess as _sp

        _sp.run(["git", "init"], cwd=str(worktree), capture_output=True, check=True)
        (worktree / "README.md").write_text("seed\n", encoding="utf-8")
        _sp.run(["git", "add", "-A"], cwd=str(worktree), capture_output=True, check=True)
        _sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "seed"],
                cwd=str(worktree), capture_output=True, check=True)
    constraint = TaskConstraint(
        mode="acting_subagent" if acting else "local_readonly_subagent",
        surface="self_worktree" if acting else "",
        write_root=str(worktree) if acting else "",
    )
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path, task_constraint=constraint)
    if acting:
        ctx.workspace_root = str(worktree)
        ctx.workspace_mode = "self_worktree"
    ctx.task_id = task_id
    ctx.task_metadata = {"root_task_id": "t-root", "parent_task_id": "t-root"}
    # A child's root exists before the child runs: its lineage is the delegated start's billing
    # authority (a missing root is refused, never a stand-in bucket).
    from ouroboros.task_results import load_task_result, write_task_result
    if load_task_result(tmp_path, "t-root") is None:
        write_task_result(tmp_path, "t-root", "running", root_task_id="t-root")
    return ctx


def _started_request(tmp_path, *, acting: bool, monkeypatch,
                     engine_version=CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION, expect="started",
                     start_kwargs=None, access=None):
    """Run _delegate_start against a stubbed gateway and return the wire request.

    ``start_kwargs`` forwards optional model-issued arguments (folder geometry)
    unchanged, so a caller can compare an explicitly named shape against omission
    on the SAME transport instead of rebuilding the stub. ``access`` is the selected
    row's captured technical access (``None``: the legacy writable default).
    """
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    if access is not None:
        _select_transport_access(monkeypatch, access)
    seen = {}

    class _Stub:
        engine_version = ""

        def handshake(self, **_kw): return {}
        def agent_capabilities(self):
            return {"harnesses": [{
                "id": "some-route", "enabled": True, "status": "ok",
                "accessProfilesSupported": ["readonly", "workspace_write"],
            }]}
        def quota_snapshots(self): return []
        def find_project_id(self, root): return "prj-existing"
        def register_project(self, root): raise AssertionError("must reuse the registration")
        def start_run(self, request, *, idempotency_key=""):
            seen["request"] = request
            run_id = f"run-{'write' if acting else 'read'}"
            return {"runId": run_id, "runDir": f"/tmp/{run_id}"}
        def close(self): pass

    _Stub.engine_version = engine_version
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    # C1: mutating starts provision a private execution snapshot under the
    # worktree-service root; keep it inside the test tmp tree.
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snap_root"))
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    task_id = f"t-nanny-{'write' if acting else 'read'}"
    payload = json.loads(delegate._delegate_start(
        _delegating_ctx(tmp_path, acting=acting, task_id=task_id), "edit the README",
        **(start_kwargs or {})
    ).text)
    delegate._CUSTODY.clear()
    assert payload["status"] == expect, payload
    return seen.get("request"), payload


def _plain_ctx(tmp_path):
    """A read-only nanny context: the smallest thing `_delegate_start` will accept."""
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "t-nanny"
    ctx.task_metadata = {"root_task_id": "t-root", "parent_task_id": "t-root"}
    from ouroboros.task_results import load_task_result, write_task_result
    if load_task_result(tmp_path, "t-root") is None:  # the nanny's real root
        write_task_result(tmp_path, "t-root", "running", root_task_id="t-root")
    return ctx


def _isolation_stub(monkeypatch, *, run_dir, engine_version=CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION,
                    effective_access="workspace_write", state="running"):
    """A daemon serving one run whose artifacts sit under ``run_dir``."""
    from ouroboros.gateways import claudexor as gw

    cancelled = {}

    class _Stub:
        engine_version = ""

        def handshake(self, **_kw): return {}
        def get_run(self, rid, *, timeout_sec=None):
            return {"lastSeq": 7, "summary": {
                "state": "cancelled" if cancelled else state,
                "effectiveAccess": effective_access,
                "runDir": str(run_dir),
            }}
        def cancel_run(self, rid, reason=""):
            cancelled["reason"] = reason
            return {"accepted": True}
        def remove_project(self, pid): pass
        def close(self): pass

    _Stub.engine_version = engine_version
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    return cancelled


def _write_attempt(run_dir, *, isolated, home_dir, attempt="a01", mechanism="seatbelt",
                   unavailable_reason=None):
    """One clean `attempt.yaml`, in Claudexor's own applied-facts shape.

    `mechanism=None` is the record an engine writes when it applied NO OS boundary —
    3.3.0/3.3.1, which have no confinement fields at all, and any host whose engine
    ships a mechanism it cannot use here. It is a supported outcome, not a malformed
    record, which is why it is a parameter of the ordinary helper.
    `unavailable_reason` is the engine's typed explanation for a missing boundary
    (phase A3) — telemetry the disclosure amplifies, never an admission token.
    """
    attempt_dir = run_dir / "attempts" / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    record = {"attempt_id": attempt, "harness_id": "some-route", "harness_home_dir": home_dir}
    if isolated is not None:
        record["harness_home_isolated"] = isolated
    if mechanism is not None:
        record["confinement_mechanism"] = mechanism
        record["confinement_profile_digest"] = "sha256:" + "0" * 64
        record["confinement_verified_denied_path"] = "/Users/op/.claudexor/v3/daemon"
    if unavailable_reason is not None:
        record["confinement_unavailable_reason"] = unavailable_reason
    lines = [f"{k}: {json.dumps(v)}" for k, v in record.items()]
    (attempt_dir / "attempt.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_failed_attempt(run_dir, *, attempt="a01"):
    """An errored attempt.yaml with NO harness-HOME fields. `AC.attemptFailureRecord`
    (orchestrator.ts:3512 and :5088) spreads the applied facts in today, but
    `harness_home_isolated` is the one optional member — absent when the attempt died
    before its home was decided — and an engine older than 3.3.2 wrote none of them."""
    attempt_dir = run_dir / "attempts" / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "attempt.yaml").write_text(
        "\n".join([
            f"attempt_id: {json.dumps(attempt)}",
            'harness_id: "some-route"', "cost_usd: 0.4", "cost_estimated: true",
            "errored: true", 'phase: "harness"', 'errors:\n  - "stream ended early"',
        ]) + "\n",
        encoding="utf-8",
    )


def _waiting(tmp_path, monkeypatch, *, acting=True, access=""):
    """One wake's read of ``run-1`` (``tools/delegate._read_run``).

    ``access`` records the GRANTED shape on the custody entry, as a STARTED row does
    (#1609: technical access alone decides it), whose floor the body holds; "" keeps a
    legacy entry with none, whose read re-derives its entitlement from the live context.
    """
    import ouroboros.tools.delegate as delegate

    ctx = _delegating_ctx(tmp_path, acting=acting)
    granted = subagents.delegated_run_shape(acting, access) if access else None
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        task_id="t-nanny", route_id="some-route", model="m",
        project_id="prj", project_owned=False,
        **({"access": granted.access, "mode": granted.mode, "isolation": granted.isolation,
            "delegated": granted.delegated} if granted else {}),
    )
    # since_seq=0 so a HEALTHY run answers `progress`; a BREACH is halted as a containment fault.
    out = json.loads(delegate._read_run(ctx, "run-1", 0))
    delegate._CUSTODY.clear()
    return out


def _nanny_ctx(tmp_path, task_id="t-a"):
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = task_id
    ctx.task_metadata = {"root_task_id": task_id, "parent_task_id": task_id}
    return ctx


def _event_types(tmp_path):
    path = tmp_path / "logs" / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line).get("type") for line in path.read_text().splitlines() if line.strip()]


class _LiveRunStub:
    """A daemon whose run starts and keeps running."""

    def __init__(self, run_id="run-live", state="running"):
        self.run_id, self.state, self.cancels = run_id, state, []

    def handshake(self, **_kw): return {}
    def agent_capabilities(self):
        return {"harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                               "accessProfilesSupported": ["readonly"]}]}
    def quota_snapshots(self): return []
    def find_project_id(self, root): return "prj-existing"
    def start_run(self, request, *, idempotency_key=""): return {"runId": self.run_id}
    # `effectiveAccess` is what the daemon DERIVES, and the containment reader treats an
    # undisclosed profile on a run that has already produced journal events as unverified.
    # A read-only fixture that omits it is not a narrower daemon, it is an unfaithful one.
    def get_run(self, rid, *, timeout_sec=None):
        return {"lastSeq": 1, "summary": {"state": self.state, "effectiveAccess": "readonly"}}
    def cancel_run(self, rid, reason=""):
        self.cancels.append((rid, reason))
        return {"accepted": True, "status": "accepted"}
    def remove_project(self, pid): pass
    def close(self): pass


def _health_invariants(tmp_path):
    """Run the real health-invariant builder over a drive with nothing else in it."""
    from ouroboros.context import build_health_invariants

    class _Env:
        drive_root = tmp_path

        def drive_path(self, rel=""):
            return tmp_path / rel

        def repo_path(self, rel=""):
            return tmp_path / "repo" / rel

    return build_health_invariants(_Env())


class _StreamingStub:
    """A daemon whose journal cursor advances on EVERY poll — i.e. a healthy run.

    `_LiveRunStub` and `_AliveStub` both hold `lastSeq` constant, so every existing wait
    test exercises the silent path. This is the busy one, and it is the shape that used
    to cost a full-context nanny round per event batch.
    """

    def __init__(self, *, state="running", batch=1, title="running tests"):
        self.seq, self.state, self.batch, self.title = 0, state, batch, title

    def handshake(self, **_kw): return {"compatible": True, "protocolMajor": 3}

    def get_run(self, rid, *, timeout_sec=None):
        self.seq += 1
        return {
            "lastSeq": self.seq,
            "summary": {"state": self.state, "effectiveAccess": "readonly"},
            "timeline": [{"type": "tool", "title": self.title, "severity": "info"}
                         for _ in range(self.seq * self.batch)],
        }

    def close(self): pass


def _delegate_registry(tmp_path, monkeypatch, task_id="t-family"):
    """A real registry, with the family's five entries registered as production does."""
    from ouroboros.tools.registry import ToolRegistry
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = task_id
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, task_id, "running", root_task_id=task_id, task_attempt=1)
    registry._ctx.task_metadata = {"root_task_id": task_id, "parent_task_id": task_id}
    assert {"delegate_start", "delegate_wait", "delegate_cancel", "delegate_answer",
            "delegate_message"} <= set(registry._entries)
    return registry


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


def _bind_without_state(root, gateway, engine, rid):
    from ouroboros import delegate_accepted
    from ouroboros.delegate_custody_reconcile import recovered_custody

    record = {**c.invocation_record(root, "lost"), "invocation_id": "lost"}
    entry, shape = recovered_custody(root, gateway, record, rid, engine.store)
    assert delegate_accepted.bind_started(root, entry, shape=shape).durable
    entry = current.run(root, rid)
    assert entry.resume_start_state == "unknown"
    return entry


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
