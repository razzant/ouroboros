"""Fixtures, stubs and context builders shared by the delegated-transport suites.

Split out of ``tests/test_delegated_subagent_transport.py`` when that module was
divided by theme (the v7 S7a split, re-cut on the v7next tip bytes); every
definition is the tip's exact bytes, so each sibling suite keeps the exact
semantics it was written against. ``_owned_gateway_uses_each_test_transport`` is
autouse, so importing it into a test module re-applies it there.
"""

from __future__ import annotations

import json
import httpx
import pytest
from ouroboros import subagents
from ouroboros.config import CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION
from ouroboros.gateways import claudexor as cx


def _transport_snapshot(route):
    target = route.route_id + (f"={route.model}" if route.model else "")
    return {
        "schema": 1, "selected_subagent_id": "transport-fixture",
        "config_fingerprint": "transport-fixture-v1",
        "route": {"kind": "agent_session", "target_id": target,
                  "credential_profile_id": route.profile_id},
        "effort": route.effort,
    }


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
    original_actor = delegate.prepare_delegate_start_actor

    def explicit_transport_actor(ctx, drive_root, **kwargs):
        if kwargs.get("recovering"):
            return original_actor(ctx, drive_root, **kwargs)
        route = subagents.get_subagent_harness()
        if route is None:
            return original_actor(ctx, drive_root, **kwargs)
        token = subagent_runtime._EXACT_START_SELECTION.set({
            "snapshot": _transport_snapshot(route),
        })
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

    def close(self): pass


def _dispatch(requested, *, route="some-route=weak:low", stub=None, monkeypatch=None,
              raises=None, acting=False):
    from ouroboros.gateways import claudexor as gw
    from ouroboros.subagents import dispatch_executor_resolution

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", route)

    def _make(*a, **k):
        if raises is not None:
            raise raises
        return stub if stub is not None else _HealthStub()

    monkeypatch.setattr(gw, "ClaudexorGateway", _make)
    task = {"delegation_role": "subagent", "requested_executor": requested}
    if acting:
        task["task_constraint"] = {"mode": "acting_subagent", "surface": "self_worktree"}
    return dispatch_executor_resolution(task)


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
    return ctx


def _started_request(tmp_path, *, acting: bool, monkeypatch,
                     engine_version=CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION, expect="started"):
    """Run _delegate_start against a stubbed gateway and return the wire request."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

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
        _delegating_ctx(tmp_path, acting=acting, task_id=task_id), "edit the README"
    ))
    delegate._CUSTODY.clear()
    assert payload["status"] == expect, payload
    return seen.get("request"), payload


def _plain_ctx(tmp_path):
    """A read-only nanny context: the smallest thing `_delegate_start` will accept."""
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "t-nanny"
    ctx.task_metadata = {"root_task_id": "t-root", "parent_task_id": "t-root"}
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


def _waiting(tmp_path, monkeypatch, *, acting=True):
    import ouroboros.tools.delegate as delegate

    ctx = _delegating_ctx(tmp_path, acting=acting)
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        task_id="t-nanny", route_id="some-route", model="m",
        project_id="prj", project_owned=False,
    )
    # since_seq=0 so a HEALTHY run records its advance and answers `progress` when the
    # one-second window expires; a BREACH is what returns immediately, mid-window. The
    # distinguishing signal is "was this halted as a containment fault", not the timing.
    out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1, since_seq=0))
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
