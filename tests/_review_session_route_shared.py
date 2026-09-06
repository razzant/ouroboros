"""Offline fixtures shared by the agent-session review-route suites.

Split out of ``tests/test_review_agent_session_route.py`` when that module was
divided by theme; the fixtures are verbatim (owner test rule: cheap, weak, no
live harness). A FakeGateway stands in for the Claudexor /v2 control plane with
the same semantics the real engine documents and a FakeLLM answers the one
sanctioned light-model extraction call. The autouse transport fixture rides
along so every sibling suite keeps patching the owned gateway it was written
against instead of silently reaching a real one.
"""

import json

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.review_execution import (
    REVIEW_SESSION_ROUTE_ENV,
    ReviewRouteKind,
)
from ouroboros.review_substrate import (
    ReviewRequest,
    ReviewSlot,
)

@pytest.fixture(autouse=True)
def _owned_gateway_uses_each_test_transport(monkeypatch):
    from ouroboros import claudexor_daemon
    from ouroboros.gateways import claudexor as gateway_module

    monkeypatch.setattr(
        claudexor_daemon,
        "ensure_owned_gateway",
        lambda: gateway_module.ClaudexorGateway(),
    )

# ---------------------------------------------------------------------------
# Offline fixtures
# ---------------------------------------------------------------------------

def _terminal_detail(text, *, state="succeeded", conformance="", truncated=False,
                     path="", reported_bytes=None, model="fake-small"):
    summary = {
        "state": state,
        "model": model,
        "spendUsd": 0.0,
        "spendEstimated": False,
    }
    if conformance:
        summary["outputConformance"] = conformance
    primary = {"text": text, "truncated": truncated}
    if path:
        primary["path"] = path
    if reported_bytes is not None:
        primary["bytes"] = reported_bytes
    return {"summary": summary, "primaryOutput": primary, "lastSeq": 3}

class FakeGateway:
    """The /v2 surface the executor drives, with recorded evidence."""

    instances = []
    catalog_entry = {}
    manifest_capabilities = {}
    detail = {}
    # Optional scripted behaviors.
    start_error = None            # exception raised on the FIRST start only
    poll_error = None             # exception raised on the FIRST terminal read only
    artifact_bytes = None
    artifact_error = None
    nonterminal = False
    project_unregistered = False
    telemetry = None
    run_dir = None

    def __init__(self, *args, **kwargs):
        FakeGateway.instances.append(self)
        self.start_requests = []
        self.start_keys = []
        self.cancels = []
        self.artifact_gets = []
        self.health_asked = []
        self.run_gets = []
        self.project_lookups = []
        self.registrations = []
        self.removals = []
        self.engine_version = "3.3.7"

    @classmethod
    def reset(cls):
        # Faithful to the real /v2 split: the agent-capability catalog row
        # (CatalogHarness) carries NO transport flags — json_schema_output and
        # interactive live only on the /v2/harnesses row's manifest. A fixture
        # that invents a catalog flag would keep alive exactly the dead read
        # this suite exists to catch.
        cls.instances = []
        cls.catalog_entry = {
            "id": "fake-review", "enabled": True, "status": "ok",
            "accessProfilesSupported": ["readonly", "workspace_write"],
        }
        cls.manifest_capabilities = {"json_schema_output": True}
        cls.detail = _terminal_detail('{"findings": []}', conformance="passed")
        cls.start_error = None
        cls.poll_error = None
        cls.artifact_bytes = None
        cls.artifact_error = None
        cls.nonterminal = False
        cls.project_unregistered = False
        cls.telemetry = None
        cls.run_dir = None

    def handshake(self, **_kw):
        return {"compatible": True, "protocolMajor": 3, "engine": {"version": self.engine_version}}

    def agent_capabilities(self):
        return {"harnesses": [dict(FakeGateway.catalog_entry)]}

    def harnesses(self):
        return [{
            "id": FakeGateway.catalog_entry["id"],
            "status": FakeGateway.catalog_entry.get("status", "ok"),
            "manifest": {"capabilities": dict(FakeGateway.manifest_capabilities)},
        }]

    def quota_snapshots(self):
        return []

    def find_project_id(self, root):
        self.project_lookups.append(root)
        return "" if FakeGateway.project_unregistered else "proj-1"

    def register_project(self, root):
        self.registrations.append(root)
        return "proj-new"

    def remove_project(self, project_id):
        self.removals.append(project_id)
        return {"removed": True}

    def start_run(self, request, *, idempotency_key=""):
        self.start_requests.append(dict(request))
        self.start_keys.append(str(idempotency_key))
        if FakeGateway.start_error is not None:
            exc = FakeGateway.start_error
            FakeGateway.start_error = None
            raise exc
        return {"runId": "run-1", "runDir": "/tmp/fake-run"}

    def get_run(self, run_id, **_kw):
        self.run_gets.append(run_id)
        if FakeGateway.poll_error is not None:
            exc = FakeGateway.poll_error
            FakeGateway.poll_error = None
            raise exc
        if FakeGateway.nonterminal:
            return {"summary": {"state": "running"}, "lastSeq": 1}
        detail = json.loads(json.dumps(FakeGateway.detail))
        if self.run_dir is not None:
            telemetry = FakeGateway.telemetry
            if telemetry is None:
                telemetry = {"run_id": run_id, "final_attempt_id": "a01", "attempts": [{
                    "attempt_id": "a01", "harness_id": "fake-review",
                    "observed_model": detail.get("summary", {}).get("model"), "profile_id": None,
                }]}
            final = self.run_dir / "final"
            final.mkdir(parents=True, exist_ok=True)
            # JSON is a YAML subset: same separate artifact as the real engine.
            (final / "telemetry.yaml").write_text(json.dumps(telemetry), encoding="utf-8")
            detail.setdefault("summary", {})["runDir"] = str(self.run_dir)
        return detail

    def get_run_artifact(self, run_id, path):
        self.artifact_gets.append((run_id, path))
        if FakeGateway.artifact_error is not None:
            raise FakeGateway.artifact_error
        return FakeGateway.artifact_bytes or b""

    def cancel_run(self, run_id, *, reason=""):
        self.cancels.append((run_id, reason))
        return {"accepted": True}

    def close(self):
        pass

class FakeLLM:
    """Answers only the light-model extraction call."""

    def __init__(self, reply="[]"):
        self.reply = reply
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": self.reply}, {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.0001}

@pytest.fixture()
def fake_route(monkeypatch, tmp_path):
    FakeGateway.reset()
    FakeGateway.run_dir = tmp_path / "review-run"
    monkeypatch.setattr("ouroboros.gateways.claudexor.ClaudexorGateway", FakeGateway)
    monkeypatch.setenv(REVIEW_SESSION_ROUTE_ENV, "fake-review=fake-small:low")
    # ABI-10: the phase-5 per-row route envs are retired and IGNORED; nothing
    # to clear — rows built from plain model lists are pinned api_chat.
    # Custody memoization is process-local; a stale entry from another test's
    # run-1 would confuse ownership replay.
    custody._CUSTODY.clear()
    return FakeGateway

def _agent_request(**overrides):
    base = dict(
        surface="scope_review",
        goal="Review the staged change.",
        task_id="t-agent",
        call_type="scope_review",
        session_root="/tmp/fake-repo",
        session_task="Review the staged diff of this repository: run `git diff --cached`.",
    )
    base.update(overrides)
    return ReviewRequest(**base)

def _agent_slot(**overrides):
    base = dict(slot_id="scope_slot_1", model="api/model-a", timeout_sec=30,
                route=ReviewRouteKind.AGENT_SESSION)
    base.update(overrides)
    return ReviewSlot(**base)


def _run_session_directly(tmp_path, **overrides):
    """Call the shared session runner with explicit knobs (the B-class surface)."""
    from ouroboros.review_execution import (
        SessionInvocation, run_delegated_review_session,
    )

    invocation = dict(task_id="t-b", surface="scope_review", slot_id="scope_slot_1",
                      timeout_sec=30)
    kwargs = dict(prompt="review this", root="/tmp/fake-repo", custody_drive=tmp_path)
    for key in list(overrides):
        if key in ("task_id", "surface", "slot_id", "timeout_sec", "logical_key_extra",
                   "output_schema", "session_route", "instructions", "retry_state",
                   "operation_id", "pending_invocation_checkpoint"):
            invocation[key] = overrides.pop(key)
    kwargs.update(overrides)
    return run_delegated_review_session(invocation=SessionInvocation(**invocation), **kwargs)
