"""The owner API and native tools use one host service and existing authority."""
from __future__ import annotations

import json
from copy import deepcopy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import harness_maintenance as service
from ouroboros.gateway import harness_maintenance as api
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.tools import control_maintenance as tools
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_harness_maintenance_transport import OPERATION, REQUEST


@pytest.fixture
def owned(monkeypatch):
    calls, opened = [], []
    inventory = {"observedAt": "2026-10-09T00:00:00Z", "harnesses": [{"harness": "agy", "targets": ["latest"]}]}
    gateway = SimpleNamespace(
        maintenance_harnesses=lambda names, **kw: calls.append(("inspect", names, kw)) or deepcopy(inventory),
        maintenance_create=lambda body, key: calls.append(("create", body, key)) or deepcopy(OPERATION),
        maintenance_operation=lambda identity: calls.append(("status", identity)) or deepcopy(OPERATION),
        maintenance_cancel=lambda identity: calls.append(("cancel", identity)) or deepcopy(OPERATION),
    )

    @contextmanager
    def connection(kind):
        opened.append(kind)
        try:
            yield gateway
        finally:
            opened.append("closed")

    monkeypatch.setattr(service, "read_owned_gateway", lambda: connection("read"))
    monkeypatch.setattr(service, "ensure_owned_gateway", lambda: connection("ensure"))
    return SimpleNamespace(calls=calls, opened=opened, inventory=inventory, gateway=gateway)


@pytest.fixture
def http():
    with TestClient(Starlette(routes=[
        Route("/api/claudexor/maintenance/harnesses", api.api_harness_maintenance_inventory),
        Route("/api/claudexor/maintenance/operations", api.api_harness_maintenance_create, methods=["POST"]),
        Route("/api/claudexor/maintenance/operations/{operation_id}", api.api_harness_maintenance_operation),
        Route("/api/claudexor/maintenance/operations/{operation_id}/cancel", api.api_harness_maintenance_operation, methods=["POST"]),
    ])) as client:
        yield client


def test_owner_aliases_are_exact_engine_envelopes_and_passive_reads_never_ensure(http, owned):
    answer = http.get("/api/claudexor/maintenance/harnesses?harness=agy&harness=codex&fresh=true&checkLatest=true")
    assert answer.status_code == 200 and answer.json() == owned.inventory
    assert owned.calls == [("inspect", ["agy", "codex"], {"fresh": True, "check_latest": True})]
    assert "ensure" not in owned.opened
    answer = http.post("/api/claudexor/maintenance/operations", json=REQUEST, headers={"Idempotency-Key": "owner-one"})
    assert answer.status_code == 202 and answer.json() == OPERATION
    assert owned.calls[-1] == ("create", REQUEST, "owner-one")
    assert http.get("/api/claudexor/maintenance/operations/maintenance-one").json() == OPERATION
    cancel = http.post("/api/claudexor/maintenance/operations/maintenance-one/cancel")
    assert cancel.json() == OPERATION and cancel.json()["termination"] == "unconfirmed"
    assert owned.opened.count("ensure") == 1


def test_invalid_owner_input_refuses_before_prepare(http, owned):
    assert http.post("/api/claudexor/maintenance/operations", json=REQUEST).status_code == 400
    assert http.post("/api/claudexor/maintenance/operations", content="{").status_code == 400
    assert http.get("/api/claudexor/maintenance/harnesses?fresh=maybe").status_code == 400
    assert http.get("/api/claudexor/maintenance/harnesses?fresh=true&fresh=false").status_code == 400
    assert not owned.calls and not owned.opened


def test_owner_errors_keep_structured_problem_and_active_handle(http, owned):
    problem = {"code": "maintenance_already_active", "message": "Already active",
               "context": {"operationId": "maintenance-old"}, "requiredActions": ["inspect"]}

    def reject(*_args):
        error = ClaudexorUnavailable(problem["code"], problem["message"], status_code=409)
        error.problem = problem
        raise error

    owned.gateway.maintenance_create = reject
    reply = http.post("/api/claudexor/maintenance/operations", json=REQUEST, headers={"Idempotency-Key": "owner-one"})
    assert reply.status_code == 409 and reply.json() == {"error": problem}
    owned.gateway.maintenance_harnesses = lambda *_args, **_kw: (_ for _ in ()).throw(
        ClaudexorUnavailable("maintenance_unavailable", "Older engine", status_code=503))
    unavailable = http.get("/api/claudexor/maintenance/harnesses")
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "maintenance_unavailable"


def context(tmp_path, *, profile="root", metadata=None):
    constraint = ({"mode": profile, "surface": "external_workspace", "write_root": str(tmp_path)}
                  if profile != "root" else None)
    return ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="maintenance-task",
                       task_constraint=constraint, task_metadata=dict(metadata or {}))


def test_tools_share_service_and_update_crosses_existing_pause_boundary(tmp_path, monkeypatch, owned):
    from ouroboros import owner_pause
    passed = []
    monkeypatch.setattr(owner_pause, "run_operation", lambda ctx, fn, *a, **kw: passed.append(ctx) or fn(*a, **kw))
    ctx = context(tmp_path)
    assert json.loads(tools._inspect_harness(ctx, harnesses=["agy"])) == owned.inventory
    assert json.loads(tools._maintain_harness(ctx, "update", harness="agy", request_id="tool-one")) == OPERATION
    assert owned.calls[-1] == ("create", REQUEST, "tool-one")
    assert passed == [ctx]
    assert json.loads(tools._inspect_harness(ctx, operation_id="maintenance-one")) == OPERATION
    assert json.loads(tools._maintain_harness(ctx, "cancel", operation_id="maintenance-one")) == OPERATION
    assert passed == [ctx, ctx]


@pytest.mark.parametrize("arguments", [
    {"action": "update", "request_id": "missing-harness"},
    {"action": "update", "harness": "agy"},
    {"action": "update", "harness": "agy", "request_id": "bad\nkey"},
    {"action": "update", "harness": "agy", "request_id": "key", "operation_id": "existing"},
    {"action": "update", "harness": "agy", "request_id": "key", "target": "version"},
    {"action": "update", "harness": "agy", "request_id": "key", "version": "1.2.3"},
    {"action": "update", "harness": "agy", "request_id": "key", "target": "unknown"},
    {"action": "cancel"},
    {"action": "unknown"},
])
def test_flat_schema_keeps_action_requirements_enforced_before_engine(tmp_path, owned, arguments):
    answer = json.loads(tools._maintain_harness(context(tmp_path), **arguments))
    assert answer["error"]["code"] == "invalid_request"
    assert not owned.opened and not owned.calls


@pytest.mark.parametrize("profile", ["local_readonly_subagent", "acting_subagent"])
def test_limited_child_inspection_works_but_mutation_never_reaches_engine(tmp_path, monkeypatch, owned, profile):
    from ouroboros import config
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "pro")
    ctx = context(tmp_path, profile=profile)
    assert json.loads(tools._inspect_harness(ctx)) == owned.inventory
    refused = json.loads(tools._maintain_harness(ctx, "update", harness="agy", request_id="child-one"))
    assert refused["error"]["code"] == "maintenance_authority_refused"
    assert [call[0] for call in owned.calls] == ["inspect"]
    assert "ensure" not in owned.opened


def test_cyber_acting_profile_uses_existing_full_matrix(tmp_path, monkeypatch, owned):
    from ouroboros import config, owner_pause
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr(owner_pause, "run_operation", lambda ctx, fn, *a, **kw: fn(*a, **kw))
    ctx = context(tmp_path, profile="acting_subagent")
    assert json.loads(tools._maintain_harness(ctx, "update", harness="agy", request_id="cyber-one")) == OPERATION
    assert owned.calls[-1][0] == "create"


def test_observe_and_network_boundaries_preserve_inspection(tmp_path, owned):
    observe = {"initiator": "consciousness", "consciousness_autonomy": "observe"}
    ctx = context(tmp_path, metadata=observe)
    assert json.loads(tools._inspect_harness(ctx)) == owned.inventory
    assert json.loads(tools._maintain_harness(ctx, "update", harness="agy", request_id="observe-one"))["error"]["code"] == "maintenance_authority_refused"
    ctx = context(tmp_path, metadata={"allowed_resources": {"network": False}})
    assert json.loads(tools._inspect_harness(ctx)) == owned.inventory
    assert json.loads(tools._inspect_harness(ctx, check_latest=True))["error"]["code"] == "maintenance_authority_refused"
    assert json.loads(tools._maintain_harness(ctx, "update", harness="agy", request_id="network-one"))["error"]["code"] == "maintenance_authority_refused"
    assert all(call[0] == "inspect" for call in owned.calls)


@pytest.mark.parametrize("granted", [False, True])
def test_verified_presence_ceiling_is_the_authority_not_actor_identity(tmp_path, monkeypatch, owned, granted):
    from ouroboros import owner_pause
    from ouroboros.presence_authority import build_presence_capability_ceiling, presence_ceiling_payload
    from ouroboros.presence_capabilities import PresenceToolTarget
    from tests.test_presence_authority import _resolution

    monkeypatch.setattr(owner_pause, "run_operation", lambda ctx, fn, *a, **kw: fn(*a, **kw))
    selections = [PresenceToolTarget("builtin", "inspect_harness")]
    if granted:
        selections.append(PresenceToolTarget("builtin", "maintain_harness"))
    ceiling = build_presence_capability_ceiling(skill_name="maintenance-room",
        skill_content_hash="c" * 64, state_fingerprint="d" * 64, resolution=_resolution(*selections))
    ctx = context(tmp_path)
    ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
    assert json.loads(tools._inspect_harness(ctx)) == owned.inventory
    answer = json.loads(tools._maintain_harness(ctx, "update", harness="agy", request_id="presence-one"))
    if granted:
        assert answer == OPERATION
        assert owned.calls[-1] == ("create", REQUEST, "presence-one")
    else:
        assert answer["error"]["code"] == "maintenance_authority_refused"
        assert "ensure" not in owned.opened


def test_pause_prevents_daemon_preparation_and_submit(tmp_path, monkeypatch, owned):
    from ouroboros import owner_pause

    def paused(*_args, **_kw):
        raise owner_pause.OwnerPauseRefused("owner_pause")

    monkeypatch.setattr(owner_pause, "run_operation", paused)
    with pytest.raises(owner_pause.OwnerPauseRefused):
        tools._maintain_harness(context(tmp_path), "update", harness="agy", request_id="paused-one")
    assert not owned.opened


def test_registry_exports_tools_once_and_skip_policy_needs_no_model(tmp_path):
    from ouroboros import safety
    from ouroboros.consciousness_authority import disabled_tools_for
    from ouroboros.tools.control import get_tools
    names = [entry.name for entry in get_tools()]
    assert names.count("inspect_harness") == names.count("maintain_harness") == 1
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    assert registry.get_schema_by_name("inspect_harness")
    assert registry.get_schema_by_name("maintain_harness")
    for name in ("inspect_harness", "maintain_harness"):
        assert safety.TOOL_POLICY[name] == safety.POLICY_SKIP
        assert safety.check_safety(name, {}) == (True, "")
    assert "maintain_harness" in disabled_tools_for("observe")
    assert "inspect_harness" not in disabled_tools_for("observe")
