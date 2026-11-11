"""The owner API and native tools use one host service and existing authority."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import harness_maintenance as service
from ouroboros.gateway import harness_maintenance as api
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.tools import control_maintenance as tools
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_harness_maintenance_transport import ENTRY, OPERATION, PROGRAM, REQUEST, client  # noqa: F401  (client: wire fixture)


@pytest.fixture
def owned(monkeypatch):
    calls, opened, budgets = [], [], []
    inventory = {"observedAt": "2026-10-09T00:00:00Z", "harnesses": [{**deepcopy(ENTRY), "harness": "agy"}]}
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

    monkeypatch.setattr(service, "read_owned_gateway", lambda **kw: budgets.append(kw) or connection("read"))
    monkeypatch.setattr(service, "ensure_owned_gateway", lambda: connection("ensure"))
    return SimpleNamespace(calls=calls, opened=opened, inventory=inventory, gateway=gateway, budgets=budgets)


@pytest.fixture
def http():
    with TestClient(Starlette(routes=[
        Route("/api/claudexor/maintenance/harnesses", api.api_harness_maintenance_inventory),
        Route("/api/claudexor/maintenance/operations", api.api_harness_maintenance_create, methods=["POST"]),
        Route("/api/claudexor/maintenance/operations/{operation_id}", api.api_harness_maintenance_operation),
        Route("/api/claudexor/maintenance/operations/{operation_id}/cancel", api.api_harness_maintenance_operation, methods=["POST"]),
    ])) as owner_client:
        yield owner_client


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


def test_owner_and_tool_reads_feed_the_same_local_fact_without_changing_the_reply(http, owned, tmp_path, monkeypatch):
    from ouroboros import config, harness_program_facts

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert http.get("/api/claudexor/maintenance/harnesses").json() == owned.inventory
    assert "agy 1.2.3 (newest not read)" in harness_program_facts.vendor_programs_line(tmp_path)
    owned.inventory["harnesses"][0]["selection"]["version"] = "vendor-build-z"
    owned.inventory["observedAt"] = "2026-12-12T00:00:00Z"
    assert json.loads(tools._inspect_harness(context(tmp_path), harnesses=["agy"])) == owned.inventory
    assert "agy vendor-build-z (newest not read)" in harness_program_facts.vendor_programs_line(tmp_path)
    assert owned.opened == ["read", "closed", "read", "closed"]


@pytest.mark.parametrize("action", ["status", "create", "cancel"])
@pytest.mark.parametrize("termination", ["confirmed", "unconfirmed"])
def test_settled_service_receipts_request_an_observation_only_after_custody_settles(owned, monkeypatch, action, termination):
    from ouroboros import harness_program_facts

    calls = []
    monkeypatch.setattr(harness_program_facts, "_pending", set())
    monkeypatch.setattr(harness_program_facts, "_launch", lambda *args: calls.append(args))
    owned.gateway._endpoint = object()
    receipt = {**deepcopy(OPERATION), "state": "succeeded", "termination": termination,
               "finishedAt": "2026-11-10T12:00:00Z"}
    owned.gateway.maintenance_operation = lambda *_: receipt
    owned.gateway.maintenance_create = lambda *_: receipt
    owned.gateway.maintenance_cancel = lambda *_: receipt
    result = {"status": lambda: service.inspect_operation("one"),
              "create": lambda: service.start_maintenance(REQUEST, "key"),
              "cancel": lambda: service.cancel_maintenance("one")}[action]()
    assert result is receipt
    assert len(calls) == (1 if termination == "confirmed" else 0)


def test_invalid_owner_input_refuses_before_engine_acquisition(http, owned):
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
    {"action": "update", "harness": "agy", "request_id": "key", "target": "version"},
    {"action": "update", "harness": "agy", "request_id": "key", "target": "unknown"},
    {"action": "update", "harness": "agy", "request_id": "key", "target": "previous"},
    {"action": "update", "harness": "agy", "request_id": "key", "target": "baseline"},
    {"action": "cancel"},
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
    from ouroboros.presence_authority import presence_ceiling_payload
    from tests.frozen_presence_authority import build_presence_capability_ceiling
    from tests.frozen_presence_capabilities import PresenceToolTarget
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


def test_pause_prevents_engine_acquisition_and_submit(tmp_path, monkeypatch, owned):
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


@pytest.mark.parametrize("target", [{"kind": "latest"}, {"kind": "version", "version": "1.2.3"}])
def test_update_submits_directly_and_same_key_recovers_a_lost_reply(http, owned, target):
    body = {**REQUEST, "target": target}
    submitted = []
    owned.gateway.maintenance_harnesses = lambda *_a, **_kw: pytest.fail("Update needs no inventory preflight")

    def create(received, key):
        submitted.append((received, key))
        if len(submitted) == 1:
            error = ClaudexorUnavailable("daemon_unreachable", "lost POST reply")
            error.problem = {"code": error.code, "context": {"acceptance": "unknown", "requestId": key}}
            raise error
        return deepcopy(OPERATION)

    owned.gateway.maintenance_create = create
    first = http.post("/api/claudexor/maintenance/operations", json=body,
                      headers={"Idempotency-Key": "same-key"})
    assert first.status_code == 503
    assert first.json()["error"]["context"] == {"acceptance": "unknown", "requestId": "same-key"}
    assert submitted == [(body, "same-key")], "no implicit retry"
    second = http.post("/api/claudexor/maintenance/operations", json=body,
                       headers={"Idempotency-Key": "same-key"})
    assert second.status_code == 202 and second.json() == OPERATION
    assert submitted == [(body, "same-key"), (body, "same-key")]


@pytest.mark.parametrize("code", ["daemon_unreachable", "daemon_spawn_failed", "attach_only_engine"])
def test_gateway_acquisition_failure_remains_before_submit(http, owned, monkeypatch, code):
    def offline():
        raise ClaudexorUnavailable(code, "cannot acquire gateway")
    monkeypatch.setattr(service, "ensure_owned_gateway", offline)
    response = http.post("/api/claudexor/maintenance/operations", json=REQUEST,
                         headers={"Idempotency-Key": "same-key"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["message"] == "cannot acquire gateway"
    assert response.json()["error"]["context"] == {
        "acceptance": "not_submitted", "scope": "this_attempt", "requestId": "same-key"}
    assert not owned.calls


def test_unrelated_submission_fault_is_not_swallowed(owned):
    def broken(*_args, **_kwargs):
        raise TypeError("programming fault")
    owned.gateway.maintenance_create = broken
    with pytest.raises(TypeError, match="programming fault"):
        service.start_maintenance(REQUEST, "same-key")


def test_os_failure_acquiring_gateway_is_proved_before_submit(http, owned, monkeypatch):
    def offline():
        raise OSError("cannot acquire gateway")
    monkeypatch.setattr(service, "ensure_owned_gateway", offline)
    reply = http.post("/api/claudexor/maintenance/operations", json=REQUEST,
                      headers={"Idempotency-Key": "same-key"})
    assert reply.status_code == 503
    assert reply.json()["error"]["code"] == "engine_unavailable"
    assert reply.json()["error"]["context"] == {
        "acceptance": "not_submitted", "scope": "this_attempt", "requestId": "same-key"}
    assert not owned.calls


@pytest.mark.parametrize("body", [
    {"harness": "agy", "target": {"kind": []}},
    {"harness": "agy", "target": {"kind": "version", "version": ""}},
    {"harness": "agy", "target": {"kind": "latest"}, "unexpected": True},
])
def test_engine_owns_update_body_validation(http, owned, body):
    def invalid(received, _key):
        assert received == body
        raise ValueError("engine rejects malformed request")
    owned.gateway.maintenance_create = invalid
    reply = http.post("/api/claudexor/maintenance/operations", json=body, headers={"Idempotency-Key": "bad-body"})
    assert reply.status_code == 400
    assert not owned.calls, "no inventory preflight for a malformed Update"


# ---- A current inspection whose reported executable is gone (#2461) ----
PINNED_SPAWN_FAILURE = json.loads((Path(__file__).parent / "fixtures/claudexor_maintenance_spawn_enoent.json").read_text(encoding="utf-8"))
RESTART_ADVICE = "Restart Claudexor in Accounts"


def spawn_failure_body(path):
    return {**deepcopy(PINNED_SPAWN_FAILURE["body"]), "message": f"spawn {path} ENOENT"}


@pytest.fixture
def real_wire(client, monkeypatch):  # noqa: F811
    """A supported engine returning the captured missing-executable problem bytes."""
    from ouroboros import review_run_isolation
    monkeypatch.setattr(review_run_isolation, "attach_home", lambda: None)
    monkeypatch.delenv("OUROBOROS_CLAUDEXOR_BIN", raising=False)

    def build(respond):
        gateway, seen = client(respond)
        gateway._engine_version = "6.0.0"
        gateway._engine_build_sha = PINNED_SPAWN_FAILURE["provenance"]["engine_build_sha"]

        @contextmanager
        def connection():
            yield gateway

        monkeypatch.setattr(service, "read_owned_gateway", connection)
        return seen
    return build


def failing_inventory(path):
    return lambda request: httpx.Response(PINNED_SPAWN_FAILURE["status"], json=spawn_failure_body(path))


def test_pinned_wire_reports_the_missing_executable_over_http_and_tool(http, real_wire, tmp_path, monkeypatch):
    missing = tmp_path / "moved-app" / "Contents" / "node"
    seen = real_wire(failing_inventory(missing))
    monkeypatch.setattr(service, "ensure_owned_gateway", lambda: pytest.fail("Inspection never starts an engine"))
    reply = http.get("/api/claudexor/maintenance/harnesses?harness=cursor&fresh=true")
    assert reply.status_code == PINNED_SPAWN_FAILURE["status"]
    problem = reply.json()["error"]
    # The engine's own facts stay exactly as received; only this host's observation is added.
    expected = spawn_failure_body(missing)
    assert {key: problem[key] for key in expected if key != "context"} == {key: value for key, value in expected.items() if key != "context"}
    assert problem["requiredActions"] == [], "may help is never a required action"
    facts = problem["context"]
    assert facts["observation"] == "reported_executable_missing"
    assert facts["engineVersion"] == "6.0.0"
    assert facts["engineBuildSha"] == PINNED_SPAWN_FAILURE["provenance"]["engine_build_sha"]
    assert facts["observedAt"]
    assert facts["remedy"].startswith("The executable reported by this inspection is no longer present.")
    assert RESTART_ADVICE in facts["remedy"] and "ends every run on the engine" in facts["remedy"]
    assert "not guaranteed" in facts["remedy"]
    assert [request.method for request in seen if request.url.path != "/v2/operations"] == ["GET"]
    answer = json.loads(tools._inspect_harness(context(tmp_path), harnesses=["cursor"], fresh=True))
    assert answer["error"]["context"]["observation"] == "reported_executable_missing"
    assert answer["error"]["message"] == f"spawn {missing} ENOENT"
    assert {request.method for request in seen} == {"GET"}


def test_a_broken_link_is_a_missing_executable(http, real_wire, tmp_path):
    link = tmp_path / "node"
    link.symlink_to(tmp_path / "gone")
    real_wire(failing_inventory(link))
    facts = http.get("/api/claudexor/maintenance/harnesses?fresh=true").json()["error"]["context"]
    assert facts["observation"] == "reported_executable_missing"


def unchanged(http, body, status=500):
    reply = http.get("/api/claudexor/maintenance/harnesses?fresh=true")
    assert reply.status_code == status
    assert reply.json()["error"] == body


def test_a_present_file_is_not_called_missing(http, real_wire, tmp_path):
    present = tmp_path / "node"
    present.write_text("")
    real_wire(failing_inventory(present))
    unchanged(http, spawn_failure_body(present))


@pytest.mark.parametrize("message", [
    "spawn node ENOENT", "spawn {missing} ENOENT\nspawn {missing} ENOENT", "ENOENT: no such file, open '{missing}'",
])
def test_unknown_message_shapes_pass_unchanged(http, real_wire, tmp_path, message):
    body = {**spawn_failure_body(tmp_path), "message": message.format(missing=tmp_path / "gone")}
    real_wire(lambda request: httpx.Response(500, json=body))
    unchanged(http, body)


@pytest.mark.parametrize("code", ["EACCES", "maintenance_inspection_failed", None])
def test_other_codes_pass_unchanged(http, real_wire, tmp_path, code):
    body = {**spawn_failure_body(tmp_path / "gone"), "code": code}
    real_wire(lambda request: httpx.Response(500, json=body))
    reply = http.get("/api/claudexor/maintenance/harnesses?fresh=true").json()["error"]
    assert "observation" not in (reply.get("context") or {})
    assert RESTART_ADVICE not in json.dumps(reply)


def test_a_path_that_cannot_be_examined_is_not_called_missing(http, real_wire, tmp_path, monkeypatch):
    body = spawn_failure_body(tmp_path / "gone")
    real_wire(lambda request: httpx.Response(500, json=body))
    real_stat = os.stat
    monkeypatch.setattr(service.os, "stat", lambda path, *a, **kw: (_ for _ in ()).throw(PermissionError("denied"))
                        if str(path) == str(tmp_path / "gone") else real_stat(path, *a, **kw))
    unchanged(http, body)


@pytest.mark.parametrize("selection", ["external", "attach_only"])
def test_a_selected_external_engine_gets_no_restart_advice(http, real_wire, tmp_path, monkeypatch, selection):
    from ouroboros import review_run_isolation
    body = spawn_failure_body(tmp_path / "gone")
    real_wire(lambda request: httpx.Response(500, json=body))
    if selection == "external":
        monkeypatch.setenv("OUROBOROS_CLAUDEXOR_BIN", "/external/claudexord")
    else:
        monkeypatch.setattr(review_run_isolation, "attach_home", lambda: "/external-home")
    unchanged(http, body)


def test_engine_context_survives_and_a_bad_context_is_not_trusted(http, real_wire, tmp_path):
    gone = tmp_path / "gone"
    real_wire(lambda request: httpx.Response(500, json={**spawn_failure_body(gone), "context": {"engineFact": 1}}))
    facts = http.get("/api/claudexor/maintenance/harnesses").json()["error"]["context"]
    assert facts["engineFact"] == 1 and facts["observation"] == "reported_executable_missing"
    real_wire(lambda request: httpx.Response(500, json={**spawn_failure_body(gone), "context": "text"}))
    assert http.get("/api/claudexor/maintenance/harnesses").json()["error"]["context"]["observation"] == "reported_executable_missing"


def historical_operation(path):
    return {**deepcopy(OPERATION), "state": "failed", "phase": "settled", "mutation": "unknown", "termination": "confirmed",
            "before": {**deepcopy(PROGRAM), "version": "2.0.0"},
            "finishedAt": "2026-10-09T12:00:00Z", "problem": spawn_failure_body(path)}


def test_retained_operation_after_replacement_never_advises_another_restart(http, real_wire, tmp_path):
    """The old operation names a path gone for ever: a healthy replacement must not read as needing Restart."""
    gone = tmp_path / "old-app" / "node"
    assert not gone.exists()
    failed = historical_operation(gone)
    replacement = {"observedAt": "2026-10-09T12:02:00Z", "harnesses": [{
        **deepcopy(ENTRY), "harness": "agy", "observedAt": "2026-10-09T12:02:00Z", "operation": failed}]}
    calls = []

    def replacement_engine(request):
        calls.append((request.method, request.url.path, request.headers.get("Idempotency-Key")))
        if request.url.path.endswith("/harnesses"):
            return httpx.Response(200, json=replacement)
        return httpx.Response(202 if request.method == "POST" else 200, json=failed)

    real_wire(replacement_engine)
    assert http.get("/api/claudexor/maintenance/harnesses?fresh=true").json() == replacement
    assert http.get("/api/claudexor/maintenance/operations/maintenance-one").json() == failed
    assert http.post("/api/claudexor/maintenance/operations/maintenance-one/cancel").json() == failed
    assert json.loads(tools._inspect_harness(context(tmp_path), operation_id="maintenance-one")) == failed
    assert RESTART_ADVICE not in json.dumps(failed)
    assert not any(method == "POST" and path.endswith("/operations") for method, path, _ in calls)


def test_create_and_rejoin_keep_acceptance_facts_and_inventory_failure_adds_no_precondition(http, real_wire, tmp_path, monkeypatch):
    """Inventory failure adds no precondition; the same-key rejoin returns the retained failure unchanged."""
    from ouroboros import claudexor_runtime as runtime
    gone = tmp_path / "gone"
    failed = historical_operation(gone)
    keys = []

    def engine(request):
        if request.url.path.endswith("/harnesses"):
            return failing_inventory(gone)(request)
        keys.append(request.headers["Idempotency-Key"])
        return httpx.Response(202, json=failed)

    real_wire(engine)
    monkeypatch.setattr(service, "ensure_owned_gateway", service.read_owned_gateway)
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: pytest.fail("A read failure prepares nothing"))
    for _attempt in range(2):
        reply = http.post("/api/claudexor/maintenance/operations", json=REQUEST, headers={"Idempotency-Key": "same-key"})
        assert reply.status_code == 202 and reply.json() == failed
    assert keys == ["same-key", "same-key"]


def test_an_engine_refusal_of_create_is_never_reclassified(http, real_wire, tmp_path, monkeypatch):
    gone = tmp_path / "gone"

    def engine(request):
        if request.url.path.endswith("/harnesses"):
            return httpx.Response(200, json={"harnesses": []})
        return httpx.Response(500, json=spawn_failure_body(gone))

    real_wire(engine)
    monkeypatch.setattr(service, "ensure_owned_gateway", service.read_owned_gateway)
    reply = http.post("/api/claudexor/maintenance/operations", json=REQUEST, headers={"Idempotency-Key": "same-key"})
    assert reply.status_code == 500 and reply.json()["error"] == spawn_failure_body(gone)


@pytest.fixture
def connect_clock(monkeypatch):
    from ouroboros import config
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(service, "time", SimpleNamespace(
        monotonic=lambda: clock.now, sleep=lambda delay: setattr(clock, "now", clock.now + delay)))
    monkeypatch.setattr(config, "get_claudexor_harness_install_timeout_sec", lambda: 3)
    return clock


def connected_program(**changes):
    return {**deepcopy(OPERATION), "harness": "future-program", "state": "succeeded", "phase": "settled",
            "termination": "confirmed", "outcome": "changed", "after": deepcopy(PROGRAM), **changes}


def test_connect_waits_on_the_same_operation_without_inventory_or_recreation(owned, connect_clock):
    accepted = connected_program(state="queued", phase="accepted", after=None, termination="not_applicable")
    reads = iter([connected_program(state="running", phase="installing", after=None), connected_program()])
    submitted, observed = [], []
    owned.gateway.maintenance_create = lambda body, key: submitted.append((body, key)) or accepted
    owned.gateway.maintenance_harnesses = lambda *_a, **_kw: pytest.fail("no latest-read or install preflight")
    owned.gateway.maintenance_operation = lambda identity, **kw: observed.append((identity, kw)) or next(reads)
    receipt = service.install_for_connect("future-program")
    assert receipt == connected_program()
    assert len(submitted) == 1
    assert submitted[0][0] == {"harness": "future-program", "target": {"kind": "latest"}}
    assert submitted[0][1]
    assert observed == [(accepted["id"], {"timeout_sec": 2.0}), (accepted["id"], {"timeout_sec": 1.0})]
    assert connect_clock.now == 2 and not owned.calls
    assert owned.budgets == [{"timeout_sec": 2.0}, {"timeout_sec": 1.0}]


@pytest.mark.parametrize("changes", [
    {"state": "failed"}, {"state": "cancelled"}, {"state": "interrupted"}, {"harness": "another-program"},
    {"after": None}, {"after": {**PROGRAM, "path": None}}, {"after": {**PROGRAM, "version": None}},
    {"after": {**PROGRAM, "versionOutput": None}}, {"after": {**PROGRAM, "version": "  "}},
])
def test_connect_rejects_a_settled_receipt_without_selected_version_evidence(owned, connect_clock, changes):
    receipt = connected_program(**changes)
    owned.gateway.maintenance_create = lambda *_a: receipt
    with pytest.raises(ClaudexorUnavailable) as failure:
        service.install_for_connect("future-program")
    assert failure.value.code == "harness_install_unverified"
    assert failure.value.problem["context"]["operation"] == receipt
    assert receipt["id"] in str(failure.value)
    assert not owned.calls and connect_clock.now == 0


@pytest.mark.parametrize("outcome", ["changed", "not_changed"])
def test_connect_accepts_a_selected_program_that_answers_version_without_ordering(owned, connect_clock, outcome):
    receipt = connected_program(outcome=outcome, after={**PROGRAM, "version": "vendor-build", "versionOutput": "vendor-build"})
    owned.gateway.maintenance_create = lambda *_a: receipt
    assert service.install_for_connect("future-program") == receipt
    assert not owned.calls and connect_clock.now == 0


@pytest.mark.parametrize("cancel_lost", [False, True])
def test_connect_timeout_keeps_unconfirmed_custody_and_never_claims_cancelled(owned, connect_clock, cancel_lost):
    receipt = connected_program(state="cancelled", termination="unconfirmed", after=None)
    cancels, reads = [], []
    owned.gateway.maintenance_create = lambda *_a: receipt
    owned.gateway.maintenance_operation = lambda identity, **kw: reads.append(identity) or receipt

    def cancel(identity, **kwargs):
        cancels.append((identity, kwargs))
        if cancel_lost:
            raise ClaudexorUnavailable("daemon_unreachable", "cancel reply lost")
        return receipt

    owned.gateway.maintenance_cancel = cancel
    with pytest.raises(ClaudexorUnavailable) as failure:
        service.install_for_connect("future-program")
    assert failure.value.code == "harness_install_timeout"
    assert failure.value.problem["context"]["operation"]["termination"] == "unconfirmed"
    assert cancels == [(receipt["id"], {"timeout_sec": 5})]
    assert len(reads) == 2 and connect_clock.now == 3
    assert owned.budgets == [{"timeout_sec": 2.0}, {"timeout_sec": 1.0}, {"timeout_sec": 5}]


@pytest.mark.parametrize("failure_at", ["create", "read"])
def test_connect_does_not_repeat_an_unanswered_submission_or_observation(owned, connect_clock, failure_at):
    calls = []
    receipt = connected_program(state="running", phase="installing", after=None)

    def create(body, key):
        calls.append(("create", body, key))
        if failure_at == "create":
            error = ClaudexorUnavailable("daemon_unreachable", "lost POST")
            error.problem = {"code": error.code, "context": {"requestId": key, "acceptance": "unknown"}}
            raise error
        return receipt

    def read(identity, **_kw):
        calls.append(("read", identity))
        raise ClaudexorUnavailable("daemon_unreachable", "lost read")

    owned.gateway.maintenance_create, owned.gateway.maintenance_operation = create, read
    with pytest.raises(ClaudexorUnavailable) as failure:
        service.install_for_connect("future-program")
    facts = failure.value.problem["context"]
    assert facts["requestId"] == calls[0][2]
    assert [call[0] for call in calls] == (["create"] if failure_at == "create" else ["create", "read"])
    if failure_at == "create":
        assert facts["acceptance"] == "unknown"
    else:
        assert facts["operationId"] == receipt["id"]
    assert not owned.calls


@pytest.mark.parametrize(("target", "version"), [("latest", ""), ("version", "1.2.3")])
def test_mind_targets_preserve_the_requested_selector_and_pause(tmp_path, monkeypatch, owned, target, version):
    from ouroboros import owner_pause
    ctx, crossed = context(tmp_path), []
    monkeypatch.setattr(owner_pause, "run_operation", lambda ctx, fn: crossed.append(ctx) or fn())
    result = json.loads(tools._maintain_harness(ctx, "update", harness="future-program", target=target,
                                              version=version, request_id="mind-key"))
    assert result == OPERATION
    requested = {"kind": target, **({"version": version} if version else {})}
    assert owned.calls == [("create", {"harness": "future-program", "target": requested}, "mind-key")]
    assert crossed == [ctx]
    schemas = {entry.name: entry.schema for entry in tools.maintenance_tool_entries()}
    assert schemas["maintain_harness"]["parameters"]["properties"]["target"]["enum"] == ["latest", "version"]


@pytest.mark.parametrize("surface", ["connect", "tool", "row", "background"])
def test_old_engine_cannot_receive_any_vendor_contract_request(client, monkeypatch, tmp_path, surface):
    from ouroboros import harness_program_facts as facts
    gateway, seen = client(lambda _: pytest.fail("No contract request to an old engine"), version="5.1.0")
    monkeypatch.setattr(service, "ensure_owned_gateway", lambda: gateway)
    monkeypatch.setattr(service, "read_owned_gateway", lambda: gateway)
    monkeypatch.setattr(facts, "_launch", lambda *_: pytest.fail("No old-engine background observation"))
    if surface == "tool":
        answer = json.loads(tools._maintain_harness(context(tmp_path), "update", harness="agy", request_id="old"))
        assert answer["error"]["code"] == "maintenance_unavailable"
    elif surface == "background":
        facts.engine_ready(gateway)
    else:
        with pytest.raises(ClaudexorUnavailable, match="Program inspection unavailable"):
            service.install_for_connect("agy") if surface == "connect" else service.inspect_harnesses()
    assert seen == []
