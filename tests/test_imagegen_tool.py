"""Fixture-only image wire, accounting, and ordinary tool/artifact consumers."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from ouroboros.gateways import image_api
from ouroboros.tools import imagegen as ig
from ouroboros.tools.registry import ToolContext, ToolRegistry

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
BODY = {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ZAI_API_KEY"):
        monkeypatch.setenv(key, "fixture-secret")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(image_api, "verify_kwargs", lambda: {})
    def no_engine():
        from ouroboros.gateways.claudexor import ClaudexorUnavailable
        raise ClaudexorUnavailable("fixture_offline", "No fixture engine")
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", no_engine)
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", no_engine)
    # A missed mock must not become a paid network call.
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda *args: pytest.fail("unexpected real HTTP"))


@pytest.fixture
def ctx(tmp_path):
    # An external workspace must not overlap the (fixture) Ouroboros repo or
    # the drive: the registry refuses such a task before any tool runs.
    workspace, repo = tmp_path / "workspace", tmp_path / "repo"
    workspace.mkdir()
    repo.mkdir()
    return ToolContext(repo_dir=repo, drive_root=tmp_path / "drive", task_id="image-task",
                       workspace_root=workspace, workspace_mode="external")


def _transport(monkeypatch, handler):
    # The client is built on the shared net_transport substrate; replace that
    # transport with the fixture so the request bytes themselves are observed.
    seen = {}
    def shared(*args, **kwargs):
        seen["built"] = True
        return httpx.MockTransport(handler)
    monkeypatch.setattr(image_api, "remote_httpx_transport", shared)
    monkeypatch.setattr(image_api, "env_proxies_configured", lambda: False)
    # Fixture CDNs have synthetic public DNS; address checks still run normally.
    import ipaddress
    from ouroboros import browser_policy
    resolve = browser_policy._resolved_addresses
    monkeypatch.setattr(browser_policy, "_resolved_addresses", lambda host:
                        {ipaddress.ip_address("8.8.8.8")} if host.endswith(".example") else resolve(host))
    return seen


def _account(monkeypatch):
    attempts = []
    def execute(request, send, **kwargs):
        from types import SimpleNamespace
        import uuid
        attempts.append(request)
        if kwargs.get("before_dispatch"):
            kwargs["before_dispatch"](SimpleNamespace(attempt_id=uuid.uuid4().hex))
        return send()
    monkeypatch.setattr(ig, "execute_physical_attempt", execute)
    return attempts


def _summary(text):
    return json.loads(text.split("\n", 1)[1])


def test_image_target_comes_from_the_shared_llm_route_resolution(monkeypatch):
    from ouroboros import llm

    calls = []
    original = llm.LLMClient.resolve_remote_target
    def spy(self, model, settings=None):
        calls.append(model)
        return original(self, model, settings)
    monkeypatch.setattr(llm.LLMClient, "resolve_remote_target", spy)
    target = image_api.resolve_image_target("openai::gpt-image-2")
    assert calls == ["openai::gpt-image-2"]
    assert target == {"provider": "openai", "model": "gpt-image-2", "base_url": "https://api.openai.com/v1",
                      "api_key": "fixture-secret", "headers": {}}
    monkeypatch.delenv("OPENAI_API_KEY")
    assert image_api.resolve_image_target("openai::gpt-image-2")["api_key"] == ""
    with pytest.raises(ValueError, match="no documented direct image endpoint"):
        image_api.resolve_image_target("anthropic::claude")


@pytest.mark.parametrize(("source", "endpoint", "image_field"), [
    ("openai", "/v1/images/edits", "images"),
    ("openrouter", "/api/v1/images", "input_references"),
])
def test_edit_sends_exact_original_bytes(monkeypatch, ctx, source, endpoint, image_field):
    # Valid PNG followed by application metadata survives byte-for-byte.
    original = PNG + b"original metadata: do not downscale or reencode"
    (ctx.workspace_root / "input.png").write_bytes(original)
    calls = []
    def handle(request):
        calls.append(request)
        assert request.method == "POST" and request.url.path == endpoint
        wire = json.loads(request.content)
        assert wire["model"] == "future-image-model"
        assert wire["prompt"] == "make the sky blue"
        data_url = (wire[image_field][0]["image_url"] if source == "openai"
                    else wire[image_field][0]["image_url"]["url"])
        assert base64.b64decode(data_url.split(",", 1)[1]) == original
        assert "response_format" not in wire
        return httpx.Response(200, json=BODY)
    built = _transport(monkeypatch, handle)
    attempts = _account(monkeypatch)
    output = ig._generate_image(ctx, "make the sky blue", f"{source}::future-image-model",
                               images=[{"root": "active_workspace", "path": "input.png"}])
    summary = _summary(output)
    assert output.startswith("OK:") and len(calls) == len(attempts) == 1 and built["built"]
    assert attempts[0].force_unknown_reservation
    assert summary["actual_cost_usd"] is None and summary["cost_final"] is False
    (image,) = summary["images"]
    assert Path(image["absolute_path"]).read_bytes() == PNG
    assert image["sha256"] == hashlib.sha256(PNG).hexdigest()
    assert image["root"] == "artifact_store" and image["mime"] == "image/png"
    assert base64.b64encode(PNG).decode() not in output
    assert ctx.pending_events == []


@pytest.mark.parametrize(("source", "endpoint"), [
    ("openai", "/v1/images/generations"), ("openrouter", "/api/v1/images"),
])
def test_generation_json_request(monkeypatch, ctx, source, endpoint):
    seen = []
    def handle(request):
        seen.append(request)
        assert request.url.path == endpoint
        assert json.loads(request.content) == {"model": "model-x", "prompt": "a castle", "size": "1024x1024"}
        return httpx.Response(200, json={**BODY, "usage": {"cost": 0.04}})
    _transport(monkeypatch, handle)
    _account(monkeypatch)
    summary = _summary(ig._generate_image(ctx, "a castle", f"{source}::model-x", options={"size": "1024x1024"}))
    assert len(seen) == 1 and summary["generated"] == 1
    assert summary["actual_cost_usd"] == 0.04 and summary["cost_final"]


def test_openrouter_endpoint_specific_options_pass_through(monkeypatch, ctx):
    # OpenRouter publishes per-endpoint supported_parameters; no local allowlist
    # may pretend to know them. Only custody-changing fields are refused locally.
    seen = []
    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=BODY)
    _transport(monkeypatch, handle)
    attempts = _account(monkeypatch)
    options = {"resolution": "2K", "aspect_ratio": "16:9", "background": "transparent",
               "provider": {"only": ["google-ai-studio"]}, "seed": 7}
    assert ig._generate_image(ctx, "a castle", "openrouter::future/image", options=options).startswith("OK:")
    assert seen == [{"model": "future/image", "prompt": "a castle", **options}]
    text = ig._generate_image(ctx, "a castle", "openrouter::future/image", options={"stream": True})
    assert "may not set" in text and len(attempts) == 1


def test_zai_uses_payg_single_url_and_does_not_leak_key(monkeypatch, ctx):
    # The shared route resolution selects the Coding Plan text endpoint; the
    # documented image endpoint is PAYG-only, so the image path maps back to it.
    monkeypatch.setenv("ZAI_PLAN", "coding")
    calls = []
    def handle(request):
        calls.append(request)
        if request.method == "POST":
            assert str(request.url) == "https://api.z.ai/api/paas/v4/images/generations"
            assert json.loads(request.content) == {"model": "glm-image", "prompt": "a castle", "quality": "hd"}
            assert request.headers["authorization"] == "Bearer fixture-secret"
            return httpx.Response(200, json={"data": [{"url": "https://cdn.example/image.png?signature=token"}]})
        assert "authorization" not in request.headers
        return httpx.Response(200, content=PNG)
    _transport(monkeypatch, handle)
    _account(monkeypatch)
    text = ig._generate_image(ctx, "a castle", "zai::glm-image", options={"quality": "hd"})
    summary = _summary(text)
    assert [request.method for request in calls] == ["POST", "GET"]
    assert summary["generated"] == 1 and summary["actual_cost_usd"] is None
    assert "signature=token" not in text and "fixture-secret" not in text


@pytest.mark.parametrize("source", ["zai", "openai"])
def test_native_options_are_validated_by_the_selected_provider(monkeypatch, ctx, source):
    seen = []
    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"code": "unsupported_parameter"}})
    _transport(monkeypatch, handle)
    attempts = _account(monkeypatch)
    options = {"future_native_option": "owner-choice"}
    result = ig._generate_image(ctx, "a castle", f"{source}::future-image", options=options)
    assert "HTTP 400" in result and len(attempts) == 1
    assert seen == [{"model": "future-image", "prompt": "a castle", **options}]


def test_zai_cannot_edit(monkeypatch, ctx):
    (ctx.workspace_root / "input.png").write_bytes(PNG)
    attempts = _account(monkeypatch)
    result = ig._generate_image(ctx, "edit", "zai::glm-image", images=[{"root": "active_workspace", "path": "input.png"}])
    assert "text only" in result and not attempts


def test_route_is_required_and_missing_credential_never_falls_back(monkeypatch, ctx):
    attempts = _account(monkeypatch)
    assert "no default is chosen" in ig._generate_image(ctx, "paint", "")
    monkeypatch.delenv("OPENAI_API_KEY")
    assert "No openai API credential" in ig._generate_image(ctx, "paint", "openai::model-x")
    assert not attempts


def test_timeout_is_one_attempt_with_durable_unknown(monkeypatch, ctx):
    calls = []
    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("lost response")
    _transport(monkeypatch, handle)
    attempts = _account(monkeypatch)
    text = ig._generate_image(ctx, "a castle", "openai::model-x")
    assert len(calls) == len(attempts) == 1 and "do not repeat automatically" in text
    rows = (ctx.drive_root / "logs" / "events.jsonl").read_text()
    assert "image_outcome_unknown" in rows and "fixture-secret" not in rows


def test_connect_failure_is_a_released_not_dispatched_attempt(monkeypatch, ctx):
    from ouroboros import usage_accounting as ua
    def handle(request):
        raise httpx.ConnectError("refused")
    _transport(monkeypatch, handle)
    text = ig._generate_image(ctx, "a castle", "openai::model-x")
    assert "nothing was billed" in text
    from ouroboros.tools.tool_result import TOOL_CODE_SPECS
    assert TOOL_CODE_SPECS["IMAGE_NOT_DISPATCHED"].status == "unavailable"
    rows = ua.read_usage_records(ctx.drive_root)
    assert [row["state"] for row in rows] == ["released"], rows
    events = [json.loads(line) for line in (ctx.drive_root / "logs" / "events.jsonl").read_text().splitlines()]
    assert events, "the ordinary ledger records its import checkpoint"
    assert not any(event.get("type") == "image_outcome_unknown" for event in events), events


@pytest.mark.parametrize(("status", "code"), [(400, "invalid_parameter"), (429, "image_generation_limit_reached"), (503, "upstream_error")])
def test_provider_failure_is_typed_not_retried(monkeypatch, ctx, status, code):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"code": code, "message": "echo fixture-secret"}}, headers={"retry-after": "60"})
    _transport(monkeypatch, handle)
    _account(monkeypatch)
    text = ig._generate_image(ctx, "paint", "openai::model-x")
    assert len(calls) == 1 and f"HTTP {status}" in text and code in text
    assert "fixture-secret" not in text and _summary(text)["retry_after"] == "60"


def test_owner_controls_raised_around_the_send_are_not_billed_unknowns(monkeypatch, ctx):
    from ouroboros.owner_pause import OwnerPauseRefused
    def refuse(request, send, **kwargs):
        raise OwnerPauseRefused("owner_pause")
    monkeypatch.setattr(ig, "execute_physical_attempt", refuse)
    with pytest.raises(OwnerPauseRefused):
        ig._generate_image(ctx, "paint", "openai::model-x")
    assert not (ctx.drive_root / "logs" / "events.jsonl").exists()


def test_download_failure_retains_recovery_source_and_never_resubmits(monkeypatch, ctx):
    calls = []
    def handle(request):
        calls.append(request)
        return (httpx.Response(200, json={"data": [{"url": "https://cdn.example/paid.png"}]})
                if request.method == "POST" else httpx.Response(503))
    _transport(monkeypatch, handle)
    _account(monkeypatch)
    summary = _summary(ig._generate_image(ctx, "paint", "zai::glm-image"))
    assert summary["generated"] == 0 and summary["failures"][0]["code"] == "IMAGE_DOWNLOAD_FAILED"
    from ouroboros.artifacts import task_artifact_dir_path
    retained = task_artifact_dir_path(ctx.drive_root, ctx.task_id) / summary["response"]["path"]
    assert json.loads(retained.read_text())["data"][0]["url"].endswith("paid.png")
    assert [request.method for request in calls] == ["POST", "GET"]


def test_partial_result_keeps_good_image_and_names_bad_row(monkeypatch, ctx):
    _account(monkeypatch)
    monkeypatch.setattr(image_api, "send_image_request", lambda *a: {"data": [BODY["data"][0], {"b64_json": "bad!!"}]})
    summary = _summary(ig._generate_image(ctx, "paint", "openai::model-x"))
    assert summary["generated"] == 1 and summary["failures"][0]["index"] == 1
    assert Path(summary["images"][0]["absolute_path"]).read_bytes() == PNG


def test_output_total_limit_is_enforced(monkeypatch, ctx):
    _account(monkeypatch)
    monkeypatch.setattr(image_api, "MAX_RESULT_BYTES", len(PNG))
    monkeypatch.setattr(image_api, "send_image_request", lambda *a: {"data": BODY["data"] * 2})
    summary = _summary(ig._generate_image(ctx, "paint", "openai::model-x"))
    assert summary["generated"] == 1 and "Combined output" in summary["failures"][0]["error"]


def test_api_discovery_is_source_qualified_and_unknown_stays_unknown(monkeypatch, ctx):
    monkeypatch.setattr(image_api, "fetch_image_models", lambda base_url: [
        {"id": "google/image-a", "supported_parameters": {"input_references": {"type": "array"}}, "pricing": [{"unit": "image", "price": 0.1}]},
        {"id": "other/image-b"},
    ])
    catalog = _summary(ig._list_image_models(ctx))
    rows = {row["model"]: row for row in catalog["models"]}
    assert rows["openrouter::google/image-a"]["editing"] is True
    assert rows["openrouter::other/image-b"]["editing"] is None
    assert rows["zai::glm-image"]["editing"] is False
    assert rows["zai::glm-image"]["pricing"] is None, "no manual tariff: price is unknown until reported"
    assert rows["zai::glm-image"]["parameters"]["quality"] == "hd | standard"
    assert "openai::gpt-image-2.5-sunburst" in rows and "openai::gpt-image-2.5" not in rows
    assert all(row["account_availability"] == "unverified" for row in rows.values())
    unknown = _summary(ig._list_image_models(ctx, model="openai::unlisted-image"))["models"][0]
    assert unknown["generation"] is None and unknown["editing"] is None
    monkeypatch.delenv("OPENAI_API_KEY")
    assert not _summary(ig._list_image_models(ctx, model="openai::unlisted-image"))["models"]


def test_an_added_image_key_lists_its_models_on_its_own_spelling(monkeypatch, ctx):
    """Each image connection that can serve lists its models under the key that pays for them; an
    exact inspection keeps the connection it names, and a switched-off key lists nothing."""
    import json as _json

    monkeypatch.setattr(image_api, "fetch_image_models", lambda base_url: [])
    rows = [{"kind": "openai", "name": "pictures", "api_key": "fixture-added"},
            {"kind": "openai", "name": "off", "api_key": "fixture-off", "enabled": False}]
    monkeypatch.setenv("OUROBOROS_CONNECTIONS", _json.dumps(rows))
    listed = {row["model"]: row for row in _summary(ig._list_image_models(ctx))["models"]}
    assert listed["openai@pictures::gpt-image-2"]["source"] == "openai@pictures"
    assert "openai::gpt-image-2" in listed and not any(model.startswith("openai@off::") for model in listed)
    exact = _summary(ig._list_image_models(ctx, model="openai@pictures::unlisted-image"))["models"]
    assert [row["model"] for row in exact] == ["openai@pictures::unlisted-image"]
    monkeypatch.delenv("OPENAI_API_KEY")  # an added key alone still serves its own spelling
    assert [row["model"] for row in _summary(ig._list_image_models(ctx, model="openai::unlisted-image"))["models"]] == []
    assert _summary(ig._list_image_models(ctx, model="openai@pictures::gpt-image-2"))["models"]


def test_catalog_failure_does_not_deny_unknown_explicit_model(monkeypatch, ctx):
    def fail(base_url):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(image_api, "fetch_image_models", fail)
    catalog = _summary(ig._list_image_models(ctx, model="openrouter::future/image"))
    assert catalog["models"][0]["generation"] is None
    assert any(error["code"] == "IMAGE_CATALOG_UNAVAILABLE" for error in catalog["errors"])
    _account(monkeypatch)
    monkeypatch.setattr(image_api, "send_image_request", lambda *args: BODY)
    assert ig._generate_image(ctx, "paint", "openrouter::future/image").startswith("OK:")


def test_catalog_fetch_uses_shared_transport_without_credentials(monkeypatch):
    seen = []
    def handle(request):
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "a/b"}, "junk"]})
    _transport(monkeypatch, handle)
    assert image_api.fetch_image_models("https://openrouter.ai/api/v1") == [{"id": "a/b"}]
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/images/models"
    assert "authorization" not in seen[0].headers


def test_artifact_and_edit_inputs_cannot_escape_resource_root(monkeypatch, ctx, tmp_path):
    forbidden = tmp_path / "outside.png"
    forbidden.write_bytes(PNG)
    attempts = _account(monkeypatch)
    text = ig._generate_image(ctx, "edit", "openai::model-x", images=[{"root": "active_workspace", "path": "../outside.png"}])
    assert not text.startswith("OK:") and not attempts
    (ctx.workspace_root / "link.png").symlink_to(forbidden)
    text = ig._generate_image(ctx, "edit", "openai::model-x", images=[{"root": "active_workspace", "path": "link.png"}])
    assert not text.startswith("OK:") and not attempts


def test_image_accounting_does_not_invent_text_price(monkeypatch, ctx):
    from ouroboros import usage_accounting as ua
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *args, **kwargs: pytest.fail("image priced as text"))
    monkeypatch.setattr(image_api, "send_image_request", lambda *args: {**BODY, "usage": {"input_tokens": 5, "output_tokens": 100}})
    summary = _summary(ig._generate_image(ctx, "paint", "openai::model-x"))
    rows = ua.read_usage_records(ctx.drive_root)
    assert len(rows) == 1 and rows[0]["cost_usd"] is None and not rows[0]["cost_final"]
    assert rows[0]["cost_evidence"]["usage"] == summary["usage"]


def test_budget_fence_prevents_provider_dispatch(monkeypatch, ctx):
    from ouroboros import usage_accounting as ua
    monkeypatch.setattr(image_api, "send_image_request", lambda *args: pytest.fail("provider dispatched"))
    # A real zero remaining bound, enforced by the existing task-tree fence. The
    # refusal is the loop's own budget rail (loop.py `except BudgetExceeded`),
    # not an image-tool text: the tool never swallows it as a billed unknown.
    from ouroboros.task_results import write_task_result
    write_task_result(ctx.drive_root, "parent-task", "running", root_task_id="parent-task")  # the child's real root
    scope = ua.UsageScope(drive_root=ctx.drive_root, task_id=ctx.task_id, root_task_id="parent-task", root_limit_usd=0)
    with ua.usage_scope(scope), pytest.raises(ua.BudgetExceeded):
        ig._generate_image(ctx, "paint", "openai::model-x")
    events = ctx.drive_root / "logs" / "events.jsonl"
    assert not events.exists() or "image_outcome_unknown" not in events.read_text()


def test_ordinary_acting_child_calls_tool_and_parent_receives_image(monkeypatch, tmp_path):
    from ouroboros.artifacts import collect_task_artifact_records, task_artifact_dir_path
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.headless import copy_child_task_result, prepare_task_drive
    from ouroboros.task_results import write_task_result
    parent, workspace, tid = tmp_path / "canonical", tmp_path / "workspace", "image-child"
    repo = tmp_path / "repo"
    workspace.mkdir()
    repo.mkdir()
    child = prepare_task_drive(parent, tid, "empty")
    ctx = ToolContext(repo_dir=repo, drive_root=child, budget_drive_root=str(parent), task_id=tid,
        workspace_root=workspace, workspace_mode="external",
        task_constraint=TaskConstraint(mode="acting_subagent", surface="external_workspace", write_root=str(workspace), allow_enable=False),
        task_metadata={"parent_task_id": "parent", "delegation_role": "subagent"})
    registry = ToolRegistry(repo_dir=repo, drive_root=child)
    registry.set_context(ctx)
    _account(monkeypatch)
    monkeypatch.setattr(image_api, "send_image_request", lambda *args: BODY)
    assert registry.get_schema_by_name("generate_image") is not None
    result = registry.execute("generate_image", {"prompt": "a website hero", "model": "openai::model-x"})
    summary = _summary(result)
    assert summary["generated"] == 1 and not ctx.pending_events
    write_task_result(child, tid, "completed", artifacts=collect_task_artifact_records(child, tid), artifact_status="ready")
    copied = copy_child_task_result(parent, {"id": tid, "drive_root": str(child)})
    image = next(row for row in copied["artifacts"] if row["kind"] == "generated_image")
    assert Path(image["path"]).parent == task_artifact_dir_path(parent, tid)
    assert Path(image["path"]).read_bytes() == PNG and image["immutable"]
    assert image["sha256"] == hashlib.sha256(PNG).hexdigest()


def test_readonly_and_network_constraints_block_generation(monkeypatch, ctx):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.contracts.task_contract import build_task_contract
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
    registry.set_context(ctx)
    assert registry.get_schema_by_name("generate_image") is None
    assert "LOCAL_READONLY_SUBAGENT_BLOCKED" in registry.execute("generate_image", {"prompt": "paint", "model": "openai::model-x"})
    ctx.task_constraint = None
    ctx.task_contract = build_task_contract({"id": ctx.task_id, "allowed_resources": {"network": False}})
    registry.set_context(ctx)
    for tool, args in [("generate_image", {"prompt": "paint", "model": "openai::model-x"}),
                       ("list_image_models", {}), ("recover_image", {"request_id": "no-request"}), ("vcs_pull_ff", {})]:
        refusal = registry.execute(tool, args)
        assert "RESOURCE_CONSTRAINT_BLOCKED" in refusal and f"blocks {tool!r}." in refusal


def test_presence_requires_exact_tool_and_input_grants(monkeypatch, ctx):
    from ouroboros.presence_authority import presence_ceiling_payload
    from tests.frozen_presence_authority import build_presence_capability_ceiling
    from tests.frozen_presence_capabilities import PresenceResourceTarget, PresenceToolTarget
    from tests.test_presence_authority import _resolution
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    def grant(*targets):
        ceiling = build_presence_capability_ceiling(skill_name="fixture", skill_content_hash="c" * 64,
            state_fingerprint="d" * 64, resolution=_resolution(*targets))
        ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
        registry.set_context(ctx)
    grant()
    assert registry.get_schema_by_name("generate_image") is None
    grant(PresenceToolTarget("builtin", "generate_image"))
    _account(monkeypatch)
    monkeypatch.setattr(image_api, "send_image_request", lambda *args: BODY)
    assert registry.execute("generate_image", {"prompt": "paint", "model": "openai::model-x"}).startswith("OK:")
    assert registry.get_schema_by_name("send_photo") is None and not ctx.pending_events
    (ctx.workspace_root / "input.png").write_bytes(PNG)
    args = {"prompt": "edit", "model": "openai::model-x", "images": [{"root": "active_workspace", "path": "input.png"}]}
    assert "outside the Presence file grant" in registry.execute("generate_image", args)
    grant(PresenceToolTarget("builtin", "generate_image"), PresenceResourceTarget("active_workspace", ("read",), "input.png"))
    assert registry.execute("generate_image", args).startswith("OK:")


def test_prior_result_is_an_ordinary_artifact_edit_input(monkeypatch, ctx):
    # The retained file from one generation is edited by the next through the
    # ordinary artifact_store root, byte for byte, with no vision projection.
    _account(monkeypatch)
    monkeypatch.setattr(image_api, "send_image_request", lambda *args: BODY)
    first = _summary(ig._generate_image(ctx, "a castle", "openai::model-x"))
    received = []
    def send(url, body, key, headers=None):
        received.append(base64.b64decode(body["images"][0]["image_url"].split(",", 1)[1]))
        return BODY
    monkeypatch.setattr(image_api, "send_image_request", send)
    result = ig._generate_image(ctx, "evening light", "openai::model-x",
                                images=[{"root": first["images"][0]["root"], "path": first["images"][0]["path"]}])
    assert result.startswith("OK:") and received == [PNG]


def test_storage_failure_does_not_resubmit_successful_generation(monkeypatch, ctx):
    calls = []
    _account(monkeypatch)
    monkeypatch.setattr(image_api, "send_image_request", lambda *args: calls.append(args) or BODY)
    original_store = ig.store_task_artifact_bytes
    def fail_image(*args, **kwargs):
        if kwargs.get("kind") == "generated_image":
            raise OSError("fixture full disk")
        return original_store(*args, **kwargs)
    monkeypatch.setattr(ig, "store_task_artifact_bytes", fail_image)
    summary = _summary(ig._generate_image(ctx, "paint", "openai::model-x"))
    assert len(calls) == 1 and summary["response"]
    assert summary["generated"] == 0 and summary["failures"][0]["code"] == "IMAGE_ARTIFACT_ERROR"


def test_http_and_encoded_result_limits(monkeypatch):
    _transport(monkeypatch, lambda request: httpx.Response(200, content=b"x" * 40))
    monkeypatch.setattr(image_api, "MAX_JSON_BYTES", 20)
    with pytest.raises(image_api.ImageBackendError) as caught:
        image_api.send_image_request("https://api.openai.com/v1/images/generations", {}, "fixture-key")
    assert caught.value.code == "IMAGE_OUTCOME_UNKNOWN"
    monkeypatch.setattr(image_api, "MAX_IMAGE_BYTES", 1)
    with pytest.raises(image_api.ImageBackendError, match="per-image limit"):
        image_api.read_result_image(BODY["data"][0])


def test_proxy_routed_install_keeps_env_proxy_mounts(monkeypatch):
    # Mirrors net_transport.keepalive_http_client: with env proxies and no trust
    # bundle, no explicit transport is attached (httpx would drop the mounts).
    monkeypatch.setattr(image_api, "env_proxies_configured", lambda: True)
    monkeypatch.setattr(image_api, "extra_ca_bundle", lambda: None)
    monkeypatch.setattr(image_api, "remote_httpx_transport", lambda *a, **k: pytest.fail("explicit transport on a proxied install"))
    built = {}
    cls = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: built.update(kwargs) or cls(**kwargs))
    image_api._client(5).close()
    assert "transport" not in built


def test_subscription_unavailable_does_not_fall_back(monkeypatch, ctx):
    attempts = _account(monkeypatch)
    result = ig._generate_image(ctx, "paint", "claudexor::codex=model-x")
    assert "refused before dispatch" in result and attempts == []


def test_api_recovery_downloads_again_without_new_attempt(monkeypatch, ctx):
    from ouroboros.usage_accounting import read_usage_records
    calls = []
    unavailable = True
    def handle(request):
        calls.append(request.method)
        if request.method == "POST":
            return httpx.Response(200, json={"data": [{"url": "https://cdn.example/image.png"}], "usage": {"cost": 0.02}})
        return httpx.Response(503) if unavailable else httpx.Response(200, content=PNG)
    _transport(monkeypatch, handle)
    first = _summary(ig._generate_image(ctx, "paint", "openrouter::image"))
    assert first["generated"] == 0
    unavailable = False
    recovered = _summary(ig._recover_image(ctx, first["request_id"]))
    assert recovered["generated"] == 1 and recovered["actual_cost_usd"] == 0.02
    assert calls.count("POST") == len(read_usage_records(ctx.drive_root)) == 1
    assert calls.count("GET") == 2
    ctx.task_id = "other-task"
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root); registry.set_context(ctx)
    assert registry.execute_result("recover_image", {"request_id": first["request_id"]}).code == "TOOL_ARG_ERROR"
    assert calls.count("GET") == 2


@pytest.mark.parametrize("partial", [False, True])
def test_registered_artifact_is_effect_without_automatic_delivery(monkeypatch, ctx, partial):
    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.outcomes import reviewable_effect_projection
    body = {"data": [*BODY["data"], *([{"b64_json": "broken"}] if partial else [])]}
    _transport(monkeypatch, lambda request: httpx.Response(200, json=body))
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root); registry.set_context(ctx)
    call = {"id": "gen", "function": {"name": "generate_image", "arguments": json.dumps({"prompt": "paint", "model": "openai::image"})}}
    row = _execute_single_tool(registry, call, ctx.drive_root / "logs", ctx.task_id)
    assert row["is_error"] is partial
    meta = row["result_meta"]
    assert meta["artifact_registered"] is True
    effects = reviewable_effect_projection({"tool_calls": [{"tool": "generate_image", "is_error": partial, **meta}]})
    assert len(effects) == 1 and not ctx.pending_events


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("control", ["budget", "interrupt"])
def test_registered_and_parallel_control_reaches_loop_owner(monkeypatch, ctx, parallel, control):
    from ouroboros import loop_tool_execution as execution, usage_accounting as ua
    from ouroboros.model_wait import ModelWaitInterrupted
    # Use the actual registry and wrapper. Generation is sequential in production;
    # a parallel-safe helper may also invoke a provider, so exercise that rail too.
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root); registry.set_context(ctx)
    name = "list_image_models"
    error = ua.BudgetExceeded("fixture budget") if control == "budget" else ModelWaitInterrupted("owner_stop")
    def handler(_ctx):
        raise error
    registry.override_handler(name, handler)
    monkeypatch.setattr(execution, "_PARALLEL_SAFE_TOOLS", frozenset({name}))
    calls = [{"id": f"control-{i}", "function": {"name": name, "arguments": "{}"}} for i in range(2 if parallel else 1)]
    with pytest.raises(type(error)) as caught:
        execution.handle_tool_calls(calls, registry, ctx.drive_root / "logs", ctx.task_id,
                                    object(), [], {"tool_calls": []}, lambda _: None)
    assert caught.value is error


@pytest.mark.parametrize("tool", ["generate_image", "list_image_models", "recover_image"])
def test_image_network_tools_share_the_caller_deadline_clamp(monkeypatch, ctx, tool):
    from ouroboros import loop_tool_execution as execution, task_pacing
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root); registry.set_context(ctx)
    monkeypatch.setattr(execution, "deadline_remaining_sec", lambda _: 40)
    monkeypatch.setattr(task_pacing, "effective_finalization_reserve_sec", lambda _: 15)
    assert execution._get_tool_timeout(registry, tool, {}) == 25
