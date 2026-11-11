"""Cited model choices preserve owner authority and explicit route identities."""
import json

import pytest

from tests.test_place_focus import focus as focus


@pytest.mark.parametrize("failure", ["policy", "storage", None])
def test_cited_main_write_distinguishes_refusal_uncertainty_and_success(model_start, monkeypatch, failure):
    from ouroboros.deployment_policy import DeploymentPolicyRefusal
    from ouroboros.tools.control_models import cited_switch

    ctx, _events = model_start
    writes = []
    def write(*args, **kwargs):
        writes.append(True)
        if failure == "policy":
            raise DeploymentPolicyRefusal("managed route")
        if failure == "storage":
            raise OSError("storage unavailable")
        return {"ok": True}
    monkeypatch.setattr("ouroboros.gateway.owner_effort._write_main_route", write)
    monkeypatch.setattr("ouroboros.gateway.ui_preferences.record_model_recent", lambda *_: None)
    result = cited_switch(ctx, model="vendor/chosen", owner_words="Please use the other one",
                          scope="main", project_id="", model_as_typed=True)
    assert writes == [True]
    assert ("refused by deployment policy" in result) is (failure == "policy")
    assert ("not confirmed" in result) is (failure == "storage")
    assert ('"ok": true' in result) is (failure is None)


@pytest.fixture
def model_start(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools import control_routing

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="caller", task_metadata={})
    ctx._owner_directives = [{"source": "initial_user", "content": "Please use the other one, then keep the report complete."}]
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    events = []
    monkeypatch.setattr(control_routing, "_emit_and_wait_for_routing",
                        lambda _ctx, event: (events.append(event) or "live", {"status": "scheduled"}))
    monkeypatch.setattr(control_routing, "_promotion_pool_disabled_from_snapshot", lambda _ctx: "")
    from ouroboros.tools.tool_result import _install_tool_result_sidecar, _restore_tool_result_sidecar
    token = _install_tool_result_sidecar(ctx, object())
    try:
        yield ctx, events
    finally:
        _restore_tool_result_sidecar(token)


@pytest.fixture
def configured_model_start(model_start, monkeypatch):
    from ouroboros.tools import control_models

    monkeypatch.setattr(control_models, "known_models", lambda: [
        {"model": "vendor/fast-one", "use_local": False}, {"model": "vendor/fast-two", "use_local": False}])
    return model_start


@pytest.mark.parametrize("cite", ["", "words never spoken"])
def test_a_model_without_verified_words_is_a_typed_refusal_with_no_uncited_retry(configured_model_start, cite):
    from ouroboros.tools.control_routing import _start_task
    from ouroboros.tools.tool_result import _published_tool_result

    ctx, events = configured_model_start
    text = _start_task(ctx, "Write the report", title="Report", model="vendor/fast-one", owner_words=cite)
    assert not events
    result = _published_tool_result(ctx, None)
    assert result.status == "error" and result.code == "TOOL_ARG_ERROR"
    assert "omit owner_words" not in text and "as your own" not in text


@pytest.mark.parametrize("model,starts", [("fast", False), ("vendor/fast-one", True), ("vendor/unlisted", True)])
def test_model_suggestions_never_guess_or_admit_a_route(configured_model_start, model, starts):
    from ouroboros.tools.control_routing import _start_task

    ctx, events = configured_model_start
    text = _start_task(ctx, "Write the report", title="Report", model=model, owner_words="use the other one")
    assert bool(events) is starts
    if not starts:
        assert "vendor/fast-one" in text and "vendor/fast-two" in text
    else:
        assert events[0]["metadata"]["lead"] == {"source": "owner_words", "model": model, "use_local": False,
                                                   "cited": "use the other one"}
        assert events[0]["owner_corpus"][-1]["content"] == ctx._owner_directives[0]["content"]
        if model == "vendor/unlisted":
            assert "not verified" in text


def test_an_unreadable_local_catalog_preserves_the_typed_choice(model_start, monkeypatch):
    from ouroboros.tools import control_models, control_routing

    ctx, events = model_start
    def unreadable():
        raise OSError("unavailable")
    monkeypatch.setattr(control_models, "known_models", unreadable)
    text = control_routing._start_task(ctx, "Write", title="Report", model="arbitrary/new", owner_words="use the other one")
    assert "not verified" in text and events[0]["metadata"]["lead"]["model"] == "arbitrary/new"


def test_cyber_start_names_the_minds_choice_without_overruling_a_room_owner(configured_model_start, monkeypatch):
    from ouroboros import route_choice, lead_fact
    from ouroboros.tools import control_models, control_routing

    ctx, events = configured_model_start
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    control_routing._start_task(ctx, "Write", title="Report", model="vendor/fast-one")
    ctx.task_metadata = events[0]["metadata"]
    assert ctx.task_metadata == {"lead": {"source": "mind", "model": "vendor/fast-one", "use_local": False}}
    route_choice.bind_task_route(ctx, ctx.task_metadata, presence=False)
    assert ctx.task_model_override == "vendor/fast-one" and lead_fact.primary_basis(ctx) == "mind"
    monkeypatch.setattr(control_models, "read_model_choice", lambda *a: {"kind": "model", "model": "owner/choice", "use_local": False})
    events.clear()
    text = control_routing._start_task(ctx, "Write", title="Report", model="vendor/fast-two")
    assert not events and "choice for this room binds" in text
    control_routing._start_task(ctx, "Write", title="Report", model="vendor/fast-two", owner_words="use the other one")
    assert events[0]["metadata"]["lead"]["source"] == "owner_words"


@pytest.fixture(params=["start_task", "switch_model"])
def local_model_choices(model_start, monkeypatch, request):
    """Real catalog/roster readers, with only control effects and transports replaced."""
    import requests
    from ouroboros import capability_evidence, config, llm_capability_policy, provider_connections, route_choice, settings_integrity
    from ouroboros.tools import control_models, control_routing, control_runtime
    from tests._socket_seal import block_network_except_socketpair

    ctx, events = model_start
    settings = {}
    monkeypatch.setattr(settings_integrity, "runtime_setting", settings.get)
    monkeypatch.setattr(provider_connections, "runtime_setting", settings.get)
    monkeypatch.setattr(config, "get_fallback_models", lambda: [])
    monkeypatch.setattr(route_choice, "following_route", lambda _role: ("vendor/main", False))
    # The host catalogue is shared, not a task/child-drive file.
    host_root = ctx.drive_root / "host"
    monkeypatch.setattr(capability_evidence, "canonical_evidence_root", lambda: host_root)
    catalog = host_root / "state/openrouter_models.json"
    catalog.parent.mkdir(parents=True)
    reads = []
    reader = llm_capability_policy.recorded_openrouter_catalog
    monkeypatch.setattr(llm_capability_policy, "recorded_openrouter_catalog",
                        lambda: reads.append("host catalogue") or reader())
    transports = []

    def no_transport(*_args, **_kwargs):
        transports.append("unexpected transport")
        raise AssertionError("Model suggestions must use only local observations")

    monkeypatch.setattr(requests, "get", no_transport)
    monkeypatch.setattr(llm_capability_policy, "read_owned_gateway", no_transport)
    block_network_except_socketpair(monkeypatch, no_transport)

    def work_choice(_ctx, choice):
        events.append({"metadata": {"lead": choice}})
        return "Primary saved", True

    monkeypatch.setattr(control_models, "request_work_model", work_choice)

    def choose(model, **kwargs):
        if request.param == "start_task":
            return control_routing._start_task(ctx, "Write", title="Report", model=model, owner_words="use the other one", **kwargs)
        return control_runtime._switch_model(ctx, model=model, owner_words="use the other one", **kwargs)

    yield choose, events, settings, catalog, reads, ctx
    assert not transports


@pytest.mark.parametrize("scope", ["room", "main"])
@pytest.mark.parametrize("actor", ["root", "presence", "child", "nanny", "reviewer"])
def test_only_integrating_foci_may_carry_owner_choices_to_room_or_main(configured_model_start, focus, monkeypatch, scope, actor):
    from ouroboros.owner_words import PLACE_CONFIRMED
    from ouroboros.tools.control_runtime import _switch_model
    from ouroboros.tools.tool_result import _published_tool_result

    ctx, _events = configured_model_start
    ctx.task_metadata = {"delegation_role": "subagent", "parent_task_id": "parent"} if actor == "child" else {}
    if actor == "nanny":
        ctx._nanny_route_dispatched = True
    if actor == "reviewer":
        ctx.task_metadata["review_role"] = "result"
    if actor == "presence":
        from ouroboros.presence_context import focus_metadata

        monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "full")
        root, _place, admission, event = focus
        ctx.task_metadata = focus_metadata(admission, event, ctx.task_id, 7, root)
        ctx.current_task_type = "presence"
        assert ctx.task_metadata["initiator"] == "consciousness"
        assert ctx.task_metadata["consciousness_autonomy"] == "full"
        ctx._owner_directives[0]["source"] = PLACE_CONFIRMED
    writes, recents = [], []
    monkeypatch.setattr("ouroboros.composer_cognition.update_project_model_choice",
                        lambda root, pid, choice: writes.append(choice) or ({"kind": "main"}, choice))
    monkeypatch.setattr("ouroboros.gateway.owner_effort._write_main_route",
                        lambda _request, choice, **kw: writes.append(choice) or {"ok": True, "main_route": choice})
    monkeypatch.setattr("ouroboros.gateway.ui_preferences.record_model_recent", lambda *a: recents.append(a))
    text = _switch_model(ctx, model="vendor/fast-one", owner_words="use the other one", scope=scope,
                         project_id="alpha" if scope == "room" else "")
    if actor in {"root", "presence"}:
        assert json.loads(text.splitlines()[0])["ok"] is True
        assert len(writes) == len(recents) == 1 and writes[0]["model"] == "vendor/fast-one"
    else:
        assert not writes and not recents
        result = _published_tool_result(ctx, None)
        assert result.status == "error" and result.code == "TOOL_ARG_ERROR"
        assert "Ask the work that delegated you" in text
        monkeypatch.setattr("ouroboros.llm.LLMClient.available_models", lambda _self: ["vendor/fast-one"])
        _switch_model(ctx, model="vendor/fast-one")
        assert ctx.active_model_override == "vendor/fast-one" and not writes


@pytest.mark.parametrize("cache", [True, False], ids=["recorded", "absent"])
def test_inexact_model_uses_the_host_catalogue_without_fetching(local_model_choices, cache):
    from ouroboros.tools.tool_result import _published_tool_result

    choose, events, settings, catalog, reads, ctx = local_model_choices
    settings["OPENROUTER_API_KEY"] = "test-only-key"
    if cache:
        catalog.write_text(json.dumps({"observed_at": "2026-01-02T03:04:05Z", "models": {
            "moonshotai/kimi-k3": {"context_length": 200000}}}), encoding="utf-8")
    text = choose("kimi")
    assert reads == ["host catalogue"]
    assert "live catalog check" in text.lower()
    if cache:
        assert not events and "moonshotai/kimi-k3" in text
        assert "cached OpenRouter catalogue" in text and "2026-01-02T03:04:05Z" in text
        result = _published_tool_result(ctx, None)
        assert result.status == "error" and result.code == "TOOL_ARG_ERROR"
    else:
        assert "not verified" in text and "used as typed" in text
        assert events[0]["metadata"]["lead"]["model"] == "kimi"


@pytest.mark.parametrize("fragment", [False, True])
def test_use_as_typed_can_confirm_an_unlisted_id_even_when_a_longer_route_is_known(local_model_choices, monkeypatch, fragment):
    import uuid
    from ouroboros.tools import control_models
    from ouroboros.tools.tool_result import _published_tool_result

    choose, events, _settings, _catalog, _reads, ctx = local_model_choices
    name = uuid.uuid4().hex
    exact = "openai::" + name
    known = exact + "-small"
    monkeypatch.setattr(control_models, "known_models", lambda: [{"model": known, "use_local": False}])
    requested = name if fragment else exact
    text = choose(requested)
    assert not events and known in text and "model_as_typed=true" in text
    assert _published_tool_result(ctx, None).status == "error"
    text = choose(requested, model_as_typed=True)
    assert events[-1]["metadata"]["lead"]["model"] == requested
    assert "not verified" in text and "used as typed" in text and known in text
    events.clear()
    choose(known)
    assert events[-1]["metadata"]["lead"]["model"] == known


@pytest.mark.parametrize("role", ["main", "light"])
def test_role_words_are_refused_as_ids_but_their_routes_remain_usable(local_model_choices, monkeypatch, role):
    from ouroboros.tools.control_runtime import _switch_model
    from ouroboros.tools.tool_result import _published_tool_result

    choose, events, _settings, _catalog, _reads, ctx = local_model_choices
    routes = {"main": ("openai::current-main", False), "light": ("local-light", True)}
    monkeypatch.setattr("ouroboros.route_choice.following_route", lambda role: routes.get(role, routes["main"]))
    text = choose(role, model_as_typed=True)
    assert not events and _published_tool_result(ctx, None).status == "error"
    assert "Main: openai::current-main" in text and "Light: local-light (local)" in text
    assert "needs no cite" in text
    choose(routes[role][0])
    assert events[-1]["metadata"]["lead"]["model"] == routes[role][0]
    monkeypatch.setattr("ouroboros.llm.LLMClient.available_models", lambda _self: [r[0] for r in routes.values()])
    _switch_model(ctx, model=routes[role][0])
    assert ctx.active_model_override == routes[role][0]


def test_main_role_cannot_be_saved_as_a_route_but_room_follow_main_remains(configured_model_start, monkeypatch):
    from ouroboros.tools.control_runtime import _switch_model

    ctx, _events = configured_model_start
    writes = []
    monkeypatch.setattr("ouroboros.gateway.owner_effort._write_main_route",
                        lambda _request, choice, **kw: writes.append(choice) or {"ok": True})
    monkeypatch.setattr("ouroboros.composer_cognition.update_project_model_choice",
                        lambda root, pid, choice: writes.append(choice) or ({"kind": "main"}, choice))
    monkeypatch.setattr("ouroboros.gateway.ui_preferences.record_model_recent", lambda *a: None)
    text = _switch_model(ctx, model="main", scope="main", owner_words="use the other one")
    assert "roles, not model ids" in text and not writes
    _switch_model(ctx, model="vendor/fast-one", scope="main", owner_words="use the other one")
    assert writes.pop()["model"] == "vendor/fast-one"
    _switch_model(ctx, model="main", scope="room", project_id="alpha", owner_words="use the other one")
    assert writes == [{"kind": "main"}]


@pytest.mark.parametrize("connection,route", [
    ("missing", ""), ("disabled", ""), ("builtin", "moonshotai/kimi-k3"),
    ("named", "openrouter@research::moonshotai/kimi-k3"),
])
def test_catalogue_suggestions_require_the_connection_to_have_credentials(local_model_choices, connection, route):
    choose, events, settings, catalog, reads, _ctx = local_model_choices
    catalog.write_text(json.dumps({"observed_at": "2026-11-10T00:00:00Z", "models": {
        "moonshotai/kimi-k3": {}}}), encoding="utf-8")
    if connection in {"builtin", "disabled"}:
        settings["OPENROUTER_API_KEY"] = "test-only-key"
    if connection == "disabled":
        settings["OUROBOROS_CONNECTIONS"] = [{"kind": "openrouter", "name": "default", "enabled": False}]
    if connection == "named":
        settings["OUROBOROS_CONNECTIONS"] = [{"kind": "openrouter", "name": "research", "api_key": "test-only-key"}]
    text = choose("kimi")
    if route:
        assert not events and route in text and reads == ["host catalogue"]
        if connection == "named":
            assert "\n- moonshotai/" not in text
    else:
        assert reads == [] and "moonshotai/kimi-k3" not in text
        assert "not verified" in text and events[0]["metadata"]["lead"]["model"] == "kimi"
    assert "test-only-key" not in text


@pytest.mark.parametrize("count", [12, 15])
def test_model_suggestions_are_bounded_counted_and_prefixes_come_first(local_model_choices, count):
    choose, events, settings, catalog, _reads, _ctx = local_model_choices
    settings["OPENROUTER_API_KEY"] = "test-only-key"
    models = {f"aaa/wrapped-kimi-{index:02}": {} for index in range(count - 1)}
    models["zzz/kimi-k3"] = {}  # Last in the record and alphabet, first by the typed prefix.
    catalog.write_text(json.dumps({"observed_at": "2026-11-10T00:00:00Z", "models": models}), encoding="utf-8")
    text = choose("KIMI")
    suggestions = [line for line in text.splitlines() if line.startswith("- ")]
    assert not events and len(suggestions) == 12
    assert suggestions[0].startswith("- zzz/kimi-k3 [")
    assert all("cached OpenRouter catalogue" in line for line in suggestions)
    if count > 12:
        assert "3 more matched; use a more specific name" in text
        assert "aaa/wrapped-kimi-11" not in text
    else:
        assert "more matched" not in text


@pytest.mark.parametrize("kind,review,local", [
    ("api_model", False, False), ("api_model", True, False),
    ("api_model", False, True), ("agent_session", False, False),
])
def test_configured_subagents_and_review_rows_suggest_only_model_routes(local_model_choices, kind, review, local):
    choose, events, settings, _catalog, reads, _ctx = local_model_choices
    target = "local-kimi.gguf (local)" if local else "moonshotai/kimi-k3"
    if kind == "agent_session":
        target = "codex=kimi-k3"
    settings["OUROBOROS_SUBAGENTS"] = json.dumps({"enabled": not review, "items": [{
        "subagent_id": "chosen", "recommended_use": "A configured helper or reviewer.",
        "route": {"kind": kind, "target_id": target}, "review_eligible": review}]})
    text = choose("kimi")
    assert reads == []
    if kind == "api_model":
        assert not events and target in text
        assert ("configured review-pool row" if review else "configured subagent row") in text
    else:
        assert target not in text and "not verified" in text
        assert events[0]["metadata"]["lead"]["model"] == "kimi"


def test_exact_catalogue_id_passes_with_its_date_and_configured_duplicate_wins(local_model_choices):
    choose, events, settings, catalog, _reads, _ctx = local_model_choices
    settings["OPENROUTER_API_KEY"] = "test-only-key"
    model = "moonshotai/kimi-k3"
    catalog.write_text(json.dumps({"observed_at": "2026-11-10T00:00:00Z", "models": {model: {}}}), encoding="utf-8")
    text = choose(model)
    assert events[0]["metadata"]["lead"]["model"] == model
    assert "cached OpenRouter catalogue" in text and "2026-11-10T00:00:00Z" in text
    settings["OUROBOROS_SUBAGENTS"] = {"enabled": True, "items": [{
        "subagent_id": "chosen", "recommended_use": "A configured helper.",
        "route": {"kind": "api_model", "target_id": model}}]}
    events.clear()
    text = choose("kimi")
    assert not events and text.count(model) == 1
    assert "configured subagent row" in text and "cached OpenRouter" not in text


@pytest.mark.parametrize("record", ["not json", '{"models": []}', '{"models": {"moonshotai/kimi-k3": {}}}'])
def test_broken_or_undated_catalogue_never_claims_verification(local_model_choices, record):
    choose, events, settings, catalog, _reads, _ctx = local_model_choices
    settings["OPENROUTER_API_KEY"] = "test-only-key"
    catalog.write_text(record, encoding="utf-8")
    text = choose("kimi")
    if "kimi-k3" in record:
        assert not events and "cached OpenRouter catalogue (date unavailable)" in text
    else:
        assert "not verified" in text and events[0]["metadata"]["lead"]["model"] == "kimi"
    assert "live catalog check" in text.lower()


@pytest.mark.parametrize("scope", ["room", "main"])
def test_uncited_cyber_scope_hint_names_its_writer_without_claiming_both_are_settings(model_start, monkeypatch, scope):
    from ouroboros.tools.control_runtime import _switch_model
    from ouroboros.tools.tool_result import _published_tool_result

    ctx, events = model_start
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    text = _switch_model(ctx, model="vendor/fast-one", scope=scope)
    assert not events and _published_tool_result(ctx, None).code == "TOOL_ARG_ERROR"
    assert "through its existing writer" in text and "existing settings writer" not in text
