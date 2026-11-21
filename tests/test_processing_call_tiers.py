"""Main Ultrafast and its path through call receipts, review admission and inherited helper tiers."""
from __future__ import annotations

import asyncio
import copy
import json
import time
from types import SimpleNamespace
from typing import get_args, get_type_hints

import pytest

from ouroboros import route_choice
from ouroboros.model_slots import (
    CALL_TIERS,
    PROCESSING_PREFERENCES,
    normalize_call_tier,
    normalize_model_role_options,
    normalize_processing_preference,
    resolve_processing_preference,
    task_processing_preference,
)
from ouroboros.model_slots import MODEL_PROCESSING_PREFERENCES_KEY as ROLES
from ouroboros.model_slots import PROCESSING_PREFERENCE_KEY as GLOBAL
from tests.test_llm_claudexor import MODEL, ledger, result, setup  # noqa: F401 -- the Claudexor engine double
from tests.test_processing_claudexor import receipt, refusal
from tests.test_processing_transport import transport  # noqa: F401 -- the API provider double
from tests.test_usage_accounting import data_root  # noqa: F401 -- a usage ledger root

ULTRAFAST = {"source": "owner", "tier": "ultrafast"}


def _literal(annotation) -> set:
    """The string values of a (NotRequired-wrapped) Literal annotation."""
    while get_args(annotation) and not all(isinstance(value, str) for value in get_args(annotation)):
        annotation = get_args(annotation)[0]
    return set(get_args(annotation))


# --- The vocabulary ------------------------------------------------------------------------------------------------


def test_ultrafast_is_a_call_tier_and_can_be_saved_only_for_main():
    assert PROCESSING_PREFERENCES == ("standard", "fast", "economy")
    assert CALL_TIERS == ("standard", "fast", "economy", "ultrafast")
    assert normalize_call_tier(" Ultrafast ") == "ultrafast" and normalize_call_tier(None) == normalize_call_tier("") == ""
    for bad in ("turbo", "auto", 1, {}, ["fast"]):
        with pytest.raises(ValueError):
            normalize_call_tier(bad)
    with pytest.raises(ValueError):
        normalize_processing_preference("ultrafast")
    for options in ({"light": "ultrafast"}, {"safety": "ultrafast"}, {"fallback": ["", "ultrafast"]}):
        with pytest.raises(ValueError):
            normalize_model_role_options(ROLES, options)
    assert resolve_processing_preference(override="ultrafast") == "ultrafast"
    settings = {ROLES: '{"main":"ultrafast"}', GLOBAL: "economy"}
    assert resolve_processing_preference("main", settings=settings) == "ultrafast"
    assert resolve_processing_preference("main", override="", settings=settings) == ""
    assert resolve_processing_preference("light", settings=settings) == "economy"
    with pytest.raises(ValueError):
        resolve_processing_preference(settings={GLOBAL: "ultrafast"})


def test_an_agents_row_keeps_the_settings_vocabulary():
    from ouroboros.configured_subagents import normalize_configured_subagents
    from tests.test_processing_preferences import actor_settings

    with pytest.raises(ValueError):
        normalize_configured_subagents(actor_settings("ultrafast")["OUROBOROS_SUBAGENTS"])


def test_the_tier_contracts_name_ultrafast_where_a_room_a_work_or_a_call_holds_it():
    from ouroboros.gateway.client_contracts import OwnerChatProcessingRequest, OwnerChatProcessingResponse
    from ouroboros.gateway.contracts import ProjectEntry
    from ouroboros.gateway.work_contracts import ProcessingTier, ProcessingTiers, TaskModelRequest, WorkProcessing

    calls, settings = {"", *CALL_TIERS}, {"", *PROCESSING_PREFERENCES}
    assert _literal(ProcessingTier) == calls
    for cls, field in ((OwnerChatProcessingRequest, "processing"), (OwnerChatProcessingResponse, "processing_choice"),
                       (OwnerChatProcessingResponse, "previous_processing_choice"), (ProjectEntry, "processing_choice"),
                       (TaskModelRequest, "processing"), (WorkProcessing, "choice")):
        assert _literal(get_type_hints(cls, include_extras=True)[field]) == calls, (cls.__name__, field)
    response = get_type_hints(OwnerChatProcessingResponse, include_extras=True)
    tiers = get_type_hints(ProcessingTiers)
    assert _literal(response["main_processing"]) == _literal(response["previous_main_processing"]) == _literal(tiers["main"]) == calls
    assert _literal(response["processing_default"]) == _literal(tiers["default"]) == settings


def test_a_carrier_on_ultrafast_answers_for_the_roots_own_calls_and_says_so():
    from ouroboros.lead_fact import processing_fact
    from ouroboros.owner_model import PROCESSING_WORDS

    for source in ("room", "owner"):
        carrier = {"source": source, "tier": "ultrafast"}
        assert route_choice.processing_snapshot(carrier) == carrier
        assert route_choice.task_processing({"processing": carrier}) == "ultrafast"
        assert task_processing_preference({"metadata": {"processing": carrier}}) == "ultrafast"
        assert task_processing_preference({"metadata": {"processing": carrier}}, model_role="fallback:0") == "ultrafast"
        assert processing_fact({"processing": carrier}) == {"choice": "ultrafast", "source": source}
    assert route_choice.task_processing({"processing": ULTRAFAST, "delegation_role": "subagent"}) == ""
    assert PROCESSING_WORDS["ultrafast"] == "Ultrafast"


# --- One call through a surface without a native Ultrafast carrier --------------------------------------------------


@pytest.mark.parametrize(("provider", "asynchronous"), [("openai", True), ("anthropic", False)])
def test_an_api_call_on_ultrafast_sends_no_invented_native_field(transport, monkeypatch, provider, asynchronous):  # noqa: F811
    from tests._usage_store_testing import ledger_rows

    root, client, sent = transport
    monkeypatch.setenv("OPENAI_API_SURFACE", "chat_completions")
    kwargs = dict(messages=[{"role": "user", "content": "unchanged input"}], model=f"{provider}::test-model",
                  model_role="light", processing_preference="ultrafast", max_tokens=12)
    message, usage = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    assert message["content"] == "ok" and len(sent) == 1
    candidate = sent[0]["payload"] if provider == "anthropic" else sent[0]
    assert "service_tier" not in candidate and "speed" not in candidate and "processing_preference" not in candidate
    if provider == "anthropic":
        assert "fast-mode" not in sent[0]["headers"].get("anthropic-beta", "")
    assert usage["processing"]["requested"] == "ultrafast"
    assert usage["processing"]["submitted"] is None and usage["processing"]["submittedNative"] is None
    assert usage["processing"]["reason"] == "processing_not_submitted"
    assert usage["processing"]["observed"] == "unknown"
    rows = ledger_rows(root)
    assert all(row["processing_preference"] == "ultrafast" for row in rows)
    assert all(not row.get("submitted_processing_mode") for row in rows)
    assert rows[-1]["processing"] == usage["processing"]


@pytest.mark.parametrize("provider", ["openai", "openrouter", "anthropic", "claudexor", "openai-compatible"])
def test_ultrafast_needs_a_known_adapter_endpoint_and_surface(provider):
    from ouroboros.llm_attempt import apply_processing_preference

    target, payload = {"provider": provider, "processing_preference": "ultrafast"}, {"model": "m"}
    apply_processing_preference(target, payload)
    assert payload == {"model": "m"} and "processing_native_origin" not in target
    # An explicit native choice still wins, as for every tier.
    target, payload = {"provider": "openai", "processing_preference": "ultrafast"}, {"service_tier": "flex"}
    apply_processing_preference(target, payload)
    assert payload == {"service_tier": "flex"} and target["processing_native_origin"] == "native_override"


def test_the_receipt_of_an_unsubmitted_ultrafast_says_so():
    from ouroboros._usage_response import processing_receipt

    assert processing_receipt("openai", {}, requested="ultrafast") == {
        "requested": "ultrafast", "submitted": None, "submittedNative": None, "observed": "unknown",
        "observedNative": [], "reason": "processing_not_submitted", "source": "host_request"}
    assert processing_receipt("anthropic", {"speed": "standard"}, requested="ultrafast")["observed"] == "standard"


_FILTERS = {"require_parameters": True, "allow_fallbacks": False}  # the adapter's own pins: they drop endpoints


@pytest.mark.parametrize(("routing", "preference", "native", "pinned"), [
    (None, "fast", "priority", False),
    (_FILTERS, "fast", "priority", False),
    (None, "economy", "flex", False),
    (_FILTERS, "economy", "flex", True),  # dropping every Flex endpoint may route the call normally
    ({"allow_fallbacks": False, "only": ["vendor-a"]}, "fast", "priority", True),  # the router re-ask rung
    ({"order": ["vendor-b"]}, "fast", "priority", True),
    ("unreadable", "fast", "priority", True),
])
def test_a_reservation_prices_the_routing_its_sealed_candidate_carries(monkeypatch, routing, preference, native,
                                                                       pinned):
    """The tier pool's routing fact is read off the exact candidate, so a connection whose routing changes before
    reserve, or a rung that adds its own provider selection, is priced as sent; the settings are not reread."""
    from ouroboros import llm_routing
    from ouroboros import usage_accounting as ua
    from ouroboros.llm import LLMClient
    from ouroboros.llm_attempt import _attempt_request, apply_processing_preference

    slug = "vendor/synthetic-routing"
    payload = {"model": slug, "messages": [{"role": "user", "content": "input"}],
               **({"extra_body": {"provider": copy.deepcopy(routing)}} if routing is not None else {})}
    target = {**LLMClient().resolve_remote_target(slug), "processing_preference": preference}
    apply_processing_preference(target, payload)
    request = _attempt_request(target, payload)
    assert request.submitted_processing_mode == native and request.provider_routing_pinned is pinned
    monkeypatch.setattr(llm_routing, "_resolve_or_provider", lambda *_a: pytest.fail("settings reread at reserve"))
    seen = []
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda model, *a, **kw: seen.append(
        (kw["processing_mode"], kw["routing_pinned"])) or 1.0)
    assert ua._reservation_cost(request) == 1.0 and seen == [(native, pinned)]


@pytest.mark.parametrize(("add_on", "unknown"), [
    ({"tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]}, False),
    ({"tools": [{"type": "openrouter:web_search", "parameters": {"max_total_results": 10}}]}, True),
    ({"extra_body": {"plugins": [{"id": "web"}]}}, True),
    ({"web_search_options": {"search_context_size": "low"}}, True),
    ({"model": "vendor/synthetic-routing:online"}, True),
], ids=["function_tool", "server_tool", "plugin", "web_search_options", "online_shortcut"])
def test_an_add_on_charge_leaves_the_reservation_unknown_never_a_token_bound(monkeypatch, add_on, unknown):
    """A server tool or plugin is billed per use beside the tokens, so no token tariff bounds that call: its
    reservation is unknown whatever tier it carries, while an ordinary function tool keeps its tier pool."""
    from ouroboros import usage_accounting as ua
    from ouroboros.llm import LLMClient
    from ouroboros.llm_attempt import _attempt_request, apply_processing_preference

    slug = "vendor/synthetic-routing"
    payload = {"model": slug, "messages": [{"role": "user", "content": "input"}], **copy.deepcopy(add_on)}
    target = {**LLMClient().resolve_remote_target(slug), "processing_preference": "fast"}
    apply_processing_preference(target, payload)
    request = _attempt_request(target, payload)
    assert request.submitted_processing_mode == "priority" and request.force_unknown_reservation is unknown
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **kw: 1.0)
    assert ua._reservation_cost(request) == (None if unknown else 1.0)


def test_a_review_preflight_projects_its_seats_routing_and_locality(monkeypatch):
    """Before any candidate exists the projection carries what the adapter will add from the seat's resolved connection
    (its provider routing), and a local seat is the local lane's own request, never its remote model's tariff."""
    from ouroboros import usage_accounting as ua
    from ouroboros.llm import LLMClient
    from ouroboros.usage_admission import projected_pricing_request

    monkeypatch.delenv("OUROBOROS_OR_PROVIDER", raising=False)
    slug = "vendor/synthetic-routing"
    base = LLMClient().resolve_remote_target(slug)
    for routing, pinned in ((None, False), ("repro", False), ({"only": ["vendor-a"]}, True)):
        request = projected_pricing_request(slug, "fast", lambda _model, value=routing: {**base, "provider_routing": value},
                                            review=True, use_local=False)
        assert (request.submitted_processing_mode, request.provider_routing_pinned) == ("priority", pinned)
    local = projected_pricing_request(slug, "fast", lambda _model: pytest.fail("a local seat resolves no remote route"),
                                      review=True, use_local=True)
    assert (local.provider, local.model, local.force_unknown_reservation) == ("local", "local-model", False)
    assert ua._reservation_cost(local) == 0.0
    online = projected_pricing_request(slug + ":online", "fast", LLMClient().resolve_remote_target, review=True,
                                       use_local=False)
    assert online.force_unknown_reservation and ua._reservation_cost(online) is None


@pytest.mark.parametrize(("slug", "sealed", "routing", "wire_usd", "preflight_usd"), [
    ("anthropic/fixture", False, {"require_parameters": True}, 0.009, 0.009),
    ("vendor/fixture", True, {"allow_fallbacks": False}, 0.009, 0.009),
    ("vendor/fixture", False, {}, 0.001, 0.009),
    ("google/gemini-fixture", True, {}, 0.001, 0.001),
])
def test_flex_preflight_bounds_the_chat_adapters_actual_provider_pool(
        monkeypatch, slug, sealed, routing, wire_usd, preflight_usd):
    from ouroboros import pricing, usage_accounting as ua
    from ouroboros.llm import LLMClient
    from ouroboros.llm_attempt import _attempt_request

    monkeypatch.setenv("OUROBOROS_OR_PROVIDER", "")
    key = ("openrouter", slug)
    tariffs = {tier: pricing.PricingSchedule((0, 0, 0, rate), cache_write_1h=0, source={"processing": tier})
               for tier, rate in [("economy", 1), ("standard", 9)]}
    monkeypatch.setattr(pricing, "_cached_pricing", {key: tariffs})
    monkeypatch.setattr(pricing, "_pricing_fetched_at", {key: time.time()})
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda *args: pytest.fail(f"unexpected fetch {args}"))
    client = LLMClient()
    target = {**client.resolve_remote_target(slug), "processing_preference": "economy"}
    messages = [{"role": "user", "content": "input"}]
    if sealed:
        messages.append({"role": "assistant", "content": "output",
                         "reasoning_details": [{"type": "reasoning.encrypted", "data": "fixture"}]})
    payload = client._build_remote_kwargs(target, messages, "medium", 1000, "auto", None, None,
                                          skip_capability_fetch=True)
    assert payload["extra_body"].get("provider", {}) == routing and payload["service_tier"] == "flex"
    physical = _attempt_request(target, payload)
    assert ua._reservation_cost(physical) == pytest.approx(wire_usd)
    wave = ua.review_wave_admission(root_task_id="review", models=[slug], use_local=False,
                                   prompt_chars=physical.prompt_tokens_estimate * 4, max_completion_tokens=1000,
                                   processing_preferences="economy", remaining_usd_override=1)
    assert wave["slot_bounds"] == [preflight_usd] and wave["unpriced_slots"] == 0


def test_review_cannot_price_ultrafast_on_openai_chat_completions(data_root, monkeypatch):  # noqa: F811
    from ouroboros import usage_accounting as ua

    monkeypatch.setenv("OPENAI_API_SURFACE", "chat_completions")
    priced, real = [], ua._reservation_cost

    def reservation(request):
        priced.append((request.processing_preference, request.submitted_processing_mode,
                       request.force_unknown_reservation))
        return real(request)
    monkeypatch.setattr(ua, "_reservation_cost", reservation)
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **kw: 0.25)
    result = ua.review_wave_admission(data_root, root_task_id="review", models=["openai::same"] * 5,
                                      prompt_chars=400, remaining_usd_override=10,
                                      processing_preferences=["standard", "ultrafast", " Ultrafast ", "", "fast"])
    assert priced == [("standard", "default", False), ("ultrafast", "", True), (" Ultrafast ", "", True),
                      ("", "", False), ("fast", "priority", False)]
    bounds = result["slot_bounds"]
    assert bounds[1] is None and bounds[2] is None
    assert all(isinstance(bound, float) and bound > 0 for bound in (bounds[0], bounds[3], bounds[4]))
    assert result["unpriced_slots"] == 2
    assert result["estimated_wave_usd"] == round(bounds[0] + bounds[3] + bounds[4], 6)


# --- Claudexor: submitted where advertised; the proved advisory refusal retries Standard once -----------------------


def _advertise(monkeypatch, *preferences):
    from ouroboros import llm_claudexor as cx

    monkeypatch.setattr(cx, "model_sources", lambda **kwargs: {
        "sources": [{"id": "codex", "processingPreferences": list(preferences)}]})


def test_a_source_that_advertises_ultrafast_receives_it(setup, monkeypatch):  # noqa: F811
    root, gateway, client = setup
    _advertise(monkeypatch, "standard", "fast", "economy", "ultrafast")
    observed = receipt("ultrafast", "ultrafast", "ultrafast")
    gateway.results = [{**result(), "processing": observed}]
    _message, usage = client.chat([{"role": "user", "content": "input"}], MODEL, processing_preference="ultrafast")
    assert gateway.uploads[0][0]["options"]["processingPreference"] == "ultrafast"
    assert usage["processing"] == observed and ledger(root)[-1]["processing"] == observed


def test_a_source_that_does_not_advertise_ultrafast_never_receives_it(setup, monkeypatch):  # noqa: F811
    root, gateway, client = setup
    _advertise(monkeypatch, "standard", "fast", "economy")
    _message, usage = client.chat([{"role": "user", "content": "input"}], MODEL, processing_preference="ultrafast")
    assert "processingPreference" not in gateway.uploads[0][0]["options"]
    assert usage["processing"]["requested"] == "ultrafast" and usage["processing"]["submitted"] is None
    assert ledger(root)[-1]["processing"] == usage["processing"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_proved_pre_generation_ultrafast_refusal_retries_standard_once(setup, monkeypatch, asynchronous):  # noqa: F811
    root, gateway, client = setup
    _advertise(monkeypatch, "standard", "fast", "economy", "ultrafast")
    gateway.results = [refusal("ultrafast", "unsupported"), {**result(), "processing": receipt()}]
    gateway.dispatch = ["response_received", "response_received"]
    kwargs = dict(messages=[{"role": "user", "content": "input"}], model=MODEL, processing_preference="ultrafast")
    _message, usage = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    first, second = [row[0] for row in gateway.uploads]
    assert first["options"]["processingPreference"] == "ultrafast"
    assert {**first, "options": {**first["options"], "processingPreference": "standard"}} == second
    finals = ledger(root)
    assert [row["state"] for row in finals] == ["settled", "settled"]
    assert all(row["processing_preference"] == "ultrafast" for row in finals)
    # The call asked Ultrafast and was sent and served Standard: its receipt says so (owner 13.11: continue on
    # Standard, the choice and the outcome visible); the physical attempt keeps the engine's own receipt.
    assert usage["processing"] == {**receipt(), "requested": "ultrafast", "reason": "submitted_mode_differs"}
    assert finals[-1]["processing"] == receipt()


def test_an_ultrafast_refusal_without_proof_is_not_retried(setup, monkeypatch):  # noqa: F811
    from ouroboros import llm_claudexor as cx

    _root, gateway, client = setup
    _advertise(monkeypatch, "standard", "fast", "economy", "ultrafast")
    response = refusal("ultrafast", "unsupported")
    response["problem"]["context"]["generationStarted"] = True
    gateway.results = [response]
    with pytest.raises(cx.ClaudexorModelError):
        client.chat([], MODEL, processing_preference="ultrafast")
    assert len(gateway.accepted_operations) == 1


# --- A fallback row serves the work's own tier ---------------------------------------------------------------------


NATIVE = {"standard": "default", "fast": "priority", "economy": "flex"}


@pytest.mark.parametrize("carrier,expected", [
    ({"source": "owner", "tier": "economy"}, "economy"),  # the card's Flex: what it promised from the next call
    ({"source": "owner", "tier": "standard"}, "standard"),  # an explicit Standard is a choice too
    ({"source": "room", "tier": "economy"}, "economy"),
    ({"source": "default"}, "fast"),  # Default: the row's own tier stands
    (None, "fast"),  # admitted before tiers: the default resolution, the row's own tier
])
def test_a_round_on_a_fallback_row_sends_the_works_own_tier_over_the_rows(setup, monkeypatch, carrier, expected):  # noqa: F811
    from ouroboros.loop_model_call import _RoundModelCallContext, _call_round_model
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry

    root, gateway, client = setup
    _advertise(monkeypatch, "standard", "fast", "economy")
    monkeypatch.setenv(GLOBAL, "")
    monkeypatch.setenv(ROLES, json.dumps({"main": "standard", "fallback": ["fast"]}))
    write_task_result(root, "task-one", "running", root_task_id="task-one", _attempt=1)
    gateway.results = [{**result(), "processing": receipt(expected, NATIVE[expected], expected)}]
    tools = ToolRegistry(repo_dir=root.parent, drive_root=root)
    tools._ctx.task_metadata = {"lead": {"source": "main"}, **({"processing": carrier} if carrier else {})}
    tools._ctx.active_model = MODEL
    # The chain's row serves this round (its route as the engine names it, so nothing is rebound after the answer).
    tools._ctx.context_fit_plan = SimpleNamespace(model_role="fallback:0", model_route=copy.deepcopy(gateway.results[0]["route"]))
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    message, _cost, _mode = _call_round_model(_RoundModelCallContext(
        llm=client, messages=[{"role": "system", "content": "Own SYSTEM"}, {"role": "user", "content": "go on"}],
        tools=tools, context_fit_plan=None, active_model=MODEL, tool_schemas=[], active_effort="high", max_retries=1,
        drive_logs=logs, task_id="task-one", round_idx=1, event_queue=None, accumulated_usage={}, task_type="task",
        active_use_local=False, active_context_mode="max", drive_root=root, attempt_cap=1))
    assert message["content"] == "Ответ 🐍"
    [(payload, _key)] = gateway.uploads
    assert payload["options"]["processingPreference"] == expected
    assert ledger(root)[-1]["processing_preference"] == expected


# --- A helper's tier ------------------------------------------------------------------------------------------------


def _rows(*rows) -> dict:
    return {"OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": list(rows)}), GLOBAL: "economy"}


def _session(row_id="session-builder", processing=None):
    from tests.test_available_subagents_runtime import _session_row

    row = _session_row(row_id)
    if processing is not None:
        row["processing_preference"] = processing
    return row


def _runs_on(snapshot) -> str:
    """What a helper's run asks for (the one seam both doors use): the work's tier a row without one took."""
    from ouroboros.subagent_runtime import run_processing

    return run_processing(snapshot)


@pytest.mark.parametrize("work_tier,own,expected", [
    ("fast", None, "fast"),  # a row without a tier of its own takes the work's
    ("ultrafast", None, "ultrafast"),
    ("fast", "standard", "standard"),  # a row with its own keeps it
    (None, None, "economy"),  # nothing known of the work's tier: the inherited default stands
    ("", None, ""),  # a work resolved to no tier hands none
    ("", "standard", "standard"),
])
def test_a_helper_without_a_tier_of_its_own_takes_the_works(work_tier, own, expected):
    from ouroboros.subagent_history import execution_identity
    from ouroboros.subagent_runtime import select_subagent_snapshot, validate_subagent_snapshot

    snapshot, _legacy = select_subagent_snapshot(_rows(_session(processing=own)), subagent_id="session-builder",
                                                 work_tier=work_tier)
    assert _runs_on(snapshot) == expected
    # The row's identity (its history and statistics) stays the row's own; the work's tier rides beside it.
    row_only, _ = select_subagent_snapshot(_rows(_session(processing=own)), subagent_id="session-builder")
    assert execution_identity(snapshot) == execution_identity(row_only)
    # A started helper's snapshot is frozen: a later reading keeps what it captured, Ultrafast included.
    assert _runs_on(validate_subagent_snapshot(copy.deepcopy(snapshot))) == expected
    child = {"task_metadata": {"configured_subagent": snapshot, "delegation_role": "subagent",
                               "processing": {"source": "owner", "tier": "standard"}}}
    assert task_processing_preference(child, model_role="subagent:session-builder") == expected


@pytest.mark.parametrize("terminal", ["settled", "start_failed"])
def test_a_session_run_on_its_works_tier_is_filed_under_its_rows_own(tmp_path, terminal):
    """The run asks the work's tier; its receipt and measurements stay the row's (``row_processing``, as ``row_effort``
    keeps the row's pin): the row's own read finds the run, and its receipt never reads as "Earlier settings"."""
    from ouroboros import delegate_custody as custody
    from ouroboros import maintenance
    from ouroboros import subagent_metrics as metrics
    from ouroboros.subagent_history import (
        execution_identity,
        record_session_execution,
        session_request_facts,
        subagent_last_delegation,
    )
    from ouroboros.subagent_runtime import select_subagent_snapshot
    from ouroboros.task_results import write_task_result

    snapshot, _ = select_subagent_snapshot(_rows(_session()), subagent_id="session-builder", work_tier="fast")
    assert (_runs_on(snapshot), snapshot["processing_preference"]) == ("fast", "economy")
    identity = execution_identity(snapshot)
    route_id, model = identity["target_id"].split("=", 1)
    row = {"row_effort": snapshot["effort"], "row_processing": snapshot["processing_preference"]}
    if terminal == "settled":
        write_task_result(tmp_path, "root", "completed")
        record_session_execution(tmp_path, custody.RunCustody(
            task_id="root", run_id="run", route_id=route_id, model=model, selected_subagent_id="session-builder",
            profile_id=identity["credential_profile_id"], access=identity["access"], effort="high",
            processing_preference="fast", mode="execution", started_ts="2026-11-14T00:00:00Z", **row),
            {"summary": {"state": "succeeded", "answer_received": True, "startedAt": "2026-11-14T00:00:00Z",
                         "finishedAt": "2026-11-14T00:00:05Z", "durationSec": 5}}, {"model": model})
        maintenance.run_job(tmp_path, "subagent_metrics")
        assert metrics.read(tmp_path, "session-builder", row_identity=identity)["rows"], "the row's own read finds it"
        assert metrics.fact_line({"configured_subagent": snapshot}, drive_root=tmp_path)
    else:
        facts = session_request_facts({"model": model, "credentialProfileId": identity["credential_profile_id"],
                                       "effort": "high", "access": identity["access"]},
                                      selected_subagent_id="session-builder", task_id="root", route=route_id,
                                      processing={"requested": "fast"}, **row)
        assert facts["processing_preference"] == "fast", "the run's request stays a fact of the failure"
        assert custody.emit(tmp_path, custody.START_FAILED, {**facts, "invocation_id": "attempt", "definite": True,
                                                             "reason": "start_refused"})
    recorded = subagent_last_delegation(tmp_path)["identity"]
    assert {key: recorded.get(key) for key in identity} == identity


def _probe_start(tmp_path, monkeypatch, metadata, settings=None, active_model="", acting=None, **spec):
    """``exact_start`` up to the snapshot it would start with (the leaf-effort step is where the probe stops)."""
    import ouroboros.subagent_runtime as runtime
    from ouroboros.subagent_runtime import SubagentSelectionError
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setattr("ouroboros.config.load_settings", lambda: copy.deepcopy(settings or _rows(_session())))
    seen = []

    def stop(_ctx, snapshot, *_args, **_kwargs):
        seen.append(copy.deepcopy(snapshot))
        raise SubagentSelectionError("probe_stop", "The probe stops before any start.")
    monkeypatch.setattr(runtime, "_leaf_effort_choice", stop)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id, ctx.task_metadata = "t-a", copy.deepcopy(metadata)
    if active_model:
        ctx.active_model = active_model  # the work's own send binding, read for the tier it runs on
    for key, value in (acting or {}).items():  # the work's live call binding (a fallback row it runs on)
        setattr(ctx, key, copy.deepcopy(value))
    payload = json.loads(runtime.exact_start(ctx, "bounded leaf", {"subagent_id": "session-builder", **spec}).text)
    assert payload["reason"] == "probe_stop", payload
    return seen[-1]


@pytest.mark.parametrize("metadata,expected", [
    ({"processing": {"source": "owner", "tier": "fast"}}, "fast"),
    ({"processing": {"source": "room", "tier": "ultrafast"}}, "ultrafast"),
    ({"processing": {"source": "default"}}, "economy"),  # the decided Default: the old resolution
    ({}, "economy"),
    ({"processing": {"source": "owner", "tier": "fast"}, "delegation_role": "subagent"}, "economy"),  # never a child's
])
def test_a_fresh_delegated_start_captures_the_works_tier(tmp_path, monkeypatch, metadata, expected):
    # This inheritance matrix intentionally exercises an Economy default; pin it so the official Standard default
    # composed by the parent cannot silently change the fixture under test.
    monkeypatch.setenv(GLOBAL, "economy")
    assert _runs_on(_probe_start(tmp_path, monkeypatch, metadata)) == expected


def _start_actor(tmp_path, monkeypatch, metadata, settings=None, **spec) -> dict:
    """``exact_start`` through ``_delegate_start`` to the run's actor (``prepare_delegate_start_actor``, real), the one
    hop that carries the selection's tier into the run's request; the start stops right after it, before any engine."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.delegate_shared import _fail
    import ouroboros.subagent_runtime as runtime
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setattr("ouroboros.config.load_settings", lambda: copy.deepcopy(settings or _rows(_session())))
    real, seen = delegate.prepare_delegate_start_actor, []

    def prepared(*args, **kwargs):
        actor, refusal = real(*args, **kwargs)
        assert refusal is None, refusal
        seen.append(copy.deepcopy(actor))
        return actor, _fail("delegate_start", "probe_stop", "The probe stops before the engine is touched.", definitely_unrun=True)
    monkeypatch.setattr(delegate, "prepare_delegate_start_actor", prepared)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id, ctx.task_metadata = "t-a", copy.deepcopy(metadata)
    payload = json.loads(runtime.exact_start(ctx, "bounded leaf", {"subagent_id": "session-builder", **spec}).text)
    assert payload["reason"] == "probe_stop", payload
    return seen[-1]


@pytest.mark.parametrize("metadata,own,expected", [
    ({"processing": {"source": "owner", "tier": "fast"}}, None, "fast"),  # a row without a tier takes the work's
    ({"processing": {"source": "room", "tier": "ultrafast"}}, None, "ultrafast"),
    ({"processing": {"source": "owner", "tier": "fast"}}, "standard", "standard"),  # a row with its own keeps it
    ({}, None, "economy"),  # nothing carried: the default resolution
])
def test_a_fresh_delegated_start_asks_its_run_for_the_tier_it_selected(tmp_path, monkeypatch, metadata, own, expected):
    # This inheritance matrix intentionally exercises an Economy default; pin it so the official Standard default
    # composed by the parent cannot silently change the fixture under test.
    monkeypatch.setenv(GLOBAL, "economy")
    actor = _start_actor(tmp_path, monkeypatch, metadata, settings=_rows(_session(processing=own)))
    assert actor["selected_subagent_id"] == "session-builder" and actor["processing_preference"] == expected


MAIN_FAST = {ROLES: json.dumps({"main": "fast"})}
WORK_MODEL = "claudexor::codex=gpt-5.6-sol"
OWN_MODEL = {"lead": {"source": "room", "model": "claudexor::codex=gpt-6-astra", "use_local": False}}


@pytest.mark.parametrize("metadata,expected", [
    ({}, "fast"),  # a work that follows Main runs on Main's tier, a Fast saved before this version included
    ({"processing": {"source": "default"}}, "fast"),  # the decided Default is that same resolution
    ({"processing": {"source": "room", "tier": "economy"}}, "economy"),  # a carrier of the work's own wins
    (OWN_MODEL, "economy"),  # a work on its own model runs on the default tier, never Main's
])
def test_a_new_helper_takes_the_tier_its_work_runs_on_mains_included(monkeypatch, metadata, expected):
    """Owner 13.11, 17:22: a new helper without a tier of its own takes the tier its work runs on, Main's included."""
    from ouroboros.subagent_runtime import select_subagent_snapshot
    from ouroboros.supervisor_route import work_tier
    from tests.test_supervisor_route import parent

    settings = {**_rows(_session()), **MAIN_FAST}
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: copy.deepcopy(settings))
    monkeypatch.setenv(GLOBAL, "economy")
    monkeypatch.setenv(ROLES, MAIN_FAST[ROLES])  # Main's tier as the live settings read it
    work = parent(task_metadata=copy.deepcopy(metadata), active_model=WORK_MODEL)
    assert work_tier(work, settings) == expected
    assert task_processing_preference({"task_metadata": metadata}) == expected or metadata == OWN_MODEL, \
        "the helper takes what the work's own calls ask"
    snapshot, _ = select_subagent_snapshot(settings, subagent_id="session-builder", work_tier=work_tier(work, settings))
    assert _runs_on(snapshot) == expected
    # A tier equal to the row's own resolution rides as none: the snapshot stays the row's.
    assert "work_processing" not in snapshot if expected == snapshot["processing_preference"] else snapshot["work_processing"] == expected
    own_row, _ = select_subagent_snapshot({**settings, "OUROBOROS_SUBAGENTS": _rows(_session(processing="standard"))[
        "OUROBOROS_SUBAGENTS"]}, subagent_id="session-builder", work_tier=work_tier(work, settings))
    assert _runs_on(own_row) == "standard", "an Agents row with a tier of its own keeps it"
    # An unreadable binding inherits nothing beyond the work's own carrier (None: unknown) and never blocks a start.
    import ouroboros.supervisor_route as supervisor_route

    def unreadable(*_args):
        raise ValueError("the binding cannot be read")
    monkeypatch.setattr(supervisor_route, "_inherited_cognition", unreadable)
    assert work_tier(work, settings) == (route_choice.task_processing(metadata) or None)


@pytest.mark.parametrize("work,expected", [("ultrafast", "ultrafast"), ("fast", "fast"), ("", ""), (None, "economy")])
def test_the_base_agent_of_an_empty_catalog_takes_the_works_tier_like_every_row_without_one(work, expected):
    """A switched-on catalog without rows offers Main's model as its base agent; Main's own tier is no tier of that
    row's own, so a helper started from it takes its work's (Main's reaches it as the work's, for work following Main)."""
    from ouroboros.subagent_runtime import select_subagent_snapshot

    settings = {**_rows(), "OUROBOROS_MODEL": "openai/gpt-5.6-sol", "OPENROUTER_API_KEY": "sk-or-present",
                ROLES: json.dumps({"main": "standard"})}
    snapshot, _ = select_subagent_snapshot(settings, subagent_id="main", work_tier=work)
    assert (snapshot["selected_subagent_id"], _runs_on(snapshot)) == ("main", expected)


@pytest.mark.parametrize("metadata,expected", [({}, "fast"), (OWN_MODEL, "economy")])
def test_both_doors_start_a_helper_on_the_tier_its_work_runs_on(tmp_path, monkeypatch, metadata, expected):
    from ouroboros.subagent_runtime import SubagentSelectionError
    from ouroboros.tools import control_scheduling
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext

    settings = {**_rows(_session()), **MAIN_FAST}
    monkeypatch.setenv(GLOBAL, "economy")
    monkeypatch.setenv(ROLES, MAIN_FAST[ROLES])
    assert _runs_on(_probe_start(tmp_path, monkeypatch, metadata, settings=settings, active_model=WORK_MODEL)) == expected
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", settings["OUROBOROS_SUBAGENTS"])
    real, seen = control_scheduling.select_subagent_snapshot, []

    def select(*args, **kwargs):
        seen.append(_runs_on(real(*args, **kwargs)[0]))
        raise SubagentSelectionError("probe_stop", "The probe stops before any child is admitted.")
    monkeypatch.setattr(control_scheduling, "select_subagent_snapshot", select)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id, ctx.task_metadata, ctx.active_model = "t-a", copy.deepcopy(metadata), WORK_MODEL
    assert "probe_stop" in _schedule_task(ctx, subagent_id="session-builder", objective="Do the part", expected_output="Done")
    assert seen == [expected]


@pytest.mark.parametrize("work,own,expected", [
    (None, None, "fast"),  # nothing known of the work's tier: the row's own resolution (today's global) stands
    ("", None, ""),  # the work resolved to no tier: its new helper asks none, never today's global Fast
    ("standard", None, "standard"), ("fast", None, "fast"), ("economy", None, "economy"),
    ("ultrafast", None, "ultrafast"),
    ("", "standard", "standard"), (None, "economy", "economy"),  # a row's own tier keeps priority
])
def test_a_resolved_empty_work_tier_is_inherited_never_replaced_by_todays_global(work, own, expected):
    """``work_tier=None`` is "unknown", ``""`` a resolved tier of none (Astra rl-20261115T134200-6ffa9baa207d)."""
    from ouroboros.subagent_history import execution_identity
    from ouroboros.subagent_runtime import select_subagent_snapshot, validate_subagent_snapshot

    settings = {**_rows(_session(processing=own)), GLOBAL: "fast"}
    snapshot, _ = select_subagent_snapshot(settings, subagent_id="session-builder", work_tier=work)
    assert _runs_on(snapshot) == expected
    assert _runs_on(validate_subagent_snapshot(copy.deepcopy(snapshot))) == expected, "frozen as captured"
    omitted, _ = select_subagent_snapshot(settings, subagent_id="session-builder")
    assert execution_identity(snapshot) == execution_identity(omitted), "the row's identity stays its own"
    assert _runs_on(omitted) == (own or "fast"), "no argument: the row's own resolution, as before"


API_ROW = "api-builder"


def _api(processing=None):
    from tests.test_available_subagents_runtime import _api_row

    row = _api_row(API_ROW)
    if processing is not None:
        row["processing_preference"] = processing
    return row


def _scheduled(monkeypatch, tmp_path, metadata, row_id, settings):
    """``schedule_subagent`` up to the snapshot its child would be admitted with (the real selection's result)."""
    from ouroboros.subagent_runtime import SubagentSelectionError
    from ouroboros.tools import control, control_scheduling
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext

    # control binds load_settings at its first import (possibly another test's double): this door reads these.
    monkeypatch.setattr(control, "load_settings", lambda: copy.deepcopy(settings))
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", settings["OUROBOROS_SUBAGENTS"])
    real, seen = control_scheduling.select_subagent_snapshot, []

    def select(*args, **kwargs):
        seen.append((kwargs.get("work_tier"), real(*args, **kwargs)[0]))
        raise SubagentSelectionError("probe_stop", "The probe stops before any child is admitted.")
    monkeypatch.setattr(control_scheduling, "select_subagent_snapshot", select)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id, ctx.task_metadata = "t-a", copy.deepcopy(metadata)
    assert "probe_stop" in _schedule_task(ctx, subagent_id=row_id, objective="Do the part", expected_output="Done")
    monkeypatch.setattr(control_scheduling, "select_subagent_snapshot", real)
    [(work, snapshot)] = seen
    return work, snapshot


def _again(tmp_path, monkeypatch, original, name, metadata, settings) -> dict:
    """``_start_actor`` once more in one test: its own drive, and the real actor hop restored, not the last probe's."""
    import ouroboros.tools.delegate as delegate

    monkeypatch.setattr(delegate, "prepare_delegate_start_actor", original)
    drive = tmp_path / f"start-{name}"
    drive.mkdir()
    return _start_actor(drive, monkeypatch, metadata, settings=settings)


def _session_request(actor) -> tuple[dict, dict]:
    from ouroboros.tools.delegate import _processing_start_request

    gateway = SimpleNamespace(agent_capabilities=lambda: {"harnesses": [{"id": "codex", "processingPreferences": [
        "standard", "fast", "economy", "ultrafast"]}]})
    return _processing_start_request({}, actor, gateway, SimpleNamespace(route_id="codex"))


@pytest.mark.parametrize("captured,later", [
    ("", "fast"),  # Astra's case: an API helper captured no tier, then the owner made the global Fast
    ("standard", "fast"), ("economy", "fast"), ("ultrafast", "fast"), ("fast", "economy"),
])
def test_two_generations_of_api_helpers_keep_the_tier_captured_before_the_global_changed(
        tmp_path, monkeypatch, captured, later):
    """Both doors, two API generations, to the request each new helper would make; a row's own tier wins."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.subagent_runtime import select_subagent_snapshot
    from ouroboros.supervisor_route import work_tier
    from tests.test_supervisor_route import parent

    original = delegate.prepare_delegate_start_actor

    def settings_at(global_tier, own=None):
        return {"OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [
            _session(processing=own), _api(processing=own)]}), GLOBAL: global_tier}

    monkeypatch.delenv(ROLES, raising=False)
    monkeypatch.setenv(GLOBAL, "")
    root = parent(task_metadata={"processing": {"source": "owner", "tier": captured}} if captured else {},
                  active_model=WORK_MODEL)
    assert work_tier(root, settings_at("")) == captured
    first, _ = select_subagent_snapshot(settings_at(""), subagent_id=API_ROW, work_tier=work_tier(root, settings_at("")))
    assert _runs_on(first) == captured
    # The owner changes the global default; the started API helper keeps asking what it captured.
    monkeypatch.setenv(GLOBAL, later)
    settings = settings_at(later)
    generation = {"configured_subagent": first, "delegation_role": "subagent"}
    assert task_processing_preference({"task_metadata": generation}, model_role=f"subagent:{API_ROW}") == captured
    for depth in (2, 3):
        # delegate_start: the session run's actor and the request it would send to the engine (one drive per start).
        actor = _again(tmp_path, monkeypatch, original, depth, generation, settings)
        request, receipt = _session_request(actor)
        assert actor["processing_preference"] == captured and receipt["requested"] == captured
        assert request.get("processingPreference") == (captured or None)
        # schedule_subagent: the next API generation and what its own calls ask.
        work, child = _scheduled(monkeypatch, tmp_path, generation, API_ROW, settings)
        assert work == captured and _runs_on(child) == captured
        generation = {"configured_subagent": child, "delegation_role": "subagent"}
        assert task_processing_preference({"task_metadata": generation}, model_role=f"subagent:{API_ROW}") == captured
    # A row's own tier keeps priority at both doors.
    own = "standard" if captured != "standard" else "economy"
    owned = settings_at(later, own=own)
    assert _again(tmp_path, monkeypatch, original, "own", generation, owned)["processing_preference"] == own
    assert _runs_on(_scheduled(monkeypatch, tmp_path, generation, API_ROW, owned)[1]) == own


def test_an_unreadable_work_binding_keeps_the_rows_resolution_and_never_blocks_a_start(tmp_path, monkeypatch):
    """Unknown is not a captured empty tier: both doors start, on the row's own resolution (today's global)."""
    import ouroboros.supervisor_route as supervisor_route
    from ouroboros.subagent_runtime import select_subagent_snapshot
    from ouroboros.supervisor_route import work_tier

    settings = {"OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [_session(), _api()]}), GLOBAL: "fast"}
    monkeypatch.setenv(GLOBAL, "fast")
    captured_empty, _ = select_subagent_snapshot({**settings, GLOBAL: ""}, subagent_id=API_ROW, work_tier="")
    generation = {"configured_subagent": captured_empty, "delegation_role": "subagent"}

    def unreadable(*_args):
        raise ValueError("the binding cannot be read")
    monkeypatch.setattr(supervisor_route, "_inherited_cognition", unreadable)
    assert work_tier(SimpleNamespace(task_metadata=generation), settings) is None
    assert _start_actor(tmp_path, monkeypatch, generation, settings=settings)["processing_preference"] == "fast"
    work, child = _scheduled(monkeypatch, tmp_path, generation, API_ROW, settings)
    assert work is None and _runs_on(child) == "fast" and "work_processing" not in child
    # A root's own carrier needs no binding read and still answers.
    assert work_tier(SimpleNamespace(task_metadata={"processing": {"source": "owner", "tier": "economy"}}), settings) == "economy"


def _catalog(global_tier, own=None, *extra) -> dict:
    """The owner's Agents rows (a session and an API row, a tier of their own or none) and global tier."""
    return {"OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [
        _session(processing=own), _api(processing=own), *extra]}), GLOBAL: global_tier}


def _engine(monkeypatch) -> list:
    """The serving engine of a session start (``_LiveRunStub``, tests._delegated_transport_shared), advertising every
    tier and ``continueFrom`` for the rows' ``codex`` route; returns the start requests it receives, as JSON."""
    from ouroboros import claudexor_daemon
    from ouroboros.gateways import claudexor as gateway
    from tests._delegated_transport_shared import _LiveRunStub

    requests = []

    class Engine(_LiveRunStub):
        def agent_capabilities(self, **_kw):
            return {"runControlKeys": ["continueFrom"], "harnesses": [{
                "id": "codex", "enabled": True, "status": "ok", "accessProfilesSupported": ["readonly"],
                "processingPreferences": ["standard", "fast", "economy", "ultrafast"]}]}

        def start_run(self, request, *, idempotency_key=""):
            requests.append(json.loads(json.dumps(request)))
            return {"runId": f"run-{len(requests)}"}

    monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *_a, **_k: Engine())
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda **_kw: Engine())
    return requests


def _owner_saves(monkeypatch, drive, settings):
    """The owner's settings as both doors read them now (the saved document and control's reader: ``root_context``,
    tests.test_supervisor_route_flow; the config reader and the environment); returns a root work's context."""
    from tests.test_supervisor_route_flow import root_context

    monkeypatch.setattr("ouroboros.config.load_settings", lambda: copy.deepcopy(settings))
    for key in ("OUROBOROS_SUBAGENTS", GLOBAL):
        monkeypatch.setenv(key, settings[key])
    return root_context(drive, settings, monkeypatch)


def _started(drive, work, **spec) -> dict:
    """``delegate_start`` by its tool entry (``exact_start``: the real selection, actor and start) from a work, to the
    start's answer; the read-only nanny is ``_plain_ctx`` (tests._delegated_transport_shared)."""
    import ouroboros.subagent_runtime as runtime
    import ouroboros.tools.delegate as delegate
    from tests._delegated_transport_shared import _plain_ctx

    ctx = _plain_ctx(drive)
    ctx.task_metadata = {**copy.deepcopy(work), **ctx.task_metadata}
    delegate._CUSTODY.clear()
    payload = json.loads(runtime.exact_start(ctx, "Inspect the fixture.", {"subagent_id": "session-builder", **spec}).text)
    delegate._CUSTODY.clear()
    assert payload["status"] == "started", payload
    return payload


def _submitted(ctx, task_id, work, **spec) -> dict:
    """``schedule_subagent`` from a work to the request it submits: the event the supervisor's live queue receives,
    through the pickling a process queue applies; the durable requested row holds the same frozen row."""
    import pickle
    import queue
    from multiprocessing.reduction import ForkingPickler

    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control_scheduling import _schedule_task

    ctx.task_id, ctx.task_metadata = task_id, {**copy.deepcopy(work), "root_task_id": "root"}
    ctx.event_queue = queue.Queue()
    answer = _schedule_task(ctx, objective="Inspect the assigned files", expected_output="Findings",
                            memory_mode="empty", **spec)
    assert answer.startswith("Subagent request queued"), answer
    event = pickle.loads(ForkingPickler.dumps(ctx.event_queue.get_nowait()))
    assert ctx.event_queue.empty() and event["type"] == "schedule_subagent"
    assert load_task_result(ctx.drive_root, event["task_id"])["configured_subagent"] == event["configured_subagent"]
    return event


def _child(event) -> dict:
    """What the submitted child's own calls read: its frozen row and its role."""
    return {key: event[key] for key in ("configured_subagent", "delegation_role")}


@pytest.mark.parametrize("main,own,expected", [
    ("", None, ""),  # a helper that captured no tier hands none down
    ("", "standard", "standard"),
    ("ultrafast", None, "ultrafast"),  # Main changes after admission: the helper hands down its captured tier
    ("ultrafast", "standard", "standard"),  # rows with a tier of their own keep it
])
def test_both_doors_send_their_real_requests_on_the_tier_captured_before_the_global_became_fast(
        tmp_path, monkeypatch, main, own, expected):
    """An API helper keeps what it took from Main after its owner changes Settings. Each door,
    from that helper and from the API child it schedules, is read where its request leaves the host: delegate_start
    in the engine's start request, schedule_subagent in the event submitted to the supervisor; continuing the last
    run keeps the tier it was captured with."""
    from ouroboros import delegate_custody as custody
    from tests.test_continuation_access_inheritance import _settle

    requests = _engine(monkeypatch)
    monkeypatch.delenv(ROLES, raising=False)
    initial = {**_catalog(""), ROLES: json.dumps({"main": main})}
    first = _submitted(_owner_saves(monkeypatch, tmp_path, initial), "root", {}, subagent_id=API_ROW)
    assert _runs_on(first["configured_subagent"]) == main
    assert first["configured_subagent"]["processing_preference"] == ""
    ctx = _owner_saves(monkeypatch, tmp_path, {**_catalog("fast", own), ROLES: "{}"})
    assert task_processing_preference({}) == "fast"
    seen, work, task_id = {}, _child(first), first["task_id"]
    for generation in (1, 2):
        run_work, started = work, _started(tmp_path, work, access="readonly")
        seen[f"delegate_start {generation}"] = requests[-1].get("processingPreference", "")
        seen[f"its run's custody {generation}"] = custody.replay(tmp_path)[started["run_id"]].processing_preference
        event = _submitted(ctx, task_id, work, subagent_id=API_ROW)
        seen[f"schedule_subagent {generation}"] = _runs_on(event["configured_subagent"])
        work, task_id = _child(event), event["task_id"]
        seen[f"its child's calls {generation}"] = task_processing_preference(
            {"task_metadata": work}, model_role=f"subagent:{API_ROW}")
    _settle(tmp_path, started["run_id"])
    continued = _started(tmp_path, run_work, continue_from=started["run_id"])
    assert requests[-1]["continueFrom"] == started["run_id"] != continued["run_id"]
    seen["continuation"] = requests[-1].get("processingPreference", "")
    assert seen == dict.fromkeys(seen, expected)


def test_an_unreadable_work_binding_reaches_both_real_requests_on_the_rows_resolution(tmp_path, monkeypatch):
    """Unknown is not a captured empty tier at the end of either door: the helper's start asks the row's resolution
    (today's Fast), and its session child with a named supervisor row is submitted on both rows' resolution."""
    import ouroboros.supervisor_route as supervisor_route
    from tests.test_supervisor_route import cognition_row

    requests = _engine(monkeypatch)
    monkeypatch.delenv(ROLES, raising=False)
    first = _submitted(_owner_saves(monkeypatch, tmp_path, _catalog("")), "root", {}, subagent_id=API_ROW)
    assert _runs_on(first["configured_subagent"]) == ""
    supervisor = cognition_row()
    supervisor.pop("processing_preference")  # a supervisor row without a tier of its own
    ctx = _owner_saves(monkeypatch, tmp_path, _catalog("fast", None, supervisor))

    def unreadable(*_args):
        raise ValueError("the binding cannot be read")
    monkeypatch.setattr(supervisor_route, "_inherited_cognition", unreadable)
    _started(tmp_path, _child(first), access="readonly")
    event = _submitted(ctx, first["task_id"], _child(first), subagent_id="session-builder", supervisor="cognition")
    selected = event["parent_cognitive_route"]["selected_supervisor_snapshot"]
    seen = {"delegate_start": requests[-1].get("processingPreference", ""),
            "schedule_subagent leaf": _runs_on(event["configured_subagent"]),
            "its supervisor": event["parent_cognitive_route"]["processing_preference"]}
    assert seen == dict.fromkeys(seen, "fast")
    assert "work_processing" not in event["configured_subagent"] and "work_processing" not in selected


ON_FALLBACK = {"context_fit_plan": SimpleNamespace(model_role="fallback:0"), "route_choice": route_choice.explicit("fallback"),
               "primary_route": {"model": WORK_MODEL, "use_local": False, "role": "main"},
               "active_model": "claudexor::codex=gpt-6-astra"}


@pytest.mark.parametrize("metadata,own,expected", [
    ({}, None, "fast"),  # a Default work on fallback:0 Fast runs Fast: its new helper does too
    ({"processing": {"source": "default"}}, None, "fast"),
    ({"processing": {"source": "owner", "tier": "standard"}}, None, "standard"),  # the work's explicit Standard wins
    ({}, "economy", "economy"),  # a row with a tier of its own keeps it
])
def test_a_work_on_its_fallback_starts_a_helper_on_the_tier_it_actually_runs(tmp_path, monkeypatch, metadata, own, expected):
    """The helper takes the tier of the call the work is making (fallback:0 Fast with Main Standard); the nanny's
    model inheritance stays the primary's, and a work back on its primary hands Main's tier as before."""
    from ouroboros.subagent_runtime import SubagentSelectionError
    from ouroboros.supervisor_route import _inherited_cognition, work_tier
    from ouroboros.tools import control_scheduling
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext

    tiers = json.dumps({"main": "standard", "fallback": ["fast"]})
    settings = {**_rows(_session(processing=own)), ROLES: tiers}
    monkeypatch.setenv(GLOBAL, "economy")
    monkeypatch.setenv(ROLES, tiers)
    monkeypatch.setenv("OUROBOROS_MODEL", WORK_MODEL)
    assert _runs_on(_probe_start(tmp_path, monkeypatch, metadata, settings=settings, acting=ON_FALLBACK)) == expected
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", settings["OUROBOROS_SUBAGENTS"])
    real, seen = control_scheduling.select_subagent_snapshot, []

    def select(*args, **kwargs):
        seen.append(_runs_on(real(*args, **kwargs)[0]))
        raise SubagentSelectionError("probe_stop", "The probe stops before any child is admitted.")
    monkeypatch.setattr(control_scheduling, "select_subagent_snapshot", select)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id, ctx.task_metadata = "t-a", copy.deepcopy(metadata)
    for key, value in ON_FALLBACK.items():
        setattr(ctx, key, copy.deepcopy(value))
    assert "probe_stop" in _schedule_task(ctx, subagent_id="session-builder", objective="Do the part", expected_output="Done")
    assert seen == [expected]
    # The nanny's model still inherits the primary (Main on Standard), not the fallback route or tier.
    nanny = _inherited_cognition(ctx, settings)
    assert (nanny["model"], nanny["source_role"], nanny["processing_preference"]) == (WORK_MODEL, "main", "standard")
    # Negative control: the same work back on its primary hands Main's Standard, not the fallback's Fast.
    ctx.context_fit_plan, ctx.route_choice = SimpleNamespace(model_role="main"), route_choice.following("main")
    assert work_tier(ctx, settings) == (route_choice.task_processing(metadata) or "standard")


@pytest.mark.parametrize("captured", ["standard", "", "ultrafast"])
def test_a_continuation_keeps_the_tier_its_run_was_captured_with(tmp_path, monkeypatch, captured):
    from ouroboros import delegate_continuation as continuation
    from ouroboros.tools.delegate import _processing_start_request
    from tests.test_delegate_continuation import ACTOR, _seed

    _seed(tmp_path, "run-captured", actor=ACTOR, processing_preference=captured)
    assert continuation.inherited_processing(SimpleNamespace(drive_root=tmp_path, task_id="t-a"), "run-captured") == captured
    snapshot = _probe_start(tmp_path, monkeypatch, {"processing": {"source": "owner", "tier": "fast"}},
                            continue_from="run-captured")
    # The run's captured tier, never the work's current one nor today's row or global one (Economy here): a run that
    # asked for no tier keeps asking for none.
    assert snapshot["work_processing"] == captured and _runs_on(snapshot) == captured
    gateway = SimpleNamespace(agent_capabilities=lambda: {"harnesses": [{"id": "codex", "processingPreferences": [
        "standard", "fast", "economy", "ultrafast"]}]})
    request, receipt = _processing_start_request({}, {"processing_preference": _runs_on(snapshot)}, gateway,
                                                 SimpleNamespace(route_id="codex"))
    assert request.get("processingPreference") == (captured or None)
    assert receipt["requested"] == captured


def test_a_continuation_without_a_captured_tier_takes_the_works(tmp_path, monkeypatch):
    from ouroboros import delegate_continuation as continuation
    from tests.test_delegate_continuation import ACTOR, _seed

    _seed(tmp_path, "run-legacy", actor=ACTOR)  # recorded before tiers were captured
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="t-a")
    assert continuation.inherited_processing(ctx, "run-legacy") is None
    assert continuation.inherited_processing(ctx, "run-unknown") is None
    snapshot = _probe_start(tmp_path, monkeypatch, {"processing": {"source": "owner", "tier": "fast"}},
                            continue_from="run-legacy")
    assert _runs_on(snapshot) == "fast"


def test_a_retry_replays_without_consulting_the_continued_runs_tier(tmp_path, monkeypatch):
    import ouroboros.subagent_runtime as runtime
    from ouroboros import delegate_continuation as continuation
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setattr(continuation, "inherited_processing",
                        lambda *_a, **_k: pytest.fail("a retry replays its recorded body"))
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "t-a"
    payload = json.loads(runtime.exact_start(ctx, "", {"retry_of": "no-such-invocation"}).text)
    assert payload["status"] == "refused" and payload["reason"] == "unknown_invocation"


@pytest.mark.parametrize("metadata,expected", [
    ({"processing": {"source": "owner", "tier": "ultrafast"}}, "ultrafast"),
    ({"processing": {"source": "default"}}, "economy"),
    ({"processing": {"source": "room", "tier": "fast"}, "delegation_role": "subagent"}, "economy"),
])
def test_a_scheduled_helper_captures_the_works_tier(tmp_path, monkeypatch, metadata, expected):
    from ouroboros.subagent_runtime import SubagentSelectionError
    from ouroboros.tools import control_scheduling
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_SUBAGENTS", _rows(_session())["OUROBOROS_SUBAGENTS"])
    monkeypatch.setenv(GLOBAL, "economy")
    real, seen = control_scheduling.select_subagent_snapshot, []

    def select(*args, **kwargs):
        seen.append((kwargs.get("work_tier"), _runs_on(real(*args, **kwargs)[0])))
        raise SubagentSelectionError("probe_stop", "The probe stops before any child is admitted.")
    monkeypatch.setattr(control_scheduling, "select_subagent_snapshot", select)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id, ctx.task_metadata = "t-a", copy.deepcopy(metadata)
    answer = _schedule_task(ctx, subagent_id="session-builder", objective="Do the part", expected_output="Done")
    assert "probe_stop" in answer
    assert seen == [(expected, expected)], "the work's tier is what its own calls ask: its carrier, else the default"


@pytest.mark.parametrize("carrier,own,expected", [
    ({"source": "owner", "tier": "fast"}, None, "fast"),  # a named row without a tier of its own takes the work's
    (ULTRAFAST, None, "ultrafast"),
    ({"source": "owner", "tier": "fast"}, "standard", "standard"),  # a row with one keeps it
    (None, None, "economy"),  # without a carrier: what the work's calls ask by the default resolution
])
def test_a_named_supervisor_row_takes_the_works_tier_unless_it_has_its_own(monkeypatch, carrier, own, expected):
    from ouroboros.subagent_history import execution_identity
    from ouroboros.supervisor_route import capture_cognitive_route
    from tests.test_available_subagents_runtime import _settings, _snapshot
    from tests.test_supervisor_route import cognition_row, parent

    monkeypatch.setenv(GLOBAL, "economy")
    row = cognition_row()
    row.pop("processing_preference")
    if own:
        row["processing_preference"] = own
    settings = _settings(_session(), row)
    leaf = _snapshot(settings, "session-builder")
    work = parent(task_metadata={"processing": carrier} if carrier else {})
    route = capture_cognitive_route(work, leaf, settings, supervisor="cognition")
    assert route["chosen_by"] == "delegator" and route["processing_preference"] == expected
    assert task_processing_preference({"configured_subagent": leaf, "parent_cognitive_route": route}) == expected
    # The row's own stays its identity (history, statistics); the work's tier rides beside it.
    selected = route["selected_supervisor_snapshot"]
    assert execution_identity(selected) == execution_identity(_snapshot(settings, "cognition"))


@pytest.mark.parametrize("carrier,expected", [(ULTRAFAST, "ultrafast"), ({"source": "room", "tier": "fast"}, "fast")])
def test_a_nannys_inherited_cognition_takes_the_parents_carried_tier(monkeypatch, carrier, expected):
    from ouroboros.supervisor_route import capture_cognitive_route
    from tests.test_available_subagents_runtime import _settings, _snapshot
    from tests.test_supervisor_route import parent

    monkeypatch.setenv(GLOBAL, "economy")
    settings = _settings(_session())
    leaf = _snapshot(settings, "session-builder")
    route = capture_cognitive_route(parent(task_metadata={"processing": carrier}), leaf, settings)
    assert route["processing_preference"] == expected
    assert task_processing_preference({"configured_subagent": leaf, "parent_cognitive_route": route}) == expected
