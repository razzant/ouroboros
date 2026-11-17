"""Known 80k routes budget both the request and its reply, including unbound helpers."""
import json
import math
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce, context_fit, usage_accounting as ua
from ouroboros.llm import LLMClient
from tests._first_contact import ANSWER, _episode, _run
from tests.system_e2e.harness import MOCK_SLUG, ScriptedStubModel, StubHttpError

WINDOW = 80_000


@pytest.mark.parametrize("catalog,refusal,expected_reply", [(20000, 0, 65536), (80000, 0, 35000),
    (0, 80000, 35000), (120000, 80000, 35000)])
def test_helper_acceptance_is_not_total_capacity(tmp_path, monkeypatch, catalog, refusal, expected_reply):
    from ouroboros import context_window
    from ouroboros.response_limits import dispatch_target
    from ouroboros.config import runtime_settings
    with ScriptedStubModel(final_answer=ANSWER) as server:
        episode = _episode(tmp_path, monkeypatch, server)
        target = dispatch_target(MOCK_SLUG, runtime_settings())
        route = dict(provider="openai-compatible", model=MOCK_SLUG, base_url=server.base_url,
                     options={}, use_local=False)
        scope = context_window.route_observation_scope(route, runtime_settings())
        evidence = ce.CapabilityEvidence(catalog, "confirmed", ce.SOURCE_PROVIDER_METADATA,
            episode.plan.route_fp, MOCK_SLUG, "openai-compatible", ts=ce.utc_now_iso())
        monkeypatch.setattr(ce, "probe", lambda *a, **kw: evidence)
        context_window.record_observation(episode.drive, scope, accepted=40000)
        if refusal:
            context_window.record_observation(episode.drive, scope, refused=refusal, kind="input")
        # Pin only the estimator for the review's exact 35k case; dispatch/SDK/HTTP stay real.
        monkeypatch.setattr(context_fit, "bounded_prompt_tokens_for_payload", lambda *a, **kw: 35000)
        sent = []
        original = server._completion
        def completion(body):
            sent.append(body)
            return original(body)
        server._completion = completion
        with ua.usage_scope(ua.UsageScope(drive_root=episode.drive, task_id="helper", root_task_id="helper")):
            result, _ = LLMClient(api_key="unused-keyless").chat(
                [{"role": "user", "content": "a" * 140000}], MOCK_SLUG, max_tokens=65536)
        assert target["provider"] == "openai-compatible" and ANSWER in str(result)
        assert len(sent) == 1 and sent[0]["max_tokens"] == expected_reply


class StrictWindowServer(ScriptedStubModel):
    """Real HTTP/SDK boundary; token usage is denser than the host's chars/4 proxy."""
    def __init__(self, *, chars_per_token=3.5):
        super().__init__(final_answer=ANSWER)
        self.window_tokens = WINDOW
        self.chars_per_token = chars_per_token
        self.posts = []

    def _completion(self, body):
        chars = len(json.dumps({"messages": body.get("messages"), "tools": body.get("tools")}, ensure_ascii=False))
        count = math.ceil(chars / self.chars_per_token)
        reply = body.get("max_tokens") or body.get("max_completion_tokens")
        self.posts.append({"input": count, "reply": reply})
        if count + reply > WINDOW:
            raise StubHttpError(400, f"input length and max_tokens exceed context limit: {count} + {reply} > {WINDOW}")
        result = super()._completion(body)
        result["usage"] = {"prompt_tokens": count, "completion_tokens": 5, "total_tokens": count + 5}
        return result


@pytest.mark.parametrize("role", ["main", "consciousness", "subagent:fixture"])
def test_each_agent_role_sends_input_and_reply_within_80k(tmp_path, monkeypatch, role):
    with StrictWindowServer() as server:
        episode = _episode(tmp_path, monkeypatch, server)
        episode.task["model_role"] = role
        if role == "consciousness":
            episode.task["usage_category"] = "consciousness"
        if role.startswith("subagent:"):
            episode.task["delegation_role"] = "subagent"
        episode.plan = context_fit.build_context_fit_plan(
            SimpleNamespace(repo_dir=episode.tools._ctx.repo_dir, drive_root=episode.drive),
            episode.plan.core, episode.task, preferred_mode="max", route_resolver=context_fit.resolve_context_fit_route,
            tool_schemas=json.loads(episode.plan.start_tool_schemas_json))
        episode.tools._ctx.context_fit_plan = episode.plan
        text, _usage, _trace = _run(episode)
    assert ANSWER in str(text)
    assert episode.plan.view_kind == {"main": "owner_turn", "consciousness": "own_wake", "subagent:fixture": "child"}[role]
    assert server.posts and all(post["input"] + post["reply"] <= WINDOW for post in server.posts)


def test_unbound_helper_uses_known_window_and_its_own_accepted_density(tmp_path, monkeypatch):
    with StrictWindowServer(chars_per_token=2.5) as server:
        episode = _episode(tmp_path, monkeypatch, server)  # reads metadata through the normal start
        before = ce._read(ce.canonical_evidence_root()).get("token_density", {})
        assert not any(pair.get("view_kind") == "helper" for value in before.values() for pair in value.get("pairs", []))
        import httpx
        fetched = []

        def recorded_get(url, *_args, **_kwargs):  # the metadata readers' own client; every read fails busy
            fetched.append(url)
            raise httpx.ConnectError("busy", request=httpx.Request("GET", url))

        monkeypatch.setattr("httpx.get", recorded_get)
        ce.probe(ce.canonical_evidence_root(), provider="openai-compatible", model=MOCK_SLUG,
                 base_url=server.base_url, force=True)
        assert fetched, "a forced start read reaches the metadata client: the recorder is live"
        fetched.clear()
        client = LLMClient(api_key="unused-keyless")
        scope = ua.UsageScope(drive_root=episode.drive, task_id="helper", root_task_id="helper", category="task")
        with ua.usage_scope(scope):
            first, _ = client.chat([{"role": "user", "content": "a" * 40_000}], MOCK_SLUG, max_tokens=65_536)
            assert ANSWER in str(first)
            density, basis = ce.resolve_main_token_density(ce.canonical_evidence_root(), episode.plan.route_fp,
                                                           MOCK_SLUG, estimate_tokens=18_100, view_kind="helper")
            assert basis == "fresh_route_usage" and density > 1.5
            # A helper observation cannot calibrate a consciousness or ordinary memory view.
            assert ce.resolve_main_token_density(ce.canonical_evidence_root(), episode.plan.route_fp,
                                                 MOCK_SLUG, estimate_tokens=18_100, view_kind="own_wake")[1] == "cold_estimate"
            second, _ = client.chat([{"role": "user", "content": "a" * 72_000}], MOCK_SLUG, max_tokens=65_536)
            assert ANSWER in str(second)
    assert not fetched, "helper finalization must not fetch metadata"
    assert len(server.posts) == 2
    assert all(post["input"] + post["reply"] <= WINDOW for post in server.posts)
    assert server.posts[1]["reply"] < WINDOW - 18_000 - 10_000  # calibrated, not the cold host estimate



def test_a_helper_attempt_builds_its_window_context_once(tmp_path, monkeypatch):
    from ouroboros import llm_attempt, llm_helper_window

    built, original = [], llm_helper_window._helper_attempt_context

    def counted(target, payload):
        built.append(payload.get("max_tokens"))
        return original(target, payload)

    with StrictWindowServer(chars_per_token=2.5) as server:
        episode = _episode(tmp_path, monkeypatch, server)
        monkeypatch.setattr(llm_attempt, "_helper_attempt_context", counted)
        monkeypatch.setattr(llm_helper_window, "_helper_attempt_context", counted)
        scope = ua.UsageScope(drive_root=episode.drive, task_id="helper", root_task_id="helper", category="task")
        with ua.usage_scope(scope):
            text, _ = LLMClient(api_key="unused-keyless").chat(
                [{"role": "user", "content": "a" * 40_000}], MOCK_SLUG, max_tokens=65_536)
    assert ANSWER in str(text)
    assert len(server.posts) == 1 and len(built) == 1, built  # priced once; its request reuses that context


def test_a_main_attempt_leaves_no_priced_candidate_behind(monkeypatch):
    """Main brings its own context: nothing is bound for its candidate, before or after its request is built."""
    from ouroboros import llm_attempt, llm_helper_window
    from ouroboros.request_wire_recovery import request_wire_call_scope

    physical = ua.PhysicalAttemptContext(profile="owner_max", rendered_mode="max", measurement_basis="cold_estimate",
        route_fp="route", round_id="round", target_total_tokens=None, capacity_total_tokens=1_000_000,
        context_target_miss=False, automatic_pass_used=False)
    target = {"provider": "openai-compatible", "resolved_model": "fixture", "base_url": "http://127.0.0.1:1/v1"}
    monkeypatch.setattr("ouroboros.response_limits.response_limit_for_target",
                        lambda *_a, **_kw: SimpleNamespace(ceiling=lambda value: value))
    llm_helper_window._PRICED.set((None, None))
    with request_wire_call_scope(), ua.bind_physical_attempt_context(physical):
        candidate = llm_attempt._finalized_physical_candidate(
            target, {"model": "fixture", "messages": [{"role": "user", "content": "x" * 4096}], "max_tokens": 1000},
            "chat.completions")
        assert llm_helper_window._PRICED.get() == (None, None)
        assert llm_attempt._attempt_request(target, candidate).physical_context is physical
    assert llm_helper_window._PRICED.get() == (None, None)

def test_final_admission_charges_the_wire_reply_not_only_the_nano_floor():
    from ouroboros.llm_attempt import main_context_fit_refusal
    physical = ua.PhysicalAttemptContext(profile="task_local_nano", rendered_mode="nano",
        measurement_basis="fresh_route_usage", route_fp="route", round_id="round", target_total_tokens=None,
        capacity_total_tokens=WINDOW, context_target_miss=False, automatic_pass_used=False,
        measurement_density=1.5, response_reserve_tokens=8192)
    request = ua.AttemptRequest(model=MOCK_SLUG, provider="openai-compatible", physical_context=physical,
                                prompt_tokens_bounded_estimate=40_000, max_completion_tokens=25_000,
                                candidate_measurement_kind="canonical_json_v1")
    refused = main_context_fit_refusal(request)
    assert refused is not None and refused["predicted_free_tokens"] == -5_000
    from dataclasses import replace
    assert main_context_fit_refusal(replace(request, candidate_measurement_kind="opaque")) is None
    # Native subscriptions seal canonical JSON too; their logical ceiling is not sent as max_tokens.
    assert main_context_fit_refusal(replace(request, provider="claudexor")) is None


def test_unbound_helper_sends_intact_input_for_the_servers_typed_refusal(tmp_path, monkeypatch):
    from openai import BadRequestError
    from ouroboros.loop_llm_call import classify_llm_exception
    with StrictWindowServer() as server:
        episode = _episode(tmp_path, monkeypatch, server)
        client = LLMClient(api_key="unused-keyless")
        original = [{"role": "user", "content": "keep this whole source " * 16_000}]
        snapshot = json.dumps(original)
        with ua.usage_scope(ua.UsageScope(drive_root=episode.drive, task_id="large-helper", root_task_id="large-helper")):
            with pytest.raises(BadRequestError) as caught:
                client.chat(original, MOCK_SLUG, max_tokens=65_536)
        assert len(server.posts) == 1
        assert json.dumps(original) == snapshot
        assert classify_llm_exception(caught.value).kind == "context_overflow"


def test_current_local_serving_window_outranks_older_helper_acceptance(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from ouroboros import config, context_window
    from ouroboros.llm_attempt import bound_reply_allowance
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(context_window, "route_observation_scope", lambda *_, **kw: "local-helper")
    with monkeypatch.context() as older:
        older.setattr(ce, "utc_now_iso", lambda: (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat())
        context_window.record_observation(tmp_path, "local-helper", accepted=100_000)
    payload = {"messages": [{"role": "user", "content": "a" * 160_000}], "max_tokens": 65_536}
    candidate = bound_reply_allowance({"provider": "local", "resolved_model": "local-model",
        "context_window_confirmed": True, "context_window_tokens": WINDOW}, payload)
    raw = context_fit.bounded_prompt_tokens_for_payload(payload, 0)
    assert candidate["max_tokens"] == WINDOW - raw - 10_000


def test_child_light_and_safety_share_the_parents_learned_window(tmp_path, monkeypatch):
    from ouroboros import consolidator, safety
    with StrictWindowServer() as server:
        server.window_tokens = 0  # parent must learn capacity from the one real SDK refusal
        episode = _episode(tmp_path, monkeypatch, server)
        text, _, _ = _run(episode)
        assert ANSWER in str(text)
        assert sum(p["input"] + p["reply"] > WINDOW for p in server.posts) == 1
        parent_posts = len(server.posts)
        # Same connection, new child's task/plan: it inherits the route receipt, not a parent's plan.
        episode.task.update(id="child", parent_task_id="first-contact", root_task_id="first-contact",
                            delegation_role="subagent", model_role="subagent:fixture")
        episode.tools._ctx.task_id = "child"
        episode.plan = context_fit.build_context_fit_plan(
            SimpleNamespace(repo_dir=episode.tools._ctx.repo_dir, drive_root=episode.drive),
            episode.plan.core, episode.task, preferred_mode="max", route_resolver=context_fit.resolve_context_fit_route,
            tool_schemas=json.loads(episode.plan.start_tool_schemas_json))
        episode.tools._ctx.context_fit_plan = episode.plan
        assert episode.plan.initial_mode == "nano" and episode.plan.view_kind == "child"
        text, _, _ = _run(episode)
        assert ANSWER in str(text) and len(server.posts) > parent_posts
        client = LLMClient(api_key="unused-keyless")
        before_helpers = len(server.posts)
        with ua.usage_scope(ua.UsageScope(drive_root=episode.drive, task_id="child", root_task_id="first-contact")):
            text, _ = consolidator._call_consolidation_llm(client, "a" * 120000, "window inheritance")
            assert ANSWER in text
            ok, detail = safety._run_llm_check("fixture_tool", {"text": "a" * 120000}, [], episode.tools._ctx)
            assert ok, detail
        assert all(p["reply"] < 65536 for p in server.posts[before_helpers:])
        assert len(server.posts) == before_helpers + 2  # both actual helper transports sent
        later = server.posts[parent_posts:]
        assert later and all(p["input"] + p["reply"] <= WINDOW for p in later), later


@pytest.mark.parametrize("model", ["openrouter@work::fixture", "openai::fixture"])
def test_helper_reads_the_canonical_capability_and_selected_account(tmp_path, monkeypatch, model):
    from ouroboros import config, context_window
    from ouroboros.llm_attempt import bound_reply_allowance
    from ouroboros.provider_models import capability_base_url, CONNECTIONS_KEY
    from ouroboros.response_limits import dispatch_target
    settings = {"OPENROUTER_API_KEY": "default-fixture", "OPENAI_API_KEY": "openai-fixture",
                CONNECTIONS_KEY: [{"kind": "openrouter", "name": "work", "api_key": "named-fixture"}]}
    monkeypatch.setattr(config, "runtime_settings", lambda: settings)
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    target = dispatch_target(model, settings)
    route = dict(provider=target["provider"], model=model, base_url=capability_base_url(model, settings),
                 options={}, use_local=False)
    ce.record_owner_ack(tmp_path, provider=route["provider"], model=model, base_url=route["base_url"], window_tokens=80000)
    if model.startswith("openrouter"):
        default = {**route, "model": "fixture"}
        context_window.record_observation(tmp_path, context_window.route_observation_scope(default, settings), refused=120000)
        scope = context_window.route_observation_scope(route, settings)
        context_window.record_observation(tmp_path, scope, refused=80000)
        assert scope != context_window.route_observation_scope(default, settings)
    monkeypatch.setattr(context_fit, "bounded_prompt_tokens_for_payload", lambda *a, **kw: 35000)
    payload = {"messages": [{"role": "user", "content": "fixture"}], "max_tokens": 65536}
    assert bound_reply_allowance(target, payload)["max_tokens"] == 35000


@pytest.mark.parametrize("main_view", [False, True], ids=["unbound_helper", "main_view"])
def test_a_responses_body_leaves_the_reply_room_of_its_chat_twin(tmp_path, monkeypatch, main_view):
    """A direct OpenAI connection on Responses carries its rows in ``input``, not ``messages``. The helper's known
    window and the reply allowance read every body through ``context_fit.payload_prompt_rows``: the same rows on
    the same route leave the same room, whichever API the connection chose (its capability identity is one)."""
    from contextlib import nullcontext
    from ouroboros import config
    from ouroboros.llm_attempt import bound_reply_allowance
    from ouroboros.response_limits import dispatch_target
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    rows = [{"role": "system", "content": "fixture rules"}, {"role": "user", "content": "a" * 160_000}]
    room, routes = {}, {}
    for surface, body, field in (
            ("chat_completions", {"messages": rows, "max_completion_tokens": 65_536}, "max_completion_tokens"),
            ("responses", {"input": rows, "max_output_tokens": 65_536, "store": False}, "max_output_tokens")):
        settings = {"OPENAI_API_KEY": "openai-fixture", "OPENAI_API_SURFACE": surface}
        monkeypatch.setattr(config, "runtime_settings", lambda settings=settings: settings)
        target = dispatch_target("openai::fixture", settings)
        routes[surface] = target["capability_route"]
        if surface == "chat_completions":
            ce.record_owner_ack(tmp_path, provider="openai", model="openai::fixture",
                                base_url=target["capability_route"]["base_url"], window_tokens=WINDOW)
        context = ua.PhysicalAttemptContext(profile="task_local_nano", rendered_mode="nano",
            measurement_basis="fresh_route_usage", route_fp="route", round_id="round", target_total_tokens=None,
            capacity_total_tokens=WINDOW, context_target_miss=False, automatic_pass_used=False,
            measurement_density=1.0)
        with ua.bind_physical_attempt_context(context) if main_view else nullcontext():
            room[surface] = bound_reply_allowance(target, body)[field]
    assert routes["responses"] == routes["chat_completions"]
    assert room["responses"] == room["chat_completions"] < WINDOW - 40_000


@pytest.mark.parametrize('declared', [0, 120000])
def test_helper_uses_input_refusal_beside_total(tmp_path, monkeypatch, declared):
    from ouroboros.context_window import record_observation
    from ouroboros.llm_helper_window import _helper_attempt_context
    monkeypatch.setattr(ce, 'canonical_evidence_root', lambda: tmp_path)
    monkeypatch.setattr('ouroboros.context_window.route_observation_scope', lambda *a, **k: 'r')
    monkeypatch.setattr(ce, 'probe', lambda *a, **k: ce.CapabilityEvidence(declared,
        'confirmed' if declared else 'unprobeable', 'metadata', 'r'))
    record_observation(tmp_path, 'r', refused=80000, kind='input')
    context = _helper_attempt_context(dict(provider='openai', model='fixture'), {'messages': [{'role': 'user', 'content':'x'}]})
    assert context is not None and context.capacity_total_tokens == 80000


def test_vllm_reply_alone_over_window_changes_the_next_real_request(tmp_path, monkeypatch):
    from pathlib import Path
    from ouroboros import loop
    original_call = loop.call_llm_with_retry
    def call(*args, **kwargs):
        kwargs.setdefault('max_tokens', 90000)
        return original_call(*args, **kwargs)
    monkeypatch.setattr(loop, 'call_llm_with_retry', call)
    sentence = json.loads((Path(__file__).parent / 'fixtures/size_refusals/bodies/'
                           'vllm-0.11.2-max_tokens-alone.body.json').read_text())['error']['message']
    with ScriptedStubModel(final_answer=ANSWER) as server:
        handler = server._server.RequestHandlerClass
        class VllmHandler(handler):
            def _send(self, payload, *, stream=False, status=200):
                if status == 400:
                    payload['error'].update(type='BadRequestError', code=400, param=None)
                return super()._send(payload, stream=stream, status=status)
        server._server.RequestHandlerClass = VllmHandler
        # The route has a known catalogue window, but the serving instance shrank.
        episode = _episode(tmp_path, monkeypatch, server)
        assert episode.plan.window_record.total.tokens == server.window_tokens
        posts = []
        complete = server._completion
        def completion(body):
            count = math.ceil(len(json.dumps({'messages': body.get('messages'), 'tools': body.get('tools')},
                                            ensure_ascii=False)) / 1000)  # synthetic tokenizer: input fits, reply alone does not
            reply = body.get('max_tokens') or body.get('max_completion_tokens')
            posts.append({'input': count, 'reply': reply, 'body': body})
            if count + reply > WINDOW:
                message = sentence.replace('90000', str(reply)).replace('1000', str(count))
                raise StubHttpError(400, message)
            return complete(body)
        server._completion = completion
        text, usage, _ = _run(episode)
    assert posts[0]['reply'] > WINDOW
    assert len(posts) > 1, (text, [(p['input'], p['reply']) for p in posts])
    assert posts[1]['body'] != posts[0]['body']
    assert posts[1]['reply'] <= WINDOW - posts[1]['input'], [(p['input'], p['reply']) for p in posts]
    assert ANSWER in str(text)
