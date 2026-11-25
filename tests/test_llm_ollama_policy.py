"""Native Ollama contracts for llm_ollama_policy."""
import pytest

from tests._ollama import (
    TARGET, install_http as install_http,
)


def test_unchanged_window_is_not_rewritten_but_a_change_or_aging_is(tmp_path, monkeypatch):
    from ouroboros import capability_evidence as ce, context_window
    from ouroboros.llm_ollama_policy import record_window

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    writes = []
    store = ce._save
    key = "ollama:" + context_window.bound_target_scope(TARGET, TARGET["requested_model"])
    monkeypatch.setattr(ce, "_save", lambda root, data: writes.append(data["probes"][key]["window_tokens"]) or store(root, data))
    for _ in range(3):
        record_window(dict(TARGET), 80_000)
    assert writes == [80_000], "the same window on every answer is one write"
    record_window(dict(TARGET), 32_768)
    assert writes == [80_000, 32_768]
    monkeypatch.setattr(context_window, "_window_age", lambda stamp: ce._CONFIRMED_TTL_SEC / 2 + 1)
    record_window(dict(TARGET), 32_768)
    assert writes == [80_000, 32_768, 32_768], "a receipt past half its TTL is refreshed"


@pytest.mark.parametrize("bound", [False, True], ids=["resolve-at-probe", "bound-target"])
def test_settings_window_replaces_cached_loaded_window_and_respects_model_max(install_http, monkeypatch, bound):
    import httpx
    from tests._ollama import OllamaHTTP, _compatible_target
    from ouroboros import capability_evidence as ce, config
    from ouroboros.llm_ollama_policy import record_window
    target = _compatible_target(monkeypatch, TARGET['base_url'])
    settings = config.runtime_settings()
    server = OllamaHTTP(window=4096)
    def handler(request):
        if request.url.path == '/api/show':
            return httpx.Response(200, json={'model_info': {'general.architecture': 'qwen35', 'qwen35.context_length': 131072}})
        return server(request)
    install_http(handler)
    route = dict(provider=target['provider'], model=TARGET['requested_model'], base_url=target['base_url'])
    if bound:
        route.update(target=target, api_key=target['api_key'])
    root = ce.canonical_evidence_root()
    assert ce.probe(root, **route).window_tokens == 4096
    settings['OUROBOROS_MODEL_CONTEXT_WINDOWS'] = {'main': 80000}
    ev = ce.probe(root, **route)
    assert ev.server_window['tokens'] == 80000
    assert ce.with_role_window(ev, 80000).server_window['tokens'] == 80000
    record_window(target, 80000, requested=80000)
    assert ce.probe(root, **route).server_window['tokens'] == 80000
    settings['OUROBOROS_MODEL_CONTEXT_WINDOWS'] = {}
    assert ce.probe(root, **route).window_tokens == 4096


@pytest.mark.parametrize("light_window", [0, 80_000], ids=["light-auto", "light-explicit"])
def test_role_load_windows_on_two_named_connections_at_one_url(install_http, monkeypatch, tmp_path, light_window):
    import httpx
    from ouroboros import capability_evidence as ce, config
    from ouroboros.llm import LLMClient
    from ouroboros.response_limits import bind_context_route, probe_context_route
    from tests._ollama import MESSAGES, OllamaHTTP
    from tests._shared import saved_settings

    main = "openai-compatible@alpha::mock-model"
    light = "openai-compatible@beta::mock-model"
    saved_settings(
        monkeypatch, tmp_path,
        OUROBOROS_CONNECTIONS=[
            {"kind": "openai-compatible", "name": name, "base_url": TARGET["base_url"], "api_key": f"test-{name}"}
            for name in ("alpha", "beta")
        ],
        OUROBOROS_MODEL=main, OUROBOROS_MODEL_LIGHT=light,
        OUROBOROS_MODEL_SAFETY=main + ":latest",
        OUROBOROS_MODEL_VISION="openai-compatible@alpha::another-model",
        OUROBOROS_MODEL_CONTEXT_WINDOWS={"main": 16_384, "safety": 32_768, "light": light_window, "vision": 96_000,
                                        "deep_review": 250_000, "websearch": 200_000},
    )
    servers = {f"Bearer test-{name}": OllamaHTTP(window=4096) for name in ("alpha", "beta")}

    def gateway(request):
        server = servers[request.headers["Authorization"]]
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {"text.context_length": 131072}})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "mock-model", "context_length": 999999}]})
        return server(request)

    install_http(gateway)
    client = LLMClient()
    settings = config.runtime_settings()
    for role, model, name, load in (("main", main, "alpha", 32_768), ("light", light, "beta", light_window)):
        route = {"provider": "openai-compatible", "model": model, "use_local": False, "options": {}}
        bind_context_route(route, settings)
        evidence = probe_context_route(ce.canonical_evidence_root(), route, task={"route_start_read": True})
        assert evidence.server_window["tokens"] == (load or 4096)
        client.chat(MESSAGES, model, model_role=role, no_proxy=True, max_tokens=64)
        (post,) = servers[f"Bearer test-{name}"].posts
        if load:
            assert post["options"]["num_ctx"] == load
        else:
            assert "num_ctx" not in post["options"]


def test_repointed_connection_does_not_size_its_old_endpoint(install_http, monkeypatch):
    from ouroboros import config
    from ouroboros.llm_routing import resolve_target
    from tests._ollama import OllamaHTTP, _compatible_target, chat

    old_target = _compatible_target(monkeypatch, TARGET["base_url"])
    config.runtime_settings().update(OPENAI_COMPATIBLE_BASE_URL="http://runner.lan/v1",
                                     OUROBOROS_MODEL_CONTEXT_WINDOWS={"main": 80_000})
    old_server, new_server = OllamaHTTP(window=4096), OllamaHTTP(window=4096)
    servers = {"127.0.0.1": old_server, "runner.lan": new_server}
    install_http(lambda request: servers[request.url.host](request))
    chat(target=old_target)
    chat(target=resolve_target(TARGET["requested_model"]))
    assert "num_ctx" not in old_server.posts[0]["options"]
    assert new_server.posts[0]["options"]["num_ctx"] == 80_000


def test_native_usage_calibrates_density_without_teaching_padding(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import capability_evidence as ce
    measured = []
    monkeypatch.setattr(ce, 'record_token_density', lambda *args, **kw: measured.append(kw))
    request = SimpleNamespace(provider='openai-compatible', model=TARGET['requested_model'], source='review_substrate.extraction',
        prompt_count_basis='evaluated',
        drive_root=tmp_path, prompt_tokens_bounded_estimate=70000, max_completion_tokens=8192,
        physical_context=SimpleNamespace(view_kind='nano', route_fp='route', window_observation_fp='route',
                                         window_witness_bound=True, response_reserve_tokens=8192))
    counts = {'prompt_tokens': 80042, 'cached_tokens': 40021, 'completion_tokens': 1}
    ce.observe_token_density(request, counts, drive_root_resolver=lambda root: root)
    assert measured[0]['prompt_tokens'] == 80042 and measured[0]['prompt_chars'] == 280000
    request.prompt_count_basis = 'prompt'
    ce.observe_token_density(request, counts, drive_root_resolver=lambda root: root)
    assert len(measured) == 1
    assert not ce._read(tmp_path).get('main_windows'), 'padding is usage, not permission to enlarge the next window'


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("no_proxy", [False, True])
def test_cold_or_stale_load_reads_model_maximum_on_same_client(install_http, monkeypatch, asynchronous, stale, no_proxy):
    import httpx
    from tests._ollama import OllamaHTTP, chat
    from ouroboros import capability_evidence as ce
    server = OllamaHTTP()
    reads = []
    def handler(request):
        if request.url.path == "/api/show":
            reads.append(request)
            return httpx.Response(200, json={"model_info": {"qwen35.context_length": 131072}})
        return server(request)
    install_http(handler)
    if stale:
        from ouroboros.llm_ollama import _identity
        fp = ce.route_fingerprint(provider=TARGET["provider"], model=TARGET["requested_model"], base_url=TARGET["base_url"])
        evidence = ce.CapabilityEvidence(200000, ce.STATUS_CONFIRMED, ce.SOURCE_SERVER_METADATA,
            fp, TARGET["requested_model"], TARGET["provider"], ts="2000-01-01T00:00:00Z", detail="Ollama /api/show (200000)")
        ce._store_evidence(ce.canonical_evidence_root(), "probes", _identity(TARGET), evidence.to_json())
    # Watch the actual constructors before the fixture replaces trust_env.
    constructions = []
    for cls in (httpx.Client, httpx.AsyncClient):
        original = cls.__init__
        def init(self, *a, _original=original, **kw):
            constructions.append((type(self), kw.get("trust_env")))
            return _original(self, *a, **kw)
        monkeypatch.setattr(cls, "__init__", init)
    chat(asynchronous=asynchronous, window=200000, no_proxy=no_proxy)
    assert server.posts[0]["options"]["num_ctx"] == 131072
    assert reads and len(constructions) == 1
    assert constructions[0][1] is (not no_proxy)


@pytest.mark.parametrize("show_status", [200, 404, 503])
def test_explicit_window_needs_a_model_maximum_on_an_identified_runner(install_http, show_status):
    import httpx
    from ouroboros.llm_ollama import OllamaRouteUnsettledError
    from tests._ollama import OllamaHTTP, chat

    server = OllamaHTTP()
    def handle(request):
        if request.url.path == "/api/show":
            return httpx.Response(show_status, json={"capabilities": ["completion"]} if show_status == 200
                                  else {"error": "model 'mock-model' not found"})
        return server(request)
    install_http(handle)
    expected = "HTTP 503" if show_status == 503 else "context maximum.*Pull the model or use Auto"
    with pytest.raises(OllamaRouteUnsettledError, match=expected) as caught:
        chat(window=200000)
    assert caught.value.retryable is (show_status == 503)
    assert not server.posts


def test_changed_load_statement_retires_only_the_old_refusal(install_http, monkeypatch):
    import json
    import httpx
    from pathlib import Path
    from tests._ollama import OllamaHTTP, _compatible_target, chat
    from ouroboros import capability_evidence as ce, config, context_window as cw
    from ouroboros.size_refusal import stamp_failure
    from types import SimpleNamespace
    target = _compatible_target(monkeypatch, TARGET["base_url"])
    settings = config.runtime_settings()
    body = json.loads((Path(__file__).parent / "fixtures/size_refusals/bodies/stand-ollama-0.40.16-native-90k.body.json").read_text())
    server = OllamaHTTP(body, status=400)
    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {"qwen35.context_length": 131072}})
        return server(request)
    install_http(handler)
    root = ce.canonical_evidence_root()
    route = dict(provider=target["provider"], model=TARGET["requested_model"], base_url=target["base_url"])
    scope = cw.route_observation_scope({**route, "use_local": False}, settings, target=target)
    def plan(window):
        settings["OUROBOROS_MODEL_CONTEXT_WINDOWS"] = {"main": window}
        evidence = ce.with_role_window(ce.probe(root, **route, target=target, force=True), window)
        return cw.resolve_window(evidence, 8192, root=root, observation_fp=scope)
    for requested in (80000, 120000):
        assert plan(requested).total.tokens == requested
        with pytest.raises(httpx.HTTPStatusError) as caught:
            chat(target=target)
        assert server.posts[-1]["options"]["num_ctx"] == requested
        usage = {}
        refusal = stamp_failure(usage, caught.value)
        assert refusal.window == 80128 and refusal.token_window == min(80128, requested)
        cw.record_refusal(usage, SimpleNamespace(provider=target["provider"], route_fp=scope, window_record=None))
        assert plan(requested).total.tokens == min(80128, requested)
    assert plan(120000).total.tokens == 80128


@pytest.mark.parametrize("old_window", [0, 80_000], ids=["auto", "explicit"])
def test_inflight_old_generation_cannot_teach_a_new_load(install_http, tmp_path, old_window):
    from types import SimpleNamespace
    from ouroboros import capability_evidence as ce, context_window as cw, usage_accounting as ua
    from ouroboros.llm_ollama_policy import record_window
    from ouroboros.size_refusal import stamp_failure
    from tests._ollama import OllamaHTTP, chat

    # The second worker publishes its 120k load while the first physical POST is
    # in flight. It has not reached the server yet, so /ps still says 80k.
    def refuse(_body):
        record_window(TARGET, 120_000, requested=120_000)
        return {"error": "request (90068 tokens) exceeds the available context size (80128 tokens), try increasing it"}

    server = install_http(OllamaHTTP(refuse, status=400, loaded=bool(old_window)))
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="old", root_task_id="old")), \
            pytest.raises(Exception) as caught:
        chat(window=old_window)
    scope = cw.bound_target_scope(TARGET, TARGET["requested_model"])
    usage = {}
    refusal = stamp_failure(usage, caught.value)
    assert refusal.window == 80128 and refusal.ollama_scope == scope
    assert caught.value.physical_attempt_capture.attempt_id and len(server.posts) == 1
    plan = SimpleNamespace(provider=TARGET["provider"], route_fp=scope, window_record=None)
    cw.record_refusal(usage, plan, refused_input=90_068)
    root = ce.canonical_evidence_root()
    for _ in range(2):  # an equal read cannot increment the statement just to hide a poisoned observation
        record_window(TARGET, 120_000, requested=120_000)
        current = ce._read(root)["main_windows"][scope]
        assert current["reads"][0]["tokens"] == 120_000
        assert not current.get("observations")


def test_positive_probe_publishes_through_the_conditional_writer_and_carries_its_order(install_http, monkeypatch):
    """A positive native sizing read is one runner statement: published under the
    store transaction with its receipt, and its assigned order rides back to the
    planner instead of being re-numbered at resolution."""
    from ouroboros import capability_evidence as ce, context_window as cw
    from ouroboros.llm_ollama import _identity
    from tests._ollama import OllamaHTTP, _compatible_target

    target = _compatible_target(monkeypatch, TARGET["base_url"])
    install_http(OllamaHTTP(window=8_000))
    root = ce.canonical_evidence_root()
    route = dict(provider=target["provider"], model=TARGET["requested_model"], base_url=target["base_url"])
    evidence = ce.probe(root, **route, target=target, force=True)
    key = _identity(dict(target))
    scope = key.removeprefix("ollama:")
    data = ce._read(root)
    assert data["probes"][key]["window_tokens"] == 8_000
    [read] = data["main_windows"][scope]["reads"]
    assert (read["tokens"], read["sequence"], read["source"]) == (8_000, 1, ce.SOURCE_SERVER_METADATA)
    assert evidence.window_tokens == 8_000 and evidence.read == "answered"
    assert (evidence.window_sequence, evidence.window_sequence_route) == (1, scope)
    # An unchanged re-read coalesces: one frontier entry, the same order, no second write.
    again = ce.probe(root, **route, target=target, force=True)
    [read] = ce._read(root)["main_windows"][scope]["reads"]
    assert read["sequence"] == 1 and again.window_sequence == 1 and again.read == "answered"
    # The planner keeps the assigned order: resolving under the bound scope does not re-number it.
    record = cw.resolve_window(evidence, 0, root=root, observation_fp=scope)
    assert record.total.tokens == 8_000
    [read] = ce._read(root)["main_windows"][scope]["reads"]
    assert read["sequence"] == 1


def test_late_positive_probe_neither_overwrites_nor_renumbers_a_newer_statement(install_http, monkeypatch):
    """The Auto probe reads 8k, pauses, and a concurrent sender publishes its 120k
    dispatch statement; the resumed probe must lose the conditional write, keep the
    winner untouched, and return its order — never replace the current load with
    the stale read."""
    from ouroboros import capability_evidence as ce
    from ouroboros.llm_ollama import _identity
    from ouroboros.llm_ollama_policy import record_window
    from tests._ollama import OllamaHTTP, _compatible_target

    target = _compatible_target(monkeypatch, TARGET["base_url"])
    key = _identity(dict(target))
    scope = key.removeprefix("ollama:")
    server = OllamaHTTP(window=8_000)
    rivaled = []

    def handle(request):
        # The rival sender publishes while this probe's own discovery read is in flight,
        # i.e. after the probe took its store snapshot and before its own publication.
        if request.url.path == "/api/ps" and not rivaled:
            rivaled.append(request.url.path)
            record_window(dict(target), 120_000, requested=120_000)
        return server(request)

    install_http(handle)
    root = ce.canonical_evidence_root()
    route = dict(provider=target["provider"], model=TARGET["requested_model"], base_url=target["base_url"])
    evidence = ce.probe(root, **route, target=target, force=True)
    assert rivaled  # the rival statement landed during the probe's own read
    data = ce._read(root)
    assert data["probes"][key]["window_tokens"] == 120_000  # the winner is kept as published
    [read] = data["main_windows"][scope]["reads"]
    assert (read["tokens"], read["sequence"]) == (120_000, 1)  # one frontier entry, never re-numbered
    assert evidence.window_tokens == 120_000  # the loser returns the current load...
    assert evidence.read == ""  # ...as a recorded statement, not this call's live read
    assert (evidence.window_sequence, evidence.window_sequence_route) == (1, scope)


def test_a_probe_overtaken_after_its_own_write_is_not_its_live_answer(install_http, monkeypatch):
    """The probe's write wins, then a concurrent sender publishes its 120k statement before the probe reads its
    row back: the planner gets the current statement as recorded, never as what this call just measured."""
    from ouroboros import capability_evidence as ce, llm_ollama_policy as policy
    from ouroboros.llm_ollama import _identity
    from tests._ollama import OllamaHTTP, _compatible_target

    target = _compatible_target(monkeypatch, TARGET["base_url"])
    scope = _identity(dict(target)).removeprefix("ollama:")
    install_http(OllamaHTTP(window=8_000))
    write, rival = policy.record_window, []

    def overtaken(*args, **kwargs):
        sequence = write(*args, **kwargs)
        rival.append(write(dict(target), 120_000, requested=120_000))  # published after this write, before its read-back
        return sequence

    monkeypatch.setattr(policy, "record_window", overtaken)
    root = ce.canonical_evidence_root()
    evidence = ce.probe(root, provider=target["provider"], model=TARGET["requested_model"],
                        base_url=target["base_url"], target=target, force=True)
    assert rival == [2]
    [read] = ce._read(root)["main_windows"][scope]["reads"]
    assert (read["tokens"], read["sequence"]) == (120_000, 2)
    assert evidence.window_tokens == 120_000 and evidence.read == ""  # recorded, not this call's live read
    assert (evidence.window_sequence, evidence.window_sequence_route) == (2, scope)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("window", [0, 80_000], ids=["auto", "explicit"])
def test_successful_generation_carries_its_dispatch_load_statement(install_http, window, asynchronous):
    """A success keeps the load statement it was dispatched under, in the refusal's
    own shape: the accepted input is durable evidence only for that load."""
    from ouroboros import capability_evidence as ce
    from ouroboros.llm_ollama import _identity
    from tests._ollama import OllamaHTTP, chat, completed

    install_http(OllamaHTTP(completed(prompt_eval_count=60_000), window=80_000))
    _, usage = chat(asynchronous=asynchronous, window=window)
    scope, sequence = usage["ollama_load_statement"]
    assert scope == _identity(dict(TARGET)).removeprefix("ollama:")
    root = ce.canonical_evidence_root()
    record = ce._read(root)["main_windows"][scope]
    assert record["reads"][0]["sequence"] == sequence  # the statement names the current frontier
    assert usage["capacity_input_tokens"] == 60_000


def test_stale_success_cannot_publish_over_a_newer_smaller_load(install_http, monkeypatch):
    """The acceptance-side mirror of ``test_inflight_old_generation_cannot_teach_a_new_load``:
    a 60k success served by an 80k load settles after another worker published a 32k
    load; the durable acceptance must be dropped and the 32k load's floor stays unset,
    while the paid usage, the anchor and the density witness survive."""
    from types import SimpleNamespace
    from ouroboros import capability_evidence as ce
    from ouroboros.context_anchor import ACCEPTED_INPUTS_KEY, main_input_anchor, record_input_anchor
    from ouroboros.context_window import WindowRecord
    from ouroboros.llm_ollama import _identity
    from ouroboros.llm_ollama_policy import record_window
    from tests._ollama import OllamaHTTP, chat, completed

    def reply(body):
        record_window(dict(TARGET), 32_000, requested=32_000)  # a concurrent re-load mid-POST
        return completed(prompt_eval_count=60_000)

    install_http(OllamaHTTP(reply))
    density = []
    monkeypatch.setattr(ce, "record_token_density", lambda *args, **kw: density.append(kw))
    _, usage = chat(window=80_000)
    scope = _identity(dict(TARGET)).removeprefix("ollama:")
    root = ce.canonical_evidence_root()
    [read] = ce._read(root)["main_windows"][scope]["reads"]
    assert read["tokens"] == 32_000  # the newer, smaller load is the frontier
    assert usage["ollama_load_statement"] == [scope, read["sequence"] - 1]  # dispatched under the retired statement

    plan = SimpleNamespace(provider=TARGET["provider"], model=TARGET["requested_model"], route_fp=scope,
                           window_record=WindowRecord(scope, load=80_000),  # a native load's plan
                           window_witness_bound=True, output_reserve_tokens=8192, view_kind="")
    accumulated = {
        "_last_round_cache_usage": {"round": 1, "model": TARGET["resolved_model"], "provider": TARGET["provider"],
                                    "prompt_tokens": 60_000, "cached_tokens": 0, "cache_write_tokens": 0,
                                    "capacity_input_tokens": usage["capacity_input_tokens"],
                                    "ollama_load_statement": list(usage["ollama_load_statement"])},
        "_context_raw_input_tokens": 58_000,
        "_model_substitutions": [],
    }
    record_input_anchor(accumulated, plan, round_idx=1, estimate_tokens=58_000)
    record = ce._read(root)["main_windows"][scope]
    assert not record.get("observations")  # the stale acceptance is no durable window evidence
    # The acceptance is kept under the statement it was dispatched under; the 32k load's floor is not it.
    assert list(accumulated[ACCEPTED_INPUTS_KEY].values()) == [60_000]
    anchor, floor = main_input_anchor(accumulated, plan, rewrite_epoch=0)
    assert (anchor.input_tokens, floor) == (60_000, None)
    assert density and density[0]["prompt_tokens"] == 60_000  # density is model-scoped, not load-scoped
    # A statement of another scope binds nothing either (an Auto rotation answered elsewhere).
    accumulated["_last_round_cache_usage"]["ollama_load_statement"] = ["another-scope", read["sequence"]]
    record_input_anchor(accumulated, plan, round_idx=2, estimate_tokens=58_000)
    assert not ce._read(root)["main_windows"][scope].get("observations")


def test_current_load_success_publishes_its_accepted_input(install_http, monkeypatch):
    """With no rival load, the same settlement writes the durable acceptance under
    the dispatch statement's own sequence."""
    from types import SimpleNamespace
    from ouroboros import capability_evidence as ce
    from ouroboros.context_anchor import record_input_anchor
    from ouroboros.llm_ollama import _identity
    from tests._ollama import OllamaHTTP, chat, completed

    monkeypatch.setattr(ce, "record_token_density", lambda *args, **kw: None)
    install_http(OllamaHTTP(completed(prompt_eval_count=60_000)))
    _, usage = chat(window=80_000)
    scope = _identity(dict(TARGET)).removeprefix("ollama:")
    root = ce.canonical_evidence_root()
    plan = SimpleNamespace(provider=TARGET["provider"], model=TARGET["requested_model"], route_fp=scope,
                           window_record=None, window_witness_bound=True, output_reserve_tokens=8192,
                           view_kind="")
    accumulated = {
        "_last_round_cache_usage": {"round": 1, "model": TARGET["resolved_model"], "provider": TARGET["provider"],
                                    "prompt_tokens": 60_000, "cached_tokens": 0, "cache_write_tokens": 0,
                                    "capacity_input_tokens": usage["capacity_input_tokens"],
                                    "ollama_load_statement": list(usage["ollama_load_statement"])},
        "_context_raw_input_tokens": 58_000,
        "_model_substitutions": [],
    }
    record_input_anchor(accumulated, plan, round_idx=1, estimate_tokens=58_000)
    record = ce._read(root)["main_windows"][scope]
    [observation] = record["observations"]
    assert observation["accepted"] == 60_000 and observation["kind"] == "input"
    assert observation["sequence"] > record["reads"][0]["sequence"]


def test_round_cache_projection_carries_the_load_statement():
    """The carrier between the settled native usage and ``record_input_anchor``:
    the round projection keeps the statement, and a route without one projects none."""
    from ouroboros.loop_llm_call import _record_round_cache_facts

    accumulated = {}
    usage = {"prompt_tokens": 60_000, "cached_tokens": 0, "capacity_input_tokens": 60_000,
             "ollama_load_statement": ["scope", 3]}
    _record_round_cache_facts(accumulated, usage, round_idx=1, round_id="r1", model="m",
                              provider="openai-compatible")
    last = accumulated["_last_round_cache_usage"]
    assert last["ollama_load_statement"] == ["scope", 3] and last["capacity_input_tokens"] == 60_000
    plain = {}
    _record_round_cache_facts(plain, {"prompt_tokens": 10}, round_idx=1, round_id="r2", model="m", provider="p")
    assert "ollama_load_statement" not in plain["_last_round_cache_usage"]


@pytest.mark.parametrize("number", ["current", "ordinary", "stale", "missing", "another scope", "malformed"])
def test_an_answer_without_a_current_load_number_raises_no_capacity_in_the_next_fit(install_http, monkeypatch, number):
    """A 100k answer from a 120k load settles after the runner reloaded at 80k. Read as the next round reads it
    (``measure_main_fit`` over the plan the new load's own probe builds), a number that is missing (the store was
    busy at dispatch), stale, of another scope or malformed adds no capacity, durable or this task's own floor,
    while the answer and its counts stay recorded and anchor the estimate. A current number, and an ordinary
    provider's answer which carries none, are recorded as before."""
    from dataclasses import replace
    from types import SimpleNamespace
    from ouroboros import capability_evidence as ce, context_window as cw, llm_ollama
    from ouroboros.context_anchor import main_input_anchor, record_input_anchor
    from ouroboros.context_fit import measure_main_fit
    from ouroboros.llm_ollama import _identity
    from ouroboros.loop_llm_call import _record_round_cache_facts
    from tests._ollama import OllamaHTTP, _compatible_target, chat, completed
    from tests.test_context_anchor import _plan

    monkeypatch.setattr(ce, "record_token_density", lambda *args, **kw: None)
    monkeypatch.setattr(ce, "resolve_main_token_density", lambda *args, **kw: (1.0, "cold_estimate"))
    target = _compatible_target(monkeypatch, TARGET["base_url"])
    scope = _identity(dict(target)).removeprefix("ollama:")
    server = OllamaHTTP(completed(prompt_eval_count=100_000), window=120_000)
    install_http(server)
    root = ce.canonical_evidence_root()
    route = dict(provider=target["provider"], model=TARGET["requested_model"], base_url=target["base_url"])
    ce.probe(root, **route, target=target, force=True)  # the plan's own read of the 120k load
    if number == "missing":
        monkeypatch.setattr(llm_ollama, "record_window", lambda *args, **kw: None)  # the store was busy at dispatch
    message, usage = chat(target=dict(target), window=0)
    assert message["content"] == "391" and (usage["prompt_tokens"], usage["completion_tokens"]) == (100_000, 4)
    if number == "missing":
        assert usage["ollama_load_statement"] == [scope, None]  # a native answer says it has no number
    if number not in ("current", "ordinary"):
        server.window = 80_000  # another sender reloads the runner while this answer settles
    evidence = ce.probe(root, **route, target=target, force=True)

    accumulated = {"_context_raw_input_tokens": 98_000, "_model_substitutions": []}
    _record_round_cache_facts(accumulated, usage, round_idx=1, round_id="r1", model=TARGET["resolved_model"],
                              provider=TARGET["provider"])
    last = accumulated["_last_round_cache_usage"]
    if number == "ordinary":
        del last["capacity_input_tokens"], last["ollama_load_statement"]
    elif number == "another scope":
        last["ollama_load_statement"][0] = "another-scope"
    elif number == "malformed":
        last["ollama_load_statement"][1] = True
    plan = SimpleNamespace(provider=TARGET["provider"], model=TARGET["requested_model"], route_fp=scope,
                           window_record=None if number == "ordinary" else cw.WindowRecord(scope, load=0),  # Auto's
                           window_witness_bound=True, output_reserve_tokens=8192, view_kind="")
    record_input_anchor(accumulated, plan, round_idx=1, estimate_tokens=98_000)
    anchor, floor = main_input_anchor(accumulated, plan, rewrite_epoch=0)
    recorded = number in ("current", "ordinary")
    # The paid usage still anchors this task; it floors it only under the load statement it was accepted under.
    assert (anchor.input_tokens, floor) == (100_000, 100_000 if recorded else None)

    window = cw.resolve_window(evidence, 8_192, root=root, observation_fp=scope)
    fit_plan = replace(_plan(), provider=TARGET["provider"], model=TARGET["requested_model"], route_fp=scope,
                       window_tokens=window.total_equivalent, window_record=window, evidence_source=window.source,
                       output_reserve_tokens=8_192)
    fit = measure_main_fit(fit_plan, fit_plan.messages_for("max"), [], profile="owner_max", rendered_mode="max",
                           round_id="r", accepted_input_tokens=floor)
    assert fit.measurement.capacity_total_tokens == (120_000 if recorded else 80_000)  # the load the next fit meets
    assert cw.resolve_window(fit_plan, 8_192).accepted == (100_000 if recorded else 0)  # durable acceptance


@pytest.mark.parametrize("binding", ["routing-header", "api-key"])
def test_named_targets_use_the_same_window_receipt_for_planning_and_dispatch(install_http, monkeypatch, binding):
    import httpx
    from tests._ollama import OllamaHTTP, chat
    from ouroboros import capability_evidence as ce, config
    from ouroboros.context_window import route_observation_scope
    from ouroboros.llm_routing import resolve_target
    from ouroboros.provider_models import CONNECTIONS_KEY
    from ouroboros.response_limits import probe_context_route

    model = "openai-compatible@lab::mock-model"
    settings = {CONNECTIONS_KEY: [dict(kind="openai-compatible", name="lab", base_url=TARGET["base_url"])],
                "OUROBOROS_MODEL": model, "OUROBOROS_MODEL_CONTEXT_WINDOWS": {"main": 200000}}
    monkeypatch.setattr(config, "runtime_settings", lambda: settings)
    base = resolve_target(model, settings=settings)
    servers = {key: OllamaHTTP() for key in ("a", "b")}
    maxima = {"a": 80000, "b": 131072}
    shows = []

    def handle(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        key = (request.headers.get("X-Runner") if binding == "routing-header"
               else request.headers.get("Authorization", "").removeprefix("Bearer "))
        if key not in servers:
            return httpx.Response(404)
        if request.url.path == "/api/show":
            shows.append(key)
            return httpx.Response(200, json={"model_info": {"text.context_length": maxima[key]}})
        return servers[key](request)

    install_http(handle)
    root = ce.canonical_evidence_root()
    route = dict(provider=base["provider"], model=model, base_url=base["capability_route"]["base_url"],
                 use_local=False, options={})
    a, b = ({**base, **({"default_headers": {"X-Runner": key}} if binding == "routing-header"
                        else {"api_key": key})} for key in ("a", "b"))
    assert route_observation_scope(route, settings, target=a) != route_observation_scope(route, settings, target=b)
    evidence_a = probe_context_route(root, {**route, "_resolved_target": a}, task={"route_start_read": True})
    assert evidence_a.window_tokens == maxima["a"]
    reads = list(shows)
    chat(target=a)
    assert shows == reads
    chat(target=b)
    assert shows == [*reads, "b"]
    evidence_b = probe_context_route(root, {**route, "_resolved_target": b}, task={}, allow_fetch=False)
    assert evidence_b.window_tokens == maxima["b"]
    if binding == "routing-header":
        assert evidence_a.route_fp != evidence_b.route_fp
    assert probe_context_route(root, {**route, "_resolved_target": a}, task={}, allow_fetch=False).window_tokens == maxima["a"]
    assert {key: server.posts[0]["options"]["num_ctx"] for key, server in servers.items()} == maxima


@pytest.mark.parametrize("first", ["native", "compatible"])
def test_named_gateway_protocol_is_bound_to_dispatched_key_even_after_restart(install_http, monkeypatch, tmp_path, first):
    from tests._ollama import MESSAGES, LlamaCppHTTP, OllamaHTTP
    from tests._shared import saved_settings
    from ouroboros import capability_evidence as ce, llm_ollama
    from ouroboros.llm import LLMClient
    from ouroboros.server_window import server_root

    servers = {"native": OllamaHTTP(), "compatible": LlamaCppHTTP()}
    model = "openai-compatible@lab::mock-model"
    install_http(lambda request: servers[request.headers["Authorization"].removeprefix("Bearer test-")](request))
    legacy = ce.route_fingerprint(provider=TARGET["provider"], base_url=server_root(TARGET["base_url"]),
                                  model=TARGET["resolved_model"])
    root = ce.canonical_evidence_root()
    ce._store_evidence(root, "native_protocols", legacy, {"protocol": "ollama"})

    for cold in (False, True):
        if cold:
            llm_ollama._rejected.clear()
        client = LLMClient()
        for name in (first, next(n for n in servers if n != first)):
            saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL=model, OUROBOROS_CONNECTIONS=[{
                "kind": "openai-compatible", "name": "lab", "base_url": TARGET["base_url"], "api_key": "test-" + name}])
            message, _ = client.chat(MESSAGES, model, no_proxy=True, max_tokens=64)
            assert message["content"] == ("391" if name == "native" else "from llama.cpp")
    assert servers["native"].gets.count("/api/version") == 3  # discovery and one guard check per send
    assert servers["compatible"].paths.count(("GET", "/api/version")) == 2
    assert ("POST", "/api/chat") not in servers["compatible"].paths

    assert "test-native" not in str(ce._read(root)["native_protocols"])

@pytest.mark.parametrize("asynchronous", [False, True])
def test_model_pull_and_changed_loaded_digest_refresh_thinking_before_chat(install_http, monkeypatch, asynchronous):
    import asyncio
    import httpx
    from ouroboros import capability_evidence as ce, llm_ollama
    from ouroboros.llm import LLMClient
    from tests._ollama import MESSAGES, OllamaHTTP, _compatible_target

    target = _compatible_target(monkeypatch, TARGET["base_url"])
    server = OllamaHTTP()
    digest, shows = "", []
    def handle(request):
        if request.url.path == "/api/show":
            shows.append(digest)
            if not digest:
                return httpx.Response(404, json={"error": "model 'mock-model' not found"})
            return httpx.Response(200, json={"capabilities": ["completion", "thinking"] if digest == "first"
                                                      else ["completion"]})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "mock-model", "digest": digest,
                                                         "context_length": 80000}] if digest else []})
        return server(request)
    install_http(handle)
    assert llm_ollama.is_ollama(target)
    assert "thinking_mode" not in ce._read(ce.canonical_evidence_root())["native_protocols"][llm_ollama._identity(target)]
    client = LLMClient()
    for digest in ("first", "second"):
        args = dict(model=TARGET["requested_model"], messages=MESSAGES, reasoning_effort="high", no_proxy=True)
        asyncio.run(client.chat_async(**args)) if asynchronous else client.chat(**args)
    assert [post.get("think") for post in server.posts] == ["high", None]
    assert shows == ["", "first", "second"]

def test_failed_metadata_leaves_explicit_main_and_reviewer_windows(install_http, monkeypatch, tmp_path):
    import httpx
    from ouroboros import capability_evidence as ce, config
    from ouroboros.response_limits import bind_context_route, probe_context_route
    from ouroboros.reviewer_window import resolve_reviewer_window
    from tests._shared import saved_settings

    model = TARGET["requested_model"]
    saved_settings(monkeypatch, tmp_path, OPENAI_COMPATIBLE_BASE_URL=TARGET["base_url"], OUROBOROS_MODEL=model,
                   OUROBOROS_MODEL_CONTEXT_WINDOWS={"main": 120000, "light": 80000})
    paths = []
    def busy(request):
        paths.append(request.url.path)
        return httpx.Response(429)
    install_http(busy)
    route = dict(provider=TARGET["provider"], model=model, use_local=False, options={})
    bind_context_route(route, config.runtime_settings())
    root = ce.canonical_evidence_root()
    probe_context_route(root, route, task={}, allow_fetch=False)
    assert paths == []
    evidence = probe_context_route(root, route, task={"route_start_read": True})
    assert evidence.read == "failed" and ce.with_role_window(evidence, 120000).window_tokens == 120000
    reviewer = resolve_reviewer_window(model, model_role="light", use_local=False)
    assert reviewer.window_tokens == reviewer.asserted_window_tokens == 80000
    assert paths == ["/api/version", "/api/version"]


def test_first_main_start_reads_replacement_window_in_auto(install_http, monkeypatch):
    import httpx
    from ouroboros import capability_evidence as ce, config, context_window as cw
    from ouroboros.response_limits import bind_context_route, probe_context_route
    from tests._ollama import OllamaHTTP, _compatible_target

    target = _compatible_target(monkeypatch, TARGET["base_url"])
    route = dict(provider=target["provider"], model=target["capability_route"]["model"], base_url=target["base_url"],
                 use_local=False, options={})
    root = ce.canonical_evidence_root()
    def start():
        bind_context_route(route, config.runtime_settings())
        evidence = probe_context_route(root, route, task={"route_start_read": True})
        return cw.resolve_window(evidence, 0, root=root, observation_fp=route["window_observation_fp"])

    native = install_http(OllamaHTTP(window=4096))
    assert start().total.tokens == 4096
    paths = []
    def replacement(request):
        path = request.url.path
        paths.append(path)
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": target["resolved_model"]}]})
        if path == "/api/v0/models":
            return httpx.Response(200, json={"data": [{"id": target["resolved_model"], "state": "loaded",
                                                       "max_context_length": 262144, "loaded_context_length": 120000}]})
        return httpx.Response(200, json={"error": f"Unexpected endpoint or method. ({request.method} {path})"})
    install_http(replacement)
    assert start().total.tokens == 120000
    assert paths[:2] == ["/api/version", "/api/ps"] and "/api/v0/models" in paths
    assert not native.posts and "/api/chat" not in paths
    assert all(read["tokens"] != 4096 for read in ce._read(root)["main_windows"][route["window_observation_fp"]]["reads"])


def test_unloaded_auto_retires_a_prior_load_but_failed_read_does_not(install_http, monkeypatch):
    from ouroboros import capability_evidence as ce, config, context_window as cw
    from ouroboros.response_limits import bind_context_route, probe_context_route
    from tests._ollama import OllamaHTTP, _compatible_target
    import httpx

    target = _compatible_target(monkeypatch, TARGET["base_url"])
    model = target["capability_route"]["model"]
    route = dict(provider=target["provider"], model=model, base_url=target["base_url"], use_local=False, options={})
    root = ce.canonical_evidence_root()
    server = OllamaHTTP(window=4096)
    state = "loaded"
    def handle(request):
        if state == "busy" and request.url.path == "/api/ps":
            return httpx.Response(429)
        return server(request)
    install_http(handle)
    bind_context_route(route, config.runtime_settings())
    def read(fetch=True):
        return probe_context_route(root, route, task={}, allow_fetch=fetch)
    assert read().window_tokens == 4096
    assert read().window_tokens == 4096
    state = "busy"
    read()
    assert read(False).window_tokens == 4096
    server.loaded, state = False, "absent"
    absent = read()
    assert not absent.window_tokens and not read(False).window_tokens
    record = cw.resolve_window(absent, 0, root=root, observation_fp=route["window_observation_fp"])
    assert record.total is None
    server.window, server.loaded = 120000, True
    assert read().window_tokens == 120000


def test_old_server_reports_upgrade_in_settings_and_cached_task_fit(install_http, monkeypatch):
    from ouroboros import capability_evidence as ce, config
    from ouroboros.context_fit import resolve_context_fit_route
    from ouroboros.llm import LLMClient
    from ouroboros.llm_ollama import OllamaUpgradeRequired
    from ouroboros.llm_probe import probe_provider_readiness
    from tests._ollama import OllamaHTTP, _compatible_target
    import httpx

    target = _compatible_target(monkeypatch, TARGET["base_url"])
    monkeypatch.setattr(config, "DATA_DIR", ce.canonical_evidence_root())
    server, version = OllamaHTTP(), "0.40.16"
    def handle(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": version})
        return server(request)
    install_http(handle)
    model = target["capability_route"]["model"]
    task = {"model": model, "use_local_model": False}
    assert resolve_context_fit_route(task, allow_fetch=True)[1].window_tokens == server.window
    version = "0.12.5"
    with pytest.raises(OllamaUpgradeRequired, match="0.12.5.*Upgrade"):
        resolve_context_fit_route(task, allow_fetch=True)
    result = probe_provider_readiness(LLMClient(), model, settings=config.runtime_settings())
    assert not result["ok"] and "Upgrade to 0.12.6" in result["error"]
    assert server.posts == []


def test_async_native_metadata_publication_runs_off_the_event_loop(install_http, monkeypatch):
    import asyncio
    from threading import get_ident
    from ouroboros import capability_evidence as ce
    from ouroboros.llm import LLMClient
    from tests._ollama import MESSAGES, OllamaHTTP, _compatible_target

    model = _compatible_target(monkeypatch, TARGET["base_url"])["capability_route"]["model"]
    install_http(OllamaHTTP())
    transactions = []
    transaction = ce._store_transaction
    def observed(root):
        transactions.append(get_ident())
        return transaction(root)
    monkeypatch.setattr(ce, "_store_transaction", observed)
    loop_thread = get_ident()
    assert asyncio.run(LLMClient().chat_async(MESSAGES, model=model))[0]["content"] == "391"
    assert transactions and loop_thread not in transactions
