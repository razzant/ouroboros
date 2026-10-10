"""Main's host clock line: sampled per new physical candidate, sealed with it, replayed once consumed.

Real consumers only: the OpenAI-compatible sync/async ladder, Anthropic Messages,
the local and GigaChat lanes, Claudexor invocations, the forced-final admission and
``call_llm_with_retry``'s canonical replay. Each physical send is checked at the
ledger: the bytes measured and priced (the reservation row), sealed (the candidate
manifest) and sent (the transport's arguments) are one candidate.
"""

from __future__ import annotations

import asyncio
import copy
import datetime as dt
import hashlib
import itertools
import json
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import send_clock as sc, usage_accounting as ua
from ouroboros.client_surface import normalize_client_surface, normalize_timezone_name
from ouroboros.llm import LLMClient
from ouroboros.llm_attempt import _attempt_request, _canonical_candidate_bytes
from ouroboros.loop_llm_call import call_llm_with_retry
from tests.test_processing_transport import transport  # noqa: F401 - fixture

MOSCOW = sc.SendClockPolicy(zone="Europe/Moscow")
T = dt.datetime(2026, 9, 26, 14, 23, 5, tzinfo=dt.timezone.utc)


@pytest.fixture
def ticking(monkeypatch):
    """Every sample one minute later, so a re-sample is visible in the bytes."""
    counter = itertools.count()
    monkeypatch.setattr(sc, "_now", lambda: T + dt.timedelta(minutes=next(counter)))


def _digest(payload):
    return hashlib.sha256(_canonical_candidate_bytes(payload)).hexdigest()


def _clock_rows(messages):
    return [row for row in messages if isinstance(row, dict) and str(row.get("content") or "").startswith(sc.CLOCK_NOTE_PREFIX)]


def _ledger(root: Path):
    return ledger_rows(root)


def _sealed(row):
    """The persisted candidate manifest: the bytes sealed before the provider saw them."""
    return json.loads(Path(row["candidate_manifest_ref"]["path"]).read_text())["candidate_raw_sha256"]


# --- policy and ingress ------------------------------------------------------------


def test_only_a_main_loop_opts_in_and_the_zone_is_the_validated_sender_zone():
    surface = {"client_surface": {"timezone": "Europe/Moscow"}}
    assert sc.main_clock_policy(surface, task_type="task") == MOSCOW
    assert sc.main_clock_policy({}, task_type="task") == sc.SendClockPolicy(zone="")
    assert sc.main_clock_policy({**surface, "delegation_role": "subagent"}) is None
    for helper in ("presence", "review", "summarize"):
        assert sc.main_clock_policy(surface, task_type=helper) is None
    # A zone that stopped loading (or never did) is unknown, never the server's zone.
    assert sc.main_clock_policy({"client_surface": {"timezone": "Mars/Olympus"}}).zone == ""


@pytest.mark.parametrize("raw, expected", [
    ("Europe/Moscow", "Europe/Moscow"), ("UTC", "UTC"), ("America/Argentina/Buenos_Aires",
                                                         "America/Argentina/Buenos_Aires"),
    ("  Asia/Tokyo ", "Asia/Tokyo"), ("../../etc/passwd", ""), ("/etc/localtime", ""), ("Not/AZone", ""),
    ("Europe/" + "x" * 80, ""), (42, ""), (None, ""), ("", ""),
])
def test_the_ingress_keeps_only_a_loadable_iana_name(raw, expected):
    assert normalize_timezone_name(raw) == expected
    fact = normalize_client_surface({"ua": "browser", "timezone": raw})
    assert fact.get("timezone", "") == expected and fact["ua"] == "browser"


def test_the_line_names_utc_and_the_sender_zone_at_a_fixed_width():
    assert sc.render_clock_note(MOSCOW, T) == (
        "[Host clock at this request: 2026-09-26T14:23:05Z UTC; "
        "sender's zone Europe/Moscow: 2026-09-26 17:23:05 (UTC+03:00)]")
    assert sc.render_clock_note(sc.SendClockPolicy(), T) == (
        "[Host clock at this request: 2026-09-26T14:23:05Z UTC; sender's time zone unknown]")
    new_york = sc.SendClockPolicy(zone="America/New_York")
    winter, summer = (sc.render_clock_note(new_york, dt.datetime(2026, month, 1, tzinfo=dt.timezone.utc))
                      for month in (1, 7))
    assert "(UTC-05:00)" in winter and "(UTC-04:00)" in summer and len(winter) == len(summer)


def test_stamping_replaces_its_pending_line_and_never_mutates(ticking):
    source = {"messages": [{"role": "user", "content": "work"}], "model": "m"}
    before = copy.deepcopy(source)
    assert sc.stamp_clock_note(source) is source  # no Main scope: nothing changes
    with sc.MainSendClock(MOSCOW).bound() as clock:
        first = sc.stamp_clock_note(source)
        again = sc.stamp_clock_note(first)
        with sc.MainSendClock(None).bound():
            assert sc.stamp_clock_note(source) is source  # a helper call inside Main
    assert source == before
    assert [len(_clock_rows(item["messages"])) for item in (first, again)] == [1, 1]
    assert first["messages"][-1] != again["messages"][-1]
    assert clock.notes == [first["messages"][-1]["content"], again["messages"][-1]["content"]]


def test_the_anthropic_line_is_exactly_what_the_next_canonical_request_replays(ticking):
    """Coalesced into the trailing user turn as the Messages builder does, so round N's
    request is a byte prefix of round N+1's."""
    client = LLMClient(api_key="unused")
    canonical = [{"role": "system", "content": "policy"}, {"role": "user", "content": "go"},
                 {"role": "assistant", "content": "", "tool_calls": [
                     {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": "c1", "content": "file body"}]
    with sc.MainSendClock(MOSCOW).bound():
        _system, native = client._build_anthropic_messages(canonical)
        sent = sc.stamp_clock_note({"messages": native}, blocks=True)["messages"]
    line = sent[-1]["content"][-1]["text"]
    assert sent[-1]["content"][0]["type"] == "tool_result" and len(sent) == len(native)
    replay = canonical + [{"role": "user", "content": line}, {"role": "assistant", "content": "done"},
                          {"role": "user", "content": "next"}]
    _system, following = client._build_anthropic_messages(replay)
    assert following[:len(sent)] == sent


# --- real provider lanes --------------------------------------------------------------


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider", ["openai", "openrouter", "anthropic"])
def test_each_remote_lane_measures_prices_seals_and_sends_one_clocked_candidate(
        transport, ticking, provider, asynchronous):  # noqa: F811
    root, client, sent = transport
    kwargs = dict(messages=[{"role": "user", "content": "unchanged input"}], model=f"{provider}::test-model",
                  model_role="main", max_tokens=123, reasoning_effort="high")
    with sc.MainSendClock(MOSCOW).bound() as clock:
        if asynchronous:
            asyncio.run(client.chat_async(**kwargs))
        else:
            client.chat(**kwargs)
    candidate = sent[0]["payload"] if provider == "anthropic" else sent[0]
    messages = candidate["messages"]
    if provider == "anthropic":
        assert messages[-1]["content"][-1] == {"type": "text", "text": clock.notes[0]}
    else:
        assert messages[-1] == {"role": "user", "content": clock.notes[0]}
    assert clock.notes[0].startswith(sc.CLOCK_NOTE_PREFIX) and len(clock.notes) == 1
    rows = _ledger(root)
    assert [row["state"] for row in rows] == ["settled"]
    assert {row["candidate_raw_sha256"] for row in rows} == {_digest(candidate)} == {_sealed(rows[-1])}
    assert rows[-1]["candidate_raw_size_bytes"] == len(_canonical_candidate_bytes(candidate))
    if not asynchronous:  # the async driver runs in its own context copy; the sync Main send is the consumer
        assert clock.consumed_note() == clock.notes[0]


def test_the_clock_free_identity_is_the_candidate_without_its_line():
    payload = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    with sc.MainSendClock(MOSCOW).bound() as clock:
        stamped = sc.stamp_clock_note(payload)
        request = _attempt_request({"provider": "openai", "usage_model": "m"}, stamped)
    assert request.candidate_raw_sha256 == _digest(stamped)
    assert request.candidate_clock_free_sha256 == _digest(payload)
    assert clock.by_candidate == {request.candidate_raw_sha256: stamped["messages"][-1]["content"]}
    assert _attempt_request({"provider": "openai", "usage_model": "m"}, stamped).candidate_clock_free_sha256 is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_every_compatibility_attempt_refreshes_its_clock_and_preserves_recovery(tmp_path, monkeypatch, ticking, asynchronous):
    import ouroboros.request_wire_contract as wire
    from tests.test_request_wire_recovery_phase2b import _Rejected, _Response, _payload, _target

    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: tmp_path / "wire")
    client = LLMClient(api_key="unused")

    def run(payload, rejection):
        sent = []

        def create(**candidate):
            sent.append(copy.deepcopy(candidate))
            if len(sent) == 1:
                raise _Rejected(rejection)
            return _Response()

        from ouroboros.request_wire_recovery import request_wire_call_scope

        with request_wire_call_scope(), ua.usage_scope(ua.UsageScope(drive_root=tmp_path / "ledger", task_id="wire")), \
                sc.MainSendClock(MOSCOW).bound():
            if asynchronous:
                async def create_async(**candidate):
                    return create(**candidate)
                async def send_and_normalize():
                    response = await client._create_chat_completion_with_retries_async(
                        create_async, payload, _target(provider="openai"))
                    return client._normalize_remote_response(
                        response.model_dump(), _target(provider="openai"), skip_cost_fetch=True)[1]
                usage = asyncio.run(send_and_normalize())
            else:
                response = client._create_chat_completion_with_retries(create, payload, _target(provider="openai"))
                _message, usage = client._normalize_remote_response(
                    response.model_dump(), _target(provider="openai"), skip_cost_fetch=True)
        return sent, usage

    # A rejected attempt creates a new physical preparation; the canonical recovery source survives.
    sent, usage = run(_payload(effort="high"), "reasoning_effort value 'high' is not supported")
    assert [item["reasoning_effort"] for item in sent] == ["high", "medium"]
    assert usage["request_wire"]["applied_effort"] == "medium"  # the recovery action is retained
    first, second = (_clock_rows(item["messages"]) for item in sent)
    assert len(first) == len(second) == 1 and first != second
    # A carrier-less optional-field repair rebuilds from the logical request: a new sample.
    sent, _usage = run(_payload(effort="", toolful=False), "temperature is unsupported")
    assert "temperature" in sent[0] and "temperature" not in sent[1]
    first, second = (_clock_rows(item["messages"]) for item in sent)
    assert len(first) == len(second) == 1 and first != second


def test_the_local_lane_measures_and_sends_the_same_single_line(tmp_path, monkeypatch, ticking):
    import ouroboros.local_model as local_model
    from tests.test_physical_candidate_capture import _Response, _physical_context, _scope

    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **kw: 0.0)
    client = LLMClient(api_key="unused")
    measured, sent = [], {}

    class _Completions:
        @staticmethod
        def create(**candidate):
            sent.update(copy.deepcopy(candidate))
            return _Response(text="local")

    monkeypatch.setattr(client, "_get_local_client",
                        lambda: SimpleNamespace(chat=SimpleNamespace(completions=_Completions())))
    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(
        serving_context_evidence=lambda: {"context_window": 8192, "confirmed": True},
        measure_prepared_input=lambda payload: measured.append(copy.deepcopy(payload)) or {"supported": False}))
    with ua.usage_scope(_scope(root, "task-local")), ua.bind_physical_attempt_context(_physical_context()), \
            sc.MainSendClock(MOSCOW).bound() as clock:
        client._chat_local([{"role": "user", "content": "local input"}], None, max_tokens=512, tool_choice="auto")
    assert len(clock.notes) == 1 and measured[0]["messages"] == sent["messages"]
    assert sent["messages"][-1] == {"role": "user", "content": clock.notes[0]}
    rows = _ledger(root)
    assert {row["candidate_raw_sha256"] for row in rows} == {_digest(sent)}


def test_the_gigachat_lane_seals_its_line_before_measurement(tmp_path, monkeypatch, ticking):
    from tests.test_physical_candidate_capture import _scope

    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **kw: 0.0)
    client, sent = LLMClient(api_key="unused"), []
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", function_call=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=1, precached_prompt_tokens=0))
    monkeypatch.setattr(client, "_get_gigachat_client", lambda target, timeout=None: SimpleNamespace(
        chat=lambda candidate: sent.append(copy.deepcopy(candidate)) or completion))
    target = {"provider": "gigachat", "resolved_model": "GigaChat-2", "usage_model": "gigachat::GigaChat-2"}
    with ua.usage_scope(_scope(root, "task-giga")), sc.MainSendClock(MOSCOW).bound() as clock:
        client._chat_gigachat(target, [{"role": "user", "content": "привет"}], None, "high", 64, "auto")
    assert sent[0]["messages"][-1] == {"role": "user", "content": clock.notes[0]}
    assert {row["candidate_raw_sha256"] for row in _ledger(root)} == {_digest(sent[0])}


def _real_gigachat(monkeypatch, chat_statuses):
    """The installed ``gigachat`` SDK itself (pinned in requirements-runtime.lock), with only its
    two httpx transports replaced: every chat POST body and every auth POST is recorded."""
    import httpx

    from ouroboros.llm_gigachat import _GigaChatLaneMixin

    posts = {"chat": [], "auth": 0}
    statuses = iter(chat_statuses)

    def chat(request):
        posts["chat"].append(json.loads(request.content))
        status = next(statuses)
        body = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0, "finish_reason": "stop"}],
                "created": 1, "model": "GigaChat-2", "object": "chat.completion",
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}
        return httpx.Response(status, json=body if status == 200 else {"status": status, "message": "refused"})

    def auth(_request):
        posts["auth"] += 1
        return httpx.Response(200, json={"access_token": f"token-{posts['auth']}", "expires_at": 4_102_444_800_000})

    real = _GigaChatLaneMixin._new_gigachat_client
    built = []

    def new(target, timeout=None, max_retries=None):
        client = real(target, timeout=timeout, max_retries=max_retries)
        client._client_instance = httpx.Client(base_url=client._settings.base_url, transport=httpx.MockTransport(chat))
        client._auth_client_instance = httpx.Client(transport=httpx.MockTransport(auth))
        built.append(client)
        return client

    monkeypatch.setattr(LLMClient, "_new_gigachat_client", staticmethod(new))
    return posts, built


def test_the_gigachat_sdk_sends_each_accounted_attempt_once_despite_its_env_retries(tmp_path, monkeypatch, ticking):
    """``GIGACHAT_MAX_RETRIES`` would make the SDK re-send a 5xx on its own, unseen by the ledger
    (shown on a bare client); the lane passes ``max_retries=0``, the SDK's supported control."""
    import gigachat
    import httpx

    from gigachat.exceptions import ServerError
    from tests.test_physical_candidate_capture import _scope

    monkeypatch.setenv("GIGACHAT_MAX_RETRIES", "3")
    monkeypatch.setenv("GIGACHAT_RETRY_BACKOFF_FACTOR", "0")
    bare_posts = []
    bare = gigachat.GigaChat(access_token="t", base_url="https://giga.test/api/v1")
    bare._client_instance = httpx.Client(base_url="https://giga.test/api/v1", transport=httpx.MockTransport(
        lambda request: bare_posts.append(request) or httpx.Response(503, json={"message": "busy"})))
    with pytest.raises(ServerError):
        bare.chat({"model": "GigaChat-2", "messages": [{"role": "user", "content": "x"}]})
    assert bare._settings.max_retries == 3 and len(bare_posts) == 4  # what the environment alone would do

    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **kw: 0.0)
    posts, built = _real_gigachat(monkeypatch, [503])
    target = {"provider": "gigachat", "resolved_model": "GigaChat-2", "usage_model": "gigachat::GigaChat-2",
              "access_token": "t", "base_url": "https://giga.test/api/v1"}
    with ua.usage_scope(_scope(root, "task-giga")), sc.MainSendClock(MOSCOW).bound():
        with pytest.raises(ServerError):
            LLMClient(api_key="unused")._chat_gigachat(target, [{"role": "user", "content": "привет"}], None,
                                                       "high", 64, "auto")
    assert built[0]._settings.max_retries == 0 and len(posts["chat"]) == 1
    assert [row["state"] for row in _ledger(root)] == ["unresolved"]


def test_the_gigachat_sdk_401_resend_repeats_the_sealed_attempt_and_claims_no_fresh_clock(
        tmp_path, monkeypatch, ticking):
    """The one hidden re-send the SDK has no switch for, qualified exactly: with a token it
    believed usable, a 401 resets it, re-authenticates and sends the SAME body again — inside
    the one accounted attempt, with the clock line sampled before the first send."""
    from tests.test_physical_candidate_capture import _scope

    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **kw: 0.0)
    posts, _built = _real_gigachat(monkeypatch, [200, 401, 200])
    target = {"provider": "gigachat", "resolved_model": "GigaChat-2", "usage_model": "gigachat::GigaChat-2",
              "api_key": "credentials", "base_url": "https://giga.test/api/v1"}
    client = LLMClient(api_key="unused")
    with ua.usage_scope(_scope(root, "task-giga")):
        for text in ("first", "second"):
            with sc.MainSendClock(MOSCOW).bound():
                client._chat_gigachat(target, [{"role": "user", "content": text}], None, "high", 64, "auto")
    assert posts["auth"] == 2 and len(posts["chat"]) == 3
    first, refused, resent = posts["chat"]
    assert refused == resent and refused["messages"][0]["content"] == "second"
    assert refused["messages"][-1]["content"] == resent["messages"][-1]["content"] != first["messages"][-1]["content"]
    rows = _ledger(root)
    assert [row["state"] for row in rows] == ["settled"] * 2  # two attempts, not three


def test_every_claudexor_invocation_samples_anew_and_uploads_its_sealed_bytes(main_call, ticking):
    from tests.test_subscription_main_wait import _dispatch, _failed

    ctx, gateway, _controller, _events, _decide, _observations = main_call
    ctx.messages[:] = ctx.messages[:2]
    gateway.results = [_failed("subscription_window_exhausted"), gateway.results[0]]
    gateway.dispatch = ["not_started", "response_received"]
    answer, _cost, _mode = _dispatch(ctx)
    assert answer and len(gateway.uploads) == 2
    lines = [upload[0]["messages"][-1]["content"] for upload in gateway.uploads]
    assert all(line.startswith(sc.CLOCK_NOTE_PREFIX) for line in lines) and lines[0] != lines[1]
    rows = _ledger(ctx.drive_root)  # one current row per attempt; a dispatched one carries its manifest
    for upload, row in zip(gateway.uploads, [r for r in rows if r.get("candidate_manifest_ref")]):
        assert {r["candidate_raw_sha256"] for r in rows if r["attempt_id"] == row["attempt_id"]} == {
            _digest(upload[0]), _sealed(row)}
    # The consumed (second) line joined the canonical transcript; the refused one did not.
    assert ctx.messages[-1]["content"] == lines[1] and lines[0] not in json.dumps(ctx.messages)


# --- Main's canonical replay and scope ---------------------------------------------------


@pytest.fixture
def unstreamed(monkeypatch):
    """The fixture's provider answers whole; Main's ``stream=True`` send stays byte-identical."""
    monkeypatch.setattr("ouroboros.llm_fallback.consume_stream", lambda response, **_kw: response)


def test_a_routing_act_recorded_during_the_turn_rides_the_next_real_request(main_call, ticking):
    """Through the real Main round and Claudexor lane: the receipts note is part of the one
    candidate measured, priced, sealed and sent, before that send's clock line."""
    from ouroboros import loop
    from ouroboros.loop_model_call import ROUTING_RECEIPTS_HEADER
    from ouroboros.project_dialogue import append_chat_annotation
    from tests.test_llm_claudexor import ledger

    ctx, gateway, _controller, _events, _decide, _observations = main_call
    ctx.tools._ctx.task_metadata = {"client_message_id": "cm-7", "routing_contract": {"llm_first": True}}
    append_chat_annotation(ctx.drive_root, "cm-7", action="steer_task", target="root-b", target_label="Build",
                           status="delivered", routing_token="tok-1")
    answer, _cost, _mode = loop._call_round_model(ctx)
    assert answer and len(gateway.uploads) == 1
    sent = gateway.uploads[0][0]
    from tests.test_subscription_main_wait import _without_context_facts

    protocol = _without_context_facts(sent["messages"], physical=True)
    texts = [json.dumps(message["content"], ensure_ascii=False) for message in protocol]
    assert ROUTING_RECEIPTS_HEADER in texts[-2] and "steer_task → Build (root-b): delivered" in texts[-2]
    assert sent["messages"][-1]["content"].startswith(sc.CLOCK_NOTE_PREFIX)
    settled = [row for row in ledger(ctx.drive_root) if row["state"] == "settled"][-1]
    assert settled["candidate_raw_sha256"] == _digest(sent)
    # Canonical history (the loop appends the answer next): the note, then the consumed clock line.
    assert _without_context_facts(ctx.messages)[-2]["content"].startswith(ROUTING_RECEIPTS_HEADER)
    assert ctx.messages[-1]["content"] == sent["messages"][-1]["content"]


def test_the_consumed_line_makes_each_request_extend_the_last_byte_for_byte(transport, ticking, unstreamed):  # noqa: F811
    root, client, sent = transport
    logs = root / "logs"
    logs.mkdir()
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "first"}]
    usage: dict = {}
    for round_idx in (1, 2):
        msg, _cost = call_llm_with_retry(client, messages, "openai::test-model", None, "high", 1, logs, "main-task",
                                         round_idx, queue.Queue(), usage, send_clock_policy=MOSCOW)
        assert msg is not None, {k: v for k, v in usage.items() if "error" in k}
        messages.append({"role": "assistant", "content": msg["content"]})
        messages.append({"role": "user", "content": f"after round {round_idx}"})
    first, second = sent[0]["messages"], sent[1]["messages"]
    assert second[:len(first)] == first  # the previous request is an exact prefix of the next
    assert len(_clock_rows(first)) == 1 and len(_clock_rows(second)) == 2
    assert not _clock_rows(messages[:2]) and _clock_rows(messages) == [first[-1], second[-1]]


def test_a_child_or_helper_call_sends_no_clock_line(transport, unstreamed):  # noqa: F811
    root, client, sent = transport
    (root / "logs").mkdir()
    messages = [{"role": "user", "content": "child work"}]
    call_llm_with_retry(client, messages, "openai::test-model", None, "high", 1, root / "logs", "child", 1,
                        queue.Queue(), {}, send_clock_policy=sc.main_clock_policy({"delegation_role": "subagent"}))
    assert not _clock_rows(sent[0]["messages"]) and messages == [{"role": "user", "content": "child work"}]


# --- the forced final is admitted on its fresh request ------------------------------------


@pytest.mark.parametrize("model,env", [("openai::gpt-test", {"OPENAI_API_KEY": "unused"}),
                                       ("anthropic::claude-test", {"ANTHROPIC_API_KEY": "unused"})])
def test_the_forced_final_admits_the_fresh_clocked_send_without_drift(monkeypatch, tmp_path, ticking, model, env):
    from ouroboros import llm as llm_module, task_pacing
    from ouroboros.llm_claudexor import cache_key_for_model
    from ouroboros import loop_forced_finalization as forced
    from tests.test_tree_cost_ceiling import _patch_execute_candidate
    from tests.test_wrapup_real_send_parity import _Captured, _MESSAGES, _TOOLS

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True, raising=False)
    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", lambda *_a, **_kw: {})
    captured = {}

    def execute(request, send, before_dispatch):
        captured["request"] = request
        raise _Captured()

    _patch_execute_candidate(monkeypatch, llm_module, execute)
    client = LLMClient(api_key="unused")
    (tmp_path / "logs").mkdir()
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="forced", root_task_id="forced")):
        with sc.MainSendClock(MOSCOW).bound():  # as ``prepared_wrapup_candidate`` binds it
            admitted = task_pacing.prospective_wrapup_attempt_request(
                llm=client, messages=_MESSAGES, model=model, reasoning_effort="high", tools=_TOOLS,
                cache_affinity=cache_key_for_model(model))
        try:  # the lane stops at the executor seam, once the physical request exists
            call_llm_with_retry(client, _MESSAGES, model, _TOOLS, "high", 1, tmp_path / "logs", "forced", 1,
                                queue.Queue(), {}, initial_messages=_MESSAGES, send_clock_policy=MOSCOW)
        except _Captured:
            pass
    actual = captured["request"]
    # Two samples, one candidate: different raw bytes, same size and clock-free identity.
    assert actual.candidate_raw_sha256 != admitted.candidate_raw_sha256
    assert actual.candidate_raw_size_bytes == admitted.candidate_raw_size_bytes
    assert actual.candidate_clock_free_sha256 == admitted.candidate_clock_free_sha256 is not None
    predicate = _forced_predicate(forced, admitted)
    assert predicate(actual) is True
    changed = SimpleNamespace(**{**vars(actual), "candidate_clock_free_sha256": "0" * 64})
    assert predicate(changed) is False  # a genuinely different candidate still drifts


def _tokenizer_sees_digits(monkeypatch):
    """Equal width is not equal tokens: this estimator charges 1000 more for one minute's digits.

    Wraps the real ``_attempt_request`` everywhere a candidate is measured (the OpenAI
    lane and the wrap-up lookahead); identity facts stay the real ones.
    """
    from dataclasses import replace

    from ouroboros import llm, llm_fallback

    real = llm_fallback._attempt_request

    def measured(target, payload, **kwargs):
        request = real(target, payload, **kwargs)
        line = next((row.get("content") for row in reversed(payload.get("messages") or [])
                     if isinstance(row, dict) and str(row.get("content") or "").startswith(sc.CLOCK_NOTE_PREFIX)), "")
        return replace(request, prompt_tokens_estimate=request.prompt_tokens_estimate + (1000 if "14:24:05" in line else 0))

    monkeypatch.setattr(llm_fallback, "_attempt_request", measured)
    monkeypatch.setattr(llm, "_attempt_request", measured)
    # One price per token for the fence and the admission alike (the same shared function).
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: round(request.prompt_tokens_estimate * 1e-4, 6))


def _forced_ctx(client, root, model):
    from ouroboros import loop

    return SimpleNamespace(
        tools=SimpleNamespace(_ctx=SimpleNamespace(task_metadata={}, context_fit_plan=None, model_turn_state=None)),
        llm=client, messages=[{"role": "user", "content": "finish the report"}], active_model=model,
        tool_schemas=None, active_effort="high", max_retries=1, drive_logs=root / "logs", task_id="processing",
        round_idx=3, event_queue=queue.Queue(), accumulated_usage={}, task_type="task", active_use_local=False,
        deadline_ts=None, model_role=""), loop


@pytest.mark.parametrize("room_usd,admitted", [(1.0, True), (0.05, False)])
def test_the_forced_final_is_admitted_on_the_fresh_requests_own_price(  # noqa: F811
        transport, monkeypatch, ticking, unstreamed, room_usd, admitted):  # noqa: F811
    """The lookahead is advice: the FRESH request — the bytes measured, priced, sealed and sent —
    is admitted against the same balances at its own price. Here its re-sampled clock costs
    1000 more estimated tokens at the same width; when that no longer fits, the send is refused
    unsent and NOT handed to the drift rail's unpredicated resend."""
    from ouroboros import loop_forced_finalization as forced, task_pacing
    from ouroboros.llm_claudexor import cache_key_for_model

    root, client, sent = transport
    _tokenizer_sees_digits(monkeypatch)
    model = "openai::gpt-test"
    ctx, loop = _forced_ctx(client, root, model)
    monkeypatch.setattr(loop, "_server_web_allowed_by_task", lambda _ctx: False)
    (root / "logs").mkdir(exist_ok=True)
    with sc.MainSendClock(sc.SendClockPolicy()).bound():  # as ``prepared_wrapup_candidate`` binds it
        lookahead = task_pacing.prospective_wrapup_attempt_request(
            llm=client, messages=ctx.messages, model=model, reasoning_effort="high", tools=None,
            cache_affinity=cache_key_for_model(model))
    admission = {"root_cap_usd": room_usd, "deciding_usd": 0.0, "global_remaining_usd": None}
    assert task_pacing.wrapup_reservation_fits(request=lookahead, **admission) is True  # the advice said yes
    drifts = []
    monkeypatch.setattr(forced.log, "warning", lambda *args, **_kw: drifts.append(args))
    if admitted:
        text = forced._send_admitted_forced_candidate(ctx, ctx.messages, lookahead, "budget_exhausted",
                                                      admission=admission)
        assert text == "ok" and len(sent) == 1 and drifts == []
        rows = _ledger(root)
        fresh = rows[-1]  # the one attempt's current row keeps its reservation bound
        body = {key: value for key, value in sent[0].items() if key != "timeout"}  # an SDK option, not the body
        assert {row["candidate_raw_sha256"] for row in rows} == {_digest(body)} == {_sealed(rows[-1])}
        # The fence reserved the FRESH price (1000 more estimated tokens), not the lookahead's.
        assert fresh["reservation_upper_bound_usd"] == pytest.approx(lookahead.prompt_tokens_estimate * 1e-4 + 0.1)
        assert sent[0]["messages"][-1]["content"].startswith(sc.CLOCK_NOTE_PREFIX + "2026-09-26T14:24:05Z")
    else:
        with pytest.raises(forced.ForcedCandidateUnaffordable):
            forced._send_admitted_forced_candidate(ctx, ctx.messages, lookahead, "budget_exhausted",
                                                   admission=admission)
        assert sent == [] and drifts == []  # refused before a byte left; no unpredicated resend
        assert [row["state"] for row in _ledger(root)] == ["released"]


def test_a_refused_fresh_forced_candidate_lands_on_the_unaffordable_fallback(monkeypatch):
    from ouroboros import loop, loop_forced_finalization as forced

    def refuse(*_args, **_kwargs):
        raise forced.ForcedCandidateUnaffordable("does not fit")

    checkpoints, fallbacks = [], []
    monkeypatch.setattr(loop, "_call_forced_model_once", refuse)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *args: checkpoints.append(args[-1]))
    monkeypatch.setattr(loop, "_forced_fallback_result",
                        lambda _ctx, _trace, text, reason, **kw: fallbacks.append((text, reason, kw)) or (text, {}, {}))
    ctx = SimpleNamespace(llm_trace={}, deadline_ts=None, messages=[], accumulated_usage={}, event_queue=None,
                          task_id="t", drive_logs=None, tools=None)
    text, _usage, _trace = forced._forced_final_answer(
        ctx, prompt="finish", fallback_text="Budget exhausted.", reason_code="budget_exhausted",
        _prompt_prepared=True, _initial_messages=[], _admitted_request=object(), _admission={"root_cap_usd": 1.0})
    assert text == "Budget exhausted." and checkpoints == [{"checkpoint_kind": "forced_candidate_unaffordable",
                                                            "reason_code": "budget_exhausted"}]
    assert fallbacks[0][2]["source"] == "budget_wrapup_unaffordable"


def _forced_predicate(forced, admitted):
    """The predicate ``_call_forced_model_once`` hands to its send, read off a real call."""
    seen = {}

    def call(*_args, **kwargs):
        seen.update(kwargs)
        return {"content": ""}, 0.0

    ctx = SimpleNamespace(
        tools=SimpleNamespace(_ctx=SimpleNamespace(task_metadata={}, context_fit_plan=None, model_turn_state=None)),
        llm=None, messages=[], active_model="m", tool_schemas=[], active_effort="high", max_retries=1,
        drive_logs=None, task_id="forced", round_idx=1, event_queue=None, accumulated_usage={}, task_type="task",
        active_use_local=False, deadline_ts=None, model_role="")
    original = forced._loop().call_llm_with_retry
    forced._loop().call_llm_with_retry = call
    try:
        forced._call_forced_model_once(ctx, admitted_request=admitted)
    finally:
        forced._loop().call_llm_with_retry = original
    assert seen["send_clock_policy"] == sc.SendClockPolicy()
    return seen["candidate_predicate"]


from tests.test_llm_claudexor import setup as _gateway_setup  # noqa: E402
from tests.test_model_wait import live_wait as _live_wait  # noqa: E402
from tests.test_subscription_main_wait import main_call as _main_call  # noqa: E402
from tests._usage_store_testing import ledger_rows

setup = _gateway_setup
live_wait = _live_wait
main_call = _main_call


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_anthropic_compatibility_retry_has_a_fresh_sealed_clock(transport, ticking, monkeypatch, asynchronous):  # noqa: F811
    import requests
    from tests.test_anthropic_native_custody import _NativeResponse, _target

    root, client, _ = transport
    sent = []
    def post(*_args, **kwargs):
        sent.append(copy.deepcopy(kwargs["json"]))
        return _NativeResponse(reject=len(sent) == 1)
    monkeypatch.setattr(requests, "post", post)
    with sc.MainSendClock(MOSCOW).bound() as clock:
        if asynchronous:
            message, usage = asyncio.run(client.chat_async(
                [{"role": "user", "content": "hi"}], "anthropic::test-model", reasoning_effort="none", max_tokens=64))
        else:
            message, usage = client._chat_anthropic(_target(), [{"role": "user", "content": "hi"}], None,
                                                   "none", 64, "auto")
    assert message["content"] == "ok" and len(sent) == 2
    assert "thinking" in sent[0] and "thinking" not in sent[1]
    assert len(clock.notes) == 2 and clock.notes[0] != clock.notes[1]
    for payload, note in zip(sent, clock.notes):
        assert payload["messages"][-1]["content"][-1] == {"type": "text", "text": note}
    rows = _ledger(root)  # one current row per attempt
    assert len(rows) == 2
    assert [r["candidate_raw_sha256"] for r in rows] == [_digest(p) for p in sent]
    assert [_sealed(r) for r in _ledger(root) if r.get("candidate_manifest_ref")] == [_digest(p) for p in sent]
    assert usage["request_wire"]["applied_effort"] == "provider_default"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("dialect_first", [False, True])
def test_clock_refresh_preserves_composed_tool_dialect_and_effort_recovery(
        tmp_path, monkeypatch, ticking, asynchronous, dialect_first):
    import ouroboros.request_wire_contract as wire
    from ouroboros.request_wire_recovery import current_wire_candidate, request_wire_call_scope
    from tests.test_openai_chat_dispatch import _DialectError
    from tests.test_request_wire_recovery_phase2b import _Rejected, _Response, _payload, _target

    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: tmp_path / "wire")
    client, sent, catalogs = LLMClient(api_key="unused"), [], []
    source = _payload(effort="high")
    errors = [_DialectError("custom tools are not supported", param="tools[0].type"),
              _Rejected("reasoning_effort value 'high' is not supported")]
    if not dialect_first:
        errors.reverse()

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        catalogs.append(current_wire_candidate().custom_catalog_sha256)
        if len(sent) <= len(errors):
            raise errors[len(sent) - 1]
        return _Response()

    async def create_async(**candidate):
        return create(**candidate)

    def normalize(response):
        return client._normalize_remote_response(response.model_dump(), _target(provider="openai"), skip_cost_fetch=True)[1]

    async def asynchronous_send():
        return normalize(await client._create_chat_completion_with_retries_async(create_async, source, _target(provider="openai")))

    with request_wire_call_scope(), ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="composed-clock")), \
            sc.MainSendClock(MOSCOW).bound() as clock:
        usage = (asyncio.run(asynchronous_send()) if asynchronous else
                 normalize(client._create_chat_completion_with_retries(create, source, _target(provider="openai"))))
        assert len(clock.notes) == len(set(clock.notes)) == 3
        assert [sc.split_clock_note(candidate)[0] for candidate in sent] == clock.notes
    assert [wire.infer_tool_dialect(p) for p in sent] == (
        ["openai_chat_custom", "function", "function"] if dialect_first else ["openai_chat_custom", "openai_chat_custom", "function"])
    # The existing dialect owner restarts requested effort on function tools:
    # a custom-profile rejection is not authority for the function profile.
    assert [wire.payload_effort(p) for p in sent] == (["high", "high", "medium"] if dialect_first else ["high", "medium", "high"])
    assert catalogs[0] and (dialect_first or catalogs[0] == catalogs[1])
    assert sent[-1]["tools"] == source["tools"] and source["messages"] == [{"role": "user", "content": "probe"}]
    assert usage["request_wire"]["applied_effort"] == ("medium" if dialect_first else "high")
    assert usage["request_wire"]["candidate_sha256"] == _digest(sent[-1])
    assert [_sealed(row) for row in _ledger(tmp_path) if row.get("candidate_manifest_ref")] == [_digest(p) for p in sent]
