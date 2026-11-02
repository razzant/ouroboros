"""#1578: a provider's raw counter presence survives its adapter into the v2 rows.

Recorded raw provider bodies are replayed through the REAL adapters on the offline
golden transports (``tests/test_llm_provider_golden.py``), then through the real
loop round, ``emit_llm_usage_event`` and the supervisor's ``llm_usage`` handler. A
fake LLM that already returns normalized usage cannot see this boundary: the
Anthropic/GigaChat adapters used to synthesize missing counters as 0, and the
OpenAI-compatible adapter left a reported nested cache 0 behind its detail block.
Billing-facing usage stays byte-identical; the golden replays pin that.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import queue
import socket
from types import SimpleNamespace

import pytest

from tests import test_llm_provider_golden as golden

COUNTERS = ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens")


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """The golden isolation (env, data root, class caches) with any socket use failing the test."""
    from ouroboros import config
    from ouroboros.llm import LLMClient

    def forbidden(*_args, **_kwargs):
        pytest.fail("provider presence test crossed the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for name in golden._ROUTE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(data / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    for name in golden._CLASS_CACHE_NAMES:
        monkeypatch.setattr(LLMClient, name, type(getattr(LLMClient, name))())
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True)
    monkeypatch.setattr(LLMClient, "_CAPABILITIES_FETCH_OK", True)
    return data


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _persisted_usage(event, root):
    from supervisor.events_budget import _handle_llm_usage

    _handle_llm_usage(event, SimpleNamespace(DRIVE_ROOT=root))
    [row] = [row for row in _rows(root / "logs" / "events.jsonl") if row["type"] == "llm_usage"]
    assert row["record_contract_version"] == 2
    return [row[key] for key in COUNTERS]


def _sse(frames):
    return b"".join((f"event: {event}\n" if event else "").encode() + b"data: "
                    + json.dumps(data).encode() + b"\n\n" for event, data in frames)


class _StreamedNative(golden._FakeRequestsResponse):
    """The recorded native body served as Anthropic SSE frames: the main loop always streams."""

    headers: dict = {}

    def iter_content(self, chunk_size=8192):
        body, usage = self._json, dict(self._json.get("usage") or {})
        output = usage.pop("output_tokens", None)
        start = {**{key: value for key, value in body.items() if key not in {"content", "stop_reason", "usage"}},
                 "content": [], "stop_reason": None, "usage": usage}
        frames = [("message_start", {"type": "message_start", "message": start})]
        for index, block in enumerate(body.get("content") or []):
            frames += [
                ("content_block_start", {"type": "content_block_start", "index": index,
                                         "content_block": {"type": "text", "text": ""}}),
                ("content_block_delta", {"type": "content_block_delta", "index": index,
                                         "delta": {"type": "text_delta", "text": block["text"]}}),
                ("content_block_stop", {"type": "content_block_stop", "index": index})]
        frames += [("message_delta", {"type": "message_delta", "delta": {"stop_reason": body["stop_reason"]},
                                      "usage": {} if output is None else {"output_tokens": output}}),
                   ("message_stop", {"type": "message_stop"})]
        yield _sse(frames)

    def close(self):
        return None


class _StreamedChat(golden._FakeResponse):
    """The recorded chat completion served as chunk frames ending in the usage frame."""

    headers: dict = {}

    @property
    def response(self):
        return self

    def iter_bytes(self):
        body = self._body
        frames = []
        for choice in body.get("choices") or []:
            message = choice.get("message") or {}
            frames += [("", {"id": body["id"], "object": "chat.completion.chunk", "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": message.get("content")}}]}),
                ("", {"id": body["id"], "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]})]
        frames.append(("", {"id": body["id"], "object": "chat.completion.chunk", "choices": [],
                            "usage": body.get("usage")}))
        yield _sse(frames) + b"data: [DONE]\n\n"

    def close(self):
        return None


def _loop_round(tmp_path, monkeypatch, model, env, step, *, round_idx=1, accumulated=None):
    """One real main-loop round over a recorded provider body; returns llm_round,
    persisted llm_usage counters and the adapter usage the loop received."""
    from ouroboros.llm import LLMClient
    from ouroboros.loop_llm_call import call_llm_with_retry

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(golden, "_FakeRequestsResponse", _StreamedNative)
    monkeypatch.setattr(golden, "_FakeResponse", _StreamedChat)
    recorder, events, logs = golden._Recorder([step]), queue.Queue(), tmp_path / "logs"
    with contextlib.ExitStack() as stack:
        golden._install_fakes(stack, recorder, {"pricing_estimate": 0.001})
        message, _cost = contextvars.Context().run(
            call_llm_with_retry, LLMClient(), [{"role": "user", "content": "hi"}], model, None,
            "low", 1, logs, "presence", round_idx, events, {} if accumulated is None else accumulated, "task", False)
    assert message and recorder.script == [] and len(recorder.sends) == 1
    [round_row] = [row for row in _rows(logs / "events.jsonl") if row["type"] == "llm_round"]
    assert round_row["record_contract_version"] == 2
    usage_events = []
    while not events.empty():
        event = events.get_nowait()
        if event.get("type") == "llm_usage":
            usage_events.append(event)
    [usage_event] = usage_events
    persisted = _persisted_usage(usage_event, tmp_path / "supervisor")
    return [round_row[key] for key in COUNTERS], persisted, usage_event["usage"]


ANTHROPIC = {"ANTHROPIC_API_KEY": "anthropic-fixture-key"}


@pytest.mark.parametrize("raw,expected", [
    # The golden fixture's shape: no cache counters, so the total prompt is unknown.
    ({"input_tokens": 20, "output_tokens": 3}, [None, 3, None, None]),
    ({"input_tokens": 7, "output_tokens": 0, "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0}, [7, 0, 0, 0]),
    ({"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 5}, [None, 2, 5, None]),
    ({"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": None,
      "cache_creation_input_tokens": None}, [None, 2, None, None]),
    ({"input_tokens": 100, "output_tokens": 6, "cache_read_input_tokens": 40,
      "cache_creation_input_tokens": 25}, [165, 6, 40, 25]),
])
def test_anthropic_native_presence_reaches_round_and_usage_rows(tmp_path, monkeypatch, offline, raw, expected):
    step = {"kind": "response", "status_code": 200, "json": {
        "id": "msg_presence", "role": "assistant", "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn", "usage": raw}}
    round_counts, usage_counts, usage = _loop_round(
        tmp_path, monkeypatch, "anthropic::claude-sonnet-4.6", ANTHROPIC, step)
    assert round_counts == usage_counts == expected
    # Billing-facing counters keep their historical synthesis; only the event fact is new.
    parts = [int(raw.get(key) or 0) for key in ("input_tokens", "cache_read_input_tokens",
                                                 "cache_creation_input_tokens")]
    assert [usage[key] for key in COUNTERS] == [sum(parts), int(raw.get("output_tokens") or 0), parts[1], parts[2]]


def _switch_facts(accumulated, model, round_idx):
    """The real ``[ROUTE FACTS]`` renderer over the usage one real adapter round left behind."""
    from ouroboros.primary_route_observation import route_facts_text

    owner = SimpleNamespace(primary_route={"model": "claudexor::codex=gpt-test", "use_local": False, "role": "main"})
    return route_facts_text(owner, model=model, use_local=False, role="fallback:0", failed_model="claudexor::codex=gpt-test",
                            failure={"failure_code": "provider_outcome_unknown"}, waiter=None, usage=accumulated,
                            round_idx=round_idx)


@pytest.mark.parametrize("raw,shown", [
    ({"input_tokens": 20, "output_tokens": 3}, "input unknown, cache read unknown, cache write unknown"),
    ({"input_tokens": 100, "output_tokens": 6, "cache_read_input_tokens": 40, "cache_creation_input_tokens": 25},
     "input 165, cache read 40, cache write 25"),
], ids=["cache-fields-absent", "fully-reported"])
def test_switch_facts_show_the_answering_calls_tokens_as_reported(tmp_path, monkeypatch, offline, raw, shown):
    """#1580 r2: the real Anthropic adapter keeps billing-shaped 20/0/0 for cache routing; the switch facts
    read that same call's reported counters, so an absent cache field stays unknown, never a 0."""
    accumulated = {}
    step = {"kind": "response", "status_code": 200, "json": {
        "id": "msg_facts", "role": "assistant", "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn", "usage": raw}}
    _loop_round(tmp_path, monkeypatch, "anthropic::claude-sonnet-4.6", ANTHROPIC, step, round_idx=2,
                accumulated=accumulated)
    parts = [int(raw.get(key) or 0) for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")]
    recorded = accumulated["_last_round_cache_usage"]
    assert [recorded[key] for key in ("prompt_tokens", "cached_tokens", "cache_write_tokens")] == [sum(parts), *parts[1:]]
    facts = _switch_facts(accumulated, "anthropic::claude-sonnet-4.6", 2)
    assert f"; tokens {shown}; tariff unknown." in facts
    if "input_tokens" in raw and "cache_read_input_tokens" not in raw:
        assert "input 20" not in facts and "cache read 0" not in facts and "cache write 0" not in facts


def test_switch_facts_never_size_a_tariff_tier_by_an_unreported_input(tmp_path, monkeypatch, offline):
    """The real OpenAI-compatible adapter: a tiered cached tariff is sized only by the answer's reported
    input, so an unreported count leaves the tier unmeasured and a reported one selects its own tier."""
    import time

    from ouroboros import pricing
    from ouroboros.provider_models import normalize_model_identity

    key = ("openrouter", normalize_model_identity("x-ai/grok-4.5"))
    tiered = pricing.PricingSchedule((1.0, 0.1, 1.25, 5.0), tiers=((200_000, (2.0, 0.2, 2.5, 10.0)),),
                                     source={"service_tier": "default"})
    monkeypatch.setitem(pricing._cached_pricing, key, {"endpoint-a": tiered})
    monkeypatch.setitem(pricing._pricing_fetched_at, key, time.time())
    reported = {}
    for raw, round_idx in (({"completion_tokens": 10}, 1), ({"prompt_tokens": 250_000, "completion_tokens": 10}, 2)):
        accumulated = {}
        step = {"kind": "response", "body": {
            "id": f"gen-tier-{round_idx}", "choices": [{"index": 0, "finish_reason": "stop",
                                                        "message": {"role": "assistant", "content": "ok"}}],
            "usage": raw}}
        _loop_round(tmp_path / str(round_idx), monkeypatch, "x-ai/grok-4.5", {"OPENROUTER_API_KEY": "or-fixture-key"},
                    step, round_idx=round_idx, accumulated=accumulated)
        reported[round_idx] = (accumulated["_last_round_cache_usage"], _switch_facts(accumulated, "x-ai/grok-4.5", round_idx))
    unreported, facts = reported[1]
    assert "tokens input unknown, cache read unknown, cache write unknown; tariff unknown (input tier unmeasured)." in facts
    assert "$1.00/M" not in facts and "$2.00/M" not in facts
    assert unreported["reported_tokens"]["prompt_tokens"] is None
    _measured, facts = reported[2]  # a reported input selects its own tier
    assert ("tokens input 250,000, cache read unknown, cache write unknown; cached catalog tariff $2.00/M in, "
            "$10.00/M out, cache read $0.20/M (estimated input tier).") in facts


@pytest.mark.parametrize("raw,expected", [
    ({"prompt_tokens": 11, "completion_tokens": 2}, [11, 2, None, None]),
    ({"prompt_tokens": 11, "completion_tokens": 2, "precached_prompt_tokens": 0}, [11, 2, 0, None]),
    ({"prompt_tokens": 50, "completion_tokens": 7, "precached_prompt_tokens": 10}, [50, 7, 10, None]),
    (None, [None, None, None, None]),  # the library returned no usage object at all
])
def test_gigachat_presence_reaches_round_and_usage_rows(tmp_path, monkeypatch, offline, raw, expected):
    body = {"message": {"content": "done"}, **({"usage": raw} if raw is not None else {})}
    round_counts, usage_counts, usage = _loop_round(
        tmp_path, monkeypatch, "gigachat::GigaChat-3-Pro", {"GIGACHAT_CREDENTIALS": "gigachat-fixture-key"},
        {"kind": "response", "body": body})
    assert round_counts == usage_counts == expected
    raw = raw or {}
    assert [usage[key] for key in ("prompt_tokens", "completion_tokens", "cached_tokens")] == [
        int(raw.get(key) or 0) for key in ("prompt_tokens", "completion_tokens", "precached_prompt_tokens")]


@pytest.mark.parametrize("raw,expected", [
    ({"prompt_tokens": 200, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}},
     [200, 10, 0, None]),
    ({"prompt_tokens": 200, "completion_tokens": 10,
      "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}}, [200, 10, 0, 0]),
    ({"prompt_tokens": 120, "completion_tokens": 8, "prompt_tokens_details": {"cached_tokens": 40}},
     [120, 8, 40, None]),
    ({"prompt_tokens": 200, "completion_tokens": 10}, [200, 10, None, None]),
    # A provider extension cannot supply the host's event fact.
    ({"prompt_tokens": 200, "completion_tokens": 10, "reported_tokens": dict.fromkeys(COUNTERS, 999)},
     [200, 10, None, None]),
])
def test_openai_compatible_reported_nested_zero_reaches_round_and_usage_rows(
    tmp_path, monkeypatch, offline, raw, expected,
):
    step = {"kind": "response", "body": {
        "id": "gen-presence", "choices": [{"index": 0, "finish_reason": "stop",
                                           "message": {"role": "assistant", "content": "ok"}}],
        "usage": raw}}
    round_counts, usage_counts, usage = _loop_round(
        tmp_path, monkeypatch, "x-ai/grok-4.5", {"OPENROUTER_API_KEY": "or-fixture-key"}, step)
    assert round_counts == usage_counts == expected
    assert "cached_tokens" not in usage or usage["cached_tokens"] == 40  # only a positive count is lifted


@pytest.mark.parametrize("raw,expected", [
    ({"input_tokens": 9, "output_tokens": 4}, [None, 4, None, None]),
    ({"input_tokens": 9, "output_tokens": 4, "cache_read_input_tokens": 300,
      "cache_creation_input_tokens": 0}, [309, 4, 300, 0]),
])
def test_anthropic_web_search_leg_reports_the_native_total(tmp_path, monkeypatch, offline, raw, expected):
    """The provider-owned web search reads raw native usage outside the adapter: its uncached
    ``input_tokens`` is not the total prompt the v2 row claims."""
    from ouroboros.tools.search import _web_search_anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-fixture-key")
    recorder = golden._Recorder([{"kind": "response", "body": {"usage": raw}}])
    ctx = SimpleNamespace(task_id="search", task_metadata={}, drive_root=offline, pending_events=[],
                          emit_progress_fn=None)
    with contextlib.ExitStack() as stack:
        golden._install_fakes(stack, recorder, {"pricing_estimate": 0.001})
        contextvars.Context().run(_web_search_anthropic, ctx, "presence", model="anthropic::claude-sonnet-4-6")
    [event] = [event for event in ctx.pending_events if event["type"] == "llm_usage"]
    assert [event[key] for key in COUNTERS] == expected
    assert _persisted_usage(event, tmp_path / "supervisor") == expected


@pytest.mark.parametrize("raw,expected", [
    ({"input_tokens": 3, "output_tokens": float("inf"), "cache_read_input_tokens": float("nan"),
      "cache_creation_input_tokens": 0}, [None, None, None, 0]),
    ({"input_tokens": -1, "output_tokens": True, "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 2.5}, [None, None, 0, None]),
    ({"input_tokens": 4.0, "output_tokens": 1, "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0}, [4, 1, 0, 0]),
    ("not a usage object", [None, None, None, None]),
])
def test_anthropic_counts_keep_each_counter_independent_of_bad_neighbours(raw, expected):
    from ouroboros.event_usage import anthropic_token_counts

    assert [anthropic_token_counts(raw)[key] for key in COUNTERS] == expected


@pytest.mark.parametrize("usage,expected", [
    # Detail blocks keep a reported zero the adapters do not lift (OpenAI chat / Responses, DeepSeek).
    ({"prompt_tokens": 5, "prompt_tokens_details": {"cached_tokens": 0}}, [5, None, 0, None]),
    ({"input_tokens": 5, "input_tokens_details": {"cached_tokens": 0}}, [5, None, 0, None]),
    ({"prompt_tokens": 5, "prompt_cache_hit_tokens": 0}, [5, None, 0, None]),
    ({"prompt_tokens": 5, "prompt_tokens_details": {"cache_creation_tokens": 0}}, [5, None, None, 0]),
    ({"prompt_tokens": 5, "prompt_tokens_details": None}, [5, None, None, None]),
    ({"prompt_tokens": 5, "cached_tokens": 7, "prompt_tokens_details": {"cached_tokens": 0}}, [5, None, 7, None]),
    # The adapter's own fact outranks billing-shaped counters, unknowns included.
    ({"prompt_tokens": 20, "cached_tokens": 0, "reported_tokens": {"prompt_tokens": None, "cached_tokens": None,
                                                                    "completion_tokens": 3}},
     [None, 3, None, None]),
])
def test_event_normalizer_reads_reported_detail_blocks(usage, expected):
    from ouroboros.event_usage import event_token_counts

    assert [event_token_counts(usage)[key] for key in COUNTERS] == expected


# --- #1578 completion: aggregates and derived round fields -------------------------------
# Raw provider bodies -> the REAL adapter (golden offline transport) -> the actual accumulator
# (``add_usage`` via the compaction recorder or the native review episode) -> the actual event
# consumers. Billing sums stay what the pre-#1578 accumulator produced.

FULL = {"input_tokens": 100, "output_tokens": 6, "cache_read_input_tokens": 40, "cache_creation_input_tokens": 25}
PARTIAL = {"input_tokens": 20, "output_tokens": 3}  # no cache counters: the total prompt is unknown
ZERO = {"input_tokens": 7, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
SILENT = {}  # the provider reported no counter at all
MALFORMED = {"input_tokens": -5, "output_tokens": 2.5, "cache_read_input_tokens": True, "cache_creation_input_tokens": 0}
BILLING = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens")
FACT = "reported_tokens"


def _adapter_usage(monkeypatch, raw):
    """One real Anthropic adapter call over a recorded raw body (golden offline transport)."""
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-fixture-key")
    recorder = golden._Recorder([{"kind": "response", "status_code": 200, "json": {
        "id": "msg_aggregate", "role": "assistant", "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn", "usage": raw}}])
    with contextlib.ExitStack() as stack:
        golden._install_fakes(stack, recorder, {"pricing_estimate": 0.001})
        _message, usage = contextvars.Context().run(
            LLMClient().chat, messages=[{"role": "user", "content": "hi"}], model="anthropic::claude-sonnet-4.6")
    return usage


def _billing_reference(total, *usages):
    """The accumulator's numeric fold exactly as it was before #1578 (cost and finality included)."""
    total = dict(total)
    for usage in usages:
        for key in BILLING:
            total[key] = int(total.get(key) or 0) + int(usage.get(key) or 0)
        if usage.get("cost") is not None:
            total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])
            if usage.get("cost_final") is False or usage.get("cost_estimated"):
                total["cost_final"] = False
        else:
            total["cost_final"] = False
    return {key: total.get(key) for key in (*BILLING, "cost", "cost_final")}


@pytest.mark.parametrize("seed,raws,expected", [
    ({}, (FULL, ZERO), [172, 6, 40, 25]),  # every call reported every counter
    ({"_task_attempt": 1}, (ZERO, ZERO), [14, 0, 0, 0]),  # the loop's real seed; true zeros stay zeros
    ({}, (FULL, PARTIAL), [None, 9, None, None]),  # one partial report: those totals are lower bounds only
    ({}, (FULL, SILENT), [None, None, None, None]),
    ({}, (FULL, MALFORMED), [None, None, None, 25]),  # negative, fractional and boolean counters are unknown
    # A preseeded zero and a pre-fact checkpoint's restored zero sum look alike: neither is a reported 0.
    ({**dict.fromkeys(("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens"), 0), "cost": 0.0},
     (ZERO,), [None, None, None, None]),
    ({**dict.fromkeys(BILLING, 0), "cost": 0.0, "cost_final": True}, (ZERO,), [None, None, None, None]),
    # Counts summed before any fact existed came from unreported calls; absent counters start empty.
    ({"prompt_tokens": 50, "completion_tokens": 5}, (FULL,), [None, None, 40, 25]),
    # A checkpoint written with the fact resumes it, unknowns included.
    ({"prompt_tokens": 172, "completion_tokens": 6, "cached_tokens": 40, "cache_write_tokens": 25,
      FACT: {"prompt_tokens": 172, "completion_tokens": 6, "cached_tokens": 40, "cache_write_tokens": 25}},
     (ZERO,), [179, 6, 40, 25]),
    ({"prompt_tokens": 20, "completion_tokens": 3, FACT: {"prompt_tokens": None, "completion_tokens": 3,
                                                          "cached_tokens": "?", "cache_write_tokens": None}},
     (ZERO,), [None, 3, None, None]),
])
def test_add_usage_keeps_billing_sums_and_reports_known_only_while_every_call_reported(
    monkeypatch, offline, seed, raws, expected,
):
    from ouroboros.event_usage import event_token_counts
    from ouroboros.llm import add_usage

    usages = [_adapter_usage(monkeypatch, raw) for raw in raws]
    total = dict(seed)
    for usage in usages:
        add_usage(total, usage)
    assert {key: total.get(key) for key in (*BILLING, "cost", "cost_final")} == _billing_reference(seed, *usages)
    assert [event_token_counts(total)[key] for key in COUNTERS] == expected


@pytest.mark.parametrize("raws,expected", [((FULL, ZERO), [172, 6, 40, 25]), ((FULL, PARTIAL), [None, 9, None, None])])
def test_native_review_episode_total_reaches_its_llm_usage_row_truthfully(tmp_path, monkeypatch, offline, raws, expected):
    """A native multi-round review episode folds its rounds with ``add_usage``; deep_self_review
    returns that episode total and ``agent.py`` emits it as one ``llm_usage`` (row shape below verbatim)."""
    import copy

    from ouroboros.review_execution import ReviewAssignment, ReviewRouteKind
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    usages = [_adapter_usage(monkeypatch, raw) for raw in raws]
    repo = tmp_path / "subject"
    repo.mkdir()
    (repo / "notes.txt").write_text("subject under review\n", encoding="utf-8")
    replies = [{"tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "read_file", "arguments": json.dumps({"path": "notes.txt"})}}]},
               {"content": '[{"severity": "advisory", "item": "x", "evidence": "e", "recommendation": "r"}]'}]

    class Rounds:
        def chat(self, **_kwargs):
            return replies.pop(0), copy.deepcopy(usages[len(usages) - len(replies) - 1])

    request = ReviewRequest(surface="multi_model_review", goal="review", task_id="t-native",
                            session_root=str(repo), session_task="Review the subject.",
                            policy={"output_contract": "JSON array of findings"}, no_proxy=True)
    slot = ReviewSlot(slot_id="deep", model="anthropic::claude-sonnet-4.6", effort="low",
                      route=ReviewRouteKind.API_CHAT, subagent_id="api-critic", native_retrieval_override=True)
    episode = NativeToolRoundReviewExecutor(ReviewAssignment(request=request, slot=slot, call_id="op-1"), llm=Rounds())
    usage = episode.execute().usage
    assert usage["native_rounds"] == 2 and not replies
    reference = _billing_reference({}, *usages)
    assert {key: usage.get(key) for key in BILLING} == {key: reference[key] for key in BILLING}
    event = {"type": "llm_usage", "ts": "2026-10-11T00:00:00+00:00", "task_id": "t-native",
             "model": str(usage.get("resolved_model") or ""), "usage": usage, "category": "deep_self_review"}
    assert _persisted_usage(event, tmp_path / "supervisor") == expected


@pytest.mark.parametrize("raw,round_idx,hit,cold", [
    (FULL, 2, 40 / 165, False),
    (PARTIAL, 1, None, False),  # a first round is never a RE-start, whatever was reported
    (PARTIAL, 2, None, None),  # total prompt and cache unknown: no ratio, no verdict
    ({"input_tokens": 10, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 990},
     2, 0.0, True),
])
def test_round_cache_fields_derive_from_reported_counters(tmp_path, monkeypatch, offline, raw, round_idx, hit, cold):
    accumulated = {}
    step = {"kind": "response", "status_code": 200, "json": {
        "id": "msg_round", "role": "assistant", "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn", "usage": raw}}
    counters, _persisted, adapter = _loop_round(tmp_path, monkeypatch, "anthropic::claude-sonnet-4.6", ANTHROPIC, step,
                                                round_idx=round_idx, accumulated=accumulated)
    [row] = [row for row in _rows(tmp_path / "logs" / "events.jsonl") if row["type"] == "llm_round"]
    assert (row["cache_hit_rate"], row["cache_cold_restart"]) == (pytest.approx(hit) if hit else hit, cold)
    # Cache routing keeps reading the billing-shaped counters it always read.
    parts = [int(raw.get(key) or 0) for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")]
    assert accumulated["_last_round_cache_usage"] == {
        "round": round_idx, "model": accumulated["_last_round_cache_usage"]["model"],
        "provider": "anthropic", "prompt_tokens": sum(parts), "cached_tokens": parts[1], "cache_write_tokens": parts[2],
        # This accepted call's own money as its producer stated it (route facts read it); no engine billing here.
        "cost": adapter["cost"], "cost_final": adapter["cost_final"], "billing": None,
        # Beside them, the counters as reported: the durable llm_round row's own values, unknown kept unknown.
        "reported_tokens": dict(zip(COUNTERS, counters))}
